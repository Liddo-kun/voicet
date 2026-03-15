# Phase 3: Hotkey Mode + Polish

## Features Implemented

### CLI with clap derive (`src/main.rs`)
- Replaced `std::env::args().nth(1)` with clap `#[derive(Parser)]`
- Arguments: `wav_file` (positional, optional), `--device`, `--delay`, `--silence-threshold`, `--silence-flush`, `--min-speech`, `--rms-ema`, `--hotkey`, `--type`
- Runtime validation of `--delay` range (1-30) and warning for `--type` without `--hotkey`
- Config table prints actual runtime values including hotkey, type mode, and all silence/speech detection parameters

### Hotkey toggle with rdev (`src/hotkey.rs`)
- `parse_hotkey()` maps strings (F1-F12, ScrollLock, Pause, PrintScreen) to `rdev::Key`
- Unified `spawn_listener()` handles both hotkey toggle and Ctrl+C detection via rdev's low-level keyboard hook, replacing the `ctrlc` crate (which didn't work reliably in PowerShell 7)
- 200ms debounce on hotkey to prevent auto-repeat from toggling state rapidly

### State machine + rolling prebuffer (`src/streaming.rs`)
- **Hotkey mode:** Model prefills on silence (instant startup), enters Ready state. Mic audio feeds a rolling `VecDeque<f32>` prebuffer capped at `delay_samples` (~400ms) so speech just before pressing the hotkey isn't lost. No silence/speech detection in hotkey mode — user controls pausing explicitly.
- **Always-on mode:** Original Phase 2 behavior preserved, with enhanced silence/speech detection.
- **Resume (Ready/Paused → Active):** Prebuffer flushed into IncrementalMel.
- **Pause (Active → Paused):** Final audio fed to IncrementalMel, then `(delay_tokens + 4) * SAMPLES_PER_TOKEN` of silence pushed through the pipeline at GPU speed (~30ms/token) to flush the decoder's lookahead buffer. Only then truncate mel_buffer and go idle.

### `--type` mode with enigo (`src/streaming.rs`)
- `OutputSink` enum: `Stdout` (print to terminal) or `Keyboard` (inject keystrokes via enigo)
- `emit_text()` and `emit_newline()` dispatch to the appropriate output
- All token emission and silence flush use the sink

### Silence & speech detection (always-on mode only)
- **Silence detection:** Raw RMS per 80ms chunk compared against `--silence-threshold` (default 0.007). After `--silence-flush` consecutive silent chunks (default delay+9), emits paragraph break.
- **Speech detection:** EMA-smoothed RMS (`--rms-ema`, default 0.3) to ride over natural inter-syllable dips. Must accumulate `--min-speech` consecutive non-silent chunks (default 8 = 640ms) before silence detection can trigger a paragraph break. Prevents breaks after 1-2 word utterances.
- **Pre-speech suppression:** `silence_emitted` starts `true` on startup and on resume, preventing paragraph breaks before the user has spoken.
- **Debug status line:** ANSI escape codes render a live RMS bar with threshold marker, raw/smoothed RMS values, silence counter, speech counter, and emitted flag at row 1 of the terminal without disturbing transcription output.

### StreamConfig (`src/streaming.rs`)
- Runtime-configurable struct carrying `delay_tokens`, `silence_threshold`, `silence_chunks`, `min_speech_chunks`, `rms_ema_alpha`, `hotkey`, `type_mode`
- Replaces compile-time constants `SILENCE_THRESHOLD`, `SILENCE_CHUNKS`, `NUM_DELAY_TOKENS`
- `decoder::prefill_len(delay_tokens)` computes prefill length at runtime

### Multi-channel mic support (`src/streaming.rs`)
- `open_mic()` accepts the device's native channel count instead of forcing mono
- Downmixes to mono by averaging all channels per frame in the audio callback

### Other changes
- `IncrementalMel::new()` constructor added to `src/mel.rs` for hotkey mode (fresh start with reflect padding on first push)
- `decoder::prepare_prefill()` now takes `delay_tokens` parameter
- `ctrlc` crate removed from dependencies

## CLI Reference

```
voicet [OPTIONS] [WAV_FILE]

Options:
  --device <N>                CUDA device index [default: 0]
  --delay <N>                 Delay tokens, 1-30 [default: 3]
  --silence-threshold <F>     RMS threshold for silence [default: 0.007]
  --silence-flush <N>         Silent chunks before paragraph break [default: delay+9]
  --min-speech <N>            Speech chunks before silence detection arms [default: 8]
  --rms-ema <F>               EMA smoothing factor for speech detection [default: 0.3]
  --hotkey <KEY>              Toggle key (F1-F12, ScrollLock, Pause, PrintScreen)
  --type                      Type as keystrokes into focused app
```

## File Change Summary

| File | Action | What changed |
|------|--------|-------------|
| `Cargo.toml` | Modified | Added `rdev`, `enigo`; removed `ctrlc` |
| `src/main.rs` | Modified | clap Parser, StreamConfig construction, `mod hotkey`, config table |
| `src/hotkey.rs` | New | ~80 lines: `parse_hotkey()`, `spawn_listener()` |
| `src/streaming.rs` | Modified | StreamConfig, OutputSink, state machine, prebuffer, pipeline flush, silence/speech detection with EMA, debug status line, multi-channel downmix |
| `src/decoder.rs` | Modified | `prefill_len()` helper, `prepare_prefill` accepts `delay_tokens` |
| `src/mel.rs` | Modified | `IncrementalMel::new()` constructor |
