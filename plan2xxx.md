Implement the following plan:

# Phase 3: Hotkey Mode + Polish

## Context

Phases 1 (offline transcription) and 2 (streaming mic capture) are complete. The model loads, the mic streams, tokens emit in real time. Phase 3 adds the features needed for daily-driver use: a global hotkey to toggle dictation on/off, a prebuffer so speech just before pressing the key isn't lost, `--type` mode to inject text as keystrokes into any application, and proper CLI argument parsing.

## Implementation Steps

### Step 3a: CLI with clap derive

**Files:** `src/main.rs`, `src/streaming.rs`, `src/decoder.rs`

Replace `std::env::args().nth(1)` with clap derive. Create a `StreamConfig` struct to carry runtime-configurable values through the pipeline.

1. Add `#[derive(Parser)]` struct `Cli` to `main.rs`:
   - `wav_file: Option<String>` — positional, backward-compatible
   - `--device` (usize, default 0) — CUDA device index
   - `--delay` (usize, default 4, valid 1-30) — delay tokens
   - `--silence-threshold` (f32, default 0.01)
   - `--silence-flush` (Option<usize>, default = delay + 6)
   - `--hotkey` (Option<String>) — e.g. "F9", "ScrollLock"
   - `--type` (bool flag) — keyboard output mode

2. Create `StreamConfig` in `streaming.rs`:
   ```rust
   pub struct StreamConfig {
       pub delay_tokens: usize,
       pub silence_threshold: f32,
       pub silence_chunks: usize,
       pub hotkey: Option<rdev::Key>,  // None = always-on mode
       pub type_mode: bool,
   }
   ```

3. Make `NUM_DELAY_TOKENS` and `PREFILL_LEN` in `decoder.rs` remain as defaults but accept runtime overrides. The `sinusoidal_embedding()` and `prepare_prefill()` already take delay as parameter — just thread the config value through.

4. Compute `DELAY_SAMPLES`, `SILENCE_CHUNKS` from config at runtime instead of from constants.

5. Update the startup config table to print actual runtime values.

6. Change `run_streaming()` signature to accept `&StreamConfig`.

### Step 3b: Hotkey toggle with rdev

**Files:** new `src/hotkey.rs`, `Cargo.toml`

1. Add `rdev = "0.5"` to `Cargo.toml`.

2. Create `src/hotkey.rs` (~60 lines) with:
   - State constants: `STATE_READY=0`, `STATE_ACTIVE=1`, `STATE_PAUSED=2`
   - `parse_hotkey(s: &str) -> Result<rdev::Key>` — maps strings like "F9", "ScrollLock", "Pause", "RightAlt" to `rdev::Key` variants. Lists valid keys on error.
   - `spawn_hotkey_listener(key: rdev::Key, state: Arc<AtomicU8>)` — spawns a thread calling `rdev::listen()`, toggles state atomically on key-down. Prints `[recording]`/`[paused]` to stderr. Includes 200ms debounce.

3. Add `mod hotkey;` to `main.rs`.

### Step 3c: State machine + rolling prebuffer in streaming loop

**Files:** `src/streaming.rs`

This is the core change. Restructure `run_streaming()` to handle three states.

**State machine:**
```
Loading → Ready → Active ↔ Paused
                              ↓
                          (Ctrl+C) → Stopped
```

**When `--hotkey` is specified:**
- Startup with **silence** (not real mic audio). The model prefills on zeros so it's warm and ready before the first hotkey press. Delay_samples of silence → `run_startup()`.
- Enter `Ready` state. Spawn hotkey listener thread.
- Print "Press <key> to start/stop recording".

**When `--hotkey` is NOT specified:**
- Current behavior preserved exactly: buffer real mic audio for startup, enter `Active` immediately.

**Main loop changes:**
- Track `prev_state` and `current_state` each iteration.
- **Ready/Paused branch:**
  - Drain audio from mpsc channel (prevent backlog).
  - Resample as usual.
  - Feed into rolling prebuffer (`VecDeque<f32>`, capped at `delay_samples` = ~400ms).
  - Do NOT push to IncrementalMel. Do NOT call `run_processing_loop()`.
- **Active branch:**
  - On transition from Ready/Paused → Active (`just_resumed`):
    - Flush prebuffer into `IncrementalMel` (and `sample_buf_for_silence`).
    - Reset `silence_counter` and `silence_emitted`.
  - Normal processing (unchanged from Phase 2).
- **On transition Active → Paused:**
  - Truncate `mel_buffer` to last `CONV_CTX` frames.
  - Clear `sample_buf_for_silence`.
  - Emit newline to close current line cleanly.

**Why no cache reset on resume:** KV caches are intact. RoPE positions continue correctly via `base_offset + current_len`. The prebuffer bridges the audio gap (~400ms of audio captured right before the hotkey press). The encoder's sliding window (750 frames = 15s) ages out stale context naturally. First token after resume: ~53ms (one inference step).

### Step 3d: `--type` mode with enigo

**Files:** `src/streaming.rs`, `Cargo.toml`

1. Add `enigo = "0.2"` to `Cargo.toml`.

2. Create `OutputSink` enum in `streaming.rs`:
   ```rust
   enum OutputSink {
       Stdout,
       Keyboard(enigo::Enigo),
   }
   ```
   With methods:
   - `emit_text(&mut self, text: &str)` — `print!` or `enigo.text()`
   - `emit_newline(&mut self)` — `print!("\n\n")` or `enigo.key(Return, Click)` x2

3. Change `emit_token()` to accept `&mut OutputSink` instead of writing to stdout directly.

4. Initialize `OutputSink` at the top of `run_streaming()` based on `config.type_mode`.

5. Replace all `print!`/`stdout().write_all()` in token emission and silence flush to use the sink.

### Step 3e: Wire together + polish

**Files:** `src/main.rs`

1. Build `StreamConfig` from parsed CLI args in `main()`.
2. Pass config to `run_streaming()`.
3. For offline mode: `--delay` applies to `sinusoidal_embedding()` and `PREFILL_LEN` computation. Pass delay as parameter.
4. Validate `--delay` range (1-30) and `--type` requires `--hotkey` (warn if used without it — typing continuously without a toggle would be chaotic).

## File Change Summary

| File | Action | What changes |
|------|--------|-------------|
| `Cargo.toml` | Modify | Add `rdev = "0.5"`, `enigo = "0.2"` |
| `src/main.rs` | Modify | clap Parser struct, StreamConfig construction, `mod hotkey`, pass config everywhere, update config table |
| `src/hotkey.rs` | **New** | ~60 lines: `parse_hotkey()`, `spawn_hotkey_listener()`, state constants |
| `src/streaming.rs` | Modify | StreamConfig, OutputSink, state machine + prebuffer in main loop, emit_token takes sink, constants → config |
| `src/decoder.rs` | Modify | Thread `delay_tokens` as runtime parameter through `prepare_prefill`/`PREFILL_LEN` |

## Implementation Order

3a → 3b → 3c → 3d → 3e (each step compiles and works before the next starts)

## Verification

1. `voicet` (no args) — identical to Phase 2, always-on streaming
2. `voicet test.wav` — offline transcription still works
3. `voicet --delay 6` — adjusts delay, verify config table prints 6
4. `voicet --hotkey F9` — press F9 to toggle, verify pause/resume works
5. `voicet --hotkey F9 --type` — open Notepad, press F9, speak, verify text appears
6. Long session with multiple pause/resume cycles — verify no memory leak or quality degradation


If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you generated), read the full transcript at: C:\Users\Jon\.claude\projects\C--Users-Jon-working1-without-debugging\4cd94afa-602b-4663-a613-60329be96120.jsonl
