# Plan: Voicet Rust Rewrite (candle + CUDA, BF16)

## References

- **[Voxtral Realtime paper](https://arxiv.org/html/2602.11298v1)** — Model architecture, training approach, delay conditioning, streaming inference protocol.
- **[Mistral realtime transcription docs](https://docs.mistral.ai/capabilities/audio_transcription/realtime_transcription)** — Official API spec: audio format, chunk sizing, streaming events.
- **[voxtral.c](https://github.com/antirez/voxtral.c)** — Reference C implementation by antirez (Metal GPU backend).
- **[voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs)** — Reference Rust implementation using Burn framework (WebGPU, Q4 quantization).

## Why Rust + candle

|                 | Python (current)                                    | Rust + candle (target)                    |
| --------------- | --------------------------------------------------- | ----------------------------------------- |
| Dependencies    | 78 packages, 5GB venv                               | ~10 crates, ~2MB binary                   |
| Startup         | ~15s (Python imports + model load)                  | < 2s (mmap weights, no interpreter)       |
| Model precision | BF16 via PyTorch                                    | BF16 via candle + cuBLAS                  |
| VRAM            | ~9GB model + PyTorch overhead                       | ~9GB model + ~100MB KV caches             |
| Token emission  | Held back by TextIteratorStreamer word buffering    | Immediate — decode each token ID directly |
| Audio capture   | sounddevice (PortAudio Python bindings)             | cpal (native Rust, cross-platform)        |
| GIL             | Python GIL serializes mic thread + inference thread | No GIL — true parallelism                 |

**Why not C (voxtral.c):** macOS/Metal focused. We'd need CUDA kernels from scratch for Windows.
**Why not C + ggml:** No Voxtral architecture support. More work than using candle.

## What candle gives us

- `candle-core`: Tensor operations with CUDA backend and BF16 support, cuBLAS-backed matmul
- `candle-nn`: `Linear` layer, `VarBuilder` for safetensors loading (mmap'd)
- `candle-transformers`: Has an offline Voxtral module we used as **reference only** — realtime variant differs too much

## Dependencies

```toml
[dependencies]
candle-core = { git = "...", features = ["cuda"] }   # tensor ops, CUDA
candle-nn = { git = "...", features = ["cuda"] }     # Linear, VarBuilder, fused RMSNorm/softmax CUDA kernels
candle-flash-attn = { git = "..." }                  # flash attention v2 (fused Q@K+softmax+@V)
safetensors = "0.7"                                   # weight loading (must match candle's version)
serde_json = "1"                                      # tekken.json parsing
base64 = "0.22"                                       # tekken token_bytes decoding
memmap2 = "0.9"                                       # mmap safetensors
hound = "3"                                           # WAV file reading
num_cpus = "1"                                        # mel spectrogram threading
clap = { version = "4", features = ["derive"] }       # CLI args
anyhow = "1"                                          # error handling
cpal = "0.15"                                         # mic capture
rdev = "0.5"                                          # global hotkeys + Ctrl+C
enigo = "0.2"                                         # keyboard output (--type mode)
```

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│  voicet (single binary, ~2MB + model weights 9GB)   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  mic (cpal) ──160 samples/10ms──► chunk assembler   │
│                                      │              │
│                              mel spectrogram        │
│                              (FFT + filterbank)     │
│                                      │              │
│                              conv1d stem (2 layers)  │
│                              + padding cache        │
│                                      │              │
│                              causal encoder         │
│                              (32 layers, sliding    │
│                               window 750)           │
│                                      │              │
│                              adapter (4x downsample)│
│                                      │              │
│              ┌───────────────────────┤              │
│              │                       │              │
│         time embedding          fuse: audio_embed   │
│         (delay τ → g(τ))        + last_token_embed  │
│              │                       │              │
│              └───► decoder (26 layers)              │
│                    Ada-RMSNorm(g(τ))                │
│                    GQA, RoPE, SwiGLU                │
│                           │                         │
│                    token → BPE decode → stdout      │
│                                                     │
│  silence detector ──► streamer flush                │
│  hotkey listener  ──► session control               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## What we wrote from scratch

| Component           | File               | Notes                                                                                                                                          |
| ------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Mel spectrogram     | `src/mel.rs`       | Batch FFT + mel filterbank (multi-threaded), `IncrementalMel` for streaming, reflect-pad, last-frame drop                                      |
| Conv1d stem         | `src/encoder.rs`   | 2 causal conv layers, left-padded, GELU, no frame truncation. Incremental mode: 4-frame context window                                         |
| Encoder (32 layers) | `src/encoder.rs`   | Causal sliding window attn, flash attention, RoPE θ=1M, SwiGLU, fused RMSNorm, Q/K deinterleave, KV cache trimming                             |
| Adapter             | `src/adapter.rs`   | 4x downsample via concat + 2 linear layers + GELU                                                                                              |
| Decoder (26 layers) | `src/decoder.rs`   | GQA 32Q/8KV, flash attention, Ada-RMSNorm bottleneck (precomputed), sinusoidal delay embed, Q/K deinterleave, KV cache trimming                |
| KV cache            | `src/common.rs`    | Shared KV cache in flash attention layout `[batch, seq, heads, hdim]` with `trim()` and `base_offset` tracking for correct RoPE after trimming |
| Tokenizer           | `src/tokenizer.rs` | Parses tekken.json directly (Mistral native format)                                                                                            |
| Streaming pipeline  | `src/streaming.rs` | Mic capture (cpal), resampling, incremental mel/conv/encoder/decoder loop, silence detection                                                   |
| Entry point         | `src/main.rs`      | CLI dispatch (offline vs streaming), config table, offline WAV transcription                                                                   |

## Implementation Phases

### Phase 1: Offline transcription — DONE

Goal: `voicet transcribe audio.wav` produces correct text. No streaming, no mic.

**Phase 1a — Scaffold + weight loading (DONE)**

- Project scaffold, candle deps, CUDA builds verified
- Weight loading: mmap `consolidated.safetensors` via `VarBuilder::from_slice_safetensors`
- Tokenizer: parse tekken.json directly (serde_json + base64)

**Phase 1b — Audio encoder (DONE)**

- Mel spectrogram from scratch in `src/mel.rs`
- Encoder from scratch in `src/encoder.rs`
- Performance: 33s audio → 0.69s forward, 2s → 0.043s

**Phase 1c — Adapter + decoder + transcription (DONE)**

- Adapter, decoder, delay conditioning, generation loop all working
- Validated against Python implementation: output matches on 3.7s and 28s test clips

**Key bugs fixed during Phase 1:**

- Mel: reflect-pad instead of zero-pad, last-frame drop, no Whisper-style frame rounding
- Conv embedder: was truncating frames to CHUNK_SIZE multiple (313→312), fixed to process all frames
- Q/K weight deinterleaving: `consolidated.safetensors` stores interleaved head dimensions, `deinterleave_qk()` converts at load time

### Phase 2: Streaming mic capture with zero latency — DONE

Goal: `voicet` opens mic, streams audio through the model, emits tokens in real time.

This combines streaming inference + mic capture into one phase — there's no point building streaming without audio input to drive it.

**2a — cpal mic capture + chunk assembly (DONE)**

1. **Audio capture** — cpal input stream, mono f32 at native rate, push samples via mpsc channel to inference thread
2. **Resampling** — Integer-ratio decimation (e.g. 48kHz→16kHz by factor 3) on the inference thread
3. **Chunk assembly** — `IncrementalMel` accumulates 16kHz samples and drains mel frames as they become available
4. **Zero-latency goal** — Audio callback never blocks; inference thread pulls from channel at its own pace

**2b — Streaming encoder (DONE)**
4. **Incremental conv stem** — Keeps only 4 mel frames of context between iterations. Runs conv stem on [4 context + 8 new] frames, skips first 2 outputs (zero-padding artifacts), takes 4 new frames = 1 encoder chunk. O(1) per token, not O(n) over session history.
5. **Encoder KV cache streaming** — `forward_chunk()` processes one 4-frame chunk through 32 transformer layers using existing KV cache (no reset). KV caches are trimmed to the sliding window after each chunk to bound GPU memory.
6. **Adapter streaming** — Each encoder chunk produces exactly 4 frames → adapter produces 1 adapter frame per chunk.

**2c — Streaming decoder (DONE)**
7. **One token per adapter frame** — For each adapter output:

- Fuse audio embed + last token embed (element-wise addition)
- Decoder forward with KV cache → next token logits → argmax
- Append to decoder KV cache
- KV caches trimmed to sliding window after each forward
  8. **Token emission** — Decode each token ID immediately:
- `[STREAMING_PAD]` (131104) → skip (model is waiting for more context)
- `[STREAMING_WORD]` (131105) → emit space
- Normal tokens → emit text
- No word buffering (unlike Python's TextIteratorStreamer)

**2d — Silence detection + flush (DONE)**
9. **Silence detection** — Monitor RMS energy of 1280-sample blocks. After `SILENCE_CHUNKS` consecutive silent chunks, emit newline.
10. **Ctrl+C handling** — Atomic flag, clean exit, prints "--- Stopped ---".

**2e — Bug fixes and memory bounding (DONE)**
11. **Unbounded mel growth** — Fixed: conv stem was re-run on entire mel history every 80ms (O(n) per token). Now uses incremental 4-frame context window (O(1) per token).
12. **Encoder KV cache growth** — Fixed: `KvCache::trim()` drops entries beyond sliding window. Tracks `base_offset` for correct RoPE after trimming.
13. **Decoder KV cache growth** — Fixed: same trim mechanism.
14. **Unified delay configuration** — `NUM_DELAY_TOKENS` in `decoder.rs` is the single constant driving prefill padding, sinusoidal embedding, and startup buffer size.
15. **Config table** — All tuneable parameters printed at startup.

**Validated** — Offline mode (`voicet <file.wav>`) produces identical output before and after all changes.

### Phase 3: Hotkey mode + polish — DONE

See [phase3.md](phase3.md) for full details.

Goal: daily-driver features — hotkey toggle, keyboard output, configurable CLI, robust silence detection.

1. **CLI with clap derive** — `--device`, `--delay`, `--silence-threshold`, `--silence-flush`, `--min-speech`, `--rms-ema`, `--hotkey`, `--type`. Config table prints all runtime values at startup.
2. **Hotkey toggle** — `rdev` low-level keyboard hook. Unified `spawn_listener()` handles both hotkey toggle (200ms debounce) and Ctrl+C detection (replaced `ctrlc` crate, which failed in PowerShell 7).
3. **State machine** — Ready → Active ↔ Paused. Hotkey mode prefills on silence (instant startup). On pause, flushes `(delay_tokens + 4)` tokens of silence through the pipeline at GPU speed to emit remaining in-flight text. No silence/speech detection in hotkey mode.
4. **Rolling prebuffer** — `VecDeque<f32>` capped at `delay_samples` (~400ms) captures audio while paused so speech before the hotkey press isn't lost. Flushed to IncrementalMel on resume.
5. **`--type` mode** — `enigo` crate injects keystrokes into the focused app via `OutputSink` enum.
6. **Silence/speech detection** (always-on mode) — Raw RMS for silence counting, EMA-smoothed RMS for speech detection (rides over inter-syllable dips). Minimum speech duration (`--min-speech`, default 8 chunks = 640ms) prevents paragraph breaks after short utterances.
7. **Multi-channel mic** — Accepts native channel count, averages all channels to mono in the callback.
8. **Dependencies** — Added `rdev`, `enigo`; removed `ctrlc`.

## What we're NOT doing

- No quantization (BF16 on 24GB VRAM is fine)
- No WASM/browser support
- No multi-GPU, no batched inference
- No speaker diarization
- No hand-written CUDA kernels (using candle's cuBLAS matmul + candle-flash-attn + candle-nn fused kernels)

## Success criteria

- Single static binary (+ model weights)
- < 10 crate dependencies
- Startup time < 2s
- Same transcription quality as Python version
- Lower latency (no GIL, no TextIteratorStreamer word buffering)
- Memory: ~9GB VRAM for model + ~100MB for KV caches
