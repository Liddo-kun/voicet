# Voicet Architecture — How It Works

## The Big Picture

Voicet turns your voice into text in real time. You speak into a microphone, and words appear on screen as you say them — not after you're done, but *while* you're still talking.

It does this by running a neural network called **Voxtral Mini 4B** locally on your GPU. No cloud, no API, no internet required. The model was built by Mistral AI specifically for this use case: streaming speech-to-text with minimal delay.

---

## The Model: What's Inside Voxtral Mini 4B

The model has three parts that form a pipeline:

### 1. Audio Encoder (970M parameters)

Takes raw sound and converts it into a compressed representation the text decoder can understand.

- Input: raw audio at 16 kHz (16,000 samples per second)
- First converts audio into a **mel spectrogram** — a visual fingerprint of the sound's frequency content over time. Uses 128 frequency bins, computed every 10ms (hop length of 160 samples, N_FFT=400). Uses reflect-padding (matching `torch.stft(center=True)`) and drops the last STFT frame (matching Python's `stft[..., :-1]`).
- A small convolutional layer (2 causal Conv1d layers with GELU) downsamples this 2x (from 100 Hz to 50 Hz frame rate). All output frames are preserved — no truncation to chunk boundaries.
- Then 32 layers of Transformer process these frames — but **causally**: each frame can only see itself and past frames, never the future. This is what makes streaming possible. (Whisper, by contrast, uses bidirectional attention — it needs the full audio before it can transcribe anything.)
- Uses a sliding window with KV cache trimming so memory stays bounded no matter how long you talk (see config table below for window sizes).

### 2. Adapter Layer

A simple bottleneck that further compresses the encoder output by 4x, reducing the frame rate from 50 Hz down to 12.5 Hz. After this step, each frame represents **80ms of audio**. This is the model's fundamental "tick rate" — it makes one transcription decision every 80ms.

### 3. Text Decoder (3.4B parameters)

An autoregressive language model that generates text tokens one at a time. At each 80ms step:

- Takes the current audio embedding from the adapter
- Fuses it (by addition) with the embedding of the most recently generated token
- Runs through 26 Transformer layers to predict the next token
- Emits either a real text token, a `[STREAMING_WORD]` word boundary marker, or a `[STREAMING_PAD]` padding token (meaning "I don't have enough evidence yet, wait")
- Uses a sliding window with periodic KV cache trimming to bound memory during long streaming sessions

### The Delay Mechanism

The model can be configured to look ahead by N steps before committing to text output. This is controlled by **`NUM_DELAY_TOKENS`** in `decoder.rs` (see config table below) — a single constant that drives both the prefill padding count and the sinusoidal conditioning embedding. The embedding is injected into every decoder layer via adaptive normalization (Ada-RMSNorm). Valid range: 1–30 (80ms–2400ms).

The delay conditioning uses a **per-layer bottleneck** architecture. Each of the 26 decoder layers has its own `ada_rms_norm_t_cond` with two linear projections and a GELU activation: `Linear(3072→32) → GELU → Linear(32→3072)`. The sinusoidal embedding of `num_delay_tokens` is projected through this per-layer bottleneck to modulate **only the FFN-path RMSNorm** (not the attention norm). The modulation formula is: `ffn_norm(x) * (1 + ada_rms_norm(t_cond))`.

This is NOT a buffer or queue. It's a **learned behavior**: during training, the model was shown audio with various delay settings (80ms to 2400ms) and learned to adjust its confidence-vs-speed tradeoff accordingly. Higher delay = the model waits for more context before committing to a word = more accurate. Lower delay = faster but more likely to need correction.

---

## The Timing Chain

```
16kHz mono f32 audio
  → Mel spectrogram (N_FFT=400, HOP=160) → 1 frame per 10ms
    → Conv stem (stride-2) → 1 encoder frame per 20ms
      → Encoder (chunks of 4 frames) → 4 frames per 80ms
        → Adapter (4:1 downsample) → 1 adapter frame per 80ms
          → Decoder (fuse token+audio) → 1 token per 80ms
```

**Fundamental tick: 1280 samples = 8 mel frames = 4 encoder frames = 1 adapter frame = 1 decoder token = 80ms.**

---

## Offline Pipeline

`voicet audio.wav` — transcribes a WAV file. Auto-uses delay=20 for maximum accuracy.

1. Load WAV → resample to 16kHz mono if needed
2. Prepend 40,960 silence zeros (32 tokens × 1280 samples)
3. Batch mel spectrogram on entire [silence + audio]
4. `enc.forward()` — resets KV caches, runs conv stem + 32 transformer layers in large chunks (375 frames) for GPU efficiency, trims KV caches after each chunk
5. `adapter.forward()` — batch 4x downsample
6. Decoder prefill: BOS + (32 + delay) PAD tokens, each fused (element-wise add) with adapter frames
7. Decode loop: for each remaining adapter frame, fuse with last token embedding, decoder forward, argmax → emit token. KV caches trimmed periodically.

---

## The Streaming Protocol

### Startup

1. Buffer `(1 + NUM_DELAY_TOKENS) * 1280` samples of real mic audio
2. Prepend 40,960 samples of silence (32 tokens worth)
3. Batch mel → full encoder forward → adapter → decoder prefill (BOS + 38 PAD tokens)
4. Save last 4 mel frames as conv stem context for the incremental loop

### Steady-state loop

Each 80ms tick:

1. Pull mic audio → resample to 16kHz → incremental mel spectrogram
2. When 8 new mel frames accumulate: run conv stem on [4 context + 8 new] frames
3. Skip first 2 conv outputs (zero-padding artifacts), take 4 new frames (= 1 encoder chunk)
4. Encoder `forward_chunk` through 32 transformer layers using KV cache
5. Adapter (4:1 downsample) → 1 adapter frame
6. Fuse with last token embedding → decoder forward → argmax → emit token
7. Trim encoder and decoder KV caches to their sliding window sizes

The conv stem runs on a fixed 12-frame window each tick (O(1) per token, not O(n) over the full session). Only 4 mel frames of context are retained between iterations.

---

## Configurable Parameters

Printed at startup. CLI flags override defaults; architectural constants are fixed.

| CLI flag              | Default   | Effect                                                                                                                                |
| --------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--delay`             | 3 (240ms) | Accuracy vs latency tradeoff. Higher = more lookahead = better accuracy but slower response. Valid: 1-30.                             |
| `--silence-threshold` | 0.007     | Raw RMS energy below which a chunk counts as silent.                                                                                  |
| `--silence-flush`     | delay+9   | Consecutive silent chunks before emitting a paragraph break.                                                                          |
| `--min-speech`        | 8 (640ms) | Minimum consecutive non-silent chunks (EMA-smoothed) before silence detection can trigger. Prevents breaks after 1-2 word utterances. |
| `--rms-ema`           | 0.3       | EMA smoothing factor for speech detection. Lower = smoother, rides over inter-syllable dips.                                          |
| `--hotkey`            | none      | Global hotkey to toggle recording (F1-F12, ScrollLock, Pause, PrintScreen). Enables hotkey mode with prebuffer.                       |
| `--type`              | off       | Inject text as keystrokes into focused app via enigo instead of printing to stdout.                                                   |
| `--device`            | 0         | CUDA device index.                                                                                                                    |

| Architectural constant | Location     | Value          | Notes                                 |
| ---------------------- | ------------ | -------------- | ------------------------------------- |
| Encoder sliding window | `encoder.rs` | 750 (15s)      | Fixed by model architecture.          |
| Decoder sliding window | `decoder.rs` | 2048 (~2.7min) | Max decoder KV cache before trimming. |
| Compute dtype          | `main.rs`    | BF16           | Matches PyTorch default.              |

---

## Dependencies

Voicet uses the [candle](https://github.com/huggingface/candle) ML framework for Rust. Three crates are used: `candle-core`, `candle-nn`, and `candle-flash-attn`. All are vendored locally in `candle-fork/` (referenced via `path` in `Cargo.toml`) so builds work offline.

The `candle-flash-attn` crate is a fork ([Liddo-kun/candle](https://github.com/Liddo-kun/candle), branch `voicet-minimal-kernels`) that compiles only the CUDA kernels this model needs: BF16, head_dim 64 (encoder) and 128 (decoder). The upstream crate compiles all 32 variants (8 head dims × 2 dtypes × 2 causal modes), which inflated the binary from ~10 MB to ~190 MB. The fork reduces it to ~35 MB.

Other dependencies: `rdev` (global hotkey + Ctrl+C via low-level keyboard hook), `enigo` (keystroke injection for `--type` mode), `cpal` (mic capture), `hound` (WAV reading), `clap` (CLI parsing).

---

## Comparison: What Others Do Differently

### voxtral.c (antirez)

- **Zero dependencies** beyond system BLAS (Accelerate on macOS, OpenBLAS on Linux)
- Implements EVERYTHING from scratch: mel spectrogram, transformer inference, tokenizer, safetensors loading
- Metal GPU backend with custom kernels for Apple Silicon
- Memory-maps weights directly from safetensors — near-instant startup
- ~8 C files, compiles to a single binary
- **Limitation**: macOS-focused (Metal backend), BLAS fallback for CPU

### voxtral-mini-realtime-rs (TrevorS)

- Uses **Burn** ML framework (Rust) instead of PyTorch
- Custom WGSL shaders for WebGPU inference
- Supports **Q4 quantization** (2.5GB vs 9GB) with fused dequant+matmul kernels
- Runs in the browser via WASM + WebGPU
- **703MB peak RAM** with Q4 vs our multi-GB footprint
- Real-time factor of 0.416 (transcribes faster than real-time)

---

## Performance Optimizations

### GPU Kernel Optimizations

| #   | Optimization              | File(s)      | What it does                                                                                                                                                                                                                                                                                                             |
| --- | ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Flash Attention (encoder) | `encoder.rs` | `flash_attn_windowed(q, k, v, scale, Some(749), Some(0))` — single fused CUDA kernel per layer handles Q@K, causal sliding window masking, softmax, and @V. No intermediate attention matrix materialized. 32 layers × 1 kernel replaces 32 × (matmul + transpose + mask + softmax + matmul + transpose + 2 contiguous). |
| 2   | Flash Attention (decoder) | `decoder.rs` | `flash_attn_windowed(q, k, v, scale, Some(2047), Some(0))` — same as encoder, plus native GQA support (32Q/8KV heads handled internally, no `repeat_kv()` needed). 26 layers × 1 kernel.                                                                                                                                 |
| 3   | Fused RMSNorm             | `common.rs`  | `candle_nn::ops::rms_norm` — single CUDA kernel replaces 7 separate ops (to_dtype, sqr, mean, add, sqrt, div, mul, to_dtype). Affects both encoder (65 calls) and decoder (53 calls) per forward. Saves ~530 kernel launches/token.                                                                                      |
| 4   | Precomputed Ada-RMSNorm   | `decoder.rs` | `precompute_t_cond()` computes all 26 per-layer Ada-RMSNorm scales once (Linear→GELU→Linear→unsqueeze→add). Stored in `ada_scales: Option<Vec<Tensor>>`. Called once after t_cond is created; forward uses cached values. Saves ~130 kernel launches/token.                                                              |

### Memory Layout Optimizations

| #   | Optimization      | File(s)      | What it does                                                                                                                                                                             |
| --- | ----------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5   | KV cache layout   | `common.rs`  | `KvCache` uses `[batch, seq, heads, dim]` layout (flash attention's expected format). Append/trim/len all operate on dim 1. No transposes needed between cache and attention.            |
| 6   | RoPE layout       | `common.rs`  | `RotaryEmbedding::apply` broadcasts over dim 2 (heads), matching the `[batch, seq, heads, dim]` tensor layout. Seq_len read from `dim(1)`, cos/sin unsqueeze on dim 2.                   |
| 7   | Zero-copy lm_head | `decoder.rs` | `tok_embeddings.t()` is a zero-copy view (stride change). cuBLAS handles the transpose natively via its GEMM transpose flags. Saves ~800MB VRAM vs caching a contiguous transposed copy. |

### KV Cache Management

| #   | Optimization                | File(s)      | What it does                                                                                                                                                                                                                                                     |
| --- | --------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8   | KV cache trim with headroom | `common.rs`  | When cache exceeds sliding window, trims to 75% of max instead of `max - 1`. This means trimming (which copies the entire cache) happens every N/4 tokens instead of every token. For the decoder (window 2048): trims every ~512 tokens instead of every token. |
| 9   | Offline encoder KV trim     | `encoder.rs` | Offline `forward()` trims encoder KV caches after each chunk. Without this, 5 minutes of audio accumulated unbounded KV entries across 32 layers, causing OOM.                                                                                                   |
| 10  | Offline decoder KV trim     | `main.rs`    | Offline generation loop trims decoder KV caches. Same OOM fix as encoder — without trimming, long files exhausted VRAM.                                                                                                                                          |

### Offline Mode Optimizations

| #   | Optimization              | File(s)      | What it does                                                                                                                                                                                                                                                       |
| --- | ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 11  | Large encoder chunks      | `encoder.rs` | Offline `forward()` uses chunks of `SLIDING_WINDOW / 2` (375 frames) instead of `CHUNK_SIZE` (4 frames). Larger matmuls saturate GPU compute units. Encoder time on 5min audio: 53s → 1s (52x speedup). Streaming still uses 4-frame chunks via `forward_chunk()`. |
| 12  | Auto delay=20 for offline | `main.rs`    | Offline transcription auto-overrides delay to 20 (1600ms lookahead) regardless of `--delay` flag. No latency penalty offline, so maximum accuracy is free.                                                                                                         |

### Model Loading Optimizations

Reduced end to end loading of voice..exe: CUDA runtimes + model load to GPU from ~4s to 2.94s on Arrow Lake + RTX 5080 by:

| #   | Optimization                     | File(s)                                 | What it does                                                                                                                                                                                                                                          |
| --- | -------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13  | Precomputed deinterleave indices | `common.rs`, `encoder.rs`, `decoder.rs` | `DeinterleaveIdx` struct creates even/odd GPU index tensors once per head_dim (64 for encoder, 128 for decoder). Reused across all layers. 4 total CUDA allocations + H2D transfers instead of 232 (116 calls × 2 tensors each).                      |
| 14  | mmap readahead thread            | `main.rs`                               | Background thread pre-faults mmap pages at full sequential NVMe bandwidth. Spawned *before* CUDA init so it gets ~0.5-1s head start. Pages are in OS cache by the time `vb.get()` copies tensors to GPU — eliminates per-tensor page-fault stalls.    |
| 15  | Deferred mmap cleanup            | `main.rs`                               | `vb` and `st_data` are not explicitly dropped after loading — they live until process exit. Avoids a ~0.6s `munmap` syscall that tears down page table entries for ~2.25M pages (9GB / 4KB). OS reclaims physical pages lazily under memory pressure. |

### Measured Performance (RTX 5080)

- **Cold model load**: 2.94s (Arrow Lake + RTX 5080)
- **Streaming**: ~30ms per tick (encoder chunk + decoder step), well within 80ms budget
- **Offline decoding**: 63 tok/s (0.20x real-time factor on 5min audio)
- **Offline encoder**: 1.0s for 5min audio (14,880 frames)
- **VRAM usage**: ~10.6GB (model weights + KV caches)

---

## Key Weight Name Mapping

The safetensors file uses **Mistral native naming** (not HuggingFace). 711 tensors total. We load from `consolidated.safetensors`.

```
Safetensors name                                                          → Rust struct field
────────────────────────────────────────────────────────────────────────────────────────────────
ENCODER (421 tensors)
mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.{weight,bias}  → encoder.conv_stem[0]
mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.{weight,bias}  → encoder.conv_stem[1]
mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{i}.attention.{wq,wk,wv,wo}.{weight,bias}  → encoder.layers[i].attn
mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{i}.feed_forward.{w1,w2,w3}.{weight,bias}  → encoder.layers[i].mlp
mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{i}.{attention_norm,ffn_norm}.weight  → encoder.layers[i].norms
mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight  → encoder.final_norm

ADAPTER (2 tensors)
mm_streams_embeddings.embedding_module.audio_language_projection.0.weight  → adapter.linear_in   [3072, 5120]
mm_streams_embeddings.embedding_module.audio_language_projection.2.weight  → adapter.linear_out  [3072, 3072]

SHARED TOK_EMBEDDINGS (1 tensor — serves as embed + lm_head, tied weights)
mm_streams_embeddings.embedding_module.tok_embeddings.weight  → shared.tok_embeddings  [131072, 3072]

DECODER (287 tensors)
layers.{i}.attention.{wq,wk,wv,wo}.weight         → decoder.layers[i].attn (GQA: 32Q/8KV heads)
layers.{i}.{attention_norm,ffn_norm}.weight         → decoder.layers[i].norms
layers.{i}.feed_forward.{w1,w2,w3}.weight           → decoder.layers[i].mlp
layers.{i}.ada_rms_norm_t_cond.{0,2}.weight         → decoder.layers[i].ada_norm (bottleneck dim=32)
norm.weight                                         → decoder.final_norm
```

**Implementation notes:**

- Q/K weights must be deinterleaved at load time via `deinterleave_qk()` — both encoder (32 heads, head_dim=64) and decoder (32Q/8KV heads, head_dim=128). V and O weights need no conversion.
- Encoder has biases on wq, wv, wo, w2. Decoder has NO biases on any linear layer.
- `candle-nn` must have `features = ["cuda"]` to unlock fused CUDA kernels (`rms_norm`, `softmax_last_dim`). Without it, these silently fall back to multi-op defaults.
- KV cache uses flash attention layout `[batch, seq, heads, hdim]`. RoPE applies on this layout directly (no transpose needed).
- candle `Tensor::sub` does not broadcast — use `.broadcast_sub()` etc.

**Special tokens (from tekken.json, IDs = 131072 + rank):**

- `<s>` → 131073 (BOS), `</s>` → 131074 (EOS)
- `[STREAMING_PAD]` → 131104 — "waiting" token
- `[STREAMING_WORD]` → 131105 — word boundary
