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

Printed at startup. All values reference constants in the source code.

| Parameter | Constant | Location | Default | Effect |
| --- | --- | --- | --- | --- |
| Delay tokens | `NUM_DELAY_TOKENS` | `decoder.rs` | 4 (320ms) | Accuracy vs latency tradeoff. Higher = more lookahead = better accuracy but slower response. |
| Encoder sliding window | `SLIDING_WINDOW` | `encoder.rs` | 750 (15s) | How far back the encoder can attend. Fixed by model architecture. |
| Decoder sliding window | `SLIDING_WINDOW` | `decoder.rs` | 2048 (~2.7min) | Max decoder context before KV cache is trimmed. Increase for very long sessions if GPU memory allows. |
| Silence threshold | `SILENCE_THRESHOLD` | `streaming.rs` | 0.01 | RMS energy below which audio is considered silence. |
| Silence newline after | `SILENCE_CHUNKS` | `streaming.rs` | 10 (800ms) | Consecutive silent chunks before emitting a newline. |
| Compute dtype | — | `main.rs` | BF16 | Model precision. BF16 matches PyTorch default. |

---

## Dependencies

Voicet uses the [candle](https://github.com/huggingface/candle) ML framework for Rust. Three crates are used: `candle-core`, `candle-nn`, and `candle-flash-attn`. All are vendored locally in `candle-fork/` (referenced via `path` in `Cargo.toml`) so builds work offline.

The `candle-flash-attn` crate is a fork ([Liddo-kun/candle](https://github.com/Liddo-kun/candle), branch `voicet-minimal-kernels`) that compiles only the CUDA kernels this model needs: BF16, head_dim 64 (encoder) and 128 (decoder). The upstream crate compiles all 32 variants (8 head dims × 2 dtypes × 2 causal modes), which inflated the binary from ~10 MB to ~190 MB. The fork reduces it to ~35 MB.

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
