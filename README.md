# Voicet

Real-time speech-to-text on your GPU. No cloud, no Python, no API keys.

Voicet runs [Voxtral Mini 4B](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime) — Mistral AI's streaming speech model — as a single Rust binary with CUDA acceleration. Speak into your mic and see words appear as you say them.

## Why Rust instead of Python?

The official HuggingFace pipeline works, but it carries a lot of weight:

|  | **Voicet (Rust)** | **HF Transformers (Python)** |
|---|---|---|
| Runtime | Single 35 MB binary | Python + PyTorch + Transformers (~5 GB installed) |
| Startup | ~3s (mmap weights directly) | ~15s (Python imports + model load) |
| Throughput | ~73 tok/s (0.16x real-time) | ~45 tok/s (0.25x real-time) |
| Streaming | Native — causal architecture, incremental mel/encoder/decoder | Requires custom pipeline code |
| Dependencies | Just CUDA runtime | Python ecosystem, pip, conda, venv |
| Deployment | Copy one binary + model weights | Reproduce Python environment |

Performance comes from:
- **Flash Attention v2** — fused CUDA kernels for both encoder (32 layers) and decoder (26 layers)
- **Fused RMSNorm** — single CUDA kernel replaces 7 ops, saves ~530 kernel launches per token
- **Precomputed Ada-RMSNorm conditioning** — 26 per-layer scale tensors computed once, not every forward pass
- **Cached lm_head transpose** — avoids transposing 800 MB every token
- **BF16 throughout** — matches PyTorch default, half the memory of FP32

## Quick start

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX series)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed
- Model weights: download [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime) into a `Voxtral-Mini-4B-Realtime/` directory

### Build

```bash
cargo build --release
```

### Run

**Live streaming** — speak into your mic:
```bash
./target/release/voicet
```

**Offline transcription** — transcribe a WAV file:
```bash
./target/release/voicet path/to/audio.wav
```

WAV files are automatically resampled to 16 kHz mono if needed.

## How it works

```
16 kHz audio
  → Mel spectrogram (128 bins, 10ms frames)
    → Conv stem (stride-2 downsample)
      → 32-layer causal encoder (sliding window attention, 750 frames)
        → Adapter (4:1 downsample)
          → 26-layer decoder (GQA, Ada-RMSNorm delay conditioning)
            → Token output every 80ms
```

The model is **causal** — each frame only sees past context, never the future. This is what makes real-time streaming possible. (Whisper, by contrast, is bidirectional and needs the full audio before it can transcribe.)

The streaming pipeline buffers 320ms of audio for startup, then processes incrementally: 8 mel frames accumulate, run through the conv stem, encoder chunk, adapter, and decoder to emit one token every 80ms.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Configuration

Tunable parameters are constants in the source code:

| Parameter | Default | Location | Effect |
|---|---|---|---|
| Delay tokens | 4 (320ms) | `decoder.rs` | Accuracy vs latency. Higher = more lookahead = better accuracy. |
| Encoder window | 750 (15s) | `encoder.rs` | How far back the encoder attends. |
| Decoder window | 2048 (~2.7min) | `decoder.rs` | Max decoder context before KV cache trim. |
| Silence threshold | 0.01 | `streaming.rs` | RMS energy below which audio is silence. |
| Silence newline | 10 chunks (800ms) | `streaming.rs` | Consecutive silent chunks before paragraph break. |

## Dependencies

Built on [candle](https://github.com/huggingface/candle), a minimal ML framework for Rust. A [vendored fork](https://github.com/Liddo-kun/candle/tree/voicet-minimal-kernels) of `candle-flash-attn` is included in `candle-fork/` that compiles only the CUDA kernels this model needs (BF16, head_dim 64/128), reducing binary size from 190 MB to 35 MB. Builds work fully offline.

## License

MIT
