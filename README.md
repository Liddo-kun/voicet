# Voicet

![Demo](Demo.gif)

Ultrafast, ultra efficient Real-time (live) speech-to-text on your GPU. No cloud, no Python, no API keys. Words appear as you speak them inside your app of choice, hotkey optional.

Support for RTX 3000 series and up (>11gb VRAM); DGX Spark [untested]. Windows/Linux

I found it to be slightly faster than speechmatics (probably because it's local) and just as accurate.

[Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) is a relatively large model for speech to text, but in return it gets you insanely low latency and high accuracy. So I built this app on top of it to optimize the best I could.

Result: <3s cold load time, only 51w power consumption on an RTX 5080. 

Automatic paragraph breaks on speech pauses (configurable)

## Features

Fastest implementation of Voxtral Mini 4B Realtime in CUDA on the web. 5x realtime on RTX 5080.

Type mode [--type] supports dictation directly into your text editor  (Word, Claude Code). Just speak and words appear.

Automatic paragraph breaks via silence detection (configurable)

Offline transcription mode with enhanced accuracy of wav files

Hotkey mode in case you want to pause the model while not in use, keeps model preloaded and primed on GPU

## Why Rust instead of Python?

The official HuggingFace pipeline works, but it carries a lot of weight:

|                                       | **Voicet (Rust)**                                             | **HF Transformers (Python)**                      |
| ------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------- |
| Runtime                               | Single 35 MB binary                                           | Python + PyTorch + Transformers (~5 GB installed) |
| Cold Startup                          | 2.94s (mmap weights directly)                                 | 14.4s (Python imports +weight loading)            |
| Throughput (RTX 5080)                 | **63 tok/s** (5x real-time)                                   | 24.5 tok/s (2.1x real-time)                       |
| Realtime Power Consumption (RTX 5080) | 51W                                                           | 110W                                              |
| Streaming                             | Native — causal architecture, incremental mel/encoder/decoder | Requires custom pipeline code                     |
| Dependencies                          | Just CUDA runtime                                             | Python ecosystem, pip, conda, venv                |
| Deployment                            | Copy one binary + model weights                               | Reproduce Python environment                      |

Performance comes from:

- **Flash Attention v2** — fused CUDA kernels for both encoder (32 layers) and decoder (26 layers)
- **Fused RMSNorm** — single CUDA kernel replaces 7 ops, saves ~530 kernel launches per token
- **Precomputed Ada-RMSNorm conditioning** — 26 per-layer scale tensors computed once, not every forward pass
- **Cached lm_head transpose** — avoids transposing 800 MB every token
- **BF16 throughout** — matches PyTorch default, half the memory of FP32

Other features: Resumable requests. Processes audio pipeline in parallel with gpu compute, eliminating gpu compute latency

## Quick start

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 5080)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed (Linux only - included for Windows release)
- Model weights: download [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime) into the voicet.exe directory. You only need consolidated.safetensors and tekken.json

### Build

```bash
cargo build --release
```

### Run

**Live streaming** — speak into your mic:

```bash
./target/release/voicet
```

**Type Mode** — speak into your mic,  words are typed into your document

```bash
./target/release/voicet --type
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

| CLI flag              | Default   | Effect                                                                        |
| --------------------- | --------- | ----------------------------------------------------------------------------- |
| `--delay`             | 4 (320ms) | Accuracy vs latency. Higher = more lookahead. Auto-set to 20 in offline mode. |
| `--silence-threshold` | 0.006     | RMS energy below which audio counts as silence.                               |
| `--silence-flush`     | delay+14  | Consecutive silent chunks before paragraph break.                             |
| `--min-speech`        | 12 (960ms)| Minimum speech duration before silence detection activates.                   |
| `--rms-ema`           | 0.3       | EMA smoothing factor for speech detection.                                    |
| `--hotkey`            | none      | Global hotkey to toggle recording (F1-F12, ScrollLock, Pause).                |
| `--type`              | off       | Type transcribed words directly into the focused app.                         |
| `--model-dir`         | `.`       | Directory containing model files.                                             |
| `--device`            | 0         | CUDA device index.                                                            |

## Dependencies

Built on [candle](https://github.com/huggingface/candle), a minimal ML framework for Rust. A [vendored fork](https://github.com/Liddo-kun/candle/tree/voicet-minimal-kernels) of `candle-flash-attn` is included in `candle-fork/` that compiles only the CUDA kernels this model needs (BF16, head_dim 64/128), reducing binary size from 190 MB to 35 MB. Builds work fully offline.

## License

MIT