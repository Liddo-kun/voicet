# Build

## Windows

cargo build --release --target-dir target

## Linux (x86_64 or aarch64/DGX Spark)

# Prerequisites: apt install pkg-config libasound2-dev libx11-dev libxtst-dev libxdo-dev

# CUDA toolkit must be installed (nvcc in PATH or /usr/local/cuda)

export CUDA_PATH=/usr/local/cuda  # override .cargo/config.toml's Windows default
cargo build --release

# Or use the build script (auto-detects arch, checks deps, sets CUDA_PATH):

./scripts/build-linux.sh              # build only
./scripts/build-linux.sh --package    # build + create release tarball

# Test

## Streaming (mic)

./target/release/voicet.exe --model-dir Voxtral-Mini-4B-Realtime

## Offline (WAV file)

./target/release/voicet.exe --model-dir Voxtral-Mini-4B-Realtime church.wav

## Full featured

./target/release/voicet.exe --model-dir Voxtral-Mini-4B-Realtime --hotkey F9 --type

# Model files

Dev: model weights are in Voxtral-Mini-4B-Realtime/ subfolder
Release: users place them next to the binary (--model-dir defaults to ".")

# Release process

## Windows

1. Build, then copy exe + CUDA DLLs + mel_filters.bin + voicet_typemode.bat to release/ folder
   - CUDA DLLs: cublas64_13.dll, curand64_10.dll (check with: dumpbin //DEPENDENTS voicet.exe | grep -i cu)
2. Zip with PowerShell: Compress-Archive
3. gh release create vX.Y.Z <zip> --title "..." --notes "..."
4. Version in commit message, tag on commit, release tied to tag

## Linux

1. Run ./scripts/build-linux.sh --package (on target machine or matching arch)
2. Produces voicet-v{VERSION}-linux-{x64|arm64}-cuda.tar.gz
3. Tarball contains: voicet binary + CUDA .so libs + mel_filters.bin + run-voicet.sh wrapper
4. gh release create vX.Y.Z <tarball> --title "..." --notes "..."

# Folder structure notes

- target/release/ — cargo build output
- release/ — Windows distribution staging (exe + CUDA DLLs + mel_filters.bin)
- release-linux/ — Linux distribution staging (created by build-linux.sh --package)
- scripts/ — build scripts (build-linux.sh)
- candle-fork/ — vendored candle dependency, don't update without reason
- 

# Platform notes

- .cargo/config.toml sets CUDA_PATH to Windows default; Linux builds override via env var
- No source code changes needed for Linux — rdev, enigo, cpal all support Linux natively
- DGX Spark = aarch64 (Grace ARM CPU + Blackwell GPU), other Linux RTX = x86_64
- Linux build deps: ALSA (cpal), X11+Xtst (rdev), libxdo (enigo)
- Linux release bundles .so files + LD_LIBRARY_PATH wrapper; Windows bundles .dll files

# Docs

- ARCHITECTURE.md — current state reference (model, protocols, optimizations)
- PLAN_voicet-rust-rewrite.md — historical record of what was built per phase
- README.md — user-facing project overview

# GitHub

- Repo: github.com/Liddo-kun/voicet (public)
- Versioning: v0.3, v0.5, v0.5.1, v0.5.2 — include version in commit messages
