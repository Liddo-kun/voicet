"""A/B comparison: HuggingFace VoxtralRealtime streaming mic transcription.
Usage: python test_hf_streaming.py [--delay N] [--device N]
Compare against: voicet.exe --model-dir . --hotkey F9
"""

import time
import sys
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor, TextStreamer

# --- Config ---
SAMPLE_RATE = 16000
DELAY_TOKENS = int(sys.argv[sys.argv.index("--delay") + 1]) if "--delay" in sys.argv else 3
CUDA_DEVICE = int(sys.argv[sys.argv.index("--device") + 1]) if "--device" in sys.argv else 0
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

print(f"Delay tokens: {DELAY_TOKENS}")
print(f"CUDA device: {CUDA_DEVICE}")

# --- Load model ---
t0 = time.time()
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Loading model to cuda:{CUDA_DEVICE}...")
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=f"cuda:{CUDA_DEVICE}",
)
print(f"Model loaded in {time.time() - t0:.2f}s")

# Get chunk sizes from processor
first_chunk_samples = processor.num_samples_first_audio_chunk
chunk_samples = processor.num_samples_per_audio_chunk
print(f"First chunk: {first_chunk_samples} samples ({first_chunk_samples/SAMPLE_RATE*1000:.0f}ms)")
print(f"Per chunk: {chunk_samples} samples ({chunk_samples/SAMPLE_RATE*1000:.0f}ms)")

# --- Live token streamer that skips PAD/WORD tokens ---
class LiveStreamer(TextStreamer):
    """Prints tokens to stdout as they're generated, skipping streaming control tokens."""
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, skip_special_tokens=True, **kwargs)

# --- Audio capture ---
audio_queue = queue.Queue()
running = threading.Event()
running.set()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio] {status}", file=sys.stderr)
    audio_queue.put(indata[:, 0].copy())

# --- Audio chunk generator for HF streaming ---
def audio_chunk_generator():
    """Yields audio chunks for the HF streaming API."""
    buffer = np.array([], dtype=np.float32)

    # First chunk (larger - includes delay)
    print(f"Buffering first chunk ({first_chunk_samples/SAMPLE_RATE*1000:.0f}ms)...")
    while len(buffer) < first_chunk_samples and running.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
            buffer = np.concatenate([buffer, chunk])
        except queue.Empty:
            continue

    if not running.is_set():
        return

    yield buffer[:first_chunk_samples].copy()
    buffer = buffer[first_chunk_samples:]

    # Subsequent chunks
    while running.is_set():
        while len(buffer) < chunk_samples and running.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, chunk])
            except queue.Empty:
                continue

        if not running.is_set():
            break

        yield buffer[:chunk_samples].copy()
        buffer = buffer[chunk_samples:]

# --- Main ---
print(f"\nOpening mic at {SAMPLE_RATE}Hz...")
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback,
    blocksize=int(SAMPLE_RATE * 0.01),  # 10ms blocks
)

streamer = LiveStreamer(processor.tokenizer)

with stream:
    print("\n=== HF VoxtralRealtime Streaming Mode ===")
    print("Press Ctrl+C to stop.\n")

    chunk_gen = audio_chunk_generator()

    try:
        # Process first chunk
        first_audio = next(chunk_gen)
        inputs = processor(
            first_audio,
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
        )
        inputs["num_delay_tokens"] = DELAY_TOKENS
        inputs = inputs.to(model.device, dtype=model.dtype)

        # Wrap remaining chunks as input_features generator
        def feature_generator():
            for audio_chunk in chunk_gen:
                chunk_inputs = processor(
                    audio_chunk,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                )
                yield chunk_inputs["input_features"].to(model.device, dtype=model.dtype)

        inputs["input_features"] = feature_generator()

        # Generate with live token streaming
        output_ids = model.generate(
            **inputs,
            use_cache=True,
            streamer=streamer,
        )

    except KeyboardInterrupt:
        running.clear()
        print("\n\n--- Stopped ---")
