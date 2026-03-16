"""Benchmark HF Voxtral Realtime — clean generation, no hooks."""
import time
import torch
import soundfile as sf

MODEL_DIR = "Voxtral-Mini-4B-Realtime"
WAV_PATH = f"{MODEL_DIR}/test01_16khz_3.7s.wav"

print("Loading model...")
t0 = time.time()
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()
load_time = time.time() - t0
print(f"Model load: {load_time:.2f}s")

audio, sr = sf.read(WAV_PATH, dtype="float32")
audio_secs = len(audio) / sr
print(f"Audio: {len(audio)} samples, {sr} Hz, {audio_secs:.2f}s")

inputs = processor(audio, is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
input_features = inputs["input_features"].to("cuda", dtype=torch.bfloat16)
num_delay_tokens = inputs.get("num_delay_tokens", 6)
prefill_len = input_ids.shape[1]

# Warmup run
print("Warmup run...")
with torch.no_grad():
    _ = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        input_features=input_features, num_delay_tokens=num_delay_tokens,
        do_sample=False, max_new_tokens=200,
    )
torch.cuda.synchronize()

# Timed runs
N_RUNS = 5
print(f"\nBenchmarking {N_RUNS} runs...")
times = []
for i in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, num_delay_tokens=num_delay_tokens,
            do_sample=False, max_new_tokens=200,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    times.append(elapsed)

    generated = output_ids[0][prefill_len:]
    n_tokens = len(generated)
    print(f"  Run {i+1}: {n_tokens} tokens in {elapsed:.3f}s ({n_tokens/elapsed:.1f} tok/s)")

# Results
avg = sum(times) / len(times)
best = min(times)
decoded = processor.decode(output_ids[0][prefill_len:], skip_special_tokens=False)
n_tokens = len(output_ids[0][prefill_len:])

print(f"\n=== Results ===")
print(f"Output: '{decoded}'")
print(f"Tokens: {n_tokens}")
print(f"Avg: {avg:.3f}s ({n_tokens/avg:.1f} tok/s, {avg/audio_secs:.2f}x real-time)")
print(f"Best: {best:.3f}s ({n_tokens/best:.1f} tok/s, {best/audio_secs:.2f}x real-time)")
print(f"Model load: {load_time:.2f}s")
