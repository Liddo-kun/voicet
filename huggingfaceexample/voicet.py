"""Voicet: Real-time microphone transcription using Voxtral-Mini-4B."""

import argparse
import queue
import signal
import sys
import threading
import time
import warnings
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, TextIteratorStreamer, VoxtralRealtimeForConditionalGeneration

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def load_model(device: str):
    """Load model and processor, printing progress."""
    print(f"Loading {MODEL_ID} on {device}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    return model, processor


def mic_thread_fn(audio_queue, stop_event):
    """Capture microphone audio and push small buffers to queue."""
    sample_rate = 16000

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                            blocksize=160, callback=callback):  # 10ms blocks
            stop_event.wait()
    except Exception as e:
        print(f"Microphone error: {e}", file=sys.stderr)


def run_session(model, processor, device, stop_event, emit_token, audio_queue, prebuffer=None):
    """Run one transcription session. Returns transcript string.

    audio_queue: queue receiving mic audio buffers (caller manages the mic)
    prebuffer: optional pre-captured audio to use for the first chunk
    """
    first_chunk_samples = processor.num_samples_first_audio_chunk
    chunk_samples = processor.num_samples_per_audio_chunk
    alt = processor.audio_length_per_tok
    ndt = processor.num_delay_tokens

    # Get first chunk audio — use prebuffer if we have enough, otherwise collect more
    if prebuffer is not None and len(prebuffer) >= first_chunk_samples:
        first_audio = prebuffer[-first_chunk_samples:]
        leftover = np.array([], dtype=np.float32)
    else:
        collected = prebuffer.copy() if prebuffer is not None else np.array([], dtype=np.float32)
        while not stop_event.is_set():
            try:
                buf = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            collected = np.concatenate([collected, buf])
            if len(collected) >= first_chunk_samples:
                break

        if stop_event.is_set() and len(collected) < first_chunk_samples:
            return ""

        first_audio = collected[:first_chunk_samples]
        leftover = collected[first_chunk_samples:]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Passing audio with", category=FutureWarning)
        first_inputs = processor(
            first_audio,
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
        )

    input_ids = first_inputs["input_ids"].to(device)
    attention_mask = first_inputs["attention_mask"].to(device)
    num_delay_tokens = first_inputs["num_delay_tokens"]
    first_features = first_inputs["input_features"].to(device, dtype=torch.bfloat16)

    def streaming_features():
        # Yield first chunk features frame by frame
        num_frames = first_features.shape[-1]
        for i in range(0, num_frames, alt):
            yield first_features[:, :, i : i + alt]

        # Then yield subsequent chunks
        buf = leftover.copy()
        while not stop_event.is_set() or len(buf) >= chunk_samples:
            while len(buf) < chunk_samples:
                if stop_event.is_set():
                    break
                try:
                    audio = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, audio])

            if len(buf) < chunk_samples:
                break

            chunk_audio = buf[:chunk_samples]
            buf = buf[chunk_samples:]

            chunk_inputs = processor(
                chunk_audio,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            chunk_feats = chunk_inputs["input_features"].to(device, dtype=torch.bfloat16)
            num_f = chunk_feats.shape[-1]
            for i in range(0, num_f, alt):
                yield chunk_feats[:, :, i : i + alt]

        # Flush the delay buffer by feeding silence chunks
        silence = np.zeros(chunk_samples, dtype=np.float32)
        for _ in range(ndt):
            chunk_inputs = processor(
                silence,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            chunk_feats = chunk_inputs["input_features"].to(device, dtype=torch.bfloat16)
            num_f = chunk_feats.shape[-1]
            for i in range(0, num_f, alt):
                yield chunk_feats[:, :, i : i + alt]

    # Set up text streamer for real-time output
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=streaming_features(),
        num_delay_tokens=num_delay_tokens,
        do_sample=False,
        streamer=streamer,
    )

    warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

    gen_thread = threading.Thread(target=lambda: model.generate(**gen_kwargs), daemon=True)
    gen_thread.start()

    transcript = []
    for text in streamer:
        if text:
            transcript.append(text)
            emit_token(text)

    gen_thread.join(timeout=10)

    return "".join(transcript)


def _print_model_info(processor, model, device):
    """Print model parameters once at startup."""
    sample_rate = processor.feature_extractor.sampling_rate
    first_chunk_samples = processor.num_samples_first_audio_chunk
    chunk_samples = processor.num_samples_per_audio_chunk
    alt = processor.audio_length_per_tok
    ndt = processor.num_delay_tokens
    ms_per_tok = alt * processor.feature_extractor.hop_length / sample_rate * 1000
    chunk_ms = chunk_samples / sample_rate * 1000
    prefill_ms = first_chunk_samples / sample_rate * 1000

    print(f"  sample_rate:            {sample_rate} Hz")
    print(f"  blocksize:              160 samples ({160 / sample_rate * 1000:.0f}ms)")
    print(f"  num_delay_tokens:       {ndt} x {ms_per_tok:.0f}ms = {ndt * ms_per_tok:.0f}ms")
    print(f"  first_chunk_samples:    {first_chunk_samples} ({prefill_ms:.0f}ms)")
    print(f"  chunk_samples:          {chunk_samples} ({chunk_ms:.0f}ms)")
    print(f"  audio_length_per_tok:   {alt}")
    print(f"  dtype:                  {model.dtype}")
    print(f"  device:                 {device}")


def _run_single_session(model, processor, device, emit):
    """Original single-session mode (Ctrl+C to stop)."""
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    mic = threading.Thread(target=mic_thread_fn, args=(audio_queue, stop_event), daemon=True)
    mic.start()

    def on_sigint(sig, frame):
        print("\nFlushing...", flush=True)
        stop_event.set()
    signal.signal(signal.SIGINT, on_sigint)

    print(f"\nListening... Press Ctrl+C to stop.\n")

    transcript = run_session(model, processor, device, stop_event, emit, audio_queue)
    mic.join(timeout=2)

    if transcript:
        print("\n\n--- Full Transcript ---")
        print(transcript)


def _run_hotkey_mode(model, processor, device, hotkey, emit):
    """Hotkey-toggled mode: press hotkey to start/stop recording sessions.

    Mic stays open permanently. A rolling buffer of recent audio means the first
    chunk is already available the instant the hotkey is pressed — no startup delay.
    """
    from pynput import keyboard

    first_chunk_samples = processor.num_samples_first_audio_chunk

    # Persistent mic — stays open for the lifetime of the program
    audio_queue = queue.Queue()
    stop_mic = threading.Event()
    mic = threading.Thread(target=mic_thread_fn, args=(audio_queue, stop_mic), daemon=True)
    mic.start()

    hotkey_queue = queue.Queue()
    exit_event = threading.Event()
    active_stop = [None]

    def on_hotkey():
        hotkey_queue.put("toggle")

    hotkey_listener = keyboard.GlobalHotKeys({hotkey: on_hotkey})
    hotkey_listener.daemon = True
    hotkey_listener.start()

    def on_sigint(sig, frame):
        exit_event.set()
        stop_mic.set()
        if active_stop[0]:
            active_stop[0].set()
        hotkey_queue.put("exit")
    signal.signal(signal.SIGINT, on_sigint)

    display = hotkey.replace("<", "").replace(">", "").title()
    print(f"\nReady. Press {display} to start recording. Ctrl+C to quit.\n")

    while not exit_event.is_set():
        # Drain mic audio into a rolling buffer while waiting for hotkey
        recent = deque()
        recent_len = 0

        while not exit_event.is_set():
            # Check for hotkey (non-blocking)
            try:
                hotkey_queue.get_nowait()
                break
            except queue.Empty:
                pass

            # Drain audio into rolling buffer, keep last first_chunk_samples
            try:
                buf = audio_queue.get(timeout=0.05)
                recent.append(buf)
                recent_len += len(buf)
                while recent_len - len(recent[0]) >= first_chunk_samples:
                    recent_len -= len(recent.popleft())
            except queue.Empty:
                pass

        if exit_event.is_set():
            break

        prebuffer = np.concatenate(list(recent)) if recent else np.array([], dtype=np.float32)

        print("[Recording...]", flush=True)
        stop_event = threading.Event()
        active_stop[0] = stop_event

        # Background thread watches for the next hotkey press to stop the session
        def wait_for_stop():
            while not exit_event.is_set():
                try:
                    hotkey_queue.get(timeout=0.1)
                    print("\nFlushing...", flush=True)
                    stop_event.set()
                    return
                except queue.Empty:
                    continue
            stop_event.set()

        stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
        stop_thread.start()

        transcript = run_session(model, processor, device, stop_event, emit, audio_queue, prebuffer)
        active_stop[0] = None
        stop_thread.join(timeout=1)

        if transcript:
            print(f"\n--- Transcript ---\n{transcript}\n")

        if not exit_event.is_set():
            print(f"Ready. Press {display} to start recording.\n")

    stop_mic.set()
    mic.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="Real-time microphone transcription with Voxtral-Mini-4B")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--type", action="store_true",
                        help="Type transcription into the active window instead of printing to stdout")
    parser.add_argument("--hotkey", nargs="?", const="<ctrl>+<space>", default=None, metavar="COMBO",
                        help="Toggle recording with a hotkey (default: Ctrl+Space). "
                             "Uses pynput format, e.g. '<ctrl>+<space>', '<alt>+s'")
    args = parser.parse_args()

    device = args.device
    model, processor = load_model(device)
    _print_model_info(processor, model, device)

    if args.type:
        from pynput.keyboard import Controller
        kb = Controller()
        emit = lambda text: kb.type(text)
    else:
        emit = lambda text: print(text, end="", flush=True)

    if args.hotkey:
        _run_hotkey_mode(model, processor, device, args.hotkey, emit)
    else:
        _run_single_session(model, processor, device, emit)


if __name__ == "__main__":
    main()
