// Streaming mic capture + real-time inference pipeline for Voicet Phase 2
//
// Architecture: cpal audio callback -> mpsc channel -> main thread inference loop
// Timing: 80ms of audio (1280 samples, 8 mel frames) = 1 decoder token
//
// Key insight: the startup must process [silence + delay_audio] together through
// the batch pipeline so decoder prefill positions 33–38 see real audio adapter
// frames, matching offline behavior exactly.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;


use crate::adapter::Adapter;
use crate::common;
use crate::decoder::{self, TextDecoder};
use crate::encoder::{self, AudioEncoder};
use crate::mel::{self, IncrementalMel};
use crate::tokenizer::{self, Tokenizer};

pub const SILENCE_THRESHOLD: f32 = 0.01;
pub const SILENCE_CHUNKS: usize = decoder::NUM_DELAY_TOKENS + 6;

/// PCM samples per decoder token: MEL_FRAMES_PER_TOKEN × HOP_LENGTH.
const SAMPLES_PER_TOKEN: usize = common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;

// Incremental conv stem: keep 4 mel frames as context between iterations.
// Conv1 (stride=1, k=3) needs 2 frames of left context; conv2 (stride=2, k=3) needs 1.
// With 4 context frames, the first 2 conv2 outputs are wrong; skip them, keep CHUNK_SIZE.
const CONV_CTX: usize = 4;
const CONV_SKIP: usize = 2;
const NEW_MEL_PER_CHUNK: usize = common::MEL_FRAMES_PER_TOKEN; // 8 mel frames → 4 conv outputs = CHUNK_SIZE

// (1 + NUM_DELAY_TOKENS) adapter frames of real audio needed for prefill.
// Silence covers LEFT_PAD_TOKENS adapter frames; remaining (1 + delay) come from real audio.
const DELAY_SAMPLES: usize = (1 + decoder::NUM_DELAY_TOKENS) * SAMPLES_PER_TOKEN;

/// Startup result passed to the streaming loop.
struct StartupState {
    last_token: u32,
    /// All mel frames from startup [silence + delay], used as conv stem prefix
    mel_frames: Vec<[f32; mel::N_MELS]>,
}

/// Process [silence + delay_audio] through batch mel → enc.forward() → adapter → prefill,
/// exactly matching the offline pipeline. Returns state for the streaming loop.
fn run_startup(
    delay_samples: &[f32],
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<StartupState> {
    let left_pad_samples = common::LEFT_PAD_TOKENS * common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;

    // Concatenate silence + delay audio, just like offline
    let mut padded = vec![0.0f32; left_pad_samples];
    padded.extend_from_slice(delay_samples);

    // Batch mel on the whole thing
    let mel_data = mel::log_mel_spectrogram(&padded, filters);
    let mel_time = mel_data.len() / mel::N_MELS;

    // Save mel frames for conv stem prefix in streaming loop
    let mut mel_frames = Vec::with_capacity(mel_time);
    for t in 0..mel_time {
        let mut frame = [0.0f32; mel::N_MELS];
        for b in 0..mel::N_MELS {
            frame[b] = mel_data[b * mel_time + t];
        }
        mel_frames.push(frame);
    }

    let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, mel_time), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

    // Full encoder (resets KV caches, processes all chunks)
    let enc_out = enc.forward(&mel_tensor)?;
    let total_enc_frames = enc_out.dim(1)?;

    // Adapter
    let adapter_out = adapter.forward(&enc_out)?;
    let n_adapter = adapter_out.dim(1)?;

    // Delay conditioning
    let t_cond = decoder::sinusoidal_embedding(decoder::NUM_DELAY_TOKENS as f32, device, dtype)?;

    // Prefill: BOS + (LEFT_PAD_TOKENS + NUM_DELAY_TOKENS) PAD tokens
    let prefill_embeds = dec.prepare_prefill(&adapter_out, device, dtype)?;

    dec.reset_caches();
    dec.precompute_t_cond(&t_cond)?;
    let logits = dec.forward(&prefill_embeds)?;
    let mut last_token = common::argmax_last(&logits)?;
    emit_token(last_token, tok);

    // Decode any remaining adapter frames beyond prefill (from delay audio)
    for pos in decoder::PREFILL_LEN..n_adapter {
        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&audio_frame)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;
        emit_token(next_token, tok);
        last_token = next_token;
        if last_token == tokenizer::EOS_ID { break; }
    }

    println!("Startup: {} mel frames, {} encoder frames, {} adapter frames",
        mel_time, total_enc_frames, n_adapter);

    Ok(StartupState {
        last_token,
        mel_frames,
    })
}

/// The core streaming processing loop.
/// `mel_buffer` holds CONV_CTX context frames followed by unprocessed new frames.
fn run_processing_loop(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    state: &mut StartupState,
    mel_buffer: &mut Vec<[f32; mel::N_MELS]>,
    device: &Device,
    dtype: DType,
) -> Result<bool> {
    // Process tokens while we have enough new mel frames beyond context
    while mel_buffer.len() >= CONV_CTX + NEW_MEL_PER_CHUNK {
        let process_len = CONV_CTX + NEW_MEL_PER_CHUNK;

        // Build mel tensor from context + new frames only (not full history)
        let mut mel_data = vec![0.0f32; mel::N_MELS * process_len];
        for (frame_idx, frame) in mel_buffer[..process_len].iter().enumerate() {
            for mel_bin in 0..mel::N_MELS {
                mel_data[mel_bin * process_len + frame_idx] = frame[mel_bin];
            }
        }
        let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, process_len), device)?
            .to_dtype(dtype)?
            .unsqueeze(0)?;

        let conv_out = enc.conv_stem(&mel_tensor)?;
        if conv_out.dim(1)? < CONV_SKIP + encoder::CHUNK_SIZE {
            break;
        }

        // Skip first CONV_SKIP outputs (wrong due to zero-padding over context),
        // take CHUNK_SIZE new frames
        let new_conv_frames = conv_out.narrow(1, CONV_SKIP, encoder::CHUNK_SIZE)?;

        // Remove processed mel frames; remaining starts with new context
        mel_buffer.drain(..NEW_MEL_PER_CHUNK);

        // Encoder → trim → adapter → decoder → argmax
        let enc_out = enc.forward_chunk(&new_conv_frames)?;
        enc.trim_caches();

        let adapter_out = adapter.forward(&enc_out)?;

        let tok_embed = dec.embed_tokens(&[state.last_token], device)?;
        let fused = tok_embed.add(&adapter_out)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;

        emit_token(next_token, tok);
        dec.trim_caches();

        state.last_token = next_token;
        if state.last_token == tokenizer::EOS_ID {
            return Ok(true); // EOS
        }
    }
    Ok(false)
}

/// Live streaming from microphone.
pub fn run_streaming(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<()> {
    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    // Open mic BEFORE startup (buffer audio during GPU warm-up)
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (_stream, native_rate) = open_mic(tx)?;

    // Resample state
    let resample_ratio = native_rate as f64 / 16000.0;
    let need_resample = (resample_ratio - 1.0).abs() > 0.01;
    let ratio_int = resample_ratio as usize;

    // Collect enough audio for delay (DELAY_SAMPLES at 16kHz)
    println!("Buffering {:.0}ms of audio for startup...", DELAY_SAMPLES as f64 / 16.0);
    let mut raw_buf: Vec<f32> = Vec::new();
    let mut samples_16k: Vec<f32> = Vec::new();

    while samples_16k.len() < DELAY_SAMPLES && running.load(Ordering::SeqCst) {
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(chunk) => raw_buf.extend_from_slice(&chunk),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        }
        // Drain any extra
        while let Ok(more) = rx.try_recv() {
            raw_buf.extend_from_slice(&more);
        }
        // Resample
        resample(&mut raw_buf, &mut samples_16k, need_resample, ratio_int);
    }

    if samples_16k.len() < DELAY_SAMPLES {
        anyhow::bail!("Not enough audio for startup");
    }

    let delay = samples_16k[..DELAY_SAMPLES].to_vec();
    let leftover = samples_16k[DELAY_SAMPLES..].to_vec();

    println!("Running startup with {} delay samples...", delay.len());
    let mut state = run_startup(&delay, enc, adapter, dec, tok, filters, device, dtype)?;

    println!("\n--- Listening (Ctrl+C to stop) ---\n");

    // Keep only last CONV_CTX mel frames as context for incremental conv stem
    let mel_len = state.mel_frames.len();
    let ctx_start = mel_len.saturating_sub(CONV_CTX);
    let mut mel_buffer: Vec<[f32; mel::N_MELS]> = state.mel_frames[ctx_start..].to_vec();
    drop(std::mem::take(&mut state.mel_frames)); // free startup mel history

    // Seed IncrementalMel with left context from tail of delay audio
    let left_ctx_len = (mel::N_FFT / 2).min(delay.len());
    let left_ctx = &delay[delay.len() - left_ctx_len..];
    let mut inc_mel = IncrementalMel::with_left_context(filters, left_ctx);

    // Feed leftover samples from buffer
    if !leftover.is_empty() {
        inc_mel.push_samples(&leftover);
    }

    let mut silence_counter: usize = 0;
    let mut silence_emitted = false;
    let mut sample_buf_for_silence: Vec<f32> = leftover;

    while running.load(Ordering::SeqCst) {
        // Pull audio from channel
        match rx.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(chunk) => {
                raw_buf.extend_from_slice(&chunk);
                while let Ok(more) = rx.try_recv() {
                    raw_buf.extend_from_slice(&more);
                }
                let mut new_16k = Vec::new();
                resample(&mut raw_buf, &mut new_16k, need_resample, ratio_int);
                if !new_16k.is_empty() {
                    inc_mel.push_samples(&new_16k);
                    sample_buf_for_silence.extend_from_slice(&new_16k);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Drain mel frames
        mel_buffer.extend(inc_mel.drain_frames());

        // Silence detection
        while sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
            let rms = (sample_buf_for_silence[..SAMPLES_PER_TOKEN].iter()
                .map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
            sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN);
            if rms < SILENCE_THRESHOLD {
                silence_counter += 1;
            } else {
                silence_counter = 0;
                silence_emitted = false;
            }
        }

        // Process tokens
        let eos = run_processing_loop(enc, adapter, dec, tok, &mut state,
            &mut mel_buffer, device, dtype)?;
        if eos { break; }

        // Silence flush: emit paragraph break once, then wait for speech to resume
        if silence_counter >= SILENCE_CHUNKS && !silence_emitted {
            print!("\n\n");
            let _ = std::io::stdout().flush();
            silence_emitted = true;
        }
    }

    println!("\n\n--- Stopped ---");
    Ok(())
}

// ---- Helpers ----

fn resample(raw_buf: &mut Vec<f32>, out: &mut Vec<f32>, need_resample: bool, ratio_int: usize) {
    if !need_resample {
        out.extend(raw_buf.drain(..));
        return;
    }
    // Decimation by integer ratio: take every ratio_int-th sample
    let n_out = raw_buf.len() / ratio_int;
    out.reserve(n_out);
    for i in 0..n_out {
        out.push(raw_buf[i * ratio_int]);
    }
    raw_buf.drain(..n_out * ratio_int);
}

fn emit_token(token: u32, tok: &Tokenizer) {
    if token == tokenizer::STREAMING_PAD_ID { return; }
    if token == tokenizer::STREAMING_WORD_ID { return; }
    if token == tokenizer::EOS_ID { return; }
    if let Some(bytes) = tok.decode_token(token) {
        let _ = std::io::stdout().write_all(&bytes);
        let _ = std::io::stdout().flush();
    }
}

fn open_mic(tx: mpsc::Sender<Vec<f32>>) -> Result<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let input_device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    println!("Input device: {}", input_device.name().unwrap_or_default());

    let default_config = input_device.default_input_config()?;
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    println!("Native config: {}Hz, {} ch", native_rate, native_channels);

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    if native_rate != 16000 {
        println!("Will resample {}Hz -> 16kHz on inference thread", native_rate);
    }

    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = tx.send(data.to_vec());
        },
        |err| {
            eprintln!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    Ok((stream, native_rate))
}
