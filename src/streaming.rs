// Streaming mic capture + real-time inference pipeline
//
// Architecture: cpal audio callback -> mpsc channel -> main thread inference loop
// Timing: 80ms of audio (1280 samples, 8 mel frames) = 1 decoder token
//
// Phase 3 additions:
// - StreamConfig for runtime-configurable parameters
// - State machine: Ready → Active ↔ Paused (hotkey mode)
// - Rolling prebuffer so speech before hotkey press isn't lost
// - OutputSink for stdout vs keyboard injection

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::adapter::Adapter;
use crate::common;
use crate::decoder::{self, TextDecoder};
use crate::encoder::{self, AudioEncoder};
use crate::hotkey;
use crate::mel::{self, IncrementalMel};
use crate::tokenizer::{self, Tokenizer};

/// Runtime-configurable streaming parameters.
pub struct StreamConfig {
    pub delay_tokens: usize,
    pub silence_threshold: f32,
    pub silence_chunks: usize,
    pub min_speech_chunks: usize,
    pub rms_ema_alpha: f32,
    pub hotkey: Option<rdev::Key>,
    pub type_mode: bool,
}

/// PCM samples per decoder token: MEL_FRAMES_PER_TOKEN × HOP_LENGTH.
const SAMPLES_PER_TOKEN: usize = common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;


/// Debug status line at top of terminal (row 1). Uses ANSI save/restore cursor.
fn debug_status(rms: f32, smooth: f32, config: &StreamConfig, silence_ctr: usize, speech_ctr: usize, emitted: bool) {
    let w = 30;
    let max = 0.03f32;
    let level = ((smooth / max).min(1.0) * w as f32) as usize;
    let tp = ((config.silence_threshold / max).min(1.0) * w as f32) as usize;
    let mut bar = String::with_capacity(w);
    for i in 0..w {
        if i == tp { bar.push('|'); }
        else if i < level { bar.push('\u{2588}'); }
        else { bar.push('\u{2591}'); }
    }
    eprint!(
        "\x1b[s\x1b[1;1H\x1b[K [{}] raw={:.4} ema={:.4} sil={}/{} spk={}/{} em={}\x1b[u",
        bar, rms, smooth, silence_ctr, config.silence_chunks,
        speech_ctr, config.min_speech_chunks, emitted,
    );
}

// Incremental conv stem: keep 4 mel frames as context between iterations.
const CONV_CTX: usize = 4;
const CONV_SKIP: usize = 2;
const NEW_MEL_PER_CHUNK: usize = common::MEL_FRAMES_PER_TOKEN;

/// Output destination for transcribed text.
enum OutputSink {
    Stdout,
    Keyboard(enigo::Enigo),
}

impl OutputSink {
    fn emit_text(&mut self, text: &[u8]) {
        match self {
            OutputSink::Stdout => {
                let _ = std::io::stdout().write_all(text);
                let _ = std::io::stdout().flush();
            }
            OutputSink::Keyboard(enigo) => {
                use enigo::Keyboard;
                if let Ok(s) = std::str::from_utf8(text) {
                    let _ = enigo.text(s);
                }
            }
        }
    }

    fn emit_newline(&mut self) {
        match self {
            OutputSink::Stdout => {
                print!("\n\n");
                let _ = std::io::stdout().flush();
            }
            OutputSink::Keyboard(enigo) => {
                use enigo::{Direction, Key, Keyboard};
                let _ = enigo.key(Key::Shift, Direction::Press);
                let _ = enigo.key(Key::Return, Direction::Click);
                let _ = enigo.key(Key::Return, Direction::Click);
                let _ = enigo.key(Key::Shift, Direction::Release);
            }
        }
    }
}

/// Startup result passed to the streaming loop.
struct StartupState {
    last_token: u32,
    mel_frames: Vec<[f32; mel::N_MELS]>,
}

/// Process [silence + delay_audio] through batch mel → enc → adapter → prefill.
fn run_startup(
    delay_samples: &[f32],
    delay_tokens: usize,
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
    sink: &mut OutputSink,
) -> Result<StartupState> {
    let left_pad_samples = common::LEFT_PAD_TOKENS * common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;

    let mut padded = vec![0.0f32; left_pad_samples];
    padded.extend_from_slice(delay_samples);

    let mel_data = mel::log_mel_spectrogram(&padded, filters);
    let mel_time = mel_data.len() / mel::N_MELS;

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

    let enc_out = enc.forward(&mel_tensor)?;
    let total_enc_frames = enc_out.dim(1)?;

    let adapter_out = adapter.forward(&enc_out)?;
    let n_adapter = adapter_out.dim(1)?;

    let t_cond = decoder::sinusoidal_embedding(delay_tokens as f32, device, dtype)?;
    let prefill_len = decoder::prefill_len(delay_tokens);

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, device, dtype)?;

    dec.reset_caches();
    dec.precompute_t_cond(&t_cond)?;
    let logits = dec.forward(&prefill_embeds)?;
    let mut last_token = common::argmax_last(&logits)?;
    emit_token(last_token, tok, sink);

    for pos in prefill_len..n_adapter {
        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = adapter_out.narrow(1, pos, 1)?;
        let fused = tok_embed.add(&audio_frame)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;
        emit_token(next_token, tok, sink);
        last_token = next_token;
        if last_token == tokenizer::EOS_ID { break; }
    }

    eprintln!("Startup: {} mel frames, {} encoder frames, {} adapter frames",
        mel_time, total_enc_frames, n_adapter);

    Ok(StartupState { last_token, mel_frames })
}

/// The core streaming processing loop.
fn run_processing_loop(
    enc: &mut AudioEncoder,
    adapter: &Adapter,
    dec: &mut TextDecoder,
    tok: &Tokenizer,
    state: &mut StartupState,
    mel_buffer: &mut Vec<[f32; mel::N_MELS]>,
    device: &Device,
    dtype: DType,
    sink: &mut OutputSink,
) -> Result<bool> {
    while mel_buffer.len() >= CONV_CTX + NEW_MEL_PER_CHUNK {
        let process_len = CONV_CTX + NEW_MEL_PER_CHUNK;

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

        let new_conv_frames = conv_out.narrow(1, CONV_SKIP, encoder::CHUNK_SIZE)?;
        mel_buffer.drain(..NEW_MEL_PER_CHUNK);

        let enc_out = enc.forward_chunk(&new_conv_frames)?;
        enc.trim_caches();

        let adapter_out = adapter.forward(&enc_out)?;

        let tok_embed = dec.embed_tokens(&[state.last_token], device)?;
        let fused = tok_embed.add(&adapter_out)?;
        let logits = dec.forward(&fused)?;
        let next_token = common::argmax_last(&logits)?;

        emit_token(next_token, tok, sink);
        dec.trim_caches();

        state.last_token = next_token;
        if state.last_token == tokenizer::EOS_ID {
            return Ok(true);
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
    config: &StreamConfig,
) -> Result<()> {
    let delay_samples_count = (1 + config.delay_tokens) * SAMPLES_PER_TOKEN;
    let hotkey_mode = config.hotkey.is_some();

    // Initialize output sink
    let mut sink = if config.type_mode {
        OutputSink::Keyboard(enigo::Enigo::new(&enigo::Settings::default())
            .map_err(|e| anyhow::anyhow!("Failed to initialize keyboard output: {}", e))?)
    } else {
        OutputSink::Stdout
    };

    let running = Arc::new(AtomicBool::new(true));

    // Open mic
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (_stream, native_rate) = open_mic(tx)?;

    // Resample state
    let resample_ratio = native_rate as f64 / 16000.0;
    let need_resample = (resample_ratio - 1.0).abs() > 0.01;
    let ratio_int = resample_ratio as usize;
    let mut raw_buf: Vec<f32> = Vec::new();

    if hotkey_mode {
        // ---- Hotkey mode: startup with silence, toggle with hotkey ----
        let silence = vec![0.0f32; delay_samples_count];
        eprintln!("Running startup with silence ({} samples)...", silence.len());
        let mut state = run_startup(
            &silence, config.delay_tokens, enc, adapter, dec, tok, filters, device, dtype, &mut sink,
        )?;

        // Set up hotkey state machine + Ctrl+C listener
        let hotkey_state = Arc::new(AtomicU8::new(hotkey::STATE_READY));
        let hotkey_key = config.hotkey.unwrap();
        hotkey::spawn_listener(running.clone(), Some(hotkey_key), Some(hotkey_state.clone()));

        eprintln!("Press {:?} to start/stop recording. Ctrl+C to quit.", hotkey_key);

        // Keep only last CONV_CTX mel frames as conv stem context
        let mel_len = state.mel_frames.len();
        let ctx_start = mel_len.saturating_sub(CONV_CTX);
        let mut mel_buffer: Vec<[f32; mel::N_MELS]> = state.mel_frames[ctx_start..].to_vec();
        drop(std::mem::take(&mut state.mel_frames));

        // Fresh IncrementalMel — will apply reflect padding on first real audio push
        let mut inc_mel = IncrementalMel::new(filters);

        // Rolling prebuffer: captures audio while paused so speech just before
        // pressing the hotkey isn't lost
        let mut prebuffer: VecDeque<f32> = VecDeque::with_capacity(delay_samples_count);

        let mut prev_state_val = hotkey::STATE_READY;

        while running.load(Ordering::SeqCst) {
            let current_state_val = hotkey_state.load(Ordering::SeqCst);

            // Pull audio from channel
            let mut new_16k = Vec::new();
            match rx.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(chunk) => {
                    raw_buf.extend_from_slice(&chunk);
                    while let Ok(more) = rx.try_recv() {
                        raw_buf.extend_from_slice(&more);
                    }
                    resample(&mut raw_buf, &mut new_16k, need_resample, ratio_int);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }

            if current_state_val == hotkey::STATE_READY || current_state_val == hotkey::STATE_PAUSED {
                // ---- Ready/Paused: feed rolling prebuffer, don't process ----

                // Detect Active → Paused transition: flush decoder's lookahead
                if prev_state_val == hotkey::STATE_ACTIVE {
                    // Feed any final real audio still in new_16k
                    if !new_16k.is_empty() {
                        inc_mel.push_samples(&new_16k);
                        new_16k.clear();
                    }

                    // Push (delay_tokens + 4) tokens of silence to flush the
                    // decoder's lookahead buffer. The decoder will emit PAD tokens
                    // for the silence, which emit_token ignores.
                    let flush_samples = (config.delay_tokens + 4) * SAMPLES_PER_TOKEN;
                    let silence = vec![0.0f32; flush_samples];
                    inc_mel.push_samples(&silence);

                    // Drain all mel frames and run processing to completion
                    mel_buffer.extend(inc_mel.drain_frames());
                    let _ = run_processing_loop(
                        enc, adapter, dec, tok, &mut state,
                        &mut mel_buffer, device, dtype, &mut sink,
                    )?;

                    // Now truncate and go idle
                    if mel_buffer.len() > CONV_CTX {
                        mel_buffer.drain(..mel_buffer.len() - CONV_CTX);
                    }
                    sink.emit_newline();
                }

                // Feed prebuffer (capped at delay_samples_count)
                for &s in &new_16k {
                    if prebuffer.len() >= delay_samples_count {
                        prebuffer.pop_front();
                    }
                    prebuffer.push_back(s);
                }
            } else {
                // ---- Active: process audio ----
                let just_resumed = prev_state_val != hotkey::STATE_ACTIVE;

                if just_resumed {
                    let prebuf_samples: Vec<f32> = prebuffer.drain(..).collect();
                    if !prebuf_samples.is_empty() {
                        inc_mel.push_samples(&prebuf_samples);
                    }
                }

                if !new_16k.is_empty() {
                    inc_mel.push_samples(&new_16k);
                }

                mel_buffer.extend(inc_mel.drain_frames());

                let eos = run_processing_loop(enc, adapter, dec, tok, &mut state,
                    &mut mel_buffer, device, dtype, &mut sink)?;
                if eos { break; }
            }

            prev_state_val = current_state_val;
        }
    } else {
        // ---- Always-on mode: original behavior ----
        hotkey::spawn_listener(running.clone(), None, None);
        let mut samples_16k: Vec<f32> = Vec::new();
        eprintln!("Buffering {:.0}ms of audio for startup...", delay_samples_count as f64 / 16.0);

        while samples_16k.len() < delay_samples_count && running.load(Ordering::SeqCst) {
            match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            }
            while let Ok(more) = rx.try_recv() {
                raw_buf.extend_from_slice(&more);
            }
            resample(&mut raw_buf, &mut samples_16k, need_resample, ratio_int);
        }

        if samples_16k.len() < delay_samples_count {
            anyhow::bail!("Not enough audio for startup");
        }

        let delay = samples_16k[..delay_samples_count].to_vec();
        let leftover = samples_16k[delay_samples_count..].to_vec();

        eprintln!("Running startup with {} delay samples...", delay.len());
        let mut state = run_startup(
            &delay, config.delay_tokens, enc, adapter, dec, tok, filters, device, dtype, &mut sink,
        )?;

        eprintln!("\n--- Listening (Ctrl+C to stop) ---\n");

        let mel_len = state.mel_frames.len();
        let ctx_start = mel_len.saturating_sub(CONV_CTX);
        let mut mel_buffer: Vec<[f32; mel::N_MELS]> = state.mel_frames[ctx_start..].to_vec();
        drop(std::mem::take(&mut state.mel_frames));

        let left_ctx_len = (mel::N_FFT / 2).min(delay.len());
        let left_ctx = &delay[delay.len() - left_ctx_len..];
        let mut inc_mel = IncrementalMel::with_left_context(filters, left_ctx);

        if !leftover.is_empty() {
            inc_mel.push_samples(&leftover);
        }

        let mut silence_counter: usize = 0;
        let mut speech_counter: usize = 0;
        let mut smoothed_rms: f32 = 0.0;
        // Start true so we don't emit a paragraph break before the user speaks.
        let mut silence_emitted = true;
        let mut sample_buf_for_silence: Vec<f32> = leftover;

        while running.load(Ordering::SeqCst) {
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

            mel_buffer.extend(inc_mel.drain_frames());

            while sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
                let rms = (sample_buf_for_silence[..SAMPLES_PER_TOKEN].iter()
                    .map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
                sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN);
                smoothed_rms = config.rms_ema_alpha * rms + (1.0 - config.rms_ema_alpha) * smoothed_rms;
                // Raw RMS for silence detection (responsive to actual quiet)
                if rms < config.silence_threshold {
                    silence_counter += 1;
                } else {
                    silence_counter = 0;
                }
                // Smoothed RMS for speech detection (rides over inter-syllable dips)
                if smoothed_rms >= config.silence_threshold {
                    speech_counter += 1;
                    if speech_counter >= config.min_speech_chunks {
                        silence_emitted = false;
                    }
                } else {
                    speech_counter = 0;
                }
                debug_status(rms, smoothed_rms, config, silence_counter, speech_counter, silence_emitted);
            }

            let eos = run_processing_loop(enc, adapter, dec, tok, &mut state,
                &mut mel_buffer, device, dtype, &mut sink)?;
            if eos { break; }

            if silence_counter >= config.silence_chunks && !silence_emitted {
                sink.emit_newline();
                silence_emitted = true;
            }
        }
    }

    eprintln!("\n--- Stopped ---");
    Ok(())
}

// ---- Helpers ----

fn resample(raw_buf: &mut Vec<f32>, out: &mut Vec<f32>, need_resample: bool, ratio_int: usize) {
    if !need_resample {
        out.extend(raw_buf.drain(..));
        return;
    }
    let n_out = raw_buf.len() / ratio_int;
    out.reserve(n_out);
    for i in 0..n_out {
        out.push(raw_buf[i * ratio_int]);
    }
    raw_buf.drain(..n_out * ratio_int);
}

fn emit_token(token: u32, tok: &Tokenizer, sink: &mut OutputSink) {
    if token == tokenizer::STREAMING_PAD_ID { return; }
    if token == tokenizer::STREAMING_WORD_ID { return; }
    if token == tokenizer::EOS_ID { return; }
    if let Some(bytes) = tok.decode_token(token) {
        sink.emit_text(&bytes);
    }
}

fn open_mic(tx: mpsc::Sender<Vec<f32>>) -> Result<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let input_device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    eprintln!("Input device: {}", input_device.name().unwrap_or_default());

    let default_config = input_device.default_input_config()?;
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    eprintln!("Native config: {}Hz, {} ch", native_rate, native_channels);

    // Use native channel count — some devices reject mono requests
    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    if native_rate != 16000 {
        eprintln!("Will resample {}Hz -> 16kHz on inference thread", native_rate);
    }

    let ch = native_channels as usize;
    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if ch == 1 {
                let _ = tx.send(data.to_vec());
            } else {
                // Downmix to mono by averaging all channels per frame
                let mut mono = vec![0.0f32; data.len() / ch];
                let scale = 1.0 / ch as f32;
                for (i, frame) in data.chunks_exact(ch).enumerate() {
                    let mut sum = 0.0f32;
                    for &s in frame {
                        sum += s;
                    }
                    mono[i] = sum * scale;
                }
                let _ = tx.send(mono);
            }
        },
        |err| {
            eprintln!("Audio input error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    Ok((stream, native_rate))
}
