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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::adapter::Adapter;
use crate::common;
use crate::decoder::{self, TextDecoder};
use crate::encoder::{self, AudioEncoder};
use crate::hotkey;
use crate::mel::{self, IncrementalMel};
use crate::settings::SharedSettings;
use crate::tokenizer::{self, Tokenizer};

/// PCM samples per decoder token: MEL_FRAMES_PER_TOKEN × HOP_LENGTH.
const SAMPLES_PER_TOKEN: usize = common::MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;


/// Two-stage silence detection and paragraph break state machine.
///
/// Stage 1 — Silence detection: counts consecutive silent chunks. Once the count
///   reaches `silence_chunks`, silence is "detected". Only arms after speech has
///   occurred (`silence_emitted == false`).
///
/// Stage 2 — Paragraph delay: once silence is detected, a separate counter ticks
///   every chunk (even if speech resumes). When it reaches `delay_tokens + offset`,
///   the paragraph break fires. Both stages then reset.
struct SilenceDetector {
    silence_counter: usize,
    speech_counter: usize,
    smoothed_rms: f32,
    silence_emitted: bool,
    silence_detected: bool,
    paragraph_delay_counter: usize,
}

impl SilenceDetector {
    fn new() -> Self {
        Self {
            silence_counter: 0,
            speech_counter: 0,
            smoothed_rms: 0.0,
            silence_emitted: true,
            silence_detected: false,
            paragraph_delay_counter: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// Process one 80ms chunk of audio. Returns true if a paragraph break should fire.
    fn process_chunk(&mut self, rms: f32, settings: &SharedSettings) -> bool {
        let sil_thresh = settings.silence_threshold.load(Ordering::Relaxed);
        let rms_alpha = settings.rms_ema_alpha.load(Ordering::Relaxed);
        self.smoothed_rms = rms_alpha * rms + (1.0 - rms_alpha) * self.smoothed_rms;

        if settings.silence_chunks.load(Ordering::Relaxed) == 0 {
            return false;
        }

        // Silence counting
        if rms < sil_thresh {
            self.silence_counter += 1;
            // Stage 1: detect silence (only after speech has occurred)
            if !self.silence_detected && !self.silence_emitted
                && self.silence_counter >= settings.silence_chunks.load(Ordering::Relaxed)
            {
                self.silence_detected = true;
                self.paragraph_delay_counter = 0;
            }
        } else {
            self.silence_counter = 0;
        }

        // Stage 2: paragraph delay ticks unconditionally once triggered
        if self.silence_detected {
            self.paragraph_delay_counter += 1;
        }

        // Speech tracking — arms silence detection after enough speech
        if self.smoothed_rms >= sil_thresh {
            self.speech_counter += 1;
            if self.speech_counter >= settings.min_speech_chunks.load(Ordering::Relaxed) {
                self.silence_emitted = false;
            }
        } else {
            self.speech_counter = 0;
        }

        // Check if paragraph break should fire
        if self.silence_detected && !self.silence_emitted {
            if self.paragraph_delay_counter >= Self::para_delay(settings) {
                self.silence_emitted = true;
                self.silence_detected = false;
                self.paragraph_delay_counter = 0;
                return true;
            }
        }
        false
    }

    fn para_delay(settings: &SharedSettings) -> usize {
        settings.delay_tokens.load(Ordering::Relaxed)
            + settings.paragraph_delay_offset.load(Ordering::Relaxed)
    }

    fn debug_status(&self, rms: f32, settings: &SharedSettings) {
        let w = 30;
        let max = 0.03f32;
        let level = ((self.smoothed_rms / max).min(1.0) * w as f32) as usize;
        let threshold = settings.silence_threshold.load(Ordering::Relaxed);
        let tp = ((threshold / max).min(1.0) * w as f32) as usize;
        let mut bar = String::with_capacity(w);
        for i in 0..w {
            if i == tp { bar.push('|'); }
            else if i < level { bar.push('\u{2588}'); }
            else { bar.push('\u{2591}'); }
        }
        let sil_chunks = settings.silence_chunks.load(Ordering::Relaxed);
        let para_delay = Self::para_delay(settings);
        eprint!(
            "\x1b[s\x1b[1;1H\x1b[K [{}] raw={:.4} ema={:.4} sil={}/{} det={} pd={}/{} spk={}/{} em={}\x1b[u",
            bar, rms, self.smoothed_rms,
            self.silence_counter, sil_chunks, self.silence_detected,
            self.paragraph_delay_counter, para_delay,
            self.speech_counter, settings.min_speech_chunks.load(Ordering::Relaxed), self.silence_emitted,
        );
    }
}

// Incremental conv stem: keep 4 mel frames as context between iterations.
const CONV_CTX: usize = 4;
const CONV_SKIP: usize = 2;
const NEW_MEL_PER_CHUNK: usize = common::MEL_FRAMES_PER_TOKEN;

/// Output destination for transcribed text.
enum OutputSink {
    Stdout,
    Keyboard(enigo::Enigo),
    Discard,
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
            OutputSink::Discard => {}
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
            OutputSink::Discard => {}
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

        // let t0 = std::time::Instant::now();

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
    settings: &Arc<SharedSettings>,
    running: &Arc<AtomicBool>,
    hotkey_thread_id: &Arc<AtomicU32>,
) -> Result<()> {
    let delay = settings.delay_tokens.load(Ordering::Relaxed);
    let delay_samples_count = (1 + delay) * SAMPLES_PER_TOKEN;

    // Initialize output sink
    let mut sink = if settings.type_mode.load(Ordering::Relaxed) {
        OutputSink::Keyboard(enigo::Enigo::new(&enigo::Settings::default())
            .map_err(|e| anyhow::anyhow!("Failed to initialize keyboard output: {}", e))?)
    } else {
        OutputSink::Stdout
    };

    // Open mic
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (_stream, native_rate) = open_mic(tx)?;

    // Resample state
    let resample_ratio = native_rate as f64 / 16000.0;
    let need_resample = (resample_ratio - 1.0).abs() > 0.01;
    let ratio_int = resample_ratio as usize;
    let mut raw_buf: Vec<f32> = Vec::new();

    // Startup with silence — model is ready, real audio goes through incremental path
    let silence = vec![0.0f32; delay_samples_count];
    eprintln!("Running startup with silence ({} samples)...", silence.len());
    let mut state = run_startup(
        &silence, delay, enc, adapter, dec, tok, filters, device, dtype, &mut sink,
    )?;

    // Spawn hotkey listener
    hotkey::spawn_listener(running.clone(), settings.clone(), hotkey_thread_id.clone());

    let hotkey_key = settings.hotkey.lock().unwrap().clone();
    if let Some(key) = hotkey_key {
        eprintln!("Press {:?} to start/stop recording. Ctrl+C to quit.", key);
    } else {
        eprintln!("\n--- Listening (Ctrl+C to stop) ---\n");
    }

    // Keep only last CONV_CTX mel frames as conv stem context
    let mel_len = state.mel_frames.len();
    let ctx_start = mel_len.saturating_sub(CONV_CTX);
    let mut mel_buffer: Vec<[f32; mel::N_MELS]> = state.mel_frames[ctx_start..].to_vec();
    drop(std::mem::take(&mut state.mel_frames));

    let mut inc_mel = IncrementalMel::new(filters);
    let mut prebuffer: VecDeque<f32> = VecDeque::with_capacity(delay_samples_count);
    let mut local_delay = delay;

    let mut sil = SilenceDetector::new();
    let mut sample_buf_for_silence: Vec<f32> = Vec::new();

    let mut prev_state_val = settings.state.load(Ordering::SeqCst);

    while running.load(Ordering::SeqCst) {
        let current_state_val = settings.state.load(Ordering::SeqCst);

        // Pull audio from channel
        let mut new_16k = Vec::new();
        match rx.recv_timeout(std::time::Duration::from_secs(2)) {
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

        // Check for delay change — restart model with new delay
        let new_delay = settings.delay_tokens.load(Ordering::SeqCst);
        if new_delay != local_delay {
            local_delay = new_delay;
            let silence = vec![0.0f32; (1 + local_delay) * SAMPLES_PER_TOKEN];
            state = run_startup(
                &silence, local_delay, enc, adapter, dec, tok, filters, device, dtype, &mut sink,
            )?;
            let mel_len = state.mel_frames.len();
            mel_buffer = state.mel_frames[mel_len.saturating_sub(CONV_CTX)..].to_vec();
            drop(std::mem::take(&mut state.mel_frames));
            inc_mel = IncrementalMel::new(filters);
            prev_state_val = current_state_val;
            continue;
        }

        // Output mode swap
        let want_type = settings.type_mode.load(Ordering::Relaxed);
        if want_type != matches!(sink, OutputSink::Keyboard(_)) {
            sink = if want_type {
                OutputSink::Keyboard(enigo::Enigo::new(&enigo::Settings::default())
                    .map_err(|e| anyhow::anyhow!("Failed to initialize keyboard output: {}", e))?)
            } else {
                OutputSink::Discard
            };
        }

        if current_state_val == hotkey::STATE_PAUSED {
            // ---- Paused: feed rolling prebuffer, don't process ----

            // Detect Active → Paused transition: flush decoder's lookahead
            if prev_state_val == hotkey::STATE_ACTIVE {
                if !new_16k.is_empty() {
                    inc_mel.push_samples(&new_16k);
                    new_16k.clear();
                }

                let flush_samples = (local_delay + 4) * SAMPLES_PER_TOKEN;
                let flush_silence = vec![0.0f32; flush_samples];
                inc_mel.push_samples(&flush_silence);

                mel_buffer.extend(inc_mel.drain_frames());
                let _ = run_processing_loop(
                    enc, adapter, dec, tok, &mut state,
                    &mut mel_buffer, device, dtype, &mut sink,
                )?;

                if mel_buffer.len() > CONV_CTX {
                    mel_buffer.drain(..mel_buffer.len() - CONV_CTX);
                }
                sil.reset();
                sample_buf_for_silence.clear();
            }

            // Feed prebuffer (160ms — captures speech just before hotkey press)
            let prebuf_cap = 2 * SAMPLES_PER_TOKEN;
            for &s in &new_16k {
                if prebuffer.len() >= prebuf_cap {
                    prebuffer.pop_front();
                }
                prebuffer.push_back(s);
            }
        } else {
            // ---- Active: process audio ----
            if prev_state_val != hotkey::STATE_ACTIVE {
                // Just resumed: flush prebuffer into mel
                let prebuf_samples: Vec<f32> = prebuffer.drain(..).collect();
                if !prebuf_samples.is_empty() {
                    inc_mel.push_samples(&prebuf_samples);
                }
            }

            if !new_16k.is_empty() {
                inc_mel.push_samples(&new_16k);
                sample_buf_for_silence.extend_from_slice(&new_16k);
            }

            mel_buffer.extend(inc_mel.drain_frames());

            // Silence detection + paragraph break
            while sample_buf_for_silence.len() >= SAMPLES_PER_TOKEN {
                let rms = (sample_buf_for_silence[..SAMPLES_PER_TOKEN].iter()
                    .map(|s| s * s).sum::<f32>() / SAMPLES_PER_TOKEN as f32).sqrt();
                sample_buf_for_silence.drain(..SAMPLES_PER_TOKEN);
                if sil.process_chunk(rms, settings) {
                    sink.emit_newline();
                }
                sil.debug_status(rms, settings);
            }

            let eos = run_processing_loop(enc, adapter, dec, tok, &mut state,
                &mut mel_buffer, device, dtype, &mut sink)?;
            if eos { break; }
        }

        prev_state_val = current_state_val;
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
