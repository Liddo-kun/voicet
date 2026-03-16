mod adapter;
mod common;
mod decoder;
mod encoder;
mod hotkey;
mod mel;
mod streaming;
mod tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

use common::{argmax_last, MEL_FRAMES_PER_TOKEN};

#[derive(Parser)]
#[command(name = "voicet", about = "Real-time speech transcription")]
struct Cli {
    /// WAV file for offline transcription (omit for streaming mode)
    wav_file: Option<String>,

    /// Directory containing model files (consolidated.safetensors, tekken.json, mel_filters.bin)
    #[arg(long, default_value = ".")]
    model_dir: String,

    /// CUDA device index
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Delay tokens (1-30, higher = more accuracy, more latency; each token = 80ms)
    #[arg(long, default_value_t = 3)]
    delay: usize,

    /// Silence RMS threshold for paragraph breaks
    #[arg(long, default_value_t = 0.007)]
    silence_threshold: f32,

    /// Silence chunks before paragraph break (default: delay + 9)
    #[arg(long)]
    silence_flush: Option<usize>,

    /// Minimum speech chunks before silence detection activates (each chunk = 80ms)
    #[arg(long, default_value_t = 8)]
    min_speech: usize,

    /// EMA smoothing factor for speech detection (0.0-1.0, lower = smoother)
    #[arg(long, default_value_t = 0.3)]
    rms_ema: f32,

    /// Hotkey to toggle recording (e.g. F9, ScrollLock, Pause)
    #[arg(long)]
    hotkey: Option<String>,

    /// Type text as keystrokes into the focused application
    #[arg(long = "type")]
    type_mode: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate --delay range
    if cli.delay < 1 || cli.delay > 30 {
        anyhow::bail!("--delay must be between 1 and 30 (got {})", cli.delay);
    }

    // Validate: --type without --hotkey is probably a mistake
    if cli.type_mode && cli.hotkey.is_none() {
        eprintln!("Warning: --type without --hotkey will type continuously into the focused app.");
        eprintln!("         Consider adding --hotkey F9 to toggle recording.");
    }

    // Parse hotkey early so we fail fast on invalid key names
    let hotkey_key = cli.hotkey.as_deref()
        .map(hotkey::parse_hotkey)
        .transpose()?;

    let silence_chunks = cli.silence_flush.unwrap_or(cli.delay + 9);

    // mmap the safetensors file (instant — no disk I/O yet) and spawn readahead
    // thread to pre-fault pages at full sequential bandwidth. Starting this before
    // CUDA init gives the readahead a ~0.5-1s head start.
    println!("Loading safetensors...");
    let st_path = format!("{}/consolidated.safetensors", cli.model_dir);
    let st_data = Arc::new(unsafe { memmap2::Mmap::map(&fs::File::open(&st_path)?)? });
    let readahead = {
        let data = Arc::clone(&st_data);
        std::thread::spawn(move || {
            let mut dummy = 0u8;
            for i in (0..data.len()).step_by(4096) {
                dummy = dummy.wrapping_add(data[i]);
            }
            std::hint::black_box(dummy);
        })
    };

    // Match PyTorch's default BF16 matmul behavior (reduced precision accumulation)
    candle_core::cuda_backend::set_gemm_reduced_precision_bf16(true);

    // --- Setup (CUDA init runs while readahead pre-faults pages) ---
    let device = Device::cuda_if_available(cli.device)?;
    let dtype = DType::BF16;

    let is_offline = cli.wav_file.is_some();
    let effective_delay = if is_offline { 20 } else { cli.delay };

    println!("\n{:<28} {}", "Parameter", "Value");
    println!("{:-<28} {:-<32}", "", "");
    println!("{:<28} {} ({}ms){}",
        "Delay tokens", effective_delay, effective_delay * 80,
        if is_offline { " (offline max accuracy)" } else { "" });
    println!("{:<28} {} ({}s)",
        "Encoder sliding window", encoder::SLIDING_WINDOW, encoder::SLIDING_WINDOW * 20 / 1000);
    println!("{:<28} {} ({:.0}min)",
        "Decoder sliding window", decoder::SLIDING_WINDOW, decoder::SLIDING_WINDOW as f64 * 0.08 / 60.0);
    println!("{:<28} {} (RMS < {})",
        "Silence threshold", cli.silence_threshold, cli.silence_threshold);
    println!("{:<28} {} ({}ms)",
        "Silence newline after", silence_chunks, silence_chunks * 80);
    println!("{:<28} {} ({}ms)",
        "Min speech to activate", cli.min_speech, cli.min_speech * 80);
    println!("{:<28} {}", "RMS EMA alpha", cli.rms_ema);
    if let Some(ref key) = hotkey_key {
        println!("{:<28} {:?}", "Hotkey", key);
    }
    if cli.type_mode {
        println!("{:<28} enabled", "Keyboard output");
    }
    println!("{:<28} {:?}", "Compute dtype", dtype);
    println!();

    let vb = VarBuilder::from_slice_safetensors(&st_data, dtype, &device)?;

    let t_total = Instant::now();

    println!("Loading tokenizer...");
    let tok = tokenizer::Tokenizer::load(&cli.model_dir)?;

    println!("Loading encoder...");
    let mut enc = encoder::AudioEncoder::load(&vb, &device, dtype)?;

    println!("Loading adapter...");
    let adapter = adapter::Adapter::load(&vb)?;

    println!("Loading decoder...");
    let mut dec = decoder::TextDecoder::load(&vb, &device, dtype)?;

    println!("Total model load: {:.2}s", t_total.elapsed().as_secs_f64());
    let _ = readahead.join();

    let filters = mel::mel_filters(&cli.model_dir);

    let config = streaming::StreamConfig {
        delay_tokens: cli.delay,
        silence_threshold: cli.silence_threshold,
        silence_chunks,
        min_speech_chunks: cli.min_speech,
        rms_ema_alpha: cli.rms_ema,
        hotkey: hotkey_key,
        type_mode: cli.type_mode,
    };

    match cli.wav_file {
        Some(path) => {
            run_offline(&path, effective_delay, &mut enc, &adapter, &mut dec, &tok, &filters, &device, dtype)
        }
        None => {
            println!("\n=== Voicet Streaming Mode ===\n");
            streaming::run_streaming(&mut enc, &adapter, &mut dec, &tok, &filters, &device, dtype, &config)
        }
    }
}

fn run_offline(
    wav_path: &str,
    delay_tokens: usize,
    enc: &mut encoder::AudioEncoder,
    adapter: &adapter::Adapter,
    dec: &mut decoder::TextDecoder,
    tok: &tokenizer::Tokenizer,
    filters: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<()> {
    println!("\n=== Voicet Offline Transcription ===\n");

    // --- Load audio ---
    println!("Loading audio: {}", wav_path);
    let samples = load_wav(wav_path)?;
    let audio_secs = samples.len() as f64 / 16000.0;
    println!("Audio: {} samples ({:.2}s at 16kHz)", samples.len(), audio_secs);

    // --- Mel spectrogram ---
    let t0 = Instant::now();
    let left_pad_samples = common::LEFT_PAD_TOKENS * MEL_FRAMES_PER_TOKEN * mel::HOP_LENGTH;
    let mut padded_samples = vec![0.0f32; left_pad_samples];
    padded_samples.extend_from_slice(&samples);
    let mel_data = mel::log_mel_spectrogram(&padded_samples, filters);
    let mel_time = mel_data.len() / mel::N_MELS;
    println!("Mel: [{}, {}] in {:.3}s", mel::N_MELS, mel_time, t0.elapsed().as_secs_f64());

    let mel_tensor = Tensor::from_vec(mel_data, (mel::N_MELS, mel_time), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

    // --- Encoder ---
    let t0 = Instant::now();
    let enc_out = enc.forward(&mel_tensor)?;
    println!("Encoder: {:?} in {:.3}s", enc_out.shape(), t0.elapsed().as_secs_f64());

    // --- Adapter ---
    let t0 = Instant::now();
    let adapter_out = adapter.forward(&enc_out)?;
    println!("Adapter: {:?} in {:.3}s", adapter_out.shape(), t0.elapsed().as_secs_f64());

    // --- Generation ---
    let t0 = Instant::now();
    let n_audio_frames = adapter_out.dim(1)?;

    let t_cond = decoder::sinusoidal_embedding(delay_tokens as f32, device, dtype)?;
    let prefill_len = decoder::prefill_len(delay_tokens);

    let prefill_embeds = dec.prepare_prefill(&adapter_out, delay_tokens, device, dtype)?;

    dec.reset_caches();
    dec.precompute_t_cond(&t_cond)?;
    let logits = dec.forward(&prefill_embeds)?;

    let first_token = argmax_last(&logits)?;
    println!("Prefill -> first token: {} ({})", first_token, tok.decode(&[first_token]));
    let mut generated_tokens: Vec<u32> = vec![first_token];

    let max_tokens = n_audio_frames;
    let mut pos = prefill_len;

    loop {
        if pos >= max_tokens {
            break;
        }
        let last_token = *generated_tokens.last().unwrap();
        if last_token == tokenizer::EOS_ID {
            break;
        }

        let tok_embed = dec.embed_tokens(&[last_token], device)?;
        let audio_frame = if pos < n_audio_frames {
            adapter_out.narrow(1, pos, 1)?
        } else {
            Tensor::zeros((1, 1, decoder::HIDDEN_SIZE), dtype, device)?
        };
        let input_embed = tok_embed.add(&audio_frame)?;

        let logits = dec.forward(&input_embed)?;
        let next_token = argmax_last(&logits)?;
        generated_tokens.push(next_token);
        pos += 1;
        dec.trim_caches();
    }

    let gen_time = t0.elapsed().as_secs_f64();
    let n_tokens = generated_tokens.len();
    let text = tok.decode(&generated_tokens);

    println!("\n=== Transcription Result ===");
    println!("{}", text);
    println!("\n--- Stats ---");
    println!("Generated {} tokens in {:.2}s ({:.1} tok/s)",
        n_tokens, gen_time, n_tokens as f64 / gen_time);
    println!("Audio duration: {:.2}s", audio_secs);
    println!("Real-time factor: {:.2}x", gen_time / audio_secs);

    let n_pad = generated_tokens.iter().filter(|&&t| t == tokenizer::STREAMING_PAD_ID).count();
    let n_word = generated_tokens.iter().filter(|&&t| t == tokenizer::STREAMING_WORD_ID).count();
    let n_eos = generated_tokens.iter().filter(|&&t| t == tokenizer::EOS_ID).count();
    let n_text = n_tokens - n_pad - n_word - n_eos;
    println!("Tokens: {} text, {} pad, {} word-boundary, {} eos",
        n_text, n_pad, n_word, n_eos);

    println!("\nNon-PAD tokens:");
    for (i, &t) in generated_tokens.iter().enumerate() {
        if t != tokenizer::STREAMING_PAD_ID {
            println!("  pos={}: id={} '{}'", i + prefill_len, t, tok.decode(&[t]));
        }
    }

    Ok(())
}

fn load_wav(path: &str) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    println!("  WAV spec: {} Hz, {} ch, {:?}, {} bits",
        spec.sample_rate, spec.channels, spec.sample_format, spec.bits_per_sample);

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max_val).collect()
        }
    };

    let samples = if spec.channels > 1 {
        samples.iter().step_by(spec.channels as usize).copied().collect()
    } else {
        samples
    };

    if spec.sample_rate != 16000 {
        println!("  Resampling from {} Hz to 16000 Hz", spec.sample_rate);
        let ratio = 16000.0 / spec.sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = (i as f64 / ratio) as usize;
            resampled.push(samples[src_idx.min(samples.len() - 1)]);
        }
        return Ok(resampled);
    }

    Ok(samples)
}
