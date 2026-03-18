// Mel spectrogram computation, adapted from whisper.cpp via candle's whisper/audio.rs
// Parameters: N_FFT=400, HOP_LENGTH=160, N_MELS=128
// Fixed: load HF filterbank, center padding, no conjugate mirror, global_log_mel_max

use std::sync::Arc;
use std::thread;

pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
const GLOBAL_LOG_MEL_MAX: f32 = 1.5;

fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    if n == 1 {
        return vec![inp[0], 0.0];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![0.0f32; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &v) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(v);
        } else {
            odd.push(v);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let two_pi = 2.0 * std::f32::consts::PI;
    let n_f = n as f32;
    for k in 0..n / 2 {
        let theta = two_pi * k as f32 / n_f;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

fn dft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    let two_pi = 2.0 * std::f32::consts::PI;
    let n_f = n as f32;
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        let k_f = k as f32;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &v) in inp.iter().enumerate() {
            let angle = two_pi * k_f * j as f32 / n_f;
            re += v * angle.cos();
            im -= v * angle.sin();
        }
        out.push(re);
        out.push(im);
    }
    out
}

fn log_mel_spectrogram_worker(
    ith: usize,
    hann: &[f32],
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    n_len: usize,
    n_mel: usize,
    n_threads: usize,
) -> Vec<f32> {
    let n_fft = 1 + fft_size / 2;
    let mut fft_in = vec![0.0f32; fft_size];
    let mut mel = vec![0.0f32; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / fft_step + 1, n_len);

    for i in (ith..end).step_by(n_threads) {
        let offset = i * fft_step;

        // apply Hanning window
        for j in 0..std::cmp::min(fft_size, n_samples.saturating_sub(offset)) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        if n_samples.saturating_sub(offset) < fft_size {
            let start = if offset < n_samples { n_samples - offset } else { 0 };
            fft_in[start..].fill(0.0);
        }

        let fft_out = fft(&fft_in);

        // magnitude squared — one-sided spectrum only, NO conjugate mirror
        // fft_out is interleaved [re0, im0, re1, im1, ...] with fft_size complex values
        // We only need bins 0..n_fft (= fft_size/2 + 1)

        // mel filterbank
        for j in 0..n_mel {
            let mut sum = 0.0f32;
            for k in 0..n_fft {
                let mag_sq = fft_out[2 * k] * fft_out[2 * k] + fft_out[2 * k + 1] * fft_out[2 * k + 1];
                sum += mag_sq * filters[j * n_fft + k];
            }
            mel[j * n_len + i] = sum.max(1e-10).log10();
        }
    }
    mel
}

/// Compute log mel spectrogram from PCM samples.
/// Returns a flat Vec<f32> of shape [n_mel, n_len] in row-major order.
pub fn log_mel_spectrogram(samples: &[f32], filters: &[f32]) -> Vec<f32> {
    let fft_size = N_FFT;
    let fft_step = HOP_LENGTH;
    let n_mel = N_MELS;

    let hann: Vec<f32> = (0..fft_size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos())
        })
        .collect();

    // Center padding: reflect-pad N_FFT/2 on each side (matching torch.stft center=True)
    let center_pad = N_FFT / 2; // 200
    let n_samples = samples.len();
    let mut padded_samples = Vec::with_capacity(n_samples + 2 * center_pad);

    // Reflect pad at the start: samples[center_pad], samples[center_pad-1], ..., samples[1]
    for i in (1..=center_pad).rev() {
        padded_samples.push(samples[i.min(n_samples - 1)]);
    }
    padded_samples.extend_from_slice(samples);
    // Reflect pad at the end: samples[n-2], samples[n-3], ...
    for i in 0..center_pad {
        let idx = n_samples.saturating_sub(2 + i);
        padded_samples.push(samples[idx]);
    }

    // torch.stft frame count: 1 + (padded_len - n_fft) / hop_length
    // Then drop the last frame (Python does stft[..., :-1])
    let stft_frames = 1 + (padded_samples.len() - fft_size) / fft_step;
    let n_len = stft_frames - 1; // drop last frame to match HF

    // Ensure padded_samples is long enough for all frames
    let needed = (stft_frames - 1) * fft_step + fft_size;
    if padded_samples.len() < needed {
        padded_samples.resize(needed, 0.0f32);
    }

    let n_threads = std::cmp::min(
        std::cmp::max(num_cpus::get() - num_cpus::get() % 2, 2),
        12,
    );

    let hann = Arc::new(hann);
    let samples_arc = Arc::new(padded_samples);
    let filters = Arc::new(filters.to_vec());

    let all_outputs = thread::scope(|s| {
        (0..n_threads)
            .map(|tid| {
                let hann = Arc::clone(&hann);
                let samples = Arc::clone(&samples_arc);
                let filters = Arc::clone(&filters);
                s.spawn(move || {
                    log_mel_spectrogram_worker(
                        tid, &hann, &samples, &filters, fft_size, fft_step, n_len,
                        n_mel, n_threads,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("mel thread panicked"))
            .collect::<Vec<_>>()
    });

    let l = all_outputs[0].len();
    let mut mel = vec![0.0f32; l];

    for thread_out in &all_outputs {
        for (i, &val) in thread_out.iter().enumerate() {
            mel[i] += val;
        }
    }

    // Normalize: use global_log_mel_max = 1.5 (from config.json / params.json)
    let min_val = GLOBAL_LOG_MEL_MAX - 8.0; // -6.5
    for m in mel.iter_mut() {
        *m = m.max(min_val) / 4.0 + 1.0;
    }

    mel
}

// ---- Incremental Mel Spectrogram (for streaming) ----

pub struct IncrementalMel {
    sample_buf: Vec<f32>,
    hop_pos: usize,       // next hop position in sample_buf
    hann: Vec<f32>,
    filters: Vec<f32>,
    initialized: bool,    // whether reflect padding has been applied
}

impl IncrementalMel {
    /// Create a fresh IncrementalMel that will apply reflect padding on first push.
    pub fn new(filters: &[f32]) -> Self {
        let hann: Vec<f32> = (0..N_FFT)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
            .collect();
        Self {
            sample_buf: Vec::new(),
            hop_pos: 0,
            hann,
            filters: filters.to_vec(),
            initialized: false,
        }
    }
    /// Push new audio samples. On the first call, prepend reflect padding.
    pub fn push_samples(&mut self, samples: &[f32]) {
        if !self.initialized && !samples.is_empty() {
            // Reflect-pad 200 samples at the start (matching offline center padding)
            let center_pad = N_FFT / 2; // 200
            let n = samples.len();
            for i in (1..=center_pad).rev() {
                self.sample_buf.push(samples[i.min(n - 1)]);
            }
            self.initialized = true;
        }
        self.sample_buf.extend_from_slice(samples);
    }

    /// Try to compute the next mel frame. Returns None if not enough samples.
    pub fn next_frame(&mut self) -> Option<[f32; N_MELS]> {
        if self.hop_pos + N_FFT > self.sample_buf.len() {
            return None;
        }

        let n_fft_bins = 1 + N_FFT / 2; // 201

        // Apply Hann window and compute FFT
        let mut fft_in = vec![0.0f32; N_FFT];
        for j in 0..N_FFT {
            fft_in[j] = self.hann[j] * self.sample_buf[self.hop_pos + j];
        }
        let fft_out = fft(&fft_in);

        // Mel filterbank dot product
        let mut mel = [0.0f32; N_MELS];
        let min_val = GLOBAL_LOG_MEL_MAX - 8.0; // -6.5
        for j in 0..N_MELS {
            let mut sum = 0.0f32;
            for k in 0..n_fft_bins {
                let mag_sq = fft_out[2 * k] * fft_out[2 * k] + fft_out[2 * k + 1] * fft_out[2 * k + 1];
                sum += mag_sq * self.filters[j * n_fft_bins + k];
            }
            mel[j] = sum.max(1e-10).log10().max(min_val) / 4.0 + 1.0;
        }

        self.hop_pos += HOP_LENGTH;
        Some(mel)
    }

    /// Drain all available mel frames.
    pub fn drain_frames(&mut self) -> Vec<[f32; N_MELS]> {
        let mut frames = Vec::new();
        while let Some(frame) = self.next_frame() {
            frames.push(frame);
        }
        // Periodically trim consumed samples to prevent unbounded growth.
        // Keep everything from hop_pos onward (needed for the next FFT window).
        if self.hop_pos > 16000 {
            self.sample_buf.drain(..self.hop_pos);
            self.hop_pos = 0;
        }
        frames
    }
}

/// Load precomputed mel filterbank from binary file.
/// Returns flat array of shape [n_mel, n_fft/2+1] = [128, 201].
pub fn mel_filters(model_dir: &str) -> Vec<f32> {
    let path = format!("{model_dir}/mel_filters.bin");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    assert_eq!(data.len(), N_MELS * (N_FFT / 2 + 1) * 4,
        "mel_filters.bin has wrong size: expected {}, got {}",
        N_MELS * (N_FFT / 2 + 1) * 4, data.len());
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
