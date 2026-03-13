// Shared types used by both encoder and decoder with different hyperparameters:
// RmsNorm, RotaryEmbedding, KvCache, deinterleave_qk
//
// Also holds protocol constants derived from the model architecture.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;

// ---- Protocol constants (architecture-derived, not tunable) ----

/// Number of silence tokens prepended to align the prefill.
/// 32 tokens × 1280 samples = 40960 samples of silence.
pub const LEFT_PAD_TOKENS: usize = 32;

/// Mel frames consumed per decoder token: conv stem stride-2 × adapter group-4.
pub const MEL_FRAMES_PER_TOKEN: usize = 8;

// ---- RMS Norm ----

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(vb: &VarBuilder, name: &str, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(&[size], name)
            .with_context(|| format!("loading RmsNorm: {name}"))?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Fused CUDA kernel: single launch with F32 accumulation internally.
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

// ---- Rotary Position Embedding ----

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f32, device: &Device, dtype: DType) -> candle_core::Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin, head_dim })
    }

    /// Apply rotary embeddings. Input layout: [batch, seq_len, heads, head_dim].
    pub fn apply(&self, x: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let seq_len = x.dim(1)?; // [batch, seq_len, heads, head_dim]
        let cos = self.cos.i(offset..offset + seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
        let sin = self.sin.i(offset..offset + seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
        // cos/sin shape: [1, seq_len, 1, head_dim] — broadcasts over batch and heads

        let half = self.head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        x.broadcast_mul(&cos)?.broadcast_add(&rotated.broadcast_mul(&sin)?)
    }
}

// ---- KV Cache ----

pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// Absolute position of the first cached entry (non-zero after trimming).
    base_offset: usize,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None, base_offset: 0 }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.base_offset = 0;
    }

    /// Append new K, V to cache and return the full (cached + new) K, V.
    /// Layout: [batch, seq, heads, head_dim] (flash attention format).
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(ck), Some(cv)) => {
                let k = Tensor::cat(&[ck, k], 1)?; // [batch, total_seq, heads, hdim]
                let v = Tensor::cat(&[cv, v], 1)?;
                (k, v)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn current_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(1).unwrap_or(0)).unwrap_or(0)
    }

    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Trim cache to at most `max_len` entries, dropping the oldest.
    /// Layout: [batch, seq, heads, head_dim].
    pub fn trim(&mut self, max_len: usize) {
        if let (Some(k), Some(v)) = (&self.k, &self.v) {
            let len = k.dim(1).unwrap_or(0);
            if len > max_len {
                let drop = len - max_len;
                self.k = Some(k.narrow(1, drop, max_len).unwrap().contiguous().unwrap());
                self.v = Some(v.narrow(1, drop, max_len).unwrap().contiguous().unwrap());
                self.base_offset += drop;
            }
        }
    }
}

/// Greedy argmax on logits [batch, 1, vocab]. Returns token ID.
pub fn argmax_last(logits: &Tensor) -> candle_core::Result<u32> {
    logits.i((0, 0, ..))?.argmax(0)?.to_scalar::<u32>()
}

// ---- Deinterleave Q/K weights ----
// Mistral's consolidated.safetensors stores Q/K weights in interleaved format.
// Deinterleave at load time so rotate_half RoPE works correctly.
pub fn deinterleave_qk(w: &Tensor, num_heads: usize, head_dim: usize) -> candle_core::Result<Tensor> {
    let in_features = w.dim(1)?;
    let half = head_dim / 2;
    let w = w.reshape((num_heads, head_dim, in_features))?;
    let device = w.device();
    let even_idx: Vec<u32> = (0..head_dim as u32).step_by(2).collect();
    let odd_idx: Vec<u32> = (1..head_dim as u32).step_by(2).collect();
    let even_idx = Tensor::from_vec(even_idx, (half,), device)?;
    let odd_idx = Tensor::from_vec(odd_idx, (half,), device)?;
    let first_half = w.index_select(&even_idx, 1)?;
    let second_half = w.index_select(&odd_idx, 1)?;
    Tensor::cat(&[&first_half, &second_half], 1)?.reshape((num_heads * head_dim, in_features))
}
