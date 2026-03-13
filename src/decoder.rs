// Text decoder for Voxtral Realtime
// 26 layers, GQA (32Q/8KV heads, head_dim=128), RoPE theta=1M, SwiGLU, Ada-RMSNorm
// No biases on any decoder linear layer.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::common::{self, deinterleave_qk, KvCache, RmsNorm, RotaryEmbedding};
use crate::tokenizer;

// Decoder hyperparameters
pub const HIDDEN_SIZE: usize = 3072;
pub const INTERMEDIATE_SIZE: usize = 9216;
pub const NUM_LAYERS: usize = 26;
pub const NUM_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const VOCAB_SIZE: usize = 131072;
pub const RMS_NORM_EPS: f64 = 1e-5;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const SLIDING_WINDOW: usize = 2048; // ~2.7 minutes of tokens at 80ms each

/// Number of delay tokens for the streaming protocol.
/// Controls both the prefill padding count and the Ada-RMSNorm conditioning signal.
/// Higher = more lookahead = better accuracy but more latency.
/// Valid range: 1–30 (80ms–2400ms). Default 6 = 480ms.
pub const NUM_DELAY_TOKENS: usize = 4;

/// Total prefill length: BOS + LEFT_PAD_TOKENS silence PADs + NUM_DELAY_TOKENS delay PADs.
pub const PREFILL_LEN: usize = 1 + common::LEFT_PAD_TOKENS + NUM_DELAY_TOKENS;

// ---- GQA Attention ----

struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
}

impl Attention {
    fn load(vb: &VarBuilder, layer_idx: usize) -> Result<Self> {
        let prefix = format!("layers.{layer_idx}.attention");

        let wq_w = vb.get(&[NUM_HEADS * HEAD_DIM, HIDDEN_SIZE], &format!("{prefix}.wq.weight"))?;
        let wq_w = deinterleave_qk(&wq_w, NUM_HEADS, HEAD_DIM)?;
        let wq = Linear::new(wq_w, None);

        let wk_w = vb.get(&[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE], &format!("{prefix}.wk.weight"))?;
        let wk_w = deinterleave_qk(&wk_w, NUM_KV_HEADS, HEAD_DIM)?;
        let wk = Linear::new(wk_w, None);

        let wv_w = vb.get(&[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE], &format!("{prefix}.wv.weight"))?;
        let wv = Linear::new(wv_w, None);

        let wo_w = vb.get(&[HIDDEN_SIZE, NUM_HEADS * HEAD_DIM], &format!("{prefix}.wo.weight"))?;
        let wo = Linear::new(wo_w, None);

        Ok(Self { wq, wk, wv, wo })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cache: &mut KvCache,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        // Absolute position for RoPE (accounts for trimmed entries)
        let offset = cache.base_offset() + cache.current_len();

        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape to [batch, seq_len, num_heads, head_dim] — flash attention layout
        let q = q.reshape((batch, seq_len, NUM_HEADS, HEAD_DIM))?;
        let k = k.reshape((batch, seq_len, NUM_KV_HEADS, HEAD_DIM))?;
        let v = v.reshape((batch, seq_len, NUM_KV_HEADS, HEAD_DIM))?;

        // Apply RoPE
        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;

        // Append to KV cache
        let (k_full, v_full) = cache.append(&k, &v)?;

        // Flash attention with causal sliding window (GQA handled natively: 32Q/8KV)
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let out = candle_flash_attn::flash_attn_windowed(
            &q, &k_full, &v_full,
            scale,
            Some(SLIDING_WINDOW - 1), // window_size_left
            Some(0),                   // window_size_right (causal)
        )?;

        // Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
        let out = out.reshape((batch, seq_len, NUM_HEADS * HEAD_DIM))?;
        self.wo.forward(&out)
    }
}

// ---- SwiGLU MLP ----

struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn load(vb: &VarBuilder, layer_idx: usize) -> Result<Self> {
        let prefix = format!("layers.{layer_idx}.feed_forward");

        let w1_w = vb.get(&[INTERMEDIATE_SIZE, HIDDEN_SIZE], &format!("{prefix}.w1.weight"))?;
        let w1 = Linear::new(w1_w, None);

        let w2_w = vb.get(&[HIDDEN_SIZE, INTERMEDIATE_SIZE], &format!("{prefix}.w2.weight"))?;
        let w2 = Linear::new(w2_w, None);

        let w3_w = vb.get(&[INTERMEDIATE_SIZE, HIDDEN_SIZE], &format!("{prefix}.w3.weight"))?;
        let w3 = Linear::new(w3_w, None);

        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

// ---- Ada-RMSNorm (per-layer delay conditioning bottleneck) ----

struct AdaRmsNorm {
    linear1: Linear, // [32, 3072] bottleneck down
    linear2: Linear, // [3072, 32] bottleneck up
}

impl AdaRmsNorm {
    fn load(vb: &VarBuilder, layer_idx: usize) -> Result<Self> {
        let prefix = format!("layers.{layer_idx}.ada_rms_norm_t_cond");

        let w1 = vb.get(&[32, HIDDEN_SIZE], &format!("{prefix}.0.weight"))?;
        let linear1 = Linear::new(w1, None);

        let w2 = vb.get(&[HIDDEN_SIZE, 32], &format!("{prefix}.2.weight"))?;
        let linear2 = Linear::new(w2, None);

        Ok(Self { linear1, linear2 })
    }

    /// Forward: Linear(3072->32) -> GELU -> Linear(32->3072)
    fn forward(&self, t_cond: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.linear1.forward(t_cond)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x)
    }
}

// ---- Decoder Layer ----

struct DecoderLayer {
    attention: Attention,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    ada_norm: AdaRmsNorm,
}

impl DecoderLayer {
    fn load(vb: &VarBuilder, layer_idx: usize) -> Result<Self> {
        let attention = Attention::load(vb, layer_idx)?;
        let attention_norm = RmsNorm::load(vb, &format!("layers.{layer_idx}.attention_norm.weight"), HIDDEN_SIZE, RMS_NORM_EPS)?;
        let mlp = Mlp::load(vb, layer_idx)?;
        let ffn_norm = RmsNorm::load(vb, &format!("layers.{layer_idx}.ffn_norm.weight"), HIDDEN_SIZE, RMS_NORM_EPS)?;
        let ada_norm = AdaRmsNorm::load(vb, layer_idx)?;

        Ok(Self { attention, attention_norm, mlp, ffn_norm, ada_norm })
    }

    /// `ada_scale` is precomputed `(1 + ada_rms_norm(t_cond)).unsqueeze(1)` [1, 1, HIDDEN_SIZE].
    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cache: &mut KvCache,
        ada_scale: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // Pre-norm attention + residual
        let residual = x;
        let h = self.attention_norm.forward(x)?;
        let h = self.attention.forward(&h, rope, cache)?;
        let x = (h + residual)?;

        // Pre-norm MLP with Ada-RMSNorm modulation + residual
        let residual = &x;
        let h = self.ffn_norm.forward(&x)?;
        let h = h.broadcast_mul(ada_scale)?;
        let h = self.mlp.forward(&h)?;
        h + residual
    }
}

// ---- Sinusoidal Time Embedding ----

/// Compute sinusoidal delay embedding matching HF's VoxtralRealtimeTimeEmbedding.
/// Returns [1, hidden_size] tensor on device with given dtype.
pub fn sinusoidal_embedding(delay_value: f32, device: &Device, dtype: DType) -> candle_core::Result<Tensor> {
    let half_dim = HIDDEN_SIZE / 2;
    let theta = 10000.0f64;

    // Compute inv_freq in F32, then convert to BF16 to match HF behavior
    let inv_freq_f32: Vec<f32> = (0..half_dim)
        .map(|i| (-(theta.ln()) * i as f64 / half_dim as f64).exp() as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq_f32, (half_dim,), device)?;
    let inv_freq_bf16 = inv_freq.to_dtype(dtype)?;

    let delay_tensor = Tensor::from_vec(vec![delay_value], (1,), device)?.to_dtype(dtype)?;
    let emb = delay_tensor.broadcast_mul(&inv_freq_bf16)?;

    let cos_part = emb.cos()?;
    let sin_part = emb.sin()?;

    let result = Tensor::cat(&[&cos_part, &sin_part], 0)?;
    result.reshape((1, HIDDEN_SIZE))
}

// ---- Full Decoder ----

pub struct TextDecoder {
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    tok_embeddings: Tensor,   // [131072, 3072] for embed_tokens (index_select)
    tok_embeddings_t: Tensor, // [3072, 131072] cached transpose for lm_head matmul
    rope: RotaryEmbedding,
    caches: Vec<KvCache>,
    ada_scales: Option<Vec<Tensor>>, // precomputed per-layer Ada-RMSNorm scales
}

impl TextDecoder {
    pub fn load(vb: &VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        println!("  Loading {} decoder layers...", NUM_LAYERS);
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(DecoderLayer::load(vb, i)
                .with_context(|| format!("loading decoder layer {i}"))?);
            if (i + 1) % 10 == 0 || i + 1 == NUM_LAYERS {
                println!("    Loaded {}/{} layers", i + 1, NUM_LAYERS);
            }
        }

        println!("  Loading final norm...");
        let final_norm = RmsNorm::load(vb, "norm.weight", HIDDEN_SIZE, RMS_NORM_EPS)?;

        println!("  Loading tok_embeddings...");
        let tok_embeddings = vb.get(
            &[VOCAB_SIZE, HIDDEN_SIZE],
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
        ).context("loading tok_embeddings")?;

        println!("  Caching transposed lm_head...");
        let tok_embeddings_t = tok_embeddings.t()?.contiguous()?;

        println!("  Creating decoder RoPE...");
        let rope = RotaryEmbedding::new(131072, HEAD_DIM, ROPE_THETA, device, dtype)?;

        let caches: Vec<KvCache> = (0..NUM_LAYERS).map(|_| KvCache::new()).collect();

        Ok(Self { layers, final_norm, tok_embeddings, tok_embeddings_t, rope, caches, ada_scales: None })
    }

    /// Reset KV caches for a new generation.
    pub fn reset_caches(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Trim all decoder KV caches to the sliding window size.
    pub fn trim_caches(&mut self) {
        for cache in &mut self.caches {
            cache.trim(SLIDING_WINDOW);
        }
    }

    /// Precompute per-layer Ada-RMSNorm scales from delay conditioning embedding.
    /// Call once after creating t_cond, before any forward() calls.
    /// Eliminates 26 × (2 matmuls + GELU + unsqueeze + add) per forward call.
    pub fn precompute_t_cond(&mut self, t_cond: &Tensor) -> candle_core::Result<()> {
        let scales = self.layers.iter()
            .map(|layer| {
                let s = layer.ada_norm.forward(t_cond)?;
                s.unsqueeze(1)? + 1.0
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        self.ada_scales = Some(scales);
        Ok(())
    }

    /// Build prefill embeddings: BOS + (LEFT_PAD_TOKENS + NUM_DELAY_TOKENS) PAD tokens,
    /// each fused (element-wise add) with the corresponding adapter frame.
    /// Returns the fused embeddings tensor [1, PREFILL_LEN, HIDDEN_SIZE].
    pub fn prepare_prefill(
        &self,
        adapter_out: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        let n_adapter = adapter_out.dim(1)?;

        let mut prefill_ids = vec![tokenizer::STREAMING_PAD_ID; PREFILL_LEN];
        prefill_ids[0] = tokenizer::BOS_ID;

        let tok_embeds = self.embed_tokens(&prefill_ids, device)?;
        let audio_slice = if PREFILL_LEN <= n_adapter {
            adapter_out.narrow(1, 0, PREFILL_LEN)?
        } else {
            let avail = adapter_out.narrow(1, 0, n_adapter.min(PREFILL_LEN))?;
            let pad_len = PREFILL_LEN - n_adapter.min(PREFILL_LEN);
            if pad_len > 0 {
                let zeros = Tensor::zeros((1, pad_len, HIDDEN_SIZE), dtype, device)?;
                Tensor::cat(&[&avail, &zeros], 1)?
            } else {
                avail
            }
        };
        tok_embeds.add(&audio_slice)
    }

    /// Get token embedding for a batch of token IDs.
    /// Output: [1, seq_len, hidden_size]
    pub fn embed_tokens(&self, ids: &[u32], device: &Device) -> candle_core::Result<Tensor> {
        let id_tensor = Tensor::from_vec(
            ids.to_vec(),
            (1, ids.len()),
            device,
        )?;
        self.tok_embeddings.index_select(&id_tensor.flatten_all()?, 0)?
            .reshape((1, ids.len(), HIDDEN_SIZE))
    }

    /// Forward pass through decoder layers. `precompute_t_cond` must be called first.
    /// Input: [batch, seq_len, hidden_size]
    /// Output: logits [batch, 1, vocab_size]
    pub fn forward(&mut self, inputs_embeds: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = inputs_embeds.clone();

        let ada_scales = self.ada_scales.as_ref()
            .expect("precompute_t_cond must be called before forward");

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &self.rope, &mut self.caches[i], &ada_scales[i])?;
        }

        x = self.final_norm.forward(&x)?;

        // lm_head: x @ tok_embeddings^T (cached transpose)
        let seq_len = x.dim(1)?;
        let last = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let logits = last.matmul(&self.tok_embeddings_t)?;
        logits.unsqueeze(1)
    }
}
