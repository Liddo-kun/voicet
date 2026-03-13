// Audio encoder for Voxtral Realtime
// Conv1d stem (2 layers) + 32 causal transformer layers with sliding window attention, RoPE, SwiGLU, RMSNorm
// Processes transformer layers in chunks of 4 frames with KV cache to match HF pipeline.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use crate::common::{deinterleave_qk, KvCache, RmsNorm, RotaryEmbedding};

// Encoder hyperparameters from config.json
pub const HIDDEN_SIZE: usize = 1280;
pub const INTERMEDIATE_SIZE: usize = 5120;
pub const NUM_LAYERS: usize = 32;
pub const NUM_HEADS: usize = 32;
pub const HEAD_DIM: usize = 64;
pub const NUM_KV_HEADS: usize = 32;
pub const SLIDING_WINDOW: usize = 750;
pub const RMS_NORM_EPS: f64 = 1e-5;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const NUM_MEL_BINS: usize = 128;
pub const CHUNK_SIZE: usize = 4;

/// Weight name prefix for the encoder
const PREFIX: &str = "mm_streams_embeddings.embedding_module.whisper_encoder";

// ---- Causal Sliding Window Attention ----

struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    cache: KvCache,
}

impl Attention {
    fn load(vb: &VarBuilder, layer_prefix: &str) -> Result<Self> {
        let attn_prefix = format!("{layer_prefix}.attention");
        let qkv_in = HIDDEN_SIZE;
        let qkv_out = NUM_HEADS * HEAD_DIM; // 32 * 64 = 2048

        // wq has bias — weight is in Mistral interleaved format, deinterleave to HF paired-halves
        let wq_w = vb.get(&[qkv_out, qkv_in], &format!("{attn_prefix}.wq.weight"))?;
        let wq_w = deinterleave_qk(&wq_w, NUM_HEADS, HEAD_DIM)?;
        let wq_b = vb.get(&[qkv_out], &format!("{attn_prefix}.wq.bias"))?;
        let wq = Linear::new(wq_w, Some(wq_b));

        // wk has NO bias — also deinterleave
        let wk_w = vb.get(&[qkv_out, qkv_in], &format!("{attn_prefix}.wk.weight"))?;
        let wk_w = deinterleave_qk(&wk_w, NUM_KV_HEADS, HEAD_DIM)?;
        let wk = Linear::new(wk_w, None);

        // wv has bias
        let wv_w = vb.get(&[qkv_out, qkv_in], &format!("{attn_prefix}.wv.weight"))?;
        let wv_b = vb.get(&[qkv_out], &format!("{attn_prefix}.wv.bias"))?;
        let wv = Linear::new(wv_w, Some(wv_b));

        // wo has bias
        let wo_w = vb.get(&[qkv_in, qkv_out], &format!("{attn_prefix}.wo.weight"))?;
        let wo_b = vb.get(&[qkv_in], &format!("{attn_prefix}.wo.bias"))?;
        let wo = Linear::new(wo_w, Some(wo_b));

        Ok(Self { wq, wk, wv, wo, cache: KvCache::new() })
    }

    fn reset_cache(&mut self) {
        self.cache.reset();
    }

    fn forward(&mut self, x: &Tensor, rope: &RotaryEmbedding) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let offset = self.cache.base_offset() + self.cache.current_len();

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
        let (k_full, v_full) = self.cache.append(&k, &v)?;

        // Flash attention with causal sliding window
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
    w1: Linear, // gate projection [5120, 1280] no bias
    w2: Linear, // down projection [1280, 5120] WITH bias
    w3: Linear, // up projection   [5120, 1280] no bias
}

impl Mlp {
    fn load(vb: &VarBuilder, layer_prefix: &str) -> Result<Self> {
        let ff_prefix = format!("{layer_prefix}.feed_forward");

        let w1_w = vb.get(&[INTERMEDIATE_SIZE, HIDDEN_SIZE], &format!("{ff_prefix}.w1.weight"))?;
        let w1 = Linear::new(w1_w, None);

        let w2_w = vb.get(&[HIDDEN_SIZE, INTERMEDIATE_SIZE], &format!("{ff_prefix}.w2.weight"))?;
        let w2_b = vb.get(&[HIDDEN_SIZE], &format!("{ff_prefix}.w2.bias"))?;
        let w2 = Linear::new(w2_w, Some(w2_b));

        let w3_w = vb.get(&[INTERMEDIATE_SIZE, HIDDEN_SIZE], &format!("{ff_prefix}.w3.weight"))?;
        let w3 = Linear::new(w3_w, None);

        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // SwiGLU: w2(silu(w1(x)) * w3(x))
        let gate = candle_nn::Activation::Silu.forward(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

// ---- Transformer Layer ----

struct EncoderLayer {
    attention: Attention,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
}

impl EncoderLayer {
    fn load(vb: &VarBuilder, layer_idx: usize) -> Result<Self> {
        let layer_prefix = format!("{PREFIX}.transformer.layers.{layer_idx}");

        let attention = Attention::load(vb, &layer_prefix)?;
        let attention_norm = RmsNorm::load(vb, &format!("{layer_prefix}.attention_norm.weight"), HIDDEN_SIZE, RMS_NORM_EPS)?;
        let mlp = Mlp::load(vb, &layer_prefix)?;
        let ffn_norm = RmsNorm::load(vb, &format!("{layer_prefix}.ffn_norm.weight"), HIDDEN_SIZE, RMS_NORM_EPS)?;

        Ok(Self { attention, attention_norm, mlp, ffn_norm })
    }

    fn reset_cache(&mut self) {
        self.attention.reset_cache();
    }

    fn trim_cache(&mut self, max_len: usize) {
        self.attention.cache.trim(max_len);
    }

    fn forward(&mut self, x: &Tensor, rope: &RotaryEmbedding) -> candle_core::Result<Tensor> {
        // Pre-norm attention + residual
        let residual = x;
        let h = self.attention_norm.forward(x)?;
        let h = self.attention.forward(&h, rope)?;
        let x = (h + residual)?;

        // Pre-norm MLP + residual
        let residual = &x;
        let h = self.ffn_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        h + residual
    }
}

// ---- Conv1d Stem ----

struct Conv1dLayer {
    weight: Tensor, // [out_channels, in_channels, kernel_size]
    bias: Tensor,   // [out_channels]
    stride: usize,
}

impl Conv1dLayer {
    fn load(vb: &VarBuilder, idx: usize, in_ch: usize, out_ch: usize, stride: usize) -> Result<Self> {
        let name = format!("{PREFIX}.conv_layers.{idx}.conv");
        let weight = vb.get(&[out_ch, in_ch, 3], &format!("{name}.weight"))?;
        let bias = vb.get(&[out_ch], &format!("{name}.bias"))?;
        Ok(Self { weight, bias, stride })
    }

    /// Causal conv1d: left-pad so output only depends on current and past inputs.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let kernel_size = 3usize;
        let left_pad = kernel_size - self.stride;

        // x shape: [batch, channels, time]
        let x = x.pad_with_zeros(2, left_pad, 0)?;

        let out = x.conv1d(&self.weight, 0, self.stride, 1, 1)?;
        out.broadcast_add(&self.bias.unsqueeze(0)?.unsqueeze(D::Minus1)?)
    }
}

// ---- Full Encoder ----

pub struct AudioEncoder {
    conv1: Conv1dLayer,
    conv2: Conv1dLayer,
    layers: Vec<EncoderLayer>,
    final_norm: RmsNorm,
    rope: RotaryEmbedding,
}

impl AudioEncoder {
    /// Load the audio encoder from safetensors weights.
    pub fn load(vb: &VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        println!("  Loading conv stem...");
        let conv1 = Conv1dLayer::load(vb, 0, NUM_MEL_BINS, HIDDEN_SIZE, 1)?;
        let conv2 = Conv1dLayer::load(vb, 1, HIDDEN_SIZE, HIDDEN_SIZE, 2)?;

        println!("  Loading {} transformer layers...", NUM_LAYERS);
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(EncoderLayer::load(vb, i)
                .with_context(|| format!("loading encoder layer {i}"))?);
            if (i + 1) % 8 == 0 {
                println!("    Loaded {}/{} layers", i + 1, NUM_LAYERS);
            }
        }

        println!("  Loading final norm...");
        let final_norm = RmsNorm::load(vb, &format!("{PREFIX}.transformer.norm.weight"), HIDDEN_SIZE, RMS_NORM_EPS)?;

        println!("  Creating RoPE embeddings...");
        let rope = RotaryEmbedding::new(131072, HEAD_DIM, ROPE_THETA, device, dtype)?;

        Ok(Self { conv1, conv2, layers, final_norm, rope })
    }

    /// Reset all KV caches for a fresh sequence.
    #[allow(dead_code)]
    pub fn reset_caches(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_cache();
        }
    }

    /// Trim all encoder KV caches to the sliding window size.
    pub fn trim_caches(&mut self) {
        for layer in &mut self.layers {
            layer.trim_cache(SLIDING_WINDOW);
        }
    }

    /// Run the conv stem on mel input. Returns [batch, time, hidden].
    pub fn conv_stem(&self, mel: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv1.forward(mel)?;
        let x = x.gelu_erf()?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu_erf()?;
        x.transpose(1, 2)?.contiguous()
    }

    /// Process a single chunk through all transformer layers using existing KV caches.
    /// Does NOT reset caches. RoPE offset is computed from cache state.
    pub fn forward_chunk(&mut self, chunk: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = chunk.clone();
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, &self.rope)?;
        }
        self.final_norm.forward(&h)
    }

    /// Run the full encoder: mel -> conv stem -> chunked transformer -> final norm.
    /// Resets KV caches, then processes all frames in CHUNK_SIZE chunks.
    pub fn forward(&mut self, mel: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv_stem(mel)?;
        let total_frames = x.dim(1)?;

        for layer in self.layers.iter_mut() {
            layer.reset_cache();
        }

        let n_full_chunks = total_frames / CHUNK_SIZE;
        let remainder = total_frames % CHUNK_SIZE;
        let n_chunks = n_full_chunks + if remainder > 0 { 1 } else { 0 };
        let mut chunk_outputs = Vec::with_capacity(n_chunks);

        for chunk_idx in 0..n_chunks {
            let chunk_len = if chunk_idx < n_full_chunks { CHUNK_SIZE } else { remainder };
            let chunk = x.narrow(1, chunk_idx * CHUNK_SIZE, chunk_len)?;
            chunk_outputs.push(self.forward_chunk(&chunk)?);
        }

        let chunk_refs: Vec<&Tensor> = chunk_outputs.iter().collect();
        Tensor::cat(&chunk_refs, 1)
    }
}
