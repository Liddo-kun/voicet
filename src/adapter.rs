// Adapter: 4x downsample by concatenating encoder frames, then project to decoder hidden size
// audio_language_projection: Linear(5120→3072, no bias) + GELU + Linear(3072→3072, no bias)

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

const PREFIX: &str = "mm_streams_embeddings.embedding_module.audio_language_projection";

pub struct Adapter {
    linear1: Linear,
    linear2: Linear,
}

impl Adapter {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let w1 = vb.get(&[3072, 5120], &format!("{PREFIX}.0.weight"))
            .context("loading adapter linear1 weight")?;
        let linear1 = Linear::new(w1, None);

        let w2 = vb.get(&[3072, 3072], &format!("{PREFIX}.2.weight"))
            .context("loading adapter linear2 weight")?;
        let linear2 = Linear::new(w2, None);

        Ok(Self { linear1, linear2 })
    }

    /// Forward pass: reshape encoder output to 4x downsample, then project.
    /// Input: [batch, frames, 1280]
    /// Output: [batch, frames/4, 3072]
    pub fn forward(&self, encoder_output: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, frames, hidden) = encoder_output.dims3()?;
        let frames4 = frames / 4; // truncate to multiple of 4

        // Truncate to multiple of 4
        let x = if frames4 * 4 != frames {
            encoder_output.narrow(1, 0, frames4 * 4)?
        } else {
            encoder_output.clone()
        };

        // Reshape: [batch, frames4*4, 1280] -> [batch, frames4, 1280*4=5120]
        let x = x.reshape((batch, frames4, hidden * 4))?;

        // Project: Linear(5120→3072) + GELU + Linear(3072→3072)
        let x = self.linear1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x)
    }
}
