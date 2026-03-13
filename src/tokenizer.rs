// Tekken tokenizer: decode-only for Voxtral Realtime
// Token IDs 0-999: special tokens (rank = token ID)
// Token IDs 1000-131071: BPE tokens (ID - 1000 = rank in vocab array)

use anyhow::{Context, Result};
use base64::Engine;

pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const STREAMING_PAD_ID: u32 = 32;
pub const STREAMING_WORD_ID: u32 = 33;

const NUM_SPECIAL: u32 = 1000;

pub struct Tokenizer {
    /// BPE vocab: index = rank, value = token bytes
    vocab: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn load(model_dir: &str) -> Result<Self> {
        let path = format!("{model_dir}/tekken.json");
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {path}"))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .context("parsing tekken.json")?;

        let vocab_arr = json["vocab"].as_array()
            .context("tekken.json missing 'vocab' array")?;

        // Build vocab indexed by rank
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_arr.len());
        let engine = base64::engine::general_purpose::STANDARD;

        for entry in vocab_arr {
            let token_bytes_b64 = entry["token_bytes"].as_str()
                .context("missing token_bytes")?;
            let bytes = engine.decode(token_bytes_b64)
                .context("base64 decode failed")?;
            vocab.push(bytes);
        }

        Ok(Self { vocab })
    }

    /// Decode a single token ID to bytes. Returns None for special tokens that should be skipped.
    pub fn decode_token(&self, id: u32) -> Option<Vec<u8>> {
        if id == STREAMING_PAD_ID {
            return None; // skip padding tokens
        }
        if id == STREAMING_WORD_ID {
            return None; // control token, not text — BPE tokens carry their own spacing
        }
        if id < NUM_SPECIAL {
            return None; // other special tokens: skip
        }
        let rank = (id - NUM_SPECIAL) as usize;
        if rank < self.vocab.len() {
            Some(self.vocab[rank].clone())
        } else {
            None
        }
    }

    /// Decode a sequence of token IDs to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.decode_token(id) {
                bytes.extend_from_slice(&b);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}
