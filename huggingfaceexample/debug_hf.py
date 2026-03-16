"""
Run HF Voxtral Realtime on test WAV with hooks to extract intermediates.
This is the KNOWN-WORKING reference to compare with Rust.
"""
import torch
import soundfile as sf
import numpy as np

MODEL_DIR = "Voxtral-Mini-4B-Realtime"

# Use PyTorch's default BF16 matmul behavior (reduced precision accumulation)
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True by default
WAV_PATH = f"{MODEL_DIR}/test01_16khz_3.7s.wav"

print("=== Loading HF model ===")
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()

print(f"\n=== Loading audio: {WAV_PATH} ===")
audio, sr = sf.read(WAV_PATH, dtype="float32")
print(f"Audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s")

# Debug Step 1: Save raw audio samples
np.save("debug_hf_step1_audio.npy", audio)
print(f"  Saved debug_hf_step1_audio.npy shape={audio.shape}")

# Process with processor (streaming mode, matching voicet.py)
inputs = processor(audio, is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
input_features = inputs["input_features"].to("cuda", dtype=torch.bfloat16)

# Debug Step 2: Save mel spectrogram
mel_step2 = input_features[0].float().cpu().numpy()
np.save("debug_hf_step2_mel.npy", mel_step2)
print(f"  Saved debug_hf_step2_mel.npy shape={mel_step2.shape}")

# Debug Step 3: Save input_ids (prefill tokens)
ids_step3 = input_ids[0].cpu().numpy().astype(np.int32)
np.save("debug_hf_step3_input_ids.npy", ids_step3)
print(f"  Saved debug_hf_step3_input_ids.npy shape={ids_step3.shape}, tokens={ids_step3.tolist()}")
num_delay_tokens = inputs.get("num_delay_tokens", 6)

print(f"\ninput_ids: {input_ids[0].tolist()}")
print(f"input_features shape: {input_features.shape}")

# ---- Register hooks ----
intermediates = {}

def stat(name, t):
    tf = t.float()
    m = tf.mean().item()
    v = tf.var().item()
    s = v**0.5 if v > 0 else 0
    first5 = tf.flatten()[:5].tolist()
    intermediates[name] = (t.shape, m, s, first5)
    print(f"  {name}: shape={tuple(t.shape)}, mean={m:.6f}, std={s:.6f}, first5={first5}")

hooks = []
all_encoder_outputs = []
all_projector_outputs = []
all_conv1_outputs = []
all_conv2_outputs = []
all_layer0_inputs = []
all_layer0_outputs = []

# Discover the audio_tower structure
print("\n=== Audio tower structure ===")
for name, mod in model.audio_tower.named_children():
    print(f"  {name}: {type(mod).__name__}")
    if hasattr(mod, 'named_children'):
        for n2, m2 in mod.named_children():
            print(f"    {n2}: {type(m2).__name__}")

# Hook conv layers inside the audio_tower
# Try common naming patterns
at = model.audio_tower
conv1_mod = None
conv2_mod = None
layer0_mod = None

# Find conv and transformer layers
for name, mod in at.named_modules():
    if 'conv' in name.lower() and hasattr(mod, 'weight') and mod.weight is not None:
        if conv1_mod is None:
            conv1_mod = (name, mod)
        elif conv2_mod is None:
            conv2_mod = (name, mod)
    # Find first transformer layer
    if name.endswith('.0') and 'layer' in name.lower():
        layer0_mod = (name, mod)

if conv1_mod:
    print(f"Found conv1: {conv1_mod[0]}")
    def conv1_hook(mod, inp, out):
        all_conv1_outputs.append(out.detach().float().cpu())
    hooks.append(conv1_mod[1].register_forward_hook(conv1_hook))

if conv2_mod:
    print(f"Found conv2: {conv2_mod[0]}")
    def conv2_hook(mod, inp, out):
        all_conv2_outputs.append(out.detach().float().cpu())
    hooks.append(conv2_mod[1].register_forward_hook(conv2_hook))

if layer0_mod:
    print(f"Found layer0: {layer0_mod[0]}")
    def layer0_hook(mod, inp, out):
        h_in = inp[0] if isinstance(inp, tuple) else inp
        all_layer0_inputs.append(h_in.detach().float().cpu())
        h_out = out[0] if isinstance(out, tuple) else out
        all_layer0_outputs.append(h_out.detach().float().cpu())
    hooks.append(layer0_mod[1].register_forward_hook(layer0_hook))

# Debug Step 4: Hook conv stem output (= input to layer 0 = output of embedder)
# The layer0_inputs already captures this, but we also hook the embedder directly
all_step4_conv_embed = []
def embedder_hook(mod, inp, out):
    # out is [batch, channels, time] -> transpose to [time, channels]
    all_step4_conv_embed.append(out.detach().float().cpu())
hooks.append(at.embedder.register_forward_hook(embedder_hook))

# Debug Step 5: Hook Q,K after RoPE in layer 0 attention
all_step5_rope_q = []
all_step5_rope_k = []
_step5_capture_count = [0]

# Patch apply_rotary_pos_emb globally to capture Q,K from layer 0's first call
import transformers.models.voxtral_realtime.modeling_voxtral_realtime as _vrt_mod
_orig_apply_rope = _vrt_mod.apply_rotary_pos_emb

def _patched_apply_rope(q, k, cos, sin, *a, **kw):
    result = _orig_apply_rope(q, k, cos, sin, *a, **kw)
    q_rot, k_rot = result
    if _step5_capture_count[0] == 0:
        # First call = first chunk of layer 0
        all_step5_rope_q.append(q_rot.detach().float().cpu())
        all_step5_rope_k.append(k_rot.detach().float().cpu())
    _step5_capture_count[0] += 1
    return result

_vrt_mod.apply_rotary_pos_emb = _patched_apply_rope

# Hook layer 0 attention sub-components
all_l0_attn_norm = []
all_l0_attn_out = []
all_l0_after_attn_res = []

layer0 = at.layers[0]
print(f"Layer 0 children: {[n for n,_ in layer0.named_children()]}")

# Find norm and attention modules
for name, mod in layer0.named_children():
    if 'norm' in name.lower() and 'attn' in name.lower() or name == 'input_layernorm':
        print(f"  Using '{name}' as attention norm")
        def l0_attn_norm_hook(mod, inp, out):
            all_l0_attn_norm.append(out.detach().float().cpu())
        hooks.append(mod.register_forward_hook(l0_attn_norm_hook))
        break
else:
    # Try first norm
    for name, mod in layer0.named_children():
        if 'norm' in name.lower():
            print(f"  Using '{name}' as attention norm")
            def l0_attn_norm_hook(mod, inp, out):
                all_l0_attn_norm.append(out.detach().float().cpu())
            hooks.append(mod.register_forward_hook(l0_attn_norm_hook))
            break

# Find attention module
for name, mod in layer0.named_children():
    if 'attn' in name.lower():
        print(f"  Using '{name}' as attention module")
        def l0_attn_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            all_l0_attn_out.append(h.detach().float().cpu())
        hooks.append(mod.register_forward_hook(l0_attn_hook))
        break

# Hook encoder output
def enc_hook(mod, inp, out):
    if hasattr(out, 'last_hidden_state'):
        stat('encoder_output', out.last_hidden_state)
        all_encoder_outputs.append(out.last_hidden_state.detach().float().cpu())
hooks.append(model.audio_tower.register_forward_hook(enc_hook))

# Hook projector output
def proj_hook(mod, inp, out):
    stat('projector_output', out)
    all_projector_outputs.append(out.detach().float().cpu())
hooks.append(model.multi_modal_projector.register_forward_hook(proj_hook))

# Hook time embedding
def time_hook(mod, inp, out):
    stat('time_embedding', out)
hooks.append(model.time_embedding.register_forward_hook(time_hook))

# Hook the fusion point: capture inputs_embeds before and after audio addition
original_forward = model.forward.__wrapped__ if hasattr(model.forward, '__wrapped__') else None

# Hook decoder layers 0, 12, 25 and final norm
lm = model.language_model.model
for li in [0, 12, 25]:
    if li < len(lm.layers):
        def make_layer_hook(layer_idx):
            def layer_hook(mod, inp, out):
                # out is a tuple, first element is hidden_states
                h = out[0] if isinstance(out, tuple) else out
                stat(f'decoder_layer_{layer_idx}', h)
            return layer_hook
        hooks.append(lm.layers[li].register_forward_hook(make_layer_hook(li)))

# Hook final norm
def norm_hook(mod, inp, out):
    stat('final_norm', out)
hooks.append(lm.norm.register_forward_hook(norm_hook))

# Hook embed_tokens to see token embeddings
def embed_hook(mod, inp, out):
    stat('tok_embeddings', out)
hooks.append(lm.embed_tokens.register_forward_hook(embed_hook))

# Also capture the fused input (tok_embed + audio) going into the LM
# We need to hook the language_model.forward to capture inputs_embeds
lm_model = model.language_model.model
original_lm_forward = lm_model.forward

def patched_lm_forward(input_ids=None, inputs_embeds=None, **kwargs):
    if inputs_embeds is not None:
        stat('lm_input_embeds', inputs_embeds)
    return original_lm_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

lm_model.forward = patched_lm_forward

print(f"\n=== Running generation ===")
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        num_delay_tokens=num_delay_tokens,
        do_sample=False,
        max_new_tokens=200,
    )

# Decode
generated = output_ids[0][input_ids.shape[1]:]
all_tokens = generated.tolist()
n_pad = all_tokens.count(32)
n_word = all_tokens.count(33)
n_eos = all_tokens.count(2)
n_text = len(all_tokens) - n_pad - n_word - n_eos

decoded = processor.decode(generated, skip_special_tokens=False)
print(f"\n=== Result ===")
print(f"Decoded: '{decoded}'")
print(f"Tokens: {n_text} text, {n_pad} pad, {n_word} word, {n_eos} eos, total={len(all_tokens)}")
print(f"\nAll generated token IDs: {all_tokens}")

print(f"\nNon-PAD tokens:")
for i, t in enumerate(all_tokens):
    if t != 32:
        print(f"  gen_pos={i}: id={t} '{processor.decode([t])}'")

# Save collected intermediates for comparison
print(f"\n=== Saving debug dumps ===")

# Debug Step 4: Save conv embedder output
# embedder output is already [batch, time, channels] (permuted inside embedder)
if all_step4_conv_embed:
    emb_stacked = torch.cat([c.squeeze(0) for c in all_step4_conv_embed], dim=0)  # [total_time, channels]
    emb_np = emb_stacked.numpy()  # [time, channels]
    np.save("debug_hf_step4_conv_embed.npy", emb_np)
    print(f"  Saved debug_hf_step4_conv_embed.npy shape={emb_np.shape}")

# Debug Step 5: Save Q,K after RoPE from layer 0 attention, first chunk, head 0
if all_step5_rope_q:
    # Q is [batch, heads, seq, head_dim] - take head 0 from first chunk
    q0 = all_step5_rope_q[0][0, 0].numpy()  # [seq, head_dim]
    np.save("debug_hf_step5_rope_q.npy", q0)
    print(f"  Saved debug_hf_step5_rope_q.npy shape={q0.shape}")

if all_step5_rope_k:
    k0 = all_step5_rope_k[0][0, 0].numpy()  # [seq, head_dim]
    np.save("debug_hf_step5_rope_k.npy", k0)
    print(f"  Saved debug_hf_step5_rope_k.npy shape={k0.shape}")

# Save conv stem outputs
if all_conv1_outputs:
    # Each is [1, 1280, chunk_time] after GELU activation
    # Stack along time dimension
    conv1_stacked = torch.cat([c.squeeze(0) for c in all_conv1_outputs], dim=1)  # [1280, total_time]
    print(f"Conv1 (GELU) frames collected: {conv1_stacked.shape}")
    # Transpose to [time, channels] for comparison
    np.save("debug_hf_conv1.npy", conv1_stacked.numpy().T)  # [time, 1280]
    print(f"  Saved debug_hf_conv1.npy shape={conv1_stacked.shape[1]}x{conv1_stacked.shape[0]}")

if all_conv2_outputs:
    conv2_stacked = torch.cat([c.squeeze(0) for c in all_conv2_outputs], dim=1)  # [1280, total_time]
    print(f"Conv2 (GELU) frames collected: {conv2_stacked.shape}")
    np.save("debug_hf_conv2.npy", conv2_stacked.numpy().T)  # [time, 1280]
    print(f"  Saved debug_hf_conv2.npy shape={conv2_stacked.shape[1]}x{conv2_stacked.shape[0]}")

if all_layer0_inputs:
    l0i_stacked = torch.cat([c.squeeze(0) for c in all_layer0_inputs], dim=0)  # [total_frames, 1280]
    print(f"Layer 0 inputs collected: {l0i_stacked.shape}")
    np.save("debug_hf_conv_stem_out.npy", l0i_stacked.numpy())
    print(f"  Saved debug_hf_conv_stem_out.npy")

if all_layer0_outputs:
    l0o_stacked = torch.cat([c.squeeze(0) for c in all_layer0_outputs], dim=0)  # [total_frames, 1280]
    print(f"Layer 0 outputs collected: {l0o_stacked.shape}")
    np.save("debug_hf_enc_layer0.npy", l0o_stacked.numpy())
    print(f"  Saved debug_hf_enc_layer0.npy")

if all_l0_attn_norm:
    stacked = torch.cat([c.squeeze(0) for c in all_l0_attn_norm], dim=0)
    print(f"Layer 0 attn_norm collected: {stacked.shape}")
    np.save("debug_hf_enc_l0_attn_norm.npy", stacked.numpy())

if all_l0_attn_out:
    stacked = torch.cat([c.squeeze(0) for c in all_l0_attn_out], dim=0)
    print(f"Layer 0 attn_out collected: {stacked.shape}")
    np.save("debug_hf_enc_l0_attn_out.npy", stacked.numpy())

# Save all projector (adapter) outputs: each is [1, 1, 3072], stack to [n_frames, 3072]
if all_projector_outputs:
    proj_stacked = torch.cat([p.squeeze(0) for p in all_projector_outputs], dim=0)  # [n_frames, 3072]
    print(f"Projector frames collected: {proj_stacked.shape}")
    np.save("debug_hf_adapter.npy", proj_stacked.numpy())
    print(f"  Saved debug_hf_adapter.npy")

# Save all encoder outputs: each is [1, 4, 1280], stack to [n_frames*4, 1280]
if all_encoder_outputs:
    enc_stacked = torch.cat([e.squeeze(0) for e in all_encoder_outputs], dim=0)  # [n_steps*4, 1280]
    print(f"Encoder frames collected: {enc_stacked.shape}")
    np.save("debug_hf_encoder.npy", enc_stacked.numpy())
    print(f"  Saved debug_hf_encoder.npy")

# Also dump mel spectrogram (input_features)
mel_np = input_features[0].float().cpu().numpy()
print(f"Mel spectrogram: {mel_np.shape}")
np.save("debug_hf_mel.npy", mel_np)
print(f"  Saved debug_hf_mel.npy")

# Remove hooks
for h in hooks:
    h.remove()
lm_model.forward = original_lm_forward
