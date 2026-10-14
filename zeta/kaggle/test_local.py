import os
import sys
import tempfile
from contextlib import nullcontext

import imageio_ffmpeg
import torch
from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXPipeline,
                       LTXVideoTransformer3DModel)
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from transformers import AutoTokenizer, T5EncoderModel

sys.path.insert(0, os.path.dirname(__file__))
import train_zeta as tz

CLIP = os.path.join(os.path.dirname(__file__), "..", "dataset", "clips", "zeta_s049_2.mp4")
W, H, F = 64, 64, 9
tz.WIDTH, tz.HEIGHT, tz.FRAMES = W, H, F

torch.manual_seed(0)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

video = tz.load_clip(CLIP, ffmpeg, W, H, F)
assert video.shape == (1, 3, F, H, W) and -1.01 <= video.min() and video.max() <= 1.01, video.shape
print("load_clip ok", tuple(video.shape))

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
(emb, mask), = tz.encode_prompts(tokenizer, text_encoder, ["Peter talks to Lois"], max_len=16)
neg = tz.encode_prompts(tokenizer, text_encoder, ["blurry"], max_len=16)[0]
print("encode_prompts ok", tuple(emb.shape), tuple(mask.shape))

vae = AutoencoderKLLTXVideo(
    in_channels=3, out_channels=3, latent_channels=8,
    block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8),
    layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1, 1),
    spatio_temporal_scaling=(True, True, False, False),
    decoder_spatio_temporal_scaling=(True, True, False, False),
    decoder_inject_noise=(False, False, False, False, False),
    upsample_residual=(False, False, False, False), upsample_factor=(1, 1, 1, 1),
    timestep_conditioning=False, patch_size=1, patch_size_t=1,
    encoder_causal=True, decoder_causal=False)
vae.use_framewise_encoding = False
vae.use_framewise_decoding = False
latents, dims = tz.encode_video(vae, video, 1, 1)
print("encode_video ok", tuple(latents.shape), dims)

transformer = LTXVideoTransformer3DModel(
    in_channels=8, out_channels=8, patch_size=1, patch_size_t=1, num_attention_heads=4,
    attention_head_dim=8, cross_attention_dim=32, num_layers=1, caption_channels=emb.shape[-1])
transformer.requires_grad_(False)
transformer.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian",
                                   target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
cast_training_params(transformer, dtype=torch.float32)
transformer.enable_gradient_checkpointing()
transformer.train()
params = [p for p in transformer.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-3)
before = {k: v.clone() for k, v in tz.lora_state(transformer).items()}
for _ in range(3):
    loss = tz.train_step(transformer, latents, dims, emb, mask, tz.shifted_sigma(latents.shape[1]),
                         torch.float32, nullcontext())
    loss.backward()
    opt.step()
    opt.zero_grad()
after = tz.lora_state(transformer)
changed = sum(not torch.equal(before[k], after[k]) for k in before)
assert torch.isfinite(loss) and changed > 0, (loss, changed)
print(f"train_step ok, loss {loss.item():.4f}, {changed}/{len(before)} LoRA tensors updated")

with tempfile.TemporaryDirectory() as d:
    tz.save_lora(transformer, d)
    print("save_lora ok", os.listdir(d))
    transformer.eval()
    pipe = LTXPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=None,
                       tokenizer=None, transformer=transformer)
    with torch.no_grad():
        tz.render(pipe, (emb, mask), neg, os.path.join(d, "s.mp4"), 2, 0, torch.float32, "cpu")
        transformer.disable_adapters()
        tz.render(pipe, (emb, mask), neg, os.path.join(d, "b.mp4"), 2, 0, torch.float32, "cpu")
        transformer.enable_adapters()
    print("render ok", os.path.getsize(os.path.join(d, "s.mp4")), "bytes")

    fresh = LTXPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=None,
                        tokenizer=None, transformer=LTXVideoTransformer3DModel(
                            in_channels=8, out_channels=8, patch_size=1, patch_size_t=1,
                            num_attention_heads=4, attention_head_dim=8, cross_attention_dim=32,
                            num_layers=1, caption_channels=emb.shape[-1]))
    fresh.load_lora_weights(d)
    print("load_lora_weights ok")
print("ALL OK")
