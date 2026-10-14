import gc
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext

STEPS = 1500
SAVE_EVERY = 250
SAMPLE_STEPS = 40
SMOKE_TEST = False

MODEL_ID = "Lightricks/LTX-Video-0.9.1"
WIDTH, HEIGHT, FRAMES, FPS = 512, 288, 73, 24
RANK = 32
LR = 1e-4
WARMUP = 50
SEED = 42
CAPTION_PREFIX = "Family Guy style cartoon. "
NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"
SAMPLE_PROMPTS = [
    "Peter Griffin sits on the purple couch in the Griffin living room, laughing, "
    "while Lois stands beside him with her arms crossed.",
    "Brian and Stewie walk down a sunny suburban street past yellow houses, talking to each other.",
    "Quagmire in a red Hawaiian shirt talks to a woman at the counter of a busy diner.",
]
WORK = "/kaggle/working"
CACHE = "/tmp/zeta_cache"

if SMOKE_TEST:
    STEPS, SAVE_EVERY, SAMPLE_STEPS = 12, 6, 8
    SAMPLE_PROMPTS = SAMPLE_PROMPTS[:1]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "diffusers", "peft", "sentencepiece", "imageio", "imageio-ffmpeg"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)


def load_clip(path, ffmpeg, width=WIDTH, height=HEIGHT, frames=FRAMES):
    import numpy as np
    import torch
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", path,
                          "-vf", f"scale={width}:{height}:flags=lanczos",
                          "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    if len(arr) < frames:
        return None
    video = torch.from_numpy(arr.copy()).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    return video.unsqueeze(0)


def encode_video(vae, video, patch_size, patch_size_t):
    import torch
    from diffusers import LTXPipeline
    with torch.no_grad():
        latents = vae.encode(video.to(vae.device, vae.dtype)).latent_dist.mode()
    latents = LTXPipeline._normalize_latents(latents, vae.latents_mean, vae.latents_std,
                                             vae.config.scaling_factor)
    _, _, f, h, w = latents.shape
    packed = LTXPipeline._pack_latents(latents, patch_size, patch_size_t)
    return packed.float().cpu(), (f, h, w)


def encode_prompts(tokenizer, text_encoder, prompts, max_len=128):
    import torch
    out = []
    for p in prompts:
        tok = tokenizer(p, padding="max_length", max_length=max_len, truncation=True,
                        add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            emb = text_encoder(tok.input_ids.to(text_encoder.device),
                               attention_mask=tok.attention_mask.to(text_encoder.device))[0]
        out.append((emb.float().cpu(), tok.attention_mask.cpu()))
    return out


def shifted_sigma(n_tokens, generator=None):
    import torch
    m = (1.15 - 0.5) / (4096 - 256)
    mu = n_tokens * m + (0.5 - m * 256)
    u = torch.sigmoid(torch.randn(1, generator=generator))
    return math.exp(mu) * u / (math.exp(mu) * u + 1 - u)


def train_step(transformer, latents, dims, emb, mask, sigma, dtype, autocast):
    import torch
    import torch.nn.functional as F
    device = transformer.device
    x0 = latents.to(device)
    noise = torch.randn_like(x0)
    s = sigma.to(device).view(-1, 1, 1)
    noisy = (1 - s) * x0 + s * noise
    f, h, w = dims
    with autocast:
        pred = transformer(hidden_states=noisy.to(dtype),
                           encoder_hidden_states=emb.to(device, dtype),
                           timestep=(sigma * 1000).to(device),
                           encoder_attention_mask=mask.to(device),
                           num_frames=f, height=h, width=w,
                           rope_interpolation_scale=(8 / FPS, 32, 32),
                           return_dict=False)[0]
    return F.mse_loss(pred.float(), (noise - x0).float())


def lora_state(transformer):
    from peft.utils import get_peft_model_state_dict
    return get_peft_model_state_dict(transformer)


def save_lora(transformer, path):
    from diffusers import LTXPipeline
    LTXPipeline.save_lora_weights(path, transformer_lora_layers=lora_state(transformer))


def render(pipe, emb, neg, path, steps, seed, dtype, device="cuda"):
    import torch
    from diffusers.utils import export_to_video
    (pe, pm), (ne, nm) = emb, neg
    frames = pipe(prompt_embeds=pe.to(device, dtype), prompt_attention_mask=pm.to(device),
                  negative_prompt_embeds=ne.to(device, dtype), negative_prompt_attention_mask=nm.to(device),
                  width=WIDTH, height=HEIGHT, num_frames=FRAMES, frame_rate=FPS,
                  num_inference_steps=steps, guidance_scale=3.0,
                  decode_timestep=0.03, decode_noise_scale=0.025,
                  generator=torch.Generator(device).manual_seed(seed)).frames[0]
    export_to_video(frames, path, fps=FPS)


def main():
    install()
    import imageio_ffmpeg
    import torch
    from diffusers import AutoencoderKLLTXVideo, LTXPipeline, LTXVideoTransformer3DModel
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from transformers import T5EncoderModel, T5Tokenizer

    os.environ.setdefault("HF_HOME", "/tmp/hf")
    random.seed(SEED)
    torch.manual_seed(SEED)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cc = torch.cuda.get_device_capability()
    native_bf16 = cc[0] >= 8
    dtype = torch.bfloat16 if native_bf16 else torch.float32
    te_dtype = torch.bfloat16 if native_bf16 else torch.float16
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if native_bf16 else nullcontext()
    log(f"GPU {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}, "
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}GB, "
        f"training in {'bf16' if native_bf16 else 'fp32'}, torch {torch.__version__}")

    meta = glob.glob("/kaggle/input/**/metadata.jsonl", recursive=True)
    if not meta:
        sys.exit("dataset not attached: no metadata.jsonl under /kaggle/input")
    data_dir = os.path.dirname(meta[0])
    rows = [json.loads(l) for l in open(meta[0], encoding="utf-8") if l.strip()]
    log(f"{len(rows)} clips in {data_dir}")

    os.makedirs(CACHE, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder",
                                                  torch_dtype=te_dtype).to("cuda")
    captions = encode_prompts(tokenizer, text_encoder,
                              [CAPTION_PREFIX + r["caption"] for r in rows])
    samples = encode_prompts(tokenizer, text_encoder,
                             [CAPTION_PREFIX + p for p in SAMPLE_PROMPTS])
    negative = encode_prompts(tokenizer, text_encoder, [NEGATIVE])[0]
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    log("captions encoded")

    transformer_cfg = LTXVideoTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")
    vae = AutoencoderKLLTXVideo.from_pretrained(MODEL_ID, subfolder="vae",
                                                torch_dtype=torch.float32).to("cuda")
    data = []
    for r, (emb, mask) in zip(rows, captions):
        video = load_clip(os.path.join(data_dir, os.path.basename(r["file_name"])), ffmpeg)
        if video is None:
            log(f"skipping short clip {r['file_name']}")
            continue
        latents, dims = encode_video(vae, video, transformer_cfg.get("patch_size", 1),
                                     transformer_cfg.get("patch_size_t", 1))
        data.append((latents, dims, emb, mask))
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    log(f"{len(data)} clips encoded, latent grid {data[0][1]} = {data[0][0].shape[1]} tokens")

    transformer = LTXVideoTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer",
                                                             torch_dtype=dtype).to("cuda")
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian",
                                       target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    cast_training_params(transformer, dtype=torch.float32)
    transformer.enable_gradient_checkpointing()
    transformer.train()
    params = [p for p in transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    log(f"LoRA rank {RANK}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable params")

    history, window, t0 = [], [], time.time()
    for step in range(1, STEPS + 1):
        latents, dims, emb, mask = random.choice(data)
        loss = train_step(transformer, latents, dims, emb, mask,
                          shifted_sigma(latents.shape[1]), dtype, autocast)
        if not torch.isfinite(loss):
            save_lora(transformer, f"{WORK}/checkpoints/nan_at_{step}")
            sys.exit(f"loss became {loss.item()} at step {step}; stopping")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        window.append(loss.item())
        if step % 10 == 0 or step == STEPS:
            per = (time.time() - t0) / step
            history.append({"step": step, "loss": sum(window) / len(window)})
            log(f"step {step}/{STEPS}  loss {history[-1]['loss']:.4f}  {per:.1f}s/step  "
                f"eta {per * (STEPS - step) / 60:.0f}min  "
                f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
            window = []
        if step % SAVE_EVERY == 0 and step != STEPS:
            save_lora(transformer, f"{WORK}/checkpoints/step_{step}")
    save_lora(transformer, f"{WORK}/zeta_lora")
    json.dump(history, open(f"{WORK}/train_log.json", "w"))
    log(f"training done in {(time.time() - t0) / 60:.0f}min")

    transformer.eval()
    pipe = LTXPipeline.from_pretrained(MODEL_ID, transformer=transformer, text_encoder=None,
                                       tokenizer=None, torch_dtype=dtype).to("cuda")
    os.makedirs(f"{WORK}/samples", exist_ok=True)
    with torch.no_grad(), autocast:
        for i, emb in enumerate(samples):
            render(pipe, emb, negative, f"{WORK}/samples/zeta_{i}.mp4", SAMPLE_STEPS, SEED + i, dtype)
            log(f"sample {i} rendered")
        transformer.disable_adapters()
        render(pipe, samples[0], negative, f"{WORK}/samples/base_0.mp4", SAMPLE_STEPS, SEED, dtype)
        transformer.enable_adapters()
        log("base model comparison rendered")
    log("all done")


if __name__ == "__main__":
    main()
