import gc
import html
import os
import random
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))

import foley
from bundle import SafetensorsFile, attach, convert_to_bundle
from media import finish_video
from storage import data_root, fast_root

VERSION = "1.0"
REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
LLM_REPO = "Qwen/Qwen2.5-3B-Instruct"
WAN_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "transformer/*", "vae/*", "text_encoder/*"]
LLM_FILES = ["*.json", "*.safetensors", "*.txt"]
OUTPUTS = ROOT / "omega" / "outputs"
FPS = 24
TIMESTEPS = (1000, 757, 522)
SIZES = {"480": (480, 832), "720": (704, 1280)}
LENGTHS = {"2": 49, "3": 73, "5": 121}
PARTS = ("transformer", "text_encoder", "enhancer")
TAE_FILE = "taew2_2_super.pth"
ENHANCE_SYSTEM = (
    "You turn short video ideas into one detailed prompt for a text-to-video model. "
    "Keep the user's subject, action and any style they ask for. Add concrete visual detail: "
    "who or what is in the scene and how they look, the setting, the motion that unfolds over a "
    "few seconds, the lighting, the camera framing and movement, and the overall visual style. "
    "Write a single paragraph of 60 to 100 words in English, in the present tense. "
    "No lists, no quotation marks, no titles, and nothing except the prompt itself."
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cache_dir():
    return data_root() / "hf" / "hub"


def store():
    return fast_root() / "omega" / "models"


def prepared():
    try:
        files = [f"{p}.safetensors" for p in PARTS] + ["vae", "taehv.py", TAE_FILE]
        return all((store() / f).exists() for f in files)
    except RuntimeError:
        return False


def _local(repo, patterns, online=False):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, allow_patterns=patterns, cache_dir=cache_dir(),
                                  local_files_only=not online))


def _convert(files, model, part, extra=None):
    out = store() / f"{part}.safetensors"
    tmp = out.with_suffix(".tmp")
    convert_to_bundle(files, model, tmp, extra)
    tmp.replace(out)


def prepare(log=print):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from diffusers import WanTransformer3DModel
    from transformers import AutoConfig, AutoModelForCausalLM, UMT5Config, UMT5EncoderModel

    store().mkdir(parents=True, exist_ok=True)
    log("  downloading the video model (~23GB)...")
    wan = _local(REPO, WAN_FILES, online=True)
    log("  downloading the prompt enhancer (~6GB)...")
    llm = _local(LLM_REPO, LLM_FILES, online=True)
    if not (store() / "transformer.safetensors").exists():
        log("  converting the video model to 8-bit...")
        with torch.device("meta"):
            model = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(wan / "transformer"))
        _convert(sorted((wan / "transformer").glob("*.safetensors")), model, "transformer")
    if not (store() / "text_encoder.safetensors").exists():
        log("  converting the text reader to 8-bit...")
        with torch.device("meta"):
            model = UMT5EncoderModel(UMT5Config.from_pretrained(wan / "text_encoder"))
        _convert(sorted((wan / "text_encoder").glob("*.safetensors")), model, "text_encoder")
    if not (store() / "enhancer.safetensors").exists():
        log("  converting the prompt enhancer to 8-bit...")
        config = AutoConfig.from_pretrained(llm)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        files = sorted(llm.glob("*.safetensors"))
        extra = {}
        if config.tie_word_embeddings:
            key = "model.embed_tokens.weight"
            embed = next(sf for sf in map(SafetensorsFile, files) if key in sf.header)
            extra["p.lm_head.weight"] = (torch.float8_e4m3fn, embed.header[key]["shape"],
                                         lambda: embed.get(key).float().clamp(-448, 448))
        _convert(files, model, "enhancer", extra)
    if not (store() / "vae").exists():
        log("  converting the decoder to 16-bit...")
        from diffusers import AutoencoderKLWan
        AutoencoderKLWan.from_pretrained(wan / "vae", torch_dtype=torch.bfloat16).save_pretrained(store() / "vae")
    for name in ("taehv.py", TAE_FILE):
        if not (store() / name).exists():
            log(f"  downloading {name}...")
            urlretrieve(f"https://github.com/madebyollin/taehv/raw/main/{name}", store() / name)
    log("  Omega ready")


def _clean(text):
    return " ".join(html.unescape(html.unescape(text)).split())


def _write_video(path, frames):
    import imageio_ffmpeg
    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=FPS, codec="libx264", quality=None,
                                         macro_block_size=1, output_params=["-crf", "18"])
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()


def _free(*modules):
    for m in modules:
        src = getattr(m, "_bundle_src", None)
        if src:
            src.close()
        m.to("meta")
    gc.collect()
    torch.cuda.empty_cache()


class Omega:
    def __init__(self):
        if not prepared():
            raise RuntimeError("Omega isn't set up yet - type: setup omega (downloads about 30GB)")
        from diffusers import WanTransformer3DModel
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
        from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.5, (free - 300 * 2**20) / total))
        self.wan = _local(REPO, WAN_FILES)
        self.last_prompt = None
        log("loading text reader...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.wan / "tokenizer")
        with torch.device("meta"):
            text_encoder = UMT5EncoderModel(UMT5Config.from_pretrained(self.wan / "text_encoder"))
        text_encoder.encoder.embed_tokens = text_encoder.shared
        self.text_encoder = attach(text_encoder, store() / "text_encoder.safetensors", "cuda",
                                   stream_embeddings=("shared",))
        log("loading video model...")
        config = WanTransformer3DModel.load_config(self.wan / "transformer")
        with torch.device("meta"):
            transformer = WanTransformer3DModel.from_config(config)
        with torch.device("cuda"):
            transformer.rope = WanRotaryPosEmbed(config["attention_head_dim"], tuple(config["patch_size"]),
                                                 config["rope_max_seq_len"])
        self.transformer = attach(transformer, store() / "transformer.safetensors", "cuda")
        log(f"ready - GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f}GB")

    @torch.no_grad()
    def enhance(self, prompt, seed):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        path = _local(LLM_REPO, LLM_FILES)
        config = AutoConfig.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        with torch.device("meta"):
            llm = AutoModelForCausalLM.from_config(config)
        with torch.device("cuda"):
            llm.model.rotary_emb = Qwen2RotaryEmbedding(config=config)
        attach(llm, store() / "enhancer.safetensors", "cuda", stream_linear=False)
        try:
            chat = [{"role": "system", "content": ENHANCE_SYSTEM}, {"role": "user", "content": prompt}]
            enc = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt").to("cuda")
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = llm.generate(**enc, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9,
                                   pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            _free(llm)
        return _clean(text) or prompt

    @torch.no_grad()
    def encode(self, text, max_len=512):
        tok = self.tokenizer(_clean(text), max_length=max_len, truncation=True, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = self.text_encoder(tok.input_ids.to("cuda"), attention_mask=tok.attention_mask.to("cuda"))[0]
        out = torch.zeros(1, max_len, emb.shape[-1], device="cuda", dtype=torch.bfloat16)
        out[:, :emb.shape[1]] = emb
        return out

    @torch.no_grad()
    def decode(self, latents, fast=True):
        if fast:
            sys.path.insert(0, str(store()))
            from taehv import TAEHV
            tae = TAEHV(str(store() / TAE_FILE)).to("cuda", torch.float16).eval()
            try:
                video = tae.decode_video(latents.permute(0, 2, 1, 3, 4).half(), parallel=False,
                                         show_progress_bar=False)
                frames = (video[0].float().clamp(0, 1) * 255).round().to(torch.uint8)
                return frames.permute(0, 2, 3, 1).cpu().numpy()
            finally:
                _free(tae)
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(store() / "vae", torch_dtype=torch.bfloat16, device_map="cuda")
        try:
            mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
            z = (latents * std + mean).to(torch.bfloat16)
            for tile, stride in ((384, 320), (256, 192)):
                vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                                  tile_sample_stride_height=stride, tile_sample_stride_width=stride)
                try:
                    video = vae.decode(z, return_dict=False)[0]
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
            frames = ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            return frames.permute(1, 2, 3, 0).cpu().numpy()
        finally:
            _free(vae)

    @torch.no_grad()
    def sample(self, text, length, res, seed, progress=None):
        emb = self.encode(text)
        height, width = SIZES[res]
        frames = LENGTHS[length]
        shape = (1, self.transformer.config.in_channels, (frames - 1) // 4 + 1, height // 16, width // 16)
        gen = torch.Generator("cuda").manual_seed(seed)
        x = torch.randn(shape, generator=gen, device="cuda")
        for i, t in enumerate(TIMESTEPS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self.transformer(hidden_states=x.to(torch.bfloat16),
                                     timestep=torch.full((1,), float(t), device="cuda"),
                                     encoder_hidden_states=emb, return_dict=False)[0]
            x0 = x - (t / 1000) * v.float()
            if i + 1 < len(TIMESTEPS):
                s = TIMESTEPS[i + 1] / 1000
                x = (1 - s) * x0 + s * torch.randn(shape, generator=gen, device="cuda")
            if progress:
                progress((i + 1) / len(TIMESTEPS), desc=f"step {i + 1}/{len(TIMESTEPS)}")
        del x, emb, v
        torch.cuda.empty_cache()
        return x0

    @torch.no_grad()
    def generate(self, prompt, length="5", res="720", seed=None, enhance=True, sound=True, smooth=False,
                 fast_decode=True, progress=None, stage=None):
        stage = stage or (lambda msg: None)
        seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)
        text = prompt.strip()
        if enhance:
            stage("enhancing prompt")
            text = self.enhance(text, seed)
        self.last_prompt = text
        stage("reading prompt")
        latents = self.sample(text, length, res, seed, progress)
        stage("decoding")
        video = self.decode(latents, fast=fast_decode)
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        name = f"omega_{time.strftime('%Y%m%d_%H%M%S')}_{seed}"
        raw = OUTPUTS / f".{name}_raw.mp4"
        _write_video(raw, video)
        audio = None
        if sound and foley.prepared():
            stage("adding sound")
            audio = foley.make_sound(raw, prompt.strip(), video.shape[0] / FPS, seed=seed)
        if smooth:
            stage("smoothing")
        final = OUTPUTS / f"{name}.mp4"
        finish_video(raw, final, audio, smooth, False)
        return str(final), seed


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    else:
        print("usage: python omega_app.py prepare")
