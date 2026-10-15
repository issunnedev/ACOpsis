import os
import random
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

import foley
from bundle import SafetensorsFile, load_bundle, save_bundle
from media import finish_video

VERSION = "1.2"
HERE = Path(__file__).resolve().parent
ZETA = HERE.parent
LORA = ZETA / "models" / "zeta-v1" / "pytorch_lora_weights.safetensors"
OUTPUTS = ZETA / "outputs"
MODEL_ID = "Lightricks/LTX-Video-0.9.1"
T5_REPO, T5_FILE = "comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"

WIDTH, HEIGHT, FPS = 512, 288, 24
PREFIX = "Family Guy style cartoon. "
NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"
LENGTHS = {"2 seconds": 49, "3 seconds": 73, "4 seconds": 97}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def models_cached():
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    ltx = hub / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    t5 = hub / ("models--" + T5_REPO.replace("/", "--")) / "snapshots"
    return any(ltx.glob("*/transformer/config.json")) and any(t5.glob(f"*/{T5_FILE}"))


class Zeta:
    def __init__(self):
        from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler,
                               LTXPipeline, LTXVideoTransformer3DModel)

        log("loading video model (fusing Zeta into it)...")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda")
        transformer.load_lora_adapter(str(LORA.parent), weight_name=LORA.name,
                                      prefix="transformer", adapter_name="zeta")
        transformer.fuse_lora(lora_scale=1.0)
        transformer.unload_lora()
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                             compute_dtype=torch.bfloat16)
        torch.cuda.empty_cache()

        log("loading decoder...")
        vae = AutoencoderKLLTXVideo.from_pretrained(MODEL_ID, subfolder="vae",
                                                    torch_dtype=torch.bfloat16, device_map="cuda")
        vae.enable_tiling()
        self.transformer, self.vae, self.offloaded = transformer, vae, False
        self.pipe = LTXPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler"),
            vae=vae, transformer=transformer, text_encoder=None, tokenizer=None)
        self.pipe.set_progress_bar_config(disable=True)

        log("loading prompt reader...")
        self.tokenizer, self.text_encoder = load_t5()
        self.negative = self.encode(NEGATIVE)
        log(f"ready - GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f}GB")

    @torch.no_grad()
    def encode(self, text, max_len=128):
        tok = self.tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
        emb = self.text_encoder(tok.input_ids, attention_mask=tok.attention_mask)[0]
        n = emb.shape[1]
        padded = torch.zeros(1, max_len, emb.shape[-1])
        padded[:, :n] = emb
        mask = torch.zeros(1, max_len, dtype=torch.long)
        mask[:, :n] = 1
        return padded.to("cuda", torch.bfloat16), mask.to("cuda")

    def _gpu_cache(self):
        folder = LORA.parent / "gpu_cache"
        stat = LORA.stat()
        key = f"{stat.st_size}-{int(stat.st_mtime)}"
        return folder, key, (folder / "key.txt").exists() and (folder / "key.txt").read_text() == key

    def offload(self):
        if self.offloaded:
            return
        folder, key, fresh = self._gpu_cache()
        if not fresh:
            folder.mkdir(exist_ok=True)
            save_bundle(self.transformer, folder / "transformer.safetensors")
            save_bundle(self.vae, folder / "vae.safetensors")
            (folder / "key.txt").write_text(key)
        self.transformer.to("meta")
        self.vae.to("meta")
        self.offloaded = True
        torch.cuda.empty_cache()

    def restore(self):
        if not self.offloaded:
            return
        folder = self._gpu_cache()[0]
        load_bundle(self.transformer, folder / "transformer.safetensors")
        load_bundle(self.vae, folder / "vae.safetensors")
        self.offloaded = False

    @torch.no_grad()
    def generate(self, prompt, frames=73, steps=40, guidance=3.0, seed=None, progress=None,
                 sound=True, smooth=False, hd=False, stage=None):
        from diffusers.utils import export_to_video
        stage = stage or (lambda msg: None)
        seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)
        self.restore()
        emb, mask = self.encode(PREFIX + prompt.strip())
        neg, neg_mask = self.negative

        def on_step(pipe, i, t, kwargs):
            if progress:
                progress((i + 1) / steps, desc=f"step {i + 1}/{steps}")
            return kwargs

        video = self.pipe(prompt_embeds=emb, prompt_attention_mask=mask,
                          negative_prompt_embeds=neg, negative_prompt_attention_mask=neg_mask,
                          width=WIDTH, height=HEIGHT, num_frames=frames, frame_rate=FPS,
                          num_inference_steps=steps, guidance_scale=guidance,
                          decode_timestep=0.03, decode_noise_scale=0.025,
                          generator=torch.Generator("cuda").manual_seed(seed),
                          callback_on_step_end=on_step).frames[0]
        OUTPUTS.mkdir(exist_ok=True)
        name = f"zeta_{time.strftime('%Y%m%d_%H%M%S')}_{seed}"
        raw = OUTPUTS / f".{name}_raw.mp4"
        export_to_video(video, str(raw), fps=FPS)

        audio = None
        if sound and foley.prepared():
            stage("adding sound")
            self.offload()
            audio = foley.make_sound(raw, prompt.strip(), frames / FPS, seed=seed)
        if smooth or hd:
            stage("smoothing and upscaling" if smooth and hd else "smoothing" if smooth else "upscaling")
        final = OUTPUTS / f"{name}.mp4"
        finish_video(raw, final, audio, smooth, hd)
        return str(final), seed


def load_t5():
    from huggingface_hub import hf_hub_download
    from transformers import T5Config, T5EncoderModel, T5Tokenizer
    from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    config = T5Config.from_pretrained(MODEL_ID, subfolder="text_encoder")
    with torch.device("meta"):
        model = T5EncoderModel(config)
    src = SafetensorsFile(hf_hub_download(T5_REPO, T5_FILE))
    model.encoder.embed_tokens = model.shared
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if isinstance(module, torch.nn.Linear):
            module.weight = None
            module._weight_src = (src, f"{prefix}weight")
            module.forward = types.MethodType(_streamed_linear_forward, module)
            continue
        if isinstance(module, T5DenseGatedActDense):
            module.forward = types.MethodType(_t5_ffn_forward, module)
        for pname in list(module._parameters):
            module._parameters[pname] = torch.nn.Parameter(src.get(prefix + pname).float(),
                                                           requires_grad=False)
    return tokenizer, model.eval()


def _streamed_linear_forward(self, x):
    src, key = self._weight_src
    return torch.nn.functional.linear(x, src.get(key).to(x.dtype), self.bias)


def _t5_ffn_forward(self, hidden_states):
    hidden = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
    return self.wo(self.dropout(hidden))


def build_ui(zeta):
    import gradio as gr

    def run(prompt, length, steps, seed, sound, smooth, hd, progress=gr.Progress()):
        if not prompt.strip():
            raise gr.Error("Type a prompt first.")
        start = time.time()
        path, used = zeta.generate(prompt, LENGTHS[length], int(steps),
                                   seed=None if seed in (None, "", -1) else int(seed),
                                   progress=progress, sound=sound, smooth=smooth, hd=hd,
                                   stage=lambda msg: progress(1.0, desc=msg))
        return path, f"Seed {used} · made in {time.time() - start:.0f}s · saved to {path}"

    with gr.Blocks(title=f"ACOpsis Zeta {VERSION}") as ui:
        gr.Markdown(f"# ACOpsis Zeta {VERSION}\nDescribe a scene and Zeta animates it.")
        prompt = gr.Textbox(label="Prompt", lines=3,
                            placeholder="Peter and Brian sit at the kitchen table arguing about pizza.")
        with gr.Row():
            sound = gr.Checkbox(value=foley.prepared(), interactive=foley.prepared(),
                                label="Sound" if foley.prepared() else "Sound (not installed)")
            smooth = gr.Checkbox(value=False, label="Smooth (48fps)")
            hd = gr.Checkbox(value=False, label="HD (1024×576)")
        go = gr.Button("Generate", variant="primary")
        with gr.Accordion("Settings", open=False):
            length = gr.Radio(list(LENGTHS), value="3 seconds", label="Length")
            steps = gr.Slider(10, 50, value=40, step=1, label="Steps (more = cleaner, slower)")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random; reuse one to remake a video)")
        video = gr.Video(label="Result", autoplay=True, loop=True)
        info = gr.Markdown()
        inputs = [prompt, length, steps, seed, sound, smooth, hd]
        go.click(run, inputs, [video, info])
        prompt.submit(run, inputs, [video, info])
    return ui


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if models_cached():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise SystemExit("No NVIDIA GPU visible to PyTorch.")
    zeta = Zeta()
    build_ui(zeta).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", inbrowser=True, show_error=True, allowed_paths=[str(OUTPUTS)])
