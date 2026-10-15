import dataclasses
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

from bundle import load_bundle, save_bundle

ROOT = Path(__file__).resolve().parents[1]
SOUND = ROOT / "sound"
MODEL = "large_44k_v2"
CLIP_REPO = "apple/DFN5B-CLIP-ViT-H-14-384"
VOCODER_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"
PARTS = ("clip", "sync", "net", "vae", "vocoder")
DTYPE = torch.bfloat16
NEGATIVE = "music, singing"


def prepared():
    return all((SOUND / f"{p}.safetensors").exists() for p in PARTS) and (SOUND / "configs.json").exists()


def _build_clip(cfg):
    from mmaudio.model.utils.features_utils import patch_clip
    from open_clip.model import CLIP
    return patch_clip(CLIP(**cfg))


def _build_sync():
    from mmaudio.ext.synchformer import Synchformer
    return Synchformer()


def _build_net():
    from mmaudio.model.networks import get_my_mmaudio
    return get_my_mmaudio(MODEL)


def _build_vae():
    from mmaudio.ext.autoencoder.vae import get_my_vae
    vae = get_my_vae("44k").remove_weight_norm()
    del vae.encoder
    return vae


def _build_vocoder(cfg):
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN
    from mmaudio.ext.bigvgan_v2.env import AttrDict
    voc = BigVGAN(AttrDict(cfg), use_cuda_kernel=False)
    voc.remove_weight_norm()
    return voc


def _load_meta(build, part, *args):
    with torch.inference_mode(False):
        try:
            with torch.device("meta"):
                module = build(*args)
        except RuntimeError:
            module = build(*args)
        return load_bundle(module, SOUND / f"{part}.safetensors", "cuda")


def _download(dest, log):
    import hashlib
    import requests
    from mmaudio.utils.download_utils import links
    link = next(l for l in links if l["name"] == dest.name)

    def md5_ok():
        h = hashlib.md5()
        with open(dest, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest() == link["md5"]

    if dest.exists() and md5_ok():
        return
    log(f"  downloading {dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(link["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    if not md5_ok():
        raise RuntimeError(f"{dest.name} downloaded incorrectly - run the install again")


def _free(*modules):
    for m in modules:
        m.to("meta")
    gc.collect()
    torch.cuda.empty_cache()


def prepare(log=print):
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    from mmaudio.eval_utils import all_model_cfg
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN

    SOUND.mkdir(exist_ok=True)
    dl = SOUND / "_download"
    cfg = all_model_cfg[MODEL]
    files = {"net": dl / cfg.model_path.name, "vae": dl / cfg.vae_path.name,
             "sync": dl / cfg.synchformer_ckpt.name}

    def load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:
            return torch.load(path, map_location="cpu", weights_only=True)

    def done(part, module):
        module.to(DTYPE)
        save_bundle(module, SOUND / f"{part}.safetensors")
        log(f"  {part} ready")
        del module
        gc.collect()

    configs = {"clip": json.load(open(hf_hub_download(CLIP_REPO, "open_clip_config.json")))["model_cfg"],
               "vocoder": json.load(open(hf_hub_download(VOCODER_REPO, "config.json")))}
    if not (SOUND / "clip.safetensors").exists():
        log("  downloading the vision model (~3.8GB)...")
        clip = _build_clip(configs["clip"])
        clip.load_state_dict(load(hf_hub_download(CLIP_REPO, "open_clip_pytorch_model.bin")))
        done("clip", clip)
    for part in ("sync", "net", "vae"):
        if (SOUND / f"{part}.safetensors").exists():
            continue
        _download(files[part], log)
        if part == "sync":
            module = _build_sync()
            module.load_state_dict(load(files[part]))
        elif part == "net":
            module = _build_net()
            module.load_weights(load(files[part]))
        else:
            from mmaudio.ext.autoencoder.vae import get_my_vae
            module = get_my_vae("44k")
            module.load_state_dict(load(files[part]))
            module.remove_weight_norm()
            del module.encoder
        done(part, module)
    if not (SOUND / "vocoder.safetensors").exists():
        log("  downloading the vocoder...")
        voc = BigVGAN.from_pretrained(VOCODER_REPO, use_cuda_kernel=False)
        voc.remove_weight_norm()
        done("vocoder", voc)

    json.dump(configs, open(SOUND / "configs.json", "w"))
    shutil.rmtree(dl, ignore_errors=True)
    _drop_hub_cache(CLIP_REPO, VOCODER_REPO)
    log("  sound models ready")


def _drop_hub_cache(*repos):
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
    except Exception:
        return
    hashes = [rev.commit_hash for repo in cache.repos if repo.repo_id in repos for rev in repo.revisions]
    if hashes:
        cache.delete_revisions(*hashes).execute()


@torch.inference_mode()
def make_sound(video_path, prompt, duration, seed=0, steps=25, cfg_strength=4.5):
    import open_clip
    from mmaudio.eval_utils import load_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.sequence_config import CONFIG_44K
    from mmaudio.model.utils.features_utils import FeaturesUtils
    from torchvision.transforms import Normalize

    configs = json.load(open(SOUND / "configs.json"))
    info = load_video(Path(video_path), duration, load_all_frames=False)
    clip_frames = info.clip_frames.unsqueeze(0).to("cuda", DTYPE)
    sync_frames = info.sync_frames.unsqueeze(0).to("cuda", DTYPE)

    clip = _load_meta(_build_clip, "clip", configs["clip"])
    sync = _load_meta(_build_sync, "sync")
    watcher = FeaturesUtils.__new__(FeaturesUtils)
    torch.nn.Module.__init__(watcher)
    watcher.clip_model, watcher.synchformer = clip, sync
    watcher.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
    watcher.tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    clip_f = watcher.encode_video_with_clip(clip_frames, batch_size=40)
    sync_f = watcher.encode_video_with_sync(sync_frames, batch_size=40)
    text_f = watcher.encode_text([prompt])
    neg_f = watcher.encode_text([NEGATIVE])
    del watcher
    _free(clip, sync)

    seq = dataclasses.replace(CONFIG_44K, duration=info.duration_sec)
    net = _load_meta(_build_net, "net")
    net.update_seq_lengths(seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len)
    rng = torch.Generator("cuda").manual_seed(seed)
    x0 = torch.randn(1, net.latent_seq_len, net.latent_dim, device="cuda", dtype=DTYPE, generator=rng)
    cond = net.preprocess_conditions(clip_f, sync_f, text_f)
    empty = net.get_empty_conditions(1, negative_text_features=neg_f)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
    latents = net.unnormalize(fm.to_data(lambda t, x: net.ode_wrapper(t, x, cond, empty, cfg_strength), x0))
    _free(net)

    vae = _load_meta(_build_vae, "vae")
    vocoder = _load_meta(_build_vocoder, "vocoder", configs["vocoder"])
    audio = vocoder(vae.decode(latents.transpose(1, 2)))
    _free(vae, vocoder)
    return audio.float().cpu().numpy().reshape(-1), seq.sampling_rate


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    else:
        print(__doc__)
