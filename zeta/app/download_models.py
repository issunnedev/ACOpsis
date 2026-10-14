import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download, snapshot_download

from zeta_app import MODEL_ID, T5_FILE, T5_REPO

print("downloading the video model (~5GB)...", flush=True)
snapshot_download(MODEL_ID, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*",
                                            "tokenizer/*", "text_encoder/config.json"])
print("downloading the prompt reader (~4.6GB)...", flush=True)
hf_hub_download(T5_REPO, T5_FILE)
print("models ready", flush=True)
