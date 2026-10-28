#!/usr/bin/env python3
"""
ACOpsis: collect a small forum corpus and train a tiny GPT-NeoX model from scratch.

    python train.py collect --boards g sci --max-threads 10
    python train.py train   --epochs 3
    python train.py generate --prompt "The problem with"
    python train.py generate -i          # interactive prompt loop
    python train.py stats                # corpus summary
    python train.py train --dry-run      # preview config + memory, no training

Dependencies:  pip install requests beautifulsoup4 numpy torch transformers datasets rich
(`rich` is optional; without it you get plain output.)
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH = True
except ImportError:  # graceful fallback to plain output
    RICH = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_CONFIG = "EleutherAI/pythia-160m"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "acopsis_corpus.jsonl"
THREADS_FILE = DATA_DIR / "acopsis_threads.jsonl"
OUTPUT_DIR = Path("checkpoints/acopsis-160m")

# Start small. Only collect from boards you have reviewed.
DEFAULT_BOARDS = ["g", "sci"]

API = "https://a.4cdn.org"
HEADERS = {"User-Agent": "ACOpsisResearch/0.1"}

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
# ">>12345678", optionally followed by "(OP)" / "(You)". Not cross-board links.
QUOTE_RE = re.compile(r"(?<!>)>>(\d+)(?:\s*\((?:OP|You)\))?")
# ">>>/g/" or ">>>/g/12345"
CROSSLINK_RE = re.compile(r">>>/\w+/(?:\d+)?")
BOARD_RE = re.compile(r"^[a-z0-9]{1,10}$")


# --------------------------------------------------------------------------- #
# Console UI (rich if available, plain text otherwise)
# --------------------------------------------------------------------------- #

console = Console() if RICH else None


def _say(icon: str, style: str, msg: str) -> None:
    if RICH:
        console.print(f"[{style}]{icon}[/] {escape(str(msg))}", highlight=False)
    else:
        print(f"{icon} {msg}")


def info(msg: str) -> None:
    _say("•", "cyan", msg)


def ok(msg: str) -> None:
    _say("✔", "green", msg)


def warn(msg: str) -> None:
    _say("!", "yellow", msg)


def error(msg: str) -> None:
    _say("✘", "bold red", msg)


def banner(title: str, subtitle: str = "") -> None:
    if RICH:
        body = f"[bold]{escape(title)}[/]"
        if subtitle:
            body += f"\n[dim]{escape(subtitle)}[/]"
        console.print(Panel.fit(body, border_style="magenta", padding=(0, 2)))
    else:
        print(f"\n=== {title} ===")
        if subtitle:
            print(subtitle)


def section(title: str) -> None:
    if RICH:
        console.rule(f"[bold]{escape(title)}[/]", style="dim")
    else:
        print(f"\n--- {title} ---")


def panel(text: str, title: str = "", style: str = "cyan") -> None:
    if RICH:
        console.print(Panel(
            escape(text), title=escape(title) if title else None,
            title_align="left", border_style=style,
        ))
    else:
        print(f"\n[{title}]\n{text}\n" if title else f"\n{text}\n")


def kv_table(title: str, rows: list[tuple[str, object]]) -> None:
    if RICH:
        table = Table(title=title, box=box.ROUNDED, show_header=False, title_style="bold")
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        for key, value in rows:
            table.add_row(escape(str(key)), escape(str(value)))
        console.print(table)
    else:
        print(f"\n{title}")
        for key, value in rows:
            print(f"  {key:<22} {value}")


class _PlainBar:
    def __init__(self, desc: str, total: int):
        self.desc, self.total, self.n = desc, total, 0

    def advance(self, n: int = 1, status: str = "") -> None:
        self.n += n
        print(f"  [{self.n}/{self.total}] {self.desc} {status}".rstrip())

    def status(self, text: str) -> None:
        pass


class _RichBar:
    def __init__(self, progress: "Progress", task_id):
        self._p, self._t = progress, task_id

    def advance(self, n: int = 1, status: str | None = None) -> None:
        fields = {} if status is None else {"status": escape(status)}
        self._p.update(self._t, advance=n, **fields)

    def status(self, text: str) -> None:
        self._p.update(self._t, status=escape(text))


@contextmanager
def progress(desc: str, total: int):
    if not RICH:
        yield _PlainBar(desc, total)
        return
    with Progress(
        SpinnerColumn(style="magenta"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        console=console,
    ) as prog:
        task = prog.add_task(desc, total=total, status="")
        yield _RichBar(prog, task)


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

def parse_comment(raw: str | None) -> tuple[str, list[int]]:
    """Turn an API HTML comment into (clean_text, ids_of_posts_it_replies_to)."""
    if not raw:
        return "", []

    soup = BeautifulSoup(raw, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for wbr in soup.find_all("wbr"):  # 4chan inserts these inside long URLs
        wbr.decompose()

    text = soup.get_text()  # entities are already decoded by the parser

    reply_ids = sorted({int(m) for m in QUOTE_RE.findall(text)})

    text = CROSSLINK_RE.sub("", text)
    text = QUOTE_RE.sub("", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)

    lines = (re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln), reply_ids


def clean_post(raw: str | None, min_length: int = 20) -> str:
    """Convenience wrapper: cleaned text, or '' if shorter than min_length."""
    text, _ = parse_comment(raw)
    return text if len(text) >= min_length else ""


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

class FourChanAPI:
    """Polite client: rate-limited, with retry/backoff on transient errors."""

    def __init__(self, delay: float = 1.1):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        retry = Retry(
            total=4,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last = 0.0

    def get_json(self, path: str):
        """Return parsed JSON, or None if the resource no longer exists (404)."""
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)

        response = self.session.get(f"{API}/{path.lstrip('/')}", timeout=20)
        self._last = time.monotonic()

        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def load_jsonl(path: Path):
    """Yield dict records from a JSONL file, skipping blank/corrupt lines."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def process_thread(board, thread_id, data, seen_texts, corpus, min_length):
    """Parse one thread. Writes new training posts; returns (posts, new_count)."""
    raw_posts = [p for p in data.get("posts", []) if p.get("no") is not None]
    raw_posts.sort(key=lambda p: int(p["no"]))

    thread_posts, new_count = [], 0
    for post in raw_posts:
        post_id = int(post["no"])
        text, reply_ids = parse_comment(post.get("com"))

        # Thread archive keeps every post (even image-only) to preserve structure.
        thread_posts.append({
            "post_id": post_id,
            "is_op": post_id == thread_id,
            "text": text,
            "reply_to": reply_ids,
        })

        # Training corpus keeps only substantial, de-duplicated text.
        if len(text) >= min_length and text not in seen_texts:
            corpus.write(json.dumps(
                {"text": text, "board": board, "post_id": post_id},
                ensure_ascii=False,
            ) + "\n")
            seen_texts.add(text)
            new_count += 1

    return thread_posts, new_count


def collect(boards, max_threads, min_length=20, delay=1.1):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    banner("ACOpsis · collect", f"boards: {', '.join('/' + b + '/' for b in boards)}")

    seen_texts = {
        o["text"].strip() for o in load_jsonl(DATA_FILE)
        if isinstance(o.get("text"), str) and o["text"].strip()
    }
    seen_threads = set()
    for o in load_jsonl(THREADS_FILE):
        try:
            seen_threads.add((o["board"], int(o["thread_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    info(f"Resuming with {len(seen_texts):,} known posts and {len(seen_threads):,} archived threads")

    api = FourChanAPI(delay=delay)
    rng = random.Random()
    stats = {"threads": 0, "posts": 0, "failed": 0, "gone": 0}

    try:
        with (
            DATA_FILE.open("a", encoding="utf-8") as corpus,
            THREADS_FILE.open("a", encoding="utf-8") as archive,
        ):
            for board in boards:
                section(f"/{board}/")
                try:
                    catalog = api.get_json(f"{board}/catalog.json")
                except (requests.RequestException, ValueError) as exc:
                    error(f"Catalog failed for /{board}/: {exc}")
                    continue
                if catalog is None:
                    warn(f"/{board}/ does not exist")
                    continue

                ids = [
                    int(t["no"])
                    for page in catalog
                    for t in page.get("threads", [])
                    if t.get("no") is not None
                ]
                candidates = [t for t in ids if (board, t) not in seen_threads]
                rng.shuffle(candidates)
                candidates = candidates[:max_threads]

                if not candidates:
                    info(f"No new threads on /{board}/")
                    continue

                with progress(f"/{board}/", len(candidates)) as bar:
                    for thread_id in candidates:
                        try:
                            data = api.get_json(f"{board}/thread/{thread_id}.json")
                        except (requests.RequestException, ValueError) as exc:
                            warn(f"Thread {thread_id} failed: {exc}")
                            stats["failed"] += 1
                            bar.advance()
                            continue

                        if data is None:
                            stats["gone"] += 1
                            bar.advance(status=f"thread {thread_id} gone")
                            continue

                        posts, new_posts = process_thread(
                            board, thread_id, data, seen_texts, corpus, min_length
                        )
                        archive.write(json.dumps(
                            {"board": board, "thread_id": thread_id, "posts": posts},
                            ensure_ascii=False,
                        ) + "\n")
                        corpus.flush()
                        archive.flush()

                        seen_threads.add((board, thread_id))
                        stats["threads"] += 1
                        stats["posts"] += new_posts
                        bar.advance(status=f"thread {thread_id} · {len(posts)} posts · +{new_posts} new")
    except KeyboardInterrupt:
        warn("Interrupted. Everything collected so far has been saved.")

    kv_table("Collection summary", [
        ("New threads archived", f"{stats['threads']:,}"),
        ("New training posts", f"{stats['posts']:,}"),
        ("Threads gone (404)", stats["gone"]),
        ("Threads failed", stats["failed"]),
        ("Training corpus", DATA_FILE),
        ("Thread archive", THREADS_FILE),
    ])


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

BASE_SOURCES = {
    # Big general-English corpora, streamed so only what you need is downloaded.
    "fineweb-edu": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "field": "text"},
    "tinystories": {"path": "roneneldan/TinyStories", "name": None, "field": "text"},
    "wikitext": {"path": "Salesforce/wikitext", "name": "wikitext-103-raw-v1", "field": "text"},
}


def holdout(items, frac: float, rng: random.Random):
    """Shuffle and split into (train, held_out). No hold-out if too few items."""
    items = list(items)
    rng.shuffle(items)
    k = max(1, round(len(items) * frac)) if len(items) >= 10 else 0
    return items[k:], items[:k]


def encode_docs(tokenizer, texts: list[str]):
    """Tokenize documents; each one ends with EOS. Returns a list of int32 arrays."""
    import numpy as np

    eos = tokenizer.eos_token_id
    docs = []
    for i in range(0, len(texts), 2000):
        for ids in tokenizer(texts[i:i + 2000], add_special_tokens=False)["input_ids"]:
            docs.append(np.asarray(ids + [eos], dtype=np.int32))
    return docs


def pack_blocks(docs, block_size: int):
    """Join tokenized docs and cut them into a (n_blocks, block_size) array."""
    import numpy as np

    if not len(docs):
        return np.empty((0, block_size), dtype=np.int32)
    stream = np.concatenate(docs)
    n = stream.size // block_size
    return stream[: n * block_size].reshape(n, block_size)


def _base_texts(spec: dict, cache: Path):
    """Yield base-corpus documents: the on-disk cache first, then the network (appending to
    the cache as we go, so later runs and larger budgets only download what's missing)."""
    last = -1
    for rec in load_jsonl(cache):
        if isinstance(rec.get("text"), str):
            last = max(last, int(rec.get("i", last)))
            yield rec["text"]

    from datasets import load_dataset

    ds = load_dataset(spec["path"], name=spec["name"], split="train", streaming=True)
    if last >= 0:
        ds = ds.skip(last + 1)
    with cache.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ds, start=last + 1):
            text = (row.get(spec["field"]) or "").strip()
            if len(text) < spec.get("min_chars", 200) or text.startswith("="):
                continue
            text = text[:20000]
            f.write(json.dumps({"i": i, "text": text}, ensure_ascii=False) + "\n")
            yield text


def fetch_base_tokens(source: str, budget: int, tokenizer):
    """Return a flat int32 token stream (docs joined by EOS) of up to `budget` tokens."""
    import numpy as np

    spec = BASE_SOURCES.get(source.lower()) or {"path": source, "name": None, "field": "text"}
    slug = re.sub(r"[^\w.-]+", "_", source.lower())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"base_{slug}.jsonl"

    parts, total, batch = [], 0, []

    def flush():
        nonlocal total
        added = 0
        for arr in encode_docs(tokenizer, batch):
            parts.append(arr)
            added += arr.size
        batch.clear()
        total += added
        return added

    with progress(f"Base corpus · {source}", budget) as bar:
        for text in _base_texts(spec, cache):
            batch.append(text)
            if len(batch) >= 256:
                bar.advance(flush())
                if total >= budget:
                    break
        if batch:
            bar.advance(flush())

    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(parts)[:budget]


def patch_attnres_for_training(model) -> bool:
    """Make KLM-style 'Block Attention Residual' source buffers autograd-safe.

    MinimaLabs/KLM-1.5's modeling code writes every layer output *in place* into one
    preallocated (B, T, layers, dim) buffer. That is a fine decoding trick, but with
    gradients on, a slice of the buffer has already been saved for backward, so the next
    write raises "modified by an inplace operation". While grad is enabled we keep the
    sources in lists and stack them instead; decoding (no_grad) keeps the fast original.
    Returns True if a patch was applied.
    """
    import torch

    for module in model.modules():
        forward = getattr(type(module), "forward", None)
        namespace = getattr(forward, "__globals__", None)
        if namespace and "AttnResSources" in namespace:
            break
    else:
        return False

    orig = namespace["AttnResSources"]
    if getattr(orig, "_autograd_safe", False):
        return False

    class SafeSources(orig):
        _autograd_safe = True

        def __init__(self, first, capacity, eps):
            self._safe = torch.is_grad_enabled()
            if not self._safe:
                super().__init__(first, capacity, eps)
                return
            self.eps, self.n, self._dtype = eps, 0, first.dtype
            self._vals, self._norms, self._stack = [], [], None
            self.add(first)

        def add(self, v):
            if not self._safe:
                return super().add(v)
            v = v.to(self._dtype)
            self._vals.append(v)
            self._norms.append(self._norm(v))
            self._stack = None
            self.n += 1

        def mix(self, query, partial=None):
            if not self._safe:
                return super().mix(query, partial)
            if self._stack is None:  # shared by the mixer and FFN mixes of one layer
                self._stack = (torch.stack(self._norms, 2), torch.stack(self._vals, 2))
            normed, values = self._stack
            n = self.n
            with torch.autocast(query.device.type, enabled=False):
                logits = normed @ query
                if partial is not None:
                    logits = torch.cat([logits, (self._norm(partial) @ query).unsqueeze(-1)], -1)
                weights = logits.softmax(-1)
            out = (weights[..., :n, None].to(values.dtype) * values).sum(-2)
            if partial is not None:
                out = out + weights[..., n:].to(partial.dtype) * partial
            return out

    namespace["AttnResSources"] = SafeSources
    return True


def reinit_weights(model, std: float) -> None:
    """GPT-style N(0, std) init for Linear/Embedding. Some remote-code models ship a
    no-op `_init_weights` (their weights always come from a checkpoint), which leaves
    tied embeddings at N(0, 1) and gives enormous initial logits when training from scratch."""
    import torch.nn as nn

    count = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
            count += 1
    info(f"Re-initialized {count} Linear/Embedding layers with N(0, {std})")


def train(a):
    import torch
    from transformers import (
        AutoConfig,
        EarlyStoppingCallback,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    banner("ACOpsis · train", f"{a.model} architecture · random init · from scratch")
    set_seed(a.seed)
    rng = random.Random(a.seed)

    import numpy as np

    # ---- forum data ------------------------------------------------------- #
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Corpus not found: {DATA_FILE}. Run `collect` first.")

    forum = list(dict.fromkeys(
        o["text"].strip() for o in load_jsonl(DATA_FILE)
        if isinstance(o.get("text"), str) and o["text"].strip()
    ))
    info(f"Loaded {len(forum):,} unique forum posts")
    if len(forum) < 20:
        raise ValueError("Too few forum posts. Run `collect` with more --max-threads first.")
    forum_train, forum_eval = holdout(forum, 0.1, rng)  # split BEFORE any repetition

    # ---- tokenizer / model config ----------------------------------------- #
    try:
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
        config = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)  # architecture only
    except ValueError as exc:
        if "trust_remote_code" in str(exc):
            error(f"{a.model} needs custom code that your transformers install doesn't include.")
            info("Option 1: pip install -U transformers  (newer versions may support it natively)")
            info(f"Option 2: review the code at https://hf.co/{a.model}, then re-run with --trust-remote-code")
            raise SystemExit(1)
        raise
    # Training doesn't need KV/conv caches, and in-place cache updates inside
    # hybrid blocks can break autograd ("modified by an inplace operation").
    config.use_cache = False
    apply_overrides(config, a.set)
    if tokenizer.pad_token is None:  # Pythia ships without one
        tokenizer.pad_token = tokenizer.eos_token
    block_size = min(a.block_size, config.max_position_embeddings)

    # ---- token budgets: base corpus (fluency) + forum (voice) -------------- #
    forum_docs = encode_docs(tokenizer, forum_train)
    forum_tokens = int(sum(d.size for d in forum_docs))
    forum_eval_blocks = pack_blocks(encode_docs(tokenizer, forum_eval), block_size)
    if forum_tokens < block_size:
        raise ValueError("Forum text is shorter than one block. Collect more data.")

    reps = max(1, a.forum_repeat)
    share = min(max(a.forum_share, 0.05), 1.0)
    empty = np.empty((0, block_size), dtype=np.int32)
    base_eval = base_replay = base_warm = empty

    if a.base_corpus.lower() != "none" and a.base_tokens > 0:
        replay = 0 if share >= 1 else int(forum_tokens * reps * (1 - share) / share)
        if replay > a.base_tokens:
            warn(f"--base-tokens is too small for a {share:.0%} forum share; using all of it as replay")
            replay = a.base_tokens
        eval_tokens = max(block_size * 32, min(100_000, a.base_tokens // 100))
        try:
            stream = fetch_base_tokens(a.base_corpus, a.base_tokens + eval_tokens, tokenizer)
        except Exception as exc:  # offline, bad dataset id, etc.
            warn(f"Could not load base corpus '{a.base_corpus}' ({exc}); using forum text only")
            stream = np.empty(0, dtype=np.int32)
        base_blocks = stream[: (stream.size // block_size) * block_size].reshape(-1, block_size)
        n = len(base_blocks)
        n_eval = min(eval_tokens // block_size, n // 20)
        n_replay = min(replay // block_size, n - n_eval)
        n_warm = max(0, min((a.base_tokens - replay) // block_size, n - n_eval - n_replay))
        base_eval = base_blocks[:n_eval]                                   # held out
        base_replay = base_blocks[n_eval:n_eval + n_replay]                # mixed into forum phase
        base_warm = base_blocks[n_eval + n_replay:n_eval + n_replay + n_warm]  # phase 1

    # Forum epochs get a fresh document order each time, so block boundaries differ.
    forum_blocks = np.concatenate([
        pack_blocks([forum_docs[i] for i in rng.sample(range(len(forum_docs)), len(forum_docs))], block_size)
        for _ in range(reps)
    ])
    nprng = np.random.default_rng(a.seed)
    phase_two = np.concatenate([forum_blocks, base_replay])
    phase_two = phase_two[nprng.permutation(len(phase_two))]
    phase_one = base_warm[nprng.permutation(len(base_warm))]
    train_blocks = np.concatenate([phase_one, phase_two])  # curriculum order
    if len(train_blocks) == 0:
        raise ValueError("No complete training blocks. Collect more text or lower --block-size.")
    real_share = forum_blocks.size / max(1, forum_blocks.size + base_replay.size)

    class BlockDataset(torch.utils.data.Dataset):
        def __init__(self, blocks):
            self.blocks = blocks

        def __len__(self):
            return len(self.blocks)

        def __getitem__(self, i):
            return {"input_ids": self.blocks[i]}

    def lm_collator(features):
        # Labels = inputs; the model shifts them internally, every token incl. EOS counts.
        ids = torch.from_numpy(np.stack([f["input_ids"] for f in features]).astype("int64"))
        return {"input_ids": ids, "labels": ids.clone()}

    train_data = BlockDataset(train_blocks)
    eval_sets = {}
    if len(forum_eval_blocks):
        eval_sets["forum"] = BlockDataset(forum_eval_blocks)
    if len(base_eval):
        eval_sets["base"] = BlockDataset(base_eval)
    has_eval = bool(eval_sets)
    primary = next(iter(eval_sets), None)  # forum if we have it
    if not has_eval:
        warn("Not enough held-out text for an eval block; evaluation disabled")

    # Count parameters on the meta device (no memory used) so --dry-run is instant.
    model = None
    try:
        with torch.device("meta"):
            n_params = sum(p.numel() for p in AutoModelForCausalLM.from_config(config, trust_remote_code=a.trust_remote_code).parameters())
    except Exception:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=a.trust_remote_code)
        n_params = sum(p.numel() for p in model.parameters())

    # ---- hyperparameters -------------------------------------------------- #
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    major = torch.cuda.get_device_capability(0)[0] if cuda else 0

    # bf16 only counts as native on Ampere (compute capability 8.0) or newer.
    # Older GPUs report "supported" but emulate it, which is slow and can break.
    precision = a.precision
    if precision == "auto":
        precision = "bf16" if major >= 8 else "fp32"
    elif precision != "fp32" and not cuda:
        warn(f"{precision} needs a CUDA GPU; falling back to fp32")
        precision = "fp32"
    elif precision == "bf16" and major < 8:
        warn(f"{gpu} has no native bf16; it will be emulated (slow, may error)")
    elif precision == "fp16":
        warn("fp16 is prone to overflow/NaN when training from scratch; prefer fp32 or bf16")
    use_bf16, use_fp16 = precision == "bf16", precision == "fp16"
    steps_per_epoch = max(1, math.ceil(math.ceil(len(train_data) / a.batch_size) / a.grad_accum))
    total_steps = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    warmup = max(1, int(0.05 * total_steps))
    interval = max(10, total_steps // 10)
    log_every = max(1, min(10, total_steps // 50))

    kv_table("Run configuration", [
        ("Parameters", f"{n_params:,} (random init, no pretrained weights)"),
        ("Memory estimate", f"≈ {n_params * 16 / 1e9:.1f} GB weights+grads+Adam, plus activations"),
        ("Device", gpu),
        ("Precision", precision + (" (auto)" if a.precision == "auto" else "")),
        ("Block size", block_size),
        ("Train blocks", f"{len(train_data):,}  ({len(base_warm):,} base warm-up → {len(phase_two):,} mixed)"),
        ("Eval blocks", ", ".join(f"{k} {len(v):,}" for k, v in eval_sets.items()) or "none"),
        ("Forum", f"{forum_tokens:,} tokens × {reps} epochs"),
        ("Base corpus", f"{a.base_corpus}: {(len(base_warm) + len(base_replay)) * block_size:,} tokens"),
        ("Forum share, final phase", f"{real_share:.0%}"),
        ("Order", "base → forum-heavy (curriculum)" if not a.no_curriculum else "fully shuffled"),
        ("Effective batch", f"{a.batch_size * a.grad_accum} sequences"),
        ("Optimizer steps", f"{total_steps:,}" + (" (--max-steps)" if a.max_steps > 0 else f" ({a.epochs} epochs)")),
        ("Learning rate", f"{a.lr:g} (cosine, {warmup} warmup steps)"),
    ])

    if a.dry_run:
        ok("Dry run complete; nothing was trained or written.")
        return

    if model is None:
        torch.manual_seed(a.seed)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=a.trust_remote_code)

    if patch_attnres_for_training(model):
        ok("Patched in-place attention-residual buffers so backward works")
    std = a.init_std if a.init_std is not None else (0.02 if a.trust_remote_code else 0.0)
    if std > 0:
        reinit_weights(model, std)

    eval_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    kwargs = dict(
        output_dir=str(a.output),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps if a.max_steps > 0 else -1,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        weight_decay=0.01,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=log_every,
        save_strategy="steps",
        save_steps=interval,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=cuda,
        logging_nan_inf_filter=False,
        disable_tqdm=RICH,  # our rich bar replaces it
        seed=a.seed,
        **{eval_key: "steps" if has_eval else "no"},
    )
    if has_eval:
        kwargs.update(
            eval_steps=interval,
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_{primary}_loss",
            greater_is_better=False,
        )
    args = TrainingArguments(**kwargs)

    # ---- trainer ---------------------------------------------------------- #
    class FiniteTrainer(Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            # Keep our curriculum order (base first, forum-heavy last) instead of shuffling.
            if a.no_curriculum:
                return super()._get_train_sampler(*args, **kwargs)
            return torch.utils.data.SequentialSampler(self.train_dataset)

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not torch.isfinite(loss).all():
                raise RuntimeError("Non-finite training loss detected; stopping.")
            return loss

    class RichProgress(TrainerCallback):
        def __init__(self):
            self._cm = self.bar = None

        def on_train_begin(self, args, state, control, **kw):
            self._cm = progress("Training", state.max_steps)
            self.bar = self._cm.__enter__()

        def on_step_end(self, args, state, control, **kw):
            self.bar.advance()

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.bar.status(f"loss {logs['loss']:.3f} · lr {logs.get('learning_rate', 0):.1e}")

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            parts = [
                f"{(k[5:-5] or 'eval')} {v:.3f} (ppl {math.exp(min(v, 20)):.1f})"
                for k, v in (metrics or {}).items()
                if k.startswith("eval_") and k.endswith("_loss")
            ]
            if parts:
                info(f"step {state.global_step}: eval loss · " + " · ".join(parts))

        def close(self):
            if self._cm is not None:
                self._cm.__exit__(None, None, None)
                self._cm = None

    class SampleCallback(TrainerCallback):
        """Print a text sample after every evaluation so you can watch it learn."""

        def on_evaluate(self, args, state, control, model=None, **kw):
            if model is None:
                return
            try:
                panel(sample_text(model, tokenizer, a.sample_prompt, 80),
                      f"sample @ step {state.global_step}", "magenta")
            except Exception as exc:  # never let sampling kill a run
                warn(f"Sampling failed: {exc}")

    callback = RichProgress()
    callbacks = [callback] if RICH else []
    if a.sample_prompt is not None:
        callbacks.append(SampleCallback())
    if a.patience > 0:
        if has_eval:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=a.patience))
        else:
            warn("--patience needs evaluation data; ignoring")

    trainer = FiniteTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_sets or None,
        data_collator=lm_collator,
        callbacks=callbacks,
    )
    # Our blocks are equal-sized, so the plain mean loss is correct. Without this, models
    # whose forward takes **kwargs skip Trainer's grad-accumulation scaling (inflated loss logs).
    trainer.model_accepts_loss_kwargs = False
    if not a.no_curriculum and not hasattr(Trainer, "_get_train_sampler"):
        warn("This transformers version has no _get_train_sampler hook; data will be shuffled")

    resume = None
    if a.resume:
        resume = get_last_checkpoint(str(a.output)) if a.output.exists() else None
        if resume:
            info(f"Resuming from {resume}")
        else:
            warn("No checkpoint found; starting fresh")

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "run_config.json").write_text(json.dumps({
        "started": datetime.now().isoformat(timespec="seconds"),
        "args": vars(a),
        "parameters": n_params,
        "block_size": block_size,
        "train_blocks": len(train_data),
        "eval_blocks": {k: len(v) for k, v in eval_sets.items()},
        "forum_posts": len(forum),
        "forum_tokens": forum_tokens,
        "base_warmup_blocks": len(base_warm),
        "base_replay_blocks": len(base_replay),
        "forum_share_final_phase": round(real_share, 3),
        "total_steps": total_steps,
    }, indent=2, default=str), encoding="utf-8")

    if a.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        warn("Anomaly detection is on: much slower, but errors will point at the exact op")

    section("Training")
    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        warn("Interrupted; saving current weights")
    finally:
        callback.close()

    trainer.save_model(str(a.output))
    tokenizer.save_pretrained(str(a.output))

    rows = [("Saved to", a.output)]
    if has_eval:
        for key, val in trainer.evaluate().items():
            if key.startswith("eval_") and key.endswith("_loss"):
                name = key[5:-5]
                rows += [(f"Eval loss ({name})", f"{val:.3f}"),
                         (f"Perplexity ({name})", f"{math.exp(min(val, 20)):.1f}")]
    kv_table("Done", rows)
    ok(f"Try it:  python train.py generate --prompt \"The problem with\"")


# --------------------------------------------------------------------------- #
# Corpus stats
# --------------------------------------------------------------------------- #

def stats(a):
    banner("ACOpsis · stats", str(DATA_FILE))
    rows = [o for o in load_jsonl(DATA_FILE) if isinstance(o.get("text"), str)]
    if not rows:
        warn(f"No corpus at {DATA_FILE}. Run `collect` first.")
        return

    texts = [o["text"] for o in rows]
    lengths = sorted(len(t) for t in texts)
    n, chars = len(lengths), sum(lengths)

    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
        total = 0
        for i in range(0, n, 2000):
            total += sum(len(x) for x in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"])
        tokens = f"{total:,} (exact, {a.model})"
    else:
        tokens = f"≈ {chars // 4:,} (estimate; pass --model for exact)"

    boards = Counter(o.get("board", "unknown") for o in rows)
    top = max(boards.values())
    board_rows = [
        (f"/{b}/", f"{c:,}  " + "█" * max(1, round(24 * c / top)))
        for b, c in boards.most_common()
    ]

    kv_table("Corpus", [
        ("Posts", f"{n:,}"),
        ("Unique posts", f"{len(set(texts)):,}"),
        ("Threads archived", f"{sum(1 for _ in load_jsonl(THREADS_FILE)):,}"),
        ("Characters", f"{chars:,}"),
        ("Tokens", tokens),
        ("Length (median / p95 / max)", f"{lengths[n // 2]:,} / {lengths[min(n - 1, int(n * 0.95))]:,} / {lengths[-1]:,}"),
        ("File size", f"{DATA_FILE.stat().st_size / 1e6:.2f} MB"),
    ])
    kv_table("Posts per board", board_rows)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def sample_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Sample a continuation. Safe to call mid-training (restores train mode)."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt or tokenizer.eos_token, return_tensors="pt").to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                use_cache=True,  # config has it off for training; sampling wants it on
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if was_training:
            model.train()
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not a.output.exists():
        raise FileNotFoundError(f"No model at {a.output}. Run `train` first.")
    if a.seed is not None:
        set_seed(a.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    banner("ACOpsis · generate", f"{a.output} · {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(a.output), trust_remote_code=a.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(str(a.output), trust_remote_code=a.trust_remote_code).to(device).eval()

    def run(prompt):
        panel(sample_text(model, tokenizer, prompt, a.max_new_tokens, a.temperature, a.top_p), "output")

    if not a.interactive:
        run(a.prompt)
        return

    info("Interactive: type a prompt + Enter. Commands: /temp X, /len N, /quit (or Ctrl-D).")
    while True:
        try:
            prompt = console.input("[bold magenta]»[/] ") if RICH else input("» ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = prompt.strip()
        if cmd in ("", "/quit", "/exit"):
            break
        if cmd.startswith(("/temp ", "/len ")):
            name, _, value = cmd.partition(" ")
            try:
                if name == "/temp":
                    a.temperature = max(0.05, float(value))
                    ok(f"temperature = {a.temperature}")
                else:
                    a.max_new_tokens = max(1, int(value))
                    ok(f"max new tokens = {a.max_new_tokens}")
            except ValueError:
                warn(f"Bad value: {value!r}")
            continue
        run(prompt)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def board_name(value: str) -> str:
    value = value.strip("/ ").lower()
    if not BOARD_RE.match(value):
        raise argparse.ArgumentTypeError(f"invalid board name: {value!r}")
    return value


def model_id(value: str) -> str:
    """Accept a repo id or a huggingface.co URL; map -GGUF repos to the HF-format repo."""
    v = value.strip()
    v = re.sub(r"^https?://(www\.)?huggingface\.co/", "", v)
    v = re.sub(r"/(tree|blob|resolve)/.*$", "", v).strip("/")
    if v.lower().endswith("-gguf"):
        fixed = v[: -len("-gguf")]
        warn(f"{v} is a GGUF (inference-only) repo with no config/tokenizer for transformers; using {fixed}")
        v = fixed
    if not re.match(r"^[\w.-]+/[\w.-]+$", v) and not Path(v).exists():
        raise argparse.ArgumentTypeError(f"expected 'namespace/repo' or a local folder, got {value!r}")
    return v


def apply_overrides(config, pairs):
    """Apply KEY=VALUE overrides (values parsed as JSON) to shrink/reshape an architecture."""
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        if not hasattr(config, key):
            warn(f"config has no '{key}'; setting it anyway")
        setattr(config, key, value)
        info(f"override: {key} = {value!r}")

    layers = getattr(config, "num_hidden_layers", None)
    types = getattr(config, "layer_types", None)
    if pairs and isinstance(types, list) and layers and len(types) != layers:
        config.layer_types = (types * math.ceil(layers / len(types)))[:layers]
        warn(f"layer_types resized to {layers} entries; check any other per-layer settings")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("collect", help="download threads into the corpus")
    c.add_argument("--boards", nargs="+", type=board_name, default=DEFAULT_BOARDS)
    c.add_argument("--max-threads", type=int, default=10, help="new threads per board")
    c.add_argument("--min-length", type=int, default=20, help="min characters for a training post")
    c.add_argument("--delay", type=float, default=1.1, help="seconds between requests (>= 1)")

    t = sub.add_parser("train", help="train from scratch on the corpus")
    t.add_argument("--model", type=model_id, default=MODEL_CONFIG,
                   help="HF repo to copy the architecture/tokenizer from (weights are NOT loaded), "
                        "e.g. LiquidAI/LFM3-1.7B")
    t.add_argument("--epochs", type=int, default=1,
                   help="passes over the whole curriculum (forum repetition is --forum-repeat)")
    t.add_argument("--trust-remote-code", action="store_true",
                   help="allow executing custom modeling code from the model repo (inspect it first)")
    t.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                   help="override config values to shrink the model, e.g. "
                        "--set hidden_size=512 num_hidden_layers=8 (check with --dry-run)")
    t.add_argument("--init-std", type=float, default=None,
                   help="re-init Linear/Embedding weights with N(0, std); default 0.02 for "
                        "--trust-remote-code models (many skip init), 0 = keep model's own")
    t.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                   help="auto = bf16 on Ampere+ GPUs, else fp32 (never fp16)")
    t.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    t.add_argument("--detect-anomaly", action="store_true",
                   help="debug autograd errors (slow): reports the forward op that caused them")
    t.add_argument("--dry-run", action="store_true", help="show config + memory estimate, then exit")
    t.add_argument("--sample-prompt", default=None, help="print a generated sample after every eval")
    t.add_argument("--patience", type=int, default=0, help="early-stop after N evals without improvement")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--block-size", type=int, default=256)
    t.add_argument("--base-corpus", default="fineweb-edu",
                   help="general English that teaches fluency: fineweb-edu | tinystories | wikitext | "
                        "none | any HF dataset id with a 'text' column (streamed + cached in data/)")
    t.add_argument("--base-tokens", type=int, default=20_000_000,
                   help="tokens of base corpus to use (more = more coherent, slower)")
    t.add_argument("--forum-repeat", type=int, default=3,
                   help="passes over the forum text in the final phase")
    t.add_argument("--forum-share", type=float, default=0.7,
                   help="fraction of final-phase tokens that are forum text (rest is base replay)")
    t.add_argument("--no-curriculum", action="store_true",
                   help="shuffle base and forum together instead of base-first ordering")
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    t.add_argument("--output", type=Path, default=OUTPUT_DIR)

    g = sub.add_parser("generate", help="sample text from the trained model")
    g.add_argument("--prompt", default="The ")
    g.add_argument("--trust-remote-code", action="store_true",
                   help="allow executing custom modeling code from the model repo (inspect it first)")
    g.add_argument("-i", "--interactive", action="store_true", help="prompt loop; model loads once")
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--seed", type=int, default=None, help="make sampling reproducible")
    g.add_argument("--max-new-tokens", type=int, default=120)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--output", type=Path, default=OUTPUT_DIR)

    st = sub.add_parser("stats", help="summarize the collected corpus")
    st.add_argument("--trust-remote-code", action="store_true",
                   help="allow executing custom modeling code from the model repo (inspect it first)")
    st.add_argument("--model", type=model_id, default=None, help="tokenizer repo for exact token counts")

    return parser


def main():
    a = build_parser().parse_args()
    if a.mode == "collect":
        collect(a.boards, max(1, a.max_threads), a.min_length, max(1.0, a.delay))
    elif a.mode == "train":
        train(a)
    elif a.mode == "stats":
        stats(a)
    else:
        generate(a)


if __name__ == "__main__":
    main()