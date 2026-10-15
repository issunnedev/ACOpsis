"""ACOpsis - command-line home for the ACOpsis models.

    acopsis                                   start the interactive prompt
    acopsis zeta prompt "Peter on the couch"  run one command and exit

Inside the prompt: `help` lists every command.
"""
import logging
import os
import shlex
import sys
import time
import warnings
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "zeta" / "app"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODELS = {
    "zeta": "text-to-video, Family Guy style. 2-4 second clips, about 40s each.",
    "Epsilon": "Zeta fork with optimization, YouTube fetch, and 4K resolution",
    "Omega": "More powerful and resource-demanding fork of Zeta",
}
HELP = """\
commands:
  models                          list the models
  model <name>                    load a model (e.g. model zeta)
  epsilon                         Start epsilon internal manager
  zeta prompt "<text>"            make a video
      options: --length 2|3|4     seconds (default 3)
               --steps N          10-50, fewer = faster but rougher (default 40)
               --seed N           reuse a seed to remake or vary a video
  again                           re-run the last prompt with a new seed
  open                            play the last video
  videos                          open the videos folder
  unload                          free the GPU
  help                            this list
  exit                            leave ACOpsis"""

LENGTHS = {"2": 49, "3": 73, "4": 97}

EPSILON = ROOT / "ε-Epsilon" / "Manager.py"

class Shell: 
    def __init__(self):
        self.loaded = {}        # name -> model
        self.current = None
        self.last_video = None
        self.last_args = None

    # ---- commands -------------------------------------------------------
    def cmd_help(self, args):
        print(HELP)

    def cmd_models(self, args):
        for name, desc in MODELS.items():
            mark = "*" if name == self.current else " "
            state = "loaded" if name in self.loaded else "not loaded"
            print(f" {mark} {name:<8} {desc} ({state})")

    def cmd_model(self, args):
        if len(args) != 1 or args[0] not in MODELS:
            print(f"usage: model <name>   (models: {', '.join(MODELS)})")
            return
        self.load(args[0])


        if "zeta" in self.loaded:
            self.cmd_unload([])

        import subprocess

        print("Starting Epsilon manager...")

        try:
            subprocess.run(
                [sys.executable, str(EPSILON), *args],
                cwd=str(EPSILON.parent),
                check=False,
            )
        except KeyboardInterrupt:
            print("\nEpsilon stopped.")
        except OSError as e:
            print(f"Couldn't start Epsilon: {e}")

    def cmd_unload(self, args):
        if not self.loaded:
            print("nothing loaded")
            return
        import gc
        import torch
        self.loaded.clear()
        self.current = None
        gc.collect()
        torch.cuda.empty_cache()
        print("unloaded - GPU freed")

    def cmd_open(self, args):
        if not self.last_video:
            print("no video yet this session")
            return
        os.startfile(self.last_video)

    def cmd_videos(self, args):
        from zeta_app import OUTPUTS
        OUTPUTS.mkdir(exist_ok=True)
        os.startfile(OUTPUTS)

    def cmd_again(self, args):
        if not self.last_args:
            print("nothing to repeat yet")
            return
        self.cmd_zeta(_drop_option(self.last_args, "--seed"))

    def cmd_zeta(self, args):
        if not args or args[0] != "prompt" or len(args) < 2:
            print('usage: zeta prompt "<text>" [--length 2|3|4] [--steps N] [--seed N]')
            return
        try:
            text, opts = _parse_prompt(args[1:])
        except ValueError as e:
            print(e)
            return
        zeta = self.load("zeta")
        if zeta is None:
            return
        self.last_args = list(args)
        steps = opts["steps"]
        start = time.time()

        def progress(frac, desc=""):
            done = round(frac * steps)
            bar = "#" * (done * 24 // steps) + "-" * (24 - done * 24 // steps)
            print(f"\r  generating [{bar}] {done}/{steps}", end="", flush=True)

        print("  reading prompt...", end="", flush=True)
        try:
            path, seed = zeta.generate(text, LENGTHS[opts["length"]], steps,
                                       seed=opts["seed"], progress=progress)
        except KeyboardInterrupt:
            print("\n  cancelled")
            return
        print(f"\r  saved {Path(path).name}  (seed {seed}, {time.time() - start:.0f}s)" + " " * 20)
        self.last_video = path
        os.startfile(path)

    # ---- plumbing -------------------------------------------------------
    def load(self, name):
        if name in self.loaded:
            self.current = name
            return self.loaded[name]
        print(f"loading {name}... (about 30 seconds)")
        start = time.time()
        try:
            import zeta_app
            if zeta_app.models_cached():   # skip the Hub; otherwise let it download what's missing
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            self.loaded[name] = zeta_app.Zeta()
        except Exception as e:
            print(f"couldn't load {name}: {e}")
            return None
        self.current = name
        print(f"{name} ready ({time.time() - start:.0f}s)")
        return self.loaded[name]

    def run(self, line):
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as e:
            print(f"couldn't read that: {e} (check your quotes)")
            return True
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("exit", "quit"):
            return False
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            print(f"unknown command '{cmd}' - type help")
        else:
            handler(args)
        return True

    def repl(self):
        print("ACOpsis - type 'help' for commands")
        while True:
            try:
                line = input(f"acopsis{f' [{self.current}]' if self.current else ''}> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not self.run(line):
                break


def _parse_prompt(tokens):
    opts = {"length": "3", "steps": 40, "seed": None}
    words, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("--length", "--steps", "--seed"):
            if i + 1 >= len(tokens):
                raise ValueError(f"{t} needs a value")
            val = tokens[i + 1]
            if t == "--length":
                if val.rstrip("s") not in LENGTHS:
                    raise ValueError("--length must be 2, 3 or 4")
                opts["length"] = val.rstrip("s")
            elif not val.lstrip("-").isdigit():
                raise ValueError(f"{t} must be a whole number")
            elif t == "--steps":
                opts["steps"] = max(10, min(50, int(val)))
            else:
                opts["seed"] = int(val)
            i += 2
        else:
            words.append(t)
            i += 1
    text = " ".join(words).strip()
    if not text:
        raise ValueError('the prompt is empty - e.g. zeta prompt "Peter sits on the couch"')
    return text, opts


def _drop_option(tokens, name):
    out, skip = [], False
    for t in tokens:
        if skip:
            skip = False
        elif t == name:
            skip = True
        else:
            out.append(t)
    return out


if __name__ == "__main__":
    shell = Shell()
    if len(sys.argv) > 1:
        shell.run(shlex.join(sys.argv[1:]))
    else:
        shell.repl()
