import gc
import logging
import os
import shlex
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EPSILON = ROOT / "ε-Epsilon" / "Manager.py"
EPSILON_DESC = "Epsilon - Zeta fork with optimization, YouTube fetch, and 4K resolution (run: epsilon)"
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "zeta" / "app"))
sys.path.insert(0, str(ROOT / "omega" / "app"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODELS = {
    "zeta": {
        "desc": "Zeta 1.2 - Family Guy style text-to-video with sound. 2-4s clips, ~1 min each.",
        "flags": ("silent", "smooth", "hd"),
        "choices": {"length": ("2", "3", "4")},
        "numbers": {"steps": (10, 50), "seed": None},
        "defaults": {"length": "3", "steps": 40},
    },
    "omega": {
        "desc": "Omega 1.0 - high quality text-to-video with sound and a prompt enhancer. 2-5s clips.",
        "flags": ("silent", "smooth", "raw", "best"),
        "choices": {"length": ("2", "3", "5"), "res": ("480", "720")},
        "numbers": {"seed": None},
        "defaults": {"length": "5", "res": "720"},
    },
}
HELP = """\
commands:
  models                          list the models
  model <name>                    load a model (e.g. model omega)
  zeta prompt "<text>"            make a Family Guy style video (with sound)
      options: --length 2|3|4     seconds (default 3)
               --steps N          10-50, fewer = faster but rougher (default 40)
               --hd               upscale to 1024x576
  omega prompt "<text>"           make a high quality video (with sound)
      options: --length 2|3|5     seconds (default 5)
               --res 480|720      480 is faster, 720 is sharper (default 720)
               --raw              use your prompt as-is, skip the enhancer
               --best             full-quality decoder (slower: ~30s at 480p, ~3 min at 720p)
  options for both:
               --seed N           reuse a seed to remake or vary a video
               --silent           skip the sound
               --smooth           interpolate to 48fps, less choppy
  setup omega                     download and prepare Omega (one time, ~30GB)
  epsilon                         start the Epsilon manager
  again                           re-run the last prompt with a new seed
  open                            play the last video
  videos                          open the videos folder
  unload                          free the GPU
  help                            this list
  exit                            leave ACOpsis"""


class Shell:
    def __init__(self):
        self.loaded = {}
        self.current = None
        self.last_video = None
        self.last = None

    def cmd_help(self, args):
        print(HELP)

    def cmd_models(self, args):
        for name, spec in MODELS.items():
            mark = "*" if name == self.current else " "
            state = "loaded" if name in self.loaded else "not loaded"
            print(f" {mark} {name:<7} {spec['desc']} ({state})")
        print(f"   {'epsilon':<7} {EPSILON_DESC}")

    def cmd_epsilon(self, args):
        if not EPSILON.is_file():
            print(f"Epsilon manager not found: {EPSILON}")
            return
        if self.loaded:
            self._unload_all()
        _show_art("epsilon")
        print("Starting Epsilon manager...")
        try:
            subprocess.run([sys.executable, str(EPSILON), *args], cwd=str(EPSILON.parent), check=False)
        except KeyboardInterrupt:
            print("\nEpsilon stopped.")
        except OSError as e:
            print(f"Couldn't start Epsilon: {e}")

    def cmd_model(self, args):
        if len(args) != 1 or args[0] not in MODELS:
            print(f"usage: model <name>   (models: {', '.join(MODELS)})")
            return
        self.load(args[0])

    def cmd_unload(self, args):
        if not self.loaded:
            print("nothing loaded")
            return
        self._unload_all()
        print("unloaded - GPU freed")

    def cmd_open(self, args):
        if not self.last_video:
            print("no video yet this session")
            return
        os.startfile(self.last_video)

    def cmd_videos(self, args):
        folder = ROOT / (self.current or "zeta") / "outputs"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def cmd_again(self, args):
        if not self.last:
            print("nothing to repeat yet")
            return
        name, prev = self.last
        self._make(name, _drop_option(prev, "--seed"))

    def cmd_setup(self, args):
        if args != ["omega"]:
            print("usage: setup omega")
            return
        import omega_app
        from storage import data_root, fast_root
        if self.loaded:
            self._unload_all()
        print(f"setting up Omega: downloads go to {data_root()}, working files to {fast_root()}")
        try:
            omega_app.prepare(log=print)
        except Exception as e:
            print(f"setup failed: {e}")

    def cmd_zeta(self, args):
        self._make("zeta", args)

    def cmd_omega(self, args):
        self._make("omega", args)

    def _make(self, name, args):
        if not args or args[0] != "prompt" or len(args) < 2:
            print(f'usage: {name} prompt "<text>"   (type help for options)')
            return
        try:
            text, opts = _parse_prompt(args[1:], MODELS[name])
        except ValueError as e:
            print(e)
            return
        model = self.load(name)
        if model is None:
            return
        self.last = (name, list(args))
        start = time.time()
        steps = opts.get("steps", 3)

        def progress(frac, desc=""):
            done = round(frac * steps)
            bar = "#" * (done * 24 // steps) + "-" * (24 - done * 24 // steps)
            print(f"\r  generating [{bar}] {done}/{steps}" + " " * 10, end="", flush=True)

        def stage(msg):
            print(f"\r  {msg}..." + " " * 30, end="", flush=True)

        print("  reading prompt...", end="", flush=True)
        common = dict(seed=opts["seed"], progress=progress, stage=stage, sound=not opts["silent"],
                      smooth=opts["smooth"])
        try:
            if name == "zeta":
                path, seed = model.generate(text, {"2": 49, "3": 73, "4": 97}[opts["length"]], steps,
                                            hd=opts["hd"], **common)
            else:
                path, seed = model.generate(text, length=opts["length"], res=opts["res"],
                                            enhance=not opts["raw"], fast_decode=not opts["best"], **common)
        except KeyboardInterrupt:
            print("\n  cancelled")
            return
        print(f"\r  saved {Path(path).name}  (seed {seed}, {time.time() - start:.0f}s)" + " " * 20)
        if name == "omega" and not opts["raw"] and model.last_prompt:
            print(textwrap.fill(model.last_prompt, width=96, initial_indent="  prompt used: ",
                                subsequent_indent=" " * 15))
        self.last_video = path
        os.startfile(path)

    def _unload_all(self):
        import torch
        self.loaded.clear()
        self.current = None
        gc.collect()
        torch.cuda.empty_cache()

    def load(self, name):
        if name in self.loaded:
            self.current = name
            return self.loaded[name]
        if self.loaded:
            print(f"unloading {', '.join(self.loaded)} to make room...")
            self._unload_all()
        print(f"loading {name}... (about 40 seconds)")
        start = time.time()
        try:
            if name == "zeta":
                import zeta_app
                if zeta_app.models_cached():
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self.loaded[name] = zeta_app.Zeta()
            else:
                import omega_app
                self.loaded[name] = omega_app.Omega()
        except Exception as e:
            print(f"couldn't load {name}: {e}")
            return None
        self.current = name
        _show_art(name)
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


def _show_art(name):
    path = ROOT / "art" / f"{name}.txt"
    if path.exists():
        print(path.read_text(encoding="utf-8"))


def _parse_prompt(tokens, spec):
    opts = {"seed": None, "silent": False, "smooth": False, "hd": False, "raw": False, "best": False,
            **spec["defaults"]}
    valued = set(spec["choices"]) | set(spec["numbers"])
    words, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        key = t[2:] if t.startswith("--") else None
        if key in spec["flags"]:
            opts[key] = True
            i += 1
        elif key in valued:
            if i + 1 >= len(tokens):
                raise ValueError(f"{t} needs a value")
            val = tokens[i + 1].rstrip("sp")
            if key in spec["choices"]:
                if val not in spec["choices"][key]:
                    raise ValueError(f"{t} must be {' or '.join(spec['choices'][key])}")
                opts[key] = val
            elif not val.lstrip("-").isdigit():
                raise ValueError(f"{t} must be a whole number")
            elif spec["numbers"][key]:
                low, high = spec["numbers"][key]
                opts[key] = max(low, min(high, int(val)))
            else:
                opts[key] = int(val)
            i += 2
        elif key is not None:
            raise ValueError(f"unknown option {t} - type help")
        else:
            words.append(t)
            i += 1
    text = " ".join(words).strip()
    if not text:
        raise ValueError('the prompt is empty - e.g. omega prompt "a cat surfing a wave at sunset"')
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
