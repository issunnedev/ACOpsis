<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ACOpsisWhite.png">
    <img src="ACOpsisBlack.png" alt="ACOpsis logo" width="180">
  </picture>
</p>

# ACOpsis

AutoCognitive Opsis: a small family of AI models, run from one terminal app.

| Model | What it does |
|---|---|
| **ω-Omega 1.0** | High-quality text-to-video with sound and a built-in prompt enhancer (2–5 second clips, up to 720p) |
| **ζ-Zeta 1.2** | Text-to-video with sound, Family Guy style (2–4 second clips) |
| **ε-Epsilon E2** | ~Early tech demo~ MESS with many bugs and no functional model (yet) **NOTE! Currently on developmental pause** |
| **δ-delta Alpha 0.1** | Highly customizable LM trained exclusively off 4Chan and basic English |

# Updates
## Omega 1.0

- New model: built on **FastWan 2.2** (Wan 2.2 5B, distilled to 3 steps), no training involved
- **Prompt enhancer:** a local Qwen2.5-3B rewrites short prompts into detailed ones (`--raw` to skip)
- **Sound:** matching sound effects, same engine as Zeta
- **`--res 480|720`**, **`--length 2|3|5`**, and **`--best`** for the full-quality decoder
- Runs on a 6GB laptop GPU by streaming the model from disk
## Zeta 1.2

- **Sound:** every video gets matching sound effects automatically (MMAudio)
- **`--smooth`:** interpolates to 48fps so motion is less choppy
- **`--hd`:** upscales to 1024×576 for sharing
- Runs on a 6GB laptop GPU with 16GB of RAM
## Epsilon E2

- Added updated parser
- Configured ACOpsis shell to integrate (may not work)
- Configured new CLI update (has issues now that I notice)

## Installer

- Picks a supported Python automatically and explains what to install if there isn't one (Python 3.15 isn't supported by PyTorch yet)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Models

## ζ-Zeta
### Install ζ-Zeta

1. Download this repo and double-click `install.bat`.
2. Pick where to install, then install:
   ```
   installer> location "D:\Apps"
   installer> install
   ```
   It downloads about 23GB, so give it a while.
3. Open a **new** terminal.

If the installer says no supported Python was found, install [Python 3.12](https://www.python.org/downloads/release/python-31210/) and run `install.bat` again. It removes any broken environment from a previous attempt by itself.

### Use ζ-Zeta

```
acopsis                                   start ACOpsis
model zeta                                load Zeta (about 40 seconds)
zeta prompt "Peter sits on the couch"     make a video with sound (about 1 minute)
zeta prompt "..." --silent                skip the sound (faster)
zeta prompt "..." --smooth --hd           smoother and bigger
help                                      every command
```

Videos are saved in `zeta/outputs` and open automatically. For a browser page instead, double-click `zeta/app/Zeta.bat`.

### ζ-Zeta Requirements

- Windows on an Intel or AMD PC with an NVIDIA GPU (6GB+ of video memory, RTX 30-series or newer)
- 16GB of RAM (close other big apps while installing)
- About 35GB of free disk to install, 25GB after
- **64-bit Python 3.10–3.14** from [python.org](https://www.python.org/downloads/release/python-31210/) (3.12 recommended, tick **Add python.exe to PATH**)

## ω-Omega
### Install ω-Omega

1. Install ζ-Zeta first (Omega uses the same app and installer).
2. Open a terminal, type `acopsis`, then:
   ```
   setup omega
   ```
   It downloads about 30GB and keeps about 16GB of working files.

### Use ω-Omega

```
omega prompt "a cat surfing a wave at sunset"    720p, 5 seconds, with sound (about 2½ minutes)
omega prompt "..." --res 480 --length 2          quick version (about 1 minute)
omega prompt "..." --raw                         use your prompt as-is, skip the enhancer
omega prompt "..." --best                        full-quality decoder (slower)
omega prompt "..." --silent --smooth             no sound, smoother motion
```

After each video, ACOpsis prints the expanded prompt the enhancer wrote. Videos are saved in `omega/outputs`.

### ω-Omega Requirements

- Everything ζ-Zeta needs
- About 45GB more free disk (30GB of downloads, which can go on another drive, plus 16GB of working files)

### Storage

Big downloads go next to ACOpsis by default. To put them on another drive, write its path (e.g. `D:\ACOpsis`) into a file called `storage.txt` in the ACOpsis folder. Omega's working files are read on every video, so they stay on the fastest drive; to move them too, write the path into `fast_storage.txt`.

## ε-Epsilon
### Install ε-Epsilon

1. Open directory ε-Epsilon
2. Open a terminal/Command line window
3. run python3/python Manager.py to start the manager CLI
4. select option 1 (setup)
5. let setup run (~1-3GB required)

### Using ε-Epsilon

1. Start Manager.py (inside ε-Epsilon inside the ACOpsis root) using python in a terminal (or type `epsilon` inside `acopsis` shell)
2. Choose a option (at-least 1 run of gather is required for parser to be useful)

### ε-Epsilon Requirements:
- ACOpsis Model 1 ζ-Zeta (w/ shared parent directory)
- Nvidia GPU with Tensor cores and minimum 6GB VRAM (Tested on RTX 3060 Laptop GPU)
- 16GB RAM DDR4/DDR5 (DDR3 or below may be too slow)
- Decently Fast SSD or HDD (NVMe SSD recommended)
- Moderate CPU
- Fast LAN/wLAN (downloading)

## Tips

### ζ-Zeta
- Simple scenes (people sitting, talking, standing) look best.
- Describe how characters look, not just their names. Zeta mixes them up.
- Talking scenes get babble, not real words. The sound model makes effects, not speech.
- Omega follows detailed prompts best; the enhancer handles that for you unless you use `--raw`.

## Training Zeta

`zeta/prep_dataset.py` cuts a source video into 3-second clips, `zeta/captions.json` holds their captions, and `zeta/kaggle/train_zeta.py` trains the LoRA on a free Kaggle T4 (about 3 hours). The training clips themselves aren't included.

## Licenses

Two parts are licensed for non-commercial use only: the MMAudio sound model's weights and Omega's Qwen2.5-3B prompt enhancer (Qwen Research License). FastWan 2.2 is Apache 2.0 and the TAEHV decoder is MIT.

### Extra
REPORT ALL BUGS/ISSUES
