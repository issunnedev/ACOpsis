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
| **Zeta 1.2** | Text-to-video with sound, Family Guy style (2–4 second clips) |
| **ε-Epsilon** | Early demo |

### Updates
## Zeta 1.2

- **Sound:** every video gets matching sound effects automatically (MMAudio)
- **`--smooth`:** interpolates to 48fps so motion is less choppy
- **`--hd`:** upscales to 1024×576 for sharing
- Runs on a 6GB laptop GPU with 16GB of RAM
## Epsilon E0.1

- Added full CLI manager interface
- Added Youtube fetching and downloading scripts

# Info

## Requirements

- Windows with an NVIDIA GPU (6GB+ of video memory, RTX 30-series or newer)
- 16GB of RAM (close other big apps while installing)
- About 35GB of free disk to install, 25GB after
- Python 3.10–3.12 from [python.org](https://www.python.org/downloads/) (tick **Add python.exe to PATH**)

## Install ζ-Zeta

1. Download this repo and double-click `install.bat`.
2. Pick where to install, then install:
   ```
   installer> location "D:\Apps"
   installer> install
   ```
   It downloads about 23GB, so give it a while.
3. Open a **new** terminal.

## Use ζ-Zeta

```
acopsis                                   start ACOpsis
model zeta                                load Zeta (about 40 seconds)
zeta prompt "Peter sits on the couch"     make a video with sound (about 1 minute)
zeta prompt "..." --silent                skip the sound (faster)
zeta prompt "..." --smooth --hd           smoother and bigger
help                                      every command
```

Videos are saved in `zeta/outputs` and open automatically. For a browser page instead, double-click `zeta/app/Zeta.bat`.

## Install ε-Epsilon

1. Open directory ε-Epsilon
2. Open a terminal/Command line window
3. run python3/python Manager.py to start the manager CLI
4. select option 1 (setup)
5. let setup run (~1-3GB required)

## Using ε-Epsilon

1. Start Manager.py using python in a terminal
2. Choose a option (atleast 1 run of gather is required for parser to be useful)

## Tips

- Simple scenes (people sitting, talking, standing) look best.
- Describe how characters look, not just their names. Zeta mixes them up.
- Talking scenes get babble, not real words. The sound model makes effects, not speech.

## Training Zeta

`zeta/prep_dataset.py` cuts a source video into 3-second clips, `zeta/captions.json` holds their captions, and `zeta/kaggle/train_zeta.py` trains the LoRA on a free Kaggle T4 (about 3 hours). The training clips themselves aren't included.

## Licenses

The MMAudio sound model's weights are licensed for non-commercial use only.
