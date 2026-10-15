ACOpsis
=======

Models
  zeta   text-to-video in a Family Guy style (2-4 second clips)

You need
  - Windows with an NVIDIA GPU (6GB+ of video memory, RTX 30-series or newer is best)
  - 16GB of RAM and about 16GB of free disk space
  - Python 3.10-3.12 from python.org (tick "Add python.exe to PATH" when installing)

Setup (once)
  1. Unzip this folder and double-click install.bat.
  2. Pick where to install (it uses ~16GB there), e.g.:
         location "D:\Apps"
     then type:
         install
     It downloads about 13GB, so give it a while.
  3. Open a NEW terminal window.

Use
  acopsis                                      start ACOpsis
  model zeta                                   load Zeta (about 30 seconds)
  zeta prompt "Peter sits on the couch"        make a video (about 40 seconds)
  help                                         every command

  Videos are saved in zeta\outputs and open automatically.
  Prefer a browser page? Double-click zeta\app\Zeta.bat instead.

Tips
  - Simple scenes (people sitting, talking, standing) look best.
  - Describe how characters look, not just their names - Zeta mixes them up.
