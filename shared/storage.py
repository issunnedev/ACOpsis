from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pointer(name):
    pointer = ROOT / name
    if not pointer.exists():
        return ROOT
    path = Path(pointer.read_text(encoding="utf-8").strip())
    if not Path(path.anchor).exists():
        raise RuntimeError(f"the ACOpsis storage drive ({path.anchor}) isn't connected")
    return path


def data_root():
    return _pointer("storage.txt")


def fast_root():
    return _pointer("fast_storage.txt")
