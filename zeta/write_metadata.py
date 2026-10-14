import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"

captions = json.loads((HERE / "captions.json").read_text(encoding="utf-8"))
clips = sorted(p.stem for p in (DATASET / "clips").glob("*.mp4"))

missing = [c for c in clips if c not in captions]
if missing:
    raise SystemExit(f"no caption for: {', '.join(missing)}")

rows = [{"file_name": f"clips/{c}.mp4", "caption": captions[c]} for c in clips]
with open(DATASET / "metadata.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
(DATASET / "videos.txt").write_text("\n".join(r["file_name"] for r in rows) + "\n", encoding="utf-8")
(DATASET / "prompts.txt").write_text("\n".join(r["caption"] for r in rows) + "\n", encoding="utf-8")
print(f"{len(rows)} clips captioned")
