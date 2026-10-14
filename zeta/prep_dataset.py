import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

FPS = 24
CLIP_FRAMES = 73
HEIGHT = 720
SCENE_THRESHOLD = 0.3
EDGE_MARGIN = 0.1

HERE = Path(__file__).resolve().parent
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def duration(src):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def detect_cuts(src):
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", str(src), "-an",
                           "-filter:v", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
                           "-f", "null", "-"], capture_output=True, text=True)
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]


def plan_clips(src):
    bounds = [0.0, *detect_cuts(src), duration(src)]
    clip_len = CLIP_FRAMES / FPS
    clips = []
    for shot, (start, end) in enumerate(zip(bounds, bounds[1:])):
        usable = end - start - 2 * EDGE_MARGIN
        for k in range(int(usable // clip_len)):
            clips.append({"name": f"zeta_s{shot:03d}_{k}",
                          "shot": shot,
                          "start": round(start + EDGE_MARGIN + k * clip_len, 3)})
    return bounds, clips


def export_clip(src, clip, out_dir):
    out = out_dir / f"{clip['name']}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(clip["start"]), "-i", str(src),
                    "-an", "-vf", f"fps={FPS},scale=-2:{HEIGHT}", "-frames:v", str(CLIP_FRAMES),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def review_sheets(clips, clip_dir, sheet_dir, per_sheet=24):
    frames_dir = sheet_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    picks = [0.2, CLIP_FRAMES / FPS / 2, CLIP_FRAMES / FPS - 0.2]
    for page in range(0, len(clips), per_sheet):
        for f in frames_dir.glob("*.png"):
            f.unlink()
        batch = clips[page:page + per_sheet]
        n = 0
        for clip in batch:
            for t in picks:
                subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
                                "-i", str(clip_dir / f"{clip['name']}.mp4"), "-frames:v", "1",
                                "-vf", "scale=256:144", str(frames_dir / f"{n:04d}.png")], check=True)
                n += 1
        rows = (len(batch) + 1) // 2
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(frames_dir / "%04d.png"),
                        "-vf", f"tile=6x{rows}:padding=4:color=black", "-frames:v", "1",
                        str(sheet_dir / f"sheet_{page // per_sheet:02d}.png")], check=True)
        print(f"sheet_{page // per_sheet:02d}: " + ", ".join(c["name"] for c in batch))
    shutil.rmtree(frames_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "dataset")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    exclude_file = HERE / "exclude.txt"
    exclude = set(exclude_file.read_text().split()) if exclude_file.exists() else set()

    clip_dir = args.out / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    bounds, clips = plan_clips(args.source)
    clips = [c for c in clips if c["name"] not in exclude]
    print(f"{len(bounds) - 1} shots -> {len(clips)} clips ({len(exclude)} excluded)")

    for old in clip_dir.glob("*.mp4"):
        old.unlink()
    for clip in clips:
        export_clip(args.source, clip, clip_dir)
    (args.out / "clips.json").write_text(json.dumps(clips, indent=1))

    if args.review:
        review_sheets(clips, clip_dir, HERE / "review")


if __name__ == "__main__":
    main()
