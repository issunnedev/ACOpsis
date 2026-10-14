import json
import re
import sys
from pathlib import Path

import yt_dlp

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "youtube_cache"

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def find_prompts():
    """Find cache folders and group them by prefix before '_'."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prompts = {}

    for folder in sorted(CACHE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        prefix = folder.name.split("_", 1)[0]
        prompts.setdefault(prefix, []).append(folder)

    return prompts


def choose_prompt(prompts):
    """Display prompt prefixes and return the selected folders."""
    names = sorted(
        prompts,
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
    )

    if not names:
        print("No prompt folders found in youtube_cache.")
        return []

    print("\nAvailable prompts:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. Prompt {name}")

    print("  A. All prompts")
    print("  Q. Quit")

    while True:
        choice = input("\nSelect a prompt: ").strip().lower()

        if choice == "q":
            return []

        if choice == "a":
            return [folder for name in names for folder in prompts[name]]

        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return prompts[names[int(choice) - 1]]

        print("Invalid selection. Try again.")


def find_json_files(folder):
    """Find JSON files in the selected folder and its subfolders."""
    return [
        path for path in folder.rglob("*.json")
        if path.is_file()
    ]


def extract_videos(value, videos):
    """Recursively extract YouTube URLs and video IDs from JSON."""
    if isinstance(value, dict):
        for key, item in value.items():
            # Common YouTube API fields.
            if key == "videoId" and isinstance(item, str):
                if VIDEO_ID_RE.fullmatch(item):
                    videos.add(item)

            extract_videos(item, videos)

    elif isinstance(value, list):
        for item in value:
            extract_videos(item, videos)

    elif isinstance(value, str):
        # Find YouTube links anywhere in a string.
        patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?[^\s\"'<>]*",
            r"(?:https?://)?youtu\.be/[A-Za-z0-9_-]{11}",
            r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[A-Za-z0-9_-]{11}",
            r"(?:https?://)?(?:www\.)?youtube\.com/embed/[A-Za-z0-9_-]{11}",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, value):
                url = match.rstrip(".,);]}")
                video_id = get_video_id(url)
                if video_id:
                    videos.add(video_id)

        # Also support strings that are bare video IDs.
        if VIDEO_ID_RE.fullmatch(value.strip()):
            videos.add(value.strip())


def get_video_id(value):
    """Convert a YouTube URL or ID into a video ID."""
    value = value.strip()

    if VIDEO_ID_RE.fullmatch(value):
        return value

    match = re.search(
        r"(?:youtu\.be/|youtube\.com/(?:watch\?.*?v=|shorts/|embed/))"
        r"([A-Za-z0-9_-]{11})",
        value,
        re.IGNORECASE,
    )

    return match.group(1) if match else None


def load_videos(folder):
    """Read all JSON files in a prompt folder."""
    videos = set()
    json_files = find_json_files(folder)

    if not json_files:
        print(f"  No JSON files found in {folder.name}")
        return []

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)

            before = len(videos)
            extract_videos(data, videos)
            print(
                f"  Read {json_file.relative_to(folder)} "
                f"({len(videos) - before} new video IDs)"
            )

        except (OSError, json.JSONDecodeError) as error:
            print(f"  Could not read {json_file.name}: {error}")

    return sorted(videos)


def download_videos(folder, video_ids):
    """Download videos into the selected cache folder."""
    if not video_ids:
        print("  No YouTube video links or IDs found.")
        return

    options = {
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
            "/best[ext=mp4]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(folder / "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "windowsfilenames": True,
        "ignoreerrors": True,
        "overwrites": False,
    }

    print(f"  Found {len(video_ids)} video(s).")

    with yt_dlp.YoutubeDL(options) as downloader:
        for video_id in video_ids:
            # Skip if an MP4 with this video ID already exists.
            existing = list(folder.glob(f"*[{video_id}].mp4"))
            if existing:
                print(f"  Already downloaded: {video_id}")
                continue

            print(f"  Downloading: {video_id}")

            try:
                downloader.download([
                    f"https://www.youtube.com/watch?v={video_id}"
                ])
            except Exception as error:
                print(f"  Failed {video_id}: {error}")


def main():
    prompts = find_prompts()
    selected_folders = choose_prompt(prompts)

    if not selected_folders:
        print("Nothing selected.")
        return

    for folder in selected_folders:
        print(f"\nProcessing prompt folder: {folder.name}")

        video_ids = load_videos(folder)
        download_videos(folder, video_ids)

    print("\nFinished.")


if __name__ == "__main__":
    main()
