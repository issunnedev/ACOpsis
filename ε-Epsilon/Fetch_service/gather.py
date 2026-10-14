import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"

MAX_DURATION = 300           # Strictly under 5 minutes
TARGET_RESULTS = 100
CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_CACHE_DIR = Path("youtube_cache")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_duration(duration):
    """Convert ISO 8601 YouTube durations to seconds."""
    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        duration,
    )
    if not match:
        return 0

    hours, minutes, seconds = (
        int(value or 0) for value in match.groups()
    )
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def api_get(endpoint, params):
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params={"key": API_KEY, **params},
        timeout=30,
    )

    if not response.ok:
        try:
            detail = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = response.text[:500]

        raise RuntimeError(
            f"YouTube API error ({response.status_code}): {detail}"
        )

    return response.json()


def make_cache_folder(prompt, cache_dir):
    """Create a stable, filesystem-safe folder for each prompt."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower())
    slug = slug.strip("_")[:45] or "search"

    digest = hashlib.sha256(
        prompt.strip().lower().encode("utf-8")
    ).hexdigest()[:10]

    folder = cache_dir / f"{slug}_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_cache(folder, prompt):
    json_path = folder / "videos.json"

    if not json_path.exists():
        return None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("prompt", "").lower() != prompt.strip().lower():
            return None

        cached_at = datetime.fromisoformat(data["cached_at"])
        age = (
            datetime.now(timezone.utc)
            - cached_at.astimezone(timezone.utc)
        ).total_seconds()

        if age < 0 or age > CACHE_TTL_SECONDS:
            return None

        videos = data.get("videos", [])
        if not isinstance(videos, list):
            return None

        print(f"Using cache: {json_path}")
        print(f"Cached videos: {len(videos)}")
        return videos

    except (OSError, ValueError, KeyError, TypeError):
        return None


def search_video_ids(prompt, max_pages):
    """Gather unique candidates; each search page returns up to 50."""
    video_ids = []
    seen = set()
    page_token = None

    for page in range(max_pages):
        params = {
            "part": "snippet",
            "q": prompt,
            "type": "video",
            "order": "viewCount",
            "maxResults": 50,
            "safeSearch": "moderate",
        }

        if page_token:
            params["pageToken"] = page_token

        data = api_get("search", params)

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if video_id and video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)

        page_token = data.get("nextPageToken")
        print(
            f"Search page {page + 1}: "
            f"{len(video_ids)} unique candidates"
        )

        if not page_token:
            break

        # Avoid hammering the API.
        time.sleep(0.1)

    return video_ids


def fetch_video_details(video_ids):
    """Retrieve details in batches of 50 IDs."""
    videos = []

    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]

        data = api_get(
            "videos",
            {
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(batch),
                "maxResults": 50,
            },
        )

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            details = item.get("contentDetails", {})
            stats = item.get("statistics", {})

            duration = parse_duration(details.get("duration", ""))

            if not 0 < duration < MAX_DURATION:
                continue

            video_id = item["id"]

            videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "duration_seconds": duration,
                "duration": format_duration(duration),
                "views": int(stats.get("viewCount", 0)),
                "likes": (
                    int(stats["likeCount"])
                    if "likeCount" in stats else None
                ),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            })

    # Rank the discovered short videos by views.
    videos.sort(key=lambda video: video["views"], reverse=True)
    return videos[:TARGET_RESULTS]


def save_results(folder, prompt, videos):
    """Write JSON cache and a spreadsheet-friendly CSV."""
    payload = {
        "prompt": prompt,
        "cached_at": utc_now(),
        "result_count": len(videos),
        "max_duration_seconds": MAX_DURATION,
        "ranking": "views_descending",
        "videos": videos,
    }

    json_path = folder / "videos.json"
    csv_path = folder / "videos.csv"

    # Write to temporary files before replacing existing results.
    json_tmp = folder / "videos.json.tmp"
    csv_tmp = folder / "videos.csv.tmp"

    json_tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "video_id", "title", "channel", "published_at",
        "duration_seconds", "duration", "views", "likes", "url",
    ]

    with csv_tmp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(videos)

    json_tmp.replace(json_path)
    csv_tmp.replace(csv_path)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def collect(prompt, cache_dir, max_pages, refresh=False):
    folder = make_cache_folder(prompt, cache_dir)

    if not refresh:
        cached = read_cache(folder, prompt)
        if cached is not None:
            return cached, folder

    print(f"\nSearching for: {prompt}")
    print(f"Candidate pages: {max_pages}")

    ids = search_video_ids(prompt, max_pages)

    print(f"Fetching details for {len(ids)} candidates...")
    videos = fetch_video_details(ids)

    save_results(folder, prompt, videos)

    if len(videos) < TARGET_RESULTS:
        print(
            f"Note: Found {len(videos)} qualifying videos, "
            f"not {TARGET_RESULTS}. Try increasing --pages."
        )

    return videos, folder


def main():
    parser = argparse.ArgumentParser(
        description="Find and locally cache short YouTube videos."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help='Search prompt, e.g. "quick pasta recipes"',
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Maximum search pages (1-10; default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Root folder for cached results",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached results and search again",
    )

    args = parser.parse_args()

    if not API_KEY:
        sys.exit(
            "Missing YOUTUBE_API_KEY. Add it to your .env file."
        )

    if not 1 <= args.pages <= 10:
        sys.exit("--pages must be between 1 and 10.")

    prompt = args.prompt or input("Enter a YouTube search prompt: ")
    prompt = prompt.strip()

    if not prompt:
        sys.exit("Search prompt cannot be empty.")

    videos, folder = collect(
        prompt,
        Path(args.cache_dir),
        args.pages,
        refresh=args.refresh,
    )

    print(f"\nTop {len(videos)} results:")
    for rank, video in enumerate(videos, 1):
        print(
            f"{rank:03}. {video['title']} | "
            f"{video['duration']} | "
            f"{video['views']:,} views | {video['url']}"
        )

    print(f"\nCache folder: {folder.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, RuntimeError) as error:
        sys.exit(f"Error: {error}")