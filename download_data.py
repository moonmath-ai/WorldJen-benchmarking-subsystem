"""
download_data.py
────────────────
Download the WorldJen dataset from HuggingFace into a local data/ directory.

Usage:
    # Download everything except videos (fast, ~50MB)
    python download_data.py

    # Also download all videos (~3 GB)
    python download_data.py --include-videos

    # Download to a custom location
    python download_data.py --dest /path/to/data

    # Download only a specific subset (e.g. ablation videos)
    python download_data.py --include-videos --video-subset ablation
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

REPO_ID = "ik6626/WorldJen-benchmarking-subsystem"
DEFAULT_DEST = Path(__file__).resolve().parent / "data"

# Glob patterns that match video files in the dataset
VIDEO_PATTERNS = [
    "videos/**",
    "videos_ablation_a1/**",
]
VIDEO_SUBSET_PATTERNS = {
    "main":     ["videos/**"],
    "ablation": ["videos_ablation_a1/**"],
}


def download(dest: Path, include_videos: bool, video_subset: str | None = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("  Run:  pip install huggingface_hub")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    if include_videos:
        if video_subset and video_subset in VIDEO_SUBSET_PATTERNS:
            ignore_patterns = [
                p for key, patterns in VIDEO_SUBSET_PATTERNS.items()
                if key != video_subset
                for p in patterns
            ]
        else:
            ignore_patterns = []
        print(f"Downloading full dataset (including videos) to {dest} ...")
    else:
        ignore_patterns = VIDEO_PATTERNS
        print(f"Downloading dataset (no videos) to {dest} ...")
        print("  Use --include-videos to also download video files (~3 GB).")

    token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        ignore_patterns=ignore_patterns,
        token=token or None,
    )
    print(f"\nDone. Data available at: {dest}")
    print("\nTo get started:")
    print("  python vlm_eval/unified_analyzer.py --data-root", dest)


def main():
    ap = argparse.ArgumentParser(description="Download WorldJen dataset from HuggingFace")
    ap.add_argument(
        "--dest", default=str(DEFAULT_DEST),
        help=f"Destination directory (default: {DEFAULT_DEST})"
    )
    ap.add_argument(
        "--include-videos", action="store_true",
        help="Also download video files (~3 GB total)"
    )
    ap.add_argument(
        "--video-subset", choices=["main", "ablation"], default=None,
        help="Download only a subset of videos (requires --include-videos)"
    )
    args = ap.parse_args()

    if args.video_subset and not args.include_videos:
        print("Warning: --video-subset has no effect without --include-videos. Ignoring.")

    download(
        dest=Path(args.dest),
        include_videos=args.include_videos,
        video_subset=args.video_subset if args.include_videos else None,
    )


if __name__ == "__main__":
    main()
