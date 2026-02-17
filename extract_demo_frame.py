"""
Extract a frame from demo.mp4 to a JPG for API testing.

Usage (PowerShell):
  python extract_demo_frame.py --frame 10 --out demo_frame.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="demo.mp4", help="Path to input video")
    ap.add_argument("--frame", type=int, default=10, help="0-based frame index to extract")
    ap.add_argument("--out", default="demo_frame.jpg", help="Output image path")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit(f"Failed to read frame {args.frame} from {video_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()

