#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract every frame from an MP4 file into a directory."
    )
    parser.add_argument("input_video", help="Path to input .mp4 video")
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Directory for extracted frames (default: <video_name>_frames)",
    )
    parser.add_argument(
        "--ext",
        default="png",
        choices=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Output image extension (default: png)",
    )
    parser.add_argument(
        "--prefix",
        default="frame_",
        help="Output frame filename prefix (default: frame_)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting frame index (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.input_video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        print(f"Warning: input extension is {video_path.suffix}, expected .mp4")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = video_path.with_name(f"{video_path.stem}_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = args.start_index
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_name = f"{args.prefix}{frame_idx:06d}.{args.ext}"
        out_path = out_dir / out_name
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"Failed to write frame: {out_path}")
        frame_idx += 1
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to: {out_dir}")


if __name__ == "__main__":
    main()
