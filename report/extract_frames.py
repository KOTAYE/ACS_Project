#!/usr/bin/env python3
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video using OpenCV")
    p.add_argument("--input", default="output.mp4", help="Input video path (default: output.mp4)")
    p.add_argument("--out-dir", default="Frames", help="Output frames directory (default: Frames)")
    p.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import cv2
    except Exception:
        print("OpenCV (cv2) is not installed. Install with:")
        print("  pip install opencv-python")
        raise

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out_path = os.path.join(args.out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, frame)
        idx += 1

        if args.max_frames and idx >= args.max_frames:
            break

    cap.release()
    print(f"Extracted {idx} frames into {args.out_dir}/")


if __name__ == "__main__":
    main()

