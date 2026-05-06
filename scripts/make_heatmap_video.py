#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def map_to_heat(map_path: Path, blocks_x: int, blocks_y: int, width: int, height: int):
    data = np.fromfile(map_path, dtype=np.float32)
    if data.size != blocks_x * blocks_y:
        return None
    block_map = data.reshape((blocks_y, blocks_x))

    # Scale < 1.0 means higher quality (less compression).
    # Scale > 1.0 means stronger compression.
    score = np.clip((block_map - 0.6) / (1.6 - 0.6), 0.0, 1.0)
    score_u8 = (score * 255.0).astype(np.uint8)
    up = cv2.resize(score_u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(up, cv2.COLORMAP_TURBO), float(np.mean(block_map))


def main():
    parser = argparse.ArgumentParser(description="Build heatmap video from ROI compression maps.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.tsv")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--fps", type=float, default=24.0, help="Output FPS")
    parser.add_argument("--alpha", type=float, default=0.40, help="Heatmap overlay alpha")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"No frames in manifest: {manifest_path}")

    base_dir = manifest_path.parent
    first_frame = cv2.imread(rows[0]["source_frame"], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {rows[0]['source_frame']}")
    height, width = first_frame.shape[:2]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {out_path}")

    for row in rows:
        frame = cv2.imread(row["source_frame"], cv2.IMREAD_COLOR)
        if frame is None:
            continue

        blocks_x = int(row["blocks_x"])
        blocks_y = int(row["blocks_y"])
        map_path = base_dir / row["map_file"]
        result = map_to_heat(map_path, blocks_x, blocks_y, width, height)
        if result is None:
            writer.write(frame)
            continue
        heat, mean_scale = result
        overlay = cv2.addWeighted(frame, 1.0 - args.alpha, heat, args.alpha, 0.0)
        cv2.putText(
            overlay,
            f"Compression heatmap | mean q-scale: {mean_scale:.3f}",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Blue/green: preserve quality  Red: stronger compression",
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        writer.write(overlay)

    writer.release()
    print(f"Heatmap video written: {out_path}")


if __name__ == "__main__":
    main()
