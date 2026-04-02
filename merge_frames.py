#!/usr/bin/env python3
"""Merge a folder of PNG/JPEG frames into output.mp4."""

import cv2
import os
import re
import sys

if len(sys.argv) < 2:
    print("Usage: python merge_frames.py <frames_dir> [fps]")
    sys.exit(1)

image_folder = sys.argv[1]
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 24
output_video = "output.mp4"

if not os.path.isdir(image_folder):
    print("error: directory not found", file=sys.stderr)
    sys.exit(1)


def numeric_sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else -1


images = sorted(
    [img for img in os.listdir(image_folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))],
    key=numeric_sort_key,
)

if not images:
    print("error: no images in folder", file=sys.stderr)
    sys.exit(1)

first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

video.release()
print(output_video)
