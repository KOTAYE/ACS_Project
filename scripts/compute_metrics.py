#!/usr/bin/env python3
"""Compute PSNR/SSIM from benchmark logs and reconstructed frame folders."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Install pillow: pip install -r requirements-bench.txt", file=sys.stderr)
    raise


def image_sort_key(path: str) -> tuple:
    name = os.path.basename(path)
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return tuple(nums) if nums else (name,)


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def compute_ssim_channel(a: np.ndarray, b: np.ndarray, window: int = 8) -> float:
    k1, k2, luma = 0.01, 0.03, 255.0
    c1 = (k1 * luma) ** 2
    c2 = (k2 * luma) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    h, w = a.shape
    vals = []
    for y in range(0, h - window + 1, window):
        for x in range(0, w - window + 1, window):
            wa = a[y : y + window, x : x + window]
            wb = b[y : y + window, x : x + window]
            mu_a, mu_b = wa.mean(), wb.mean()
            sigma_a2, sigma_b2 = wa.var(), wb.var()
            sigma_ab = np.mean((wa - mu_a) * (wb - mu_b))
            num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
            den = (mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2)
            vals.append(num / den)
    return float(np.mean(vals)) if vals else 1.0


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        return compute_ssim_channel(a, b)
    return float(np.mean([compute_ssim_channel(a[:, :, c], b[:, :, c]) for c in range(a.shape[2])]))


def parse_benchmark_log(log_path: str) -> dict:
    out: dict = {}
    if not os.path.isfile(log_path):
        return out
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"\[BENCHMARK\]\s+(.*)", line)
            if not m:
                continue
            for kv in m.group(1).split():
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def discover_qualities(results_dir: str, impl: str) -> list[int]:
    pat = re.compile(rf"^{impl}_compress_q(\d+)\.log$")
    qs: set[int] = set()
    try:
        for name in os.listdir(results_dir):
            m = pat.match(name)
            if m:
                qs.add(int(m.group(1)))
    except OSError:
        pass
    return sorted(qs)


def list_frames(directory: str) -> list[str]:
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.exr"):
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths, key=image_sort_key)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build metrics.json from benchmark_results/")
    ap.add_argument("orig_dir", help="Original frames directory")
    ap.add_argument("results_dir", help="Directory with logs and *_recon folders")
    args = ap.parse_args()

    orig_frames = list_frames(args.orig_dir)
    if not orig_frames:
        print(f"No frames in {args.orig_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(orig_frames)} original frames")

    all_results: dict = {}
    for impl in ("gpu", "cpu"):
        for q in discover_qualities(args.results_dir, impl):
            key = f"{impl}_q{q}"
            recon_dir = os.path.join(args.results_dir, f"{key}_recon")
            entry = {
                "impl": impl,
                "quality": q,
                **parse_benchmark_log(os.path.join(args.results_dir, f"{impl}_compress_q{q}.log")),
                **parse_benchmark_log(os.path.join(args.results_dir, f"{impl}_decompress_q{q}.log")),
            }

            if os.path.isdir(recon_dir):
                recon_frames = list_frames(recon_dir)
                n = min(len(orig_frames), len(recon_frames))
                if len(orig_frames) != len(recon_frames):
                    print(
                        f"  {key}: frame count orig={len(orig_frames)} recon={len(recon_frames)}",
                        file=sys.stderr,
                    )

                psnr_vals: list[float] = []
                ssim_vals: list[float] = []
                for i in range(n):
                    orig_img = load_image(orig_frames[i])
                    recon_img = load_image(recon_frames[i])
                    if orig_img.shape != recon_img.shape:
                        min_h = min(orig_img.shape[0], recon_img.shape[0])
                        min_w = min(orig_img.shape[1], recon_img.shape[1])
                        orig_img = orig_img[:min_h, :min_w]
                        recon_img = recon_img[:min_h, :min_w]
                    psnr_vals.append(compute_psnr(orig_img, recon_img))
                    ssim_vals.append(compute_ssim(orig_img, recon_img))

                if psnr_vals:
                    entry["avg_psnr"] = float(np.mean(psnr_vals))
                    entry["avg_ssim"] = float(np.mean(ssim_vals))
                    entry["min_psnr"] = float(np.min(psnr_vals))
                    entry["max_psnr"] = float(np.max(psnr_vals))
                    print(f"  {key}: PSNR={entry['avg_psnr']:.2f} dB, SSIM={entry['avg_ssim']:.4f}")

            all_results[key] = entry

    out_path = os.path.join(args.results_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
