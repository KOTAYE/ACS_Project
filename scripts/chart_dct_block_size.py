#!/usr/bin/env python3
"""Графік compress_fps та compression_ratio для -b 8/16/32 (flipbook_cuda)."""
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.size": 11,
    }
)


def parse_benchmark(text: str) -> dict:
    out = {}
    for m in re.finditer(r"\[BENCHMARK\]\s+(.+)", text):
        for kv in m.group(1).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: chart_dct_block_size.py <frames_dir> <flipbook_cuda_exe> <out_charts_dir>")
        sys.exit(1)
    frames = sys.argv[1]
    exe = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    blocks = [8, 16, 32]
    rows = []
    for b in blocks:
        bin_path = os.path.join(out_dir, f"_tmp_block_{b}.bin")
        p = subprocess.run(
            [exe, "compress", "-q", "50", "-b", str(b), frames, bin_path],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            print(f"Warning: flipbook_cuda -b {b} failed (code {p.returncode})", file=sys.stderr)
            continue
        m = parse_benchmark(text)
        sz = os.path.getsize(bin_path) if os.path.isfile(bin_path) else 0
        try:
            os.remove(bin_path)
        except OSError:
            pass
        raw = m.get("raw_bytes") or 0
        ratio = m.get("compression_ratio")
        if ratio is None and raw and sz:
            ratio = float(raw) / float(sz)
        rows.append(
            {
                "block": b,
                "compress_fps": m.get("compress_fps", 0),
                "compression_ratio": ratio or 0,
                "compress_ms": m.get("compress_ms", 0),
            }
        )

    bench_json = os.path.join(os.path.dirname(out_dir), "benchmark_results", "block_size_metrics.json")
    try:
        os.makedirs(os.path.dirname(bench_json), exist_ok=True)
        with open(bench_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {bench_json}")
    except OSError:
        with open(os.path.join(out_dir, "block_size_metrics.json"), "w") as f:
            json.dump(rows, f, indent=2)

    if not rows:
        print("No block-size data; skip chart.", file=sys.stderr)
        return

    labels = [str(r["block"]) for r in rows]
    x = range(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fps = [r["compress_fps"] for r in rows]
    ratios = [r["compression_ratio"] for r in rows]
    ax1.bar(labels, fps, color="#58a6ff", edgecolor="white")
    ax1.set_xlabel("DCT block size")
    ax1.set_ylabel("Compress FPS")
    ax1.set_title("CUDA: compression speed vs block size (q=50)")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar(labels, ratios, color="#3fb950", edgecolor="white")
    ax2.set_xlabel("DCT block size")
    ax2.set_ylabel("Compression ratio (×)")
    ax2.set_title("CUDA: compression ratio vs block size (q=50)")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "chart_dct_block_size.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
