#!/usr/bin/env python3
"""Plot charts from benchmark_results/metrics.json."""

from __future__ import annotations

import argparse
import json
import os
import re
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
        "grid.alpha": 0.8,
        "font.size": 11,
    }
)

GPU_COLOR = "#58a6ff"
CPU_COLOR = "#f0883e"
ACCENT1 = "#3fb950"
ACCENT2 = "#bc8cff"
ACCENT3 = "#ff7b72"


def load_metrics(results_dir: str) -> dict:
    path = os.path.join(results_dir, "metrics.json")
    if not os.path.isfile(path):
        print(f"Missing {path}. Run compute_metrics.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def qualities_in_data(data: dict) -> list[int]:
    qs: set[int] = set()
    for key in data:
        m = re.match(r"(?:gpu|cpu)_q(\d+)$", key)
        if m:
            qs.add(int(m.group(1)))
    return sorted(qs)


def get_series(data: dict, impl: str, field: str, qualities: list[int]) -> tuple[list[int], list[float]]:
    x, y = [], []
    for q in qualities:
        key = f"{impl}_q{q}"
        if key in data and field in data[key]:
            x.append(q)
            y.append(data[key][field])
    return x, y


def chart_psnr(data: dict, qualities: list[int], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    gq, gv = get_series(data, "gpu", "avg_psnr", qualities)
    cq, cv = get_series(data, "cpu", "avg_psnr", qualities)
    if gq:
        ax.plot(gq, gv, "o-", color=GPU_COLOR, linewidth=2, markersize=7, label="CUDA")
    if cq:
        ax.plot(cq, cv, "s--", color=CPU_COLOR, linewidth=2, markersize=7, label="OpenMP")
    ax.set_xlabel("Quality")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs quality")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_psnr_quality.png"), dpi=150)
    plt.close()


def chart_ssim(data: dict, qualities: list[int], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    gq, gv = get_series(data, "gpu", "avg_ssim", qualities)
    cq, cv = get_series(data, "cpu", "avg_ssim", qualities)
    if gq:
        ax.plot(gq, gv, "o-", color=GPU_COLOR, linewidth=2, markersize=7, label="CUDA")
    if cq:
        ax.plot(cq, cv, "s--", color=CPU_COLOR, linewidth=2, markersize=7, label="OpenMP")
    ax.set_xlabel("Quality")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs quality")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_ssim_quality.png"), dpi=150)
    plt.close()


def chart_field(data: dict, qualities: list[int], field: str, ylabel: str, title: str, fname: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    gq, gv = get_series(data, "gpu", field, qualities)
    cq, cv = get_series(data, "cpu", field, qualities)
    if gq:
        ax.plot(gq, gv, "o-", color=GPU_COLOR, linewidth=2, markersize=7, label="CUDA")
    if cq:
        ax.plot(cq, cv, "s--", color=CPU_COLOR, linewidth=2, markersize=7, label="OpenMP")
    ax.set_xlabel("Quality")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()


def chart_rate_distortion(data: dict, qualities: list[int], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for impl, color, marker, label in [
        ("gpu", GPU_COLOR, "o", "CUDA"),
        ("cpu", CPU_COLOR, "s", "OpenMP"),
    ]:
        bpp_vals, psnr_vals, q_labels = [], [], []
        for q in qualities:
            key = f"{impl}_q{q}"
            row = data.get(key, {})
            if "avg_psnr" not in row or "compressed_bytes" not in row or "raw_bytes" not in row:
                continue
            raw, comp = row["raw_bytes"], row["compressed_bytes"]
            if raw <= 0:
                continue
            bpp_vals.append(8.0 * comp / raw)
            psnr_vals.append(row["avg_psnr"])
            q_labels.append(q)
        if bpp_vals:
            ax.plot(bpp_vals, psnr_vals, f"{marker}-", color=color, linewidth=2, markersize=7, label=label)
            for bpp, psnr, q in zip(bpp_vals, psnr_vals, q_labels):
                ax.annotate(
                    f"q={q}",
                    (bpp, psnr),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color=color,
                )
    ax.set_xlabel("Bits per pixel")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-distortion")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_rate_distortion.png"), dpi=150)
    plt.close()


def chart_speedup(data: dict, qualities: list[int], out_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cq, comp_up = [], []
    dq, dec_up = [], []
    for q in qualities:
        gk, ck = f"gpu_q{q}", f"cpu_q{q}"
        if gk not in data or ck not in data:
            continue
        g, c = data[gk], data[ck]
        if "compress_ms" in g and "compress_ms" in c and c["compress_ms"] > 0:
            cq.append(q)
            comp_up.append(c["compress_ms"] / max(g["compress_ms"], 1e-6))
        if "decode_total_ms" in g and "decode_total_ms" in c and g["decode_total_ms"] > 0:
            dq.append(q)
            dec_up.append(c["decode_total_ms"] / g["decode_total_ms"])

    if cq:
        ax1.bar(cq, comp_up, width=6, color=GPU_COLOR, alpha=0.85)
        ax1.axhline(1, color=ACCENT3, linestyle=":", alpha=0.7)
        ax1.set_xlabel("Quality")
        ax1.set_ylabel("CPU ms / GPU ms")
        ax1.set_title("Compress speedup")
        ax1.grid(True, alpha=0.3, axis="y")

    if dq:
        ax2.bar(dq, dec_up, width=6, color=ACCENT1, alpha=0.85)
        ax2.axhline(1, color=ACCENT3, linestyle=":", alpha=0.7)
        ax2.set_xlabel("Quality")
        ax2.set_ylabel("CPU ms / GPU ms")
        ax2.set_title("Decompress speedup")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_speedup.png"), dpi=150)
    plt.close()


def chart_summary_table(data: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    metrics = [
        ("Compression ratio", "compression_ratio", "x", ".1f"),
        ("PSNR (dB)", "avg_psnr", "", ".2f"),
        ("SSIM", "avg_ssim", "", ".4f"),
        ("Compress FPS", "compress_fps", "", ".1f"),
        ("Decode FPS", "decode_fps", "", ".1f"),
        ("Compress ms", "compress_ms", "", ".1f"),
        ("Decode ms", "decode_total_ms", "", ".1f"),
    ]

    gpu = data.get("gpu_q50", {})
    cpu = data.get("cpu_q50", {})
    rows = []
    for name, field, suffix, fmt in metrics:
        gv, cv = gpu.get(field), cpu.get(field)
        gs = f"{gv:{fmt}}{suffix}" if gv is not None else "n/a"
        cs = f"{cv:{fmt}}{suffix}" if cv is not None else "n/a"
        if gv and cv and field in ("compress_ms", "decode_total_ms"):
            sp = f"{cv / gv:.1f}x"
        elif gv and cv and field in ("compress_fps", "decode_fps"):
            sp = f"{gv / cv:.1f}x"
        else:
            sp = "-"
        rows.append([name, gs, cs, sp])

    table = ax.table(cellText=rows, colLabels=["Metric", "CUDA", "OpenMP", "Ratio"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title("Summary at q=50", fontsize=14, pad=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="benchmark_results directory")
    ap.add_argument("--charts-dir", default="charts", help="output directory for PNGs")
    args = ap.parse_args()

    data = load_metrics(args.results_dir)
    qualities = qualities_in_data(data)
    if not qualities:
        print("No gpu_q*/cpu_q* entries in metrics.json", file=sys.stderr)
        return 1

    os.makedirs(args.charts_dir, exist_ok=True)

    chart_psnr(data, qualities, args.charts_dir)
    chart_ssim(data, qualities, args.charts_dir)
    chart_field(
        data,
        qualities,
        "compression_ratio",
        "Ratio",
        "Compression ratio",
        "chart_compression_ratio.png",
        args.charts_dir,
    )
    chart_field(
        data, qualities, "decode_fps", "FPS", "Decode FPS", "chart_decode_fps.png", args.charts_dir
    )
    chart_field(
        data, qualities, "compress_fps", "FPS", "Compress FPS", "chart_compress_fps.png", args.charts_dir
    )
    chart_rate_distortion(data, qualities, args.charts_dir)
    chart_speedup(data, qualities, args.charts_dir)
    chart_summary_table(data, args.charts_dir)

    print(f"Wrote charts to {args.charts_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
