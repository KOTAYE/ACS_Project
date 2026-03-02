#!/usr/bin/env python3
"""
Usage:
    python benchmark.py --runs 5 --input-dir Frames --quality 50
"""

import argparse
import subprocess
import time
import os
import sys
import shutil


def parse_args():
    p = argparse.ArgumentParser(description="Flipbook encoding benchmark")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of timed runs per backend (default: 3)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warm-up runs before timing (default: 1)")
    p.add_argument("--input-dir", default="Frames",
                   help="Input frames directory (default: Frames)")
    p.add_argument("--quality", type=int, default=50,
                   help="Compression quality 1-100 (default: 50)")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip CMake configure + build step")
    return p.parse_args()


def build_all(project_root):
    bench_src = os.path.join(project_root, "bench")
    build_dir = os.path.join(project_root, "build_bench")

    if not os.path.exists(build_dir):
        print("[BUILD] Configuring CMake...")
        r = subprocess.run(
            ["cmake", "-S", bench_src, "-B", build_dir,
             "-G", "Visual Studio 17 2022"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"CMake configure failed:\n{r.stderr}")
            sys.exit(1)

    print("[BUILD] Building all targets (Release)...")
    r = subprocess.run(
        ["cmake", "--build", build_dir, "--config", "Release"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"Build failed:\n{r.stderr}\n{r.stdout}")
        sys.exit(1)

    print("[BUILD] Done.\n")
    return build_dir


def robust_remove(path, retries=5, delay=0.2):
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"Warning: Could not remove {path}")

def run_once(exe, input_dir, quality, output_bin):
    """Run one compression, return elapsed seconds or None on failure."""
    robust_remove(output_bin)

    start = time.perf_counter()
    r = subprocess.run(
        [exe, "compress", "-q", str(quality), input_dir, output_bin],
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - start

    if r.returncode != 0:
        return None
    return elapsed


def benchmark_backend(name, exe, input_dir, quality, warmup, runs):
    """Warm up, then run `runs` timed iterations. Return list of times."""
    output_bin = f"_bench_{name.lower()}.bin"

    # Warm-up
    for w in range(warmup):
        t = run_once(exe, input_dir, quality, output_bin)
        if t is None:
            print(f"  Warm-up {w+1} FAILED")
            return []
        print(f"  Warm-up {w+1}/{warmup}: {t:.2f}s")

    # Timed runs
    times = []
    for i in range(runs):
        t = run_once(exe, input_dir, quality, output_bin)
        if t is None:
            print(f"  Run {i+1} FAILED")
            continue
        times.append(t)
        print(f"  Run {i+1}/{runs}: {t:.2f}s")

    # Clean up output binary
    robust_remove(output_bin)

    return times


def print_table(results):
    serial_avg = results.get("Serial", {}).get("avg")
    print("\n" + "=" * 52)
    print(f"  {'Backend':<10} {'Avg (s)':<12} {'Min (s)':<12} {'Speedup':<10}")
    print("-" * 52)
    for name in ["Serial", "OpenMP", "CUDA"]:
        if name not in results:
            continue
        d = results[name]
        sp = f"{serial_avg / d['avg']:.2f}x" if serial_avg else "—"
        print(f"  {name:<10} {d['avg']:<12.3f} {d['min']:<12.3f} {sp:<10}")
    print("=" * 52)


def plot_chart(results, runs, quality):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[CHART] matplotlib not installed — skipping chart.")
        print("        Install with: pip install matplotlib")
        return

    order = [n for n in ["Serial", "OpenMP", "CUDA"] if n in results]
    avgs = [results[n]["avg"] for n in order]
    mins = [results[n]["min"] for n in order]
    colors = {"CUDA": "#76b900", "OpenMP": "#0071c5", "Serial": "#888888"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, avgs,
                  color=[colors[n] for n in order],
                  edgecolor="white", linewidth=1.5, zorder=3)

    # Error bars showing min
    for bar, avg, mn in zip(bars, avgs, mins):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, avg + max(avgs) * 0.02,
                f"{avg:.2f}s", ha="center", va="bottom",
                fontweight="bold", fontsize=13)

    serial_avg = results.get("Serial", {}).get("avg")
    if serial_avg:
        for bar, avg in zip(bars, avgs):
            x = bar.get_x() + bar.get_width() / 2
            sp = serial_avg / avg
            if sp != 1.0:
                ax.text(x, avg / 2, f"{sp:.1f}x",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white")

    ax.set_ylabel("Average Time (seconds)", fontsize=12)
    ax.set_title(f"Encoding Benchmark  ({runs} runs, quality={quality})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(avgs) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    chart_path = "benchmark_results.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    print(f"\n[CHART] Saved to: {chart_path}")

    # Try to open
    if sys.platform == "win32":
        os.startfile(chart_path)


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))

    if not args.skip_build:
        build_dir = build_all(project_root)
    else:
        build_dir = os.path.join(project_root, "build_bench")

    exe_dir = os.path.join(build_dir, "Release")
    backends = {
        "CUDA":   os.path.join(exe_dir, "bench_cuda.exe"),
        "OpenMP": os.path.join(exe_dir, "bench_omp.exe"),
        "Serial": os.path.join(exe_dir, "bench_serial.exe"),
    }

    results = {}

    for name, exe in backends.items():
        if not os.path.exists(exe):
            print(f"[{name}] executable not found ({exe}), skipping.\n")
            continue

        print(f"[{name}] {args.warmup} warm-up + {args.runs} timed runs:")
        times = benchmark_backend(name, exe, args.input_dir,
                                  args.quality, args.warmup, args.runs)
        if times:
            results[name] = {
                "times": times,
                "avg": sum(times) / len(times),
                "min": min(times),
            }
            print(f"  → Average: {results[name]['avg']:.3f}s\n")
        else:
            print(f"  → All runs failed.\n")

    if not results:
        print("No results collected.")
        return

    print_table(results)
    plot_chart(results, args.runs, args.quality)

    # Final cleanup
    for f in os.listdir(project_root):
        if f.startswith("_bench_") and f.endswith(".bin"):
            robust_remove(os.path.join(project_root, f))

    print("\nDone. Temporary files cleaned up.")


if __name__ == "__main__":
    main()
