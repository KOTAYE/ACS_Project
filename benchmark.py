#!/usr/bin/env python3

"""
Usage:
    python3 benchmark.py --runs 3 --input-dir Frames --quality 80
"""

import argparse
import subprocess
import time
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Flipbook encoding benchmark")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of timed runs per backend (default: 3)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warm-up runs before timing (default: 1)")
    p.add_argument("--input-dir", default="Frames",
                   help="Input frames directory (default: Frames)")
    p.add_argument("--quality", type=int, default=80,
                   help="Compression quality 1-100 (default: 80)")
    return p.parse_args()

def robust_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except:
            pass

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
    """Warm up, then run iterations. Return list of times."""
    output_bin = f"_bench_{name.lower()}.bin"
    
    for w in range(warmup):
        t = run_once(exe, input_dir, quality, output_bin)
        if t is None:
            print(f"  Warm-up {w+1} FAILED")
            return []
        print(f"  Warm-up {w+1}/{warmup}: {t:.2f}s")

    times = []
    for i in range(runs):
        t = run_once(exe, input_dir, quality, output_bin)
        if t is None:
            print(f"  Run {i+1} FAILED")
            continue
        times.append(t)
        print(f"  Run {i+1}/{runs}: {t:.2f}s")

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

def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_root, "build")

    if not os.path.exists(build_dir):
        print(f"Error: Build directory not found at {build_dir}. Please run 'make' first.")
        sys.exit(1)

    backends = {
        "CUDA":   os.path.join(build_dir, "flipbook_cuda"),
        "OpenMP": os.path.join(build_dir, "flipbook_omp"),
        "Serial": os.path.join(build_dir, "flipbook_serial"),
    }

    results = {}

    for name, exe in backends.items():
        if not os.path.exists(exe):
            print(f"[{name}] binary not found at {exe}, skipping.\n")
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
            print(f"  → FAILED.\n")

    if not results:
        print("No results collected.")
        return

    print_table(results)

if __name__ == "__main__":
    main()
