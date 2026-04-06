#!/usr/bin/env python3
"""
Знімає baseline продуктивності flipbook_cuda (compress + decompress) для порівняння після оптимізацій.

Парсить рядки [BENCHMARK] з stdout і зберігає JSON у baseline/runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_short(root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return (r.stdout or "").strip() or "unknown"
    except OSError:
        return "unknown"


def _parse_compress(line: str) -> dict | None:
    m = re.search(
        r"compress_ms=([0-9.]+)\s+frames=(\d+)\s+compress_fps=([0-9.]+)", line
    )
    if not m:
        return None
    return {
        "compress_ms": float(m.group(1)),
        "frames": int(m.group(2)),
        "compress_fps": float(m.group(3)),
    }


def _parse_decompress(line: str) -> dict | None:
    m = re.search(
        r"decode_total_ms=([0-9.]+)\s+frames=(\d+)\s+avg_ms=([0-9.]+)\s+decode_fps=([0-9.]+)",
        line,
    )
    if not m:
        return None
    return {
        "decode_total_ms": float(m.group(1)),
        "frames": int(m.group(2)),
        "avg_ms_per_frame": float(m.group(3)),
        "decode_fps": float(m.group(4)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture flipbook_cuda performance baseline")
    ap.add_argument("frames_dir", type=Path, help="Directory with PNG/JPEG frames")
    ap.add_argument(
        "--exe",
        type=Path,
        default=Path("build/flipbook_cuda"),
        help="Path to flipbook_cuda executable",
    )
    ap.add_argument("-q", "--quality", type=int, default=50)
    ap.add_argument("-b", "--block-size", type=int, default=8, choices=(8, 16, 32))
    ap.add_argument("--no-ycbcr", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("baseline/runs"),
        help="Directory for JSON output",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    exe = args.exe if args.exe.is_absolute() else root / args.exe
    frames = args.frames_dir if args.frames_dir.is_absolute() else root / args.frames_dir

    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 1
    if not frames.is_dir():
        print(f"Frames directory not found: {frames}", file=sys.stderr)
        return 1

    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_bin = out_dir / f"_baseline_tmp_{datetime.now(timezone.utc).strftime('%H%M%S')}.bin"
    tmp_recon = out_dir / "_baseline_recon"

    ycbcr_flag = ["--no-ycbcr"] if args.no_ycbcr else []

    cmd_c = [
        str(exe),
        "compress",
        "-q",
        str(args.quality),
        "-b",
        str(args.block_size),
        *ycbcr_flag,
        str(frames),
        str(tmp_bin),
    ]
    cmd_d = [str(exe), "decompress", str(tmp_bin), str(tmp_recon)]

    env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "0"}

    r1 = subprocess.run(cmd_c, cwd=str(root), capture_output=True, text=True, env=env)
    text_c = r1.stdout + r1.stderr
    if r1.returncode != 0:
        print(text_c, file=sys.stderr)
        print(f"compress failed ({r1.returncode})", file=sys.stderr)
        return r1.returncode

    r2 = subprocess.run(cmd_d, cwd=str(root), capture_output=True, text=True, env=env)
    text_d = r2.stdout + r2.stderr
    if r2.returncode != 0:
        print(text_d, file=sys.stderr)
        print(f"decompress failed ({r2.returncode})", file=sys.stderr)
        if tmp_bin.exists():
            tmp_bin.unlink()
        return r2.returncode

    compress_stats = None
    for line in text_c.splitlines():
        compress_stats = _parse_compress(line) or compress_stats

    decompress_stats = None
    for line in text_d.splitlines():
        decompress_stats = _parse_decompress(line) or decompress_stats

    if tmp_bin.exists():
        tmp_bin.unlink()
    if tmp_recon.exists():
        import shutil

        shutil.rmtree(tmp_recon, ignore_errors=True)

    record = {
        "schema": "flipbook_baseline_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev_short": _git_short(root),
        "executable": str(exe),
        "frames_dir": str(frames),
        "quality": args.quality,
        "block_size": args.block_size,
        "use_ycbcr": not args.no_ycbcr,
        "compress": compress_stats,
        "decompress": decompress_stats,
        "raw_tail_compress": "\n".join(text_c.splitlines()[-8:]),
        "raw_tail_decompress": "\n".join(text_d.splitlines()[-8:]),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"baseline_{stamp}_{record['git_rev_short']}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
