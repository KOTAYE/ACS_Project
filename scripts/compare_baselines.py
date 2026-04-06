#!/usr/bin/env python3
"""Порівняння двох JSON з capture_baseline.py (відносні зміни %)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pct(old: float, new: float) -> str:
    if old == 0:
        return "n/a"
    return f"{100.0 * (new - old) / old:+.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    args = ap.parse_args()
    a = json.loads(args.before.read_text(encoding="utf-8"))
    b = json.loads(args.after.read_text(encoding="utf-8"))
    ca, da = a.get("compress") or {}, a.get("decompress") or {}
    cb, db = b.get("compress") or {}, b.get("decompress") or {}
    if not ca or not cb:
        print("Missing compress stats in one of files", file=sys.stderr)
        return 1
    print("compress_fps:", ca.get("compress_fps"), "→", cb.get("compress_fps"), pct(ca["compress_fps"], cb["compress_fps"]))
    print("compress_ms: ", ca.get("compress_ms"), "→", cb.get("compress_ms"), pct(ca["compress_ms"], cb["compress_ms"]))
    if da and db:
        print("decode_fps:  ", da.get("decode_fps"), "→", db.get("decode_fps"), pct(da["decode_fps"], db["decode_fps"]))
        print("avg_ms/frame:", da.get("avg_ms_per_frame"), "→", db.get("avg_ms_per_frame"),
              pct(da["avg_ms_per_frame"], db["avg_ms_per_frame"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
