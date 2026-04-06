#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${FRAMES:=Frames}"
: "${Q:=50}"
EXE="${EXE:-$ROOT/build/flipbook_cuda}"
FR_PATH="$ROOT/$FRAMES"

if [[ ! -x "$EXE" ]]; then
  echo "Зберіть: cmake -S . -B build && cmake --build build --target flipbook_cuda"
  exit 1
fi
if [[ ! -d "$FR_PATH" ]]; then
  echo "Немає $FR_PATH (задайте FRAMES=...)"
  exit 1
fi

echo "# Таблиця: розмір блоку DCT (Phase 2, §4)"
echo ""
echo "| block | compress_ms | fps | compressed_bytes | ratio vs raw |"
echo "|------:|------------:|----:|-----------------:|-------------:|"

RAW_BYTES=""
for B in 8 16 32; do
  BIN="$ROOT/benchmark_results/_phase2_b${B}.bin"
  mkdir -p "$ROOT/benchmark_results"
  rm -f "$BIN"
  LOG=$(mktemp)
  "$EXE" compress -q "$Q" -b "$B" "$FR_PATH" "$BIN" 2>&1 | tee "$LOG" | tail -n 3
  LINE=$(grep '\[BENCHMARK\]' "$LOG" || true)
  rm -f "$BIN" "$LOG"

  CM=$(echo "$LINE" | sed -n 's/.*compress_ms=\([0-9.]*\).*/\1/p')
  FPS=$(echo "$LINE" | sed -n 's/.*compress_fps=\([0-9.]*\).*/\1/p')
  CB=$(echo "$LINE" | sed -n 's/.*compressed_bytes=\([0-9]*\).*/\1/p')
  RAT=$(echo "$LINE" | sed -n 's/.*compression_ratio=\([0-9.]*\).*/\1/p')
  if [[ -z "$RAW_BYTES" ]]; then
    RAW=$(echo "$LINE" | sed -n 's/.*raw_bytes=\([0-9]*\).*/\1/p')
    RAW_BYTES="$RAW"
  fi
  echo "| $B | ${CM:-—} | ${FPS:-—} | ${CB:-—} | ${RAT:-—} |"
done
echo ""
echo "Параметри: FRAMES=$FRAMES Q=$Q. Скопіюйте таблицю в docs/BLOCK_SIZE_SCALING.md за потреби."
