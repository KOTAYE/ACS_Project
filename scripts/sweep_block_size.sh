#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${FRAMES:=Frames}"
EXE="${EXE:-$ROOT/build/flipbook_cuda}"
Q="${Q:-50}"

if [[ ! -x "$EXE" ]]; then
  echo "Зберіть flipbook_cuda: cmake -S . -B build && cmake --build build"
  exit 1
fi
FR_PATH="$ROOT/$FRAMES"
if [[ ! -d "$FR_PATH" ]]; then
  echo "Немає каталогу $FR_PATH (задайте FRAMES=...)"
  exit 1
fi

OUT_DIR="$ROOT/reports"
mkdir -p "$OUT_DIR"

echo "=== Block size sweep: q=$Q frames=$FRAMES ==="
for B in 8 16 32; do
  BIN="$OUT_DIR/_sweep_b${B}.bin"
  rm -f "$BIN"
  echo ""
  echo "--- block_size=$B ---"
  "$EXE" compress -q "$Q" -b "$B" "$FR_PATH" "$BIN" 2>&1 | tee "$OUT_DIR/sweep_compress_b${B}.log" | tail -n 5
  rm -f "$BIN"
done
echo ""
echo "Повні логи: $OUT_DIR/sweep_compress_b*.log"