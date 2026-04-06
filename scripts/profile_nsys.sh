#!/usr/bin/env bash


set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p reports

: "${FRAMES:=Frames}"
: "${OUT_REP:=reports/nsys_flipbook}"
EXE="${EXE:-$ROOT/build/flipbook_cuda}"

if [[ ! -x "$EXE" ]]; then
  echo "Зберіть flipbook_cuda: mkdir -p build && cd build && cmake .. && cmake --build ."
  echo "Або вкажіть EXE=/шлях/до/flipbook_cuda"
  exit 1
fi
if [[ ! -d "$ROOT/$FRAMES" ]]; then
  echo "Немає каталогу кадрів: $ROOT/$FRAMES (задайте FRAMES=...)"
  exit 1
fi

TMP_BIN="$ROOT/reports/_nsys_tmp.bin"
rm -f "$TMP_BIN"

echo "=== Nsight Systems → ${OUT_REP}.nsys-rep (compress) ==="
nsys profile \
  --trace=cuda,nvtx,osrt,syscall,cublas,cudnn \
  --cuda-memory-usage=true \
  --output "$OUT_REP" \
  --force-overwrite true \
  "$EXE" compress -q 50 -b 8 "$ROOT/$FRAMES" "$TMP_BIN"

echo ""
echo "Відкрийте звіт: nsys-ui ${OUT_REP}.nsys-rep"
echo "Шукайте: великі проміжки між cudaMemcpy (sync) і наступним kernel, cudaStreamSynchronize, CPU gaps."
