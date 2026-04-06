#!/usr/bin/env bash
# Повний прогін: збірка, self-tests, round-trip (CUDA/OMP/Serial), легкий бенчмарк,
# metrics.json, графіки в charts/, JSON розмірів блоків DCT.
set -euo pipefail
# Щоб кирилиця в echo не перетворювалась на "????" у терміналі без UTF-8
export LC_ALL=C.UTF-8 LANG=C.UTF-8 2>/dev/null || true
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="${BUILD_DIR:-build}"
RESULTS="${BENCHMARK_RESULTS_DIR:-$ROOT/benchmark_results}"
CHARTS="${CHARTS_DIR:-$ROOT/charts}"
FRAMES="${TEST_FRAMES:-$ROOT/_test_frames}"

echo "=== [1/6] CMake build (Release) ==="
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

echo ""
echo "=== [2/6] entropy_reference --self-test ==="
"$BUILD/entropy_reference" --self-test

echo ""
echo "=== [3/6] Тестові кадри + round-trip (CUDA, OpenMP, Serial) ==="
if [[ ! -d "$FRAMES" ]] || [[ -z "$(find "$FRAMES" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' \) -print -quit 2>/dev/null)" ]]; then
  echo "Генерую $ROOT/_test_frames (4× PNG 256×128 RGB)…"
  mkdir -p "$ROOT/_test_frames"
  python3 -c "
from PIL import Image
from pathlib import Path
out = Path('$ROOT') / '_test_frames'
out.mkdir(exist_ok=True)
for i in range(4):
    Image.new('RGB', (256, 128), (i * 60, 100, 200)).save(out / f'frame_{i:04d}.png')
print('OK', out)
"
  FRAMES="$ROOT/_test_frames"
fi

BIN_CUDA="$ROOT/_ci_cuda.bin"
BIN_OMP="$ROOT/_ci_omp.bin"
BIN_SER="$ROOT/_ci_serial.bin"
OUT_C="$ROOT/_ci_recon_cuda"
OUT_O="$ROOT/_ci_recon_omp"
OUT_S="$ROOT/_ci_recon_serial"
rm -rf "$OUT_C" "$OUT_O" "$OUT_S"

if [[ -x "$BUILD/flipbook_cuda" ]]; then
  echo "CUDA compress/decompress…"
  "$BUILD/flipbook_cuda" compress -q 85 "$FRAMES" "$BIN_CUDA"
  "$BUILD/flipbook_cuda" decompress "$BIN_CUDA" "$OUT_C"
else
  echo "Пропуск CUDA (немає flipbook_cuda)."
fi

echo "OpenMP compress/decompress…"
"$BUILD/flipbook_omp" compress -q 85 "$FRAMES" "$BIN_OMP"
"$BUILD/flipbook_omp" decompress "$BIN_OMP" "$OUT_O"

echo "Serial compress/decompress…"
"$BUILD/flipbook_serial" compress -q 85 "$FRAMES" "$BIN_SER"
"$BUILD/flipbook_serial" decompress "$BIN_SER" "$OUT_S"

echo ""
echo "=== [4/6] Легкий бенчмарк (GPU+CPU, q ∈ {40,50,70,100}) → $RESULTS ==="
mkdir -p "$RESULTS"
# Щоб compute_metrics не змішував кадри з попередніх прогонів (інша кількість/розмір)
rm -rf "$RESULTS"/gpu_q*_recon "$RESULTS"/cpu_q*_recon
rm -f "$RESULTS"/gpu_compress_q*.log "$RESULTS"/gpu_decompress_q*.log \
      "$RESULTS"/cpu_compress_q*.log "$RESULTS"/cpu_decompress_q*.log
for Q in 40 50 70 100; do
  if [[ -x "$BUILD/flipbook_cuda" ]]; then
    echo "--- GPU q=$Q ---"
    "$BUILD/flipbook_cuda" compress -q "$Q" "$FRAMES" "$RESULTS/_gpu_q${Q}.bin" 2>&1 | tee "$RESULTS/gpu_compress_q${Q}.log"
    "$BUILD/flipbook_cuda" decompress "$RESULTS/_gpu_q${Q}.bin" "$RESULTS/gpu_q${Q}_recon" 2>&1 | tee "$RESULTS/gpu_decompress_q${Q}.log"
    rm -f "$RESULTS/_gpu_q${Q}.bin"
  else
    echo "--- GPU q=$Q (пропуск) ---"
    : > "$RESULTS/gpu_compress_q${Q}.log"
    : > "$RESULTS/gpu_decompress_q${Q}.log"
  fi
  echo "--- CPU q=$Q ---"
  "$BUILD/flipbook_omp" compress -q "$Q" "$FRAMES" "$RESULTS/_cpu_q${Q}.bin" 2>&1 | tee "$RESULTS/cpu_compress_q${Q}.log"
  "$BUILD/flipbook_omp" decompress "$RESULTS/_cpu_q${Q}.bin" "$RESULTS/cpu_q${Q}_recon" 2>&1 | tee "$RESULTS/cpu_decompress_q${Q}.log"
  rm -f "$RESULTS/_cpu_q${Q}.bin"
done

echo ""
echo "=== [5/6] metrics.json + charts (Python: pillow, matplotlib) ==="
python3 "$ROOT/scripts/compute_metrics.py" "$FRAMES" "$RESULTS"
python3 "$ROOT/scripts/generate_real_charts.py" "$RESULTS"

echo ""
echo "=== [6/6] Графік DCT block size (CUDA, -b 8/16/32, q=50) ==="
if [[ -x "$BUILD/flipbook_cuda" ]]; then
  python3 "$ROOT/scripts/chart_dct_block_size.py" "$FRAMES" "$BUILD/flipbook_cuda" "$CHARTS"
else
  echo "Пропуск chart_dct_block_size (немає CUDA)."
fi

echo ""
echo "Готово. Графіки: $CHARTS/"
echo "Метрики: $RESULTS/metrics.json"
ls -la "$CHARTS"/*.png 2>/dev/null | head -20 || true
