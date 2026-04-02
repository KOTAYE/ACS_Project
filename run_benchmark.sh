#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

REL_INPUT="${1:-Frames}"
if [[ "$REL_INPUT" = /* ]]; then
  FRAMES_DIR="$REL_INPUT"
else
  FRAMES_DIR="$ROOT/$REL_INPUT"
fi

if [[ ! -d "$FRAMES_DIR" ]]; then
  echo "error: frames directory not found: $FRAMES_DIR"
  echo "usage: $0 [path/to/frames]"
  exit 1
fi

if ! find "$FRAMES_DIR" -maxdepth 1 \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.exr' \) -print -quit | grep -q .; then
  echo "error: no PNG/JPEG/EXR in $FRAMES_DIR"
  exit 1
fi

BUILD="${BUILD_DIR:-build}"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$ROOT/benchmark_results}"
CHARTS_DIR="${CHARTS_DIR:-$ROOT/charts}"

echo "=== Build ==="
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"

EXE_CUDA="$BUILD/flipbook_cuda"
EXE_OMP="$BUILD/flipbook_omp"

mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/gpu_compress_q*.log "$RESULTS_DIR"/gpu_decompress_q*.log \
      "$RESULTS_DIR"/cpu_compress_q*.log "$RESULTS_DIR"/cpu_decompress_q*.log

echo ""
echo "=== GPU benchmarks ==="
for Q in 10 20 30 40 50 60 70 80 90 100; do
  BIN="$RESULTS_DIR/_gpu_q${Q}.bin"
  RECON="$RESULTS_DIR/gpu_q${Q}_recon"
  echo "--- GPU q=$Q ---"
  "$EXE_CUDA" compress -q "$Q" "$FRAMES_DIR" "$BIN" 2>&1 | tee "$RESULTS_DIR/gpu_compress_q${Q}.log"
  "$EXE_CUDA" decompress "$BIN" "$RECON" 2>&1 | tee "$RESULTS_DIR/gpu_decompress_q${Q}.log"
  rm -f "$BIN"
done

echo ""
echo "=== CPU (OpenMP) benchmarks ==="
for Q in 10 20 30 40 50 60 70 80 90 100; do
  BIN="$RESULTS_DIR/_cpu_q${Q}.bin"
  RECON="$RESULTS_DIR/cpu_q${Q}_recon"
  echo "--- CPU q=$Q ---"
  "$EXE_OMP" compress -q "$Q" "$FRAMES_DIR" "$BIN" 2>&1 | tee "$RESULTS_DIR/cpu_compress_q${Q}.log"
  "$EXE_OMP" decompress "$BIN" "$RECON" 2>&1 | tee "$RESULTS_DIR/cpu_decompress_q${Q}.log"
  rm -f "$BIN"
done

echo ""
echo "=== Metrics and charts ==="
python3 "$ROOT/scripts/compute_metrics.py" "$FRAMES_DIR" "$RESULTS_DIR"
python3 "$ROOT/scripts/generate_real_charts.py" "$RESULTS_DIR" --charts-dir "$CHARTS_DIR"

echo "Done. Charts: $CHARTS_DIR/  Metrics: $RESULTS_DIR/metrics.json"
