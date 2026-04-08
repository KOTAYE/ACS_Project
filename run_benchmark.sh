#!/bin/bash
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
    if [[ "${1:-}" == "frames_cuda" ]] && [[ -d "$ROOT/Frames" ]]; then
        echo "Каталог frames_cuda/ не знайдено (наприклад, після прибирання репозиторію). Використовую Frames/."
        FRAMES_DIR="$ROOT/Frames"
    else
        echo "Помилка: немає каталогу з кадрами: $FRAMES_DIR"
        echo "Створіть його або передайте шлях: $0 /шлях/до/png"
        exit 1
    fi
fi

if ! find "$FRAMES_DIR" -maxdepth 1 \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.exr' \) -print -quit | grep -q .; then
    echo "Помилка: у $FRAMES_DIR немає PNG/JPEG/EXR файлів (перевірте шлях)."
    exit 1
fi

BUILD_DIR="build"
RESULTS_DIR="benchmark_results"

mkdir -p "$RESULTS_DIR"

echo "=== Building project ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

echo ""
echo "=== Running GPU benchmarks ==="
for Q in 10 20 30 40 50 60 70 80 90 100; do
    OUT_BIN="$RESULTS_DIR/gpu_q${Q}.bin"
    OUT_DIR="$RESULTS_DIR/gpu_q${Q}_recon"
    echo "--- GPU quality=$Q ---"
    ./$BUILD_DIR/flipbook_cuda compress -q $Q "$FRAMES_DIR" "$OUT_BIN" 2>&1 | tee "$RESULTS_DIR/gpu_compress_q${Q}.log"
    ./$BUILD_DIR/flipbook_cuda decompress "$OUT_BIN" "$OUT_DIR" 2>&1 | tee "$RESULTS_DIR/gpu_decompress_q${Q}.log"
done

echo ""
echo "=== Running CPU (OpenMP) benchmarks ==="
for Q in 10 20 30 40 50 60 70 80 90 100; do
    OUT_BIN="$RESULTS_DIR/cpu_q${Q}.bin"
    OUT_DIR="$RESULTS_DIR/cpu_q${Q}_recon"
    echo "--- CPU quality=$Q ---"
    ./$BUILD_DIR/flipbook_omp compress -q $Q "$FRAMES_DIR" "$OUT_BIN" 2>&1 | tee "$RESULTS_DIR/cpu_compress_q${Q}.log"
    ./$BUILD_DIR/flipbook_omp decompress "$OUT_BIN" "$OUT_DIR" 2>&1 | tee "$RESULTS_DIR/cpu_decompress_q${Q}.log"
done

echo ""
echo "=== Computing PSNR and SSIM ==="
python3 scripts/compute_metrics.py "$FRAMES_DIR" "$RESULTS_DIR"

echo ""
echo "=== Generating charts ==="
python3 scripts/generate_real_charts.py "$RESULTS_DIR"

echo ""
echo "=== Done! Charts saved in charts/ ==="