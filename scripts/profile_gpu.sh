#!/bin/bash

INPUT_DIR=${1:-Frames}
QUALITY=${2:-10}
OUTPUT_BIN="/tmp/profile_bench.bin"

echo "=== Running Nsight Systems Profiling ==="
mkdir -p profiler_reports

nsys profile \
    --trace=cuda,osrt,nvtx \
    --stats=true \
    --force-overwrite=true \
    --output=profiler_reports/flipbook_profile_q${QUALITY} \
    ./build/flipbook_cuda compress -q ${QUALITY} "${INPUT_DIR}" "${OUTPUT_BIN}"

echo ""
echo "=== Done! Report saved in profiler_reports/ ==="
echo "You can open the .nsys-rep file in the Nsight Systems GUI for a visual timeline analysis."