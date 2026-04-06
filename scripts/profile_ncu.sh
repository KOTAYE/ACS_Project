#!/usr/bin/env bash


set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p reports

: "${FRAMES:=Frames}"
EXE="${EXE:-$ROOT/build/flipbook_cuda}"
OUT_BASE="${OUT_BASE:-reports/ncu_flipbook}"

if [[ ! -x "$EXE" ]]; then
  echo "Зберіть flipbook_cuda або задайте EXE="
  exit 1
fi
if [[ ! -d "$ROOT/$FRAMES" ]]; then
  echo "Немає каталогу кадрів: $ROOT/$FRAMES"
  exit 1
fi

TMP_BIN="$ROOT/reports/_ncu_tmp.bin"
rm -f "$TMP_BIN"

echo "=== Nsight Compute → ${OUT_BASE}.ncu-rep ==="
ncu --target-processes all \
  --set full \
  -f \
  -o "$OUT_BASE" \
  "$EXE" compress -q 50 -b 8 "$ROOT/$FRAMES" "$TMP_BIN"

echo ""
echo "Відкрийте ${OUT_BASE}.ncu-rep у Nsight Compute."
echo "Експорт CSV для analyze_ncu.py: у GUI Section → Export (або Summary table → CSV)."
echo "Потім: python3 analyze_ncu.py <файл.csv>"
