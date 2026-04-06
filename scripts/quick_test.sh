#!/usr/bin/env bash
# Build + minimal GPU round-trip + PSNR (CUDA + Python3 + Pillow)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="${BUILD_DIR:-build}"
FRAMES="${1:-$ROOT/_test_frames}"

mkdir -p "$BUILD"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

if [[ ! -d "$FRAMES" ]] || [[ -z "$(find "$FRAMES" -maxdepth 1 -name '*.png' -print -quit)" ]]; then
  echo "No PNG frames in $FRAMES — generating $ROOT/_test_frames (256x128 RGB x4)"
  mkdir -p "$ROOT/_test_frames"
  python3 -c "
from PIL import Image
from pathlib import Path
out = Path('$ROOT') / '_test_frames'
out.mkdir(exist_ok=True)
for i in range(4):
    Image.new('RGB', (256, 128), (i * 60, 100, 200)).save(out / f'frame_{i:04d}.png')
print(out)
"
  FRAMES="$ROOT/_test_frames"
fi

BIN="$ROOT/_quick_cuda.bin"
OUT="$ROOT/_quick_recon"
rm -rf "$OUT"
"$BUILD/flipbook_cuda" compress -q 100 --no-ycbcr "$FRAMES" "$BIN"
"$BUILD/flipbook_cuda" decompress "$BIN" "$OUT"

python3 -c "
import glob, os, re, sys
import numpy as np
from PIL import Image
root = '$ROOT'
fr = '$FRAMES'
rec = '$OUT'
def key(p):
    n = [int(x) for x in re.findall(r'\d+', os.path.basename(p))]
    return tuple(n) if n else (p,)
origs = sorted(glob.glob(os.path.join(fr, '*.png')), key=key)
recons = sorted(glob.glob(os.path.join(rec, '*.png')), key=key)
if not origs or not recons:
    print('Missing images'); sys.exit(1)
psnrs = []
for o, r in zip(origs, recons):
    a = np.array(Image.open(o)).astype(float)
    b = np.array(Image.open(r)).astype(float)
    if a.shape != b.shape:
        print('Shape mismatch', o, a.shape, r, b.shape); sys.exit(1)
    mse = ((a - b) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse) if mse > 0 else 99
    psnrs.append(psnr)
avg = sum(psnrs) / len(psnrs)
print('GPU round-trip (--no-ycbcr q=100): avg PSNR = {:.2f} dB ({} frames)'.format(avg, len(psnrs)))
if min(psnrs) < 25:
    print('ERROR: PSNR too low (expected ~35+ at q=100 for this synthetic data).', file=sys.stderr)
    sys.exit(2)
"

echo "quick_test OK"
