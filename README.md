# Flipbook codec

JPEG-style flipbook compression: YCbCr → DCT → quantize → Huffman. CUDA backend plus OpenMP and serial CPU builds.

## Authors

- Denys Maletskiy ([@maletsden](https://github.com/maletsden))
- Viktor Syrotiuk ([@KOTAYE](https://github.com/KOTAYE))
- Yulian Zaiats ([@Scorpion1355](https://github.com/Scorpion1355))
- Artem Onyshchuk ([@Sneezyan123](https://github.com/Sneezyan123))
- Yarema Mykhasiak ([@YarkoMarko](https://github.com/YarkoMarko))

## Build

Requirements: CMake 3.20+, C++20, CUDA 11+, OpenMP, zlib.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Binaries: `flipbook_cuda`, `flipbook_omp`, `flipbook_serial` (under `build/` or `build/Release/`).

## Usage

```bash
./flipbook_cuda compress -q 75 -b 16 ./input_frames/ output.bin
./flipbook_cuda decompress output.bin ./restored_frames/
```

| Flag | Meaning |
|------|---------|
| `-q` | Quality 1–100 (default 50) |
| `-b` | DCT block size 8, 16, or 32 (default 8) |
| `--no-ycbcr` | Compress RGB planes without YCbCr |

Container format: `FLI3` (`.bin`), interchangeable across backends.

## Scripts

```bash
bash scripts/quick_test.sh          # smoke test
bash scripts/run_tests_and_charts.sh # metrics + charts (needs Python deps)
bash scripts/profile_gpu.sh         # nsys wrapper
pip install -r requirements-bench.txt
```

## Docs

- [Block sizes](docs/BLOCK_SIZE_SCALING.md)
- [GPU Huffman layout](docs/GPU_HUFFMAN_BITSTREAM.md)
- [Profiling & baselines](docs/PROFILING_AND_BASELINE.md)
