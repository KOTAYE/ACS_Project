# Flipbook Compression Pipeline

High-performance flipbook sequence compression for GPU and CPU. This project implements a JPEG-like pipeline (YCbCr → DCT → Quantization → Huffman) optimized for NVIDIA GPUs (CUDA) and multi-core CPUs (OpenMP).

## Authors (Team)
- **Denys Maletskiy** ([@maletsden](https://github.com/maletsden))
- **Viktor Syrotiuk** ([@KOTAYE](https://github.com/KOTAYE))
- **Yulian Zaiats** ([@Scorpion1355](https://github.com/Scorpion1355))
- **Artem Onyshchuk** ([@Sneezyan123](https://github.com/Sneezyan123))
- **Yarema Mykhasiak** ([@YarkoMarko](https://github.com/YarkoMarko))

---

## Prerequisites

Before you start, make sure you have these tools installed:

- **CMake** (v3.20 or newer)
- **C++ Compiler** (GCC, MSVC, or Clang with C++20 support)
- **CUDA Toolkit** (v11.0 or newer) - *Required for GPU acceleration*
- **OpenMP** - *Required for CPU parallelization*
- **zlib** - *Required for .exr file support*

### Python setup (for benchmarks and charts)
```bash
pip install matplotlib pillow numpy opencv-python
```

---

## Getting Started

### 1. Build the Project
Use CMake to configure and build the executables:

```bash
# Configure the build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build --config Release -j
```

After building, you will find 3 main programs in `build/` (or `build/Release/`):
- `flipbook_cuda`: Uses NVIDIA GPU for maximum speed.
- `flipbook_omp`: Uses multi-core CPU via OpenMP.
- `flipbook_serial`: Standard single-threaded CPU version.

---

## How to Use

### Compression
To compress a folder of images (PNG, JPG, or EXR):
```bash
./flipbook_cuda compress -q 75 -b 16 ./input_frames/ output.bin
```
**Flags:**
- `-q <1-100>`: Set quality (higher is better). Default is 50.
- `-b <8|16|32>`: Set DCT block size. Larger blocks can improve compression. Default is 8.
- `--no-ycbcr`: Skip color conversion (compress RGB directly).

### Decompression
To restore images from a binary file:
```bash
./flipbook_cuda decompress output.bin ./restored_frames/
```

---

## Analytics and Scripts

We provide several scripts to test performance and quality:

- **Full Benchmark**: Run all backends and generate charts.
  ```bash
  bash scripts/run_tests_and_charts.sh
  ```
- **Quick Test**: Verify end-to-end functionality of the CUDA backend.
  ```bash
  bash scripts/quick_test.sh
  ```
- **Nsight Profiling**: Generate GPU performance reports.
  ```bash
  bash scripts/profile_gpu.sh
  ```

---

## Technical Documentation

For deep technical details, check the internal documentation:

- **[Block Size Scaling](docs/BLOCK_SIZE_SCALING.md)**: How block sizes (8, 16, 32) affect speed and quality.
- **[CUDA Streams & Memory](docs/CUDA_STREAMS_PINNED.md)**: Details on Pinned Memory and asynchronous data transfers.
- **[GPU Huffman Bitstream](docs/GPU_HUFFMAN_BITSTREAM.md)**: Our parallel bit-packing format for GPU entropy coding.
- **[Pipeline Threading](docs/PIPELINE_THREADING.md)**: Information on the producer-consumer model for I/O and parsing.
- **[Profiling & Benchmarking](docs/PROFILING_AND_BASELINE.md)**: Guide on using Nsight Systems and capturing performance baselines.

---

## File Formats
The project uses a custom `.bin` format (`FLI3`) which is compatible across all backends. You can compress on a GPU and decompress with OpenMP seamlessly.
