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
- `--adaptive-roi`: Enable adaptive ROI compression (available for all backends).
- `--roi-strength <0.0-1.0>`: Control adaptive ROI intensity (default: `0.55`).
- `--heatmap-video <out.mp4>`: Generate a separate compression heatmap video after compression.

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
- **Compression Heatmap Video**: Build an overlay video that shows where compression was stronger/weaker.
  ```bash
  python scripts/make_heatmap_video.py --manifest output.bin.heatmap_data/manifest.tsv --output heatmap.mp4
  ```

- **Live camera compression (target Mbps)**: Capture webcam frames and compress in real-time chunks using CUDA backend.
  ```bash
  python scripts/live_camera_compress.py --target-mbps 8 --camera 0 --fps 30 --chunk-seconds 1.0 --quality 70 --scene-cut-threshold 22
  ```
  Output chunks are written to `live_stream_output/chunks/` with `live_stream_output/manifest.tsv`.

- **Live video chat transport (no .bin files)**: real-time camera sender/receiver over TCP with compressed frames and target Mbps control.
  ```bash
  # Terminal 1 (receiver)
  python scripts/live_video_chat.py receiver --bind 0.0.0.0 --port 5000 --show-latency

  # Terminal 2 (sender)
  python scripts/live_video_chat.py sender --host 127.0.0.1 --port 5000 --camera 0 --target-mbps 3.0 --preview --quality-slider --max-fps
  ```
  Press `q` in sender/receiver window to stop.
  Use the `Quality` slider in sender window to change compression on the fly (bitrate changes accordingly).

- **Live video chat with your CUDA codec (`flipbook_cuda`)**: uses your project compressor/decompressor in real-time chunks over TCP.
  Build the runtime library first:
  ```bash
  cmake --build build --config Release --target realtime_codec
  ```
  ```bash
  # Terminal 1 (receiver)
  python scripts/live_video_chat_cuda.py receiver --bind 0.0.0.0 --port 6000 --show-latency

  # Terminal 2 (sender)
  python scripts/live_video_chat_cuda.py sender --host 127.0.0.1 --port 6000 --camera 0 --target-mbps 2.0 --preview --quality-slider --max-fps
  ```
  This mode uses in-memory codec packets (no per-frame subprocess/temp chunk files).

---

## Technical Documentation

For deep technical details, check the internal documentation:

- **[Block Size Scaling](docs/BLOCK_SIZE_SCALING.md)**: How block sizes (8, 16, 32) affect speed and quality.
- **[CUDA Streams & Memory](docs/CUDA_STREAMS_PINNED.md)**: Details on Pinned Memory and asynchronous data transfers.
- **[GPU Huffman Bitstream](docs/GPU_HUFFMAN_BITSTREAM.md)**: Our parallel bit-packing format for GPU entropy coding.
- **[Pipeline Threading](docs/PIPELINE_THREADING.md)**: Information on the producer-consumer model for I/O and parsing.
- **[Profiling & Benchmarking](docs/PROFILING_AND_BASELINE.md)**: Guide on using Nsight Systems and capturing performance baselines.

---

## Realtime In-Memory API (CUDA)

For low-latency transport without per-frame disk I/O, a manual in-memory codec API is available in:

- `src/realtime/in_memory_codec.h`
- `src/realtime/in_memory_codec.cpp`

It provides:
- `CudaRealtimeEncoder::encode_frame(...)` -> `RealtimeEncodedFrame` packet in memory
- `CudaRealtimeDecoder::decode_frame(...)` -> decoded interleaved RGB frame in memory

Use this API inside your own TCP/UDP/WebRTC loop to avoid subprocess+temp-file overhead.

---

## File Formats
The project uses a custom `.bin` format (`FLI3`) which is compatible across all backends. You can compress on a GPU and decompress with OpenMP seamlessly.
