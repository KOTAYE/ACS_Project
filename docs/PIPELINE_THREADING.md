# Multi-threaded I/O Pipeline

Our codec uses a **Parallel Pipeline** to make sure the GPU is never waiting for the next image. We use a "Producer-Consumer" model with multiple threads.

## Theoretical Architecture

The pipeline is split into three main stages that run at the same time:

### 1. The I/O Thread (Producer)
This thread does nothing but read raw bytes from the disk.
- It finds all images in the folder.
- It reads them into memory as fast as possible.
- It puts these "raw jobs" into a **`ThreadSafeQueue`**.

### 2. The Parser Pool (Workers)
We launch a pool of CPU threads (usually 8-16 threads depending on your CPU).
- Each thread takes a raw image from the queue.
- It decodes the format (PNG via `stb_image` or EXR via `tinyexr`).
- It performs color conversion (RGB to YCbCr).
- It pushes the ready-to-process frame into the **`OrderedFrameBuffer`**.

### 3. The GPU Thread (Consumer)
This is the main thread that talks to the CUDA API.
- It waits for frames to appear in the `OrderedFrameBuffer`.
- It takes them **in the correct order** (e.g., Frame 0, then 1, then 2).
- It sends them to the GPU for compression.

## Why use an Ordered Buffer?
Since we parse images in parallel, Frame 5 might finish before Frame 2. However, for video compression (especially if we use delta-encoding), we need the frames in the right order. The **`OrderedFrameBuffer`** acts like a "sorting station" that holds Frame 5 until Frames 2, 3, and 4 are ready.

## Scalability
- **CPU Bound?** If decoding PNGs is too slow, you can increase the number of parser threads.
- **I/O Bound?** If reading from HDD is the bottleneck, the GPU will wait. We recommend using an SSD for best results (490+ FPS).

## Error Handling
If any image fails to load (wrong size or corrupted file), the pipeline calls **`set_fail()`**. This immediately stops all other stages and prints an error message, preventing the program from crashing or saving a broken binary.
