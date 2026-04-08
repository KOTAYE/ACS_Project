# CUDA Streams and Pinned Memory

To achieve high throughput (490+ FPS), we use a combination of **Pinned Memory** and **CUDA Streams**. This allows us to hide the delay of moving data between the CPU and GPU.

## Pinned Memory (H2D Staging)

Standard CPU memory (pageable) is slow for GPU transfers because the OS can move it around. We use **`cudaHostAlloc`** to reserve "Pinned" (non-pageable) memory.

- **Fast DMA**: The GPU can read pinned memory directly.
- **Async ready**: Only pinned memory supports truly asynchronous (non-blocking) transfers.

We use a **Ping-Pong Buffer** strategy for the input frames. While the GPU is processing Frame 0, the CPU is already copying Frame 1 into the second pinned buffer.

## CUDA Streams (Parallel Tasks)

By default, CUDA does one thing at a time. We use multiple **Streams** to run tasks in parallel:

1. **`g_transfer_stream`**: Handles copying raw image data from CPU to GPU (H2D).
2. **`g_stream[0..2]`**: Three separate streams (one per color channel) that run the actual compression kernels.

### The Overlap Workflow
Because we use separate streams, the GPU can do two things at the exact same time:
- **Stream A**: Finish compression for the current frame.
- **Stream B**: Start downloading the next frame from the CPU.

This "hiding" of the transfer time is why the CUDA backend is so much faster than the CPU versions.

## Synchronization with Events

To make sure we don't start processing before the data arrives, we use **CUDA Events**:
- `g_evt_h2d_done`: Tells the GPU kernels it is safe to start.
- `g_evt_encode_slot_done`: Tells the CPU it is safe to overwrite the pinned buffer with a new frame.

## Summary Diagram
```text
[ CPU ] -> (Copy to Pinned) -> [ DMA ] -> (H2D Transfer Stream) -> [ GPU ]
                                  |
                                  +--> [ Compute Streams ] -> (DCT/Huffman)
```
The goal is to keep the [ Compute Streams ] busy 100% of the time.
