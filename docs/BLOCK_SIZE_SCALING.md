# DCT Block Size Scaling (8x8, 16x16, 32x32)

Our codec supports three block sizes: **8x8**, **16x16**, and **32x32**. You can select the size using the `-b` flag during compression.

## Implementation Details

To keep the code clean, we use **C++ Templates**. This means we write the DCT (Discrete Cosine Transform) logic once, and the compiler generates specialized versions for each block size.

- **GPU**: Uses `encode_blocks_kernel<BS>` where `BS` is the block size.
- **CPU**: Uses a separable DCT implementation that scales based on the input size.

## Quality vs Performance Trade-offs

Choosing a block size is a balance between compression quality and processing speed.

### Larger Blocks (16x16, 32x32)
- **Better Quality/Ratio**: Larger blocks allow the DCT to capture more structure in smooth areas (like gradients). This usually results in smaller files and fewer "blocking" artifacts.
- **Memory Pressure**: On the GPU, larger blocks use more **Shared Memory**. A 32x32 block needs 12 KiB of shared memory, while an 8x8 block only needs 768 bytes.
- **Throughput**: Larger blocks mean there are fewer blocks overall (a 32x32 block covers 16 times more area than 8x8), which can reduce overhead.

### Smaller Blocks (8x8)
- **Fine Details**: Better at preserving sharp edges and high-frequency textures.
- **Higher Speed**: Calculations are simpler and finish faster per block.
- **Industry Standard**: 8x8 is the classic size used by JPEG.

## Resource Usage Table

| Block Size | Pixels | Shared Memory (Bytes) | Ideal Use Case |
|------------|--------|-----------------------|----------------|
| **8x8**    | 64     | 768                   | General purpose, high detail |
| **16x16**  | 256    | 3,072                 | High-resolution frames (2K/4K) |
| **32x32**  | 1,024  | 12,288                | Smooth gradients, maximum ratio |

## Benchmarking Different Sizes

You can automatically compare all three sizes by running our sweep script:
```bash
bash scripts/sweep_block_size.sh
```
This will print a performance table showing FPS and compression ratios for each size.
