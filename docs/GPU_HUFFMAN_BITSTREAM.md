# GPU Huffman Bitstream (FLI3 Format)

Standard Huffman coding is a serial process—you usually need to finish reading one symbol before you can find the next one. To make this work on a GPU, we use a **per-block bitstream** format.

## The Problem: Parallelism
If we write one long bitstream for the whole image, only one GPU thread can decode it. This would be a massive bottleneck.

## The Solution: Segmented Packing
We split the image into 8x8 (or 16x16/32x32) blocks and give each block its own starting position in the file.

### Step 1: Bit Counting (Analysis)
Before we save anything, a GPU kernel (`HuffmanBlockBitLengthKernel`) calculates exactly how many bits each block will take after compression.

### Step 2: Offset Calculation (Scan)
We use a **Prefix Sum** (via NVIDIA CUB) to calculate the "Start Bit" for every block. 
- Block 0 starts at bit 0.
- Block 1 starts at bit [length of Block 0].
- Block 2 starts at bit [length of Block 0 + length of Block 1].

### Step 3: Parallel Packing
Now, every block has its own "reserved" space in the global bitstream. Thousands of GPU threads can write their data simultaneously because they never overlap.

## Binary Format Structure

Each color plane in our binary file follows this structure:

1. **Header**:
   - `rle_bytes`: Total size if it were uncompressed.
   - `enc_len`: Final size in bytes after Huffman compression.
   - `num_blocks`: Total number of tiles.
   - `huffman_freq`: Frequency table (used to rebuild the tree).
2. **Indexing**:
   - `block_bit_lengths`: An array of sizes (in bits) for every single block.
3. **Payload**:
   - The actual compressed bits.

## Why this is fast
During decompression, the GPU reads the `block_bit_lengths` array. It immediately knows exactly where in the file every block starts. It can then launch one GPU thread per block, achieving **2400+ FPS** during playback.
