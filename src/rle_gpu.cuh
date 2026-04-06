#pragma once
#include <cstdint>
#include <cstddef>

struct GpuHuffNode {
    int16_t children[2]; // [0]=left (bit 0), [1]=right (bit 1)
    int16_t symbol;      // >=0 for leaf, -1 for internal
};

void rle_gpu_init(int ch, size_t max_elements);
void rle_gpu_cleanup();

void cuda_compute_histogram(int ch, uint32_t* h_hist, void* stream = nullptr);

void cuda_rle_encode_indexed(int ch, const int16_t* d_coeffs, int num_blocks, int block_size,
                             uint32_t* out_rle_bytes, void* stream = nullptr);

// Copy packed RLE bytes from device (after cuda_rle_encode_indexed + sync)
void cuda_rle_download_to_host(int ch, void* dst, size_t nbytes);

// After GPU RLE: HuffmanBlockBitLengthKernel → CUB exclusive bit offsets → HuffmanBlockPackSerialKernel (same bits as huffman.cpp write_bits).
// Decode: HuffmanDecodePerBlockKernel (same semantics as huffman_decode_bit_window) → RleDecodePerBlockKernel.
void cuda_huffman_pack_gpu_indexed(int ch, int num_blocks,
                                   const uint32_t* h_code_bits, const uint8_t* h_code_lens,
                                   uint8_t** d_out_packed, size_t* out_bytes,
                                   uint32_t* d_block_bit_lengths,
                                   void* stream = nullptr);

// After cuda_huffman_pack_gpu_indexed + stream sync: copy per-block Huffman bit counts to host.
void cuda_huffman_download_block_bit_lengths(int ch, uint32_t* h_dst, int num_blocks);

// GPU-side entropy decoding (Huffman decode + RLE decode)
void cuda_gpu_decode_entropy(int ch,
                              const uint8_t* h_packed_data, size_t packed_bytes,
                              const uint32_t* h_block_bit_lengths, int num_blocks,
                              const uint16_t* h_freq, int block_size,
                              void* stream = nullptr);

// Get pointer to decoded coefficients after gpu_decode_entropy
int16_t* cuda_get_decoded_coeffs(int ch);
