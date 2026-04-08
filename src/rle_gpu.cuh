#pragma once
#include <cstdint>
#include <cstddef>

struct GpuHuffNode {
    int16_t children[2];
    int16_t symbol;
};

void rle_gpu_init(int ch, size_t max_elements);
void rle_gpu_cleanup();

void cuda_compute_histogram(int ch, uint32_t* h_hist, void* stream = nullptr);

void cuda_rle_encode_indexed(int ch, const int16_t* d_coeffs, int num_blocks, int block_size,
                             void* d_metadata_ptr, void* stream = nullptr);

void cuda_rle_download_to_host(int ch, void* dst, size_t nbytes);

void cuda_huffman_pack_gpu_indexed(int ch, int num_blocks,
                                   const uint32_t* h_code_bits, const uint8_t* h_code_lens,
                                   uint8_t** d_out_packed, size_t* out_bytes,
                                   uint32_t* d_block_bit_lengths,
                                   void* d_metadata_ptr = nullptr,
                                   void* stream = nullptr);

void cuda_huffman_download_block_bit_lengths(int ch, uint32_t* h_dst, int num_blocks);

void cuda_prepare_huffman_codebook_gpu(int ch, void* stream_ptr);

void cuda_gpu_decode_entropy(int ch,
                              const uint8_t* h_packed_data, size_t packed_bytes,
                              const uint32_t* h_block_bit_lengths, int num_blocks,
                              const uint32_t* h_freq, int block_size,
                              void* stream = nullptr);

int16_t* cuda_get_decoded_coeffs(int ch);
