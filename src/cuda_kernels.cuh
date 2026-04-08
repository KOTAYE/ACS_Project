#pragma once

#include <cstdint>

void cuda_init();
void cuda_cleanup();
void cuda_init_dct_constants();
void cuda_alloc_frame_buffers(int width, int height, int channels,
                              const float* luma_qm, const float* chroma_qm,
                              const int* zigzag, bool use_ycbcr, int block_size = 8);
void cuda_free_frame_buffers();

int16_t* cuda_alloc_pinned_coeffs(size_t num_elements);
void cuda_free_pinned_coeffs(int16_t* ptr);


void cuda_submit_frame_h2d(int frame_index, const uint8_t* ptr[3], int channels);
void cuda_download_planes(uint8_t* ptr[3], int channels);

void cuda_encode_channel(int ch, int pw, int ph, int block_size, bool is_keyframe, int src_slot);

void cuda_record_encode_slot_done(int slot, int last_ch);
void cuda_download_coeffs(int ch, int16_t* host_dst, int num_coeffs);
void cuda_rle_channel(int ch, int pw, int ph, int block_size, uint32_t* out_rle_bytes);
void cuda_hist_channel(int ch, uint32_t h_hist[256]);
void cuda_pack_channel(int ch, const uint32_t h_code_bits[256], const uint8_t h_code_lens[256],
                       uint8_t** d_packed_ptr, size_t* out_packed_bytes);

void cuda_decode_channel(int ch, const int16_t* coeff_in,
                         int pw, int ph, int block_size, bool is_keyframe);

void cuda_rle_channel_indexed(int ch, int num_blocks, int block_size, uint32_t* out_rle_bytes);
void cuda_rle_encode_async(int ch, int num_blocks, int block_size);
uint8_t* cuda_get_bitstream_ptr(int ch);
void cuda_get_pinned_metadata(int ch, uint32_t* rle_bytes, uint32_t* pack_bytes);
void cuda_hist_channel_new(int ch, uint32_t* h_hist);

void cuda_get_block_bit_lengths(int ch, int num_blocks, uint32_t* h_lengths);
void cuda_pack_channel_indexed(int ch, int num_blocks, int block_size,
                               const uint32_t h_code_bits[256], const uint8_t h_code_lens[256],
                               uint8_t** d_packed_ptr, size_t* out_packed_bytes,
                               uint32_t* d_block_bit_lengths_ptr);

void cuda_full_decode_channel(int ch,
                               const uint8_t* h_packed_data, size_t packed_bytes,
                               const uint32_t* h_block_bit_lengths, int num_blocks,
                               const uint32_t* h_freq,
                               int pw, int ph, int block_size, bool is_keyframe);

int16_t* cuda_channel_d_coeff(int ch);
void* cuda_channel_stream_ptr(int ch);

void cuda_swap_recon();
void cuda_sync_channel(int ch);
void cuda_sync_all();

void cuda_memcpy_to_host(void* host_ptr, const void* device_ptr, size_t bytes);
