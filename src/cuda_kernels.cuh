#pragma once

#include <cstdint>

void cuda_init();
void cuda_cleanup();
void cuda_alloc_frame_buffers(int width, int height, int channels,
                              const float* luma_qm, const float* chroma_qm,
                              bool use_ycbcr = true);
void cuda_free_frame_buffers();

void cuda_upload_and_convert_rgb(const uint8_t* rgb_host,
                                 int width, int height, int channels);

void cuda_encode_channel(int ch, int16_t* coeff_out,
                         int padded_w, int padded_h, bool is_keyframe);

void cuda_decode_channel(int ch, const int16_t* coeff_in,
                         int padded_w, int padded_h, bool is_keyframe);

void cuda_download_and_convert_rgb(uint8_t* rgb_host,
                                   int width, int height, int channels);

void cuda_swap_recon();

void cuda_sync_channel(int ch);
void cuda_sync_all();
