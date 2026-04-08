#pragma once
#include <cstdint>

void extract_block_8x8(const uint8_t* in, int stride, int bx, int by, float out[64]);
void insert_block_8x8(uint8_t* out, int stride, int bx, int by, const float in[64]);

void extract_block_n(const uint8_t* in, int stride, int bx, int by, int bs, float* out);
void insert_block_n(uint8_t* out, int stride, int bx, int by, int bs, const float* in);
