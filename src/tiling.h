#pragma once

void extract_block_n(const uint8_t* in, int stride, int bx, int by, int bs, float* out);
void insert_block_n(uint8_t* out, int stride, int bx, int by, int bs, const float* in);
