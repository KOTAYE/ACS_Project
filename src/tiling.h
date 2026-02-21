#pragma once

void extract_block_8x8(const float* channel, int padded_w,
                        int bx, int by, float block[64]);
void insert_block_8x8(float* channel, int padded_w,
                       int bx, int by, const float block[64]);
