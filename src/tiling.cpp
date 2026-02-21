#include "tiling.h"

void extract_block_8x8(const float* channel, int padded_w,
                        int bx, int by, float block[64])
{
    const int x0 = bx * 8;
    const int y0 = by * 8;

    for (int r = 0; r < 8; ++r) {
        const float* row = channel + (y0 + r) * padded_w + x0;
        for (int c = 0; c < 8; ++c) {
            block[r * 8 + c] = row[c];
        }
    }
}

void insert_block_8x8(float* channel, int padded_w,
                       int bx, int by, const float block[64])
{
    const int x0 = bx * 8;
    const int y0 = by * 8;

    for (int r = 0; r < 8; ++r) {
        float* row = channel + (y0 + r) * padded_w + x0;
        for (int c = 0; c < 8; ++c) {
            row[c] = block[r * 8 + c];
        }
    }
}
