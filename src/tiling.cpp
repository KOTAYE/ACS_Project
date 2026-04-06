#include "tiling.h"
#include <cmath>
#include <algorithm>
#include <cstdint>

void extract_block_8x8(const uint8_t* in, int stride, int bx, int by, float out[64]) {
    int start_x = bx * 8;
    int start_y = by * 8;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            out[y * 8 + x] = static_cast<float>(in[(start_y + y) * stride + start_x + x]);
        }
    }
}

void insert_block_8x8(uint8_t* out, int stride, int bx, int by, const float in[64]) {
    int start_x = bx * 8;
    int start_y = by * 8;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            out[(start_y + y) * stride + start_x + x] = 
                static_cast<uint8_t>(std::clamp(std::roundf(in[y * 8 + x]), 0.0f, 255.0f));
        }
    }
}

void extract_block_n(const uint8_t* in, int stride, int bx, int by, int bs, float* out) {
    int x0 = bx * bs;
    int y0 = by * bs;
    for (int y = 0; y < bs; ++y)
        for (int x = 0; x < bs; ++x)
            out[y * bs + x] = static_cast<float>(in[(y0 + y) * stride + x0 + x]);
}

void insert_block_n(uint8_t* out, int stride, int bx, int by, int bs, const float* in) {
    int x0 = bx * bs;
    int y0 = by * bs;
    for (int y = 0; y < bs; ++y)
        for (int x = 0; x < bs; ++x)
            out[(y0 + y) * stride + x0 + x] =
                static_cast<uint8_t>(std::clamp(std::roundf(in[y * bs + x]), 0.0f, 255.0f));
}
