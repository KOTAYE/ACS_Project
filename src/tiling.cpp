#include "tiling.h"

#include <algorithm>
#include <cmath>

void extract_block_n(const uint8_t* in, int stride, int bx, int by, int bs, float* out) {
    const int x0 = bx * bs;
    const int y0 = by * bs;
    for (int y = 0; y < bs; ++y)
        for (int x = 0; x < bs; ++x)
            out[y * bs + x] = static_cast<float>(in[(y0 + y) * stride + x0 + x]);
}

void insert_block_n(uint8_t* out, int stride, int bx, int by, int bs, const float* in) {
    const int x0 = bx * bs;
    const int y0 = by * bs;
    for (int y = 0; y < bs; ++y)
        for (int x = 0; x < bs; ++x)
            out[(y0 + y) * stride + x0 + x] =
                static_cast<uint8_t>(std::clamp(std::roundf(in[y * bs + x]), 0.0f, 255.0f));
}
