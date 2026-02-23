#include "quant.h"
#include <cmath>


void quantize_block_64(float block[64], float scale) {
    if (scale <= 0.0f) return;
    for (int i = 0; i < 64; ++i)
        block[i] = std::roundf(block[i] / scale);
}

void dequantize_block_64(float block[64], float scale) {
    int i;
    for (i = 0; i < 64; ++i)
        block[i] *= scale;
}
