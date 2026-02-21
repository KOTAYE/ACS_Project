#include "quant.h"
#include <cmath>

#define BLOCK_SIZE 64

void quantize_block_64(float block[64], float scale) {
    int i;
    if (scale <= 0.0f) return;
    for (i = 0; i < BLOCK_SIZE; i++)
        block[i] = (float)(int)roundf(block[i] / scale);
}

void dequantize_block_64(float block[64], float scale) {
    int i;
    for (i = 0; i < BLOCK_SIZE; i++)
        block[i] *= scale;
}
