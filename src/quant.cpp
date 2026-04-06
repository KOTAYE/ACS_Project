#include "quant.h"
#include <cmath>
#include <algorithm>

QuantMatrix make_quant_matrix(const QuantMatrix8& base, int quality, int block_size) {
    int q = std::clamp(quality, 1, 100);
    const float scale = (q < 50) ? 50.0f / (float)q : 2.0f - (float)q / 50.0f;

    QuantMatrix qm(block_size * block_size);
    if (block_size == 8) {
        for (int i = 0; i < 64; ++i) qm[i] = std::clamp(std::roundf(base[i] * scale), 1.0f, 255.0f);
    } else {
        
        float step = 7.0f / (block_size - 1);
        for (int r = 0; r < block_size; ++r) {
            for (int c = 0; c < block_size; ++c) {
                float y = r * step;
                float x = c * step;
                int y0 = (int)y, y1 = std::min(y0 + 1, 7);
                int x0 = (int)x, x1 = std::min(x0 + 1, 7);
                float dy = y - y0, dx = x - x0;
                float v = (1-dy)*(1-dx)*base[y0*8+x0] + dy*(1-dx)*base[y1*8+x0] + 
                          (1-dy)*dx*base[y0*8+x1] + dy*dx*base[y1*8+x1];
                qm[r * block_size + c] = std::clamp(std::roundf(v * scale), 1.0f, 255.0f);
            }
        }
    }
    return qm;
}

void quantize_block(float* block, const QuantMatrix& qm, int size) {
    for (int i = 0; i < size; ++i) block[i] = std::roundf(block[i] / qm[i]);
}

void dequantize_block(float* block, const QuantMatrix& qm, int size) {
    for (int i = 0; i < size; ++i) block[i] *= qm[i];
}
