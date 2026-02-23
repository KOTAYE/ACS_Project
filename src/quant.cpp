#include "quant.h"
#include <cmath>
#include <algorithm>

QuantMatrix make_quant_matrix(const QuantMatrix& base, int quality) {
    quality = std::clamp(quality, 1, 100);
    const float scale = (quality < 50)
        ? 50.0f / static_cast<float>(quality)
        :  2.0f - static_cast<float>(quality) / 50.0f;

    QuantMatrix qm;
    for (int i = 0; i < 64; ++i) {
        const float v = std::roundf(base[i] * scale);
        qm[i] = std::clamp(v, 1.0f, 255.0f);
    }
    return qm;
}

void quantize_block_64(float block[64], const QuantMatrix& qm) {
    for (int i = 0; i < 64; ++i)
        block[i] = std::roundf(block[i] / qm[i]);
}

void dequantize_block_64(float block[64], const QuantMatrix& qm) {
    for (int i = 0; i < 64; ++i)
        block[i] *= qm[i];
}
