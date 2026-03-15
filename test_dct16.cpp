#include <iostream>
#include <cmath>

#include "dct.h"

static double compute_mse_16(const float* a, const float* b) {
    double sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum / 256.0;
}

int main() {
    dct_init_lut();

    float in[256];
    float dct[256];
    float out[256];

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            in[y * 16 + x] = static_cast<float>(x + y);
        }
    }

    dct2d_naive_16(in, dct);
    idct2d_naive_16(dct, out);

    double mse = compute_mse_16(in, out);
    double max_err = 0.0;
    for (int i = 0; i < 256; ++i) {
        double d = std::fabs(static_cast<double>(in[i]) - static_cast<double>(out[i]));
        if (d > max_err) max_err = d;
    }

    std::cout << "16x16 DCT/IDCT test\n";
    std::cout << "  MSE     = " << mse << "\n";
    std::cout << "  max err = " << max_err << "\n";

    return 0;
}

