#include "metrics.h"
#include <cmath>
#include <limits>
#include <algorithm>

double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum / n;
}

double compute_psnr(double mse, double max_val) {
    if (mse < 1e-10)
        return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10((max_val * max_val) / mse);
}

double compute_mse_u8(const uint8_t* a, const uint8_t* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum / n;
}

double compute_psnr_u8(const uint8_t* a, const uint8_t* b, int n) {
    return compute_psnr(compute_mse_u8(a, b, n), 255.0);
}

double compute_ssim(const uint8_t* a, const uint8_t* b, int width, int height, int channels) {
    constexpr int WINDOW = 8;
    constexpr double K1 = 0.01, K2 = 0.03, L = 255.0;
    const double C1 = (K1 * L) * (K1 * L);
    const double C2 = (K2 * L) * (K2 * L);

    int total_pixels = width * height * channels;
    if (total_pixels == 0) return 1.0;

    int num_ch = channels;
    double ssim_sum = 0.0;
    int window_count = 0;

    for (int c = 0; c < num_ch; ++c) {
        for (int y = 0; y <= height - WINDOW; y += WINDOW) {
            for (int x = 0; x <= width - WINDOW; x += WINDOW) {
                double sum_a = 0, sum_b = 0;
                double sum_a2 = 0, sum_b2 = 0, sum_ab = 0;
                int N = WINDOW * WINDOW;

                for (int wy = 0; wy < WINDOW; ++wy) {
                    for (int wx = 0; wx < WINDOW; ++wx) {
                        int idx = ((y + wy) * width + (x + wx)) * num_ch + c;
                        double va = a[idx], vb = b[idx];
                        sum_a += va;  sum_b += vb;
                        sum_a2 += va * va;  sum_b2 += vb * vb;
                        sum_ab += va * vb;
                    }
                }

                double mu_a = sum_a / N, mu_b = sum_b / N;
                double sigma_a2 = sum_a2 / N - mu_a * mu_a;
                double sigma_b2 = sum_b2 / N - mu_b * mu_b;
                double sigma_ab = sum_ab / N - mu_a * mu_b;

                double num = (2.0 * mu_a * mu_b + C1) * (2.0 * sigma_ab + C2);
                double den = (mu_a * mu_a + mu_b * mu_b + C1) * (sigma_a2 + sigma_b2 + C2);

                ssim_sum += num / den;
                window_count++;
            }
        }
    }

    return window_count > 0 ? ssim_sum / window_count : 1.0;
}
