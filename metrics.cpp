#include "metrics.h"
#include <cmath>
#include <limits>

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
