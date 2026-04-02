#include "dct.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float cos_table[8][8];
static bool lut_ready = false;

void dct_init_lut() {
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 8; ++n)
            cos_table[k][n] = std::cos(M_PI * (2.0 * n + 1.0) * k / 16.0);
    lut_ready = true;
}

static inline float C8(int u) {
    return (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
}

static void dct1d_row_8(const float* row_in, float* row_out) {
    for (int k = 0; k < 8; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < 8; ++n)
            sum += row_in[n] * cos_table[k][n];
        row_out[k] = 0.5f * C8(k) * sum;
    }
}

static void dct1d_col_8(const float* data, int col, int stride, float* out, int out_stride) {
    for (int k = 0; k < 8; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < 8; ++n)
            sum += data[n * stride + col] * cos_table[k][n];
        out[k * out_stride + col] = 0.5f * C8(k) * sum;
    }
}

static void dct2d_separable_8(const float in[64], float out[64]) {
    float tmp[64];
    for (int r = 0; r < 8; ++r)
        dct1d_row_8(in + r * 8, tmp + r * 8);
    for (int c = 0; c < 8; ++c)
        dct1d_col_8(tmp, c, 8, out, 8);
}

static void idct1d_row_8(const float* row_in, float* row_out) {
    for (int n = 0; n < 8; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k)
            sum += C8(k) * row_in[k] * cos_table[k][n];
        row_out[n] = 0.5f * sum;
    }
}

static void idct1d_col_8(const float* data, int col, int stride, float* out, int out_stride) {
    for (int n = 0; n < 8; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k)
            sum += C8(k) * data[k * stride + col] * cos_table[k][n];
        out[n * out_stride + col] = 0.5f * sum;
    }
}

static void idct2d_separable_8(const float in[64], float out[64]) {
    float tmp[64];
    for (int c = 0; c < 8; ++c)
        idct1d_col_8(in, c, 8, tmp, 8);
    for (int r = 0; r < 8; ++r)
        idct1d_row_8(tmp + r * 8, out + r * 8);
}

static void dct1d_fwd_n(const float* row_in, float* row_out, int N) {
    for (int k = 0; k < N; ++k) {
        float s = 0.f;
        for (int n = 0; n < N; ++n)
            s += row_in[n] * std::cos(static_cast<float>(M_PI) * (2.f * n + 1.f) * k / (2.f * N));
        const float ck = (k == 0) ? 0.70710678118f : 1.0f;
        row_out[k] = 0.5f * ck * s;
    }
}

static void dct1d_col_fwd_n(const float* data, int col, int stride, float* out, int out_stride, int N) {
    for (int k = 0; k < N; ++k) {
        float s = 0.f;
        for (int n = 0; n < N; ++n)
            s += data[n * stride + col] * std::cos(static_cast<float>(M_PI) * (2.f * n + 1.f) * k / (2.f * N));
        const float ck = (k == 0) ? 0.70710678118f : 1.0f;
        out[k * out_stride + col] = 0.5f * ck * s;
    }
}

static void idct1d_row_n(const float* row_in, float* row_out, int N) {
    for (int out_n = 0; out_n < N; ++out_n) {
        float s = 0.f;
        for (int k = 0; k < N; ++k) {
            const float ck = (k == 0) ? 0.70710678118f : 1.0f;
            s += ck * row_in[k] * std::cos(static_cast<float>(M_PI) * (2.f * out_n + 1.f) * k / (2.f * N));
        }
        row_out[out_n] = 0.5f * s;
    }
}

static void idct1d_col_n(const float* data, int col, int stride, float* out, int out_col, int out_stride, int N) {
    for (int out_n = 0; out_n < N; ++out_n) {
        float s = 0.f;
        for (int k = 0; k < N; ++k) {
            const float ck = (k == 0) ? 0.70710678118f : 1.0f;
            s += ck * data[k * stride + col] * std::cos(static_cast<float>(M_PI) * (2.f * out_n + 1.f) * k / (2.f * N));
        }
        out[out_n * out_stride + out_col] = 0.5f * s;
    }
}

void dct2d_separable_n(const float* in, float* out, int n) {
    if (n == 8) {
        (void)lut_ready;
        dct2d_separable_8(in, out);
        return;
    }
    std::vector<float> tmp(static_cast<size_t>(n) * n);
    for (int r = 0; r < n; ++r)
        dct1d_fwd_n(in + r * n, tmp.data() + r * n, n);
    for (int c = 0; c < n; ++c)
        dct1d_col_fwd_n(tmp.data(), c, n, out, c, n, n);
}

void idct2d_separable_n(const float* in, float* out, int n) {
    if (n == 8) {
        idct2d_separable_8(in, out);
        return;
    }
    std::vector<float> tmp(static_cast<size_t>(n) * n);
    for (int c = 0; c < n; ++c)
        idct1d_col_n(in, c, n, tmp.data(), c, n, n);
    for (int r = 0; r < n; ++r)
        idct1d_row_n(tmp.data() + r * n, out + r * n, n);
}
