#include "dct.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float cos_table[8][8];
static bool  lut_ready = false;

void dct_init_lut() {
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 8; ++n)
            cos_table[k][n] = std::cos(M_PI * (2.0 * n + 1.0) * k / 16.0);
    lut_ready = true;
}

static inline float C(int u) {
    return (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
}

void level_shift(float block[64], float offset) {
    for (int i = 0; i < 64; ++i)
        block[i] += offset;
}

void dct2d_naive(const float in[64], float out[64]) {
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            float sum = 0.0f;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    sum += in[y * 8 + x]
                           * cos_table[u][y]
                           * cos_table[v][x];
                }
            }
            out[u * 8 + v] = 0.25f * C(u) * C(v) * sum;
        }
    }
}

void idct2d_naive(const float in[64], float out[64]) {
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float sum = 0.0f;
            for (int u = 0; u < 8; ++u) {
                for (int v = 0; v < 8; ++v) {
                    sum += C(u) * C(v)
                           * in[u * 8 + v]
                           * cos_table[u][y]
                           * cos_table[v][x];
                }
            }
            out[y * 8 + x] = 0.25f * sum;
        }
    }
}

static void dct1d_row(const float* row_in, float* row_out) {
    for (int k = 0; k < 8; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < 8; ++n)
            sum += row_in[n] * cos_table[k][n];
        row_out[k] = 0.5f * C(k) * sum;
    }
}

static void dct1d_col(const float* data, int col, int stride,
                       float* out_data, int out_col, int out_stride)
{
    for (int k = 0; k < 8; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < 8; ++n)
            sum += data[n * stride + col] * cos_table[k][n];
        out_data[k * out_stride + out_col] = 0.5f * C(k) * sum;
    }
}

void dct2d_separable(const float in[64], float out[64]) {
    float tmp[64];

    for (int r = 0; r < 8; ++r)
        dct1d_row(in + r * 8, tmp + r * 8);

    for (int c = 0; c < 8; ++c)
        dct1d_col(tmp, c, 8, out, c, 8);
}

static void idct1d_row(const float* row_in, float* row_out) {
    for (int n = 0; n < 8; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k)
            sum += C(k) * row_in[k] * cos_table[k][n];
        row_out[n] = 0.5f * sum;
    }
}

static void idct1d_col(const float* data, int col, int stride,
                        float* out_data, int out_col, int out_stride)
{
    for (int n = 0; n < 8; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k)
            sum += C(k) * data[k * stride + col] * cos_table[k][n];
        out_data[n * out_stride + out_col] = 0.5f * sum;
    }
}

void idct2d_separable(const float in[64], float out[64]) {
    float tmp[64];

    for (int c = 0; c < 8; ++c)
        idct1d_col(in, c, 8, tmp, c, 8);

    for (int r = 0; r < 8; ++r)
        idct1d_row(tmp + r * 8, out + r * 8);
}
