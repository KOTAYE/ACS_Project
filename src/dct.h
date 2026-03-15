#pragma once

void dct_init_lut();

void level_shift(float block[64], float offset);

void dct2d_naive(const float in[64], float out[64]);
void idct2d_naive(const float in[64], float out[64]);

void dct2d_separable(const float in[64], float out[64]);
void idct2d_separable(const float in[64], float out[64]);

// 16x16 variants (CPU, naive reference)
void dct2d_naive_16(const float in[256], float out[256]);
void idct2d_naive_16(const float in[256], float out[256]);
