#pragma once

void dct_init_lut();

void level_shift(float block[64], float offset);

void dct2d_naive(const float in[64], float out[64]);
void idct2d_naive(const float in[64], float out[64]);

void dct2d_separable(const float in[64], float out[64]);
void idct2d_separable(const float in[64], float out[64]);

/** Separable DCT-II / inverse for block edge N in {8,16,32}; matches CUDA scaling. */
void dct2d_separable_n(const float* in, float* out, int n);
void idct2d_separable_n(const float* in, float* out, int n);
