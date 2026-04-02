#pragma once

void dct_init_lut();

void dct2d_separable_n(const float* in, float* out, int n);
void idct2d_separable_n(const float* in, float* out, int n);
