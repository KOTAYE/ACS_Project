#pragma once

#include <cstdint>

double compute_mse(const float* a, const float* b, int n);
double compute_psnr(double mse, double max_val = 255.0);

double compute_mse_u8(const uint8_t* a, const uint8_t* b, int n);
double compute_psnr_u8(const uint8_t* a, const uint8_t* b, int n);
double compute_ssim(const uint8_t* a, const uint8_t* b, int width, int height, int channels);
