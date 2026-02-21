#pragma once

double compute_mse(const float* a, const float* b, int n);

double compute_psnr(double mse, double max_val = 255.0);
