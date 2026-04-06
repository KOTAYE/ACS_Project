#pragma once

#include <vector>

/** Same diagonal scan as CUDA encode (must match generate_zigzag in codec history). */
inline std::vector<int> codec_zigzag_scan_table(int n) {
    std::vector<int> zigzag(n * n);
    int i = 0, j = 0;
    bool up = true;
    for (int k = 0; k < n * n; ++k) {
        zigzag[k] = i * n + j;
        if (up) {
            if (j == n - 1) {
                i++;
                up = false;
            } else if (i == 0) {
                j++;
                up = false;
            } else {
                i--;
                j++;
            }
        } else {
            if (i == n - 1) {
                j++;
                up = true;
            } else if (j == 0) {
                i++;
                up = true;
            } else {
                i++;
                j--;
            }
        }
    }
    return zigzag;
}
