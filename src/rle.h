#pragma once

#include <cstdint>
#include <vector>

std::vector<int16_t> rle_encode_zeros(const std::vector<int16_t>& in);
void rle_decode_zeros(const std::vector<int16_t>& in, std::vector<int16_t>& out);
