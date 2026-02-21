#pragma once

#include <cstdint>

int huffman_encode_block_64(const float block[64], uint8_t* encoded, int encoded_max_size);
int huffman_decode_block_64(const uint8_t* encoded, int encoded_len, float block[64]);
