#pragma once

#include <cstdint>

int huffman_encode_bytes(const uint8_t* in_data, int in_len, uint8_t* encoded, int encoded_max_size);

int huffman_codebook_from_freq32(const uint32_t freq[256], uint32_t out_bits[256], uint8_t out_lens[256]);
int huffman_decode_bytes(const uint8_t* encoded, int encoded_len, uint8_t* out_data, int out_len);

int huffman_decode_bit_window(const uint32_t freq[256], const uint8_t* packed, int packed_len_bytes,
                              int bit_start, int num_bits, uint8_t* out, int out_cap);
