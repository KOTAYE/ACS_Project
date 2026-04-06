#pragma once

#include <cstdint>

int huffman_encode_block_64(const float block[64], uint8_t* encoded, int encoded_max_size);
int huffman_decode_block_64(const uint8_t* encoded, int encoded_len, float block[64]);

int huffman_encode_bytes(const uint8_t* in_data, int in_len, uint8_t* encoded, int encoded_max_size);
void huffman_prepare_codebook(const uint32_t histogram[256], uint32_t code_bits[256], uint8_t code_lens[256], uint16_t out_freq[256]);

// Same code lengths/bits as huffman_decode_bit_window uses for `freq` (uint16_t channel freqs from bitstream).
int huffman_codebook_from_freq16(const uint16_t freq[256], uint32_t out_bits[256], uint8_t out_lens[256]);
int huffman_decode_bytes(const uint8_t* encoded, int encoded_len, uint8_t* out_data, int out_len);

// Decode Huffman symbols from a contiguous bit substring of `packed` (MSB-first within each byte).
// Consumes exactly `num_bits` bits starting at `bit_start`. Returns number of output bytes, or -1 on error.
int huffman_decode_bit_window(const uint16_t freq[256], const uint8_t* packed, int packed_len_bytes,
                              int bit_start, int num_bits, uint8_t* out, int out_cap);
