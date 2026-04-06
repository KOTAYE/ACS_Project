#pragma once

#include <cstdint>
#include <vector>

// Еталон: нижня межа довжини (512 B як у Huffman + ceil(H/8) для ідеального AC).
int arithmetic_order0_bound_total_bytes(const uint8_t* data, int len);

// Статичний order-0 range coding: заголовок ARQ0 + int32 len + uint16×256 + payload.
// Повертає розмір out або -1 при помилці.
int arithmetic_order0_encode(const uint8_t* data, int len, std::vector<uint8_t>& out);
// Повертає розмір відновлених даних або -1.
int arithmetic_order0_decode(const uint8_t* enc, int enc_len, std::vector<uint8_t>& out);

bool arithmetic_order0_roundtrip_selftest();
