#pragma once

#include <cstdint>
#include <vector>


int arithmetic_order0_bound_total_bytes(const uint8_t* data, int len);



int arithmetic_order0_encode(const uint8_t* data, int len, std::vector<uint8_t>& out);

int arithmetic_order0_decode(const uint8_t* enc, int enc_len, std::vector<uint8_t>& out);

bool arithmetic_order0_roundtrip_selftest();
