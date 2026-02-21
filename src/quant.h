#pragma once

void quantize_block_64(float block[64], float scale);
void dequantize_block_64(float block[64], float scale);
