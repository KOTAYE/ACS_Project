#pragma once

#include <string>

void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality, bool use_ycbcr = true);
void decompress_flipbook(const std::string& in_path, const std::string& out_dir);
