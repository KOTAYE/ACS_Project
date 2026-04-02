#pragma once
#include "frame.h"
#include <cstdint>
#include <vector>
#include <string>

Frame rgb_to_planes_parallel(uint8_t* raw, int w, int h, int ch, bool use_ycbcr,
                             int block_align = 8);
void planes_to_rgb_parallel(const Frame& f, std::vector<uint8_t>& out_rgb, bool use_ycbcr);


void sort_frame_paths(std::vector<std::string>& paths);


std::vector<uint8_t> read_file_bytes(const std::string& path);


bool decode_image_file_bytes(const std::string& path_hint, const uint8_t* data, size_t len,
                             int& out_w, int& out_h, int& out_ch, std::vector<uint8_t>& interleaved);
