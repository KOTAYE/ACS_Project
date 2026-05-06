#pragma once

#include <string>

void compress_flipbook(const std::string& in_dir,
                       const std::string& out_path,
                       int quality,
                       int block_size = 8,
                       bool use_ycbcr = true,
                       bool adaptive_roi = false,
                       float roi_strength = 0.55f,
                       const std::string& heatmap_video_path = "",
                       float target_size_mb = 0.0f,
                       float scene_cut_threshold = 22.0f,
                       bool motion_predict = false,
                       int motion_search_radius = 8);
void decompress_flipbook(const std::string& in_path, const std::string& out_dir);
