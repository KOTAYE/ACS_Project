#pragma once
#include "frame.h"
#include <cstdint>
#include <vector>
#include <string>

Frame load_image(const char* path);

bool save_image(const char* path, const Frame& frame);

// Вхід: інтерлейв RGB uint8 (як з stbi_load). Вихід: Frame з планарними uint8 каналами (компактно для GPU upload).
Frame rgb_to_planes_parallel(uint8_t* raw, int w, int h, int ch, bool use_ycbcr,
                             int block_align = 8);
void planes_to_rgb_parallel(const Frame& f, std::vector<uint8_t>& out_rgb, bool use_ycbcr);

/** Sort paths like compute_metrics.py (numeric runs in filename), not plain lexicographic order. */
void sort_frame_paths(std::vector<std::string>& paths);

/** Читає весь файл у пам'ять (для I/O-потоку конвеєра). Порожній вектор при помилці. */
std::vector<uint8_t> read_file_bytes(const std::string& path);

/** Декод у інтерлейв uint8 (stb: PNG/JPEG/…; за розширенням .exr — TinyEXR, float→uint8 з простим tone-map). */
bool decode_image_file_bytes(const std::string& path_hint, const uint8_t* data, size_t len,
                             int& out_w, int& out_h, int& out_ch, std::vector<uint8_t>& interleaved);
