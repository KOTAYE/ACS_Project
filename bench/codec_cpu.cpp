#include "codec.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <future>
#include <chrono>
#include <array>
#include <cstring>
#include <cmath>
#include <cstdlib>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "frame.h"
#include "tiling.h"
#include "dct.h"
#include "quant.h"
#include "huffman.h"
#include "zigzag.h"

#include "stb_image_write.h"

namespace fs = std::filesystem;

#pragma pack(push, 1)
struct BinHeader {
    char magic[4] = {'F', 'L', 'I', '3'};
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t quality;
    int32_t block_size;
    int32_t frame_count;
    int32_t use_ycbcr = 1;
};
#pragma pack(pop)

namespace {

inline int codec_pad(int n, int bs) { return ((n + bs - 1) / bs) * bs; }

inline void level_shift_n(float* b, int n, float d) {
    for (int i = 0; i < n; ++i) b[i] += d;
}

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

inline float block_quant_scale(const float* residual, int n, bool adaptive_roi, float roi_strength) {
    if (!adaptive_roi) {
        return 1.0f;
    }
    float sum_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_abs += std::fabs(residual[i]);
    }
    const float avg_abs = sum_abs / static_cast<float>(n);
    const float importance = 1.0f - std::exp(-avg_abs / 18.0f);
    const float scale = 1.0f + roi_strength * (0.5f - importance);
    return clampf(scale, 0.60f, 1.60f);
}

inline void quantize_block_scaled(float* block, const QuantMatrix& qm, int size, float scale) {
    for (int i = 0; i < size; ++i) {
        block[i] = std::roundf(block[i] / (qm[i] * scale));
    }
}

inline void dequantize_block_scaled(float* block, const QuantMatrix& qm, int size, float scale) {
    for (int i = 0; i < size; ++i) {
        block[i] *= (qm[i] * scale);
    }
}

inline double mean_abs_diff_u8(const uint8_t* a, const uint8_t* b, size_t n) {
    if (!a || !b || n == 0) return 0.0;
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<uint32_t>(std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    }
    return static_cast<double>(sum) / static_cast<double>(n);
}

inline int clamp_quality(int q) {
    return std::max(5, std::min(100, q));
}

} 

std::vector<int16_t> rle_encode_zeros(const std::vector<int16_t>& in) {
    std::vector<int16_t> out;
    out.reserve(in.size() / 2);
    int i = 0;
    const int n = static_cast<int>(in.size());
    while (i < n) {
        if (in[i] == 0) {
            int run = 1;
            while (i + run < n && in[i + run] == 0 && run < 32767) run++;
            out.push_back(0);
            out.push_back(static_cast<int16_t>(run));
            i += run;
        } else {
            out.push_back(in[i]);
            i++;
        }
    }
    return out;
}

void rle_decode_zeros(const std::vector<int16_t>& in, std::vector<int16_t>& out) {
    int i = 0, o = 0;
    const int n = static_cast<int>(in.size());
    const int out_n = static_cast<int>(out.size());
    while (i < n && o < out_n) {
        if (in[i] == 0) {
            if (i + 1 >= n) break;
            int run = in[i + 1];
            for (int k = 0; k < run && o < out_n; ++k) out[o++] = 0;
            i += 2;
        } else {
            out[o++] = in[i];
            i++;
        }
    }
}


static void rle_decode_macroblock(const int16_t* rle, int rle_elem_count, int16_t* coef_out, int bs2) {
    int in_idx = 0, out_idx = 0;
    while (in_idx < rle_elem_count && out_idx < bs2) {
        if (rle[in_idx] == 0) {
            if (in_idx + 1 >= rle_elem_count) break;
            const int run = rle[in_idx + 1];
            for (int k = 0; k < run && out_idx < bs2; ++k) coef_out[out_idx++] = 0;
            in_idx += 2;
        } else {
            coef_out[out_idx++] = rle[in_idx];
            in_idx++;
        }
    }
    while (out_idx < bs2) coef_out[out_idx++] = 0;
}

#include "image_io.h"

void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality, int block_size,
                       bool use_ycbcr, bool adaptive_roi, float roi_strength, const std::string& heatmap_video_path,
                       float target_size_mb, float scene_cut_threshold, bool /*motion_predict*/,
                       int /*motion_search_radius*/) {
    if (block_size != 8 && block_size != 16 && block_size != 32) {
        std::cerr << "Error: block_size must be 8, 16, or 32.\n";
        return;
    }

    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Input must be a valid directory containing frames.\n";
        return;
    }

    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(in_dir)) {
        if (entry.is_regular_file()) frames.push_back(entry.path().string());
    }
    sort_frame_paths(frames);

    if (frames.empty()) {
        std::cerr << "No frames found in directory " << in_dir << "\n";
        return;
    }

    int img_w = 0, img_h = 0, img_ch = 0;
    std::vector<uint8_t> first_interleaved;
    {
        std::vector<uint8_t> fb = read_file_bytes(frames[0]);
        if (!decode_image_file_bytes(frames[0], fb.data(), fb.size(), img_w, img_h, img_ch, first_interleaved)) {
            std::cerr << "Failed to decode the first frame: " << frames[0] << "\n";
            return;
        }
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << out_path << " for writing\n";
        return;
    }

    BinHeader header;
    header.magic[3] = '5';
    header.width = img_w;
    header.height = img_h;
    header.channels = img_ch;
    header.quality = quality;
    header.block_size = block_size;
    header.frame_count = static_cast<int32_t>(frames.size());
    header.use_ycbcr = use_ycbcr ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const bool rc_enabled = target_size_mb > 0.0f;
    const size_t target_size_bytes = rc_enabled
        ? static_cast<size_t>(target_size_mb * 1024.0f * 1024.0f)
        : 0u;
    size_t encoded_bytes_so_far = sizeof(header);
    int rc_quality = clamp_quality(quality);

    const std::vector<int> zigzag = codec_zigzag_scan_table(block_size);
    const int bs = block_size;
    const int bpp = bs * bs;

    Frame prev_recon = frame_create(img_w, img_h, img_ch, use_ycbcr, block_size);
    Frame curr_recon = frame_create(img_w, img_h, img_ch, use_ycbcr, block_size);

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;
    std::vector<float> frame_luma_scales;
    std::string heatmap_data_dir;
    std::ofstream heatmap_manifest;
    bool emit_heatmap = !heatmap_video_path.empty();
    if (emit_heatmap) {
        heatmap_data_dir = out_path + ".heatmap_data";
        fs::create_directories(heatmap_data_dir);
        heatmap_manifest.open(heatmap_data_dir + "/manifest.tsv", std::ios::trunc);
        if (!heatmap_manifest) {
            std::cerr << "Warning: failed to create heatmap manifest; heatmap video disabled.\n";
            emit_heatmap = false;
        } else {
            heatmap_manifest << "frame_idx\tblocks_x\tblocks_y\tmap_file\tsource_frame\n";
        }
    }

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";
    auto t_compress_start = std::chrono::high_resolution_clock::now();

#ifdef USE_OMP
    std::future<std::vector<uint8_t>> prefetch_interleaved;
#endif

    std::vector<uint8_t> prev_input_luma;

    for (size_t f_idx = 0; f_idx < frames.size(); ++f_idx) {
        std::vector<uint8_t> interleaved;
        if (f_idx == 0) {
            interleaved = std::move(first_interleaved);
        } else {
#ifdef USE_OMP
            interleaved = prefetch_interleaved.get();
#else
            std::vector<uint8_t> fb = read_file_bytes(frames[f_idx]);
            int w = 0, h = 0, c = 0;
            if (!decode_image_file_bytes(frames[f_idx], fb.data(), fb.size(), w, h, c, interleaved) ||
                w != img_w || h != img_h || c != img_ch) {
                std::cerr << "\nError: Invalid frame or dimensions mismatched at " << frames[f_idx] << "\n";
                break;
            }
#endif
        }

        if (interleaved.empty()) {
            std::cerr << "\nError: empty decode at " << frames[f_idx] << "\n";
            break;
        }

#ifdef USE_OMP
        if (f_idx + 1 < frames.size()) {
            const std::string next_path = frames[f_idx + 1];
            const int ew = img_w, eh = img_h, ec = img_ch;
            prefetch_interleaved = std::async(std::launch::async, [next_path, ew, eh, ec]() {
                std::vector<uint8_t> fb = read_file_bytes(next_path);
                std::vector<uint8_t> out;
                int w = 0, h = 0, c = 0;
                if (!decode_image_file_bytes(next_path, fb.data(), fb.size(), w, h, c, out)) return std::vector<uint8_t>{};
                if (w != ew || h != eh || c != ec) return std::vector<uint8_t>{};
                return out;
            });
        }
#endif

        Frame current = rgb_to_planes_parallel(interleaved.data(), img_w, img_h, img_ch, use_ycbcr, block_size);

        const int luma_pixels = current.padded_width[0] * current.padded_height[0];
        const uint8_t* curr_luma = current.channel_ptr(0);
        const bool scene_cut = (f_idx > 0 && scene_cut_threshold > 0.0f && !prev_input_luma.empty() &&
            mean_abs_diff_u8(curr_luma, prev_input_luma.data(), static_cast<size_t>(luma_pixels)) >= scene_cut_threshold);
        bool is_keyframe = (f_idx == 0) || scene_cut;
        const int frame_quality = rc_enabled ? rc_quality : quality;
        const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, frame_quality, block_size);
        const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, frame_quality, block_size);

        uint8_t keyframe_marker = is_keyframe ? 1u : 0u;
        uint8_t frame_quality_marker = static_cast<uint8_t>(frame_quality);
        out.write(reinterpret_cast<const char*>(&keyframe_marker), 1);
        out.write(reinterpret_cast<const char*>(&frame_quality_marker), 1);
        size_t frame_bytes = 2;

        for (int ch = 0; ch < current.channels; ++ch) {
            const QuantMatrix& qm = (ch == 0) ? luma_qm : chroma_qm;
            const uint8_t* src_channel = current.channel_ptr(ch);
            const uint8_t* prev_channel = prev_recon.channel_ptr(ch);
            uint8_t* recon_channel = curr_recon.channel_ptr(ch);

            const int padded_w = current.padded_width[ch];
            const int padded_h = current.padded_height[ch];
            const int blocks_x = padded_w / bs;
            const int blocks_y = padded_h / bs;
            const int total_blocks = blocks_x * blocks_y;
            const int channel_samples = total_blocks * bpp;
            if (ch == 0 && emit_heatmap) {
                frame_luma_scales.assign(static_cast<size_t>(total_blocks), 1.0f);
            }

            if (static_cast<int>(channel_buffer.size()) < channel_samples)
                channel_buffer.resize(channel_samples);

#ifdef USE_OMP
#pragma omp parallel
            {
                std::vector<float> block_in(static_cast<size_t>(bpp));
                std::vector<float> prev_block(static_cast<size_t>(bpp));
                std::vector<float> dct_out(static_cast<size_t>(bpp));
                std::vector<float> idct_out(static_cast<size_t>(bpp));
#pragma omp for
                for (int by = 0; by < blocks_y; ++by) {
                    for (int bx = 0; bx < blocks_x; ++bx) {
                        int sample_idx = (by * blocks_x + bx) * bpp;
                        extract_block_n(src_channel, padded_w, bx, by, bs, block_in.data());

                        if (!is_keyframe) {
                            extract_block_n(prev_channel, padded_w, bx, by, bs, prev_block.data());
                            for (int i = 0; i < bpp; ++i) block_in[i] -= prev_block[i];
                        } else {
                            level_shift_n(block_in.data(), bpp, -128.0f);
                        }

                        dct2d_separable_n(block_in.data(), dct_out.data(), bs);
                        const int block_id = by * blocks_x + bx;
                        const float q_scale = block_quant_scale(block_in.data(), bpp, adaptive_roi, roi_strength);
                        if (ch == 0 && emit_heatmap) {
                            frame_luma_scales[static_cast<size_t>(block_id)] = q_scale;
                        }
                        quantize_block_scaled(dct_out.data(), qm, bpp, q_scale);

                        for (int i = 0; i < bpp; ++i)
                            channel_buffer[sample_idx + zigzag[i]] = static_cast<int16_t>(dct_out[i]);

                        dequantize_block_scaled(dct_out.data(), qm, bpp, q_scale);
                        idct2d_separable_n(dct_out.data(), idct_out.data(), bs);

                        if (!is_keyframe) {
                            for (int i = 0; i < bpp; ++i) idct_out[i] += prev_block[i];
                        } else {
                            level_shift_n(idct_out.data(), bpp, 128.0f);
                        }

                        insert_block_n(recon_channel, padded_w, bx, by, bs, idct_out.data());
                    }
                }
            }
#else
            std::vector<float> block_in(static_cast<size_t>(bpp));
            std::vector<float> prev_block(static_cast<size_t>(bpp));
            std::vector<float> dct_out(static_cast<size_t>(bpp));
            std::vector<float> idct_out(static_cast<size_t>(bpp));
            for (int by = 0; by < blocks_y; ++by) {
                for (int bx = 0; bx < blocks_x; ++bx) {
                    int sample_idx = (by * blocks_x + bx) * bpp;
                    extract_block_n(src_channel, padded_w, bx, by, bs, block_in.data());

                    if (!is_keyframe) {
                        extract_block_n(prev_channel, padded_w, bx, by, bs, prev_block.data());
                        for (int i = 0; i < bpp; ++i) block_in[i] -= prev_block[i];
                    } else {
                        level_shift_n(block_in.data(), bpp, -128.0f);
                    }

                    dct2d_separable_n(block_in.data(), dct_out.data(), bs);
                    const int block_id = by * blocks_x + bx;
                    const float q_scale = block_quant_scale(block_in.data(), bpp, adaptive_roi, roi_strength);
                    if (ch == 0 && emit_heatmap) {
                        frame_luma_scales[static_cast<size_t>(block_id)] = q_scale;
                    }
                    quantize_block_scaled(dct_out.data(), qm, bpp, q_scale);

                    for (int i = 0; i < bpp; ++i)
                        channel_buffer[sample_idx + zigzag[i]] = static_cast<int16_t>(dct_out[i]);

                    dequantize_block_scaled(dct_out.data(), qm, bpp, q_scale);
                    idct2d_separable_n(dct_out.data(), idct_out.data(), bs);

                    if (!is_keyframe) {
                        for (int i = 0; i < bpp; ++i) idct_out[i] += prev_block[i];
                    } else {
                        level_shift_n(idct_out.data(), bpp, 128.0f);
                    }

                    insert_block_n(recon_channel, padded_w, bx, by, bs, idct_out.data());
                }
            }
#endif

            std::vector<int16_t> current_ch_buffer(channel_buffer.begin(),
                                                   channel_buffer.begin() + channel_samples);
            std::vector<int16_t> rle_buffer = rle_encode_zeros(current_ch_buffer);
            const uint8_t* raw_bytes = reinterpret_cast<const uint8_t*>(rle_buffer.data());
            const int raw_len = static_cast<int>(rle_buffer.size() * sizeof(int16_t));

            if (encoded.size() < static_cast<size_t>(raw_len) + 16384u)
                encoded.resize(static_cast<size_t>(raw_len) + 16384u);

            int enc_total = 0;
            if (raw_len > 0) {
                enc_total = huffman_encode_bytes(raw_bytes, raw_len, encoded.data(),
                                                   static_cast<int>(encoded.size()));
                if (enc_total < 0) {
                    std::cerr << "\nCompression failed (buffer overflow)\n";
                    enc_total = 0;
                }
            }

            uint32_t rle_bytes_len = static_cast<uint32_t>(raw_len);
            uint32_t payload =
                (enc_total >= 512) ? static_cast<uint32_t>(enc_total - 512) : 0u;
            uint32_t nb = static_cast<uint32_t>(total_blocks);

            out.write(reinterpret_cast<const char*>(&rle_bytes_len), sizeof(rle_bytes_len));
            out.write(reinterpret_cast<const char*>(&payload), sizeof(payload));
            out.write(reinterpret_cast<const char*>(&nb), sizeof(nb));

            std::array<uint32_t, 256> freq{};
            if (enc_total >= sizeof(uint32_t) * 256)
                std::memcpy(freq.data(), encoded.data(), sizeof(uint32_t) * 256);
            out.write(reinterpret_cast<const char*>(freq.data()), sizeof(freq));

            std::vector<uint32_t> bl(static_cast<size_t>(nb), 0u);
            out.write(reinterpret_cast<const char*>(bl.data()), nb * sizeof(uint32_t));
            if (payload > 0)
                out.write(reinterpret_cast<const char*>(encoded.data() + sizeof(uint32_t) * 256), payload);
            frame_bytes += 4 + 4 + 4 + (256 * sizeof(uint32_t)) + (static_cast<size_t>(nb) * sizeof(uint32_t)) + payload;
        }

        prev_input_luma.assign(curr_luma, curr_luma + static_cast<size_t>(luma_pixels));
        if (rc_enabled) {
            const size_t remain_before = (target_size_bytes > encoded_bytes_so_far)
                ? (target_size_bytes - encoded_bytes_so_far) : 1u;
            const double budget_this = static_cast<double>(remain_before) /
                                       static_cast<double>(std::max<size_t>(1, frames.size() - f_idx));
            encoded_bytes_so_far += frame_bytes;
            if (f_idx + 1 < frames.size()) {
                const double rel_err = (budget_this > 1.0) ? (static_cast<double>(frame_bytes) / budget_this - 1.0) : 0.0;
                int step = static_cast<int>(std::round(rel_err * 10.0));
                if (step > 0) {
                    rc_quality = clamp_quality(rc_quality - std::min(8, step));
                } else if (step < 0) {
                    rc_quality = clamp_quality(rc_quality + std::min(4, -step));
                }
            }
        }

        std::swap(curr_recon.data, prev_recon.data);
        frame_destroy(current);
        if (emit_heatmap && !frame_luma_scales.empty()) {
            const int blocks_x = prev_recon.padded_width[0] / bs;
            const int blocks_y = prev_recon.padded_height[0] / bs;
            const fs::path map_name = "frame_" + std::to_string(f_idx) + ".bin";
            const fs::path map_path = fs::path(heatmap_data_dir) / map_name;
            std::ofstream map_out(map_path, std::ios::binary);
            if (map_out) {
                map_out.write(reinterpret_cast<const char*>(frame_luma_scales.data()),
                              static_cast<std::streamsize>(frame_luma_scales.size() * sizeof(float)));
                heatmap_manifest << f_idx << "\t" << blocks_x << "\t" << blocks_y << "\t"
                                << map_name.string() << "\t" << frames[f_idx] << "\n";
            }
        }
        std::cout << "\r  Progress: " << f_idx + 1 << "/" << frames.size() << std::flush;
    }
    auto t_compress_end = std::chrono::high_resolution_clock::now();
    double compress_ms = std::chrono::duration<double, std::milli>(t_compress_end - t_compress_start).count();
    double compress_fps = 1000.0 * frames.size() / compress_ms;

    auto file_size = fs::file_size(out_path);
    size_t raw_size = static_cast<size_t>(img_w) * img_h * img_ch * frames.size();
    double ratio = (double)raw_size / (double)file_size;

    std::cout << "\n  Finished compressing flipbook.\n";
    std::cout << "  [BENCHMARK] compress_ms=" << compress_ms
              << " frames=" << frames.size()
              << " compress_fps=" << compress_fps
              << " raw_bytes=" << raw_size
              << " compressed_bytes=" << file_size
              << " compression_ratio=" << ratio << "\n";
    if (rc_enabled) {
        std::cout << "  [RATE_CONTROL] target_mb=" << target_size_mb
                  << " actual_mb=" << (static_cast<double>(file_size) / (1024.0 * 1024.0))
                  << " final_quality=" << rc_quality << "\n";
    }
    if (emit_heatmap) {
        heatmap_manifest.close();
        std::string cmd = "python scripts/make_heatmap_video.py --manifest \"" + heatmap_data_dir +
                          "/manifest.tsv\" --output \"" + heatmap_video_path + "\"";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::cerr << "Heatmap video generation failed (exit code " << rc
                      << "). Run the script manually.\n";
        } else {
            std::cout << "  Heatmap video written to: " << heatmap_video_path << "\n";
        }
    }

    frame_destroy(prev_recon);
    frame_destroy(curr_recon);
}

void decompress_flipbook(const std::string& in_path, const std::string& out_dir) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << in_path << " for reading\n";
        return;
    }

    BinHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    const bool valid_magic = (in.gcount() == sizeof(header) && header.magic[0] == 'F' &&
        header.magic[1] == 'L' && header.magic[2] == 'I' &&
        (header.magic[3] == '3' || header.magic[3] == '4' || header.magic[3] == '5'));
    if (!valid_magic) {
        std::cerr << "Invalid or corrupted bin file: " << in_path << "\n";
        return;
    }
    const bool has_frame_keyflags = (header.magic[3] == '4' || header.magic[3] == '5');
    const bool has_frame_quality = (header.magic[3] == '5');

    if (header.block_size != 8 && header.block_size != 16 && header.block_size != 32) {
        std::cerr << "Unsupported block_size " << header.block_size << " in file.\n";
        return;
    }

    fs::create_directories(out_dir);

    bool use_ycbcr = (header.use_ycbcr != 0);
    const int bs = header.block_size;
    const int bpp = bs * bs;
    const std::vector<int> zigzag = codec_zigzag_scan_table(bs);

    Frame prev_recon = frame_create(header.width, header.height, header.channels, use_ycbcr, bs);
    Frame curr_recon = frame_create(header.width, header.height, header.channels, use_ycbcr, bs);

    stbi_write_png_compression_level = 1;

    constexpr int NUM_WRITE_BUFS = 4;
    const size_t rgb_size = static_cast<size_t>(header.width) * header.height * header.channels;

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;
    std::vector<std::vector<uint8_t>> rgb_ring(NUM_WRITE_BUFS);
    for (auto& buf : rgb_ring) buf.resize(rgb_size);
    std::future<void> write_futures[NUM_WRITE_BUFS];

    std::cout << "Decompressing " << header.frame_count << " frames to " << out_dir << "...\n";

    double total_decode_ms = 0.0;

    for (int f_idx = 0; f_idx < header.frame_count; ++f_idx) {
        bool is_keyframe = (f_idx == 0);
        int frame_quality = header.quality;
        if (has_frame_keyflags) {
            uint8_t keyflag = 0;
            in.read(reinterpret_cast<char*>(&keyflag), 1);
            if (!in) {
                std::cerr << "\nCorrupt frame keyflag at frame " << f_idx << "\n";
                return;
            }
            is_keyframe = (keyflag != 0);
        }
        if (has_frame_quality) {
            uint8_t q = 0;
            in.read(reinterpret_cast<char*>(&q), 1);
            if (!in) {
                std::cerr << "\nCorrupt frame quality at frame " << f_idx << "\n";
                return;
            }
            frame_quality = clamp_quality(static_cast<int>(q));
        }
        const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, frame_quality, bs);
        const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, frame_quality, bs);
        auto t_decode_start = std::chrono::high_resolution_clock::now();

        for (int ch = 0; ch < curr_recon.channels; ++ch) {
            const QuantMatrix& qm = (ch == 0) ? luma_qm : chroma_qm;
            uint8_t* recon_channel = curr_recon.channel_ptr(ch);
            const uint8_t* prev_channel = prev_recon.channel_ptr(ch);

            const int padded_w = curr_recon.padded_width[ch];
            const int padded_h = curr_recon.padded_height[ch];
            const int blocks_x = padded_w / bs;
            const int blocks_y = padded_h / bs;
            const int total_blocks = blocks_x * blocks_y;
            const int channel_samples = total_blocks * bpp;

            if (static_cast<int>(channel_buffer.size()) < channel_samples)
                channel_buffer.resize(channel_samples);

            uint32_t rle_bytes_len = 0, len32 = 0, num_blocks_file = 0;
            in.read(reinterpret_cast<char*>(&rle_bytes_len), sizeof(rle_bytes_len));
            in.read(reinterpret_cast<char*>(&len32), sizeof(len32));
            in.read(reinterpret_cast<char*>(&num_blocks_file), sizeof(num_blocks_file));

            std::vector<uint32_t> h_freq(256);
            in.read(reinterpret_cast<char*>(h_freq.data()), h_freq.size() * sizeof(uint32_t));

            if (!in || num_blocks_file > 10000000u) {
                std::cerr << "\nCorrupt channel header (frame " << f_idx << " ch " << ch << ")\n";
                return;
            }

            std::vector<uint32_t> block_bit_lengths(static_cast<size_t>(num_blocks_file));
            in.read(reinterpret_cast<char*>(block_bit_lengths.data()),
                    static_cast<std::streamsize>(num_blocks_file * sizeof(uint32_t)));

            encoded.resize(len32);
            if (len32 > 0)
                in.read(reinterpret_cast<char*>(encoded.data()), len32);

            bool per_block_entropy = false;
            for (uint32_t bl : block_bit_lengths) {
                if (bl != 0) {
                    per_block_entropy = true;
                    break;
                }
            }

            std::fill(channel_buffer.begin(), channel_buffer.end(), 0);

            if (len32 > 0 && per_block_entropy) {
                if (static_cast<int>(num_blocks_file) != total_blocks) {
                    std::cerr << "\nnum_blocks mismatch (ch=" << ch << " frame=" << f_idx << ")\n";
                } else {
                    std::vector<uint8_t> byte_scratch(static_cast<size_t>(bpp * 2 + 64));
                    int bit_cursor = 0;
                    for (int bid = 0; bid < total_blocks; ++bid) {
                        const int nbits = static_cast<int>(block_bit_lengths[static_cast<size_t>(bid)]);
                        int16_t* dst = channel_buffer.data() + static_cast<size_t>(bid) * bpp;
                        if (nbits == 0) {
                            std::fill(dst, dst + bpp, static_cast<int16_t>(0));
                            continue;
                        }
                        const int nbytes = huffman_decode_bit_window(
                            h_freq.data(), encoded.data(), static_cast<int>(encoded.size()), bit_cursor, nbits,
                            byte_scratch.data(), static_cast<int>(byte_scratch.size()));
                        if (nbytes < 0 || (nbytes & 1) != 0) {
                            std::cerr << "\nPer-block Huffman decode failed (ch=" << ch << " frame=" << f_idx
                                      << " block=" << bid << ")\n";
                            std::fill(dst, dst + bpp, static_cast<int16_t>(0));
                        } else {
                            const int rle_elems = nbytes / static_cast<int>(sizeof(int16_t));
                            rle_decode_macroblock(reinterpret_cast<const int16_t*>(byte_scratch.data()), rle_elems,
                                                  dst, bpp);
                        }
                        bit_cursor += nbits;
                    }
                }
            } else if (len32 > 0 && rle_bytes_len > 0) {
                std::vector<uint8_t> enc_with_hdr(sizeof(uint32_t) * 256 + len32);
                std::memcpy(enc_with_hdr.data(), h_freq.data(), sizeof(uint32_t) * 256);
                std::memcpy(enc_with_hdr.data() + sizeof(uint32_t) * 256, encoded.data(), len32);

                std::vector<int16_t> rle_buf(rle_bytes_len / sizeof(int16_t));
                if (huffman_decode_bytes(enc_with_hdr.data(), static_cast<int>(enc_with_hdr.size()),
                                         reinterpret_cast<uint8_t*>(rle_buf.data()),
                                         static_cast<int>(rle_bytes_len)) != 0) {
                    std::cerr << "\nHuffman decode failed (ch=" << ch << " frame=" << f_idx << ")\n";
                } else {
                    rle_decode_zeros(rle_buf, channel_buffer);
                }
            }

#ifdef USE_OMP
#pragma omp parallel
            {
                std::vector<float> dct_out(static_cast<size_t>(bpp));
                std::vector<float> idct_out(static_cast<size_t>(bpp));
                std::vector<float> prev_block(static_cast<size_t>(bpp));
#pragma omp for
                for (int by = 0; by < blocks_y; ++by) {
                    for (int bx = 0; bx < blocks_x; ++bx) {
                        int sample_idx = (by * blocks_x + bx) * bpp;
                        for (int j = 0; j < bpp; ++j)
                            dct_out[j] =
                                static_cast<float>(channel_buffer[sample_idx + zigzag[j]]) * qm[j];

                        idct2d_separable_n(dct_out.data(), idct_out.data(), bs);

                        if (!is_keyframe) {
                            extract_block_n(prev_channel, padded_w, bx, by, bs, prev_block.data());
                            for (int i = 0; i < bpp; ++i) idct_out[i] += prev_block[i];
                        } else {
                            level_shift_n(idct_out.data(), bpp, 128.0f);
                        }

                        insert_block_n(recon_channel, padded_w, bx, by, bs, idct_out.data());
                    }
                }
            }
#else
            std::vector<float> dct_out(static_cast<size_t>(bpp));
            std::vector<float> idct_out(static_cast<size_t>(bpp));
            std::vector<float> prev_block(static_cast<size_t>(bpp));
            for (int by = 0; by < blocks_y; ++by) {
                for (int bx = 0; bx < blocks_x; ++bx) {
                    int sample_idx = (by * blocks_x + bx) * bpp;
                    for (int j = 0; j < bpp; ++j)
                        dct_out[j] =
                            static_cast<float>(channel_buffer[sample_idx + zigzag[j]]) * qm[j];

                    idct2d_separable_n(dct_out.data(), idct_out.data(), bs);

                    if (!is_keyframe) {
                        extract_block_n(prev_channel, padded_w, bx, by, bs, prev_block.data());
                        for (int i = 0; i < bpp; ++i) idct_out[i] += prev_block[i];
                    } else {
                        level_shift_n(idct_out.data(), bpp, 128.0f);
                    }

                    insert_block_n(recon_channel, padded_w, bx, by, bs, idct_out.data());
                }
            }
#endif
        }

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        total_decode_ms += std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        int slot = f_idx % NUM_WRITE_BUFS;

        if (write_futures[slot].valid()) write_futures[slot].get();

        planes_to_rgb_parallel(curr_recon, rgb_ring[slot], use_ycbcr);

        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        std::string out_path_png = out_dir + filename;
        int w = header.width, h = header.height, ch_count = header.channels;
        uint8_t* buf_ptr = rgb_ring[slot].data();

        write_futures[slot] = std::async(std::launch::async,
                                         [out_path_png, buf_ptr, w, h, ch_count]() {
                                             stbi_write_png(out_path_png.c_str(), w, h, ch_count, buf_ptr,
                                                            w * ch_count);
                                         });

        std::cout << "\r  Progress: " << f_idx + 1 << "/" << header.frame_count << std::flush;

        std::swap(curr_recon.data, prev_recon.data);
    }

    for (int i = 0; i < NUM_WRITE_BUFS; ++i)
        if (write_futures[i].valid()) write_futures[i].get();

    double avg_ms = total_decode_ms / header.frame_count;
    double fps = 1000.0 / avg_ms;
    std::cout << "\n  Finished decompressing flipbook.\n";
    std::cout << "  [BENCHMARK] decode_total_ms=" << total_decode_ms
              << " frames=" << header.frame_count
              << " avg_ms=" << avg_ms
              << " decode_fps=" << fps << "\n";

    frame_destroy(prev_recon);
    frame_destroy(curr_recon);
}
