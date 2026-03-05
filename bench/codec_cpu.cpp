#include "codec.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <future>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "frame.h"
#include "tiling.h"
#include "dct.h"
#include "quant.h"
#include "huffman.h"
#include "metrics.h"

#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

inline int codec_pad8(int n) { return ((n + 7) / 8) * 8; }

#pragma pack(push, 1)
struct BinHeader {
    char magic[4] = {'F', 'L', 'I', '2'};
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t quality;
    int32_t frame_count;
    int32_t use_ycbcr = 1;
};
#pragma pack(pop)

const int ZIGZAG_ORDER[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

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

Frame rgb_to_planes_parallel(uint8_t* raw, int w, int h, int ch, bool use_ycbcr) {
    Frame f = frame_create(w, h, ch, use_ycbcr);
    if (use_ycbcr && ch >= 3) {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float R = static_cast<float>(raw[(y * w + x) * ch + 0]);
                float G = static_cast<float>(raw[(y * w + x) * ch + 1]);
                float B = static_cast<float>(raw[(y * w + x) * ch + 2]);

                float Y  =  0.299f * R + 0.587f * G + 0.114f * B;
                f.at(0, x, y) = Y;

                if (x % 2 == 0 && y % 2 == 0) {
                    float Cb = -0.168736f * R - 0.331264f * G + 0.5f * B + 128.0f;
                    float Cr =  0.5f * R - 0.418688f * G - 0.081312f * B + 128.0f;
                    f.at(1, x / 2, y / 2) = Cb;
                    f.at(2, x / 2, y / 2) = Cr;
                }
            }
        }
    } else {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ch; ++c) {
                    f.at(c, x, y) = static_cast<float>(raw[(y * w + x) * ch + c]);
                }
            }
        }
    }
    return f;
}

void planes_to_rgb_parallel(const Frame& f, std::vector<uint8_t>& out_rgb, bool use_ycbcr) {
    int w = f.width;
    int h = f.height;
    int ch = f.channels;

    if (use_ycbcr && ch >= 3) {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float Y  = f.at(0, x, y);
                float Cb = f.at(1, x / 2, y / 2) - 128.0f;
                float Cr = f.at(2, x / 2, y / 2) - 128.0f;

                float R = Y + 1.402f * Cr;
                float G = Y - 0.344136f * Cb - 0.714136f * Cr;
                float B = Y + 1.772f * Cb;

                out_rgb[(y * w + x) * ch + 0] = static_cast<uint8_t>(std::clamp(R + 0.5f, 0.0f, 255.0f));
                out_rgb[(y * w + x) * ch + 1] = static_cast<uint8_t>(std::clamp(G + 0.5f, 0.0f, 255.0f));
                out_rgb[(y * w + x) * ch + 2] = static_cast<uint8_t>(std::clamp(B + 0.5f, 0.0f, 255.0f));

                if (ch == 4) out_rgb[(y * w + x) * ch + 3] = 255;
            }
        }
    } else {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ch; ++c) {
                    float val = f.at(c, x, y);
                    out_rgb[(y * w + x) * ch + c] = static_cast<uint8_t>(std::clamp(val + 0.5f, 0.0f, 255.0f));
                }
            }
        }
    }
}

void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality, bool use_ycbcr) {
    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Input must be a valid directory containing frames.\n";
        return;
    }

    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(in_dir)) {
        if (entry.is_regular_file()) frames.push_back(entry.path().string());
    }
    std::sort(frames.begin(), frames.end());

    if (frames.empty()) {
        std::cerr << "No frames found in directory " << in_dir << "\n";
        return;
    }

    int img_w, img_h, img_ch;
    uint8_t* first_raw = stbi_load(frames[0].c_str(), &img_w, &img_h, &img_ch, 0);
    if (!first_raw) {
        std::cerr << "Failed to load the first frame: " << frames[0] << "\n";
        return;
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << out_path << " for writing\n";
        stbi_image_free(first_raw);
        return;
    }

    BinHeader header;
    header.width = img_w;
    header.height = img_h;
    header.channels = img_ch;
    header.quality = quality;
    header.frame_count = static_cast<int32_t>(frames.size());
    header.use_ycbcr = use_ycbcr ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, quality);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality);

    Frame prev_recon = frame_create(img_w, img_h, img_ch, use_ycbcr);
    Frame curr_recon = frame_create(img_w, img_h, img_ch, use_ycbcr);

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";

#ifdef USE_OMP
    std::future<uint8_t*> prefetch;
#endif

    for (size_t f_idx = 0; f_idx < frames.size(); ++f_idx) {
        uint8_t* raw = nullptr;
        if (f_idx == 0) {
            raw = first_raw;
        } else {
#ifdef USE_OMP
            raw = prefetch.get();
#else
            int w, h, c;
            raw = stbi_load(frames[f_idx].c_str(), &w, &h, &c, 0);
#endif
            if (!raw) {
                std::cerr << "\nError: Invalid frame or dimensions mismatched at " << frames[f_idx] << "\n";
                break;
            }
        }

#ifdef USE_OMP
        if (f_idx + 1 < frames.size()) {
            prefetch = std::async(std::launch::async, [path = frames[f_idx + 1]]() -> uint8_t* {
                int w, h, c;
                return stbi_load(path.c_str(), &w, &h, &c, 0);
            });
        }
#endif

        Frame current = rgb_to_planes_parallel(raw, img_w, img_h, img_ch, use_ycbcr);
        stbi_image_free(raw);

        bool is_keyframe = (f_idx == 0);

        for (int ch = 0; ch < current.channels; ++ch) {
            const QuantMatrix& qm = (ch == 0) ? luma_qm : chroma_qm;
            const float* src_channel = current.channel_ptr(ch);
            const float* prev_channel = prev_recon.channel_ptr(ch);
            float* recon_channel = curr_recon.channel_ptr(ch);

            const int padded_w = current.padded_width[ch];
            const int padded_h = current.padded_height[ch];
            const int blocks_x = padded_w / 8;
            const int blocks_y = padded_h / 8;
            const int total_blocks = blocks_x * blocks_y;
            const int channel_samples = total_blocks * 64;

            if (static_cast<int>(channel_buffer.size()) < channel_samples)
                channel_buffer.resize(channel_samples);

            #ifdef USE_OMP
            #pragma omp parallel for
            #endif
            for (int by = 0; by < blocks_y; ++by) {
                float block_in[64], prev_block[64], dct_out[64], idct_out[64];
                for (int bx = 0; bx < blocks_x; ++bx) {
                    int sample_idx = (by * blocks_x + bx) * 64;
                    extract_block_8x8(src_channel, padded_w, bx, by, block_in);

                    if (!is_keyframe) {
                        extract_block_8x8(prev_channel, padded_w, bx, by, prev_block);
                        for (int i = 0; i < 64; ++i) block_in[i] -= prev_block[i];
                    } else {
                        level_shift(block_in, -128.0f);
                    }

                    dct2d_separable(block_in, dct_out);
                    quantize_block_64(dct_out, qm);

                    for (int i = 0; i < 64; ++i)
                        channel_buffer[sample_idx + i] = static_cast<int16_t>(dct_out[ZIGZAG_ORDER[i]]);

                    dequantize_block_64(dct_out, qm);
                    idct2d_separable(dct_out, idct_out);

                    if (!is_keyframe) {
                        for (int i = 0; i < 64; ++i) idct_out[i] += prev_block[i];
                    } else {
                        level_shift(idct_out, +128.0f);
                    }

                    insert_block_8x8(recon_channel, padded_w, bx, by, idct_out);
                }
            }

            std::vector<int16_t> current_ch_buffer(channel_buffer.begin(), channel_buffer.begin() + channel_samples);
            std::vector<int16_t> rle_buffer = rle_encode_zeros(current_ch_buffer);
            const uint8_t* raw_bytes = reinterpret_cast<const uint8_t*>(rle_buffer.data());
            const int raw_len = static_cast<int>(rle_buffer.size() * sizeof(int16_t));

            if (encoded.size() < raw_len + 1024) encoded.resize(raw_len + 1024);

            int encoded_len = huffman_encode_bytes(raw_bytes, raw_len, encoded.data(), encoded.size());
            if (encoded_len < 0) {
                std::cerr << "\nCompression failed (buffer overflow)\n";
                encoded_len = 0;
            }

            uint32_t rle_bytes_len = static_cast<uint32_t>(raw_len);
            out.write(reinterpret_cast<const char*>(&rle_bytes_len), sizeof(rle_bytes_len));

            uint32_t len_u32 = static_cast<uint32_t>(encoded_len);
            out.write(reinterpret_cast<const char*>(&len_u32), sizeof(len_u32));
            if (encoded_len > 0)
                out.write(reinterpret_cast<const char*>(encoded.data()), encoded_len);
        }

        std::swap(curr_recon.data, prev_recon.data);
        frame_destroy(current);
        std::cout << "\r  Progress: " << f_idx + 1 << "/" << frames.size() << std::flush;
    }
    std::cout << "\n  Finished compressing flipbook.\n";

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
    if (in.gcount() != sizeof(header) || header.magic[0] != 'F' || header.magic[1] != 'L' ||
        header.magic[2] != 'I' || header.magic[3] != '2') {
        std::cerr << "Invalid or corrupted bin file: " << in_path << "\n";
        return;
    }

    fs::create_directories(out_dir);

    bool use_ycbcr = (header.use_ycbcr != 0);

    Frame prev_recon = frame_create(header.width, header.height, header.channels, use_ycbcr);
    Frame curr_recon = frame_create(header.width, header.height, header.channels, use_ycbcr);

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, header.quality);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, header.quality);

    // Use fast PNG compression (level 1 instead of default 8)
    stbi_write_png_compression_level = 1;

    // Ring buffer of N frames for concurrent PNG writes
    constexpr int NUM_WRITE_BUFS = 4;
    const size_t rgb_size = static_cast<size_t>(header.width) * header.height * header.channels;

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;
    std::vector<std::vector<uint8_t>> rgb_ring(NUM_WRITE_BUFS);
    for (auto& buf : rgb_ring) buf.resize(rgb_size);
    std::future<void> write_futures[NUM_WRITE_BUFS];

    std::cout << "Decompressing " << header.frame_count << " frames to " << out_dir << "...\n";

    for (int f_idx = 0; f_idx < header.frame_count; ++f_idx) {
        bool is_keyframe = (f_idx == 0);

        for (int ch = 0; ch < curr_recon.channels; ++ch) {
            const QuantMatrix& qm = (ch == 0) ? luma_qm : chroma_qm;
            float* recon_channel = curr_recon.channel_ptr(ch);
            const float* prev_channel = prev_recon.channel_ptr(ch);

            const int blocks_x = prev_recon.padded_width[ch] / 8;
            const int blocks_y = prev_recon.padded_height[ch] / 8;
            const int total_blocks = blocks_x * blocks_y;
            const int channel_samples = total_blocks * 64;

            if (static_cast<int>(channel_buffer.size()) < channel_samples)
                channel_buffer.resize(channel_samples);

            uint32_t rle_bytes_len = 0;
            in.read(reinterpret_cast<char*>(&rle_bytes_len), sizeof(rle_bytes_len));

            uint32_t len_u32 = 0;
            in.read(reinterpret_cast<char*>(&len_u32), sizeof(len_u32));

            std::fill(channel_buffer.begin(), channel_buffer.end(), 0);

            if (len_u32 > 0) {
                encoded.resize(len_u32);
                in.read(reinterpret_cast<char*>(encoded.data()), len_u32);

                std::vector<int16_t> rle_buffer(rle_bytes_len / sizeof(int16_t));
                uint8_t* raw_bytes = reinterpret_cast<uint8_t*>(rle_buffer.data());

                huffman_decode_bytes(encoded.data(), len_u32, raw_bytes, rle_bytes_len);
                rle_decode_zeros(rle_buffer, channel_buffer);
            }

            #ifdef USE_OMP
            #pragma omp parallel for
            #endif
            for (int by = 0; by < blocks_y; ++by) {
                float dct_out[64], idct_out[64], prev_block[64];
                for (int bx = 0; bx < blocks_x; ++bx) {
                    int sample_idx = (by * blocks_x + bx) * 64;
                    for (int i = 0; i < 64; ++i)
                        dct_out[ZIGZAG_ORDER[i]] = static_cast<float>(channel_buffer[sample_idx + i]);

                    dequantize_block_64(dct_out, qm);
                    idct2d_separable(dct_out, idct_out);

                    if (!is_keyframe) {
                        extract_block_8x8(prev_channel, prev_recon.padded_width[ch], bx, by, prev_block);
                        for (int i = 0; i < 64; ++i) idct_out[i] += prev_block[i];
                    } else {
                        level_shift(idct_out, +128.0f);
                    }

                    insert_block_8x8(recon_channel, curr_recon.padded_width[ch], bx, by, idct_out);
                }
            }
        }

        int slot = f_idx % NUM_WRITE_BUFS;

        // Wait for this slot's previous write to complete before reusing its buffer
        if (write_futures[slot].valid()) write_futures[slot].get();

        // YCbCr/raw -> RGB into this slot's buffer
        planes_to_rgb_parallel(curr_recon, rgb_ring[slot], use_ycbcr);

        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        std::string out_path = out_dir + filename;
        int w = header.width, h = header.height, ch_count = header.channels;
        uint8_t* buf_ptr = rgb_ring[slot].data();

        write_futures[slot] = std::async(std::launch::async,
            [out_path, buf_ptr, w, h, ch_count]() {
                stbi_write_png(out_path.c_str(), w, h, ch_count, buf_ptr, w * ch_count);
            });

        std::cout << "\r  Progress: " << f_idx + 1 << "/" << header.frame_count << std::flush;

        std::swap(curr_recon.data, prev_recon.data);
    }

    // Wait for all remaining writes
    for (int i = 0; i < NUM_WRITE_BUFS; ++i)
        if (write_futures[i].valid()) write_futures[i].get();

    std::cout << "\n  Finished decompressing flipbook.\n";

    frame_destroy(prev_recon);
    frame_destroy(curr_recon);
}
