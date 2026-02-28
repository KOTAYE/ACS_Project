#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include "frame.h"
#include "image_io.h"
#include "tiling.h"
#include "dct.h"
#include "quant.h"
#include "huffman.h"
#include "metrics.h"

namespace fs = std::filesystem;

#pragma pack(push, 1)
struct BinHeader {
    char magic[4] = {'F', 'L', 'I', '2'};
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t quality;
    int32_t frame_count;
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
            while (i + run < n && in[i + run] == 0 && run < 32767) {
                run++;
            }
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
            for (int k = 0; k < run && o < out_n; ++k) {
                out[o++] = 0;
            }
            i += 2;
        } else {
            out[o++] = in[i];
            i++;
        }
    }
}

void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality) {
    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Input must be a valid directory containing frames.\n";
        return;
    }

    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(in_dir)) {
        if (entry.is_regular_file()) {
            frames.push_back(entry.path().string());
        }
    }
    std::sort(frames.begin(), frames.end());

    if (frames.empty()) {
        std::cerr << "No frames found in directory " << in_dir << "\n";
        return;
    }

    Frame first_frame = load_image(frames[0].c_str());
    if (!first_frame.data) {
        std::cerr << "Failed to load the first frame: " << frames[0] << "\n";
        return;
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << out_path << " for writing\n";
        frame_destroy(first_frame);
        return;
    }

    BinHeader header;
    header.width = first_frame.width;
    header.height = first_frame.height;
    header.channels = first_frame.channels;
    header.quality = quality;
    header.frame_count = static_cast<int32_t>(frames.size());
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, quality);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality);

    Frame prev_recon = frame_create(first_frame.width, first_frame.height, first_frame.channels);
    Frame curr_recon = frame_create(first_frame.width, first_frame.height, first_frame.channels);

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;

    float block_in[64];
    float prev_block[64];
    float dct_out[64];
    float idct_out[64];

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";

    for (size_t f_idx = 0; f_idx < frames.size(); ++f_idx) {
        Frame current;
        if (f_idx == 0) {
            current = first_frame;
        } else {
            current = load_image(frames[f_idx].c_str());
            if (!current.data || current.width != first_frame.width || current.height != first_frame.height || current.channels != first_frame.channels) {
                std::cerr << "\nError: Invalid frame or dimensions mismatched at " << frames[f_idx] << "\n";
                if (current.data) frame_destroy(current);
                break;
            }
        }

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

            if (channel_buffer.size() < channel_samples) channel_buffer.resize(channel_samples);

            int sample_idx = 0;
            for (int by = 0; by < blocks_y; ++by) {
                for (int bx = 0; bx < blocks_x; ++bx) {
                    extract_block_8x8(src_channel, padded_w, bx, by, block_in);
                    
                    if (!is_keyframe) {
                        extract_block_8x8(prev_channel, padded_w, bx, by, prev_block);
                        for (int i = 0; i < 64; ++i) block_in[i] -= prev_block[i];
                    } else {
                        level_shift(block_in, -128.0f);
                    }

                    dct2d_separable(block_in, dct_out);
                    quantize_block_64(dct_out, qm);

                    for (int i = 0; i < 64; ++i) {
                        channel_buffer[sample_idx++] = static_cast<int16_t>(dct_out[ZIGZAG_ORDER[i]]);
                    }

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

            if (encoded.size() < raw_len + 1024) {
                encoded.resize(raw_len + 1024);
            }

            int encoded_len = huffman_encode_bytes(raw_bytes, raw_len, encoded.data(), encoded.size());
            if (encoded_len < 0) {
                std::cerr << "\nCompression failed (buffer overflow)\n";
                encoded_len = 0;
            }

            uint32_t rle_bytes_len = static_cast<uint32_t>(raw_len);
            out.write(reinterpret_cast<const char*>(&rle_bytes_len), sizeof(rle_bytes_len));

            uint32_t len_u32 = static_cast<uint32_t>(encoded_len);
            out.write(reinterpret_cast<const char*>(&len_u32), sizeof(len_u32));
            if (encoded_len > 0) {
                out.write(reinterpret_cast<const char*>(encoded.data()), encoded_len);
            }
        }
        
        std::swap(curr_recon.data, prev_recon.data);
        
        if (f_idx != 0) frame_destroy(current);
        std::cout << "\r  Progress: " << f_idx + 1 << "/" << frames.size() << std::flush;
    }
    std::cout << "\n  Finished compressing flipbook.\n";

    if (first_frame.data) frame_destroy(first_frame);
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

    Frame prev_recon = frame_create(header.width, header.height, header.channels);
    Frame curr_recon = frame_create(header.width, header.height, header.channels);

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, header.quality);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, header.quality);

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;

    float dct_out[64];
    float idct_out[64];
    float prev_block[64];

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

            if (channel_buffer.size() < channel_samples) channel_buffer.resize(channel_samples);

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

            int sample_idx = 0;
            for (int by = 0; by < blocks_y; ++by) {
                for (int bx = 0; bx < blocks_x; ++bx) {
                    for (int i = 0; i < 64; ++i) {
                        dct_out[ZIGZAG_ORDER[i]] = static_cast<float>(channel_buffer[sample_idx++]);
                    }

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
        
        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        std::string out_path = out_dir + filename;
        
        save_image(out_path.c_str(), curr_recon);
        std::cout << "\r  Progress: " << f_idx + 1 << "/" << header.frame_count << std::flush;
        
        std::swap(curr_recon.data, prev_recon.data);
    }
    std::cout << "\n  Finished decompressing flipbook.\n";
    
    frame_destroy(prev_recon);
    frame_destroy(curr_recon);
}

void print_usage(const char* prog_name) {
    std::cerr << "Usage:\n"
              << "  " << prog_name << " compress [-q <quality>] <input_directory> <output.bin>\n"
              << "  " << prog_name << " decompress <input.bin> <output_directory>\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    dct_init_lut();

    std::string mode = argv[1];

    if (mode == "compress") {
        int quality = 50;
        int in_arg_idx = 2;
        
        if (argc >= 4 && (std::string_view(argv[2]) == "-q" || std::string_view(argv[2]) == "--quality")) {
            quality = std::stoi(argv[3]);
            if (quality < 1 || quality > 100) {
                std::cerr << "Error: quality must be between 1 and 100.\n";
                return 1;
            }
            in_arg_idx = 4;
        }

        if (argc < in_arg_idx + 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string in_dir = argv[in_arg_idx];
        std::string out_path = argv[in_arg_idx + 1];
        compress_flipbook(in_dir, out_path, quality);

    } else if (mode == "decompress") {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        std::string in_path = argv[2];
        std::string out_dir = argv[3];
        decompress_flipbook(in_path, out_dir);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}