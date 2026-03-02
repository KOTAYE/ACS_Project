#include "codec.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <future>

#include "cuda_kernels.cuh"

#include "frame.h"
#include "image_io.h"
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
};
#pragma pack(pop)


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


void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality) {
    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Input must be a valid directory containing frames.\n";
        return;
    }

    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(in_dir))
        if (entry.is_regular_file()) frames.push_back(entry.path().string());
    std::sort(frames.begin(), frames.end());

    if (frames.empty()) { std::cerr << "No frames found in " << in_dir << "\n"; return; }

    int img_w, img_h, img_ch;
    uint8_t* first_raw = stbi_load(frames[0].c_str(), &img_w, &img_h, &img_ch, 0);
    if (!first_raw) { std::cerr << "Failed to load first frame\n"; return; }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::cerr << "Failed to open " << out_path << "\n"; stbi_image_free(first_raw); return; }

    BinHeader header;
    header.width = img_w;   header.height = img_h;
    header.channels = img_ch; header.quality = quality;
    header.frame_count = static_cast<int32_t>(frames.size());
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, quality);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality);

    cuda_alloc_frame_buffers(img_w, img_h, img_ch, luma_qm.data(), chroma_qm.data());

    int pw[3], ph[3];
    for (int ch = 0; ch < img_ch; ++ch) {
        int w = img_w, h = img_h;
        if (ch > 0 && img_ch == 3) { w = (img_w+1)/2; h = (img_h+1)/2; }
        pw[ch] = codec_pad8(w);  ph[ch] = codec_pad8(h);
    }

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";

    std::future<uint8_t*> prefetch;

    for (size_t f_idx = 0; f_idx < frames.size(); ++f_idx) {
        uint8_t* raw;
        if (f_idx == 0) {
            raw = first_raw;
        } else {
            raw = prefetch.get();
            if (!raw) { std::cerr << "\nFailed to load frame " << frames[f_idx] << "\n"; break; }
        }

        if (f_idx + 1 < frames.size()) {
            prefetch = std::async(std::launch::async,
                [path = frames[f_idx + 1]]() -> uint8_t* {
                    int w, h, c;
                    return stbi_load(path.c_str(), &w, &h, &c, 0);
                });
        }

        cuda_upload_and_convert_rgb(raw, img_w, img_h, img_ch);
        stbi_image_free(raw);

        bool is_keyframe = (f_idx == 0);

        for (int ch = 0; ch < img_ch; ++ch) {
            int total_blocks = (pw[ch] / 8) * (ph[ch] / 8);
            int samples = total_blocks * 64;
            if (static_cast<int>(channel_buffer.size()) < samples) channel_buffer.resize(samples);

            cuda_encode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], is_keyframe);
            cuda_sync_channel(ch);

            std::vector<int16_t> rle_in(channel_buffer.begin(), channel_buffer.begin() + samples);
            std::vector<int16_t> rle_buf = rle_encode_zeros(rle_in);
            const uint8_t* raw_bytes = reinterpret_cast<const uint8_t*>(rle_buf.data());
            const int raw_len = static_cast<int>(rle_buf.size() * sizeof(int16_t));

            if (encoded.size() < raw_len + 1024) encoded.resize(raw_len + 1024);

            int enc_len = huffman_encode_bytes(raw_bytes, raw_len, encoded.data(), encoded.size());
            if (enc_len < 0) { std::cerr << "\nHuffman overflow\n"; enc_len = 0; }

            uint32_t rle_bytes = static_cast<uint32_t>(raw_len);
            out.write(reinterpret_cast<const char*>(&rle_bytes), sizeof(rle_bytes));
            uint32_t len32 = static_cast<uint32_t>(enc_len);
            out.write(reinterpret_cast<const char*>(&len32), sizeof(len32));
            if (enc_len > 0) out.write(reinterpret_cast<const char*>(encoded.data()), enc_len);
        }

        cuda_swap_recon();
        std::cout << "\r  Progress: " << f_idx + 1 << "/" << frames.size() << std::flush;
    }
    std::cout << "\n  Finished compressing flipbook.\n";
    cuda_free_frame_buffers();
}


void decompress_flipbook(const std::string& in_path, const std::string& out_dir) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << in_path << "\n"; return; }

    BinHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (in.gcount() != sizeof(header) || header.magic[0] != 'F' ||
        header.magic[1] != 'L' || header.magic[2] != 'I' || header.magic[3] != '2') {
        std::cerr << "Invalid bin file: " << in_path << "\n"; return;
    }

    fs::create_directories(out_dir);

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, header.quality);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, header.quality);

    cuda_alloc_frame_buffers(header.width, header.height, header.channels,
                             luma_qm.data(), chroma_qm.data());

    int pw[3], ph[3];
    for (int ch = 0; ch < header.channels; ++ch) {
        int w = header.width, h = header.height;
        if (ch > 0 && header.channels == 3) { w = (header.width+1)/2; h = (header.height+1)/2; }
        pw[ch] = codec_pad8(w);  ph[ch] = codec_pad8(h);
    }

    std::vector<uint8_t> rgb_buf(static_cast<size_t>(header.width) * header.height * header.channels);

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> encoded;

    std::cout << "Decompressing " << header.frame_count << " frames to " << out_dir << "...\n";

    for (int f_idx = 0; f_idx < header.frame_count; ++f_idx) {
        bool is_keyframe = (f_idx == 0);

        for (int ch = 0; ch < header.channels; ++ch) {
            int total_blocks = (pw[ch] / 8) * (ph[ch] / 8);
            int samples = total_blocks * 64;
            if (static_cast<int>(channel_buffer.size()) < samples) channel_buffer.resize(samples);

            uint32_t rle_bytes_len = 0, len32 = 0;
            in.read(reinterpret_cast<char*>(&rle_bytes_len), sizeof(rle_bytes_len));
            in.read(reinterpret_cast<char*>(&len32), sizeof(len32));

            std::fill(channel_buffer.begin(), channel_buffer.end(), 0);

            if (len32 > 0) {
                encoded.resize(len32);
                in.read(reinterpret_cast<char*>(encoded.data()), len32);
                std::vector<int16_t> rle_buf(rle_bytes_len / sizeof(int16_t));
                huffman_decode_bytes(encoded.data(), len32,
                                    reinterpret_cast<uint8_t*>(rle_buf.data()), rle_bytes_len);
                rle_decode_zeros(rle_buf, channel_buffer);
            }

            cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], is_keyframe);
        }

        cuda_sync_all();

        cuda_download_and_convert_rgb(rgb_buf.data(), header.width, header.height, header.channels);

        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        std::string fpath = out_dir + filename;
        stbi_write_png(fpath.c_str(), header.width, header.height, header.channels,
                       rgb_buf.data(), header.width * header.channels);

        std::cout << "\r  Progress: " << f_idx + 1 << "/" << header.frame_count << std::flush;
        cuda_swap_recon();
    }
    std::cout << "\n  Finished decompressing flipbook.\n";
    cuda_free_frame_buffers();
}
