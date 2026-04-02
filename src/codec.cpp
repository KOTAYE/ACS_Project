#include "codec.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <cstring>

#include <cuda_runtime.h>
#include "cuda_kernels.cuh"
#include "rle_gpu.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
#endif

#include "frame.h"
#include "image_io.h"
#include "quant.h"
#include "huffman.h"
#include "rle.h"

#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

inline int codec_pad(int n, int bs) { return ((n + bs - 1) / bs) * bs; }

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

struct EncodedChannel {
    uint32_t rle_bytes;
    uint32_t enc_len;
    uint32_t huffman_freq[256];
    std::vector<uint32_t> block_bit_lengths;
    std::vector<uint8_t> data;
};

static void write_encoded_channel(std::ofstream& out, const EncodedChannel& ch) {
    out.write(reinterpret_cast<const char*>(&ch.rle_bytes), 4);
    out.write(reinterpret_cast<const char*>(&ch.enc_len), 4);
    const uint32_t num_blocks = static_cast<uint32_t>(ch.block_bit_lengths.size());
    out.write(reinterpret_cast<const char*>(&num_blocks), 4);
    out.write(reinterpret_cast<const char*>(ch.huffman_freq), sizeof(ch.huffman_freq));
    out.write(reinterpret_cast<const char*>(ch.block_bit_lengths.data()), num_blocks * sizeof(uint32_t));
    if (!ch.data.empty())
        out.write(reinterpret_cast<const char*>(ch.data.data()), static_cast<std::streamsize>(ch.data.size()));
}

void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality, int block_size, bool use_ycbcr) {
    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Input must be a valid directory containing frames.\n";
        return;
    }

    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(in_dir))
        if (entry.is_regular_file()) frames.push_back(entry.path().string());
    sort_frame_paths(frames);

    if (frames.empty()) {
        std::cerr << "No frames found in " << in_dir << "\n";
        return;
    }

    int img_w = 0, img_h = 0, img_ch = 0;
    {
        const std::vector<uint8_t> probe_file = read_file_bytes(frames[0]);
        std::vector<uint8_t> probe_rgb;
        if (!decode_image_file_bytes(frames[0], probe_file.data(), probe_file.size(), img_w, img_h, img_ch, probe_rgb)) {
            std::cerr << "Failed to decode first frame: " << frames[0] << "\n";
            return;
        }
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << out_path << "\n";
        return;
    }

    BinHeader header;
    header.width = img_w;
    header.height = img_h;
    header.channels = img_ch;
    header.quality = quality;
    header.block_size = block_size;
    header.frame_count = static_cast<int32_t>(frames.size());
    header.use_ycbcr = use_ycbcr ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, quality, block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality, block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(block_size);

    cuda_alloc_frame_buffers(img_w, img_h, img_ch, luma_qm.data(), chroma_qm.data(), zigzag.data(), use_ycbcr,
                             block_size);

    int pw[3], ph[3];
    for (int ch = 0; ch < img_ch; ++ch) {
        int w = img_w, h = img_h;
        if (use_ycbcr && ch > 0 && img_ch == 3) {
            w = (img_w + 1) / 2;
            h = (img_h + 1) / 2;
        }
        pw[ch] = codec_pad(w, block_size);
        ph[ch] = codec_pad(h, block_size);
    }

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";
    const auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t f_idx = 0; f_idx < frames.size(); ++f_idx) {
        const std::vector<uint8_t> file_bytes = read_file_bytes(frames[f_idx]);
        if (file_bytes.empty()) {
            std::cerr << "Failed to read: " << frames[f_idx] << "\n";
            cuda_cleanup();
            return;
        }

        std::vector<uint8_t> interleaved;
        int w = 0, h = 0, c = 0;
        if (!decode_image_file_bytes(frames[f_idx], file_bytes.data(), file_bytes.size(), w, h, c, interleaved)) {
            std::cerr << "Decode failed: " << frames[f_idx] << "\n";
            cuda_cleanup();
            return;
        }
        if (w != img_w || h != img_h || c != img_ch) {
            std::cerr << "Frame size mismatch: " << frames[f_idx] << "\n";
            cuda_cleanup();
            return;
        }

        Frame f = rgb_to_planes_parallel(interleaved.data(), img_w, img_h, img_ch, use_ycbcr, block_size);
        const bool is_keyframe = (f_idx == 0);
        const uint8_t* up[3] = {f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2)};

        cuda_upload_frame(up, f.channels);

        for (int ch = 0; ch < img_ch; ++ch)
            cuda_encode_channel(ch, pw[ch], ph[ch], block_size, is_keyframe);

        for (int ch = 0; ch < img_ch; ++ch) {
            const int num_blocks = (pw[ch] / block_size) * (ph[ch] / block_size);
            void* stream = cuda_channel_stream_ptr(ch);
            cuda_rle_encode_async(ch, num_blocks, block_size);
            cuda_compute_histogram(ch, nullptr, stream);
            cuda_prepare_huffman_codebook_gpu(ch, stream);
            cuda_pack_channel_indexed(ch, num_blocks, block_size, nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        cuda_sync_all();

        for (int ch = 0; ch < img_ch; ++ch) {
            const int num_blocks = (pw[ch] / block_size) * (ph[ch] / block_size);
            EncodedChannel ec;
            cuda_get_pinned_metadata(ch, &ec.rle_bytes, &ec.enc_len);
            ec.block_bit_lengths.resize(static_cast<size_t>(num_blocks));

            if (ec.rle_bytes > 0) {
                cuda_huffman_download_block_bit_lengths(ch, ec.block_bit_lengths.data(), num_blocks);
                ec.data.resize(ec.enc_len);
                uint8_t* d_pack = cuda_get_bitstream_ptr(ch);
                if (ec.enc_len > 0 && d_pack)
                    cuda_memcpy_to_host(ec.data.data(), d_pack, ec.enc_len);
                cuda_compute_histogram(ch, ec.huffman_freq, nullptr);
            } else {
                std::memset(ec.huffman_freq, 0, sizeof(ec.huffman_freq));
                std::fill(ec.block_bit_lengths.begin(), ec.block_bit_lengths.end(), 0u);
            }
            write_encoded_channel(out, ec);
        }

        cuda_swap_recon();
        frame_destroy(f);
        std::cout << "\r  Progress: " << (f_idx + 1) << "/" << frames.size() << std::flush;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double compress_fps = 1000.0 * static_cast<double>(frames.size()) / compress_ms;
    const auto file_size = fs::file_size(out_path);
    const size_t raw_size = static_cast<size_t>(img_w) * img_h * img_ch * frames.size();
    const double ratio = static_cast<double>(raw_size) / static_cast<double>(file_size);

    std::cout << "\n[BENCHMARK] compress_ms=" << compress_ms << " frames=" << frames.size()
              << " compress_fps=" << compress_fps << " raw_bytes=" << raw_size
              << " compressed_bytes=" << file_size << " compression_ratio=" << ratio << "\n";
    cuda_cleanup();
}

void decompress_flipbook(const std::string& in_path, const std::string& out_dir) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << in_path << "\n";
        return;
    }

    BinHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (in.gcount() != sizeof(header) || header.magic[0] != 'F' || header.magic[1] != 'L' ||
        header.magic[2] != 'I' || header.magic[3] != '3') {
        std::cerr << "Invalid bin file: " << in_path << "\n";
        return;
    }

    fs::create_directories(out_dir);

    const bool use_ycbcr = (header.use_ycbcr != 0);
    const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, header.quality, header.block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, header.quality, header.block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(header.block_size);

    cuda_alloc_frame_buffers(header.width, header.height, header.channels, luma_qm.data(), chroma_qm.data(),
                             zigzag.data(), header.use_ycbcr != 0, header.block_size);

    int pw[3], ph[3];
    for (int ch = 0; ch < header.channels; ++ch) {
        int w = header.width, h = header.height;
        if (use_ycbcr && ch > 0 && header.channels == 3) {
            w = (header.width + 1) / 2;
            h = (header.height + 1) / 2;
        }
        pw[ch] = codec_pad(w, header.block_size);
        ph[ch] = codec_pad(h, header.block_size);
    }

    stbi_write_png_compression_level = 1;

    std::vector<uint8_t> encoded;
    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> rgb;
    rgb.resize(static_cast<size_t>(header.width) * header.height * header.channels);

    std::cout << "Decompressing " << header.frame_count << " frames to " << out_dir << "...\n";
    double total_decode_ms = 0.0;

    for (int f_idx = 0; f_idx < header.frame_count; ++f_idx) {
        const bool is_keyframe = (f_idx == 0);
        const auto t_decode_start = std::chrono::high_resolution_clock::now();

        for (int ch = 0; ch < header.channels; ++ch) {
            uint32_t rle_bytes_len = 0, len32 = 0, num_blocks = 0;
            in.read(reinterpret_cast<char*>(&rle_bytes_len), sizeof(rle_bytes_len));
            in.read(reinterpret_cast<char*>(&len32), sizeof(len32));
            in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));

            const int total_blocks = (pw[ch] / header.block_size) * (ph[ch] / header.block_size);
            const int samples = total_blocks * header.block_size * header.block_size;
            if (static_cast<int>(channel_buffer.size()) < samples) channel_buffer.resize(samples);

            if (len32 > 0) {
                std::vector<uint32_t> h_freq(256);
                in.read(reinterpret_cast<char*>(h_freq.data()), h_freq.size() * sizeof(uint32_t));

                std::vector<uint32_t> block_bit_lengths(num_blocks);
                in.read(reinterpret_cast<char*>(block_bit_lengths.data()), num_blocks * sizeof(uint32_t));

                encoded.resize(len32);
                in.read(reinterpret_cast<char*>(encoded.data()), len32);

                bool per_block_entropy = false;
                for (uint32_t bl : block_bit_lengths) {
                    if (bl != 0) {
                        per_block_entropy = true;
                        break;
                    }
                }

                if (per_block_entropy) {
                    if (static_cast<int>(num_blocks) != total_blocks) {
                        std::cerr << "\nnum_blocks mismatch (ch=" << ch << " frame=" << f_idx << ")\n";
                        std::fill(channel_buffer.begin(), channel_buffer.begin() + samples, 0);
                        cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], header.block_size,
                                            is_keyframe);
                    } else {
                        cuda_full_decode_channel(ch, encoded.data(), len32, block_bit_lengths.data(), total_blocks,
                                                 h_freq.data(), pw[ch], ph[ch], header.block_size, is_keyframe);
                        cuda_sync_channel(ch);
                    }
                } else {
                    std::vector<uint8_t> enc_with_hdr(sizeof(uint32_t) * 256 + len32);
                    std::memcpy(enc_with_hdr.data(), h_freq.data(), sizeof(uint32_t) * 256);
                    std::memcpy(enc_with_hdr.data() + sizeof(uint32_t) * 256, encoded.data(), len32);

                    std::vector<int16_t> rle_buf(rle_bytes_len / sizeof(int16_t));
                    if (huffman_decode_bytes(enc_with_hdr.data(), static_cast<int>(enc_with_hdr.size()),
                                             reinterpret_cast<uint8_t*>(rle_buf.data()),
                                             static_cast<int>(rle_bytes_len)) != 0) {
                        std::cerr << "\nHuffman decode failed (ch=" << ch << " frame=" << f_idx << ")\n";
                        std::fill(channel_buffer.begin(), channel_buffer.begin() + samples, 0);
                    } else {
                        std::fill(channel_buffer.begin(), channel_buffer.end(), 0);
                        rle_decode_zeros(rle_buf, channel_buffer);
                    }
                    cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], header.block_size, is_keyframe);
                }
            } else {
                std::fill(channel_buffer.begin(), channel_buffer.begin() + samples, 0);
                cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], header.block_size, is_keyframe);
            }
        }

        cuda_sync_all();

        const auto t_decode_end = std::chrono::high_resolution_clock::now();
        total_decode_ms += std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        Frame f = frame_create(header.width, header.height, header.channels, header.use_ycbcr != 0, header.block_size);
        uint8_t* ptrs[3] = {f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2)};
        cuda_download_planes(ptrs, f.channels);
        planes_to_rgb_parallel(f, rgb, header.use_ycbcr != 0);
        frame_destroy(f);

        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        const std::string fpath = out_dir + filename;
        stbi_write_png(fpath.c_str(), header.width, header.height, header.channels, rgb.data(),
                       header.width * header.channels);

        std::cout << "\r  Progress: " << f_idx + 1 << "/" << header.frame_count << std::flush;
        cuda_swap_recon();
    }

    const double avg_ms = total_decode_ms / header.frame_count;
    const double fps = 1000.0 / avg_ms;
    std::cout << "\n[BENCHMARK] decode_total_ms=" << total_decode_ms << " frames=" << header.frame_count
              << " avg_ms=" << avg_ms << " decode_fps=" << fps << "\n";
    cuda_free_frame_buffers();
}
