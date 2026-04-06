#include "codec.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <future>
#include <chrono>
#include <thread>

#include "cuda_kernels.cuh"
#include "rle_gpu.cuh"

#include "frame.h"
#include "image_io.h"
#include "tiling.h"
#include "dct.h"
#include "zigzag.h"
#include "quant.h"
#include "huffman.h"
#include "metrics.h"

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


#include "thread_safe_queue.h"
#include "ordered_frame_buffer.h"

struct PinnedCoeffs {
    int16_t* ptr = nullptr;
    size_t capacity = 0;
    
    void resize(size_t req) {
        if (req > capacity) {
            if (ptr) cuda_free_pinned_coeffs(ptr);
            ptr = cuda_alloc_pinned_coeffs(req);
            capacity = req;
        }
    }
    ~PinnedCoeffs() {
        if (ptr) cuda_free_pinned_coeffs(ptr);
    }
    PinnedCoeffs() = default;
    PinnedCoeffs(const PinnedCoeffs&) = delete;
    PinnedCoeffs& operator=(const PinnedCoeffs&) = delete;
    PinnedCoeffs(PinnedCoeffs&& other) noexcept {
        ptr = other.ptr; capacity = other.capacity;
        other.ptr = nullptr; other.capacity = 0;
    }
    PinnedCoeffs& operator=(PinnedCoeffs&& other) noexcept {
        if (this != &other) {
            if (ptr) cuda_free_pinned_coeffs(ptr);
            ptr = other.ptr; capacity = other.capacity;
            other.ptr = nullptr; other.capacity = 0;
        }
        return *this;
    }
};

struct EncodedChannel {
    uint32_t rle_bytes;
    uint32_t enc_len;
    uint16_t huffman_freq[256];
    std::vector<uint32_t> block_bit_lengths;
    std::vector<uint8_t> data;
};

struct EncodedFrame {
    int f_idx;
    bool is_keyframe;
    std::vector<EncodedChannel> channels;
};

void compress_flipbook(const std::string& in_dir, const std::string& out_path, int quality, int block_size, bool use_ycbcr) {
    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Input must be a valid directory containing frames.\n";
        return;
    }

    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(in_dir))
        if (entry.is_regular_file()) frames.push_back(entry.path().string());
    sort_frame_paths(frames);

    if (frames.empty()) { std::cerr << "No frames found in " << in_dir << "\n"; return; }

    int img_w = 0, img_h = 0, img_ch = 0;
    {
        std::vector<uint8_t> probe_file = read_file_bytes(frames[0]);
        std::vector<uint8_t> probe_rgb;
        if (!decode_image_file_bytes(frames[0], probe_file.data(), probe_file.size(), img_w, img_h, img_ch,
                                     probe_rgb)) {
            std::cerr << "Failed to decode first frame (probe): " << frames[0] << "\n";
            return;
        }
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::cerr << "Failed to open " << out_path << "\n"; return; }

    BinHeader header;
    header.width = img_w;   header.height = img_h;
    header.channels = img_ch; header.quality = quality;
    header.block_size = block_size;
    header.frame_count = static_cast<int32_t>(frames.size());
    header.use_ycbcr = use_ycbcr ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, quality, block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality, block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(block_size);

    cuda_alloc_frame_buffers(img_w, img_h, img_ch, luma_qm.data(), chroma_qm.data(), zigzag.data(), use_ycbcr, block_size);

    int pw[3], ph[3];
    for (int ch = 0; ch < img_ch; ++ch) {
        int w = img_w, h = img_h;
        if (use_ycbcr && ch > 0 && img_ch == 3) { w = (img_w+1)/2; h = (img_h+1)/2; }
        pw[ch] = codec_pad(w, block_size);  ph[ch] = codec_pad(h, block_size);
    }

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";
    auto t_compress_start = std::chrono::high_resolution_clock::now();

    ThreadSafeQueue<EncodedFrame> encode_queue;

    struct RawQueueJob {
        size_t frame_index = 0;
        std::string path;
        std::vector<uint8_t> file_bytes;
        bool worker_stop = false;
    };

    ThreadSafeQueue<RawQueueJob> raw_queue;
    OrderedFrameBuffer ordered_frames;
    const size_t n_frames = frames.size();
    const unsigned parse_workers = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));

    std::thread io_thread([&]() {
        bool read_ok = true;
        for (size_t i = 0; i < n_frames; ++i) {
            if (!read_ok) break;
            std::vector<uint8_t> bytes = read_file_bytes(frames[i]);
            if (bytes.empty()) {
                std::cerr << "Failed to read file: " << frames[i] << "\n";
                ordered_frames.set_fail();
                read_ok = false;
                break;
            }
            RawQueueJob job;
            job.frame_index = i;
            job.path = frames[i];
            job.file_bytes = std::move(bytes);
            raw_queue.push(std::move(job));
        }
        for (unsigned w = 0; w < parse_workers; ++w) {
            RawQueueJob stop;
            stop.worker_stop = true;
            raw_queue.push(std::move(stop));
        }
        raw_queue.finish();
    });

    auto parse_worker_body = [&]() {
        RawQueueJob job;
        while (raw_queue.pop(job)) {
            if (job.worker_stop) break;
            int w = 0, h = 0, c = 0;
            std::vector<uint8_t> interleaved;
            if (!decode_image_file_bytes(job.path, job.file_bytes.data(), job.file_bytes.size(), w, h, c,
                                         interleaved)) {
                std::cerr << "[parse] frame " << job.frame_index << ": decode failed\n";
                ordered_frames.set_fail();
                continue;
            }
            if (w != img_w || h != img_h || c != img_ch) {
                std::cerr << "[parse] frame " << job.frame_index << " size/channel mismatch\n";
                ordered_frames.set_fail();
                continue;
            }
            Frame f = rgb_to_planes_parallel(interleaved.data(), img_w, img_h, img_ch, use_ycbcr, block_size);
            ordered_frames.push(job.frame_index, std::move(f));
        }
    };

    std::vector<std::thread> parse_threads;
    parse_threads.reserve(parse_workers);
    for (unsigned w = 0; w < parse_workers; ++w)
        parse_threads.emplace_back(parse_worker_body);

    std::thread gpu_thread([&]() {
        if (n_frames == 0) {
            encode_queue.finish();
            return;
        }

        Frame f;
        if (!ordered_frames.wait_take(0, f)) {
            encode_queue.finish();
            return;
        }

        int f_idx = 0;
        bool h2d_current_already_queued = false;

        for (;;) {
            if (!h2d_current_already_queued) {
                const uint8_t* up[3] = { f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2) };
                cuda_submit_frame_h2d(f_idx, up, f.channels);
            }
            h2d_current_already_queued = false;

            const int src_slot = f_idx % 2;

            EncodedFrame ef;
            ef.f_idx = f_idx;
            ef.is_keyframe = (f_idx == 0);
            ef.channels.resize(img_ch);

            for (int ch = 0; ch < img_ch; ++ch) {
                cuda_encode_channel(ch, pw[ch], ph[ch], block_size, ef.is_keyframe, src_slot);
                if (ch + 1 == img_ch)
                    cuda_record_encode_slot_done(src_slot, ch);
                cuda_sync_channel(ch);
            }

            Frame f_next;
            const bool have_next_frame = (static_cast<size_t>(f_idx) + 1 < n_frames);
            if (have_next_frame) {
                if (!ordered_frames.wait_take(static_cast<size_t>(f_idx) + 1, f_next)) break;
                const uint8_t* upn[3] = { f_next.channel_ptr(0), f_next.channel_ptr(1), f_next.channel_ptr(2) };
                cuda_submit_frame_h2d(f_idx + 1, upn, f_next.channels);
                h2d_current_already_queued = true;
            }

            for (int ch = 0; ch < img_ch; ++ch) {
                const int num_blocks = (pw[ch] / block_size) * (ph[ch] / block_size);
                void* stream = cuda_channel_stream_ptr(ch);
                uint32_t rle_byte_len = 0;
                cuda_rle_encode_indexed(ch, cuda_channel_d_coeff(ch), num_blocks, block_size, &rle_byte_len,
                                        stream);
                cuda_sync_channel(ch);

                ef.channels[ch].rle_bytes = rle_byte_len;
                ef.channels[ch].block_bit_lengths.resize(static_cast<size_t>(num_blocks));

                if (rle_byte_len == 0) {
                    std::memset(ef.channels[ch].huffman_freq, 0, sizeof(ef.channels[ch].huffman_freq));
                    ef.channels[ch].enc_len = 0;
                    ef.channels[ch].data.clear();
                    std::fill(ef.channels[ch].block_bit_lengths.begin(), ef.channels[ch].block_bit_lengths.end(),
                              0u);
                } else {
                    uint32_t h_hist[256];
                    cuda_compute_histogram(ch, h_hist, stream);
                    cuda_sync_channel(ch);

                    uint32_t code_bits[256];
                    uint8_t code_lens[256];
                    uint16_t freq_out[256];
                    huffman_prepare_codebook(h_hist, code_bits, code_lens, freq_out);
                    std::memcpy(ef.channels[ch].huffman_freq, freq_out, sizeof(ef.channels[ch].huffman_freq));

                    uint8_t* d_pack = nullptr;
                    size_t pack_sz = 0;
                    cuda_huffman_pack_gpu_indexed(ch, num_blocks, code_bits, code_lens, &d_pack, &pack_sz,
                                                  nullptr, stream);
                    cuda_sync_channel(ch);

                    cuda_huffman_download_block_bit_lengths(ch, ef.channels[ch].block_bit_lengths.data(),
                                                            num_blocks);
                    ef.channels[ch].enc_len = static_cast<uint32_t>(pack_sz);
                    ef.channels[ch].data.resize(pack_sz);
                    if (pack_sz > 0 && d_pack)
                        cuda_memcpy_to_host(ef.channels[ch].data.data(), d_pack, pack_sz);
                }
            }

            cuda_swap_recon();
            frame_destroy(f);
            encode_queue.push(std::move(ef));

            if (!have_next_frame) break;
            f = std::move(f_next);
            ++f_idx;
        }
        encode_queue.finish();
    });

    EncodedFrame ef;
    std::future<void> write_future;

    while (encode_queue.pop(ef)) {
        struct FrameWriteData {
            struct ChData {
                uint32_t rle_bytes;
                uint32_t enc_len;
                uint16_t huffman_freq[256];
                std::vector<uint32_t> block_bit_lengths;
                std::vector<uint8_t> data;
            };
            std::vector<ChData> channels;
        };

        FrameWriteData fwd;
        fwd.channels.resize(img_ch);
        for (int ch = 0; ch < img_ch; ++ch) {
            fwd.channels[ch].rle_bytes = ef.channels[ch].rle_bytes;
            fwd.channels[ch].enc_len = ef.channels[ch].enc_len;
            std::memcpy(fwd.channels[ch].huffman_freq, ef.channels[ch].huffman_freq, 512);
            fwd.channels[ch].block_bit_lengths = std::move(ef.channels[ch].block_bit_lengths);
            fwd.channels[ch].data = std::move(ef.channels[ch].data);
        }

        if (write_future.valid()) write_future.get();
        write_future = std::async(std::launch::async, [&out, data = std::move(fwd)]() {
            for (auto& cd : data.channels) {
                out.write(reinterpret_cast<const char*>(&cd.rle_bytes), 4);
                out.write(reinterpret_cast<const char*>(&cd.enc_len), 4);
                uint32_t num_blocks = static_cast<uint32_t>(cd.block_bit_lengths.size());
                out.write(reinterpret_cast<const char*>(&num_blocks), 4);
                out.write(reinterpret_cast<const char*>(cd.huffman_freq), 512);
                out.write(reinterpret_cast<const char*>(cd.block_bit_lengths.data()), num_blocks * sizeof(uint32_t));
                if (!cd.data.empty())
                    out.write(reinterpret_cast<const char*>(cd.data.data()), cd.data.size());
            }
        });

        std::cout << "\r  Progress: " << ef.f_idx + 1 << "/" << frames.size() << std::flush;
    }

    io_thread.join();
    for (auto& t : parse_threads) t.join();
    gpu_thread.join();
    if (write_future.valid()) write_future.get();

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
    cuda_cleanup();
}


void decompress_flipbook(const std::string& in_path, const std::string& out_dir) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << in_path << "\n"; return; }

    BinHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (in.gcount() != sizeof(header) || header.magic[0] != 'F' ||
        header.magic[1] != 'L' || header.magic[2] != 'I' || header.magic[3] != '3') {
        std::cerr << "Invalid bin file: " << in_path << "\n"; return;
    }

    fs::create_directories(out_dir);

    bool use_ycbcr = (header.use_ycbcr != 0);

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, header.quality, header.block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, header.quality, header.block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(header.block_size);

    cuda_alloc_frame_buffers(header.width, header.height, header.channels,
                             luma_qm.data(), chroma_qm.data(), zigzag.data(), 
                             header.use_ycbcr != 0, header.block_size);

    int pw[3], ph[3];
    for (int ch = 0; ch < header.channels; ++ch) {
        int w = header.width, h = header.height;
        if (use_ycbcr && ch > 0 && header.channels == 3) { w = (header.width+1)/2; h = (header.height+1)/2; }
        pw[ch] = codec_pad(w, header.block_size);  ph[ch] = codec_pad(h, header.block_size);
    }

    stbi_write_png_compression_level = 1;

    constexpr int NUM_WRITE_BUFS = 4;
    const size_t rgb_size = static_cast<size_t>(header.width) * header.height * header.channels;
    std::vector<std::vector<uint8_t>> rgb_ring(NUM_WRITE_BUFS);
    for (auto& buf : rgb_ring) buf.resize(rgb_size);
    std::future<void> write_futures[NUM_WRITE_BUFS];

    std::vector<uint8_t> encoded;
    std::vector<int16_t> channel_buffer;

    std::cout << "Decompressing " << header.frame_count << " frames to " << out_dir << "...\n";

    double total_decode_ms = 0.0;

    for (int f_idx = 0; f_idx < header.frame_count; ++f_idx) {
        bool is_keyframe = (f_idx == 0);

        auto t_decode_start = std::chrono::high_resolution_clock::now();

        for (int ch = 0; ch < header.channels; ++ch) {
            uint32_t rle_bytes_len = 0, len32 = 0, num_blocks = 0;
            in.read(reinterpret_cast<char*>(&rle_bytes_len), sizeof(rle_bytes_len));
            in.read(reinterpret_cast<char*>(&len32), sizeof(len32));
            in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));

            int total_blocks = (pw[ch] / header.block_size) * (ph[ch] / header.block_size);
            int samples = total_blocks * header.block_size * header.block_size;
            if (static_cast<int>(channel_buffer.size()) < samples) channel_buffer.resize(samples);

            if (len32 > 0) {
                std::vector<uint16_t> h_freq(256);
                in.read(reinterpret_cast<char*>(h_freq.data()), 512);

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
                        cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], header.block_size, is_keyframe);
                    } else {
                        cuda_full_decode_channel(ch, encoded.data(), len32, block_bit_lengths.data(), total_blocks,
                                                 h_freq.data(), pw[ch], ph[ch], header.block_size, is_keyframe);
                        cuda_sync_channel(ch);
                    }
                } else {
                    std::vector<uint8_t> enc_with_hdr(512 + len32);
                    std::memcpy(enc_with_hdr.data(), h_freq.data(), 512);
                    std::memcpy(enc_with_hdr.data() + 512, encoded.data(), len32);

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

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        total_decode_ms += frame_ms;

        int slot = f_idx % NUM_WRITE_BUFS;

        Frame f = frame_create(header.width, header.height, header.channels, header.use_ycbcr != 0,
                               header.block_size);
        uint8_t* ptrs[3] = { f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2) };
        cuda_download_planes(ptrs, f.channels);
        planes_to_rgb_parallel(f, rgb_ring[slot], header.use_ycbcr != 0);
        frame_destroy(f);

        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        std::string fpath = out_dir + filename;
        int w = header.width, h = header.height, ch_count = header.channels;
        uint8_t* buf_ptr = rgb_ring[slot].data();

        write_futures[slot] = std::async(std::launch::async,
            [fpath, buf_ptr, w, h, ch_count]() {
                stbi_write_png(fpath.c_str(), w, h, ch_count, buf_ptr, w * ch_count);
            });

        std::cout << "\r  Progress: " << f_idx + 1 << "/" << header.frame_count << std::flush;
        cuda_swap_recon();
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
    cuda_free_frame_buffers();
}
