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
#include <cstdlib>
#include <cmath>

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

namespace {
double mean_abs_diff_u8(const uint8_t* a, const uint8_t* b, size_t n) {
    if (!a || !b || n == 0) return 0.0;
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<uint32_t>(std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    }
    return static_cast<double>(sum) / static_cast<double>(n);
}

int clamp_quality(int q) {
    return std::max(5, std::min(100, q));
}
}

#pragma pack(push, 1)
struct BinHeaderV5 {
    char magic[4] = {'F', 'L', 'I', '5'};
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t quality;
    int32_t block_size;
    int32_t frame_count;
    int32_t use_ycbcr = 1;
};
struct BinHeaderV6 {
    BinHeaderV5 base{};
    int32_t codec_flags = 0;
};
#pragma pack(pop)

constexpr int32_t kCodecFlagBlockMotionMV = 1;
constexpr int32_t kCodecFlagPerFrameMotionFlag = 2;
constexpr int32_t kCodecFlagPackedMotionPayload = 4;
constexpr int kDefaultMotionSearchRadius = 8;
constexpr double kMotionEnableMadThreshold = 2.0;

std::vector<int8_t> pack_motion_vectors(const std::vector<int8_t>& dense_mv) {
    const size_t pairs = dense_mv.size() / 2u;
    if (pairs == 0) return {};
    size_t nonzero = 0;
    for (size_t i = 0; i < pairs; ++i) {
        if (dense_mv[i * 2] != 0 || dense_mv[i * 2 + 1] != 0) ++nonzero;
    }
    if (nonzero == 0) return {};

    const size_t dense_size = 1u + dense_mv.size();
    const size_t bitmap_bytes = (pairs + 7u) / 8u;
    const size_t sparse_size = 1u + bitmap_bytes + nonzero * 2u;

    std::vector<int8_t> out;
    if (sparse_size < dense_size) {
        out.reserve(sparse_size);
        out.push_back(1); // bitmap-packed
        const size_t header = out.size();
        out.resize(header + bitmap_bytes, 0);
        for (size_t i = 0; i < pairs; ++i) {
            const int8_t mx = dense_mv[i * 2];
            const int8_t my = dense_mv[i * 2 + 1];
            if (mx != 0 || my != 0) {
                out[header + (i >> 3)] = static_cast<int8_t>(out[header + (i >> 3)] | (1 << (i & 7)));
                out.push_back(mx);
                out.push_back(my);
            }
        }
    } else {
        out.reserve(dense_size);
        out.push_back(0); // dense
        out.insert(out.end(), dense_mv.begin(), dense_mv.end());
    }
    return out;
}

bool unpack_motion_vectors(const std::vector<int8_t>& payload, int num_pairs, std::vector<int8_t>& out_dense) {
    out_dense.assign(static_cast<size_t>(num_pairs) * 2u, 0);
    if (payload.empty() || num_pairs <= 0) return false;
    const uint8_t mode = static_cast<uint8_t>(payload[0]);
    if (mode == 0) {
        const size_t need = static_cast<size_t>(num_pairs) * 2u;
        if (payload.size() < 1u + need) return false;
        std::memcpy(out_dense.data(), payload.data() + 1, need);
        return true;
    }
    if (mode == 1) {
        const size_t bitmap_bytes = (static_cast<size_t>(num_pairs) + 7u) / 8u;
        if (payload.size() < 1u + bitmap_bytes) return false;
        const int8_t* bits = payload.data() + 1;
        const int8_t* vals = bits + bitmap_bytes;
        const int8_t* end = payload.data() + payload.size();
        for (int i = 0; i < num_pairs; ++i) {
            const bool nz = (static_cast<uint8_t>(bits[i >> 3]) >> (i & 7)) & 1u;
            if (nz) {
                if (vals + 1 >= end) return false;
                out_dense[static_cast<size_t>(i) * 2u] = *vals++;
                out_dense[static_cast<size_t>(i) * 2u + 1u] = *vals++;
            }
        }
        return true;
    }
    return false;
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
    uint32_t huffman_freq[256];
    std::vector<uint32_t> block_bit_lengths;
    std::vector<uint8_t> data;
};

struct EncodedFrame {
    int f_idx;
    bool is_keyframe;
    int frame_quality = 50;
    std::vector<EncodedChannel> channels;
    int heat_blocks_x = 0;
    int heat_blocks_y = 0;
    std::vector<float> heat_qscale_map;
    bool use_motion_vectors = false;
    std::vector<int8_t> mv_payload;
};

static size_t encoded_frame_size_bytes(const EncodedFrame& ef, bool motion_predict) {
    size_t total = 2; // per-frame keyframe marker + per-frame quality
    if (motion_predict) total += 1; // per-frame motion flag
    if (motion_predict && !ef.is_keyframe && !ef.mv_payload.empty())
        total += ef.mv_payload.size();
    for (const auto& ch : ef.channels) {
        total += 4 + 4 + 4;               // rle_bytes + enc_len + num_blocks
        total += 256 * sizeof(uint32_t);  // huffman_freq
        total += ch.block_bit_lengths.size() * sizeof(uint32_t);
        total += ch.data.size();
    }
    return total;
}

void compress_flipbook(const std::string& in_dir,
                       const std::string& out_path,
                       int quality,
                       int block_size,
                       bool use_ycbcr,
                       bool adaptive_roi,
                       float roi_strength,
                       const std::string& heatmap_video_path,
                       float target_size_mb,
                       float scene_cut_threshold,
                       bool motion_predict,
                       int motion_search_radius) {
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

    const bool use_motion_bin = motion_predict;
    if (use_motion_bin) {
        BinHeaderV6 header{};
        header.base.magic[0] = 'F';
        header.base.magic[1] = 'L';
        header.base.magic[2] = 'I';
        header.base.magic[3] = '6';
        header.base.width = img_w;
        header.base.height = img_h;
        header.base.channels = img_ch;
        header.base.quality = quality;
        header.base.block_size = block_size;
        header.base.frame_count = static_cast<int32_t>(frames.size());
        header.base.use_ycbcr = use_ycbcr ? 1 : 0;
        header.codec_flags = kCodecFlagBlockMotionMV | kCodecFlagPerFrameMotionFlag | kCodecFlagPackedMotionPayload;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    } else {
        BinHeaderV5 header{};
        header.magic[0] = 'F';
        header.magic[1] = 'L';
        header.magic[2] = 'I';
        header.magic[3] = '5';
        header.width = img_w;
        header.height = img_h;
        header.channels = img_ch;
        header.quality = quality;
        header.block_size = block_size;
        header.frame_count = static_cast<int32_t>(frames.size());
        header.use_ycbcr = use_ycbcr ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    const bool rc_enabled = target_size_mb > 0.0f;
    const size_t target_size_bytes = rc_enabled
        ? static_cast<size_t>(target_size_mb * 1024.0f * 1024.0f)
        : 0u;
    const size_t header_bytes_out = use_motion_bin ? sizeof(BinHeaderV6) : sizeof(BinHeaderV5);
    size_t encoded_bytes_so_far = header_bytes_out;
    int rc_quality = clamp_quality(quality);

    const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, quality, block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality, block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(block_size);

    cuda_alloc_frame_buffers(img_w, img_h, img_ch, luma_qm.data(), chroma_qm.data(), zigzag.data(), use_ycbcr, block_size);
    cuda_set_adaptive_roi(adaptive_roi, roi_strength);
    cuda_set_motion_predict(use_motion_bin, std::max(1, std::min(32, motion_search_radius)));

    int pw[3], ph[3];
    for (int ch = 0; ch < img_ch; ++ch) {
        int w = img_w, h = img_h;
        if (use_ycbcr && ch > 0 && img_ch == 3) { w = (img_w+1)/2; h = (img_h+1)/2; }
        pw[ch] = codec_pad(w, block_size);  ph[ch] = codec_pad(h, block_size);
    }

    std::cout << "Compressing " << frames.size() << " frames into " << out_path << "...\n";
    auto t_compress_start = std::chrono::high_resolution_clock::now();
    uint64_t total_pixels = static_cast<uint64_t>(img_w) * img_h * img_ch * frames.size();

    cudaEvent_t ev_start[3], ev_stop[3];
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaEventCreate(&ev_start[i]));
        CUDA_CHECK(cudaEventCreate(&ev_stop[i]));
    }
    float ms_stage[3] = {0.0f, 0.0f, 0.0f};

    ThreadSafeQueue<EncodedFrame> encode_queue;
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
        std::vector<uint8_t> prev_input_luma;
        int applied_quality = clamp_quality(quality);

        for (;;) {
            if (!h2d_current_already_queued) {
                const uint8_t* up[3] = { f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2) };
                cuda_submit_frame_h2d(f_idx, up, f.channels);
            }
            h2d_current_already_queued = false;

            const int src_slot = f_idx % 2;

            EncodedFrame ef;
            ef.f_idx = f_idx;
            const int luma_pixels = pw[0] * ph[0];
            const uint8_t* curr_luma = f.channel_ptr(0);
            const double luma_mad = (f_idx > 0 && !prev_input_luma.empty())
                ? mean_abs_diff_u8(curr_luma, prev_input_luma.data(), static_cast<size_t>(luma_pixels))
                : 0.0;
            const bool scene_cut = (f_idx > 0 && scene_cut_threshold > 0.0f && !prev_input_luma.empty() &&
                luma_mad >= scene_cut_threshold);
            ef.is_keyframe = (f_idx == 0) || scene_cut;
            ef.use_motion_vectors = use_motion_bin && !ef.is_keyframe && (luma_mad >= kMotionEnableMadThreshold);
            ef.channels.resize(img_ch);
            if (emit_heatmap && img_ch > 0) {
                ef.heat_blocks_x = pw[0] / block_size;
                ef.heat_blocks_y = ph[0] / block_size;
                ef.heat_qscale_map.resize(static_cast<size_t>(ef.heat_blocks_x * ef.heat_blocks_y));
            }

            if (rc_enabled && rc_quality != applied_quality) {
                const QuantMatrix luma_qm_dyn = make_quant_matrix(kJpegLumaQuant, rc_quality, block_size);
                const QuantMatrix chroma_qm_dyn = make_quant_matrix(kJpegChromaQuant, rc_quality, block_size);
                cuda_update_quant_matrices(luma_qm_dyn.data(), chroma_qm_dyn.data(), block_size);
                applied_quality = rc_quality;
            }
            ef.frame_quality = applied_quality;

            cuda_prepare_encode_prediction(src_slot, ef.is_keyframe, ef.use_motion_vectors, use_ycbcr, img_ch);

            CUDA_CHECK(cudaEventRecord(ev_start[0], 0));
            for (int ch = 0; ch < img_ch; ++ch) {
                cuda_encode_channel(ch, pw[ch], ph[ch], block_size, ef.is_keyframe, src_slot);
                if (ch + 1 == img_ch)
                    cuda_record_encode_slot_done(src_slot, ch);
            }
            CUDA_CHECK(cudaEventRecord(ev_stop[0], 0));

            Frame f_next;
            const bool have_next_frame = (static_cast<size_t>(f_idx) + 1 < n_frames);
            if (have_next_frame) {
                if (!ordered_frames.wait_take(static_cast<size_t>(f_idx) + 1, f_next)) break;
                const uint8_t* upn[3] = { f_next.channel_ptr(0), f_next.channel_ptr(1), f_next.channel_ptr(2) };
                cuda_submit_frame_h2d(f_idx + 1, upn, f_next.channels);
                h2d_current_already_queued = true;
            }

            CUDA_CHECK(cudaEventRecord(ev_start[1], 0));
            for (int ch = 0; ch < img_ch; ++ch) {
                const int num_blocks = (pw[ch] / block_size) * (ph[ch] / block_size);
                void* stream = cuda_channel_stream_ptr(ch);
                
                cuda_rle_encode_async(ch, num_blocks, block_size);
                cuda_compute_histogram(ch, nullptr, stream);
                cuda_prepare_huffman_codebook_gpu(ch, stream);
                cuda_pack_channel_indexed(ch, num_blocks, block_size, nullptr, nullptr, nullptr, nullptr, nullptr);
            }
            CUDA_CHECK(cudaEventRecord(ev_stop[1], 0));

            cuda_sync_all();

            float t_ms = 0;
            cudaEventElapsedTime(&t_ms, ev_start[0], ev_stop[0]); ms_stage[0] += t_ms;
            cudaEventElapsedTime(&t_ms, ev_start[1], ev_stop[1]); ms_stage[1] += t_ms;

            for (int ch = 0; ch < img_ch; ++ch) {
                const int num_blocks = (pw[ch] / block_size) * (ph[ch] / block_size);
                uint32_t rle_byte_len = 0;
                uint32_t pack_len = 0;
                
                cuda_get_pinned_metadata(ch, &rle_byte_len, &pack_len);
                ef.channels[ch].rle_bytes = rle_byte_len;
                ef.channels[ch].enc_len = pack_len;
                ef.channels[ch].block_bit_lengths.resize(static_cast<size_t>(num_blocks));
                
                if (rle_byte_len > 0) {
                    cuda_huffman_download_block_bit_lengths(ch, ef.channels[ch].block_bit_lengths.data(), num_blocks);
                    ef.channels[ch].data.resize(pack_len);
                    uint8_t* d_pack = cuda_get_bitstream_ptr(ch);
                    if (pack_len > 0 && d_pack) {
                        cuda_memcpy_to_host(ef.channels[ch].data.data(), d_pack, pack_len);
                    }
                    cuda_compute_histogram(ch, (uint32_t*)ef.channels[ch].huffman_freq, nullptr);
                } else {
                    std::memset(ef.channels[ch].huffman_freq, 0, sizeof(ef.channels[ch].huffman_freq));
                    ef.channels[ch].data.clear();
                    std::fill(ef.channels[ch].block_bit_lengths.begin(), ef.channels[ch].block_bit_lengths.end(), 0u);
                }
            }
            if (emit_heatmap && !ef.heat_qscale_map.empty()) {
                cuda_download_qscale_map(0, ef.heat_qscale_map.data(),
                                         static_cast<int>(ef.heat_qscale_map.size()));
            }

            if (use_motion_bin && ef.use_motion_vectors && !ef.is_keyframe) {
                const int nb_mv = (pw[0] / block_size) * (ph[0] / block_size);
                std::vector<int8_t> mv_dense(static_cast<size_t>(nb_mv) * 2u);
                cuda_download_mv_luma(mv_dense.data(), nb_mv);
                ef.mv_payload = pack_motion_vectors(mv_dense);
                if (ef.mv_payload.empty()) {
                    ef.use_motion_vectors = false;
                }
            }

            cuda_swap_recon();
            prev_input_luma.assign(curr_luma, curr_luma + static_cast<size_t>(luma_pixels));

            if (rc_enabled) {
                const size_t frame_bytes = encoded_frame_size_bytes(ef, use_motion_bin);
                const size_t frames_done = static_cast<size_t>(f_idx) + 1;
                const size_t frames_left = n_frames - frames_done;
                const size_t remain_before = (target_size_bytes > encoded_bytes_so_far)
                    ? (target_size_bytes - encoded_bytes_so_far) : 1u;
                const double budget_this = static_cast<double>(remain_before) /
                                           static_cast<double>(std::max<size_t>(1, n_frames - static_cast<size_t>(f_idx)));
                encoded_bytes_so_far += frame_bytes;
                if (frames_left > 0) {
                    const double rel_err = (budget_this > 1.0) ? (static_cast<double>(frame_bytes) / budget_this - 1.0) : 0.0;
                    int step = static_cast<int>(std::round(rel_err * 10.0));
                    if (step > 0) {
                        rc_quality = clamp_quality(rc_quality - std::min(8, step));
                    } else if (step < 0) {
                        rc_quality = clamp_quality(rc_quality + std::min(4, -step));
                    }
                }
            }

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
            uint8_t is_keyframe = 0;
            uint8_t frame_quality = 50;
            bool write_mv_payload = false;
            uint8_t motion_flag = 0;
            std::vector<int8_t> mv_payload;
            struct ChData {
                uint32_t rle_bytes;
                uint32_t enc_len;
                uint32_t huffman_freq[256];
                std::vector<uint32_t> block_bit_lengths;
                std::vector<uint8_t> data;
            };
            std::vector<ChData> channels;
        };

        FrameWriteData fwd;
        fwd.channels.resize(img_ch);
        fwd.is_keyframe = ef.is_keyframe ? 1u : 0u;
        fwd.frame_quality = static_cast<uint8_t>(clamp_quality(ef.frame_quality));
        fwd.write_mv_payload = use_motion_bin;
        fwd.motion_flag = ef.use_motion_vectors ? 1u : 0u;
        if (use_motion_bin && ef.use_motion_vectors && !ef.is_keyframe)
            fwd.mv_payload = std::move(ef.mv_payload);
        for (int ch = 0; ch < img_ch; ++ch) {
            fwd.channels[ch].rle_bytes = ef.channels[ch].rle_bytes;
            fwd.channels[ch].enc_len = ef.channels[ch].enc_len;
            std::memcpy(fwd.channels[ch].huffman_freq, ef.channels[ch].huffman_freq, sizeof(fwd.channels[ch].huffman_freq));
            fwd.channels[ch].block_bit_lengths = std::move(ef.channels[ch].block_bit_lengths);
            fwd.channels[ch].data = std::move(ef.channels[ch].data);
        }

        if (write_future.valid()) write_future.get();
        write_future = std::async(std::launch::async, [&out, data = std::move(fwd)]() {
            out.write(reinterpret_cast<const char*>(&data.is_keyframe), 1);
            out.write(reinterpret_cast<const char*>(&data.frame_quality), 1);
            if (data.write_mv_payload) {
                out.write(reinterpret_cast<const char*>(&data.motion_flag), 1);
            }
            if (data.write_mv_payload && data.motion_flag && !data.is_keyframe && !data.mv_payload.empty()) {
                out.write(reinterpret_cast<const char*>(data.mv_payload.data()),
                          static_cast<std::streamsize>(data.mv_payload.size() * sizeof(int8_t)));
            }
            for (auto& cd : data.channels) {
                out.write(reinterpret_cast<const char*>(&cd.rle_bytes), 4);
                out.write(reinterpret_cast<const char*>(&cd.enc_len), 4);
                uint32_t num_blocks = static_cast<uint32_t>(cd.block_bit_lengths.size());
                out.write(reinterpret_cast<const char*>(&num_blocks), 4);
                out.write(reinterpret_cast<const char*>(cd.huffman_freq), sizeof(cd.huffman_freq));
                out.write(reinterpret_cast<const char*>(cd.block_bit_lengths.data()), num_blocks * sizeof(uint32_t));
                if (!cd.data.empty())
                    out.write(reinterpret_cast<const char*>(cd.data.data()), cd.data.size());
            }
        });
        if (emit_heatmap && !ef.heat_qscale_map.empty()) {
            const fs::path map_name = "frame_" + std::to_string(ef.f_idx) + ".bin";
            const fs::path map_path = fs::path(heatmap_data_dir) / map_name;
            std::ofstream map_out(map_path, std::ios::binary);
            if (map_out) {
                map_out.write(reinterpret_cast<const char*>(ef.heat_qscale_map.data()),
                              static_cast<std::streamsize>(ef.heat_qscale_map.size() * sizeof(float)));
                heatmap_manifest << ef.f_idx << "\t" << ef.heat_blocks_x << "\t" << ef.heat_blocks_y
                                << "\t" << map_name.string() << "\t" << frames[ef.f_idx] << "\n";
            }
        }

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
    
    double dct_gb = (total_pixels * 5.0) / 1e9;
    double dct_bw = dct_gb / (ms_stage[0] / 1000.0);
    
    double rle_gb = (total_pixels * 2.5) / 1e9;
    double rle_bw = rle_gb / (ms_stage[1] / 1000.0);

    for (int i = 0; i < 3; ++i) {
        cudaEventDestroy(ev_start[i]);
        cudaEventDestroy(ev_stop[i]);
    }

    std::cout << "  [PERFORMANCE] DCT_BW=" << dct_bw << " GB/s, RLE_Huffman_BW=" << rle_bw << " GB/s\n";
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
    cuda_cleanup();
}


void decompress_flipbook(const std::string& in_path, const std::string& out_dir) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << in_path << "\n"; return; }

    BinHeaderV5 h5{};
    in.read(reinterpret_cast<char*>(&h5), sizeof(h5));
    const bool valid_magic = (in.gcount() == sizeof(h5) &&
        h5.magic[0] == 'F' && h5.magic[1] == 'L' && h5.magic[2] == 'I' &&
        (h5.magic[3] == '3' || h5.magic[3] == '4' || h5.magic[3] == '5' || h5.magic[3] == '6'));
    if (!valid_magic) {
        std::cerr << "Invalid bin file: " << in_path << "\n"; return;
    }
    int32_t codec_flags = 0;
    if (h5.magic[3] == '6') {
        in.read(reinterpret_cast<char*>(&codec_flags), sizeof(codec_flags));
        if (!in || in.gcount() != static_cast<std::streamsize>(sizeof(codec_flags))) {
            std::cerr << "Invalid FLI6 header (codec_flags): " << in_path << "\n";
            return;
        }
    }
    const bool has_frame_keyflags = (h5.magic[3] == '4');
    const bool has_frame_quality = (h5.magic[3] == '5' || h5.magic[3] == '6');
    const bool motion_decode = (h5.magic[3] == '6') && ((codec_flags & kCodecFlagBlockMotionMV) != 0);
    const bool has_per_frame_motion_flag = (h5.magic[3] == '6') && ((codec_flags & kCodecFlagPerFrameMotionFlag) != 0);
    const bool has_packed_motion_payload = (h5.magic[3] == '6') && ((codec_flags & kCodecFlagPackedMotionPayload) != 0);

    fs::create_directories(out_dir);

    bool use_ycbcr = (h5.use_ycbcr != 0);

    const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, h5.quality, h5.block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, h5.quality, h5.block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(h5.block_size);

    cuda_alloc_frame_buffers(h5.width, h5.height, h5.channels,
                             luma_qm.data(), chroma_qm.data(), zigzag.data(),
                             h5.use_ycbcr != 0, h5.block_size);
    cuda_set_motion_predict(motion_decode, kDefaultMotionSearchRadius);

    int pw[3], ph[3];
    for (int ch = 0; ch < h5.channels; ++ch) {
        int w = h5.width, h = h5.height;
        if (use_ycbcr && ch > 0 && h5.channels == 3) { w = (h5.width+1)/2; h = (h5.height+1)/2; }
        pw[ch] = codec_pad(w, h5.block_size);  ph[ch] = codec_pad(h, h5.block_size);
    }

    const int mv_luma_blocks = (pw[0] / h5.block_size) * (ph[0] / h5.block_size);
    const size_t mv_payload_bytes = static_cast<size_t>(mv_luma_blocks) * 2u * sizeof(int8_t);

    stbi_write_png_compression_level = 1;

    constexpr int NUM_WRITE_BUFS = 4;
    const size_t rgb_size = static_cast<size_t>(h5.width) * h5.height * h5.channels;
    std::vector<std::vector<uint8_t>> rgb_ring(NUM_WRITE_BUFS);
    for (auto& buf : rgb_ring) buf.resize(rgb_size);
    std::future<void> write_futures[NUM_WRITE_BUFS];

    std::vector<uint8_t> encoded;
    std::vector<int16_t> channel_buffer;

    std::cout << "Decompressing " << h5.frame_count << " frames to " << out_dir << "...\n";

    double total_decode_ms = 0.0;

    std::vector<int8_t> mv_read;

    for (int f_idx = 0; f_idx < h5.frame_count; ++f_idx) {
        bool is_keyframe = (f_idx == 0);
        int frame_quality = h5.quality;
        if (has_frame_keyflags || has_frame_quality) {
            uint8_t keyflag = 0;
            in.read(reinterpret_cast<char*>(&keyflag), 1);
            if (!in) {
                std::cerr << "Corrupt frame keyflag at frame " << f_idx << "\n";
                break;
            }
            is_keyframe = (keyflag != 0);
        }
        if (has_frame_quality) {
            uint8_t q = 0;
            in.read(reinterpret_cast<char*>(&q), 1);
            if (!in) {
                std::cerr << "Corrupt frame quality at frame " << f_idx << "\n";
                break;
            }
            frame_quality = clamp_quality(static_cast<int>(q));
        }

        bool use_motion_for_frame = motion_decode && !is_keyframe;
        if (motion_decode && has_per_frame_motion_flag) {
            uint8_t mflag = 0;
            in.read(reinterpret_cast<char*>(&mflag), 1);
            if (!in) {
                std::cerr << "Corrupt frame motion flag at frame " << f_idx << "\n";
                break;
            }
            use_motion_for_frame = (mflag != 0) && !is_keyframe;
        }

        if (use_motion_for_frame) {
            if (has_packed_motion_payload) {
                uint8_t mode = 0;
                in.read(reinterpret_cast<char*>(&mode), 1);
                if (!in) {
                    std::cerr << "Corrupt packed motion mode at frame " << f_idx << "\n";
                    break;
                }
                if (mode == 0) {
                    std::vector<int8_t> payload(1 + mv_luma_blocks * 2);
                    payload[0] = static_cast<int8_t>(mode);
                    in.read(reinterpret_cast<char*>(payload.data() + 1), static_cast<std::streamsize>(mv_payload_bytes));
                    if (!in || static_cast<size_t>(in.gcount()) != mv_payload_bytes) {
                        std::cerr << "Corrupt dense motion payload at frame " << f_idx << "\n";
                        break;
                    }
                    if (!unpack_motion_vectors(payload, mv_luma_blocks, mv_read)) {
                        std::cerr << "Failed to unpack dense motion payload at frame " << f_idx << "\n";
                        break;
                    }
                } else if (mode == 1) {
                    const size_t bitmap_bytes = (static_cast<size_t>(mv_luma_blocks) + 7u) / 8u;
                    std::vector<int8_t> payload(1 + bitmap_bytes);
                    payload[0] = static_cast<int8_t>(mode);
                    in.read(reinterpret_cast<char*>(payload.data() + 1), static_cast<std::streamsize>(bitmap_bytes));
                    if (!in || static_cast<size_t>(in.gcount()) != bitmap_bytes) {
                        std::cerr << "Corrupt sparse motion bitmap at frame " << f_idx << "\n";
                        break;
                    }
                    size_t nonzero = 0;
                    for (int i = 0; i < mv_luma_blocks; ++i) {
                        const uint8_t b = static_cast<uint8_t>(payload[1 + (i >> 3)]);
                        if ((b >> (i & 7)) & 1u) ++nonzero;
                    }
                    const size_t vals_bytes = nonzero * 2u;
                    const size_t old_size = payload.size();
                    payload.resize(old_size + vals_bytes);
                    in.read(reinterpret_cast<char*>(payload.data() + old_size), static_cast<std::streamsize>(vals_bytes));
                    if (!in || static_cast<size_t>(in.gcount()) != vals_bytes) {
                        std::cerr << "Corrupt sparse motion values at frame " << f_idx << "\n";
                        break;
                    }
                    if (!unpack_motion_vectors(payload, mv_luma_blocks, mv_read)) {
                        std::cerr << "Failed to unpack sparse motion payload at frame " << f_idx << "\n";
                        break;
                    }
                } else {
                    std::cerr << "Unknown motion payload mode at frame " << f_idx << "\n";
                    break;
                }
            } else {
                mv_read.resize(mv_payload_bytes / sizeof(int8_t));
                in.read(reinterpret_cast<char*>(mv_read.data()), static_cast<std::streamsize>(mv_payload_bytes));
                if (!in || static_cast<size_t>(in.gcount()) != mv_payload_bytes) {
                    std::cerr << "Corrupt motion-vector payload at frame " << f_idx << "\n";
                    break;
                }
            }
        } else {
            mv_read.clear();
        }

        cuda_prepare_decode_prediction(is_keyframe || !use_motion_for_frame,
                                       mv_read.empty() ? nullptr : mv_read.data(),
                                       mv_read.size() * sizeof(int8_t),
                                       use_ycbcr, h5.channels);

        const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, frame_quality, h5.block_size);
        const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, frame_quality, h5.block_size);
        cuda_update_quant_matrices(luma_qm.data(), chroma_qm.data(), h5.block_size);

        auto t_decode_start = std::chrono::high_resolution_clock::now();

        for (int ch = 0; ch < h5.channels; ++ch) {
            uint32_t rle_bytes_len = 0, len32 = 0, num_blocks = 0;
            in.read(reinterpret_cast<char*>(&rle_bytes_len), sizeof(rle_bytes_len));
            in.read(reinterpret_cast<char*>(&len32), sizeof(len32));
            in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));

            int total_blocks = (pw[ch] / h5.block_size) * (ph[ch] / h5.block_size);
            int samples = total_blocks * h5.block_size * h5.block_size;
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
                        cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], h5.block_size, is_keyframe);
                    } else {
                        cuda_full_decode_channel(ch, encoded.data(), len32, block_bit_lengths.data(), total_blocks,
                                                 h_freq.data(), pw[ch], ph[ch], h5.block_size, is_keyframe);
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

                    cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], h5.block_size, is_keyframe);
                }
            } else {
                std::fill(channel_buffer.begin(), channel_buffer.begin() + samples, 0);
                cuda_decode_channel(ch, channel_buffer.data(), pw[ch], ph[ch], h5.block_size, is_keyframe);
            }
        }

        cuda_sync_all();

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        total_decode_ms += frame_ms;

        int slot = f_idx % NUM_WRITE_BUFS;

        Frame f = frame_create(h5.width, h5.height, h5.channels, h5.use_ycbcr != 0,
                               h5.block_size);
        uint8_t* ptrs[3] = { f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2) };
        cuda_download_planes(ptrs, f.channels);
        planes_to_rgb_parallel(f, rgb_ring[slot], h5.use_ycbcr != 0);
        frame_destroy(f);

        char filename[256];
        std::snprintf(filename, sizeof(filename), "/frame_%04d.png", f_idx);
        std::string fpath = out_dir + filename;
        int w = h5.width, h = h5.height, ch_count = h5.channels;
        uint8_t* buf_ptr = rgb_ring[slot].data();

        write_futures[slot] = std::async(std::launch::async,
            [fpath, buf_ptr, w, h, ch_count]() {
                stbi_write_png(fpath.c_str(), w, h, ch_count, buf_ptr, w * ch_count);
            });

        std::cout << "\r  Progress: " << f_idx + 1 << "/" << h5.frame_count << std::flush;
        cuda_swap_recon();
    }

    for (int i = 0; i < NUM_WRITE_BUFS; ++i)
        if (write_futures[i].valid()) write_futures[i].get();

    double avg_ms = total_decode_ms / h5.frame_count;
    double fps = 1000.0 / avg_ms;
    std::cout << "\n  Finished decompressing flipbook.\n";
    std::cout << "  [BENCHMARK] decode_total_ms=" << total_decode_ms
              << " frames=" << h5.frame_count
              << " avg_ms=" << avg_ms
              << " decode_fps=" << fps << "\n";
    cuda_free_frame_buffers();
}
