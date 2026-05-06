#include "realtime/in_memory_codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "cuda_kernels.cuh"
#include "frame.h"
#include "huffman.h"
#include "image_io.h"
#include "quant.h"
#include "rle_gpu.cuh"
#include "zigzag.h"

namespace {
inline int pad_to_bs(int n, int bs) { return ((n + bs - 1) / bs) * bs; }

inline int clamp_quality(int q) {
    return std::max(5, std::min(100, q));
}

double mean_abs_diff_u8(const uint8_t* a, const uint8_t* b, size_t n) {
    if (!a || !b || n == 0) return 0.0;
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<uint32_t>(std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    }
    return static_cast<double>(sum) / static_cast<double>(n);
}

void rle_decode_zeros_to_coeffs(const int16_t* rle, int rle_count, int16_t* coeffs, int coeff_count) {
    int in_idx = 0;
    int out_idx = 0;
    while (in_idx < rle_count && out_idx < coeff_count) {
        if (rle[in_idx] == 0) {
            if (in_idx + 1 >= rle_count) break;
            const int run = static_cast<int>(rle[in_idx + 1]);
            for (int k = 0; k < run && out_idx < coeff_count; ++k) coeffs[out_idx++] = 0;
            in_idx += 2;
        } else {
            coeffs[out_idx++] = rle[in_idx++];
        }
    }
    while (out_idx < coeff_count) coeffs[out_idx++] = 0;
}
}

CudaRealtimeEncoder::CudaRealtimeEncoder() = default;

CudaRealtimeEncoder::~CudaRealtimeEncoder() {
    reset();
}

bool CudaRealtimeEncoder::init(int width, int height, int channels, const RealtimeEncodeParams& params) {
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 3) return false;
    if (params.block_size != 8 && params.block_size != 16 && params.block_size != 32) return false;

    reset();

    width_ = width;
    height_ = height;
    channels_ = channels;
    params_ = params;
    current_quality_ = clamp_quality(params.quality);
    frame_index_ = 0;
    prev_input_luma_.clear();

    cuda_init();
    const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, current_quality_, params_.block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, current_quality_, params_.block_size);
    const std::vector<int> zigzag = codec_zigzag_scan_table(params_.block_size);
    cuda_alloc_frame_buffers(width_, height_, channels_, luma_qm.data(), chroma_qm.data(),
                             zigzag.data(), params_.use_ycbcr, params_.block_size);
    cuda_set_adaptive_roi(params_.adaptive_roi, params_.roi_strength);
    initialized_ = true;
    return true;
}

void CudaRealtimeEncoder::reset() {
    if (initialized_) {
        cuda_cleanup();
    }
    initialized_ = false;
    width_ = height_ = channels_ = 0;
    frame_index_ = 0;
    current_quality_ = 50;
    prev_input_luma_.clear();
}

bool CudaRealtimeEncoder::encode_frame(const uint8_t* interleaved, size_t bytes, RealtimeEncodedFrame& out_frame) {
    if (!initialized_ || !interleaved) return false;
    const size_t expected = static_cast<size_t>(width_) * static_cast<size_t>(height_) * static_cast<size_t>(channels_);
    if (bytes < expected) return false;

    std::vector<uint8_t> frame_copy(interleaved, interleaved + expected);
    Frame f = rgb_to_planes_parallel(frame_copy.data(), width_, height_, channels_, params_.use_ycbcr, params_.block_size);

    const int luma_pixels = f.padded_width[0] * f.padded_height[0];
    const uint8_t* curr_luma = f.channel_ptr(0);
    const bool scene_cut = (frame_index_ > 0 && params_.scene_cut_threshold > 0.0f && !prev_input_luma_.empty() &&
        mean_abs_diff_u8(curr_luma, prev_input_luma_.data(), static_cast<size_t>(luma_pixels)) >= params_.scene_cut_threshold);

    const int requested_q = clamp_quality(params_.quality);
    const bool quality_changed = (requested_q != current_quality_);
    if (quality_changed) {
        const QuantMatrix luma_qm_dyn = make_quant_matrix(kJpegLumaQuant, requested_q, params_.block_size);
        const QuantMatrix chroma_qm_dyn = make_quant_matrix(kJpegChromaQuant, requested_q, params_.block_size);
        cuda_update_quant_matrices(luma_qm_dyn.data(), chroma_qm_dyn.data(), params_.block_size);
        current_quality_ = requested_q;
    }
    const bool periodic_keyframe = (frame_index_ % 30) == 0;
    const bool is_keyframe = (frame_index_ == 0) || scene_cut || quality_changed || periodic_keyframe;

    const uint8_t* up[3] = { f.channel_ptr(0), f.channel_ptr(1), f.channel_ptr(2) };
    cuda_submit_frame_h2d(frame_index_, up, channels_);

    int pw[3] = {0}, ph[3] = {0};
    out_frame = {};
    out_frame.width = static_cast<uint32_t>(width_);
    out_frame.height = static_cast<uint32_t>(height_);
    out_frame.channels = static_cast<uint32_t>(channels_);
    out_frame.block_size = static_cast<uint32_t>(params_.block_size);
    out_frame.frame_index = static_cast<uint32_t>(frame_index_);
    out_frame.is_keyframe = is_keyframe ? 1u : 0u;
    out_frame.use_ycbcr = params_.use_ycbcr ? 1u : 0u;
    out_frame.quality = static_cast<uint8_t>(current_quality_);
    out_frame.channel_data.resize(static_cast<size_t>(channels_));

    const int src_slot = frame_index_ % 2;
    for (int ch = 0; ch < channels_; ++ch) {
        int w = width_, h = height_;
        if (params_.use_ycbcr && ch > 0 && channels_ == 3) { w = (width_ + 1) / 2; h = (height_ + 1) / 2; }
        pw[ch] = pad_to_bs(w, params_.block_size);
        ph[ch] = pad_to_bs(h, params_.block_size);
    }

    for (int ch = 0; ch < channels_; ++ch) {
        cuda_encode_channel(ch, pw[ch], ph[ch], params_.block_size, is_keyframe, src_slot);
        if (ch + 1 == channels_) cuda_record_encode_slot_done(src_slot, ch);
    }
    for (int ch = 0; ch < channels_; ++ch) {
        const int num_blocks = (pw[ch] / params_.block_size) * (ph[ch] / params_.block_size);
        void* stream = cuda_channel_stream_ptr(ch);
        cuda_rle_encode_async(ch, num_blocks, params_.block_size);
        cuda_compute_histogram(ch, nullptr, stream);
        cuda_prepare_huffman_codebook_gpu(ch, stream);
        cuda_pack_channel_indexed(ch, num_blocks, params_.block_size, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    cuda_sync_all();

    for (int ch = 0; ch < channels_; ++ch) {
        const int num_blocks = (pw[ch] / params_.block_size) * (ph[ch] / params_.block_size);
        uint32_t rle_byte_len = 0, pack_len = 0;
        cuda_get_pinned_metadata(ch, &rle_byte_len, &pack_len);
        auto& dst = out_frame.channel_data[static_cast<size_t>(ch)];
        dst.rle_bytes = rle_byte_len;
        dst.enc_len = pack_len;
        dst.block_bit_lengths.resize(static_cast<size_t>(num_blocks), 0u);

        if (rle_byte_len > 0) {
            cuda_huffman_download_block_bit_lengths(ch, dst.block_bit_lengths.data(), num_blocks);
            dst.data.resize(pack_len);
            uint8_t* d_pack = cuda_get_bitstream_ptr(ch);
            if (pack_len > 0 && d_pack) cuda_memcpy_to_host(dst.data.data(), d_pack, pack_len);
            cuda_compute_histogram(ch, dst.huffman_freq.data(), nullptr);
        } else {
            dst.data.clear();
        }
    }

    prev_input_luma_.assign(curr_luma, curr_luma + static_cast<size_t>(luma_pixels));
    cuda_swap_recon();
    frame_destroy(f);
    ++frame_index_;
    return true;
}

void CudaRealtimeEncoder::set_quality(int quality) {
    params_.quality = clamp_quality(quality);
}

CudaRealtimeDecoder::CudaRealtimeDecoder() = default;

CudaRealtimeDecoder::~CudaRealtimeDecoder() {
    reset();
}

bool CudaRealtimeDecoder::init(int width, int height, int channels, int block_size, bool use_ycbcr, int quality) {
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 3) return false;
    if (block_size != 8 && block_size != 16 && block_size != 32) return false;

    reset();
    width_ = width;
    height_ = height;
    channels_ = channels;
    block_size_ = block_size;
    use_ycbcr_ = use_ycbcr;
    current_quality_ = clamp_quality(quality);

    cuda_init();
    const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, current_quality_, block_size_);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, current_quality_, block_size_);
    const std::vector<int> zigzag = codec_zigzag_scan_table(block_size_);
    cuda_alloc_frame_buffers(width_, height_, channels_, luma_qm.data(), chroma_qm.data(),
                             zigzag.data(), use_ycbcr_, block_size_);
    initialized_ = true;
    return true;
}

void CudaRealtimeDecoder::reset() {
    if (initialized_) {
        cuda_cleanup();
    }
    initialized_ = false;
    width_ = height_ = channels_ = 0;
    block_size_ = 8;
    use_ycbcr_ = true;
    current_quality_ = 50;
}

bool CudaRealtimeDecoder::decode_frame(const RealtimeEncodedFrame& frame, std::vector<uint8_t>& out_interleaved) {
    if (!initialized_) return false;
    if (static_cast<int>(frame.width) != width_ || static_cast<int>(frame.height) != height_ ||
        static_cast<int>(frame.channels) != channels_ || static_cast<int>(frame.block_size) != block_size_) {
        return false;
    }

    const int frame_q = clamp_quality(static_cast<int>(frame.quality));
    if (frame_q != current_quality_) {
        const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, frame_q, block_size_);
        const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, frame_q, block_size_);
        cuda_update_quant_matrices(luma_qm.data(), chroma_qm.data(), block_size_);
        current_quality_ = frame_q;
    }

    int pw[3] = {0}, ph[3] = {0};
    for (int ch = 0; ch < channels_; ++ch) {
        int w = width_, h = height_;
        if (use_ycbcr_ && ch > 0 && channels_ == 3) { w = (width_ + 1) / 2; h = (height_ + 1) / 2; }
        pw[ch] = pad_to_bs(w, block_size_);
        ph[ch] = pad_to_bs(h, block_size_);
    }

    const bool is_keyframe = (frame.is_keyframe != 0);
    for (int ch = 0; ch < channels_; ++ch) {
        if (static_cast<size_t>(ch) >= frame.channel_data.size()) return false;
        const auto& src = frame.channel_data[static_cast<size_t>(ch)];
        const int total_blocks = (pw[ch] / block_size_) * (ph[ch] / block_size_);
        const int coeff_count = total_blocks * block_size_ * block_size_;
        std::vector<int16_t> coeffs(static_cast<size_t>(coeff_count), 0);

        if (src.enc_len > 0 && !src.data.empty()) {
            bool per_block_entropy = false;
            for (uint32_t bl : src.block_bit_lengths) {
                if (bl != 0u) { per_block_entropy = true; break; }
            }

            if (per_block_entropy && static_cast<int>(src.block_bit_lengths.size()) == total_blocks) {
                std::vector<uint8_t> byte_scratch(static_cast<size_t>(2 * block_size_ * block_size_ + 64));
                int bit_cursor = 0;
                for (int bid = 0; bid < total_blocks; ++bid) {
                    const int nbits = static_cast<int>(src.block_bit_lengths[static_cast<size_t>(bid)]);
                    int16_t* dst = coeffs.data() + static_cast<size_t>(bid) * block_size_ * block_size_;
                    if (nbits == 0) continue;
                    const int nbytes = huffman_decode_bit_window(
                        src.huffman_freq.data(),
                        src.data.data(),
                        static_cast<int>(src.data.size()),
                        bit_cursor,
                        nbits,
                        byte_scratch.data(),
                        static_cast<int>(byte_scratch.size()));
                    if (nbytes > 0 && (nbytes & 1) == 0) {
                        const int rle_elems = nbytes / static_cast<int>(sizeof(int16_t));
                        rle_decode_zeros_to_coeffs(reinterpret_cast<const int16_t*>(byte_scratch.data()), rle_elems,
                                                   dst, block_size_ * block_size_);
                    }
                    bit_cursor += nbits;
                }
            } else if (src.rle_bytes > 0) {
                std::vector<uint8_t> enc_with_hdr(sizeof(uint32_t) * 256 + src.data.size());
                std::memcpy(enc_with_hdr.data(), src.huffman_freq.data(), sizeof(uint32_t) * 256);
                std::memcpy(enc_with_hdr.data() + sizeof(uint32_t) * 256, src.data.data(), src.data.size());
                std::vector<int16_t> rle_buf(src.rle_bytes / sizeof(int16_t));
                if (huffman_decode_bytes(enc_with_hdr.data(), static_cast<int>(enc_with_hdr.size()),
                                         reinterpret_cast<uint8_t*>(rle_buf.data()),
                                         static_cast<int>(src.rle_bytes)) == 0) {
                    rle_decode_zeros_to_coeffs(rle_buf.data(), static_cast<int>(rle_buf.size()),
                                               coeffs.data(), coeff_count);
                }
            }
        }
        cuda_decode_channel(ch, coeffs.data(), pw[ch], ph[ch], block_size_, is_keyframe);
    }
    cuda_sync_all();

    Frame decoded = frame_create(width_, height_, channels_, use_ycbcr_, block_size_);
    uint8_t* ptrs[3] = { decoded.channel_ptr(0), decoded.channel_ptr(1), decoded.channel_ptr(2) };
    cuda_download_planes(ptrs, decoded.channels);
    out_interleaved.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * static_cast<size_t>(channels_));
    planes_to_rgb_parallel(decoded, out_interleaved, use_ycbcr_);
    frame_destroy(decoded);

    cuda_swap_recon();
    return true;
}
