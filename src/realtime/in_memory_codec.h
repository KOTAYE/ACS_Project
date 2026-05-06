#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct RealtimeEncodedChannel {
    uint32_t rle_bytes = 0;
    uint32_t enc_len = 0;
    std::array<uint32_t, 256> huffman_freq{};
    std::vector<uint32_t> block_bit_lengths;
    std::vector<uint8_t> data;
};

struct RealtimeEncodedFrame {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    uint32_t block_size = 8;
    uint32_t frame_index = 0;
    uint8_t is_keyframe = 0;
    uint8_t use_ycbcr = 1;
    uint8_t quality = 50;
    std::vector<RealtimeEncodedChannel> channel_data;
};

struct RealtimeEncodeParams {
    int quality = 50;
    int block_size = 8;
    bool use_ycbcr = true;
    bool adaptive_roi = false;
    float roi_strength = 0.55f;
    float scene_cut_threshold = 22.0f;
};

class CudaRealtimeEncoder {
public:
    CudaRealtimeEncoder();
    ~CudaRealtimeEncoder();

    bool init(int width, int height, int channels, const RealtimeEncodeParams& params);
    void reset();

    bool encode_frame(const uint8_t* interleaved, size_t bytes, RealtimeEncodedFrame& out_frame);
    void set_quality(int quality);

private:
    bool initialized_ = false;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    int frame_index_ = 0;
    int current_quality_ = 50;
    RealtimeEncodeParams params_{};
    std::vector<uint8_t> prev_input_luma_;
};

class CudaRealtimeDecoder {
public:
    CudaRealtimeDecoder();
    ~CudaRealtimeDecoder();

    bool init(int width, int height, int channels, int block_size, bool use_ycbcr, int quality);
    void reset();

    bool decode_frame(const RealtimeEncodedFrame& frame, std::vector<uint8_t>& out_interleaved);

private:
    bool initialized_ = false;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    int block_size_ = 8;
    bool use_ycbcr_ = true;
    int current_quality_ = 50;
};
