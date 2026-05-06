#include "realtime/in_memory_codec_c_api.h"

#include "realtime/in_memory_codec.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x52465431u;

struct EncoderWrap {
    CudaRealtimeEncoder encoder;
};

struct DecoderWrap {
    CudaRealtimeDecoder decoder;
    bool initialized = false;
    int width = 0;
    int height = 0;
    int channels = 0;
    int block_size = 8;
    bool use_ycbcr = true;
    int quality = 50;
};

void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xffu));
}

bool read_u32(const uint8_t* data, size_t size, size_t& off, uint32_t& out) {
    if (off + 4 > size) return false;
    out = static_cast<uint32_t>(data[off]) |
          (static_cast<uint32_t>(data[off + 1]) << 8) |
          (static_cast<uint32_t>(data[off + 2]) << 16) |
          (static_cast<uint32_t>(data[off + 3]) << 24);
    off += 4;
    return true;
}

bool serialize_frame(const RealtimeEncodedFrame& frame, std::vector<uint8_t>& out) {
    out.clear();
    append_u32(out, kMagic);
    append_u32(out, frame.width);
    append_u32(out, frame.height);
    append_u32(out, frame.channels);
    append_u32(out, frame.block_size);
    append_u32(out, frame.frame_index);
    out.push_back(frame.is_keyframe);
    out.push_back(frame.use_ycbcr);
    out.push_back(frame.quality);
    out.push_back(0u);
    append_u32(out, static_cast<uint32_t>(frame.channel_data.size()));

    for (const auto& ch : frame.channel_data) {
        append_u32(out, ch.rle_bytes);
        append_u32(out, ch.enc_len);
        append_u32(out, static_cast<uint32_t>(ch.block_bit_lengths.size()));
        for (uint32_t f : ch.huffman_freq) append_u32(out, f);
        for (uint32_t bl : ch.block_bit_lengths) append_u32(out, bl);
        out.insert(out.end(), ch.data.begin(), ch.data.end());
    }
    return true;
}

bool deserialize_frame(const uint8_t* data, size_t size, RealtimeEncodedFrame& frame) {
    frame = {};
    size_t off = 0;
    uint32_t magic = 0;
    if (!read_u32(data, size, off, magic) || magic != kMagic) return false;
    if (!read_u32(data, size, off, frame.width)) return false;
    if (!read_u32(data, size, off, frame.height)) return false;
    if (!read_u32(data, size, off, frame.channels)) return false;
    if (!read_u32(data, size, off, frame.block_size)) return false;
    if (!read_u32(data, size, off, frame.frame_index)) return false;
    if (off + 4 > size) return false;
    frame.is_keyframe = data[off++];
    frame.use_ycbcr = data[off++];
    frame.quality = data[off++];
    off += 1;

    uint32_t channel_count = 0;
    if (!read_u32(data, size, off, channel_count)) return false;
    frame.channel_data.resize(channel_count);

    for (uint32_t ci = 0; ci < channel_count; ++ci) {
        auto& ch = frame.channel_data[ci];
        uint32_t num_blocks = 0;
        if (!read_u32(data, size, off, ch.rle_bytes)) return false;
        if (!read_u32(data, size, off, ch.enc_len)) return false;
        if (!read_u32(data, size, off, num_blocks)) return false;
        for (size_t i = 0; i < ch.huffman_freq.size(); ++i) {
            if (!read_u32(data, size, off, ch.huffman_freq[i])) return false;
        }
        ch.block_bit_lengths.resize(num_blocks);
        for (uint32_t bi = 0; bi < num_blocks; ++bi) {
            if (!read_u32(data, size, off, ch.block_bit_lengths[bi])) return false;
        }
        if (off + ch.enc_len > size) return false;
        ch.data.assign(data + off, data + off + ch.enc_len);
        off += ch.enc_len;
    }
    return off == size;
}
}

extern "C" {

void* rtc_encoder_create() {
    return new EncoderWrap();
}

void rtc_encoder_destroy(void* encoder) {
    auto* w = reinterpret_cast<EncoderWrap*>(encoder);
    delete w;
}

int rtc_encoder_init(void* encoder, int width, int height, int channels,
                     int quality, int block_size, int use_ycbcr,
                     int adaptive_roi, float roi_strength, float scene_cut_threshold) {
    if (!encoder) return 0;
    auto* w = reinterpret_cast<EncoderWrap*>(encoder);
    RealtimeEncodeParams p;
    p.quality = quality;
    p.block_size = block_size;
    p.use_ycbcr = (use_ycbcr != 0);
    p.adaptive_roi = (adaptive_roi != 0);
    p.roi_strength = roi_strength;
    p.scene_cut_threshold = scene_cut_threshold;
    return w->encoder.init(width, height, channels, p) ? 1 : 0;
}

void rtc_encoder_set_quality(void* encoder, int quality) {
    if (!encoder) return;
    auto* w = reinterpret_cast<EncoderWrap*>(encoder);
    w->encoder.set_quality(quality);
}

int rtc_encoder_encode_packet(void* encoder, const uint8_t* interleaved, int bytes,
                              void** out_packet, int* out_size) {
    if (!encoder || !interleaved || bytes <= 0 || !out_packet || !out_size) return 0;
    auto* w = reinterpret_cast<EncoderWrap*>(encoder);
    RealtimeEncodedFrame frame;
    if (!w->encoder.encode_frame(interleaved, static_cast<size_t>(bytes), frame)) return 0;
    std::vector<uint8_t> serialized;
    if (!serialize_frame(frame, serialized)) return 0;
    auto* mem = static_cast<uint8_t*>(std::malloc(serialized.size()));
    if (!mem) return 0;
    std::memcpy(mem, serialized.data(), serialized.size());
    *out_packet = static_cast<void*>(mem);
    *out_size = static_cast<int>(serialized.size());
    return 1;
}

void* rtc_decoder_create() {
    return new DecoderWrap();
}

void rtc_decoder_destroy(void* decoder) {
    auto* w = reinterpret_cast<DecoderWrap*>(decoder);
    delete w;
}

int rtc_decoder_decode_packet(void* decoder, const uint8_t* packet, int packet_size,
                              void** out_rgb, int* out_size,
                              int* out_width, int* out_height, int* out_channels) {
    if (!decoder || !packet || packet_size <= 0 || !out_rgb || !out_size ||
        !out_width || !out_height || !out_channels) {
        return 0;
    }
    auto* w = reinterpret_cast<DecoderWrap*>(decoder);
    RealtimeEncodedFrame frame;
    if (!deserialize_frame(packet, static_cast<size_t>(packet_size), frame)) return 0;

    const int width = static_cast<int>(frame.width);
    const int height = static_cast<int>(frame.height);
    const int channels = static_cast<int>(frame.channels);
    const int block_size = static_cast<int>(frame.block_size);
    const bool use_ycbcr = (frame.use_ycbcr != 0);
    const int quality = static_cast<int>(frame.quality);

    if (!w->initialized || w->width != width || w->height != height || w->channels != channels ||
        w->block_size != block_size || w->use_ycbcr != use_ycbcr || w->quality != quality) {
        if (!w->decoder.init(width, height, channels, block_size, use_ycbcr, quality)) return 0;
        w->initialized = true;
        w->width = width;
        w->height = height;
        w->channels = channels;
        w->block_size = block_size;
        w->use_ycbcr = use_ycbcr;
        w->quality = quality;
    }

    std::vector<uint8_t> rgb;
    if (!w->decoder.decode_frame(frame, rgb)) return 0;
    auto* mem = static_cast<uint8_t*>(std::malloc(rgb.size()));
    if (!mem) return 0;
    std::memcpy(mem, rgb.data(), rgb.size());
    *out_rgb = static_cast<void*>(mem);
    *out_size = static_cast<int>(rgb.size());
    *out_width = width;
    *out_height = height;
    *out_channels = channels;
    return 1;
}

void rtc_free_buffer(void* buffer) {
    std::free(buffer);
}

}
