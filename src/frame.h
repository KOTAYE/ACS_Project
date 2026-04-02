#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

inline int pad_to_block(int n, int align) {
    if (align <= 1) return n;
    return ((n + align - 1) / align) * align;
}

struct Frame {
    int width = 0;
    int height = 0;
    int channels = 0;
    int padded_width[3] = {0};
    int padded_height[3] = {0};
    uint8_t* data[3] = {nullptr};

    uint8_t* channel_ptr(int ch) { return data[ch]; }
    const uint8_t* channel_ptr(int ch) const { return data[ch]; }

    uint8_t& at(int ch, int x, int y) { return data[ch][y * padded_width[ch] + x]; }
    uint8_t at(int ch, int x, int y) const { return data[ch][y * padded_width[ch] + x]; }
};

inline size_t frame_plane_size_bytes(const Frame& f, int ch) {
    return static_cast<size_t>(f.padded_width[ch]) * static_cast<size_t>(f.padded_height[ch]);
}

inline Frame frame_create(int width, int height, int channels, bool use_ycbcr = true,
                          int block_align = 8) {
    Frame f;
    f.width = width;
    f.height = height;
    f.channels = channels;
    for (int ch = 0; ch < channels; ++ch) {
        int w = width;
        int h = height;
        if (use_ycbcr && ch > 0 && channels == 3) {
            w = (width + 1) / 2;
            h = (height + 1) / 2;
        }
        f.padded_width[ch] = pad_to_block(w, block_align);
        f.padded_height[ch] = pad_to_block(h, block_align);
        const size_t total = static_cast<size_t>(f.padded_width[ch]) * f.padded_height[ch];
        f.data[ch] = static_cast<uint8_t*>(std::calloc(total, sizeof(uint8_t)));
        assert(f.data[ch]);
    }
    return f;
}

inline void frame_destroy(Frame& f) {
    for (int ch = 0; ch < f.channels; ++ch) {
        std::free(f.data[ch]);
        f.data[ch] = nullptr;
    }
}
