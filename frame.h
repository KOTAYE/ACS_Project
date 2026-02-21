    #pragma once
    #include <cstdlib>
    #include <cstring>
    #include <cassert>

    inline int pad8(int n) { return ((n + 7) / 8) * 8; }

    struct Frame {
        int width = 0;
        int height = 0;
        int channels = 0;
        int padded_width = 0;
        int padded_height = 0;
        float* data = nullptr; // CHW-буфер

        float* channel_ptr(int ch) {
            return data + ch * padded_width * padded_height;
        }
        const float* channel_ptr(int ch) const {
            return data + ch * padded_width * padded_height;
        }

        float& at(int ch, int x, int y) {
            return data[ch * padded_width * padded_height + y * padded_width + x];
        }
        float at(int ch, int x, int y) const {
            return data[ch * padded_width * padded_height + y * padded_width + x];
        }
    };

    inline Frame frame_create(int width, int height, int channels) {
        Frame f;
        f.width = width;
        f.height = height;
        f.channels = channels;
        f.padded_width = pad8(width);
        f.padded_height = pad8(height);
        size_t total = (size_t)f.channels * f.padded_width * f.padded_height;
        f.data = (float*)std::calloc(total, sizeof(float));
        assert(f.data);
        return f;
    }

    inline void frame_destroy(Frame& f) {
        std::free(f.data);
        f.data = nullptr;
    }

    struct Flipbook {
        int frame_count = 0;
        Frame* frames = nullptr;
    };

    inline Flipbook flipbook_create(int count, int w, int h, int ch) {
        Flipbook fb;
        fb.frame_count = count;
        fb.frames = new Frame[count];
        for (int i = 0; i < count; ++i)
            fb.frames[i] = frame_create(w, h, ch);
        return fb;
    }

    inline void flipbook_destroy(Flipbook& fb) {
        for (int i = 0; i < fb.frame_count; ++i)
            frame_destroy(fb.frames[i]);
        delete[] fb.frames;
        fb.frames = nullptr;
        fb.frame_count = 0;
    }
