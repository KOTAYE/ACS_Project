    #pragma once
    #include <cstdlib>
    #include <cstring>
    #include <cassert>

    inline int pad8(int n) { return ((n + 7) / 8) * 8; }

    struct Frame {
        int width = 0;
        int height = 0;
        int channels = 0;
        int padded_width[3] = {0};
        int padded_height[3] = {0};
        float* data[3] = {nullptr};

        float* channel_ptr(int ch) {
            return data[ch];
        }
        const float* channel_ptr(int ch) const {
            return data[ch];
        }

        float& at(int ch, int x, int y) {
            return data[ch][y * padded_width[ch] + x];
        }
        float at(int ch, int x, int y) const {
            return data[ch][y * padded_width[ch] + x];
        }
    };

    inline Frame frame_create(int width, int height, int channels) {
        Frame f;
        f.width = width;
        f.height = height;
        f.channels = channels;
        for (int ch = 0; ch < channels; ++ch) {
            int w = width;
            int h = height;
            if (ch > 0 && channels == 3) {
                // 4:2:0 Subsampling for Chroma
                w = (width + 1) / 2;
                h = (height + 1) / 2;
            }
            f.padded_width[ch] = pad8(w);
            f.padded_height[ch] = pad8(h);
            size_t total = (size_t)f.padded_width[ch] * f.padded_height[ch];
            f.data[ch] = (float*)std::calloc(total, sizeof(float));
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
