#include "image_io.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstring>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string_view>
#include <memory>

Frame load_image(const char* path) {
    int w = 0, h = 0, ch = 0;
    
    struct StbiDeleter {
        void operator()(unsigned char* p) const { stbi_image_free(p); }
    };
    std::unique_ptr<unsigned char, StbiDeleter> raw(stbi_load(path, &w, &h, &ch, 0));
    
    if (!raw) {
        std::fprintf(stderr, "[load_image] Cannot load '%s': %s\n",
                     path, stbi_failure_reason());
        return {};
    }

    Frame f = frame_create(w, h, ch);

    if (ch >= 3) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float R = static_cast<float>(raw.get()[(y * w + x) * ch + 0]);
                float G = static_cast<float>(raw.get()[(y * w + x) * ch + 1]);
                float B = static_cast<float>(raw.get()[(y * w + x) * ch + 2]);
                
                float Y  =  0.299f * R + 0.587f * G + 0.114f * B;
                float Cb = -0.168736f * R - 0.331264f * G + 0.5f * B + 128.0f;
                float Cr =  0.5f * R - 0.418688f * G - 0.081312f * B + 128.0f;
                
                f.at(0, x, y) = Y;
                
                if (x % 2 == 0 && y % 2 == 0) {
                    f.at(1, x / 2, y / 2) = Cb;
                    f.at(2, x / 2, y / 2) = Cr;
                }
            }
        }
    } else {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ch; ++c) {
                    const float val = static_cast<float>(raw.get()[(y * w + x) * ch + c]);
                    f.at(c, x, y) = val;
                }
            }
        }
    }

    return f;
}

bool save_image(const char* path, const Frame& frame) {
    const int w  = frame.width;
    const int h  = frame.height;
    const int ch = frame.channels;

    std::vector<unsigned char> buf(w * h * ch);

    if (ch >= 3) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float Y  = frame.at(0, x, y);
                float Cb = frame.at(1, x / 2, y / 2) - 128.0f;
                float Cr = frame.at(2, x / 2, y / 2) - 128.0f;
                
                float R = Y + 1.402f * Cr;
                float G = Y - 0.344136f * Cb - 0.714136f * Cr;
                float B = Y + 1.772f * Cb;
                
                buf[(y * w + x) * ch + 0] = static_cast<unsigned char>(std::clamp(R + 0.5f, 0.0f, 255.0f));
                buf[(y * w + x) * ch + 1] = static_cast<unsigned char>(std::clamp(G + 0.5f, 0.0f, 255.0f));
                buf[(y * w + x) * ch + 2] = static_cast<unsigned char>(std::clamp(B + 0.5f, 0.0f, 255.0f));
                
                if (ch == 4) {
                    buf[(y * w + x) * ch + 3] = 255;
                }
            }
        }
    } else {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ch; ++c) {
                    const float val = frame.at(c, x, y);
                    const float clamped = std::clamp(val, 0.0f, 255.0f);
                    buf[(y * w + x) * ch + c] = static_cast<unsigned char>(clamped + 0.5f);
                }
            }
        }
    }

    const std::string_view p(path);
    int ok = 0;
    
    if (p.ends_with(".png")) {
        ok = stbi_write_png(path, w, h, ch, buf.data(), w * ch);
    } else if (p.ends_with(".bmp")) {
        ok = stbi_write_bmp(path, w, h, ch, buf.data());
    } else if (p.ends_with(".jpg") || p.ends_with(".jpeg")) {
        ok = stbi_write_jpg(path, w, h, ch, buf.data(), 95);
    } else {
        ok = stbi_write_png(path, w, h, ch, buf.data(), w * ch);
    }

    if (!ok) {
        std::fprintf(stderr, "[save_image] Failed to write '%s'\n", path);
        return false;
    }
    return true;
}
