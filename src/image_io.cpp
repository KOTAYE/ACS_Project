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

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                const float val = static_cast<float>(raw.get()[(y * w + x) * ch + c]);
                f.at(c, x, y) = val;
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

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                const float val = frame.at(c, x, y);
                const float clamped = std::clamp(val, 0.0f, 255.0f);
                buf[(y * w + x) * ch + c] = static_cast<unsigned char>(clamped + 0.5f);
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
