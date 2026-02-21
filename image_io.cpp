#include "image_io.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstring>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>


Frame load_image(const char* path) {
    int w, h, ch;
    unsigned char* raw = stbi_load(path, &w, &h, &ch, 0);
    if (!raw) {
        std::fprintf(stderr, "[load_image] Cannot load '%s': %s\n",
                     path, stbi_failure_reason());
        return {};
    }

    Frame f = frame_create(w, h, ch);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                float val = static_cast<float>(raw[(y * w + x) * ch + c]);
                f.at(c, x, y) = val;
            }
        }
    }

    stbi_image_free(raw);
    return f;
}


bool save_image(const char* path, const Frame& frame) {
    int w  = frame.width;
    int h  = frame.height;
    int ch = frame.channels;

    std::vector<unsigned char> buf(w * h * ch);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                float val = frame.at(c, x, y);
                val = std::max(0.0f, std::min(255.0f, val));
                buf[(y * w + x) * ch + c] = static_cast<unsigned char>(val + 0.5f);
            }
        }
    }

    std::string p(path);
    int ok = 0;
    if (p.size() >= 4 && p.substr(p.size() - 4) == ".png") {
        ok = stbi_write_png(path, w, h, ch, buf.data(), w * ch);
    } else if (p.size() >= 4 && p.substr(p.size() - 4) == ".bmp") {
        ok = stbi_write_bmp(path, w, h, ch, buf.data());
    } else if (p.size() >= 4 &&
              (p.substr(p.size() - 4) == ".jpg" || p.substr(p.size() - 5) == ".jpeg")) {
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
