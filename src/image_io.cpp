#include "image_io.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstring>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string_view>
#include <memory>
#include <filesystem>
#include <cctype>
#include <fstream>
#include <string>

#include "tinyexr.h"

namespace fs = std::filesystem;

static bool path_ends_with_exr(const std::string& p) {
    if (p.size() < 4) return false;
    char a = p[p.size() - 4], b = p[p.size() - 3], c = p[p.size() - 2], d = p[p.size() - 1];
    auto lo = [](char x) { return (x >= 'A' && x <= 'Z') ? static_cast<char>(x + 32) : x; };
    return lo(a) == '.' && lo(b) == 'e' && lo(c) == 'x' && lo(d) == 'r';
}

static uint8_t tonemap_exr_float_to_u8(float v) {
    v = std::max(0.f, v);
    if (v > 1.f) v = v / (1.f + v);
    return static_cast<uint8_t>(std::clamp(std::round(v * 255.f), 0.f, 255.f));
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    in.seekg(0, std::ios::end);
    const std::streamoff len = in.tellg();
    if (len <= 0) return {};
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    if (!in.read(reinterpret_cast<char*>(buf.data()), len)) return {};
    return buf;
}

bool decode_image_file_bytes(const std::string& path_hint, const uint8_t* data, size_t len,
                             int& out_w, int& out_h, int& out_ch, std::vector<uint8_t>& interleaved) {
    interleaved.clear();
    if (!data || len == 0) return false;

    if (path_ends_with_exr(path_hint)) {
        float* rgba = nullptr;
        const char* err = nullptr;
        int w = 0, h = 0;
        const int ret = LoadEXRFromMemory(&rgba, &w, &h, data, len, &err);
        if (ret != TINYEXR_SUCCESS) {
            if (err) FreeEXRErrorMessage(err);
            return false;
        }
        out_w = w;
        out_h = h;
        out_ch = 4;
        const size_t pix = static_cast<size_t>(w) * static_cast<size_t>(h);
        interleaved.resize(pix * 4);
        for (size_t i = 0; i < pix; ++i) {
            for (int c = 0; c < 4; ++c)
                interleaved[i * 4 + c] = tonemap_exr_float_to_u8(rgba[i * 4 + c]);
        }
        free(rgba);
        return true;
    }

    int w = 0, h = 0, c = 0;
    uint8_t* raw = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &c, 0);
    if (!raw) return false;
    out_w = w;
    out_h = h;
    out_ch = c;
    interleaved.assign(raw, raw + static_cast<size_t>(w) * h * c);
    stbi_image_free(raw);
    return true;
}

static std::vector<int> frame_path_numeric_key(std::string_view name) {
    std::vector<int> nums;
    for (size_t i = 0; i < name.size(); ) {
        unsigned char c = static_cast<unsigned char>(name[i]);
        if (std::isdigit(c)) {
            int v = 0;
            while (i < name.size()) {
                c = static_cast<unsigned char>(name[i]);
                if (!std::isdigit(c)) break;
                v = v * 10 + static_cast<int>(c - '0');
                ++i;
            }
            nums.push_back(v);
        } else {
            ++i;
        }
    }
    return nums;
}

void sort_frame_paths(std::vector<std::string>& paths) {
    std::sort(paths.begin(), paths.end(), [](const std::string& a, const std::string& b) {
        std::string na = fs::path(a).filename().string();
        std::string nb = fs::path(b).filename().string();
        std::vector<int> ka = frame_path_numeric_key(na);
        std::vector<int> kb = frame_path_numeric_key(nb);
        if (!ka.empty() && !kb.empty()) {
            size_t n = std::min(ka.size(), kb.size());
            for (size_t i = 0; i < n; ++i) {
                if (ka[i] != kb[i]) return ka[i] < kb[i];
            }
            if (ka.size() != kb.size()) return ka.size() < kb.size();
        }
        return na < nb;
    });
}

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

Frame rgb_to_planes_parallel(uint8_t* raw, int w, int h, int ch, bool use_ycbcr,
                             int block_align) {
    Frame f = frame_create(w, h, ch, use_ycbcr, block_align);
    if (use_ycbcr && ch >= 3) {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float R = static_cast<float>(raw[(y * w + x) * ch + 0]);
                float G = static_cast<float>(raw[(y * w + x) * ch + 1]);
                float B = static_cast<float>(raw[(y * w + x) * ch + 2]);

                float Y  =  0.299f * R + 0.587f * G + 0.114f * B;
                f.at(0, x, y) = static_cast<uint8_t>(std::clamp(std::round(Y), 0.0f, 255.0f));

                if (x % 2 == 0 && y % 2 == 0) {
                    float Cb = -0.168736f * R - 0.331264f * G + 0.5f * B + 128.0f;
                    float Cr =  0.5f * R - 0.418688f * G - 0.081312f * B + 128.0f;
                    f.at(1, x / 2, y / 2) = static_cast<uint8_t>(std::clamp(std::round(Cb), 0.0f, 255.0f));
                    f.at(2, x / 2, y / 2) = static_cast<uint8_t>(std::clamp(std::round(Cr), 0.0f, 255.0f));
                }
            }
        }
    } else {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ch; ++c) {
                    f.at(c, x, y) = raw[(y * w + x) * ch + c];
                }
            }
        }
    }
    return f;
}

void planes_to_rgb_parallel(const Frame& f, std::vector<uint8_t>& out_rgb, bool use_ycbcr) {
    int w = f.width;
    int h = f.height;
    int ch = f.channels;

    if (use_ycbcr && ch >= 3) {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float Y  = static_cast<float>(f.at(0, x, y));
                float Cb = static_cast<float>(f.at(1, x / 2, y / 2)) - 128.0f;
                float Cr = static_cast<float>(f.at(2, x / 2, y / 2)) - 128.0f;

                float R = Y + 1.402f * Cr;
                float G = Y - 0.344136f * Cb - 0.714136f * Cr;
                float B = Y + 1.772f * Cb;

                out_rgb[(y * w + x) * ch + 0] = static_cast<uint8_t>(std::clamp(std::round(R), 0.0f, 255.0f));
                out_rgb[(y * w + x) * ch + 1] = static_cast<uint8_t>(std::clamp(std::round(G), 0.0f, 255.0f));
                out_rgb[(y * w + x) * ch + 2] = static_cast<uint8_t>(std::clamp(std::round(B), 0.0f, 255.0f));

                if (ch == 4) out_rgb[(y * w + x) * ch + 3] = 255;
            }
        }
    } else {
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ch; ++c) {
                    out_rgb[(y * w + x) * ch + c] = f.at(c, x, y);
                }
            }
        }
    }
}
