// Еталонне порівняння стиснення: RLE, Huffman, RLE+Huffman, RLE+Arithmetic (CPU).
// Опційно: кадри з каталогу (колір), синтетичні нормалі / глибина.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "arithmetic_ref.h"
#include "dct.h"
#include "frame.h"
#include "huffman.h"
#include "image_io.h"
#include "quant.h"
#include "tiling.h"
#include "zigzag.h"

namespace fs = std::filesystem;

static std::vector<int16_t> rle_encode_zeros_vec(const std::vector<int16_t>& in) {
    std::vector<int16_t> out;
    out.reserve(in.size() / 2);
    int i = 0;
    const int n = static_cast<int>(in.size());
    while (i < n) {
        if (in[i] == 0) {
            int run = 1;
            while (i + run < n && in[i + run] == 0 && run < 32767) run++;
            out.push_back(0);
            out.push_back(static_cast<int16_t>(run));
            i += run;
        } else {
            out.push_back(in[i]);
            i++;
        }
    }
    return out;
}

struct ChannelStats {
    std::string label;
    int64_t raw_coeff_bytes = 0;
    int64_t rle_bytes = 0;
    int64_t huff_total = 0;
    int64_t order0_bound = 0; // заголовок + ceil(H/8) (ідеальна межа)
    int64_t arithmetic_actual = 0; // фактичний статичний order-0 range code (roundtrip)
};

static void synth_normals_rgb(int w, int h, std::vector<uint8_t>& raw) {
    raw.resize(static_cast<size_t>(w * h * 3));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(w > 1 ? w - 1 : 1);
            const float v = static_cast<float>(y) / static_cast<float>(h > 1 ? h - 1 : 1);
            float nx = std::sin(u * 6.2831853f) * 0.5f;
            float ny = std::cos(v * 6.2831853f) * 0.5f;
            float nz = std::sqrt(std::max(0.f, 1.f - nx * nx - ny * ny));
            const size_t o = (static_cast<size_t>(y) * w + x) * 3;
            raw[o + 0] = static_cast<uint8_t>(std::clamp(std::round((nx + 0.5f) * 255.f), 0.f, 255.f));
            raw[o + 1] = static_cast<uint8_t>(std::clamp(std::round((ny + 0.5f) * 255.f), 0.f, 255.f));
            raw[o + 2] = static_cast<uint8_t>(std::clamp(std::round(nz * 255.f), 0.f, 255.f));
        }
    }
}

static void synth_depth_grey(int w, int h, std::vector<uint8_t>& raw) {
    raw.resize(static_cast<size_t>(w * h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float z = 1.f + 0.3f * std::sin(static_cast<float>(x + y) * 0.05f);
            const float d = 1.f / z;
            raw[static_cast<size_t>(y) * w + x] =
                static_cast<uint8_t>(std::clamp(std::round(d * 200.f), 0.f, 255.f));
        }
    }
}

static void process_frame_like_codec(const Frame& current, Frame& prev_recon, Frame& curr_recon, bool is_keyframe,
                                     int quality, int block_size, const std::vector<int>& zigzag,
                                     std::vector<ChannelStats>& acc) {
    if (acc.empty()) {
        acc.resize(static_cast<size_t>(current.channels));
        for (int c = 0; c < current.channels; ++c) {
            acc[static_cast<size_t>(c)].label =
                (current.channels >= 3) ? (c == 0 ? "Y" : (c == 1 ? "Cb" : "Cr")) : ("ch" + std::to_string(c));
        }
    }

    const int bs = block_size;
    const int bpp = bs * bs;
    const QuantMatrix luma_qm = make_quant_matrix(kJpegLumaQuant, quality, block_size);
    const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality, block_size);

    std::vector<int16_t> channel_buffer;
    std::vector<uint8_t> enc_huff;

    for (int ch = 0; ch < current.channels; ++ch) {
        const QuantMatrix& qm = (ch == 0) ? luma_qm : chroma_qm;
        const uint8_t* src_channel = current.channel_ptr(ch);
        const uint8_t* prev_channel = prev_recon.channel_ptr(ch);
        uint8_t* recon_channel = curr_recon.channel_ptr(ch);

        const int padded_w = current.padded_width[ch];
        const int padded_h = current.padded_height[ch];
        const int blocks_x = padded_w / bs;
        const int blocks_y = padded_h / bs;
        const int total_blocks = blocks_x * blocks_y;
        const int channel_samples = total_blocks * bpp;

        channel_buffer.resize(channel_samples);

        for (int by = 0; by < blocks_y; ++by) {
            std::vector<float> block_in(static_cast<size_t>(bpp));
            std::vector<float> prev_block(static_cast<size_t>(bpp));
            std::vector<float> dct_out(static_cast<size_t>(bpp));
            std::vector<float> idct_out(static_cast<size_t>(bpp));
            for (int bx = 0; bx < blocks_x; ++bx) {
                const int sample_idx = (by * blocks_x + bx) * bpp;
                extract_block_n(src_channel, padded_w, bx, by, bs, block_in.data());

                if (!is_keyframe) {
                    extract_block_n(prev_channel, padded_w, bx, by, bs, prev_block.data());
                    for (int i = 0; i < bpp; ++i) block_in[i] -= prev_block[i];
                } else {
                    for (int i = 0; i < bpp; ++i) block_in[i] -= 128.0f;
                }

                dct2d_separable_n(block_in.data(), dct_out.data(), bs);
                quantize_block(dct_out.data(), qm, bpp);

                for (int i = 0; i < bpp; ++i) channel_buffer[sample_idx + zigzag[i]] = static_cast<int16_t>(dct_out[i]);

                dequantize_block(dct_out.data(), qm, bpp);
                idct2d_separable_n(dct_out.data(), idct_out.data(), bs);

                if (!is_keyframe) {
                    for (int i = 0; i < bpp; ++i) idct_out[i] += prev_block[i];
                } else {
                    for (int i = 0; i < bpp; ++i) idct_out[i] += 128.0f;
                }

                insert_block_n(recon_channel, padded_w, bx, by, bs, idct_out.data());
            }
        }

        std::vector<int16_t> rle = rle_encode_zeros_vec(channel_buffer);
        const uint8_t* rle_u8 = reinterpret_cast<const uint8_t*>(rle.data());
        const int rle_len = static_cast<int>(rle.size() * sizeof(int16_t));

        const int64_t raw_b = static_cast<int64_t>(channel_samples * static_cast<int>(sizeof(int16_t)));
        const int64_t rle_b = rle_len;

        enc_huff.resize(static_cast<size_t>(rle_len + 8192));
        int huff_total = 0;
        if (rle_len > 0)
            huff_total = huffman_encode_bytes(rle_u8, rle_len, enc_huff.data(), static_cast<int>(enc_huff.size()));

        const int bound_total = (rle_len > 0) ? arithmetic_order0_bound_total_bytes(rle_u8, rle_len) : 512;

        int64_t ac_sz = 0;
        if (rle_len > 0) {
            std::vector<uint8_t> ac_enc;
            const int enc_n = arithmetic_order0_encode(rle_u8, rle_len, ac_enc);
            if (enc_n > 0) ac_sz = static_cast<int64_t>(enc_n);
        }

        acc[static_cast<size_t>(ch)].raw_coeff_bytes += raw_b;
        acc[static_cast<size_t>(ch)].rle_bytes += rle_b;
        acc[static_cast<size_t>(ch)].huff_total += (huff_total >= 512) ? static_cast<int64_t>(huff_total) : 0;
        acc[static_cast<size_t>(ch)].order0_bound += static_cast<int64_t>(bound_total);
        acc[static_cast<size_t>(ch)].arithmetic_actual += ac_sz;
    }

    std::swap(curr_recon.data, prev_recon.data);
}

static void print_report(const std::string& title, std::vector<ChannelStats>& acc) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "RLE = run-length нулів після квантування (int16 → байти). Huffman = еталон CPU (як flipbook_omp).\n";
    std::cout << "Order0-ideal = 512 + ceil(H/8) (як блок частот Huffman). AC(actual) = range code, заголовок 520 B (ARQ0).\n\n";
    std::cout << "channel | raw_coef | RLE     | Huffman | Order0-ideal | AC(actual) | Huff/raw | Ideal/raw | AC/raw\n";
    int64_t t_raw = 0, t_rle = 0, t_h = 0, t_a = 0, t_ac = 0;
    for (auto& c : acc) {
        t_raw += c.raw_coeff_bytes;
        t_rle += c.rle_bytes;
        t_h += c.huff_total;
        t_a += c.order0_bound;
        t_ac += c.arithmetic_actual;
        auto ratio = [](int64_t a, int64_t b) {
            return b ? (100.0 * static_cast<double>(a) / static_cast<double>(b)) : 0.0;
        };
        std::cout << c.label << "\t| " << c.raw_coeff_bytes << "\t| " << c.rle_bytes << "\t| " << c.huff_total
                  << "\t| " << c.order0_bound << "\t| " << c.arithmetic_actual << "\t| "
                  << ratio(c.huff_total, c.raw_coeff_bytes) << "%\t| "
                  << ratio(c.order0_bound, c.raw_coeff_bytes) << "%\t| "
                  << ratio(c.arithmetic_actual, c.raw_coeff_bytes) << "%\n";
    }
    auto ratio = [](int64_t a, int64_t b) {
        return b ? (100.0 * static_cast<double>(a) / static_cast<double>(b)) : 0.0;
    };
    std::cout << "ALL\t| " << t_raw << "\t| " << t_rle << "\t| " << t_h << "\t| " << t_a << "\t| " << t_ac << "\t| "
              << ratio(t_h, t_raw) << "%\t| " << ratio(t_a, t_raw) << "%\t| " << ratio(t_ac, t_raw) << "%\n";
}

static int self_test_bound() {
    if (!arithmetic_order0_roundtrip_selftest()) {
        std::cerr << "self-test: arithmetic roundtrip failed\n";
        return 1;
    }

    std::vector<uint8_t> msg(800);
    for (size_t i = 0; i < msg.size(); ++i) msg[i] = static_cast<uint8_t>(i * 13 + 7);

    std::vector<uint8_t> enc_h(msg.size() + 4096);
    const int hlen =
        huffman_encode_bytes(msg.data(), static_cast<int>(msg.size()), enc_h.data(), static_cast<int>(enc_h.size()));
    const int bound = arithmetic_order0_bound_total_bytes(msg.data(), static_cast<int>(msg.size()));
    std::vector<uint8_t> enc_ac;
    const int ac_len = arithmetic_order0_encode(msg.data(), static_cast<int>(msg.size()), enc_ac);
    std::vector<uint8_t> dec;
    if (arithmetic_order0_decode(enc_ac.data(), static_cast<int>(enc_ac.size()), dec) != static_cast<int>(msg.size()) ||
        dec != msg) {
        std::cerr << "self-test: arithmetic decode mismatch\n";
        return 1;
    }
    if (hlen < 0 || bound < 512 || ac_len < 520) {
        std::cerr << "self-test encode failed\n";
        return 1;
    }
    if (hlen < bound) {
        std::cerr << "unexpected: Huffman total " << hlen << " < order0 bound " << bound << "\n";
        return 1;
    }
    const int bound_ac = bound - 512 + 520; // той самий H, заголовок як у AC
    if (ac_len > bound_ac + 64) {
        std::cerr << "unexpected: AC actual " << ac_len << " >> bound_ac " << bound_ac << "\n";
        return 1;
    }
    std::cout << "self-test: AC roundtrip OK; Huffman=" << hlen << " order0_bound=" << bound << " AC_actual=" << ac_len
              << "\n";
    return 0;
}

int main(int argc, char** argv) {
    dct_init_lut();

    if (argc >= 2 && std::string(argv[1]) == "--self-test") return self_test_bound();

    int quality = 50;
    int block_size = 8;
    int max_frames = 5;
    std::string mode = "color";
    std::string input_dir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-q" && i + 1 < argc) quality = std::stoi(argv[++i]);
        else if (a == "-b" && i + 1 < argc) block_size = std::stoi(argv[++i]);
        else if (a == "-n" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a[0] != '-') input_dir = a;
    }

    if (block_size != 8 && block_size != 16 && block_size != 32) {
        std::cerr << "block_size must be 8, 16, or 32\n";
        return 1;
    }

    const std::vector<int> zigzag = codec_zigzag_scan_table(block_size);

    if (mode == "color") {
        if (input_dir.empty()) {
            std::cerr << "Usage: entropy_reference [--self-test] [DIR] [-q Q] [-b B] [-n NFRAMES]\n"
                      << "       entropy_reference --mode normals|depth [-q Q] [-b B]\n"
                      << "  Порівнює: raw, RLE, Huffman, межа order-0 AC, фактичний AC (range code).\n";
            return 1;
        }
        std::vector<std::string> frames;
        for (const auto& e : fs::directory_iterator(input_dir)) {
            if (e.is_regular_file()) frames.push_back(e.path().string());
        }
        sort_frame_paths(frames);
        if (frames.empty()) {
            std::cerr << "No frames in " << input_dir << "\n";
            return 1;
        }
        if (max_frames > 0 && static_cast<int>(frames.size()) > max_frames) frames.resize(static_cast<size_t>(max_frames));

        int img_w = 0, img_h = 0, img_ch = 0;
        std::vector<uint8_t> first_file = read_file_bytes(frames[0]);
        std::vector<uint8_t> first_rgb;
        if (!decode_image_file_bytes(frames[0], first_file.data(), first_file.size(), img_w, img_h, img_ch,
                                     first_rgb)) {
            std::cerr << "Failed to decode first frame\n";
            return 1;
        }

        Frame prev = frame_create(img_w, img_h, img_ch, true, block_size);
        Frame curr = frame_create(img_w, img_h, img_ch, true, block_size);
        std::vector<ChannelStats> acc;

        for (size_t fi = 0; fi < frames.size(); ++fi) {
            std::vector<uint8_t> file_b = read_file_bytes(frames[fi]);
            std::vector<uint8_t> inter;
            int w = 0, h = 0, c = 0;
            if (!decode_image_file_bytes(frames[fi], file_b.data(), file_b.size(), w, h, c, inter)) continue;
            if (w != img_w || h != img_h || c != img_ch) continue;
            Frame current = rgb_to_planes_parallel(inter.data(), img_w, img_h, img_ch, true, block_size);

            process_frame_like_codec(current, prev, curr, fi == 0, quality, block_size, zigzag, acc);
            frame_destroy(current);
        }
        frame_destroy(prev);
        frame_destroy(curr);

        print_report("Color (YCbCr planes, DCT+RLE reference)", acc);
        return 0;
    }

    const int W = 512, H = 512;
    std::vector<uint8_t> raw;
    int ch = 3;
    bool ycbcr = true;
    if (mode == "normals") {
        synth_normals_rgb(W, H, raw);
    } else if (mode == "depth") {
        synth_depth_grey(W, H, raw);
        ch = 1;
        ycbcr = false;
    } else {
        std::cerr << "Unknown --mode (use color, normals, depth)\n";
        return 1;
    }

    Frame prev = frame_create(W, H, ch, ycbcr, block_size);
    Frame curr = frame_create(W, H, ch, ycbcr, block_size);
    Frame current = rgb_to_planes_parallel(raw.data(), W, H, ch, ycbcr, block_size);
    std::vector<ChannelStats> acc;
    process_frame_like_codec(current, prev, curr, true, quality, block_size, zigzag, acc);
    frame_destroy(current);
    frame_destroy(prev);
    frame_destroy(curr);

    print_report(mode == "normals" ? "Synthetic packed normals (as RGB)" : "Synthetic depth (grey, no YCbCr)", acc);
    return 0;
}
