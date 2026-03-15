#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include <cstring>
#include <cmath>

#include "huffman.h"

static std::vector<int16_t> rle_encode_zeros(const std::vector<int16_t>& in) {
    std::vector<int16_t> out;
    out.reserve(in.size() / 2);
    int i = 0;
    const int n = static_cast<int>(in.size());
    while (i < n) {
        if (in[i] == 0) {
            int run = 1;
            while (i + run < n && in[i + run] == 0 && run < 32767) {
                run++;
            }
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

static void rle_decode_zeros(const std::vector<int16_t>& in,
                             std::vector<int16_t>& out) {
    int i = 0;
    int o = 0;
    const int n = static_cast<int>(in.size());
    const int out_n = static_cast<int>(out.size());
    while (i < n && o < out_n) {
        if (in[i] == 0) {
            if (i + 1 >= n) break;
            int run = in[i + 1];
            for (int k = 0; k < run && o < out_n; ++k) {
                out[o++] = 0;
            }
            i += 2;
        } else {
            out[o++] = in[i];
            i++;
        }
    }
}

static bool equal_vectors(const std::vector<int16_t>& a,
                          const std::vector<int16_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    const int N = 1 << 16;
    std::vector<int16_t> coeffs(N);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> val_dist(-10, 10);
    std::uniform_real_distribution<float> zero_dist(0.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        if (zero_dist(rng) < 0.7f) {
            coeffs[i] = 0;
        } else {
            coeffs[i] = static_cast<int16_t>(val_dist(rng));
        }
    }

    const int raw_bytes = static_cast<int>(coeffs.size() * sizeof(int16_t));

    std::vector<int16_t> rle_buf = rle_encode_zeros(coeffs);
    const int rle_bytes = static_cast<int>(rle_buf.size() * sizeof(int16_t));

    std::vector<uint8_t> enc_rle(rle_bytes + 1024u);
    std::vector<uint8_t> enc_raw(raw_bytes + 1024u);

    const uint8_t* rle_bytes_ptr =
        reinterpret_cast<const uint8_t*>(rle_buf.data());
    int rle_huff_len = huffman_encode_bytes(rle_bytes_ptr, rle_bytes,
                                            enc_rle.data(), static_cast<int>(enc_rle.size()));
    if (rle_huff_len < 0) rle_huff_len = 0;

    const uint8_t* raw_bytes_ptr =
        reinterpret_cast<const uint8_t*>(coeffs.data());
    int raw_huff_len = huffman_encode_bytes(raw_bytes_ptr, raw_bytes,
                                            enc_raw.data(), static_cast<int>(enc_raw.size()));
    if (raw_huff_len < 0) raw_huff_len = 0;

    std::vector<int16_t> rle_dec(rle_buf.size());
    if (rle_huff_len > 0) {
        std::vector<uint8_t> tmp(rle_bytes);
        huffman_decode_bytes(enc_rle.data(), rle_huff_len,
                             tmp.data(), rle_bytes);
        std::memcpy(rle_dec.data(), tmp.data(), rle_bytes);
    }

    std::vector<int16_t> coeffs_rec(coeffs.size(), 0);
    rle_decode_zeros(rle_dec, coeffs_rec);

    bool ok_rle = equal_vectors(coeffs, coeffs_rec);

    std::vector<int16_t> coeffs_dec(coeffs.size());
    if (raw_huff_len > 0) {
        std::vector<uint8_t> tmp(raw_bytes);
        huffman_decode_bytes(enc_raw.data(), raw_huff_len,
                             tmp.data(), raw_bytes);
        std::memcpy(coeffs_dec.data(), tmp.data(), raw_bytes);
    }

    bool ok_raw = equal_vectors(coeffs, coeffs_dec);

    std::cout << "Entropy test on synthetic coefficients (N=" << N << ")\n";
    std::cout << "  Raw bytes          : " << raw_bytes << " (100%)\n";
    std::cout << "  RLE bytes          : " << rle_bytes
              << " (" << (100.0 * rle_bytes / raw_bytes) << "%)\n";
    std::cout << "  Huffman only bytes : " << raw_huff_len
              << " (" << (100.0 * raw_huff_len / raw_bytes) << "%)\n";
    std::cout << "  RLE+Huffman bytes  : " << rle_huff_len
              << " (" << (100.0 * rle_huff_len / raw_bytes) << "%)\n";

    std::cout << "  Decode raw Huffman ok : " << (ok_raw ? "yes" : "no") << "\n";
    std::cout << "  Decode RLE+Huffman ok : " << (ok_rle ? "yes" : "no") << "\n";

    return 0;
}

