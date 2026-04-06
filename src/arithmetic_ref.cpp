#include "arithmetic_ref.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

int arithmetic_order0_bound_total_bytes(const uint8_t* data, int len) {
    // Теоретична межа при тому ж розмірі блоку частот, що й Huffman (512 B).
    constexpr int kHeader = 512;
    if (len <= 0) return kHeader;

    std::array<uint64_t, 256> hist{};
    for (int i = 0; i < len; ++i) ++hist[data[i]];

    long double bits = 0.0L;
    const long double n = static_cast<long double>(len);
    for (unsigned s = 0; s < 256; ++s) {
        if (hist[s] == 0) continue;
        const long double c = static_cast<long double>(hist[s]);
        bits -= c * std::log2(c / n);
    }
    const int payload = static_cast<int>(std::ceil(static_cast<double>(bits / 8.0L)));
    return kHeader + std::max(0, payload);
}

namespace {

constexpr int kModelTotal = 65536; // сума масштабованих частот для стабільного ділення

void build_histogram(const uint8_t* data, int len, uint32_t raw[256]) {
    std::memset(raw, 0, 256 * sizeof(uint32_t));
    for (int i = 0; i < len; ++i) ++raw[data[i]];
}

// Масштабує частоти до суми kModelTotal; нульові лишаються нулями.
void scale_to_model(const uint32_t raw[256], uint16_t scaled[256]) {
    uint64_t sum = 0;
    for (int i = 0; i < 256; ++i) sum += raw[i];
    if (sum == 0) {
        std::memset(scaled, 0, 256 * sizeof(uint16_t));
        return;
    }
    uint32_t placed = 0;
    int last_nz = -1;
    for (int i = 0; i < 256; ++i) {
        if (raw[i] == 0) {
            scaled[i] = 0;
            continue;
        }
        uint64_t v = std::max<uint64_t>(1, raw[i] * static_cast<uint64_t>(kModelTotal) / sum);
        if (v > 65535) v = 65535;
        scaled[i] = static_cast<uint16_t>(v);
        placed += scaled[i];
        last_nz = i;
    }
    if (last_nz < 0) return;
    if (placed > kModelTotal) {
        uint32_t over = placed - kModelTotal;
        while (over > 0 && scaled[last_nz] > 1) {
            uint32_t d = std::min<uint32_t>(over, scaled[last_nz] - 1);
            scaled[last_nz] = static_cast<uint16_t>(scaled[last_nz] - d);
            over -= d;
        }
    } else if (placed < kModelTotal) {
        scaled[last_nz] = static_cast<uint16_t>(scaled[last_nz] + (kModelTotal - placed));
    }
}

void cum_from_scaled(const uint16_t scaled[256], uint32_t cum[257]) {
    cum[0] = 0;
    for (int i = 0; i < 256; ++i) cum[i + 1] = cum[i] + scaled[i];
}

struct ByteWriter {
    std::vector<uint8_t>& out;
    explicit ByteWriter(std::vector<uint8_t>& o) : out(o) {}
    void push(uint8_t b) { out.push_back(b); }
};

void range_encode_bytes(const uint8_t* data, int len, const uint32_t cum[257], ByteWriter& w) {
    const uint32_t total = cum[256];
    if (total == 0) return;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFu;

    for (int i = 0; i < len; ++i) {
        const unsigned sym = data[i];
        const uint32_t c0 = cum[sym];
        const uint32_t c1 = cum[sym + 1];
        const uint64_t range = static_cast<uint64_t>(high) - low + 1u;
        high = static_cast<uint32_t>(low + (range * static_cast<uint64_t>(c1)) / total - 1u);
        low = static_cast<uint32_t>(low + (range * static_cast<uint64_t>(c0)) / total);

        for (;;) {
            if ((low ^ high) < (1u << 24)) {
                w.push(static_cast<uint8_t>(low >> 24));
                low <<= 8;
                high = (high << 8) | 0xFFu;
            } else
                break;
        }
    }

    for (int k = 0; k < 4; ++k) {
        w.push(static_cast<uint8_t>(low >> 24));
        low <<= 8;
    }
}

struct ByteReader {
    const uint8_t* p;
    int n;
    int pos = 0;
    ByteReader(const uint8_t* ptr, int len) : p(ptr), n(len) {}
    uint8_t get() {
        if (pos >= n) return 0;
        return p[pos++];
    }
};

void range_decode_bytes(ByteReader& r, const uint32_t cum[257], int out_len, uint8_t* out) {
    const uint32_t total = cum[256];
    if (total == 0) return;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFu;
    uint32_t code = 0;
    for (int k = 0; k < 4; ++k) code = (code << 8) | r.get();

    for (int i = 0; i < out_len; ++i) {
        const uint64_t range = static_cast<uint64_t>(high) - low + 1u;
        const uint32_t cum_val =
            static_cast<uint32_t>(((static_cast<uint64_t>(code - low) + 1u) * total - 1u) / range);

        int sym = 0;
        while (cum[sym + 1] <= cum_val) ++sym;

        out[i] = static_cast<uint8_t>(sym);
        const uint32_t c0 = cum[sym];
        const uint32_t c1 = cum[sym + 1];

        high = static_cast<uint32_t>(low + (range * static_cast<uint64_t>(c1)) / total - 1u);
        low = static_cast<uint32_t>(low + (range * static_cast<uint64_t>(c0)) / total);

        for (;;) {
            if ((low ^ high) < (1u << 24)) {
                code = (code << 8) | r.get();
                low <<= 8;
                high = (high << 8) | 0xFFu;
            } else
                break;
        }
    }
}

} // namespace

int arithmetic_order0_encode(const uint8_t* data, int len, std::vector<uint8_t>& out) {
    out.clear();
    constexpr int kHdr = 8 + 512; // magic+len + uint16×256 scaled model
    out.resize(kHdr);
    std::memcpy(out.data(), "ARQ0", 4);
    *reinterpret_cast<int32_t*>(out.data() + 4) = len;

    uint32_t raw[256];
    build_histogram(data, len, raw);
    uint16_t scaled[256];
    scale_to_model(raw, scaled);
    std::memcpy(out.data() + 8, scaled, sizeof(scaled));

    uint32_t cum[257];
    cum_from_scaled(scaled, cum);
    if (cum[256] == 0) {
        return static_cast<int>(out.size());
    }

    ByteWriter w(out);
    range_encode_bytes(data, len, cum, w);
    return static_cast<int>(out.size());
}

int arithmetic_order0_decode(const uint8_t* enc, int enc_len, std::vector<uint8_t>& out) {
    out.clear();
    constexpr int kHdr = 8 + 512;
    if (enc_len < kHdr) return -1;
    if (std::memcmp(enc, "ARQ0", 4) != 0) return -1;
    const int payload_len = *reinterpret_cast<const int32_t*>(enc + 4);
    if (payload_len < 0) return -1;

    uint16_t scaled[256];
    std::memcpy(scaled, enc + 8, sizeof(scaled));

    uint32_t cum[257];
    cum_from_scaled(scaled, cum);
    if (cum[256] == 0) {
        if (payload_len != 0) return -1;
        return 0;
    }

    out.resize(static_cast<size_t>(payload_len));
    ByteReader r(enc + kHdr, enc_len - kHdr);
    range_decode_bytes(r, cum, payload_len, out.data());
    return static_cast<int>(out.size());
}

bool arithmetic_order0_roundtrip_selftest() {
    std::vector<uint8_t> msg(5000);
    for (size_t i = 0; i < msg.size(); ++i) msg[i] = static_cast<uint8_t>((i * 17 + 31) % 251);

    std::vector<uint8_t> enc;
    if (arithmetic_order0_encode(msg.data(), static_cast<int>(msg.size()), enc) < 0) return false;

    std::vector<uint8_t> dec;
    if (arithmetic_order0_decode(enc.data(), static_cast<int>(enc.size()), dec) < 0) return false;
    return dec == msg;
}
