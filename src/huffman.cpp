#include "huffman.h"

#include <algorithm>
#include <array>
#include <cstring>

static constexpr int kNumSymbols = 256;
static constexpr int kMaxNodes = 512;

struct Node {
    int left;
    int right;
    int symbol;
    uint32_t count;
};

static std::array<uint8_t, kNumSymbols> code_len;
static std::array<uint32_t, kNumSymbols> code_bits;
static std::array<Node, kMaxNodes> nodes;
static int num_nodes;

static int build_tree(const uint32_t* freq) {
    num_nodes = 0;
    std::array<int, kNumSymbols> roots;
    int num_roots = 0;

    for (int i = 0; i < kNumSymbols; ++i) {
        if (freq[i] == 0) continue;
        const int id = num_nodes++;
        nodes[id] = { -1, -1, i, freq[i] };
        roots[num_roots++] = id;
    }

    if (num_roots == 0) return -1;

    while (num_roots > 1) {
        int i0 = 0, i1 = 1;
        if (nodes[roots[i1]].count < nodes[roots[i0]].count) std::swap(i0, i1);

        for (int i = 2; i < num_roots; ++i) {
            const uint32_t c = nodes[roots[i]].count;
            if (c < nodes[roots[i0]].count) {
                i1 = i0;
                i0 = i;
            } else if (c < nodes[roots[i1]].count) {
                i1 = i;
            }
        }

        const int id0 = roots[i0];
        const int id1 = roots[i1];
        const int parent = num_nodes++;
        nodes[parent] = { id0, id1, -1, nodes[id0].count + nodes[id1].count };

        roots[i0] = parent;
        roots[i1] = roots[num_roots - 1];
        --num_roots;
    }
    return roots[0];
}

struct CodeFrame { int node; int depth; uint32_t bits; };

static void build_codes(int root) {
    std::array<CodeFrame, kMaxNodes> stack;
    int top = 0;
    stack[top++] = { root, 0, 0u };

    while (top > 0) {
        const auto [node, depth, bits] = stack[--top];
        const Node& n = nodes[node];

        if (n.left < 0 && n.right < 0) {
            code_len [n.symbol] = static_cast<uint8_t>(depth);
            code_bits[n.symbol] = bits;
            continue;
        }
        if (n.right >= 0) {
            stack[top++] = { n.right, depth + 1, (bits << 1) | 1u };
        }
        if (n.left  >= 0) {
            stack[top++] = { n.left,  depth + 1, bits << 1 };
        }
    }
}

static void write_bit(uint8_t* buf, int& byte_pos, int& bit_pos, int max_bytes, bool b) {
    if (byte_pos >= max_bytes) return;
    if (b) buf[byte_pos] |= static_cast<uint8_t>(1u << (7 - bit_pos));
    if (++bit_pos == 8) { bit_pos = 0; ++byte_pos; }
}

static void write_bits(uint8_t* buf, int& byte_pos, int& bit_pos,
                       int max_bytes, uint32_t bits, int n) {
    for (int i = n - 1; i >= 0; --i)
        write_bit(buf, byte_pos, bit_pos, max_bytes, static_cast<bool>((bits >> i) & 1u));
}

static int read_bit(const uint8_t* buf, int& byte_pos, int& bit_pos, int byte_len) {
    if (byte_pos >= byte_len) return 0;
    const int b = (buf[byte_pos] >> (7 - bit_pos)) & 1;
    if (++bit_pos == 8) { bit_pos = 0; ++byte_pos; }
    return b;
}

static int decode_symbol(const uint8_t* buf, int& byte_pos, int& bit_pos,
                         int byte_len, int root) {
    int cur = root;
    if (nodes[cur].left < 0 && nodes[cur].right < 0) {
        const int sym = nodes[cur].symbol;
        for (int k = 0; k < static_cast<int>(code_len[sym]); ++k)
            (void)read_bit(buf, byte_pos, bit_pos, byte_len);
        return sym;
    }
    while (nodes[cur].left >= 0 || nodes[cur].right >= 0) {
        const int b = read_bit(buf, byte_pos, bit_pos, byte_len);
        cur = b ? nodes[cur].right : nodes[cur].left;
        if (cur < 0) return -1;
    }
    return nodes[cur].symbol;
}

static int read_bit_abs(const uint8_t* packed, int packed_len_bytes, int abs_bit) {
    const int bi = abs_bit >> 3;
    if (bi >= packed_len_bytes) return 0;
    return (packed[bi] >> (7 - (abs_bit & 7))) & 1;
}

int huffman_encode_bytes(const uint8_t* in_data, int in_len, uint8_t* encoded, int encoded_max_size) {
    constexpr int header_size = kNumSymbols * static_cast<int>(sizeof(uint32_t));
    if (encoded_max_size < header_size) return -1;

    auto* freq = reinterpret_cast<uint32_t*>(encoded);
    std::fill(freq, freq + kNumSymbols, 0u);
    for (int i = 0; i < in_len; ++i) ++freq[in_data[i]];

    const int root = build_tree(freq);
    if (root < 0) return -1;
    build_codes(root);

    if (nodes[root].left < 0 && nodes[root].right < 0) {
        code_len [nodes[root].symbol] = 1;
        code_bits[nodes[root].symbol] = 0;
    }

    const int max_bytes = encoded_max_size - header_size;
    uint8_t* wr_buf = encoded + header_size;
    std::memset(wr_buf, 0, static_cast<std::size_t>(max_bytes));

    int byte_pos = 0, bit_pos = 0;
    for (int i = 0; i < in_len; ++i) {
        const int s = in_data[i];
        write_bits(wr_buf, byte_pos, bit_pos, max_bytes, code_bits[s], code_len[s]);
        if (byte_pos >= max_bytes) return -1;
    }
    if (bit_pos != 0) ++byte_pos;
    return header_size + byte_pos;
}

int huffman_codebook_from_freq32(const uint32_t freq[256], uint32_t out_bits[256], uint8_t out_lens[256]) {
    const int root = build_tree(freq);
    if (root < 0) {
        std::memset(out_bits, 0, 256 * sizeof(uint32_t));
        std::memset(out_lens, 0, 256);
        return -1;
    }
    code_len.fill(0);
    code_bits.fill(0);
    build_codes(root);
    if (nodes[root].left < 0 && nodes[root].right < 0) {
        code_len [nodes[root].symbol] = 1;
        code_bits[nodes[root].symbol] = 0;
    }
    std::memcpy(out_bits, code_bits.data(), 256 * sizeof(uint32_t));
    std::memcpy(out_lens, code_len.data(), 256 * sizeof(uint8_t));
    return 0;
}

int huffman_decode_bit_window(const uint32_t freq[256], const uint8_t* packed, int packed_len_bytes,
                              int bit_start, int num_bits, uint8_t* out, int out_cap) {
    const int root = build_tree(freq);
    if (root < 0) return -1;
    code_len.fill(0);
    code_bits.fill(0);
    build_codes(root);

    if (nodes[root].left < 0 && nodes[root].right < 0) {
        code_len [nodes[root].symbol] = 1;
        code_bits[nodes[root].symbol] = 0;
    }

    int bits_used = 0;
    int out_len = 0;
    while (bits_used < num_bits) {
        int cur = root;
        if (nodes[cur].left < 0 && nodes[cur].right < 0) {
            const int sym = nodes[cur].symbol;
            const int L = static_cast<int>(code_len[sym]);
            if (bits_used + L > num_bits) return -1;
            for (int k = 0; k < L; ++k) {
                (void)read_bit_abs(packed, packed_len_bytes, bit_start + bits_used);
                ++bits_used;
            }
            if (out_len >= out_cap) return -1;
            out[out_len++] = static_cast<uint8_t>(sym);
            continue;
        }
        while (nodes[cur].left >= 0 || nodes[cur].right >= 0) {
            if (bits_used >= num_bits) return -1;
            const int b = read_bit_abs(packed, packed_len_bytes, bit_start + bits_used);
            ++bits_used;
            cur = b ? nodes[cur].right : nodes[cur].left;
            if (cur < 0) return -1;
        }
        if (out_len >= out_cap) return -1;
        out[out_len++] = static_cast<uint8_t>(nodes[cur].symbol);
    }
    if (bits_used != num_bits) return -1;
    return out_len;
}

int huffman_decode_bytes(const uint8_t* encoded, int encoded_len, uint8_t* out_data, int out_len) {
    constexpr int header_size = kNumSymbols * static_cast<int>(sizeof(uint32_t));
    if (encoded_len < header_size) return -1;

    const auto* freq = reinterpret_cast<const uint32_t*>(encoded);
    const int root = build_tree(freq);
    if (root < 0) return -1;
    build_codes(root);
    if (nodes[root].left < 0 && nodes[root].right < 0) {
        code_len [nodes[root].symbol] = 1;
        code_bits[nodes[root].symbol] = 0;
    }

    int byte_pos = 0, bit_pos = 0;
    const uint8_t* rd_buf = encoded + header_size;
    const int rd_len = encoded_len - header_size;

    for (int i = 0; i < out_len; ++i) {
        const int s = decode_symbol(rd_buf, byte_pos, bit_pos, rd_len, root);
        if (s < 0) return -1;
        out_data[i] = static_cast<uint8_t>(s);
    }
    return 0;
}
