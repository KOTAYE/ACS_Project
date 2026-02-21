#include "huffman.h"

#define BLOCK_BYTES (64 * (int)sizeof(float))
#define NUM_SYMBOLS 256
#define MAX_NODES 512

typedef struct {
    int left;
    int right;
    int symbol;
    uint32_t count;
} Node;

static uint8_t code_len[NUM_SYMBOLS];
static uint32_t code_bits[NUM_SYMBOLS];
static Node nodes[MAX_NODES];
static int num_nodes;

static int build_tree(const uint16_t* freq) {
    num_nodes = 0;
    int roots[256];
    int num_roots = 0;

    for (int i = 0; i < NUM_SYMBOLS; i++) {
        if (freq[i] == 0) continue;
        int id = num_nodes++;
        nodes[id].symbol = i;
        nodes[id].count = freq[i];
        nodes[id].left = -1;
        nodes[id].right = -1;
        roots[num_roots++] = id;
    }

    if (num_roots == 0) return -1;

    while (num_roots > 1) {
        int i0 = 0, i1 = 1;
        if (num_roots > 1 && nodes[roots[1]].count < nodes[roots[0]].count) {
            int t = i0; i0 = i1; i1 = t;
        }
        for (int i = 2; i < num_roots; i++) {
            uint32_t c = nodes[roots[i]].count;
            if (c < nodes[roots[i0]].count) {
                i1 = i0;
                i0 = i;
            } else if (c < nodes[roots[i1]].count) {
                i1 = i;
            }
        }
        int id0 = roots[i0], id1 = roots[i1];
        int parent = num_nodes++;
        nodes[parent].left = id0;
        nodes[parent].right = id1;
        nodes[parent].symbol = -1;
        nodes[parent].count = nodes[id0].count + nodes[id1].count;

        roots[i0] = parent;
        roots[i1] = roots[num_roots - 1];
        num_roots--;
    }
    return roots[0];
}

static void build_codes(int root, int depth, uint32_t bits) {
    if (nodes[root].left < 0 && nodes[root].right < 0) {
        code_len[nodes[root].symbol] = (uint8_t)depth;
        code_bits[nodes[root].symbol] = bits;
        return;
    }
    if (nodes[root].left >= 0)
        build_codes(nodes[root].left, depth + 1, bits << 1);
    if (nodes[root].right >= 0)
        build_codes(nodes[root].right, depth + 1, (bits << 1) | 1);
}

static void write_bit(uint8_t* buf, int* byte_pos, int* bit_pos, int max_bytes, int b) {
    if (*byte_pos >= max_bytes) return;
    if (b)
        buf[*byte_pos] |= (uint8_t)(1u << (7 - *bit_pos));
    (*bit_pos)++;
    if (*bit_pos == 8) {
        *bit_pos = 0;
        (*byte_pos)++;
    }
}

static void write_bits(uint8_t* buf, int* byte_pos, int* bit_pos, int max_bytes, uint32_t bits, int n) {
    for (int i = n - 1; i >= 0; i--)
        write_bit(buf, byte_pos, bit_pos, max_bytes, (int)((bits >> i) & 1));
}

static int read_bit(const uint8_t* buf, int* byte_pos, int* bit_pos, int byte_len) {
    if (*byte_pos >= byte_len) return 0;
    int b = (buf[*byte_pos] >> (7 - *bit_pos)) & 1;
    (*bit_pos)++;
    if (*bit_pos == 8) {
        *bit_pos = 0;
        (*byte_pos)++;
    }
    return b;
}

static int decode_symbol(const uint8_t* buf, int* byte_pos, int* bit_pos, int byte_len, int root) {
    int cur = root;
    while (nodes[cur].left >= 0 || nodes[cur].right >= 0) {
        int b = read_bit(buf, byte_pos, bit_pos, byte_len);
        cur = b ? nodes[cur].right : nodes[cur].left;
        if (cur < 0) return -1;
    }
    return nodes[cur].symbol;
}

int huffman_encode_block_64(const float* block, uint8_t* encoded, int encoded_max_size) {
    const int header_size = NUM_SYMBOLS * (int)sizeof(uint16_t);
    if (encoded_max_size < header_size) return -1;

    uint16_t* freq = (uint16_t*)encoded;
    for (int i = 0; i < NUM_SYMBOLS; i++)
        freq[i] = 0;
    const uint8_t* bytes = (const uint8_t*)block;
    for (int i = 0; i < BLOCK_BYTES; i++)
        freq[bytes[i]]++;

    int root = build_tree(freq);
    if (root < 0) return -1;
    build_codes(root, 0, 0);
    if (nodes[root].left < 0 && nodes[root].right < 0) {
        code_len[nodes[root].symbol] = 1;
        code_bits[nodes[root].symbol] = 0;
    }

    int byte_pos = 0;
    int bit_pos = 0;
    int max_bytes = encoded_max_size - header_size;
    uint8_t* wr_buf = encoded + header_size;

    for (int i = 0; i < BLOCK_BYTES; i++) {
        int s = bytes[i];
        write_bits(wr_buf, &byte_pos, &bit_pos, max_bytes, code_bits[s], code_len[s]);
    }
    if (bit_pos != 0) byte_pos++;
    return header_size + byte_pos;
}

int huffman_decode_block_64(const uint8_t* encoded, int encoded_len, float* block) {
    const int header_size = NUM_SYMBOLS * (int)sizeof(uint16_t);
    if (encoded_len < header_size) return -1;

    const uint16_t* freq = (const uint16_t*)encoded;
    int root = build_tree(freq);
    if (root < 0) return -1;

    int byte_pos = 0;
    int bit_pos = 0;
    const uint8_t* rd_buf = encoded + header_size;
    int rd_len = encoded_len - header_size;
    uint8_t* out = (uint8_t*)block;

    for (int i = 0; i < BLOCK_BYTES; i++) {
        int s = decode_symbol(rd_buf, &byte_pos, &bit_pos, rd_len, root);
        if (s < 0) return -1;
        out[i] = (uint8_t)s;
    }
    return 0;
}
