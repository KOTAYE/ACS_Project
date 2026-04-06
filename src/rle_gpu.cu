#include "rle_gpu.cuh"
#include "huffman.h"

#include <cub/cub.cuh>
#include <cstring>
#include <iostream>
// Per-block Huffman bitstream: Step A = HuffmanBlockBitLengthKernel, Step B = CUB ExclusiveSum,
// Step C = HuffmanPackAllBlocksSerialKernel (see docs/GPU_HUFFMAN_BITSTREAM.md).

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit((int)err); \
        } \
    } while (0)

struct RleContext {
    int16_t* d_unique = nullptr;
    int* d_counts = nullptr;
    int* d_num_runs_out = nullptr;
    int* d_sizes = nullptr;
    int* d_offsets = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int16_t* d_final_out = nullptr;
    size_t capacity = 0;

    int*      d_block_rle_counts = nullptr;
    int*      d_block_rle_offsets = nullptr;
    uint32_t* d_block_bit_lengths = nullptr;
    uint32_t* d_block_bit_offsets = nullptr;
    
    uint32_t* d_code_bits = nullptr;
    uint8_t*  d_code_lens = nullptr;
    uint8_t*  d_bitstream = nullptr;
    size_t    bitstream_cap = 0;
    uint32_t* d_hist = nullptr;

    size_t rle_len_bytes = 0;

    // Decode-specific fields
    GpuHuffNode* d_decode_tree = nullptr;
    uint8_t*     d_decode_packed = nullptr;
    size_t       decode_packed_cap = 0;
    int16_t*     d_decode_rle_buf = nullptr;
    size_t       decode_rle_buf_cap = 0;
    int*         d_decode_rle_counts = nullptr;
    size_t       decode_rle_counts_cap = 0;
};

struct HuffTreeBuildNode {
    int left;
    int right;
    int symbol;
    uint32_t count;
};

// Same tree shape as huffman.cpp build_tree(freq) for uint16_t frequencies.
static int build_huffman_tree_u16(const uint16_t* freq, HuffTreeBuildNode* nodes, int* out_num_nodes) {
    int num_nodes = 0;
    int roots[256];
    int num_roots = 0;

    for (int i = 0; i < 256; ++i) {
        if (freq[i] == 0) continue;
        const int id = num_nodes++;
        nodes[id] = { -1, -1, i, freq[i] };
        roots[num_roots++] = id;
    }

    if (num_roots == 0) {
        *out_num_nodes = 0;
        return -1;
    }

    while (num_roots > 1) {
        int i0 = 0, i1 = 1;
        if (nodes[roots[i1]].count < nodes[roots[i0]].count) {
            const int t = i0;
            i0 = i1;
            i1 = t;
        }

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

    *out_num_nodes = num_nodes;
    return roots[0];
}

static void ht_to_gpu_nodes(const HuffTreeBuildNode* nodes, int num_nodes, GpuHuffNode* out) {
    for (int i = 0; i < num_nodes; ++i) {
        if (nodes[i].left < 0 && nodes[i].right < 0) {
            out[i].children[0] = -1;
            out[i].children[1] = -1;
            out[i].symbol = static_cast<int16_t>(nodes[i].symbol);
        } else {
            out[i].children[0] = static_cast<int16_t>(nodes[i].left);
            out[i].children[1] = static_cast<int16_t>(nodes[i].right);
            out[i].symbol = -1;
        }
    }
}

// ----------------------------------------------------------------------------
// Step 5: Per-block Kernels
// ----------------------------------------------------------------------------

__global__ void RleCountPerBlockKernel(const int16_t* in, int num_blocks, int block_size_sq, int* counts) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const int16_t* b_ptr = &in[bid * block_size_sq];
    int output_elems = 0;
    int i = 0;
    while (i < block_size_sq) {
        if (b_ptr[i] == 0) {
            int run = 1;
            while (i + run < block_size_sq && b_ptr[i + run] == 0 && run < 32767) run++;
            output_elems += 2; // zero marker + run length
            i += run;
        } else {
            output_elems += 1; // single non-zero value
            i++;
        }
    }
    counts[bid] = output_elems;
}

__global__ void RleScatterPerBlockKernel(const int16_t* in, int num_blocks, int block_size_sq, 
                                         const int* offsets, int16_t* out) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const int16_t* b_ptr = &in[bid * block_size_sq];
    int16_t* o_ptr = &out[offsets[bid]];
    int i = 0;
    while (i < block_size_sq) {
        if (b_ptr[i] == 0) {
            int run = 1;
            while (i + run < block_size_sq && b_ptr[i + run] == 0 && run < 32767) run++;
            *o_ptr++ = 0;
            *o_ptr++ = (int16_t)run;
            i += run;
        } else {
            *o_ptr++ = b_ptr[i];
            i++;
        }
    }
}

__global__ void HuffmanBlockBitLengthKernel(const int16_t* rle_in, const int* rle_offsets, const int* rle_counts,
                                            int num_blocks, const uint8_t* lens, uint32_t* bit_lengths) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    int r_off = rle_offsets[bid];
    int r_cnt = rle_counts[bid];
    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(&rle_in[r_off]);
    const int n_bytes = r_cnt * static_cast<int>(sizeof(int16_t));

    uint32_t total_bits = 0;
    for (int i = 0; i < n_bytes; ++i)
        total_bits += lens[b_ptr[i]];
    bit_lengths[bid] = total_bits;
}

// Same bit layout as huffman.cpp write_bits / write_bit (MSB of codeword first, bit index within stream).
__device__ void dev_write_huff_bits(uint8_t* out, uint32_t& global_bit_off, uint32_t code, int n) {
    for (int i = n - 1; i >= 0; --i) {
        const int bit = (int)((code >> i) & 1u);
        const uint32_t pos = global_bit_off++;
        const int bi = static_cast<int>(pos >> 3);
        const int bo = 7 - (static_cast<int>(pos) & 7);
        if (bit) out[bi] |= static_cast<uint8_t>(1u << bo);
    }
}

// One thread packs all blocks in order. Parallel <<<num_blocks,1>>> races when adjacent blocks share a byte
// (bit ranges are disjoint but byte writes use |= without atomics).
__global__ void HuffmanPackAllBlocksSerialKernel(const int16_t* rle_in, const int* rle_offsets, const int* rle_counts,
                                                 int num_blocks, const uint32_t* bits, const uint8_t* lens,
                                                 const uint32_t* bit_offsets, uint8_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int bid = 0; bid < num_blocks; ++bid) {
        const int r_off = rle_offsets[bid];
        const int r_cnt = rle_counts[bid];
        const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(&rle_in[r_off]);
        uint32_t global_bit_off = bit_offsets[bid];

        for (int i = 0; i < r_cnt * (int)sizeof(int16_t); ++i) {
            const uint8_t sym = b_ptr[i];
            const uint32_t b = bits[sym];
            const int l = (int)lens[sym];
            if (l <= 0) continue;
            dev_write_huff_bits(out, global_bit_off, b, l);
        }
    }
}

__global__ void HuffmanHistKernel(const uint8_t* in, int n, uint32_t* hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[in[i]], 1);
}

// ─── GPU Decode Kernels ─────────────────────────────────────────────────

__device__ int gpu_read_bit(const uint8_t* stream, uint32_t bit_pos) {
    return (stream[bit_pos >> 3] >> (7 - (bit_pos & 7))) & 1;
}

__device__ bool gpu_huff_is_leaf(const GpuHuffNode* tree, int n) {
    return tree[n].children[0] < 0 && tree[n].children[1] < 0;
}

// Matches huffman.cpp huffman_decode_bit_window: root-leaf consumes code_len bits (forced to 1 for single-symbol tree).
__global__ void HuffmanDecodePerBlockKernel(
    const uint8_t* __restrict__ bitstream,
    const uint32_t* __restrict__ bit_starts,
    const uint32_t* __restrict__ bit_lengths,
    int num_blocks,
    const GpuHuffNode* __restrict__ tree,
    int tree_root,
    const uint8_t* __restrict__ code_lens,
    int16_t* __restrict__ rle_out,
    int max_rle_elems_per_block,
    int* __restrict__ rle_elem_counts)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const uint32_t b_start = bit_starts[bid];
    const int b_len = static_cast<int>(bit_lengths[bid]);
    int16_t* out = &rle_out[static_cast<size_t>(bid) * static_cast<size_t>(max_rle_elems_per_block)];

    int bits_used = 0;
    int out_idx = 0;
    uint8_t byte_pair[2] = {0, 0};
    int byte_idx = 0;

    while (bits_used < b_len && out_idx < max_rle_elems_per_block) {
        int cur = tree_root;
        if (gpu_huff_is_leaf(tree, cur)) {
            const int sym = static_cast<int>(tree[cur].symbol);
            const int L = static_cast<int>(code_lens[sym]);
            if (L <= 0) {
                rle_elem_counts[bid] = 0;
                return;
            }
            if (bits_used + L > b_len) {
                rle_elem_counts[bid] = 0;
                return;
            }
            for (int k = 0; k < L; ++k) {
                (void)gpu_read_bit(bitstream, b_start + static_cast<uint32_t>(bits_used));
                ++bits_used;
            }
            byte_pair[byte_idx++] = static_cast<uint8_t>(sym);
            if (byte_idx == 2) {
                out[out_idx++] =
                    static_cast<int16_t>(static_cast<uint16_t>(byte_pair[0]) |
                                         (static_cast<uint16_t>(byte_pair[1]) << 8));
                byte_idx = 0;
            }
        } else {
            while (!gpu_huff_is_leaf(tree, cur)) {
                if (bits_used >= b_len) {
                    rle_elem_counts[bid] = 0;
                    return;
                }
                const int bit = gpu_read_bit(bitstream, b_start + static_cast<uint32_t>(bits_used));
                ++bits_used;
                cur = bit ? static_cast<int>(tree[cur].children[1]) : static_cast<int>(tree[cur].children[0]);
                if (cur < 0) {
                    rle_elem_counts[bid] = 0;
                    return;
                }
            }
            const int sym = static_cast<int>(tree[cur].symbol);
            byte_pair[byte_idx++] = static_cast<uint8_t>(sym);
            if (byte_idx == 2) {
                out[out_idx++] =
                    static_cast<int16_t>(static_cast<uint16_t>(byte_pair[0]) |
                                         (static_cast<uint16_t>(byte_pair[1]) << 8));
                byte_idx = 0;
            }
        }
    }

    if (bits_used != b_len || byte_idx != 0) {
        rle_elem_counts[bid] = 0;
        return;
    }
    rle_elem_counts[bid] = out_idx;
}

__global__ void RleDecodePerBlockKernel(
    const int16_t* __restrict__ rle_in,
    const int* __restrict__ rle_elem_counts,
    int max_rle_elems_per_block,
    int num_blocks,
    int block_size_sq,
    int16_t* __restrict__ coeffs_out)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const int16_t* in = &rle_in[bid * max_rle_elems_per_block];
    int16_t* out = &coeffs_out[bid * block_size_sq];
    int in_count = rle_elem_counts[bid];

    int in_idx = 0, out_idx = 0;
    while (in_idx < in_count && out_idx < block_size_sq) {
        if (in[in_idx] == 0) {
            if (in_idx + 1 >= in_count) break;
            int run = in[in_idx + 1];
            for (int k = 0; k < run && out_idx < block_size_sq; ++k)
                out[out_idx++] = 0;
            in_idx += 2;
        } else {
            out[out_idx++] = in[in_idx];
            in_idx++;
        }
    }
    while (out_idx < block_size_sq) out[out_idx++] = 0;
}

static RleContext* g_rle_ctx[3] = {nullptr, nullptr, nullptr};

void rle_gpu_init(int ch, size_t max_elements) {
    if (!g_rle_ctx[ch]) g_rle_ctx[ch] = new RleContext();
    auto& ctx = *g_rle_ctx[ch];
    if (ctx.capacity >= max_elements) return;
    
    if (ctx.capacity > 0) {
        cudaFree(ctx.d_unique); cudaFree(ctx.d_counts); cudaFree(ctx.d_sizes);
        cudaFree(ctx.d_offsets); cudaFree(ctx.d_num_runs_out);
        cudaFree(ctx.d_temp_storage); cudaFree(ctx.d_final_out);
        cudaFree(ctx.d_code_bits); cudaFree(ctx.d_code_lens);
        cudaFree(ctx.d_bitstream); ctx.bitstream_cap = 0;
        cudaFree(ctx.d_hist);
        cudaFree(ctx.d_block_rle_counts);
        cudaFree(ctx.d_block_rle_offsets);
        cudaFree(ctx.d_block_bit_lengths);
        cudaFree(ctx.d_block_bit_offsets);
        cudaFree(ctx.d_decode_tree);
        cudaFree(ctx.d_decode_packed);
        cudaFree(ctx.d_decode_rle_buf);
        cudaFree(ctx.d_decode_rle_counts);
        ctx.d_decode_tree = nullptr;
        ctx.d_decode_packed = nullptr;
        ctx.decode_packed_cap = 0;
        ctx.d_decode_rle_buf = nullptr;
        ctx.decode_rle_buf_cap = 0;
        ctx.d_decode_rle_counts = nullptr;
        ctx.decode_rle_counts_cap = 0;
    }
    
    ctx.capacity = max_elements;
    int max_blocks = (max_elements + 63) / 64; // Approx for 8x8 blocks

    CUDA_CHECK(cudaMalloc(&ctx.d_unique, max_elements * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_counts, max_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sizes, max_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_offsets, max_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_num_runs_out, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_final_out, 2 * max_elements * sizeof(int16_t)));

    CUDA_CHECK(cudaMalloc(&ctx.d_code_bits, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_code_lens, 256 * sizeof(uint8_t)));
    {
        size_t bs_alloc = (2 * max_elements * sizeof(uint8_t) + 3) & ~3;
        CUDA_CHECK(cudaMalloc(&ctx.d_bitstream, bs_alloc));
        ctx.bitstream_cap = bs_alloc;
    }
    CUDA_CHECK(cudaMalloc(&ctx.d_hist, 256 * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&ctx.d_block_rle_counts, max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_rle_offsets, max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_bit_lengths, max_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_bit_offsets, max_blocks * sizeof(uint32_t)));

    constexpr int kGpuHuffNodes = 512;
    CUDA_CHECK(cudaMalloc(&ctx.d_decode_tree, static_cast<size_t>(kGpuHuffNodes) * sizeof(GpuHuffNode)));
    constexpr size_t kInitialPacked = 1u << 20;
    CUDA_CHECK(cudaMalloc(&ctx.d_decode_packed, kInitialPacked));
    ctx.decode_packed_cap = kInitialPacked;

    // We'll use ExclusiveSum for various things, so find max required temp storage
    size_t rb1 = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, rb1, (int*)nullptr, (int*)nullptr, (int)max_elements, nullptr);
    size_t rb2 = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, rb2, (uint32_t*)nullptr, (uint32_t*)nullptr, (int)max_blocks, nullptr);
    
    ctx.temp_storage_bytes = (rb1 > rb2 ? rb1 : rb2) + 16384;
    CUDA_CHECK(cudaMalloc(&ctx.d_temp_storage, ctx.temp_storage_bytes));
}

void rle_gpu_cleanup() {
    for (int ch = 0; ch < 3; ++ch) {
        if (g_rle_ctx[ch]) {
            auto& ctx = *g_rle_ctx[ch];
            if (ctx.capacity > 0) {
                cudaFree(ctx.d_unique); cudaFree(ctx.d_counts); cudaFree(ctx.d_sizes);
                cudaFree(ctx.d_offsets); cudaFree(ctx.d_num_runs_out);
                cudaFree(ctx.d_temp_storage); cudaFree(ctx.d_final_out);
                cudaFree(ctx.d_code_bits); cudaFree(ctx.d_code_lens);
                cudaFree(ctx.d_bitstream); ctx.bitstream_cap = 0;
                cudaFree(ctx.d_hist);
                cudaFree(ctx.d_block_rle_counts);
                cudaFree(ctx.d_block_rle_offsets);
                cudaFree(ctx.d_block_bit_lengths);
                cudaFree(ctx.d_block_bit_offsets);
                cudaFree(ctx.d_decode_tree);
                cudaFree(ctx.d_decode_packed);
                cudaFree(ctx.d_decode_rle_buf);
                cudaFree(ctx.d_decode_rle_counts);
            }
            delete g_rle_ctx[ch];
            g_rle_ctx[ch] = nullptr;
        }
    }
}

void cuda_compute_histogram(int ch, uint32_t* h_hist, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    auto& ctx = *g_rle_ctx[ch];
    if (ctx.rle_len_bytes == 0) {
        std::memset(h_hist, 0, 256 * sizeof(uint32_t));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(ctx.d_hist, 0, 256 * sizeof(uint32_t), stream));
    
    int threads = 256;
    int blocks = (ctx.rle_len_bytes + threads - 1) / threads;
    HuffmanHistKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint8_t*>(ctx.d_final_out), (int)ctx.rle_len_bytes, ctx.d_hist);
    CUDA_CHECK(cudaMemcpyAsync(h_hist, ctx.d_hist, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
}

void cuda_rle_encode_indexed(int ch, const int16_t* d_coeffs, int num_blocks, int block_size,
                             uint32_t* out_rle_bytes, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    auto& ctx = *g_rle_ctx[ch];
    int bs2 = block_size * block_size;

    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    RleCountPerBlockKernel<<<blocks, threads, 0, stream>>>(d_coeffs, num_blocks, bs2, ctx.d_block_rle_counts);

    size_t tb = ctx.temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(ctx.d_temp_storage, tb, ctx.d_block_rle_counts, ctx.d_block_rle_offsets, num_blocks, stream);

    RleScatterPerBlockKernel<<<blocks, threads, 0, stream>>>(d_coeffs, num_blocks, bs2, ctx.d_block_rle_offsets, ctx.d_final_out);

    uint32_t last_cnt, last_off;
    CUDA_CHECK(cudaMemcpyAsync(&last_cnt, &ctx.d_block_rle_counts[num_blocks - 1], 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_off, &ctx.d_block_rle_offsets[num_blocks - 1], 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ctx.rle_len_bytes = (last_off + last_cnt) * sizeof(int16_t);
    if (out_rle_bytes) *out_rle_bytes = (uint32_t)ctx.rle_len_bytes;
}

void cuda_rle_download_to_host(int ch, void* dst, size_t nbytes) {
    auto& ctx = *g_rle_ctx[ch];
    if (nbytes == 0 || !dst) return;
    size_t avail = ctx.rle_len_bytes;
    if (nbytes > avail) {
        std::cerr << "cuda_rle_download_to_host: nbytes " << nbytes << " > rle_len " << avail << "\n";
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(dst, ctx.d_final_out, nbytes, cudaMemcpyDeviceToHost));
}

void cuda_huffman_pack_gpu_indexed(int ch, int num_blocks,
                                   const uint32_t* h_code_bits, const uint8_t* h_code_lens,
                                   uint8_t** d_out_packed, size_t* out_bytes,
                                   uint32_t* d_block_bit_lengths_out,
                                   void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    auto& ctx = *g_rle_ctx[ch];

    if (num_blocks <= 0) {
        *out_bytes = 0;
        *d_out_packed = ctx.d_bitstream;
        return;
    }

    if (ctx.rle_len_bytes == 0) {
        CUDA_CHECK(cudaMemsetAsync(ctx.d_block_bit_lengths, 0, static_cast<size_t>(num_blocks) * sizeof(uint32_t),
                                   stream));
        if (d_block_bit_lengths_out && d_block_bit_lengths_out != ctx.d_block_bit_lengths) {
            CUDA_CHECK(cudaMemcpyAsync(d_block_bit_lengths_out, ctx.d_block_bit_lengths,
                                     static_cast<size_t>(num_blocks) * sizeof(uint32_t),
                                     cudaMemcpyDeviceToDevice, stream));
        }
        *out_bytes = 0;
        *d_out_packed = ctx.d_bitstream;
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_bits, h_code_bits, 256 * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_lens, h_code_lens, 256, cudaMemcpyHostToDevice, stream));

    const int threads = 256;
    const int blocks_k = (num_blocks + threads - 1) / threads;
    HuffmanBlockBitLengthKernel<<<blocks_k, threads, 0, stream>>>(
        ctx.d_final_out, ctx.d_block_rle_offsets, ctx.d_block_rle_counts, num_blocks, ctx.d_code_lens,
        ctx.d_block_bit_lengths);

    size_t tb = ctx.temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(ctx.d_temp_storage, tb, ctx.d_block_bit_lengths, ctx.d_block_bit_offsets,
                                  num_blocks, stream);

    uint32_t last_off = 0, last_len = 0;
    CUDA_CHECK(cudaMemcpyAsync(&last_off, &ctx.d_block_bit_offsets[num_blocks - 1], sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_len, &ctx.d_block_bit_lengths[num_blocks - 1], sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const uint64_t total_bits_u64 = static_cast<uint64_t>(last_off) + static_cast<uint64_t>(last_len);
    const size_t need_bytes = static_cast<size_t>((total_bits_u64 + 7u) / 8u);
    *out_bytes = need_bytes;

    if (need_bytes > ctx.bitstream_cap) {
        if (ctx.d_bitstream) CUDA_CHECK(cudaFree(ctx.d_bitstream));
        ctx.bitstream_cap = need_bytes + (need_bytes / 2 > 65536 ? need_bytes / 2 : 65536);
        CUDA_CHECK(cudaMalloc(&ctx.d_bitstream, ctx.bitstream_cap));
    }

    if (need_bytes > 0)
        CUDA_CHECK(cudaMemsetAsync(ctx.d_bitstream, 0, need_bytes, stream));

    HuffmanPackAllBlocksSerialKernel<<<1, 1, 0, stream>>>(
        ctx.d_final_out, ctx.d_block_rle_offsets, ctx.d_block_rle_counts, num_blocks, ctx.d_code_bits,
        ctx.d_code_lens, ctx.d_block_bit_offsets, ctx.d_bitstream);

    if (d_block_bit_lengths_out && d_block_bit_lengths_out != ctx.d_block_bit_lengths) {
        CUDA_CHECK(cudaMemcpyAsync(d_block_bit_lengths_out, ctx.d_block_bit_lengths,
                                 static_cast<size_t>(num_blocks) * sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                                 stream));
    }

    *d_out_packed = ctx.d_bitstream;
}

void cuda_huffman_download_block_bit_lengths(int ch, uint32_t* h_dst, int num_blocks) {
    auto& ctx = *g_rle_ctx[ch];
    if (!h_dst || num_blocks <= 0) return;
    CUDA_CHECK(cudaMemcpy(h_dst, ctx.d_block_bit_lengths, (size_t)num_blocks * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}

void cuda_gpu_decode_entropy(int ch,
                              const uint8_t* h_packed_data, size_t packed_bytes,
                              const uint32_t* h_block_bit_lengths, int num_blocks,
                              const uint16_t* h_freq, int block_size,
                              void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    auto& ctx = *g_rle_ctx[ch];
    const int bs2 = block_size * block_size;
    const int max_rle_per_block = bs2;

    if (num_blocks <= 0) return;

    size_t rle_buf_needed = static_cast<size_t>(num_blocks) * static_cast<size_t>(max_rle_per_block) *
                            sizeof(int16_t);
    if (rle_buf_needed > ctx.decode_rle_buf_cap) {
        if (ctx.d_decode_rle_buf) cudaFree(ctx.d_decode_rle_buf);
        ctx.decode_rle_buf_cap = rle_buf_needed;
        CUDA_CHECK(cudaMalloc(&ctx.d_decode_rle_buf, rle_buf_needed));
    }
    const size_t counts_needed = static_cast<size_t>(num_blocks) * sizeof(int);
    if (counts_needed > ctx.decode_rle_counts_cap) {
        if (ctx.d_decode_rle_counts) cudaFree(ctx.d_decode_rle_counts);
        ctx.decode_rle_counts_cap = counts_needed;
        CUDA_CHECK(cudaMalloc(&ctx.d_decode_rle_counts, counts_needed));
    }

    HuffTreeBuildNode hnodes[512];
    int hn = 0;
    const int root = build_huffman_tree_u16(h_freq, hnodes, &hn);

    if (root < 0 || hn <= 0) {
        CUDA_CHECK(cudaMemsetAsync(ctx.d_decode_rle_counts, 0, counts_needed, stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.d_final_out, 0,
                                   static_cast<size_t>(num_blocks) * static_cast<size_t>(bs2) * sizeof(int16_t),
                                   stream));
        return;
    }

    GpuHuffNode gpu_nodes[512];
    ht_to_gpu_nodes(hnodes, hn, gpu_nodes);

    uint32_t dummy_bits[256];
    uint8_t h_code_lens[256];
    if (huffman_codebook_from_freq16(h_freq, dummy_bits, h_code_lens) != 0) {
        CUDA_CHECK(cudaMemsetAsync(ctx.d_decode_rle_counts, 0, counts_needed, stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.d_final_out, 0,
                                   static_cast<size_t>(num_blocks) * static_cast<size_t>(bs2) * sizeof(int16_t),
                                   stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_decode_tree, gpu_nodes, static_cast<size_t>(hn) * sizeof(GpuHuffNode),
                             cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_lens, h_code_lens, 256, cudaMemcpyHostToDevice, stream));

    if (packed_bytes > ctx.decode_packed_cap) {
        if (ctx.d_decode_packed) CUDA_CHECK(cudaFree(ctx.d_decode_packed));
        const size_t grow = packed_bytes + (packed_bytes / 2 > (1u << 20) ? packed_bytes / 2 : (1u << 20));
        ctx.decode_packed_cap = grow;
        CUDA_CHECK(cudaMalloc(&ctx.d_decode_packed, ctx.decode_packed_cap));
    }
    if (packed_bytes > 0)
        CUDA_CHECK(
            cudaMemcpyAsync(ctx.d_decode_packed, h_packed_data, packed_bytes, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_block_bit_lengths, h_block_bit_lengths,
                             static_cast<size_t>(num_blocks) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    size_t tb = ctx.temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(ctx.d_temp_storage, tb, ctx.d_block_bit_lengths, ctx.d_block_bit_offsets,
                                num_blocks, stream);

    const int threads = 256;
    const int blocks_k = (num_blocks + threads - 1) / threads;
    HuffmanDecodePerBlockKernel<<<blocks_k, threads, 0, stream>>>(
        ctx.d_decode_packed, ctx.d_block_bit_offsets, ctx.d_block_bit_lengths, num_blocks, ctx.d_decode_tree, root,
        ctx.d_code_lens, ctx.d_decode_rle_buf, max_rle_per_block, ctx.d_decode_rle_counts);

    CUDA_CHECK(cudaMemsetAsync(ctx.d_final_out, 0,
                             static_cast<size_t>(num_blocks) * static_cast<size_t>(bs2) * sizeof(int16_t),
                             stream));
    RleDecodePerBlockKernel<<<blocks_k, threads, 0, stream>>>(ctx.d_decode_rle_buf, ctx.d_decode_rle_counts,
                                                             max_rle_per_block, num_blocks, bs2, ctx.d_final_out);
    CUDA_CHECK(cudaGetLastError());
}

int16_t* cuda_get_decoded_coeffs(int ch) {
    return g_rle_ctx[ch] ? g_rle_ctx[ch]->d_final_out : nullptr;
}
