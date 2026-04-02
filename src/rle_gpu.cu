#include "rle_gpu.cuh"
#include "huffman.h"

#include <cub/cub.cuh>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " " \
                      << cudaGetErrorString(err) << "\n";                      \
            exit(static_cast<int>(err));                                       \
        }                                                                      \
    } while (0)

template <typename T>
static void cuda_free_ptr(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

struct GpuMetadata {
    uint32_t rle_bytes;
    uint32_t pack_bytes;
    uint32_t num_blocks;
};

struct RleContext {
    int16_t* d_rle = nullptr;
    size_t rle_cap = 0;

    int* d_block_rle_counts = nullptr;
    int* d_block_rle_offsets = nullptr;

    uint32_t* d_code_bits = nullptr;
    uint8_t* d_code_lens = nullptr;

    uint32_t* d_block_bit_lengths = nullptr;
    uint32_t* d_block_bit_offsets = nullptr;
    uint8_t* d_bitstream = nullptr;
    size_t bitstream_cap = 0;

    uint32_t* d_hist = nullptr;

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int max_blocks = 0;
    int last_num_blocks = 0;
};

static RleContext* g_ctx[3] = {nullptr, nullptr, nullptr};

static void rle_count_block(const int16_t* blk, int bs2, int& out_elems) {
    out_elems = 0;
    int i = 0;
    while (i < bs2) {
        if (blk[i] == 0) {
            int run = 1;
            while (i + run < bs2 && blk[i + run] == 0 && run < 32767) ++run;
            out_elems += 2;
            i += run;
        } else {
            out_elems += 1;
            ++i;
        }
    }
}

static void rle_scatter_block(const int16_t* blk, int bs2, int16_t* out) {
    int i = 0;
    while (i < bs2) {
        if (blk[i] == 0) {
            int run = 1;
            while (i + run < bs2 && blk[i + run] == 0 && run < 32767) ++run;
            *out++ = 0;
            *out++ = static_cast<int16_t>(run);
            i += run;
        } else {
            *out++ = blk[i++];
        }
    }
}

static void rle_decode_coeff_block(const int16_t* rle, int rle_count, int16_t* coef, int bs2) {
    int in = 0, out = 0;
    while (in < rle_count && out < bs2) {
        if (rle[in] == 0) {
            if (in + 1 >= rle_count) break;
            const int run = rle[in + 1];
            for (int k = 0; k < run && out < bs2; ++k) coef[out++] = 0;
            in += 2;
        } else {
            coef[out++] = rle[in++];
        }
    }
    while (out < bs2) coef[out++] = 0;
}

__global__ void RleCountPerBlockKernel(const int16_t* in, int num_blocks, int block_size_sq, int* counts) {
    extern __shared__ int16_t shared_blk[];
    const int bid = blockIdx.x;
    if (bid >= num_blocks) return;

    const int16_t* src = in + static_cast<size_t>(bid) * block_size_sq;
    for (int i = threadIdx.x; i < block_size_sq; i += blockDim.x) shared_blk[i] = src[i];
    __syncthreads();

    if (threadIdx.x == 0) rle_count_block(shared_blk, block_size_sq, counts[bid]);
}

__global__ void RleScatterPerBlockKernel(const int16_t* in, int num_blocks, int block_size_sq,
                                         const int* offsets, int16_t* out) {
    extern __shared__ int16_t shared_blk[];
    const int bid = blockIdx.x;
    if (bid >= num_blocks) return;

    const int16_t* src = in + static_cast<size_t>(bid) * block_size_sq;
    for (int i = threadIdx.x; i < block_size_sq; i += blockDim.x) shared_blk[i] = src[i];
    __syncthreads();

    if (threadIdx.x == 0) rle_scatter_block(shared_blk, block_size_sq, out + offsets[bid]);
}

__global__ void HuffmanHistKernel(const int16_t* rle, const int* offsets, const int* counts, int num_blocks,
                                  uint32_t* hist) {
    const int bid = blockIdx.x;
    if (bid >= num_blocks) return;

    const auto* bytes = reinterpret_cast<const uint8_t*>(rle + offsets[bid]);
    const int nbytes = counts[bid] * 2;
    for (int i = threadIdx.x; i < nbytes; i += blockDim.x) atomicAdd(&hist[bytes[i]], 1u);
}

__global__ void HuffmanBlockBitLengthKernel(const int16_t* rle, const int* offsets, const int* counts,
                                            int num_blocks, const uint8_t* lens, uint32_t* bit_lengths) {
    const int bid = blockIdx.x;
    if (bid >= num_blocks || threadIdx.x != 0) return;

    const auto* bytes = reinterpret_cast<const uint8_t*>(rle + offsets[bid]);
    const int nbytes = counts[bid] * 2;
    uint32_t total_bits = 0;
    for (int i = 0; i < nbytes; ++i) total_bits += lens[bytes[i]];
    bit_lengths[bid] = total_bits;
}

__device__ void write_bits_msb(uint8_t* out, uint32_t& bit_pos, uint32_t code, int nbits) {
    for (int i = nbits - 1; i >= 0; --i) {
        if ((code >> i) & 1u) out[bit_pos >> 3] |= static_cast<uint8_t>(1u << (7 - (bit_pos & 7)));
        ++bit_pos;
    }
}

__global__ void HuffmanPackKernel(const int16_t* rle, const int* offsets, const int* counts, int num_blocks,
                                  const uint32_t* code_bits, const uint8_t* code_lens,
                                  const uint32_t* bit_offsets, uint8_t* out) {
    const int bid = blockIdx.x;
    if (bid >= num_blocks || threadIdx.x != 0) return;

    const auto* bytes = reinterpret_cast<const uint8_t*>(rle + offsets[bid]);
    const int nbytes = counts[bid] * 2;
    uint32_t pos = bit_offsets[bid];

    for (int i = 0; i < nbytes; ++i) {
        const uint8_t sym = bytes[i];
        write_bits_msb(out, pos, code_bits[sym], static_cast<int>(code_lens[sym]));
    }
}

__global__ void UpdateRleMetaKernel(const int* offsets, const int* counts, int num_blocks, GpuMetadata* meta) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        meta->rle_bytes = static_cast<uint32_t>((offsets[num_blocks - 1] + counts[num_blocks - 1]) * 2);
}

__global__ void UpdatePackMetaKernel(const uint32_t* offsets, const uint32_t* lengths, int num_blocks,
                                     GpuMetadata* meta) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        meta->pack_bytes = (offsets[num_blocks - 1] + lengths[num_blocks - 1] + 7u) / 8u;
}

void rle_gpu_init(int ch, size_t max_elements) {
    if (!g_ctx[ch]) g_ctx[ch] = new RleContext();
    auto& ctx = *g_ctx[ch];
    if (ctx.rle_cap >= max_elements) return;

    cuda_free_ptr(ctx.d_rle);
    cuda_free_ptr(ctx.d_code_bits);
    cuda_free_ptr(ctx.d_code_lens);
    cuda_free_ptr(ctx.d_bitstream);
    cuda_free_ptr(ctx.d_hist);
    cuda_free_ptr(ctx.d_block_rle_counts);
    cuda_free_ptr(ctx.d_block_rle_offsets);
    cuda_free_ptr(ctx.d_block_bit_lengths);
    cuda_free_ptr(ctx.d_block_bit_offsets);
    cuda_free_ptr(ctx.d_temp_storage);

    ctx.rle_cap = max_elements;
    ctx.max_blocks = static_cast<int>((max_elements + 63) / 64);

    constexpr size_t kPad = 65536;
    CUDA_CHECK(cudaMalloc(&ctx.d_rle, max_elements * sizeof(int16_t) + kPad));
    CUDA_CHECK(cudaMalloc(&ctx.d_code_bits, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_code_lens, 256));
    ctx.bitstream_cap = (max_elements * 2 + kPad) & ~size_t{4095};
    CUDA_CHECK(cudaMalloc(&ctx.d_bitstream, ctx.bitstream_cap));
    CUDA_CHECK(cudaMalloc(&ctx.d_hist, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_rle_counts, static_cast<size_t>(ctx.max_blocks) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_rle_offsets, static_cast<size_t>(ctx.max_blocks) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_bit_lengths, static_cast<size_t>(ctx.max_blocks) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_bit_offsets, static_cast<size_t>(ctx.max_blocks) * sizeof(uint32_t)));

    size_t scan_int_bytes = 0;
    size_t scan_u32_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_int_bytes, static_cast<int*>(nullptr),
                                  static_cast<int*>(nullptr), static_cast<int>(max_elements), nullptr);
    cub::DeviceScan::ExclusiveSum(nullptr, scan_u32_bytes, static_cast<uint32_t*>(nullptr),
                                  static_cast<uint32_t*>(nullptr), ctx.max_blocks, nullptr);
    ctx.temp_storage_bytes = std::max(scan_int_bytes, scan_u32_bytes) + kPad;
    CUDA_CHECK(cudaMalloc(&ctx.d_temp_storage, ctx.temp_storage_bytes));
}

void rle_gpu_cleanup() {
    for (int ch = 0; ch < 3; ++ch) {
        if (!g_ctx[ch]) continue;
        auto& ctx = *g_ctx[ch];
        cuda_free_ptr(ctx.d_rle);
        cuda_free_ptr(ctx.d_code_bits);
        cuda_free_ptr(ctx.d_code_lens);
        cuda_free_ptr(ctx.d_bitstream);
        cuda_free_ptr(ctx.d_hist);
        cuda_free_ptr(ctx.d_block_rle_counts);
        cuda_free_ptr(ctx.d_block_rle_offsets);
        cuda_free_ptr(ctx.d_block_bit_lengths);
        cuda_free_ptr(ctx.d_block_bit_offsets);
        cuda_free_ptr(ctx.d_temp_storage);
        delete g_ctx[ch];
        g_ctx[ch] = nullptr;
    }
}

void cuda_rle_encode_indexed(int ch, const int16_t* d_coeffs, int num_blocks, int block_size, void* d_meta,
                             void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    auto& ctx = *g_ctx[ch];
    ctx.last_num_blocks = num_blocks;
    const int bs2 = block_size * block_size;
    const int tpb = bs2 > 256 ? 256 : bs2;

    RleCountPerBlockKernel<<<num_blocks, tpb, static_cast<size_t>(bs2) * sizeof(int16_t), st>>>(
        d_coeffs, num_blocks, bs2, ctx.d_block_rle_counts);

    size_t temp_bytes = ctx.temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(ctx.d_temp_storage, temp_bytes, ctx.d_block_rle_counts,
                                    ctx.d_block_rle_offsets, num_blocks, st);

    RleScatterPerBlockKernel<<<num_blocks, tpb, static_cast<size_t>(bs2) * sizeof(int16_t), st>>>(
        d_coeffs, num_blocks, bs2, ctx.d_block_rle_offsets, ctx.d_rle);

    if (d_meta)
        UpdateRleMetaKernel<<<1, 1, 0, st>>>(ctx.d_block_rle_offsets, ctx.d_block_rle_counts, num_blocks,
                                               static_cast<GpuMetadata*>(d_meta));
}

void cuda_compute_histogram(int ch, uint32_t* h_hist, void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    auto& ctx = *g_ctx[ch];

    CUDA_CHECK(cudaMemsetAsync(ctx.d_hist, 0, 256 * sizeof(uint32_t), st));
    const int nb = ctx.last_num_blocks;
    HuffmanHistKernel<<<nb, 256, 0, st>>>(ctx.d_rle, ctx.d_block_rle_offsets, ctx.d_block_rle_counts, nb, ctx.d_hist);
    if (h_hist) {
        if (st) CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(h_hist, ctx.d_hist, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
}

void cuda_prepare_huffman_codebook_gpu(int ch, void* stream) {
    if (!g_ctx[ch]) return;

    cudaStream_t st = static_cast<cudaStream_t>(stream);
    auto& ctx = *g_ctx[ch];
    if (st) CUDA_CHECK(cudaStreamSynchronize(st));

    uint32_t hist[256] = {};
    CUDA_CHECK(cudaMemcpy(hist, ctx.d_hist, sizeof(hist), cudaMemcpyDeviceToHost));

    uint32_t code_bits[256] = {};
    uint8_t code_lens[256] = {};
    if (huffman_codebook_from_freq32(hist, code_bits, code_lens) != 0) {
        std::memset(code_bits, 0, sizeof(code_bits));
        std::memset(code_lens, 0, sizeof(code_lens));
    }

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_bits, code_bits, sizeof(code_bits), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_lens, code_lens, sizeof(code_lens), cudaMemcpyHostToDevice, st));
}

void cuda_huffman_pack_gpu_indexed(int ch, int num_blocks, const uint32_t* h_bits, const uint8_t* h_lens,
                                   uint8_t** d_out, size_t* out_cap, uint32_t* h_block_lengths, void* d_meta,
                                   void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    auto& ctx = *g_ctx[ch];

    if (h_bits) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_bits, h_bits, 256 * sizeof(uint32_t), cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_code_lens, h_lens, 256, cudaMemcpyHostToDevice, st));
    }

    HuffmanBlockBitLengthKernel<<<num_blocks, 1, 0, st>>>(ctx.d_rle, ctx.d_block_rle_offsets, ctx.d_block_rle_counts,
                                                          num_blocks, ctx.d_code_lens, ctx.d_block_bit_lengths);

    size_t temp_bytes = ctx.temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(ctx.d_temp_storage, temp_bytes, ctx.d_block_bit_lengths, ctx.d_block_bit_offsets,
                                  num_blocks, st);

    if (d_meta)
        UpdatePackMetaKernel<<<1, 1, 0, st>>>(ctx.d_block_bit_offsets, ctx.d_block_bit_lengths, num_blocks,
                                              static_cast<GpuMetadata*>(d_meta));

    CUDA_CHECK(cudaMemsetAsync(ctx.d_bitstream, 0, ctx.bitstream_cap, st));
    HuffmanPackKernel<<<num_blocks, 1, 0, st>>>(ctx.d_rle, ctx.d_block_rle_offsets, ctx.d_block_rle_counts,
                                                  num_blocks, ctx.d_code_bits, ctx.d_code_lens, ctx.d_block_bit_offsets,
                                                  ctx.d_bitstream);

    if (d_out) *d_out = ctx.d_bitstream;
    if (out_cap) *out_cap = ctx.bitstream_cap;
}

void cuda_huffman_download_block_bit_lengths(int ch, uint32_t* dst, int num_blocks) {
    if (!g_ctx[ch] || !dst) return;
    CUDA_CHECK(cudaMemcpy(dst, g_ctx[ch]->d_block_bit_lengths, static_cast<size_t>(num_blocks) * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}

void cuda_gpu_decode_entropy(int ch, const uint8_t* packed, size_t packed_bytes, const uint32_t* bit_lengths,
                             int num_blocks, const uint32_t* freq, int block_size, void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    auto& ctx = *g_ctx[ch];
    if (st) CUDA_CHECK(cudaStreamSynchronize(st));

    const int bs2 = block_size * block_size;
    std::vector<uint32_t> bit_start(static_cast<size_t>(num_blocks));
    uint32_t pos = 0;
    for (int i = 0; i < num_blocks; ++i) {
        bit_start[static_cast<size_t>(i)] = pos;
        pos += bit_lengths[i];
    }

    std::vector<int16_t> coeffs(static_cast<size_t>(num_blocks) * bs2, 0);
    std::vector<uint8_t> rle_bytes(static_cast<size_t>(bs2) * 2 + 64);

    for (int bid = 0; bid < num_blocks; ++bid) {
        int16_t* coef = coeffs.data() + static_cast<size_t>(bid) * bs2;
        if (bit_lengths[bid] == 0) continue;

        const int nbytes = huffman_decode_bit_window(
            freq, packed, static_cast<int>(packed_bytes), static_cast<int>(bit_start[static_cast<size_t>(bid)]),
            static_cast<int>(bit_lengths[bid]), rle_bytes.data(), static_cast<int>(rle_bytes.size()));
        if (nbytes < 0 || (nbytes & 1) != 0) continue;

        rle_decode_coeff_block(reinterpret_cast<const int16_t*>(rle_bytes.data()), nbytes / 2, coef, bs2);
    }

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_rle, coeffs.data(), coeffs.size() * sizeof(int16_t), cudaMemcpyHostToDevice, st));
}

int16_t* cuda_get_decoded_coeffs(int ch) {
    return g_ctx[ch] ? g_ctx[ch]->d_rle : nullptr;
}

void cuda_rle_download_to_host(int ch, void* dst, size_t nbytes) {
    if (!g_ctx[ch] || !dst || nbytes == 0) return;
    CUDA_CHECK(cudaMemcpy(dst, g_ctx[ch]->d_rle, nbytes, cudaMemcpyDeviceToHost));
}
