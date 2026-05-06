#include "rle_gpu.cuh"
#include "huffman.h"

#include <cub/cub.cuh>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <utility>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit((int)err); \
        } \
    } while (0)

template<typename T>
void robust_cuda_free(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

struct RleContext {
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

    void*  d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int    h_metadata_count = 0;

    GpuHuffNode* d_decode_tree = nullptr;
    uint8_t*     d_decode_packed = nullptr;
    int16_t*     d_decode_rle_buf = nullptr;
    int*         d_decode_rle_counts = nullptr;
    size_t       decode_rle_buf_cap = 0;
    size_t       decode_rle_counts_cap = 0;
};

struct GPU_PinnedMetadata {
    uint32_t rle_bytes;
    uint32_t pack_bytes;
    uint32_t num_blocks;
};

struct HuffTreeBuildNode {
    int left, right, symbol;
    uint32_t count;
};

static int build_huffman_tree_u16(const uint32_t* freq, HuffTreeBuildNode* nodes, int* out_num_nodes) {
    int num_nodes = 0; int roots[256]; int num_roots = 0;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] == 0) continue;
        const int id = num_nodes++; nodes[id] = { -1, -1, i, freq[i] }; roots[num_roots++] = id;
    }
    if (num_roots == 0) { *out_num_nodes = 0; return -1; }
    while (num_roots > 1) {
        int i0 = 0, i1 = 1; if (nodes[roots[i1]].count < nodes[roots[i0]].count) std::swap(i0, i1);
        for (int i = 2; i < num_roots; ++i) {
            if (nodes[roots[i]].count < nodes[roots[i0]].count) { i1 = i0; i0 = i; }
            else if (nodes[roots[i]].count < nodes[roots[i1]].count) i1 = i;
        }
        int id0 = roots[i0], id1 = roots[i1], parent = num_nodes++;
        nodes[parent] = { id0, id1, -1, nodes[id0].count + nodes[id1].count };
        roots[i0] = parent; roots[i1] = roots[num_roots - 1]; --num_roots;
    }
    *out_num_nodes = num_nodes; return roots[0];
}

static void ht_to_gpu_nodes(const HuffTreeBuildNode* nodes, int num_nodes, GpuHuffNode* out) {
    for (int i = 0; i < num_nodes; ++i) {
        if (nodes[i].left < 0) { 
            out[i].children[0] = -1; out[i].children[1] = -1; 
            out[i].symbol = (int16_t)nodes[i].symbol; 
        } else { 
            out[i].children[0] = (int16_t)nodes[i].left; 
            out[i].children[1] = (int16_t)nodes[i].right; 
            out[i].symbol = -1; 
        }
    }
}


__global__ void RleCountPerBlockKernel(const int16_t* __restrict__ in, int num_blocks, int block_size_sq, int* counts) {
    extern __shared__ int16_t s_blk[];
    int bid = blockIdx.x;
    if (bid >= num_blocks) return;
    int tid = threadIdx.x;
    const int16_t* b_ptr = &in[(size_t)bid * block_size_sq];

    for (int i = tid; i < block_size_sq; i += blockDim.x) {
        s_blk[i] = b_ptr[i];
    }
    __syncthreads();

    if (tid == 0) {
        int output_elems = 0, i = 0;
        while (i < block_size_sq) {
            if (s_blk[i] == 0) { 
                int run = 1; 
                while (i + run < block_size_sq && s_blk[i+run] == 0 && run < 32767) run++; 
                output_elems += 2; i += run; 
            } else { output_elems += 1; i++; }
        }
        counts[bid] = output_elems;
    }
}

__global__ void RleScatterPerBlockKernel(const int16_t* __restrict__ in, int num_blocks, int block_size_sq, const int* offsets, int16_t* out) {
    extern __shared__ int16_t s_blk[];
    int bid = blockIdx.x;
    if (bid >= num_blocks) return;
    int tid = threadIdx.x;
    const int16_t* b_ptr = &in[(size_t)bid * block_size_sq];

    for (int i = tid; i < block_size_sq; i += blockDim.x) {
        s_blk[i] = b_ptr[i];
    }
    __syncthreads();

    if (tid == 0) {
        int16_t* o_ptr = &out[(size_t)offsets[bid]];
        int i = 0;
        while (i < block_size_sq) {
            if (s_blk[i] == 0) {
                int run = 1;
                while (i + run < block_size_sq && s_blk[i+run] == 0 && run < 32767) run++;
                *o_ptr++ = 0; *o_ptr++ = (int16_t)run; i += run;
            } else { *o_ptr++ = s_blk[i]; i++; }
        }
    }
}

__global__ void HuffmanBlockBitLengthKernel(const int16_t* __restrict__ rle_in, const int* rle_offsets, const int* rle_counts, int num_blocks, const uint8_t* __restrict__ lens, uint32_t* blen) {
    extern __shared__ uint8_t s_lens[];
    int tid = threadIdx.x;
    if (tid < 256) s_lens[tid] = lens[tid];
    __syncthreads();

    int bid = blockIdx.x;
    if (bid >= num_blocks) return;
    
    const uint8_t* b_ptr = (const uint8_t*)&rle_in[(size_t)rle_offsets[bid]];
    int n = rle_counts[bid] * 2;
    
    uint32_t total = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        total += s_lens[b_ptr[i]];
    }
    
    for (int mask = 16; mask > 0; mask >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, mask);
    }

    __shared__ uint32_t s_partials[32];
    if ((tid & 31) == 0) {
        s_partials[tid >> 5] = total;
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t final_total = 0;
        for (int i = 0; i < (blockDim.x + 31) / 32; ++i) final_total += s_partials[i];
        blen[bid] = final_total;
    }
}

__device__ void dev_write_bits(uint8_t* out, uint32_t& off, uint32_t code, int n) {
    if (n <= 0) return;
    for (int i = n - 1; i >= 0; --i) {
        if ((code >> i) & 1u) {
            uint32_t p = off;
            atomicOr((unsigned int*)&out[(p >> 3) & ~3], (1u << ((7 - (p & 7)) + 8 * (p >> 3 & 3))));
        }
        off++;
    }
}

__global__ void HuffmanPackKernel(const int16_t* __restrict__ rle_in, const int* rle_off, const int* rle_cnt, int num_blocks, const uint32_t* __restrict__ bits, const uint8_t* __restrict__ lens, const uint32_t* bit_off, uint8_t* out) {
    extern __shared__ uint32_t s_pack_buf[];
    uint32_t* s_bits = s_pack_buf;
    uint8_t*  s_lens = (uint8_t*)&s_pack_buf[256];
    
    int tid = threadIdx.x;
    if (tid < 256) {
        s_bits[tid] = bits[tid];
        s_lens[tid] = lens[tid];
    }
    __syncthreads();

    int bid = blockIdx.x;
    if (bid >= num_blocks) return;
    
    const uint8_t* b_ptr = (const uint8_t*)&rle_in[(size_t)rle_off[bid]];
    uint32_t goff = bit_off[bid]; 
    int n = rle_cnt[bid] * 2;
    
    if (tid == 0) {
        for (int i = 0; i < n; ++i) {
            uint8_t s = b_ptr[i];
            dev_write_bits(out, goff, s_bits[s], (int)s_lens[s]);
        }
    }
}

__global__ void HuffmanHistKernel(const int16_t* rle_in, const int* offsets, const int* counts, int num_blocks, uint32_t* hist) {
    int bid = blockIdx.x; if (bid >= num_blocks) return;
    const uint8_t* b_ptr = (const uint8_t*)&rle_in[(size_t)offsets[bid]];
    int n = counts[bid] * 2;
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&hist[b_ptr[i]], 1);
}

struct GPUHuffNodeIntern { uint32_t count; int16_t parent, left, right; };
__global__ void HuffmanPrepareCodebookKernel(const uint32_t* hist, uint32_t* out_bits, uint8_t* out_lens) {
    __shared__ GPUHuffNodeIntern nodes[512]; __shared__ int roots[256]; __shared__ int nr, nn, cr;
    int tid = threadIdx.x; if (tid < 256) nodes[tid] = { hist[tid], -1, -1, -1 }; __syncthreads();
    if (tid == 0) { int c = 0; for (int i = 0; i < 256; ++i) if (nodes[i].count > 0) roots[c++] = i; nr = cr = c; nn = 256; } __syncthreads();
    if (nr == 0) { if (tid < 256) { out_bits[tid] = 0; out_lens[tid] = 0; } return; }
    if (nr == 1) { if (tid == 0) { out_bits[roots[0]] = 0; out_lens[roots[0]] = 1; } if (tid < 256 && tid != roots[0]) { out_bits[tid] = 0; out_lens[tid] = 0; } return; }
    while (cr > 1) {
        if (tid == 0) {
            int i1 = 0, i2 = 1; if (nodes[roots[i2]].count < nodes[roots[i1]].count) { int t=roots[i1]; roots[i1]=roots[i2]; roots[i2]=t; }
            for (int i = 2; i < cr; ++i) { if (nodes[roots[i]].count < nodes[roots[i1]].count) { i2 = i1; i1 = i; } else if (nodes[roots[i]].count < nodes[roots[i2]].count) i2 = i; }
            int id1 = roots[i1], id2 = roots[i2], p = nn++; nodes[p] = { nodes[id1].count + nodes[id2].count, -1, (int16_t)id1, (int16_t)id2 };
            nodes[id1].parent = nodes[id2].parent = (int16_t)p; roots[i1] = p; roots[i2] = roots[cr - 1]; cr--;
        } __syncthreads();
    }
    if (tid < 256) {
        if (nodes[tid].count > 0) {
            uint32_t c = 0; int l = 0, cu = tid; 
            while (nodes[cu].parent != -1) { 
                int p = nodes[cu].parent; 
                if (nodes[p].right == cu) c |= (1u << l); 
                l++; cu = p; 
            }
            out_bits[tid] = c; out_lens[tid] = (uint8_t)l;
        } else { out_bits[tid] = 0; out_lens[tid] = 0; }
    }
}

__global__ void UpdateRleMeta(const int* off, const int* cnt, int nb, GPU_PinnedMetadata* meta) { 
    if (threadIdx.x==0 && blockIdx.x==0) meta->rle_bytes = (nb > 0) ? (uint32_t)((off[nb-1]+cnt[nb-1])*2) : 0u;
}
__global__ void UpdatePackMeta(const uint32_t* off, const uint32_t* len, int nb, GPU_PinnedMetadata* meta) { 
    if (threadIdx.x==0 && blockIdx.x==0) meta->pack_bytes = (nb > 0) ? (off[nb-1]+len[nb-1]+7u)/8u : 0u;
}


static RleContext* g_rle_ctx[3] = {nullptr, nullptr, nullptr};

void rle_gpu_init(int ch, size_t max_elements) {
    if (!g_rle_ctx[ch]) g_rle_ctx[ch] = new RleContext();
    auto& ctx = *g_rle_ctx[ch];
    if (ctx.capacity >= max_elements) return;

    robust_cuda_free(ctx.d_final_out); robust_cuda_free(ctx.d_code_bits); robust_cuda_free(ctx.d_code_lens);
    robust_cuda_free(ctx.d_bitstream); robust_cuda_free(ctx.d_hist); robust_cuda_free(ctx.d_block_rle_counts);
    robust_cuda_free(ctx.d_block_rle_offsets); robust_cuda_free(ctx.d_block_bit_lengths); robust_cuda_free(ctx.d_block_bit_offsets);
    robust_cuda_free(ctx.d_temp_storage); robust_cuda_free(ctx.d_decode_tree); robust_cuda_free(ctx.d_decode_packed);
    robust_cuda_free(ctx.d_decode_rle_buf); robust_cuda_free(ctx.d_decode_rle_counts);

    ctx.capacity = max_elements;
    int mb = (int)((max_elements + 63) / 64);
    ctx.h_metadata_count = mb;

    const size_t padding = 65536;
    CUDA_CHECK(cudaMalloc(&ctx.d_final_out, max_elements * 4 + padding));
    CUDA_CHECK(cudaMalloc(&ctx.d_code_bits, 1024));
    CUDA_CHECK(cudaMalloc(&ctx.d_code_lens, 256));
    ctx.bitstream_cap = (max_elements * 2 + padding) & ~4095;
    CUDA_CHECK(cudaMalloc(&ctx.d_bitstream, ctx.bitstream_cap));
    CUDA_CHECK(cudaMalloc(&ctx.d_hist, 1024));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_rle_counts, mb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_rle_offsets, mb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_bit_lengths, mb * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_block_bit_offsets, mb * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_decode_tree, 512 * sizeof(GpuHuffNode)));

    size_t rb1=0, rb2=0; 
    cub::DeviceScan::ExclusiveSum(0, rb1, (int*)0, (int*)0, (int)max_elements, 0); 
    cub::DeviceScan::ExclusiveSum(0, rb2, (uint32_t*)0, (uint32_t*)0, mb, 0);
    ctx.temp_storage_bytes = std::max(rb1, rb2) + padding;
    CUDA_CHECK(cudaMalloc(&ctx.d_temp_storage, ctx.temp_storage_bytes));
}

void rle_gpu_cleanup() {
    for (int ch=0; ch<3; ++ch) {
        if (!g_rle_ctx[ch]) continue;
        auto& ctx = *g_rle_ctx[ch];
        robust_cuda_free(ctx.d_final_out); robust_cuda_free(ctx.d_code_bits); robust_cuda_free(ctx.d_code_lens);
        robust_cuda_free(ctx.d_bitstream); robust_cuda_free(ctx.d_hist); robust_cuda_free(ctx.d_block_rle_counts);
        robust_cuda_free(ctx.d_block_rle_offsets); robust_cuda_free(ctx.d_block_bit_lengths); robust_cuda_free(ctx.d_block_bit_offsets);
        robust_cuda_free(ctx.d_temp_storage); robust_cuda_free(ctx.d_decode_tree); robust_cuda_free(ctx.d_decode_packed);
        robust_cuda_free(ctx.d_decode_rle_buf); robust_cuda_free(ctx.d_decode_rle_counts);
        delete g_rle_ctx[ch]; g_rle_ctx[ch] = nullptr;
    }
}

void cuda_rle_encode_indexed(int ch, const int16_t* d_c, int nb, int bs, void* d_m, void* s) {
    cudaStream_t st = (cudaStream_t)s; auto& cx = *g_rle_ctx[ch]; int bs2 = bs*bs;
    
    int tpb = (bs2 > 256) ? 256 : bs2;
    RleCountPerBlockKernel<<<nb, tpb, bs2 * sizeof(int16_t), st>>>(d_c, nb, bs2, cx.d_block_rle_counts);
    
    size_t tb = cx.temp_storage_bytes; cub::DeviceScan::ExclusiveSum(cx.d_temp_storage, tb, cx.d_block_rle_counts, cx.d_block_rle_offsets, nb, st);
    
    RleScatterPerBlockKernel<<<nb, tpb, bs2 * sizeof(int16_t), st>>>(d_c, nb, bs2, cx.d_block_rle_offsets, cx.d_final_out);
    
    if (d_m) UpdateRleMeta<<<1,1,0,st>>>(cx.d_block_rle_offsets, cx.d_block_rle_counts, nb, (GPU_PinnedMetadata*)d_m);
}

void cuda_compute_histogram(int ch, uint32_t* hh, void* s) {
    cudaStream_t st = (cudaStream_t)s; auto& cx = *g_rle_ctx[ch]; CUDA_CHECK(cudaMemsetAsync(cx.d_hist, 0, 1024, st));
    HuffmanHistKernel<<<cx.h_metadata_count, 256, 0, st>>>(cx.d_final_out, cx.d_block_rle_offsets, cx.d_block_rle_counts, cx.h_metadata_count, cx.d_hist);
    if (hh) {
        if (st) CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(hh, cx.d_hist, 1024, cudaMemcpyDeviceToHost));
    }
}

void cuda_prepare_huffman_codebook_gpu(int ch, void* s) { 
    if (!g_rle_ctx[ch]) return;
    HuffmanPrepareCodebookKernel<<<1, 256, 0, (cudaStream_t)s>>>(g_rle_ctx[ch]->d_hist, g_rle_ctx[ch]->d_code_bits, g_rle_ctx[ch]->d_code_lens); 
}

void cuda_huffman_pack_gpu_indexed(int ch, int nb, const uint32_t* h_b, const uint8_t* h_l, uint8_t** d_o, size_t* os, uint32_t* d_bl, void* d_m, void* s) {
    cudaStream_t st = (cudaStream_t)s; auto& cx = *g_rle_ctx[ch];
    if (h_b) { CUDA_CHECK(cudaMemcpyAsync(cx.d_code_bits, h_b, 1024, cudaMemcpyHostToDevice, st)); CUDA_CHECK(cudaMemcpyAsync(cx.d_code_lens, h_l, 256, cudaMemcpyHostToDevice, st)); }
    
    int tpb = 256;
    HuffmanBlockBitLengthKernel<<<nb, tpb, 256, st>>>(cx.d_final_out, cx.d_block_rle_offsets, cx.d_block_rle_counts, nb, cx.d_code_lens, cx.d_block_bit_lengths);
    
    size_t tb = cx.temp_storage_bytes; cub::DeviceScan::ExclusiveSum(cx.d_temp_storage, tb, cx.d_block_bit_lengths, cx.d_block_bit_offsets, nb, st);
    if (d_m) UpdatePackMeta<<<1,1,0,st>>>(cx.d_block_bit_offsets, cx.d_block_bit_lengths, nb, (GPU_PinnedMetadata*)d_m);
    CUDA_CHECK(cudaMemsetAsync(cx.d_bitstream, 0, cx.bitstream_cap, st));
    
    HuffmanPackKernel<<<nb, tpb, 1280, st>>>(cx.d_final_out, cx.d_block_rle_offsets, cx.d_block_rle_counts, nb, cx.d_code_bits, cx.d_code_lens, cx.d_block_bit_offsets, cx.d_bitstream);
    
    if (d_o) *d_o = cx.d_bitstream; if (os) *os = cx.bitstream_cap;
}

void cuda_huffman_download_block_bit_lengths(int ch, uint32_t* d, int nb) { 
    if (!g_rle_ctx[ch]) return;
    CUDA_CHECK(cudaMemcpy(d, g_rle_ctx[ch]->d_block_bit_lengths, (size_t)nb*4, cudaMemcpyDeviceToHost)); 
}

__device__ int gpu_rb(const uint8_t* s, uint32_t p) { return (s[p>>3] >> (7-(p&7))) & 1; }
__device__ bool gpu_hd_isl(const GpuHuffNode* t, int n) { return t[n].children[0] < 0; }
__global__ void HuffmanDecodeKernel(const uint8_t* bi, const uint32_t* bs, const uint32_t* bl, int nb, const GpuHuffNode* t, int tr, int16_t* ro, int mr, int* c) {
    int bid = blockIdx.x*blockDim.x+threadIdx.x; if (bid >= nb) return;
    const uint32_t s = bs[bid], l = bl[bid]; int16_t* o = &ro[(size_t)bid*mr]; int bu = 0, oi = 0; uint8_t bp[2]; int bii = 0;
    while (bu < l && oi < mr) { int cu = tr; while (!gpu_hd_isl(t, cu)) { cu = gpu_rb(bi, s+bu++) ? t[cu].children[1] : t[cu].children[0]; } bp[bii++] = (uint8_t)t[cu].symbol; if (bii == 2) { o[oi++] = (int16_t)(((uint16_t)bp[1]<<8)|bp[0]); bii = 0; } }
    c[bid] = oi;
}
__global__ void RleDecodeKernel(const int16_t* ri, int nb, int bs2, const int* c, int mr, int16_t* co) {
    int bid = blockIdx.x*blockDim.x+threadIdx.x; if (bid >= nb) return;
    const int16_t* in = &ri[(size_t)bid*mr]; int16_t* out = &co[(size_t)bid*bs2]; int cnt = c[bid], ii = 0, oi = 0;
    while (ii<cnt && oi<bs2) { if (in[ii]==0) { int r=in[ii+1]; for (int k=0; k<r && oi<bs2; ++k) out[oi++]=0; ii+=2; } else { out[oi++]=in[ii++]; } }
    while (oi<bs2) out[oi++]=0;
}

void cuda_gpu_decode_entropy(int ch, const uint8_t* hp, size_t pb, const uint32_t* hbl, int nb, const uint32_t* hf, int bs, void* s) {
    cudaStream_t st = (cudaStream_t)s; auto& cx = *g_rle_ctx[ch]; int bs2 = bs*bs, mr = 2*bs2;
    if (cx.decode_rle_buf_cap < (size_t)nb*mr) { robust_cuda_free(cx.d_decode_rle_buf); cx.decode_rle_buf_cap = (size_t)nb*mr*2; CUDA_CHECK(cudaMalloc(&cx.d_decode_rle_buf, cx.decode_rle_buf_cap*sizeof(int16_t))); }
    if (cx.decode_rle_counts_cap < (size_t)nb) { robust_cuda_free(cx.d_decode_rle_counts); cx.decode_rle_counts_cap = nb*2; CUDA_CHECK(cudaMalloc(&cx.d_decode_rle_counts, cx.decode_rle_counts_cap*sizeof(int))); }
    if (cx.bitstream_cap < pb) { robust_cuda_free(cx.d_bitstream); cx.bitstream_cap = pb+65536; CUDA_CHECK(cudaMalloc(&cx.d_bitstream, cx.bitstream_cap)); }
    CUDA_CHECK(cudaMemcpyAsync(cx.d_bitstream, hp, pb, cudaMemcpyHostToDevice, st)); CUDA_CHECK(cudaMemcpyAsync(cx.d_block_bit_lengths, hbl, (size_t)nb*4, cudaMemcpyHostToDevice, st));
    cub::DeviceScan::ExclusiveSum(cx.d_temp_storage, cx.temp_storage_bytes, cx.d_block_bit_lengths, cx.d_block_bit_offsets, nb, st);
    HuffTreeBuildNode hn[512]; int hn_c = 0; int r = build_huffman_tree_u16(hf, hn, &hn_c); GpuHuffNode gn[512]; ht_to_gpu_nodes(hn, hn_c, gn);
    CUDA_CHECK(cudaMemcpyAsync(cx.d_decode_tree, gn, hn_c*sizeof(GpuHuffNode), cudaMemcpyHostToDevice, st));
    int tpb = 256, bl = (nb+tpb-1)/tpb; HuffmanDecodeKernel<<<bl, tpb, 0, st>>>(cx.d_bitstream, cx.d_block_bit_offsets, cx.d_block_bit_lengths, nb, cx.d_decode_tree, r, cx.d_decode_rle_buf, mr, cx.d_decode_rle_counts);
    RleDecodeKernel<<<bl, tpb, 0, st>>>(cx.d_decode_rle_buf, nb, bs2, cx.d_decode_rle_counts, mr, cx.d_final_out);
}
int16_t* cuda_get_decoded_coeffs(int ch) { if (!g_rle_ctx[ch]) return 0; return g_rle_ctx[ch]->d_final_out; }
void cuda_rle_download_to_host(int ch, void* d, size_t n) { if (!g_rle_ctx[ch] || !d || n==0) return; CUDA_CHECK(cudaMemcpy(d, g_rle_ctx[ch]->d_final_out, n, cudaMemcpyDeviceToHost)); }
