#include "cuda_kernels.cuh"
#include "rle_gpu.cuh"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <utility>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
#define M_PI_F 3.14159265358979323846f

__device__ __forceinline__ float ld_u8_as_float(const uint8_t* p) {
    return static_cast<float>(__ldg(p));
}







template<int BS>
__device__ void dct2d_device(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ shared) {
    float* tmp = shared;
    for (int r = 0; r < BS; ++r) {
        const float* row_in = in + r * BS;
        float* row_tmp = tmp + r * BS;
        for (int k = 0; k < BS; ++k) {
            float s = 0.f;
            for (int n = 0; n < BS; ++n)
                s += row_in[n] * cosf(M_PI_F * (2.0f * n + 1.0f) * k / (2.0f * BS));
            float ck = (k == 0) ? 0.70710678118f : 1.0f;
            row_tmp[k] = 0.5f * ck * s;
        }
    }
    for (int c = 0; c < BS; ++c) {
        for (int k = 0; k < BS; ++k) {
            float s = 0.f;
            for (int n = 0; n < BS; ++n)
                s += tmp[n * BS + c] * cosf(M_PI_F * (2.0f * n + 1.0f) * k / (2.0f * BS));
            float ck = (k == 0) ? 0.70710678118f : 1.0f;
            out[k * BS + c] = 0.5f * ck * s;
        }
    }
}

template<int BS>
__device__ void idct2d_device(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ shared) {
    float* tmp = shared;
    for (int c = 0; c < BS; ++c) {
        for (int n = 0; n < BS; ++n) {
            float s = 0.f;
            for (int k = 0; k < BS; ++k) {
                float ck = (k == 0) ? 0.70710678118f : 1.0f;
                s += ck * in[k * BS + c] * cosf(M_PI_F * (2.0f * n + 1.0f) * k / (2.0f * BS));
            }
            tmp[n * BS + c] = 0.5f * s;
        }
    }
    for (int r = 0; r < BS; ++r) {
        const float* row_in = tmp + r * BS;
        float* row_out = out + r * BS;
        for (int n = 0; n < BS; ++n) {
            float s = 0.f;
            for (int k = 0; k < BS; ++k) {
                float ck = (k == 0) ? 0.70710678118f : 1.0f;
                s += ck * row_in[k] * cosf(M_PI_F * (2.0f * n + 1.0f) * k / (2.0f * BS));
            }
            row_out[n] = 0.5f * s;
        }
    }
}

template<int BS>
__global__ void encode_blocks_kernel(
    const uint8_t* src, const uint8_t* prev, uint8_t* recon,
    int16_t* coeffs, const float* qm, const int* zigzag,
    int padded_w, int padded_h, bool is_keyframe)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x0 = bx * BS;
    int y0 = by * BS;
    int grid_w = padded_w / BS;
    int block_idx = by * grid_w + bx;

    extern __shared__ float s_buf[];
    float* s_blk = s_buf;
    float* s_mid = s_buf + BS * BS;
    float* s_tmp = s_buf + 2 * BS * BS;

    for (int r = 0; r < BS; ++r) {
        for (int c = 0; c < BS; ++c) {
            const int row = y0 + r;
            const int col = x0 + c;
            const int idx = row * padded_w + col;
            const float val = ld_u8_as_float(src + idx);
            if (!is_keyframe) {
                const float pval = ld_u8_as_float(prev + idx);
                s_blk[r * BS + c] = val - pval;
            } else {
                s_blk[r * BS + c] = val - 128.0f;
            }
        }
    }

    dct2d_device<BS>(s_blk, s_mid, s_tmp);

    for (int i = 0; i < BS * BS; ++i) {
        coeffs[block_idx * BS * BS + zigzag[i]] = static_cast<int16_t>(roundf(s_mid[i] / qm[i]));
    }

    idct2d_device<BS>(s_mid, s_blk, s_tmp);

    for (int r = 0; r < BS; ++r) {
        for (int c = 0; c < BS; ++c) {
            float rval = s_blk[r * BS + c];
            const int row = y0 + r;
            const int col = x0 + c;
            const int idx = row * padded_w + col;
            if (!is_keyframe) {
                rval += ld_u8_as_float(prev + idx);
            } else {
                rval += 128.0f;
            }
            recon[idx] = static_cast<uint8_t>(fminf(fmaxf(roundf(rval), 0.0f), 255.0f));
        }
    }
}

template<int BS>
__global__ void decode_blocks_kernel(
        const int16_t* __restrict__ coeff_in,
        const uint8_t*   __restrict__ prev,
        uint8_t*         __restrict__ recon,
        const float*   __restrict__ qm,
        const int*     __restrict__ zigzag,
        int padded_w, int padded_h, bool is_keyframe)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x0 = bx * BS;
    int y0 = by * BS;
    int grid_w = padded_w / BS;
    int block_idx = by * grid_w + bx;

    extern __shared__ float s_buf[];
    float* s_coeff = s_buf;
    float* s_spat  = s_buf + BS * BS;
    float* s_tmp   = s_buf + 2 * BS * BS;

    
    for (int j = 0; j < BS * BS; ++j) {
        s_coeff[j] = static_cast<float>(coeff_in[block_idx * BS * BS + zigzag[j]]) * qm[j];
    }

    idct2d_device<BS>(s_coeff, s_spat, s_tmp);

    for (int r = 0; r < BS; ++r) {
        for (int c = 0; c < BS; ++c) {
            float rval = s_spat[r * BS + c];
            const int row = y0 + r;
            const int col = x0 + c;
            const int idx = row * padded_w + col;
            if (!is_keyframe) {
                rval += ld_u8_as_float(prev + idx);
            } else {
                rval += 128.0f;
            }
            recon[idx] = static_cast<uint8_t>(fminf(fmaxf(roundf(rval), 0.0f), 255.0f));
        }
    }
}


struct ChannelBuf {
    uint8_t *d_src_ping[2] = {nullptr, nullptr};
    uint8_t *d_prev = nullptr, *d_curr = nullptr;
    int16_t *d_coeff = nullptr;
    size_t pixels = 0, coeffs = 0;
    
    
    int16_t* d_rle = nullptr;
    size_t rle_len = 0;
    uint8_t* d_packed = nullptr;
    size_t packed_bytes = 0;

    
    uint32_t* d_block_bit_lengths = nullptr;
    int pw = 0, ph = 0;
};

static constexpr int MAX_CH = 3;
static ChannelBuf   g_ch[MAX_CH];
static cudaStream_t g_stream[MAX_CH] = {};
static cudaStream_t g_transfer_stream = nullptr;
static float*       g_d_qm[2]       = {};
static int*         g_d_zigzag      = nullptr;
static int          g_num_ch         = 0;
static bool         g_allocated      = false;

static int g_bs = 8;
static int g_pw[MAX_CH] = {}, g_ph[MAX_CH] = {};

static uint8_t* g_h_rgb_in  = nullptr;
static uint8_t* g_h_rgb_out = nullptr;
static size_t   g_total_bytes = 0;
static cudaEvent_t g_evt_h2d_done[2]         = {};
static cudaEvent_t g_evt_encode_slot_done[2] = {};

static inline int pad_bs(int n, int bs) { return ((n + bs - 1) / bs) * bs; }



void cuda_init() {
    CUDA_CHECK(cudaSetDevice(0));
    for (int i = 0; i < MAX_CH; ++i)
        CUDA_CHECK(cudaStreamCreate(&g_stream[i]));
    CUDA_CHECK(cudaStreamCreate(&g_transfer_stream));
}

void cuda_cleanup() {
    cuda_free_frame_buffers();
    rle_gpu_cleanup();
    for (int i = 0; i < MAX_CH; ++i)
        if (g_stream[i]) { cudaStreamDestroy(g_stream[i]); g_stream[i] = nullptr; }
    if (g_transfer_stream) { cudaStreamDestroy(g_transfer_stream); g_transfer_stream = nullptr; }
}

void cuda_alloc_frame_buffers(int width, int height, int channels,
                              const float* luma_qm, const float* chroma_qm,
                              const int* zigzag, bool use_ycbcr, int block_size) {
    if (g_allocated) cuda_free_frame_buffers();
    g_num_ch = channels;
    g_bs     = block_size;

    int total_coeffs_per_block = block_size * block_size;
    g_total_bytes = 0;
    for (int ch = 0; ch < channels; ++ch) {
        int w = width, h = height;
        if (use_ycbcr && ch > 0 && channels == 3) { w = (width+1)/2; h = (height+1)/2; }
        g_pw[ch] = pad_bs(w, block_size);
        g_ph[ch] = pad_bs(h, block_size);
        auto& b = g_ch[ch];
        b.pixels = size_t(g_pw[ch]) * g_ph[ch];
        b.coeffs = size_t(g_pw[ch]/block_size) * (g_ph[ch]/block_size) * total_coeffs_per_block;
        b.pw = g_pw[ch]; b.ph = g_ph[ch];
        rle_gpu_init(ch, b.coeffs);
        g_total_bytes += b.pixels;
        
        size_t bytes = b.pixels * sizeof(uint8_t);
        CUDA_CHECK(cudaMalloc(&b.d_src_ping[0], bytes));
        CUDA_CHECK(cudaMalloc(&b.d_src_ping[1], bytes));
        CUDA_CHECK(cudaMalloc(&b.d_prev, bytes));
        CUDA_CHECK(cudaMalloc(&b.d_curr, bytes));
        CUDA_CHECK(cudaMemset(b.d_prev, 0, bytes));
        CUDA_CHECK(cudaMemset(b.d_curr, 0, bytes));
        CUDA_CHECK(cudaMalloc(&b.d_coeff, b.coeffs * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&b.d_block_bit_lengths, (b.coeffs / (block_size * block_size)) * sizeof(uint32_t)));
    }

    CUDA_CHECK(cudaMalloc(&g_d_qm[0], total_coeffs_per_block*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_qm[1], total_coeffs_per_block*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_d_qm[0], luma_qm,   total_coeffs_per_block*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_qm[1], chroma_qm, total_coeffs_per_block*sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&g_d_zigzag, total_coeffs_per_block*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(g_d_zigzag, zigzag, total_coeffs_per_block*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g_h_rgb_in),  2 * g_total_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g_h_rgb_out), g_total_bytes, cudaHostAllocDefault));

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_evt_h2d_done[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&g_evt_encode_slot_done[s], cudaEventDisableTiming));
    }

    g_allocated = true;
}

void cuda_free_frame_buffers() {
    if (!g_allocated) return;
    for (int ch = 0; ch < g_num_ch; ++ch) {
        auto& b = g_ch[ch];
        cudaFree(b.d_src_ping[0]); cudaFree(b.d_src_ping[1]);
        cudaFree(b.d_prev);
        cudaFree(b.d_curr); cudaFree(b.d_coeff);
        cudaFree(b.d_block_bit_lengths);
        
        g_ch[ch] = {};
    }
    for (auto& p : g_d_qm) { cudaFree(p); p = nullptr; }
    if (g_d_zigzag) { cudaFree(g_d_zigzag); g_d_zigzag = nullptr; }
    if (g_h_rgb_in)  { cudaFreeHost(g_h_rgb_in);  g_h_rgb_in  = nullptr; }
    if (g_h_rgb_out) { cudaFreeHost(g_h_rgb_out); g_h_rgb_out = nullptr; }
    for (int s = 0; s < 2; ++s) {
        if (g_evt_h2d_done[s])         { cudaEventDestroy(g_evt_h2d_done[s]);         g_evt_h2d_done[s]         = {}; }
        if (g_evt_encode_slot_done[s]) { cudaEventDestroy(g_evt_encode_slot_done[s]); g_evt_encode_slot_done[s] = {}; }
    }
    g_allocated = false;
}

void cuda_submit_frame_h2d(int frame_index, const uint8_t* ptr[3], int channels) {
    const int slot = frame_index % 2;
    
    if (frame_index >= 2)
        CUDA_CHECK(cudaStreamWaitEvent(g_transfer_stream, g_evt_encode_slot_done[slot]));

    uint8_t* pin_slot = g_h_rgb_in + slot * g_total_bytes;
    size_t offset = 0;
    for (int ch = 0; ch < channels; ++ch) {
        size_t bytes = g_ch[ch].pixels * sizeof(uint8_t);
        std::memcpy(pin_slot + offset, ptr[ch], bytes);
        CUDA_CHECK(cudaMemcpyAsync(g_ch[ch].d_src_ping[slot], pin_slot + offset, bytes,
                                   cudaMemcpyHostToDevice, g_transfer_stream));
        offset += bytes;
    }
    CUDA_CHECK(cudaEventRecord(g_evt_h2d_done[slot], g_transfer_stream));
}

void cuda_download_planes(uint8_t* ptr[3], int channels) {
    
    
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t offset = 0;
    for (int ch = 0; ch < channels; ++ch) {
        size_t bytes = g_ch[ch].pixels * sizeof(uint8_t);
        CUDA_CHECK(cudaMemcpyAsync(g_h_rgb_out + offset, g_ch[ch].d_curr, bytes,
                                   cudaMemcpyDeviceToHost, g_transfer_stream));
        offset += bytes;
    }
    CUDA_CHECK(cudaStreamSynchronize(g_transfer_stream));
    offset = 0;
    for (int ch = 0; ch < channels; ++ch) {
        size_t bytes = g_ch[ch].pixels * sizeof(uint8_t);
        std::memcpy(ptr[ch], g_h_rgb_out + offset, bytes);
        offset += bytes;
    }
}

void cuda_download_coeffs(int ch, int16_t* host_dst, int num_coeffs) {
    if (!host_dst || num_coeffs <= 0) return;
    CUDA_CHECK(cudaMemcpy(host_dst, g_ch[ch].d_coeff, (size_t)num_coeffs * sizeof(int16_t),
                          cudaMemcpyDeviceToHost));
}

void cuda_encode_channel(int ch, int pw, int ph, int block_size, bool is_keyframe, int src_slot) {
    auto& buf    = g_ch[ch];
    auto  stream = g_stream[ch];
    const float* qm = (ch == 0) ? g_d_qm[0] : g_d_qm[1];
    const int s = src_slot & 1;
    CUDA_CHECK(cudaStreamWaitEvent(stream, g_evt_h2d_done[s]));

    uint8_t* d_src = buf.d_src_ping[s];
    dim3 grid(pw / block_size, ph / block_size);
    const size_t shared_mem = 3u * (size_t)block_size * (size_t)block_size * sizeof(float);

    if (block_size == 8) {
        encode_blocks_kernel<8><<<grid, 1, shared_mem, stream>>>(
            d_src, buf.d_prev, buf.d_curr, buf.d_coeff, qm, g_d_zigzag, pw, ph, is_keyframe);
    } else if (block_size == 16) {
        encode_blocks_kernel<16><<<grid, 1, shared_mem, stream>>>(
            d_src, buf.d_prev, buf.d_curr, buf.d_coeff, qm, g_d_zigzag, pw, ph, is_keyframe);
    } else if (block_size == 32) {
        encode_blocks_kernel<32><<<grid, 1, shared_mem, stream>>>(
            d_src, buf.d_prev, buf.d_curr, buf.d_coeff, qm, g_d_zigzag, pw, ph, is_keyframe);
    }
    CUDA_CHECK(cudaGetLastError());
}

void cuda_record_encode_slot_done(int slot, int last_ch) {
    const int s = slot & 1;
    const int lc = last_ch < 0 ? 0 : (last_ch > MAX_CH - 1 ? MAX_CH - 1 : last_ch);
    CUDA_CHECK(cudaEventRecord(g_evt_encode_slot_done[s], g_stream[lc]));
}

void cuda_rle_channel_indexed(int ch, int num_blocks, int block_size, uint32_t* out_rle_bytes) {
    auto& buf = g_ch[ch];
    auto  stream = g_stream[ch];
    cuda_rle_encode_indexed(ch, buf.d_coeff, num_blocks, block_size, out_rle_bytes, stream);
}

void cuda_hist_channel_new(int ch, uint32_t* h_hist) {
    auto  stream = g_stream[ch];
    cuda_compute_histogram(ch, h_hist, stream);
}

void cuda_rle_channel(int ch, int pw, int ph, int block_size, uint32_t* out_rle_elements) {
    int num_blocks = (pw / block_size) * (ph / block_size);
    cuda_rle_channel_indexed(ch, num_blocks, block_size, out_rle_elements);
}

void cuda_hist_channel(int ch, uint32_t h_hist[256]) {
    cuda_hist_channel_new(ch, h_hist);
}

void cuda_pack_channel(int ch, const uint32_t h_code_bits[256], const uint8_t h_code_lens[256],
                       uint8_t** d_packed_ptr, size_t* out_packed_bytes) {
    auto& buf = g_ch[ch];
    int num_blocks = (buf.pw / g_bs) * (buf.ph / g_bs);
    cuda_pack_channel_indexed(ch, num_blocks, g_bs, h_code_bits, h_code_lens, d_packed_ptr, out_packed_bytes, nullptr);
}

void cuda_decode_channel(int ch, const int16_t* coeff_in,
                         int pw, int ph, int block_size, bool is_keyframe) {
    auto& buf    = g_ch[ch];
    auto  stream = g_stream[ch];
    const float* qm = (ch == 0) ? g_d_qm[0] : g_d_qm[1];

    int coeffs_to_copy = (pw / block_size) * (ph / block_size) * block_size * block_size;
    CUDA_CHECK(cudaMemcpyAsync(buf.d_coeff, coeff_in,
               coeffs_to_copy * sizeof(int16_t), cudaMemcpyHostToDevice, stream));

    dim3 grid(pw / block_size, ph / block_size);
    const size_t shared_mem = 3u * (size_t)block_size * (size_t)block_size * sizeof(float);

    if (block_size == 8) {
        decode_blocks_kernel<8><<<grid, 1, shared_mem, stream>>>(
            buf.d_coeff, buf.d_prev, buf.d_curr, qm, g_d_zigzag, pw, ph, is_keyframe);
    } else if (block_size == 16) {
        decode_blocks_kernel<16><<<grid, 1, shared_mem, stream>>>(
            buf.d_coeff, buf.d_prev, buf.d_curr, qm, g_d_zigzag, pw, ph, is_keyframe);
    } else if (block_size == 32) {
        decode_blocks_kernel<32><<<grid, 1, shared_mem, stream>>>(
            buf.d_coeff, buf.d_prev, buf.d_curr, qm, g_d_zigzag, pw, ph, is_keyframe);
    }
    CUDA_CHECK(cudaGetLastError());
}

int16_t* cuda_channel_d_coeff(int ch) { return g_allocated ? g_ch[ch].d_coeff : nullptr; }

void* cuda_channel_stream_ptr(int ch) { return g_stream[ch]; }

void cuda_swap_recon() {
    for (int ch = 0; ch < g_num_ch; ++ch)
        std::swap(g_ch[ch].d_prev, g_ch[ch].d_curr);
}

void cuda_sync_channel(int ch) { CUDA_CHECK(cudaStreamSynchronize(g_stream[ch])); }
void cuda_sync_all()           { for (int i = 0; i < g_num_ch; ++i) cuda_sync_channel(i); }

int16_t* cuda_alloc_pinned_coeffs(size_t num_elements) {
    int16_t* ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr), num_elements * sizeof(int16_t), cudaHostAllocDefault));
    return ptr;
}

void cuda_free_pinned_coeffs(int16_t* ptr) {
    if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
}

void cuda_memcpy_to_host(void* host_ptr, const void* device_ptr, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost));
}

void cuda_full_decode_channel(int ch,
                               const uint8_t* h_packed_data, size_t packed_bytes,
                               const uint32_t* h_block_bit_lengths, int num_blocks,
                               const uint16_t* h_freq,
                               int pw, int ph, int block_size, bool is_keyframe) {
    auto& buf    = g_ch[ch];
    auto  stream = g_stream[ch];
    const float* qm = (ch == 0) ? g_d_qm[0] : g_d_qm[1];

    
    cuda_gpu_decode_entropy(ch, h_packed_data, packed_bytes,
                            h_block_bit_lengths, num_blocks, h_freq, block_size, stream);

    int total_coeffs = num_blocks * block_size * block_size;
    int16_t* decoded = cuda_get_decoded_coeffs(ch);
    CUDA_CHECK(cudaMemcpyAsync(buf.d_coeff, decoded,
               total_coeffs * sizeof(int16_t), cudaMemcpyDeviceToDevice, stream));

    
    dim3 grid(pw / block_size, ph / block_size);
    const size_t shared_mem = 3u * (size_t)block_size * (size_t)block_size * sizeof(float);

    if (block_size == 8) {
        decode_blocks_kernel<8><<<grid, 1, shared_mem, stream>>>(
            buf.d_coeff, buf.d_prev, buf.d_curr, qm, g_d_zigzag, pw, ph, is_keyframe);
    } else if (block_size == 16) {
        decode_blocks_kernel<16><<<grid, 1, shared_mem, stream>>>(
            buf.d_coeff, buf.d_prev, buf.d_curr, qm, g_d_zigzag, pw, ph, is_keyframe);
    } else if (block_size == 32) {
        decode_blocks_kernel<32><<<grid, 1, shared_mem, stream>>>(
            buf.d_coeff, buf.d_prev, buf.d_curr, qm, g_d_zigzag, pw, ph, is_keyframe);
    }
    CUDA_CHECK(cudaGetLastError());
}

void cuda_get_block_bit_lengths(int ch, int num_blocks, uint32_t* h_lengths) {
    CUDA_CHECK(cudaMemcpy(h_lengths, g_ch[ch].d_block_bit_lengths, num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void cuda_pack_channel_indexed(int ch, int num_blocks, int block_size,
                               const uint32_t h_code_bits[256], const uint8_t h_code_lens[256],
                               uint8_t** d_packed_ptr, size_t* out_packed_bytes,
                               uint32_t* d_block_bit_lengths_ptr) {
    auto& buf = g_ch[ch];
    auto  stream = g_stream[ch];

    cuda_huffman_pack_gpu_indexed(ch, num_blocks,
                                  h_code_bits, h_code_lens, &buf.d_packed, &buf.packed_bytes, 
                                  buf.d_block_bit_lengths, stream);
    
    *d_packed_ptr = buf.d_packed;
    *out_packed_bytes = buf.packed_bytes;
}
