#include "cuda_kernels.cuh"

#include <cstdio>
#include <cmath>
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


__constant__ float d_cos[8][8];

__constant__ int d_zigzag[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

__device__ __forceinline__ float d_C(int u) {
    return (u == 0) ? 0.70710678118f : 1.0f;
}

static constexpr int TILES_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLK = TILES_PER_BLOCK * 64;


__global__ void rgb_to_ycbcr_kernel(
        const uint8_t* __restrict__ rgb,
        float* __restrict__ y_plane,
        float* __restrict__ cb_plane,
        float* __restrict__ cr_plane,
        int width, int height, int channels,
        int pw_y, int pw_cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = (y * width + x) * channels;
    const float R = static_cast<float>(rgb[idx + 0]);
    const float G = static_cast<float>(rgb[idx + 1]);
    const float B = static_cast<float>(rgb[idx + 2]);

    const float Y  =  0.299f * R + 0.587f * G + 0.114f * B;
    const float Cb = -0.168736f * R - 0.331264f * G + 0.5f * B + 128.0f;
    const float Cr =  0.5f * R - 0.418688f * G - 0.081312f * B + 128.0f;

    y_plane[y * pw_y + x] = Y;

    if ((x & 1) == 0 && (y & 1) == 0) {
        cb_plane[(y / 2) * pw_cb + (x / 2)] = Cb;
        cr_plane[(y / 2) * pw_cb + (x / 2)] = Cr;
    }
}

__global__ void ycbcr_to_rgb_kernel(
        const float* __restrict__ y_plane,
        const float* __restrict__ cb_plane,
        const float* __restrict__ cr_plane,
        uint8_t* __restrict__ rgb,
        int width, int height, int channels,
        int pw_y, int pw_cb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float Y  = y_plane[y * pw_y + x];
    const float Cb = cb_plane[(y / 2) * pw_cb + (x / 2)] - 128.0f;
    const float Cr = cr_plane[(y / 2) * pw_cb + (x / 2)] - 128.0f;

    const float R = Y + 1.402f * Cr;
    const float G = Y - 0.344136f * Cb - 0.714136f * Cr;
    const float B = Y + 1.772f * Cb;

    const int idx = (y * width + x) * channels;
    rgb[idx + 0] = static_cast<uint8_t>(fminf(fmaxf(R + 0.5f, 0.0f), 255.0f));
    rgb[idx + 1] = static_cast<uint8_t>(fminf(fmaxf(G + 0.5f, 0.0f), 255.0f));
    rgb[idx + 2] = static_cast<uint8_t>(fminf(fmaxf(B + 0.5f, 0.0f), 255.0f));
    if (channels == 4) rgb[idx + 3] = 255;
}


__global__ void encode_blocks_kernel(
        const float* __restrict__ src,
        const float* __restrict__ prev,
        float*       __restrict__ recon,
        int16_t*     __restrict__ coeff_out,
        const float* __restrict__ qm,
        int padded_w, int blocks_x, int total_blocks,
        bool is_keyframe)
{
    const int tile     = threadIdx.x >> 6;
    const int tid      = threadIdx.x & 63;
    const int block_id = blockIdx.x * TILES_PER_BLOCK + tile;
    const bool active  = (block_id < total_blocks);

    const int bx  = active ? (block_id % blocks_x) : 0;
    const int by  = active ? (block_id / blocks_x) : 0;
    const int row = tid >> 3, col = tid & 7;
    const int px  = bx * 8 + col, py = by * 8 + row;

    __shared__ float s_blk[TILES_PER_BLOCK][64];
    __shared__ float s_tmp[TILES_PER_BLOCK][64];

    float val = 0.0f;
    if (active) {
        val = src[py * padded_w + px];
        if (!is_keyframe) val -= prev[py * padded_w + px];
        else              val -= 128.0f;
    }
    s_blk[tile][tid] = val;
    __syncthreads();

    { float s = 0; for (int n = 0; n < 8; ++n) s += s_blk[tile][row*8+n] * d_cos[col][n];
      s_tmp[tile][tid] = 0.5f * d_C(col) * s; }
    __syncthreads();

    { float s = 0; for (int n = 0; n < 8; ++n) s += s_tmp[tile][n*8+col] * d_cos[row][n];
      s_blk[tile][tid] = 0.5f * d_C(row) * s; }
    __syncthreads();

    const float qm_val = qm[tid];
    const float q = roundf(s_blk[tile][tid] / qm_val);
    s_blk[tile][tid] = q;
    __syncthreads();

    if (active) coeff_out[block_id * 64 + tid] = static_cast<int16_t>(s_blk[tile][d_zigzag[tid]]);
    __syncthreads();

    s_blk[tile][tid] = q * qm_val;
    __syncthreads();

    { float s = 0; for (int k = 0; k < 8; ++k) s += d_C(k) * s_blk[tile][k*8+col] * d_cos[k][row];
      s_tmp[tile][tid] = 0.5f * s; }
    __syncthreads();

    { float s = 0; for (int k = 0; k < 8; ++k) s += d_C(k) * s_tmp[tile][row*8+k] * d_cos[k][col];
      s_blk[tile][tid] = 0.5f * s; }
    __syncthreads();

    if (active) {
        float r = s_blk[tile][tid];
        if (!is_keyframe) r += prev[py * padded_w + px]; else r += 128.0f;
        recon[py * padded_w + px] = r;
    }
}


__global__ void decode_blocks_kernel(
        const int16_t* __restrict__ coeff_in,
        const float*   __restrict__ prev,
        float*         __restrict__ recon,
        const float*   __restrict__ qm,
        int padded_w, int blocks_x, int total_blocks,
        bool is_keyframe)
{
    const int tile     = threadIdx.x >> 6;
    const int tid      = threadIdx.x & 63;
    const int block_id = blockIdx.x * TILES_PER_BLOCK + tile;
    const bool active  = (block_id < total_blocks);

    const int bx = active ? (block_id % blocks_x) : 0;
    const int by = active ? (block_id / blocks_x) : 0;
    const int row = tid >> 3, col = tid & 7;

    __shared__ float s_blk[TILES_PER_BLOCK][64];
    __shared__ float s_tmp[TILES_PER_BLOCK][64];

    float c = 0.0f;
    if (active) c = static_cast<float>(coeff_in[block_id * 64 + tid]);
    s_blk[tile][d_zigzag[tid]] = c;
    __syncthreads();

    s_blk[tile][tid] *= qm[tid];
    __syncthreads();

    { float s = 0; for (int k = 0; k < 8; ++k) s += d_C(k) * s_blk[tile][k*8+col] * d_cos[k][row];
      s_tmp[tile][tid] = 0.5f * s; }
    __syncthreads();

    { float s = 0; for (int k = 0; k < 8; ++k) s += d_C(k) * s_tmp[tile][row*8+k] * d_cos[k][col];
      s_blk[tile][tid] = 0.5f * s; }
    __syncthreads();

    if (active) {
        const int px = bx * 8 + col, py = by * 8 + row;
        float v = s_blk[tile][tid];
        if (!is_keyframe) v += prev[py * padded_w + px]; else v += 128.0f;
        recon[py * padded_w + px] = v;
    }
}


struct ChannelBuf {
    float *d_src = nullptr, *d_prev = nullptr, *d_curr = nullptr;
    int16_t *d_coeff = nullptr;
    size_t pixels = 0, coeffs = 0;
};

static constexpr int MAX_CH = 3;
static ChannelBuf   g_ch[MAX_CH];
static cudaStream_t g_stream[MAX_CH] = {};
static float*       g_d_qm[2]       = {};
static int          g_num_ch         = 0;
static bool         g_allocated      = false;

static int g_width = 0, g_height = 0, g_channels = 0;
static int g_pw[MAX_CH] = {}, g_ph[MAX_CH] = {};

static uint8_t* g_d_rgb_in  = nullptr;
static uint8_t* g_d_rgb_out = nullptr;
static size_t   g_rgb_bytes = 0;

static inline int pad8(int n) { return ((n + 7) / 8) * 8; }


void cuda_init() {
    float h_cos[8][8];
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 8; ++n)
            h_cos[k][n] = cosf(float(M_PI) * (2.f * n + 1.f) * k / 16.f);
    CUDA_CHECK(cudaMemcpyToSymbol(d_cos, h_cos, sizeof(h_cos)));
    for (int i = 0; i < MAX_CH; ++i)
        CUDA_CHECK(cudaStreamCreate(&g_stream[i]));
}

void cuda_cleanup() {
    cuda_free_frame_buffers();
    for (int i = 0; i < MAX_CH; ++i)
        if (g_stream[i]) { cudaStreamDestroy(g_stream[i]); g_stream[i] = nullptr; }
}

void cuda_alloc_frame_buffers(int width, int height, int channels,
                              const float* luma_qm, const float* chroma_qm) {
    if (g_allocated) cuda_free_frame_buffers();
    g_num_ch = channels; g_width = width; g_height = height; g_channels = channels;

    for (int ch = 0; ch < channels; ++ch) {
        int w = width, h = height;
        if (ch > 0 && channels == 3) { w = (width+1)/2; h = (height+1)/2; }
        g_pw[ch] = pad8(w); g_ph[ch] = pad8(h);
        auto& b = g_ch[ch];
        b.pixels = size_t(g_pw[ch]) * g_ph[ch];
        b.coeffs = size_t(g_pw[ch]/8) * (g_ph[ch]/8) * 64;
        CUDA_CHECK(cudaMalloc(&b.d_src,   b.pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b.d_prev,  b.pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b.d_curr,  b.pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b.d_coeff, b.coeffs * sizeof(int16_t)));
        CUDA_CHECK(cudaMemset(b.d_src,  0, b.pixels * sizeof(float)));
        CUDA_CHECK(cudaMemset(b.d_prev, 0, b.pixels * sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc(&g_d_qm[0], 64*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_qm[1], 64*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_d_qm[0], luma_qm,   64*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_qm[1], chroma_qm, 64*sizeof(float), cudaMemcpyHostToDevice));

    g_rgb_bytes = size_t(width) * height * channels;
    CUDA_CHECK(cudaMalloc(&g_d_rgb_in,  g_rgb_bytes));
    CUDA_CHECK(cudaMalloc(&g_d_rgb_out, g_rgb_bytes));

    g_allocated = true;
}

void cuda_free_frame_buffers() {
    if (!g_allocated) return;
    for (int ch = 0; ch < g_num_ch; ++ch) {
        cudaFree(g_ch[ch].d_src);  cudaFree(g_ch[ch].d_prev);
        cudaFree(g_ch[ch].d_curr); cudaFree(g_ch[ch].d_coeff);
        g_ch[ch] = {};
    }
    for (auto& p : g_d_qm) { cudaFree(p); p = nullptr; }
    cudaFree(g_d_rgb_in);  g_d_rgb_in  = nullptr;
    cudaFree(g_d_rgb_out); g_d_rgb_out = nullptr;
    g_allocated = false;
}


void cuda_upload_and_convert_rgb(const uint8_t* rgb_host,
                                 int width, int height, int channels) {
    CUDA_CHECK(cudaMemcpy(g_d_rgb_in, rgb_host, g_rgb_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    rgb_to_ycbcr_kernel<<<grid, block>>>(
        g_d_rgb_in,
        g_ch[0].d_src, g_ch[1].d_src, g_ch[2].d_src,
        width, height, channels, g_pw[0], g_pw[1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void cuda_encode_channel(int ch, int16_t* coeff_out,
                         int padded_w, int padded_h, bool is_keyframe) {
    auto& buf    = g_ch[ch];
    auto  stream = g_stream[ch];
    const float* qm = (ch == 0) ? g_d_qm[0] : g_d_qm[1];

    const int bx = padded_w / 8, by = padded_h / 8;
    const int total = bx * by;
    const int grid  = (total + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;

    encode_blocks_kernel<<<grid, THREADS_PER_BLK, 0, stream>>>(
        buf.d_src, buf.d_prev, buf.d_curr, buf.d_coeff,
        qm, padded_w, bx, total, is_keyframe);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(coeff_out, buf.d_coeff,
               buf.coeffs * sizeof(int16_t), cudaMemcpyDeviceToHost, stream));
}


void cuda_decode_channel(int ch, const int16_t* coeff_in,
                         int padded_w, int padded_h, bool is_keyframe) {
    auto& buf    = g_ch[ch];
    auto  stream = g_stream[ch];
    const float* qm = (ch == 0) ? g_d_qm[0] : g_d_qm[1];

    const int bx = padded_w / 8, by = padded_h / 8;
    const int total = bx * by;
    const int grid  = (total + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;

    CUDA_CHECK(cudaMemcpyAsync(buf.d_coeff, coeff_in,
               buf.coeffs * sizeof(int16_t), cudaMemcpyHostToDevice, stream));

    decode_blocks_kernel<<<grid, THREADS_PER_BLK, 0, stream>>>(
        buf.d_coeff, buf.d_prev, buf.d_curr,
        qm, padded_w, bx, total, is_keyframe);
    CUDA_CHECK(cudaGetLastError());
}


void cuda_download_and_convert_rgb(uint8_t* rgb_host,
                                   int width, int height, int channels) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    ycbcr_to_rgb_kernel<<<grid, block>>>(
        g_ch[0].d_curr, g_ch[1].d_curr, g_ch[2].d_curr,
        g_d_rgb_out, width, height, channels, g_pw[0], g_pw[1]);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(rgb_host, g_d_rgb_out, g_rgb_bytes, cudaMemcpyDeviceToHost));
}


void cuda_swap_recon() {
    for (int ch = 0; ch < g_num_ch; ++ch)
        std::swap(g_ch[ch].d_prev, g_ch[ch].d_curr);
}

void cuda_sync_channel(int ch) { CUDA_CHECK(cudaStreamSynchronize(g_stream[ch])); }
void cuda_sync_all()           { for (int i = 0; i < g_num_ch; ++i) cuda_sync_channel(i); }
