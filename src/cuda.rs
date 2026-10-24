//! NVIDIA GPU backend. Loads the CUDA driver (nvcuda.dll) and NVRTC at runtime, compiles the kernels
//! below to PTX for the installed GPU, and runs the whole transformer on the device. No build-time CUDA
//! dependency: without a GPU the binary still works on the CPU.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::{Arc, OnceLock};

use crate::gguf::{GgmlType, Gguf};
use crate::model::Config;
use crate::profile;
use crate::quant::f16_to_f32;

const MAX_BATCH: usize = 1024; // prompt tokens per GPU pass: big batches keep the attention kernel busy
const MATMUL_TB: usize = 8; // tokens per matmul block row (must match TB in the kernel source)
/// Rows per warp in the K-quant generation matmuls (passed to the kernels as -DRPW).
/// BETTERLLAMA_RPW overrides it for tuning.
fn kq_rpw() -> usize {
    static RPW: OnceLock<usize> = OnceLock::new();
    *RPW.get_or_init(|| std::env::var("BETTERLLAMA_RPW").ok().and_then(|v| v.parse().ok()).filter(|&v| (1..=16).contains(&v)).unwrap_or(1))
}
/// Rows per warp in the K-quant decode-batch matmuls (speculative verify; -DDP4A_BATCH_RPW).
fn batch_rpw() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| std::env::var("BETTERLLAMA_BATCH_RPW").ok().and_then(|v| v.parse().ok()).filter(|&v| (1..=8).contains(&v)).unwrap_or(2))
}

/// Activation values per int8 scale in the K-quant prompt matmuls (passed as -DQCH): 128 or 256.
/// 128 is ~1% slower than 256 and about halves the log-probability drift; BETTERLLAMA_QCH=256 overrides.
fn qch() -> usize {
    static QCH: OnceLock<usize> = OnceLock::new();
    *QCH.get_or_init(|| std::env::var("BETTERLLAMA_QCH").ok().and_then(|v| v.parse().ok()).filter(|&v| v == 128 || v == 256).unwrap_or(128))
}
const KQ_TB: usize = 4; // tokens per block row in the K-quant generation matmuls (must match TBK)
const TILE: usize = 64; // must match TILE in the kernel source
const TILED_MIN_BATCH: usize = 16; // from this many tokens, the int8 tiled matmul beats one-warp-per-row
const ATT_MAX_CHUNK: usize = 256; // must match ATT_MAX_CHUNK
const ATT_MAX_HD: usize = 128; // must match ATT_MAX_HD
const ATT_MAX_G: usize = 8; // query heads per kv head the GQA attention kernel handles (must match ATT_MAX_G)
const SPLIT_MIN_SPAN: usize = 32; // contexts up to this many positions are not split across blocks (must match ATT_SPLIT_MIN_SPAN)
const ATT_TARGET_BLOCKS: usize = 96; // aim for a few blocks per SM when splitting attention for one token
const PREFILL_QT: usize = 32; // query tokens per prompt-attention block (must match PQT)
const PREFILL_KT: usize = 32; // key positions per shared-memory tile (must match PKT)
const PREFILL_TC_QT: usize = 64; // query tokens per tensor-core prompt-attention block (must match BQ)
const SPLIT_MAX_BATCH: usize = TILED_MIN_BATCH - 1; // tokens per split-attention launch
/// Most tokens per decode batch: one token, or a speculative verify batch.
pub const DECODE_MAX_BATCH: usize = 8;

const KERNELS: &str = r#"
#define TB 8
#define NEG_INF __int_as_float(0xff800000) // NVRTC has no <math.h> INFINITY
extern "C" {

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
// block-wide reductions; blockDim.x must be a multiple of 32
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : NEG_INF;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// ---- weight formats (repacked at upload so every field is naturally aligned) ----
// fmt 0, Q8_0: int8 values (rows x cols) + a separate f32 scale per 32
// fmt 1, Q4_K: the GGUF block as is, 144 bytes per 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins,
//              128 bytes of nibbles
// fmt 2, Q6_K: 208-byte blocks of 256 (128 bytes low nibbles, 64 bytes 2-bit highs, 16 int8 scales), and
//              the f32 d of each block in a separate array (wd)
// Both block sizes are multiples of 16 bytes, so the quant bytes can be read with 128-bit loads.
#define Q4K_BYTES 144
#define Q6K_BYTES 208

__device__ __forceinline__ float h2f(unsigned int h) {
    float f;
    asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.f32.f16 %0, t;}" : "=f"(f) : "r"(h));
    return f;
}
// d and dmin of a Q4_K block (one 32-bit load)
__device__ __forceinline__ void q4k_dm(const unsigned char* b, float* d, float* dmin) {
    unsigned int w = *reinterpret_cast<const unsigned int*>(b);
    *d = h2f(w & 0xFFFF);
    *dmin = h2f(w >> 16);
}

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* s, int* sc, int* m) {
    if (j < 4) { *sc = s[j] & 63; *m = s[j + 4] & 63; }
    else { *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}

// Same, for sub-blocks j and j+1 (j even), from the 12 scale bytes as three words.
__device__ __forceinline__ void q4k_scale_min2w(int j, unsigned int w0, unsigned int w1, unsigned int w2, int* sc1, int* m1, int* sc2, int* m2) {
    #define SB(i) ((int)(((i) < 4 ? w0 : (i) < 8 ? w1 : w2) >> (8 * ((i) & 3)) & 0xFF))
    if (j < 4) {
        *sc1 = SB(j) & 63;            *m1 = SB(j + 4) & 63;
        *sc2 = SB(j + 1) & 63;        *m2 = SB(j + 5) & 63;
    } else {
        *sc1 = (SB(j + 4) & 0xF) | ((SB(j - 4) >> 6) << 4); *m1 = (SB(j + 4) >> 4) | ((SB(j) >> 6) << 4);
        *sc2 = (SB(j + 5) & 0xF) | ((SB(j - 3) >> 6) << 4); *m2 = (SB(j + 5) >> 4) | ((SB(j + 1) >> 6) << 4);
    }
    #undef SB
}

// value i of a Q4_K / Q6_K row
__device__ float deq_q4k(const unsigned char* row, int i) {
    const unsigned char* b = row + (size_t)(i >> 8) * Q4K_BYTES;
    int k = i & 255, j2 = k >> 6, half = (k >> 5) & 1, l = k & 31;
    int sc, m;
    q4k_scale_min(2 * j2 + half, b + 4, &sc, &m);
    int q = b[16 + 32 * j2 + l];
    q = half ? (q >> 4) : (q & 15);
    float d, dmin;
    q4k_dm(b, &d, &dmin);
    return d * sc * q - dmin * m;
}
__device__ float deq_q6k(const unsigned char* row, const float* drow, int i) {
    const unsigned char* b = row + (size_t)(i >> 8) * Q6K_BYTES;
    int k = i & 255, half = k >> 7, g = (k >> 5) & 3, l = k & 31;
    int lo = b[64 * half + (g & 1) * 32 + l];
    lo = (g >= 2) ? (lo >> 4) : (lo & 15);
    int hi = (b[128 + 32 * half + l] >> (2 * g)) & 3;
    return drow[i >> 8] * (float)(signed char)b[192 + 8 * half + 2 * g + (l >> 4)] * ((lo | (hi << 4)) - 32);
}

// x[b] = embedding row of tokens[b], any weight format
__global__ void embed(int fmt, const unsigned char* __restrict__ w, const float* __restrict__ wd,
                      const int* __restrict__ tokens, float* __restrict__ x, int cols) {
    int b = blockIdx.x;
    size_t tok = tokens[b];
    float* o = x + (size_t)b * cols;
    if (fmt == 0) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            o[i] = wd[tok * (cols >> 5) + (i >> 5)] * (float)(signed char)w[tok * cols + i];
    } else {
        size_t row_bytes = (size_t)(cols >> 8) * (fmt == 1 ? Q4K_BYTES : Q6K_BYTES);
        const unsigned char* row = w + tok * row_bytes;
        const float* drow = wd + tok * (cols >> 8); // Q6_K
        for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = fmt == 1 ? deq_q4k(row, i) : deq_q6k(row, drow, i);
    }
}

// Per head of the packed [q | k | v] rows: RMSNorm of q and k (Qwen3; skipped when qn is null), RoPE on
// q and k, and k, v copied into the cache. grid (tokens, n_heads + 2 * n_kv), block = 64.
// neox: pairs (x[i], x[i + rope_dim/2]) (Qwen); otherwise adjacent pairs (x[2i], x[2i+1]) (GGUF llama).
__global__ void qk_norm_rope(float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                             float* __restrict__ kc, float* __restrict__ vc, int pos0, const int* __restrict__ posp, int n_heads,
                             int n_kv, int hd, int rope_dim, float base, int neox, float eps) {
    __shared__ float sh[32];
    if (posp) pos0 = *posp; // CUDA graphs: the position comes from device memory
    int b = blockIdx.x, h = blockIdx.y, pos = pos0 + b;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    float* row = qkv + (size_t)b * (q_dim + 2 * kv_dim);
    if (h >= n_heads + n_kv) { // a v head: straight into the cache (same for the whole block)
        int vh = h - n_heads - n_kv;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) vc[(size_t)pos * kv_dim + vh * hd + d] = row[q_dim + kv_dim + vh * hd + d];
        return;
    }
    float* vec = row + h * hd; // k heads directly follow the q heads
    if (qn) {
        const float* w = h < n_heads ? qn : kn;
        float ss = 0.f;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += vec[d] * vec[d];
        ss = block_sum(ss, sh);
        float scale = rsqrtf(ss / hd + eps);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) vec[d] = vec[d] * scale * w[d];
        __syncthreads();
    }
    int half = rope_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(base, -2.0f * (float)i / (float)rope_dim);
        float s, c;
        sincosf((float)pos * freq, &s, &c);
        int ia = neox ? i : 2 * i, ib = neox ? i + half : 2 * i + 1;
        float a = vec[ia], bb = vec[ib];
        vec[ia] = a * c - bb * s;
        vec[ib] = a * s + bb * c;
    }
    if (h >= n_heads) {
        __syncthreads();
        int kh = h - n_heads;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) kc[(size_t)pos * kv_dim + kh * hd + d] = vec[d];
    }
}

// int8 quantisation (as quantize_q8) of 32-value blocks held 4 per lane by 8 adjacent lanes; the whole
// warp must call it. Lane i of the row's float4s writes its 4 bytes; the block's first lane its scale and sum.
__device__ __forceinline__ void quant8x4(float4 v, bool ok, int i, signed char* __restrict__ q, float* __restrict__ d, int* __restrict__ sums) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    int q0 = (int)roundf(v.x * inv), q1 = (int)roundf(v.y * inv), q2 = (int)roundf(v.z * inv), q3 = (int)roundf(v.w * inv);
    int s = q0 + q1 + q2 + q3;
    for (int o = 1; o < 8; o <<= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (ok) {
        *reinterpret_cast<int*>(q + 4 * (size_t)i) = (q0 & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((unsigned)q3 << 24);
        if ((i & 7) == 0) { d[i >> 3] = dd; sums[i >> 3] = s; }
    }
}

// one block per row
__global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                        int cols, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* o = out + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    float scale = rsqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = xr[i] * scale * w[i];
}

// RMSNorm straight to int8 activations (one token per block, up to 1024 threads, cols a multiple of 32):
// the one-token K-quant matmuls take xq / xd directly, so there is no f32 copy and no quantize launch.
__global__ void rmsnorm_q8(const float* __restrict__ x, const float* __restrict__ w, signed char* __restrict__ xq,
                           float* __restrict__ xd, int* __restrict__ xsum, int cols, float eps) {
    __shared__ float sh[32];
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * cols);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    int nv = cols >> 2;
    float ss = 0.f;
    for (int i = threadIdx.x; i < nv; i += blockDim.x) { float4 v = xr[i]; ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
    ss = block_sum(ss, sh);
    float scale = rsqrtf(ss / cols + eps);
    size_t r = blockIdx.x;
    for (int base = 0; base < nv; base += blockDim.x) { // every lane takes part in the shuffles
        int i = base + threadIdx.x;
        bool ok = i < nv;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ok) {
            float4 a = xr[i], g = w4[i];
            v = make_float4(a.x * scale * g.x, a.y * scale * g.y, a.z * scale * g.z, a.w * scale * g.w);
        }
        quant8x4(v, ok, i, xq + r * cols, xd + r * (cols >> 5), xsum + r * (cols >> 5));
    }
}

// Generation matmuls: out[b*out_stride + out_off + r] (+)= dot(W[r], x[b]), f32 activations.
// One warp per row, TB tokens per block in y.

// Q8_0: W is int8 (rows x cols) with one f32 scale per 32 values.
__global__ void matmul_q8(const signed char* __restrict__ wq, const float* __restrict__ wd,
                          const float* __restrict__ x, float* __restrict__ out,
                          int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int b0 = blockIdx.y * TB;
    int nb = min(TB, n - b0);
    float acc[TB];
    #pragma unroll
    for (int j = 0; j < TB; j++) acc[j] = 0.f;
    const signed char* wr = wq + (size_t)row * cols;
    const float* dr = wd + (size_t)row * (cols >> 5);
    for (int i = lane * 4; i < cols; i += 128) {
        int packed = *reinterpret_cast<const int*>(wr + i);
        float d = dr[i >> 5];
        float w0 = d * (float)(signed char)(packed);
        float w1 = d * (float)(signed char)(packed >> 8);
        float w2 = d * (float)(signed char)(packed >> 16);
        float w3 = d * (float)(signed char)(packed >> 24);
        #pragma unroll
        for (int j = 0; j < TB; j++) {
            if (j < nb) {
                float4 xv = *reinterpret_cast<const float4*>(x + (size_t)(b0 + j) * cols + i);
                acc[j] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < TB; j++) {
        float v = warp_sum(acc[j]);
        if (lane == 0 && j < nb) {
            size_t idx = (size_t)(b0 + j) * out_stride + out_off + row;
            out[idx] = accumulate ? out[idx] + v : v;
        }
    }
}

// K-quant generation matmuls: each warp handles RPW rows so the activations a lane needs are loaded
// once into registers and reused for all of them (4-bit weights are ~7x smaller than f32 activations,
// so re-reading the activations per row would dominate). TBK tokens per block in y.
// Compiled twice: TBK = 1 for generation (few registers, so many warps keep loads in flight) and
// TBK = 4 for short multi-token batches. RPW (rows per warp) comes from the compile options.

}  // extern "C": the templates below need C++ linkage

// Q4_K: per super-block each lane takes 4 bytes of a 64-value chunk, i.e. 4 values of two sub-blocks.
template <int TBK>
__device__ void q4k_matvec(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ out,
                           int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RPW;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int b0 = blockIdx.y * TBK, nb = min(TBK, n - b0);
    int nsb = cols >> 8;
    int j2 = lane >> 3, sub = lane & 7;
    size_t row_bytes = (size_t)nsb * Q4K_BYTES;
    float acc[RPW][TBK];
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) acc[r][j] = 0.f;
    #pragma unroll 4 // several super-blocks' loads in flight before any is used
    for (int sb = 0; sb < nsb; sb++) {
        int base = sb * 256 + 64 * j2 + 4 * sub;
        float4 xl[TBK], xh[TBK];
        float sl[TBK], sh[TBK];
        #pragma unroll
        for (int j = 0; j < TBK; j++) {
            if (j < nb) {
                const float* xr = x + (size_t)(b0 + j) * cols + base;
                xl[j] = *reinterpret_cast<const float4*>(xr);
                xh[j] = *reinterpret_cast<const float4*>(xr + 32);
                sl[j] = xl[j].x + xl[j].y + xl[j].z + xl[j].w;
                sh[j] = xh[j].x + xh[j].y + xh[j].z + xh[j].w;
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            if (row0 + r < rows) {
                const unsigned char* b = w + (size_t)(row0 + r) * row_bytes + (size_t)sb * Q4K_BYTES;
                float d, dmin;
                q4k_dm(b, &d, &dmin);
                int sc1, m1, sc2, m2;
                q4k_scale_min(2 * j2, b + 4, &sc1, &m1);
                q4k_scale_min(2 * j2 + 1, b + 4, &sc2, &m2);
                int p = *(const int*)(b + 16 + 32 * j2 + 4 * sub);
                float a1 = d * sc1, c1 = dmin * m1, a2 = d * sc2, c2 = dmin * m2;
                float lo0 = p & 15, lo1 = (p >> 8) & 15, lo2 = (p >> 16) & 15, lo3 = (p >> 24) & 15;
                float hi0 = (p >> 4) & 15, hi1 = (p >> 12) & 15, hi2 = (p >> 20) & 15, hi3 = (p >> 28) & 15;
                #pragma unroll
                for (int j = 0; j < TBK; j++) {
                    if (j < nb) {
                        acc[r][j] += a1 * (lo0 * xl[j].x + lo1 * xl[j].y + lo2 * xl[j].z + lo3 * xl[j].w) - c1 * sl[j]
                                   + a2 * (hi0 * xh[j].x + hi1 * xh[j].y + hi2 * xh[j].z + hi3 * xh[j].w) - c2 * sh[j];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < nb && row0 + r < rows) {
                size_t idx = (size_t)(b0 + j) * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Q6_K: 16 lanes per 128-value half; each lane takes 2 adjacent positions of all 4 groups (8 values).
template <int TBK>
__device__ void q6k_matvec(const unsigned char* __restrict__ w, const float* __restrict__ wd, const float* __restrict__ x,
                           float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RPW;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int b0 = blockIdx.y * TBK, nb = min(TBK, n - b0);
    int nsb = cols >> 8;
    int half = lane >> 4, l = 2 * (lane & 15), is = l >> 4;
    size_t row_bytes = (size_t)nsb * Q6K_BYTES;
    float acc[RPW][TBK];
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) acc[r][j] = 0.f;
    #pragma unroll 4 // several super-blocks' loads in flight before any is used
    for (int sb = 0; sb < nsb; sb++) {
        float2 xv[4][TBK]; // [group][token]
        #pragma unroll
        for (int g = 0; g < 4; g++)
            #pragma unroll
            for (int j = 0; j < TBK; j++)
                if (j < nb) xv[g][j] = *reinterpret_cast<const float2*>(x + (size_t)(b0 + j) * cols + sb * 256 + 128 * half + 32 * g + l);
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            if (row0 + r < rows) {
                const unsigned char* b = w + (size_t)(row0 + r) * row_bytes + (size_t)sb * Q6K_BYTES;
                float d = wd[(size_t)(row0 + r) * nsb + sb];
                int qla =*(const unsigned short*)(b + 64 * half + l);
                int qlb = *(const unsigned short*)(b + 64 * half + 32 + l);
                int qh = *(const unsigned short*)(b + 128 + 32 * half + l);
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int ql = (g & 1) ? qlb : qla;
                    int s4 = (g >= 2) ? 4 : 0;
                    float q0 = (float)((((ql >> s4) & 15) | (((qh >> (2 * g)) & 3) << 4)) - 32);
                    float q1 = (float)((((ql >> (8 + s4)) & 15) | (((qh >> (8 + 2 * g)) & 3) << 4)) - 32);
                    float s = d * (float)(signed char)b[192 + 8 * half + 2 * g + is];
                    #pragma unroll
                    for (int j = 0; j < TBK; j++)
                        if (j < nb) acc[r][j] += s * (q0 * xv[g][j].x + q1 * xv[g][j].y);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < nb && row0 + r < rows) {
                size_t idx = (size_t)(b0 + j) * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Decode matmuls with int8 activations (quantize_q8 / rmsnorm_q8 / silu_q8 first): weights stay packed
// and go straight into __dp4a (4 multiply-adds per instruction), instead of being unpacked to floats.
// Templated on NT, the most tokens per launch: NT = 1 for normal decoding, NT = 8 for speculative verify
// batches, where each weight block is read and decoded once and applied to every token (each token's
// activations come from cache as needed), and on R, the rows per warp. A token's arithmetic is the same sequence of operations in
// both instances, so it gets bit-identical results alone or in a batch: speculative decoding reproduces
// plain greedy output exactly. RPW rows per warp (K-quants).
#define DP4A_PARAMS const unsigned char* __restrict__ w, const float* __restrict__ wd, const signed char* __restrict__ xq, \
                    const float* __restrict__ xd, const int* __restrict__ xs, float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off

// Q4_K: each lane takes 16 bytes of a 32-byte nibble chunk = 16 values of each of two sub-blocks (32
// values), plus the block's 16-byte header (d, dmin and all 12 scale bytes) in one load; 8 lanes cover a
// super-block, so the warp walks four super-blocks per step. The R rows of a warp are decoded first, then
// each token's activations are loaded once for all of them. The "min" term uses the activation block
// sums from the quantiser (xs): of the two lanes sharing a 32-value block, the first adds it.
template <int NT, int R>
__device__ __forceinline__ void q4k_dp4a(DP4A_PARAMS) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int nsb = cols >> 8;
    int qsel = lane >> 3, l8 = lane & 7, j2 = l8 >> 1, hf = l8 & 1;
    size_t row_bytes = (size_t)nsb * Q4K_BYTES;
    float acc[R][NT];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) acc[r][j] = 0.f;
    #pragma unroll 2 // more super-blocks' loads in flight before any is used
    for (int sb = qsel; sb < nsb; sb += 4) {
        int pos = sb * 256 + 64 * j2 + 16 * hf;
        int4 lo4[R], hi4[R];
        float d[R], dmin[R];
        int sc1[R], m1[R], sc2[R], m2[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int row = min(row0 + r, rows - 1); // past the end: computed, never stored
            const unsigned char* b = w + (size_t)row * row_bytes + (size_t)sb * Q4K_BYTES;
            int4 hdr = *reinterpret_cast<const int4*>(b);
            int4 p = *reinterpret_cast<const int4*>(b + 16 + 32 * j2 + 16 * hf);
            d[r] = h2f((unsigned)hdr.x & 0xFFFF);
            dmin[r] = hf ? 0.f : h2f((unsigned)hdr.x >> 16);
            q4k_scale_min2w(2 * j2, hdr.y, hdr.z, hdr.w, &sc1[r], &m1[r], &sc2[r], &m2[r]);
            lo4[r] = make_int4(p.x & 0x0F0F0F0F, p.y & 0x0F0F0F0F, p.z & 0x0F0F0F0F, p.w & 0x0F0F0F0F);
            hi4[r] = make_int4((p.x >> 4) & 0x0F0F0F0F, (p.y >> 4) & 0x0F0F0F0F, (p.z >> 4) & 0x0F0F0F0F, (p.w >> 4) & 0x0F0F0F0F);
        }
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                const signed char* x = xq + (size_t)j * cols;
                const float* dx = xd + (size_t)j * (cols >> 5);
                const int* sx = xs + (size_t)j * (cols >> 5);
                int4 xl = *reinterpret_cast<const int4*>(x + pos), xh = *reinterpret_cast<const int4*>(x + pos + 32);
                float dl = dx[pos >> 5], dh = dx[(pos >> 5) + 1];
                float sl = dl * (float)sx[pos >> 5], sh = dh * (float)sx[(pos >> 5) + 1];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    int lo = __dp4a(lo4[r].x, xl.x, __dp4a(lo4[r].y, xl.y, __dp4a(lo4[r].z, xl.z, __dp4a(lo4[r].w, xl.w, 0))));
                    int hi = __dp4a(hi4[r].x, xh.x, __dp4a(hi4[r].y, xh.y, __dp4a(hi4[r].z, xh.z, __dp4a(hi4[r].w, xh.w, 0))));
                    acc[r][j] += d[r] * (sc1[r] * dl * (float)lo + sc2[r] * dh * (float)hi) - dmin[r] * (m1[r] * sl + m2[r] * sh);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < n && row0 + r < rows) {
                size_t idx = (size_t)j * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Q6_K: per 128-value half, 4 lanes take groups {0,2} and 4 lanes groups {1,3}; each lane 8 consecutive
// positions of both groups (they share the low-nibble bytes: low nibble = first group, high = second).
// A half-warp covers a super-block, so the warp walks two super-blocks per step. R rows decoded first.
template <int NT, int R>
__device__ __forceinline__ void q6k_dp4a(DP4A_PARAMS) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int nsb = cols >> 8;
    int hl = lane >> 4, l16 = lane & 15;
    int half = l16 >> 3, gsel = (l16 >> 2) & 1, l = 8 * (l16 & 3), is = l >> 4;
    int ga = gsel, gb = gsel + 2;
    size_t row_bytes = (size_t)nsb * Q6K_BYTES;
    float acc[R][NT];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) acc[r][j] = 0.f;
    #pragma unroll 2 // more super-blocks' loads in flight before any is used
    for (int sb = hl; sb < nsb; sb += 2) {
        int pa = sb * 256 + 128 * half + 32 * ga + l, pb = pa + 64;
        int qa0[R], qa1[R], qb0[R], qb1[R], sa[R], sbb[R];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int row = min(row0 + r, rows - 1); // past the end: computed, never stored
            const unsigned char* b = w + (size_t)row * row_bytes + (size_t)sb * Q6K_BYTES;
            d[r] = wd[(size_t)row * nsb + sb];
            int ql0 = *(const int*)(b + 64 * half + 32 * gsel + l), ql1 = *(const int*)(b + 64 * half + 32 * gsel + l + 4);
            int qh0 = *(const int*)(b + 128 + 32 * half + l), qh1 = *(const int*)(b + 132 + 32 * half + l);
            qa0[r] = __vsub4((ql0 & 0x0F0F0F0F) | (((qh0 >> (2 * ga)) & 0x03030303) << 4), 0x20202020);
            qa1[r] = __vsub4((ql1 & 0x0F0F0F0F) | (((qh1 >> (2 * ga)) & 0x03030303) << 4), 0x20202020);
            qb0[r] = __vsub4(((ql0 >> 4) & 0x0F0F0F0F) | (((qh0 >> (2 * gb)) & 0x03030303) << 4), 0x20202020);
            qb1[r] = __vsub4(((ql1 >> 4) & 0x0F0F0F0F) | (((qh1 >> (2 * gb)) & 0x03030303) << 4), 0x20202020);
            const signed char* sc = (const signed char*)(b + 192 + 8 * half + is);
            sa[r] = sc[2 * ga];
            sbb[r] = sc[2 * gb];
        }
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                const signed char* x = xq + (size_t)j * cols;
                const float* dx = xd + (size_t)j * (cols >> 5);
                int2 xa = *reinterpret_cast<const int2*>(x + pa), xb = *reinterpret_cast<const int2*>(x + pb);
                float da = dx[pa >> 5], db = dx[pb >> 5];
                #pragma unroll
                for (int r = 0; r < R; r++)
                    acc[r][j] += d[r] * (sa[r] * da * (float)__dp4a(qa0[r], xa.x, __dp4a(qa1[r], xa.y, 0))
                                      + sbb[r] * db * (float)__dp4a(qb0[r], xb.x, __dp4a(qb1[r], xb.y, 0)));
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < n && row0 + r < rows) {
                size_t idx = (size_t)j * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}
// Q8_0 (used for draft models): one row per warp, each lane 16 int8 weights and 16 activations per step.
template <int NT>
__device__ __forceinline__ void q8_dp4a(DP4A_PARAMS) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const signed char* wr = reinterpret_cast<const signed char*>(w) + (size_t)row * cols;
    const float* dr = wd + (size_t)row * (cols >> 5);
    float acc[NT];
    #pragma unroll
    for (int j = 0; j < NT; j++) acc[j] = 0.f;
    #pragma unroll 2
    for (int i = lane * 16; i < cols; i += 512) {
        int4 wv = *reinterpret_cast<const int4*>(wr + i);
        float dw = dr[i >> 5];
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                int4 xv = *reinterpret_cast<const int4*>(xq + (size_t)j * cols + i);
                int s = __dp4a(wv.x, xv.x, __dp4a(wv.y, xv.y, __dp4a(wv.z, xv.z, __dp4a(wv.w, xv.w, 0))));
                acc[j] += dw * xd[(size_t)j * (cols >> 5) + (i >> 5)] * (float)s;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; j++) {
        float v = warp_sum(acc[j]);
        if (lane == 0 && j < n) {
            size_t idx = (size_t)j * out_stride + out_off + row;
            out[idx] = accumulate ? out[idx] + v : v;
        }
    }
}

extern "C" {

// the batch instances get a register budget, so the compiler cannot hoist every token's loads at once
#define DP4A_ARGS w, wd, xq, xd, xs, out, rows, cols, n, accumulate, out_stride, out_off
__global__ void matmul_q4k_dp4a(DP4A_PARAMS) { q4k_dp4a<1, RPW>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q4k_dp4a8(DP4A_PARAMS) { q4k_dp4a<8, DP4A_BATCH_RPW>(DP4A_ARGS); }
__global__ void matmul_q6k_dp4a(DP4A_PARAMS) { q6k_dp4a<1, RPW>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q6k_dp4a8(DP4A_PARAMS) { q6k_dp4a<8, DP4A_BATCH_RPW>(DP4A_ARGS); }
__global__ void matmul_q8_dp4a(DP4A_PARAMS) { q8_dp4a<1>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q8_dp4a8(DP4A_PARAMS) { q8_dp4a<8>(DP4A_ARGS); }
#define KQ_ARGS const unsigned char* __restrict__ w, const float* __restrict__ wd, const float* __restrict__ x, \
                float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off
__global__ void matmul_q4k_1(KQ_ARGS) { q4k_matvec<1>(w, x, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void matmul_q4k_4(KQ_ARGS) { q4k_matvec<4>(w, x, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void matmul_q6k_1(KQ_ARGS) { q6k_matvec<1>(w, wd, x, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void matmul_q6k_4(KQ_ARGS) { q6k_matvec<4>(w, wd, x, out, rows, cols, n, accumulate, out_stride, out_off); }

#define ATT_MAX_CHUNK 256
#define ATT_MAX_HD 128
#define ATT_MAX_G 8
#define ATT_SPLIT_MIN_SPAN 32

// Context slice per decode-attention block for a token that sees span positions: whole contexts up to
// ATT_SPLIT_MIN_SPAN, otherwise about want slices of 32..ATT_MAX_CHUNK positions. It depends on the
// token's own span only, so a token is attended the same way alone or inside a batch.
__device__ __forceinline__ int att_chunk_for(int span, int want) {
    return span <= ATT_SPLIT_MIN_SPAN ? ATT_MAX_CHUNK : min(max((span + want - 1) / want, 32), ATT_MAX_CHUNK);
}

// Flash-decoding attention for grouped-query models, part 1: one block per (kv head, token, slice of
// `chunk` positions) computes all G = n_heads / n_kv query heads that share the kv head, so each key
// and value row is read once. A warp scores one position at a time: every lane loads hd/32 consecutive
// floats of the key (one coalesced row read) and the dot products are reduced across the warp.
// Writes each head's unnormalised output, max score and exp-sum to `part`, or (nsplit == 1) the final
// normalised result to `out`. Needs hd a multiple of 32 (<= 128) and G <= ATT_MAX_G.
__global__ void attention_partial_gqa(const float* __restrict__ qkv, const float* __restrict__ kc, const float* __restrict__ vc,
                                      float* __restrict__ part, float* __restrict__ out, int pos0, const int* __restrict__ posp,
                                      int n_heads, int n_kv, int hd, float scale, int nsplit, int want) {
    if (posp) pos0 = *posp;
    __shared__ float sc[ATT_MAX_G][ATT_MAX_CHUNK];
    __shared__ float red[32];
    int kvh = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int G = n_heads / n_kv;
    int span = pos0 + b + 1; // causal
    int chunk = att_chunk_for(span, want);
    int t0 = s * chunk, len = min(span, t0 + chunk) - t0;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    if (len <= 0) { // same for the whole block, so returning before the barriers is safe
        for (int g = 0; g < G; g++) {
            float* p = part + (((size_t)b * n_heads + kvh * G + g) * nsplit + s) * (hd + 2);
            for (int d = tid; d < hd; d += blockDim.x) p[d] = 0.f;
            if (tid == 0) { p[hd] = NEG_INF; p[hd + 1] = 0.f; }
        }
        return;
    }
    int per = hd >> 5; // floats per lane: 2 (hd 64) or 4 (hd 128)
    float q[ATT_MAX_G][4];
    const float* qrow = qkv + (size_t)b * (q_dim + 2 * kv_dim) + kvh * G * hd + lane * per;
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++)
        #pragma unroll
        for (int e = 0; e < 4; e++) q[g][e] = (g < G && e < per) ? qrow[g * hd + e] * scale : 0.f;

    for (int t = warp; t < len; t += nwarps) {
        const float* kt = kc + (size_t)(t0 + t) * kv_dim + kvh * hd + lane * per;
        float k4[4] = {0.f, 0.f, 0.f, 0.f};
        if (per == 4) { float4 v = *reinterpret_cast<const float4*>(kt); k4[0] = v.x; k4[1] = v.y; k4[2] = v.z; k4[3] = v.w; }
        else { float2 v = *reinterpret_cast<const float2*>(kt); k4[0] = v.x; k4[1] = v.y; }
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) {
            if (g < G) {
                float d = q[g][0] * k4[0] + q[g][1] * k4[1] + q[g][2] * k4[2] + q[g][3] * k4[3];
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
                if (lane == 0) sc[g][t] = d;
            }
        }
    }
    __syncthreads();
    float mxg[ATT_MAX_G], seg[ATT_MAX_G];
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++) {
        if (g < G) {
            float m = NEG_INF;
            for (int t = tid; t < len; t += blockDim.x) m = fmaxf(m, sc[g][t]);
            m = block_max(m, red);
            float se = 0.f;
            for (int t = tid; t < len; t += blockDim.x) {
                float e = expf(sc[g][t] - m);
                sc[g][t] = e;
                se += e;
            }
            se = block_sum(se, red); // barrier: the probabilities are visible to every thread
            mxg[g] = m;
            seg[g] = se;
        }
    }
    if (tid < hd) {
        float a[ATT_MAX_G];
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) a[g] = 0.f;
        const float* vcol = vc + (size_t)t0 * kv_dim + kvh * hd + tid;
        for (int t = 0; t < len; t++) {
            float v = vcol[(size_t)t * kv_dim];
            #pragma unroll
            for (int g = 0; g < ATT_MAX_G; g++) if (g < G) a[g] += sc[g][t] * v;
        }
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) {
            if (g < G) {
                int h = kvh * G + g;
                if (nsplit == 1) {
                    out[(size_t)b * q_dim + h * hd + tid] = a[g] * (1.f / seg[g]); // as attention_combine
                } else {
                    float* p = part + (((size_t)b * n_heads + h) * nsplit + s) * (hd + 2);
                    p[tid] = a[g];
                    if (tid == 0) { p[hd] = mxg[g]; p[hd + 1] = seg[g]; }
                }
            }
        }
    }
}

// Flash-decoding attention, part 1 (one head per block; used when attention_partial_gqa does not apply):
// one block per (head, token, slice of `chunk` positions). Writes the slice's unnormalised output plus
// its max score and exp-sum to `part` (hd + 2 floats).
// With nsplit == 1 there is nothing to merge: the block writes the normalised result straight to `out`.
__global__ void attention_partial(const float* __restrict__ qkv, const float* __restrict__ kc, const float* __restrict__ vc,
                                  float* __restrict__ part, float* __restrict__ out, int pos0, const int* __restrict__ posp, int n_heads, int n_kv, int hd,
                                  float scale, int nsplit, int want) {
    if (posp) pos0 = *posp;
    __shared__ float sc[ATT_MAX_CHUNK];
    __shared__ float qs[ATT_MAX_HD];
    __shared__ float red[32];
    __shared__ float accs[4][ATT_MAX_HD];
    int h = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int span = pos0 + b + 1; // causal
    int chunk = att_chunk_for(span, want);
    int t0 = s * chunk, len = min(span, t0 + chunk) - t0;
    float* p = part + (((size_t)b * n_heads + h) * nsplit + s) * (hd + 2);
    if (len <= 0) { // same for the whole block, so returning before the barriers is safe
        for (int d = threadIdx.x; d < hd; d += blockDim.x) p[d] = 0.f;
        if (threadIdx.x == 0) { p[hd] = NEG_INF; p[hd + 1] = 0.f; }
        return;
    }
    int kvh = h / (n_heads / n_kv);
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    const float* qh = qkv + (size_t)b * (q_dim + 2 * kv_dim) + h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qh[d];
    __syncthreads();

    float mx = NEG_INF;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        const float4* kt = reinterpret_cast<const float4*>(kc + (size_t)(t0 + t) * kv_dim + kvh * hd);
        float acc = 0.f;
        for (int d = 0; d < hd / 4; d++) {
            float4 k4 = kt[d];
            acc += qs[4 * d] * k4.x + qs[4 * d + 1] * k4.y + qs[4 * d + 2] * k4.z + qs[4 * d + 3] * k4.w;
        }
        acc *= scale;
        sc[t] = acc;
        mx = fmaxf(mx, acc);
    }
    mx = block_max(mx, red);
    float se = 0.f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        float e = expf(sc[t] - mx);
        sc[t] = e;
        se += e;
    }
    se = block_sum(se, red); // barrier: all scores visible

    // threads form `groups` copies of the head dimension, each summing every groups-th position
    int groups = blockDim.x / hd;
    int g = threadIdx.x / hd, d = threadIdx.x % hd;
    if (g < groups) {
        float acc = 0.f;
        for (int t = g; t < len; t += groups) acc += sc[t] * vc[(size_t)(t0 + t) * kv_dim + kvh * hd + d];
        accs[g][d] = acc;
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        float acc = 0.f;
        for (int k = 0; k < groups; k++) acc += accs[k][threadIdx.x];
        if (nsplit == 1) out[(size_t)b * q_dim + h * hd + threadIdx.x] = acc * (1.f / se); // as attention_combine
        else p[threadIdx.x] = acc;
    }
    if (threadIdx.x == 0 && nsplit > 1) { p[hd] = mx; p[hd + 1] = se; }
}

// FlashAttention-style kernel for prompts: one block per (head, tile of PQT query tokens). Key/value tiles
// of PKT positions are staged in shared memory once for all PQT queries, with a running (online) softmax
// per query. 128 threads: each owns 2 query rows (16 row pairs) x one of 8 lanes ("g"); for scores a thread
// covers keys g, g+8, g+16, g+24 of the tile, for outputs the dims g, g+8, ... (interleaved: no bank conflicts).
#define PQT 32
#define PKT 32
__global__ void attention_prefill(const float* __restrict__ qkv, const float* __restrict__ kc, const float* __restrict__ vc,
                                  float* __restrict__ out, int pos0, int n, int n_heads, int n_kv, int hd, float scale) {
    extern __shared__ float smem[];
    int ld = hd + 1; // padded row stride: rows land on different banks
    float* Ks = smem;
    float* Vs = Ks + PKT * ld;
    float* Qs = Vs + PKT * ld;
    float* S = Qs + PQT * ld; // PQT x (PKT + 1)
    const int sld = PKT + 1;
    int h = blockIdx.x, q0 = blockIdx.y * PQT;
    int kvh = h / (n_heads / n_kv);
    int q_dim = n_heads * hd, kv_dim = n_kv * hd, row_stride = q_dim + 2 * kv_dim;
    int tid = threadIdx.x, g = tid & 7;
    int r0 = (tid >> 3) * 2, r1 = r0 + 1;
    int dpt = hd >> 3; // output dims per thread (hd <= 128)
    int kv_len = pos0 + n;

    for (int idx = tid; idx < PQT * hd; idx += blockDim.x) {
        int r = idx / hd, d = idx % hd;
        Qs[r * ld + d] = (q0 + r < n) ? qkv[(size_t)(q0 + r) * row_stride + h * hd + d] : 0.f;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.f, 0.f}, acc[2][16];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[0][i] = 0.f; acc[1][i] = 0.f; }
    int lim0 = pos0 + q0 + r0, lim1 = lim0 + 1; // causal limits of the two rows
    int kmax = min(kv_len, pos0 + q0 + PQT);
    for (int t0 = 0; t0 < kmax; t0 += PKT) {
        __syncthreads(); // previous tile consumed (and Qs complete, the first time)
        for (int idx = tid; idx < PKT * hd; idx += blockDim.x) {
            int k = idx / hd, d = idx % hd, kp = t0 + k;
            bool ok = kp < kv_len;
            Ks[k * ld + d] = ok ? kc[(size_t)kp * kv_dim + kvh * hd + d] : 0.f;
            Vs[k * ld + d] = ok ? vc[(size_t)kp * kv_dim + kvh * hd + d] : 0.f;
        }
        __syncthreads();

        // scores for 2 rows x 4 keys, operands from registers: 6 shared reads per 8 FMAs
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int d = 0; d < hd; d++) {
            float qa = Qs[r0 * ld + d], qb = Qs[r1 * ld + d];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float kv = Ks[(g + 8 * j) * ld + d];
                s[0][j] += qa * kv;
                s[1][j] += qb * kv;
            }
        }
        // online softmax per row; the row's 8 threads are adjacent lanes of one warp
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int lim = i ? lim1 : lim0, r = i ? r1 : r0;
            float mx = NEG_INF;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int kp = t0 + g + 8 * j;
                s[i][j] = (kp < kv_len && kp <= lim) ? s[i][j] * scale : NEG_INF;
                mx = fmaxf(mx, s[i][j]);
            }
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            float m_new = fmaxf(m[i], mx);
            float alpha = (m[i] == NEG_INF) ? 0.f : expf(m[i] - m_new);
            float ls = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float pv = (s[i][j] == NEG_INF) ? 0.f : expf(s[i][j] - m_new);
                S[r * sld + g + 8 * j] = pv;
                ls += pv;
            }
            for (int o = 1; o < 8; o <<= 1) ls += __shfl_xor_sync(0xffffffff, ls, o);
            l[i] = l[i] * alpha + ls;
            m[i] = m_new;
            #pragma unroll
            for (int dd = 0; dd < 16; dd++) acc[i][dd] *= alpha;
        }
        __syncwarp(); // the rows' probabilities, written by their 8 lanes, are read by all 8

        // out += P V: per key 2 probabilities + dpt values for 2*dpt FMAs
        for (int k = 0; k < PKT; k++) {
            float pa = S[r0 * sld + k], pb = S[r1 * sld + k];
            const float* vr = Vs + k * ld + g;
            #pragma unroll
            for (int dd = 0; dd < 16; dd++) {
                if (dd < dpt) {
                    float v = vr[8 * dd];
                    acc[0][dd] += pa * v;
                    acc[1][dd] += pb * v;
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int r = i ? r1 : r0;
        if (q0 + r < n) {
            float inv = 1.f / l[i];
            for (int dd = 0; dd < dpt; dd++) out[(size_t)(q0 + r) * q_dim + h * hd + g + 8 * dd] = acc[i][dd] * inv;
        }
    }
}

}  // extern "C": the template below needs C++ linkage

// two floats -> one f16x2 word (low half first); sm_75 has no packed f32->f16x2 convert
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    unsigned r;
    asm("{.reg .f16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h};}" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}
#define MMA_F16(c, a0, a1, b) \
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
        : "+f"((c)[0]), "+f"((c)[1]), "+f"((c)[2]), "+f"((c)[3]) : "r"(a0), "r"(a1), "r"(b))
#define LDSM_X4(r, addr, trans) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4" trans ".shared.b16 {%0,%1,%2,%3}, [%4];" \
                 : "=r"((r)[0]), "=r"((r)[1]), "=r"((r)[2]), "=r"((r)[3]) : "r"(addr))

// Tensor-core version of attention_prefill (sm_75+, FlashAttention-2 style): one block per (head, 64
// query tokens), 4 warps of 16 query rows. Q, K and V go through f16 m16n8k8 mma with f32 accumulators:
// a warp keeps its Q rows as fragments in registers, S = Q K^T comes out in the accumulator layout,
// which is also the A-fragment layout, so the probabilities feed O += P V without a trip through
// shared memory. K/V tiles of 32 positions are converted to f16 into shared memory (rows padded by 16
// bytes: ldmatrix rows hit different banks); the same buffer stages Q at the start. Online softmax in
// base 2 (Q is pre-scaled by scale * log2 e). The row sums come from the tensor cores too (P x ones), so they\n// add up exactly the rounded probabilities that P V uses.
template <int HD>
__device__ void attn_prefill_tc(const float* __restrict__ qkv, const float* __restrict__ kc, const float* __restrict__ vc,
                                float* __restrict__ out, int pos0, int n, int n_heads, int n_kv, float scale) {
    constexpr int BQ = 64, BK = 32, LDW = HD / 2 + 4; // row stride in f16x2 words
    __shared__ __align__(16) unsigned sm[2 * BK * LDW]; // Q staging (BQ rows), then K (BK rows) | V (BK rows)
    unsigned* Ks = sm;
    unsigned* Vs = sm + BK * LDW;
    int h = blockIdx.x;
    int q0 = (gridDim.y - 1 - blockIdx.y) * BQ; // the longest (latest) query tiles start first
    int kvh = h / (n_heads / n_kv);
    int q_dim = n_heads * HD, kv_dim = n_kv * HD, row_stride = q_dim + 2 * kv_dim;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int kv_len = pos0 + n;
    const float qs = scale * 1.4426950408889634f;

    // stage Q (rows past n are zeros), then each warp takes its 16 rows as A fragments
    for (int idx = tid; idx < BQ * HD / 4; idx += 128) {
        int r = idx / (HD / 4), c4 = idx % (HD / 4);
        float4 v = q0 + r < n ? *reinterpret_cast<const float4*>(qkv + (size_t)(q0 + r) * row_stride + h * HD + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<uint2*>(sm + r * LDW + 2 * c4) = make_uint2(pack_h2(v.x * qs, v.y * qs), pack_h2(v.z * qs, v.w * qs));
    }
    __syncthreads();
    unsigned qf[HD / 8][2]; // per k8 step: rows 0-7, rows 8-15
    {
        int m = lane >> 3; // ldmatrix.x4 matrices: (rows 0-7 | 8-15) x (k0 | k0+8)
        unsigned base = (unsigned)__cvta_generic_to_shared(sm + (16 * warp + (lane & 7) + 8 * (m & 1)) * LDW + 4 * (m >> 1));
        #pragma unroll
        for (int k = 0; k < HD / 16; k++) {
            unsigned r[4];
            LDSM_X4(r, base + 32 * k, "");
            qf[2 * k][0] = r[0]; qf[2 * k][1] = r[1]; qf[2 * k + 1][0] = r[2]; qf[2 * k + 1][1] = r[3];
        }
    }

    float o[HD / 8][4]; // 16 rows x HD dims: per dim tile of 8, rows lo (c0, c1) and hi (c2, c3)
    #pragma unroll
    for (int d = 0; d < HD / 8; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
    float m_lo = NEG_INF, m_hi = NEG_INF;
    float l[4] = {0.f, 0.f, 0.f, 0.f}; // row sums of the f16 probabilities: P x ones (c0 = row lo, c2 = row hi)
    int p_lo = pos0 + q0 + 16 * warp + (lane >> 2), p_hi = p_lo + 8; // positions of this thread's two rows
    int warp_last = pos0 + q0 + 16 * warp + 15;
    int kmax = min(kv_len, pos0 + q0 + BQ);
    // ldmatrix row addresses: K (keys as rows, non-transposed = B fragments of K^T), V (transposed)
    unsigned kaddr = (unsigned)__cvta_generic_to_shared(Ks + (8 * 0 + (lane & 7)) * LDW + 4 * (lane >> 3));
    unsigned vaddr = (unsigned)__cvta_generic_to_shared(Vs + lane * LDW);

    for (int t0 = 0; t0 < kmax; t0 += BK) {
        __syncthreads(); // previous tile (or the Q staging) consumed
        for (int idx = tid; idx < BK * HD / 4; idx += 128) {
            int r = idx / (HD / 4), c4 = idx % (HD / 4), kp = t0 + r;
            float4 k4 = make_float4(0.f, 0.f, 0.f, 0.f), v4 = k4; // positions past the end must be finite: P = 0 there
            if (kp < kv_len) {
                k4 = *reinterpret_cast<const float4*>(kc + (size_t)kp * kv_dim + kvh * HD + 4 * c4);
                v4 = *reinterpret_cast<const float4*>(vc + (size_t)kp * kv_dim + kvh * HD + 4 * c4);
            }
            *reinterpret_cast<uint2*>(Ks + r * LDW + 2 * c4) = make_uint2(pack_h2(k4.x, k4.y), pack_h2(k4.z, k4.w));
            *reinterpret_cast<uint2*>(Vs + r * LDW + 2 * c4) = make_uint2(pack_h2(v4.x, v4.y), pack_h2(v4.z, v4.w));
        }
        __syncthreads();
        if (t0 > warp_last) continue; // causal: nothing of this tile is visible to the warp's rows

        // S = Q K^T: 4 key tiles of 8; per key tile ldmatrix.x4 covers 4 k8 steps (32 dims)
        float s[BK / 8][4];
        #pragma unroll
        for (int j = 0; j < BK / 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            #pragma unroll
            for (int k = 0; k < HD / 32; k++) {
                unsigned b[4];
                LDSM_X4(b, kaddr + (8 * j * LDW + 16 * k) * 4, "");
                #pragma unroll
                for (int q = 0; q < 4; q++) MMA_F16(s[j], qf[4 * k + q][0], qf[4 * k + q][1], b[q]);
            }
        }
        // causal / length mask, online softmax (row max over the 4 lanes of a quad)
        float mx_lo = NEG_INF, mx_hi = NEG_INF;
        #pragma unroll
        for (int j = 0; j < BK / 8; j++)
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int kp = t0 + 8 * j + 2 * (lane & 3) + e;
                if (kp > p_lo || kp >= kv_len) s[j][e] = NEG_INF;
                if (kp > p_hi || kp >= kv_len) s[j][2 + e] = NEG_INF;
                mx_lo = fmaxf(mx_lo, s[j][e]);
                mx_hi = fmaxf(mx_hi, s[j][2 + e]);
            }
        #pragma unroll
        for (int x = 1; x < 4; x <<= 1) {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, x));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, x));
        }
        // every row sees position 0 in the first tile, so the running max is finite from then on
        float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
        float a_lo = exp2f(m_lo - mn_lo), a_hi = exp2f(m_hi - mn_hi);
        m_lo = mn_lo; m_hi = mn_hi;
        unsigned pf[BK / 8][2]; // probabilities as A fragments, one per k8 step of keys
        #pragma unroll
        for (int j = 0; j < BK / 8; j++) {
            float p0 = exp2f(s[j][0] - mn_lo), p1 = exp2f(s[j][1] - mn_lo);
            float p2 = exp2f(s[j][2] - mn_hi), p3 = exp2f(s[j][3] - mn_hi);
            pf[j][0] = pack_h2(p0, p1);
            pf[j][1] = pack_h2(p2, p3);
        }
        // normalise by the sum of the rounded probabilities that P V actually uses
        l[0] *= a_lo; l[1] *= a_lo; l[2] *= a_hi; l[3] *= a_hi;
        #pragma unroll
        for (int j = 0; j < BK / 8; j++) MMA_F16(l, pf[j][0], pf[j][1], 0x3C003C00u); // f16 1.0 x2
        // O = O * alpha + P V: per dim tile one ldmatrix.x4.trans gives the B fragments of all 4 key steps
        #pragma unroll
        for (int d = 0; d < HD / 8; d++) {
            o[d][0] *= a_lo; o[d][1] *= a_lo; o[d][2] *= a_hi; o[d][3] *= a_hi;
            unsigned b[4];
            LDSM_X4(b, vaddr + 16 * d, ".trans");
            #pragma unroll
            for (int j = 0; j < BK / 8; j++) MMA_F16(o[d], pf[j][0], pf[j][1], b[j]);
        }
    }
    int r_lo = q0 + 16 * warp + (lane >> 2), r_hi = r_lo + 8;
    float i_lo = 1.f / l[0], i_hi = 1.f / l[2];
    #pragma unroll
    for (int d = 0; d < HD / 8; d++) {
        int col = h * HD + 8 * d + 2 * (lane & 3);
        if (r_lo < n) *reinterpret_cast<float2*>(out + (size_t)r_lo * q_dim + col) = make_float2(o[d][0] * i_lo, o[d][1] * i_lo);
        if (r_hi < n) *reinterpret_cast<float2*>(out + (size_t)r_hi * q_dim + col) = make_float2(o[d][2] * i_hi, o[d][3] * i_hi);
    }
}

extern "C" {

#define ATT_TC_ARGS const float* __restrict__ qkv, const float* __restrict__ kc, const float* __restrict__ vc, \
                    float* __restrict__ out, int pos0, int n, int n_heads, int n_kv, int hd, float scale
__global__ void __launch_bounds__(128) attention_prefill_tc64(ATT_TC_ARGS) { attn_prefill_tc<64>(qkv, kc, vc, out, pos0, n, n_heads, n_kv, scale); }
__global__ void __launch_bounds__(128) attention_prefill_tc128(ATT_TC_ARGS) { attn_prefill_tc<128>(qkv, kc, vc, out, pos0, n, n_heads, n_kv, scale); }

// Flash-decoding attention, part 2: merge the slices of each (head, token) into the normalised output.
__global__ void attention_combine(const float* __restrict__ part, float* __restrict__ out,
                                  int pos0, const int* __restrict__ posp, int n_heads, int hd, int nsplit, int want) {
    if (posp) pos0 = *posp;
    int h = blockIdx.x, b = blockIdx.y;
    int span = pos0 + b + 1;
    int chunk = att_chunk_for(span, want);
    int used = min(nsplit, (span + chunk - 1) / chunk);
    const float* p = part + ((size_t)b * n_heads + h) * nsplit * (hd + 2);
    float m = NEG_INF;
    for (int s = 0; s < used; s++) m = fmaxf(m, p[s * (hd + 2) + hd]);
    float total = 0.f;
    for (int s = 0; s < used; s++) total += p[s * (hd + 2) + hd + 1] * expf(p[s * (hd + 2) + hd] - m);
    float inv = 1.f / total;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < used; s++) acc += p[s * (hd + 2) + d] * expf(p[s * (hd + 2) + hd] - m);
        out[(size_t)b * n_heads * hd + h * hd + d] = acc * inv;
    }
}

// Activations to int8 in blocks of 32 (one warp per block): q = round(x / d), d = max|x| / 127,
// plus each block's sum of q (for the K-quant "min" terms).
__global__ void quantize_q8(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ d,
                            int* __restrict__ sums, int n_blocks) {
    int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (blk >= n_blocks) return;
    float v = x[(size_t)blk * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    int qi = (int)roundf(v * inv);
    q[(size_t)blk * 32 + lane] = (signed char)qi;
    for (int o = 16; o > 0; o >>= 1) qi += __shfl_xor_sync(0xffffffff, qi, o);
    if (lane == 0) { d[blk] = dd; sums[blk] = qi; }
}

// Batched int8 matmul for prompts: out[t*out_stride + out_off + r] (+)= sum_k W[r][k] * X[t][k].
// Activations are int8 + scale per 32. Each weight 32-block becomes int8 values plus a scale for each
// 16-value half and a "min": Q8_0 (one scale, no min), Q4_K (values 0..15, scale and min),
// Q6_K (values -32..31, two scales). 64x64 output tile per 256-thread block; each thread owns
// 4 rows x 4 tokens; __dp4a does 4 int8 multiply-adds.
#define TILE 64
__global__ void matmul_tiled(int fmt, const unsigned char* __restrict__ wq, const float* __restrict__ wd,
                             const signed char* __restrict__ xq, const float* __restrict__ xd, const int* __restrict__ xsum,
                             float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    // one step covers up to 8 blocks of 32 (a whole K-quant super-block): fewer barriers and scale decodes
    __shared__ int ws[TILE][65]; // 64 ints = 256 int8 per row, +1 to spread rows over banks
    __shared__ int xs[TILE][65];
    __shared__ float wsa[TILE][8], wsb[TILE][8], wmn[TILE][8], xds[TILE][8], xss[TILE][8];
    int tid = threadIdx.x;
    int tx = tid & 15, ty = tid >> 4; // tx picks rows (coalesced stores), ty picks tokens
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int nb = cols >> 5, nsb = cols >> 8;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int kc = 0; kc < nb; kc += 8) {
        int nk = min(8, nb - kc);
        for (int idx = tid; idx < TILE * 64; idx += 256) {
            int r = idx >> 6, wi = idx & 63, k8 = wi >> 3, w = wi & 7;
            int gr = row0 + r, gt = tok0 + r, kb = kc + k8;
            int v = 0, xv = 0;
            if (k8 < nk) {
                if (gr < rows) {
                    if (fmt == 0) {
                        v = *reinterpret_cast<const int*>(wq + (size_t)gr * cols + kb * 32 + w * 4);
                    } else if (fmt == 1) {
                        const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                        int p = *reinterpret_cast<const int*>(b + 16 + 32 * (k8 >> 1) + 4 * w);
                        v = (k8 & 1) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                    } else {
                        const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                        int half = k8 >> 2, g = k8 & 3;
                        int p = *reinterpret_cast<const int*>(b + 64 * half + (g & 1) * 32 + 4 * w);
                        int lo = (g >= 2) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                        int h = *reinterpret_cast<const int*>(b + 128 + 32 * half + 4 * w);
                        v = __vsub4(lo | (((h >> (2 * g)) & 0x03030303) << 4), 0x20202020); // bytewise -32
                    }
                }
                if (gt < n) xv = *reinterpret_cast<const int*>(xq + (size_t)gt * cols + kb * 32 + w * 4);
            }
            ws[r][wi] = v;
            xs[r][wi] = xv;
        }
        // per-32 scales: 64 rows x 8 blocks for the weights, the same for the activations
        for (int idx = tid; idx < TILE * 8; idx += 256) {
            int r = idx >> 3, k8 = idx & 7, kb = kc + k8;
            int gr = row0 + r, gt = tok0 + r;
            float sa = 0.f, sb = 0.f, mn = 0.f;
            if (k8 < nk && gr < rows) {
                if (fmt == 0) {
                    sa = sb = wd[(size_t)gr * nb + kb];
                } else if (fmt == 1) {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                    int sc, m;
                    q4k_scale_min(k8, b + 4, &sc, &m);
                    float d, dmin;
                    q4k_dm(b, &d, &dmin);
                    sa = sb = d * sc;
                    mn = dmin * m;
                } else {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                    float d = wd[(size_t)gr * nsb + (kb >> 3)];
                    const signed char* sc = (const signed char*)(b + 192 + 8 * (k8 >> 2) + 2 * (k8 & 3));
                    sa = d * sc[0];
                    sb = d * sc[1];
                }
            }
            wsa[r][k8] = sa; wsb[r][k8] = sb; wmn[r][k8] = mn;
            bool tok_ok = k8 < nk && gt < n;
            xds[r][k8] = tok_ok ? xd[(size_t)gt * nb + kb] : 0.f;
            xss[r][k8] = tok_ok ? (float)xsum[(size_t)gt * nb + kb] : 0.f;
        }
        __syncthreads();
        for (int k8 = 0; k8 < nk; k8++) {
            int wr[4][8], xr[4][8];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int w = 0; w < 8; w++) { wr[i][w] = ws[tx + 16 * i][k8 * 8 + w]; xr[i][w] = xs[ty + 16 * i][k8 * 8 + w]; }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = tx + 16 * i;
                float sa = wsa[r][k8], sb = wsb[r][k8], mn = wmn[r][k8];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int lo = 0, hi = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; w++) lo = __dp4a(wr[i][w], xr[j][w], lo);
                    #pragma unroll
                    for (int w = 4; w < 8; w++) hi = __dp4a(wr[i][w], xr[j][w], hi);
                    int t = ty + 16 * j;
                    acc[i][j] += xds[t][k8] * (sa * (float)lo + sb * (float)hi - mn * xss[t][k8]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + tx + 16 * i;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int t = tok0 + ty + 16 * j;
            if (r < rows && t < n) {
                size_t idx = (size_t)t * out_stride + out_off + r;
                out[idx] = accumulate ? out[idx] + acc[i][j] : acc[i][j];
            }
        }
    }
}

// Tensor-core version of matmul_tiled (sm_75+): int8 mma.sync.m8n8k16, whose register layout is
// documented, so every thread knows the (row, token) of its two results and can apply the per-block
// weight and activation scales. One k16 step is exactly one Q6_K sub-block (half a 32-block).
// 128 threads = 2x2 warps over a 64x64 output tile; each warp owns 32 rows x 32 tokens (4x4 mma tiles).
// K advances 128 values (4 blocks of 32) per shared-memory stage. Rows are padded to 36 words so the
// fragment loads (8 rows x 4 words per warp) hit 32 different banks.
#define MMA_KB 4
#define MMA_LD 36
__global__ void matmul_mma(int fmt, const unsigned char* __restrict__ wq, const float* __restrict__ wd,
                           const signed char* __restrict__ xq, const float* __restrict__ xd, const int* __restrict__ xsum,
                           float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    __shared__ int ws[TILE * MMA_LD];
    __shared__ int xs[TILE * MMA_LD];
    __shared__ float wsa[TILE][MMA_KB], wsb[TILE][MMA_KB], wmn[TILE][MMA_KB], xds[TILE][MMA_KB], xms[TILE][MMA_KB];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int wr = (warp >> 1) * 32, wt = (warp & 1) * 32; // this warp's rows / tokens inside the tile
    int nb = cols >> 5, nsb = cols >> 8;
    int fr = lane >> 2, fk = lane & 3; // fragment row (or token) and word within the k16 step
    float acc[4][4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; }

    // Software pipeline: each thread fetches the next stage into registers (its global loads stay in
    // flight) while the warps run tensor-core math on the current stage in shared memory, then writes
    // it to shared memory. Per stage and thread (128 threads): 16 weight words, 16 activation words,
    // 2 (row, block) scale entries.
    int wreg[16], xreg[16];
    float sreg[2][5];
    auto fetch = [&](int kc) {
        int nk = min(MMA_KB, nb - kc);
        if (fmt == 1) {
            // Q4_K: one word holds the low nibbles of block 2p and the high nibbles of block 2p+1
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                int idx = tid + 128 * s;
                int r = idx >> 4, pi = (idx >> 3) & 1, w = idx & 7;
                int gr = row0 + r, kk = 2 * pi, kb = kc + kk;
                int p = 0;
                if (kk < nk && gr < rows) {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                    p = *reinterpret_cast<const int*>(b + 16 + 32 * ((kb & 7) >> 1) + 4 * w);
                }
                wreg[2 * s] = p & 0x0F0F0F0F;
                wreg[2 * s + 1] = (kk + 1 < nk) ? ((p >> 4) & 0x0F0F0F0F) : 0;
            }
        } else {
            #pragma unroll
            for (int s = 0; s < 16; s++) {
                int idx = tid + 128 * s;
                int r = idx >> 5, wi = idx & 31, kk = wi >> 3, w = wi & 7;
                int gr = row0 + r, kb = kc + kk, k8 = kb & 7;
                int v = 0;
                if (kk < nk && gr < rows) {
                    if (fmt == 0) {
                        v = *reinterpret_cast<const int*>(wq + (size_t)gr * cols + kb * 32 + w * 4);
                    } else {
                        const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                        int half = k8 >> 2, g = k8 & 3;
                        int p = *reinterpret_cast<const int*>(b + 64 * half + (g & 1) * 32 + 4 * w);
                        int lo = (g >= 2) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                        int h = *reinterpret_cast<const int*>(b + 128 + 32 * half + 4 * w);
                        v = __vsub4(lo | (((h >> (2 * g)) & 0x03030303) << 4), 0x20202020);
                    }
                }
                wreg[s] = v;
            }
        }
        #pragma unroll
        for (int s = 0; s < 16; s++) {
            int idx = tid + 128 * s;
            int r = idx >> 5, wi = idx & 31, kk = wi >> 3, w = wi & 7;
            int gt = tok0 + r;
            xreg[s] = (kk < nk && gt < n) ? *reinterpret_cast<const int*>(xq + (size_t)gt * cols + (kc + kk) * 32 + w * 4) : 0;
        }
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int idx = tid + 128 * s;
            int r = idx / MMA_KB, kk = idx % MMA_KB, kb = kc + kk, k8 = kb & 7;
            int gr = row0 + r, gt = tok0 + r;
            float sa = 0.f, sb = 0.f, mn = 0.f;
            if (kk < nk && gr < rows) {
                if (fmt == 0) {
                    sa = sb = wd[(size_t)gr * nb + kb];
                } else if (fmt == 1) {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                    int sc, m;
                    q4k_scale_min(k8, b + 4, &sc, &m);
                    float d, dmin;
                    q4k_dm(b, &d, &dmin);
                    sa = sb = d * sc;
                    mn = dmin * m;
                } else {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                    float d = wd[(size_t)gr * nsb + (kb >> 3)];
                    const signed char* sc = (const signed char*)(b + 192 + 8 * (k8 >> 2) + 2 * (k8 & 3));
                    sa = d * sc[0];
                    sb = d * sc[1];
                }
            }
            bool ok = kk < nk && gt < n;
            float dx = ok ? xd[(size_t)gt * nb + kb] : 0.f;
            sreg[s][0] = sa; sreg[s][1] = sb; sreg[s][2] = mn;
            sreg[s][3] = dx;
            sreg[s][4] = ok ? dx * (float)xsum[(size_t)gt * nb + kb] : 0.f;
        }
    };
    auto commit = [&]() {
        if (fmt == 1) {
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                int idx = tid + 128 * s;
                int r = idx >> 4, kk = 2 * ((idx >> 3) & 1), w = idx & 7;
                ws[r * MMA_LD + kk * 8 + w] = wreg[2 * s];
                ws[r * MMA_LD + (kk + 1) * 8 + w] = wreg[2 * s + 1];
            }
        } else {
            #pragma unroll
            for (int s = 0; s < 16; s++) {
                int idx = tid + 128 * s;
                ws[(idx >> 5) * MMA_LD + (idx & 31)] = wreg[s];
            }
        }
        #pragma unroll
        for (int s = 0; s < 16; s++) {
            int idx = tid + 128 * s;
            xs[(idx >> 5) * MMA_LD + (idx & 31)] = xreg[s];
        }
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int idx = tid + 128 * s;
            int r = idx / MMA_KB, kk = idx % MMA_KB;
            wsa[r][kk] = sreg[s][0]; wsb[r][kk] = sreg[s][1]; wmn[r][kk] = sreg[s][2];
            xds[r][kk] = sreg[s][3]; xms[r][kk] = sreg[s][4];
        }
    };

    fetch(0);
    commit();
    __syncthreads();
    for (int kc = 0; kc < nb; kc += MMA_KB) {
        int nk = min(MMA_KB, nb - kc);
        bool more = kc + MMA_KB < nb;
        if (more) fetch(kc + MMA_KB); // issued now, consumed after the math below

        for (int kk = 0; kk < nk; kk++) {
            // this thread's scales for the block, in registers: 4 rows, 8 tokens
            float ra[4], rb[4], ts[4][2];
            #pragma unroll
            for (int i = 0; i < 4; i++) { ra[i] = wsa[wr + i * 8 + fr][kk]; rb[i] = wsb[wr + i * 8 + fr][kk]; }
            #pragma unroll
            for (int j = 0; j < 4; j++) { ts[j][0] = xds[wt + j * 8 + fk * 2][kk]; ts[j][1] = xds[wt + j * 8 + fk * 2 + 1][kk]; }
            int c[4][4][2];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) { c[i][j][0] = 0; c[i][j][1] = 0; }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int kw = kk * 8 + h * 4 + fk;
                int a[4], bq[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) a[i] = ws[(wr + i * 8 + fr) * MMA_LD + kw];
                #pragma unroll
                for (int j = 0; j < 4; j++) bq[j] = xs[(wt + j * 8 + fr) * MMA_LD + kw];
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                            : "+r"(c[i][j][0]), "+r"(c[i][j][1]) : "r"(a[i]), "r"(bq[j]));
                // Q6_K has a scale per 16 values (one k16 step): fold in each half separately.
                // Q8_0 / Q4_K have one scale per 32: integers keep accumulating, folded in once below.
                if (fmt == 2) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float s = h ? rb[i] : ra[i];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            acc[i][j][0] += (float)c[i][j][0] * (s * ts[j][0]);
                            acc[i][j][1] += (float)c[i][j][1] * (s * ts[j][1]);
                            c[i][j][0] = 0; c[i][j][1] = 0;
                        }
                    }
                }
            }
            if (fmt != 2) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float mn = wmn[wr + i * 8 + fr][kk]; // Q4_K mins (0 for Q8_0)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int t = wt + j * 8 + fk * 2;
                        acc[i][j][0] += (float)c[i][j][0] * (ra[i] * ts[j][0]) - mn * xms[t][kk];
                        acc[i][j][1] += (float)c[i][j][1] * (ra[i] * ts[j][1]) - mn * xms[t + 1][kk];
                    }
                }
            }
        }
        __syncthreads(); // everyone done reading this stage
        if (more) {
            commit();
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + wr + i * 8 + fr;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int t = tok0 + wt + j * 8 + fk * 2 + e;
                if (r < rows && t < n) {
                    size_t idx = (size_t)t * out_stride + out_off + r;
                    out[idx] = accumulate ? out[idx] + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
    }
}

// Activations for the K-quant prompt matmuls: int8 with one scale per QCH values (128 or 256, set by the
// compile options; one warp per chunk), plus each 32-block's sum split into a high part (sum >> 6) and the
// low 6 bits, so the Q4_K min term fits int8 tensor cores. Per chunk, QCH/64 sum words: the QCH/32 high
// bytes, then the QCH/32 low bytes.
#define QVPL (QCH / 32) // values per lane
__global__ void quantize_q8k(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ d,
                             int* __restrict__ sums, int n_chunks) {
    int c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (c >= n_chunks) return;
    const float4* src = reinterpret_cast<const float4*>(x + (size_t)c * QCH + lane * QVPL);
    float v[QVPL];
    #pragma unroll
    for (int i = 0; i < QVPL / 4; i++) { float4 f = src[i]; v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w; }
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < QVPL; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    int s = 0, packed[QVPL / 4];
    #pragma unroll
    for (int i = 0; i < QVPL / 4; i++) {
        int q0 = (int)roundf(v[4 * i] * inv), q1 = (int)roundf(v[4 * i + 1] * inv);
        int q2 = (int)roundf(v[4 * i + 2] * inv), q3 = (int)roundf(v[4 * i + 3] * inv);
        packed[i] = (q0 & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((unsigned)q3 << 24);
        s += q0 + q1 + q2 + q3;
    }
    int* dst = reinterpret_cast<int*>(q + (size_t)c * QCH + lane * QVPL);
    #pragma unroll
    for (int i = 0; i < QVPL / 4; i++) dst[i] = packed[i];
    for (int o = 1; o < 32 / QVPL; o <<= 1) s += __shfl_xor_sync(0xffffffff, s, o); // lanes of one 32-block
    if (lane == 0) d[c] = dd;
    if ((lane & (32 / QVPL - 1)) == 0) {
        signed char* bs = reinterpret_cast<signed char*>(sums + (size_t)c * (QCH / 64));
        int j = lane / (32 / QVPL);
        bs[j] = (signed char)(s >> 6);
        bs[QCH / 32 + j] = (signed char)(s & 63);
    }
}

}  // extern "C": the template below needs C++ linkage

// K-quant prompt matmul with an integer epilogue (activations from quantize_q8k). Within one activation
// scale (QCH values) everything stays integer: the 6-bit (Q4_K) or int8 (Q6_K) sub-block scales multiply
// the tensor-core sums with IMAD, the Q4_K mins go through one extra pair of mma (mins x the split
// activation sums), and floats are touched once per QCH values instead of once per 32 or 16.
// Same tiling as matmul_mma: 64x64 tile, 2x2 warps of 32x32, 128 K-values per shared-memory stage, so
// two stages per super-block. Per tile and 32-block the two k16 mma run back to back into a fresh pair
// of registers, which keeps the per-block integer sums out of the register file.
#define XBW (QCH / 64) // split-sum words per token and activation chunk
template <int FMT>
__device__ void mma_kquant(const unsigned char* __restrict__ wq, const float* __restrict__ wd, const signed char* __restrict__ xq,
                           const float* __restrict__ xd, const int* __restrict__ xb, float* __restrict__ out,
                           int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    __shared__ __align__(16) int ws[TILE * MMA_LD];
    __shared__ __align__(16) int xs[TILE * MMA_LD];
    __shared__ float wdm[TILE][2]; // d, and dmin for Q4_K
    __shared__ int wsc[TILE][2];   // this stage's sub-block scales as packed bytes (Q4_K: 4, Q6_K: 8)
    __shared__ int wmn[TILE][2];   // Q4_K: the super-block's 8 mins as packed bytes
    __shared__ float xds[TILE];    // activation scale of each token for the current QCH chunk
    __shared__ int xbs[TILE][4];   // its split 32-block sums (XBW words used)
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int wr = (warp >> 1) * 32, wt = (warp & 1) * 32;
    int nsb = cols >> 8, nst = 2 * nsb;
    int fr = lane >> 2, fk = lane & 3;
    const int bytes = FMT == 1 ? Q4K_BYTES : Q6K_BYTES;
    float acc[4][4][2];
    int iacc[4][4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { acc[i][j][0] = acc[i][j][1] = 0.f; iacc[i][j][0] = iacc[i][j][1] = 0; }

    // Register-staged pipeline as in matmul_mma. fetch() only issues loads into raw words; all the
    // unpacking happens in commit(), after the math, so no thread waits on its loads early.
    // Quant bytes and activations move in 16-byte pieces (128-bit loads and shared stores). Each group
    // of 8 lanes reads one cache line (activations: 8 pieces of one token) or one piece of 8 rows
    // (weights), and the 8 lanes of a shared store hit 32 different banks. Rows and tokens past the end
    // are clamped instead of tested; they only feed results that are never stored.
    int4 wraw[FMT == 1 ? 2 : 3], xreg[4];
    const signed char* xp[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) xp[s] = xq + (size_t)min(tok0 + (tid >> 3) + 16 * s, n - 1) * cols + 16 * (tid & 7);
    // weights: Q4_K piece q (of 4 per row and stage) of rows wl + 32*s; Q6_K piece half wh of row wl
    int wl = FMT == 1 ? (tid & 7) + 8 * (tid >> 5) : (tid & 7) + 8 * (tid >> 4);
    int wqp = FMT == 1 ? (tid >> 3) & 3 : (tid >> 3) & 1;
    const unsigned char* wp[FMT == 1 ? 2 : 1];
    #pragma unroll
    for (int s = 0; s < (FMT == 1 ? 2 : 1); s++) wp[s] = wq + (size_t)min(row0 + wl + 32 * s, rows - 1) * nsb * bytes;
    // metadata: two threads per row and per token
    int lr = tid >> 1, hh = tid & 1;
    int grc = min(row0 + lr, rows - 1), gtc = min(tok0 + lr, n - 1);
    const unsigned char* mrow = wq + (size_t)grc * nsb * bytes;
    const float* wdrow = wd + (size_t)grc * nsb; // Q6_K
    float xf;
    unsigned int wm, ma0, ma1; // wm: Q4_K d|dmin as two f16, Q6_K the f32 d's bits
    int xi0, xi1;
    // Two halves, issued at different points of the math so the load queue doesn't overflow:
    // the weights and their metadata (half 0), the activations and theirs (half 1).
    auto fetch = [&](int st, int half) {
        int sb = st >> 1, hs = st & 1;
        if (half == 1) {
            #pragma unroll
            for (int s = 0; s < 4; s++) xreg[s] = *reinterpret_cast<const int4*>(xp[s] + st * 128);
            size_t c = (size_t)gtc * (cols / QCH) + st * 128 / QCH;
            xf = xd[c];
            xi0 = xb[c * XBW + hh * (XBW / 2)];
            xi1 = XBW == 4 ? xb[c * XBW + hh * (XBW / 2) + 1] : 0;
            return;
        }
        if (FMT == 1) {
            // nibble bytes 64*hs .. +63 = [blocks 0 (low) / 1 (high): 32 bytes] [blocks 2 / 3: 32 bytes]
            #pragma unroll
            for (int s = 0; s < 2; s++) wraw[s] = *reinterpret_cast<const int4*>(wp[s] + (size_t)sb * bytes + 16 + 64 * hs + 16 * wqp);
        } else {
            // half wqp (16 bytes) of the low-nibble words of blocks 0/2, of blocks 1/3, and of the 2-bit highs
            const unsigned char* b = wp[0] + (size_t)sb * bytes;
            wraw[0] = *reinterpret_cast<const int4*>(b + 64 * hs + 16 * wqp);
            wraw[FMT - 1] = *reinterpret_cast<const int4*>(b + 64 * hs + 32 + 16 * wqp);
            wraw[2 * (FMT - 1)] = *reinterpret_cast<const int4*>(b + 128 + 32 * hs + 16 * wqp);
        }
        const unsigned char* b = mrow + (size_t)sb * bytes;
        if (FMT == 1) {
            wm = *reinterpret_cast<const unsigned int*>(b);                     // d, dmin
            ma0 = *reinterpret_cast<const unsigned int*>(b + 4 + 4 * hh);       // scales / mins of sub-blocks 0..3
            ma1 = *reinterpret_cast<const unsigned int*>(b + 12);               // low 4 bits of both for 4..7
        } else {
            wm = __float_as_uint(wdrow[sb]);
            ma0 = *reinterpret_cast<const unsigned int*>(b + 192 + 8 * hs + 4 * hh);
        }
    };
    auto commit = [&](int st) {
        int hs = st & 1;
        float wf = FMT == 1 ? h2f(hh ? wm >> 16 : wm & 0xFFFF) : __uint_as_float(wm);
        if (FMT == 1) {
            // piece q = words 4*(q&1) .. +3 of the 32 bytes of blocks 2*(q>>1) (low) and 2*(q>>1)+1 (high)
            int kb = 2 * (wqp >> 1), h = wqp & 1;
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                int4 p = wraw[s];
                int* dst = ws + (wl + 32 * s) * MMA_LD + kb * 8 + h * 4;
                *reinterpret_cast<int4*>(dst) = make_int4(p.x & 0x0F0F0F0F, p.y & 0x0F0F0F0F, p.z & 0x0F0F0F0F, p.w & 0x0F0F0F0F);
                *reinterpret_cast<int4*>(dst + 8) = make_int4((p.x >> 4) & 0x0F0F0F0F, (p.y >> 4) & 0x0F0F0F0F, (p.z >> 4) & 0x0F0F0F0F, (p.w >> 4) & 0x0F0F0F0F);
            }
        } else {
            int l0[4] = {wraw[0].x, wraw[0].y, wraw[0].z, wraw[0].w};
            int l1[4] = {wraw[FMT - 1].x, wraw[FMT - 1].y, wraw[FMT - 1].z, wraw[FMT - 1].w};
            int hb[4] = {wraw[2 * (FMT - 1)].x, wraw[2 * (FMT - 1)].y, wraw[2 * (FMT - 1)].z, wraw[2 * (FMT - 1)].w};
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int v[4];
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    int p = (g & 1) ? l1[f] : l0[f];
                    int lo = (g >= 2) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                    v[f] = __vsub4(lo | (((hb[f] >> (2 * g)) & 0x03030303) << 4), 0x20202020); // bytewise -32
                }
                *reinterpret_cast<int4*>(ws + wl * MMA_LD + g * 8 + wqp * 4) = make_int4(v[0], v[1], v[2], v[3]);
            }
        }
        #pragma unroll
        for (int s = 0; s < 4; s++) *reinterpret_cast<int4*>(xs + ((tid >> 3) + 16 * s) * MMA_LD + 4 * (tid & 7)) = xreg[s];
        if (FMT == 1) {
            // 6-bit fields: sub-blocks 0..3 in the low 6 bits of ma0's bytes; 4..7 from ma1's nibbles + ma0's top 2 bits
            unsigned int hi = ((ma1 >> (4 * hh)) & 0x0F0F0F0F) | ((ma0 >> 2) & 0x30303030);
            wdm[lr][hh] = wf;
            if (hh == 0) wsc[lr][0] = (int)(hs ? hi : ma0 & 0x3F3F3F3F);
            else { wmn[lr][0] = (int)(ma0 & 0x3F3F3F3F); wmn[lr][1] = (int)hi; }
        } else {
            if (hh == 0) wdm[lr][0] = wf;
            wsc[lr][hh] = (int)ma0;
        }
        if (hh == 0) xds[lr] = xf;
        xbs[lr][hh * (XBW / 2)] = xi0;
        if (XBW == 4) xbs[lr][2 * hh + 1] = xi1;
    };

    // shared-memory byte addresses of this lane's ldmatrix rows (rows 36 words apart: conflict-free)
    unsigned lda = (unsigned)__cvta_generic_to_shared(ws + (wr + (lane >> 3) * 8 + (lane & 7)) * MMA_LD);
    unsigned ldb = (unsigned)__cvta_generic_to_shared(xs + (wt + (lane >> 3) * 8 + (lane & 7)) * MMA_LD);

    // ldmatrix.x4: four 8x8 b16 matrices (= 8 rows x 16 int8) per instruction, already in the m8n8k16
    // fragment layout; lane l gives the address of row l&7 of matrix l>>3. Block kb's fragments are
    // loaded while block kb-1 is still in the tensor cores.
    int fa[2][4][2], fb[2][4][2];
    auto frags = [&](int kb, int buf) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(fa[buf][0][h]), "=r"(fa[buf][1][h]), "=r"(fa[buf][2][h]), "=r"(fa[buf][3][h]) : "r"(lda + (kb * 8 + h * 4) * 4));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(fb[buf][0][h]), "=r"(fb[buf][1][h]), "=r"(fb[buf][2][h]), "=r"(fb[buf][3][h]) : "r"(ldb + (kb * 8 + h * 4) * 4));
        }
    };

    fetch(0, 0);
    fetch(0, 1);
    commit(0);
    __syncthreads();
    for (int st = 0; st < nst; st++) {
        bool more = st + 1 < nst;
        frags(0, 0);
        if (more) fetch(st + 1, 0); // issued now, consumed after the math below

        int scw[4][FMT == 1 ? 1 : 2];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            scw[i][0] = wsc[wr + i * 8 + fr][0];
            if (FMT == 2) scw[i][FMT - 1] = wsc[wr + i * 8 + fr][1];
        }
        #pragma unroll
        for (int kb = 0; kb < 4; kb++) {
            if (kb < 3) frags(kb + 1, (kb + 1) & 1);
            if (kb == 1 && more) fetch(st + 1, 1);
            int (&a)[4][2] = fa[kb & 1];
            int (&bq)[4][2] = fb[kb & 1];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (FMT == 1) {
                        // Q4_K: one scale per 32 values
                        int sc = (scw[i][0] >> (8 * kb)) & 0xFF;
                        int c0 = 0, c1 = 0;
                        #pragma unroll
                        for (int h = 0; h < 2; h++)
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(c0), "+r"(c1) : "r"(a[i][h]), "r"(bq[j][h]));
                        iacc[i][j][0] += sc * c0;
                        iacc[i][j][1] += sc * c1;
                    } else {
                        // Q6_K: one signed scale per 16 values, i.e. per k16 step
                        #pragma unroll
                        for (int h = 0; h < 2; h++) {
                            int sc = (int)(signed char)(scw[i][(FMT - 1) * (kb >> 1)] >> (8 * ((2 * kb + h) & 3)));
                            int c0 = 0, c1 = 0;
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(c0), "+r"(c1) : "r"(a[i][h]), "r"(bq[j][h]));
                            iacc[i][j][0] += sc * c0;
                            iacc[i][j][1] += sc * c1;
                        }
                    }
                }
            }
        }
        if ((st + 1) % (QCH / 128) == 0) {
            // activation chunk done: one float fold per result, plus the Q4_K mins of its sub-blocks
            const int nf = QCH / 128; // fragment words holding those mins / sums (4 per word)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = wr + i * 8 + fr;
                float d = wdm[r][0], dm = FMT == 1 ? wdm[r][1] : 0.f;
                int ma = (FMT == 1 && fk < nf) ? wmn[r][(nf == 1 ? (st & 1) : 0) + fk] : 0; // A fragment: mins, zeros above
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int t = wt + j * 8 + fk * 2;
                    float D0 = xds[t], D1 = xds[t + 1];
                    int m0 = 0, m1 = 0;
                    if (FMT == 1) {
                        int tb = wt + j * 8 + fr; // B fragment: column = token
                        int bh = fk < nf ? xbs[tb][fk] : 0, bl = fk < nf ? xbs[tb][nf + fk] : 0;
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                            : "+r"(m0), "+r"(m1) : "r"(ma), "r"(bh));
                        m0 *= 64; m1 *= 64;
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                            : "+r"(m0), "+r"(m1) : "r"(ma), "r"(bl));
                    }
                    acc[i][j][0] += D0 * (d * (float)iacc[i][j][0] - dm * (float)m0);
                    acc[i][j][1] += D1 * (d * (float)iacc[i][j][1] - dm * (float)m1);
                    iacc[i][j][0] = iacc[i][j][1] = 0;
                }
            }
        }
        __syncthreads(); // everyone done reading this stage
        if (more) {
            commit(st + 1);
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + wr + i * 8 + fr;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int t = tok0 + wt + j * 8 + fk * 2 + e;
                if (r < rows && t < n) {
                    size_t idx = (size_t)t * out_stride + out_off + r;
                    out[idx] = accumulate ? out[idx] + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
    }
}

extern "C" {

#define MMA_K_ARGS const unsigned char* __restrict__ wq, const float* __restrict__ wd, const signed char* __restrict__ xq, const float* __restrict__ xd, \
                   const int* __restrict__ xb, float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off
__global__ void __launch_bounds__(128) matmul_mma_q4k(MMA_K_ARGS) { mma_kquant<1>(wq, wd, xq, xd, xb, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void __launch_bounds__(128) matmul_mma_q6k(MMA_K_ARGS) { mma_kquant<2>(wq, wd, xq, xd, xb, out, rows, cols, n, accumulate, out_stride, out_off); }

// Decode: h = silu(gate) * up straight to int8 activations for the down projection (ffn a multiple of 32),
// one token per grid row.
__global__ void silu_q8(const float* __restrict__ gu, signed char* __restrict__ xq, float* __restrict__ xd,
                        int* __restrict__ xsum, int ffn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, nv = ffn >> 2;
    size_t r = blockIdx.y;
    gu += r * 2 * ffn;
    xq += r * ffn;
    xd += r * (ffn >> 5);
    xsum += r * (ffn >> 5);
    bool ok = i < nv;
    float4 h = make_float4(0.f, 0.f, 0.f, 0.f);
    if (ok) {
        float4 g = reinterpret_cast<const float4*>(gu)[i], u = reinterpret_cast<const float4*>(gu + ffn)[i];
        h = make_float4(g.x / (1.f + expf(-g.x)) * u.x, g.y / (1.f + expf(-g.y)) * u.y,
                        g.z / (1.f + expf(-g.z)) * u.z, g.w / (1.f + expf(-g.w)) * u.w);
    }
    quant8x4(h, ok, i, xq, xd, xsum);
}

// Index of each row's largest value, as the greedy sampler picks it: the first maximum, NaN never wins,
// 0 when nothing beats -inf. One block per row.
__global__ void argmax_rows(const float* __restrict__ x, int cols, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* r = x + (size_t)blockIdx.x * cols;
    float bv = NEG_INF;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = r[i];
        if (v > bv) { bv = v; bi = i; }
    }
    // better: larger value, or the same value at a lower index
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : NEG_INF;
        bi = lane < nw ? si[lane] : 0x7FFFFFFF;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = bi == 0x7FFFFFFF ? 0 : bi;
    }
}

// gu rows are [gate | up]; h = silu(gate) * up
__global__ void silu_mul(const float* __restrict__ gu, float* __restrict__ h, int ffn, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * ffn) return;
    size_t b = i / ffn, j = i % ffn;
    float g = gu[b * 2 * ffn + j], u = gu[b * 2 * ffn + ffn + j];
    h[i] = g / (1.f + expf(-g)) * u;
}

}
"#;

// ---------------------------------------------------------------- dynamic loading

type CuResult = i32;
type Ptr = *mut c_void;

#[cfg(windows)]
mod os {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn SetDllDirectoryW(dir: *const u16) -> i32;
    }
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;

    pub fn open(path: &str) -> Option<*mut c_void> {
        // nvrtc loads nvrtc-builtins*.dll lazily at compile time; make its folder searchable
        if let Some(dir) = std::path::Path::new(path).parent() {
            let wide: Vec<u16> = dir.to_string_lossy().encode_utf16().chain(Some(0)).collect();
            unsafe { SetDllDirectoryW(wide.as_ptr()) };
        }
        let wide: Vec<u16> = path.encode_utf16().chain(Some(0)).collect();
        let h = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        (!h.is_null()).then_some(h)
    }

    pub fn symbol(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        let p = unsafe { GetProcAddress(lib, c.as_ptr() as *const u8) };
        (!p.is_null()).then_some(p)
    }
}

#[cfg(not(windows))]
mod os {
    use std::ffi::c_void;
    pub fn open(_: &str) -> Option<*mut c_void> {
        None
    }
    pub fn symbol(_: *mut c_void, _: &str) -> Option<*mut c_void> {
        None
    }
}

macro_rules! api {
    ($($name:ident : fn($($arg:ty),*) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        struct Fns { $($name: unsafe extern "system" fn($($arg),*) -> $ret,)* }
        impl Fns {
            fn load(cuda: Ptr, nvrtc: Ptr) -> Result<Fns, String> {
                Ok(Fns { $($name: {
                    let lib = if stringify!($name).starts_with("nvrtc") { nvrtc } else { cuda };
                    let p = os::symbol(lib, stringify!($name)).ok_or_else(|| format!("missing CUDA symbol {}", stringify!($name)))?;
                    unsafe { std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($arg),*) -> $ret>(p) }
                },)* })
            }
        }
    };
}

api! {
    cuInit: fn(u32) -> CuResult;
    cuDeviceGet: fn(*mut i32, i32) -> CuResult;
    cuDeviceGetAttribute: fn(*mut i32, i32, i32) -> CuResult;
    cuDeviceGetName: fn(*mut c_char, i32, i32) -> CuResult;
    cuCtxCreate_v2: fn(*mut Ptr, u32, i32) -> CuResult;
    cuModuleLoadData: fn(*mut Ptr, *const c_void) -> CuResult;
    cuModuleGetFunction: fn(*mut Ptr, Ptr, *const c_char) -> CuResult;
    cuMemAlloc_v2: fn(*mut u64, usize) -> CuResult;
    cuMemFree_v2: fn(u64) -> CuResult;
    cuMemcpyHtoD_v2: fn(u64, *const c_void, usize) -> CuResult;
    cuMemcpyDtoH_v2: fn(*mut c_void, u64, usize) -> CuResult;
    cuLaunchKernel: fn(Ptr, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut Ptr, *mut Ptr) -> CuResult;
    cuFuncSetAttribute: fn(Ptr, i32, i32) -> CuResult;
    cuCtxSynchronize: fn() -> CuResult;
    cuStreamCreate: fn(*mut Ptr, u32) -> CuResult;
    cuStreamBeginCapture_v2: fn(Ptr, i32) -> CuResult;
    cuStreamEndCapture: fn(Ptr, *mut Ptr) -> CuResult;
    cuGraphInstantiateWithFlags: fn(*mut Ptr, Ptr, u64) -> CuResult;
    cuGraphLaunch: fn(Ptr, Ptr) -> CuResult;
    cuGraphDestroy: fn(Ptr) -> CuResult;
    cuGraphExecDestroy: fn(Ptr) -> CuResult;
    cuGetErrorString: fn(CuResult, *mut *const c_char) -> CuResult;
    nvrtcCreateProgram: fn(*mut Ptr, *const c_char, *const c_char, i32, *const *const c_char, *const *const c_char) -> i32;
    nvrtcCompileProgram: fn(Ptr, i32, *const *const c_char) -> i32;
    nvrtcGetPTXSize: fn(Ptr, *mut usize) -> i32;
    nvrtcGetPTX: fn(Ptr, *mut c_char) -> i32;
    nvrtcGetProgramLogSize: fn(Ptr, *mut usize) -> i32;
    nvrtcGetProgramLog: fn(Ptr, *mut c_char) -> i32;
    nvrtcDestroyProgram: fn(*mut Ptr) -> i32;
}

struct Api {
    f: Fns,
    pub device_name: String,
    pub compute: (i32, i32),
    pub nvrtc_path: String,
}

// the raw function pointers and library handles are process-global and immutable once loaded
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn check(&self, r: CuResult, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut msg: *const c_char = std::ptr::null();
        unsafe { (self.f.cuGetErrorString)(r, &mut msg) };
        let text = if msg.is_null() { "unknown error".to_string() } else { unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned() };
        Err(format!("CUDA {what} failed: {text} ({r})"))
    }
}

/// Newest NVRTC under the CUDA toolkit folder, or BETTERLLAMA_NVRTC.
fn find_nvrtc() -> Option<String> {
    if let Ok(p) = std::env::var("BETTERLLAMA_NVRTC") {
        return Some(p);
    }
    let mut roots = Vec::new();
    if let Ok(p) = std::env::var("CUDA_PATH") {
        roots.push(std::path::PathBuf::from(&p));
        roots.push(std::path::PathBuf::from(&p).join("bin"));
    }
    if let Ok(dirs) = std::fs::read_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
        roots.extend(dirs.flatten().map(|d| d.path().join("bin")));
    }
    let mut found: Vec<std::path::PathBuf> = roots
        .iter()
        .filter_map(|r| std::fs::read_dir(r).ok())
        .flat_map(|d| d.flatten().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.starts_with("nvrtc64_") && n.ends_with(".dll") && !n.contains(".alt")
        })
        .collect();
    found.sort();
    found.pop().map(|p| p.to_string_lossy().into_owned())
}

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let cuda = os::open("nvcuda.dll").ok_or("no NVIDIA driver (nvcuda.dll) found")?;
        let nvrtc_path = find_nvrtc().ok_or("no NVRTC found (install the CUDA toolkit or set BETTERLLAMA_NVRTC)")?;
        let nvrtc = os::open(&nvrtc_path).ok_or_else(|| format!("cannot load {nvrtc_path}"))?;
        let f = Fns::load(cuda, nvrtc)?;
        let mut api = Api { f, device_name: String::new(), compute: (0, 0), nvrtc_path };
        unsafe {
            api.check((api.f.cuInit)(0), "init")?;
            let mut dev = 0;
            api.check((api.f.cuDeviceGet)(&mut dev, 0), "device")?;
            let mut name = [0 as c_char; 256];
            api.check((api.f.cuDeviceGetName)(name.as_mut_ptr(), 256, dev), "device name")?;
            api.device_name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let (mut major, mut minor) = (0, 0);
            api.check((api.f.cuDeviceGetAttribute)(&mut major, 75, dev), "compute capability")?;
            api.check((api.f.cuDeviceGetAttribute)(&mut minor, 76, dev), "compute capability")?;
            api.compute = (major, minor);
            let mut ctx: Ptr = std::ptr::null_mut();
            api.check((api.f.cuCtxCreate_v2)(&mut ctx, 0, dev), "context")?;
        }
        Ok(api)
    })
    .as_ref()
    .map_err(|e| e.clone())
}

// ---------------------------------------------------------------- device memory and kernels

struct DeviceBuf {
    ptr: u64,
    bytes: usize,
}

impl DeviceBuf {
    fn alloc(bytes: usize) -> Result<DeviceBuf, String> {
        let a = api()?;
        let mut ptr = 0u64;
        a.check(unsafe { (a.f.cuMemAlloc_v2)(&mut ptr, bytes.max(4)) }, &format!("alloc {} MB", bytes >> 20))?;
        Ok(DeviceBuf { ptr, bytes })
    }

    fn upload<T: Copy>(data: &[T]) -> Result<DeviceBuf, String> {
        let bytes = std::mem::size_of_val(data);
        let buf = DeviceBuf::alloc(bytes)?;
        buf.write(data)?;
        Ok(buf)
    }

    fn write<T: Copy>(&self, data: &[T]) -> Result<(), String> {
        let a = api()?;
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes);
        a.check(unsafe { (a.f.cuMemcpyHtoD_v2)(self.ptr, data.as_ptr() as *const c_void, bytes) }, "copy to GPU")
    }

    /// Byte offset into the buffer, for passing a sub-range to a kernel.
    fn at(&self, bytes: usize) -> u64 {
        self.ptr + bytes as u64
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if let Ok(a) = api() {
            unsafe { (a.f.cuMemFree_v2)(self.ptr) };
        }
    }
}

#[derive(Clone, Copy)]
enum Arg {
    P(u64),
    I(i32),
    F(f32),
}

struct Kernels {
    embed: Ptr,
    rmsnorm: Ptr,
    rmsnorm_q8: Ptr, // one token: RMSNorm straight to int8 activations
    qk_norm_rope: Ptr,
    matmul_q8: Ptr,
    matmul_q4k: [Ptr; 2], // f32 activations: one-token and four-token variants
    matmul_q6k: [Ptr; 2],
    matmul_dp4a: [[Ptr; 2]; 3], // decode, int8 activations: [Q8_0 | Q4_K | Q6_K][1 token | up to 8]
    argmax_rows: Ptr,
    matmul_tiled: Ptr,
    matmul_mma: Option<Ptr>, // int8 tensor cores, sm_75 and newer
    matmul_mma_k: Option<[Ptr; 2]>, // the same for Q4_K / Q6_K with per-256 activation scales
    quantize_q8: Ptr,
    quantize_q8k: Ptr,
    attention_partial: Ptr,
    attention_partial_gqa: Ptr,
    attention_combine: Ptr,
    attention_prefill: Ptr,
    attention_prefill_tc: Option<[Ptr; 2]>, // f16 tensor cores (sm_75+), head size 64 / 128
    silu_mul: Ptr,
    silu_q8: Ptr,
    /// every launch goes here (a blocking stream: the synchronous copies still order against it); CUDA
    /// graphs need a stream other than the legacy default one
    stream: Ptr,
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    fn compile() -> Result<Kernels, String> {
        let a = api()?;
        let f = &a.f;
        unsafe {
            let src = CString::new(KERNELS).unwrap();
            let name = CString::new("betterllama.cu").unwrap();
            let mut prog: Ptr = std::ptr::null_mut();
            if (f.nvrtcCreateProgram)(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
                return Err("nvrtcCreateProgram failed".into());
            }
            let opts: Vec<CString> = [format!("--gpu-architecture=compute_{}{}", a.compute.0, a.compute.1), "-std=c++14".into(), format!("-DRPW={}", kq_rpw()), format!("-DQCH={}", qch()), format!("-DDP4A_BATCH_MINB={}", std::env::var("BETTERLLAMA_BATCH_MINB").unwrap_or("3".into())), format!("-DDP4A_BATCH_RPW={}", batch_rpw())]
                .into_iter()
                .map(|o| CString::new(o).unwrap())
                .collect();
            let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
            let rc = (f.nvrtcCompileProgram)(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
            if rc != 0 {
                let mut n = 0usize;
                (f.nvrtcGetProgramLogSize)(prog, &mut n);
                let mut log = vec![0 as c_char; n.max(1)];
                (f.nvrtcGetProgramLog)(prog, log.as_mut_ptr());
                (f.nvrtcDestroyProgram)(&mut prog);
                return Err(format!("kernel compile failed:\n{}", CStr::from_ptr(log.as_ptr()).to_string_lossy()));
            }
            let mut n = 0usize;
            (f.nvrtcGetPTXSize)(prog, &mut n);
            let mut ptx = vec![0 as c_char; n];
            (f.nvrtcGetPTX)(prog, ptx.as_mut_ptr());
            (f.nvrtcDestroyProgram)(&mut prog);

            let mut module: Ptr = std::ptr::null_mut();
            a.check((f.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void), "load kernels")?;
            let get = |name: &str| -> Result<Ptr, String> {
                let c = CString::new(name).unwrap();
                let mut func: Ptr = std::ptr::null_mut();
                a.check((f.cuModuleGetFunction)(&mut func, module, c.as_ptr()), name)?;
                Ok(func)
            };
            Ok(Kernels {
                embed: get("embed")?,
                rmsnorm: get("rmsnorm")?,
                rmsnorm_q8: get("rmsnorm_q8")?,
                qk_norm_rope: get("qk_norm_rope")?,
                matmul_q8: get("matmul_q8")?,
                matmul_q4k: [get("matmul_q4k_1")?, get("matmul_q4k_4")?],
                matmul_q6k: [get("matmul_q6k_1")?, get("matmul_q6k_4")?],
                matmul_dp4a: [
                    [get("matmul_q8_dp4a")?, get("matmul_q8_dp4a8")?],
                    [get("matmul_q4k_dp4a")?, get("matmul_q4k_dp4a8")?],
                    [get("matmul_q6k_dp4a")?, get("matmul_q6k_dp4a8")?],
                ],
                argmax_rows: get("argmax_rows")?,
                matmul_tiled: get("matmul_tiled")?,
                matmul_mma: if a.compute >= (7, 5) && std::env::var_os("BETTERLLAMA_NO_MMA").is_none() { Some(get("matmul_mma")?) } else { None },
                // BETTERLLAMA_NO_Q8K=1 keeps K-quant prompts on matmul_mma (per-32 activation scales)
                matmul_mma_k: if a.compute >= (7, 5) && std::env::var_os("BETTERLLAMA_NO_MMA").is_none() && std::env::var_os("BETTERLLAMA_NO_Q8K").is_none() {
                    Some([get("matmul_mma_q4k")?, get("matmul_mma_q6k")?])
                } else {
                    None
                },
                quantize_q8: get("quantize_q8")?,
                quantize_q8k: get("quantize_q8k")?,
                attention_partial: get("attention_partial")?,
                attention_partial_gqa: get("attention_partial_gqa")?,
                attention_combine: get("attention_combine")?,
                attention_prefill: get("attention_prefill")?,
                // BETTERLLAMA_NO_TC_ATT=1 keeps prompts on the f32 attention kernel
                attention_prefill_tc: if a.compute >= (7, 5) && std::env::var_os("BETTERLLAMA_NO_TC_ATT").is_none() {
                    Some([get("attention_prefill_tc64")?, get("attention_prefill_tc128")?])
                } else {
                    None
                },
                silu_mul: get("silu_mul")?,
                silu_q8: get("silu_q8")?,
                stream: {
                    let mut st: Ptr = std::ptr::null_mut();
                    a.check((f.cuStreamCreate)(&mut st, 0), "stream")?;
                    st
                },
            })
        }
    }

    fn launch(&self, func: Ptr, grid: (u32, u32), block: u32, shared: u32, args: &[Arg]) -> Result<(), String> {
        self.launch3(func, (grid.0, grid.1, 1), block, shared, args)
    }

    fn launch3(&self, func: Ptr, grid: (u32, u32, u32), block: u32, shared: u32, args: &[Arg]) -> Result<(), String> {
        let a = api()?;
        // each argument lives in a u64 slot; the kernel reads the first 4 or 8 bytes (little-endian)
        let mut slots: Vec<u64> = args
            .iter()
            .map(|arg| match *arg {
                Arg::P(p) => p,
                Arg::I(i) => i as u32 as u64,
                Arg::F(f) => f.to_bits() as u64,
            })
            .collect();
        let mut params: Vec<Ptr> = slots.iter_mut().map(|s| s as *mut u64 as Ptr).collect();
        a.check(
            unsafe { (a.f.cuLaunchKernel)(func, grid.0, grid.1, grid.2, block, 1, 1, shared, self.stream, params.as_mut_ptr(), std::ptr::null_mut()) },
            "kernel launch",
        )
    }
}

// ---------------------------------------------------------------- model

/// Weight format on the GPU (matches `fmt` in the kernels).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fmt {
    /// int8 values + separate f32 scale per 32
    Q8 = 0,
    /// the GGUF blocks as is: 144 bytes per 256 (f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes)
    Q4K = 1,
    /// 208-byte blocks of 256 (128 low-nibble bytes, 64 high-bit bytes, 16 int8 scales) + f32 d per block
    Q6K = 2,
}

/// A weight matrix on the GPU, repacked from its GGUF blocks so every field is naturally aligned and
/// every block starts on a 16-byte boundary.
struct GpuMatrix {
    fmt: Fmt,
    q: DeviceBuf,         // Q8: int8 values; K-quants: the blocks
    d: Option<DeviceBuf>, // Q8: scale per 32; Q6K: d per block
    rows: usize,
    cols: usize,
}

impl GpuMatrix {
    fn d_ptr(&self) -> u64 {
        self.d.as_ref().map_or(0, |d| d.ptr)
    }
}

/// Appends one tensor's blocks to `q`/`d` in the GPU layout of its format.
fn repack(ty: GgmlType, raw: &[u8], q: &mut Vec<u8>, d: &mut Vec<f32>) -> Result<Fmt, String> {
    let f16 = |b: &[u8]| f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    match ty {
        GgmlType::Q8_0 => {
            for blk in raw.chunks_exact(34) {
                d.push(f16(blk));
                q.extend_from_slice(&blk[2..]);
            }
            Ok(Fmt::Q8)
        }
        GgmlType::Q4K => {
            q.extend_from_slice(raw); // 144-byte blocks: already 16-byte multiples
            Ok(Fmt::Q4K)
        }
        GgmlType::Q6K => {
            for blk in raw.chunks_exact(210) {
                q.extend_from_slice(&blk[..208]);
                d.push(f16(&blk[208..]));
            }
            Ok(Fmt::Q6K)
        }
        other => Err(format!("GPU backend does not handle {} weights yet", other.name())),
    }
}

/// Uploads tensors whose rows are concatenated in one output (e.g. Q, K and V). Neighbours of the same
/// format are merged into one matrix (one launch); returns each matrix with its output row offset.
fn upload_parts(g: &Gguf, names: &[String]) -> Result<Vec<(GpuMatrix, usize)>, String> {
    let mut parts = Vec::new();
    let mut i = 0;
    let mut row_off = 0;
    while i < names.len() {
        let ty = g.tensor(&names[i]).ok_or_else(|| format!("missing tensor {}", names[i]))?.ty;
        let (mut q, mut d) = (Vec::new(), Vec::new());
        let (mut rows, mut cols, mut fmt) = (0, 0, Fmt::Q8);
        while i < names.len() {
            let t = g.tensor(&names[i]).ok_or_else(|| format!("missing tensor {}", names[i]))?;
            if t.ty != ty {
                break;
            }
            cols = t.dims[0] as usize;
            if (cols % 256 != 0 && ty != GgmlType::Q8_0) || cols % 32 != 0 {
                return Err(format!("unsupported row length {cols} in {}", names[i]));
            }
            rows += t.dims[1] as usize;
            fmt = repack(t.ty, &g.mmap[t.offset..t.offset + t.n_bytes], &mut q, &mut d)?;
            i += 1;
        }
        let d = if fmt != Fmt::Q4K { Some(DeviceBuf::upload(&d)?) } else { None };
        parts.push((GpuMatrix { fmt, q: DeviceBuf::upload(&q)?, d, rows, cols }, row_off));
        row_off += rows;
    }
    Ok(parts)
}

fn upload_one(g: &Gguf, name: &str) -> Result<GpuMatrix, String> {
    Ok(upload_parts(g, &[name.to_string()])?.pop().unwrap().0)
}

fn upload_vec(g: &Gguf, name: &str) -> Result<DeviceBuf, String> {
    let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if t.ty != GgmlType::F32 {
        return Err(format!("GPU backend expects F32 norm weights ({name} is {})", t.ty.name()));
    }
    let v: Vec<f32> = g.mmap[t.offset..t.offset + t.n_bytes].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    DeviceBuf::upload(&v)
}

struct GpuLayer {
    attn_norm: DeviceBuf,
    qkv: Vec<(GpuMatrix, usize)>, // with output row offsets in the packed [q | k | v] rows
    q_norm: Option<DeviceBuf>,    // Qwen3 per-head norms
    k_norm: Option<DeviceBuf>,
    wo: GpuMatrix,
    ffn_norm: DeviceBuf,
    gate_up: Vec<(GpuMatrix, usize)>, // [gate | up]
    down: GpuMatrix,
}

pub struct GpuModel {
    pub cfg: Config,
    k: Arc<Kernels>,
    embd: GpuMatrix,
    output: Option<GpuMatrix>, // None = tied to the embedding
    output_norm: DeviceBuf,
    layers: Vec<GpuLayer>,
    prefill_shared: Option<usize>,
    pub device: String,
    /// Q8_0 weights decode with int8 activations too (for draft models; otherwise exact f32)
    pub int8_q8: bool,
}

impl GpuModel {
    pub fn load(g: &Gguf, cfg: Config) -> Result<GpuModel, String> {
        if cfg.head_dim > ATT_MAX_HD || ATT_MAX_HD % cfg.head_dim != 0 || cfg.head_dim % 4 != 0 {
            return Err(format!("GPU backend: head size {} not supported (needs a divisor of {ATT_MAX_HD})", cfg.head_dim));
        }
        let a = api()?;
        let k = Arc::new(Kernels::compile()?);
        let opt_vec = |name: String| g.tensor(&name).map(|_| upload_vec(g, &name)).transpose();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(GpuLayer {
                attn_norm: upload_vec(g, &p("attn_norm"))?,
                qkv: upload_parts(g, &[p("attn_q"), p("attn_k"), p("attn_v")])?,
                q_norm: opt_vec(p("attn_q_norm"))?,
                k_norm: opt_vec(p("attn_k_norm"))?,
                wo: upload_one(g, &p("attn_output"))?,
                ffn_norm: upload_vec(g, &p("ffn_norm"))?,
                gate_up: upload_parts(g, &[p("ffn_gate"), p("ffn_up")])?,
                down: upload_one(g, &p("ffn_down"))?,
            });
        }
        let embd = upload_one(g, "token_embd.weight")?;
        let output = if g.tensor("output.weight").is_some() { Some(upload_one(g, "output.weight")?) } else { None };

        // prompt attention: dynamic shared memory above the default 48 KB has to be requested per kernel
        let ld = cfg.head_dim + 1;
        let bytes = ((2 * PREFILL_KT + PREFILL_QT) * ld + PREFILL_QT * (PREFILL_KT + 1)) * 4;
        const MAX_DYNAMIC_SHARED: i32 = 8; // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        let prefill_shared = (bytes <= 48 * 1024 || unsafe { (a.f.cuFuncSetAttribute)(k.attention_prefill, MAX_DYNAMIC_SHARED, bytes as i32) } == 0).then_some(bytes);

        let device = format!("{} (sm_{}{}, NVRTC {})", a.device_name, a.compute.0, a.compute.1, std::path::Path::new(&a.nvrtc_path).file_name().and_then(|n| n.to_str()).unwrap_or("?"));
        // the logits buffer is sized from cfg.vocab; a mismatch would write past it
        let out_rows = output.as_ref().unwrap_or(&embd).rows;
        if out_rows != cfg.vocab {
            return Err(format!("output layer has {out_rows} rows but the vocabulary has {} tokens", cfg.vocab));
        }
        Ok(GpuModel { output_norm: upload_vec(g, "output_norm.weight")?, cfg, k, embd, output, layers, prefill_shared, device, int8_q8: false })
    }

    /// With BETTERLLAMA_PROFILE set, waits for the GPU after `f` so its time lands in bucket `op`.
    fn timed(&self, op: usize, f: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
        if !profile::enabled() {
            return f();
        }
        profile::time(op, || {
            f()?;
            let a = api()?;
            a.check(unsafe { (a.f.cuCtxSynchronize)() }, "synchronize")
        })
    }

    /// For each part, `out[t*out_stride + off + r] (+)= W[r] · x[t]` over `n` input rows of `cols` floats.
    fn matmul(&self, parts: &[(&GpuMatrix, usize)], x: u64, out: u64, out_stride: usize, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        let k = &self.k;
        let cols = parts[0].0.cols;
        if n >= TILED_MIN_BATCH {
            if let Some(mk) = k.matmul_mma_k.filter(|_| parts.iter().all(|(m, _)| m.fmt != Fmt::Q8)) {
                // K-quant prompts: one activation scale per 128 (or 256) values keeps the tensor-core epilogue integer
                let chunks = n * cols / qch();
                k.launch(k.quantize_q8k, ((chunks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(chunks as i32)])?;
                for &(m, off) in parts {
                    k.launch(
                        mk[(m.fmt == Fmt::Q6K) as usize],
                        (m.rows.div_ceil(TILE) as u32, n.div_ceil(TILE) as u32),
                        128,
                        0,
                        &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
                    )?;
                }
                return Ok(());
            }
            // prompts: quantise the activations once, then int8 tiles with dp4a
            let blocks = n * cols / 32;
            k.launch(k.quantize_q8, ((blocks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(blocks as i32)])?;
            for &(m, off) in parts {
                let (func, threads) = match k.matmul_mma {
                    Some(f) => (f, 128),
                    None => (k.matmul_tiled, 256),
                };
                k.launch(
                    func,
                    (m.rows.div_ceil(TILE) as u32, n.div_ceil(TILE) as u32),
                    threads,
                    0,
                    &[Arg::I(m.fmt as i32), Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
                )?;
            }
            return Ok(());
        }
        // decoding (up to DECODE_MAX_BATCH tokens): int8 activations + dp4a (BETTERLLAMA_EXACT=1 keeps f32)
        if self.dp4a_path(parts, n) {
            let blocks = n * cols / 32;
            k.launch(k.quantize_q8, ((blocks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(blocks as i32)])?;
            return self.matmul_dp4a(parts, out, out_stride, n, accumulate, s);
        }
        // otherwise f32 activations; 256 threads = 8 warps, one row each for Q8_0, RPW rows each for K-quants
        for &(m, off) in parts {
            let tail = [Arg::P(x), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)];
            let (variant, tb) = if n == 1 { (0, 1) } else { (1, KQ_TB) };
            let kq_grid = (m.rows.div_ceil(8 * kq_rpw()) as u32, n.div_ceil(tb) as u32);
            match m.fmt {
                Fmt::Q8 => k.launch(k.matmul_q8, (m.rows.div_ceil(8) as u32, n.div_ceil(MATMUL_TB) as u32), 256, 0, &[&[Arg::P(m.q.ptr), Arg::P(m.d_ptr())][..], &tail].concat())?,
                Fmt::Q4K => k.launch(k.matmul_q4k[variant], kq_grid, 256, 0, &[&[Arg::P(m.q.ptr), Arg::P(0)][..], &tail].concat())?,
                Fmt::Q6K => k.launch(k.matmul_q6k[variant], kq_grid, 256, 0, &[&[Arg::P(m.q.ptr), Arg::P(m.d_ptr())][..], &tail].concat())?,
            }
        }
        Ok(())
    }

    /// Decoding batches (1..=DECODE_MAX_BATCH tokens) with K-quant weights go through int8 activations and
    /// dp4a; Q8_0 too when the model is a draft (`int8_q8`), keeping exact f32 decoding otherwise.
    fn dp4a_path(&self, parts: &[(&GpuMatrix, usize)], n: usize) -> bool {
        n <= DECODE_MAX_BATCH && !crate::quant::exact_mode() && parts.iter().all(|(m, _)| m.fmt != Fmt::Q8 || self.int8_q8)
    }

    /// The dp4a matmuls on `n` tokens' activations already quantised into `s.xq` / `s.xd`. One token uses
    /// the 1-token kernels, batches the 8-token ones; both give a token bit-identical results.
    fn matmul_dp4a(&self, parts: &[(&GpuMatrix, usize)], out: u64, out_stride: usize, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        let k = &self.k;
        for &(m, off) in parts {
            let rpw = if m.fmt == Fmt::Q8 { 1 } else if n > 1 { batch_rpw() } else { kq_rpw() };
            k.launch(
                k.matmul_dp4a[m.fmt as usize][(n > 1) as usize],
                (m.rows.div_ceil(8 * rpw) as u32, 1),
                256,
                0,
                &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
            )?;
        }
        Ok(())
    }

    /// RMSNorm of `x` into `s.xb`, then the matmul; for one token on the dp4a path the norm writes the
    /// int8 activations directly.
    fn norm_matmul(&self, norm: &DeviceBuf, parts: &[(&GpuMatrix, usize)], out: u64, out_stride: usize, n: usize, s: &GpuSession) -> Result<(), String> {
        let (k, c) = (&self.k, &self.cfg);
        if self.dp4a_path(parts, n) {
            k.launch(k.rmsnorm_q8, (n as u32, 1), 1024, 0, &[Arg::P(s.x.ptr), Arg::P(norm.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(c.dim as i32), Arg::F(c.norm_eps)])?;
            return self.matmul_dp4a(parts, out, out_stride, n, false, s);
        }
        k.launch(k.rmsnorm, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(norm.ptr), Arg::P(s.xb.ptr), Arg::I(c.dim as i32), Arg::F(c.norm_eps)])?;
        self.matmul(parts, s.xb.ptr, out, out_stride, n, false, s)
    }

    fn matmul_one(&self, m: &GpuMatrix, x: u64, out: u64, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        self.matmul(&[(m, 0)], x, out, m.rows, n, accumulate, s)
    }

    /// The grouped-query attention kernel applies (head size a multiple of 32, at most 8 heads per kv head).
    fn gqa_kernel(&self) -> bool {
        let c = &self.cfg;
        c.head_dim % 32 == 0 && c.n_heads % c.n_kv_heads == 0 && c.n_heads / c.n_kv_heads <= ATT_MAX_G
    }

    /// Slices to aim for per token when generating: enough that a single token spreads over the GPU.
    /// Each token's slice size then follows from its own span alone (att_chunk_for in the kernels).
    fn att_want(&self) -> usize {
        ATT_TARGET_BLOCKS.div_ceil(if self.gqa_kernel() { self.cfg.n_kv_heads } else { self.cfg.n_heads })
    }

    /// Mirror of the kernels' att_chunk_for: short contexts in one block per head (no merge pass),
    /// longer ones split.
    fn att_chunk(span: usize, want: usize) -> usize {
        if span <= SPLIT_MIN_SPAN {
            return ATT_MAX_CHUNK;
        }
        span.div_ceil(want).clamp(32, ATT_MAX_CHUNK)
    }

    /// Most slices any token of a ctx-position session needs: the fixed split grid of graph decoding.
    fn max_splits(&self, ctx: usize) -> usize {
        let want = self.att_want();
        (1..=ctx).map(|span| span.div_ceil(Self::att_chunk(span, want))).max().unwrap_or(1)
    }

    /// posp: device pointer to the position (CUDA graphs), else 0 and s.pos is used. ixed_split:
    /// one-token steps use the session's widest split grid, so a captured graph fits every position.
    fn attention(&self, s: &GpuSession, n: usize, kc: u64, vc: u64, posp: u64, fixed_split: bool) -> Result<(), String> {
        let c = &self.cfg;
        let k = &self.k;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let tc = k.attention_prefill_tc.and_then(|f| match c.head_dim {
            64 => Some(f[0]),
            128 => Some(f[1]),
            _ => None,
        });
        if let (true, Some(func)) = (n >= TILED_MIN_BATCH, tc) {
            return k.launch(
                func,
                (c.n_heads as u32, n.div_ceil(PREFILL_TC_QT) as u32),
                128,
                0,
                &[Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::P(s.att.ptr), Arg::I(s.pos as i32), Arg::I(n as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale)],
            );
        }
        if let (true, Some(shared)) = (n >= TILED_MIN_BATCH, self.prefill_shared) {
            return k.launch(
                k.attention_prefill,
                (c.n_heads as u32, n.div_ceil(PREFILL_QT) as u32),
                128,
                shared as u32,
                &[Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::P(s.att.ptr), Arg::I(s.pos as i32), Arg::I(n as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale)],
            );
        }
        // split-context path, a few tokens at a time so the partial-results buffer stays small
        let (q_dim, kv_dim) = (c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
        let row = (q_dim + 2 * kv_dim) * 4;
        for b0 in (0..n).step_by(SPLIT_MAX_BATCH) {
            let m = (n - b0).min(SPLIT_MAX_BATCH);
            let pos0 = s.pos + b0;
            // the grid covers the token needing the most slices; the others' extra slices are empty
            let want = self.att_want();
            let nsplit = if fixed_split {
                self.max_splits(s.ctx)
            } else {
                (pos0 + 1..=pos0 + m).map(|span| span.div_ceil(Self::att_chunk(span, want))).max().unwrap_or(1)
            };
            let (qkv, out) = (s.qkv.at(b0 * row), s.att.at(b0 * q_dim * 4));
            let (func, heads) = if self.gqa_kernel() { (k.attention_partial_gqa, c.n_kv_heads) } else { (k.attention_partial, c.n_heads) };
            k.launch3(
                func,
                (heads as u32, m as u32, nsplit as u32),
                128,
                0,
                &[Arg::P(qkv), Arg::P(kc), Arg::P(vc), Arg::P(s.att_part.ptr), Arg::P(out), Arg::I(pos0 as i32), Arg::P(posp), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale), Arg::I(nsplit as i32), Arg::I(want as i32)],
            )?;
            if nsplit > 1 {
                k.launch(
                    k.attention_combine,
                    (c.n_heads as u32, m as u32),
                    128,
                    0,
                    &[Arg::P(s.att_part.ptr), Arg::P(out), Arg::I(pos0 as i32), Arg::P(posp), Arg::I(c.n_heads as i32), Arg::I(c.head_dim as i32), Arg::I(nsplit as i32), Arg::I(want as i32)],
                )?;
            }
        }
        Ok(())
    }

    pub fn prefill(&self, tokens: &[u32], s: &mut GpuSession) -> Result<(), String> {
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_BATCH).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            self.forward_batch(chunk, s, i + 1 == chunks.len())?;
        }
        Ok(())
    }

    /// Runs `tokens` at positions `s.pos..`; copies next-token logits into `s.logits` if asked.
    pub fn forward_batch(&self, tokens: &[u32], s: &mut GpuSession, want_logits: bool) -> Result<(), String> {
        self.forward(tokens, s, if want_logits { Out::Last } else { Out::None }).map(|_| ())
    }

    /// Runs up to DECODE_MAX_BATCH `tokens` and returns the greedy next token after each of them (argmax on
    /// the GPU). Each position gets exactly the logits a one-token step would give it, so this is how a
    /// speculative draft is checked.
    pub fn forward_argmax(&self, tokens: &[u32], s: &mut GpuSession) -> Result<Vec<u32>, String> {
        if tokens.len() > DECODE_MAX_BATCH {
            return Err(format!("argmax batch of {} exceeds {DECODE_MAX_BATCH}", tokens.len()));
        }
        self.forward(tokens, s, Out::Argmax)
    }

    fn forward(&self, tokens: &[u32], s: &mut GpuSession, want: Out) -> Result<Vec<u32>, String> {
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > MAX_BATCH {
            return Err(format!("GPU batch of {n} exceeds {MAX_BATCH}"));
        }
        if s.pos + n > s.ctx {
            return Err(format!("context full ({} tokens)", s.ctx));
        }
        let k = &self.k;
        let a = api()?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        s.tokens.write(&ids)?;
        // one-token steps replay a CUDA graph captured on the session's first one (BETTERLLAMA_NO_GRAPH=1:
        // plain launches); the kernels then read the position from s.pos_dev
        if n == 1 && want != Out::None && graphs_enabled() {
            s.pos_dev.write(&[s.pos as i32])?;
            let slot = (want == Out::Argmax) as usize;
            if s.graphs[slot].is_null() {
                const RELAXED: i32 = 2; // CU_STREAM_CAPTURE_MODE_RELAXED
                a.check(unsafe { (a.f.cuStreamBeginCapture_v2)(k.stream, RELAXED) }, "begin graph capture")?;
                let launched = self.launches(1, s, want, s.pos_dev.ptr, true);
                let mut graph: Ptr = std::ptr::null_mut();
                let ended = unsafe { (a.f.cuStreamEndCapture)(k.stream, &mut graph) };
                launched?;
                a.check(ended, "end graph capture")?;
                let mut exec: Ptr = std::ptr::null_mut();
                let made = unsafe { (a.f.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) };
                unsafe { (a.f.cuGraphDestroy)(graph) };
                a.check(made, "instantiate graph")?;
                s.graphs[slot] = exec;
            }
            a.check(unsafe { (a.f.cuGraphLaunch)(s.graphs[slot], k.stream) }, "graph launch")?;
        } else {
            self.launches(n, s, want, 0, false)?;
        }
        let mut picks = Vec::new();
        if want == Out::Last {
            a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(s.logits.as_mut_ptr() as *mut c_void, s.logits_dev.ptr, s.logits.len() * 4) }, "copy logits")?;
        } else if want == Out::Argmax {
            let mut ids = vec![0i32; n];
            a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(ids.as_mut_ptr() as *mut c_void, s.picks.ptr, n * 4) }, "copy argmax")?;
            picks = ids.into_iter().map(|i| i as u32).collect();
        }
        s.pos += n;
        Ok(picks)
    }

    /// Every kernel of a forward pass over `n` tokens at `s.pos..` (`posp`: the position from device memory
    /// instead, for graphs), ending with the last token's logits (Last) or every token's argmax (Argmax) on
    /// the device. Launches only, so it can be captured.
    fn launches(&self, n: usize, s: &GpuSession, want: Out, posp: u64, fixed_split: bool) -> Result<(), String> {
        let c = &self.cfg;
        let (dim, hd) = (c.dim, c.head_dim);
        let kv_dim = c.n_kv_heads * hd;
        let k = &self.k;

        // profile buckets match profile::OPS; with profiling off `timed` is a plain call
        self.timed(0, || {
            k.launch(k.embed, (n as u32, 1), 256, 0, &[Arg::I(self.embd.fmt as i32), Arg::P(self.embd.q.ptr), Arg::P(self.embd.d_ptr()), Arg::P(s.tokens.ptr), Arg::P(s.x.ptr), Arg::I(dim as i32)])
        })?;
        let layer_kv = s.ctx * kv_dim * 4;
        let qkv_rows = (c.n_heads + 2 * c.n_kv_heads) * hd;
        for (l, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = (s.k_cache.at(l * layer_kv), s.v_cache.at(l * layer_kv));
            let qkv: Vec<(&GpuMatrix, usize)> = layer.qkv.iter().map(|(m, off)| (m, *off)).collect();
            self.timed(2, || self.norm_matmul(&layer.attn_norm, &qkv, s.qkv.ptr, qkv_rows, n, s))?;
            self.timed(3, || {
                let (qn, kn) = match (&layer.q_norm, &layer.k_norm) {
                    (Some(qn), Some(kn)) => (qn.ptr, kn.ptr),
                    _ => (0, 0),
                };
                k.launch(
                    k.qk_norm_rope,
                    (n as u32, (c.n_heads + 2 * c.n_kv_heads) as u32),
                    64,
                    0,
                    &[Arg::P(s.qkv.ptr), Arg::P(qn), Arg::P(kn), Arg::P(kc), Arg::P(vc), Arg::I(s.pos as i32), Arg::P(posp), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(hd as i32), Arg::I(c.rope_dim as i32), Arg::F(c.rope_base), Arg::I(c.rope_neox as i32), Arg::F(c.norm_eps)],
                )
            })?;
            self.timed(4, || self.attention(s, n, kc, vc, posp, fixed_split))?;
            self.timed(5, || self.matmul_one(&layer.wo, s.att.ptr, s.x.ptr, n, true, s))?;

            self.timed(6, || {
                let gate_up: Vec<(&GpuMatrix, usize)> = layer.gate_up.iter().map(|(m, off)| (m, *off)).collect();
                self.norm_matmul(&layer.ffn_norm, &gate_up, s.gu.ptr, 2 * c.ffn_dim, n, s)?;
                let down = [(&layer.down, 0)];
                if self.dp4a_path(&down, n) {
                    // decoding: silu(gate) * up straight to int8 for the down projection
                    let nv = (c.ffn_dim / 4) as u32;
                    k.launch(k.silu_q8, (nv.div_ceil(256), n as u32), 256, 0, &[Arg::P(s.gu.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(c.ffn_dim as i32)])?;
                    return self.matmul_dp4a(&down, s.x.ptr, dim, n, true, s);
                }
                let total = (n * c.ffn_dim) as u32;
                k.launch(k.silu_mul, (total.div_ceil(256), 1), 256, 0, &[Arg::P(s.gu.ptr), Arg::P(s.h.ptr), Arg::I(c.ffn_dim as i32), Arg::I(n as i32)])?;
                self.matmul_one(&layer.down, s.h.ptr, s.x.ptr, n, true, s)
            })?;
        }

        if want != Out::None {
            self.timed(7, || {
                // Last: the last token's logits; Argmax: every token's, reduced on the GPU
                let rows = if want == Out::Last { 1 } else { n };
                let first = s.x.at((n - rows) * dim * 4);
                k.launch(k.rmsnorm, (rows as u32, 1), 256, 0, &[Arg::P(first), Arg::P(self.output_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)])?;
                self.matmul_one(self.output.as_ref().unwrap_or(&self.embd), s.xb.ptr, s.logits_dev.ptr, rows, false, s)?;
                if want == Out::Argmax {
                    k.launch(k.argmax_rows, (rows as u32, 1), 1024, 0, &[Arg::P(s.logits_dev.ptr), Arg::I(c.vocab as i32), Arg::P(s.picks.ptr)])?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

/// What a forward pass hands back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Out {
    None,
    /// the last token's logits, copied to `GpuSession::logits`
    Last,
    /// the greedy next token after every token of the batch
    Argmax,
}

pub struct GpuSession {
    pub pos: usize,
    pub ctx: usize,
    tokens: DeviceBuf,
    x: DeviceBuf,
    xb: DeviceBuf,
    qkv: DeviceBuf,
    att: DeviceBuf,
    gu: DeviceBuf,
    h: DeviceBuf,
    k_cache: DeviceBuf,
    v_cache: DeviceBuf,
    xq: DeviceBuf,       // int8 activations for the tiled matmul
    xd: DeviceBuf,       // their per-32 scales
    xsum: DeviceBuf,     // and per-32 sums of the int8 values
    att_part: DeviceBuf, // flash-decoding partial results
    logits_dev: DeviceBuf, // up to DECODE_MAX_BATCH rows
    picks: DeviceBuf,      // their argmax
    pos_dev: DeviceBuf,    // the position, for CUDA graphs
    graphs: [Ptr; 2],      // captured one-token steps: Last, Argmax
    pub logits: Vec<f32>,
}

impl Drop for GpuSession {
    fn drop(&mut self) {
        if let Ok(a) = api() {
            for g in self.graphs.iter().filter(|g| !g.is_null()) {
                unsafe { (a.f.cuGraphExecDestroy)(*g) };
            }
        }
    }
}

/// One-token steps replay CUDA graphs unless BETTERLLAMA_NO_GRAPH is set or the profiler (which
/// synchronises after every operation) is on.
fn graphs_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BETTERLLAMA_NO_GRAPH").is_none()) && !profile::enabled()
}

impl GpuSession {
    pub fn new(cfg: &Config, ctx: usize) -> Result<GpuSession, String> {
        let ctx = ctx.min(cfg.max_ctx);
        let (q_dim, kv_dim) = (cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
        let f = |n: usize| DeviceBuf::alloc(n * 4);
        let widest = cfg.dim.max(q_dim).max(cfg.ffn_dim);
        // split attention runs at most SPLIT_MAX_BATCH tokens at a time, with slices of >= 32 positions
        let slices = SPLIT_MAX_BATCH * ctx.div_ceil(32);
        Ok(GpuSession {
            xq: DeviceBuf::alloc(MAX_BATCH * widest)?,
            xd: f(MAX_BATCH * widest / 32)?,
            xsum: f(MAX_BATCH * widest / 32)?,
            att_part: f(slices * cfg.n_heads * (cfg.head_dim + 2))?,
            pos: 0,
            ctx,
            tokens: f(MAX_BATCH)?,
            x: f(MAX_BATCH * cfg.dim)?,
            xb: f(MAX_BATCH * cfg.dim)?,
            qkv: f(MAX_BATCH * (q_dim + 2 * kv_dim))?,
            att: f(MAX_BATCH * q_dim)?,
            gu: f(MAX_BATCH * 2 * cfg.ffn_dim)?,
            h: f(MAX_BATCH * cfg.ffn_dim)?,
            k_cache: f(cfg.n_layers * ctx * kv_dim)?,
            v_cache: f(cfg.n_layers * ctx * kv_dim)?,
            logits_dev: f(DECODE_MAX_BATCH * cfg.vocab)?,
            picks: f(DECODE_MAX_BATCH)?,
            pos_dev: f(1)?,
            graphs: [std::ptr::null_mut(); 2],
            logits: vec![0.0; cfg.vocab],
        })
    }
}
