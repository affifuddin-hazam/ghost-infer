//! NVIDIA GPU backend. Loads the CUDA driver (nvcuda.dll) and NVRTC at runtime, compiles the kernels
//! below to PTX for the installed GPU, and runs the whole transformer on the device. No build-time CUDA
//! dependency: without a GPU the binary still works on the CPU.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::{Arc, OnceLock};

use crate::gguf::{GgmlType, Gguf, Value};
use crate::model::Config;
use crate::profile;
use crate::quant::f16_to_f32;

const MAX_BATCH: usize = 1024; // prompt tokens per GPU pass: big batches keep the attention kernel busy
const MATMUL_TB: usize = 8; // tokens per matmul block row (must match TB in the kernel source)
const TILE: usize = 64; // must match TILE in the kernel source
const TILED_MIN_BATCH: usize = 16; // from this many tokens, the int8 tiled matmul beats one-warp-per-row
const ATT_MAX_CHUNK: usize = 256; // must match ATT_MAX_CHUNK
const ATT_MAX_HD: usize = 128; // must match ATT_MAX_HD
const ATT_TARGET_BLOCKS: usize = 96; // aim for a few blocks per SM when splitting attention for one token
const PREFILL_QT: usize = 32; // query tokens per prompt-attention block (must match PQT)
const PREFILL_KT: usize = 32; // key positions per shared-memory tile (must match PKT)
const SPLIT_MAX_BATCH: usize = TILED_MIN_BATCH - 1; // tokens per split-attention launch

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

// x[b] = embedding row of tokens[b]
__global__ void embed_q8(const signed char* __restrict__ wq, const float* __restrict__ wd,
                         const int* __restrict__ tokens, float* __restrict__ x, int cols) {
    int b = blockIdx.x;
    size_t tok = tokens[b];
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        x[(size_t)b * cols + i] = wd[tok * (cols >> 5) + (i >> 5)] * (float)wq[tok * cols + i];
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

// out[b*rows + r] (+)= dot(W[r], x[b]). One warp per row, TB tokens per block in y.
// W is int8 (rows x cols) with one f32 scale per 32 values.
__global__ void matmul_q8(const signed char* __restrict__ wq, const float* __restrict__ wd,
                          const float* __restrict__ x, float* __restrict__ out,
                          int rows, int cols, int n, int accumulate) {
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
            size_t idx = (size_t)(b0 + j) * rows + row;
            out[idx] = accumulate ? out[idx] + v : v;
        }
    }
}

// qkv rows are [q | k | v]; rotates q and k (adjacent pairs) and stores k, v at their positions
__global__ void rope_cache(float* __restrict__ qkv, float* __restrict__ kc, float* __restrict__ vc,
                           int pos0, int n_heads, int n_kv, int hd, int rope_dim, float base) {
    int b = blockIdx.x, pos = pos0 + b;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    float* row = qkv + (size_t)b * (q_dim + 2 * kv_dim);
    int half = rope_dim / 2;
    for (int idx = threadIdx.x; idx < (n_heads + n_kv) * half; idx += blockDim.x) {
        int h = idx / half, i = idx % half;
        float* vec = row + h * hd; // k heads directly follow the q heads
        float freq = powf(base, -2.0f * (float)i / (float)rope_dim);
        float s, c;
        sincosf((float)pos * freq, &s, &c);
        float a = vec[2 * i], bb = vec[2 * i + 1];
        vec[2 * i] = a * c - bb * s;
        vec[2 * i + 1] = a * s + bb * c;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < kv_dim; j += blockDim.x) {
        kc[(size_t)pos * kv_dim + j] = row[q_dim + j];
        vc[(size_t)pos * kv_dim + j] = row[q_dim + kv_dim + j];
    }
}

// Flash-decoding attention, part 1: one block per (head, token, slice of `chunk` positions).
// Writes the slice's unnormalised output plus its max score and exp-sum to `part` (hd + 2 floats).
#define ATT_MAX_CHUNK 256
#define ATT_MAX_HD 128
// With nsplit == 1 there is nothing to merge: the block writes the normalised result straight to `out`.
__global__ void attention_partial(const float* __restrict__ qkv, const float* __restrict__ kc, const float* __restrict__ vc,
                                  float* __restrict__ part, float* __restrict__ out, int pos0, int n_heads, int n_kv, int hd,
                                  float scale, int nsplit, int chunk) {
    __shared__ float sc[ATT_MAX_CHUNK];
    __shared__ float qs[ATT_MAX_HD];
    __shared__ float red[32];
    __shared__ float accs[4][ATT_MAX_HD];
    int h = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int span = pos0 + b + 1; // causal
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
        if (nsplit == 1) out[(size_t)b * q_dim + h * hd + threadIdx.x] = acc / se;
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

// Flash-decoding attention, part 2: merge the slices of each (head, token) into the normalised output.
__global__ void attention_combine(const float* __restrict__ part, float* __restrict__ out,
                                  int pos0, int n_heads, int hd, int nsplit, int chunk) {
    int h = blockIdx.x, b = blockIdx.y;
    int span = pos0 + b + 1;
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

// Activations to int8 in blocks of 32 (one warp per block): q = round(x / d), d = max|x| / 127.
__global__ void quantize_q8(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ d, int n_blocks) {
    int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (blk >= n_blocks) return;
    float v = x[(size_t)blk * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    q[(size_t)blk * 32 + lane] = (signed char)roundf(v * inv);
    if (lane == 0) d[blk] = dd;
}

// Batched int8 matmul for prompts: out[t*rows + r] (+)= sum_k W[r][k] * X[t][k], both Q8 (int8 + scale per 32).
// 64x64 output tile per 256-thread block; each thread owns 4 rows x 4 tokens; __dp4a does 4 int8 MACs.
#define TILE 64
__global__ void matmul_q8_tiled(const signed char* __restrict__ wq, const float* __restrict__ wd,
                                const signed char* __restrict__ xq, const float* __restrict__ xd,
                                float* __restrict__ out, int rows, int cols, int n, int accumulate) {
    __shared__ int ws[TILE][9]; // 8 ints = 32 int8 per row, +1 to spread rows over banks
    __shared__ int xs[TILE][9];
    __shared__ float wds[TILE], xds[TILE];
    int tid = threadIdx.x;
    int tx = tid & 15, ty = tid >> 4; // tx picks rows (coalesced stores), ty picks tokens
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int nb = cols >> 5;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int kb = 0; kb < nb; kb++) {
        for (int idx = tid; idx < TILE * 8; idx += 256) {
            int r = idx >> 3, w = idx & 7;
            int gr = row0 + r, gt = tok0 + r;
            ws[r][w] = gr < rows ? *reinterpret_cast<const int*>(wq + (size_t)gr * cols + kb * 32 + w * 4) : 0;
            xs[r][w] = gt < n ? *reinterpret_cast<const int*>(xq + (size_t)gt * cols + kb * 32 + w * 4) : 0;
        }
        if (tid < TILE) {
            int gr = row0 + tid;
            wds[tid] = gr < rows ? wd[(size_t)gr * nb + kb] : 0.f;
        } else if (tid < 2 * TILE) {
            int t = tid - TILE, gt = tok0 + t;
            xds[t] = gt < n ? xd[(size_t)gt * nb + kb] : 0.f;
        }
        __syncthreads();
        int wr[4][8], xr[4][8];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int w = 0; w < 8; w++) { wr[i][w] = ws[tx + 16 * i][w]; xr[i][w] = xs[ty + 16 * i][w]; }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float dw = wds[tx + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int s = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++) s = __dp4a(wr[i][w], xr[j][w], s);
                acc[i][j] += (float)s * dw * xds[ty + 16 * j];
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
                size_t idx = (size_t)t * rows + r;
                out[idx] = accumulate ? out[idx] + acc[i][j] : acc[i][j];
            }
        }
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
    cuCtxSynchronize: fn() -> CuResult;
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
    embed_q8: Ptr,
    rmsnorm: Ptr,
    matmul_q8: Ptr,
    matmul_q8_tiled: Ptr,
    quantize_q8: Ptr,
    rope_cache: Ptr,
    attention_partial: Ptr,
    attention_combine: Ptr,
    attention_prefill: Ptr,
    silu_mul: Ptr,
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
            let opts: Vec<CString> = [format!("--gpu-architecture=compute_{}{}", a.compute.0, a.compute.1), "-std=c++14".into()]
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
                embed_q8: get("embed_q8")?,
                rmsnorm: get("rmsnorm")?,
                matmul_q8: get("matmul_q8")?,
                matmul_q8_tiled: get("matmul_q8_tiled")?,
                quantize_q8: get("quantize_q8")?,
                rope_cache: get("rope_cache")?,
                attention_partial: get("attention_partial")?,
                attention_combine: get("attention_combine")?,
                attention_prefill: get("attention_prefill")?,
                silu_mul: get("silu_mul")?,
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
            unsafe { (a.f.cuLaunchKernel)(func, grid.0, grid.1, grid.2, block, 1, 1, shared, std::ptr::null_mut(), params.as_mut_ptr(), std::ptr::null_mut()) },
            "kernel launch",
        )
    }
}

// ---------------------------------------------------------------- model

/// A Q8_0 matrix split into aligned int8 values and f32 scales (one per 32 values).
struct GpuMatrix {
    q: DeviceBuf,
    d: DeviceBuf,
    rows: usize,
    cols: usize,
}

/// Concatenates the rows of several Q8_0 tensors (e.g. Q, K and V) into one GPU matrix.
fn upload_q8(g: &Gguf, names: &[String]) -> Result<GpuMatrix, String> {
    let mut q: Vec<i8> = Vec::new();
    let mut d: Vec<f32> = Vec::new();
    let (mut rows, mut cols) = (0, 0);
    for name in names {
        let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if t.ty != GgmlType::Q8_0 {
            return Err(format!("GPU backend handles Q8_0 weights only for now ({name} is {})", t.ty.name()));
        }
        cols = t.dims[0] as usize;
        rows += t.dims[1] as usize;
        for blk in g.mmap[t.offset..t.offset + t.n_bytes].chunks_exact(34) {
            d.push(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
            q.extend(blk[2..].iter().map(|&b| b as i8));
        }
    }
    if cols % 32 != 0 {
        return Err(format!("unsupported row length {cols}"));
    }
    Ok(GpuMatrix { q: DeviceBuf::upload(&q)?, d: DeviceBuf::upload(&d)?, rows, cols })
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
    qkv: GpuMatrix,
    wo: GpuMatrix,
    ffn_norm: DeviceBuf,
    gate_up: GpuMatrix,
    down: GpuMatrix,
}

pub struct GpuModel {
    pub cfg: Config,
    k: Arc<Kernels>,
    embd: GpuMatrix,
    output: Option<GpuMatrix>, // None = tied to the embedding
    output_norm: DeviceBuf,
    layers: Vec<GpuLayer>,
    pub device: String,
}

impl GpuModel {
    pub fn load(g: &Gguf, cfg: Config) -> Result<GpuModel, String> {
        let arch = g.get("general.architecture").and_then(Value::as_str).unwrap_or("");
        if arch != "llama" {
            return Err(format!("GPU backend: architecture \"{arch}\" not supported yet"));
        }
        if cfg.head_dim > ATT_MAX_HD || ATT_MAX_HD % cfg.head_dim != 0 || cfg.head_dim % 4 != 0 {
            return Err(format!("GPU backend: head size {} not supported (needs a divisor of {ATT_MAX_HD})", cfg.head_dim));
        }
        let a = api()?;
        let k = Arc::new(Kernels::compile()?);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(GpuLayer {
                attn_norm: upload_vec(g, &p("attn_norm"))?,
                qkv: upload_q8(g, &[p("attn_q"), p("attn_k"), p("attn_v")])?,
                wo: upload_q8(g, &[p("attn_output")])?,
                ffn_norm: upload_vec(g, &p("ffn_norm"))?,
                gate_up: upload_q8(g, &[p("ffn_gate"), p("ffn_up")])?,
                down: upload_q8(g, &[p("ffn_down")])?,
            });
        }
        let embd = upload_q8(g, &["token_embd.weight".to_string()])?;
        let output = if g.tensor("output.weight").is_some() { Some(upload_q8(g, &["output.weight".to_string()])?) } else { None };
        let device = format!("{} (sm_{}{}, NVRTC {})", a.device_name, a.compute.0, a.compute.1, std::path::Path::new(&a.nvrtc_path).file_name().and_then(|n| n.to_str()).unwrap_or("?"));
        Ok(GpuModel { output_norm: upload_vec(g, "output_norm.weight")?, cfg, k, embd, output, layers, device })
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

    /// `out[t*rows + r] (+)= W[r] · x[t]` for `n` input rows of `m.cols` floats at `x`.
    fn matmul(&self, m: &GpuMatrix, x: u64, out: u64, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        let k = &self.k;
        if n >= TILED_MIN_BATCH {
            // prompts: quantise the activations once, then int8 tiles with dp4a
            let blocks = n * m.cols / 32;
            k.launch(k.quantize_q8, ((blocks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::I(blocks as i32)])?;
            return k.launch(
                k.matmul_q8_tiled,
                (m.rows.div_ceil(TILE) as u32, n.div_ceil(TILE) as u32),
                256,
                0,
                &[Arg::P(m.q.ptr), Arg::P(m.d.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32)],
            );
        }
        // generation: one warp per weight row, f32 activations
        let rows_per_block = 8; // 256 threads = 8 warps, one row each
        k.launch(
            k.matmul_q8,
            (m.rows.div_ceil(rows_per_block) as u32, n.div_ceil(MATMUL_TB) as u32),
            256,
            0,
            &[Arg::P(m.q.ptr), Arg::P(m.d.ptr), Arg::P(x), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32)],
        )
    }

    /// Context slice per attention block when generating. Up to 256 positions one block per head is
    /// enough (and needs no merge pass); beyond that the context is split so a single token still
    /// spreads over the GPU.
    fn att_chunk(&self, n: usize, span: usize) -> usize {
        if span <= ATT_MAX_CHUNK {
            return ATT_MAX_CHUNK;
        }
        let want_splits = ATT_TARGET_BLOCKS.div_ceil(self.cfg.n_heads * n);
        span.div_ceil(want_splits).clamp(64, ATT_MAX_CHUNK)
    }

    /// Shared memory for the prompt attention kernel, if it fits the default 48 KB.
    fn prefill_shared(&self) -> Option<usize> {
        let ld = self.cfg.head_dim + 1;
        let bytes = ((2 * PREFILL_KT + PREFILL_QT) * ld + PREFILL_QT * (PREFILL_KT + 1)) * 4;
        (bytes <= 48 * 1024).then_some(bytes)
    }

    fn attention(&self, s: &GpuSession, n: usize, kc: u64, vc: u64) -> Result<(), String> {
        let c = &self.cfg;
        let k = &self.k;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        if let (true, Some(shared)) = (n >= TILED_MIN_BATCH, self.prefill_shared()) {
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
            let span = pos0 + m;
            let chunk = self.att_chunk(m, span);
            let nsplit = span.div_ceil(chunk);
            let (qkv, out) = (s.qkv.at(b0 * row), s.att.at(b0 * q_dim * 4));
            k.launch3(
                k.attention_partial,
                (c.n_heads as u32, m as u32, nsplit as u32),
                128,
                0,
                &[Arg::P(qkv), Arg::P(kc), Arg::P(vc), Arg::P(s.att_part.ptr), Arg::P(out), Arg::I(pos0 as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale), Arg::I(nsplit as i32), Arg::I(chunk as i32)],
            )?;
            if nsplit > 1 {
                k.launch(
                    k.attention_combine,
                    (c.n_heads as u32, m as u32),
                    128,
                    0,
                    &[Arg::P(s.att_part.ptr), Arg::P(out), Arg::I(pos0 as i32), Arg::I(c.n_heads as i32), Arg::I(c.head_dim as i32), Arg::I(nsplit as i32), Arg::I(chunk as i32)],
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
        let c = &self.cfg;
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if n > MAX_BATCH {
            return Err(format!("GPU batch of {n} exceeds {MAX_BATCH}"));
        }
        if s.pos + n > s.ctx {
            return Err(format!("context full ({} tokens)", s.ctx));
        }
        let (dim, hd) = (c.dim, c.head_dim);
        let kv_dim = c.n_kv_heads * hd;
        let k = &self.k;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        s.tokens.write(&ids)?;

        // profile buckets match profile::OPS; with profiling off `timed` is a plain call
        self.timed(0, || k.launch(k.embed_q8, (n as u32, 1), 256, 0, &[Arg::P(self.embd.q.ptr), Arg::P(self.embd.d.ptr), Arg::P(s.tokens.ptr), Arg::P(s.x.ptr), Arg::I(dim as i32)]))?;
        let layer_kv = s.ctx * kv_dim * 4;
        for (l, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = (s.k_cache.at(l * layer_kv), s.v_cache.at(l * layer_kv));
            self.timed(1, || k.launch(k.rmsnorm, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(layer.attn_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)]))?;
            self.timed(2, || self.matmul(&layer.qkv, s.xb.ptr, s.qkv.ptr, n, false, s))?;
            self.timed(3, || {
                k.launch(
                    k.rope_cache,
                    (n as u32, 1),
                    256,
                    0,
                    &[Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::I(s.pos as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(hd as i32), Arg::I(c.rope_dim as i32), Arg::F(c.rope_base)],
                )
            })?;
            self.timed(4, || self.attention(s, n, kc, vc))?;
            self.timed(5, || self.matmul(&layer.wo, s.att.ptr, s.x.ptr, n, true, s))?;

            self.timed(1, || k.launch(k.rmsnorm, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(layer.ffn_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)]))?;
            self.timed(6, || {
                self.matmul(&layer.gate_up, s.xb.ptr, s.gu.ptr, n, false, s)?;
                let total = (n * c.ffn_dim) as u32;
                k.launch(k.silu_mul, (total.div_ceil(256), 1), 256, 0, &[Arg::P(s.gu.ptr), Arg::P(s.h.ptr), Arg::I(c.ffn_dim as i32), Arg::I(n as i32)])?;
                self.matmul(&layer.down, s.h.ptr, s.x.ptr, n, true, s)
            })?;
        }

        if want_logits {
            self.timed(7, || {
                let last = s.x.at((n - 1) * dim * 4);
                k.launch(k.rmsnorm, (1, 1), 256, 0, &[Arg::P(last), Arg::P(self.output_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)])?;
                self.matmul(self.output.as_ref().unwrap_or(&self.embd), s.xb.ptr, s.logits_dev.ptr, 1, false, s)?;
                let a = api()?;
                a.check(
                    unsafe { (a.f.cuMemcpyDtoH_v2)(s.logits.as_mut_ptr() as *mut c_void, s.logits_dev.ptr, s.logits.len() * 4) },
                    "copy logits",
                )
            })?;
        }
        s.pos += n;
        Ok(())
    }
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
    att_part: DeviceBuf, // flash-decoding partial results
    logits_dev: DeviceBuf,
    pub logits: Vec<f32>,
}

impl GpuSession {
    pub fn new(cfg: &Config, ctx: usize) -> Result<GpuSession, String> {
        let ctx = ctx.min(cfg.max_ctx);
        let (q_dim, kv_dim) = (cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
        let f = |n: usize| DeviceBuf::alloc(n * 4);
        let widest = cfg.dim.max(q_dim).max(cfg.ffn_dim);
        // split attention runs at most SPLIT_MAX_BATCH tokens at a time, with slices of >= 64 positions
        let slices = SPLIT_MAX_BATCH * ctx.div_ceil(64);
        Ok(GpuSession {
            xq: DeviceBuf::alloc(MAX_BATCH * widest)?,
            xd: f(MAX_BATCH * widest / 32)?,
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
            logits_dev: f(cfg.vocab)?,
            logits: vec![0.0; cfg.vocab],
        })
    }
}
