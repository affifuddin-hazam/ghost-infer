//! Inkling 2 (inkling2_loop) on the GPU, with the experts of the layers that do not fit on the CPU.
//!
//! The same computation as inkling.rs (the CPU reference it is checked against, slot by slot), one token at a
//! time. Dense weights go through cuda.rs's matmuls; the model's own small steps (depth mixes, short
//! convolutions, clips, the relative-bias attention with sinks, the loop controls, NVFP4 experts with SITU) are
//! kernels of this file. Experts live on the GPU for as many looped layers as fit (they run three times per
//! token), the rest go to the CPU (`experts_situ`) through a copy each way.

use super::*;
use crate::inkling::{Cfg, Inkling};

const INK_KERNELS: &str = r#"
extern "C" {
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o)); return v; }
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (w == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < nw ? sh[threadIdx.x] : -3.0e38f;
    if (w == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float h2f(unsigned int h) { float f; asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.f32.f16 %0, t;}" : "=f"(f) : "r"(h)); return f; }
__device__ __forceinline__ unsigned short f2h(float f) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }

// one row of n per block: out = x / rms(x) (* w)
__global__ void ink_rms(const float* x, const float* w, float* out, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* o = out + (size_t)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; s += v * v; }
    s = block_sum(s, sh);
    float inv = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i] * inv; o[i] = w ? v * w[i] : v; }
}

// [rms(a), rms(b)] side by side (the loop controls' gate input); block 0 a, block 1 b
__global__ void ink_rms2(const float* a, const float* b, float* out, int n) {
    __shared__ float sh[32];
    const float* xr = blockIdx.x == 0 ? a : b;
    float* o = out + blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; s += v * v; }
    s = block_sum(s, sh);
    float inv = rsqrtf(s / n + 1e-6f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * inv;
}

// out[r] = W[r] . x for f32 W (rows x cols), one warp per row
__global__ void ink_mv_f32(const float* W, const float* x, float* out, int rows, int cols) {
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (r >= rows) return;
    x += (size_t)blockIdx.y * cols; // token blockIdx.y
    out += (size_t)blockIdx.y * rows;
    const float* wr = W + (size_t)r * cols;
    float s = 0.f;
    for (int i = l; i < cols; i += 32) s += wr[i] * x[i];
    s = warp_sum(s);
    if (l == 0) out[r] = s;
}

// out[r] = W[r] . x for f16 W (the loop controls), one warp per row
__global__ void ink_mv_f16(const unsigned short* W, const float* x, float* out, int rows, int cols) {
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (r >= rows) return;
    const unsigned short* wr = W + (size_t)r * cols;
    float s = 0.f;
    for (int i = l; i < cols; i += 32) s += h2f(wr[i]) * x[i];
    s = warp_sum(s);
    if (l == 0) out[r] = s;
}

// loop controls, fused (blockIdx.y: control j of up to 3, inputs cur_j and msg row j): first
// c64[j][r] = D_j[r] . [rms(cur_j); rms(msg_j)] (f16 D: rank rows of 2 dim), 8 rows a block, each block its own norms
__global__ void ink_ctrl_down(const float* c0, const float* c1, const float* c2, const float* msg, const unsigned short* D0,
                              const unsigned short* D1, const unsigned short* D2, float* c64, int d, int rank, int B) {
    __shared__ float sh[32];
    const int j = blockIdx.y, tok = blockIdx.z;
    const float* cur = (j == 0 ? c0 : j == 1 ? c1 : c2) + (size_t)tok * d;
    const float* m = msg + ((size_t)j * B + tok) * d;
    const unsigned short* D = j == 0 ? D0 : j == 1 ? D1 : D2;
    float a = 0.f, b = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) { float u = cur[i], v = m[i]; a += u * u; b += v * v; }
    a = block_sum(a, sh);
    b = block_sum(b, sh);
    const float ia = rsqrtf(a / d + 1e-6f), ib = rsqrtf(b / d + 1e-6f);
    // a block a row: f16 pairs, all the block's threads along it
    const int r = blockIdx.x;
    const unsigned* wr = reinterpret_cast<const unsigned*>(D + (size_t)r * 2 * d);
    float s = 0.f;
    for (int i = threadIdx.x; i < d / 2; i += blockDim.x) {
        const unsigned u = wr[i], v = wr[d / 2 + i];
        s += h2f(u & 0xffff) * (cur[2 * i] * ia) + h2f(u >> 16) * (cur[2 * i + 1] * ia);
        s += h2f(v & 0xffff) * (m[2 * i] * ib) + h2f(v >> 16) * (m[2 * i + 1] * ib);
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) c64[((size_t)j * B + tok) * rank + r] = s;
}

// then out_j = (1 - tanh(F_j . c64_j)) * cur_j + tanh(G_j . c64_j) * msg_j, a warp a row (f16 G, F: dim rows of rank)
__global__ void ink_ctrl_up(const float* c0, const float* c1, const float* c2, const float* msg, const unsigned short* G0,
                            const unsigned short* G1, const unsigned short* G2, const unsigned short* F0, const unsigned short* F1,
                            const unsigned short* F2, const float* c64, float* out, int d, int rank, int B) {
    __shared__ float cs[256];
    const int j = blockIdx.y, tok = blockIdx.z;
    const size_t row = (size_t)j * B + tok;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) cs[i] = c64[row * rank + i];
    __syncthreads();
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= d) return;
    const unsigned* g = reinterpret_cast<const unsigned*>((j == 0 ? G0 : j == 1 ? G1 : G2) + (size_t)r * rank);
    const unsigned* f = reinterpret_cast<const unsigned*>((j == 0 ? F0 : j == 1 ? F1 : F2) + (size_t)r * rank);
    float gs = 0.f, fs = 0.f;
    for (int i = lane; i < rank / 2; i += 32) {
        const unsigned a = g[i], b = f[i];
        gs += h2f(a & 0xffff) * cs[2 * i] + h2f(a >> 16) * cs[2 * i + 1];
        fs += h2f(b & 0xffff) * cs[2 * i] + h2f(b >> 16) * cs[2 * i + 1];
    }
    gs = warp_sum(gs);
    fs = warp_sum(fs);
    if (lane == 0) {
        const float c = (j == 0 ? c0 : j == 1 ? c1 : c2)[(size_t)tok * d + r];
        out[row * d + r] = c - tanhf(fs) * c + tanhf(gs) * msg[row * d + r];
    }
}

// (1 - tanh(fp)) * cur + tanh(gp) * msg
__global__ void ink_control(const float* cur, const float* msg, const float* gp, const float* fp, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = cur[i];
    out[i] = c - tanhf(fp[i]) * c + tanhf(gp[i]) * msg[i];
}

__global__ void ink_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void ink_copy(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}

// depth attention: for query j (block j), softmax over the ns RMS-normed states of q_j . s / sqrt(dim), and the
// mix of the normed states
__global__ void ink_mix(const float* s0, const float* s1, const float* s2, const float* s3, const float* s4, const float* s5,
                        const float* s6, const float* s7, int ns, const float* query, float* out, int dim, int B) {
    __shared__ const float* st[8];
    __shared__ float inv[8], score[8];
    const int tok = blockIdx.y; // states are B rows each; out query-major: row (query, token)
    if (threadIdx.x < 8) {
        const float* all[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
        st[threadIdx.x] = all[threadIdx.x] ? all[threadIdx.x] + (size_t)tok * dim : 0;
    }
    __syncthreads();
    const float* q = query + (size_t)blockIdx.x * dim;
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (wp < ns) {
        const float* v = st[wp];
        float ss = 0.f, d = 0.f;
        for (int i = lane; i < dim; i += 32) { float a = v[i]; ss += a * a; d += a * q[i]; }
        ss = warp_sum(ss);
        d = warp_sum(d);
        if (lane == 0) { float iv = rsqrtf(ss / dim + 1e-6f); inv[wp] = iv; score[wp] = d * iv * rsqrtf((float)dim); }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -3.0e38f;
        for (int s = 0; s < ns; s++) m = fmaxf(m, score[s]);
        float t = 0.f;
        for (int s = 0; s < ns; s++) { score[s] = __expf(score[s] - m); t += score[s]; }
        for (int s = 0; s < ns; s++) score[s] = score[s] / t * inv[s];
    }
    __syncthreads();
    float* o = out + ((size_t)blockIdx.x * B + tok) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < ns; s++) v += score[s] * st[s][i];
        o[i] = v;
    }
}

// causal short convolution with a residual over B tokens in order; state: per channel the last conv-1 inputs,
// oldest first; dyn (when given): per token, heads x conv taps
__global__ void ink_conv(const float* cur, const float* w, const float* dyn, float* state, float* out, int n, int conv, int per_head, float div, int B) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n) return;
    float* hist = state + (size_t)ch * (conv - 1);
    float x[8];
    for (int k = 0; k < conv - 1; k++) x[k] = hist[k];
    const int heads = n / per_head;
    for (int b = 0; b < B; b++) {
        float c = cur[(size_t)b * n + ch];
        x[conv - 1] = c;
        float y = 0.f;
        for (int k = 0; k < conv; k++) y += w[ch * conv + k] * x[k];
        if (dyn) {
            const float* dy = dyn + (size_t)b * heads * conv;
            int h = ch / per_head;
            for (int lag = 0; lag < conv; lag++) y += dy[h * conv + lag] * x[conv - 1 - lag];
        }
        for (int k = 0; k < conv - 1; k++) x[k] = x[k + 1];
        out[(size_t)b * n + ch] = (c + y) * div;
    }
    for (int k = 0; k < conv - 1; k++) hist[k] = x[k];
}
// rows of n (one warp each): times the per-channel scale, down to an L2 norm of at most thr, then times tau
// (1 + alpha log(max(1, (pos + 1) / floor)), the full-attention layers' log scale) when use_tau
__global__ void ink_clip(float* x, const float* scale, int rows, int n, float thr, const int* posp, int use_tau, float alpha, float floor_, int per_tok) {
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (r >= rows) return;
    float* xr = x + (size_t)r * n;
    const float* sr = scale ? scale + (size_t)(r % per_tok) * n : 0; // the scale is per head, the same every token
    float ss = 0.f;
    for (int i = l; i < n; i += 32) { float v = xr[i] * (sr ? sr[i] : 1.f); ss += v * v; }
    ss = warp_sum(ss);
    float f = fminf(fmaxf(thr / (sqrtf(ss) + 1e-6f), 0.f), 1.f);
    if (use_tau) f *= 1.f + alpha * logf(fmaxf(1.f, (float)(*posp + r / per_tok + 1) / floor_));
    for (int i = l; i < n; i += 32) xr[i] = xr[i] * (sr ? sr[i] : 1.f) * f;
}

// this token's K and V (as f16) into the slot's cache row pos % cap
__global__ void ink_store_kv(const float* k, const float* v, unsigned short* kc, unsigned short* vc, const int* posp, int cap, int kvdim, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvdim * B) return;
    const int b = i / kvdim, e = i - b * kvdim;
    size_t row = (size_t)((*posp + b) % cap) * kvdim;
    kc[row + e] = f2h(k[i]);
    vc[row + e] = f2h(v[i]);
}

// attention of query head h (block h, 128 threads, head size 128): scores q.k / sqrt(dh) plus the relative bias of
// the distance (zero from `extent` on) minus the head's sink logit; keys within `window` (0: all); a softmax with
// an extra zero logit (the sink); the values' mix times sigmoid(gate)
__global__ void ink_attn(const float* q, const float* r, const float* sinks, const float* rel_bias, const unsigned short* kc, const unsigned short* vc,
                         const float* gate, float* out, float* scr, const int* posp, int cap, int window, int extent, int n_rel,
                         int kvdim, int group, int max_n, float scale, int b0) {
    __shared__ float sh[32];
    __shared__ float qs[128], rs[32];
    const int h = blockIdx.x, dh = 128, g = h / group, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nh = gridDim.x, b = b0 + blockIdx.y; // token b of the batch, at position *posp + b
    const int pos = *posp + b;
    q += (size_t)b * nh * dh;
    r += (size_t)b * nh * n_rel;
    sinks += (size_t)b * nh;
    gate += (size_t)b * nh * dh;
    out += (size_t)b * nh * dh;
    scr += (size_t)blockIdx.y * nh * max_n;
    int n = pos + 1;
    if (window > 0 && n > window) n = window;
    if (n > cap) n = cap;
    qs[tid] = q[h * dh + tid];
    if (tid < n_rel) rs[tid] = r[h * n_rel + tid];
    __syncthreads();
    const float t = sinks[h];
    float* sc = scr + (size_t)h * max_n;
    for (int dist = warp; dist < n; dist += 4) {
        int p = pos - dist;
        const unsigned short* kr = kc + (size_t)(p % cap) * kvdim + g * dh;
        float s = 0.f;
        for (int i = lane; i < dh; i += 32) s += qs[i] * h2f(kr[i]);
        s = warp_sum(s);
        if (lane == 0) {
            float b = 0.f;
            if (dist < extent) for (int j = 0; j < n_rel; j++) b += rel_bias[dist * n_rel + j] * rs[j];
            sc[dist] = s * scale + b - t;
        }
    }
    __syncthreads();
    float m = 0.f; // the sink's logit
    for (int i = tid; i < n; i += blockDim.x) m = fmaxf(m, sc[i]);
    m = block_max(m, sh);
    float tot = 0.f;
    for (int i = tid; i < n; i += blockDim.x) { float e = __expf(sc[i] - m); sc[i] = e; tot += e; }
    tot = block_sum(tot, sh) + __expf(-m);
    __syncthreads();
    float acc = 0.f;
    for (int dist = 0; dist < n; dist++) {
        int p = pos - dist;
        acc += sc[dist] * h2f(vc[(size_t)(p % cap) * kvdim + g * dh + tid]);
    }
    out[h * dh + tid] = acc / tot * sigm(gate[h * dh + tid]);
}

// sigmoid routing: the k best experts by sigmoid(logit) + bias, weighted by their sigmoid renormalised, times scale
__global__ void ink_topk(const float* logits, const float* bias, int n_exp, int k, float wscale, int* picks, float* weights) {
    __shared__ float sel[256], pr[256], top[32];
    logits += (size_t)blockIdx.x * n_exp; // token blockIdx.x
    picks += (size_t)blockIdx.x * k;
    weights += (size_t)blockIdx.x * k;
    int i = threadIdx.x;
    if (i < n_exp) { float p = sigm(logits[i]); pr[i] = p; sel[i] = p + bias[i]; }
    __syncthreads();
    if (i < n_exp) {
        int rank = 0;
        for (int j = 0; j < n_exp; j++) rank += sel[j] > sel[i] || (sel[j] == sel[i] && j < i);
        if (rank < k) { picks[rank] = i; top[rank] = pr[i]; }
    }
    __syncthreads();
    if (i == 0) {
        float t = 1e-20f;
        for (int j = 0; j < k; j++) t += top[j];
        for (int j = 0; j < k; j++) weights[j] = top[j] / t * wscale;
    }
}

// rows of 32 values to int8 and a scale (one warp per 32)
__global__ void ink_q8(const float* x, signed char* xq, float* xd, int n) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (b * 32 >= n) return;
    float v = x[b * 32 + l];
    float a = warp_max(fabsf(v));
    float d = a / 127.f;
    xq[b * 32 + l] = (signed char)__float2int_rn(d > 0.f ? v / d : 0.f);
    if (l == 0) xd[b] = d;
}

// Q8 rows (cuda.rs's layout: int8 values, then an f32 scale per 32) times int8 activations: 8 lanes a row,
// 16-byte loads, the activations in shared memory
// ink_mv_q8 for B tokens (rows of xq / xd; out: a row of `rows` per token): each weight read once for up to 8
// tokens (blockIdx.y picks the 8); per token the same arithmetic as ink_mv_q8
__global__ void ink_mm_q8(const signed char* Wq, const float* Wd, const signed char* xq, const float* xd, float* out, int rows, int cols, int B) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int t0 = blockIdx.y * 8, nt = min(8, B - t0);
    if (r >= rows) return;
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; t++) s[t] = 0.f;
#pragma unroll 2
    for (int i = lane * 16; i < cols; i += 512) {
        const int4 wv = *reinterpret_cast<const int4*>(Wq + (size_t)r * cols + i);
        const float dw = Wd[(size_t)r * (cols >> 5) + (i >> 5)];
#pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t < nt) {
                const int4 xv = *reinterpret_cast<const int4*>(xq + (size_t)(t0 + t) * cols + i);
                const int a = __dp4a(wv.x, xv.x, __dp4a(wv.y, xv.y, __dp4a(wv.z, xv.z, __dp4a(wv.w, xv.w, 0))));
                s[t] += dw * xd[(size_t)(t0 + t) * (cols >> 5) + (i >> 5)] * (float)a;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; t++) {
        const float v = warp_sum(s[t]);
        if (lane == 0 && t < nt) out[(size_t)(t0 + t) * rows + r] = v;
    }
}

// Q8 matrices stacked by rows that read the same input, for B tokens (blockIdx.y: 8 of them): rows [e_{s-1}, e_s)
// go to o_s (a row of e_s - e_{s-1} per token); per token the same arithmetic as ink_mv_q8
__global__ void ink_mv_q8s(const signed char* Wq, const float* Wd, const signed char* xq, const float* xd, int rows, int cols, int B,
                           float* o0, float* o1, float* o2, float* o3, float* o4, float* o5, int e0, int e1, int e2, int e3, int e4, int e5) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int t0 = blockIdx.y * 8, nt = min(8, B - t0);
    if (r >= rows) return;
    float* o;
    int st, n;
    if (r < e0) { o = o0; st = 0; n = e0; }
    else if (r < e1) { o = o1; st = e0; n = e1 - e0; }
    else if (r < e2) { o = o2; st = e1; n = e2 - e1; }
    else if (r < e3) { o = o3; st = e2; n = e3 - e2; }
    else if (r < e4) { o = o4; st = e3; n = e4 - e3; }
    else { o = o5; st = e4; n = e5 - e4; }
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; t++) s[t] = 0.f;
#pragma unroll 2
    for (int i = lane * 16; i < cols; i += 512) {
        const int4 wv = *reinterpret_cast<const int4*>(Wq + (size_t)r * cols + i);
        const float dw = Wd[(size_t)r * (cols >> 5) + (i >> 5)];
#pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t < nt) {
                const int4 xv = *reinterpret_cast<const int4*>(xq + (size_t)(t0 + t) * cols + i);
                const int a = __dp4a(wv.x, xv.x, __dp4a(wv.y, xv.y, __dp4a(wv.z, xv.z, __dp4a(wv.w, xv.w, 0))));
                s[t] += dw * xd[(size_t)(t0 + t) * (cols >> 5) + (i >> 5)] * (float)a;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; t++) {
        const float v = warp_sum(s[t]);
        if (lane == 0 && t < nt) o[(size_t)(t0 + t) * n + (r - st)] = v;
    }
}

#define MV_ROWS 1
__global__ void ink_mv_q8(const signed char* Wq, const float* Wd, const signed char* xq, const float* xd, float* out, int rows, int cols) {
    // a warp a row (MV_ROWS rows a warp), each lane 16 consecutive values a step: every load a full 512-byte line
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * MV_ROWS;
    float s[MV_ROWS];
#pragma unroll
    for (int k = 0; k < MV_ROWS; k++) s[k] = 0.f;
#pragma unroll 2
    for (int i = lane * 16; i < cols; i += 512) {
        const int4 xv = *reinterpret_cast<const int4*>(xq + i);
        const float dx = xd[i >> 5];
#pragma unroll
        for (int k = 0; k < MV_ROWS; k++) {
            if (r0 + k < rows) {
                const int4 wv = *reinterpret_cast<const int4*>(Wq + (size_t)(r0 + k) * cols + i);
                const int a = __dp4a(wv.x, xv.x, __dp4a(wv.y, xv.y, __dp4a(wv.z, xv.z, __dp4a(wv.w, xv.w, 0))));
                s[k] += Wd[(size_t)(r0 + k) * (cols >> 5) + (i >> 5)] * dx * (float)a;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < MV_ROWS; k++) {
        const float v = warp_sum(s[k]);
        if (lane == 0 && r0 + k < rows) out[r0 + k] = v;
    }
}

__device__ __forceinline__ float ue4m3(unsigned x) {
    // ggml_ue4m3_to_fp32 halved: (1 + m/8) 2^(e-8) as bits; exponent 0: m 2^-10; 0 and 0x7f are 0
    if (x == 0 || x == 0x7f) return 0.f;
    unsigned e = (x >> 3) & 15, m = x & 7;
    return e ? __int_as_float(((e + 119) << 23) | (m << 20)) : (float)m * 0.0009765625f;
}
// eight 4-bit codes (a 32-bit word) through the 16-entry FP4 table held in registers: .x the values of the low
// nibbles of its four bytes, .y of the high ones (llama.cpp's get_int_from_table_16)
__device__ __forceinline__ int2 fp4_lookup(int q4) {
    const unsigned t0 = 0x03020100u, t1 = 0x0c080604u, t2 = 0xfdfeff00u, t3 = 0xf4f8fafcu; // 0 1 2 3 4 6 8 12 0 -1 -2 -3 -4 -6 -8 -12
    unsigned tmp[2];
    const unsigned sel = 0x32103210u | (((unsigned)q4 & 0x88888888u) >> 1);
    for (int i = 0; i < 2; i++) {
        const unsigned sh = 16 * i;
        const unsigned lo = __byte_perm(t0, t1, (unsigned)q4 >> sh);
        const unsigned hi = __byte_perm(t2, t3, (unsigned)q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
// one split-NVFP4 row's dot with int8 activations (in shared memory): lanes over 32-value pieces, each one 16-byte
// load of codes (two sub-blocks: per 8 bytes, low nibbles are values 0..7, high ones 8..15) and two scale bytes
// one 32-value piece hb of a split-NVFP4 row against int8 activations (before their per-32 scale)
__device__ __forceinline__ float nvfp4s_piece(const unsigned char* q, const unsigned char* sc, const signed char* xq, int hb) {
    const int4 w = reinterpret_cast<const int4*>(q)[hb];
    const unsigned short ss = reinterpret_cast<const unsigned short*>(sc)[hb];
    // 16-byte reads of the activations: lanes 32 bytes apart reading 4 bytes at a time hit the same banks 8 ways
    const int4 x0 = reinterpret_cast<const int4*>(xq + hb * 32)[0], x1 = reinterpret_cast<const int4*>(xq + hb * 32)[1];
    int2 a = fp4_lookup(w.x), b = fp4_lookup(w.y);
    int acc = __dp4a(a.x, x0.x, 0);
    acc = __dp4a(b.x, x0.y, acc);
    acc = __dp4a(a.y, x0.z, acc);
    acc = __dp4a(b.y, x0.w, acc);
    float part = ue4m3(ss & 0xff) * (float)acc;
    a = fp4_lookup(w.z), b = fp4_lookup(w.w);
    acc = __dp4a(a.x, x1.x, 0);
    acc = __dp4a(b.x, x1.y, acc);
    acc = __dp4a(a.y, x1.z, acc);
    acc = __dp4a(b.y, x1.w, acc);
    return part + ue4m3(ss >> 8) * (float)acc;
}

__device__ __forceinline__ float nvfp4s_row_dot(const unsigned char* q, const unsigned char* sc, const signed char* xq, const float* xd, int cols, int lane) {
    float s = 0.f;
    const int pieces = cols / 32;
    for (int hb = lane; hb < pieces; hb += 32) {
        const int4 w = reinterpret_cast<const int4*>(q)[hb];
        const unsigned short ss = reinterpret_cast<const unsigned short*>(sc)[hb];
        const int* xi = reinterpret_cast<const int*>(xq + hb * 32);
        int2 a = fp4_lookup(w.x), b = fp4_lookup(w.y);
        int acc = __dp4a(a.x, xi[0], 0);
        acc = __dp4a(b.x, xi[1], acc);
        acc = __dp4a(a.y, xi[2], acc);
        acc = __dp4a(b.y, xi[3], acc);
        float part = ue4m3(ss & 0xff) * (float)acc;
        a = fp4_lookup(w.z), b = fp4_lookup(w.w);
        acc = __dp4a(a.x, xi[4], 0);
        acc = __dp4a(b.x, xi[5], acc);
        acc = __dp4a(a.y, xi[6], acc);
        acc = __dp4a(b.y, xi[7], acc);
        part += ue4m3(ss >> 8) * (float)acc;
        s += part * xd[hb];
    }
    return s;
}

// gate_up of the k picked experts: block (row group, pick); out[pick][row] = scale[e] * W_e[row] . x
__global__ void ink_moe_gu(const unsigned char* W, long long ebytes, const int* picks, const float* gscale, const signed char* xq,
                           const float* xd, float* out, int rows, int cols, const int* slot) {
    __shared__ __align__(16) signed char xs[4096];
    __shared__ float ds[128];
    // token blockIdx.z of a prompt batch: its picks, activations and rows of out
    picks += (size_t)blockIdx.z * gridDim.y;
    xq += (size_t)blockIdx.z * cols;
    xd += (size_t)blockIdx.z * (cols / 32);
    out += (size_t)blockIdx.z * gridDim.y * rows;
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) reinterpret_cast<int*>(xs)[i] = reinterpret_cast<const int*>(xq)[i];
    for (int i = threadIdx.x; i < cols / 32; i += blockDim.x) ds[i] = xd[i];
    __syncthreads();
    const int r = blockIdx.x * (blockDim.x >> 3) + (threadIdx.x >> 3), sub = threadIdx.x & 7;
    const int e0 = picks[blockIdx.y], e = slot[e0];
    const bool ok = r < rows && e >= 0;
    float s = 0.f;
    if (ok) {
        const unsigned char* base = W + e * ebytes;
        const unsigned char *q = base + (size_t)r * (cols / 2), *sc = base + (size_t)rows * (cols / 2) + (size_t)r * (cols / 16);
#pragma unroll 3
        for (int hb = sub; hb < cols / 32; hb += 8) s += ds[hb] * nvfp4s_piece(q, sc, xs, hb);
    }
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);
    if (r < rows && sub == 0) out[(size_t)blockIdx.y * rows + r] = ok ? s * gscale[e0] : 0.f;
}

// gate_up of the k picked experts in IQ3_XXS (the file's blocks: f16 d, 64 grid index bytes, 8 x u32 of four
// 7-bit sign codes and a 4-bit scale): a warp a row, a lane 8 values, so each step of the warp is one 256-value
// block; grid entries from shared memory, signs flipped with xor/sub per byte
__global__ void ink_moe_gu_iq3(const unsigned char* W, long long ebytes, const int* picks, const float* gscale, const signed char* xq,
                               const float* xd, float* out, int rows, int cols, const unsigned int* grid, const int* slot) {
    __shared__ __align__(16) signed char xs[4096];
    __shared__ float ds[128];
    __shared__ unsigned int gs[256];
    // token blockIdx.z of a prompt batch: its picks, activations and rows of out
    picks += (size_t)blockIdx.z * gridDim.y;
    xq += (size_t)blockIdx.z * cols;
    xd += (size_t)blockIdx.z * (cols / 32);
    out += (size_t)blockIdx.z * gridDim.y * rows;
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) reinterpret_cast<int*>(xs)[i] = reinterpret_cast<const int*>(xq)[i];
    for (int i = threadIdx.x; i < cols / 32; i += blockDim.x) ds[i] = xd[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) gs[i] = grid[i];
    __syncthreads();
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int e0 = picks[blockIdx.y], e = slot[e0];
    if (e < 0) {
        if (lane == 0) out[(size_t)blockIdx.y * rows + r] = 0.f;
        return;
    }
    const int nb = cols / 256;
    const unsigned char* row = W + e * ebytes + (size_t)r * nb * 98;
    const int ib = lane >> 2, sub = lane & 3;
    float s = 0.f;
    for (int b = 0; b < nb; b++) {
        const unsigned char* blk = row + b * 98;
        const unsigned short* h = reinterpret_cast<const unsigned short*>(blk);
        const float d = h2f(h[0]);
        const unsigned short qq = h[1 + 4 * ib + sub];
        const unsigned int aux = (unsigned int)h[33 + 2 * ib] | ((unsigned int)h[34 + 2 * ib] << 16);
        unsigned int sg = (aux >> (7 * sub)) & 127;
        sg |= (__popc(sg) & 1) << 7;
        const unsigned int m0 = ((sg & 15) * 0x00204081u & 0x01010101u) * 0xffu;
        const unsigned int m1 = ((sg >> 4) * 0x00204081u & 0x01010101u) * 0xffu;
        const int g0 = (int)__vsub4(gs[qq & 0xff] ^ m0, m0);
        const int g1 = (int)__vsub4(gs[qq >> 8] ^ m1, m1);
        const int* xi = reinterpret_cast<const int*>(xs + b * 256 + lane * 8);
        const int dot = __dp4a(g1, xi[1], __dp4a(g0, xi[0], 0));
        s += d * (0.5f + (float)(aux >> 28)) * 0.5f * ds[b * 8 + ib] * (float)dot;
    }
    s = warp_sum(s);
    if (lane == 0) out[(size_t)blockIdx.y * rows + r] = s * gscale[e0];
}

// a prompt batch's (token, pick) pairs bucketed by expert: cnt[e], off[e] (where e's pairs start in list) and
// list (pair t * k + j); one block
__global__ void ink_group(const int* picks, int np, int n_exp, int* cnt, int* off, int* list) {
    __shared__ int c[256], o[257];
    for (int i = threadIdx.x; i < n_exp; i += blockDim.x) c[i] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < np; p += blockDim.x) atomicAdd(&c[picks[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        o[0] = 0;
        for (int i = 0; i < n_exp; i++) o[i + 1] = o[i] + c[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_exp; i += blockDim.x) { cnt[i] = c[i]; off[i] = o[i]; c[i] = o[i]; }
    __syncthreads();
    for (int p = threadIdx.x; p < np; p += blockDim.x) list[atomicAdd(&c[picks[p]], 1)] = p;
}

// ink_moe_gu for a prompt batch grouped by expert: block (row group, expert u) takes every pair that picked u, 8
// at a time, each weight piece read once for the 8; per pair the same arithmetic as ink_moe_gu (out: a pair's
// [gate; up] row at pair * rows)
__global__ void ink_moe_gu_g(const unsigned char* W, long long ebytes, const int* cnt, const int* off, const int* list, int k,
                             const float* gscale, const signed char* xq, const float* xd, float* out, int rows, int cols, const int* slot) {
    __shared__ __align__(16) signed char xs[8 * 2304];
    __shared__ float ds[8 * 72];
    const int u = blockIdx.y, n = cnt[u];
    if (n == 0) return;
    const int e = slot[u], start = off[u];
    const int r = blockIdx.x * (blockDim.x >> 3) + (threadIdx.x >> 3), sub = threadIdx.x & 7;
    const unsigned char* base = W + (e < 0 ? 0 : e) * ebytes;
    const unsigned char *q = base + (size_t)r * (cols / 2), *sc = base + (size_t)rows * (cols / 2) + (size_t)r * (cols / 16);
    for (int c0 = 0; c0 < n; c0 += 8) {
        const int nt = min(8, n - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * cols / 16; i += blockDim.x) {
            const int t = i / (cols / 16), w = i - t * (cols / 16), tok = list[start + c0 + t] / k;
            reinterpret_cast<int4*>(xs + t * cols)[w] = reinterpret_cast<const int4*>(xq + (size_t)tok * cols)[w];
        }
        for (int i = threadIdx.x; i < nt * cols / 32; i += blockDim.x) {
            const int t = i / (cols / 32), w = i - t * (cols / 32), tok = list[start + c0 + t] / k;
            ds[t * (cols / 32) + w] = xd[(size_t)tok * (cols / 32) + w];
        }
        __syncthreads();
        float s[8];
#pragma unroll
        for (int t = 0; t < 8; t++) s[t] = 0.f;
        if (r < rows && e >= 0) {
            for (int hb = sub; hb < cols / 32; hb += 8) {
                const int4 w = reinterpret_cast<const int4*>(q)[hb];
                const unsigned short ss = reinterpret_cast<const unsigned short*>(sc)[hb];
                const float s0 = ue4m3(ss & 0xff), s1 = ue4m3(ss >> 8);
                const int2 a0 = fp4_lookup(w.x), b0 = fp4_lookup(w.y), a1 = fp4_lookup(w.z), b1 = fp4_lookup(w.w);
#pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (t < nt) {
                        const int4 x0 = reinterpret_cast<const int4*>(xs + t * cols + hb * 32)[0], x1 = reinterpret_cast<const int4*>(xs + t * cols + hb * 32)[1];
                        int acc = __dp4a(a0.x, x0.x, 0);
                        acc = __dp4a(b0.x, x0.y, acc);
                        acc = __dp4a(a0.y, x0.z, acc);
                        acc = __dp4a(b0.y, x0.w, acc);
                        float part = s0 * (float)acc;
                        acc = __dp4a(a1.x, x1.x, 0);
                        acc = __dp4a(b1.x, x1.y, acc);
                        acc = __dp4a(a1.y, x1.z, acc);
                        acc = __dp4a(b1.y, x1.w, acc);
                        s[t] += ds[t * (cols / 32) + hb] * (part + s1 * (float)acc);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) {
            float v = s[t];
            v += __shfl_xor_sync(0xffffffff, v, 4);
            v += __shfl_xor_sync(0xffffffff, v, 2);
            v += __shfl_xor_sync(0xffffffff, v, 1);
            if (r < rows && sub == 0 && t < nt) out[(size_t)list[start + c0 + t] * rows + r] = e >= 0 ? v * gscale[u] : 0.f;
        }
    }
}

// SITU of each pick's [gate; up] (ff each) into int8 per 32 for the downs
__global__ void ink_situ_q8(const float* gu, signed char* hq, float* hd, int ff, int k, float beta, float lb) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (b * 32 >= k * ff) return;
    int u = b * 32 + l, e = u / ff, i = u % ff;
    float g = gu[(size_t)e * 2 * ff + i], v = gu[(size_t)e * 2 * ff + ff + i];
    float h = beta * tanhf(g / beta) * sigm(g) * lb * tanhf(v / lb);
    float a = warp_max(fabsf(h));
    float d = a / 127.f;
    hq[u] = (signed char)__float2int_rn(d > 0.f ? h / d : 0.f);
    if (l == 0) hd[b] = d;
}

// the downs of the k picks summed: out[row] = sum_p w_p * scale[e_p] * D_e[row] . h_p
#define DOWN_ROWS 1
__global__ void ink_moe_down(const unsigned char* W, long long ebytes, const int* picks, const float* weights, const float* dscale,
                             const signed char* hq, const float* hd, float* out, int rows, int cols, int k, const int* slot) {
    __shared__ __align__(16) signed char hs[8 * 1152];
    __shared__ float ds[8 * 36];
    __shared__ int se[8];
    __shared__ float sw[8];
    // token blockIdx.z of a prompt batch
    picks += (size_t)blockIdx.z * k;
    weights += (size_t)blockIdx.z * k;
    hq += (size_t)blockIdx.z * k * cols;
    hd += (size_t)blockIdx.z * k * (cols / 32);
    out += (size_t)blockIdx.z * rows;
    for (int i = threadIdx.x; i < k * cols / 4; i += blockDim.x) reinterpret_cast<int*>(hs)[i] = reinterpret_cast<const int*>(hq)[i];
    for (int i = threadIdx.x; i < k * cols / 32; i += blockDim.x) ds[i] = hd[i];
    if (threadIdx.x < k) {
        const int e0 = picks[threadIdx.x];
        se[threadIdx.x] = slot[e0];
        sw[threadIdx.x] = weights[threadIdx.x] * dscale[e0];
    }
    __syncthreads();
    // DOWN_ROWS rows a warp, so the activations copied above serve more rows
    const int lane = threadIdx.x & 31, pieces = cols / 32;
    for (int i = 0; i < DOWN_ROWS; i++) {
        const int r = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * DOWN_ROWS + i;
        if (r >= rows) return;
        float acc = 0.f;
#pragma unroll 3
        for (int j = lane; j < k * pieces; j += 32) {
            const int p = j / pieces, hb = j - p * pieces, e = se[p];
            if (e < 0) continue;
            const unsigned char* base = W + e * ebytes;
            acc += sw[p] * ds[j] * nvfp4s_piece(base + (size_t)r * (cols / 2), base + (size_t)rows * (cols / 2) + (size_t)r * (cols / 16), hs + p * cols, hb);
        }
        acc = warp_sum(acc);
        if (lane == 0) out[r] = acc;
    }
}

// Q5_K rows (176-byte blocks of 256: f16 d, f16 dmin, 12 scale bytes, 32 high-bit bytes, 128 nibble bytes) times
// an f32 vector, one warp per row; each lane takes 4 + 4 values of each 64 (ggml's dequantize_row_q5_K)
__device__ __forceinline__ void q5k_scale_min(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__global__ void ink_mv_q5k(const unsigned char* W, const float* x, float* out, int rows, int cols) {
    __shared__ float xs[4096];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int nb = cols / 256;
    const unsigned char* row = W + (size_t)r * nb * 176;
    const int g = lane >> 3, o = (lane & 7) * 4; // the 64-group and the offset in it
    float s = 0.f;
    for (int b = 0; b < nb; b++) {
        const unsigned char* blk = row + b * 176;
        const unsigned dd = *reinterpret_cast<const unsigned*>(blk); // blocks are 176 bytes: 4-byte aligned
        const float d = h2f(dd & 0xffff), dmin = h2f(dd >> 16);
        const unsigned char* sc = blk + 4;
        int s1, m1, s2, m2;
        q5k_scale_min(2 * g, sc, &s1, &m1);
        q5k_scale_min(2 * g + 1, sc, &s2, &m2);
        const float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
        const unsigned qh4 = *reinterpret_cast<const unsigned*>(blk + 16 + o);
        const unsigned ql4 = *reinterpret_cast<const unsigned*>(blk + 48 + 32 * g + o);
        const float* xb = xs + b * 256 + 64 * g + o;
        float lo_sum = 0.f, hi_sum = 0.f, xl = 0.f, xh = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned q = (ql4 >> (8 * j)) & 0xff, h = (qh4 >> (8 * j)) & 0xff;
            const float lo = (float)((q & 0xF) | (((h >> (2 * g)) & 1) << 4));
            const float hi = (float)((q >> 4) | (((h >> (2 * g + 1)) & 1) << 4));
            lo_sum += lo * xb[j]; xl += xb[j];
            hi_sum += hi * xb[j + 32]; xh += xb[j + 32];
        }
        s += d1 * lo_sum - mm1 * xl + d2 * hi_sum - mm2 * xh;
    }
    s = warp_sum(s);
    if (lane == 0) out[r] = s;
}

// Q5_K rows times int8 activations (per-32 scales): 8 lanes a row (4 rows a warp), a lane 16 bytes of a block's
// 128 nibble bytes a step (32 values: 16 of the group's low sub-block and 16 of its high one) with their 16 high-bit
// bytes, dp4a against the activations in shared memory
__global__ void ink_mv_q5k8(const unsigned char* W, const signed char* xq, const float* xd, float* out, int rows, int cols) {
    __shared__ __align__(16) signed char xs[4096];
    __shared__ float ds[128];
    for (int i = threadIdx.x; i < cols / 16; i += blockDim.x) reinterpret_cast<int4*>(xs)[i] = reinterpret_cast<const int4*>(xq)[i];
    for (int i = threadIdx.x; i < cols / 32; i += blockDim.x) ds[i] = xd[i];
    __syncthreads();
    const int r = blockIdx.x * (blockDim.x >> 3) + (threadIdx.x >> 3), c = threadIdx.x & 7;
    const int g = c >> 1, i0 = (c & 1) * 16; // the 64-value group and the offset in its 32 bytes
    const int nb = cols / 256;
    float s = 0.f;
    if (r < rows) {
        const unsigned char* row = W + (size_t)r * nb * 176;
#pragma unroll 3
        for (int b = 0; b < nb; b++) {
            const unsigned char* blk = row + b * 176;
            // 176 = 11 x 16: every block (and row) starts 16-byte aligned
            const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            const uint4 q4 = *reinterpret_cast<const uint4*>(blk + 48 + 32 * g + i0);
            const uint4 h4 = *reinterpret_cast<const uint4*>(blk + 16 + i0);
            const float d = h2f(hdr.x & 0xffff), dmin = h2f(hdr.x >> 16);
            unsigned char scb[12];
            *reinterpret_cast<unsigned*>(scb) = hdr.y;
            *reinterpret_cast<unsigned*>(scb + 4) = hdr.z;
            *reinterpret_cast<unsigned*>(scb + 8) = hdr.w;
            int s1, m1, s2, m2;
            q5k_scale_min(2 * g, scb, &s1, &m1);
            q5k_scale_min(2 * g + 1, scb, &s2, &m2);
            const unsigned ql[4] = {q4.x, q4.y, q4.z, q4.w}, qh[4] = {h4.x, h4.y, h4.z, h4.w};
            const int* xl = reinterpret_cast<const int*>(xs + b * 256 + 64 * g + i0);
            const int* xh = reinterpret_cast<const int*>(xs + b * 256 + 64 * g + 32 + i0);
            int sl = 0, sh = 0, nl = 0, nh = 0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const unsigned q = ql[k], h = qh[k];
                const int lo = (int)((q & 0x0f0f0f0fu) | (((h >> (2 * g)) & 0x01010101u) << 4));
                const int hi = (int)(((q >> 4) & 0x0f0f0f0fu) | (((h >> (2 * g + 1)) & 0x01010101u) << 4));
                sl = __dp4a(lo, xl[k], sl);
                sh = __dp4a(hi, xh[k], sh);
                nl = __dp4a(0x01010101, xl[k], nl);
                nh = __dp4a(0x01010101, xh[k], nh);
            }
            const int sb = b * 8 + 2 * g;
            s += ds[sb] * (d * s1 * (float)sl - dmin * m1 * (float)nl) + ds[sb + 1] * (d * s2 * (float)sh - dmin * m2 * (float)nh);
        }
    }
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);
    if (r < rows && c == 0) out[r] = s;
}

// a CPU hand-off without leaving the token's graph: n floats of x (and k picks and weights when given) into
// mapped host memory, then the hand-off's number in *flag once all of it is visible to the host
__global__ void ink_publish(const float* x, const int* picks, const float* wts, float* up, int n, int k, volatile int* flag, const int* posp, int h) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) up[i] = x[i];
    if (picks)
        for (int i = threadIdx.x; i < k; i += blockDim.x) { reinterpret_cast<int*>(up)[n + i] = picks[i]; up[n + k + i] = wts[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *flag = posp[1] * 64 + h + 1;
}

// one thread waits for the host's answer to hand-off h; gives up (and says so in *err) well before the
// operating system's GPU watchdog would reset the device
__global__ void ink_await(volatile const int* ack, const int* posp, int h, volatile int* err) {
    const int want = posp[1] * 64 + h + 1;
    const long long t0 = clock64();
    while (*ack != want) {
        if (clock64() - t0 > 1200000000LL) { *err = want; return; }
        __nanosleep(200);
    }
}

__global__ void ink_softcap(float* x, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
}
"#;

/// This file's kernels.
struct InkKernels {
    rms: Ptr,
    rms2: Ptr,
    mv_f32: Ptr,
    mv_f16: Ptr,
    control: Ptr,
    add: Ptr,
    copy: Ptr,
    mix: Ptr,
    conv: Ptr,
    clip: Ptr,
    store_kv: Ptr,
    attn: Ptr,
    topk: Ptr,
    q8: Ptr,
    moe_gu: Ptr,
    moe_gu_iq3: Ptr,
    situ_q8: Ptr,
    moe_down: Ptr,
    softcap: Ptr,
    mv_q5k: Ptr,
    mv_q8: Ptr,
    ctrl_down: Ptr,
    ctrl_up: Ptr,
    mm_q8: Ptr,
    mv_q5k8: Ptr,
    mv_q8s: Ptr,
    publish: Ptr,
    await_: Ptr,
    group: Ptr,
    moe_gu_g: Ptr,
}

impl InkKernels {
    fn load() -> Result<InkKernels, String> {
        let a = api()?;
        let f = &a.f;
        let opts = [format!("--gpu-architecture=sm_{}{}", a.compute.0, a.compute.1), "-std=c++14".to_string()];
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (INK_KERNELS, &opts, &a.nvrtc_path, &a.device_name).hash(&mut h);
            format!("ink-{:016x}", h.finish())
        };
        let file = std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("ghost-infer").join("kernels").join(format!("{key}.cubin")));
        let bin = match file.as_ref().and_then(|p| std::fs::read(p).ok()) {
            Some(b) => b,
            None => unsafe {
                let src = CString::new(INK_KERNELS).unwrap();
                let name = CString::new("inkling.cu").unwrap();
                let mut prog: Ptr = std::ptr::null_mut();
                if (f.nvrtcCreateProgram)(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
                    return Err("nvrtcCreateProgram failed".into());
                }
                let copts: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
                let ptrs: Vec<*const c_char> = copts.iter().map(|o| o.as_ptr()).collect();
                if (f.nvrtcCompileProgram)(prog, ptrs.len() as i32, ptrs.as_ptr()) != 0 {
                    let mut n = 0usize;
                    (f.nvrtcGetProgramLogSize)(prog, &mut n);
                    let mut log = vec![0 as c_char; n.max(1)];
                    (f.nvrtcGetProgramLog)(prog, log.as_mut_ptr());
                    (f.nvrtcDestroyProgram)(&mut prog);
                    return Err(format!("Inkling kernel compile failed:\n{}", CStr::from_ptr(log.as_ptr()).to_string_lossy()));
                }
                let mut n = 0usize;
                (f.nvrtcGetCUBINSize)(prog, &mut n);
                let mut bin = vec![0u8; n];
                (f.nvrtcGetCUBIN)(prog, bin.as_mut_ptr() as *mut c_char);
                (f.nvrtcDestroyProgram)(&mut prog);
                if let Some(p) = &file {
                    let _ = std::fs::create_dir_all(p.parent().unwrap());
                    let _ = std::fs::write(p, &bin);
                }
                bin
            },
        };
        unsafe {
            let mut module: Ptr = std::ptr::null_mut();
            a.check((f.cuModuleLoadData)(&mut module, bin.as_ptr() as *const c_void), "load Inkling kernels")?;
            let get = |name: &str| -> Result<Ptr, String> {
                let c = CString::new(name).unwrap();
                let mut func: Ptr = std::ptr::null_mut();
                a.check((f.cuModuleGetFunction)(&mut func, module, c.as_ptr()), name)?;
                Ok(func)
            };
            Ok(InkKernels {
                rms: get("ink_rms")?,
                rms2: get("ink_rms2")?,
                mv_f32: get("ink_mv_f32")?,
                mv_f16: get("ink_mv_f16")?,
                control: get("ink_control")?,
                add: get("ink_add")?,
                copy: get("ink_copy")?,
                mix: get("ink_mix")?,
                conv: get("ink_conv")?,
                clip: get("ink_clip")?,
                store_kv: get("ink_store_kv")?,
                attn: get("ink_attn")?,
                topk: get("ink_topk")?,
                q8: get("ink_q8")?,
                moe_gu: get("ink_moe_gu")?,
                moe_gu_iq3: get("ink_moe_gu_iq3")?,
                situ_q8: get("ink_situ_q8")?,
                moe_down: get("ink_moe_down")?,
                softcap: get("ink_softcap")?,
                mv_q5k: get("ink_mv_q5k")?,
                mv_q8: get("ink_mv_q8")?,
                ctrl_down: get("ink_ctrl_down")?,
                ctrl_up: get("ink_ctrl_up")?,
                mm_q8: get("ink_mm_q8")?,
                mv_q5k8: get("ink_mv_q5k8")?,
                mv_q8s: get("ink_mv_q8s")?,
                publish: get("ink_publish")?,
                await_: get("ink_await")?,
                group: get("ink_group")?,
                moe_gu_g: get("ink_moe_gu_g")?,
            })
        }
    }
}

/// A layer's experts on the GPU: every expert's NVFP4 [gate; up] and down blocks as in the file, and their scales.
struct GExperts {
    gu: DeviceBuf,
    /// gate_up in IQ3_XXS as in the file (else split NVFP4)
    gu_iq3: Option<DeviceBuf>,
    /// expert -> its index in the buffers, -1 when it is not on the GPU (the CPU does it)
    slot: DeviceBuf,
    /// only some experts on the GPU: which
    resident: Option<Vec<bool>>,
    /// the slot table as the host holds it
    slots: Vec<i32>,
    down: DeviceBuf,
    gu_bytes: usize,
    down_bytes: usize,
    gu_scale: DeviceBuf,
    down_scale: DeviceBuf,
}

struct GLayer {
    swa: bool,
    norms: Vec<DeviceBuf>,
    wq: GpuMatrix,
    wk: GpuMatrix,
    wv: GpuMatrix,
    wo: GpuMatrix,
    wgate: GpuMatrix,
    rel: GpuMatrix,
    sink: GpuMatrix,
    k_proj: Option<DeviceBuf>,
    v_proj: Option<DeviceBuf>,
    q_clip: DeviceBuf,
    rel_bias: DeviceBuf,
    rel_clip: DeviceBuf,
    k_conv: DeviceBuf,
    v_conv: DeviceBuf,
    post_conv: DeviceBuf,
    ffn_conv: DeviceBuf,
    attn_res_q: Option<(DeviceBuf, usize)>,
    ffn_res_q: DeviceBuf,
    ffn_norm: DeviceBuf,
    router: DeviceBuf,
    bias: DeviceBuf,
    experts: Option<GExperts>,
    /// q, rel, sink and gate (and k, v when one input feeds them all) stacked into one matrix, with each one's
    /// rows; the matrices above are then views into it
    fused: Option<(GpuMatrix, Vec<usize>)>,
}

struct GLoop {
    msg_q: DeviceBuf,
    down: Vec<DeviceBuf>,
    gain: Vec<DeviceBuf>,
    forget: Vec<DeviceBuf>,
}

/// Device buffers for one token's step.
struct Work {
    x: DeviceBuf,
    inp: DeviceBuf,  // 3 rows: mixed attention inputs
    inpc: DeviceBuf, // 3 rows: after the controls
    nrm: DeviceBuf,  // 3 rows: normed
    q: DeviceBuf,
    r: DeviceBuf,
    sinks: DeviceBuf,
    gate: DeviceBuf,
    kraw: DeviceBuf,
    vraw: DeviceBuf,
    kdyn: DeviceBuf,
    vdyn: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    att: DeviceBuf,
    a: DeviceBuf,
    a2: DeviceBuf,
    fi: DeviceBuf,
    f: DeviceBuf,
    f2: DeviceBuf,
    fnorm: DeviceBuf,
    logits_e: DeviceBuf,
    picks: DeviceBuf,
    weights: DeviceBuf,
    xq: DeviceBuf,
    aq: DeviceBuf, // an attention matmul input as int8 (ink_mv_q8)
    ad: DeviceBuf,
    xd: DeviceBuf,
    gu: DeviceBuf,
    hq: DeviceBuf,
    hd: DeviceBuf,
    mo: DeviceBuf,
    mo2: DeviceBuf,
    mcpu: DeviceBuf, // the CPU's experts' part of a partly resident layer
    grp: DeviceBuf,  // a batch's pairs by expert: counts, starts, the pairs (ink_group)
    msgs: DeviceBuf, // 4 rows
    cg: DeviceBuf,   // control gate input (2 * dim)
    c64: DeviceBuf,
    cgain: DeviceBuf,
    cforget: DeviceBuf,
    scr: DeviceBuf, // attention scores
    pos: DeviceBuf,
    pool: DeviceBuf, // rows for the depth blocks, the prelude output and the pass outputs
    logits: DeviceBuf,
}

/// Mapped host memory for the CPU hand-offs of a one-graph token: GPU to host (a row, picks and weights), host
/// to GPU (the CPU's sum), and their flags.
struct Hops {
    up: MappedBuf,
    down: MappedBuf,
    flag: MappedBuf,
    ack: MappedBuf,
    err: MappedBuf,
}

pub struct GpuInkling<'a> {
    /// the hand-offs' buffers (None: GHOST_INFER_INK_SYNC_HOPS=1, a graph between hand-offs and copies at each)
    hops: Option<Hops>,
    /// per layer, how often each expert was picked lately (partly resident layers: what to keep on the GPU)
    heat: Vec<Vec<f32>>,
    pub swaps: usize,
    /// the hand-off now in the token, and tokens stepped so far (the hand-offs' numbers)
    hop: i32,
    steps: i32,
    /// a prompt batch's work buffers (swapped with `w` while one runs) and the tokens in the pass now
    wb: Option<Work>,
    nb: usize,
    cpu: Inkling<'a>,
    c: Cfg,
    mm: GpuModel,
    s: GpuSession,
    ik: InkKernels,
    layers: Vec<GLayer>,
    loops: Vec<GLoop>,
    tok_norm: DeviceBuf,
    /// the head: Q5_K blocks as in the file (ink_mv_q5k), or a matrix of cuda.rs's formats
    output: Result<(DeviceBuf, usize), GpuMatrix>,
    output_norm: DeviceBuf,
    output_res_q: DeviceBuf,
    /// per slot: K and V caches (f16), rows (n_swa for sliding-window slots, ctx for full ones), and the conv state
    kv: Vec<(DeviceBuf, DeviceBuf, usize)>,
    conv: DeviceBuf,
    w: Work,
    pub ctx: usize,
    pub pos: usize,
    pub gpu_expert_layers: Vec<usize>,
    pub trace: Option<Vec<(String, Vec<f32>)>>,
    /// seconds: waiting for the GPU before a CPU-expert layer (its queued work), the CPU experts
    pub prof: [f64; 2],
    /// host time per part: whole steps, graph launches, copies to the GPU after CPU experts, the logits' copy back
    pub hostprof: [f64; 4],
    pub prof_layer: Vec<(f64, usize)>,
    sc: crate::quant::ExpertScratch,
    kprof: Option<std::cell::RefCell<std::collections::HashMap<usize, (f64, usize)>>>,
    /// CUDA graphs: one per stretch of GPU work between two CPU-expert hand-offs (the same every token), captured
    /// on the first token and replayed after; `gmode` 0 off, 1 capturing, 2 replaying; `seg` the stretch now
    graphs: Vec<Ptr>,
    /// the graphs of a full prompt batch's stretches (swapped in while one runs: every full batch is the same work)
    graphs_b: Vec<Ptr>,
    gmode: u8,
    kev: std::cell::RefCell<Vec<(usize, usize, usize)>>,
    /// GHOST_INFER_INK_COUNT: how often each layer picks each expert (no graphs then: the picks are read back)
    counts: Option<Vec<Vec<u64>>>,
    pub segtime: Option<Vec<(f64, usize)>>,
    seg: usize,
}

fn f32_buf(up: &mut Uploader, v: &[f32]) -> Result<DeviceBuf, String> {
    up.put_vec(v)
}

fn read_f32(g: &Gguf, name: &str) -> Result<Vec<f32>, String> {
    let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let n = t.n_elements() as usize;
    let mut v = vec![0f32; n];
    crate::quant::Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n }.row_f32(0, &mut v);
    Ok(v)
}

/// A 3-D NVFP4 expert tensor uploaded with each expert's matrix split for coalesced reads: all its rows' 4-bit
/// codes (cols / 2 bytes a row, the file's byte order), then all its rows' UE4M3 scales (cols / 16 a row). The
/// same bytes per expert as the file.
fn upload_nvfp4_split(up: &mut Uploader, g: &Gguf, t: &TensorInfo, n_exp: usize, which: &[usize]) -> Result<DeviceBuf, String> {
    let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let ebytes = t.n_bytes / n_exp;
    let buf = up.alloc(ebytes * which.len())?;
    let file = std::fs::File::open(&g.path).map_err(|e| format!("cannot open {}: {e}", g.path.display()))?;
    let mut raw = vec![0u8; ebytes];
    let mut out = vec![0u8; ebytes];
    let (qrow, srow, nb) = (cols / 2, cols / 16, cols / 64);
    for (i, &e) in which.iter().enumerate() {
        crate::inkling::read_exact_at(&file, &mut raw, (t.offset + e * ebytes) as u64)?;
        nvfp4_split_into(&raw, rows, cols, &mut out);
        let _ = (qrow, srow, nb);
        up.put(buf.ptr + (i * ebytes) as u64, &out)?;
    }
    Ok(buf)
}

/// The `n` experts of layer `il` to keep on the GPU when only some fit: the most used ones by the counts in the
/// file GHOST_INFER_INK_HOT names (lines "layer: count count ..." per expert, as inkling-gpu-gen writes with
/// GHOST_INFER_INK_COUNT), else the first `n`.
/// Rows a warp of ink_mv_q8 takes (must match MV_ROWS).
const MV_ROWS: usize = 1;

/// Rows a warp of ink_moe_down takes (must match DOWN_ROWS).
const DOWN_ROWS: usize = 1;

fn hot_experts(il: usize, n_exp: usize, n: usize, model: &std::path::Path) -> Vec<usize> {
    // the counts GHOST_INFER_INK_HOT names, else <model>.hot.txt beside the model when there is one
    let file = std::env::var_os("GHOST_INFER_INK_HOT").map(std::path::PathBuf::from).unwrap_or_else(|| model.with_extension("hot.txt"));
    let counts: Option<Vec<u64>> = std::fs::read_to_string(file).ok().and_then(|t| {
        t.lines().find_map(|l| {
            let (a, b) = l.split_once(':')?;
            (a.trim().parse::<usize>().ok()? == il).then(|| b.split_whitespace().filter_map(|v| v.parse().ok()).collect::<Vec<u64>>())
        })
    });
    let mut order: Vec<usize> = (0..n_exp).collect();
    if let Some(c) = counts.filter(|c| c.len() == n_exp) {
        order.sort_by(|a, b| c[*b].cmp(&c[*a]).then(a.cmp(b)));
    }
    order.truncate(n);
    order.sort_unstable();
    order
}

/// Stacks a layer's Q8 projections that read the same input (q, rel, sink, gate; with k and v too when one
/// normed input feeds them all, the sliding-window layers) into one matrix, the originals freed and replaced by
/// views into it. Layers whose projections are in other formats stay as they are.
fn fuse_projections(l: &mut GLayer) -> Result<(), String> {
    let n = if l.swa { 6 } else { 4 };
    let (rows, cols) = {
        let ms = [&l.wq, &l.rel, &l.sink, &l.wgate, &l.wk, &l.wv];
        let cols = ms[0].cols;
        if !ms[..n].iter().all(|m| m.fmt == Fmt::Q8 && m.fold.is_none() && m.d.is_some() && m.cols == cols) || cols % 256 != 0 || cols > 4096 {
            return Ok(());
        }
        (ms[..n].iter().map(|m| m.rows).collect::<Vec<usize>>(), cols)
    };
    let total: usize = rows.iter().sum();
    let (q, d) = (DeviceBuf::alloc(total * cols)?, DeviceBuf::alloc(4 * total * cols / 32)?);
    let a = api()?;
    let mut off = 0;
    {
        let ms = [&l.wq, &l.rel, &l.sink, &l.wgate, &l.wk, &l.wv];
        for m in &ms[..n] {
            a.check(unsafe { (a.f.cuMemcpyDtoD_v2)(q.ptr + (off * cols) as u64, m.q.ptr, m.rows * cols) }, "stack projections")?;
            a.check(unsafe { (a.f.cuMemcpyDtoD_v2)(d.ptr + (4 * off * cols / 32) as u64, m.d_ptr(), 4 * m.rows * cols / 32) }, "stack projections")?;
            off += m.rows;
        }
    }
    a.check(unsafe { (a.f.cuCtxSynchronize)() }, "synchronize")?;
    let view = |off: usize, rows: usize| GpuMatrix {
        fmt: Fmt::Q8,
        q: DeviceBuf { ptr: q.ptr + (off * cols) as u64, bytes: 0, owned: false },
        d: Some(DeviceBuf { ptr: d.ptr + (4 * off * cols / 32) as u64, bytes: 0, owned: false }),
        rows,
        cols,
        coarse: false,
        iq: -1,
        fold: None,
    };
    let mut off = 0;
    for (j, &r) in rows.iter().enumerate() {
        let m = match j {
            0 => &mut l.wq,
            1 => &mut l.rel,
            2 => &mut l.sink,
            3 => &mut l.wgate,
            4 => &mut l.wk,
            _ => &mut l.wv,
        };
        *m = view(off, r); // frees the original
        off += r;
    }
    l.fused = Some((GpuMatrix { fmt: Fmt::Q8, q, d: Some(d), rows: total, cols, coarse: false, iq: -1, fold: None }, rows));
    Ok(())
}

/// Prompt tokens a GPU pass takes at most.
const PREFILL_B: usize = 128;
/// Rows the depth states take in the pool: the prelude's block and output, each pass's blocks and output, the tail's block
const POOL_ROWS: usize = 12;

/// The work buffers for passes of `nb` tokens (the attention scores for 8 of them at a time).
fn alloc_work(c: &Cfg, ctx: usize, nb: usize, logits_rows: usize) -> Result<Work, String> {
    let f = |n: usize| DeviceBuf::alloc(4 * n);
    let (d, kvd) = (c.dim, c.n_kv * c.dh);
    Ok(Work {
        x: f(d * nb)?,
        inp: f(3 * d * nb)?,
        inpc: f(3 * d * nb)?,
        nrm: f(3 * d * nb)?,
        q: f(c.n_head * c.dh * nb)?,
        r: f(c.n_head * c.n_rel * nb)?,
        sinks: f(c.n_head * nb)?,
        gate: f(c.n_head * c.dh * nb)?,
        kraw: f(kvd * nb)?,
        vraw: f(kvd * nb)?,
        kdyn: f(c.n_kv * c.conv * nb)?,
        vdyn: f(c.n_kv * c.conv * nb)?,
        k: f(kvd * nb)?,
        v: f(kvd * nb)?,
        att: f(c.n_head * c.dh * nb)?,
        a: f(d * nb)?,
        a2: f(d * nb)?,
        fi: f(d * nb)?,
        f: f(d * nb)?,
        f2: f(d * nb)?,
        fnorm: f(d * nb)?,
        logits_e: f(c.n_exp * nb)?,
        picks: f(c.k * nb)?,
        weights: f(c.k * nb)?,
        xq: DeviceBuf::alloc(d * nb)?,
        aq: DeviceBuf::alloc(4096 * nb)?,
        ad: f(128 * nb)?,
        xd: f(d / 32 * nb)?,
        gu: f(c.k * 2 * c.ff_exp * nb)?,
        hq: DeviceBuf::alloc(c.k * c.ff_exp * nb)?,
        hd: f(c.k * c.ff_exp / 32 * nb)?,
        mo: f(d * nb)?,
        mo2: f(d * nb)?,
        mcpu: f(d * nb)?,
        grp: f(2 * c.n_exp + c.k * nb)?,
        msgs: f(4 * d * nb)?,
        cg: f(2 * d)?,
        c64: f(256 * nb)?,
        cgain: f(d)?,
        cforget: f(d)?,
        scr: f(c.n_head * ctx * nb.min(8))?,
        pos: DeviceBuf::alloc(8)?, // the position, and the step count (hand-off numbers)
        pool: f(POOL_ROWS * d * nb)?,
        logits: f(logits_rows)?,
    })
}

/// One NVFP4 expert matrix (rows of 36-byte blocks, as in the file) re-laid as upload_nvfp4_split lays it.
fn nvfp4_split_into(raw: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
    let (qrow, srow, nb) = (cols / 2, cols / 16, cols / 64);
    let (q, sc) = out.split_at_mut(rows * qrow);
    for r in 0..rows {
        for b in 0..nb {
            let blk = &raw[(r * nb + b) * 36..(r * nb + b + 1) * 36];
            sc[r * srow + 4 * b..r * srow + 4 * b + 4].copy_from_slice(&blk[..4]);
            q[r * qrow + 32 * b..r * qrow + 32 * b + 32].copy_from_slice(&blk[4..]);
        }
    }
}

fn free_vram() -> Result<usize, String> {
    let a = api()?;
    let (mut free, mut total) = (0usize, 0usize);
    a.check(unsafe { (a.f.cuMemGetInfo_v2)(&mut free, &mut total) }, "memory info")?;
    Ok(free)
}

impl<'a> GpuInkling<'a> {
    pub fn load(g: &'a Gguf, ctx: usize) -> Result<GpuInkling<'a>, String> {
        let cpu = Inkling::load(g)?;
        let c = cpu.c.clone();
        if c.k > 8 || c.k * c.ff_exp > 8 * 1152 || c.dim > 4096 {
            return Err(format!("the GPU expert kernels hold 8 picks of 1152 (this file: {} of {}, dim {})", c.k, c.ff_exp, c.dim));
        }
        let cfg = Config {
            dim: c.dim,
            n_layers: 0,
            n_heads: c.n_head,
            n_kv_heads: c.n_kv,
            head_dim: c.dh,
            ffn_dim: c.n_head * c.dh,
            vocab: 32,
            max_ctx: 32,
            rope_dim: c.dh,
            rope_base: 10_000.0,
            rope_neox: false,
            rope_freqs: None,
            norm_eps: c.eps,
            n_experts: 0,
            n_experts_used: 0,
            expert_ffn_dim: 0,
            mla: None,
            n_dense_lead: 0,
            n_shared_experts: 0,
            gating_sigmoid: false,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            delta: None,
            attn_gated: false,
        };
        let k = Arc::new(Kernels::compile_on(false, false)?);
        eprintln!("[vram free after kernels: {} MB]", free_vram()? >> 20);
        let ik = InkKernels::load()?;
        let mut up = Uploader::new()?;
        let up = &mut up;
        let mut layers = Vec::new();
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let swa = c.swa[i];
            let norms = if swa { vec![f32_buf(up, &read_f32(g, &p("attn_norm"))?)?] } else { ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"].iter().map(|n| f32_buf(up, &read_f32(g, &p(n))?)).collect::<Result<_, _>>()? };
            let one = |n: &str, up: &mut Uploader| upload_one(g, &p(n), up);
            let vecf = |n: &str, up: &mut Uploader| -> Result<DeviceBuf, String> { f32_buf(up, &read_f32(g, n)?) };
            let bias = match g.tensor(&format!("blk.{i}.exp_probs_b.bias")) {
                Some(_) => read_f32(g, &format!("blk.{i}.exp_probs_b.bias"))?,
                None => vec![0f32; c.n_exp],
            };
            layers.push(GLayer {
                swa,
                norms,
                wq: one("attn_q", up)?,
                wk: one("attn_k", up)?,
                wv: one("attn_v", up)?,
                wo: one("attn_output", up)?,
                wgate: one("attn_gate", up)?,
                rel: one("attn_rel", up)?,
                sink: one("attn_sink_proj", up)?,
                k_proj: if swa { None } else { Some(vecf(&p("attn_k_conv_proj"), up)?) },
                v_proj: if swa { None } else { Some(vecf(&p("attn_v_conv_proj"), up)?) },
                q_clip: vecf(&p("attn_q_clip"), up)?,
                rel_bias: vecf(&p("attn_rel_bias"), up)?,
                rel_clip: vecf(&p("attn_rel_clip"), up)?,
                k_conv: vecf(&p("attn_k_conv"), up)?,
                v_conv: vecf(&p("attn_v_conv"), up)?,
                post_conv: vecf(&p("attn_post_conv"), up)?,
                ffn_conv: vecf(&p("ffn_post_conv"), up)?,
                attn_res_q: if i > 0 { Some((vecf(&p("attn_res_query"), up)?, if swa { 1 } else { 3 })) } else { None },
                ffn_res_q: vecf(&p("ffn_res_query"), up)?,
                ffn_norm: vecf(&p("ffn_norm"), up)?,
                router: vecf(&p("ffn_gate_inp"), up)?,
                bias: f32_buf(up, &bias)?,
                experts: None,
                fused: None,
            });
        }
        let mut loops = Vec::new();
        for i in c.loop_start..c.loop_end {
            let names = ["q", "k", "v", "ffn"];
            // the controls' matrices as f16 (half the memory of f32; their products feed tanh gates)
            let m = |kind: &str, up: &mut Uploader| -> Result<Vec<DeviceBuf>, String> {
                names.iter().map(|n| -> Result<DeviceBuf, String> {
                    let v = read_f32(g, &format!("blk.{i}.loop_{n}_ctrl_{kind}.weight"))?;
                    up.put_vec(&v.iter().map(|&x| crate::quant::f32_to_f16(x)).collect::<Vec<u16>>())
                }).collect()
            };
            loops.push(GLoop { msg_q: f32_buf(up, &read_f32(g, &format!("blk.{i}.loop_msg_query.weight"))?)?, down: m("down", up)?, gain: m("gain", up)?, forget: m("forget", up)? });
        }
        let output = match g.tensor("output.weight") {
            Some(t) if t.ty == GgmlType::Q5K => {
                let buf = up.alloc(t.n_bytes)?;
                up.put_file(buf.ptr, &g.path, t.offset as u64, t.n_bytes)?;
                Ok((buf, t.dims[1] as usize))
            }
            _ => Err(upload_one(g, "output.weight", up)?),
        };
        let tok_norm = f32_buf(up, &read_f32(g, "token_embd_norm.weight")?)?;
        let output_norm = f32_buf(up, &read_f32(g, "output_norm.weight")?)?;
        let output_res_q = f32_buf(up, &read_f32(g, "output_res_query.weight")?)?;
        up.finish()?;
        // after the uploads land: they are what gets stacked
        if std::env::var_os("GHOST_INFER_INK_NO_FUSE").is_none() {
            for l in layers.iter_mut() {
                fuse_projections(l)?;
            }
        }

        // caches: sliding-window slots hold the window, full ones the context
        let n_slots = c.n_slots();
        let kvd = c.n_kv * c.dh;
        let slot_layer = |s: usize| if s < c.n_layers { s } else { c.loop_start + (s - c.n_layers) % (c.loop_end - c.loop_start) };
        let mut kv = Vec::with_capacity(n_slots);
        for s in 0..n_slots {
            let cap = if c.swa[slot_layer(s)] { (c.n_swa + PREFILL_B).min(ctx) } else { ctx };
            kv.push((DeviceBuf::alloc(2 * cap * kvd)?, DeviceBuf::alloc(2 * cap * kvd)?, cap));
        }
        let conv_floats = (2 * kvd + 2 * c.dim) * (c.conv - 1);
        eprintln!("[vram free after weights: {} MB]", free_vram()? >> 20);
        let conv = DeviceBuf::alloc(4 * conv_floats * n_slots)?;
        zero(&conv)?;
        let w = alloc_work(&c, ctx, 1, cpu_output_rows(g)?)?;
        // a prompt batch's buffers (GHOST_INFER_INK_PREFILL=0: prompts token by token)
        let wb = if std::env::var("GHOST_INFER_INK_PREFILL").is_ok_and(|v| v == "0") { None } else { Some(alloc_work(&c, ctx, PREFILL_B, cpu_output_rows(g)?)?) };
        eprintln!("[vram free after caches and work buffers: {} MB]", free_vram()? >> 20);
        let hops = if std::env::var_os("GHOST_INFER_INK_SYNC_HOPS").is_some() {
            None
        } else {
            Some(Hops { up: MappedBuf::alloc(4 * (c.dim + 2 * c.k))?, down: MappedBuf::alloc(4 * c.dim)?, flag: MappedBuf::alloc(4)?, ack: MappedBuf::alloc(4)?, err: MappedBuf::alloc(4)? })
        };
        let s = GpuSession::new_one_row(&cfg, 32)?;
        let mm = GpuModel {
            cfg,
            k,
            embd: Embd::Vectors,
            output: None,
            output_norm: DeviceBuf::alloc(4)?,
            rope_freqs: None,
            layers: Vec::new(),
            cache: Default::default(),
            moe: None,
            prefill_shared: None,
            device: String::new(),
            int8_q8: true,
            iq_scratch: None,
            lora_t: None,
            lora: None,
            hadamard: None,
            _fold_signs: Vec::new(),
            _weights: None,
        };
        let mut m = GpuInkling { cpu, c, mm, s, ik, layers, loops, tok_norm, output, output_norm, output_res_q, kv, conv, w, wb, nb: 1, hops, heat: Vec::new(), swaps: 0, hop: 0, steps: 0, ctx, pos: 0, gpu_expert_layers: Vec::new(), trace: None, prof: [0.0; 2], hostprof: [0.0; 4], prof_layer: vec![(0.0, 0); 24], sc: Default::default(), kprof: std::env::var_os("GHOST_INFER_INK_KPROF").map(|_| Default::default()), graphs: Vec::new(), graphs_b: Vec::new(), gmode: 0, seg: 0, kev: Default::default(), segtime: std::env::var_os("GHOST_INFER_INK_SEGTIME").map(|_| Vec::new()), counts: None };
        if std::env::var_os("GHOST_INFER_INK_COUNT").is_some() {
            m.counts = Some(vec![vec![0; m.c.n_exp]; m.c.n_layers]);
        }
        m.heat = vec![vec![0f32; m.c.n_exp]; m.c.n_layers];
        eprintln!("[vram free after session: {} MB]", free_vram()? >> 20);
        m.place_experts(g)?;
        Ok(m)
    }

    /// Experts to the GPU for the looped layers stored in NVFP4 (three runs per token each), as many as fit with
    /// GHOST_INFER_INK_RESERVE_MB (default 128) left free; GHOST_INFER_INK_GPU_LAYERS=a,b,... picks them instead.
    fn place_experts(&mut self, g: &Gguf) -> Result<(), String> {
        let c = &self.c;
        // NVFP4 downs with NVFP4 or IQ3_XXS gate_ups have GPU kernels
        let ty = |i: usize, n: &str| g.tensor(&format!("blk.{i}.{n}.weight")).map(|t| t.ty);
        let nv = |i: usize| ty(i, "ffn_down_exps") == Some(GgmlType::Nvfp4) && matches!(ty(i, "ffn_gate_up_exps"), Some(GgmlType::Nvfp4 | GgmlType::Iq3Xxs));
        let wanted: Vec<usize> = match std::env::var("GHOST_INFER_INK_GPU_LAYERS") {
            Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).filter(|&i| i < c.n_layers && nv(i)).collect(),
            Err(_) => (c.loop_start..c.loop_end).filter(|&i| nv(i)).chain((0..c.n_layers).filter(|&i| !(c.loop_start..c.loop_end).contains(&i) && nv(i))).collect(),
        };
        let reserve = std::env::var("GHOST_INFER_INK_RESERVE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize) << 20;
        let mut up = Uploader::new()?;
        // looped layers first (each runs every pass), whole then in part; then the others the same way
        let in_loop = |i: &usize| (c.loop_start..c.loop_end).contains(i);
        let groups = [wanted.iter().copied().filter(|i| in_loop(i)).collect::<Vec<usize>>(), wanted.iter().copied().filter(|i| !in_loop(i)).collect()];
        for group in groups {
        let mut left = Vec::new();
        for pass in 0..2 {
        let list = if pass == 0 { group.clone() } else { left.clone() };
        for i in list {
            let (gu, dn) = (g.tensor(&format!("blk.{i}.ffn_gate_up_exps.weight")).unwrap(), g.tensor(&format!("blk.{i}.ffn_down_exps.weight")).unwrap());
            let per = (gu.n_bytes + dn.n_bytes) / c.n_exp;
            let free = free_vram()?.saturating_sub(reserve);
            // first every layer that fits whole; then what is left goes to the first layers that do not, a part of
            // their experts each (GHOST_INFER_INK_NO_PARTIAL: none)
            let which: Vec<usize> = if pass == 0 {
                if free < gu.n_bytes + dn.n_bytes {
                    left.push(i); // a smaller layer further on may still fit
                    continue;
                }
                (0..c.n_exp).collect()
            } else {
                let n = (free / per).min(c.n_exp);
                // a quarter of the experts at least: fewer save less than the hand-off to the CPU costs
                if n < (c.n_exp / 4).max(c.k) || std::env::var_os("GHOST_INFER_INK_NO_PARTIAL").is_some() {
                    continue;
                }
                hot_experts(i, c.n_exp, n, &g.path)
            };
            let resident = if which.len() == c.n_exp { None } else {
                let mut r = vec![false; c.n_exp];
                which.iter().for_each(|&e| r[e] = true);
                Some(r)
            };
            let mut slot = vec![-1i32; c.n_exp];
            which.iter().enumerate().for_each(|(j, &e)| slot[e] = j as i32);
            let iq3 = gu.ty == GgmlType::Iq3Xxs;
            let gbuf = if iq3 {
                let eb = gu.n_bytes / c.n_exp;
                let b = up.alloc(eb * which.len())?;
                for (j, &e) in which.iter().enumerate() {
                    up.put_file(b.ptr + (j * eb) as u64, &g.path, (gu.offset + e * eb) as u64, eb)?;
                }
                b
            } else {
                upload_nvfp4_split(&mut up, g, gu, c.n_exp, &which)?
            };
            let gu_iq3 = if iq3 { Some(up.put_vec(&crate::iquant::IQ3XXS_GRID)?) } else { None };
            let dbuf = upload_nvfp4_split(&mut up, g, dn, c.n_exp, &which)?;
            let ones = vec![1f32; c.n_exp];
            let gs = match g.tensor(&format!("blk.{i}.ffn_gate_up_exps.scale")) { Some(_) => read_f32(g, &format!("blk.{i}.ffn_gate_up_exps.scale"))?, None => ones.clone() };
            let ds = match g.tensor(&format!("blk.{i}.ffn_down_exps.scale")) { Some(_) => read_f32(g, &format!("blk.{i}.ffn_down_exps.scale"))?, None => ones };
            self.layers[i].experts = Some(GExperts { gu: gbuf, gu_iq3, slot: up.put_vec(&slot)?, slots: slot.clone(), resident, down: dbuf, gu_bytes: gu.n_bytes / c.n_exp, down_bytes: dn.n_bytes / c.n_exp, gu_scale: up.put_vec(&gs)?, down_scale: up.put_vec(&ds)? });
            self.gpu_expert_layers.push(i);
            if which.len() < c.n_exp {
                eprintln!("[layer {i}: {} of {} experts on the GPU, the rest on the CPU]", which.len(), c.n_exp);
            }
        }
        }
        }
        up.finish()?;
        // the CPU layers' experts into RAM of our own (GHOST_INFER_INK_MAPPED=1 reads them from the mapped file)
        if std::env::var_os("GHOST_INFER_INK_MAPPED").is_none() {
            let t = std::time::Instant::now();
            let mut bytes = 0;
            let file = std::fs::File::open(&g.path).map_err(|e| format!("cannot open {}: {e}", g.path.display()))?;
            for i in (0..self.c.n_layers).filter(|i| self.layers[*i].experts.as_ref().is_none_or(|e| e.resident.is_some())) {
                bytes += self.cpu.own_experts(i, &file, g.mmap.as_ptr() as usize)?;
            }
            eprintln!("[{} MB of CPU experts copied to RAM in {:.1} s]", bytes >> 20, t.elapsed().as_secs_f64());
        }
        Ok(())
    }

    fn launch(&self, f: Ptr, grid: (u32, u32), block: u32, args: &[Arg]) -> Result<(), String> {
        if self.gmode == 2 {
            return Ok(());
        }
        self.timed(f as usize, || self.mm.k.launch(f, grid, block, 0, args))
    }

    fn launch3(&self, f: Ptr, grid: (u32, u32, u32), block: u32, args: &[Arg]) -> Result<(), String> {
        if self.gmode == 2 {
            return Ok(());
        }
        self.timed(f as usize, || self.mm.k.launch3(f, grid, block, 0, args))
    }

    /// Starts a stretch of GPU work: capturing, its capture begins.
    fn seg_begin(&mut self) -> Result<(), String> {
        if self.gmode == 1 {
            let a = api()?;
            const RELAXED: i32 = 2;
            a.check(unsafe { (a.f.cuStreamBeginCapture_v2)(self.mm.k.stream, RELAXED) }, "begin graph capture")?;
        }
        Ok(())
    }

    /// Ends a stretch before the host needs its results: capturing, the graph is made and run; replaying, the
    /// stretch's graph runs.
    fn seg_end(&mut self) -> Result<(), String> {
        let t = std::time::Instant::now();
        let r = self.seg_end_inner();
        self.hostprof[1] += t.elapsed().as_secs_f64();
        r
    }

    fn seg_end_inner(&mut self) -> Result<(), String> {
        let a = api()?;
        let stream = self.mm.k.stream;
        match self.gmode {
            1 => {
                let mut g: Ptr = std::ptr::null_mut();
                a.check(unsafe { (a.f.cuStreamEndCapture)(stream, &mut g) }, "end graph capture")?;
                let mut exec: Ptr = std::ptr::null_mut();
                let made = unsafe { (a.f.cuGraphInstantiateWithFlags)(&mut exec, g, 0) };
                unsafe { (a.f.cuGraphDestroy)(g) };
                a.check(made, "graph instantiate")?;
                self.graphs.push(exec);
                a.check(unsafe { (a.f.cuGraphLaunch)(exec, stream) }, "graph launch")?;
            }
            2 => {
                let exec = *self.graphs.get(self.seg).ok_or("a stretch of GPU work with no graph")?;
                // GHOST_INFER_INK_SEGTIME=1: each replay's GPU time, by stretch
                if self.segtime.is_some() {
                    let (mut e0, mut e1): (Ptr, Ptr) = (std::ptr::null_mut(), std::ptr::null_mut());
                    unsafe {
                        (a.f.cuEventCreate)(&mut e0, 0);
                        (a.f.cuEventCreate)(&mut e1, 0);
                        (a.f.cuEventRecord)(e0, stream);
                        a.check((a.f.cuGraphLaunch)(exec, stream), "graph launch")?;
                        (a.f.cuEventRecord)(e1, stream);
                        (a.f.cuEventSynchronize)(e1);
                        let mut ms = 0f32;
                        (a.f.cuEventElapsedTime)(&mut ms, e0, e1);
                        (a.f.cuEventDestroy_v2)(e0);
                        (a.f.cuEventDestroy_v2)(e1);
                        let st = self.segtime.as_mut().unwrap();
                        if st.len() <= self.seg {
                            st.resize(self.seg + 1, (0.0, 0));
                        }
                        st[self.seg].0 += ms as f64;
                        st[self.seg].1 += 1;
                    }
                } else {
                    a.check(unsafe { (a.f.cuGraphLaunch)(exec, stream) }, "graph launch")?;
                }
            }
            _ => {}
        }
        self.seg += 1;
        Ok(())
    }

    /// GHOST_INFER_INK_KPROF=1: every launch synchronised and timed by kernel (the time includes the launch).
    fn timed(&self, key: usize, run: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
        let Some(kp) = &self.kprof else { return run() };
        // GPU time between two events around the launch (no synchronising: the launches queue as usual)
        let a = api()?;
        let (mut e0, mut e1): (Ptr, Ptr) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe {
            a.check((a.f.cuEventCreate)(&mut e0, 0), "event")?;
            a.check((a.f.cuEventCreate)(&mut e1, 0), "event")?;
            a.check((a.f.cuEventRecord)(e0, self.mm.k.stream), "event record")?;
        }
        run()?;
        a.check(unsafe { (a.f.cuEventRecord)(e1, self.mm.k.stream) }, "event record")?;
        let mut m = kp.borrow_mut();
        m.entry(usize::MAX).or_insert((0.0, 0)); // keeps the map's shape; the events go to the list
        drop(m);
        self.kev.borrow_mut().push((key, e0 as usize, e1 as usize));
        Ok(())
    }

    /// The kernel profile, slowest first, with names.
    pub fn kprof_report(&self) -> Option<String> {
        let a = api().ok()?;
        let mut kp: std::collections::HashMap<usize, (f64, usize)> = Default::default();
        for &(key, e0, e1) in self.kev.borrow().iter() {
            let mut ms = 0f32;
            unsafe {
                (a.f.cuEventSynchronize)(e1 as Ptr);
                (a.f.cuEventElapsedTime)(&mut ms, e0 as Ptr, e1 as Ptr);
            }
            let e = kp.entry(key).or_insert((0.0, 0));
            e.0 += ms as f64 / 1000.0;
            e.1 += 1;
        }
        let ik = &self.ik;
        let names: [(Ptr, &str); 20] = [
            (ik.rms, "rms"), (ik.rms2, "rms2"), (ik.mv_f32, "mv_f32"), (ik.mv_f16, "mv_f16"), (ik.control, "control"), (ik.add, "add"),
            (ik.copy, "copy"), (ik.mix, "mix"), (ik.conv, "conv"), (ik.clip, "clip"), (ik.store_kv, "store_kv"), (ik.attn, "attn"),
            (ik.topk, "topk"), (ik.q8, "q8"), (ik.moe_gu, "moe_gu"), (ik.situ_q8, "situ_q8"), (ik.moe_down, "moe_down"), (ik.softcap, "softcap"),
            (ik.mv_q5k, "mv_q5k"), (std::ptr::null_mut(), "matmul (cuda.rs)"),
        ];
        let total: f64 = kp.values().map(|v| v.0).sum();
        let mut rows: Vec<(f64, usize, &str)> = kp.iter().map(|(k, v)| (v.0, v.1, names.iter().find(|n| n.0 as usize == *k).map_or("?", |n| n.1))).collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        Some(format!("GPU time {:.0} ms in all
", total * 1000.0) + &rows.iter().map(|(t, n, name)| format!("{name}: {:.0} ms over {n} launches ({:.1} us each)", t * 1000.0, t * 1e6 / *n as f64)).collect::<Vec<_>>().join("
"))
    }

    /// `rows` rows of `n` a token, for the pass's tokens.
    fn rms(&self, x: u64, w: u64, out: u64, rows: usize, n: usize, eps: f32) -> Result<(), String> {
        self.launch(self.ik.rms, ((rows * self.nb) as u32, 1), 256, &[Arg::P(x), Arg::P(w), Arg::P(out), Arg::I(n as i32), Arg::F(eps)])
    }

    fn mv_f32(&self, w: u64, x: u64, out: u64, rows: usize, cols: usize) -> Result<(), String> {
        self.launch(self.ik.mv_f32, (rows.div_ceil(8) as u32, self.nb as u32), 256, &[Arg::P(w), Arg::P(x), Arg::P(out), Arg::I(rows as i32), Arg::I(cols as i32)])
    }

    /// Matrices times vectors, (matrix, input, output) in order: plain Q8 ones through ink_mv_q8, each input
    /// quantised once for the run of matrices that read it; the rest through cuda.rs.
    fn mms(&self, jobs: &[(&GpuMatrix, u64, u64)]) -> Result<(), String> {
        if self.nb > 1 {
            return self.mms_batch(jobs);
        }
        let mut quantised = 0u64;
        for &(m, x, out) in jobs {
            if m.fmt == Fmt::Q8 && m.fold.is_none() && m.cols <= 4096 && m.cols % 256 == 0 && std::env::var_os("GHOST_INFER_INK_NO_MVQ8").is_none() {
                if quantised != x {
                    self.launch(self.ik.q8, ((m.cols / 32).div_ceil(8) as u32, 1), 256, &[Arg::P(x), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::I(m.cols as i32)])?;
                    quantised = x;
                }
                self.launch(self.ik.mv_q8, (m.rows.div_ceil(8 * MV_ROWS) as u32, 1), 256, &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32)])?;
            } else {
                self.mm(m, x, out)?;
            }
        }
        Ok(())
    }

    /// `mms` for the pass's tokens (rows of the inputs and outputs): plain Q8 through ink_mm_q8, the rest a token
    /// at a time.
    fn mms_batch(&self, jobs: &[(&GpuMatrix, u64, u64)]) -> Result<(), String> {
        let nb = self.nb;
        let mut quantised = 0u64;
        for &(m, x, out) in jobs {
            if m.fmt == Fmt::Q8 && m.fold.is_none() && m.cols <= 4096 && m.cols % 256 == 0 {
                if quantised != x {
                    self.launch(self.ik.q8, ((nb * m.cols / 32).div_ceil(8) as u32, 1), 256, &[Arg::P(x), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::I((nb * m.cols) as i32)])?;
                    quantised = x;
                }
                self.launch(self.ik.mm_q8, (m.rows.div_ceil(8) as u32, nb.div_ceil(8) as u32), 256, &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(nb as i32)])?;
            } else {
                for b in 0..nb {
                    self.mm(m, x + (4 * b * m.cols) as u64, out + (4 * b * m.rows) as u64)?;
                }
            }
        }
        Ok(())
    }

    /// A stacked matrix (fuse_projections) times the pass's tokens' input x: one launch, matrix j's rows to outs[j].
    fn mm_stacked(&self, m: &GpuMatrix, x: u64, outs: &[u64], rows: &[usize]) -> Result<(), String> {
        let nb = self.nb;
        self.launch(self.ik.q8, ((nb * m.cols / 32).div_ceil(8) as u32, 1), 256, &[Arg::P(x), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::I((nb * m.cols) as i32)])?;
        // only the first rows.len() matrices of the stack
        let used: usize = rows.iter().sum();
        let mut args = vec![Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::I(used as i32), Arg::I(m.cols as i32), Arg::I(nb as i32)];
        args.extend((0..6).map(|j| Arg::P(*outs.get(j).unwrap_or(&0))));
        let mut end = 0;
        args.extend((0..6).map(|j| {
            end += rows.get(j).copied().unwrap_or(0);
            Arg::I(end as i32)
        }));
        self.launch(self.ik.mv_q8s, (used.div_ceil(8) as u32, nb.div_ceil(8) as u32), 256, &args)
    }

    /// Every 32 tokens, between them: in each partly resident layer, up to 4 of the coolest experts on the GPU
    /// give their slots to hotter ones the CPU has been doing (their weights copied up from the CPU's copy); the
    /// counts then decay, so the GPU follows what the conversation uses (GHOST_INFER_INK_NO_SWAP=1: fixed).
    fn swap_experts(&mut self) -> Result<usize, String> {
        if self.steps % 32 != 0 || std::env::var_os("GHOST_INFER_INK_NO_SWAP").is_some() {
            return Ok(0);
        }
        let mut swapped = 0;
        for il in 0..self.c.n_layers {
            let Some(e) = self.layers[il].experts.as_ref() else { continue };
            let Some(res) = e.resident.clone() else { continue };
            let heat = &self.heat[il];
            let mut out: Vec<usize> = (0..res.len()).filter(|&x| res[x]).collect();
            let mut inn: Vec<usize> = (0..res.len()).filter(|&x| !res[x]).collect();
            out.sort_by(|a, b| heat[*a].total_cmp(&heat[*b]));
            inn.sort_by(|a, b| heat[*b].total_cmp(&heat[*a]));
            let pairs: Vec<(usize, usize)> = inn.iter().zip(&out).take(4).filter(|(i, o)| heat[**i] > heat[**o] * 1.5 + 2.0).map(|(&i, &o)| (i, o)).collect();
            for (e_in, e_out) in pairs {
                let (gu_m, dn_m) = self.cpu.expert_mats(il, e_in);
                let ex = self.layers[il].experts.as_mut().unwrap();
                let j = ex.slots[e_out];
                let (gu_at, dn_at) = (ex.gu.ptr + (j as usize * ex.gu_bytes) as u64, ex.down.ptr + (j as usize * ex.down_bytes) as u64);
                if ex.gu_iq3.is_some() {
                    self.mm.k.h2d(gu_at, gu_m.data)?;
                } else {
                    let mut buf = vec![0u8; gu_m.data.len()];
                    nvfp4_split_into(gu_m.data, gu_m.rows, gu_m.cols, &mut buf);
                    self.mm.k.h2d(gu_at, &buf)?;
                }
                let mut buf = vec![0u8; dn_m.data.len()];
                nvfp4_split_into(dn_m.data, dn_m.rows, dn_m.cols, &mut buf);
                self.mm.k.h2d(dn_at, &buf)?;
                ex.slots[e_out] = -1;
                ex.slots[e_in] = j;
                let r = ex.resident.as_mut().unwrap();
                r[e_out] = false;
                r[e_in] = true;
                let (slot_dev, slots) = (ex.slot.ptr, ex.slots.clone());
                self.mm.k.h2d(slot_dev, &slots)?;
                swapped += 1;
            }
        }
        for h in self.heat.iter_mut().flatten() {
            *h *= 0.85;
        }
        Ok(swapped)
    }

    /// One-graph tokens (one token a pass, mapped hand-offs on).
    fn one_graph(&self) -> bool {
        self.hops.is_some() && self.nb == 1
    }

    /// The GPU side of a hand-off: `x` (and the picks and weights when `picks` is given) out to the host, then
    /// a wait for its answer in the mapped `down` row.
    fn hop_gpu_out(&self, x: u64, picks: u64, wts: u64) -> Result<(), String> {
        let (c, hp) = (&self.c, self.hops.as_ref().unwrap());
        self.launch(self.ik.publish, (1, 1), 256, &[Arg::P(x), Arg::P(picks), Arg::P(wts), Arg::P(hp.up.dev), Arg::I(c.dim as i32), Arg::I(c.k as i32), Arg::P(hp.flag.dev), Arg::P(self.w.pos.ptr), Arg::I(self.hop)])
    }

    fn hop_gpu_wait(&self) -> Result<(), String> {
        let hp = self.hops.as_ref().unwrap();
        self.launch(self.ik.await_, (1, 1), 1, &[Arg::P(hp.ack.dev), Arg::P(self.w.pos.ptr), Arg::I(self.hop), Arg::P(hp.err.dev)])
    }

    /// The host side of a hand-off (not while capturing: nothing runs then): waits for the GPU's row, returns
    /// it with the picks and weights.
    fn hop_host_in(&mut self) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>), String> {
        let (d, k) = (self.c.dim, self.c.k);
        let hp = self.hops.as_ref().unwrap();
        let want = self.steps * 64 + self.hop + 1;
        let t0 = std::time::Instant::now();
        let mut spins = 0u32;
        while hp.flag.get::<i32>(0) != want {
            spins = spins.wrapping_add(1);
            if spins % 4096 == 0 {
                if hp.err.get::<i32>(0) != 0 {
                    return Err(format!("the GPU stopped waiting for the CPU at hand-off {}", self.hop));
                }
                if t0.elapsed() > std::time::Duration::from_secs(20) {
                    return Err(format!("the GPU never reached hand-off {}", self.hop));
                }
            }
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        self.prof[0] += t0.elapsed().as_secs_f64();
        let x: Vec<f32> = (0..d).map(|i| hp.up.get::<f32>(i)).collect();
        let picks: Vec<i32> = (0..k).map(|i| hp.up.get::<i32>(d + i)).collect();
        let wts: Vec<f32> = (0..k).map(|i| hp.up.get::<f32>(d + k + i)).collect();
        Ok((x, picks, wts))
    }

    /// The host's answer to the hand-off: its sum into the mapped row, then the hand-off's number.
    fn hop_host_out(&mut self, out: &[f32]) {
        let hp = self.hops.as_ref().unwrap();
        for (i, &v) in out.iter().enumerate() {
            hp.down.set::<f32>(i, v);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        hp.ack.set::<i32>(0, self.steps * 64 + self.hop + 1);
    }

    fn mm(&self, m: &GpuMatrix, x: u64, out: u64) -> Result<(), String> {
        if self.gmode == 2 {
            return Ok(());
        }
        self.timed(0, || self.mm.matmul_one(m, x, out, 1, false, &self.s))
    }

    fn elem(&self, f: Ptr, args: &[Arg], n: usize) -> Result<(), String> {
        self.launch(f, (n.div_ceil(256) as u32, 1), 256, args)
    }

    fn mix(&self, states: &[u64], query: u64, nq: usize, out: u64) -> Result<(), String> {
        if states.len() > 8 {
            return Err("more than 8 depth states".into());
        }
        let mut args: Vec<Arg> = (0..8).map(|i| Arg::P(*states.get(i).unwrap_or(&0))).collect();
        args.extend([Arg::I(states.len() as i32), Arg::P(query), Arg::P(out), Arg::I(self.c.dim as i32), Arg::I(self.nb as i32)]);
        self.launch(self.ik.mix, (nq as u32, self.nb as u32), 256, &args)
    }

    fn conv(&self, cur: u64, w: u64, dynamic: u64, slot: usize, ch_off: usize, out: u64, n: usize) -> Result<(), String> {
        let c = &self.c;
        let per_slot = (2 * c.n_kv * c.dh + 2 * c.dim) * (c.conv - 1);
        let state = self.conv.ptr + 4 * (slot * per_slot + ch_off * (c.conv - 1)) as u64;
        let div = 1.0 / (((if dynamic != 0 { 2 } else { 1 }) * c.conv + 1) as f32).sqrt();
        self.elem(self.ik.conv, &[Arg::P(cur), Arg::P(w), Arg::P(dynamic), Arg::P(state), Arg::P(out), Arg::I(n as i32), Arg::I(c.conv as i32), Arg::I(c.dh as i32), Arg::F(div), Arg::I(self.nb as i32)], n)
    }

    fn clip(&self, x: u64, scale: u64, rows: usize, n: usize, thr: f32, tau: bool) -> Result<(), String> {
        let c = &self.c;
        let all = rows * self.nb;
        self.launch(self.ik.clip, (all.div_ceil(8) as u32, 1), 256, &[Arg::P(x), Arg::P(scale), Arg::I(all as i32), Arg::I(n as i32), Arg::F(thr), Arg::P(self.w.pos.ptr), Arg::I(tau as i32), Arg::F(c.log_alpha), Arg::F(c.log_floor), Arg::I(rows as i32)])
    }

    /// Controls first..first + curs.len() of a loop layer, fused: cur j against message row j of `msg`, into
    /// row j of `out` (GHOST_INFER_INK_NO_FUSED_CTRL=1: one control at a time, unfused).
    fn controls(&self, curs: &[u64], msg: u64, lp: &GLoop, first: usize, out: u64) -> Result<(), String> {
        let d = self.c.dim;
        let rank = lp.down[first].bytes / 2 / (2 * d);
        if self.nb == 1 && (std::env::var_os("GHOST_INFER_INK_NO_FUSED_CTRL").is_some() || rank * curs.len() > 256) {
            for (j, &cur) in curs.iter().enumerate() {
                self.control(cur, msg + (4 * j * d) as u64, lp, first + j, out + (4 * j * d) as u64)?;
            }
            return Ok(());
        }
        let n = curs.len();
        let at = |v: &[DeviceBuf], j: usize| v[first + j.min(n - 1)].ptr;
        let cur = |j: usize| curs[j.min(n - 1)];
        let w = &self.w;
        let nb = self.nb as u32;
        self.launch3(self.ik.ctrl_down, (rank as u32, n as u32, nb), 256, &[Arg::P(cur(0)), Arg::P(cur(1)), Arg::P(cur(2)), Arg::P(msg), Arg::P(at(&lp.down, 0)), Arg::P(at(&lp.down, 1)), Arg::P(at(&lp.down, 2)), Arg::P(w.c64.ptr), Arg::I(d as i32), Arg::I(rank as i32), Arg::I(nb as i32)])?;
        self.launch3(
            self.ik.ctrl_up,
            (d.div_ceil(8) as u32, n as u32, nb),
            256,
            &[Arg::P(cur(0)), Arg::P(cur(1)), Arg::P(cur(2)), Arg::P(msg), Arg::P(at(&lp.gain, 0)), Arg::P(at(&lp.gain, 1)), Arg::P(at(&lp.gain, 2)), Arg::P(at(&lp.forget, 0)), Arg::P(at(&lp.forget, 1)), Arg::P(at(&lp.forget, 2)), Arg::P(w.c64.ptr), Arg::P(out), Arg::I(d as i32), Arg::I(rank as i32), Arg::I(nb as i32)],
        )
    }

    fn control(&self, cur: u64, msg: u64, lp: &GLoop, ctrl: usize, out: u64) -> Result<(), String> {
        let (d, w) = (self.c.dim, &self.w);
        self.launch(self.ik.rms2, (2, 1), 256, &[Arg::P(cur), Arg::P(msg), Arg::P(w.cg.ptr), Arg::I(d as i32)])?;
        let rank = lp.down[ctrl].bytes / 2 / (2 * d);
        let mv16 = |m: u64, x: u64, out: u64, rows: usize, cols: usize| self.launch(self.ik.mv_f16, (rows.div_ceil(8) as u32, 1), 256, &[Arg::P(m), Arg::P(x), Arg::P(out), Arg::I(rows as i32), Arg::I(cols as i32)]);
        mv16(lp.down[ctrl].ptr, w.cg.ptr, w.c64.ptr, rank, 2 * d)?;
        mv16(lp.gain[ctrl].ptr, w.c64.ptr, w.cgain.ptr, d, rank)?;
        mv16(lp.forget[ctrl].ptr, w.c64.ptr, w.cforget.ptr, d, rank)?;
        self.elem(self.ik.control, &[Arg::P(cur), Arg::P(msg), Arg::P(w.cgain.ptr), Arg::P(w.cforget.ptr), Arg::P(out), Arg::I(d as i32)], d)
    }

    fn note(&mut self, name: impl FnOnce() -> String, ptr: u64, n: usize) -> Result<(), String> {
        if self.trace.is_some() {
            // a batch's rows one token after another, as one-token steps record them
            let mut v = vec![0f32; n * self.nb];
            self.mm.k.d2h(&mut v, ptr)?;
            let name = name();
            for row in v.chunks(n) {
                self.trace.as_mut().unwrap().push((name.clone(), row.to_vec()));
            }
        }
        Ok(())
    }

    /// Layer `il` at depth `ip` in `slot`, on w.x; `blocks`: the depth states' device rows; `pool_next`: the next
    /// free pool row.
    #[allow(clippy::too_many_arguments)]
    fn layer(&mut self, il: usize, slot: usize, ip: usize, blocks: &mut Vec<u64>, pool_next: &mut usize, msgs: Option<usize>) -> Result<(), String> {
        let c = self.c.clone();
        let d = c.dim;
        let kvd = c.n_kv * c.dh;
        let (x, inp, inpc, nrm) = (self.w.x.ptr, self.w.inp.ptr, self.w.inpc.ptr, self.w.nrm.ptr);
        let (nb, dn) = (self.nb, d * self.nb);
        // multi-row buffers hold row j for all the pass's tokens together
        let row = |p: u64, j: usize| p + (4 * j * dn) as u64;
        let l = &self.layers[il];
        let mut inputs = [x, x, x];
        if let Some((q, nq)) = &l.attn_res_q {
            let mut st = blocks.clone();
            st.push(x);
            self.mix(&st, q.ptr, *nq, inp)?;
            for (j, v) in inputs.iter_mut().enumerate() {
                *v = row(inp, if l.swa { 0 } else { j });
            }
        }
        if let Some(li) = msgs {
            self.controls(&inputs, self.w.msgs.ptr, &self.loops[li], 0, inpc)?;
            for (j, v) in inputs.iter_mut().enumerate() {
                *v = row(inpc, j);
            }
        }
        if l.swa && msgs.is_none() {
            self.rms(inputs[0], l.norms[0].ptr, nrm, 1, d, c.eps)?;
            inputs = [nrm; 3];
        } else {
            for (j, v) in inputs.iter_mut().enumerate() {
                self.rms(*v, l.norms[if l.swa { 0 } else { j }].ptr, row(nrm, j), 1, d, c.eps)?;
                *v = row(nrm, j);
            }
        }
        let w = &self.w;
        match &l.fused {
            Some((fm, rows)) => {
                // k and v in the stack only while one input feeds all three (not after a loop pass's controls)
                let n = if rows.len() == 6 && inputs[1] == inputs[0] && inputs[2] == inputs[0] { 6 } else { 4 };
                let outs = [w.q.ptr, w.r.ptr, w.sinks.ptr, w.gate.ptr, w.kraw.ptr, w.vraw.ptr];
                self.mm_stacked(fm, inputs[0], &outs[..n], &rows[..n])?;
                if n == 4 {
                    self.mms(&[(&l.wk, inputs[1], w.kraw.ptr), (&l.wv, inputs[2], w.vraw.ptr)])?;
                }
            }
            None => self.mms(&[(&l.wq, inputs[0], w.q.ptr), (&l.rel, inputs[0], w.r.ptr), (&l.sink, inputs[0], w.sinks.ptr), (&l.wgate, inputs[0], w.gate.ptr), (&l.wk, inputs[1], w.kraw.ptr), (&l.wv, inputs[2], w.vraw.ptr)])?,
        }
        let (mut kdyn, mut vdyn) = (0, 0);
        if let (Some(kp), Some(vp)) = (&l.k_proj, &l.v_proj) {
            self.mv_f32(kp.ptr, inputs[1], w.kdyn.ptr, c.n_kv * c.conv, d)?;
            self.mv_f32(vp.ptr, inputs[2], w.vdyn.ptr, c.n_kv * c.conv, d)?;
            (kdyn, vdyn) = (w.kdyn.ptr, w.vdyn.ptr);
        }
        self.conv(w.kraw.ptr, l.k_conv.ptr, kdyn, slot, 0, w.k.ptr, kvd)?;
        self.conv(w.vraw.ptr, l.v_conv.ptr, vdyn, slot, kvd, w.v.ptr, kvd)?;
        self.clip(w.q.ptr, l.q_clip.ptr, c.n_head, c.dh, c.qk_clip, !l.swa)?;
        self.clip(w.k.ptr, 0, c.n_kv, c.dh, c.qk_clip, false)?;
        self.rms(w.v.ptr, 0, w.v.ptr, c.n_kv, c.dh, c.eps)?;
        self.clip(w.r.ptr, l.rel_clip.ptr, c.n_head, c.n_rel, c.rel_clip, !l.swa)?;
        let (q, k, v) = (w.q.ptr, w.k.ptr, w.v.ptr);
        self.note(|| format!("Qcur-{slot}"), q, c.n_head * c.dh)?;
        self.note(|| format!("Kcur-{slot}"), k, kvd)?;
        self.note(|| format!("Vcur-{slot}"), v, kvd)?;
        let (l, w) = (&self.layers[il], &self.w);
        let (kc, vc, cap) = (&self.kv[slot].0, &self.kv[slot].1, self.kv[slot].2);
        self.elem(self.ik.store_kv, &[Arg::P(w.k.ptr), Arg::P(w.v.ptr), Arg::P(kc.ptr), Arg::P(vc.ptr), Arg::P(w.pos.ptr), Arg::I(cap as i32), Arg::I(kvd as i32), Arg::I(nb as i32)], kvd * nb)?;
        let (window, extent) = if l.swa { (c.n_swa, c.n_swa) } else { (0, c.rel_extent) };
        // queries 8 at a time (the scores' scratch holds 8)
        for b0 in (0..nb).step_by(8) {
        self.launch(
            self.ik.attn,
            (c.n_head as u32, (nb - b0).min(8) as u32),
            128,
            &[
                Arg::P(w.q.ptr),
                Arg::P(w.r.ptr),
                Arg::P(w.sinks.ptr),
                Arg::P(l.rel_bias.ptr),
                Arg::P(kc.ptr),
                Arg::P(vc.ptr),
                Arg::P(w.gate.ptr),
                Arg::P(w.att.ptr),
                Arg::P(w.scr.ptr),
                Arg::P(w.pos.ptr),
                Arg::I(cap as i32),
                Arg::I(window as i32),
                Arg::I(extent as i32),
                Arg::I(c.n_rel as i32),
                Arg::I(kvd as i32),
                Arg::I((c.n_head / c.n_kv) as i32),
                Arg::I(self.ctx as i32),
                Arg::F(1.0 / (c.dh as f32).sqrt()),
                Arg::I(b0 as i32),
            ],
        )?;
        }
        self.mms(&[(&l.wo, w.att.ptr, w.a.ptr)])?;
        self.conv(w.a.ptr, l.post_conv.ptr, 0, slot, 2 * kvd, w.a2.ptr, d)?;
        let a2 = self.w.a2.ptr;
        self.note(|| format!("attn_out-{slot}"), a2, d)?;
        let w = &self.w;
        let fi = if ip % c.block == 0 {
            let r = row(w.pool.ptr, *pool_next);
            *pool_next += 1;
            self.elem(self.ik.copy, &[Arg::P(x), Arg::P(r), Arg::I(dn as i32)], dn)?;
            blocks.push(r);
            w.a2.ptr
        } else {
            self.elem(self.ik.add, &[Arg::P(x), Arg::P(w.a2.ptr), Arg::P(w.fi.ptr), Arg::I(dn as i32)], dn)?;
            w.fi.ptr
        };
        self.note(|| format!("ffn_inp-{slot}"), fi, d)?;
        let (l, w) = (&self.layers[il], &self.w);
        let mut st = blocks.clone();
        st.push(fi);
        self.mix(&st, l.ffn_res_q.ptr, 1, w.f.ptr)?;
        let mut f = w.f.ptr;
        if let Some(li) = msgs {
            self.controls(&[f], row(w.msgs.ptr, 3), &self.loops[li], 3, w.f2.ptr)?;
            f = w.f2.ptr;
        }
        self.rms(f, l.ffn_norm.ptr, w.fnorm.ptr, 1, d, c.eps)?;
        match &l.experts {
            Some(_) => {
                self.mv_f32(l.router.ptr, w.fnorm.ptr, w.logits_e.ptr, c.n_exp, d)?;
                self.launch(self.ik.topk, (nb as u32, 1), c.n_exp.next_multiple_of(32) as u32, &[Arg::P(w.logits_e.ptr), Arg::P(l.bias.ptr), Arg::I(c.n_exp as i32), Arg::I(c.k as i32), Arg::F(c.exp_scale), Arg::P(w.picks.ptr), Arg::P(w.weights.ptr)])?;
                self.launch(self.ik.q8, ((dn / 32).div_ceil(8) as u32, 1), 256, &[Arg::P(w.fnorm.ptr), Arg::P(w.xq.ptr), Arg::P(w.xd.ptr), Arg::I(dn as i32)])?;
                if self.counts.is_some() && nb == 1 {
                    let mut picks = vec![0i32; c.k];
                    self.mm.k.d2h(&mut picks, w.picks.ptr)?;
                    let cn = self.counts.as_mut().unwrap();
                    picks.iter().for_each(|&e| cn[il][e as usize] += 1);
                }
                let (l, w) = (&self.layers[il], &self.w);
                let e = l.experts.as_ref().unwrap();
                let mapped_partial = e.resident.is_some() && self.one_graph();
                if mapped_partial {
                    self.hop_gpu_out(w.fnorm.ptr, w.picks.ptr, w.weights.ptr)?;
                }
                let rows_gu = 2 * c.ff_exp;
                // every token of the pass in one launch (blockIdx.z)
                let (picks_b, weights_b, xq_b, xd_b, mo_b) = (w.picks.ptr, w.weights.ptr, w.xq.ptr, w.xd.ptr, w.mo.ptr);
                let z = nb as u32;
                let grouped = nb > 1 && e.gu_iq3.is_none() && d <= 2304 && c.n_exp <= 256 && std::env::var_os("GHOST_INFER_INK_NO_GROUP").is_none();
                if grouped {
                    let (cnt, off, list) = (w.grp.ptr, w.grp.ptr + (4 * c.n_exp) as u64, w.grp.ptr + (8 * c.n_exp) as u64);
                    self.launch(self.ik.group, (1, 1), 256, &[Arg::P(w.picks.ptr), Arg::I((nb * c.k) as i32), Arg::I(c.n_exp as i32), Arg::P(cnt), Arg::P(off), Arg::P(list)])?;
                    self.launch(
                        self.ik.moe_gu_g,
                        (rows_gu.div_ceil(32) as u32, c.n_exp as u32),
                        256,
                        &[Arg::P(e.gu.ptr), Arg::P(e.gu_bytes as u64), Arg::P(cnt), Arg::P(off), Arg::P(list), Arg::I(c.k as i32), Arg::P(e.gu_scale.ptr), Arg::P(w.xq.ptr), Arg::P(w.xd.ptr), Arg::P(w.gu.ptr), Arg::I(rows_gu as i32), Arg::I(d as i32), Arg::P(e.slot.ptr)],
                    )?;
                }
                match &e.gu_iq3 {
                    _ if grouped => {}
                    None => self.launch3(
                        self.ik.moe_gu,
                        (rows_gu.div_ceil(32) as u32, c.k as u32, z),
                        256,
                        &[Arg::P(e.gu.ptr), Arg::P(e.gu_bytes as u64), Arg::P(picks_b), Arg::P(e.gu_scale.ptr), Arg::P(xq_b), Arg::P(xd_b), Arg::P(w.gu.ptr), Arg::I(rows_gu as i32), Arg::I(d as i32), Arg::P(e.slot.ptr)],
                    )?,
                    Some(grid) => self.launch3(
                        self.ik.moe_gu_iq3,
                        (rows_gu.div_ceil(8) as u32, c.k as u32, z),
                        256,
                        &[Arg::P(e.gu.ptr), Arg::P(e.gu_bytes as u64), Arg::P(picks_b), Arg::P(e.gu_scale.ptr), Arg::P(xq_b), Arg::P(xd_b), Arg::P(w.gu.ptr), Arg::I(rows_gu as i32), Arg::I(d as i32), Arg::P(grid.ptr), Arg::P(e.slot.ptr)],
                    )?,
                }
                // the batch's picks one after another: k * nb of them
                self.launch(self.ik.situ_q8, ((c.k * nb * c.ff_exp / 32).div_ceil(8) as u32, 1), 256, &[Arg::P(w.gu.ptr), Arg::P(w.hq.ptr), Arg::P(w.hd.ptr), Arg::I(c.ff_exp as i32), Arg::I((c.k * nb) as i32), Arg::F(c.situ_beta), Arg::F(c.situ_lb)])?;
                self.launch3(
                    self.ik.moe_down,
                    (d.div_ceil(8 * DOWN_ROWS) as u32, 1, z),
                    256,
                    &[Arg::P(e.down.ptr), Arg::P(e.down_bytes as u64), Arg::P(picks_b), Arg::P(weights_b), Arg::P(e.down_scale.ptr), Arg::P(w.hq.ptr), Arg::P(w.hd.ptr), Arg::P(mo_b), Arg::I(d as i32), Arg::I(c.ff_exp as i32), Arg::I(c.k as i32), Arg::P(e.slot.ptr)],
                )?;
                if let (true, Some(res)) = (mapped_partial, &e.resident) {
                    // the GPU's part runs while the CPU does the picks it does not hold; the GPU then waits for its sum
                    let res = res.clone();
                    self.hop_gpu_wait()?;
                    let (mo, down) = (self.w.mo.ptr, self.hops.as_ref().unwrap().down.dev);
                    self.elem(self.ik.add, &[Arg::P(mo), Arg::P(down), Arg::P(mo), Arg::I(d as i32)], d)?;
                    if self.gmode != 1 {
                        let (xn, picks, wts) = self.hop_host_in()?;
                        picks.iter().for_each(|&e| self.heat[il][e as usize] += 1.0);
                        let t1 = std::time::Instant::now();
                        let mine: Vec<(usize, f32)> = picks.iter().zip(&wts).filter(|(e, _)| !res[**e as usize]).map(|(&e, &w)| (e as usize, w)).collect();
                        let out = if mine.is_empty() { vec![0f32; d] } else { self.cpu.moe_picks(il, &xn, &mine, &mut self.sc) };
                        self.prof[1] += t1.elapsed().as_secs_f64();
                        self.prof_layer[il].0 += t1.elapsed().as_secs_f64();
                        self.prof_layer[il].1 += 1;
                        self.hop_host_out(&out);
                    }
                    self.hop += 1;
                } else if let Some(res) = &e.resident {
                    // the GPU's part runs while the CPU does the picks it does not hold
                    let res = res.clone();
                    let (picks_p, weights_p, fnorm, mcpu) = (w.picks.ptr, w.weights.ptr, w.fnorm.ptr, w.mcpu.ptr);
                    self.seg_end()?;
                    let t0 = std::time::Instant::now();
                    let (mut picks, mut wts, mut xn) = (vec![0i32; c.k * nb], vec![0f32; c.k * nb], vec![0f32; dn]);
                    self.mm.k.d2h(&mut picks, picks_p)?;
                    self.mm.k.d2h(&mut wts, weights_p)?;
                    self.mm.k.d2h(&mut xn, fnorm)?;
                    let t1 = std::time::Instant::now();
                    picks.iter().for_each(|&e| self.heat[il][e as usize] += 1.0);
                    let mine: Vec<Vec<(usize, f32)>> = picks.chunks(c.k).zip(wts.chunks(c.k)).map(|(p, w)| p.iter().zip(w).filter(|(e, _)| !res[**e as usize]).map(|(&e, &w)| (e as usize, w)).collect()).collect();
                    let out = if mine.iter().all(|m| m.is_empty()) {
                        vec![0f32; dn]
                    } else if nb == 1 {
                        self.cpu.moe_picks(il, &xn, &mine[0], &mut self.sc)
                    } else {
                        self.cpu.moe_batch(il, &xn, Some(&mine), &mut self.sc)
                    };
                    self.prof[0] += (t1 - t0).as_secs_f64();
                    self.prof[1] += t1.elapsed().as_secs_f64();
                    self.prof_layer[il].0 += t1.elapsed().as_secs_f64();
                    self.prof_layer[il].1 += 1;
                    let th = std::time::Instant::now();
                    self.mm.k.h2d(mcpu, &out)?;
                    self.hostprof[2] += th.elapsed().as_secs_f64();
                    self.seg_begin()?;
                    let w = &self.w;
                    self.elem(self.ik.add, &[Arg::P(w.mo.ptr), Arg::P(w.mcpu.ptr), Arg::P(w.mo.ptr), Arg::I(dn as i32)], dn)?;
                }
            }
            None if self.one_graph() => {
                let fnorm = self.w.fnorm.ptr;
                self.hop_gpu_out(fnorm, 0, 0)?;
                self.hop_gpu_wait()?;
                let (mo, down) = (self.w.mo.ptr, self.hops.as_ref().unwrap().down.dev);
                self.elem(self.ik.copy, &[Arg::P(down), Arg::P(mo), Arg::I(d as i32)], d)?;
                if self.gmode != 1 {
                    let (xn, _, _) = self.hop_host_in()?;
                    let t1 = std::time::Instant::now();
                    if let Some(cn) = self.counts.as_mut() {
                        for (e, _) in self.cpu.route(il, &xn) {
                            cn[il][e] += 1;
                        }
                    }
                    let out = self.cpu.moe_by_layer(il, &xn, &mut self.sc);
                    self.prof[1] += t1.elapsed().as_secs_f64();
                    self.prof_layer[il].0 += t1.elapsed().as_secs_f64();
                    self.prof_layer[il].1 += 1;
                    self.hop_host_out(&out);
                }
                self.hop += 1;
            }
            None => {
                let mut xn = vec![0f32; dn];
                let fnorm = w.fnorm.ptr;
                self.seg_end()?;
                let t0 = std::time::Instant::now();
                self.mm.k.d2h(&mut xn, fnorm)?;
                let t1 = std::time::Instant::now();
                if let Some(cn) = self.counts.as_mut().filter(|_| nb == 1) {
                    for (e, _) in self.cpu.route(il, &xn) {
                        cn[il][e] += 1;
                    }
                }
                let out = if nb == 1 { self.cpu.moe_by_layer(il, &xn, &mut self.sc) } else { self.cpu.moe_batch(il, &xn, None, &mut self.sc) };
                self.prof[0] += (t1 - t0).as_secs_f64();
                self.prof[1] += t1.elapsed().as_secs_f64();
                self.prof_layer[il].0 += t1.elapsed().as_secs_f64();
                self.prof_layer[il].1 += 1;
                let th = std::time::Instant::now();
                self.mm.k.h2d(self.w.mo.ptr, &out)?;
                self.hostprof[2] += th.elapsed().as_secs_f64();
                self.seg_begin()?;
            }
        }
        let (l, w) = (&self.layers[il], &self.w);
        self.conv(w.mo.ptr, l.ffn_conv.ptr, 0, slot, 2 * kvd + d, w.mo2.ptr, d)?;
        let mo2 = w.mo2.ptr;
        self.note(|| format!("ffn_out-{slot}"), mo2, d)?;
        let w = &self.w;
        self.elem(self.ik.add, &[Arg::P(fi), Arg::P(w.mo2.ptr), Arg::P(w.x.ptr), Arg::I(dn as i32)], dn)?;
        self.note(|| format!("l_out-{slot}"), x, d)?;
        Ok(())
    }

    /// One token; returns its logits (unpadded vocabulary).
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let t = std::time::Instant::now();
        let r = self.step_inner(&[token]);
        self.hostprof[0] += t.elapsed().as_secs_f64();
        r
    }

    /// A prompt's tokens; returns the last one's logits. Passes of up to PREFILL_B tokens (each expert read once
    /// for all of a pass's tokens that picked it), the same arithmetic per token as `step`.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut logits = Vec::new();
        for chunk in tokens.chunks(PREFILL_B) {
            if chunk.len() == 1 || self.wb.is_none() || (self.trace.is_some() && std::env::var_os("GHOST_INFER_INK_PREFILL_TRACE").is_none()) || self.counts.is_some() {
                for &t in chunk {
                    logits = self.step(t)?;
                }
                continue;
            }
            std::mem::swap(&mut self.w, self.wb.as_mut().unwrap());
            std::mem::swap(&mut self.graphs, &mut self.graphs_b);
            self.nb = chunk.len();
            let t = std::time::Instant::now();
            let r = self.step_inner(chunk);
            self.hostprof[0] += t.elapsed().as_secs_f64();
            self.nb = 1;
            std::mem::swap(&mut self.graphs, &mut self.graphs_b);
            std::mem::swap(&mut self.w, self.wb.as_mut().unwrap());
            logits = r?;
        }
        Ok(logits)
    }

    fn step_inner(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let c = self.c.clone();
        let nb = tokens.len();
        if self.pos + nb > self.ctx {
            return Err(format!("the context of {} tokens is full", self.ctx));
        }
        self.steps += 1;
        if nb == 1 && self.trace.is_none() {
            self.swaps += self.swap_experts()?;
        }
        self.mm.k.h2d(self.w.pos.ptr, &[self.pos as i32, self.steps])?;
        let e: Vec<f32> = tokens.iter().flat_map(|&t| self.cpu.embed(t)).collect();
        self.mm.k.h2d(self.w.mo.ptr, &e)?;
        // graphs from the second token on; not while tracing or timing kernels (they read and sync between launches)
        self.gmode = match self.gmode {
            _ if (nb > 1 && nb != PREFILL_B) || std::env::var_os("GHOST_INFER_INK_NO_BATCH_GRAPHS").is_some_and(|_| nb > 1) || self.trace.is_some() || self.kprof.is_some() || self.counts.is_some() || std::env::var_os("GHOST_INFER_INK_NO_GRAPHS").is_some() => 0,
            _ if self.graphs.is_empty() => 1,
            _ => 2,
        };
        if self.one_graph() {
            // the whole token one graph: captured first (the hand-offs only recorded), then run with the host
            // answering the hand-offs as the GPU reaches them
            if self.gmode == 1 {
                let a = api()?;
                const RELAXED: i32 = 2;
                a.check(unsafe { (a.f.cuStreamBeginCapture_v2)(self.mm.k.stream, RELAXED) }, "begin graph capture")?;
                self.hop = 0;
                let walked = self.walk(nb);
                let mut g: Ptr = std::ptr::null_mut();
                let ended = unsafe { (a.f.cuStreamEndCapture)(self.mm.k.stream, &mut g) };
                walked?;
                a.check(ended, "end graph capture")?;
                let mut exec: Ptr = std::ptr::null_mut();
                let made = unsafe { (a.f.cuGraphInstantiateWithFlags)(&mut exec, g, 0) };
                unsafe { (a.f.cuGraphDestroy)(g) };
                a.check(made, "graph instantiate")?;
                self.graphs.push(exec);
                self.gmode = 2;
            }
            if self.gmode == 2 {
                let a = api()?;
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                let t = std::time::Instant::now();
                a.check(unsafe { (a.f.cuGraphLaunch)(self.graphs[0], self.mm.k.stream) }, "graph launch")?;
                self.hostprof[1] += t.elapsed().as_secs_f64();
            }
            self.hop = 0;
            self.walk(nb)?;
        } else {
            self.seg = 0;
            self.seg_begin()?;
            self.walk(nb)?;
            self.seg_end()?;
        }
        let mut logits = vec![0f32; c.vocab];
        let tl = std::time::Instant::now();
        self.mm.k.d2h(&mut logits, self.w.logits.ptr)?;
        self.hostprof[3] += tl.elapsed().as_secs_f64();
        self.pos += nb;
        Ok(logits)
    }

    /// The token's (or batch's) work after its embedding: the layers, the output mix and the head.
    fn walk(&mut self, nb: usize) -> Result<(), String> {
        let c = self.c.clone();
        let d = c.dim;
        let dn = d * nb;
        self.rms(self.w.mo.ptr, self.tok_norm.ptr, self.w.x.ptr, 1, d, c.eps)?;
        let x = self.w.x.ptr;
        self.note(|| "embd_norm".into(), x, d)?;
        let row = |p: u64, j: usize| p + (4 * j * dn) as u64;
        let pool = self.w.pool.ptr;
        let mut next = 0usize;
        let mut blocks: Vec<u64> = Vec::new();
        for il in 0..c.loop_start {
            self.layer(il, il, il, &mut blocks, &mut next, None)?;
        }
        let prelude = row(pool, next);
        next += 1;
        self.elem(self.ik.copy, &[Arg::P(x), Arg::P(prelude), Arg::I(dn as i32)], dn)?;
        let prelude_blocks = blocks.clone();
        let core = c.loop_end - c.loop_start;
        for pass in 0..c.passes {
            let mut bank: Vec<u64> = Vec::new();
            if pass > 0 {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                let out = row(pool, next);
                next += 1;
                self.elem(self.ik.copy, &[Arg::P(x), Arg::P(out), Arg::I(dn as i32)], dn)?;
                bank.push(out);
                self.elem(self.ik.copy, &[Arg::P(prelude), Arg::P(x), Arg::I(dn as i32)], dn)?;
                blocks = prelude_blocks.clone();
            }
            for ip in c.loop_start..c.loop_end {
                if pass == 0 {
                    self.layer(ip, ip, ip, &mut blocks, &mut next, None)?;
                    continue;
                }
                let slot = c.n_layers + (pass - 1) * core + (ip - c.loop_start);
                let li = ip - c.loop_start;
                self.mix(&bank, self.loops[li].msg_q.ptr, 4, self.w.msgs.ptr)?;
                self.layer(ip, slot, ip, &mut blocks, &mut next, Some(li))?;
            }
        }
        for il in c.loop_end..c.n_layers {
            self.layer(il, il, il, &mut blocks, &mut next, None)?;
        }
        if next > POOL_ROWS {
            return Err(format!("{next} depth states, the pool holds {POOL_ROWS}"));
        }
        let mut st = blocks.clone();
        st.push(x);
        self.mix(&st, self.output_res_q.ptr, 1, self.w.f.ptr)?;
        self.rms(self.w.f.ptr, self.output_norm.ptr, self.w.fnorm.ptr, 1, d, c.eps)?;
        let fnorm = self.w.fnorm.ptr;
        self.note(|| "result_norm".into(), fnorm, d)?;
        // the logits of the last token only
        let last = fnorm + (4 * (nb - 1) * d) as u64;
        let nb_now = std::mem::replace(&mut self.nb, 1);
        let head = match &self.output {
            // int8 activations, as llama.cpp's head (GHOST_INFER_INK_F32_HEAD=1: the f32 kernel)
            Ok((buf, rows)) if std::env::var_os("GHOST_INFER_INK_F32_HEAD").is_none() => self
                .launch(self.ik.q8, ((d / 32).div_ceil(8) as u32, 1), 256, &[Arg::P(last), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::I(d as i32)])
                .and_then(|_| self.launch(self.ik.mv_q5k8, (rows.div_ceil(32) as u32, 1), 256, &[Arg::P(buf.ptr), Arg::P(self.w.aq.ptr), Arg::P(self.w.ad.ptr), Arg::P(self.w.logits.ptr), Arg::I(*rows as i32), Arg::I(d as i32)])),
            Ok((buf, rows)) => self.launch(self.ik.mv_q5k, (rows.div_ceil(8) as u32, 1), 256, &[Arg::P(buf.ptr), Arg::P(last), Arg::P(self.w.logits.ptr), Arg::I(*rows as i32), Arg::I(d as i32)]),
            Err(m) => self.mm(m, last, self.w.logits.ptr),
        };
        self.nb = nb_now;
        head?;
        if c.softcap != 0.0 {
            self.elem(self.ik.softcap, &[Arg::P(self.w.logits.ptr), Arg::I(c.vocab as i32), Arg::F(c.softcap)], c.vocab)?;
        }
        Ok(())
    }
}

fn cpu_output_rows(g: &Gguf) -> Result<usize, String> {
    Ok(g.tensor("output.weight").ok_or("missing output.weight")?.dims[1] as usize)
}

/// `inkling-gpu-check <model.gguf> <ids>` (as inkling-check, on the GPU) and `inkling-gpu-gen <model.gguf> <ids> <n>`.
pub fn run(path: &std::path::Path, ids: &str, n_gen: Option<usize>) -> Result<(), String> {
    crate::inkling::set_threads();
    let g = Gguf::open(path)?;
    let t0 = std::time::Instant::now();
    let ctx = std::env::var("GHOST_INFER_INK_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
    let mut m = GpuInkling::load(&g, ctx)?;
    eprintln!("[loaded in {:.1} s; experts on the GPU for layers {:?}; {} MB of VRAM free]", t0.elapsed().as_secs_f64(), m.gpu_expert_layers, free_vram()? >> 20);
    let ids: Vec<u32> = ids.split(',').map(|s| s.trim().parse().map_err(|e| format!("bad token id {s}: {e}"))).collect::<Result<_, _>>()?;
    if n_gen.is_none() {
        m.trace = Some(Vec::new());
    }
    let mut per_name: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    let mut logits = Vec::new();
    let tp = std::time::Instant::now();
    let batch_trace = std::env::var_os("GHOST_INFER_INK_PREFILL_TRACE").is_some();
    if m.trace.is_none() || batch_trace {
        logits = m.prefill(&ids)?;
        if let Some(t) = m.trace.as_mut() {
            for (name, v) in t.drain(..) {
                match per_name.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, rows)) => rows.push(v),
                    None => per_name.push((name, vec![v])),
                }
            }
        }
    }
    let tracing = m.trace.is_some() && !batch_trace;
    for &id in ids.iter().filter(|_| tracing) {
        logits = m.step(id)?;
        if let Some(t) = m.trace.as_mut() {
            for (name, v) in t.drain(..) {
                match per_name.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, rows)) => rows.push(v),
                    None => per_name.push((name, vec![v])),
                }
            }
        }
    }
    eprintln!("[prompt: {} tokens in {:.2} s ({:.1} tok/s); waiting for the GPU {:.2} s, CPU experts {:.2} s]", ids.len(), tp.elapsed().as_secs_f64(), ids.len() as f64 / tp.elapsed().as_secs_f64(), m.prof[0], m.prof[1]);
    let per: Vec<String> = m.prof_layer.iter().enumerate().filter(|(_, p)| p.1 > 0).map(|(i, p)| format!("{i}:{:.1}ms", 1000.0 * p.0 / p.1 as f64)).collect();
    eprintln!("[prompt CPU experts per pass: {}]", per.join(" "));
    let bp: Vec<u64> = crate::quant::BATCH_PROF.iter().map(|v| v.load(std::sync::atomic::Ordering::Relaxed)).collect();
    if bp[4] > 0 {
        eprintln!("[prompt CPU expert batches: {} calls, {:.1} pairs each; setup {:.2} s, gate_up {:.2} s, h to int8 {:.2} s, down {:.2} s]", bp[4], bp[5] as f64 / bp[4] as f64, bp[0] as f64 * 1e-9, bp[1] as f64 * 1e-9, bp[2] as f64 * 1e-9, bp[3] as f64 * 1e-9);
    }
    match n_gen {
        None => {
            for (name, rows) in &per_name {
                println!("{name}");
                for r in rows {
                    let n = r.len();
                    println!("  [{:10.4}, {:10.4}, {:10.4}, ..., {:10.4}, {:10.4}, {:10.4}]", r[0], r[1], r[2], r[n - 3], r[n - 2], r[n - 1]);
                }
                println!("  sum = {:.6}", rows.iter().flatten().map(|&v| v as f64).sum::<f64>());
            }
            let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!("logits [{:.4}, {:.4}, {:.4}, ...] top: {:?}", logits[0], logits[1], logits[2], &top[..5]);
        }
        Some(n) => {
            let t = std::time::Instant::now();
            let mut out = Vec::new();
            for _ in 0..n {
                let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
                out.push(next);
                logits = m.step(next)?;
            }
            println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
            eprintln!("[{n} tokens in {:.2} s, {:.2} tok/s]", t.elapsed().as_secs_f64(), n as f64 / t.elapsed().as_secs_f64());
            eprintln!("[waiting for the GPU {:.2} s, CPU experts {:.2} s (prompt included)]", m.prof[0], m.prof[1]);
            let h = m.hostprof;
            eprintln!("[steps {:.2} s; graph launches {:.3} s, copies after CPU experts {:.3} s, logits back (with the last stretch) {:.3} s]", h[0], h[1], h[2], h[3]);
            let per: Vec<String> = m.prof_layer.iter().enumerate().filter(|(_, p)| p.1 > 0).map(|(i, p)| format!("{i}:{:.1}ms", 1000.0 * p.0 / p.1 as f64)).collect();
            eprintln!("[CPU experts per pass: {}]", per.join(" "));
            if m.swaps > 0 {
                eprintln!("[{} experts swapped onto the GPU as they got hot]", m.swaps);
            }
            if let Some(st) = &m.segtime {
                let per: Vec<String> = st.iter().map(|(ms, n)| format!("{:.2}", ms / *n as f64)).collect();
                eprintln!("[GPU ms per stretch: {}; {:.1} ms a token]", per.join(" "), st.iter().map(|(ms, n)| ms / *n as f64).sum::<f64>());
            }
            if let Some(r) = m.kprof_report() {
                eprintln!("{r}");
            }
            if let (Some(cn), Some(path)) = (&m.counts, std::env::var_os("GHOST_INFER_INK_COUNT")) {
                let text: String = cn.iter().enumerate().map(|(i, v)| format!("{i}: {}\n", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "))).collect();
                std::fs::write(&path, text).map_err(|e| format!("writing the counts: {e}"))?;
                eprintln!("[expert counts written to {}]", std::path::Path::new(&path).display());
            }
        }
    }
    Ok(())
}
