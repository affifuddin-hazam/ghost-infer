//! Weight formats and the matrix kernels that read them.

use rayon::prelude::*;

use crate::gguf::GgmlType;

pub const QK8_0: usize = 32;
const Q8_0_BYTES: usize = 34; // f16 scale + 32 x i8

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1; // may carry into the exponent, which is the correct rounding
    }
    sign | r as u16
}

#[inline]
fn read_f16(b: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// A 2-D weight matrix inside the model file: `rows` rows of `cols` contiguous values.
#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
}

/// A batch of activation vectors quantised to Q8_0 blocks, so they can be dotted
/// against Q8_0 weights with integer arithmetic. Buffers are reused between calls.
#[derive(Default)]
pub struct QuantizedBatch {
    pub n: usize,
    pub cols: usize,
    scales: Vec<f32>,
    quants: Vec<i8>,
    sums: Vec<i32>, // per block: sum of its int8 values (K-quant "min" terms need it)
}

impl QuantizedBatch {
    pub fn quantize(&mut self, x: &[f32], cols: usize) {
        assert!(cols % QK8_0 == 0, "row length {cols} is not a multiple of {QK8_0}");
        self.n = x.len() / cols;
        self.cols = cols;
        self.scales.resize(x.len() / QK8_0, 0.0);
        self.quants.resize(x.len(), 0);
        self.sums.resize(x.len() / QK8_0, 0);
        for (((chunk, q), d_out), s_out) in x.chunks_exact(QK8_0).zip(self.quants.chunks_exact_mut(QK8_0)).zip(&mut self.scales).zip(&mut self.sums) {
            let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut sum = 0i32;
            for (qv, v) in q.iter_mut().zip(chunk) {
                *qv = (v * inv).round() as i8;
                sum += *qv as i32;
            }
            *s_out = sum;
            // the scale goes through f16 like ggml's, so results line up with reference engines
            *d_out = f16_to_f32(f32_to_f16(d));
        }
    }

    fn row(&self, b: usize) -> (&[f32], &[i8]) {
        let nb = self.cols / QK8_0;
        (&self.scales[b * nb..(b + 1) * nb], &self.quants[b * self.cols..(b + 1) * self.cols])
    }

    fn row3(&self, b: usize) -> XRow<'_> {
        let nb = self.cols / QK8_0;
        XRow { d: &self.scales[b * nb..(b + 1) * nb], q: &self.quants[b * self.cols..(b + 1) * self.cols], s: &self.sums[b * nb..(b + 1) * nb] }
    }
}

/// One quantised activation row: per-32 scales, int8 values, per-32 sums of the values.
#[derive(Clone, Copy)]
pub struct XRow<'a> {
    d: &'a [f32],
    q: &'a [i8],
    s: &'a [i32],
}

/// Lets parallel tasks write disjoint parts of one output buffer.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// SAFETY: callers write each index from exactly one task.
    #[inline]
    unsafe fn write(self, i: usize, v: f32) {
        unsafe { *self.0.add(i) = v }
    }
}

mod kernels {
    use super::{read_f16, Q8_0_BYTES, QK8_0};

    /// Dot products of one Q8_0 weight row with `N` quantised activation rows.
    #[inline]
    pub fn q8_row_scalar<const N: usize>(w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
        let mut out = [0f32; N];
        for (b, blk) in w.chunks_exact(Q8_0_BYTES).enumerate() {
            let dw = read_f16(blk);
            let wq = &blk[2..];
            for (o, (scales, quants)) in out.iter_mut().zip(xs.iter()) {
                let xq = &quants[b * QK8_0..(b + 1) * QK8_0];
                let mut acc = 0i32;
                for (a, c) in wq.iter().zip(xq) {
                    acc += (*a as i8 as i32) * (*c as i32);
                }
                *o += dw * scales[b] * acc as f32;
            }
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q8_row_avx2<const N: usize>(w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); N];
        for (b, blk) in w.chunks_exact(Q8_0_BYTES).enumerate() {
            // pull weights in ~8 blocks (4 cache lines) ahead; may run past the row, which is harmless
            _mm_prefetch::<_MM_HINT_T0>(blk.as_ptr().wrapping_add(8 * Q8_0_BYTES) as *const i8);
            let dw = read_f16(blk);
            let wv = unsafe { _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i) };
            // maddubs wants unsigned x signed: use |w| and move w's sign onto x
            let w_abs = _mm256_sign_epi8(wv, wv);
            for (a, (scales, quants)) in acc.iter_mut().zip(xs.iter()) {
                let xv = unsafe { _mm256_loadu_si256(quants.as_ptr().add(b * QK8_0) as *const __m256i) };
                let x_signed = _mm256_sign_epi8(xv, wv);
                let p16 = _mm256_maddubs_epi16(w_abs, x_signed);
                let p32 = _mm256_madd_epi16(p16, ones);
                *a = _mm256_fmadd_ps(_mm256_set1_ps(dw * scales[b]), _mm256_cvtepi32_ps(p32), *a);
            }
        }
        let mut out = [0f32; N];
        for (o, a) in out.iter_mut().zip(acc) {
            let hi = _mm256_extractf128_ps(a, 1);
            let s = _mm_add_ps(_mm256_castps256_ps128(a), hi);
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            *o = _mm_cvtss_f32(s);
        }
        out
    }

    // ---- K-quants: 256-value super-blocks; activation blocks of 32 line up with their sub-blocks ----

    pub const Q3K_BYTES: usize = 110; // 32 bytes high bits, 64 bytes low 2-bit pairs, 12 bytes of 6-bit scales, f16 d
    pub const Q4K_BYTES: usize = 144; // f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles
    pub const Q5K_BYTES: usize = 176; // f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 32 bytes high bits, 128 bytes of nibbles
    pub const Q6K_BYTES: usize = 210; // 128 bytes low nibbles, 64 bytes high 2-bit pairs, 16 int8 scales, f16 d

    /// The 16 sub-block scales of a Q3_K block (one per 16 values), 6 bits each packed into 12 bytes,
    /// centred to -32..31: low nibbles in bytes 0-7, the top 2 bits in bytes 8-11.
    #[inline]
    pub fn q3k_scales(s: &[u8]) -> [i8; 16] {
        let mut out = [0i8; 16];
        for (j, o) in out.iter_mut().enumerate() {
            let lo = if j < 8 { s[j] & 0x0f } else { s[j - 8] >> 4 };
            let hi = (s[8 + (j & 3)] >> (2 * (j / 4))) & 3;
            *o = (lo | (hi << 4)) as i8 - 32;
        }
        out
    }

    /// Value `l` (0..32) of group `j` (0..4) in 128-value half `half` of a Q3_K block, as -4..3:
    /// two low bits from qs, and 4 subtracted unless the high bit is set.
    #[inline]
    pub fn q3k_value(q: &[u8], hm: &[u8], half: usize, j: usize, l: usize) -> i32 {
        ((q[l] >> (2 * j)) & 3) as i32 - if (hm[l] >> (4 * half + j)) & 1 != 0 { 0 } else { 4 }
    }

    pub fn dequant_q3k(row: &[u8], out: &mut [f32]) {
        for (blk, y) in row.chunks_exact(Q3K_BYTES).zip(out.chunks_exact_mut(256)) {
            let d = read_f16(&blk[108..]);
            let (hm, sc) = (&blk[..32], q3k_scales(&blk[96..108]));
            for half in 0..2 {
                let q = &blk[32 + 32 * half..64 + 32 * half];
                for j in 0..4 {
                    for l in 0..32 {
                        y[128 * half + 32 * j + l] = d * sc[8 * half + 2 * j + l / 16] as f32 * q3k_value(q, hm, half, j, l) as f32;
                    }
                }
            }
        }
    }

    pub fn dequant_q5k(row: &[u8], out: &mut [f32]) {
        for (blk, y) in row.chunks_exact(Q5K_BYTES).zip(out.chunks_exact_mut(256)) {
            let (d, dmin) = (read_f16(blk), read_f16(&blk[2..]));
            let (sc, qh, qs) = (&blk[4..16], &blk[16..48], &blk[48..]);
            for j2 in 0..4 {
                let (s1, m1) = q4k_scale_min(2 * j2, sc);
                let (s2, m2) = q4k_scale_min(2 * j2 + 1, sc);
                let q = &qs[32 * j2..32 * j2 + 32];
                for l in 0..32 {
                    let (h1, h2) = ((qh[l] >> (2 * j2)) & 1, (qh[l] >> (2 * j2 + 1)) & 1);
                    y[64 * j2 + l] = d * s1 as f32 * ((q[l] & 0x0f) | (h1 << 4)) as f32 - dmin * m1 as f32;
                    y[64 * j2 + 32 + l] = d * s2 as f32 * ((q[l] >> 4) | (h2 << 4)) as f32 - dmin * m2 as f32;
                }
            }
        }
    }

    pub fn q3k_row_scalar<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        let mut out = [0f32; N];
        for (sb, blk) in w.chunks_exact(Q3K_BYTES).enumerate() {
            let d = read_f16(&blk[108..]);
            let (hm, sc) = (&blk[..32], q3k_scales(&blk[96..108]));
            for half in 0..2 {
                let q = &blk[32 + 32 * half..64 + 32 * half];
                for j in 0..4 {
                    let xb = sb * 8 + 4 * half + j;
                    let (sa, sbv) = (sc[8 * half + 2 * j] as f32, sc[8 * half + 2 * j + 1] as f32);
                    for (o, x) in out.iter_mut().zip(xs.iter()) {
                        let xq = &x.q[xb * QK8_0..(xb + 1) * QK8_0];
                        let (mut a, mut b) = (0i32, 0i32);
                        for l in 0..32 {
                            let p = q3k_value(q, hm, half, j, l) * xq[l] as i32;
                            if l < 16 { a += p } else { b += p }
                        }
                        *o += x.d[xb] * d * (sa * a as f32 + sbv * b as f32);
                    }
                }
            }
        }
        out
    }

    pub fn q5k_row_scalar<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        let mut out = [0f32; N];
        for (sb, blk) in w.chunks_exact(Q5K_BYTES).enumerate() {
            let (d, dmin) = (read_f16(blk), read_f16(&blk[2..]));
            let (sc, qh, qs) = (&blk[4..16], &blk[16..48], &blk[48..]);
            for j2 in 0..4 {
                let q = &qs[32 * j2..32 * j2 + 32];
                for half in 0..2 {
                    let (s, m) = q4k_scale_min(2 * j2 + half, sc);
                    let xb = sb * 8 + 2 * j2 + half;
                    for (o, x) in out.iter_mut().zip(xs.iter()) {
                        let xq = &x.q[xb * QK8_0..(xb + 1) * QK8_0];
                        let mut acc = 0i32;
                        for l in 0..32 {
                            let lo = if half == 0 { q[l] & 0x0f } else { q[l] >> 4 };
                            let v = lo | (((qh[l] >> (2 * j2 + half)) & 1) << 4);
                            acc += v as i32 * xq[l] as i32;
                        }
                        *o += x.d[xb] * (d * s as f32 * acc as f32 - dmin * m as f32 * x.s[xb] as f32);
                    }
                }
            }
        }
        out
    }

    /// Scale and min of Q4_K sub-block `j` (0..8), 6 bits each, packed into 12 bytes.
    #[inline]
    pub fn q4k_scale_min(j: usize, s: &[u8]) -> (u8, u8) {
        if j < 4 {
            (s[j] & 63, s[j + 4] & 63)
        } else {
            ((s[j + 4] & 0x0f) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
        }
    }

    /// Value `l` (0..32) of group `g` (0..4) in one 128-value half of a Q6_K block, as -32..31.
    #[inline]
    pub fn q6k_value(ql: &[u8], qh: &[u8], g: usize, l: usize) -> i32 {
        let lo = match g {
            0 => ql[l] & 0x0f,
            1 => ql[l + 32] & 0x0f,
            2 => ql[l] >> 4,
            _ => ql[l + 32] >> 4,
        };
        let hi = (qh[l] >> (2 * g)) & 3;
        (lo | (hi << 4)) as i32 - 32
    }

    pub fn dequant_q4k(row: &[u8], out: &mut [f32]) {
        for (blk, y) in row.chunks_exact(Q4K_BYTES).zip(out.chunks_exact_mut(256)) {
            let (d, dmin) = (read_f16(blk), read_f16(&blk[2..]));
            let (sc, qs) = (&blk[4..16], &blk[16..]);
            for j2 in 0..4 {
                let (s1, m1) = q4k_scale_min(2 * j2, sc);
                let (s2, m2) = q4k_scale_min(2 * j2 + 1, sc);
                let q = &qs[32 * j2..32 * j2 + 32];
                for l in 0..32 {
                    y[64 * j2 + l] = d * s1 as f32 * (q[l] & 0x0f) as f32 - dmin * m1 as f32;
                    y[64 * j2 + 32 + l] = d * s2 as f32 * (q[l] >> 4) as f32 - dmin * m2 as f32;
                }
            }
        }
    }

    pub fn dequant_q6k(row: &[u8], out: &mut [f32]) {
        for (blk, y) in row.chunks_exact(Q6K_BYTES).zip(out.chunks_exact_mut(256)) {
            let d = read_f16(&blk[208..]);
            for half in 0..2 {
                let (ql, qh, sc) = (&blk[64 * half..], &blk[128 + 32 * half..], &blk[192 + 8 * half..]);
                for g in 0..4 {
                    for l in 0..32 {
                        let s = sc[2 * g + l / 16] as i8 as f32;
                        y[128 * half + 32 * g + l] = d * s * q6k_value(ql, qh, g, l) as f32;
                    }
                }
            }
        }
    }

    pub fn q4k_row_scalar<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        let mut out = [0f32; N];
        for (sb, blk) in w.chunks_exact(Q4K_BYTES).enumerate() {
            let (d, dmin) = (read_f16(blk), read_f16(&blk[2..]));
            let (sc, qs) = (&blk[4..16], &blk[16..]);
            for j2 in 0..4 {
                let q = &qs[32 * j2..32 * j2 + 32];
                for half in 0..2 {
                    let (s, m) = q4k_scale_min(2 * j2 + half, sc);
                    let xb = sb * 8 + 2 * j2 + half; // activation block
                    for (o, x) in out.iter_mut().zip(xs.iter()) {
                        let xq = &x.q[xb * QK8_0..(xb + 1) * QK8_0];
                        let mut acc = 0i32;
                        for (wq, xv) in q.iter().zip(xq) {
                            let v = if half == 0 { wq & 0x0f } else { wq >> 4 };
                            acc += v as i32 * *xv as i32;
                        }
                        *o += x.d[xb] * (d * s as f32 * acc as f32 - dmin * m as f32 * x.s[xb] as f32);
                    }
                }
            }
        }
        out
    }

    pub fn q6k_row_scalar<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        let mut out = [0f32; N];
        for (sb, blk) in w.chunks_exact(Q6K_BYTES).enumerate() {
            let d = read_f16(&blk[208..]);
            for half in 0..2 {
                let (ql, qh, sc) = (&blk[64 * half..], &blk[128 + 32 * half..], &blk[192 + 8 * half..]);
                for g in 0..4 {
                    let xb = sb * 8 + 4 * half + g;
                    let (sa, sb_) = (sc[2 * g] as i8 as f32, sc[2 * g + 1] as i8 as f32);
                    for (o, x) in out.iter_mut().zip(xs.iter()) {
                        let xq = &x.q[xb * QK8_0..(xb + 1) * QK8_0];
                        let (mut a, mut b) = (0i32, 0i32);
                        for l in 0..32 {
                            let p = q6k_value(ql, qh, g, l) * xq[l] as i32;
                            if l < 16 { a += p } else { b += p }
                        }
                        *o += x.d[xb] * d * (sa * a as f32 + sb_ * b as f32);
                    }
                }
            }
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(a: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q4k_row_avx2<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let nib = _mm256_set1_epi8(0x0f);
        let mut acc = [_mm256_setzero_ps(); N];
        let mut mins = [0f32; N];
        for (sb, blk) in w.chunks_exact(Q4K_BYTES).enumerate() {
            _mm_prefetch::<_MM_HINT_T0>(blk.as_ptr().wrapping_add(2 * Q4K_BYTES) as *const i8);
            let (d, dmin) = (read_f16(blk), read_f16(&blk[2..]));
            let sc = &blk[4..16];
            for j2 in 0..4 {
                let qv = unsafe { _mm256_loadu_si256(blk.as_ptr().add(16 + 32 * j2) as *const __m256i) };
                let vals = [_mm256_and_si256(qv, nib), _mm256_and_si256(_mm256_srli_epi16(qv, 4), nib)];
                for half in 0..2 {
                    let (s, m) = q4k_scale_min(2 * j2 + half, sc);
                    let xb = sb * 8 + 2 * j2 + half;
                    for ((a, mn), x) in acc.iter_mut().zip(mins.iter_mut()).zip(xs.iter()) {
                        let xv = unsafe { _mm256_loadu_si256(x.q.as_ptr().add(xb * QK8_0) as *const __m256i) };
                        // weights are unsigned 0..15, exactly what maddubs wants
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(vals[half], xv), ones);
                        *a = _mm256_fmadd_ps(_mm256_set1_ps(x.d[xb] * d * s as f32), _mm256_cvtepi32_ps(p32), *a);
                        *mn += x.d[xb] * dmin * m as f32 * x.s[xb] as f32;
                    }
                }
            }
        }
        let mut out = [0f32; N];
        for ((o, a), mn) in out.iter_mut().zip(acc).zip(mins) {
            *o = unsafe { hsum(a) } - mn;
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q3k_row_avx2<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        use std::arch::x86_64::*;
        let ones16 = _mm256_set1_epi16(1);
        let ones8 = _mm256_set1_epi8(1);
        let three = _mm256_set1_epi8(3);
        let mut acc = [_mm256_setzero_ps(); N];
        for (sb, blk) in w.chunks_exact(Q3K_BYTES).enumerate() {
            _mm_prefetch::<_MM_HINT_T0>(blk.as_ptr().wrapping_add(2 * Q3K_BYTES) as *const i8);
            let d = read_f16(&blk[108..]);
            let sc = q3k_scales(&blk[96..108]);
            let hm = unsafe { _mm256_loadu_si256(blk.as_ptr() as *const __m256i) };
            for half in 0..2 {
                let q = unsafe { _mm256_loadu_si256(blk.as_ptr().add(32 + 32 * half) as *const __m256i) };
                for j in 0..4 {
                    // unsigned 0..7 values (2 low bits, high bit as 4); the -4 offset goes through the
                    // activation sums below. 16-bit shifts are safe: every byte is masked afterwards.
                    let lo = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * j as i32)), three);
                    let hi = _mm256_and_si256(_mm256_srl_epi16(hm, _mm_cvtsi32_si128((4 * half + j) as i32)), ones8);
                    let u = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 2));
                    let xb = sb * 8 + 4 * half + j;
                    let (sa, sbv) = (sc[8 * half + 2 * j] as f32, sc[8 * half + 2 * j + 1] as f32);
                    for (a, x) in acc.iter_mut().zip(xs.iter()) {
                        let xv = unsafe { _mm256_loadu_si256(x.q.as_ptr().add(xb * QK8_0) as *const __m256i) };
                        let qx = _mm256_madd_epi16(_mm256_maddubs_epi16(u, xv), ones16);
                        let sx = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, xv), ones16);
                        // lanes 0-3 cover values 0..16 (scale a), lanes 4-7 values 16..32 (scale b)
                        let p32 = _mm256_sub_epi32(qx, _mm256_slli_epi32(sx, 2));
                        let f = x.d[xb] * d;
                        let scale = _mm256_setr_ps(f * sa, f * sa, f * sa, f * sa, f * sbv, f * sbv, f * sbv, f * sbv);
                        *a = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), *a);
                    }
                }
            }
        }
        let mut out = [0f32; N];
        for (o, a) in out.iter_mut().zip(acc) {
            *o = unsafe { hsum(a) };
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q5k_row_avx2<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let nib = _mm256_set1_epi8(0x0f);
        let one = _mm256_set1_epi8(1);
        let mut acc = [_mm256_setzero_ps(); N];
        let mut mins = [0f32; N];
        for (sb, blk) in w.chunks_exact(Q5K_BYTES).enumerate() {
            _mm_prefetch::<_MM_HINT_T0>(blk.as_ptr().wrapping_add(2 * Q5K_BYTES) as *const i8);
            let (d, dmin) = (read_f16(blk), read_f16(&blk[2..]));
            let sc = &blk[4..16];
            let qh = unsafe { _mm256_loadu_si256(blk.as_ptr().add(16) as *const __m256i) };
            for j2 in 0..4 {
                let qv = unsafe { _mm256_loadu_si256(blk.as_ptr().add(48 + 32 * j2) as *const __m256i) };
                // fifth bit of each half, moved to bit 4 (16-bit shifts are safe: masked per byte)
                let h0 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2 * j2 as i32)), one), 4);
                let h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2 * j2 as i32 + 1)), one), 4);
                let vals = [
                    _mm256_or_si256(_mm256_and_si256(qv, nib), h0),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qv, 4), nib), h1),
                ];
                for half in 0..2 {
                    let (s, m) = q4k_scale_min(2 * j2 + half, sc);
                    let xb = sb * 8 + 2 * j2 + half;
                    for ((a, mn), x) in acc.iter_mut().zip(mins.iter_mut()).zip(xs.iter()) {
                        let xv = unsafe { _mm256_loadu_si256(x.q.as_ptr().add(xb * QK8_0) as *const __m256i) };
                        // weights are unsigned 0..31, exactly what maddubs wants
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(vals[half], xv), ones);
                        *a = _mm256_fmadd_ps(_mm256_set1_ps(x.d[xb] * d * s as f32), _mm256_cvtepi32_ps(p32), *a);
                        *mn += x.d[xb] * dmin * m as f32 * x.s[xb] as f32;
                    }
                }
            }
        }
        let mut out = [0f32; N];
        for ((o, a), mn) in out.iter_mut().zip(acc).zip(mins) {
            *o = unsafe { hsum(a) } - mn;
        }
        out
    }

    /// One K-quant weight row against `N` activation rows, with the format's kernel.
    #[inline]
    pub fn kq_row<const N: usize>(ty: crate::gguf::GgmlType, avx2: bool, w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        use crate::gguf::GgmlType as T;
        match (ty, avx2) {
            (T::Q3K, true) => unsafe { q3k_row_avx2(w, xs) },
            (T::Q3K, false) => q3k_row_scalar(w, xs),
            (T::Q4K, true) => unsafe { q4k_row_avx2(w, xs) },
            (T::Q4K, false) => q4k_row_scalar(w, xs),
            (T::Q5K, true) => unsafe { q5k_row_avx2(w, xs) },
            (T::Q5K, false) => q5k_row_scalar(w, xs),
            (T::Q6K, true) => unsafe { q6k_row_avx2(w, xs) },
            (T::Q6K, false) => q6k_row_scalar(w, xs),
            _ => unreachable!("not a K-quant: {}", ty.name()),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q6k_row_avx2<const N: usize>(w: &[u8], xs: [super::XRow; N]) -> [f32; N] {
        use std::arch::x86_64::*;
        let ones16 = _mm256_set1_epi16(1);
        let ones8 = _mm256_set1_epi8(1);
        let nib = _mm256_set1_epi8(0x0f);
        let two = _mm256_set1_epi8(3);
        let mut acc = [_mm256_setzero_ps(); N];
        for (sb, blk) in w.chunks_exact(Q6K_BYTES).enumerate() {
            _mm_prefetch::<_MM_HINT_T0>(blk.as_ptr().wrapping_add(2 * Q6K_BYTES) as *const i8);
            let d = read_f16(&blk[208..]);
            for half in 0..2 {
                let p = blk.as_ptr();
                let ql0 = unsafe { _mm256_loadu_si256(p.add(64 * half) as *const __m256i) };
                let ql1 = unsafe { _mm256_loadu_si256(p.add(64 * half + 32) as *const __m256i) };
                let qh = unsafe { _mm256_loadu_si256(p.add(128 + 32 * half) as *const __m256i) };
                // 2 high bits of each group, moved to bits 4-5 (16-bit shifts are safe: every byte is masked)
                let h0 = _mm256_slli_epi16(_mm256_and_si256(qh, two), 4);
                let h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), two), 4);
                let h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), two), 4);
                let h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), two), 4);
                // unsigned 0..63 values; the -32 offset is applied through the activation sums below
                let groups = [
                    _mm256_or_si256(_mm256_and_si256(ql0, nib), h0),
                    _mm256_or_si256(_mm256_and_si256(ql1, nib), h1),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), nib), h2),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), nib), h3),
                ];
                let sc = &blk[192 + 8 * half..];
                for (g, gv) in groups.iter().enumerate() {
                    let xb = sb * 8 + 4 * half + g;
                    let (sa, sbv) = (sc[2 * g] as i8 as f32, sc[2 * g + 1] as i8 as f32);
                    for (a, x) in acc.iter_mut().zip(xs.iter()) {
                        let xv = unsafe { _mm256_loadu_si256(x.q.as_ptr().add(xb * QK8_0) as *const __m256i) };
                        let qx = _mm256_madd_epi16(_mm256_maddubs_epi16(*gv, xv), ones16);
                        let sx = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, xv), ones16);
                        // lanes 0-3 cover values 0..16 (scale a), lanes 4-7 values 16..32 (scale b)
                        let p32 = _mm256_sub_epi32(qx, _mm256_slli_epi32(sx, 5));
                        let f = x.d[xb] * d;
                        let scale = _mm256_setr_ps(f * sa, f * sa, f * sa, f * sa, f * sbv, f * sbv, f * sbv, f * sbv);
                        *a = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), *a);
                    }
                }
            }
        }
        let mut out = [0f32; N];
        for (o, a) in out.iter_mut().zip(acc) {
            *o = unsafe { hsum(a) };
        }
        out
    }
}

/// f32 dot product written so LLVM keeps 8 independent lanes (vectorises cleanly).
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// BETTERLLAMA_EXACT=1: skip activation quantisation (slow; for checking accuracy).
pub fn exact_mode() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT.get_or_init(|| std::env::var_os("BETTERLLAMA_EXACT").is_some())
}

fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVX2.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

impl<'a> Matrix<'a> {
    fn row_bytes(&self) -> usize {
        let (per_block, bytes) = self.ty.block().expect("unsupported weight type");
        self.cols / per_block * bytes
    }

    fn row(&self, r: usize) -> &'a [u8] {
        let n = self.row_bytes();
        &self.data[r * n..(r + 1) * n]
    }

    /// Copies one row out as f32 (used for the token embedding lookup).
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let row = self.row(r);
        match self.ty {
            GgmlType::F32 => {
                for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                    *o = f32::from_le_bytes(b.try_into().unwrap());
                }
            }
            GgmlType::F16 => {
                for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                    *o = read_f16(b);
                }
            }
            GgmlType::Q8_0 => {
                for (blk, o) in row.chunks_exact(Q8_0_BYTES).zip(out.chunks_exact_mut(QK8_0)) {
                    let d = read_f16(blk);
                    for (v, q) in o.iter_mut().zip(&blk[2..]) {
                        *v = d * (*q as i8 as f32);
                    }
                }
            }
            GgmlType::Q3K => kernels::dequant_q3k(row, out),
            GgmlType::Q4K => kernels::dequant_q4k(row, out),
            GgmlType::Q5K => kernels::dequant_q5k(row, out),
            GgmlType::Q6K => kernels::dequant_q6k(row, out),
            GgmlType::Other(_) => panic!("unsupported weight type {}", self.ty.name()),
        }
    }

    /// Calls `emit(b, W[r] · x[b])` for every input `b` of the batch.
    #[inline]
    fn row_dots(&self, r: usize, x: &[f32], xq: &QuantizedBatch, n: usize, avx2: bool, mut emit: impl FnMut(usize, f32)) {
        let w = self.row(r);
        match self.ty {
            ty if (ty == GgmlType::Q8_0 || ty.is_kquant()) && exact_mode() => {
                // reference path: dequantised weights against unrounded f32 activations
                let mut wf = vec![0f32; self.cols];
                self.row_f32(r, &mut wf);
                for b in 0..n {
                    emit(b, dot_f32(&wf, &x[b * self.cols..(b + 1) * self.cols]));
                }
            }
            GgmlType::Q8_0 => {
                debug_assert_eq!(xq.n, n);
                let mut b = 0;
                // four inputs at a time share each weight load
                while b + 4 <= n {
                    let xs = [xq.row(b), xq.row(b + 1), xq.row(b + 2), xq.row(b + 3)];
                    let d = if avx2 { unsafe { kernels::q8_row_avx2(w, xs) } } else { kernels::q8_row_scalar(w, xs) };
                    for (i, v) in d.into_iter().enumerate() {
                        emit(b + i, v);
                    }
                    b += 4;
                }
                while b < n {
                    let xs = [xq.row(b)];
                    let d = if avx2 { unsafe { kernels::q8_row_avx2(w, xs) } } else { kernels::q8_row_scalar(w, xs) };
                    emit(b, d[0]);
                    b += 1;
                }
            }
            ty if ty.is_kquant() => {
                debug_assert_eq!(xq.n, n);
                let mut b = 0;
                while b + 4 <= n {
                    let xs = [xq.row3(b), xq.row3(b + 1), xq.row3(b + 2), xq.row3(b + 3)];
                    for (i, v) in kernels::kq_row(ty, avx2, w, xs).into_iter().enumerate() {
                        emit(b + i, v);
                    }
                    b += 4;
                }
                while b < n {
                    emit(b, kernels::kq_row(ty, avx2, w, [xq.row3(b)])[0]);
                    b += 1;
                }
            }
            GgmlType::F32 | GgmlType::F16 => {
                let mut wf = vec![0f32; self.cols];
                self.row_f32(r, &mut wf);
                for b in 0..n {
                    emit(b, dot_f32(&wf, &x[b * self.cols..(b + 1) * self.cols]));
                }
            }
            _ => panic!("unsupported weight type {}", self.ty.name()),
        }
    }

    /// `out[b * rows + r] = W[r] · x[b]` for every row `r` and every input `b`.
    /// `x` holds the inputs as f32 (`n × cols`); `xq` is the same batch quantised (used for Q8_0 weights).
    pub fn matmul(&self, x: &[f32], xq: &QuantizedBatch, out: &mut [f32]) {
        matmul_many(&mut [(self, out)], x, xq);
    }
}

fn rows_per_task(total_rows: usize) -> usize {
    // enough tasks to balance across threads, but not so many that scheduling dominates
    (total_rows / (rayon::current_num_threads() * 4)).clamp(4, 256).next_multiple_of(4)
}

/// Several matrices that share the same input (e.g. Q, K and V) in one parallel pass.
pub fn matmul_many(jobs: &mut [(&Matrix, &mut [f32])], x: &[f32], xq: &QuantizedBatch) {
    let avx2 = has_avx2();
    let n = x.len() / jobs[0].0.cols;
    let step = rows_per_task(jobs.iter().map(|j| j.0.rows).sum());
    let targets: Vec<(Matrix, SendPtr)> = jobs
        .iter_mut()
        .map(|(m, out)| {
            debug_assert_eq!(out.len(), n * m.rows);
            (**m, SendPtr(out.as_mut_ptr()))
        })
        .collect();
    let tasks: Vec<(usize, usize, usize)> = targets
        .iter()
        .enumerate()
        .flat_map(|(j, (m, _))| (0..m.rows).step_by(step).map(move |r0| (j, r0, (r0 + step).min(m.rows))))
        .collect();
    tasks.par_iter().for_each(|&(j, r0, r1)| {
        let (m, out) = &targets[j];
        for r in r0..r1 {
            // SAFETY: each (matrix, row) belongs to exactly one task
            m.row_dots(r, x, xq, n, avx2, |b, v| unsafe { out.write(b * m.rows + r, v) });
        }
    });
}

/// SwiGLU front half in one pass: `h[b * rows + r] = silu(gate[r] · x[b]) * (up[r] · x[b])`.
pub fn gate_up_silu(gate: &Matrix, up: &Matrix, x: &[f32], xq: &QuantizedBatch, h: &mut [f32]) {
    let avx2 = has_avx2();
    let n = x.len() / gate.cols;
    let rows = gate.rows;
    debug_assert_eq!(h.len(), n * rows);
    let step = rows_per_task(rows * 2);
    let out = SendPtr(h.as_mut_ptr());
    (0..rows.div_ceil(step)).into_par_iter().for_each_init(Vec::new, |g_vals, task| {
        let (r0, r1) = (task * step, ((task + 1) * step).min(rows));
        g_vals.resize(n, 0f32);
        for r in r0..r1 {
            gate.row_dots(r, x, xq, n, avx2, |b, v| g_vals[b] = v);
            up.row_dots(r, x, xq, n, avx2, |b, u| {
                let g = g_vals[b];
                // SAFETY: row r of every batch entry belongs to this task only
                unsafe { out.write(b * rows + r, g / (1.0 + (-g).exp()) * u) };
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for v in [0.0f32, 1.0, -2.5, 0.000061035156, 65504.0, 3.1415927, 1e-7] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }

    #[test]
    fn kquant_kernels_agree_with_dequantised_reference() {
        let (rows, cols, n) = (3, 512, 5);
        let mut seed = 0x2545_f491u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        };
        let x: Vec<f32> = (0..n * cols).map(|i| ((i * 37 % 97) as f32 - 48.0) * 0.03).collect();
        let mut xq = QuantizedBatch::default();
        xq.quantize(&x, cols);
        // activations exactly as the integer kernels see them
        let xd: Vec<f32> = (0..n * cols).map(|i| xq.quants[i] as f32 * xq.scales[i / QK8_0]).collect();

        for ty in [GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let (per, bytes) = ty.block().unwrap();
            let mut data: Vec<u8> = (0..rows * cols / per * bytes).map(|_| rnd()).collect();
            for blk in data.chunks_exact_mut(bytes) {
                // keep the f16 super-block scales finite and modest
                let h = f32_to_f16(0.01).to_le_bytes();
                match ty {
                    GgmlType::Q4K | GgmlType::Q5K => {
                        blk[0..2].copy_from_slice(&h);
                        blk[2..4].copy_from_slice(&f32_to_f16(0.004).to_le_bytes());
                    }
                    GgmlType::Q3K => blk[108..110].copy_from_slice(&h),
                    _ => blk[208..210].copy_from_slice(&h),
                }
            }
            let m = Matrix { data: &data, ty, rows, cols };
            let mut out = vec![0f32; n * rows];
            m.matmul(&x, &xq, &mut out);
            let mut wf = vec![0f32; cols];
            for r in 0..rows {
                m.row_f32(r, &mut wf);
                let scalar = kernels::kq_row(ty, false, m.row(r), [xq.row3(0)]);
                for b in 0..n {
                    let reference = dot_f32(&wf, &xd[b * cols..(b + 1) * cols]);
                    let got = out[b * rows + r];
                    let tol = 1e-3 * wf.iter().map(|v| v.abs()).sum::<f32>();
                    assert!((got - reference).abs() < tol, "{} r{r} b{b}: kernel {got} vs reference {reference}", ty.name());
                    if b == 0 {
                        assert!((got - scalar[0]).abs() < tol, "{} r{r}: simd {got} vs scalar {}", ty.name(), scalar[0]);
                    }
                }
            }
        }
    }

    /// cargo test --release dispatch -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dispatch_cost() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        pool.install(|| {
            for round in 0..3 {
                let t = std::time::Instant::now();
                let iters = 5000;
                for _ in 0..iters {
                    (0..6).into_par_iter().for_each(|i| {
                        std::hint::black_box(i);
                    });
                }
                println!("round {round}: {:.1} us per parallel dispatch", t.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
        });
    }

    /// cargo test --release matvec_bandwidth -- --ignored --nocapture
    #[test]
    #[ignore]
    fn matvec_bandwidth() {
        for (rows, cols) in [(576usize, 576usize), (1536, 576), (576, 1536), (49152, 576), (245_760, 576)] {
            let data = test_matrix(rows, cols);
            let m = Matrix { data: &data, ty: GgmlType::Q8_0, rows, cols };
            let x: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.1).collect();
            let mut xq = QuantizedBatch::default();
            xq.quantize(&x, cols);
            let mut out = vec![0f32; rows];
            for threads in [1usize, 6] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| {
                    let reps = (200_000_000 / data.len()).max(3);
                    m.matmul(&x, &xq, &mut out);
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        m.matmul(&x, &xq, &mut out);
                    }
                    let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
                    println!("{rows:>6}x{cols:<5} {threads} thr: {us:>8.1} us  {:>6.1} GB/s (in cache if < 16 MB: {:.1} MB)", data.len() as f64 / us / 1e3, data.len() as f64 / 1e6);
                });
            }
        }
    }

    fn test_matrix(rows: usize, cols: usize) -> Vec<u8> {
        let mut data = Vec::new();
        for r in 0..rows {
            for blk in 0..cols / QK8_0 {
                data.extend_from_slice(&f32_to_f16(0.01 * (1 + (r + blk) % 7) as f32).to_le_bytes());
                for c in 0..QK8_0 {
                    data.push((((r * 31 + blk * 7 + c * 13) % 255) as i32 - 127) as i8 as u8);
                }
            }
        }
        data
    }

    #[test]
    fn q8_matmul_matches_f32_reference_and_scalar() {
        let (rows, cols, n) = (37, 96, 7); // odd sizes: exercises the 4-wide and leftover paths
        let data = test_matrix(rows, cols);
        let m = Matrix { data: &data, ty: GgmlType::Q8_0, rows, cols };
        let x: Vec<f32> = (0..n * cols).map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.02).collect();
        let mut xq = QuantizedBatch::default();
        xq.quantize(&x, cols);
        let mut out = vec![0f32; n * rows];
        m.matmul(&x, &xq, &mut out);

        let mut wf = vec![0f32; cols];
        for r in 0..rows {
            m.row_f32(r, &mut wf);
            for b in 0..n {
                let exact = dot_f32(&wf, &x[b * cols..(b + 1) * cols]);
                let got = out[b * rows + r];
                let scalar = kernels::q8_row_scalar(m.row(r), [xq.row(b)])[0];
                assert!((got - scalar).abs() < 1e-3 * scalar.abs().max(1.0), "r{r} b{b}: simd {got} vs scalar {scalar}");
                // activation quantisation error scales with the size of the terms, not the (cancelling) sum
                let magnitude: f32 = wf.iter().zip(&x[b * cols..(b + 1) * cols]).map(|(a, c)| (a * c).abs()).sum();
                assert!((got - exact).abs() < 0.01 * magnitude, "r{r} b{b}: {got} vs exact {exact} (terms {magnitude})");
            }
        }
    }
}
