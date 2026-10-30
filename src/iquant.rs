//! llama.cpp's i-quants that small Qwen3-Next files use: IQ2_XXS, IQ2_XS and IQ2_S (2-bit: each 8 values are
//! one entry of a fixed grid of non-negative levels, with a sign mask) and IQ3_S (3-bit: 4-wide grid entries).
//! Every block holds 256 values behind one f16 scale, with a finer scale per 32 (or per 16) values.

use crate::gguf::GgmlType;
use crate::quant::f16_to_f32;

include!("iq_tables.rs");

/// Bytes per 256-value block.
pub const IQ2_XXS_BYTES: usize = 66; // f16 d, 32 x u16 (4 grid indices, then 4 x 7 sign bits and a 4-bit scale per 32)
pub const IQ2_XS_BYTES: usize = 74; // f16 d, 32 x u16 (9-bit grid index, 7 sign bits), 8 scale bytes (two 4-bit per 32)
pub const IQ2_S_BYTES: usize = 82; // f16 d, 32 grid index bytes, 32 sign bytes, 8 high-bit bytes, 8 scale bytes
pub const IQ3_S_BYTES: usize = 110; // f16 d, 64 grid index bytes, 8 high-bit bytes, 32 sign bytes, 4 scale bytes

#[inline]
fn d16(b: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// The 8 levels of a grid entry, signed by bits of `signs` (bit j set: value j is negative), times `scale`.
#[inline]
fn put8(out: &mut [f32], grid: u64, signs: u8, scale: f32) {
    for (j, (o, g)) in out.iter_mut().zip(grid.to_le_bytes()).enumerate() {
        *o = scale * g as f32 * if signs >> j & 1 != 0 { -1.0 } else { 1.0 };
    }
}

fn dequant_iq2_xxs(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(IQ2_XXS_BYTES).zip(out.chunks_exact_mut(256)) {
        let d = d16(blk);
        for (q, o) in blk[2..].chunks_exact(8).zip(o.chunks_exact_mut(32)) {
            let aux = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
            for (l, o) in o.chunks_exact_mut(8).enumerate() {
                put8(o, IQ2XXS_GRID[q[l] as usize], KSIGNS_IQ2XS[(aux >> (7 * l) & 127) as usize], db);
            }
        }
    }
}

fn dequant_iq2_xs(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(IQ2_XS_BYTES).zip(out.chunks_exact_mut(256)) {
        let d = d16(blk);
        let scales = &blk[66..74];
        for (ib, o) in o.chunks_exact_mut(32).enumerate() {
            let db = [d * (0.5 + (scales[ib] & 0xf) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
            for (l, o) in o.chunks_exact_mut(8).enumerate() {
                let at = 2 + 2 * (4 * ib + l);
                let q = u16::from_le_bytes([blk[at], blk[at + 1]]);
                put8(o, IQ2XS_GRID[(q & 511) as usize], KSIGNS_IQ2XS[(q >> 9) as usize], db[l / 2]);
            }
        }
    }
}

fn dequant_iq2_s(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(IQ2_S_BYTES).zip(out.chunks_exact_mut(256)) {
        let d = d16(blk);
        let (qs, signs, qh, scales) = (&blk[2..34], &blk[34..66], &blk[66..74], &blk[74..82]);
        for (ib, o) in o.chunks_exact_mut(32).enumerate() {
            let db = [d * (0.5 + (scales[ib] & 0xf) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
            for (l, o) in o.chunks_exact_mut(8).enumerate() {
                let idx = qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300);
                put8(o, IQ2S_GRID[idx], signs[4 * ib + l], db[l / 2]);
            }
        }
    }
}

fn dequant_iq3_s(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(IQ3_S_BYTES).zip(out.chunks_exact_mut(256)) {
        let d = d16(blk);
        let (qs, qh, signs, scales) = (&blk[2..66], &blk[66..74], &blk[74..106], &blk[106..110]);
        // 32 values at a time, two per scale byte; each 8 are two 4-wide grid entries with 9-bit indices
        for (ib, o) in o.chunks_exact_mut(32).enumerate() {
            let s = scales[ib / 2];
            let db = d * (1 + 2 * if ib % 2 == 0 { s & 0xf } else { s >> 4 }) as f32;
            let (q, h, sg) = (&qs[8 * ib..8 * ib + 8], qh[ib] as usize, &signs[4 * ib..4 * ib + 4]);
            for (l, o) in o.chunks_exact_mut(8).enumerate() {
                let g1 = IQ3S_GRID[q[2 * l] as usize | (h << (8 - 2 * l) & 256)].to_le_bytes();
                let g2 = IQ3S_GRID[q[2 * l + 1] as usize | (h << (7 - 2 * l) & 256)].to_le_bytes();
                for j in 0..4 {
                    o[j] = db * g1[j] as f32 * if sg[l] >> j & 1 != 0 { -1.0 } else { 1.0 };
                    o[j + 4] = db * g2[j] as f32 * if sg[l] >> (j + 4) & 1 != 0 { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// The 2-bit formats with a dot-product kernel (IQ3_S only holds embeddings, which are looked up).
pub fn has_dot(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Iq2Xxs | GgmlType::Iq2Xs | GgmlType::Iq2S)
}

/// Per 32 values of a block: the 4 grid entries, their 4 sign bytes, and the scales of its two halves.
#[inline]
fn sub_block(ty: GgmlType, blk: &[u8], d: f32, ib: usize) -> ([u64; 4], [u8; 4], [f32; 2]) {
    match ty {
        GgmlType::Iq2Xxs => {
            let q = &blk[2 + 8 * ib..2 + 8 * ib + 8];
            let aux = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
            let g = [0, 1, 2, 3].map(|l| IQ2XXS_GRID[q[l] as usize]);
            (g, [0, 1, 2, 3].map(|l| KSIGNS_IQ2XS[(aux >> (7 * l) & 127) as usize]), [db, db])
        }
        GgmlType::Iq2Xs => {
            let s = blk[66 + ib];
            let q = [0, 1, 2, 3].map(|l| u16::from_le_bytes([blk[2 + 2 * (4 * ib + l)], blk[3 + 2 * (4 * ib + l)]]));
            (q.map(|q| IQ2XS_GRID[(q & 511) as usize]), q.map(|q| KSIGNS_IQ2XS[(q >> 9) as usize]), [d * (0.5 + (s & 0xf) as f32) * 0.25, d * (0.5 + (s >> 4) as f32) * 0.25])
        }
        _ => {
            let (s, h) = (blk[74 + ib], blk[66 + ib] as usize);
            let g = [0, 1, 2, 3].map(|l| IQ2S_GRID[blk[2 + 4 * ib + l] as usize | (h << (8 - 2 * l) & 0x300)]);
            (g, [0, 1, 2, 3].map(|l| blk[34 + 4 * ib + l]), [d * (0.5 + (s & 0xf) as f32) * 0.25, d * (0.5 + (s >> 4) as f32) * 0.25])
        }
    }
}

/// One 2-bit weight row against `N` int8 activation rows (Q8_0 blocks of 32: `xd` scales, `xq` values).
pub fn row_dot_scalar<const N: usize>(ty: GgmlType, w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
    let bytes = ty.block().unwrap().1;
    let mut out = [0f32; N];
    for (sb, blk) in w.chunks_exact(bytes).enumerate() {
        let d = d16(blk);
        for ib in 0..8 {
            let (grid, signs, db) = sub_block(ty, blk, d, ib);
            let xb = sb * 8 + ib;
            for (o, (xd, xq)) in out.iter_mut().zip(xs.iter()) {
                let mut half = [0i32; 2];
                for l in 0..4 {
                    for (j, g) in grid[l].to_le_bytes().into_iter().enumerate() {
                        let v = g as i32 * xq[xb * 32 + 8 * l + j] as i32;
                        half[l / 2] += if signs[l] >> j & 1 != 0 { -v } else { v };
                    }
                }
                *o += xd[xb] * (db[0] * half[0] as f32 + db[1] * half[1] as f32);
            }
        }
    }
    out
}

/// Per sign byte: its 8 bits as bytes, -1 (0xff) where set and 1 where not, for `_mm256_sign_epi8`.
const SIGNS256: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut b = 0;
    while b < 256 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            v |= (if (b >> j) & 1 != 0 { 0xffu64 } else { 0x01 }) << (8 * j);
            j += 1;
        }
        t[b] = v;
        b += 1;
    }
    t
};

/// The same for the 7-bit sign codes of IQ2_XXS and IQ2_XS (their eighth bit makes the count of minuses even).
const SIGNS128: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut c = 0;
    while c < 128 {
        t[c] = SIGNS256[KSIGNS_IQ2XS[c] as usize];
        c += 1;
    }
    t
};

/// `row_dot_scalar` with AVX2: dispatches to the format's own kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn row_dot_avx2<const N: usize>(ty: GgmlType, w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
    match ty {
        GgmlType::Iq2Xxs => unsafe { dot_avx2::<0, N>(w, xs) },
        GgmlType::Iq2Xs => unsafe { dot_avx2::<1, N>(w, xs) },
        _ => unsafe { dot_avx2::<2, N>(w, xs) },
    }
}

/// One 2-bit row (T: 0 IQ2_XXS, 1 IQ2_XS, 2 IQ2_S) against N int8 activation rows. Per 32 values: the four grid
/// entries side by side (unsigned levels), the activations signed from a table (llama.cpp's way), maddubs, then
/// the 4-bit scales (2s + 1) as integers on each half, and one float multiply with d / 8 and the activations'
/// scale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const T: u8, const N: usize>(w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
    use std::arch::x86_64::*;
    let bytes = [IQ2_XXS_BYTES, IQ2_XS_BYTES, IQ2_S_BYTES][T as usize];
    let mut acc = [_mm256_setzero_ps(); N];
    for (sb, blk) in w.chunks_exact(bytes).enumerate() {
        let d = d16(blk) * 0.125;
        for ib in 0..8 {
            let (gv, sv, ls) = match T {
                0 => {
                    let q = &blk[2 + 8 * ib..2 + 8 * ib + 8];
                    let aux = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
                    let g = |l: usize| IQ2XXS_GRID[q[l] as usize] as i64;
                    let s = |l: usize| SIGNS128[(aux >> (7 * l) & 127) as usize] as i64;
                    let ls = (2 * (aux >> 28) + 1) as i16;
                    (_mm256_set_epi64x(g(3), g(2), g(1), g(0)), _mm256_set_epi64x(s(3), s(2), s(1), s(0)), _mm256_set1_epi16(ls))
                }
                1 => {
                    let at = 2 + 8 * ib;
                    let q = |l: usize| u16::from_le_bytes([blk[at + 2 * l], blk[at + 2 * l + 1]]);
                    let g = |l: usize| IQ2XS_GRID[(q(l) & 511) as usize] as i64;
                    let s = |l: usize| SIGNS128[(q(l) >> 9) as usize] as i64;
                    let sc = blk[66 + ib];
                    let (lo, hi) = ((2 * (sc & 15) + 1) as i16, (2 * (sc >> 4) + 1) as i16);
                    (_mm256_set_epi64x(g(3), g(2), g(1), g(0)), _mm256_set_epi64x(s(3), s(2), s(1), s(0)), _mm256_set_m128i(_mm_set1_epi16(hi), _mm_set1_epi16(lo)))
                }
                _ => {
                    let h = blk[66 + ib] as usize;
                    let g = |l: usize| IQ2S_GRID[blk[2 + 4 * ib + l] as usize | (h << (8 - 2 * l) & 0x300)] as i64;
                    let s = |l: usize| SIGNS256[blk[34 + 4 * ib + l] as usize] as i64;
                    let sc = blk[74 + ib];
                    let (lo, hi) = ((2 * (sc & 15) + 1) as i16, (2 * (sc >> 4) + 1) as i16);
                    (_mm256_set_epi64x(g(3), g(2), g(1), g(0)), _mm256_set_epi64x(s(3), s(2), s(1), s(0)), _mm256_set_m128i(_mm_set1_epi16(hi), _mm_set1_epi16(lo)))
                }
            };
            let xb = sb * 8 + ib;
            for (a, (xd, xq)) in acc.iter_mut().zip(xs.iter()) {
                let xv = unsafe { _mm256_loadu_si256(xq.as_ptr().add(xb * 32) as *const __m256i) };
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(gv, _mm256_sign_epi8(xv, sv)), ls);
                *a = _mm256_fmadd_ps(_mm256_set1_ps(d * xd[xb]), _mm256_cvtepi32_ps(p), *a);
            }
        }
    }
    let mut out = [0f32; N];
    for (o, a) in out.iter_mut().zip(acc) {
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        *o = _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
    }
    out
}

/// One row of an i-quant tensor as f32.
pub fn dequant(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::Iq2Xxs => dequant_iq2_xxs(row, out),
        GgmlType::Iq2Xs => dequant_iq2_xs(row, out),
        GgmlType::Iq2S => dequant_iq2_s(row, out),
        GgmlType::Iq3S => dequant_iq3_s(row, out),
        _ => unreachable!("not an i-quant: {}", ty.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pseudo-random bytes, with every block's f16 scale set to something sane.
    fn row(ty: GgmlType, blocks: usize, seed: u64) -> Vec<u8> {
        let bytes = ty.block().unwrap().1;
        let mut s = seed;
        let mut v: Vec<u8> = (0..blocks * bytes)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        for b in v.chunks_exact_mut(bytes) {
            b[..2].copy_from_slice(&crate::quant::f32_to_f16(0.0123).to_le_bytes());
        }
        v
    }

    /// cargo test --release iq2_expert_speed -- --ignored --nocapture: one token through 10 random IQ2_XXS experts
    /// (2048 -> 512 -> 2048, Qwen3-Coder-Next's shape) on the CPU expert path, with the weights in RAM.
    #[test]
    #[ignore]
    fn iq2_expert_speed() {
        let (dim, hid, k) = (2048, 512, 10);
        let ty = GgmlType::Iq2Xxs;
        let mats: Vec<Vec<u8>> = (0..3 * k).map(|i| row(ty, dim * hid / 256, 1000 + i as u64)).collect();
        let m = |i: usize, rows: usize, cols: usize| crate::quant::Matrix { data: &mats[i], ty, rows, cols };
        let experts: Vec<_> = (0..k).map(|e| (m(3 * e, hid, dim), m(3 * e + 1, hid, dim), m(3 * e + 2, dim, hid), 0.1f32)).collect();
        let x: Vec<f32> = (0..dim).map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0).collect();
        let mut acc = vec![0f32; dim];
        let mut sc = crate::quant::ExpertScratch::default();
        for _ in 0..20 {
            crate::quant::experts_one(&experts, &x, &mut acc, &mut sc);
        }
        let t = std::time::Instant::now();
        let reps = 200;
        for _ in 0..reps {
            crate::quant::experts_one(&experts, &x, &mut acc, &mut sc);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("10 IQ2_XXS experts, one token: {us:.0} us ({:.0} us per expert)", us / k as f64);
        // the row kernel alone, one thread: a 512 x 2048 matrix
        let mut xq = crate::quant::QuantizedBatch::default();
        xq.quantize(&x, dim);
        let (xd, xqv) = xq.row_parts(0);
        let w = &mats[0];
        let rb = dim / 256 * ty.block().unwrap().1;
        let t = std::time::Instant::now();
        let mut sink = 0f32;
        for _ in 0..50 {
            for r in 0..hid {
                sink += unsafe { row_dot_avx2(ty, &w[r * rb..(r + 1) * rb], [(xd, xqv)]) }[0];
            }
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (50 * hid * dim / 32) as f64;
        println!("row kernel: {ns:.1} ns per 32 values on one thread ({sink:.1})");
    }

    #[test]
    fn dot_kernels_agree_with_dequantised_rows() {
        let cols = 512;
        let x: Vec<f32> = (0..cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0).collect();
        let mut xq = crate::quant::QuantizedBatch::default();
        xq.quantize(&x, cols);
        let (xd, xqv) = xq.row_parts(0);
        // the activations as the kernels see them (int8 times scale)
        let xr: Vec<f32> = xqv.iter().enumerate().map(|(i, &q)| q as f32 * xd[i / 32]).collect();
        for ty in [GgmlType::Iq2Xxs, GgmlType::Iq2Xs, GgmlType::Iq2S] {
            let w = row(ty, cols / 256, ty.block().unwrap().1 as u64);
            let mut wf = vec![0f32; cols];
            dequant(ty, &w, &mut wf);
            let want: f32 = wf.iter().zip(&xr).map(|(a, b)| a * b).sum();
            let scalar = row_dot_scalar(ty, &w, [(xd, xqv)])[0];
            assert!((scalar - want).abs() <= 1e-3 * want.abs().max(1.0), "{} scalar {scalar} vs {want}", ty.name());
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let fast = unsafe { row_dot_avx2(ty, &w, [(xd, xqv), (xd, xqv)]) };
                for f in fast {
                    assert!((f - want).abs() <= 1e-3 * want.abs().max(1.0), "{} avx2 {f} vs {want}", ty.name());
                }
            }
        }
    }
}
