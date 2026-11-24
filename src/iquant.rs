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
pub const IQ3_XXS_BYTES: usize = 98; // f16 d, 64 grid index bytes, 8 x u32 (4 x 7 sign bits and a 4-bit scale per 32)

/// IQ3_XXS (ggml's dequantize_row_iq3_xxs): per 32 values, 8 grid indices into 4-wide entries and one u32
/// holding four 7-bit sign indices (into KSIGNS_IQ2XS) and the 4-bit scale on top.
fn dequant_iq3_xxs(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(IQ3_XXS_BYTES).zip(out.chunks_exact_mut(256)) {
        let d = d16(blk);
        let (qs, ss) = (&blk[2..66], &blk[66..98]);
        for (ib, o) in o.chunks_exact_mut(32).enumerate() {
            let aux = u32::from_le_bytes(ss[4 * ib..4 * ib + 4].try_into().unwrap());
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            let q = &qs[8 * ib..8 * ib + 8];
            for (l, o) in o.chunks_exact_mut(8).enumerate() {
                let signs = KSIGNS_IQ2XS[(aux >> (7 * l) & 127) as usize];
                let g1 = IQ3XXS_GRID[q[2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[q[2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    o[j] = db * g1[j] as f32 * if signs >> j & 1 != 0 { -1.0 } else { 1.0 };
                    o[j + 4] = db * g2[j] as f32 * if signs >> (j + 4) & 1 != 0 { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// FP4 (E2M1) values times two, as ggml keeps them (kvalues_fp4).
const FP4_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Unsigned E4M3 scale, halved to go with FP4_X2 (ggml_ue4m3_to_fp32).
#[inline]
fn ue4m3(x: u8) -> f32 {
    UE4M3[x as usize]
}

/// Every UE4M3 byte's value (halved), as ggml_ue4m3_to_fp32: 0 and 0x7f are 0; exponent 0 is m * 2^-9, else
/// (1 + m/8) * 2^(e-7). Halving keeps every value a normal f32, so each is built from its bits.
const UE4M3: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut x = 1;
    while x < 256 {
        let (e, m) = ((x >> 3 & 0xf) as u32, (x & 7) as u32);
        if x != 0x7f && !(e == 0 && m == 0) {
            t[x] = if e == 0 {
                // m * 2^-10: m = 1..7 as a normal float (m < 8: exponent of its leading bit)
                let lead = 31 - m.leading_zeros(); // 0..2
                let frac = (m << (23 - lead)) & 0x7f_ffff;
                f32::from_bits(((lead + 127 - 10) << 23) | frac)
            } else {
                f32::from_bits(((e + 127 - 8) << 23) | (m << 20))
            };
        }
        x += 1;
    }
    t
};

/// NVFP4 (ggml's dequantize_row_nvfp4): 64-value blocks of four UE4M3 scales (one per 16) and 32 bytes of
/// 4-bit E2M1 codes; in each 16, value j is the low nibble of byte j and value j + 8 the high one.
/// (The per-tensor `.scale` some files add beside the weights is applied by the caller.)
pub fn dequant_nvfp4(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(36).zip(out.chunks_exact_mut(64)) {
        for (s, o) in o.chunks_exact_mut(16).enumerate() {
            let d = ue4m3(blk[s]);
            let q = &blk[4 + 8 * s..12 + 8 * s];
            for j in 0..8 {
                o[j] = FP4_X2[(q[j] & 0xf) as usize] as f32 * d;
                o[j + 8] = FP4_X2[(q[j] >> 4) as usize] as f32 * d;
            }
        }
    }
}

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

/// The formats with an int8 dot-product kernel: the 2-bit i-quants, and the 3- and 4-bit ones Inkling 2's
/// experts use (NVFP4, IQ4_NL, IQ3_XXS, IQ3_S).
pub fn has_dot(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Iq2Xxs | GgmlType::Iq2Xs | GgmlType::Iq2S) || is_ext(ty)
}

fn is_ext(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Nvfp4 | GgmlType::Iq4Nl | GgmlType::Iq3Xxs | GgmlType::Iq3S)
}

/// Scalar dot of a row of one of the `is_ext` formats with int8 activations: each 32 values dequantised
/// and dotted with the block's int8 values.
fn ext_row_dot_scalar<const N: usize>(ty: GgmlType, w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
    let cols = w.len() / ty.block().unwrap().1 * ty.block().unwrap().0;
    let mut wf = vec![0f32; cols];
    match ty {
        GgmlType::Nvfp4 => dequant_nvfp4(w, &mut wf),
        GgmlType::Iq4Nl => {
            for (blk, o) in w.chunks_exact(18).zip(wf.chunks_exact_mut(32)) {
                let d = d16(blk);
                for j in 0..16 {
                    o[j] = IQ4NL[(blk[2 + j] & 15) as usize] as f32 * d;
                    o[j + 16] = IQ4NL[(blk[2 + j] >> 4) as usize] as f32 * d;
                }
            }
        }
        _ => dequant(ty, w, &mut wf),
    }
    let mut out = [0f32; N];
    for (o, (xd, xq)) in out.iter_mut().zip(xs.iter()) {
        for (b, (wb, qb)) in wf.chunks_exact(32).zip(xq.chunks_exact(32)).enumerate() {
            let s: f32 = wb.iter().zip(qb).map(|(a, &q)| a * q as f32).sum();
            *o += s * xd[b];
        }
    }
    out
}

/// IQ4_NL's levels (ggml's kvalues_iq4nl).
const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// One row of an `is_ext` format against N int8 activation rows, with AVX2. Per 32 values the weights become
/// 32 signed bytes (a nibble table look-up, or grid entries signed from a table), dotted with
/// maddubs(|w|, sign(x, w)); NVFP4's two 16-value halves keep their own scales.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn ext_dot_avx2<const N: usize>(ty: GgmlType, w: &[u8], xs: [(&[f32], &[i8]); N]) -> [f32; N] {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); N];
    let ones = _mm256_set1_epi16(1);
    let m4 = _mm_set1_epi8(0x0f);
    // the dot of 32 signed weight bytes with block xb of each activation row, as 8 i32 lanes, times `scale`
    macro_rules! block {
        ($wv:expr, $xb:expr, $scale:expr) => {{
            let wv = $wv;
            let aw = _mm256_sign_epi8(wv, wv);
            for (a, (xd, xq)) in acc.iter_mut().zip(xs.iter()) {
                let xv = unsafe { _mm256_loadu_si256(xq.as_ptr().add($xb * 32) as *const __m256i) };
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, wv)), ones);
                *a = _mm256_fmadd_ps(_mm256_mul_ps($scale, _mm256_set1_ps(xd[$xb])), _mm256_cvtepi32_ps(p), *a);
            }
        }};
    }
    match ty {
        GgmlType::Nvfp4 => {
            let table = unsafe { _mm_loadu_si128(FP4_X2.as_ptr() as *const __m128i) };
            for (b64, blk) in w.chunks_exact(36).enumerate() {
                for half in 0..2 {
                    let bytes = unsafe { _mm_loadu_si128(blk.as_ptr().add(4 + 16 * half) as *const __m128i) };
                    let lo = _mm_and_si128(bytes, m4);
                    let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m4);
                    let a = _mm_shuffle_epi8(table, _mm_unpacklo_epi64(lo, hi));
                    let b = _mm_shuffle_epi8(table, _mm_unpackhi_epi64(lo, hi));
                    let (sa, sb) = (ue4m3(blk[2 * half]), ue4m3(blk[2 * half + 1]));
                    let scale = _mm256_set_m128(_mm_set1_ps(sb), _mm_set1_ps(sa));
                    block!(_mm256_set_m128i(b, a), b64 * 2 + half, scale);
                }
            }
        }
        GgmlType::Iq4Nl => {
            let table = unsafe { _mm_loadu_si128(IQ4NL.as_ptr() as *const __m128i) };
            for (b, blk) in w.chunks_exact(18).enumerate() {
                let bytes = unsafe { _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i) };
                let lo = _mm_shuffle_epi8(table, _mm_and_si128(bytes, m4));
                let hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), m4));
                // F16C for the per-32 scale (every AVX2 CPU has it): the software conversion was a tenth of the time
                let d = _mm256_cvtph_ps(_mm_set1_epi16(u16::from_le_bytes([blk[0], blk[1]]) as i16));
                block!(_mm256_set_m128i(hi, lo), b, d);
            }
        }
        GgmlType::Iq3Xxs => {
            for (sb, blk) in w.chunks_exact(IQ3_XXS_BYTES).enumerate() {
                let d = d16(blk);
                let (qs, ss) = (&blk[2..66], &blk[66..98]);
                for ib in 0..8 {
                    let aux = u32::from_le_bytes(ss[4 * ib..4 * ib + 4].try_into().unwrap());
                    let q = &qs[8 * ib..8 * ib + 8];
                    let g = |i: usize| IQ3XXS_GRID[q[i] as usize] as i32;
                    let gv = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
                    let s = |l: usize| SIGNS128[(aux >> (7 * l) & 127) as usize] as i64;
                    let wv = _mm256_sign_epi8(gv, _mm256_set_epi64x(s(3), s(2), s(1), s(0)));
                    block!(wv, sb * 8 + ib, _mm256_set1_ps(d * (0.5 + (aux >> 28) as f32) * 0.5));
                }
            }
        }
        _ => {
            // IQ3_S
            for (sb, blk) in w.chunks_exact(IQ3_S_BYTES).enumerate() {
                let d = d16(blk);
                let (qs, qh, signs, scales) = (&blk[2..66], &blk[66..74], &blk[74..106], &blk[106..110]);
                for ib in 0..8 {
                    let s = scales[ib / 2];
                    let db = d * (1 + 2 * if ib % 2 == 0 { s & 0xf } else { s >> 4 }) as f32;
                    let (q, h, sg) = (&qs[8 * ib..8 * ib + 8], qh[ib] as usize, &signs[4 * ib..4 * ib + 4]);
                    let g1 = |l: usize| IQ3S_GRID[q[2 * l] as usize | (h << (8 - 2 * l) & 256)] as i32;
                    let g2 = |l: usize| IQ3S_GRID[q[2 * l + 1] as usize | (h << (7 - 2 * l) & 256)] as i32;
                    let gv = _mm256_set_epi32(g2(3), g1(3), g2(2), g1(2), g2(1), g1(1), g2(0), g1(0));
                    let sv = |l: usize| SIGNS256[sg[l] as usize] as i64;
                    let wv = _mm256_sign_epi8(gv, _mm256_set_epi64x(sv(3), sv(2), sv(1), sv(0)));
                    block!(wv, sb * 8 + ib, _mm256_set1_ps(db));
                }
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

/// Four sign bytes (bit j of byte l: value 8l + j negative) as 32 bytes of 1 / -1, the values SIGNS256 holds,
/// without a table look-up (the kernels below run short of load slots, not arithmetic).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sign_bytes(s4: u32) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let v = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(s4 as i32), shuf), bits);
    _mm256_or_si256(_mm256_cmpeq_epi8(v, bits), _mm256_set1_epi8(1))
}

/// `row_dot_avx2` for one activation row: the same arithmetic in the same order (so the same result to the
/// bit), with the accumulator in a register and no bounds checks; the generic N-row loops kept it in memory
/// and ran 2-3x slower.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn row_dot1_avx2(ty: GgmlType, w: &[u8], xd: &[f32], xq: &[i8]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    let m4 = _mm_set1_epi8(0x0f);
    let (p, xp) = (w.as_ptr(), xq.as_ptr());
    // a block's f16 scale: in software for scalar use (F16C on one lane became a pinsrw into whatever register LLVM
    // picked, here the accumulator, chaining every block to the last FMA); broadcast to all 8 lanes with F16C
    let h16 = |b: *const u8| -> f32 { crate::quant::f16_to_f32(unsafe { std::ptr::read_unaligned(b as *const u16) }) };
    let h16x8 = |b: *const u8| -> __m256 { _mm256_cvtph_ps(_mm_set1_epi16(unsafe { std::ptr::read_unaligned(b as *const u16) } as i16)) };
    // signed weights wv against block xb, times `scale` (8 lanes) and the block's activation scale
    macro_rules! ext {
        ($wv:expr, $xb:expr, $scale:expr) => {{
            let wv = $wv;
            let xv = unsafe { _mm256_loadu_si256(xp.add($xb * 32) as *const __m256i) };
            let pr = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv)), ones);
            acc = _mm256_fmadd_ps(_mm256_mul_ps($scale, _mm256_set1_ps(unsafe { *xd.get_unchecked($xb) })), _mm256_cvtepi32_ps(pr), acc);
        }};
    }
    // unsigned grid levels gv, signs sv (0x01 / 0xff bytes), integer half scales ls, times d
    macro_rules! grid {
        ($gv:expr, $sv:expr, $ls:expr, $xb:expr, $d:expr) => {{
            let xv = unsafe { _mm256_loadu_si256(xp.add($xb * 32) as *const __m256i) };
            let pr = _mm256_madd_epi16(_mm256_maddubs_epi16($gv, _mm256_sign_epi8(xv, $sv)), $ls);
            acc = _mm256_fmadd_ps(_mm256_set1_ps($d * unsafe { *xd.get_unchecked($xb) }), _mm256_cvtepi32_ps(pr), acc);
        }};
    }
    match ty {
        GgmlType::Nvfp4 => {
            let table = unsafe { _mm_loadu_si128(FP4_X2.as_ptr() as *const __m128i) };
            for b64 in 0..w.len() / 36 {
                let blk = unsafe { p.add(b64 * 36) };
                for half in 0..2 {
                    let bytes = unsafe { _mm_loadu_si128(blk.add(4 + 16 * half) as *const __m128i) };
                    let lo = _mm_and_si128(bytes, m4);
                    let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m4);
                    let a = _mm_shuffle_epi8(table, _mm_unpacklo_epi64(lo, hi));
                    let b = _mm_shuffle_epi8(table, _mm_unpackhi_epi64(lo, hi));
                    let (sa, sb) = unsafe { (ue4m3(*blk.add(2 * half)), ue4m3(*blk.add(2 * half + 1))) };
                    ext!(_mm256_set_m128i(b, a), b64 * 2 + half, _mm256_set_m128(_mm_set1_ps(sb), _mm_set1_ps(sa)));
                }
            }
        }
        GgmlType::Iq4Nl => {
            let table = unsafe { _mm_loadu_si128(IQ4NL.as_ptr() as *const __m128i) };
            for b in 0..w.len() / 18 {
                let blk = unsafe { p.add(b * 18) };
                let bytes = unsafe { _mm_loadu_si128(blk.add(2) as *const __m128i) };
                let lo = _mm_shuffle_epi8(table, _mm_and_si128(bytes, m4));
                let hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), m4));
                ext!(_mm256_set_m128i(hi, lo), b, h16x8(blk));
            }
        }
        GgmlType::Iq3Xxs => {
            for sb in 0..w.len() / IQ3_XXS_BYTES {
                let blk = unsafe { p.add(sb * IQ3_XXS_BYTES) };
                let d = h16(blk);
                for ib in 0..8 {
                    let aux = unsafe { std::ptr::read_unaligned(blk.add(66 + 4 * ib) as *const u32) };
                    let q8 = unsafe { std::ptr::read_unaligned(blk.add(2 + 8 * ib) as *const u64) };
                    let g = |i: u32| unsafe { *IQ3XXS_GRID.get_unchecked((q8 >> (8 * i)) as u8 as usize) } as i32;
                    let gv = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
                    let s = |l: usize| unsafe { *SIGNS128.get_unchecked((aux >> (7 * l) & 127) as usize) } as i64;
                    let wv = _mm256_sign_epi8(gv, _mm256_set_epi64x(s(3), s(2), s(1), s(0)));
                    ext!(wv, sb * 8 + ib, _mm256_set1_ps(d * (0.5 + (aux >> 28) as f32) * 0.5));
                }
            }
        }
        GgmlType::Iq3S => {
            for sb in 0..w.len() / IQ3_S_BYTES {
                let blk = unsafe { p.add(sb * IQ3_S_BYTES) };
                let d = h16(blk);
                for ib in 0..8 {
                    let s = unsafe { *blk.add(106 + ib / 2) };
                    let db = d * (1 + 2 * if ib % 2 == 0 { s & 0xf } else { s >> 4 }) as f32;
                    let (q8, h) = unsafe { (std::ptr::read_unaligned(blk.add(2 + 8 * ib) as *const u64), *blk.add(66 + ib) as usize) };
                    let sg = unsafe { std::ptr::read_unaligned(blk.add(74 + 4 * ib) as *const u32) };
                    let qb = |i: usize| (q8 >> (8 * i)) as u8 as usize;
                    let g1 = |l: usize| unsafe { *IQ3S_GRID.get_unchecked(qb(2 * l) | (h << (8 - 2 * l) & 256)) } as i32;
                    let g2 = |l: usize| unsafe { *IQ3S_GRID.get_unchecked(qb(2 * l + 1) | (h << (7 - 2 * l) & 256)) } as i32;
                    let gv = _mm256_set_epi32(g2(3), g1(3), g2(2), g1(2), g2(1), g1(1), g2(0), g1(0));
                    let wv = _mm256_sign_epi8(gv, unsafe { sign_bytes(sg) });
                    ext!(wv, sb * 8 + ib, _mm256_set1_ps(db));
                }
            }
        }
        GgmlType::Iq2Xxs => {
            for sb in 0..w.len() / IQ2_XXS_BYTES {
                let blk = unsafe { p.add(sb * IQ2_XXS_BYTES) };
                let d = h16(blk) * 0.125;
                for ib in 0..8 {
                    let q = unsafe { blk.add(2 + 8 * ib) };
                    let aux = unsafe { std::ptr::read_unaligned(q.add(4) as *const u32) };
                    let g = |l: usize| unsafe { *IQ2XXS_GRID.get_unchecked(*q.add(l) as usize) } as i64;
                    let ls = (2 * (aux >> 28) + 1) as i16;
                    let s = |l: usize| unsafe { *SIGNS128.get_unchecked((aux >> (7 * l) & 127) as usize) } as i64;
                    grid!(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), _mm256_set_epi64x(s(3), s(2), s(1), s(0)), _mm256_set1_epi16(ls), sb * 8 + ib, d);
                }
            }
        }
        GgmlType::Iq2Xs => {
            for sb in 0..w.len() / IQ2_XS_BYTES {
                let blk = unsafe { p.add(sb * IQ2_XS_BYTES) };
                let d = h16(blk) * 0.125;
                for ib in 0..8 {
                    let q = |l: usize| unsafe { std::ptr::read_unaligned(blk.add(2 + 8 * ib + 2 * l) as *const u16) };
                    let g = |l: usize| unsafe { *IQ2XS_GRID.get_unchecked((q(l) & 511) as usize) } as i64;
                    let s = |l: usize| unsafe { *SIGNS128.get_unchecked((q(l) >> 9) as usize) } as i64;
                    let sc = unsafe { *blk.add(66 + ib) };
                    let (lo, hi) = ((2 * (sc & 15) + 1) as i16, (2 * (sc >> 4) + 1) as i16);
                    grid!(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), _mm256_set_epi64x(s(3), s(2), s(1), s(0)), _mm256_set_m128i(_mm_set1_epi16(hi), _mm_set1_epi16(lo)), sb * 8 + ib, d);
                }
            }
        }
        _ => {
            // IQ2_S
            for sb in 0..w.len() / IQ2_S_BYTES {
                let blk = unsafe { p.add(sb * IQ2_S_BYTES) };
                let d = h16(blk) * 0.125;
                for ib in 0..8 {
                    let h = unsafe { *blk.add(66 + ib) } as usize;
                    let qs = unsafe { std::ptr::read_unaligned(blk.add(2 + 4 * ib) as *const u32) };
                    let g = |l: usize| unsafe { *IQ2S_GRID.get_unchecked((qs >> (8 * l)) as u8 as usize | (h << (8 - 2 * l) & 0x300)) } as i64;
                    let sg = unsafe { std::ptr::read_unaligned(blk.add(34 + 4 * ib) as *const u32) };
                    let sc = unsafe { *blk.add(74 + ib) };
                    let (lo, hi) = ((2 * (sc & 15) + 1) as i16, (2 * (sc >> 4) + 1) as i16);
                    grid!(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), unsafe { sign_bytes(sg) }, _mm256_set_m128i(_mm_set1_epi16(hi), _mm_set1_epi16(lo)), sb * 8 + ib, d);
                }
            }
        }
    }
    let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
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
    if is_ext(ty) {
        return ext_row_dot_scalar(ty, w, xs);
    }
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
        _ if is_ext(ty) => unsafe { ext_dot_avx2(ty, w, xs) },
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
        GgmlType::Iq3Xxs => dequant_iq3_xxs(row, out),
        _ => unreachable!("not an i-quant: {}", ty.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 dots of NVFP4, IQ4_NL, IQ3_XXS and IQ3_S agree with dequantising (random rows of 512 values).
    #[test]
    fn ext_dots_match() {
        if !crate::quant::has_avx2() {
            return;
        }
        let cols = 512;
        let mut s = 7u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        };
        let xq: Vec<i8> = (0..cols).map(|_| (rnd() as i8).max(-127)).collect(); // int8 activations are never -128
        let xd: Vec<f32> = (0..cols / 32).map(|i| 0.01 + i as f32 * 0.001).collect();
        for ty in [GgmlType::Nvfp4, GgmlType::Iq4Nl, GgmlType::Iq3Xxs, GgmlType::Iq3S] {
            let (per, bytes) = ty.block().unwrap();
            let mut w: Vec<u8> = (0..cols / per * bytes).map(|_| rnd()).collect();
            if ty != GgmlType::Nvfp4 {
                for b in w.chunks_exact_mut(bytes) {
                    b[..2].copy_from_slice(&crate::quant::f32_to_f16(0.0123).to_le_bytes());
                }
            } else {
                for b in w.chunks_exact_mut(bytes) {
                    for s in &mut b[..4] {
                        *s &= 0x6f; // finite, moderate scales
                    }
                }
            }
            let want = ext_row_dot_scalar(ty, &w, [(&xd, &xq)])[0];
            let got = unsafe { row_dot_avx2(ty, &w, [(&xd, &xq)]) }[0];
            assert_eq!(got.to_bits(), unsafe { row_dot1_avx2(ty, &w, &xd, &xq) }.to_bits(), "{}: one-row kernel", ty.name());
            assert!((want - got).abs() <= 1e-3 * want.abs().max(1.0), "{}: {want} vs {got}", ty.name());
        }
    }

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

    /// cargo test --release kernel_speed -- --ignored --nocapture: each format's AVX2 row kernel on one thread,
    /// in cache (64 rows of 2304), as ns per 32 values.
    #[test]
    #[ignore]
    fn kernel_speed() {
        let (rows, cols) = (64, 2304);
        let x: Vec<f32> = (0..cols).map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0).collect();
        let mut xq = crate::quant::QuantizedBatch::default();
        xq.quantize(&x, cols);
        let (xd, xqv) = xq.row_parts(0);
        for ty in [GgmlType::Nvfp4, GgmlType::Iq4Nl, GgmlType::Iq3Xxs, GgmlType::Iq3S, GgmlType::Iq2Xxs, GgmlType::Iq2Xs, GgmlType::Iq2S] {
            let (per, bytes) = ty.block().unwrap();
            let rb = cols / per * bytes;
            let mut w = row(ty, rows * cols / per, 99);
            if ty == GgmlType::Nvfp4 {
                for b in w.chunks_exact_mut(bytes) {
                    b[..2].copy_from_slice(&[0x30, 0x31]);
                }
            }
            let mut sink = 0f32;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..200 {
                    for r in 0..rows {
                        sink += unsafe { row_dot1_avx2(ty, &w[r * rb..(r + 1) * rb], xd, xqv) };
                    }
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e9 / (200 * rows * cols / 32) as f64);
            }
            println!("{:8} {best:5.2} ns per 32 values ({:.1} GB/s a thread) {sink:.0}", ty.name(), bytes as f64 * 32.0 / per as f64 / best);
            let mut best4 = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..50 {
                    for r in 0..rows {
                        let v = unsafe { row_dot_avx2(ty, &w[r * rb..(r + 1) * rb], [(xd, xqv); 4]) };
                        sink += v[0] + v[3];
                    }
                }
                best4 = best4.min(t.elapsed().as_secs_f64() * 1e9 / (50 * rows * cols / 32 * 4) as f64);
            }
            println!("         4 rows: {best4:5.2} ns per 32 values per row");
        }
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
                assert_eq!(fast[0].to_bits(), unsafe { row_dot1_avx2(ty, &w, xd, xqv) }.to_bits(), "{} one-row kernel", ty.name());
                for f in fast {
                    assert!((f - want).abs() <= 1e-3 * want.abs().max(1.0), "{} avx2 {f} vs {want}", ty.name());
                }
            }
        }
    }
}

#[cfg(test)]
mod ue4m3_tests {
    /// The table holds ggml_ue4m3_to_fp32's values.
    #[test]
    fn ue4m3_table() {
        for x in 0..256usize {
            let (e, m) = ((x >> 3 & 0xf) as i32, (x & 7) as f32);
            let want = if x == 0 || x == 0x7f { 0.0 } else if e == 0 { m * (2f32).powi(-9) * 0.5 } else { (1.0 + m / 8.0) * (2f32).powi(e - 7) * 0.5 };
            assert_eq!(super::UE4M3[x], want, "byte {x:#x}");
        }
    }
}
