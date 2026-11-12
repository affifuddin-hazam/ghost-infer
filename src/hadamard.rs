//! PrismML's Hadamard-folded weights (`prism.hadamard.*`, the Ternary Bonsai models). A folded weight is
//! W_f = W·D·H: the original weight with ±1 signs D (one vector per input width) and the normalized block
//! Walsh-Hadamard matrix H folded in. Since D·D = H·H = I, W·x = W_f·(H·(D·x)): the input of each folded matmul
//! gets the signs, then the transform. The token embedding stores rotated rows, z = H·D·h, so a looked-up row
//! goes back with h = D·(H·z). Follows llama.cpp's load_hparams_hadamard and build_hadamard_input (b11747).

use std::collections::{HashMap, HashSet};

use crate::gguf::Gguf;

pub struct Hadamard {
    /// the transform's block (a power of two; 1024 in the Bonsai files)
    pub block: usize,
    /// input width -> signs (empty map: identity sign mode)
    signs: HashMap<usize, Vec<f32>>,
    /// the weights folded this way
    folded: HashSet<String>,
    /// token_embd.weight stores rotated rows
    pub embd_inverse: bool,
    /// ssm_out was folded with its value heads grouped by key head, while the DeltaNet gives them tiled
    pub gdn_v_grouped: bool,
}

impl Hadamard {
    /// The file's Hadamard folding, if it has one; an error for anything this engine does not apply exactly.
    pub fn from_gguf(g: &Gguf) -> Result<Option<Hadamard>, String> {
        let key = |k: &str| g.get(&format!("prism.hadamard.{k}"));
        let Some(version) = key("version") else { return Ok(None) };
        let bad = |what: String| Err(format!("prism.hadamard: {what}"));
        if version.as_u64() != Some(1) {
            return bad(format!("version {version} is not supported (1 is)"));
        }
        if key("tied_output").and_then(|v| v.as_bool()).unwrap_or(false) {
            return bad("a tied output head is not supported".into());
        }
        let text = |k: &str| key(k).and_then(|v| v.as_str()).unwrap_or("");
        if text("transform") != "normalized-sylvester-walsh-hadamard" {
            return bad(format!("transform \"{}\" is not supported", text("transform")));
        }
        if text("axis") != "input-last-dimension" {
            return bad(format!("axis \"{}\" is not supported", text("axis")));
        }
        let block = key("block_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if block == 0 || !block.is_power_of_two() {
            return bad(format!("block size {block} is not a power of two"));
        }
        let ints = |k: &str| -> Vec<i64> {
            key(k).and_then(|v| v.as_array()).unwrap_or(&[]).iter().map(|v| match v {
                crate::gguf::Value::I32(i) => *i as i64,
                crate::gguf::Value::I64(i) => *i,
                crate::gguf::Value::I8(i) => *i as i64,
                crate::gguf::Value::I16(i) => *i as i64,
                other => other.as_u64().map(|u| u as i64).unwrap_or(0),
            }).collect()
        };
        let mut signs = HashMap::new();
        match text("sign_mode") {
            "identity" => {}
            "explicit" => {
                let (widths, values) = (ints("sign_widths"), ints("sign_values"));
                if widths.is_empty() {
                    return bad("explicit signs with no widths".into());
                }
                let mut off = 0;
                for w in widths {
                    let w = w as usize;
                    if w == 0 || w % block != 0 || off + w > values.len() || signs.contains_key(&w) {
                        return bad(format!("bad sign width {w}"));
                    }
                    let v = &values[off..off + w];
                    if v.iter().any(|&s| s != 1 && s != -1) {
                        return bad("sign values must be +1 or -1".into());
                    }
                    signs.insert(w, v.iter().map(|&s| s as f32).collect::<Vec<f32>>());
                    off += w;
                }
                if off != values.len() {
                    return bad("sign values do not match their widths".into());
                }
            }
            other => return bad(format!("sign mode \"{other}\" is not supported")),
        }
        let names = |k: &str| -> Vec<String> { key(k).and_then(|v| v.as_array()).unwrap_or(&[]).iter().filter_map(|v| v.as_str().map(String::from)).collect() };
        let folded: HashSet<String> = names("weight_names").into_iter().collect();
        if folded.is_empty() {
            return bad("no folded weights listed".into());
        }
        let inverse = names("inverse_weight_names");
        if inverse.iter().any(|n| n != "token_embd.weight") {
            return bad("only token_embd.weight can be read back with the inverse".into());
        }
        Ok(Some(Hadamard { block, signs, folded, embd_inverse: !inverse.is_empty(), gdn_v_grouped: key("gdn_v_grouped").and_then(|v| v.as_bool()).unwrap_or(false) }))
    }

    pub fn is_folded(&self, name: &str) -> bool {
        self.folded.contains(name)
    }

    /// The input widths that have signs (none in identity sign mode).
    pub fn widths(&self) -> Vec<usize> {
        self.signs.keys().copied().collect()
    }

    pub fn n_folded(&self) -> usize {
        self.folded.len()
    }

    /// The signs for inputs `width` wide (None: identity sign mode).
    pub fn signs(&self, width: usize) -> Result<Option<&[f32]>, String> {
        if self.signs.is_empty() {
            return Ok(None);
        }
        self.signs.get(&width).map(|s| Some(&s[..])).ok_or_else(|| format!("prism.hadamard: no signs for input width {width}"))
    }

    /// Checks that every folded weight of `width` columns can be transformed.
    pub fn check_width(&self, width: usize) -> Result<(), String> {
        if width % self.block != 0 {
            return Err(format!("prism.hadamard: width {width} is not a multiple of the block {}", self.block));
        }
        self.signs(width).map(|_| ())
    }

    /// The input of a folded matmul, in place, every `width`-wide row of `x`: (the grouped head order for a
    /// DeltaNet ssm_out, `perm` = (head size, key heads, value heads per key head)), the signs, then H.
    pub fn forward(&self, x: &mut [f32], width: usize, perm: Option<(usize, usize, usize)>) {
        let signs = self.signs(width).expect("checked at load");
        let mut tmp = Vec::new();
        for row in x.chunks_exact_mut(width) {
            if let Some((hd, nk, rep)) = perm {
                // tiled [hd, nk, rep] (key head fastest after the head's values) -> grouped [hd, rep, nk]
                tmp.clear();
                tmp.extend_from_slice(row);
                for k in 0..nk {
                    for r in 0..rep {
                        let src = (r * nk + k) * hd;
                        let dst = (k * rep + r) * hd;
                        row[dst..dst + hd].copy_from_slice(&tmp[src..src + hd]);
                    }
                }
            }
            if let Some(s) = signs {
                row.iter_mut().zip(s).for_each(|(v, s)| *v *= s);
            }
            for b in row.chunks_exact_mut(self.block) {
                fwht(b);
            }
        }
    }

    /// One stored row of a folded weight (or of the embedding) back to the original weight's row, in place:
    /// H, the signs, then (ssm_out) the tiled head order back. For checking the folding against an unfolded file.
    pub fn unfold_row(&self, row: &mut [f32], perm: Option<(usize, usize, usize)>) {
        let width = row.len();
        self.inverse(row, width);
        if let Some((hd, nk, rep)) = perm {
            let tmp = row.to_vec();
            for k in 0..nk {
                for r in 0..rep {
                    let (tiled, grouped) = ((r * nk + k) * hd, (k * rep + r) * hd);
                    row[tiled..tiled + hd].copy_from_slice(&tmp[grouped..grouped + hd]);
                }
            }
        }
    }

    /// A looked-up token embedding row back to the model's basis, in place: H, then the signs.
    pub fn inverse(&self, x: &mut [f32], width: usize) {
        let signs = self.signs(width).expect("checked at load");
        for row in x.chunks_exact_mut(width) {
            for b in row.chunks_exact_mut(self.block) {
                fwht(b);
            }
            if let Some(s) = signs {
                row.iter_mut().zip(s).for_each(|(v, s)| *v *= s);
            }
        }
    }
}

/// The normalized Walsh-Hadamard transform of one block (Sylvester order; its own inverse), in place.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

#[cfg(test)]
mod tests {
    use super::fwht;

    #[test]
    fn fwht_is_its_own_inverse_and_matches_the_matrix() {
        let n = 16;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut y = x.clone();
        fwht(&mut y);
        // H[i][j] = (-1)^popcount(i & j) / sqrt(n) (Sylvester)
        for (i, &yi) in y.iter().enumerate() {
            let want: f32 = (0..n).map(|j| if (i & j).count_ones() % 2 == 0 { x[j] } else { -x[j] }).sum::<f32>() / (n as f32).sqrt();
            assert!((yi - want).abs() < 1e-5);
        }
        fwht(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
