//! LoRA adapters applied at run time, in llama.cpp's GGUF adapter format (what its convert_lora_to_gguf.py
//! writes, and scripts/lora-to-gguf.py here): `general.type = "adapter"`, `adapter.type = "lora"`,
//! `adapter.lora.alpha`, and for each adapted weight `<base name>.lora_a` (ne = [cols, r]) and
//! `<base name>.lora_b` (ne = [r, rows]). A layer's output becomes W·x + scale · B·(A·x), with
//! scale = user scale · alpha / r (the user scale alone when alpha is 0, as llama.cpp does).
//!
//! The base weights stay as they are in the model file: the adapter's small matrices are kept in f32 and
//! added on top, so a delta smaller than a quantisation step is not lost to requantising.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::gguf::Gguf;
use crate::quant::Matrix;

/// One adapted weight: A is r rows of `cols`, B is `rows` rows of r, both row-major f32.
#[derive(Clone)]
pub struct Pair {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub r: usize,
    pub rows: usize,
    pub cols: usize,
    pub scale: f32,
}

impl Pair {
    /// `out[b * stride + off + o] += scale · Σ_j B[o][j] · (A·x_b)[j]` for each of the n rows of `x`.
    pub fn apply(&self, x: &[f32], out: &mut [f32], stride: usize, off: usize) {
        let n = x.len() / self.cols;
        let mut t = vec![0f32; self.r];
        for b in 0..n {
            let xb = &x[b * self.cols..(b + 1) * self.cols];
            for (j, tj) in t.iter_mut().enumerate() {
                let row = &self.a[j * self.cols..(j + 1) * self.cols];
                *tj = row.iter().zip(xb).map(|(a, x)| a * x).sum::<f32>();
            }
            let o0 = b * stride + off;
            for (o, y) in out[o0..o0 + self.rows].iter_mut().enumerate() {
                let row = &self.b[o * self.r..(o + 1) * self.r];
                *y += self.scale * row.iter().zip(&t).map(|(b, t)| b * t).sum::<f32>();
            }
        }
    }
}

/// The weights of one layer an adapter may change (the projections of attention and of a dense SwiGLU).
#[derive(Clone, Default)]
pub struct LayerLora {
    pub q: Option<Pair>,
    pub k: Option<Pair>,
    pub v: Option<Pair>,
    pub o: Option<Pair>,
    pub gate: Option<Pair>,
    pub up: Option<Pair>,
    pub down: Option<Pair>,
}

impl LayerLora {
    pub fn is_empty(&self) -> bool {
        self.pairs().iter().all(|p| p.is_none())
    }

    pub fn pairs(&self) -> [&Option<Pair>; 7] {
        [&self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down]
    }

    fn slot(&mut self, name: &str) -> Option<&mut Option<Pair>> {
        Some(match name {
            "attn_q" => &mut self.q,
            "attn_k" => &mut self.k,
            "attn_v" => &mut self.v,
            "attn_output" => &mut self.o,
            "ffn_gate" => &mut self.gate,
            "ffn_up" => &mut self.up,
            "ffn_down" => &mut self.down,
            _ => return None,
        })
    }
}

/// A loaded adapter: per layer, the weights it changes.
pub struct Adapter {
    pub path: PathBuf,
    pub alpha: f32,
    pub rank: usize,
    pub layers: Vec<LayerLora>,
}

impl Adapter {
    /// Reads an adapter file; `scale` is the user's strength (1.0 = as trained).
    pub fn load(path: &Path, scale: f32) -> Result<Adapter, String> {
        let g = Gguf::open(path)?;
        let text = |k: &str| g.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        if text("general.type") != "adapter" || text("adapter.type") != "lora" {
            return Err(format!("{} is not a LoRA adapter (general.type \"{}\", adapter.type \"{}\")", path.display(), text("general.type"), text("adapter.type")));
        }
        let alpha = g.get("adapter.lora.alpha").and_then(|v| v.as_f32()).unwrap_or(0.0);
        let mut layers: Vec<LayerLora> = Vec::new();
        let mut rank = 0;
        for t in &g.tensors {
            let Some(base) = t.name.strip_suffix(".lora_a") else {
                if t.name.ends_with(".lora_b") || t.name.ends_with("_norm.weight") {
                    continue; // B is read with its A; llama.cpp skips adapted norms too
                }
                return Err(format!("{}: tensor {} is not part of a LoRA pair", path.display(), t.name));
            };
            let b = g.tensor(&format!("{base}.lora_b")).ok_or_else(|| format!("{}: {base} has lora_a but no lora_b", path.display()))?;
            // blk.<layer>.<name>.weight
            let parts: Vec<&str> = base.split('.').collect();
            let (l, name) = match parts.as_slice() {
                ["blk", l, name, "weight"] => (l.parse::<usize>().map_err(|_| format!("{}: bad layer in {base}", path.display()))?, *name),
                _ => return Err(format!("{}: {base} is not a layer weight this engine can adapt", path.display())),
            };
            if t.dims.len() != 2 || b.dims.len() != 2 {
                return Err(format!("{}: {base}: LoRA tensors must be 2-D", path.display()));
            }
            let (cols, r) = (t.dims[0] as usize, t.dims[1] as usize);
            let (rb, rows) = (b.dims[0] as usize, b.dims[1] as usize);
            if r != rb || r == 0 {
                return Err(format!("{}: {base}: lora_a is {cols}x{r} but lora_b is {rb}x{rows} (A must be transposed as llama.cpp expects)", path.display()));
            }
            let read = |ti: &crate::gguf::TensorInfo, rows: usize, cols: usize| -> Result<Vec<f32>, String> {
                if ti.ty.block().is_none() {
                    return Err(format!("{}: {} uses {}, which cannot be read", path.display(), ti.name, ti.ty.name()));
                }
                let m = Matrix { data: &g.mmap[ti.offset..ti.offset + ti.n_bytes], ty: ti.ty, rows, cols };
                let mut out = vec![0f32; rows * cols];
                for (i, row) in out.chunks_exact_mut(cols).enumerate() {
                    m.row_f32(i, row);
                }
                Ok(out)
            };
            let pair = Pair {
                a: read(t, r, cols)?,
                b: read(b, rows, r)?,
                r,
                rows,
                cols,
                scale: if alpha != 0.0 { scale * alpha / r as f32 } else { scale },
            };
            rank = rank.max(r);
            if layers.len() <= l {
                layers.resize_with(l + 1, LayerLora::default);
            }
            let slot = layers[l].slot(name).ok_or_else(|| format!("{}: {base}: this engine adapts attn_q/k/v/output and ffn_gate/up/down only", path.display()))?;
            *slot = Some(pair);
        }
        if rank == 0 {
            return Err(format!("{} holds no LoRA weights", path.display()));
        }
        Ok(Adapter { path: path.to_path_buf(), alpha, rank, layers })
    }

    /// The pairs of layer l (an empty set past the adapter's last layer).
    pub fn layer(&self, l: usize) -> &LayerLora {
        static EMPTY: LayerLora = LayerLora { q: None, k: None, v: None, o: None, gate: None, up: None, down: None };
        self.layers.get(l).unwrap_or(&EMPTY)
    }

    /// Checks a pair against the base weight it adapts.
    pub fn check(pair: &Option<Pair>, rows: usize, cols: usize, what: &str) -> Result<(), String> {
        match pair {
            Some(p) if p.rows != rows || p.cols != cols => Err(format!("the adapter's {what} is {}x{}, the model's is {rows}x{cols}: it was trained on another model", p.rows, p.cols)),
            _ => Ok(()),
        }
    }
}

/// Loads the adapter for `model` (see `for_model`) and hands it to `attach`. One given with --lora must work;
/// one found in the nightly folder that does not fit is reported and left out, so the model still runs.
pub fn attach_to(model: &Path, attach: impl FnOnce(&Adapter) -> Result<(), String>) -> Result<Option<PathBuf>, String> {
    let explicit = CHOSEN.lock().unwrap().is_some();
    let Some((path, scale)) = for_model(model) else { return Ok(None) };
    let result = Adapter::load(&path, scale).and_then(|ad| {
        attach(&ad)?;
        eprintln!("[lora: {} (rank {}, alpha {}, scale {scale})]", path.display(), ad.rank, ad.alpha);
        Ok(())
    });
    match result {
        Ok(()) => Ok(Some(path)),
        Err(e) if explicit => Err(format!("--lora {}: {e}", path.display())),
        Err(e) => {
            eprintln!("[lora: {} left out: {e}]", path.display());
            Ok(None)
        }
    }
}

/// The adapter asked for on the command line (`--lora file[:scale]`), if any.
static CHOSEN: Mutex<Option<(PathBuf, f32)>> = Mutex::new(None);

pub fn choose(spec: &str) -> Result<(), String> {
    // a Windows path has a colon after its drive letter: only a trailing ":<number>" is a scale
    let (path, scale) = match spec.rsplit_once(':') {
        Some((p, s)) if !p.is_empty() && s.parse::<f32>().is_ok() => (p, s.parse::<f32>().unwrap()),
        _ => (spec, 1.0),
    };
    *CHOSEN.lock().unwrap() = Some((PathBuf::from(path), scale));
    Ok(())
}

/// The adapter for a model file: the one given with --lora, else `<data>\lora\<model name>.gguf` when the
/// nightly training has written one (GHOST_INFER_NO_LORA=1 turns that off). None: run the model as it is.
pub fn for_model(model: &Path) -> Option<(PathBuf, f32)> {
    if let Some(chosen) = CHOSEN.lock().unwrap().clone() {
        return Some(chosen);
    }
    // GHOST_INFER_LORA=<file>[:scale]: as --lora, for the commands that take no options (top)
    if let Some(spec) = std::env::var_os("GHOST_INFER_LORA").filter(|v| !v.is_empty()) {
        let _ = choose(&spec.to_string_lossy());
        return CHOSEN.lock().unwrap().clone();
    }
    if std::env::var_os("GHOST_INFER_NO_LORA").is_some() {
        return None;
    }
    // the model's own name, dots and all (Qwen3-4B.Q4_K_M.gguf -> lora\Qwen3-4B.Q4_K_M.gguf)
    let stem = model.file_stem()?.to_string_lossy().into_owned();
    let file = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("ghost-infer").join("lora").join(format!("{stem}.gguf"));
    file.is_file().then_some((file, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pair_adds_scaled_b_times_a_x_at_its_offset() {
        // A: 2 x 3, B: 2 x 2, scale 0.5; two rows of x into a stride-4 output at offset 1
        let p = Pair { a: vec![1.0, 0.0, 2.0, 0.0, 1.0, 0.0], b: vec![1.0, 1.0, 0.0, 2.0], r: 2, rows: 2, cols: 3, scale: 0.5 };
        let x = [1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let mut out = vec![10.0f32; 8];
        p.apply(&x, &mut out, 4, 1);
        // row 0: A x = [7, 2]; B (A x) = [9, 4]; halved and added after column 0
        assert_eq!(&out[..4], &[10.0, 14.5, 12.0, 10.0]);
        // row 1: A x = [0, 1]; B (A x) = [1, 2]
        assert_eq!(&out[4..], &[10.0, 10.5, 11.0, 10.0]);
    }

    #[test]
    fn a_trailing_number_is_the_strength_but_a_drive_letter_is_not() {
        choose(r"C:\lora\a.gguf:0.5").unwrap();
        assert_eq!(CHOSEN.lock().unwrap().clone(), Some((PathBuf::from(r"C:\lora\a.gguf"), 0.5)));
        choose(r"C:\lora\a.gguf").unwrap();
        assert_eq!(CHOSEN.lock().unwrap().clone(), Some((PathBuf::from(r"C:\lora\a.gguf"), 1.0)));
        *CHOSEN.lock().unwrap() = None;
    }
}
