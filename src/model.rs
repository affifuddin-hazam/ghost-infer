//! Llama-family transformer on the CPU: Llama 2/3, SmolLM, Mistral-style GQA, and Qwen3
//! (per-head RMSNorm on queries and keys, NEOX-style RoPE).

use rayon::prelude::*;

use crate::gguf::{Gguf, TensorInfo, Value};
use crate::profile;
use crate::quant::{dot_f32, gate_up_silu, matmul_many, Matrix, QuantizedBatch};

/// Prompts are processed this many tokens at a time (bounds scratch memory).
const MAX_BATCH: usize = 256;

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub max_ctx: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    /// NEOX RoPE rotates (x[i], x[i + rope_dim/2]); otherwise adjacent pairs (x[2i], x[2i+1]).
    /// GGUF converters permute llama's Q/K for adjacent pairs but leave Qwen's as trained.
    pub rope_neox: bool,
    pub norm_eps: f32,
}

pub const SUPPORTED_ARCHS: [&str; 2] = ["llama", "qwen3"];

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Config, String> {
        let arch = g.get("general.architecture").and_then(Value::as_str).unwrap_or("").to_string();
        if !SUPPORTED_ARCHS.contains(&arch.as_str()) {
            return Err(format!("architecture \"{arch}\" is not supported yet (supported: {})", SUPPORTED_ARCHS.join(", ")));
        }
        let u = |k: &str| g.get(&format!("{arch}.{k}")).and_then(Value::as_u64).map(|v| v as usize);
        let f = |k: &str| g.get(&format!("{arch}.{k}")).and_then(Value::as_f32);
        let need = |k: &str| u(k).ok_or_else(|| format!("missing {arch}.{k}"));

        let dim = need("embedding_length")?;
        let n_heads = need("attention.head_count")?;
        let head_dim = u("attention.key_length").unwrap_or(dim / n_heads);
        Ok(Config {
            dim,
            n_layers: need("block_count")?,
            n_heads,
            n_kv_heads: u("attention.head_count_kv").unwrap_or(n_heads),
            head_dim,
            ffn_dim: need("feed_forward_length")?,
            // not every GGUF has <arch>.vocab_size; the embedding table always says
            vocab: u("vocab_size").or_else(|| g.tensor("token_embd.weight").and_then(|t| t.dims.get(1)).map(|&v| v as usize)).ok_or("cannot tell the vocabulary size")?,
            max_ctx: need("context_length")?,
            rope_dim: u("rope.dimension_count").unwrap_or(head_dim),
            rope_base: f("rope.freq_base").unwrap_or(10_000.0),
            rope_neox: arch == "qwen3",
            norm_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
        })
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: Matrix<'a>,
    wk: Matrix<'a>,
    wv: Matrix<'a>,
    wo: Matrix<'a>,
    q_norm: Option<Vec<f32>>, // Qwen3: RMSNorm over each query head before RoPE
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: Matrix<'a>,
    w_up: Matrix<'a>,
    w_down: Matrix<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    token_embd: Matrix<'a>,
    layers: Vec<Layer<'a>>,
    output_norm: Vec<f32>,
    output: Matrix<'a>,
}

fn tensor<'g>(g: &'g Gguf, name: &str) -> Result<&'g TensorInfo, String> {
    g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))
}

fn matrix<'g>(g: &'g Gguf, name: &str) -> Result<Matrix<'g>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    if t.dims.len() != 2 {
        return Err(format!("tensor {name} is not 2-D"));
    }
    Ok(Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, cols: t.dims[0] as usize, rows: t.dims[1] as usize })
}

fn vector(g: &Gguf, name: &str) -> Result<Vec<f32>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    let n = t.n_elements() as usize;
    let mut out = vec![0f32; n];
    let m = Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n };
    m.row_f32(0, &mut out);
    Ok(out)
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Model<'a>, String> {
        let cfg = Config::from_gguf(g)?;
        let token_embd = matrix(g, "token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(Layer {
                attn_norm: vector(g, &p("attn_norm"))?,
                wq: matrix(g, &p("attn_q"))?,
                wk: matrix(g, &p("attn_k"))?,
                wv: matrix(g, &p("attn_v"))?,
                wo: matrix(g, &p("attn_output"))?,
                q_norm: g.tensor(&p("attn_q_norm")).map(|_| vector(g, &p("attn_q_norm"))).transpose()?,
                k_norm: g.tensor(&p("attn_k_norm")).map(|_| vector(g, &p("attn_k_norm"))).transpose()?,
                ffn_norm: vector(g, &p("ffn_norm"))?,
                w_gate: matrix(g, &p("ffn_gate"))?,
                w_up: matrix(g, &p("ffn_up"))?,
                w_down: matrix(g, &p("ffn_down"))?,
            });
        }
        // small models tie the output projection to the embedding table
        let output = if g.tensor("output.weight").is_some() { matrix(g, "output.weight")? } else { token_embd };
        Ok(Model { output_norm: vector(g, "output_norm.weight")?, cfg, token_embd, layers, output })
    }
}

/// Per-sequence state: KV cache plus scratch buffers that grow to the largest batch seen.
pub struct Session {
    pub pos: usize,
    pub ctx: usize,
    k_cache: Vec<f32>, // [layer][pos][kv_dim]
    v_cache: Vec<f32>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xq: QuantizedBatch,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att_out: Vec<f32>,
    att_partial: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    pub logits: Vec<f32>,
    rope_inv_freq: Vec<f32>,
}

impl Session {
    pub fn new(cfg: &Config, ctx: usize) -> Session {
        let ctx = ctx.min(cfg.max_ctx);
        let kv = cfg.n_layers * ctx * cfg.kv_dim();
        let half = cfg.rope_dim / 2;
        Session {
            pos: 0,
            ctx,
            k_cache: vec![0.0; kv],
            v_cache: vec![0.0; kv],
            x: Vec::new(),
            xb: Vec::new(),
            xq: QuantizedBatch::default(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            att_out: Vec::new(),
            att_partial: Vec::new(),
            proj: Vec::new(),
            gate: Vec::new(),
            logits: vec![0.0; cfg.vocab],
            rope_inv_freq: (0..half).map(|i| cfg.rope_base.powf(-(2.0 * i as f32) / cfg.rope_dim as f32)).collect(),
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    fn size_for(&mut self, cfg: &Config, n: usize) {
        for (buf, width) in [
            (&mut self.x, cfg.dim),
            (&mut self.xb, cfg.dim),
            (&mut self.q, cfg.q_dim()),
            (&mut self.k, cfg.kv_dim()),
            (&mut self.v, cfg.kv_dim()),
            (&mut self.att_out, cfg.q_dim()),
            (&mut self.proj, cfg.dim),
            (&mut self.gate, cfg.ffn_dim),
        ] {
            buf.resize(n * width, 0.0);
        }
    }
}

fn rms_norm_rows(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let dim = w.len();
    for (o, xr) in out.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
        let ms = dot_f32(xr, xr) / dim as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for ((ov, v), g) in o.iter_mut().zip(xr).zip(w) {
            *ov = v * scale * g;
        }
    }
}

/// RMSNorm of every `w.len()`-sized chunk in place (Qwen3's per-head query/key norm).
fn rms_norm_inplace(buf: &mut [f32], w: &[f32], eps: f32) {
    for chunk in buf.chunks_exact_mut(w.len()) {
        let scale = 1.0 / (dot_f32(chunk, chunk) / w.len() as f32 + eps).sqrt();
        for (v, g) in chunk.iter_mut().zip(w) {
            *v *= scale * g;
        }
    }
}

/// Rotary position embedding on every head of `row`. Pairs are (x[2i], x[2i+1]) for GGUF llama
/// weights, or (x[i], x[i + rope_dim/2]) for NEOX-style models such as Qwen.
fn rope(row: &mut [f32], head_dim: usize, rope_dim: usize, pos: usize, inv_freq: &[f32], neox: bool) {
    let half = rope_dim / 2;
    for head in row.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (s, c) = (pos as f32 * inv_freq[i]).sin_cos();
            let (ia, ib) = if neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let (a, b) = (head[ia], head[ib]);
            head[ia] = a * c - b * s;
            head[ib] = a * s + b * c;
        }
    }
}

impl Model<'_> {
    /// Feeds a whole prompt; leaves the logits for the token after it in `s.logits`.
    pub fn prefill(&self, tokens: &[u32], s: &mut Session) -> Result<(), String> {
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_BATCH).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            self.forward_batch(chunk, s, i + 1 == chunks.len())?;
        }
        Ok(())
    }

    /// Runs one token; leaves next-token logits in `s.logits` if `want_logits`.
    pub fn forward(&self, token: u32, s: &mut Session, want_logits: bool) -> Result<(), String> {
        self.forward_batch(&[token], s, want_logits)
    }

    /// Runs `tokens` at positions `s.pos..s.pos + n` in one pass, so every weight is read once per batch.
    fn forward_batch(&self, tokens: &[u32], s: &mut Session, want_logits: bool) -> Result<(), String> {
        let c = &self.cfg;
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if s.pos + n > s.ctx {
            return Err(format!("context full ({} tokens)", s.ctx));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= self.token_embd.rows) {
            return Err(format!("token id {bad} out of range"));
        }
        s.size_for(c, n);
        let (dim, kv_dim, q_dim, hd) = (c.dim, c.kv_dim(), c.q_dim(), c.head_dim);
        let pos0 = s.pos;

        profile::time(0, || {
            for (b, &t) in tokens.iter().enumerate() {
                self.token_embd.row_f32(t as usize, &mut s.x[b * dim..(b + 1) * dim]);
            }
        });

        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention ---
            profile::time(1, || {
                rms_norm_rows(&mut s.xb, &s.x, &layer.attn_norm, c.norm_eps);
                s.xq.quantize(&s.xb, dim);
            });
            profile::time(2, || {
                matmul_many(&mut [(&layer.wq, &mut s.q), (&layer.wk, &mut s.k), (&layer.wv, &mut s.v)], &s.xb, &s.xq);
            });
            let layer_base = l * s.ctx * kv_dim;
            profile::time(3, || {
                if let Some(w) = &layer.q_norm {
                    rms_norm_inplace(&mut s.q, w, c.norm_eps);
                }
                if let Some(w) = &layer.k_norm {
                    rms_norm_inplace(&mut s.k, w, c.norm_eps);
                }
                for b in 0..n {
                    rope(&mut s.q[b * q_dim..(b + 1) * q_dim], hd, c.rope_dim, pos0 + b, &s.rope_inv_freq, c.rope_neox);
                    rope(&mut s.k[b * kv_dim..(b + 1) * kv_dim], hd, c.rope_dim, pos0 + b, &s.rope_inv_freq, c.rope_neox);
                }
                let at = layer_base + pos0 * kv_dim;
                s.k_cache[at..at + n * kv_dim].copy_from_slice(&s.k);
                s.v_cache[at..at + n * kv_dim].copy_from_slice(&s.v);
            });
            profile::time(4, || self.attention(s, layer_base, n));

            profile::time(5, || {
                s.xq.quantize(&s.att_out, q_dim);
                layer.wo.matmul(&s.att_out, &s.xq, &mut s.proj);
                for (x, p) in s.x.iter_mut().zip(&s.proj) {
                    *x += p;
                }
            });

            // --- feed-forward (SwiGLU) ---
            profile::time(1, || {
                rms_norm_rows(&mut s.xb, &s.x, &layer.ffn_norm, c.norm_eps);
                s.xq.quantize(&s.xb, dim);
            });
            profile::time(6, || {
                gate_up_silu(&layer.w_gate, &layer.w_up, &s.xb, &s.xq, &mut s.gate);
                s.xq.quantize(&s.gate, c.ffn_dim);
                layer.w_down.matmul(&s.gate, &s.xq, &mut s.proj);
                for (x, f) in s.x.iter_mut().zip(&s.proj) {
                    *x += f;
                }
            });
        }

        if want_logits {
            profile::time(7, || {
                let last = &s.x[(n - 1) * dim..n * dim];
                let mut normed = vec![0f32; dim];
                rms_norm_rows(&mut normed, last, &self.output_norm, c.norm_eps);
                s.xq.quantize(&normed, dim);
                self.output.matmul(&normed, &s.xq, &mut s.logits);
            });
        }
        s.pos += n;
        Ok(())
    }

    /// Causal multi-head attention for the batch, reading keys/values from the cache.
    fn attention(&self, s: &mut Session, layer_base: usize, n: usize) {
        let c = &self.cfg;
        let (kv_dim, q_dim, hd) = (c.kv_dim(), c.q_dim(), c.head_dim);
        let group = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos0 = s.pos;
        let k_cache = &s.k_cache[layer_base..layer_base + (pos0 + n) * kv_dim];
        let v_cache = &s.v_cache[layer_base..layer_base + (pos0 + n) * kv_dim];
        let q = &s.q;
        let span_of = |b: usize| pos0 + b + 1; // causal: token b sees positions 0..=pos0+b

        // Work unit = (token, kv head, slice of the context). The `group` query heads sharing a kv head
        // are adjacent in q and att_out, so each key/value row is read once for all of them. When there
        // are fewer (token, kv head) pairs than threads (single-token decoding), the context is split into
        // slices whose partial softmaxes are merged afterwards ("flash-decoding").
        let pairs = n * c.n_kv_heads;
        let threads = rayon::current_num_threads();
        let splits = if pairs >= threads { 1 } else { threads.div_ceil(pairs).min(span_of(0).div_ceil(64)).max(1) };
        let unit = group * hd + 2 * group; // per slice: unnormalised outputs, then max and sum per head
        s.att_partial.resize(pairs * splits * unit, 0.0);

        let slice_task = |scores: &mut Vec<f32>, (idx, part): (usize, &mut [f32])| {
            let (pair, split) = (idx / splits, idx % splits);
            let (b, kvh) = (pair / c.n_kv_heads, pair % c.n_kv_heads);
            let span = span_of(b);
            let per = span.div_ceil(splits);
            let (t0, t1) = ((split * per).min(span), ((split + 1) * per).min(span));
            let len = t1 - t0;
            let q0 = b * q_dim + kvh * group * hd;
            let qs = &q[q0..q0 + group * hd];
            let (out, stats) = part.split_at_mut(group * hd);
            out.fill(0.0);
            if len == 0 {
                for g in 0..group {
                    stats[2 * g] = f32::NEG_INFINITY;
                    stats[2 * g + 1] = 0.0;
                }
                return;
            }
            scores.resize(group * len, 0.0);
            for t in t0..t1 {
                let kt = &k_cache[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                for (g, qh) in qs.chunks_exact(hd).enumerate() {
                    scores[g * len + (t - t0)] = dot_f32(qh, kt) * scale;
                }
            }
            for (g, row) in scores.chunks_exact_mut(len).enumerate() {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for sc in row.iter_mut() {
                    *sc = (*sc - max).exp();
                    total += *sc;
                }
                stats[2 * g] = max;
                stats[2 * g + 1] = total;
            }
            for t in t0..t1 {
                let vt = &v_cache[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                for (g, oh) in out.chunks_exact_mut(hd).enumerate() {
                    let w = scores[g * len + (t - t0)];
                    for (o, v) in oh.iter_mut().zip(vt) {
                        *o += w * v;
                    }
                }
            }
        };
        // a single token over a short context is too little work to be worth waking threads
        if n * c.n_heads * (pos0 + n) * hd < 1 << 16 {
            let mut scores = Vec::new();
            s.att_partial.chunks_mut(unit).enumerate().for_each(|t| slice_task(&mut scores, t));
        } else {
            s.att_partial.par_chunks_mut(unit).enumerate().for_each_init(Vec::new, slice_task);
        }

        // merge the slices of each (token, kv head) into normalised head outputs
        for pair in 0..pairs {
            let (b, kvh) = (pair / c.n_kv_heads, pair % c.n_kv_heads);
            let parts = &s.att_partial[pair * splits * unit..(pair + 1) * splits * unit];
            for g in 0..group {
                let m = parts.chunks_exact(unit).map(|p| p[group * hd + 2 * g]).fold(f32::NEG_INFINITY, f32::max);
                let o0 = b * q_dim + (kvh * group + g) * hd;
                let out = &mut s.att_out[o0..o0 + hd];
                out.fill(0.0);
                let mut total = 0.0;
                for p in parts.chunks_exact(unit) {
                    let w = (p[group * hd + 2 * g] - m).exp();
                    if w == 0.0 {
                        continue;
                    }
                    total += p[group * hd + 2 * g + 1] * w;
                    for (o, v) in out.iter_mut().zip(&p[g * hd..(g + 1) * hd]) {
                        *o += w * v;
                    }
                }
                let inv = 1.0 / total;
                out.iter_mut().for_each(|o| *o *= inv);
            }
        }
    }
}
