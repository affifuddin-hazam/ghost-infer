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
    /// Mixture of experts (qwen3moe; 0 for dense models): experts per layer, experts each token
    /// uses, and an expert's hidden size.
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub expert_ffn_dim: usize,
}

pub const SUPPORTED_ARCHS: [&str; 3] = ["llama", "qwen3", "qwen3moe"];

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
            rope_neox: arch.starts_with("qwen3"),
            norm_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            n_experts: u("expert_count").unwrap_or(0),
            n_experts_used: u("expert_used_count").unwrap_or(0),
            expert_ffn_dim: u("expert_feed_forward_length").unwrap_or(0),
        })
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
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
    ffn: Ffn<'a>,
}

enum Ffn<'a> {
    Dense { gate: Matrix<'a>, up: Matrix<'a>, down: Matrix<'a> },
    /// router: n_experts rows of dim f32 logit weights
    Moe { router: Vec<f32>, experts: Vec<Expert<'a>> },
}

/// One expert's SwiGLU feed-forward, inside the model file.
#[derive(Clone, Copy)]
pub struct Expert<'a> {
    pub gate: Matrix<'a>,
    pub up: Matrix<'a>,
    pub down: Matrix<'a>,
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

/// A 3-D expert tensor (cols x rows x experts, every expert's matrix stored whole) as one matrix per expert.
fn expert_matrices<'g>(g: &'g Gguf, name: &str) -> Result<Vec<Matrix<'g>>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    let [cols, rows, n] = t.dims[..] else { return Err(format!("tensor {name} is not 3-D")) };
    let (cols, rows, n) = (cols as usize, rows as usize, n as usize);
    let bytes = t.n_bytes / n;
    Ok((0..n).map(|e| Matrix { data: &g.mmap[t.offset + e * bytes..t.offset + (e + 1) * bytes], ty: t.ty, rows, cols }).collect())
}

/// Every layer's experts (empty for a dense model), for the CPU expert path of the GPU backend too.
pub fn load_experts<'g>(g: &'g Gguf, cfg: &Config) -> Result<Vec<Vec<Expert<'g>>>, String> {
    if !cfg.is_moe() {
        return Ok(Vec::new());
    }
    (0..cfg.n_layers)
        .map(|i| {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let (gate, up, down) = (expert_matrices(g, &p("ffn_gate_exps"))?, expert_matrices(g, &p("ffn_up_exps"))?, expert_matrices(g, &p("ffn_down_exps"))?);
            if gate.len() != cfg.n_experts || up.len() != cfg.n_experts || down.len() != cfg.n_experts {
                return Err(format!("layer {i}: expert tensors do not hold {} experts", cfg.n_experts));
            }
            if gate[0].rows != cfg.expert_ffn_dim || down[0].cols != cfg.expert_ffn_dim || gate[0].cols != cfg.dim {
                return Err(format!("layer {i}: expert shapes do not match expert_feed_forward_length {}", cfg.expert_ffn_dim));
            }
            Ok(gate.into_iter().zip(up).zip(down).map(|((gate, up), down)| Expert { gate, up, down }).collect())
        })
        .collect()
}

/// Qwen3-MoE routing for one token: the `k` largest router logits, weighted by a softmax over
/// just those k (the same as a softmax over all experts renormalised to the chosen ones).
pub fn route(logits: &[f32], k: usize, out: &mut Vec<(usize, f32)>) {
    out.clear();
    out.extend(logits.iter().copied().enumerate());
    // descending logit; ties to the lower expert index
    out.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let max = out[0].1;
    let total: f32 = out.iter().map(|&(_, l)| (l - max).exp()).sum();
    for e in out.iter_mut() {
        e.1 = (e.1 - max).exp() / total;
    }
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
        let mut experts = load_experts(g, &cfg)?.into_iter();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let ffn = match experts.next() {
                Some(experts) => Ffn::Moe { router: vector(g, &p("ffn_gate_inp"))?, experts },
                None => Ffn::Dense { gate: matrix(g, &p("ffn_gate"))?, up: matrix(g, &p("ffn_up"))?, down: matrix(g, &p("ffn_down"))? },
            };
            layers.push(Layer {
                attn_norm: vector(g, &p("attn_norm"))?,
                wq: matrix(g, &p("attn_q"))?,
                wk: matrix(g, &p("attn_k"))?,
                wv: matrix(g, &p("attn_v"))?,
                wo: matrix(g, &p("attn_output"))?,
                q_norm: g.tensor(&p("attn_q_norm")).map(|_| vector(g, &p("attn_q_norm"))).transpose()?,
                k_norm: g.tensor(&p("attn_k_norm")).map(|_| vector(g, &p("attn_k_norm"))).transpose()?,
                ffn_norm: vector(g, &p("ffn_norm"))?,
                ffn,
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
    picks: Vec<(usize, f32)>, // mixture of experts: (expert, weight) per token and pick
    moe: MoeScratch,
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
            picks: Vec::new(),
            moe: MoeScratch::default(),
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

/// Scratch for the CPU expert path, reused between calls.
#[derive(Default)]
pub struct MoeScratch {
    logits: Vec<f32>,
    by_expert: Vec<Vec<(usize, f32)>>, // per expert: (token, weight)
    xin: Vec<f32>,
    xq: QuantizedBatch,
    h: Vec<f32>,
    out: Vec<f32>,
}

/// Routes each of the `xb.len() / dim` (normalised) rows: `picks` gets k (expert, weight) pairs per row.
pub fn moe_route(router: &[f32], k: usize, xb: &[f32], dim: usize, picks: &mut Vec<(usize, f32)>, sc: &mut MoeScratch) {
    let n_experts = router.len() / dim;
    picks.clear();
    let mut top = Vec::with_capacity(n_experts);
    for row in xb.chunks_exact(dim) {
        sc.logits.clear();
        sc.logits.extend(router.chunks_exact(dim).map(|w| dot_f32(w, row)));
        route(&sc.logits, k, &mut top);
        picks.extend_from_slice(&top);
    }
}

/// `acc[b] += Σ weight · expert(xb[b])` over the k experts `picks` gives row b (`expert(e)` hands out
/// expert e of `n_experts`). Rows sharing an expert run through it together, so each expert's weights
/// are read once per batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_experts<'e>(n_experts: usize, expert: impl Fn(usize) -> Expert<'e>, picks: &[(usize, f32)], k: usize, xb: &[f32], dim: usize, acc: &mut [f32], sc: &mut MoeScratch) {
    sc.by_expert.resize_with(n_experts, Vec::new);
    sc.by_expert.iter_mut().for_each(Vec::clear);
    for (i, &(e, w)) in picks.iter().enumerate() {
        sc.by_expert[e].push((i / k, w));
    }
    for (e, rows) in sc.by_expert.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
        let ex = &expert(e);
        let (m, hidden) = (rows.len(), ex.gate.rows);
        sc.xin.clear();
        for &(b, _) in rows {
            sc.xin.extend_from_slice(&xb[b * dim..(b + 1) * dim]);
        }
        sc.h.resize(m * hidden, 0.0);
        sc.out.resize(m * dim, 0.0);
        sc.xq.quantize(&sc.xin, dim);
        gate_up_silu(&ex.gate, &ex.up, &sc.xin, &sc.xq, &mut sc.h);
        sc.xq.quantize(&sc.h, hidden);
        ex.down.matmul(&sc.h, &sc.xq, &mut sc.out);
        for (&(b, w), o) in rows.iter().zip(sc.out.chunks_exact(dim)) {
            for (a, v) in acc[b * dim..(b + 1) * dim].iter_mut().zip(o) {
                *a += w * v;
            }
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
            profile::time(6, || match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    gate_up_silu(gate, up, &s.xb, &s.xq, &mut s.gate);
                    s.xq.quantize(&s.gate, c.ffn_dim);
                    down.matmul(&s.gate, &s.xq, &mut s.proj);
                    for (x, f) in s.x.iter_mut().zip(&s.proj) {
                        *x += f;
                    }
                }
                Ffn::Moe { router, experts } => {
                    moe_route(router, c.n_experts_used, &s.xb, dim, &mut s.picks, &mut s.moe);
                    moe_experts(experts.len(), |e| experts[e], &s.picks, c.n_experts_used, &s.xb, dim, &mut s.x, &mut s.moe);
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

#[cfg(test)]
mod tests {
    use super::route;

    #[test]
    fn route_keeps_top_k_with_softmax_over_them() {
        let logits = [0.5f32, 2.0, -1.0, 2.0, 1.0];
        let mut picks = Vec::new();
        route(&logits, 3, &mut picks);
        // ties go to the lower expert index
        assert_eq!(picks.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 3, 4]);
        let z = 2.0f32.exp() * 2.0 + 1.0f32.exp();
        for (p, want) in picks.iter().zip([2.0f32.exp() / z, 2.0f32.exp() / z, 1.0f32.exp() / z]) {
            assert!((p.1 - want).abs() < 1e-6, "{p:?} vs {want}");
        }
    }
}
