//! Thinking Machines' Inkling 2 (llama.cpp's `inkling2_loop` GGUF layout) on the CPU, one token at a time.
//!
//! This is the reference path: plain, exact to llama.cpp's graph (src/models/inkling2.cpp and
//! inkling2-loop.cpp), and checked against its eval-callback dumps with `ghost-infer inkling-check`.
//! What makes the model unusual, per layer:
//! - no RoPE: positions enter through a learned relative-distance bias (attn_rel, attn_rel_bias) and, on the
//!   full-attention layers, a log-of-position scale on the queries;
//! - a per-query sink logit (attn_sink_proj), clipped query/key norms, RMS-normed values, a sigmoid gate on the
//!   attention output;
//! - short causal convolutions (4 taps, with a residual) on K, V, the attention output and the FFN output,
//!   whose last inputs are kept per cache slot like a recurrent state;
//! - "attention residuals": every 6 layers the stream starts afresh and later inputs are softmax mixes over the
//!   earlier block outputs (attn_res_query, ffn_res_query, output_res_query);
//! - a sigmoid-routed MoE with the SITU activation and per-expert output scales (NVFP4's second scale);
//! - layers 6..18 run three times; passes after the first restart from layer 6's input and mix in a "message"
//!   from the previous pass through tanh-gated controls. Each pass has its own KV and conv state ("slots").

use crate::gguf::{Gguf, GgmlType, TensorInfo, Value};
use crate::quant::{matmul_many, Matrix, QuantizedBatch};

#[derive(Debug, Clone)]
pub struct Cfg {
    pub dim: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub dh: usize,
    pub n_rel: usize,
    pub n_swa: usize,
    pub rel_extent: usize,
    pub conv: usize,
    pub block: usize,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub n_exp: usize,
    pub k: usize,
    pub ff_exp: usize,
    pub exp_scale: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub softcap: f32,
    pub vocab: usize,
    pub swa: Vec<bool>,
    pub loop_start: usize,
    pub loop_end: usize,
    pub passes: usize,
}

impl Cfg {
    pub fn from_gguf(g: &Gguf) -> Result<Cfg, String> {
        let a = "inkling2_loop";
        if g.get("general.architecture").and_then(Value::as_str) != Some(a) {
            return Err("not an inkling2_loop file".into());
        }
        let get = |k: &str| g.get(&format!("{a}.{k}")).ok_or_else(|| format!("missing {a}.{k}"));
        let u = |k: &str| get(k).and_then(|v| v.as_u64().map(|v| v as usize).ok_or(format!("{a}.{k} is not a number")));
        let f = |k: &str| get(k).and_then(|v| v.as_f32().ok_or(format!("{a}.{k} is not a number")));
        // per-layer head counts are arrays; this engine needs them uniform
        let uniform = |k: &str| -> Result<usize, String> {
            let v = get(k)?;
            match v.as_array() {
                Some(arr) => {
                    let first = arr.first().and_then(Value::as_u64).ok_or(format!("{a}.{k} is empty"))?;
                    if arr.iter().any(|x| x.as_u64() != Some(first)) {
                        return Err(format!("{a}.{k} differs between layers, which is not supported"));
                    }
                    Ok(first as usize)
                }
                None => v.as_u64().map(|v| v as usize).ok_or(format!("{a}.{k} is not a number")),
            }
        };
        let swa: Vec<bool> = get("attention.sliding_window_pattern")?.as_array().ok_or("sliding_window_pattern is not an array")?.iter().map(|v| v.as_bool().unwrap_or(false)).collect();
        let c = Cfg {
            dim: u("embedding_length")?,
            n_layers: u("block_count")?,
            n_head: uniform("attention.head_count")?,
            n_kv: uniform("attention.head_count_kv")?,
            dh: u("attention.key_length")?,
            n_rel: u("attention.relative_length")?,
            n_swa: u("attention.sliding_window")?,
            rel_extent: u("attention.relative_extent")?,
            conv: u("shortconv.l_cache")?,
            block: u("attn_res.block_size")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: f("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?,
            n_exp: u("expert_count")?,
            k: u("expert_used_count")?,
            ff_exp: u("expert_feed_forward_length")?,
            exp_scale: f("expert_weights_scale").unwrap_or(1.0),
            situ_beta: f("activation.situ_beta")?,
            situ_lb: f("activation.situ_linear_beta")?,
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            vocab: u("vocab_size_unpadded")?,
            swa,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            passes: u("loop.count")?,
        };
        if c.swa.len() != c.n_layers || c.n_head % c.n_kv != 0 || c.loop_start == 0 || c.loop_start >= c.loop_end || c.loop_end > c.n_layers || c.loop_start % c.block != 0 || c.loop_end % c.block != 0 || c.passes < 2 {
            return Err("inkling2_loop: sizes do not fit together".into());
        }
        Ok(c)
    }

    /// KV / conv-state slots: every physical layer once, plus the core layers again for each extra pass.
    pub fn n_slots(&self) -> usize {
        self.n_layers + (self.passes - 1) * (self.loop_end - self.loop_start)
    }

    fn kv_dim(&self) -> usize {
        self.n_kv * self.dh
    }

    /// conv-state channels: K, V, the attention output, the FFN output
    fn conv_channels(&self) -> usize {
        2 * self.kv_dim() + 2 * self.dim
    }
}

struct Layer<'a> {
    swa: bool,
    /// SWA layers: one input norm; full layers: one each for Q, K and V
    norms: Vec<Vec<f32>>,
    wq: Matrix<'a>,
    wk: Matrix<'a>,
    wv: Matrix<'a>,
    wo: Matrix<'a>,
    wgate: Matrix<'a>,
    q_clip: Vec<f32>,
    sink: Matrix<'a>,
    rel: Matrix<'a>,
    rel_bias: Vec<f32>,
    rel_clip: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    post_conv: Vec<f32>,
    ffn_conv: Vec<f32>,
    /// full layers: input-dependent taps per kv head for the K and V convolutions
    k_conv_proj: Option<Matrix<'a>>,
    v_conv_proj: Option<Matrix<'a>>,
    /// layers after the first: queries mixing the earlier blocks into the attention input (1, or 3 for Q K V)
    attn_res_q: Option<Vec<f32>>,
    ffn_res_q: Vec<f32>,
    ffn_norm: Vec<f32>,
    router: Matrix<'a>,
    bias: Option<Vec<f32>>,
    /// per expert: [gate; up] rows, and down
    gate_up: Vec<Matrix<'a>>,
    down: Vec<Matrix<'a>>,
    gu_scale: Option<Vec<f32>>,
    down_scale: Option<Vec<f32>>,
}

/// A core layer's controls for passes after the first (Q, K, V, FFN).
struct Loop<'a> {
    msg_q: Vec<f32>,
    down: Vec<Matrix<'a>>,
    gain: Vec<Matrix<'a>>,
    forget: Vec<Matrix<'a>>,
}

pub struct Inkling<'a> {
    pub c: Cfg,
    tok_embd: Matrix<'a>,
    tok_norm: Vec<f32>,
    output: Matrix<'a>,
    output_norm: Vec<f32>,
    output_res_q: Vec<f32>,
    layers: Vec<Layer<'a>>,
    loops: Vec<Loop<'a>>,
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

fn experts<'g>(g: &'g Gguf, name: &str) -> Result<Vec<Matrix<'g>>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    let [cols, rows, n] = t.dims[..] else { return Err(format!("tensor {name} is not 3-D")) };
    let (cols, rows, n) = (cols as usize, rows as usize, n as usize);
    let bytes = t.n_bytes / n;
    Ok((0..n).map(|e| Matrix { data: &g.mmap[t.offset + e * bytes..t.offset + (e + 1) * bytes], ty: t.ty, rows, cols }).collect())
}

fn vector(g: &Gguf, name: &str) -> Result<Vec<f32>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    let n = t.n_elements() as usize;
    let mut out = vec![0f32; n];
    Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n }.row_f32(0, &mut out);
    Ok(out)
}

fn opt_vector(g: &Gguf, name: &str) -> Result<Option<Vec<f32>>, String> {
    if g.tensor(name).is_some() { vector(g, name).map(Some) } else { Ok(None) }
}

impl<'a> Inkling<'a> {
    pub fn load(g: &'a Gguf) -> Result<Inkling<'a>, String> {
        let c = Cfg::from_gguf(g)?;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let swa = c.swa[i];
            let norms = if swa { vec![vector(g, &p("attn_norm"))?] } else { vec![vector(g, &p("attn_q_inp_norm"))?, vector(g, &p("attn_k_inp_norm"))?, vector(g, &p("attn_v_inp_norm"))?] };
            let l = Layer {
                swa,
                norms,
                wq: matrix(g, &p("attn_q"))?,
                wk: matrix(g, &p("attn_k"))?,
                wv: matrix(g, &p("attn_v"))?,
                wo: matrix(g, &p("attn_output"))?,
                wgate: matrix(g, &p("attn_gate"))?,
                q_clip: vector(g, &p("attn_q_clip"))?,
                sink: matrix(g, &p("attn_sink_proj"))?,
                rel: matrix(g, &p("attn_rel"))?,
                rel_bias: vector(g, &p("attn_rel_bias"))?,
                rel_clip: vector(g, &p("attn_rel_clip"))?,
                k_conv: vector(g, &p("attn_k_conv"))?,
                v_conv: vector(g, &p("attn_v_conv"))?,
                post_conv: vector(g, &p("attn_post_conv"))?,
                ffn_conv: vector(g, &p("ffn_post_conv"))?,
                k_conv_proj: if swa { None } else { Some(matrix(g, &p("attn_k_conv_proj"))?) },
                v_conv_proj: if swa { None } else { Some(matrix(g, &p("attn_v_conv_proj"))?) },
                attn_res_q: if i > 0 { Some(vector(g, &p("attn_res_query"))?) } else { None },
                ffn_res_q: vector(g, &p("ffn_res_query"))?,
                ffn_norm: vector(g, &p("ffn_norm"))?,
                router: matrix(g, &p("ffn_gate_inp"))?,
                bias: opt_vector(g, &format!("blk.{i}.exp_probs_b.bias"))?,
                gate_up: experts(g, &p("ffn_gate_up_exps"))?,
                down: experts(g, &p("ffn_down_exps"))?,
                gu_scale: opt_vector(g, &format!("blk.{i}.ffn_gate_up_exps.scale"))?,
                down_scale: opt_vector(g, &format!("blk.{i}.ffn_down_exps.scale"))?,
            };
            let extent = if swa { c.n_swa } else { c.rel_extent };
            if l.wq.rows != c.n_head * c.dh || l.wk.rows != c.kv_dim() || l.rel.rows != c.n_rel * c.n_head || l.rel_bias.len() != c.n_rel * extent || l.k_conv.len() != c.conv * c.kv_dim() || l.gate_up.len() != c.n_exp || l.gate_up[0].rows != 2 * c.ff_exp || l.down[0].cols != c.ff_exp {
                return Err(format!("layer {i}: tensor shapes do not match the model's sizes"));
            }
            layers.push(l);
        }
        let mut loops = Vec::new();
        for i in c.loop_start..c.loop_end {
            let names = ["q", "k", "v", "ffn"];
            let m = |kind: &str| -> Result<Vec<Matrix<'a>>, String> { names.iter().map(|n| matrix(g, &format!("blk.{i}.loop_{n}_ctrl_{kind}.weight"))).collect() };
            loops.push(Loop { msg_q: vector(g, &format!("blk.{i}.loop_msg_query.weight"))?, down: m("down")?, gain: m("gain")?, forget: m("forget")? });
        }
        Ok(Inkling {
            tok_embd: matrix(g, "token_embd.weight")?,
            tok_norm: vector(g, "token_embd_norm.weight")?,
            output: matrix(g, "output.weight")?,
            output_norm: vector(g, "output_norm.weight")?,
            output_res_q: vector(g, "output_res_query.weight")?,
            layers,
            loops,
            c,
        })
    }

    pub fn new_state(&self) -> State {
        let c = &self.c;
        State { pos: 0, kv: vec![(Vec::new(), Vec::new()); c.n_slots()], conv: vec![vec![0f32; c.conv_channels() * (c.conv - 1)]; c.n_slots()], trace: None }
    }
}

/// One sequence's caches: per slot, every token's K and V, and the last `conv - 1` inputs of each conv channel.
pub struct State {
    pub pos: usize,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    conv: Vec<Vec<f32>>,
    /// when set, intermediate tensors by llama.cpp's names ("l_out-12", ...) for checking
    pub trace: Option<Vec<(String, Vec<f32>)>>,
}

impl State {
    fn note(&mut self, name: impl FnOnce() -> String, v: &[f32]) {
        if let Some(t) = &mut self.trace {
            t.push((name(), v.to_vec()));
        }
    }
}

/// `m · x` for one input vector.
/// Fills `buf` from `file` at byte `offset`, without moving a shared cursor: seek_read on Windows, read_at on
/// Unix (pread).
pub fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        #[cfg(windows)]
        let n = file.seek_read(&mut buf[done..], offset + done as u64);
        #[cfg(unix)]
        let n = file.read_at(&mut buf[done..], offset + done as u64);
        match n {
            Ok(0) => return Err("reading experts: unexpected end of file".into()),
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(format!("reading experts: {e}")),
        }
    }
    Ok(())
}

fn mv(m: &Matrix, x: &[f32]) -> Vec<f32> {
    let mut xq = QuantizedBatch::default();
    xq.quantize(x, m.cols);
    let mut out = vec![0f32; m.rows];
    matmul_many(&mut [(m, &mut out)], x, &xq);
    out
}

/// Several matrices over one input.
fn mv_many(ms: &[&Matrix], x: &[f32]) -> Vec<Vec<f32>> {
    let mut xq = QuantizedBatch::default();
    xq.quantize(x, ms[0].cols);
    let mut outs: Vec<Vec<f32>> = ms.iter().map(|m| vec![0f32; m.rows]).collect();
    let mut jobs: Vec<(&Matrix, &mut [f32])> = ms.iter().copied().zip(outs.iter_mut().map(|o| o.as_mut_slice())).collect();
    matmul_many(&mut jobs, x, &xq);
    outs
}

fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let s = (1.0 / (ms + eps as f64).sqrt()) as f32;
    match w {
        Some(w) => x.iter().zip(w).map(|(v, g)| v * s * g).collect(),
        None => x.iter().map(|v| v * s).collect(),
    }
}

/// Every `n`-wide row (scaled per channel by `scale`, laid out like the rows) down to an L2 norm of at most `thr`.
fn clip(x: &mut [f32], n: usize, scale: Option<&[f32]>, thr: f32) {
    if let Some(s) = scale {
        for (v, s) in x.iter_mut().zip(s) {
            *v *= s;
        }
    }
    for row in x.chunks_exact_mut(n) {
        let norm = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt() as f32 + 1e-6;
        let f = (thr / norm).clamp(0.0, 1.0);
        for v in row {
            *v *= f;
        }
    }
}

/// Attention over the depth: for each of the `nq` queries, a softmax over the RMS-normed states (the blocks and
/// `cur`) of query · state / sqrt(dim), and the mix of those normed states.
fn mix(blocks: &[Vec<f32>], cur: &[f32], query: &[f32], dim: usize) -> Vec<Vec<f32>> {
    let states: Vec<Vec<f32>> = blocks.iter().map(|b| b.as_slice()).chain(std::iter::once(cur)).map(|s| rms(s, None, 1e-6)).collect();
    let scale = 1.0 / (dim as f32).sqrt();
    query
        .chunks_exact(dim)
        .map(|q| {
            let scores: Vec<f32> = states.iter().map(|s| crate::quant::dot_f32(s, q) * scale).collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = e.iter().sum();
            let mut out = vec![0f32; dim];
            for (s, w) in states.iter().zip(&e) {
                for (o, v) in out.iter_mut().zip(s) {
                    *o += v * w / total;
                }
            }
            out
        })
        .collect()
}

/// Causal short convolution with a residual over one token, its `cur.len()` channels at `offset` of the slot's
/// state (oldest input first, `conv - 1` per channel). `dynamic`: per head, input-dependent taps added to the
/// fixed ones (index lag + conv * head, lag 0 the current input).
fn convolve(cur: &[f32], w: &[f32], dynamic: Option<&[f32]>, state: &mut [f32], offset: usize, conv: usize) -> Vec<f32> {
    let n = cur.len();
    let past = conv - 1;
    let heads = dynamic.map_or(1, |d| d.len() / conv);
    let per_head = n / heads;
    let mut out = vec![0f32; n];
    for ch in 0..n {
        let hist = &mut state[(offset + ch) * past..(offset + ch + 1) * past];
        // the window, oldest first: the kept inputs then the current one
        let x = |k: usize| if k < past { hist[k] } else { cur[ch] };
        let mut y = 0f32;
        for k in 0..conv {
            y += w[ch * conv + k] * x(k);
        }
        if let Some(d) = dynamic {
            let h = ch / per_head;
            for lag in 0..conv {
                y += d[h * conv + lag] * x(conv - 1 - lag);
            }
        }
        out[ch] = (cur[ch] + y) / (((if dynamic.is_some() { 2 } else { 1 }) * conv + 1) as f32).sqrt();
        hist.copy_within(1.., 0);
        hist[past - 1] = cur[ch];
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl<'a> Inkling<'a> {
    /// A looped layer input with a message from the previous pass: (1 - forget) * cur + gain * message, both
    /// tanh of low-rank projections of [rms(cur), rms(message)].
    fn control(&self, cur: &[f32], msg: &[f32], lp: &Loop, ctrl: usize) -> Vec<f32> {
        let mut gate = rms(cur, None, 1e-6);
        gate.extend(rms(msg, None, 1e-6));
        let g = mv(&lp.down[ctrl], &gate);
        let gf = mv_many(&[&lp.gain[ctrl], &lp.forget[ctrl]], &g);
        cur.iter().zip(msg).zip(gf[0].iter().zip(&gf[1])).map(|((&c, &m), (&ga, &fo))| c - fo.tanh() * c + ga.tanh() * m).collect()
    }

    /// The embedding row of a token.
    /// Copies layer `il`'s experts out of the mapped file into memory of the process's own (kept for the rest of
    /// the run): the CPU expert path then reads RAM that stays put, not file pages Windows may drop and read
    /// again (slow from a USB disk). Returns the bytes copied.
    /// The bytes are read from `file` (whose mapping starts at `base`), not through the mapping: mapped pages
    /// touched here would stay in the working set beside the copy, twice the RAM.
    pub fn own_experts(&mut self, il: usize, file: &std::fs::File, base: usize) -> Result<usize, String> {
        let l = &mut self.layers[il];
        let mut total = 0;
        for ms in [&mut l.gate_up, &mut l.down] {
            let (start, end) = (ms[0].data.as_ptr() as usize, ms[ms.len() - 1].data.as_ptr() as usize + ms[ms.len() - 1].data.len());
            let mut buf = vec![0u8; end - start];
            read_exact_at(file, &mut buf, (start - base) as u64)?;
            // IQ3_S re-laid as 4-bit codes (the same values, a third of the CPU's decoding work);
            // GHOST_INFER_INK_IQ3S=1 keeps it as it is
            if ms[0].ty == GgmlType::Iq3S && std::env::var_os("GHOST_INFER_INK_IQ3S").is_none() {
                let mut nib = Vec::with_capacity(buf.len() / crate::iquant::IQ3_S_BYTES * crate::iquant::IQ3S_NIB_BYTES);
                let mut spans = Vec::with_capacity(ms.len());
                for m in ms.iter() {
                    let off = m.data.as_ptr() as usize - start;
                    let at = nib.len();
                    match crate::iquant::iq3s_to_nib(&buf[off..off + m.data.len()]) {
                        Some(v) => nib.extend_from_slice(&v),
                        None => return Err("an IQ3_S grid level is not odd".into()),
                    }
                    spans.push((at, nib.len() - at));
                }
                drop(buf);
                let owned: &'static [u8] = Box::leak(nib.into_boxed_slice());
                for (m, (at, n)) in ms.iter_mut().zip(spans) {
                    m.data = &owned[at..at + n];
                    m.ty = GgmlType::Iq3SNib;
                }
                total += owned.len();
                continue;
            }
            let owned: &'static [u8] = Box::leak(buf.into_boxed_slice());
            for m in ms.iter_mut() {
                let off = m.data.as_ptr() as usize - start;
                m.data = &owned[off..off + m.data.len()];
            }
            total += end - start;
        }
        Ok(total)
    }

    /// Expert `e` of layer `il`: its [gate; up] and down matrices (their bytes as the CPU holds them).
    pub fn expert_mats(&self, il: usize, e: usize) -> (crate::quant::Matrix<'a>, crate::quant::Matrix<'a>) {
        (self.layers[il].gate_up[e], self.layers[il].down[e])
    }

    pub fn embed(&self, token: u32) -> Vec<f32> {
        let mut e = vec![0f32; self.c.dim];
        self.tok_embd.row_f32(token as usize, &mut e);
        e
    }

    /// Layer `il`'s experts for a normed FFN input (the GPU's CPU-expert layers), with reused scratch.
    pub fn moe_by_layer(&self, il: usize, x: &[f32], sc: &mut crate::quant::ExpertScratch) -> Vec<f32> {
        self.moe_with(&self.layers[il], x, sc)
    }

    fn moe(&self, l: &Layer, x: &[f32]) -> Vec<f32> {
        self.moe_with(l, x, &mut crate::quant::ExpertScratch::default())
    }

    fn moe_with(&self, l: &Layer, x: &[f32], sc: &mut crate::quant::ExpertScratch) -> Vec<f32> {
        let c = &self.c;
        let logits = mv(&l.router, x);
        let probs: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
        let mut pick: Vec<(usize, f32)> = probs.iter().enumerate().map(|(e, &p)| (e, p + l.bias.as_ref().map_or(0.0, |b| b[e]))).collect();
        pick.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        pick.truncate(c.k);
        let total: f32 = pick.iter().map(|&(e, _)| probs[e]).sum::<f32>() + 1e-20;
        let experts: Vec<crate::quant::SituExpert> = pick
            .iter()
            .map(|&(e, _)| crate::quant::SituExpert {
                gate_up: l.gate_up[e],
                down: l.down[e],
                w: probs[e] / total * c.exp_scale,
                gu_scale: l.gu_scale.as_ref().map_or(1.0, |s| s[e]),
                down_scale: l.down_scale.as_ref().map_or(1.0, |s| s[e]),
            })
            .collect();
        let mut acc = vec![0f32; c.dim];
        crate::quant::experts_situ(&experts, x, &mut acc, sc, c.situ_beta, c.situ_lb);
        acc
    }

    /// Layer `il`'s experts `picks` (expert, routing weight) summed for one normed FFN input: the CPU's share of a
    /// layer whose other experts are on the GPU.
    pub fn moe_picks(&self, il: usize, x: &[f32], picks: &[(usize, f32)], sc: &mut crate::quant::ExpertScratch) -> Vec<f32> {
        let l = &self.layers[il];
        let experts: Vec<crate::quant::SituExpert> = picks
            .iter()
            .map(|&(e, w)| crate::quant::SituExpert {
                gate_up: l.gate_up[e],
                down: l.down[e],
                w,
                gu_scale: l.gu_scale.as_ref().map_or(1.0, |s| s[e]),
                down_scale: l.down_scale.as_ref().map_or(1.0, |s| s[e]),
            })
            .collect();
        let mut acc = vec![0f32; self.c.dim];
        crate::quant::experts_situ(&experts, x, &mut acc, sc, self.c.situ_beta, self.c.situ_lb);
        acc
    }

    /// Layer `il`'s experts for several tokens' normed FFN inputs (rows of `xs`): each token's own picks
    /// (`picks[t]`, expert and weight), or with None routed here as `moe_by_layer` routes; rows of sums.
    pub fn moe_batch(&self, il: usize, xs: &[f32], picks: Option<&[Vec<(usize, f32)>]>, sc: &mut crate::quant::ExpertScratch) -> Vec<f32> {
        let (c, l) = (&self.c, &self.layers[il]);
        let nt = xs.len() / c.dim;
        let routed: Vec<Vec<(usize, f32)>> = match picks {
            Some(p) => p.to_vec(),
            None => (0..nt).map(|t| self.route(il, &xs[t * c.dim..(t + 1) * c.dim])).collect(),
        };
        let experts: Vec<Vec<crate::quant::SituExpert>> = routed
            .iter()
            .map(|list| {
                list.iter()
                    .map(|&(e, w)| crate::quant::SituExpert {
                        gate_up: l.gate_up[e],
                        down: l.down[e],
                        w,
                        gu_scale: l.gu_scale.as_ref().map_or(1.0, |s| s[e]),
                        down_scale: l.down_scale.as_ref().map_or(1.0, |s| s[e]),
                    })
                    .collect()
            })
            .collect();
        let mut acc = vec![0f32; xs.len()];
        crate::quant::experts_situ_batch(&experts, xs, &mut acc, sc, c.situ_beta, c.situ_lb);
        acc
    }

    /// The experts' picks and weights for one normed FFN input (for the GPU's CPU-expert layers).
    pub fn route(&self, il: usize, x: &[f32]) -> Vec<(usize, f32)> {
        let (c, l) = (&self.c, &self.layers[il]);
        let probs: Vec<f32> = mv(&l.router, x).iter().map(|&v| sigmoid(v)).collect();
        let mut pick: Vec<(usize, f32)> = probs.iter().enumerate().map(|(e, &p)| (e, p + l.bias.as_ref().map_or(0.0, |b| b[e]))).collect();
        pick.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        pick.truncate(c.k);
        let total: f32 = pick.iter().map(|&(e, _)| probs[e]).sum::<f32>() + 1e-20;
        pick.iter().map(|&(e, _)| (e, probs[e] / total * c.exp_scale)).collect()
    }

    /// Layer `il` (physical) at depth `ip`, keeping its caches in `slot`.
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, il: usize, slot: usize, ip: usize, cur: Vec<f32>, blocks: &mut Vec<Vec<f32>>, msgs: Option<(&[Vec<f32>], &Loop)>, st: &mut State) -> Vec<f32> {
        let c = &self.c;
        let l = &self.layers[il];
        let (h, hkv, dh, nr) = (c.n_head, c.n_kv, c.dh, c.n_rel);
        let pos = st.pos;

        let mut inputs: [Vec<f32>; 3] = [cur.clone(), cur.clone(), cur.clone()];
        if let Some(q) = &l.attn_res_q {
            let mixed = mix(blocks, &cur, q, c.dim);
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = mixed[if l.swa { 0 } else { j }].clone();
            }
        }
        if let Some((m, lp)) = msgs {
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = self.control(inp, &m[j], lp, j);
            }
        }
        let norm = |x: &[f32], w: &[f32]| rms(x, Some(w), c.eps);
        if l.swa {
            if msgs.is_some() {
                for inp in inputs.iter_mut() {
                    *inp = norm(inp, &l.norms[0]);
                }
            } else {
                let n0 = norm(&inputs[0], &l.norms[0]);
                inputs = [n0.clone(), n0.clone(), n0];
            }
        } else {
            for (inp, w) in inputs.iter_mut().zip(&l.norms) {
                *inp = norm(inp, w);
            }
        }

        // projections
        let q_side = mv_many(&[&l.wq, &l.rel, &l.sink, &l.wgate], &inputs[0]);
        let (mut q, mut r, sinks, gate) = (q_side[0].clone(), q_side[1].clone(), q_side[2].clone(), q_side[3].clone());
        let k_side = match &l.k_conv_proj { Some(p) => mv_many(&[&l.wk, p], &inputs[1]), None => vec![mv(&l.wk, &inputs[1])] };
        let v_side = match &l.v_conv_proj { Some(p) => mv_many(&[&l.wv, p], &inputs[2]), None => vec![mv(&l.wv, &inputs[2])] };
        let kv = c.kv_dim();
        let state = &mut st.conv[slot];
        let mut k = convolve(&k_side[0], &l.k_conv, k_side.get(1).map(|v| v.as_slice()), state, 0, c.conv);
        let mut v = convolve(&v_side[0], &l.v_conv, v_side.get(1).map(|v| v.as_slice()), state, kv, c.conv);

        clip(&mut q, dh, Some(&l.q_clip), c.qk_clip);
        clip(&mut k, dh, None, c.qk_clip);
        for head in v.chunks_exact_mut(dh) {
            let n = rms(head, None, c.eps);
            head.copy_from_slice(&n);
        }
        clip(&mut r, nr, Some(&l.rel_clip), c.rel_clip);
        if !l.swa {
            let tau = 1.0 + c.log_alpha * (((pos + 1) as f32 / c.log_floor).max(1.0)).ln();
            q.iter_mut().for_each(|x| *x *= tau);
            r.iter_mut().for_each(|x| *x *= tau);
        }
        st.note(|| format!("Qcur-{slot}"), &q);
        st.note(|| format!("Kcur-{slot}"), &k);
        st.note(|| format!("Vcur-{slot}"), &v);

        // the cache, and attention with the relative bias, the sink and the window
        let (kc, vc) = &mut st.kv[slot];
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let n_kv = kc.len() / kv;
        let extent = if l.swa { c.n_swa } else { c.rel_extent };
        let scale = 1.0 / (dh as f32).sqrt();
        let mut att = vec![0f32; h * dh];
        for hd in 0..h {
            let g = hd / (h / hkv);
            let qh = &q[hd * dh..(hd + 1) * dh];
            let rh = &r[hd * nr..(hd + 1) * nr];
            let bias = |dist: usize| -> f32 {
                let b = if dist < extent { crate::quant::dot_f32(&l.rel_bias[dist * nr..(dist + 1) * nr], rh) } else { 0.0 };
                b - sinks[hd]
            };
            let mut s: Vec<(usize, f32)> = Vec::with_capacity(n_kv);
            for j in 0..n_kv {
                let dist = n_kv - 1 - j; // positions are 0..n_kv in order
                if l.swa && dist >= c.n_swa {
                    continue;
                }
                let kj = &kc[j * kv + g * dh..j * kv + (g + 1) * dh];
                s.push((j, crate::quant::dot_f32(qh, kj) * scale + bias(dist)));
            }
            let max = s.iter().map(|x| x.1).fold(0f32, f32::max); // the sink logit is 0
            let total: f32 = s.iter().map(|x| (x.1 - max).exp()).sum::<f32>() + (-max).exp();
            let out = &mut att[hd * dh..(hd + 1) * dh];
            for &(j, sc) in &s {
                let w = (sc - max).exp() / total;
                let vj = &vc[j * kv + g * dh..j * kv + (g + 1) * dh];
                for (o, x) in out.iter_mut().zip(vj) {
                    *o += w * x;
                }
            }
        }
        for (a, gt) in att.iter_mut().zip(&gate) {
            *a *= sigmoid(*gt);
        }
        let a = mv(&l.wo, &att);
        let a = convolve(&a, &l.post_conv, None, &mut st.conv[slot], 2 * kv, c.conv);
        st.note(|| format!("attn_out-{slot}"), &a);

        let ffn_inp = if ip % c.block == 0 {
            blocks.push(cur);
            a
        } else {
            cur.iter().zip(&a).map(|(x, y)| x + y).collect()
        };
        st.note(|| format!("ffn_inp-{slot}"), &ffn_inp);

        let mut f = mix(blocks, &ffn_inp, &l.ffn_res_q, c.dim).pop().unwrap();
        if let Some((m, lp)) = msgs {
            f = self.control(&f, &m[3], lp, 3);
        }
        let f = norm(&f, &l.ffn_norm);
        let f = self.moe(l, &f);
        let f = convolve(&f, &l.ffn_conv, None, &mut st.conv[slot], 2 * kv + c.dim, c.conv);
        st.note(|| format!("ffn_out-{slot}"), &f);
        let out: Vec<f32> = ffn_inp.iter().zip(&f).map(|(x, y)| x + y).collect();
        st.note(|| format!("l_out-{slot}"), &out);
        out
    }

    /// One token through the model; returns its logits (the unpadded vocabulary).
    pub fn step(&self, token: u32, st: &mut State) -> Vec<f32> {
        let c = &self.c;
        let mut e = vec![0f32; c.dim];
        self.tok_embd.row_f32(token as usize, &mut e);
        let mut x = rms(&e, Some(&self.tok_norm), c.eps);
        st.note(|| "embd_norm".into(), &x);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        for il in 0..c.loop_start {
            x = self.layer(il, il, il, x, &mut blocks, None, st);
        }
        let prelude = x.clone();
        let prelude_blocks = blocks.clone();
        let core = c.loop_end - c.loop_start;
        for pass in 0..c.passes {
            let mut bank: Vec<Vec<f32>> = Vec::new();
            let mut bank_out: Vec<f32> = Vec::new();
            if pass > 0 {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank_out = std::mem::replace(&mut x, prelude.clone());
                blocks = prelude_blocks.clone();
            }
            for ip in c.loop_start..c.loop_end {
                if pass == 0 {
                    x = self.layer(ip, ip, ip, x, &mut blocks, None, st);
                    continue;
                }
                let slot = c.n_layers + (pass - 1) * core + (ip - c.loop_start);
                let lp = &self.loops[ip - c.loop_start];
                let msgs = mix(&bank, &bank_out, &lp.msg_q, c.dim);
                x = self.layer(ip, slot, ip, x, &mut blocks, Some((&msgs, lp)), st);
            }
        }
        for il in c.loop_end..c.n_layers {
            x = self.layer(il, il, il, x, &mut blocks, None, st);
        }
        let o = mix(&blocks, &x, &self.output_res_q, c.dim).pop().unwrap();
        let o = rms(&o, Some(&self.output_norm), c.eps);
        st.note(|| "result_norm".into(), &o);
        let mut logits = mv(&self.output, &o);
        logits.truncate(c.vocab);
        if c.softcap != 0.0 {
            for v in logits.iter_mut() {
                *v = c.softcap * (*v / c.softcap).tanh();
            }
        }
        st.pos += 1;
        logits
    }
}

/// `inkling-check <model.gguf> <token ids, comma-separated>`: runs the tokens one by one and prints, per traced
/// tensor, the first and last three values of every token and the sum over all of them, as llama.cpp's
/// eval-callback does, then the top five next tokens.
pub fn check(path: &std::path::Path, ids: &str) -> Result<(), String> {
    set_threads();
    let g = Gguf::open(path)?;
    let m = Inkling::load(&g)?;
    let ids: Vec<u32> = ids.split(',').map(|s| s.trim().parse().map_err(|e| format!("bad token id {s}: {e}"))).collect::<Result<_, _>>()?;
    let mut st = m.new_state();
    st.trace = Some(Vec::new());
    let mut per_name: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    let mut logits = Vec::new();
    let t0 = std::time::Instant::now();
    for &id in &ids {
        logits = m.step(id, &mut st);
        for (name, v) in st.trace.as_mut().unwrap().drain(..) {
            match per_name.iter_mut().find(|(n, _)| *n == name) {
                Some((_, rows)) => rows.push(v),
                None => per_name.push((name, vec![v])),
            }
        }
    }
    eprintln!("[{} tokens in {:.1} s]", ids.len(), t0.elapsed().as_secs_f64());
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
    let _ = GgmlType::F32;
    Ok(())
}

/// `inkling-gen <model.gguf> <token ids> <n>`: greedy continuation, printing the token ids and the speed.
/// `inkling-moe-bench <model.gguf> <layer> [passes]`: one CPU expert pass (8 experts of the layer, picked at
/// random each time, the weights in RAM of our own) timed over many passes, with the weight bytes it reads per
/// second. GHOST_INFER_THREADS sets the pool's threads.
pub fn moe_bench(path: &std::path::Path, layer: usize, passes: usize) -> Result<(), String> {
    set_threads();
    let g = Gguf::open(path)?;
    let mut m = Inkling::load(&g)?;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    m.own_experts(layer, &file, g.mmap.as_ptr() as usize)?;
    let c = m.c.clone();
    let l = &m.layers[layer];
    let mut s = 12345u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as usize
    };
    let x: Vec<f32> = (0..c.dim).map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0).collect();
    let mut sc = crate::quant::ExpertScratch::default();
    let mut acc = vec![0f32; c.dim];
    let per_expert = l.gate_up[0].data.len() + l.down[0].data.len();
    let mut run = |n: usize| -> f64 {
        let t = std::time::Instant::now();
        for _ in 0..n {
            let mut picks: Vec<usize> = Vec::new();
            while picks.len() < c.k {
                let e = rnd() % c.n_exp;
                if !picks.contains(&e) {
                    picks.push(e);
                }
            }
            let ex: Vec<crate::quant::SituExpert> = picks.iter().map(|&e| crate::quant::SituExpert { gate_up: l.gate_up[e], down: l.down[e], w: 0.1, gu_scale: 1.0, down_scale: 1.0 }).collect();
            crate::quant::experts_situ(&ex, &x, &mut acc, &mut sc, c.situ_beta, c.situ_lb);
        }
        t.elapsed().as_secs_f64()
    };
    run(20);
    let secs = run(passes);
    let ms = secs * 1000.0 / passes as f64;
    println!(
        "layer {layer} ({} / {}): {:.2} ms per pass, {:.1} GB/s of weights, {} threads",
        l.gate_up[0].ty.name(),
        l.down[0].ty.name(),
        ms,
        (c.k * per_expert) as f64 / (ms / 1000.0) / 1e9,
        rayon::current_num_threads()
    );
    Ok(())
}

pub fn generate(path: &std::path::Path, ids: &str, n: usize) -> Result<(), String> {
    set_threads();
    let g = Gguf::open(path)?;
    let m = Inkling::load(&g)?;
    let ids: Vec<u32> = ids.split(',').map(|s| s.trim().parse().map_err(|e| format!("bad token id {s}: {e}"))).collect::<Result<_, _>>()?;
    let mut st = m.new_state();
    let mut logits = Vec::new();
    for &id in &ids {
        logits = m.step(id, &mut st);
    }
    let t0 = std::time::Instant::now();
    let mut out = Vec::new();
    for _ in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        out.push(next);
        logits = m.step(next, &mut st);
    }
    println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    eprintln!("[{n} tokens in {:.1} s, {:.2} tok/s]", t0.elapsed().as_secs_f64(), n as f64 / t0.elapsed().as_secs_f64());
    Ok(())
}

/// The CPU threads for the Inkling commands: GHOST_INFER_THREADS, else one per physical core (the spinning
/// expert pool loses most of its speed on hyper-threads: 5 ms a pass on 12 threads of a Ryzen 5 3600, 1.7 on 6).
pub fn set_threads() {
    let n = std::env::var("GHOST_INFER_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) / 2).max(1));
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
}
