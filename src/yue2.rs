//! YuE2's song writer: the AR half of YuE2-3B (a mixture of transformers whose AR path is a plain Qwen3
//! stack), which writes a song as codec tokens after a prompt of style, lyrics and an ABC score. The other
//! half (flow matching to audio latents) and the audio decoder stay in YuE2's own PyTorch code; this only
//! writes the tokens, which is most of the time a song takes.
//!
//! The GGUF comes from scripts/yue2-convert.py: a qwen3 file whose output layer has only the rows a song
//! token can be (MUSIC_END, then the 32768 codec tokens), so logit i is token `yue2.logit_from` + i, and whose
//! embedding stays in RAM (only the prompt's rows are read).
//!
//!   ghost-infer yue2 MODEL.gguf JOB.json OUT.json     (JOB: {"prefix": [ids], "seed": n, ...sampling})
//!   ghost-infer yue2-check MODEL.gguf REF.json        (logits against YuE2's own, from the converter's --ref)

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use crate::cuda::{GpuModel, GpuSession};
use crate::gguf::{Gguf, Value};
use crate::json::{self, Json};
use crate::model::Config;

/// The longest prompt + song YuE2 was trained on.
const CONTEXT: usize = 24576;

/// How YuE2 samples song tokens (its GenerationConfig.semantic), applied in its order: the codec range,
/// no end before `min_tokens`, the windowed repetition penalty, temperature, top-k, top-p.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub penalty_window: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 1.0, top_p: 0.95, top_k: 100, repetition_penalty: 1.2, penalty_window: 50, min_tokens: 200, max_tokens: 9000 }
    }
}

pub struct Yue2 {
    _g: Gguf,
    model: GpuModel,
    /// the token logit 0 is (MUSIC_END); the codec tokens follow it
    pub logit_from: u32,
}

impl Yue2 {
    pub fn load(path: &Path, ctx: usize) -> Result<Yue2, String> {
        let g = Gguf::open(path)?;
        let logit_from = match g.get("yue2.logit_from") {
            Some(v) => v.as_u64().ok_or("yue2.logit_from is not a number")? as u32,
            None => return Err(format!("{}: not a YuE2 song-writer GGUF (convert it with scripts/yue2-convert.py)", path.display())),
        };
        if !matches!(g.get("ghost.embd_on_host"), Some(Value::Bool(true))) {
            return Err("this YuE2 GGUF does not keep its embedding in RAM; convert it again".into());
        }
        let mut cfg = Config::from_gguf(&g)?;
        cfg.max_ctx = ctx.min(CONTEXT).min(cfg.max_ctx);
        let model = GpuModel::load_with(&g, cfg, false)?;
        Ok(Yue2 { model, logit_from, _g: g })
    }

    pub fn session(&self) -> Result<GpuSession, String> {
        GpuSession::new(&self.model.cfg, self.model.cfg.max_ctx)
    }

    /// Runs `tokens` in GPU-batch-sized pieces (a long song's ABC score makes a prompt of 1500+ tokens),
    /// keeping the logits after the last one.
    fn prefill(&self, tokens: &[u32], s: &mut GpuSession) -> Result<(), String> {
        let step = crate::cuda::max_batch(&self.model.cfg);
        let pieces: Vec<&[u32]> = tokens.chunks(step).collect();
        for (i, piece) in pieces.iter().enumerate() {
            self.model.forward_batch(piece, s, i + 1 == pieces.len())?;
        }
        Ok(())
    }

    /// Writes a song after `prefix` (the prompt's token IDs, ending in MUSIC_START): the codec values
    /// (tokens minus the codec offset), and whether it ended by itself (false: it hit max_tokens).
    /// `on_progress(tokens, seconds)` is called every step.
    pub fn write(&self, s: &mut GpuSession, prefix: &[u32], sampling: Sampling, seed: u64, mut on_progress: impl FnMut(usize, f64)) -> Result<(Vec<u32>, bool, f64), String> {
        if prefix.len() + sampling.max_tokens > s.ctx {
            return Err(format!("the prompt ({} tokens) and a {}-token song do not fit in {} positions", prefix.len(), sampling.max_tokens, s.ctx));
        }
        s.reset()?;
        let t = Instant::now();
        self.prefill(prefix, s)?;
        let prefill = t.elapsed().as_secs_f64();
        let mut picker = Picker::new(sampling, seed);
        let mut song = Vec::new();
        let mut ended = false;
        for step in 0..sampling.max_tokens {
            let i = picker.pick(&s.logits, step);
            if i == 0 {
                ended = true;
                break;
            }
            song.push(i - 1); // logit i is codec value i - 1
            on_progress(song.len(), t.elapsed().as_secs_f64());
            if step + 1 < sampling.max_tokens {
                self.model.forward_batch(&[self.logit_from + i], s, true)?;
            }
        }
        Ok((song, ended, prefill))
    }
}

/// YuE2's song-token sampling over the output layer's rows (row 0 is MUSIC_END).
struct Picker {
    sampling: Sampling,
    rng: u64,
    /// the rows picked so far (the repetition window reads its tail)
    history: Vec<u32>,
    counts: Vec<u16>,
    scores: Vec<f32>,
    order: Vec<u32>,
}

impl Picker {
    fn new(sampling: Sampling, seed: u64) -> Picker {
        // splitmix64 seeding, so nearby seeds start far apart
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Picker { sampling, rng: (z ^ (z >> 31)).max(1), history: Vec::new(), counts: Vec::new(), scores: Vec::new(), order: Vec::new() }
    }

    fn uniform(&mut self) -> f64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn pick(&mut self, logits: &[f32], step: usize) -> u32 {
        let sp = self.sampling;
        let n = logits.len();
        self.scores.clear();
        self.scores.extend_from_slice(logits);
        let s = &mut self.scores;
        if step < sp.min_tokens {
            s[0] = f32::NEG_INFINITY;
        }
        // repetition: each row in the last `penalty_window` picks, by how often (penalty^count), divides a
        // positive score and multiplies a negative one
        if sp.repetition_penalty != 1.0 && !self.history.is_empty() {
            self.counts.clear();
            self.counts.resize(n, 0);
            let from = self.history.len().saturating_sub(sp.penalty_window);
            for &r in &self.history[from..] {
                self.counts[r as usize] += 1;
            }
            for &r in &self.history[from..] {
                let c = self.counts[r as usize];
                if c > 0 {
                    let alpha = sp.repetition_penalty.powi(c as i32);
                    let v = &mut s[r as usize];
                    *v = if *v < 0.0 { *v * alpha } else { *v / alpha };
                    self.counts[r as usize] = 0; // once per row
                }
            }
        }
        let pick = if sp.temperature == 0.0 {
            (0..n).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap() as u32
        } else {
            if sp.temperature != 1.0 {
                s.iter_mut().for_each(|v| *v /= sp.temperature);
            }
            // top-k: the k best (ties at the k-th value stay in, as torch's masked_fill(scores < threshold) keeps them)
            self.order.clear();
            self.order.extend(0..n as u32);
            let k = sp.top_k.min(n);
            self.order.select_nth_unstable_by(k - 1, |&a, &b| s[b as usize].total_cmp(&s[a as usize]));
            let threshold = s[self.order[k - 1] as usize];
            self.order.retain(|&i| s[i as usize] >= threshold && s[i as usize].is_finite());
            self.order.sort_unstable_by(|&a, &b| s[b as usize].total_cmp(&s[a as usize]));
            // top-p over what is left: keep a token while the probability before it is within top_p (the best always)
            let top = s[self.order[0] as usize];
            let weights: Vec<f64> = self.order.iter().map(|&i| ((s[i as usize] - top) as f64).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut keep = weights.len();
            let mut before = 0.0;
            for (j, w) in weights.iter().enumerate() {
                if j > 0 && before > sp.top_p as f64 {
                    keep = j;
                    break;
                }
                before += w / total;
            }
            let kept: f64 = weights[..keep].iter().sum();
            let mut u = self.uniform() * kept;
            let mut chosen = self.order[keep - 1];
            for (j, w) in weights[..keep].iter().enumerate() {
                if u < *w {
                    chosen = self.order[j];
                    break;
                }
                u -= w;
            }
            chosen
        };
        if pick != 0 {
            self.history.push(pick);
        }
        pick
    }
}

fn ids(j: &Json, key: &str) -> Result<Vec<u32>, String> {
    j.get(key).and_then(Json::as_array).ok_or(format!("missing \"{key}\""))?.iter().map(|v| v.as_f64().map(|x| x as u32).ok_or(format!("\"{key}\" must hold token IDs"))).collect()
}

/// `yue2 MODEL.gguf JOB.json OUT.json`: writes one song's tokens. Progress goes to stderr in YuE2's own form
/// ("[YuE2] Running Generating song: N tokens | X tokens/s | elapsed Ys"), so tools reading YuE2 read this too.
pub fn run(model: &Path, job: &Path, out: &Path) -> Result<(), String> {
    let j = Json::parse(&std::fs::read_to_string(job).map_err(|e| format!("{}: {e}", job.display()))?)?;
    let prefix = ids(&j, "prefix")?;
    let seed = j.get("seed").and_then(Json::as_f64).ok_or("missing \"seed\"")? as u64;
    let d = Sampling::default();
    let f = |k: &str, v: f32| j.get(k).and_then(Json::as_f64).map_or(v, |x| x as f32);
    let u = |k: &str, v: usize| j.get(k).and_then(Json::as_f64).map_or(v, |x| x as usize);
    let sampling = Sampling {
        temperature: f("temperature", d.temperature),
        top_p: f("top_p", d.top_p),
        top_k: u("top_k", d.top_k).max(1),
        repetition_penalty: f("repetition_penalty", d.repetition_penalty),
        penalty_window: u("penalty_window", d.penalty_window).max(1),
        min_tokens: u("min_tokens", d.min_tokens),
        max_tokens: u("max_tokens", d.max_tokens).max(1),
    };
    let t = Instant::now();
    let y = Yue2::load(model, prefix.len() + sampling.max_tokens + 1)?;
    let mut s = y.session()?;
    eprintln!("[YuE2] Completed Loading model: elapsed {:.1}s", t.elapsed().as_secs_f64());
    let mut last = 0.0;
    let report = |done: &str, n: usize, secs: f64| {
        eprintln!("[YuE2] {done} Generating song: {n} tokens | {:.1} tokens/s | elapsed {secs:.1}s", n as f64 / secs.max(1e-9));
        let _ = std::io::stderr().flush();
    };
    let (song, ended, prefill) = y.write(&mut s, &prefix, sampling, seed, |n, secs| {
        if secs - last >= 5.0 {
            last = secs;
            report("Running", n, secs);
        }
    })?;
    let secs = t.elapsed().as_secs_f64();
    report("Completed", song.len(), secs);
    let doc = json::obj(vec![
        ("tokens", Json::Arr(song.iter().map(|&v| json::num(v)).collect())),
        ("ended", Json::Bool(ended)),
        ("prefill_seconds", json::num(prefill)),
        ("seconds", json::num(secs)),
        ("prefix_tokens", json::num(prefix.len() as f64)),
    ]);
    std::fs::write(out, doc.to_string()).map_err(|e| format!("{}: {e}", out.display()))
}

/// `yue2-check MODEL.gguf REF.json`: next-token logits at a few song positions against YuE2's own (PyTorch).
pub fn check(model: &Path, reference: &Path) -> Result<(), String> {
    let j = Json::parse(&std::fs::read_to_string(reference).map_err(|e| e.to_string())?)?;
    let prefix = ids(&j, "prefix")?;
    let song = ids(&j, "song")?;
    let y = Yue2::load(model, prefix.len() + song.len() + 8)?;
    let mut s = y.session()?;
    s.reset()?;
    let t = Instant::now();
    y.prefill(&prefix, &mut s)?;
    println!("prompt: {} tokens in {:.0} ms", prefix.len(), t.elapsed().as_secs_f64() * 1000.0);
    let checks = j.get("checks").and_then(Json::as_array).ok_or("missing \"checks\"")?;
    let mut fed = 0usize;
    let mut worst_kl = 0f64;
    let mut top1 = 0usize;
    for c in checks {
        let after = c.get("after").and_then(Json::as_f64).ok_or("check without \"after\"")? as usize;
        let want: Vec<f32> = c.get("logits").and_then(Json::as_array).ok_or("check without logits")?.iter().filter_map(Json::as_f64).map(|v| v as f32).collect();
        while fed < after {
            y.model.forward_batch(&[song[fed]], &mut s, true)?;
            fed += 1;
        }
        let got = &s.logits;
        if got.len() != want.len() {
            return Err(format!("{} logits, the reference has {}", got.len(), want.len()));
        }
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        let (ga, wa) = (argmax(got), argmax(&want));
        top1 += (ga == wa) as usize;
        let max_diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // KL(reference || ours) over the softmax
        let lse = |v: &[f32]| {
            let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
        };
        let (lg, lw) = (lse(got), lse(&want));
        let kl: f64 = want.iter().zip(got.iter()).map(|(&w, &g)| {
            let pw = (w as f64 - lw).exp();
            pw * ((w as f64 - lw) - (g as f64 - lg))
        }).sum();
        worst_kl = worst_kl.max(kl);
        println!("after {after:>3} song tokens: top-1 {} ({ga} vs {wa}), max logit difference {max_diff:.3}, KL {kl:.5}", if ga == wa { "same" } else { "DIFFERENT" });
    }
    println!("top-1 agreed at {top1} of {} positions; worst KL {worst_kl:.5}", checks.len());
    if worst_kl > 0.05 {
        return Err("the logits do not match YuE2's".into());
    }
    Ok(())
}
