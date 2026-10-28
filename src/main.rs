mod cuda;
mod gguf;
mod json;
mod model;
mod profile;
mod quant;
mod sampler;
mod server;
mod sysinfo;
mod tokenizer;
mod tools;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{Gguf, Value};
use model::{Config, Model, Session};
use sampler::Sampler;
use tokenizer::{Tokenizer, Utf8Stream};

const USAGE: &str = "ghost-infer - run GGUF language models on the CPU or an NVIDIA GPU

USAGE:
    ghost-infer inspect  <model.gguf>
    ghost-infer meta     <model.gguf> <key>
    ghost-infer dequant  <model.gguf> <tensor> <row>   (one row of a weight as floats, for checking formats)
    ghost-infer tokenize <model.gguf> <text>
    ghost-infer top      <model.gguf> <text> [--gpu]   (next-token probabilities after raw text)
    ghost-infer bench    <model.gguf> [-n <gen tokens>] [--prompt-len <n>] [--threads <n>] [--gpu] [--draft <gguf>]
    ghost-infer run      <model.gguf> -p <prompt> [options]
    ghost-infer chat     <model.gguf> [options]
    ghost-infer serve    <model.gguf> [--host <addr>] [--port <n>] [options]   (OpenAI-compatible HTTP API)

OPTIONS:
    -p, --prompt <text>   prompt (run)
    -n, --max <n>         max new tokens per reply          [256]
    -t, --temp <f>        temperature, 0 = greedy            [0.7]
        --top-k <n>       keep the k most likely tokens, 0 = all [40]
        --top-p <f>       nucleus sampling cut-off           [0.9]
        --seed <n>        sampling seed                      [time]
        --ctx <n>         context window in tokens  [run: 2048; chat, serve: GPU as much as fits, CPU 4096]
        --system <text>   system prompt (chat models)
        --raw             send the prompt as-is, no chat template
        --gpu             run on the NVIDIA GPU (CUDA driver + NVRTC, loaded at runtime)
        --draft <gguf>    speculative decoding with this small model (GPU, greedy; same output, faster)
        --draft-k <n>     tokens the draft proposes per step  [3]
        --kv <f16|int8>   GPU KV cache: int8 holds about twice the context  [int8, or GHOST_INFER_KV]
        --threads <n>     CPU worker threads                 [physical cores]
        --special         print special tokens
        --host <addr>     serve: address to listen on        [127.0.0.1]
        --port <n>        serve: port                        [8080]
        --no-think        serve: Qwen3 answers without thinking unless a request asks for it
        --models <dir>    serve: another folder of GGUF files for the model picker (repeatable)
";

struct Opts {
    model: PathBuf,
    prompt: Option<String>,
    max_new: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    ctx: Option<usize>, // None: the command's default (0 = fit to memory)
    system: Option<String>,
    raw: bool,
    show_special: bool,
    prompt_len: usize,
    gpu: bool,
    no_think: bool,
    draft: Option<PathBuf>,
    draft_k: usize,
    host: String,
    port: u16,
    models: Vec<PathBuf>, // serve: more folders of GGUF files to switch to, besides the model's own
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    // token generation is memory-bound, so hyper-threads only add contention: default to one thread per core
    if !args.iter().any(|a| a == "--threads") {
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let _ = rayon::ThreadPoolBuilder::new().num_threads((logical / 2).max(1)).build_global();
    }
    let mut it = args.iter();
    let model = PathBuf::from(it.next().ok_or("missing model path")?);
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    let mut o = Opts {
        model,
        prompt: None,
        max_new: 256,
        temp: 0.7,
        top_k: 40,
        top_p: 0.9,
        seed,
        ctx: None,
        system: None,
        raw: false,
        show_special: false,
        prompt_len: 256,
        gpu: false,
        no_think: false,
        draft: None,
        draft_k: 3,
        host: "127.0.0.1".into(),
        port: 8080,
        models: Vec::new(),
    };
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        let num = |s: String, name: &str| s.parse::<f64>().map_err(|_| format!("{name}: \"{s}\" is not a number"));
        match a.as_str() {
            "-p" | "--prompt" => o.prompt = Some(val(a)?),
            "-n" | "--max" => o.max_new = num(val(a)?, a)? as usize,
            "-t" | "--temp" => o.temp = num(val(a)?, a)? as f32,
            "--top-k" => o.top_k = num(val(a)?, a)? as usize,
            "--top-p" => o.top_p = num(val(a)?, a)? as f32,
            "--seed" => o.seed = num(val(a)?, a)? as u64,
            "--ctx" => o.ctx = Some(num(val(a)?, a)? as usize),
            "--prompt-len" => o.prompt_len = num(val(a)?, a)? as usize,
            "--system" => o.system = Some(val(a)?),
            "--threads" => {
                let n = num(val(a)?, a)? as usize;
                rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| e.to_string())?;
            }
            "--raw" => o.raw = true,
            "--special" => o.show_special = true,
            "--gpu" => o.gpu = true,
            "--no-think" => o.no_think = true,
            "--draft" => o.draft = Some(PathBuf::from(val(a)?)),
            "--draft-k" => o.draft_k = num(val(a)?, a)? as usize,
            "--host" => o.host = val(a)?,
            "--port" => o.port = num(val(a)?, a)? as u16,
            "--models" => o.models.push(PathBuf::from(val(a)?)),
            "--kv" => match val(a)?.as_str() {
                "f16" => cuda::set_kv_int8(false),
                "int8" => cuda::set_kv_int8(true),
                v => return Err(format!("--kv: \"{v}\" is not f16 or int8")),
            },
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(o)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or(&[]);
    let result = match args.first().map(String::as_str) {
        Some("inspect") if rest.len() == 1 => inspect(&PathBuf::from(&rest[0])),
        Some("meta") if rest.len() == 2 => meta(&PathBuf::from(&rest[0]), &rest[1]),
        Some("dequant") if rest.len() == 3 => dequant(&PathBuf::from(&rest[0]), &rest[1], &rest[2]),
        Some("tokenize") if rest.len() == 2 => tokenize(&PathBuf::from(&rest[0]), &rest[1]),
        Some("top") if rest.len() == 2 || (rest.len() == 3 && rest[2] == "--gpu") => top(&PathBuf::from(&rest[0]), &rest[1], rest.len() == 3),
        Some("bench") if !rest.is_empty() => parse_opts(rest).and_then(|o| bench(&o)),
        Some("run") if !rest.is_empty() => parse_opts(rest).and_then(|o| run(&o)),
        Some("chat") if !rest.is_empty() => parse_opts(rest).and_then(|o| chat(&o)),
        Some("serve") if !rest.is_empty() => parse_opts(rest).and_then(|o| server::serve(&o, &o.host, o.port)),
        _ => {
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &PathBuf) -> Result<(), String> {
    let g = Gguf::open(path)?;
    println!("GGUF v{}  {} metadata keys  {} tensors  {:.1} MB", g.version, g.metadata.len(), g.tensors.len(), g.mmap.len() as f64 / 1e6);
    println!("\n== metadata ==");
    for (k, v) in &g.metadata {
        println!("{k:<45} {v}");
    }
    println!("\n== tensors (first 12) ==");
    for t in g.tensors.iter().take(12) {
        println!("{:<32} {:<6} {:?}", t.name, t.ty.name(), t.dims);
    }
    let mut by_type = std::collections::BTreeMap::<String, (usize, u64)>::new();
    for t in &g.tensors {
        let e = by_type.entry(t.ty.name()).or_default();
        e.0 += 1;
        e.1 += t.n_elements();
    }
    println!("\n== tensor types ==");
    for (ty, (count, elems)) in by_type {
        println!("{ty:<8} {count:>4} tensors  {:>8.2} M params", elems as f64 / 1e6);
    }
    Ok(())
}

fn meta(path: &PathBuf, key: &str) -> Result<(), String> {
    let g = Gguf::open(path)?;
    match g.get(key).ok_or_else(|| format!("no key {key}"))? {
        Value::Str(s) => println!("{s}"),
        other => println!("{other}"),
    }
    Ok(())
}

/// Row `row` of tensor `name`, dequantised, one value per line (rows of every expert counted together).
fn dequant(path: &PathBuf, name: &str, row: &str) -> Result<(), String> {
    let g = Gguf::open(path)?;
    let t = g.tensor(name).ok_or_else(|| format!("no tensor {name}"))?;
    let cols = t.dims[0] as usize;
    let rows = (t.n_elements() as usize) / cols;
    let r: usize = row.parse().map_err(|_| format!("row \"{row}\" is not a number"))?;
    if r >= rows || t.ty.block().is_none() {
        return Err(format!("{name}: row {r} of {rows} ({})", t.ty.name()));
    }
    let m = quant::Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows, cols };
    let mut out = vec![0f32; cols];
    m.row_f32(r, &mut out);
    let mut text = String::with_capacity(cols * 12);
    for v in out {
        text.push_str(&format!("{v:e}\n"));
    }
    print!("{text}");
    Ok(())
}

fn tokenize(path: &PathBuf, text: &str) -> Result<(), String> {
    let g = Gguf::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(text);
    println!("{} tokens", ids.len());
    for id in &ids {
        println!("{id:>7}  {:?}", tok.token_text(*id));
    }
    println!("ids: {ids:?}");
    println!("round trip ok: {}", tok.decode(&ids, true) == text);
    Ok(())
}

/// Where the transformer runs. Both keep their own KV cache and expose next-token logits.
enum Backend<'a> {
    Cpu(Model<'a>, Session),
    Gpu(cuda::GpuModel, cuda::GpuSession),
}

impl<'a> Backend<'a> {
    fn new(g: &'a Gguf, gpu: bool, ctx: usize) -> Result<Backend<'a>, String> {
        if gpu {
            let cfg = Config::from_gguf(g)?;
            let model = cuda::GpuModel::load(g, cfg)?;
            let session = cuda::GpuSession::new(&model.cfg, ctx)?;
            Ok(Backend::Gpu(model, session))
        } else {
            let model = Model::load(g)?;
            let session = Session::new(&model.cfg, ctx);
            Ok(Backend::Cpu(model, session))
        }
    }

    fn describe(&self) -> String {
        match self {
            Backend::Cpu(..) => format!("CPU, {} threads", rayon::current_num_threads()),
            Backend::Gpu(m, _) => format!("GPU {}", m.device),
        }
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), String> {
        // GHOST_INFER_STEPWISE=1: feed the prompt one token at a time, to test the generation path
        if std::env::var_os("GHOST_INFER_STEPWISE").is_some() {
            return ids.iter().try_for_each(|&id| self.step(id));
        }
        match self {
            Backend::Cpu(m, s) => m.prefill(ids, s),
            Backend::Gpu(m, s) => m.prefill(ids, s),
        }
    }

    fn step(&mut self, id: u32) -> Result<(), String> {
        match self {
            Backend::Cpu(m, s) => m.forward(id, s, true),
            Backend::Gpu(m, s) => m.forward_batch(&[id], s, true),
        }
    }

    fn logits(&self) -> &[f32] {
        match self {
            Backend::Cpu(_, s) => &s.logits,
            Backend::Gpu(_, s) => &s.logits,
        }
    }

    fn pos(&self) -> usize {
        match self {
            Backend::Cpu(_, s) => s.pos,
            Backend::Gpu(_, s) => s.pos,
        }
    }

    fn ctx(&self) -> usize {
        match self {
            Backend::Cpu(_, s) => s.ctx,
            Backend::Gpu(_, s) => s.ctx,
        }
    }

    fn reset(&mut self) {
        match self {
            Backend::Cpu(_, s) => s.reset(),
            Backend::Gpu(_, s) => s.pos = 0,
        }
    }
}

/// A small model proposing tokens for the GPU model to check (speculative decoding). It sees the same
/// token sequence as the main model; between generations its cache holds exactly the main model's.
struct Draft {
    model: cuda::GpuModel,
    sess: cuda::GpuSession,
    k: usize,
    proposed: usize,
    accepted: usize,
    draft_secs: f64,
    verify_secs: f64,
    rounds: usize,
}

struct Engine<'a> {
    backend: Backend<'a>,
    tok: Arc<Tokenizer>,
    stop: Vec<u32>,
    draft: Option<Draft>,
    /// the tokens in the main model's cache, in order (generate_with keeps it; the draft catches up from it)
    hist: Vec<u32>,
}

struct Stats {
    prompt_tokens: usize,
    prompt_secs: f64,
    gen_tokens: usize,
    gen_secs: f64,
    stopped_on: Option<u32>,
}

/// Called with each generated token; Ok(false) stops after it (the token is still read into the cache).
type Emit<'e> = &'e mut dyn FnMut(u32) -> Result<bool, String>;

fn load<'a>(g: &'a Gguf, gpu: bool, ctx: usize) -> Result<Engine<'a>, String> {
    load_with_draft(g, gpu, ctx, None, 0)
}

/// ctx 0: on the GPU the longest context whose caches (the draft's too) fit in the memory the weights
/// leave free; on the CPU 4096.
fn load_with_draft<'a>(g: &'a Gguf, gpu: bool, ctx: usize, draft: Option<&PathBuf>, draft_k: usize) -> Result<Engine<'a>, String> {
    let tok = Tokenizer::from_gguf(g)?;
    // the models first, then their sessions, so that an automatic context knows what is left
    let main = if gpu { Some(cuda::GpuModel::load(g, Config::from_gguf(g)?)?) } else { None };
    let draft_model = match draft {
        None => None,
        Some(path) => {
            if !gpu {
                return Err("--draft needs --gpu".into());
            }
            if !(1..cuda::DECODE_MAX_BATCH).contains(&draft_k) {
                return Err(format!("--draft-k must be 1..{}", cuda::DECODE_MAX_BATCH - 1));
            }
            let dg = Gguf::open(path)?;
            let dtok = Tokenizer::from_gguf(&dg)?;
            let probe = "Hello, world! <|im_start|>user\nThe 3 quick brown foxes<|im_end|>";
            if dtok.vocab_size() != tok.vocab_size() || dtok.encode(probe) != tok.encode(probe) {
                return Err("the draft model's tokenizer differs from the main model's".into());
            }
            let cfg = Config::from_gguf(&dg)?;
            let mut model = cuda::GpuModel::load(&dg, cfg)?;
            model.int8_q8 = true; // a draft only has to guess well, not match f32 maths
            Some(model)
        }
    };
    let ctx = match (ctx, &main) {
        (0, Some(m)) => {
            let fit = cuda::fit_ctx(&[Some(&m.cfg), draft_model.as_ref().map(|d| &d.cfg)].into_iter().flatten().collect::<Vec<_>>())?;
            if m.cfg.is_moe() { fit.min(cuda::MOE_DEFAULT_CTX) } else { fit }
        }
        (0, None) => 4096,
        (c, _) => c,
    };
    let backend = match main {
        Some(m) => {
            let s = cuda::GpuSession::new(&m.cfg, ctx)?;
            Backend::Gpu(m, s)
        }
        None => Backend::new(g, false, ctx)?,
    };
    let draft = match draft_model {
        Some(model) => {
            let sess = cuda::GpuSession::new(&model.cfg, ctx)?;
            Some(Draft { model, sess, k: draft_k, proposed: 0, accepted: 0, draft_secs: 0.0, verify_secs: 0.0, rounds: 0 })
        }
        None => None,
    };
    let mut stop: Vec<u32> = tok.eos_id.into_iter().collect();
    // GLM ends its turn by starting the next one (<|user|>, or <|observation|> for a tool's result)
    for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|user|>", "<|observation|>"] {
        if let Some(id) = tok.token_id(t) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
    }
    Ok(Engine { backend, tok: Arc::new(tok), stop, draft, hist: Vec::new() })
}

impl Engine<'_> {
    /// Feeds `prompt` then samples up to `max_new` tokens, streaming text to stdout.
    fn generate(&mut self, prompt: &[u32], max_new: usize, sampler: &mut Sampler, show_special: bool) -> Result<Stats, String> {
        let tok = self.tok.clone(); // a shared handle: generate_with borrows the engine
        let mut out = std::io::stdout().lock();
        let mut utf8 = Utf8Stream::default();
        let st = self.generate_with(prompt, max_new, sampler, &mut |id| {
            let text = utf8.push(&tok.token_bytes(id, show_special));
            if !text.is_empty() {
                write!(out, "{text}").and_then(|_| out.flush()).map_err(|e| e.to_string())?;
            }
            Ok(true)
        })?;
        writeln!(out).map_err(|e| e.to_string())?;
        Ok(st)
    }

    /// Feeds `prompt` after what the cache holds, then samples up to `max_new` tokens, handing each to
    /// `emit`. Greedy sampling with a draft model goes through speculative decoding (same tokens).
    fn generate_with(&mut self, prompt: &[u32], max_new: usize, sampler: &mut Sampler, emit: Emit) -> Result<Stats, String> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if self.backend.pos() + prompt.len() >= self.backend.ctx() {
            return Err(format!("prompt does not fit: {} + {} tokens > context {}", self.backend.pos(), prompt.len(), self.backend.ctx()));
        }
        let t0 = Instant::now();
        let base = self.backend.pos();
        self.hist.truncate(base);
        self.backend.prefill(prompt)?;
        if let Some(d) = self.draft.as_mut().filter(|_| sampler.is_greedy()) {
            // the draft skips sampled replies; it catches up on whatever it missed, then reads the prompt
            d.sess.pos = d.sess.pos.min(base);
            let missed = self.hist[d.sess.pos..].to_vec();
            d.model.prefill(&missed, &mut d.sess)?;
            d.model.prefill(prompt, &mut d.sess)?;
        }
        self.hist.extend_from_slice(prompt);
        let prompt_secs = t0.elapsed().as_secs_f64();

        let mut ids = Vec::new();
        let t1 = Instant::now();
        if let (Some(d), true) = (self.draft.as_mut(), sampler.is_greedy()) {
            let (generated, stopped_on) = speculate(&mut self.backend, d, &self.stop, max_new, true, &mut |id| {
                ids.push(id);
                emit(id)
            })?;
            self.hist.extend_from_slice(&ids);
            return Ok(Stats { prompt_tokens: prompt.len(), prompt_secs, gen_tokens: generated, gen_secs: t1.elapsed().as_secs_f64(), stopped_on });
        }

        let mut generated = 0;
        let mut stopped_on = None;
        while generated < max_new && self.backend.pos() < self.backend.ctx() {
            let next = sampler.sample(self.backend.logits());
            generated += 1;
            if self.stop.contains(&next) {
                stopped_on = Some(next);
                break;
            }
            ids.push(next);
            let more = emit(next)?;
            self.backend.step(next)?;
            if !more {
                break;
            }
        }
        self.hist.extend_from_slice(&ids);
        Ok(Stats { prompt_tokens: prompt.len(), prompt_secs, gen_tokens: generated, gen_secs: t1.elapsed().as_secs_f64(), stopped_on })
    }

    /// Makes the cache hold only its first `pos` tokens (in the draft model too).
    fn truncate(&mut self, pos: usize) {
        self.hist.truncate(pos);
        match &mut self.backend {
            Backend::Cpu(_, s) => s.pos = s.pos.min(pos),
            Backend::Gpu(_, s) => s.pos = s.pos.min(pos),
        }
        if let Some(d) = self.draft.as_mut() {
            d.sess.pos = d.sess.pos.min(pos);
        }
    }

    fn speculate(&mut self, max_new: usize, honour_stop: bool, emit: Emit) -> Result<(usize, Option<u32>), String> {
        let d = self.draft.as_mut().ok_or("no draft model")?;
        speculate(&mut self.backend, d, &self.stop, max_new, honour_stop, emit)
    }
}

/// Greedy decoding with a draft model, after both models have read the prompt. Produces exactly the
/// tokens of the plain greedy loop, and leaves both caches as that loop leaves the main model's: every
/// emitted token read except a final stop token. Returns (tokens generated, stop token if any).
fn speculate(backend: &mut Backend, d: &mut Draft, stop: &[u32], max_new: usize, honour_stop: bool, emit: Emit) -> Result<(usize, Option<u32>), String> {
    let Backend::Gpu(tm, ts) = backend else { return Err("speculative decoding needs the GPU".into()) };
    let base = ts.pos;
    let mut seq: Vec<u32> = Vec::new(); // tokens the main model has read since `base`
    let mut last = argmax(&ts.logits); // decided but not yet read by either model
    let mut pending: Vec<u32> = Vec::new(); // read by the main model, not yet by the draft
    let mut generated = 0;
    let mut stopped_on = None;
    'decode: loop {
        generated += 1;
        if honour_stop && stop.contains(&last) {
            stopped_on = Some(last);
            break;
        }
        let more = emit(last)?;
        if !more || generated >= max_new || ts.pos + 1 >= ts.ctx {
            tm.forward_batch(&[last], ts, false)?; // the plain loop reads its last token too
            seq.push(last);
            break;
        }
        let k = d.k.min(max_new - generated).min(ts.ctx - ts.pos - 1); // >= 1 here

        // the draft catches up and guesses k tokens
        let td = Instant::now();
        let dpos0 = d.sess.pos;
        let mut feed = std::mem::take(&mut pending);
        feed.push(last);
        let mut guesses = vec![*d.model.forward_argmax(&feed, &mut d.sess)?.last().unwrap()];
        while guesses.len() < k {
            let g = d.model.forward_argmax(&[*guesses.last().unwrap()], &mut d.sess)?[0];
            guesses.push(g);
        }
        // the main model reads `last` and the guesses in one batch: its pick after each of them
        let tv = Instant::now();
        d.draft_secs += (tv - td).as_secs_f64();
        let tpos0 = ts.pos;
        let mut batch = vec![last];
        batch.extend_from_slice(&guesses);
        let picks = tm.forward_argmax(&batch, ts)?;
        d.verify_secs += tv.elapsed().as_secs_f64();
        d.rounds += 1;
        let a = guesses.iter().zip(&picks).take_while(|(g, p)| g == p).count();
        d.proposed += k;
        d.accepted += a;
        // keep `last` and the accepted guesses; positions after them get overwritten later
        ts.pos = tpos0 + 1 + a;
        seq.push(last);
        d.sess.pos = dpos0 + feed.len() + a.min(k - 1);
        if a == k {
            pending.push(guesses[k - 1]); // the draft never read its own last guess
        }
        for (i, &g) in guesses[..a].iter().enumerate() {
            generated += 1;
            if honour_stop && stop.contains(&g) {
                ts.pos = tpos0 + 1 + i; // the plain loop does not read a stop token
                stopped_on = Some(g);
                break 'decode;
            }
            let more = emit(g)?;
            seq.push(g);
            if !more || generated >= max_new {
                ts.pos = tpos0 + 2 + i;
                break 'decode;
            }
        }
        last = picks[a];
    }
    // leave the draft holding exactly the main model's sequence
    seq.truncate(ts.pos - base);
    if d.sess.pos > ts.pos {
        d.sess.pos = ts.pos;
    } else if d.sess.pos < ts.pos {
        d.model.forward_batch(&seq[d.sess.pos - base..], &mut d.sess, false)?;
    }
    Ok((generated, stopped_on))
}

impl Engine<'_> {
    /// Forgets the conversation, in the draft model too.
    fn reset(&mut self) {
        self.backend.reset();
        self.hist.clear();
        if let Some(d) = self.draft.as_mut() {
            d.sess.pos = 0;
        }
    }

    fn print_draft_stats(&self) {
        if let Some(d) = self.draft.as_ref().filter(|d| d.proposed > 0) {
            eprintln!(
                "[draft: {} of {} proposed tokens accepted ({:.0}%), {} per step | per step: draft {:.2} ms, verify {:.2} ms]",
                d.accepted,
                d.proposed,
                100.0 * d.accepted as f64 / d.proposed as f64,
                d.k,
                1000.0 * d.draft_secs / d.rounds as f64,
                1000.0 * d.verify_secs / d.rounds as f64
            );
        }
    }
}

fn warn_draft_unused(o: &Opts) {
    if o.draft.is_some() && o.temp > 0.0 {
        eprintln!("[note: the draft model is used for greedy decoding only (-t 0); sampling without it]");
    }
}

/// The greedy sampler's pick: the first maximum.
fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

fn top(path: &PathBuf, text: &str, gpu: bool) -> Result<(), String> {
    let g = Gguf::open(path)?;
    let mut engine = load(&g, gpu, 4096)?;
    let ids = engine.tok.encode(text);
    engine.backend.prefill(&ids)?;
    let logits = engine.backend.logits();
    // GHOST_INFER_LOGITS_OUT=<file>: every logit as little-endian f32, for accuracy tooling
    if let Some(out) = std::env::var_os("GHOST_INFER_LOGITS_OUT") {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out, bytes).map_err(|e| format!("cannot write {}: {e}", PathBuf::from(&out).display()))?;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    println!("{} prompt tokens: {ids:?}", ids.len());
    for &i in idx.iter().take(10) {
        let p = ((logits[i] - max) as f64).exp() / total;
        println!("{:>7}  {:>8.4}  logprob {:>8.4}  {:?}", i, p, p.ln(), engine.tok.decode(&[i as u32], true));
    }
    Ok(())
}

/// Fixed workload: a `prompt_len`-token prompt, then `max_new` greedy tokens; best of 3 runs.
fn bench(o: &Opts) -> Result<(), String> {
    let g = Gguf::open(&o.model)?;
    let mut engine = load_with_draft(&g, o.gpu, o.prompt_len + o.max_new + 8, o.draft.as_ref(), o.draft_k)?;
    let base = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg developed movable metal type. ";
    let mut ids = Vec::new();
    while ids.len() < o.prompt_len {
        ids.extend(engine.tok.encode(base));
    }
    ids.truncate(o.prompt_len);
    let mut sampler = Sampler::new(0.0, 0, 1.0, 1);
    let (mut best_pp, mut best_tg) = (0f64, 0f64);
    for round in 0..3 {
        engine.reset();
        profile::reset();
        let t0 = Instant::now();
        engine.backend.prefill(&ids)?;
        if let Some(d) = engine.draft.as_mut() {
            d.model.prefill(&ids, &mut d.sess)?;
        }
        let pp = ids.len() as f64 / t0.elapsed().as_secs_f64();
        if round == 2 && profile::enabled() {
            eprintln!("-- prompt phase --");
            profile::report(ids.len());
        }
        profile::reset();
        let t1 = Instant::now();
        if engine.draft.is_some() {
            engine.speculate(o.max_new, false, &mut |_| Ok(true))?;
        } else {
            for _ in 0..o.max_new {
                let next = sampler.sample(engine.backend.logits());
                engine.backend.step(next)?;
            }
        }
        let tg = o.max_new as f64 / t1.elapsed().as_secs_f64();
        best_pp = best_pp.max(pp);
        best_tg = best_tg.max(tg);
    }
    println!("prompt {} tok: {:>8.1} tok/s | generate {} tok: {:>7.1} tok/s | {}", ids.len(), best_pp, o.max_new, best_tg, engine.backend.describe());
    if let Backend::Gpu(m, _) = &engine.backend {
        if let Some(cache) = m.expert_cache() {
            println!("expert cache: {cache}");
        }
    }
    // GHOST_INFER_VERIFY_N=n: time one n-token decode batch (a speculative verify step) on its own
    if let (Some(n), Backend::Gpu(m, s)) = (std::env::var("GHOST_INFER_VERIFY_N").ok().and_then(|v| v.parse::<usize>().ok()), &mut engine.backend) {
        let batch: Vec<u32> = ids[..n].to_vec();
        let pos = s.pos;
        profile::reset();
        let t = Instant::now();
        for _ in 0..10 {
            s.pos = pos;
            m.forward_argmax(&batch, s)?;
        }
        println!("{n}-token decode batch: {:.2} ms", t.elapsed().as_secs_f64() * 100.0);
        if profile::enabled() {
            profile::report(10 * n);
        }
    }
    engine.print_draft_stats();
    if profile::enabled() {
        eprintln!("-- generation phase --");
        profile::report(o.max_new); // last run
    }
    Ok(())
}

/// Chat formatting. Only ChatML (<|im_start|> ... <|im_end|>) for now, which covers SmolLM2 and Qwen.
struct ChatFormat {
    default_system: Option<String>,
    /// the template has Qwen3's thinking switch (enable_thinking); models without it never think, and
    /// an empty <think></think> block where they do not expect one makes them stop after a few tokens
    thinking: bool,
    /// GLM-4's format ([gMASK]<sop>, <|user|> / <|assistant|> with no newlines) instead of ChatML
    glm: bool,
}

impl ChatFormat {
    fn detect(g: &Gguf) -> Result<ChatFormat, String> {
        let tpl = g.get("tokenizer.chat_template").and_then(Value::as_str).unwrap_or("");
        let glm = tpl.contains("[gMASK]") && tpl.contains("<|assistant|>");
        if glm {
            return Ok(ChatFormat { default_system: None, thinking: tpl.contains("enable_thinking"), glm });
        }
        if !tpl.contains("<|im_start|>") {
            return Err("this model's chat template is neither ChatML nor GLM's; use --raw".into());
        }
        // the template's built-in system prompt, if it has one: '<|im_start|>system\n…<|im_end|>'
        let default_system = tpl.find("<|im_start|>system").and_then(|i| {
            let after = &tpl[i + "<|im_start|>system".len()..];
            let after = after.strip_prefix("\\n").or_else(|| after.strip_prefix('\n')).unwrap_or(after);
            let end = after.find("<|im_end|>")?;
            let text = &after[..end];
            (!text.is_empty() && !text.contains("{{") && !text.contains("{%")).then(|| text.to_string())
        });
        Ok(ChatFormat { default_system, thinking: tpl.contains("enable_thinking"), glm })
    }

    fn first_turn(&self, system: Option<&str>, user: &str) -> String {
        if self.glm {
            let sys = system.map(|s| format!("<|system|>{s}")).unwrap_or_default();
            return format!("[gMASK]<sop>{sys}{}", self.next_turn(user));
        }
        let mut s = String::new();
        if let Some(sys) = system.or(self.default_system.as_deref()) {
            s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
        }
        s.push_str(&self.next_turn(user));
        s
    }

    fn next_turn(&self, user: &str) -> String {
        if self.glm {
            return format!("<|user|>{user}<|assistant|>{}", if self.thinking { "<think>" } else { "</think>" });
        }
        format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    }
}

fn print_stats(st: &Stats) {
    eprintln!(
        "[prompt {} tok in {:.2}s = {:.1} tok/s | generated {} tok in {:.2}s = {:.1} tok/s{}]",
        st.prompt_tokens,
        st.prompt_secs,
        st.prompt_tokens as f64 / st.prompt_secs.max(1e-9),
        st.gen_tokens,
        st.gen_secs,
        st.gen_tokens as f64 / st.gen_secs.max(1e-9),
        if st.stopped_on.is_some() { ", stopped at end-of-turn" } else { "" }
    );
}

fn run(o: &Opts) -> Result<(), String> {
    let prompt = o.prompt.as_deref().ok_or("run needs -p <prompt>")?;
    let t = Instant::now();
    let g = Gguf::open(&o.model)?;
    let mut engine = load_with_draft(&g, o.gpu, o.ctx.unwrap_or(2048), o.draft.as_ref(), o.draft_k)?;
    warn_draft_unused(o);
    let text = if o.raw { prompt.to_string() } else { ChatFormat::detect(&g)?.first_turn(o.system.as_deref(), prompt) };
    let ids = engine.tok.encode(&text);
    eprintln!("[loaded {} on {} in {:.2}s]", g.get("general.name").and_then(Value::as_str).unwrap_or("model"), engine.backend.describe(), t.elapsed().as_secs_f64());
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, o.seed);
    let st = engine.generate(&ids, o.max_new, &mut sampler, o.show_special)?;
    print_stats(&st);
    engine.print_draft_stats();
    Ok(())
}

fn chat(o: &Opts) -> Result<(), String> {
    let g = Gguf::open(&o.model)?;
    let mut engine = load_with_draft(&g, o.gpu, o.ctx.unwrap_or(0), o.draft.as_ref(), o.draft_k)?;
    warn_draft_unused(o);
    let fmt = ChatFormat::detect(&g)?;
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, o.seed);
    eprintln!(
        "chatting with {} on {} - empty line or /bye to quit, /reset to start over",
        g.get("general.name").and_then(Value::as_str).unwrap_or("model"),
        engine.backend.describe()
    );

    let stdin = std::io::stdin();
    let mut first = true;
    let mut pending_end: Option<u32> = None;
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        // text piped from PowerShell starts with a byte-order mark
        let msg = line.trim_start_matches('\u{feff}').trim();
        if msg.is_empty() || msg == "/bye" {
            break;
        }
        if msg == "/reset" {
            engine.reset();
            first = true;
            pending_end = None;
            eprintln!("(conversation cleared)");
            continue;
        }
        // close the previous assistant turn: its <|im_end|> was sampled but not yet fed to the model
        let mut ids = Vec::new();
        if let Some(end) = pending_end.take().filter(|_| !fmt.glm) {
            ids.push(end);
            ids.extend(engine.tok.encode("\n"));
        }
        let text = if first { fmt.first_turn(o.system.as_deref(), msg) } else { fmt.next_turn(msg) };
        ids.extend(engine.tok.encode(&text));
        first = false;
        if engine.backend.pos() + ids.len() + 1 >= engine.backend.ctx() {
            eprintln!("(context full - starting a new conversation)");
            engine.reset();
            ids = engine.tok.encode(&fmt.first_turn(o.system.as_deref(), msg));
        }
        let st = engine.generate(&ids, o.max_new, &mut sampler, o.show_special)?;
        pending_end = st.stopped_on.or_else(|| engine.tok.token_id("<|im_end|>"));
        print_stats(&st);
    }
    Ok(())
}
