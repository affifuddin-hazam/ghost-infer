mod cuda;
mod gguf;
mod model;
mod profile;
mod quant;
mod sampler;
mod tokenizer;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{Gguf, Value};
use model::{Config, Model, Session};
use sampler::Sampler;
use tokenizer::{Tokenizer, Utf8Stream};

const USAGE: &str = "betterllama - run GGUF language models on the CPU or an NVIDIA GPU

USAGE:
    betterllama inspect  <model.gguf>
    betterllama meta     <model.gguf> <key>
    betterllama tokenize <model.gguf> <text>
    betterllama top      <model.gguf> <text> [--gpu]   (next-token probabilities after raw text)
    betterllama bench    <model.gguf> [-n <gen tokens>] [--prompt-len <n>] [--threads <n>] [--gpu]
    betterllama run      <model.gguf> -p <prompt> [options]
    betterllama chat     <model.gguf> [options]

OPTIONS:
    -p, --prompt <text>   prompt (run)
    -n, --max <n>         max new tokens per reply          [256]
    -t, --temp <f>        temperature, 0 = greedy            [0.7]
        --top-k <n>       keep the k most likely tokens, 0 = all [40]
        --top-p <f>       nucleus sampling cut-off           [0.9]
        --seed <n>        sampling seed                      [time]
        --ctx <n>         context window in tokens           [2048]
        --system <text>   system prompt (chat models)
        --raw             send the prompt as-is, no chat template
        --gpu             run on the NVIDIA GPU (CUDA driver + NVRTC, loaded at runtime)
        --threads <n>     CPU worker threads                 [physical cores]
        --special         print special tokens
";

struct Opts {
    model: PathBuf,
    prompt: Option<String>,
    max_new: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    ctx: usize,
    system: Option<String>,
    raw: bool,
    show_special: bool,
    prompt_len: usize,
    gpu: bool,
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
        ctx: 2048,
        system: None,
        raw: false,
        show_special: false,
        prompt_len: 256,
        gpu: false,
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
            "--ctx" => o.ctx = num(val(a)?, a)? as usize,
            "--prompt-len" => o.prompt_len = num(val(a)?, a)? as usize,
            "--system" => o.system = Some(val(a)?),
            "--threads" => {
                let n = num(val(a)?, a)? as usize;
                rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| e.to_string())?;
            }
            "--raw" => o.raw = true,
            "--special" => o.show_special = true,
            "--gpu" => o.gpu = true,
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
        Some("tokenize") if rest.len() == 2 => tokenize(&PathBuf::from(&rest[0]), &rest[1]),
        Some("top") if rest.len() == 2 || (rest.len() == 3 && rest[2] == "--gpu") => top(&PathBuf::from(&rest[0]), &rest[1], rest.len() == 3),
        Some("bench") if !rest.is_empty() => parse_opts(rest).and_then(|o| bench(&o)),
        Some("run") if !rest.is_empty() => parse_opts(rest).and_then(|o| run(&o)),
        Some("chat") if !rest.is_empty() => parse_opts(rest).and_then(|o| chat(&o)),
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
        // BETTERLLAMA_STEPWISE=1: feed the prompt one token at a time, to test the generation path
        if std::env::var_os("BETTERLLAMA_STEPWISE").is_some() {
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

struct Engine<'a> {
    backend: Backend<'a>,
    tok: Tokenizer,
    stop: Vec<u32>,
}

struct Stats {
    prompt_tokens: usize,
    prompt_secs: f64,
    gen_tokens: usize,
    gen_secs: f64,
    stopped_on: Option<u32>,
}

fn load<'a>(g: &'a Gguf, gpu: bool, ctx: usize) -> Result<Engine<'a>, String> {
    let backend = Backend::new(g, gpu, ctx)?;
    let tok = Tokenizer::from_gguf(g)?;
    let mut stop: Vec<u32> = tok.eos_id.into_iter().collect();
    for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
        if let Some(id) = tok.token_id(t) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
    }
    Ok(Engine { backend, tok, stop })
}

impl Engine<'_> {
    /// Feeds `prompt` then samples up to `max_new` tokens, streaming text to stdout.
    fn generate(&mut self, prompt: &[u32], max_new: usize, sampler: &mut Sampler, show_special: bool) -> Result<Stats, String> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if self.backend.pos() + prompt.len() >= self.backend.ctx() {
            return Err(format!("prompt does not fit: {} + {} tokens > context {}", self.backend.pos(), prompt.len(), self.backend.ctx()));
        }
        let t0 = Instant::now();
        self.backend.prefill(prompt)?;
        let prompt_secs = t0.elapsed().as_secs_f64();

        let mut out = std::io::stdout().lock();
        let mut utf8 = Utf8Stream::default();
        let t1 = Instant::now();
        let mut generated = 0;
        let mut stopped_on = None;
        while generated < max_new && self.backend.pos() < self.backend.ctx() {
            let next = sampler.sample(self.backend.logits());
            generated += 1;
            if self.stop.contains(&next) {
                stopped_on = Some(next);
                break;
            }
            let text = utf8.push(&self.tok.token_bytes(next, show_special));
            if !text.is_empty() {
                write!(out, "{text}").and_then(|_| out.flush()).map_err(|e| e.to_string())?;
            }
            self.backend.step(next)?;
        }
        writeln!(out).map_err(|e| e.to_string())?;
        Ok(Stats { prompt_tokens: prompt.len(), prompt_secs, gen_tokens: generated, gen_secs: t1.elapsed().as_secs_f64(), stopped_on })
    }
}

fn top(path: &PathBuf, text: &str, gpu: bool) -> Result<(), String> {
    let g = Gguf::open(path)?;
    let mut engine = load(&g, gpu, 4096)?;
    let ids = engine.tok.encode(text);
    engine.backend.prefill(&ids)?;
    let logits = engine.backend.logits();
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
    let mut engine = load(&g, o.gpu, o.prompt_len + o.max_new + 8)?;
    let base = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg developed movable metal type. ";
    let mut ids = Vec::new();
    while ids.len() < o.prompt_len {
        ids.extend(engine.tok.encode(base));
    }
    ids.truncate(o.prompt_len);
    let mut sampler = Sampler::new(0.0, 0, 1.0, 1);
    let (mut best_pp, mut best_tg) = (0f64, 0f64);
    for round in 0..3 {
        engine.backend.reset();
        profile::reset();
        let t0 = Instant::now();
        engine.backend.prefill(&ids)?;
        let pp = ids.len() as f64 / t0.elapsed().as_secs_f64();
        if round == 2 && profile::enabled() {
            eprintln!("-- prompt phase --");
            profile::report(ids.len());
        }
        profile::reset();
        let t1 = Instant::now();
        for _ in 0..o.max_new {
            let next = sampler.sample(engine.backend.logits());
            engine.backend.step(next)?;
        }
        let tg = o.max_new as f64 / t1.elapsed().as_secs_f64();
        best_pp = best_pp.max(pp);
        best_tg = best_tg.max(tg);
    }
    println!("prompt {} tok: {:>8.1} tok/s | generate {} tok: {:>7.1} tok/s | {}", ids.len(), best_pp, o.max_new, best_tg, engine.backend.describe());
    if profile::enabled() {
        eprintln!("-- generation phase --");
        profile::report(o.max_new); // last run
    }
    Ok(())
}

/// Chat formatting. Only ChatML (<|im_start|> ... <|im_end|>) for now, which covers SmolLM2 and Qwen.
struct ChatFormat {
    default_system: Option<String>,
}

impl ChatFormat {
    fn detect(g: &Gguf) -> Result<ChatFormat, String> {
        let tpl = g.get("tokenizer.chat_template").and_then(Value::as_str).unwrap_or("");
        if !tpl.contains("<|im_start|>") {
            return Err("this model's chat template is not ChatML; use --raw".into());
        }
        // the template's built-in system prompt, if it has one: '<|im_start|>system\n…<|im_end|>'
        let default_system = tpl.find("<|im_start|>system").and_then(|i| {
            let after = &tpl[i + "<|im_start|>system".len()..];
            let after = after.strip_prefix("\\n").or_else(|| after.strip_prefix('\n')).unwrap_or(after);
            let end = after.find("<|im_end|>")?;
            let text = &after[..end];
            (!text.is_empty() && !text.contains("{{") && !text.contains("{%")).then(|| text.to_string())
        });
        Ok(ChatFormat { default_system })
    }

    fn first_turn(&self, system: Option<&str>, user: &str) -> String {
        let mut s = String::new();
        if let Some(sys) = system.or(self.default_system.as_deref()) {
            s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
        }
        s.push_str(&self.next_turn(user));
        s
    }

    fn next_turn(&self, user: &str) -> String {
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
    let mut engine = load(&g, o.gpu, o.ctx)?;
    let text = if o.raw { prompt.to_string() } else { ChatFormat::detect(&g)?.first_turn(o.system.as_deref(), prompt) };
    let ids = engine.tok.encode(&text);
    eprintln!("[loaded {} on {} in {:.2}s]", g.get("general.name").and_then(Value::as_str).unwrap_or("model"), engine.backend.describe(), t.elapsed().as_secs_f64());
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, o.seed);
    let st = engine.generate(&ids, o.max_new, &mut sampler, o.show_special)?;
    print_stats(&st);
    Ok(())
}

fn chat(o: &Opts) -> Result<(), String> {
    let g = Gguf::open(&o.model)?;
    let mut engine = load(&g, o.gpu, o.ctx)?;
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
        let msg = line.trim();
        if msg.is_empty() || msg == "/bye" {
            break;
        }
        if msg == "/reset" {
            engine.backend.reset();
            first = true;
            pending_end = None;
            eprintln!("(conversation cleared)");
            continue;
        }
        // close the previous assistant turn: its <|im_end|> was sampled but not yet fed to the model
        let mut ids = Vec::new();
        if let Some(end) = pending_end.take() {
            ids.push(end);
            ids.extend(engine.tok.encode("\n"));
        }
        let text = if first { fmt.first_turn(o.system.as_deref(), msg) } else { fmt.next_turn(msg) };
        ids.extend(engine.tok.encode(&text));
        first = false;
        if engine.backend.pos() + ids.len() + 1 >= engine.backend.ctx() {
            eprintln!("(context full - starting a new conversation)");
            engine.backend.reset();
            ids = engine.tok.encode(&fmt.first_turn(o.system.as_deref(), msg));
        }
        let st = engine.generate(&ids, o.max_new, &mut sampler, o.show_special)?;
        pending_end = st.stopped_on.or_else(|| engine.tok.token_id("<|im_end|>"));
        print_stats(&st);
    }
    Ok(())
}
