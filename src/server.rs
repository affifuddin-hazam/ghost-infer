//! OpenAI-compatible HTTP server: /v1/chat/completions, /v1/completions (both with streaming over
//! server-sent events), /v1/models, and a chat page at /. A small HTTP/1.1 implementation on std::net, one connection thread
//! each; generation runs on the main thread, one request at a time. The cache keeps the tokens of the
//! last conversation, so a request that repeats it (a chat client resending the history) only reads
//! the new part. /api/models lists the GGUF files next to the model and /api/models/load switches to
//! one of them: the model thread drops the running model (freeing its GPU memory) and loads the other.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::json::{Json, num, obj, str};
use crate::sampler::Sampler;
use crate::tokenizer::Utf8Stream;
use crate::{ChatFormat, Engine, Opts};

const MAX_BODY: usize = 32 << 20;

enum Kind {
    Chat,
    Completion,
}

/// Work for the model thread.
enum Work {
    Job(Job),
    /// switch to this GGUF file; the reply comes once a model is running again (Err: it stayed on the old one)
    Load(PathBuf, mpsc::Sender<Result<(), String>>),
}

/// A generation request handed from a connection thread to the model thread.
struct Job {
    model: String, // the model the prompt was written for: if another one runs by then, the job is refused
    kind: Kind,
    prompt: String, // the full text the model reads (chat: already in the chat template)
    max_tokens: Option<usize>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    stop: Vec<String>,
    events: mpsc::Sender<Event>,
}

enum Event {
    Text(String),
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize },
    Error(u16, String),
}

/// The running model, as the connection threads see it.
#[derive(Clone)]
struct Loaded {
    id: String,
    ctx: usize,
    chat_ok: bool,
    default_system: Option<String>,
    thinking: bool, // the chat template has a thinking switch
}

#[derive(Default)]
struct Status {
    current: Option<Loaded>,
    loading: Option<String>, // the model being loaded; requests get 503 meanwhile
}

/// What the connection threads share: the model's status, and where the models are.
struct Shared {
    status: Mutex<Status>,
    dir: PathBuf,
    defaults: (f32, usize, f32, usize),
    think_default: bool,
}

fn model_id(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string()
}

/// The GGUF files in the models folder, by name.
fn list_models(dir: &Path) -> Vec<(String, PathBuf, u64)> {
    let mut found: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
        .map(|p| (model_id(&p), std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), p))
        .map(|(id, size, p)| (id, p, size))
        .collect();
    found.sort_by_key(|m| m.0.to_lowercase());
    found
}

pub fn serve(o: &Opts, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let dir = o.model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new(".")).to_path_buf();
    let shared = Arc::new(Shared { status: Mutex::new(Status::default()), dir, defaults: (o.temp, o.top_k, o.top_p, o.max_new), think_default: !o.no_think });
    let (work, rx) = mpsc::channel::<Work>();
    let s = shared.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let (work, s) = (work.clone(), s.clone());
            std::thread::spawn(move || {
                if let Err(e) = handle(conn, &work, &s) {
                    eprintln!("[connection: {e}]");
                }
            });
        }
    });

    // the model thread: the engine (GPU state included) never leaves it. Each turn of this loop loads a
    // model and serves it until a switch is asked for; the engine and its file are dropped before the next.
    let mut path = o.model.clone();
    let mut last_good: Option<PathBuf> = None;
    let mut waiting: Option<(mpsc::Sender<Result<(), String>>, Option<String>)> = None; // who asked for this switch, and an error to report
    loop {
        let id = model_id(&path);
        shared.status.lock().unwrap().loading = Some(id.clone());
        let t = Instant::now();
        // a failed switch goes back to the model that was running; failing at startup (or on the way back) ends the server
        let mut fall_back = |e: String, path: &mut PathBuf| -> Result<(), String> {
            match (&mut waiting, &last_good) {
                (Some(w), Some(prev)) if prev != path => {
                    eprintln!("[could not load {}: {e}; going back to {}]", model_id(path), model_id(prev));
                    w.1 = Some(format!("could not load {}: {e}", model_id(path)));
                    *path = prev.clone();
                    Ok(())
                }
                _ => Err(e),
            }
        };
        let g = match crate::gguf::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                fall_back(e, &mut path)?;
                continue;
            }
        };
        // the draft model only fits the model it was given with
        let draft = if path == o.model { o.draft.as_ref() } else { None };
        let mut engine = match crate::load_with_draft(&g, o.gpu, o.ctx.unwrap_or(0), draft, o.draft_k) {
            Ok(e) => e,
            Err(e) => {
                fall_back(e, &mut path)?;
                continue;
            }
        };
        let fmt = ChatFormat::detect(&g).ok();
        eprintln!(
            "[loaded {} on {} in {:.2}s, {}-token context{}]",
            id,
            engine.backend.describe(),
            t.elapsed().as_secs_f64(),
            engine.backend.ctx(),
            if engine.draft.is_some() { ", with a draft model for greedy requests" } else { "" }
        );
        if last_good.is_none() {
            eprintln!("chat in your browser: http://{host}:{port}/");
            eprintln!("OpenAI-compatible API: http://{host}:{port}/v1 (/v1/chat/completions, /v1/completions, /v1/models)");
        }
        if fmt.is_none() {
            eprintln!("[note: no ChatML template in this model, so /v1/chat/completions is off]");
        }
        *shared.status.lock().unwrap() = Status {
            current: Some(Loaded {
                id: id.clone(),
                ctx: engine.backend.ctx(),
                chat_ok: fmt.is_some(),
                thinking: fmt.as_ref().is_some_and(|f| f.thinking),
                default_system: fmt.and_then(|f| f.default_system),
            }),
            loading: None,
        };
        if let Some((w, err)) = waiting.take() {
            let _ = w.send(err.map_or(Ok(()), Err));
        }
        last_good = Some(path.clone());

        let (next, reply) = loop {
            match rx.recv() {
                Err(_) => return Ok(()),
                Ok(Work::Job(job)) => run(&mut engine, &id, job),
                Ok(Work::Load(p, reply)) if p == path => {
                    let _ = reply.send(Ok(()));
                }
                Ok(Work::Load(p, reply)) => break (p, reply),
            }
        };
        eprintln!("[switching from {id} to {}]", model_id(&next));
        path = next;
        waiting = Some((reply, None));
    }
}

fn run(engine: &mut Engine, model: &str, job: Job) {
    if job.model != model {
        let _ = job.events.send(Event::Error(409, format!("the model changed to {model} while this request waited; send it again")));
        return;
    }
    let t0 = Instant::now();
    let ids = engine.tok.encode(&job.prompt);
    let ctx = engine.backend.ctx();
    if ids.is_empty() || ids.len() >= ctx {
        let _ = job.events.send(Event::Error(400, format!("the prompt has {} tokens; the context holds {ctx}", ids.len())));
        return;
    }
    // reuse the cache for the part of the prompt it already holds (at least one token is read anew,
    // which gives the logits for the first reply token)
    let keep = engine.hist.iter().zip(&ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
    engine.truncate(keep);
    let max_new = job.max_tokens.unwrap_or(usize::MAX).min(ctx - ids.len()).max(1);
    let mut sampler = Sampler::new(job.temperature, job.top_k, job.top_p, job.seed);

    let tok = engine.tok.clone();
    let mut utf8 = Utf8Stream::default();
    let mut held = String::new(); // text not sent yet: it could be the start of a stop string
    let mut hit_stop = false;
    let events = job.events.clone();
    let stops = job.stop.clone();
    let result = engine.generate_with(&ids[keep..], max_new, &mut sampler, &mut |id| {
        held.push_str(&utf8.push(&tok.token_bytes(id, true)));
        if let Some(at) = stops.iter().filter_map(|s| held.find(s.as_str())).min() {
            held.truncate(at);
            hit_stop = true;
            let _ = events.send(Event::Text(std::mem::take(&mut held)));
            return Ok(false);
        }
        let keep_back = stops.iter().map(|s| partial_suffix(&held, s)).max().unwrap_or(0);
        let ready = held.len() - keep_back;
        if ready > 0 {
            let rest = held.split_off(ready);
            let text = std::mem::replace(&mut held, rest);
            if events.send(Event::Text(text)).is_err() {
                return Ok(false); // the client went away
            }
        }
        Ok(true)
    });
    match result {
        Ok(st) => {
            if !held.is_empty() {
                let _ = events.send(Event::Text(held));
            }
            let finish = if hit_stop || st.stopped_on.is_some() { "stop" } else { "length" };
            let _ = job.events.send(Event::Done { finish, prompt_tokens: ids.len(), completion_tokens: st.gen_tokens });
            eprintln!(
                "[{}: prompt {} tokens ({} new), {} generated in {:.2}s ({:.1} tok/s), finish {finish}]",
                match job.kind {
                    Kind::Chat => "chat",
                    Kind::Completion => "completion",
                },
                ids.len(),
                ids.len() - keep,
                st.gen_tokens,
                t0.elapsed().as_secs_f64(),
                st.gen_tokens as f64 / st.gen_secs.max(1e-9),
            );
        }
        Err(e) => {
            // the cache may be half-written: start the next request from scratch
            engine.truncate(0);

            let _ = job.events.send(Event::Error(500, e));
        }
    }
}

/// Length of the longest end of `text` that is the start of `stop` (and could still grow into it).
fn partial_suffix(text: &str, stop: &str) -> usize {
    (1..stop.len().min(text.len() + 1))
        .rev()
        .find(|&k| stop.is_char_boundary(k) && text.is_char_boundary(text.len() - k) && text.ends_with(&stop[..k]))
        .unwrap_or(0)
}

// ------------------------------------------------------------------ HTTP

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(conn: &mut TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(conn.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let path = parts.next().ok_or("no path")?.split('?').next().unwrap_or("").to_string();
    let (mut len, mut expect_continue) = (0usize, false);
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().map_err(|_| "bad Content-Length")?,
                "expect" => expect_continue = v.trim().eq_ignore_ascii_case("100-continue"),
                _ => {}
            }
        }
    }
    if len > MAX_BODY {
        return Err(format!("request body of {len} bytes is too large"));
    }
    if expect_continue {
        conn.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(|e| e.to_string())?;
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, body })
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    }
}

fn respond(conn: &mut TcpStream, code: u16, body: &Json) -> Result<(), String> {
    let text = body.to_string();
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n",
        status_text(code),
        text.len()
    );
    conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(text.as_bytes())).map_err(|e| e.to_string())
}

/// The built-in chat page (src/chat.html, compiled into the program).
fn respond_page(conn: &mut TcpStream) -> Result<(), String> {
    respond_file(conn, include_bytes!("chat.html"), "text/html; charset=utf-8", "no-cache")
}

/// The chat page's 3D library (three.js, MIT) and typeface (Instrument Sans, OFL), from src/vendor, compiled in
/// so the page works offline. The paths name the version, so browsers may keep them for good.
const VENDOR: [(&str, &[u8], &str); 3] = [
    ("/vendor/three-0.160.0.module.min.js", include_bytes!("vendor/three.module.min.js"), "text/javascript; charset=utf-8"),
    ("/vendor/instrument-sans-5.3.0.woff2", include_bytes!("vendor/instrument-sans.woff2"), "font/woff2"),
    ("/vendor/instrument-sans-italic-5.3.0.woff2", include_bytes!("vendor/instrument-sans-italic.woff2"), "font/woff2"),
];

fn respond_file(conn: &mut TcpStream, body: &[u8], content_type: &str, cache: &str) -> Result<(), String> {
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: {cache}\r\nConnection: close\r\n\r\n", body.len());
    conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(body)).map_err(|e| e.to_string())
}

fn error_body(message: &str) -> Json {
    obj(vec![("error", obj(vec![("message", str(message)), ("type", str("invalid_request_error"))]))])
}

/// One server-sent event as one HTTP chunk.
fn sse(conn: &mut TcpStream, data: &str) -> std::io::Result<()> {
    let event = format!("data: {data}\n\n");
    write!(conn, "{:x}\r\n{event}\r\n", event.len())?;
    conn.flush()
}

fn handle(mut conn: TcpStream, work: &mpsc::Sender<Work>, shared: &Shared) -> Result<(), String> {
    let req = read_request(&mut conn)?;
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (current, loading) = {
        let s = shared.status.lock().unwrap();
        (s.current.clone(), s.loading.clone())
    };
    let (defaults, think_default) = (shared.defaults, shared.think_default);
    match (req.method.as_str(), req.path.trim_end_matches('/')) {
        // the models folder, for the chat page's model picker
        ("GET", "/api/models") => {
            let models = list_models(&shared.dir)
                .into_iter()
                .map(|(id, _, size)| obj(vec![("id", str(&id)), ("size", num(size as f64))]))
                .collect();
            let opt = |s: Option<&str>| s.map(str).unwrap_or(Json::Null);
            let body = obj(vec![
                ("current", opt(current.as_ref().map(|c| c.id.as_str()))),
                ("loading", opt(loading.as_deref())),
                ("context_length", current.as_ref().map(|c| num(c.ctx as f64)).unwrap_or(Json::Null)),
                ("thinking", Json::Bool(current.as_ref().is_some_and(|c| c.thinking))),
                ("models", Json::Arr(models)),
            ]);
            return respond(&mut conn, 200, &body);
        }
        // the prompt being read: tokens done of the total (0: none), and whether it is waiting on the disk
        ("GET", "/api/progress") => {
            let (done, total, disk_ms, disk) = crate::cuda::PROGRESS.get();
            let body = obj(vec![("done", num(done as f64)), ("total", num(total as f64)), ("disk_ms", num(disk_ms as f64)), ("disk", Json::Bool(disk))]);
            return respond(&mut conn, 200, &body);
        }
        // {"id": "<file name without .gguf>"}: answers once the switch is done (it can take a minute)
        ("POST", "/api/models/load") => {
            let id = std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse).ok().and_then(|b| b.get("id").and_then(Json::as_str).map(String::from));
            let Some(id) = id else { return respond(&mut conn, 400, &error_body("send {\"id\": \"<model>\"}")) };
            let Some((_, path, _)) = list_models(&shared.dir).into_iter().find(|m| m.0 == id) else {
                return respond(&mut conn, 404, &error_body(&format!("no {id}.gguf in {}", shared.dir.display())));
            };
            if let Some(l) = &loading {
                return respond(&mut conn, 409, &error_body(&format!("{l} is loading; wait for it")));
            }
            let (tx, done) = mpsc::channel();
            work.send(Work::Load(path, tx)).map_err(|_| "the model thread stopped")?;
            return match done.recv() {
                Ok(Ok(())) => respond(&mut conn, 200, &obj(vec![("current", str(&id))])),
                Ok(Err(e)) => respond(&mut conn, 500, &error_body(&e)),
                Err(_) => respond(&mut conn, 500, &error_body("the model thread stopped")),
            };
        }
        _ => {}
    }
    // everything else needs a running model
    let Some(Loaded { id: model_id, ctx, chat_ok, default_system, thinking }) = current.filter(|_| loading.is_none()) else {
        if matches!(req.path.trim_end_matches('/'), "/v1/chat/completions" | "/v1/completions" | "/v1/models") {
            let what = loading.as_deref().unwrap_or("the model");
            return respond(&mut conn, 503, &error_body(&format!("loading {what}; try again in a moment")));
        }
        if let ("GET", "" | "/index.html") = (req.method.as_str(), req.path.trim_end_matches('/')) {
            return respond_page(&mut conn);
        }
        if let Some((_, body, content_type)) = VENDOR.iter().find(|v| v.0 == req.path) {
            return respond_file(&mut conn, body, content_type, "public, max-age=31536000, immutable");
        }
        return respond(&mut conn, 503, &error_body("loading"));
    };
    let (model_id, default_system) = (model_id.as_str(), default_system.as_deref());
    match (req.method.as_str(), req.path.trim_end_matches('/')) {
        ("OPTIONS", _) => {
            let head = format!("HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\nConnection: close\r\n\r\n");
            return conn.write_all(head.as_bytes()).map_err(|e| e.to_string());
        }
        ("GET", "" | "/index.html") => return respond_page(&mut conn),
        ("GET", p) if VENDOR.iter().any(|v| v.0 == p) => {
            let (_, body, content_type) = VENDOR.iter().find(|v| v.0 == p).unwrap();
            return respond_file(&mut conn, body, content_type, "public, max-age=31536000, immutable");
        }
        ("GET", "/health") => return respond(&mut conn, 200, &obj(vec![("status", str("ok"))])),
        ("GET", "/v1/models") => {
            let m = obj(vec![("id", str(model_id)), ("object", str("model")), ("created", num(created as f64)), ("owned_by", str("ghost-infer")), ("context_length", num(ctx as f64))]);
            return respond(&mut conn, 200, &obj(vec![("object", str("list")), ("data", Json::Arr(vec![m]))]));
        }
        ("POST", "/v1/chat/completions") if chat_ok => {}
        ("POST", "/v1/chat/completions") => return respond(&mut conn, 400, &error_body("this model has no chat template; use /v1/completions")),
        ("POST", "/v1/completions") => {}
        (_, "/v1/chat/completions" | "/v1/completions" | "/v1/models") => return respond(&mut conn, 405, &error_body("method not allowed")),
        _ => return respond(&mut conn, 404, &error_body("not found")),
    }
    let chat = req.path.trim_end_matches('/') == "/v1/chat/completions";
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return respond(&mut conn, 400, &error_body(&format!("request body: {e}"))),
    };
    let prompt = if chat { chat_prompt(&body, default_system, think_default, thinking) } else { completion_prompt(&body) };
    let prompt = match prompt {
        Ok(p) => p,
        Err(e) => return respond(&mut conn, 400, &error_body(&e)),
    };
    let f = |k: &str| body.get(k).and_then(Json::as_f64);
    let stop = match body.get("stop") {
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(a)) => a.iter().filter_map(|s| s.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    let stream = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let include_usage = body.get("stream_options").and_then(|s| s.get("include_usage")).and_then(Json::as_bool).unwrap_or(false);
    let (tx, events) = mpsc::channel();
    let job = Job {
        model: model_id.to_string(),
        kind: if chat { Kind::Chat } else { Kind::Completion },
        prompt,
        max_tokens: f("max_completion_tokens").or(f("max_tokens")).map(|n| n.max(1.0) as usize).or(Some(defaults.3)),
        temperature: f("temperature").map(|t| t as f32).unwrap_or(defaults.0),
        top_k: f("top_k").map(|k| k as usize).unwrap_or(defaults.1),
        top_p: f("top_p").map(|p| p as f32).unwrap_or(defaults.2),
        seed: f("seed").map(|s| s as u64).unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)),
        stop,
        events: tx,
    };
    work.send(Work::Job(job)).map_err(|_| "the model thread stopped")?;

    let id = format!("{}-{created:x}{:08x}", if chat { "chatcmpl" } else { "cmpl" }, rand_suffix());
    let object = if chat { "chat.completion" } else { "text_completion" };
    let choice = |text: Option<&str>, finish: Option<&str>, first: bool| -> Json {
        let finish = finish.map(str).unwrap_or(Json::Null);
        if chat {
            let mut delta = Vec::new();
            if first {
                delta.push(("role", str("assistant")));
            }
            if let Some(t) = text {
                delta.push(("content", str(t)));
            }
            obj(vec![("index", num(0)), ("delta", obj(delta)), ("finish_reason", finish)])
        } else {
            obj(vec![("index", num(0)), ("text", str(text.unwrap_or(""))), ("logprobs", Json::Null), ("finish_reason", finish)])
        }
    };
    // tool calls (chat requests with tools): split out of the text, sent as OpenAI's `tool_calls`
    let mut calls = (chat && !request_tools(&body).is_empty()).then(ToolCalls::default);
    let call_json = |index: usize, name: &str, arguments: &str| -> Json {
        obj(vec![
            ("index", num(index as f64)),
            ("id", str(&format!("call_{:08x}{index}", rand_suffix()))),
            ("type", str("function")),
            ("function", obj(vec![("name", str(name)), ("arguments", str(arguments))])),
        ])
    };
    let usage = |p: usize, c: usize| obj(vec![("prompt_tokens", num(p as f64)), ("completion_tokens", num(c as f64)), ("total_tokens", num((p + c) as f64))]);
    let chunk = |choices: Vec<Json>, extra: Option<(&str, Json)>| -> String {
        let mut fields = vec![
            ("id", str(&id)),
            ("object", str(if chat { "chat.completion.chunk" } else { "text_completion" })),
            ("created", num(created as f64)),
            ("model", str(model_id)),
            ("choices", Json::Arr(choices)),
        ];
        if let Some(e) = extra {
            fields.push(e);
        }
        obj(fields).to_string()
    };

    if !stream {
        let mut text = String::new();
        for ev in events {
            match ev {
                Event::Text(t) => text.push_str(&t),
                Event::Error(code, e) => return respond(&mut conn, code, &error_body(&e)),
                Event::Done { finish, prompt_tokens, completion_tokens } => {
                    let c = if chat {
                        let mut message = vec![("role", str("assistant")), ("content", str(&text))];
                        let mut finish = finish;
                        if let Some(tc) = calls.as_mut() {
                            let (mut content, mut found) = (String::new(), Vec::new());
                            for part in tc.push(&text).into_iter().chain(tc.finish()) {
                                match part {
                                    Part::Text(t) => content.push_str(&t),
                                    Part::Call { name, arguments } => found.push(call_json(found.len(), &name, &arguments)),
                                }
                            }
                            if !found.is_empty() {
                                let content = content.trim_end();
                                message[1].1 = if content.is_empty() { Json::Null } else { str(content) };
                                message.push(("tool_calls", Json::Arr(found)));
                                finish = "tool_calls";
                            }
                        }
                        obj(vec![("index", num(0)), ("message", obj(message)), ("finish_reason", str(finish))])
                    } else {
                        obj(vec![("index", num(0)), ("text", str(&text)), ("logprobs", Json::Null), ("finish_reason", str(finish))])
                    };
                    let resp = obj(vec![
                        ("id", str(&id)),
                        ("object", str(object)),
                        ("created", num(created as f64)),
                        ("model", str(model_id)),
                        ("choices", Json::Arr(vec![c])),
                        ("usage", usage(prompt_tokens, completion_tokens)),
                    ]);
                    return respond(&mut conn, 200, &resp);
                }
            }
        }
        return respond(&mut conn, 500, &error_body("generation ended without a result"));
    }

    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{CORS}Connection: close\r\n\r\n");
    conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    let mut first = true;
    if chat {
        // the role first, as OpenAI does
        sse(&mut conn, &chunk(vec![choice(Some(""), None, true)], None)).map_err(|e| e.to_string())?;
        first = false;
    }
    let mut n_calls = 0;
    // a failed write means the client is gone: dropping `events` stops the generation
    let mut send_parts = |conn: &mut TcpStream, parts: Vec<Part>, first: &mut bool| -> Result<(), String> {
        for part in parts {
            let c = match part {
                Part::Text(t) => choice(Some(&t), None, *first),
                Part::Call { name, arguments } => {
                    n_calls += 1;
                    obj(vec![("index", num(0)), ("delta", obj(vec![("tool_calls", Json::Arr(vec![call_json(n_calls - 1, &name, &arguments)]))])), ("finish_reason", Json::Null)])
                }
            };
            sse(conn, &chunk(vec![c], None)).map_err(|e| e.to_string())?;
            *first = false;
        }
        Ok(())
    };
    for ev in events {
        match ev {
            Event::Text(t) if t.is_empty() => {}
            Event::Text(t) => {
                let parts = match calls.as_mut() {
                    Some(tc) => tc.push(&t),
                    None => vec![Part::Text(t)],
                };
                send_parts(&mut conn, parts, &mut first)?;
            }
            Event::Error(_, e) => {
                sse(&mut conn, &error_body(&e).to_string()).map_err(|e| e.to_string())?;
                break;
            }
            Event::Done { finish, prompt_tokens, completion_tokens } => {
                if let Some(tc) = calls.as_mut() {
                    send_parts(&mut conn, tc.finish(), &mut first)?;
                }
                let finish = if calls.as_ref().is_some_and(|tc| tc.calls > 0) { "tool_calls" } else { finish };
                let last = if chat { choice(None, Some(finish), false) } else { choice(Some(""), Some(finish), false) };
                sse(&mut conn, &chunk(vec![last], None)).map_err(|e| e.to_string())?;
                if include_usage {
                    sse(&mut conn, &chunk(Vec::new(), Some(("usage", usage(prompt_tokens, completion_tokens))))).map_err(|e| e.to_string())?;
                }
                break;
            }
        }
    }
    sse(&mut conn, "[DONE]").map_err(|e| e.to_string())?;
    conn.write_all(b"0\r\n\r\n").map_err(|e| e.to_string())
}

fn rand_suffix() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));
    h.finish() as u32
}

/// The text of a message's content: a string, or OpenAI's list of parts (text parts only).
fn content_text(c: Option<&Json>) -> Result<String, String> {
    match c {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => Ok(parts.iter().filter(|p| p.get("type").and_then(Json::as_str) == Some("text")).filter_map(|p| p.get("text").and_then(Json::as_str)).collect::<Vec<_>>().join("")),
        Some(_) => Err("message content must be a string or a list of parts".into()),
    }
}

/// The tools the model may call in this request (none when `tool_choice` is "none").
fn request_tools(body: &Json) -> &[Json] {
    if body.get("tool_choice").and_then(Json::as_str) == Some("none") {
        return &[];
    }
    body.get("tools").and_then(Json::as_array).unwrap_or(&[])
}

/// The messages in the ChatML template, ending with an open assistant turn. Tools, tool calls and
/// tool results are written as Qwen's template writes them (`<tools>`, `<tool_call>`,
/// `<tool_response>`), and the reasoning of turns before the latest user question is left out.
/// `switch`: the model's template has a thinking switch; without one, thinking settings are ignored.
fn chat_prompt(body: &Json, default_system: Option<&str>, think_default: bool, switch: bool) -> Result<String, String> {
    let messages = body.get("messages").and_then(Json::as_array).ok_or("messages must be a list")?;
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let mut roles = Vec::with_capacity(messages.len());
    let mut contents = Vec::with_capacity(messages.len());
    for m in messages {
        roles.push(match m.get("role").and_then(Json::as_str) {
            Some("developer") => "system",
            Some(r @ ("system" | "user" | "assistant" | "tool")) => r,
            Some(other) => return Err(format!("unknown message role {other}")),
            None => return Err("a message has no role".into()),
        });
        contents.push(content_text(m.get("content"))?);
    }
    let tools = request_tools(body);

    let mut s = String::new();
    let system = if roles[0] == "system" { Some(contents[0].as_str()) } else if roles.contains(&"system") { None } else { default_system };
    if !tools.is_empty() {
        s.push_str("<|im_start|>system\n");
        if let Some(sys) = system {
            s.push_str(sys);
            s.push_str("\n\n");
        }
        s.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for t in tools {
            s.push('\n');
            s.push_str(&t.to_python_string());
        }
        s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    } else if let Some(sys) = system {
        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }

    // the latest real question (a user turn that is not a tool result)
    let is_tool_result = |c: &str| c.starts_with("<tool_response>") && c.ends_with("</tool_response>");
    let last_query = (0..messages.len()).rev().find(|&i| roles[i] == "user" && !is_tool_result(&contents[i])).unwrap_or(messages.len() - 1);
    for (i, m) in messages.iter().enumerate() {
        let content = contents[i].as_str();
        match roles[i] {
            "system" if i == 0 => {}
            r @ ("system" | "user") => s.push_str(&format!("<|im_start|>{r}\n{content}<|im_end|>\n")),
            "assistant" => {
                let (mut reasoning, mut content) = (String::new(), content.to_string());
                if let Some(r) = m.get("reasoning_content").and_then(Json::as_str) {
                    reasoning = r.to_string();
                } else if let Some((before, after)) = content.split_once("</think>") {
                    reasoning = before.trim_end_matches('\n').rsplit("<think>").next().unwrap_or("").trim_start_matches('\n').to_string();
                    content = after.trim_start_matches('\n').to_string();
                }
                s.push_str("<|im_start|>assistant\n");
                if i > last_query && (i == messages.len() - 1 || !reasoning.is_empty()) {
                    s.push_str(&format!("<think>\n{}\n</think>\n\n{}", reasoning.trim_matches('\n'), content.trim_start_matches('\n')));
                } else {
                    s.push_str(&content);
                }
                for (k, call) in m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]).iter().enumerate() {
                    if k > 0 || !content.is_empty() {
                        s.push('\n');
                    }
                    let f = call.get("function").unwrap_or(call);
                    let name = f.get("name").and_then(Json::as_str).unwrap_or("");
                    let args = match f.get("arguments") {
                        Some(Json::Str(a)) => a.clone(),
                        Some(a) => a.to_python_string(),
                        None => "{}".into(),
                    };
                    s.push_str(&format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"));
                }
                s.push_str("<|im_end|>\n");
            }
            _ => {
                // tool results: consecutive ones share one user turn
                if i == 0 || roles[i - 1] != "tool" {
                    s.push_str("<|im_start|>user");
                }
                s.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
                if i + 1 == messages.len() || roles[i + 1] != "tool" {
                    s.push_str("<|im_end|>\n");
                }
            }
        }
    }
    s.push_str("<|im_start|>assistant\n");
    // thinking off: chat_template_kwargs.enable_thinking = false, or OpenAI's reasoning_effort
    // "none"/"minimal" (sent by agents such as Hermes); otherwise think_default (serve --no-think)
    let thinking = body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")).and_then(Json::as_bool).or_else(|| {
        body.get("reasoning_effort").and_then(Json::as_str).map(|e| !matches!(e.to_ascii_lowercase().as_str(), "none" | "minimal"))
    });
    if switch && !thinking.unwrap_or(think_default) {
        s.push_str("<think>\n\n</think>\n\n");
    }
    Ok(s)
}

/// One piece of a reply, once tool calls are split out of the text.
#[derive(Debug, PartialEq)]
enum Part {
    Text(String),
    Call { name: String, arguments: String },
}

/// Splits `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks out of streamed text,
/// holding back what could still become a tag. Whitespace-only text after a call is dropped; a
/// block that is not valid JSON stays text.
#[derive(Default)]
struct ToolCalls {
    buf: String,
    in_call: bool,
    calls: usize,
}

impl ToolCalls {
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn push(&mut self, text: &str) -> Vec<Part> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(Self::CLOSE) else { break };
                let block: String = self.buf.drain(..end + Self::CLOSE.len()).collect();
                self.in_call = false;
                out.push(match self.call(&block[..end]) {
                    Part::Text(t) => Part::Text(t + Self::CLOSE),
                    call => call,
                });
            } else if let Some(at) = self.buf.find(Self::OPEN) {
                let before: String = self.buf.drain(..at + Self::OPEN.len()).collect();
                self.text(&before[..at], &mut out);
                self.in_call = true;
            } else {
                let ready = self.buf.len() - partial_suffix(&self.buf, Self::OPEN);
                // after a call, wait for something other than whitespace before sending text
                let ready = if self.calls > 0 && self.buf[..ready].trim().is_empty() { 0 } else { ready };
                let rest = self.buf.split_off(ready);
                let text = std::mem::replace(&mut self.buf, rest);
                self.text(&text, &mut out);
                break;
            }
        }
        out
    }

    /// What is left at the end of the reply.
    fn finish(&mut self) -> Vec<Part> {
        let rest = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        if self.in_call {
            // cut off before the closing tag: still a call if the JSON is complete
            self.in_call = false;
            out.push(self.call(&rest));
        } else {
            self.text(&rest, &mut out);
        }
        out
    }

    fn text(&self, t: &str, out: &mut Vec<Part>) {
        if !t.is_empty() && !(self.calls > 0 && t.trim().is_empty()) {
            out.push(Part::Text(t.to_string()));
        }
    }

    fn call(&mut self, inner: &str) -> Part {
        let parsed = Json::parse(inner.trim()).ok().and_then(|j| {
            let name = j.get("name")?.as_str()?.to_string();
            let arguments = match j.get("arguments") {
                Some(Json::Str(s)) => s.clone(),
                Some(a) => a.to_python_string(),
                None => "{}".into(),
            };
            Some(Part::Call { name, arguments })
        });
        match parsed {
            Some(p) => {
                self.calls += 1;
                p
            }
            None => Part::Text(format!("{}{inner}", Self::OPEN)),
        }
    }
}

fn completion_prompt(body: &Json) -> Result<String, String> {
    match body.get("prompt") {
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(a)) if a.len() == 1 => a[0].as_str().map(String::from).ok_or_else(|| "prompt must be a string".into()),
        Some(Json::Arr(_)) => Err("only one prompt per request".into()),
        _ => Err("prompt must be a string".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_string_prefixes() {
        assert_eq!(partial_suffix("hello wor", "world"), 3);
        assert_eq!(partial_suffix("hello", "world"), 0);
        assert_eq!(partial_suffix("ab", "abc"), 2);
    }

    #[test]
    fn chat_template() {
        let body = Json::parse(r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}"#).unwrap();
        assert_eq!(chat_prompt(&body, Some("Be brief."), true, true).unwrap(), "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn tool_template() {
        // what the Qwen3 chat template (Jinja, in the GGUF) gives for this conversation
        let body = Json::parse(
            r#"{"tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}],
            "messages":[{"role":"system","content":"Sys"},{"role":"user","content":"Time?"},
            {"role":"assistant","content":"<think>\nhm\n</think>\n\nLet me check.","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_time","arguments":"{\"tz\": \"UTC\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"12:00"},{"role":"tool","tool_call_id":"c2","content":"x"}]}"#,
        )
        .unwrap();
        let want = concat!(
            "<|im_start|>system\nSys\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            r#"{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}"#,
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\nTime?<|im_end|>\n",
            "<|im_start|>assistant\n<think>\nhm\n</think>\n\nLet me check.\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\n12:00\n</tool_response>\n<tool_response>\nx\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(chat_prompt(&body, None, true, true).unwrap(), want);
    }

    #[test]
    fn reasoning_effort_none_turns_thinking_off() {
        let off = Json::parse(r#"{"reasoning_effort":"none","messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&off, None, true, true).unwrap().ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let on = Json::parse(r#"{"reasoning_effort":"medium","messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&on, None, true, true).unwrap().ends_with("<|im_start|>assistant\n"));
        // an explicit chat_template_kwargs wins over reasoning_effort
        let kw = Json::parse(r#"{"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true},"messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&kw, None, true, true).unwrap().ends_with("<|im_start|>assistant\n"));
        // serve --no-think: off unless the request asks
        let plain = Json::parse(r#"{"messages":[{"role":"user","content":"Hi"}]}"#).unwrap();
        assert!(chat_prompt(&plain, None, false, true).unwrap().ends_with("<think>\n\n</think>\n\n"));
        assert!(chat_prompt(&on, None, false, true).unwrap().ends_with("<|im_start|>assistant\n"));
        // a model without a thinking switch (Qwen3-2507 Instruct, SmolLM2) never gets the empty block
        assert!(chat_prompt(&off, None, true, false).unwrap().ends_with("<|im_start|>assistant\n"));
        assert!(chat_prompt(&plain, None, false, false).unwrap().ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn tool_calls_split_from_stream() {
        let mut tc = ToolCalls::default();
        let mut parts = Vec::new();
        let text = "<think>\nok\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"g\", \"arguments\": {}}\n</tool_call>\n";
        // one character at a time, as tokens arrive
        for c in text.chars() {
            parts.extend(tc.push(&c.to_string()));
        }
        parts.extend(tc.finish());
        let text: String = parts.iter().filter_map(|p| if let Part::Text(t) = p { Some(t.as_str()) } else { None }).collect();
        assert_eq!(text, "<think>\nok\n</think>\n\nSure.\n");
        let calls: Vec<_> = parts.iter().filter(|p| matches!(p, Part::Call { .. })).collect();
        assert_eq!(calls, [&Part::Call { name: "f".into(), arguments: "{\"a\": 1}".into() }, &Part::Call { name: "g".into(), arguments: "{}".into() }]);

        let mut tc = ToolCalls::default();
        let mut parts = tc.push("a <tool_call>not json</tool_call> b");
        parts.extend(tc.finish());
        assert_eq!(parts.iter().map(|p| if let Part::Text(t) = p { t.as_str() } else { "" }).collect::<String>(), "a <tool_call>not json</tool_call> b");
    }
}
