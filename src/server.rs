//! OpenAI-compatible HTTP server: /v1/chat/completions, /v1/completions (both with streaming over
//! server-sent events), /v1/models. A small HTTP/1.1 implementation on std::net, one connection thread
//! each; generation runs on the main thread, one request at a time. The cache keeps the tokens of the
//! last conversation, so a request that repeats it (a chat client resending the history) only reads
//! the new part.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
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

/// A generation request handed from a connection thread to the model thread.
struct Job {
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

pub fn serve(o: &Opts, host: &str, port: u16) -> Result<(), String> {
    let t = Instant::now();
    let g = crate::gguf::Gguf::open(&o.model)?;
    let mut engine = crate::load_with_draft(&g, o.gpu, o.ctx, o.draft.as_ref(), o.draft_k)?;
    let fmt = ChatFormat::detect(&g).ok();
    let model_id = o.model.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    eprintln!(
        "[loaded {} on {} in {:.2}s{}]",
        model_id,
        engine.backend.describe(),
        t.elapsed().as_secs_f64(),
        if engine.draft.is_some() { ", with a draft model for greedy requests" } else { "" }
    );
    eprintln!("listening on http://{host}:{port}/v1 (OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models)");
    if fmt.is_none() {
        eprintln!("[note: no ChatML template in this model, so /v1/chat/completions is off]");
    }

    let (jobs, rx) = mpsc::channel::<Job>();
    let defaults = (o.temp, o.top_k, o.top_p, o.max_new);
    let id = model_id.clone();
    let chat_ok = fmt.is_some();
    let default_system = fmt.as_ref().and_then(|f| f.default_system.clone());
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let (jobs, id, sys) = (jobs.clone(), id.clone(), default_system.clone());
            std::thread::spawn(move || {
                if let Err(e) = handle(conn, &jobs, &id, chat_ok, sys.as_deref(), defaults) {
                    eprintln!("[connection: {e}]");
                }
            });
        }
    });

    // the model thread: the engine (GPU state included) never leaves it
    for job in rx {
        run(&mut engine, job);
    }
    Ok(())
}

fn run(engine: &mut Engine, job: Job) {
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
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n",
        status_text(code),
        text.len()
    );
    conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(text.as_bytes())).map_err(|e| e.to_string())
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

fn handle(mut conn: TcpStream, jobs: &mpsc::Sender<Job>, model_id: &str, chat_ok: bool, default_system: Option<&str>, defaults: (f32, usize, f32, usize)) -> Result<(), String> {
    let req = read_request(&mut conn)?;
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    match (req.method.as_str(), req.path.trim_end_matches('/')) {
        ("OPTIONS", _) => {
            let head = format!("HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\nConnection: close\r\n\r\n");
            return conn.write_all(head.as_bytes()).map_err(|e| e.to_string());
        }
        ("GET", "" | "/health") => return respond(&mut conn, 200, &obj(vec![("status", str("ok"))])),
        ("GET", "/v1/models") => {
            let m = obj(vec![("id", str(model_id)), ("object", str("model")), ("created", num(created as f64)), ("owned_by", str("betterllama"))]);
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
    let prompt = if chat { chat_prompt(&body, default_system) } else { completion_prompt(&body) };
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
    jobs.send(job).map_err(|_| "the model thread stopped")?;

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
                        obj(vec![("index", num(0)), ("message", obj(vec![("role", str("assistant")), ("content", str(&text))])), ("finish_reason", str(finish))])
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
    for ev in events {
        match ev {
            Event::Text(t) if t.is_empty() => {}
            Event::Text(t) => {
                // a failed write means the client is gone: dropping `events` stops the generation
                sse(&mut conn, &chunk(vec![choice(Some(&t), None, first)], None)).map_err(|e| e.to_string())?;
                first = false;
            }
            Event::Error(_, e) => {
                sse(&mut conn, &error_body(&e).to_string()).map_err(|e| e.to_string())?;
                break;
            }
            Event::Done { finish, prompt_tokens, completion_tokens } => {
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

/// The messages in the ChatML template, ending with an open assistant turn.
fn chat_prompt(body: &Json, default_system: Option<&str>) -> Result<String, String> {
    let messages = body.get("messages").and_then(Json::as_array).ok_or("messages must be a list")?;
    if messages.is_empty() {
        return Err("messages is empty".into());
    }
    let mut s = String::new();
    let has_system = messages.iter().any(|m| matches!(m.get("role").and_then(Json::as_str), Some("system" | "developer")));
    if let (false, Some(sys)) = (has_system, default_system) {
        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }
    for m in messages {
        let role = match m.get("role").and_then(Json::as_str) {
            Some("developer") => "system",
            Some(r @ ("system" | "user" | "assistant" | "tool")) => r,
            Some(other) => return Err(format!("unknown message role {other}")),
            None => return Err("a message has no role".into()),
        };
        s.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", content_text(m.get("content"))?));
    }
    s.push_str("<|im_start|>assistant\n");
    Ok(s)
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
        assert_eq!(chat_prompt(&body, Some("Be brief.")).unwrap(), "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }
}
