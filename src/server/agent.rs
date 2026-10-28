//! The agent loop on the server, for the native app and any other client: a chat's messages go to the model
//! with the tools it may use, its tool calls run (after the user's OK where needed) and their results go
//! back, round after round. Every step streams to the client as an event, and the chat is saved as it goes
//! in %LOCALAPPDATA%\ghost-infer\chats, so every client sees the same chats.
//!
//! Events, one JSON object per server-sent event:
//!   {"type":"chat","id","title","checkpoint"}     the chat this run belongs to (new chats get an id here), and
//!                                                 the checkpoint its file changes go under (POST /api/tools/undo)
//!   {"type":"text","text"}                        reply text as it streams (a thinking reply starts with <think>)
//!   {"type":"tool_call","id","name","arguments"}  the model called a tool
//!   {"type":"approval","id","name","arguments","group"}  waiting for POST /api/agent/approve {"id","decision"}
//!   {"type":"tool_result","id","name","result","denied"}
//!   {"type":"context","prompt_tokens","completion_tokens","context_length"}  after each round
//!   {"type":"done","finish"} or {"type":"error","message"}

use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{chat_prompt, chat_prompt_glm, sse, Event, Job, Kind, Loaded, Part, Shared, ToolCalls, Work};
use crate::json::{num, obj, str, Json};

/// Model turns one reply may take (each tool round is one).
const MAX_ROUNDS: usize = 40;

// ---- tools

/// The tools, by group: what the client turns on. The system group (get_system_info) is always there.
const TOOLS: &str = r#"{
 "system": [
  {"type":"function","function":{"name":"get_system_info","description":"Get the specs of the user's computer: CPU, RAM, GPU and VRAM, operating system, free disk space, and what is running on it right now.","parameters":{"type":"object","properties":{},"required":[]}}}
 ],
 "files": [
  {"type":"function","function":{"name":"read_file","description":"Read a text file, with line numbers. For a big file, read part of it with offset and limit.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"offset":{"type":"integer","description":"First line to read, from 1"},"limit":{"type":"integer","description":"How many lines (default 300)"}},"required":["path"]}}},
  {"type":"function","function":{"name":"list_files","description":"List a folder, or with pattern every file under it matching a glob such as **/*.py.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Folder (default: the first allowed folder)"},"pattern":{"type":"string","description":"Glob, e.g. *.md or src/**/*.rs"}},"required":[]}}},
  {"type":"function","function":{"name":"search_files","description":"Search inside files for a regular expression. Returns path:line:text.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Regular expression"},"path":{"type":"string","description":"Folder or file to search"},"glob":{"type":"string","description":"Only files matching this, e.g. *.rs"},"ignore_case":{"type":"boolean","description":"Ignore upper/lower case"}},"required":["pattern"]}}}
 ],
 "edit": [
  {"type":"function","function":{"name":"write_file","description":"Create a file, or replace a file's whole content.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"The complete content"}},"required":["path","content"]}}},
  {"type":"function","function":{"name":"edit_file","description":"Replace exact text in a file. Read the file first; old_string must match exactly once, so include enough surrounding lines.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"old_string":{"type":"string","description":"The exact text to replace"},"new_string":{"type":"string","description":"The text to put in its place"},"replace_all":{"type":"boolean","description":"Replace every match"}},"required":["path","old_string","new_string"]}}}
 ],
 "cmd": [
  {"type":"function","function":{"name":"run_command","description":"Run a Windows PowerShell command and get its output (stops after 60 s unless timeout is set).","parameters":{"type":"object","properties":{"command":{"type":"string","description":"PowerShell command"},"cwd":{"type":"string","description":"Folder to run it in"},"timeout":{"type":"integer","description":"Seconds, up to 600"}},"required":["command"]}}},
  {"type":"function","function":{"name":"start_process","description":"Start a long-running PowerShell command (a server, a build) in the background. Returns its id.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"PowerShell command"},"cwd":{"type":"string","description":"Folder to run it in"}},"required":["command"]}}},
  {"type":"function","function":{"name":"check_process","description":"A background process's state and the output it printed since the last check.","parameters":{"type":"object","properties":{"id":{"type":"integer","description":"Process id"}},"required":["id"]}}},
  {"type":"function","function":{"name":"stop_process","description":"Stop a background process.","parameters":{"type":"object","properties":{"id":{"type":"integer","description":"Process id"}},"required":["id"]}}}
 ],
 "web": [
  {"type":"function","function":{"name":"web_search","description":"Search the web. Returns titles, addresses and snippets.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"What to search for"}},"required":["query"]}}},
  {"type":"function","function":{"name":"fetch_url","description":"Read a web page as text.","parameters":{"type":"object","properties":{"url":{"type":"string","description":"http(s) address"}},"required":["url"]}}}
 ]
}"#;

fn tool_groups() -> &'static Json {
    static T: OnceLock<Json> = OnceLock::new();
    T.get_or_init(|| Json::parse(TOOLS).expect("the built-in tool list is valid JSON"))
}

/// The tools of these groups (and the system one), as the chat template's "tools".
fn tools_for(groups: &[String]) -> Vec<Json> {
    let all = tool_groups();
    std::iter::once("system")
        .chain(groups.iter().map(String::as_str))
        .filter_map(|g| all.get(g).and_then(Json::as_array))
        .flatten()
        .cloned()
        .collect()
}

/// The group whose calls need the user's OK, if any.
fn approval_group(name: &str) -> Option<&'static str> {
    match name {
        "write_file" | "edit_file" => Some("edit"),
        "run_command" | "start_process" => Some("cmd"),
        "web_search" | "fetch_url" => Some("web"),
        _ => None,
    }
}

/// Told to the model when tools are on: where it may work, and how this computer runs commands.
fn tool_prompt(folders: &[PathBuf]) -> String {
    let list: Vec<String> = folders.iter().map(|f| f.display().to_string()).collect();
    let where_ = if list.is_empty() {
        "No folders are allowed yet, so file tools will fail; if you need files, ask the user to add a folder in Settings.".to_string()
    } else {
        format!("You may use files in {} (relative paths start in {}).", list.join(", "), list[0])
    };
    format!("You can use tools to do things on the user's computer. {where_} It runs Windows; commands run in Windows PowerShell. Read a file before you edit it. Use tools until the task is done, then answer briefly.")
}

// ---- approvals: the run waits on a channel the approve endpoint answers

fn pending() -> &'static Mutex<HashMap<String, mpsc::Sender<String>>> {
    static P: OnceLock<Mutex<HashMap<String, mpsc::Sender<String>>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

/// The user's answer to approval `id`: "once", "always" or "deny". False if nothing waits for it.
pub fn answer(id: &str, decision: &str) -> bool {
    match pending().lock().unwrap().remove(id) {
        Some(tx) => tx.send(decision.to_string()).is_ok(),
        None => false,
    }
}

// ---- the chat store

fn chats_dir() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("chats")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

fn chat_file(id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!("{id} is not a chat id"));
    }
    Ok(chats_dir()?.join(format!("{id}.json")))
}

fn now_ms() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0)
}

/// Every saved chat, newest first: id, title, when it last changed.
pub fn list_chats() -> Result<Json, String> {
    let mut chats: Vec<(f64, Json)> = std::fs::read_dir(chats_dir()?)
        .map_err(|e| e.to_string())?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| Json::parse(&std::fs::read_to_string(e.path()).ok()?).ok())
        .map(|c| {
            let at = c.get("updated").and_then(Json::as_f64).unwrap_or(0.0);
            let summary = obj(vec![("id", c.get("id").cloned().unwrap_or(Json::Null)), ("title", c.get("title").cloned().unwrap_or(Json::Null)), ("updated", num(at))]);
            (at, summary)
        })
        .collect();
    chats.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(Json::Arr(chats.into_iter().map(|(_, c)| c).collect()))
}

pub fn load_chat(id: &str) -> Result<Json, String> {
    let path = chat_file(id)?;
    let text = std::fs::read_to_string(&path).map_err(|_| format!("no chat {id}"))?;
    Json::parse(&text)
}

pub fn delete_chat(id: &str) -> Result<(), String> {
    std::fs::remove_file(chat_file(id)?).map_err(|_| format!("no chat {id}"))
}

/// A chat in memory: its id, title and messages (OpenAI-style, with a few extra fields the clients read).
struct Chat {
    id: String,
    title: String,
    created: f64,
    messages: Vec<Json>,
}

impl Chat {
    fn new() -> Chat {
        let t = now_ms();
        // time, then a little randomness so two chats made in the same millisecond differ
        let rnd = {
            use std::hash::{BuildHasher, Hasher};
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(t as u64);
            h.finish() & 0xffff
        };
        Chat { id: format!("{:x}-{rnd:04x}", t as u64), title: String::new(), created: t, messages: Vec::new() }
    }

    fn load(id: &str) -> Result<Chat, String> {
        let j = load_chat(id)?;
        Ok(Chat {
            id: id.to_string(),
            title: j.get("title").and_then(Json::as_str).unwrap_or("").to_string(),
            created: j.get("created").and_then(Json::as_f64).unwrap_or(0.0),
            messages: j.get("messages").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default(),
        })
    }

    fn save(&self) -> Result<(), String> {
        let j = obj(vec![
            ("id", str(&self.id)),
            ("title", str(&self.title)),
            ("created", num(self.created)),
            ("updated", num(now_ms())),
            ("messages", Json::Arr(self.messages.clone())),
        ]);
        let path = chat_file(&self.id)?;
        // written whole to a side file first, so a crash never leaves half a chat
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, j.to_string()).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
    }
}

// ---- the run

/// What the client asks for: the chat (none: a new one), a new user message (none: answer the last one
/// again), and the reply's settings.
struct Request {
    chat: Option<String>,
    message: Option<String>,
    system: String,
    thinking: bool,
    temperature: f32,
    max_tokens: usize,
    groups: Vec<String>,
}

impl Request {
    fn parse(b: &Json, shared: &Shared) -> Request {
        let f = |k: &str| b.get(k).and_then(Json::as_f64);
        Request {
            chat: b.get("chat").and_then(Json::as_str).map(String::from),
            message: b.get("message").and_then(Json::as_str).map(String::from),
            system: b.get("system").and_then(Json::as_str).unwrap_or("").to_string(),
            thinking: b.get("thinking").and_then(Json::as_bool).unwrap_or(shared.think_default),
            temperature: f("temperature").map(|t| t as f32).unwrap_or(shared.defaults.0),
            max_tokens: f("max_tokens").map(|n| n.max(1.0) as usize).unwrap_or(shared.defaults.3),
            groups: b.get("tools").and_then(Json::as_array).map(|a| a.iter().filter_map(Json::as_str).map(String::from).collect()).unwrap_or_default(),
        }
    }
}

/// One event to the client; Err when it has gone away.
fn emit(conn: &mut TcpStream, ev: Json) -> Result<(), String> {
    sse(conn, &ev.to_string()).map_err(|_| "the client went away".to_string())
}

/// A comment line, to notice a client that went away while the run waits on the user.
fn ping(conn: &mut TcpStream) -> bool {
    let c = ": waiting\n\n";
    write!(conn, "{:x}\r\n{c}\r\n", c.len()).and_then(|_| conn.flush()).is_ok()
}

/// POST /api/agent: runs the agent for one reply, streaming events (see the module comment).
pub fn run(conn: &mut TcpStream, body: &Json, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(), String> {
    let req = Request::parse(body, shared);
    let mut chat = match &req.chat {
        Some(id) => Chat::load(id)?,
        None => Chat::new(),
    };
    match &req.message {
        Some(m) => {
            if chat.title.is_empty() {
                let line = m.trim().lines().next().unwrap_or("");
                chat.title = if line.chars().count() > 60 { format!("{}…", line.chars().take(57).collect::<String>().trim_end()) } else { line.to_string() };
            }
            chat.messages.push(obj(vec![("role", str("user")), ("content", str(m))]));
        }
        // answer again: drop the reply after the last user message
        None => {
            let last_user = chat.messages.iter().rposition(|m| m.get("role").and_then(Json::as_str) == Some("user")).ok_or("this chat has no message to answer")?;
            chat.messages.truncate(last_user + 1);
        }
    }
    chat.save()?;

    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    let checkpoint = format!("{}-{}", chat.id, now_ms() as u64); // this reply's file changes, for undo
    emit(conn, obj(vec![("type", str("chat")), ("id", str(&chat.id)), ("title", str(&chat.title)), ("checkpoint", str(&checkpoint))]))?;
    let result = rounds(conn, &req, &mut chat, &checkpoint, work, shared, loaded);
    chat.save()?;
    let end = match result {
        Ok(finish) => obj(vec![("type", str("done")), ("finish", str(finish))]),
        Err(e) => obj(vec![("type", str("error")), ("message", str(&e))]),
    };
    let _ = emit(conn, end);
    let _ = sse(conn, "[DONE]");
    let _ = conn.write_all(b"0\r\n\r\n");
    Ok(())
}

/// The model's rounds for one reply; Ok is how it finished ("stop", "length", or "rounds" at MAX_ROUNDS).
fn rounds(conn: &mut TcpStream, req: &Request, chat: &mut Chat, checkpoint: &str, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<&'static str, String> {
    let reply_start = chat.messages.len();
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let mut changed = false;
    for round in 0..MAX_ROUNDS {
        // the prompt: the client's system text plus where tools may work, then the whole chat
        let folders = shared.harness.lock().unwrap().folders.clone();
        let system = [req.system.trim().to_string(), if tools.is_empty() { String::new() } else { tool_prompt(&folders) }].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(obj(vec![("role", str("system")), ("content", str(&system))]));
        }
        messages.extend(chat.messages.iter().cloned());
        let mut fields = vec![("messages", Json::Arr(messages)), ("chat_template_kwargs", obj(vec![("enable_thinking", Json::Bool(req.thinking))]))];
        if !tools.is_empty() {
            fields.push(("tools", Json::Arr(tools.clone())));
        }
        let body = obj(fields);
        let prompt = if loaded.glm { chat_prompt_glm(&body, shared.think_default, loaded.thinking) } else { chat_prompt(&body, loaded.default_system.as_deref(), shared.think_default, loaded.thinking) }?;
        let lead = if loaded.glm && prompt.ends_with("<think>") { "<think>" } else { "" };

        let (tx, events) = mpsc::channel();
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(req.max_tokens), temperature: req.temperature, top_k: shared.defaults.1, top_p: shared.defaults.2, seed, stop: Vec::new(), lead, events: tx };
        work.send(Work::Job(job)).map_err(|_| "the model thread stopped")?;

        // stream it: text to the client as it comes, tool calls split out
        let mut split = ToolCalls::default();
        let (mut text, mut calls) = (String::new(), Vec::new());
        let mut finish = "stop";
        let mut usage = (0usize, 0usize);
        let take = |parts: Vec<Part>, conn: &mut TcpStream, text: &mut String, calls: &mut Vec<(String, String, String)>| -> Result<(), String> {
            for part in parts {
                match part {
                    Part::Text(t) => {
                        text.push_str(&t);
                        emit(conn, obj(vec![("type", str("text")), ("text", str(&t))]))?;
                    }
                    Part::Call { name, arguments } => {
                        let id = format!("call_{}_{}", round, calls.len());
                        emit(conn, obj(vec![("type", str("tool_call")), ("id", str(&id)), ("name", str(&name)), ("arguments", str(&arguments))]))?;
                        calls.push((id, name, arguments));
                    }
                }
            }
            Ok(())
        };
        for ev in events {
            match ev {
                // the client going away drops `events`, which stops the generation
                Event::Text(t) => take(if tools.is_empty() { vec![Part::Text(t)] } else { split.push(&t) }, conn, &mut text, &mut calls)?,
                Event::Done { finish: f, prompt_tokens, completion_tokens } => {
                    finish = f;
                    usage = (prompt_tokens, completion_tokens);
                    break;
                }
                Event::Error(_, e) => return Err(e),
            }
        }
        take(split.finish(), conn, &mut text, &mut calls)?;
        emit(conn, obj(vec![("type", str("context")), ("prompt_tokens", num(usage.0 as f64)), ("completion_tokens", num(usage.1 as f64)), ("context_length", num(loaded.ctx as f64))]))?;

        // the round as the chat keeps it: what the model wrote, and its calls in OpenAI's form
        let mut msg = vec![("role", str("assistant")), ("content", str(text.trim())), ("checkpoint", str(checkpoint))];
        if !calls.is_empty() {
            let tc = calls.iter().map(|(id, name, args)| obj(vec![("id", str(id)), ("type", str("function")), ("function", obj(vec![("name", str(name)), ("arguments", str(args))]))])).collect();
            msg.push(("tool_calls", Json::Arr(tc)));
        }
        chat.messages.push(obj(msg));
        if calls.is_empty() {
            mark_changed(chat, reply_start, changed);
            return Ok(if finish == "length" { "length" } else { "stop" });
        }

        for (id, name, arguments) in calls {
            let (result, denied) = call(conn, &id, &name, &arguments, chat, reply_start, checkpoint, shared, loaded)?;
            if !denied && approval_group(&name) == Some("edit") && !result.starts_with("Error") {
                changed = true;
            }
            emit(conn, obj(vec![("type", str("tool_result")), ("id", str(&id)), ("name", str(&name)), ("result", str(&result)), ("denied", Json::Bool(denied))]))?;
            let mut m = vec![("role", str("tool")), ("tool_call_id", str(&id)), ("name", str(&name)), ("content", str(&result))];
            if denied {
                m.push(("denied", Json::Bool(true)));
            }
            chat.messages.push(obj(m));
        }
        mark_changed(chat, reply_start, changed);
        chat.save()?;
    }
    Ok("rounds")
}

/// Notes on the reply's last assistant message whether it changed files (the clients offer undo then).
fn mark_changed(chat: &mut Chat, reply_start: usize, changed: bool) {
    if !changed {
        return;
    }
    let last = chat.messages[reply_start..].iter().rposition(|m| m.get("role").and_then(Json::as_str) == Some("assistant"));
    for (i, m) in chat.messages[reply_start..].iter_mut().enumerate() {
        if let Json::Obj(fields) = m {
            fields.retain(|(k, _)| k != "changed");
            if Some(i) == last {
                fields.push(("changed".into(), Json::Bool(true)));
            }
        }
    }
}

/// One tool call: refused if it repeats itself, put to the user if its group needs an OK, then run.
/// Returns its result for the model and whether the user said no.
#[allow(clippy::too_many_arguments)]
fn call(conn: &mut TcpStream, id: &str, name: &str, arguments: &str, chat: &Chat, reply_start: usize, checkpoint: &str, shared: &Shared, loaded: &Loaded) -> Result<(String, bool), String> {
    // the same call a third time in one reply: a small model going round in circles
    let same = chat.messages[reply_start..]
        .iter()
        .filter_map(|m| m.get("tool_calls").and_then(Json::as_array))
        .flatten()
        .filter(|c| c.get("function").is_some_and(|f| f.get("name").and_then(Json::as_str) == Some(name) && f.get("arguments").and_then(Json::as_str) == Some(arguments)))
        .count();
    if same > 2 {
        return Ok(("You already made this exact call twice; its result is above. Do something different, or answer with what you have.".into(), false));
    }
    if let Some(group) = approval_group(name) {
        if !shared.harness.lock().unwrap().always.iter().any(|g| g == group) {
            let (tx, rx) = mpsc::channel();
            pending().lock().unwrap().insert(id.to_string(), tx);
            emit(conn, obj(vec![("type", str("approval")), ("id", str(id)), ("name", str(name)), ("arguments", str(arguments)), ("group", str(group))]))?;
            let decision = loop {
                match rx.recv_timeout(Duration::from_secs(2)) {
                    Ok(d) => break d,
                    Err(mpsc::RecvTimeoutError::Timeout) if ping(conn) => continue,
                    _ => {
                        pending().lock().unwrap().remove(id);
                        return Err("the client went away".into());
                    }
                }
            };
            match decision.as_str() {
                "always" => {
                    let mut s = shared.harness.lock().unwrap();
                    if !s.always.iter().any(|g| g == group) {
                        s.always.push(group.to_string());
                        let _ = s.save();
                    }
                }
                "once" => {}
                _ => return Ok(("The user declined this action. Do not try it again. If there is no other way to proceed, explain the situation clearly and stop.".into(), true)),
            }
        }
    }
    if name == "get_system_info" {
        let first = vec![
            ("inference_engine", str(concat!("ghost-infer ", env!("CARGO_PKG_VERSION"), ", running locally on this computer"))),
            ("model_loaded", str(&loaded.id)),
            ("model_runs_on", str(&loaded.backend)),
            ("context_length", num(loaded.ctx as f64)),
        ];
        return Ok((crate::sysinfo::report(first).to_string(), false));
    }
    let args = if arguments.trim().is_empty() { Json::Obj(Vec::new()) } else { Json::parse(arguments).unwrap_or(Json::Null) };
    if args == Json::Null {
        return Ok((format!("Error: invalid arguments for {name}: not valid JSON. Arguments given: {}. Fix them and call the tool again.", crate::tools::clip(arguments, 300)), false));
    }
    // a copy of the settings: a command may run for minutes, and the settings must stay usable meanwhile
    let settings = shared.harness.lock().unwrap().clone();
    Ok((crate::tools::run(name, &args, &settings, checkpoint).unwrap_or_else(|e| format!("Error: {e}")), false))
}
