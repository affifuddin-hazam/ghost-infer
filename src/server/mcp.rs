//! MCP servers, configured as in Claude Desktop, in %LOCALAPPDATA%\ghost-infer\mcp.json:
//!   {"mcpServers": {"name": {"command": "npx", "args": ["-y", "..."], "env": {"KEY": "..."}}}}
//! Each runs as a child process speaking newline-delimited JSON-RPC on its stdin and stdout. The client
//! speaks both eras of the protocol: it probes with server/discover in the stateless 2026-07-28 revision
//! (every request then carries its version in _meta), and falls back to the initialize handshake of
//! 2025-11-25 and earlier when the server does not answer that as a modern server would.
//! Their tools reach the model as mcp__<server>__<tool>, and each call asks the user first.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::json::{num, obj, str, Json};

const MODERN: &str = "2026-07-28";
const LEGACY: &str = "2025-11-25";

/// How long a probe waits before taking a silent server for a legacy one, and how long a server may take to
/// start (npx may download it first) or a tool call may run.
const PROBE: Duration = Duration::from_secs(15);
const START: Duration = Duration::from_secs(90);
const CALL: Duration = Duration::from_secs(300);

pub fn config_file() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d.join("mcp.json"))
}

/// The configured servers: name and its settings, in the file's order.
fn config() -> Vec<(String, Json)> {
    let text = config_file().ok().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    // a file saved by Notepad or PowerShell may start with a byte order mark
    match Json::parse(text.trim_start_matches('\u{feff}')).ok().and_then(|j| j.get("mcpServers").cloned()) {
        Some(Json::Obj(servers)) => servers,
        _ => Vec::new(),
    }
}

/// A JSON-RPC error from a server: its code and message (and data, which modern version errors use).
struct RpcError {
    code: i64,
    message: String,
    data: Json,
}

/// A running server process and the protocol it speaks.
struct Conn {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::Receiver<String>,
    next: u64,
    modern: Option<String>, // the modern version in use; None: a legacy session after initialize
}

impl Conn {
    fn send(&mut self, msg: &Json) -> Result<(), String> {
        writeln!(self.stdin, "{}", msg.to_string()).and_then(|_| self.stdin.flush()).map_err(|e| format!("the server stopped: {e}"))
    }

    /// One request; waits for its answer, replying to the few requests a legacy server may make meanwhile.
    fn request(&mut self, method: &str, params: Json, wait: Duration) -> Result<Json, Option<RpcError>> {
        self.next += 1;
        let id = self.next;
        let mut params = match params {
            Json::Obj(f) => f,
            _ => Vec::new(),
        };
        if let Some(v) = &self.modern {
            params.push(("_meta".into(), meta(v)));
        }
        let msg = obj(vec![("jsonrpc", str("2.0")), ("id", num(id as f64)), ("method", str(method)), ("params", Json::Obj(params))]);
        self.send(&msg).map_err(|e| Some(RpcError { code: 0, message: e, data: Json::Null }))?;
        let until = Instant::now() + wait;
        loop {
            let left = until.saturating_duration_since(Instant::now());
            let line = match self.lines.recv_timeout(left) {
                Ok(l) => l,
                Err(mpsc::RecvTimeoutError::Timeout) => return Err(None),
                Err(_) => return Err(Some(RpcError { code: 0, message: "the server stopped".into(), data: Json::Null })),
            };
            let Ok(m) = Json::parse(&line) else { continue };
            if let Some(method) = m.get("method").and_then(Json::as_str) {
                // a request from the server (legacy only): answer what is cheap, refuse the rest
                if let Some(rid) = m.get("id").cloned() {
                    let reply = match method {
                        "ping" => obj(vec![("jsonrpc", str("2.0")), ("id", rid), ("result", obj(vec![]))]),
                        "roots/list" => obj(vec![("jsonrpc", str("2.0")), ("id", rid), ("result", obj(vec![("roots", Json::Arr(Vec::new()))]))]),
                        _ => obj(vec![("jsonrpc", str("2.0")), ("id", rid), ("error", obj(vec![("code", num(-32601.0)), ("message", str("not supported by ghost-infer"))]))]),
                    };
                    let _ = self.send(&reply);
                }
                continue; // notifications (progress, logs) are not shown
            }
            if m.get("id").and_then(Json::as_f64) != Some(id as f64) {
                continue;
            }
            if let Some(e) = m.get("error") {
                return Err(Some(RpcError {
                    code: e.get("code").and_then(Json::as_f64).unwrap_or(0.0) as i64,
                    message: e.get("message").and_then(Json::as_str).unwrap_or("error").to_string(),
                    data: e.get("data").cloned().unwrap_or(Json::Null),
                }));
            }
            return Ok(m.get("result").cloned().unwrap_or(Json::Null));
        }
    }

    fn stop(&mut self) {
        // the whole tree: npx and cmd start node underneath
        let _ = Command::new("taskkill").args(["/T", "/F", "/PID", &self.child.id().to_string()]).stdout(Stdio::null()).stderr(Stdio::null()).status();
        let _ = self.child.kill();
    }
}

/// The per-request metadata of the modern protocol.
fn meta(version: &str) -> Json {
    obj(vec![
        ("io.modelcontextprotocol/protocolVersion", str(version)),
        ("io.modelcontextprotocol/clientInfo", client_info()),
        ("io.modelcontextprotocol/clientCapabilities", obj(vec![])),
    ])
}

fn client_info() -> Json {
    obj(vec![("name", str("ghost-infer")), ("version", str(env!("CARGO_PKG_VERSION")))])
}

/// A configured server: its state and, once started, its tools.
pub struct Server {
    name: String,
    status: String, // "starting", "ready", or why it is not
    tools: Vec<Json>,
    conn: Option<Conn>,
}

fn servers() -> &'static Mutex<Vec<Arc<Mutex<Server>>>> {
    static S: OnceLock<Mutex<Vec<Arc<Mutex<Server>>>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Vec::new()))
}

/// The server process, started without a window. A command Windows cannot start by itself (npx, uvx: they
/// are .cmd files) goes through cmd.
fn spawn(name: &str, cfg: &Json) -> Result<Conn, String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let command = cfg.get("command").and_then(Json::as_str).ok_or("no command")?;
    let args: Vec<String> = cfg.get("args").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
    let env: Vec<(String, String)> = match cfg.get("env") {
        Some(Json::Obj(f)) => f.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect(),
        _ => Vec::new(),
    };
    let log = config_file().ok().and_then(|p| p.parent().map(|d| d.join(format!("mcp-{}.log", clean(name))))).and_then(|p| std::fs::File::create(p).ok());
    let build = |direct: bool| {
        let mut c = if direct {
            let mut c = Command::new(command);
            c.args(&args);
            c
        } else {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(command).args(&args);
            c
        };
        c.envs(env.iter().cloned()).stdin(Stdio::piped()).stdout(Stdio::piped()).creation_flags(CREATE_NO_WINDOW);
        c.stderr(match log.as_ref().and_then(|f| f.try_clone().ok()) {
            Some(f) => Stdio::from(f),
            None => Stdio::null(),
        });
        c
    };
    let mut child = match build(true).spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => build(false).spawn().map_err(|e| format!("cannot start {command}: {e}"))?,
        Err(e) => return Err(format!("cannot start {command}: {e}")),
    };
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if !line.trim().is_empty() && tx.send(line).is_err() {
                break;
            }
        }
    });
    Ok(Conn { child, stdin, lines, next: 0, modern: None })
}

/// Starts a server: the process, the protocol era, and its tool list.
fn start(name: &str, cfg: &Json) -> Result<(Conn, Vec<Json>), String> {
    if cfg.get("url").is_some() || cfg.get("type").and_then(Json::as_str).is_some_and(|t| t != "stdio") {
        return Err("only servers started by a command (stdio) are supported for now".into());
    }
    let mut conn = spawn(name, cfg)?;
    // the probe: a modern server answers server/discover, or says which modern versions it speaks
    conn.modern = Some(MODERN.to_string());
    let probe = conn.request("server/discover", obj(vec![]), PROBE);
    match probe {
        Ok(r) => {
            let versions: Vec<String> = r.get("supportedVersions").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
            if !versions.is_empty() && !versions.iter().any(|v| v == MODERN) {
                conn.modern = versions.into_iter().find(|v| v.as_str() > LEGACY);
            }
        }
        Err(Some(e)) if e.code == -32022 => {
            let supported: Vec<String> = e.data.get("supported").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
            conn.modern = supported.into_iter().find(|v| v.as_str() > LEGACY);
            if conn.modern.is_none() {
                conn.stop();
                return Err(format!("it speaks no protocol version ghost-infer knows ({})", e.message));
            }
        }
        // code 0: our own error (the process ended), not the server's
        Err(Some(e)) if e.code == 0 => {
            conn.stop();
            return Err("it stopped as it started (see its log in %LOCALAPPDATA%\\ghost-infer)".into());
        }
        // anything else: a legacy server, which wants the initialize handshake
        Err(_) => {
            conn.modern = None;
            let init = obj(vec![("protocolVersion", str(LEGACY)), ("capabilities", obj(vec![])), ("clientInfo", client_info())]);
            if let Err(e) = conn.request("initialize", init, START) {
                conn.stop();
                return Err(match e {
                    Some(e) => format!("it refused to start: {}", e.message),
                    None => "it did not answer".into(),
                });
            }
            let _ = conn.send(&obj(vec![("jsonrpc", str("2.0")), ("method", str("notifications/initialized"))]));
        }
    }
    // its tools, page by page
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let params = match &cursor {
            Some(c) => obj(vec![("cursor", str(c))]),
            None => obj(vec![]),
        };
        let r = match conn.request("tools/list", params, START) {
            Ok(r) => r,
            Err(e) => {
                conn.stop();
                return Err(format!("it would not list its tools: {}", e.map_or("no answer".into(), |e| e.message)));
            }
        };
        tools.extend(r.get("tools").and_then(Json::as_array).unwrap_or(&[]).iter().cloned());
        cursor = r.get("nextCursor").and_then(Json::as_str).map(String::from);
        if cursor.is_none() || tools.len() > 500 {
            break;
        }
    }
    Ok((conn, tools))
}

/// A name as tool names allow it: letters, digits, _ and -.
fn clean(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect()
}

/// Starts every configured server (each on its own thread: some take a while), after stopping any running.
pub fn start_all() {
    stop_all();
    let list: Vec<Arc<Mutex<Server>>> = config()
        .into_iter()
        .filter(|(_, cfg)| cfg.get("disabled").and_then(Json::as_bool) != Some(true))
        .map(|(name, cfg)| {
            let s = Arc::new(Mutex::new(Server { name: name.clone(), status: "starting".into(), tools: Vec::new(), conn: None }));
            let s2 = s.clone();
            std::thread::spawn(move || {
                let result = start(&name, &cfg);
                let mut s = s2.lock().unwrap();
                match result {
                    Ok((conn, tools)) => {
                        s.status = "ready".into();
                        s.tools = tools;
                        s.conn = Some(conn);
                    }
                    Err(e) => s.status = e,
                }
            });
            s
        })
        .collect();
    *servers().lock().unwrap() = list;
}

/// Stops every server process (on reload, and before ghost-infer quits).
pub fn stop_all() {
    for s in servers().lock().unwrap().drain(..) {
        if let Ok(mut s) = s.lock() {
            if let Some(mut c) = s.conn.take() {
                c.stop();
            }
        }
    }
}

/// The servers for the clients: name, state and tool names.
pub fn status() -> Json {
    let list = servers().lock().unwrap().clone();
    Json::Arr(
        list.iter()
            .map(|s| match s.try_lock() {
                Ok(s) => obj(vec![
                    ("name", str(&s.name)),
                    ("status", str(&s.status)),
                    ("tools", Json::Arr(s.tools.iter().filter_map(|t| t.get("name").cloned()).collect())),
                ]),
                Err(_) => obj(vec![("name", str("")), ("status", str("busy")), ("tools", Json::Arr(Vec::new()))]),
            })
            .collect(),
    )
}

/// Every ready server's tools, as the chat template's tools (OpenAI form).
pub fn tools() -> Vec<Json> {
    let list = servers().lock().unwrap().clone();
    let mut out = Vec::new();
    for s in list {
        let Ok(s) = s.try_lock() else { continue };
        if s.conn.is_none() {
            continue;
        }
        for t in &s.tools {
            let Some(name) = t.get("name").and_then(Json::as_str) else { continue };
            let desc = t.get("description").and_then(Json::as_str).unwrap_or("");
            let params = t.get("inputSchema").cloned().unwrap_or_else(|| obj(vec![("type", str("object")), ("properties", obj(vec![]))]));
            out.push(obj(vec![
                ("type", str("function")),
                (
                    "function",
                    obj(vec![
                        ("name", str(&format!("mcp__{}__{}", clean(&s.name), clean(name)))),
                        ("description", str(&format!("[{} MCP] {}", s.name, crate::tools::clip(desc, 1000)))),
                        ("parameters", params),
                    ]),
                ),
            ]));
        }
    }
    out
}

/// The full name (mcp__<server>__<tool>) of a running server's tool called `tool`, if one has it.
pub fn find_tool(tool: &str) -> Option<String> {
    let list = servers().lock().unwrap().clone();
    list.iter().find_map(|s| {
        let s = s.try_lock().ok()?;
        let has = s.conn.is_some() && s.tools.iter().any(|t| t.get("name").and_then(Json::as_str) == Some(tool));
        has.then(|| format!("mcp__{}__{}", clean(&s.name), clean(tool)))
    })
}

/// Runs an mcp__<server>__<tool> call; its result for the model.
pub fn call(full: &str, args: &Json) -> String {
    let Some(rest) = full.strip_prefix("mcp__") else { return format!("Error: {full} is not an MCP tool") };
    let list = servers().lock().unwrap().clone();
    let Some(server) = list.iter().find(|s| s.lock().is_ok_and(|s| rest.starts_with(&format!("{}__", clean(&s.name))))).cloned() else {
        return format!("Error: no MCP server runs {full}");
    };
    let mut s = server.lock().unwrap();
    let tool = rest[clean(&s.name).len() + 2..].to_string();
    let Some(real) = s.tools.iter().filter_map(|t| t.get("name").and_then(Json::as_str)).find(|n| clean(n) == tool).map(String::from) else {
        return format!("Error: the {} MCP server has no tool {tool}", s.name);
    };
    let Some(conn) = s.conn.as_mut() else { return format!("Error: the {} MCP server is not running ({})", s.name, s.status) };
    let arguments = if matches!(args, Json::Obj(_)) { args.clone() } else { obj(vec![]) };
    let r = match conn.request("tools/call", obj(vec![("name", str(&real)), ("arguments", arguments)]), CALL) {
        Ok(r) => r,
        Err(Some(e)) => return format!("Error: {}", e.message),
        Err(None) => return "Error: the MCP server did not answer in time".into(),
    };
    if r.get("resultType").and_then(Json::as_str) == Some("input_required") {
        return "Error: this tool asked for more input, which ghost-infer cannot give yet".into();
    }
    let mut parts = Vec::new();
    for c in r.get("content").and_then(Json::as_array).unwrap_or(&[]) {
        match c.get("type").and_then(Json::as_str) {
            Some("text") => parts.push(c.get("text").and_then(Json::as_str).unwrap_or("").to_string()),
            Some("resource") => {
                let res = c.get("resource");
                parts.push(res.and_then(|r| r.get("text")).and_then(Json::as_str).map(String::from).unwrap_or_else(|| format!("[resource {}]", res.and_then(|r| r.get("uri")).and_then(Json::as_str).unwrap_or(""))));
            }
            Some("resource_link") => parts.push(format!("[link {}]", c.get("uri").and_then(Json::as_str).unwrap_or(""))),
            Some(other) => parts.push(format!("[{other}]")),
            None => {}
        }
    }
    if parts.is_empty() {
        if let Some(sc) = r.get("structuredContent") {
            parts.push(sc.to_string());
        }
    }
    let text = crate::tools::clip(&parts.join("\n"), crate::tools::MAX_RESULT);
    if r.get("isError").and_then(Json::as_bool) == Some(true) { format!("Error: {text}") } else { text }
}

/// Adds the servers configured in Claude Desktop (%APPDATA%\Claude\claude_desktop_config.json) to mcp.json;
/// a server already here keeps its settings. The names added.
pub fn import_claude() -> Result<Vec<String>, String> {
    let src = std::env::var_os("APPDATA").map(|d| PathBuf::from(d).join("Claude").join("claude_desktop_config.json")).ok_or("APPDATA is not set")?;
    let text = std::fs::read_to_string(&src).map_err(|_| "Claude Desktop's settings were not found on this computer".to_string())?;
    let theirs = match Json::parse(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?.get("mcpServers").cloned() {
        Some(Json::Obj(f)) => f,
        _ => Vec::new(),
    };
    let mut ours = config();
    let mut added = Vec::new();
    for (name, cfg) in theirs {
        if !ours.iter().any(|(n, _)| *n == name) {
            added.push(name.clone());
            ours.push((name, cfg));
        }
    }
    save(&ours)?;
    Ok(added)
}

fn save(list: &[(String, Json)]) -> Result<(), String> {
    let j = obj(vec![("mcpServers", Json::Obj(list.to_vec()))]);
    std::fs::write(config_file()?, j.to_string()).map_err(|e| e.to_string())
}

/// mcp.json, made with an empty list if it does not exist yet (for the clients' "edit" button). Its path.
pub fn ensure_config() -> Result<String, String> {
    let p = config_file()?;
    if !p.exists() {
        save(&[])?;
    }
    Ok(p.display().to_string())
}
