//! Plugins: tools ghost writes for itself. A plugin is a folder in %LOCALAPPDATA%\ghost-infer\plugins\<name>
//! holding a plugin.json and a small MCP server (a Python or Node script):
//!   {"name": "weather", "description": "...", "version": "0.1.0", "command": "python", "args": ["server.py"],
//!    "env": {}, "enabled": true}
//! The enabled ones run as MCP servers next to those in mcp.json (mcp.rs), so their tools reach the model as
//! mcp__<plugin>__<tool> and each call asks the user, as any MCP tool's does. Running out of ghost-infer's
//! process, a broken plugin cannot take the server down with it, and can be stopped.
//! The plugin_manager tool lets the model write, install, switch and remove them during a chat; every change
//! needs the user's OK, and saving starts the plugin and lists its tools, so the model sees at once whether it
//! works (or the plugin's log when it does not).
//!
//! A plugin can also show itself in the app, as DeepSeek Harness's plugins do:
//!   "panel": {"title": "PenipuMY status", "icon": "status", "page": "panel.html"}
//!     a row in the app's side bar (under New chat, with that icon), opening the plugin's own web page in the
//!     preview pane; the page is served at /plugin/<name>/<file>, and its script can call the plugin's
//!     functions with fetch("/plugin/<name>/call/<function>", {method: "POST", body: JSON args}).
//!   "badge": {"function": "status", "every": 60}
//!     a short status the app shows over every page (a coloured dot and a few words), from one of its
//!     functions, which returns {"text": "...", "color": "green" | "amber" | "red" | "gray"} (or plain text).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::json::{obj, str, Json};

pub fn dir() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("plugins")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

/// A plugin name as folders and tool names allow it: lowercase letters, digits and -, at most 40.
fn clean_name(name: &str) -> String {
    let mut s: String = name.trim().to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').chars().take(40).collect::<String>().trim_end_matches('-').to_string()
}

fn manifest(dir: &Path) -> Option<Json> {
    let text = std::fs::read_to_string(dir.join("plugin.json")).ok()?;
    Json::parse(text.trim_start_matches('\u{feff}')).ok()
}

/// Every plugin folder: its name (the folder's) and manifest.
fn all() -> Vec<(String, Json)> {
    let Ok(root) = dir() else { return Vec::new() };
    let mut list: Vec<(String, Json)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.'))
        .filter_map(|e| Some((e.file_name().to_string_lossy().to_string(), manifest(&e.path())?)))
        .collect();
    list.sort_by(|a, b| a.0.cmp(&b.0));
    list
}

fn enabled(m: &Json) -> bool {
    m.get("enabled").and_then(Json::as_bool) != Some(false)
}

/// The Windows launcher for Python 3 (py.exe), which finds a real Python where "python" may be the Store's stub.
fn py_launcher() -> Option<PathBuf> {
    let p = std::env::var_os("SystemRoot").map(|r| PathBuf::from(r).join("py.exe"))?;
    p.is_file().then_some(p)
}

/// The enabled plugins as MCP server settings (mcp.rs starts them): run in the plugin's folder, and Python
/// through py -3, unbuffered and in UTF-8.
pub fn servers() -> Vec<(String, Json)> {
    let Ok(root) = dir() else { return Vec::new() };
    all().into_iter().filter(|(_, m)| enabled(m)).filter_map(|(name, m)| Some((name.clone(), server(&root.join(&name), &m)?))).collect()
}

/// ghost's runner for simple plugins (RUNNER), written next to the plugins where it is missing or old.
fn runner() -> Option<PathBuf> {
    let p = dir().ok()?.join(".runtime").join("ghost_tools.py");
    if std::fs::read_to_string(&p).ok().as_deref() != Some(RUNNER) {
        std::fs::create_dir_all(p.parent()?).ok()?;
        std::fs::write(&p, RUNNER).ok()?;
    }
    Some(p)
}

fn server(dir: &Path, m: &Json) -> Option<Json> {
    // a simple plugin: its "tools" file's functions, served by the runner
    let (command, args) = match m.get("tools").and_then(Json::as_str) {
        Some(file) => ("python".to_string(), vec![str(&runner()?.display().to_string()), str(file)]),
        None => (m.get("command").and_then(Json::as_str)?.to_string(), m.get("args").and_then(Json::as_array).unwrap_or(&[]).to_vec()),
    };
    let (mut command, mut args) = (command, args);
    let mut env: Vec<(String, Json)> = match m.get("env") {
        Some(Json::Obj(f)) => f.clone(),
        _ => Vec::new(),
    };
    if matches!(command.to_lowercase().as_str(), "python" | "python3" | "python.exe" | "py") {
        if let Some(py) = py_launcher() {
            command = py.display().to_string();
            args.insert(0, str("-3"));
        }
        env.push(("PYTHONUNBUFFERED".into(), str("1")));
        env.push(("PYTHONIOENCODING".into(), str("utf-8")));
    }
    Some(obj(vec![("command", str(&command)), ("args", Json::Arr(args)), ("env", Json::Obj(env)), ("cwd", str(&dir.display().to_string()))]))
}

/// The plugin_manager actions that change nothing, which need no OK.
pub fn read_only(arguments: &str) -> bool {
    let a = Json::parse(arguments).unwrap_or(Json::Null);
    matches!(a.get("action").and_then(Json::as_str).unwrap_or(""), "guide" | "list" | "show")
}

/// The plugin_manager tool.
pub fn tool(arguments: &str) -> String {
    let a = Json::parse(arguments).unwrap_or(Json::Null);
    let s = |k: &str| a.get(k).and_then(Json::as_str).unwrap_or("").trim().to_string();
    let result = match s("action").as_str() {
        "guide" if s("name") == "server" => Ok(format!("A plugin as a full MCP server: server.py (or server.js for Node), saved as files {{\"server.py\": \"...\"}}. It talks JSON-RPC on stdin/stdout, one message a line, so it must never print anything else to stdout (logs go to stderr). Start from this template:\n```python\n{TEMPLATE}```")),
        "guide" => Ok(GUIDE.to_string()),
        "list" => Ok(list()),
        "show" => show(&s("name")),
        "save" => save(&s("name"), &s("description"), a.get("files")),
        "install" => install(&s("source"), &s("name")),
        "enable" => switch(&s("name"), true),
        "disable" => switch(&s("name"), false),
        "remove" => remove(&s("name")),
        "" => Err("say which action: guide, list, show, save, install, enable, disable or remove".into()),
        other => Err(format!("there is no action {other}: use guide, list, show, save, install, enable, disable or remove")),
    };
    result.unwrap_or_else(|e| format!("Error: {e}"))
}

/// The plugins for the clients (GET /api/plugins): name, description, on or off, state and tools.
pub fn status() -> Json {
    let running = super::mcp::status();
    Json::Arr(
        all()
            .into_iter()
            .map(|(name, m)| {
                let srv = running.as_array().unwrap_or(&[]).iter().find(|s| s.get("name").and_then(Json::as_str) == Some(name.as_str())).cloned();
                let status = if !enabled(&m) { "off".to_string() } else { srv.as_ref().and_then(|s| s.get("status")).and_then(Json::as_str).unwrap_or("stopped").to_string() };
                obj(vec![
                    ("name", str(&name)),
                    ("description", str(m.get("description").and_then(Json::as_str).unwrap_or(""))),
                    ("enabled", Json::Bool(enabled(&m))),
                    ("status", str(&status)),
                    ("tools", srv.and_then(|s| s.get("tools").cloned()).unwrap_or(Json::Arr(Vec::new()))),
                    ("panel", panel(&name, &m).unwrap_or(Json::Null)),
                    ("badge", Json::Bool(m.get("badge").is_some())),
                ])
            })
            .collect(),
    )
}

/// A plugin's panel for the app: {title, icon, url} (url: the page's path on this server), when it has one.
fn panel(name: &str, m: &Json) -> Option<Json> {
    let p = m.get("panel")?;
    let g = |k: &str| p.get(k).and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty());
    let page = g("page").unwrap_or("panel.html").trim_start_matches(['/', '\\']);
    Some(obj(vec![("title", str(g("title").unwrap_or(name))), ("icon", str(g("icon").unwrap_or("plug"))), ("url", str(&format!("/plugin/{name}/{page}")))]))
}

/// A file of an enabled plugin's folder, for its panel page (and what that page loads), never outside it.
pub fn file(name: &str, rel: &str) -> Option<PathBuf> {
    let (name, d) = existing(name).ok()?;
    let m = manifest(&d)?;
    if !enabled(&m) || name.is_empty() {
        return None;
    }
    let f = std::fs::canonicalize(inside(&d, rel).ok()?).ok()?;
    (f.starts_with(std::fs::canonicalize(&d).ok()?) && f.is_file()).then_some(f)
}

/// One of an enabled plugin's functions (its tools), called by its panel page or for its badge.
pub fn call(name: &str, function: &str, args: &Json) -> String {
    match existing(name) {
        Ok((name, d)) if manifest(&d).is_some_and(|m| enabled(&m)) => super::mcp::call(&format!("mcp__{name}__{}", function.trim()), args),
        Ok((name, _)) => format!("Error: the plugin {name} is turned off"),
        Err(e) => format!("Error: {e}"),
    }
}

/// The badges' last answers: a plugin's, and when it came.
static BADGES: Mutex<Vec<(String, Instant, Json)>> = Mutex::new(Vec::new());

/// The badges of the enabled plugins that have one (GET /api/plugins/badges): each plugin's name, title, icon,
/// panel url ("" without a panel), and its status {text, color}, asked again once `every` seconds have passed
/// (60 unless it says, at least 10).
pub fn badges() -> Json {
    let mut out = Vec::new();
    for (name, m) in all().into_iter().filter(|(_, m)| enabled(m)) {
        let Some(b) = m.get("badge") else { continue };
        let function = b.get("function").and_then(Json::as_str).unwrap_or("status").to_string();
        let every = Duration::from_secs(b.get("every").and_then(Json::as_f64).unwrap_or(60.0).clamp(10.0, 86400.0) as u64);
        let cached = BADGES.lock().unwrap().iter().find(|(n, at, _)| *n == name && at.elapsed() < every).map(|(_, _, j)| j.clone());
        let status = cached.unwrap_or_else(|| {
            let j = badge_status(&call(&name, &function, &obj(vec![])));
            let mut list = BADGES.lock().unwrap();
            list.retain(|(n, _, _)| *n != name);
            list.push((name.clone(), Instant::now(), j.clone()));
            j
        });
        let p = panel(&name, &m);
        let pg = |k: &str| p.as_ref().and_then(|p| p.get(k)).and_then(Json::as_str).unwrap_or("").to_string();
        out.push(obj(vec![
            ("name", str(&name)),
            ("title", str(&if pg("title").is_empty() { name.clone() } else { pg("title") })),
            ("icon", str(&if pg("icon").is_empty() { "plug".to_string() } else { pg("icon") })),
            ("url", str(&pg("url"))),
            ("text", status.get("text").cloned().unwrap_or(str(""))),
            ("color", status.get("color").cloned().unwrap_or(str("gray"))),
        ]));
    }
    Json::Arr(out)
}

/// A badge function's answer as {text, color}: its JSON, or its text (the first line) in grey, or its error in red.
fn badge_status(r: &str) -> Json {
    let short = |t: &str| t.lines().next().unwrap_or("").trim().chars().take(60).collect::<String>();
    if let Some(e) = r.strip_prefix("Error: ") {
        return obj(vec![("text", str(&short(e))), ("color", str("red"))]);
    }
    match Json::parse(r.trim()) {
        Ok(j @ Json::Obj(_)) => {
            let text = j.get("text").and_then(Json::as_str).map(short).unwrap_or_default();
            let color = j.get("color").and_then(Json::as_str).unwrap_or("gray").to_lowercase();
            obj(vec![("text", str(&text)), ("color", str(&color))])
        }
        _ => obj(vec![("text", str(&short(r))), ("color", str("gray"))]),
    }
}

fn list() -> String {
    let plugins = status();
    let plugins = plugins.as_array().unwrap_or(&[]);
    if plugins.is_empty() {
        return "No plugins yet. To make one, call plugin_manager with action guide first.".into();
    }
    let lines: Vec<String> = plugins
        .iter()
        .map(|p| {
            let g = |k: &str| p.get(k).and_then(Json::as_str).unwrap_or("").to_string();
            let tools: Vec<String> = p.get("tools").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(|t| format!("mcp__{}__{t}", g("name"))).collect();
            format!("- {} ({}): {}{}", g("name"), g("status"), g("description"), if tools.is_empty() { String::new() } else { format!("\n  tools: {}", tools.join(", ")) })
        })
        .collect();
    format!("Plugins in {}:\n{}", dir().map(|d| d.display().to_string()).unwrap_or_default(), lines.join("\n"))
}

fn existing(name: &str) -> Result<(String, PathBuf), String> {
    let name = clean_name(name);
    let d = dir()?.join(&name);
    if name.is_empty() || !d.join("plugin.json").is_file() {
        let names: Vec<String> = all().into_iter().map(|(n, _)| n).collect();
        return Err(format!("there is no plugin {name:?} (plugins: {})", if names.is_empty() { "none".into() } else { names.join(", ") }));
    }
    Ok((name, d))
}

/// A plugin's files (the text ones, each clipped) and the end of its log.
fn show(name: &str) -> Result<String, String> {
    let (name, d) = existing(name)?;
    let mut out = format!("Plugin {name} in {}", d.display());
    for f in files_in(&d) {
        let rel = f.strip_prefix(&d).unwrap_or(&f).display().to_string();
        match std::fs::read_to_string(&f) {
            Ok(t) => out.push_str(&format!("\n\n--- {rel}\n{}", crate::tools::clip(&t, 6000))),
            Err(_) => out.push_str(&format!("\n\n--- {rel} (not text)")),
        }
    }
    let log = super::mcp::log_tail(&name, 3000);
    if !log.is_empty() {
        out.push_str(&format!("\n\n--- its log (stderr)\n{log}"));
    }
    Ok(out)
}

/// The files under a folder (not its dependencies' folders), at most 200.
fn files_in(d: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![d.to_path_buf()];
    while let Some(p) = stack.pop() {
        for e in std::fs::read_dir(&p).into_iter().flatten().flatten() {
            let path = e.path();
            let n = e.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                if !matches!(n.as_str(), "node_modules" | "__pycache__" | ".git" | ".venv" | "venv") {
                    stack.push(path);
                }
            } else if out.len() < 200 {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// A path inside the plugin's folder: relative, without .. or a drive.
fn inside(d: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim().replace('\\', "/");
    let p = Path::new(&rel);
    if rel.is_empty() || p.is_absolute() || rel.contains(':') || p.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
        return Err(format!("{rel:?} is not a path inside the plugin's folder (use a relative one such as server.py)"));
    }
    Ok(d.join(p))
}

/// A file's content without the Markdown fence a model may wrap it in (```python ... ```), the fence's line
/// break written as \n or as a real one.
fn unfenced(text: &str) -> String {
    let t = text.trim_start();
    let Some(rest) = t.strip_prefix("```") else { return text.to_string() };
    let lang = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '+').count();
    let rest = &rest[lang..];
    let Some(body) = rest.strip_prefix("\\n").or_else(|| rest.strip_prefix("\r\n")).or_else(|| rest.strip_prefix('\n')) else { return text.to_string() };
    let body = body.trim_end();
    let body = body.strip_suffix("```").unwrap_or(body);
    let body = body.strip_suffix("\\n").unwrap_or(body);
    format!("{}\n", body.trim_end())
}

/// A file written as one line with its line breaks escaped once too often (\\n in the JSON, so "\n" in the
/// text): a small model's slip. A file with any real line break is left as it is.
fn unescaped(text: &str) -> String {
    let body = text.trim_end_matches('\n');
    if body.contains('\n') || !body.contains("\\n") {
        return text.to_string();
    }
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('n') => {
                    chars.next();
                    out.push('\n');
                    continue;
                }
                Some('t') => {
                    chars.next();
                    out.push('\t');
                    continue;
                }
                Some('"') => {
                    chars.next();
                    out.push('"');
                    continue;
                }
                Some('\\') => {
                    chars.next();
                    out.push('\\');
                    continue;
                }
                _ => {}
            }
        }
        out.push(c);
    }
    out.push('\n');
    out
}

/// Writes a plugin's files (new ones, or changed ones of an existing plugin), makes its plugin.json when it
/// has none, then starts it and lists its tools.
fn save(name: &str, description: &str, files: Option<&Json>) -> Result<String, String> {
    let name = clean_name(name);
    if name.is_empty() {
        return Err("give the plugin a name (lowercase words joined by -)".into());
    }
    if super::mcp::configured(&name) {
        return Err(format!("an MCP server in mcp.json is already called {name}: choose another name"));
    }
    let files: Vec<(String, String)> = match files {
        Some(Json::Obj(f)) => f.iter().map(|(k, v)| (k.clone(), unescaped(&unfenced(&v.as_str().map(String::from).unwrap_or_else(|| v.to_string()))))).collect(),
        None => Vec::new(),
        _ => return Err("files must be an object of relative path -> file content".into()),
    };
    let d = dir()?.join(&name);
    if files.is_empty() && !d.join("plugin.json").is_file() {
        return Err("a new plugin needs its files: files = {\"server.py\": \"...\"}".into());
    }
    // check everything before writing anything
    let mut writes = Vec::new();
    for (rel, content) in &files {
        let p = inside(&d, rel)?;
        if rel.trim().eq_ignore_ascii_case("plugin.json") {
            let m = Json::parse(content).map_err(|e| format!("plugin.json is not valid JSON: {e}"))?;
            if m.get("command").and_then(Json::as_str).is_none() && m.get("tools").and_then(Json::as_str).is_none() {
                return Err("plugin.json needs \"tools\" (a tools.py), or a \"command\" (python or node) and \"args\" (the script)".into());
            }
        }
        writes.push((p, content));
    }
    for (p, content) in writes {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("cannot make {}: {e}", parent.display()))?;
        }
        std::fs::write(&p, content).map_err(|e| format!("cannot write {}: {e}", p.display()))?;
    }
    // a manifest: the one written, the old one, or one made for the script there
    let m = match manifest(&d) {
        Some(m) => m,
        None => {
            let (command, script) = if d.join("tools.py").is_file() && !d.join("server.py").is_file() {
                let m = obj(vec![("name", str(&name)), ("description", str(description)), ("version", str("0.1.0")), ("tools", str("tools.py"))]);
                std::fs::write(d.join("plugin.json"), m.to_string()).map_err(|e| e.to_string())?;
                return finish(&d, &name, description, files.len());
            } else if d.join("server.py").is_file() {
                ("python", "server.py")
            } else if d.join("server.js").is_file() {
                ("node", "server.js")
            } else if d.join("server.mjs").is_file() {
                ("node", "server.mjs")
            } else {
                return Err("no plugin.json, and no tools.py, server.py or server.js to make one for".into());
            };
            obj(vec![("name", str(&name)), ("description", str(description)), ("version", str("0.1.0")), ("command", str(command)), ("args", Json::Arr(vec![str(script)]))])
        }
    };
    std::fs::write(d.join("plugin.json"), m.to_string()).map_err(|e| e.to_string())?;
    finish(&d, &name, description, files.len())
}

/// Turns a saved plugin on (its manifest named, described and enabled), then starts it.
fn finish(d: &Path, name: &str, description: &str, written: usize) -> Result<String, String> {
    let mut m = manifest(d).ok_or("its plugin.json is not valid JSON")?;
    if let Json::Obj(f) = &mut m {
        f.retain(|(k, _)| k != "enabled" && k != "name");
        f.insert(0, ("name".into(), str(&name)));
        if !description.is_empty() {
            f.retain(|(k, _)| k != "description");
            f.insert(1, ("description".into(), str(description)));
        }
        f.push(("enabled".into(), Json::Bool(true)));
    }
    std::fs::write(d.join("plugin.json"), m.to_string()).map_err(|e| e.to_string())?;
    started(&name, &format!("Saved {} file(s) in {}.", written, d.display()))
}

/// Starts (or restarts) a plugin and says how it went: its tools, or why it did not start and its log.
fn started(name: &str, done: &str) -> Result<String, String> {
    match super::mcp::restart(name) {
        Ok(tools) if tools.is_empty() => Ok(format!("{done} It started but has no tools: check its tools/list answer.")),
        Ok(tools) => {
            let full: Vec<String> = tools.iter().map(|t| format!("mcp__{name}__{t}")).collect();
            Ok(format!("{done} It is running with these tools: {}. Load one with tool_search (\"select:{}\"), then call it once to check it works.", full.join(", "), full[0]))
        }
        Err(e) => {
            let log = super::mcp::log_tail(name, 3000);
            let at = dir().map(|d| d.join(name).display().to_string()).unwrap_or_default();
            Ok(format!(
                "{done} But it did not start: {e}.{} Its files are in {at} (read_file can read them). Fix them, then save the corrected files whole with plugin_manager save.",
                if log.is_empty() { String::new() } else { format!("\nIts log (stderr):\n{log}\n") }
            ))
        }
    }
}

/// Copies a plugin folder from this computer into the plugins folder, then starts it.
fn install(source: &str, name: &str) -> Result<String, String> {
    let src = PathBuf::from(source.trim().trim_matches('"'));
    let m = manifest(&src).ok_or_else(|| format!("{} has no plugin.json", src.display()))?;
    let name = clean_name(if name.is_empty() { m.get("name").and_then(Json::as_str).unwrap_or_else(|| src.file_name().and_then(|n| n.to_str()).unwrap_or("")) } else { name });
    if name.is_empty() {
        return Err("give the plugin a name".into());
    }
    if super::mcp::configured(&name) {
        return Err(format!("an MCP server in mcp.json is already called {name}: give another name"));
    }
    let d = dir()?.join(&name);
    if d.exists() {
        return Err(format!("a plugin called {name} is installed already: remove it first, or give another name"));
    }
    let files = files_in(&src);
    let size: u64 = files.iter().filter_map(|f| f.metadata().ok()).map(|m| m.len()).sum();
    if size > 50 << 20 {
        return Err("that folder holds more than 50 MB: a plugin should be a few small files".into());
    }
    for f in &files {
        let to = d.join(f.strip_prefix(&src).map_err(|e| e.to_string())?);
        if let Some(p) = to.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        std::fs::copy(f, &to).map_err(|e| format!("cannot copy {}: {e}", f.display()))?;
    }
    switch_file(&d, &name, true)?;
    started(&name, &format!("Installed {name} ({} files).", files.len()))
}

fn switch_file(d: &Path, name: &str, on: bool) -> Result<(), String> {
    let mut m = manifest(d).ok_or("its plugin.json is not valid JSON")?;
    if let Json::Obj(f) = &mut m {
        f.retain(|(k, _)| k != "enabled" && k != "name");
        f.insert(0, ("name".into(), str(name)));
        f.push(("enabled".into(), Json::Bool(on)));
    }
    std::fs::write(d.join("plugin.json"), m.to_string()).map_err(|e| e.to_string())
}

fn switch(name: &str, on: bool) -> Result<String, String> {
    let (name, d) = existing(name)?;
    switch_file(&d, &name, on)?;
    if on {
        started(&name, &format!("Turned {name} on."))
    } else {
        let _ = super::mcp::restart(&name); // stops it: a disabled plugin is not started again
        Ok(format!("Turned {name} off; its tools are gone until it is turned on again."))
    }
}

/// Stops a plugin and moves its folder to plugins\.removed (so it can be brought back by hand).
fn remove(name: &str) -> Result<String, String> {
    let (name, d) = existing(name)?;
    switch_file(&d, &name, false)?;
    let _ = super::mcp::restart(&name);
    let bin = dir()?.join(".removed");
    std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
    let to = bin.join(format!("{name}-{}", super::agent::now_ms() as u64));
    std::fs::rename(&d, &to).map_err(|e| format!("cannot move it away: {e}"))?;
    Ok(format!("Removed {name} (its files are kept in {} in case it is wanted back).", to.display()))
}

const GUIDE: &str = r#"How to make a ghost plugin: new tools for you, kept for every later chat.

A plugin is one file, tools.py: plain Python 3 functions. Each function whose name does not start with _ becomes a tool. ghost does the rest (the protocol, the tool list). Example, a whole plugin:

import random

def roll_dice(count: int = 2, sides: int = 6) -> str:
    """Roll dice and give each roll and the total.
    count: how many dice
    sides: how many sides each die has
    """
    rolls = [random.randint(1, sides) for _ in range(count)]
    return f"rolls: {rolls}, total: {sum(rolls)}"

Rules:
- Write the real functions the user asked for; do not copy the example.
- Give each parameter a type (str, int, float, bool, list, dict) and a default when it is optional. The docstring's first line says what the tool does; lines "name: ..." describe the parameters.
- Return text (or a list or dict, sent as JSON). Raise an exception for an error: its message is shown.
- Python standard library only (no pip installs unless the user agrees). Keep each call under a minute or two. Never put a password or key in the code: ask the user, and read it from os.environ.
- Save: plugin_manager {"action": "save", "name": "short-name", "description": "one line", "files": {"tools.py": "<the code>"}}. Write the code as normal lines (in the JSON string, a line break is \n), with no ``` fence.
- Saving starts the plugin and answers with its tools, or with the error. To fix one: read_file its tools.py (the path is in the answer), then save the corrected tools.py whole.
- Then test it: load the tool with tool_search ("select:mcp__<name>__<function>") and call it. Say it is done only once a call works. Each call asks the user first.
- Other actions: list, show (a plugin's files and log), enable, disable, remove, install (a plugin folder on this computer). For a plugin tools.py cannot be (Node, or a server of its own), call guide with name "server" for the full template.

A plugin can also show itself in the app (a dashboard, a status, a little app of its own). Add a plugin.json to the same save:
{"tools": "tools.py", "panel": {"title": "Site status", "icon": "status", "page": "panel.html"}, "badge": {"function": "status", "every": 60}}
- panel: a row in the app's side bar, under New chat, with that icon; a click opens page (a web page in the plugin's folder, saved in the same files) in the app's preview pane. icon is one of: status, chart, globe, shield, bell, calendar, cloud, mail, money, music, home, people, shop, game, map, clock, news, code, chat, warning, heart, star, bolt, plug.
- The page is plain HTML, CSS and JavaScript in one file (no build step; a CDN script is fine). To get data, its script calls the plugin's own functions, which can reach any site (a page cannot, the browser blocks it): const r = await fetch("/plugin/<name>/call/<function>", {method: "POST", body: JSON.stringify({...args})}); const {text, data} = await r.json(); (data is the result when it is JSON: return a dict or list from the function). Refresh it with setInterval when it shows live data. Match the app: a dark page (#1f1f1f background, #e8e8e8 text), system-ui font.
- badge: a short status shown over every page of the app (a coloured dot and a few words; a click opens the panel). function is a function in tools.py with no required arguments that returns {"text": "All up", "color": "green"} (color: green, amber, red or gray); the app asks it again every `every` seconds (at least 10).
- Fetch with urllib.request (standard library), with a timeout of 10 to 20 seconds, and catch errors: a badge function should return {"text": "can't reach it", "color": "red"}, not raise."#;

const TEMPLATE: &str = r#""""A ghost plugin: a small MCP server. ghost runs it and talks JSON-RPC on stdin/stdout, one message a line."""
import json
import sys


def hello(args):
    """An example tool: replace it with the real ones."""
    name = args.get("name") or "world"
    return f"Hello, {name}!"


# name -> function, description, parameters (JSON Schema properties) and which are required
TOOLS = {
    "hello": {
        "fn": hello,
        "description": "Greet someone by name.",
        "params": {"name": {"type": "string", "description": "Who to greet"}},
        "required": ["name"],
    },
}


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return  # a notification: nothing to answer
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion", "2025-11-25")
        result = {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "plugin", "version": "0.1.0"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": n, "description": t["description"],
             "inputSchema": {"type": "object", "properties": t["params"], "required": t.get("required", [])}}
            for n, t in TOOLS.items()
        ]}
    elif method == "tools/call":
        params = msg.get("params") or {}
        tool = TOOLS.get(params.get("name"))
        if tool is None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "no such tool"}})
            return
        try:
            text, failed = str(tool["fn"](params.get("arguments") or {})), False
        except Exception as e:
            text, failed = f"{type(e).__name__}: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": failed}
    elif method == "ping":
        result = {}
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
        return
    send({"jsonrpc": "2.0", "id": mid, "result": result})


for line in sys.stdin:
    line = line.strip()
    if line:
        try:
            handle(json.loads(line))
        except Exception as e:
            print(f"bad message: {e}", file=sys.stderr, flush=True)
"#;

/// The runner for simple plugins: serves a tools.py's public functions as MCP tools (stdio, the 2025-11-25
/// handshake), their parameters from the signature and the docstring's "name: ..." lines. What a plugin
/// prints goes to stderr, so it cannot break the protocol.
const RUNNER: &str = r#""""ghost's runner for simple plugins: the public functions of a plugin's tools.py become MCP tools."""
import importlib.util
import inspect
import json
import os
import sys
import traceback
import typing

PROTOCOL = sys.stdout
sys.stdout = sys.stderr  # the plugin's own prints go to the log, not into the protocol

TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object", tuple: "array"}


def load(path):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("plugin_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        n: f for n, f in vars(mod).items()
        if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == mod.__name__
    }


def type_of(ann):
    if ann is inspect.Parameter.empty:
        return None
    origin = typing.get_origin(ann)
    if origin is typing.Union:  # Optional[int] and the like: the first real type
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        return type_of(args[0]) if args else None
    return TYPES.get(origin or ann)


def describe(name, fn):
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    sig = inspect.signature(fn)
    notes, summary = {}, []
    for line in lines:
        key, sep, text = line.strip().lstrip("-* ").partition(":")
        if sep and key.strip() in sig.parameters:
            notes[key.strip()] = text.strip()
        elif not notes:
            summary.append(line.strip())
    props, required = {}, []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = {}
        t = type_of(p.annotation) or (TYPES.get(type(p.default)) if p.default not in (p.empty, None) else None) or "string"
        prop["type"] = t
        if p.name in notes:
            prop["description"] = notes[p.name]
        if p.default is p.empty:
            required.append(p.name)
        elif isinstance(p.default, (int, float, str, bool)):
            prop["default"] = p.default
        props[p.name] = prop
    text = " ".join(s for s in summary if s) or name.replace("_", " ")
    return {"name": name, "description": text, "inputSchema": {"type": "object", "properties": props, "required": required}}


def coerce(value, t):
    """A model may send 3 as "3": give the function what its signature asks for."""
    try:
        if t == "integer" and isinstance(value, (str, float)):
            return int(float(value))
        if t == "number" and isinstance(value, str):
            return float(value)
        if t == "boolean" and isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        if t in ("array", "object") and isinstance(value, str):
            return json.loads(value)
    except (ValueError, TypeError):
        pass
    return value


def send(msg):
    PROTOCOL.write(json.dumps(msg) + "\n")
    PROTOCOL.flush()


def main():
    try:
        tools = load(sys.argv[1])
        schemas = {n: describe(n, f) for n, f in tools.items()}
    except Exception:
        traceback.print_exc()  # to the log: the plugin did not load
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            continue
        params = msg.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", "2025-11-25"), "capabilities": {"tools": {}}, "serverInfo": {"name": "ghost-plugin", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": list(schemas.values())}
        elif method == "tools/call":
            name = params.get("name")
            if name not in tools:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"no tool {name}"}})
                continue
            props = schemas[name]["inputSchema"]["properties"]
            args = {k: coerce(v, props.get(k, {}).get("type")) for k, v in (params.get("arguments") or {}).items() if k in props}
            try:
                out = tools[name](**args)
                text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
                failed = False
            except Exception as e:
                traceback.print_exc()
                text, failed = f"{type(e).__name__}: {e}", True
            result = {"content": [{"type": "text", "text": text}], "isError": failed}
        elif method == "ping":
            result = {}
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": result})


main()
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_paths() {
        assert_eq!(clean_name("My Weather  Tool!"), "my-weather-tool");
        assert_eq!(clean_name("--x--"), "x");
        let d = Path::new("C:/p/x");
        assert!(inside(d, "server.py").is_ok());
        assert!(inside(d, "lib/util.py").is_ok());
        assert!(inside(d, "../evil.py").is_err());
        assert!(inside(d, "C:/Windows/x.py").is_err());
        assert!(inside(d, "/etc/x").is_err());
        assert!(inside(d, "").is_err());
        assert_eq!(unfenced("```python\\nimport x\nprint(1)\n```"), "import x\nprint(1)\n");
        assert_eq!(unfenced("```py\nimport x\n```\n"), "import x\n");
        assert_eq!(unfenced("import x\n"), "import x\n");
        assert_eq!(unescaped("import x\\nprint(\\\"a\\\\nb\\\")"), "import x\nprint(\"a\\nb\")\n");
        assert_eq!(unescaped("a\nb\\n"), "a\nb\\n");
        assert!(read_only(r#"{"action":"list"}"#));
        assert!(!read_only(r#"{"action":"save"}"#));
    }
}
