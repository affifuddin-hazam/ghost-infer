//! ghost-infer's native desktop app: a light window (drawn on the CPU, no GPU memory) onto the local
//! ghost-infer server. The server runs the model and the agent loop and keeps the chats; this shows them.

#![windows_subsystem = "windows"]

mod api;
mod ghost;
#[path = "../../src/json.rs"]
#[allow(dead_code)]
mod json;
mod view;

use std::time::Instant;

use iced::futures::channel::{mpsc, oneshot};
use iced::widget::{markdown, text_editor};
use iced::{task, Task, Theme};

use api::Api;
use json::{num, obj, str, Json};

/// How many of the latest messages are drawn; older ones wait behind "Show earlier" (text layout is the
/// app's main memory cost, so it stays bounded however long a chat gets).
pub const SHOWN: usize = 30;

/// Command line: `ghost-app [host:port] [--open <chat id>] [--send <message>] [--menu plus|chat|settings|skill-link]` (the last three
/// for testing).
#[derive(Debug, Clone, Default)]
struct Args {
    addr: String,
    open: Option<String>,
    send: Option<String>,
    menu: Option<String>,
}

fn main() -> iced::Result {
    let mut args = Args { addr: "127.0.0.1:8080".into(), ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--open" => args.open = it.next(),
            "--send" => args.send = it.next(),
            "--menu" => args.menu = it.next(),
            _ => args.addr = a,
        }
    }
    iced::application(move || App::new(args.clone()), App::update, App::view)
        .subscription(App::subscription)
        .title(|app: &App| match &app.title {
            Some(t) => format!("{t} - ghost-infer"),
            None => "ghost-infer".into(),
        })
        .theme(|app: &App| app.theme.clone())
        // tiny-skia repaints only what it thinks changed, and misses some text changes (a growing reply's
        // last lines, an icon swapped for a wider one); a background off by a hair makes it repaint the
        // whole window, so every real change does that, while the ghost's ticks stay cheap
        .style(|app: &App, _: &Theme| {
            let s = app.skin();
            let bg = if app.repaint { iced::Color { r: s.bg.r + 0.0001, ..s.bg } } else { s.bg };
            iced::theme::Style { background_color: bg, text_color: s.text }
        })
        .font(include_bytes!("../fonts/InstrumentSans.ttf").as_slice())
        .default_font(view::BODY)
        // no Windows frame: the app draws its own title bar in the skin's colours, like Claude Desktop
        .window(iced::window::Settings {
            size: iced::Size::new(1180.0, 780.0),
            min_size: Some(iced::Size::new(560.0, 420.0)),
            decorations: false,
            icon: ghost::icon(view::GRAPHITE.accent),
            platform_specific: iced::window::settings::PlatformSpecific { undecorated_shadow: true, ..Default::default() },
            ..Default::default()
        })
        .run()
}

/// Messages drawn when a chat opens: SHOWN, or GHOST_APP_SHOWN=n (for measuring what a long chat costs).
fn window() -> usize {
    std::env::var("GHOST_APP_SHOWN").ok().and_then(|v| v.parse().ok()).unwrap_or(SHOWN)
}

/// A blocking call on its own thread, as a future iced can wait on.
fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> impl std::future::Future<Output = T> {
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    async move { rx.await.expect("the worker thread ended without an answer") }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
}

/// A tool call as the reply shows it: what was called and what came back.
#[derive(Debug)]
pub struct ToolLine {
    id: String,
    name: String,
    arguments: String,
    result: Option<String>,
    denied: bool,
    open: bool,          // its arguments and result shown
    ask: Option<String>, // waiting for the user's OK: the group it needs ("edit", "cmd", "web")
    pub round: usize,    // the reply's round that called it
}

impl ToolLine {
    fn new(id: &str, name: &str, arguments: &str) -> ToolLine {
        ToolLine { id: id.to_string(), name: name.to_string(), arguments: arguments.to_string(), result: None, denied: false, open: false, ask: None, round: 0 }
    }

    /// The harness sending the reply back, shown as a step (its words inside): to check the code it changed,
    /// or (`goal`) to carry on with its task list's open steps.
    fn nudge(text: &str, goal: bool) -> ToolLine {
        ToolLine { result: Some(text.to_string()), ..ToolLine::new("nudge", if goal { "keep_going" } else { "check_work" }, "{}") }
    }

    /// Whether it changed a file (so the reply can be undone).
    fn changed_files(&self) -> bool {
        matches!(self.name.as_str(), "write_file" | "edit_file") && !self.denied && self.result.as_deref().is_some_and(|r| !r.starts_with("Error"))
    }
}

/// The tool groups the app can turn on, with their menu labels; "files" (reading) is on by default.
pub const GROUPS: [(&str, &str); 5] = [("files", "Read files"), ("edit", "Change files"), ("cmd", "Run commands"), ("web", "Use the web"), ("mcp", "Use MCP tools")];

/// What "Always allow" covers, by approval group.
fn group_name(group: &str) -> &'static str {
    match group {
        "edit" => "file changes",
        "cmd" => "commands",
        "mcp" => "MCP tools",
        "skills" => "skill installs",
        _ => "web access",
    }
}

/// One model turn of a reply, drawn in the order it happened: its thinking, its text, then the tools it
/// called (which belong to `Msg::tools`, marked with the round's index).
#[derive(Debug)]
pub struct Round {
    raw: String, // as streamed (a thinking one starts with <think>)
    pub think: String,
    pub answer: String,
    pub md: markdown::Content, // the answer parsed
}

/// One message as drawn: a user's text, or a whole reply (its rounds and their tool calls).
#[derive(Debug)]
struct Msg {
    role: Role,
    raw: String,       // a user message's text
    rounds: Vec<Round>,
    new_round: bool,   // a tool ran since the last text: the next text starts a round
    think_open: bool,  // the thinking shown (every round's at once)
    tools: Vec<ToolLine>,
    stats: String,
    checkpoint: Option<String>, // where the reply's file changes are kept, for undo
    changed: bool,              // it changed files
    undone: bool,
    scheduled: bool, // a user message the scheduler sent (a task or the heartbeat), shown by its label
}

impl Msg {
    fn user(text: &str) -> Msg {
        Msg {
            role: Role::User,
            raw: text.to_string(),
            rounds: Vec::new(),
            new_round: true,
            think_open: false,
            tools: Vec::new(),
            stats: String::new(),
            checkpoint: None,
            changed: false,
            undone: false,
            scheduled: false,
        }
    }

    fn reply() -> Msg {
        Msg { role: Role::Assistant, ..Msg::user("") }
    }

    /// More reply text, to the current round (a new one after a tool ran). The answer is parsed as it grows
    /// (it only ever grows at its end, so the Markdown is fed just the new part).
    fn push(&mut self, more: &str) {
        if self.new_round || self.rounds.is_empty() {
            self.rounds.push(Round { raw: String::new(), think: String::new(), answer: String::new(), md: markdown::Content::new() });
            self.new_round = false;
        }
        let r = self.rounds.last_mut().unwrap();
        r.raw.push_str(more);
        let (think, answer) = split_think(&r.raw);
        let (think, answer) = (think.to_string(), answer.trim().to_string());
        r.think = think;
        if let Some(new) = answer.strip_prefix(r.answer.as_str()) {
            if !new.is_empty() {
                r.md.push_str(new);
                r.answer = answer;
            }
        } else {
            // trailing spaces trimmed away later: parse it all again
            r.md = markdown::Content::parse(&answer);
            r.answer = answer;
        }
    }

    /// A tool call, in the round that made it (a round of its own when the model wrote nothing first).
    fn add_tool(&mut self, mut t: ToolLine) {
        if self.new_round || self.rounds.is_empty() {
            self.push("");
        }
        t.round = self.rounds.len() - 1;
        self.tools.push(t);
        self.new_round = true;
    }

    /// The whole answer: every round's text, a blank line between them (for copying).
    fn answer_text(&self) -> String {
        self.rounds.iter().map(|r| r.answer.as_str()).filter(|a| !a.is_empty()).collect::<Vec<_>>().join("\n\n")
    }
}

/// A reply's thinking (inside <think>...</think> at its start) and its answer.
fn split_think(raw: &str) -> (&str, &str) {
    let t = raw.trim_start();
    let Some(rest) = t.strip_prefix("<think>") else { return ("", raw) };
    match rest.find("</think>") {
        Some(end) => (rest[..end].trim(), rest[end + 8..].trim_start()),
        None => (rest.trim(), ""),
    }
}

/// A task list as the server keeps it: [{"text","status"}].
fn tasks_of(j: Option<&Json>) -> Vec<(String, String)> {
    j.and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter_map(|t| Some((t.get("text")?.as_str()?.to_string(), t.get("status").and_then(Json::as_str).unwrap_or("pending").to_string())))
        .collect()
}

/// Text on one line, cut to `max` characters.
pub fn clip_line(s: &str, max: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() > max { format!("{}…", s.chars().take(max - 1).collect::<String>()) } else { s }
}

/// An MCP tool's server and tool, from its name mcp__<server>__<tool>.
fn mcp_parts(name: &str) -> (String, String) {
    let rest = name.trim_start_matches("mcp__");
    match rest.split_once("__") {
        Some((s, t)) => (s.to_string(), t.to_string()),
        None => (rest.to_string(), String::new()),
    }
}

/// A short label for a tool call, like the chat page's.
fn tool_label(t: &ToolLine) -> String {
    let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
    let s = |k: &str| short(a.get(k).and_then(Json::as_str).unwrap_or(""));
    let label = match t.name.as_str() {
        "read_file" => format!("Read {}", s("path")),
        "list_files" => format!("Listed {}", if s("path").is_empty() { "the folder".into() } else { s("path") }),
        "search_files" => format!("Searched for {}", s("pattern")),
        "write_file" => format!("Wrote {}", s("path")),
        "edit_file" => format!("Edited {}", s("path")),
        "run_command" => format!("Ran {}", s("command")),
        "start_process" => format!("Started {}", s("command")),
        "check_process" => format!("Checked process {}", a.get("id").map(|v| v.to_string()).unwrap_or_default()),
        "stop_process" => format!("Stopped process {}", a.get("id").map(|v| v.to_string()).unwrap_or_default()),
        "web_search" => format!("Searched the web for \"{}\"", s("query")),
        "fetch_url" => format!("Read {}", s("url")),
        "get_system_info" => "Checked this computer's specs".into(),
        "update_tasks" => "Updated the task list".into(),
        "remember" => format!("Remembered: {}", s("fact")),
        "forget" => format!("Forgot: {}", s("fact")),
        "use_skill" => format!("Used the {} skill", s("name")),
        "install_skill" => format!("Installed {}", if s("skill").is_empty() { s("source") } else { format!("{} from {}", s("skill"), s("source")) }),
        "search_chats" => format!("Searched earlier chats for \"{}\"", s("query")),
        "read_chat" => "Read an earlier chat".into(),
        "check_work" => "Asked to check its work".into(),
        "keep_going" => "Sent back to its open steps".into(),
        "schedule_task" => format!("Scheduled \"{}\"", if s("name").is_empty() { s("prompt") } else { s("name") }),
        "list_schedules" => "Listed the scheduled tasks".into(),
        "cancel_schedule" => format!("Cancelled scheduled task {}", s("id")),
        n if n.starts_with("mcp__") => {
            let (server, tool) = mcp_parts(n);
            format!("Used {tool} ({server} MCP)")
        }
        other => format!("Used {other}"),
    };
    if t.result.is_some() && !t.denied {
        return label;
    }
    // not done (waiting for an OK, or refused): "Edit notes.txt", not "Edited"
    let (verb, rest) = label.split_once(' ').unwrap_or((&label, ""));
    let verb = match verb {
        "Wrote" => "Write",
        "Edited" => "Edit",
        "Ran" => "Run",
        "Started" => "Start",
        "Searched" => "Search",
        "Installed" => "Install",
        other => other,
    };
    let label = format!("{verb} {rest}");
    if t.denied { format!("{label} (you said no)") } else { label }
}

/// The approval card's question and what it would do, like the chat page's.
fn approval_text(t: &ToolLine) -> (String, String) {
    let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
    let s = |k: &str| a.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let cwd = if s("cwd").is_empty() { String::new() } else { format!(" in {}", s("cwd")) };
    match t.name.as_str() {
        "write_file" => (format!("Write {} ({} lines)?", s("path"), s("content").lines().count()), s("content")),
        "edit_file" => {
            let lines = |k: &str, mark: &str| s(k).lines().map(|l| format!("{mark} {l}")).collect::<Vec<_>>().join("\n");
            (format!("Change {}?", s("path")), format!("{}\n{}", lines("old_string", "-"), lines("new_string", "+")))
        }
        "run_command" => (format!("Run this in PowerShell{cwd}?"), s("command")),
        "start_process" => (format!("Start this in the background{cwd}?"), s("command")),
        "web_search" => (format!("Search the web for \"{}\"?", s("query")), String::new()),
        "fetch_url" => (format!("Open {}?", s("url")), String::new()),
        "install_skill" => (format!("Install a skill from {}?", s("source")), if s("skill").is_empty() { String::new() } else { format!("Skill: {}", s("skill")) }),
        n if n.starts_with("mcp__") => {
            let (server, tool) = mcp_parts(n);
            let args = match Json::parse(&t.arguments) {
                Ok(Json::Obj(f)) => f.iter().map(|(k, v)| format!("{k}: {}", v.as_str().map(String::from).unwrap_or_else(|| v.to_string()))).collect::<Vec<_>>().join("\n"),
                _ => t.arguments.clone(),
            };
            (format!("Use {tool} from the {server} MCP server?"), args)
        }
        other => (format!("Use {other}?"), t.arguments.clone()),
    }
}

/// A call's details for the opened tool line: its arguments, one per line, then what came back (cut short).
fn tool_detail(t: &ToolLine) -> String {
    let mut out = String::new();
    if let Ok(Json::Obj(fields)) = Json::parse(&t.arguments) {
        for (k, v) in fields {
            let v = match v {
                Json::Str(s) => s,
                other => other.to_string(),
            };
            out.push_str(&format!("{k}: {}\n", clip(&v, 12, 1200)));
        }
    }
    if let Some(r) = &t.result {
        out.push_str(&format!("\n{}", clip(r, 40, 4000)));
    }
    out.trim_end().to_string()
}

/// Text cut to its first `lines` lines and `chars` characters, saying how much was left out.
fn clip(s: &str, lines: usize, chars: usize) -> String {
    let total = s.lines().count();
    let mut out: String = s.lines().take(lines).collect::<Vec<_>>().join("\n");
    if out.chars().count() > chars {
        out = out.chars().take(chars).collect();
    }
    if total > lines || out.len() < s.trim_end().len() {
        out.push_str(&format!("\n… ({total} lines in all)"));
    }
    out
}

/// A label value kept short: a long path by its last two parts, other long text cut.
fn short(v: &str) -> String {
    let v = v.trim();
    if v.chars().count() <= 60 {
        return v.to_string();
    }
    let parts: Vec<&str> = v.split(['\\', '/']).filter(|p| !p.is_empty()).collect();
    if parts.len() > 2 && !v.contains(' ') {
        return format!("…\\{}", parts[parts.len() - 2..].join("\\"));
    }
    format!("{}…", v.chars().take(57).collect::<String>())
}

/// A model in the picker: its file name, shown by a short name.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelChoice {
    pub id: String,
    pub label: String,
}

impl std::fmt::Display for ModelChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label)
    }
}

/// A model's file name without its quantisation and the noise around it, for people.
pub fn short_model(id: &str) -> String {
    let base = id.rsplit_once(['-', '.']).filter(|(_, q)| q.starts_with(['Q', 'q', 'I', 'F'])).map_or(id, |(b, _)| b);
    let base = base.trim_end_matches(".i1").trim_start_matches("Huihui-").trim_start_matches("JOSIEFIED-");
    if base.chars().count() > 28 { format!("{}…", base.chars().take(27).collect::<String>()) } else { base.to_string() }
}

/// A saved chat as the sidebar lists it.
#[derive(Debug, Clone)]
pub struct ChatItem {
    pub id: String,
    pub title: String,
    pub updated: f64, // milliseconds since 1970
    pub unread: bool, // a scheduled run wrote to it since the user last opened it
}

/// The open menu, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Menu {
    Chat,
    Settings,
    Plus,
}

/// Light or dark: Windows' own setting unless the user picked one (kept in %LOCALAPPDATA%\ghost-infer\app.json).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ThemeMode {
    System,
    Dark,
    Light,
}

fn app_file() -> Option<std::path::PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("ghost-infer").join("app.json"))
}

/// The app's own settings, kept in app.json (start-ghost-app reads load_last and last_model from it too).
struct Saved {
    mode: ThemeMode,
    tools: Vec<String>,         // the tool groups that are on
    rail: bool,                 // the rail is shown
    load_last: bool,            // the launcher starts the server with the last picked model
    last_model: Option<String>, // the model last picked in the app
    keep_going: bool,           // the goal loop: replies keep at their task list until it is done
}

fn load_settings() -> Saved {
    let j = app_file().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| Json::parse(&t).ok()).unwrap_or(Json::Null);
    let mode = match j.get("theme").and_then(Json::as_str) {
        Some("dark") => ThemeMode::Dark,
        Some("light") => ThemeMode::Light,
        _ => ThemeMode::System,
    };
    let tools = match j.get("tools").and_then(Json::as_array) {
        Some(a) => a.iter().filter_map(Json::as_str).filter(|g| GROUPS.iter().any(|(k, _)| k == g)).map(String::from).collect(),
        None => vec!["files".to_string()],
    };
    Saved {
        mode,
        tools,
        rail: j.get("rail").and_then(Json::as_bool).unwrap_or(false),
        load_last: j.get("load_last").and_then(Json::as_bool).unwrap_or(false),
        last_model: j.get("last_model").and_then(Json::as_str).map(String::from),
        keep_going: j.get("keep_going").and_then(Json::as_bool).unwrap_or(false),
    }
}

/// A link from a reply, in the default browser. Only web links: a reply could link to a program or a file.
/// Handed to Windows as it is (ShellExecuteW): `cmd /C start` would run whatever follows an & in it.
fn open_link(url: &str) {
    let lower = url.trim().to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://")) {
        return;
    }
    #[link(name = "shell32")]
    unsafe extern "system" {
        fn ShellExecuteW(hwnd: isize, op: *const u16, file: *const u16, params: *const u16, dir: *const u16, show: i32) -> isize;
    }
    const SW_SHOWNORMAL: i32 = 1;
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (op, file) = (w("open"), w(url.trim()));
    unsafe { ShellExecuteW(0, op.as_ptr(), file.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL) };
}

/// Whether Windows is set to dark mode for apps (HKCU\...\Personalize\AppsUseLightTheme = 0).
fn windows_dark() -> bool {
    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(key: isize, sub: *const u16, value: *const u16, flags: u32, kind: *mut u32, data: *mut u32, len: *mut u32) -> i32;
    }
    const HKEY_CURRENT_USER: isize = 0x8000_0001u32 as i32 as isize;
    const RRF_RT_REG_DWORD: u32 = 0x10;
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (sub, value) = (w(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"), w("AppsUseLightTheme"));
    let (mut data, mut len) = (1u32, 4u32);
    let r = unsafe { RegGetValueW(HKEY_CURRENT_USER, sub.as_ptr(), value.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(), &mut data, &mut len) };
    r == 0 && data == 0
}

struct App {
    theme: Theme,
    mode: ThemeMode,
    dark: bool,
    menu: Option<Menu>,
    clock: u64,               // milliseconds since the app started, for the ghost
    born: Instant,
    last_chunk_at: Option<u64>, // when text last arrived (the ghost's mouth moves while it does)
    happy_until: u64,         // the ghost smiles a moment after a reply
    last_active: Instant,     // the user's last action (the ghost dozes off after a while)
    searching: bool,
    filter: String,
    recents_open: bool,
    api: Option<Api>,
    status: String,
    models: Vec<ModelChoice>,
    model: Option<ModelChoice>,
    chats: Vec<ChatItem>,
    chat: Option<String>,
    title: Option<String>,
    messages: Vec<Msg>,
    shown: usize,
    composer: text_editor::Content,
    thinking: bool,
    busy: Option<task::Handle>,
    started: Option<Instant>,
    chunks: usize,               // text pieces of this round so far (about one token each)
    writing_since: Option<Instant>, // this round's first text: the speed shown counts from it
    context: (f64, f64), // tokens used, context length
    start: Args,         // what the command line asked to open or send once connected
    sidebar: bool,       // the chat list shown (it hides with the pane button)
    hovered: Option<String>, // the chat row under the pointer (its delete button shows)
    folder: Option<String>,  // the first folder file tools may use, by its name
    folders: Vec<String>,    // every folder file tools may use (the server's harness settings)
    tools: Vec<String>,      // the tool groups that are on
    keep_going: bool,        // the goal loop is on (the + menu)
    repaint: bool,           // flipped on every change but the ghost's tick: the whole window is drawn again
    maximized: bool,         // the window fills the screen (no resize edges; the button shows "restore")
    window: iced::Size,      // the window's size, to place menus
    plus_at: Option<iced::Rectangle>, // where the + button is (the + menu opens above it)
    rail: bool,              // the session rail on the right is shown
    load_last: bool,         // start with the last picked model (the launcher reads it)
    last_model: Option<String>, // the model last picked here
    system: Json,            // this computer, as /api/system reports it (for the rail)
    tasks: Vec<(String, String)>, // the model's task list for this chat: text, and pending / in_progress / done
    plan_open: Option<bool>, // the plan above the message box opened or closed by hand (None: open until it is all done)
    memory: Vec<String>,     // what the model remembers across chats
    memory_draft: String,    // a memory being typed in the rail
    instructions: Vec<(String, bool, bool)>, // the GHOST.md files: path, whether it exists, whether it is the global one
    skills: Vec<(String, String)>, // installed skills: name, description
    skill_link: String,      // a skill link being pasted in the rail
    skill_choices: Option<(String, Vec<(String, String, String)>)>, // a source with several skills: it, and each one's path, name, description
    mcp: Vec<(String, String, usize)>, // MCP servers: name, state, how many tools
    mcp_config: String,      // where mcp.json is
    background: Json,        // what runs on its own: the scheduled tasks and the heartbeat (/api/background)
    polled_at: u64,          // when /api/background was last asked (the clock), while no answer is pending
    polling: bool,
}

#[derive(Debug, Clone)]
enum Message {
    Connected(Result<(Api, Json, Json, Json), String>),
    Chats(Result<Json, String>),
    OpenChat(String),
    ChatLoaded(Result<Json, String>),
    NewChat,
    DeleteChat(String),
    PickModel(ModelChoice),
    ModelLoaded(Result<Json, String>),
    Edit(text_editor::Action),
    Send,
    Stop,
    Agent(Json),
    ToggleThinkView(usize),
    ToggleThinking,
    ShowEarlier,
    Link(String),
    Retry,
    ToggleSidebar,
    Hover(Option<String>),
    Copy(usize),
    Tick,
    Menu(Option<Menu>),
    ToggleTheme,
    SystemTheme,
    ToggleSearch,
    Filter(String),
    ToggleRecents,
    CopyChat,
    DeleteCurrent,
    ToggleTool(usize, usize), // a reply's tool line opened or closed
    ToggleGroup(&'static str),
    ToggleKeepGoing,
    Approve(String, &'static str), // a call's id, and "once", "always" or "deny"
    Approved(Result<Json, String>),
    Undo(usize),
    Undone(usize, Result<Json, String>),
    // the window, which has no Windows frame
    DragWindow,
    ToggleMaximize,
    Minimize,
    CloseWindow,
    ResizeWindow(iced::window::Direction),
    WindowResized(iced::Size),
    OpenPlus,
    PlusAt(Option<iced::Rectangle>),
    Maximized(bool),
    ToggleRail,
    TogglePlan,
    ToggleLoadLast,
    Quit,
    System(Result<Json, String>),
    UndoAll,
    AddFolder,
    FolderPicked(Option<String>),
    RemoveFolder(usize),
    Harness(Result<Json, String>),
    Memory(Result<Json, String>),
    Forget(usize),
    MemoryDraft(String),
    Instructions(Result<Json, String>),
    EditInstructions(String),
    AddMemory,
    InstallSkill(bool), // from an archive (true) or a folder
    SkillPicked(Option<String>),
    SkillLink(String),
    InstallLink,
    OpenSkillLink, // the + menu: the rail, with its link field focused
    InstallChoice(Option<String>), // one of a source's skills, by path (None: all of them)
    CancelChoice,
    Skills(Result<Json, String>),
    RemoveSkill(String),
    Mcp(Result<Json, String>),
    McpReload,
    McpImport,
    McpEdit,
    Background(Result<Json, String>),
    SetHeartbeat(u32), // minutes between checks; 0: off
    EditHeartbeat,
    RunScheduled(String), // a task's id, or "heartbeat": now
    CancelScheduled(String),
    Read(Result<Json, String>),
}

impl App {
    fn new(args: Args) -> (App, Task<Message>) {
        let addr = args.addr.clone();
        let Saved { mode, tools, rail, load_last, last_model, keep_going } = load_settings();
        let dark = match mode {
            ThemeMode::Dark => true,
            ThemeMode::Light => false,
            ThemeMode::System => windows_dark(),
        };
        let app = App {
            theme: if dark { view::GRAPHITE.theme() } else { view::FOG.theme() },
            mode,
            dark,
            menu: None,
            clock: 0,
            born: Instant::now(),
            last_chunk_at: None,
            happy_until: 0,
            last_active: Instant::now(),
            searching: false,
            filter: String::new(),
            recents_open: true,
            api: None,
            status: format!("Connecting to {addr}…"),
            models: Vec::new(),
            model: None,
            chats: Vec::new(),
            chat: None,
            title: None,
            messages: Vec::new(),
            shown: window(),
            composer: text_editor::Content::new(),
            thinking: true,
            busy: None,
            started: None,
            chunks: 0,
            writing_since: None,
            context: (0.0, 0.0),
            start: args,
            sidebar: true,
            hovered: None,
            folder: None,
            folders: Vec::new(),
            tools,
            keep_going,
            repaint: false,
            maximized: false,
            window: iced::Size::new(1180.0, 780.0),
            plus_at: None,
            rail,
            load_last,
            last_model,
            system: Json::Null,
            tasks: Vec::new(),
            plan_open: None,
            memory: Vec::new(),
            memory_draft: String::new(),
            instructions: Vec::new(),
            skills: Vec::new(),
            skill_link: String::new(),
            skill_choices: None,
            mcp: Vec::new(),
            mcp_config: String::new(),
            background: Json::Null,
            polled_at: 0,
            polling: false,
        };
        (app, Task::perform(blocking(move || connect(&addr)), Message::Connected))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        if !matches!(message, Message::Tick | Message::Agent(_) | Message::Hover(_) | Message::Background(_) | Message::Read(_)) {
            self.last_active = Instant::now();
        }
        // the background poll repaints only when it brings news (it sets `repaint` itself)
        if !matches!(message, Message::Tick | Message::Background(_)) {
            self.repaint = !self.repaint;
        }
        match message {
            Message::Connected(Ok((api, models, chats, harness))) => {
                self.api = Some(api);
                self.set_harness(&harness);
                self.set_models(&models);
                self.set_chats(&chats);
                self.status = String::new();
                let menu = self.start.menu.take();
                self.menu = match menu.as_deref() {
                    Some("chat") => Some(Menu::Chat),
                    Some("settings") => Some(Menu::Settings),
                    _ => None,
                };
                if self.model.is_none() {
                    self.status = "Loading the model…".into();
                }
                // the + menu opens as a click opens it: once the button is on screen
                let plus = match menu.as_deref() {
                    Some("plus") => Task::done(Message::OpenPlus),
                    Some("skill-link") => Task::done(Message::OpenSkillLink),
                    _ => Task::none(),
                };
                let system = Task::batch([self.refresh_system(), self.refresh_memory(), self.refresh_extras(), self.wait_for_model(), plus]);
                if let Some(id) = self.start.open.take() {
                    return Task::batch([system, self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded)]);
                }
                if let Some(text) = self.start.send.take() {
                    self.composer = text_editor::Content::with_text(&text);
                    return Task::batch([system, self.send(None)]);
                }
                return system;
            }
            // the server not up yet (the launcher starts both at once): try again in a moment
            Message::Connected(Err(_)) => {
                let addr = self.start.addr.clone();
                self.status = format!("Waiting for ghost-infer at {addr}…");
                return Task::perform(
                    blocking(move || {
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        connect(&addr)
                    }),
                    Message::Connected,
                );
            }
            Message::Chats(Ok(c)) => self.set_chats(&c),
            Message::Chats(Err(e)) | Message::ChatLoaded(Err(e)) | Message::ModelLoaded(Err(e)) => self.status = e,
            Message::OpenChat(id) => {
                if self.busy.is_none() {
                    return self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded);
                }
            }
            Message::ChatLoaded(Ok(chat)) => {
                self.show_chat(&chat);
                // opened with a scheduled run's reply in it: seen now
                if chat.get("unread").and_then(Json::as_bool).unwrap_or(false) {
                    let body = obj(vec![("id", chat.get("id").cloned().unwrap_or(Json::Null))]);
                    return self.call(
                        move |api| {
                            api.post("/api/chats/read", &body)?;
                            api.get("/api/chats")
                        },
                        Message::Read,
                    );
                }
            }
            Message::NewChat => {
                if self.busy.is_none() {
                    self.chat = None;
                    self.title = None;
                    self.messages.clear();
                    self.shown = window();
                    self.context.0 = 0.0;
                    self.tasks.clear();
                    self.plan_open = None;
                }
            }
            Message::DeleteChat(id) => {
                if self.chat.as_deref() == Some(id.as_str()) {
                    self.chat = None;
                    self.title = None;
                    self.messages.clear();
                }
                return self.call(
                    move |api| {
                        api.delete(&format!("/api/chats/{id}"))?;
                        api.get("/api/chats")
                    },
                    Message::Chats,
                );
            }
            Message::PickModel(choice) => {
                let id = choice.id;
                self.last_model = Some(id.clone());
                self.save_settings();
                self.status = format!("Loading {}…", choice.label);
                let body = obj(vec![("id", str(&id))]);
                return self.call(
                    move |api| {
                        api.post("/api/models/load", &body)?;
                        api.get("/api/models")
                    },
                    Message::ModelLoaded,
                );
            }
            Message::ModelLoaded(Ok(models)) => {
                self.set_models(&models);
                self.status = if self.model.is_none() { "Loading the model…".into() } else { String::new() };
                return Task::batch([self.wait_for_model(), self.refresh_system()]);
            }
            Message::Edit(action) => self.composer.perform(action),
            Message::Send => return self.send(None),
            Message::Retry => return self.send(Some(())),
            Message::Stop => {
                if let Some(h) = self.busy.take() {
                    h.abort();
                }
                self.finish("stopped");
            }
            Message::Agent(ev) => return self.on_event(ev),
            Message::ToggleThinkView(i) => {
                if let Some(m) = self.messages.get_mut(i) {
                    m.think_open = !m.think_open;
                }
            }
            Message::ToggleThinking => self.thinking = !self.thinking,
            Message::ShowEarlier => self.shown += SHOWN,
            Message::ToggleSidebar => self.sidebar = !self.sidebar,
            Message::Tick => {
                self.clock = self.born.elapsed().as_millis() as u64;
                // scheduled runs happen on the server on their own: a look every few seconds (the first at once)
                let due = self.background == Json::Null || self.clock.saturating_sub(self.polled_at) > 4000;
                if !self.polling && self.api.is_some() && due {
                    self.polling = true;
                    return self.call(|api| api.get("/api/background"), Message::Background);
                }
            }
            Message::Background(r) => {
                self.polling = false;
                self.polled_at = self.clock;
                let Ok(j) = r else { return Task::none() };
                let seq = |j: &Json| j.get("seq").and_then(Json::as_f64).unwrap_or(0.0);
                let running = |j: &Json| j.get("running").and_then(Json::as_str).map(String::from);
                if seq(&j) == seq(&self.background) {
                    return Task::none();
                }
                self.repaint = !self.repaint;
                let (was, now) = (running(&self.background), running(&j));
                let first = self.background == Json::Null;
                self.background = j;
                if first {
                    return Task::none();
                }
                let mut tasks = vec![self.call(|api| api.get("/api/chats"), Message::Chats)];
                // a run ended: if it wrote to the open chat, that chat is shown again with it
                if let Some(c) = was.filter(|_| now.is_none()) {
                    if self.busy.is_none() && (self.chat.as_deref() == Some(c.as_str()) || c.is_empty()) {
                        if let Some(id) = self.chat.clone() {
                            tasks.push(self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded));
                        }
                    }
                    if self.busy.is_none() {
                        self.status = String::new();
                    }
                } else if now.is_some() && self.busy.is_none() {
                    self.status = "A scheduled task is running".into();
                }
                return Task::batch(tasks);
            }
            Message::SetHeartbeat(every) => {
                let body = obj(vec![("every", num(every as f64)), ("tools", Json::Arr(self.tools.iter().map(|g| str(g)).collect())), ("thinking", Json::Bool(self.thinking))]);
                return self.call(move |api| api.post("/api/schedules/heartbeat", &body), Message::Background);
            }
            // the checklist in Notepad, made first when missing (the next check reads it)
            Message::EditHeartbeat => {
                let Some(path) = self.background.get("heartbeat").and_then(|h| h.get("file")).and_then(Json::as_str).map(String::from) else { return Task::none() };
                let file = std::path::Path::new(&path);
                if !file.exists() {
                    let note = "<!-- The heartbeat's checklist: while the heartbeat is on, ghost goes through it every so often and tells you only when something needs you (a reply of HEARTBEAT_OK is dropped). Keep it short: it is read on every check. Comments like this one are left out, and a file with only headings is not checked at all. -->\n\n# Heartbeat checklist\n\n- \n";
                    if let Err(e) = std::fs::write(file, note) {
                        self.status = format!("Couldn't make {path}: {e}");
                        return Task::none();
                    }
                }
                let _ = std::process::Command::new("notepad").arg(file).spawn();
            }
            Message::RunScheduled(id) => {
                self.status = "It runs as soon as nothing else is".into();
                let body = obj(vec![("id", str(&id))]);
                return self.call(move |api| api.post("/api/schedules/run", &body), Message::Background);
            }
            Message::CancelScheduled(id) => {
                let body = obj(vec![("id", str(&id))]);
                return self.call(move |api| api.post("/api/schedules/cancel", &body), Message::Background);
            }
            Message::Read(Ok(c)) => self.set_chats(&c),
            Message::Read(Err(_)) => {}
            Message::Menu(m) => self.menu = m,
            Message::ToggleTheme => {
                self.set_dark(!self.dark);
                self.mode = if self.dark { ThemeMode::Dark } else { ThemeMode::Light };
                self.save_settings();
                self.menu = None;
            }
            Message::SystemTheme => {
                self.mode = ThemeMode::System;
                self.set_dark(windows_dark());
                self.save_settings();
                self.menu = None;
            }
            Message::ToggleTool(i, j) => {
                if let Some(t) = self.messages.get_mut(i).and_then(|m| m.tools.get_mut(j)) {
                    t.open = !t.open;
                }
            }
            // the menu stays open, so several groups can be switched at once
            Message::ToggleGroup(g) => {
                match self.tools.iter().position(|t| t == g) {
                    Some(k) => {
                        self.tools.remove(k);
                    }
                    None => self.tools.push(g.to_string()),
                }
                self.save_settings();
            }
            Message::ToggleKeepGoing => {
                self.keep_going = !self.keep_going;
                self.save_settings();
            }
            Message::Approve(id, decision) => {
                if let Some(t) = self.messages.last_mut().and_then(|m| m.tools.iter_mut().find(|t| t.id == id)) {
                    t.ask = None;
                }
                self.status = if decision == "deny" { "Reading the results".into() } else { "Using a tool".into() };
                let body = obj(vec![("id", str(&id)), ("decision", str(decision))]);
                return self.call(move |api| api.post("/api/agent/approve", &body), Message::Approved);
            }
            Message::Approved(Ok(_)) => {}
            Message::Approved(Err(e)) => self.status = format!("Couldn't send your answer: {e}"),
            Message::Undo(i) => {
                if let Some(cp) = self.messages.get(i).and_then(|m| m.checkpoint.clone()) {
                    let body = obj(vec![("checkpoint", str(&cp))]);
                    return self.call(move |api| api.post("/api/tools/undo", &body), move |r| Message::Undone(i, r));
                }
            }
            Message::Undone(i, r) => {
                let note = match r {
                    Ok(j) => match j.get("undone").and_then(Json::as_array).map_or(0, |a| a.len()) {
                        0 => "Nothing to undo".to_string(),
                        1 => "Undone: 1 file put back".to_string(),
                        n => format!("Undone: {n} files put back"),
                    },
                    Err(e) => format!("Couldn't undo: {e}"),
                };
                if let Some(m) = self.messages.get_mut(i) {
                    m.undone = true;
                    m.stats = note;
                }
            }
            Message::ToggleSearch => {
                self.searching = !self.searching;
                if !self.searching {
                    self.filter.clear();
                }
            }
            Message::Filter(f) => self.filter = f,
            Message::ToggleRecents => self.recents_open = !self.recents_open,
            Message::CopyChat => {
                self.menu = None;
                let md: Vec<String> = self.messages.iter().map(|m| if m.role == Role::User { format!("**You:** {}", m.raw) } else { m.answer_text() }).collect();
                return iced::clipboard::write(md.join("\n\n"));
            }
            Message::DeleteCurrent => {
                self.menu = None;
                if let Some(id) = self.chat.clone() {
                    return self.update(Message::DeleteChat(id));
                }
            }
            Message::DragWindow => return iced::window::latest().and_then(iced::window::drag),
            Message::ToggleMaximize => return iced::window::latest().and_then(iced::window::toggle_maximize),
            Message::Minimize => return iced::window::latest().and_then(|id| iced::window::minimize(id, true)),
            Message::CloseWindow => return iced::exit(),
            Message::ResizeWindow(d) => return iced::window::latest().and_then(move |id| iced::window::drag_resize(id, d)),
            // maximized by the button, a double click, snapping or Win+Up: the size changes every time
            Message::WindowResized(size) => {
                self.window = size;
                return iced::window::latest().and_then(iced::window::is_maximized).map(Message::Maximized);
            }
            // the + menu: find the button on screen first, then open the menu above it
            Message::OpenPlus => return iced::widget::selector::find(iced::widget::selector::id("plus")).map(|t| Message::PlusAt(t.map(|t| t.bounds()))),
            Message::PlusAt(at) => {
                self.plus_at = at;
                self.menu = Some(Menu::Plus);
            }
            Message::Maximized(m) => self.maximized = m,
            Message::ToggleLoadLast => {
                self.load_last = !self.load_last;
                // on with nothing picked yet: the model running now is the one to start with
                if self.last_model.is_none() {
                    self.last_model = self.model.as_ref().map(|m| m.id.clone());
                }
                self.save_settings();
            }
            // the server runs without a window, so the app stops it: the model leaves the GPU, then the app closes
            Message::Quit => {
                let Some(api) = self.api.clone() else { return iced::exit() };
                return Task::perform(blocking(move || api.post("/api/quit", &Json::Null)), |_| ()).discard().chain(iced::exit());
            }
            Message::TogglePlan => self.plan_open = Some(!self.plan_shown()),
            Message::ToggleRail => {
                self.rail = !self.rail;
                self.save_settings();
                if self.rail {
                    return self.refresh_system();
                }
            }
            Message::System(Ok(j)) => self.system = j,
            Message::System(Err(_)) => {}
            Message::AddFolder => {
                self.menu = None;
                // the dialog blocks, so it opens on its own thread
                return Task::perform(blocking(|| rfd::FileDialog::new().set_title("Choose a folder the model may use").pick_folder().map(|p| p.display().to_string())), Message::FolderPicked);
            }
            Message::FolderPicked(None) => {}
            Message::FolderPicked(Some(f)) => {
                let mut folders = self.folders.clone();
                if !folders.contains(&f) {
                    folders.push(f);
                }
                return self.save_folders(folders);
            }
            Message::RemoveFolder(i) => {
                let mut folders = self.folders.clone();
                if i < folders.len() {
                    folders.remove(i);
                }
                return self.save_folders(folders);
            }
            Message::Harness(Ok(j)) => {
                self.set_harness(&j);
                return self.refresh_instructions(); // a new folder may bring its own
            }
            Message::Harness(Err(e)) => self.status = format!("Couldn't save the folders: {e}"),
            Message::Memory(Ok(j)) => self.memory = j.get("facts").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect(),
            Message::Memory(Err(e)) => self.status = format!("Couldn't read the memory: {e}"),
            Message::MemoryDraft(d) => self.memory_draft = d,
            Message::AddMemory => {
                let fact = std::mem::take(&mut self.memory_draft);
                if !fact.trim().is_empty() {
                    let body = obj(vec![("fact", str(fact.trim()))]);
                    return self.call(move |api| api.post("/api/memory/add", &body), Message::Memory);
                }
            }
            Message::InstallSkill(archive) => {
                self.menu = None;
                return Task::perform(
                    blocking(move || {
                        let d = rfd::FileDialog::new();
                        let picked = if archive { d.set_title("Choose a skill archive").add_filter("Skill", &["zip", "skill"]).pick_file() } else { d.set_title("Choose a skill's folder (it has a SKILL.md)").pick_folder() };
                        picked.map(|p| p.display().to_string())
                    }),
                    Message::SkillPicked,
                );
            }
            Message::SkillPicked(None) => {}
            Message::SkillLink(l) => self.skill_link = l,
            Message::InstallLink => {
                let link = std::mem::take(&mut self.skill_link);
                if !link.trim().is_empty() {
                    self.status = "Downloading the skill…".into();
                    let body = obj(vec![("path", str(link.trim()))]);
                    return self.call(move |api| api.post("/api/skills/install", &body), Message::Skills);
                }
            }
            Message::OpenSkillLink => {
                self.menu = None;
                // the field sits low in the rail: scrolled to, then focused
                let focus = iced::widget::operation::snap_to_end("rail").chain(iced::widget::operation::focus("skill-link"));
                if !self.rail {
                    self.rail = true;
                    self.save_settings();
                    return Task::batch([self.refresh_system(), focus]);
                }
                return focus;
            }
            Message::SkillPicked(Some(path)) => {
                let body = obj(vec![("path", str(&path))]);
                return self.call(move |api| api.post("/api/skills/install", &body), Message::Skills);
            }
            Message::InstallChoice(path) => {
                if let Some((source, _)) = &self.skill_choices {
                    self.status = "Installing…".into();
                    let body = obj(vec![("path", str(source)), ("skills", Json::Arr(vec![str(path.as_deref().unwrap_or("all"))]))]);
                    return self.call(move |api| api.post("/api/skills/install", &body), Message::Skills);
                }
            }
            Message::CancelChoice => self.skill_choices = None,
            Message::Skills(Ok(j)) => {
                self.skills = j.get("skills").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|s| Some((s.get("name")?.as_str()?.to_string(), s.get("description").and_then(Json::as_str).unwrap_or("").to_string()))).collect();
                if let Some(name) = j.get("installed").and_then(Json::as_str) {
                    self.skill_choices = None;
                    self.status = if name.contains(", ") { format!("Installed the skills {name}") } else { format!("Installed the {name} skill") };
                }
                if let Some(list) = j.get("choose").and_then(Json::as_array) {
                    let s = |c: &Json, k: &str| c.get(k).and_then(Json::as_str).unwrap_or("").to_string();
                    let choices: Vec<_> = list.iter().map(|c| (s(c, "path"), s(c, "name"), s(c, "description"))).collect();
                    self.status = format!("That link has {} skills: pick one in the rail", choices.len());
                    self.skill_choices = Some((s(&j, "source"), choices));
                }
            }
            Message::Skills(Err(e)) => self.status = format!("Couldn't install the skill: {e}"),
            Message::RemoveSkill(name) => {
                let body = obj(vec![("name", str(&name))]);
                return self.call(move |api| api.post("/api/skills/remove", &body), Message::Skills);
            }
            Message::Mcp(Ok(j)) => {
                self.mcp = j.get("servers").and_then(Json::as_array).unwrap_or(&[]).iter().filter(|s| s.get("name").and_then(Json::as_str).is_some_and(|n| !n.is_empty())).filter_map(|s| {
                    Some((s.get("name")?.as_str()?.to_string(), s.get("status").and_then(Json::as_str).unwrap_or("").to_string(), s.get("tools").and_then(Json::as_array).map_or(0, |t| t.len())))
                }).collect();
                if let Some(c) = j.get("config").and_then(Json::as_str) {
                    self.mcp_config = c.to_string();
                }
                // servers still starting: look again in a moment
                if self.mcp.iter().any(|m| m.1 == "starting") {
                    return self.call(|api| { std::thread::sleep(std::time::Duration::from_secs(3)); api.get("/api/mcp") }, Message::Mcp);
                }
            }
            Message::Mcp(Err(_)) => {}
            Message::McpReload => {
                return self.call(|api| { api.post("/api/mcp/reload", &Json::Null)?; std::thread::sleep(std::time::Duration::from_millis(500)); api.get("/api/mcp") }, Message::Mcp);
            }
            Message::McpImport => {
                return self.call(|api| { api.post("/api/mcp/import-claude", &Json::Null)?; std::thread::sleep(std::time::Duration::from_millis(500)); api.get("/api/mcp") }, Message::Mcp);
            }
            // mcp.json in Notepad; Reload picks the changes up
            Message::Instructions(Ok(j)) => {
                self.instructions = j.get("files").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|f| {
                    Some((f.get("path")?.as_str()?.to_string(), f.get("exists").and_then(Json::as_bool).unwrap_or(false), f.get("scope").and_then(Json::as_str) == Some("global")))
                }).collect();
            }
            Message::Instructions(Err(_)) => {}
            // opened in Notepad, made first when missing (the model reads it at its next reply)
            Message::EditInstructions(path) => {
                let file = std::path::Path::new(&path);
                if !file.exists() {
                    let note = "<!-- ghost reads this file into its chats, like a CLAUDE.md: write your instructions for the model below. Comments like this one are left out. -->\n\n";
                    if let Err(e) = std::fs::write(file, note) {
                        self.status = format!("Couldn't make {path}: {e}");
                        return Task::none();
                    }
                }
                let _ = std::process::Command::new("notepad").arg(file).spawn();
                return self.refresh_instructions();
            }
            Message::McpEdit => {
                if !self.mcp_config.is_empty() {
                    let _ = std::process::Command::new("notepad").arg(&self.mcp_config).spawn();
                }
            }
            Message::Forget(i) => {
                let body = obj(vec![("index", num(i as f64))]);
                return self.call(move |api| api.post("/api/memory/forget", &body), Message::Memory);
            }
            // every reply's file changes put back, the newest first, one after another (they share a journal)
            Message::UndoAll => {
                let mut task = Task::none();
                for i in (0..self.messages.len()).rev() {
                    let m = &self.messages[i];
                    if let (true, false, Some(cp)) = (m.changed, m.undone, m.checkpoint.clone()) {
                        let body = obj(vec![("checkpoint", str(&cp))]);
                        task = task.chain(self.call(move |api| api.post("/api/tools/undo", &body), move |r| Message::Undone(i, r)));
                    }
                }
                return task;
            }
            Message::Hover(h) => self.hovered = h,
            Message::Copy(i) => {
                if let Some(m) = self.messages.get(i) {
                    return iced::clipboard::write(m.answer_text());
                }
            }
            Message::Link(url) => open_link(&url),
        }
        Task::none()
    }

    /// A server call on a worker thread, its result as a message.
    fn call<T: Send + 'static>(&self, f: impl FnOnce(&Api) -> Result<T, String> + Send + 'static, done: impl FnOnce(Result<T, String>) -> Message + Send + 'static) -> Task<Message> {
        let Some(api) = self.api.clone() else { return Task::none() };
        Task::perform(blocking(move || f(&api)), done)
    }

    /// This computer's GPU, memory and model, for the rail (only fetched while it is shown).
    fn save_settings(&self) {
        let Some(p) = app_file() else { return };
        let v = match self.mode {
            ThemeMode::System => "system",
            ThemeMode::Dark => "dark",
            ThemeMode::Light => "light",
        };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let mut fields = vec![
            ("theme", str(v)),
            ("tools", Json::Arr(self.tools.iter().map(|g| str(g)).collect())),
            ("rail", Json::Bool(self.rail)),
            ("load_last", Json::Bool(self.load_last)),
            ("keep_going", Json::Bool(self.keep_going)),
        ];
        if let Some(m) = &self.last_model {
            fields.push(("last_model", str(m)));
        }
        let _ = std::fs::write(p, obj(fields).to_string());
    }

    /// The server's harness settings: the folders file tools may use.
    fn set_harness(&mut self, j: &Json) {
        self.folders = j.get("folders").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect();
        self.folder = self.folders.first().map(|f| f.rsplit(['\\', '/']).find(|p| !p.is_empty()).unwrap_or(f).to_string());
    }

    /// Sends the new folder list to the server (it checks each one is a folder).
    fn save_folders(&self, folders: Vec<String>) -> Task<Message> {
        let body = obj(vec![("folders", Json::Arr(folders.iter().map(|f| str(f)).collect()))]);
        self.call(move |api| api.post("/api/harness", &body), Message::Harness)
    }

    /// While the server is still loading its model, asks again every two seconds until one is running.
    fn wait_for_model(&self) -> Task<Message> {
        if self.model.is_some() {
            return Task::none();
        }
        self.call(
            |api| {
                std::thread::sleep(std::time::Duration::from_secs(2));
                api.get("/api/models")
            },
            Message::ModelLoaded,
        )
    }

    /// The installed skills, the MCP servers and the GHOST.md files (the rail lists them).
    fn refresh_extras(&self) -> Task<Message> {
        Task::batch([self.call(|api| api.get("/api/skills"), Message::Skills), self.call(|api| api.get("/api/mcp"), Message::Mcp), self.refresh_instructions()])
    }

    /// Whether the plan above the message box shows its steps: as set by hand, else until every step is done.
    pub fn plan_shown(&self) -> bool {
        self.plan_open.unwrap_or_else(|| self.tasks.iter().any(|t| t.1 != "done"))
    }

    fn refresh_instructions(&self) -> Task<Message> {
        self.call(|api| api.get("/api/instructions"), Message::Instructions)
    }

    /// What the model remembers (the rail shows it, with a remove button for each).
    fn refresh_memory(&self) -> Task<Message> {
        self.call(|api| api.get("/api/memory"), Message::Memory)
    }

    fn refresh_system(&self) -> Task<Message> {
        if !self.rail {
            return Task::none();
        }
        self.call(|api| api.get("/api/system"), Message::System)
    }

    /// The files this chat changed and not undone, with lines added and removed, for the rail.
    pub fn changes(&self) -> Vec<(String, usize, usize)> {
        let mut out: Vec<(String, usize, usize)> = Vec::new();
        for t in self.messages.iter().filter(|m| !m.undone).flat_map(|m| &m.tools).filter(|t| t.changed_files()) {
            let a = Json::parse(&t.arguments).unwrap_or(Json::Null);
            let s = |k: &str| a.get(k).and_then(Json::as_str).unwrap_or("").to_string();
            let (add, del) = if t.name == "write_file" { (s("content").lines().count(), 0) } else { (s("new_string").lines().count(), s("old_string").lines().count()) };
            let path = s("path");
            match out.iter_mut().find(|c| c.0 == path) {
                Some(c) => {
                    c.1 += add;
                    c.2 += del;
                }
                None => out.push((path, add, del)),
            }
        }
        out
    }

    fn set_models(&mut self, j: &Json) {
        let choice = |id: &str| ModelChoice { id: id.to_string(), label: short_model(id) };
        self.models = j.get("models").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(|m| m.get("id").and_then(Json::as_str).map(choice)).collect();
        self.model = j.get("current").and_then(Json::as_str).map(choice);
        self.context.1 = j.get("context_length").and_then(Json::as_f64).unwrap_or(0.0);
    }

    fn set_chats(&mut self, j: &Json) {
        self.chats = j
            .get("chats")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .filter_map(|c| {
                Some(ChatItem {
                    id: c.get("id")?.as_str()?.to_string(),
                    title: c.get("title").and_then(Json::as_str).unwrap_or("").to_string(),
                    updated: c.get("updated").and_then(Json::as_f64).unwrap_or(0.0),
                    unread: c.get("unread").and_then(Json::as_bool).unwrap_or(false),
                })
            })
            .collect();
    }

    pub fn skin(&self) -> &'static view::Skin {
        if self.dark { &view::GRAPHITE } else { &view::FOG }
    }

    fn set_dark(&mut self, dark: bool) {
        self.dark = dark;
        self.theme = self.skin().theme();
    }

    pub fn idle_secs(&self) -> f64 {
        self.last_active.elapsed().as_secs_f64()
    }

    /// Where file tools work, for the label under the message box (the server's first allowed folder).
    pub fn folder_label(&self) -> String {
        self.folder.clone().unwrap_or_else(|| "No folder".into())
    }

    /// The ghost's clock: a tick a few times a second (the sprite moves in steps, so no more is needed).
    fn subscription(&self) -> iced::Subscription<Message> {
        let resized = iced::window::resize_events().map(|(_, size)| Message::WindowResized(size));
        let tick = iced::Subscription::run(|| {
            use iced::futures::StreamExt;
            let (tx, rx) = mpsc::unbounded::<()>();
            std::thread::spawn(move || {
                while tx.unbounded_send(()).is_ok() {
                    std::thread::sleep(std::time::Duration::from_millis(110));
                }
            });
            rx.map(|_| Message::Tick)
        });
        iced::Subscription::batch([tick, resized])
    }

    /// A saved chat as messages: each user message, then its reply's rounds folded into one.
    fn show_chat(&mut self, chat: &Json) {
        self.chat = chat.get("id").and_then(Json::as_str).map(String::from);
        self.title = chat.get("title").and_then(Json::as_str).map(String::from);
        self.context.0 = chat.get("tokens").and_then(Json::as_f64).unwrap_or(0.0); // chats saved before this have none
        self.tasks = tasks_of(chat.get("tasks"));
        self.plan_open = None;
        self.messages.clear();
        self.shown = window();
        for m in chat.get("messages").and_then(Json::as_array).unwrap_or(&[]) {
            let content = m.get("content").and_then(Json::as_str).unwrap_or("");
            match m.get("role").and_then(Json::as_str) {
                // the harness sending the reply back (check your work, keep going): a step in the reply, not
                // a message of the user
                Some("user") if m.get("auto").and_then(Json::as_bool).unwrap_or(false) => {
                    if let Some(reply) = self.messages.last_mut().filter(|r| r.role == Role::Assistant) {
                        reply.add_tool(ToolLine::nudge(content, m.get("goal").and_then(Json::as_bool).unwrap_or(false)));
                    }
                }
                // sent by the scheduler: its short label, not the whole prompt the model got
                Some("user") if m.get("shown").is_some() => {
                    let mut msg = Msg::user(m.get("shown").and_then(Json::as_str).unwrap_or(content));
                    msg.scheduled = true;
                    self.messages.push(msg);
                }
                Some("user") => self.messages.push(Msg::user(content)),
                Some("assistant") => {
                    if self.messages.last().is_none_or(|l| l.role != Role::Assistant) {
                        self.messages.push(Msg::reply());
                    }
                    let reply = self.messages.last_mut().unwrap();
                    reply.new_round = true;
                    reply.push(content);
                    if let Some(cp) = m.get("checkpoint").and_then(Json::as_str) {
                        reply.checkpoint = Some(cp.to_string());
                    }
                    reply.changed |= m.get("changed").and_then(Json::as_bool).unwrap_or(false);
                    for c in m.get("tool_calls").and_then(Json::as_array).unwrap_or(&[]) {
                        let f = c.get("function");
                        let field = |k: &str| f.and_then(|f| f.get(k)).and_then(Json::as_str).unwrap_or("");
                        reply.add_tool(ToolLine::new(c.get("id").and_then(Json::as_str).unwrap_or(""), field("name"), field("arguments")));
                    }
                }
                Some("tool") => {
                    if let Some(reply) = self.messages.last_mut() {
                        let id = m.get("tool_call_id").and_then(Json::as_str).unwrap_or("");
                        // by its call's id; chats saved without ids answer the first open call
                        let k = reply.tools.iter().position(|t| t.result.is_none() && t.id == id).or_else(|| reply.tools.iter().position(|t| t.result.is_none()));
                        if let Some(t) = k.map(|k| &mut reply.tools[k]) {
                            t.result = Some(content.to_string());
                            t.denied = m.get("denied").and_then(Json::as_bool).unwrap_or(false);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Sends the composer's text (or, for a retry, answers the last message again) and streams the reply.
    fn send(&mut self, retry: Option<()>) -> Task<Message> {
        let Some(api) = self.api.clone() else { return Task::none() };
        if self.busy.is_some() {
            return Task::none();
        }
        let mut body = vec![("thinking", Json::Bool(self.thinking)), ("tools", Json::Arr(self.tools.iter().map(|g| str(g)).collect())), ("max_tokens", num(4096.0)), ("keep_going", Json::Bool(self.keep_going))];
        if let Some(id) = &self.chat {
            body.push(("chat", str(id)));
        }
        match retry {
            None => {
                let text = self.composer.text().trim().to_string();
                if text.is_empty() {
                    return Task::none();
                }
                self.composer = text_editor::Content::new();
                self.messages.push(Msg::user(&text));
                body.push(("message", str(&text)));
            }
            Some(()) => {
                if self.messages.last().is_some_and(|m| m.role == Role::Assistant) {
                    self.messages.pop();
                }
            }
        }
        self.messages.push(Msg::reply());
        self.started = Some(Instant::now());
        (self.chunks, self.writing_since) = (0, None);
        // the server holds this message until a scheduled run answering in this chat is done
        let waits = self.chat.is_some() && self.background.get("running").and_then(Json::as_str) == self.chat.as_deref();
        self.status = if waits { "Waiting for the scheduled task in this chat to finish".into() } else { "Reading".into() };
        let body = obj(body);
        let (tx, rx) = mpsc::unbounded();
        std::thread::spawn(move || {
            let end = match api.stream("/api/agent", &body, |ev| tx.unbounded_send(ev).is_ok()) {
                Ok(()) => obj(vec![("type", str("end"))]),
                Err(e) => obj(vec![("type", str("error")), ("message", str(&e))]),
            };
            let _ = tx.unbounded_send(end);
        });
        let (task, handle) = Task::run(rx, Message::Agent).abortable();
        self.busy = Some(handle);
        task
    }

    fn on_event(&mut self, ev: Json) -> Task<Message> {
        let kind = ev.get("type").and_then(Json::as_str).unwrap_or("");
        let s = |k: &str| ev.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        match kind {
            "chat" => {
                let new = self.chat.is_none();
                self.chat = Some(s("id"));
                self.title = Some(s("title"));
                if let Some(m) = self.messages.last_mut() {
                    m.checkpoint = Some(s("checkpoint")).filter(|c| !c.is_empty());
                }
                if new {
                    return self.call(|api| api.get("/api/chats"), Message::Chats);
                }
            }
            "text" => {
                self.chunks += 1;
                self.writing_since.get_or_insert_with(Instant::now);
                self.last_chunk_at = Some(self.born.elapsed().as_millis() as u64);
                if let Some(m) = self.messages.last_mut() {
                    m.push(&s("text"));
                    self.status = if m.rounds.last().is_some_and(|r| r.answer.is_empty() && !r.think.is_empty()) { "Thinking".into() } else { "Writing".into() };
                }
            }
            "tool_call" => {
                if let Some(m) = self.messages.last_mut() {
                    m.add_tool(ToolLine::new(&s("id"), &s("name"), &s("arguments")));
                }
                self.status = "Using a tool".into();
            }
            "tasks" => self.tasks = tasks_of(ev.get("tasks")),
            "nudge" => {
                let goal = s("kind") == "goal";
                if let Some(m) = self.messages.last_mut() {
                    m.add_tool(ToolLine::nudge(&s("text"), goal));
                }
                self.status = if goal { "Carrying on with its plan".into() } else { "Checking its work".into() };
            }
            "tool_result" => {
                let id = s("id");
                if let Some(m) = self.messages.last_mut() {
                    if let Some(t) = m.tools.iter_mut().find(|t| t.id == id) {
                        t.result = Some(s("result"));
                        t.denied = ev.get("denied").and_then(Json::as_bool).unwrap_or(false);
                        t.ask = None;
                        let changed = t.changed_files();
                        m.changed |= changed;
                    }
                }
                self.status = "Reading the results".into();
            }
            // the run waits until the card under the reply is answered
            "approval" => {
                let id = s("id");
                if let Some(t) = self.messages.last_mut().and_then(|m| m.tools.iter_mut().find(|t| t.id == id)) {
                    t.ask = Some(s("group"));
                }
                self.status = "Waiting for your OK".into();
            }
            // the model's title for a new chat, a moment after its first reply
            "title" => {
                let (id, title) = (s("id"), s("title"));
                if self.chat.as_deref() == Some(id.as_str()) {
                    self.title = Some(title.clone());
                }
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    c.title = title;
                }
            }
            // a round ended: the next one's speed counts afresh (not over the tools run in between)
            "context" => {
                let f = |k: &str| ev.get(k).and_then(Json::as_f64).unwrap_or(0.0);
                self.context = (f("prompt_tokens") + f("completion_tokens"), f("context_length"));
                (self.chunks, self.writing_since) = (0, None);
            }
            "done" => {
                self.finish(&s("finish"));
                return Task::batch([self.refresh_system(), self.refresh_memory(), self.refresh_extras()]); // the reply may have changed what the GPU holds
            }
            "error" => {
                self.status = s("message");
                self.busy = None;
            }
            "end" => {
                if self.busy.is_some() {
                    self.finish("stop");
                }
            }
            _ => {}
        }
        Task::none()
    }

    fn finish(&mut self, how: &str) {
        self.busy = None;
        if how != "stopped" {
            self.happy_until = self.born.elapsed().as_millis() as u64 + 3000;
        }
        let secs = self.started.take().map_or(0.0, |t| t.elapsed().as_secs_f64());
        if let Some(m) = self.messages.last_mut().filter(|m| m.role == Role::Assistant) {
            m.stats = format!("{secs:.1} s{}", match how {
                "length" => ", stopped at the length limit",
                "stopped" => ", stopped",
                _ => "",
            });
            // a card left open when the reply stopped has nothing to answer any more
            for t in &mut m.tools {
                t.ask = None;
            }
        }
        self.status = String::new();
    }
}

/// Connects and reads the models and chats.
fn connect(addr: &str) -> Result<(Api, Json, Json, Json), String> {
    let api = Api::connect(addr)?;
    let models = api.get("/api/models")?;
    let chats = api.get("/api/chats")?;
    let harness = api.get("/api/harness").unwrap_or(Json::Null);
    Ok((api, models, chats, harness))
}
