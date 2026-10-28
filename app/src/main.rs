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

/// Command line: `ghost-app [host:port] [--open <chat id>] [--send <message>] [--menu plus|chat|settings]` (the last three
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
        .window_size((1180.0, 780.0))
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

    /// Whether it changed a file (so the reply can be undone).
    fn changed_files(&self) -> bool {
        matches!(self.name.as_str(), "write_file" | "edit_file") && !self.denied && self.result.as_deref().is_some_and(|r| !r.starts_with("Error"))
    }
}

/// The tool groups the app can turn on, with their menu labels; "files" (reading) is on by default.
pub const GROUPS: [(&str, &str); 4] = [("files", "Read files"), ("edit", "Change files"), ("cmd", "Run commands"), ("web", "Use the web")];

/// What "Always allow" covers, by approval group.
fn group_name(group: &str) -> &'static str {
    match group {
        "edit" => "file changes",
        "cmd" => "commands",
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

/// The app's own settings: the theme and the tool groups that are on.
fn load_settings() -> (ThemeMode, Vec<String>) {
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
    (mode, tools)
}

fn save_settings(mode: ThemeMode, tools: &[String]) {
    let Some(p) = app_file() else { return };
    let v = match mode {
        ThemeMode::System => "system",
        ThemeMode::Dark => "dark",
        ThemeMode::Light => "light",
    };
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(p, obj(vec![("theme", str(v)), ("tools", Json::Arr(tools.iter().map(|g| str(g)).collect()))]).to_string());
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
    chunks: usize,
    context: (f64, f64), // tokens used, context length
    start: Args,         // what the command line asked to open or send once connected
    sidebar: bool,       // the chat list shown (it hides with the pane button)
    hovered: Option<String>, // the chat row under the pointer (its delete button shows)
    folder: Option<String>,  // the first folder file tools may use
    tools: Vec<String>,      // the tool groups that are on
    repaint: bool,           // flipped on every change but the ghost's tick: the whole window is drawn again
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
    Approve(String, &'static str), // a call's id, and "once", "always" or "deny"
    Approved(Result<Json, String>),
    Undo(usize),
    Undone(usize, Result<Json, String>),
}

impl App {
    fn new(args: Args) -> (App, Task<Message>) {
        let addr = args.addr.clone();
        let (mode, tools) = load_settings();
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
            context: (0.0, 0.0),
            start: args,
            sidebar: true,
            hovered: None,
            folder: None,
            tools,
            repaint: false,
        };
        (app, Task::perform(blocking(move || connect(&addr)), Message::Connected))
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        if !matches!(message, Message::Tick | Message::Agent(_) | Message::Hover(_)) {
            self.last_active = Instant::now();
        }
        if !matches!(message, Message::Tick) {
            self.repaint = !self.repaint;
        }
        match message {
            Message::Connected(Ok((api, models, chats, harness))) => {
                self.api = Some(api);
                self.folder = harness.get("folders").and_then(Json::as_array).and_then(|f| f.first()).and_then(Json::as_str).map(|f| f.rsplit(['\\', '/']).next().unwrap_or(f).to_string());
                self.set_models(&models);
                self.set_chats(&chats);
                self.status = String::new();
                self.menu = match self.start.menu.take().as_deref() {
                    Some("plus") => Some(Menu::Plus),
                    Some("chat") => Some(Menu::Chat),
                    Some("settings") => Some(Menu::Settings),
                    _ => None,
                };
                if let Some(id) = self.start.open.take() {
                    return self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded);
                }
                if let Some(text) = self.start.send.take() {
                    self.composer = text_editor::Content::with_text(&text);
                    return self.send(None);
                }
            }
            Message::Connected(Err(e)) => self.status = e,
            Message::Chats(Ok(c)) => self.set_chats(&c),
            Message::Chats(Err(e)) | Message::ChatLoaded(Err(e)) | Message::ModelLoaded(Err(e)) => self.status = e,
            Message::OpenChat(id) => {
                if self.busy.is_none() {
                    return self.call(move |api| api.get(&format!("/api/chats/{id}")), Message::ChatLoaded);
                }
            }
            Message::ChatLoaded(Ok(chat)) => self.show_chat(&chat),
            Message::NewChat => {
                if self.busy.is_none() {
                    self.chat = None;
                    self.title = None;
                    self.messages.clear();
                    self.shown = window();
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
                self.status = String::new();
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
            Message::Tick => self.clock = self.born.elapsed().as_millis() as u64,
            Message::Menu(m) => self.menu = m,
            Message::ToggleTheme => {
                self.set_dark(!self.dark);
                self.mode = if self.dark { ThemeMode::Dark } else { ThemeMode::Light };
                save_settings(self.mode, &self.tools);
                self.menu = None;
            }
            Message::SystemTheme => {
                self.mode = ThemeMode::System;
                self.set_dark(windows_dark());
                save_settings(self.mode, &self.tools);
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
                save_settings(self.mode, &self.tools);
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
            Message::Hover(h) => self.hovered = h,
            Message::Copy(i) => {
                if let Some(m) = self.messages.get(i) {
                    return iced::clipboard::write(m.answer_text());
                }
            }
            Message::Link(url) => {
                // open in the default browser
                let _ = std::process::Command::new("cmd").args(["/C", "start", "", &url]).spawn();
            }
        }
        Task::none()
    }

    /// A server call on a worker thread, its result as a message.
    fn call<T: Send + 'static>(&self, f: impl FnOnce(&Api) -> Result<T, String> + Send + 'static, done: impl FnOnce(Result<T, String>) -> Message + Send + 'static) -> Task<Message> {
        let Some(api) = self.api.clone() else { return Task::none() };
        Task::perform(blocking(move || f(&api)), done)
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
        iced::Subscription::run(|| {
            use iced::futures::StreamExt;
            let (tx, rx) = mpsc::unbounded::<()>();
            std::thread::spawn(move || {
                while tx.unbounded_send(()).is_ok() {
                    std::thread::sleep(std::time::Duration::from_millis(110));
                }
            });
            rx.map(|_| Message::Tick)
        })
    }

    /// A saved chat as messages: each user message, then its reply's rounds folded into one.
    fn show_chat(&mut self, chat: &Json) {
        self.chat = chat.get("id").and_then(Json::as_str).map(String::from);
        self.title = chat.get("title").and_then(Json::as_str).map(String::from);
        self.messages.clear();
        self.shown = window();
        for m in chat.get("messages").and_then(Json::as_array).unwrap_or(&[]) {
            let content = m.get("content").and_then(Json::as_str).unwrap_or("");
            match m.get("role").and_then(Json::as_str) {
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
        let mut body = vec![("thinking", Json::Bool(self.thinking)), ("tools", Json::Arr(self.tools.iter().map(|g| str(g)).collect())), ("max_tokens", num(4096.0))];
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
        self.chunks = 0;
        self.status = "Reading".into();
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
            "context" => {
                let f = |k: &str| ev.get(k).and_then(Json::as_f64).unwrap_or(0.0);
                self.context = (f("prompt_tokens") + f("completion_tokens"), f("context_length"));
            }
            "done" => self.finish(&s("finish")),
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
