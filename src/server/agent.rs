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
//!   {"type":"tasks","tasks"}                      the model's task list changed: [{"text","status"}]
//!   {"type":"title","id","title"}                 after a chat's first reply: the title the model gave it
//!   {"type":"nudge","kind","text"}                the harness sent the model back: "check" its work, or keep
//!                                                 going on its task list ("goal")
//!
//! Scheduled tasks and the heartbeat (schedule.rs) run here too, with nobody watching: their events go
//! nowhere, the chat file keeps it all, and the chat is marked unread for the clients.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{chat_prompt, chat_prompt_glm, sse, Event, Job, Kind, Loaded, Part, Shared, ToolCalls, Work};
use crate::json::{num, obj, str, Json};

/// Model turns one reply may take (each tool round is one).
const MAX_ROUNDS: usize = 40;

// ---- tools

/// The tools, by group: what the client turns on. The system group (this computer's specs, the task list
/// and memory) is always there.
const TOOLS: &str = r#"{
 "system": [
  {"type":"function","function":{"name":"get_system_info","description":"Get the specs of the user's computer: CPU, RAM, GPU and VRAM, operating system, free disk space, and what is running on it right now.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"update_tasks","description":"Keep a task list for any job of more than one step; the user sees it pinned above the message box. Send the whole list every time. Mark one task in_progress while you work on it, and done as soon as it is finished.","parameters":{"type":"object","properties":{"tasks":{"type":"array","items":{"type":"object","properties":{"text":{"type":"string","description":"The step, in a few words"},"status":{"type":"string","enum":["pending","in_progress","done"]}},"required":["text","status"]}}},"required":["tasks"]}}},
  {"type":"function","function":{"name":"remember","description":"Save a lasting fact about the user or this computer for future chats: a preference, where their projects live, how a program runs here. Not for details of one task.","parameters":{"type":"object","properties":{"fact":{"type":"string","description":"The fact, in one short sentence"}},"required":["fact"]}}},
  {"type":"function","function":{"name":"forget","description":"Forget a remembered fact that is wrong or no longer true.","parameters":{"type":"object","properties":{"fact":{"type":"string","description":"The fact as it was saved"}},"required":["fact"]}}},
  {"type":"function","function":{"name":"search_chats","description":"Search the user's earlier chats with you, for when they refer to something you talked about before. Returns matching chats with their id, title, date and the lines that match.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Words to look for"}},"required":["query"]}}},
  {"type":"function","function":{"name":"read_chat","description":"Read an earlier chat found with search_chats.","parameters":{"type":"object","properties":{"id":{"type":"string","description":"The chat's id"}},"required":["id"]}}},
  {"type":"function","function":{"name":"schedule_task","description":"Do something later, or again and again, even while the user is away: remind them, check on something, run a job. At that time the prompt is sent to you in this chat, and you work on it with the same tools (only those the user allows without asking). Give exactly one of in_minutes, at, every_minutes or daily_at.","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"What to do then, as an instruction to yourself, with every detail needed (you may not remember this chat by then)"},"name":{"type":"string","description":"A short name for it, a few words"},"in_minutes":{"type":"integer","description":"Once, this many minutes from now"},"at":{"type":"string","description":"Once, at this local time: HH:MM (today, or tomorrow if it has passed) or YYYY-MM-DD HH:MM"},"every_minutes":{"type":"integer","description":"Again and again, this many minutes apart (at least 5)"},"daily_at":{"type":"string","description":"Every day at this local time, HH:MM"},"days":{"type":"string","description":"With daily_at: only on these days, e.g. mon,tue,wed,thu,fri"}},"required":["prompt"]}}},
  {"type":"function","function":{"name":"list_schedules","description":"The scheduled tasks: id, name, when each runs next, and the time now.","parameters":{"type":"object","properties":{},"required":[]}}},
  {"type":"function","function":{"name":"cancel_schedule","description":"Cancel a scheduled task by its id from list_schedules.","parameters":{"type":"object","properties":{"id":{"type":"string","description":"The task's id"}},"required":["id"]}}}
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

/// Loading and installing skills: always offered, so a skill installed during a reply can be used in it and
/// the tool list (part of the cached prompt) does not change.
const SKILL_TOOLS: [&str; 2] = [
    r#"{"type":"function","function":{"name":"use_skill","description":"Load an installed skill's instructions (the installed skills are listed in the system prompt). Call it before a task one of them is for.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"The skill's name"}},"required":["name"]}}}"#,
    r#"{"type":"function","function":{"name":"install_skill","description":"Install a skill (a folder with a SKILL.md and any files it uses) so it can be loaded with use_skill. The source can be a GitHub link (a repository, a folder in one, or a SKILL.md in one), a link to a .zip, or a folder or .zip on this computer. Use this, not git or a download, when the user asks to install or add a skill. When the source holds several skills, the result lists them; call again with skill set to the path of the one wanted, or all.","parameters":{"type":"object","properties":{"source":{"type":"string","description":"The link, folder or .zip"},"skill":{"type":"string","description":"Which skill, when the source holds several: its path from the list, or all"}},"required":["source"]}}}"#,
];

/// The tools of these groups (and the system one), as the chat template's "tools": with the skill tools, and
/// the MCP servers' tools when the mcp group is on.
fn tools_for(groups: &[String]) -> Vec<Json> {
    let all = tool_groups();
    let mut tools: Vec<Json> = std::iter::once("system").chain(groups.iter().map(String::as_str)).filter_map(|g| all.get(g).and_then(Json::as_array)).flatten().cloned().collect();
    tools.extend(SKILL_TOOLS.iter().map(|t| Json::parse(t).expect("valid tool")));
    if groups.iter().any(|g| g == "mcp") {
        tools.extend(super::mcp::tools());
    }
    tools
}

/// The group whose calls need the user's OK, if any.
fn approval_group(name: &str) -> Option<&'static str> {
    match name {
        "write_file" | "edit_file" => Some("edit"),
        "run_command" | "start_process" => Some("cmd"),
        "web_search" | "fetch_url" => Some("web"),
        "install_skill" => Some("skills"),
        n if n.starts_with("mcp__") => Some("mcp"),
        _ => None,
    }
}

/// Told to the model when tools are on: where it may work, how this computer runs commands, and (when it may
/// run them) to check what it changes.
fn tool_prompt(folders: &[PathBuf], can_run: bool) -> String {
    let list: Vec<String> = folders.iter().map(|f| f.display().to_string()).collect();
    let where_ = if list.is_empty() {
        "No folders are allowed yet, so file tools will fail; if you need files, ask the user to add a folder in Settings.".to_string()
    } else {
        format!("You may use files in {} (relative paths start in {}).", list.join(", "), list[0])
    };
    let check = if can_run { " After you change code, check it: run it or its tests with run_command. If something fails, fix it and run it again. Say it is done only once it works, and say how you checked." } else { "" };
    format!("You can use tools to do things on the user's computer. {where_} It runs Windows; commands run in Windows PowerShell. Read a file before you edit it. Unless the task is one quick step, start by writing a short plan with update_tasks (2 to 6 steps), mark the step you start in_progress and each finished one done, and change the plan when the work shows it should change. Save lasting facts with remember. When the user refers to an earlier conversation, look it up with search_chats.{check} Use tools until the task is done, then answer briefly.")
}

/// Files whose changes can be checked by running something (a notes file is not).
const CODE: [&str; 31] = [
    "rs", "py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "go", "java", "kt", "c", "h", "cc", "cpp", "hpp", "cs", "rb", "php", "swift", "lua", "sh", "ps1", "bat", "cmd", "html", "css", "vue", "svelte", "sql", "dart",
];

fn is_code(path: &str) -> bool {
    path.rsplit_once('.').is_some_and(|(_, ext)| CODE.iter().any(|c| c.eq_ignore_ascii_case(ext)))
}

/// A `<tool_call>` block the model wrote that could not be read even mended: it stays in the chat as a call
/// of this name (its arguments the block as written), and its result asks for the call again.
const BAD_CALL: &str = "unreadable_tool_call";
const BAD_CALL_RESULT: &str = "Error: this tool call could not be read, so nothing was run. Send it again inside <tool_call></tool_call> as valid JSON: {\"name\": \"<tool>\", \"arguments\": {...}}, with every brace and bracket closed.";

/// What the harness says when a reply would end with changed code that nothing has run since.
const NUDGE: &str = "(Automatic check) You changed code in this reply and have not run anything since. Check your work now: run the code or its tests with run_command. If something fails, fix it and run it again. If it cannot be run here, or the user asked you not to run it, say so in one line.";

/// The goal loop, when the user turns it on ("keep going until done"; scheduled runs always have it): a
/// reply that stops with steps of its task list still open is sent back to them, up to this many times in a
/// row without a step getting done. A reply to the nudge with no step done is the model's answer to it (the
/// user said to wait, or a step cannot be done), so once is enough. Not on by default: told nothing, a model
/// takes the nudge for the user's word and goes past a "do step 1, then wait for me" about half the time.
const GOAL_NUDGES: usize = 1;

/// What the harness says when a reply would end with its task list unfinished.
fn goal_nudge(open: &[String]) -> String {
    format!("(Automatic check from ghost, not a message from the user) Your reply ended with steps of your task list still open: {}. If you have done them already, mark them done with update_tasks. If the user asked you to stop or wait before them, do not do them: reply in one line that they are waiting for the user. If a step cannot be done here, say why in one line. Otherwise the user's request is not finished: carry on with the next step now.", open.join("; "))
}

/// The task list's open steps, unless the reply is about to close them: when only the step in progress is
/// left, the run marks it done (close_last_task), so nothing is open.
fn open_tasks(chat: &Chat) -> Vec<String> {
    let tasks = chat.tasks.as_array().unwrap_or(&[]);
    let status = |t: &Json| t.get("status").and_then(Json::as_str).unwrap_or("").to_string();
    let open: Vec<&Json> = tasks.iter().filter(|t| status(t) != "done").collect();
    if let [only] = open[..] {
        if status(only) == "in_progress" {
            return Vec::new();
        }
    }
    open.iter().map(|t| t.get("text").and_then(Json::as_str).unwrap_or("").to_string()).collect()
}

/// Put after the user's message while tools are on: told only in the system prompt, a small model often
/// starts working without a plan.
const PLAN_REMINDER: &str = "(Reminder from ghost: unless this takes one quick step, first call update_tasks with a short plan, then keep it up to date as you work.)";

/// The chat as the model reads it for this reply: the user's message with the plan reminder after it. The
/// chat keeps the message as written, so the reminder is never shown or saved.
fn with_reminder(messages: &[Json], at: usize) -> Vec<Json> {
    let mut out = messages.to_vec();
    if let Some(Json::Obj(fields)) = out.get_mut(at) {
        if let Some((_, Json::Str(text))) = fields.iter_mut().find(|(k, _)| k == "content") {
            text.push_str("\n\n");
            text.push_str(PLAN_REMINDER);
        }
    }
    out
}

/// The harness's own messages in a chat (the nudge): not the user's, so retrying and searching pass them by.
fn is_auto(m: &Json) -> bool {
    m.get("auto").and_then(Json::as_bool).unwrap_or(false)
}

// ---- fitting a chat to the context: the chat keeps every message, marked with what the model no longer gets

/// What the model gets in place of a tool's output taken out to make room.
const TRIMMED: &str = "[This output was taken out to make room in the context. Call the tool again if it is needed.]";

/// Put before the first message the model still gets, once earlier ones were left out.
const LEFT_OUT: &str = "(Earlier messages of this chat were left out to make room in the context.)";

fn flag(m: &Json, k: &str) -> bool {
    m.get(k).and_then(Json::as_bool).unwrap_or(false)
}

fn role(m: &Json) -> &str {
    m.get("role").and_then(Json::as_str).unwrap_or("")
}

/// Sets a message's field `k`.
fn mark(m: &mut Json, k: &str, v: Json) {
    if let Json::Obj(fields) = m {
        fields.retain(|(f, _)| f != k);
        fields.push((k.into(), v));
    }
}

/// The chat as the model reads it: messages left out ("dropped") skipped, tool outputs taken out ("trimmed")
/// replaced by a note, and messages cut down ("clip": the characters kept) cut in the middle.
fn for_model(messages: &[Json]) -> Vec<Json> {
    let mut out: Vec<Json> = Vec::with_capacity(messages.len());
    let mut left_out = false;
    for m in messages {
        if flag(m, "dropped") {
            left_out = true;
            continue;
        }
        let (trimmed, clip, note) = (flag(m, "trimmed"), m.get("clip").and_then(Json::as_f64).map(|c| c as usize), left_out && out.is_empty());
        let mut m = m.clone();
        if let (true, Json::Obj(fields)) = (trimmed || clip.is_some() || note, &mut m) {
            if let Some((_, Json::Str(t))) = fields.iter_mut().find(|(k, _)| k == "content") {
                if trimmed {
                    *t = TRIMMED.into();
                } else if let Some(c) = clip {
                    *t = crate::tools::clip(t, c);
                }
                if note {
                    *t = format!("{LEFT_OUT}\n\n{t}");
                }
            }
        }
        out.push(m);
    }
    out
}

/// Makes about `need` tokens of room, the least missed first: tool outputs taken out, oldest first, all but
/// the last 3; then the oldest turns left out whole; then, when the turn being answered is too big on its
/// own, its biggest messages cut in half. Whether it found anything to take out.
fn make_room(chat: &mut Chat, need: usize, tok: &crate::tokenizer::Tokenizer) -> bool {
    // a message's text as the model gets it now, in tokens
    let cost = |m: &Json| -> usize {
        if flag(m, "dropped") {
            return 0;
        }
        if flag(m, "trimmed") {
            return 25;
        }
        let t = m.get("content").and_then(Json::as_str).unwrap_or("");
        let shown = m.get("clip").and_then(Json::as_f64).map(|c| crate::tools::clip(t, c as usize));
        tok.encode(shown.as_deref().unwrap_or(t)).len()
    };
    let msgs = &mut chat.messages;
    let current = msgs.iter().rposition(|m| role(m) == "user" && !is_auto(m)).unwrap_or(0);
    let mut freed = 0;
    let outputs: Vec<usize> = (0..msgs.len()).filter(|&i| role(&msgs[i]) == "tool" && !flag(&msgs[i], "trimmed") && !flag(&msgs[i], "dropped")).collect();
    for &i in &outputs[..outputs.len().saturating_sub(3)] {
        if freed >= need {
            break;
        }
        let c = cost(&msgs[i]);
        if c > 100 {
            mark(&mut msgs[i], "trimmed", Json::Bool(true));
            freed += c - 25;
        }
    }
    // a turn: a user's message and everything up to their next one
    let mut i = 0;
    while freed < need && i < current {
        let end = (i + 1..current).find(|&j| role(&msgs[j]) == "user" && !is_auto(&msgs[j])).unwrap_or(current);
        for j in i..end {
            freed += cost(&msgs[j]);
            mark(&mut msgs[j], "dropped", Json::Bool(true));
        }
        i = end;
    }
    let mut big: Vec<(usize, usize)> = (current..msgs.len()).map(|j| (cost(&msgs[j]), j)).filter(|(c, _)| *c > 1000).collect();
    big.sort_by(|a, b| b.cmp(a));
    for (c, j) in big {
        if freed >= need {
            break;
        }
        let chars = msgs[j].get("content").and_then(Json::as_str).map_or(0, |t| t.chars().count());
        let kept = msgs[j].get("clip").and_then(Json::as_f64).map_or(chars, |c| c as usize);
        mark(&mut msgs[j], "clip", num((kept / 2) as f64));
        freed += c / 2;
    }
    freed > 0
}

/// This round's prompt, made to fit: when it would leave the reply less room than it asks for (up to a
/// quarter of the context), make_room takes out enough that it fills at most 3/4 of the rest, so the next
/// rounds fit too and start the same way (the server's cache still holds them).
fn fitted_prompt(req: &Request, chat: &mut Chat, at: Option<usize>, tools: &[Json], shared: &Shared, loaded: &Loaded) -> Result<String, String> {
    let limit = loaded.ctx - req.max_tokens.min(loaded.ctx / 4);
    for _ in 0..8 {
        let history = match at {
            Some(at) => with_reminder(&chat.messages, at),
            None => chat.messages.clone(),
        };
        let prompt = prompt_for(req, &for_model(&history), tools, req.thinking, shared, loaded)?;
        // a token is at least one byte, so a prompt this short fits without counting
        if prompt.len() <= limit {
            return Ok(prompt);
        }
        let n = loaded.tok.encode(&prompt).len();
        if n <= limit {
            return Ok(prompt);
        }
        if !make_room(chat, n - limit * 3 / 4, &loaded.tok) {
            return Err(format!(
                "this message is too long for the model: it needs {n} tokens, and the context holds {} with {} kept for the reply. Shorten it, or start a new chat",
                loaded.ctx,
                loaded.ctx - limit
            ));
        }
    }
    Err("this chat cannot be made to fit the model's context; start a new chat".into())
}

// ---- memory: facts the model saved, kept in %LOCALAPPDATA%\ghost-infer\memory.json for every chat

/// %LOCALAPPDATA%\ghost-infer, made if missing.
pub(super) fn data_dir() -> Result<PathBuf, String> {
    let d = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer")).ok_or("LOCALAPPDATA is not set")?;
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot make {}: {e}", d.display()))?;
    Ok(d)
}

fn memory_file() -> Result<PathBuf, String> {
    Ok(data_dir()?.join("memory.json"))
}

/// The remembered facts, oldest first.
pub fn memory() -> Vec<String> {
    let text = memory_file().ok().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    let j = Json::parse(&text).unwrap_or(Json::Null);
    j.get("facts").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(String::from).collect()
}

fn save_memory(facts: &[String]) -> Result<(), String> {
    let j = obj(vec![("facts", Json::Arr(facts.iter().map(|f| str(f)).collect()))]);
    std::fs::write(memory_file()?, j.to_string()).map_err(|e| e.to_string())
}

/// Forgets the fact at `index` (the clients' remove button); the facts left.
pub fn forget_at(index: usize) -> Result<Vec<String>, String> {
    let mut facts = memory();
    if index >= facts.len() {
        return Err("no such fact".into());
    }
    facts.remove(index);
    save_memory(&facts)?;
    Ok(facts)
}

/// A fact the user adds by hand (the rail's field); the facts.
pub fn add_fact(fact: &str) -> Result<Vec<String>, String> {
    let fact = fact.trim();
    if fact.is_empty() {
        return Err("the fact is empty".into());
    }
    let mut facts = memory();
    if !facts.iter().any(|f| f.trim().eq_ignore_ascii_case(fact)) {
        facts.push(fact.to_string());
        save_memory(&facts)?;
    }
    Ok(facts)
}

// ---- earlier chats: search_chats and read_chat

/// A chat's text as the model reads it: what the user and the assistant said, without tool calls or thinking.
fn chat_text(c: &Json) -> Vec<(String, String)> {
    c.get("messages")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter_map(|m| {
            let who = match m.get("role").and_then(Json::as_str)? {
                "user" if is_auto(m) => return None,
                "user" => "User",
                "assistant" => "Assistant",
                _ => return None,
            };
            let text = m.get("content").and_then(Json::as_str)?;
            let text = text.rsplit("</think>").next().unwrap_or(text).trim();
            (!text.is_empty()).then(|| (who.to_string(), text.to_string()))
        })
        .collect()
}

/// The search_chats tool: chats whose title or text has the query's words, best first, with matching lines.
fn search_chats(query: &str, current: &str) -> String {
    let words: Vec<String> = query.to_lowercase().split_whitespace().filter(|w| w.len() > 1).map(String::from).collect();
    if words.is_empty() {
        return "Error: give some words to look for".into();
    }
    let Ok(dir) = chats_dir() else { return "Error: the chats cannot be read".into() };
    let mut hits: Vec<(usize, f64, String)> = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let Some(c) = std::fs::read_to_string(e.path()).ok().and_then(|t| Json::parse(&t).ok()) else { continue };
        let id = c.get("id").and_then(Json::as_str).unwrap_or("").to_string();
        if id == current || id.is_empty() {
            continue;
        }
        let title = c.get("title").and_then(Json::as_str).unwrap_or("").to_string();
        let lines = chat_text(&c);
        let score_of = |t: &str| {
            let t = t.to_lowercase();
            words.iter().filter(|w| t.contains(w.as_str())).count()
        };
        let mut score = 3 * score_of(&title);
        let mut matched = Vec::new();
        for (who, text) in &lines {
            for line in text.lines() {
                let n = score_of(line);
                if n > 0 {
                    score += n;
                    if matched.len() < 3 {
                        matched.push(format!("  {who}: {}", crate::tools::clip(line.trim(), 200)));
                    }
                }
            }
        }
        if score > 0 {
            let at = c.get("updated").and_then(Json::as_f64).unwrap_or(0.0);
            let days = ((now_ms() - at) / 86_400_000.0).floor() as i64;
            let when = match days {
                ..=0 => "today".to_string(),
                1 => "yesterday".to_string(),
                d => format!("{d} days ago"),
            };
            hits.push((score, at, format!("- id {id}: \"{title}\" ({when})\n{}", matched.join("\n"))));
        }
    }
    if hits.is_empty() {
        return format!("No earlier chat mentions {query}.");
    }
    hits.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
    let list: Vec<String> = hits.into_iter().take(5).map(|h| h.2).collect();
    format!("{}\n\nRead one with read_chat and its id.", list.join("\n"))
}

/// The read_chat tool: an earlier chat's conversation (its end, when long).
fn read_chat(id: &str) -> String {
    match load_chat(id.trim()) {
        Ok(c) => {
            let title = c.get("title").and_then(Json::as_str).unwrap_or("");
            let text: Vec<String> = chat_text(&c).into_iter().map(|(who, t)| format!("{who}: {t}")).collect();
            crate::tools::clip(&format!("Chat \"{title}\":\n\n{}", text.join("\n\n")), 12000)
        }
        Err(_) => format!("Error: no chat with id {id}"),
    }
}

/// The remember and forget tools; their result for the model.
fn memory_tool(name: &str, args: &Json) -> String {
    let fact = args.get("fact").and_then(Json::as_str).unwrap_or("").trim().to_string();
    if fact.is_empty() {
        return "Error: give the fact".into();
    }
    let mut facts = memory();
    let same = |f: &String| f.trim().eq_ignore_ascii_case(&fact);
    let result = if name == "remember" {
        if facts.iter().any(same) {
            return "Already remembered.".into();
        }
        facts.push(fact);
        "Remembered.".to_string()
    } else {
        let before = facts.len();
        facts.retain(|f| !same(f));
        if facts.len() == before {
            return format!("Error: no remembered fact reads exactly that. The facts are: {}", memory().join(" | "));
        }
        "Forgotten.".to_string()
    };
    match save_memory(&facts) {
        Ok(()) => result,
        Err(e) => format!("Error: {e}"),
    }
}

/// The facts as the system prompt gives them, or nothing.
fn memory_prompt() -> String {
    let facts = memory();
    if facts.is_empty() {
        return String::new();
    }
    format!("What you remember from earlier chats:\n{}", facts.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n"))
}

// ---- instructions: GHOST.md files, read into every reply's system prompt (as Claude Code reads CLAUDE.md)

/// A folder's instructions are its GHOST.md, else a CLAUDE.md or AGENTS.md written for other agents.
const INSTRUCTION_FILES: [&str; 3] = ["GHOST.md", "CLAUDE.md", "AGENTS.md"];
const INSTRUCTIONS_MAX: usize = 12000; // characters, all the files together

/// The instruction files, each with whether it exists: the global GHOST.md, then one per folder the model may
/// use (the GHOST.md it would have when it has none of the three).
pub fn instruction_files(folders: &[PathBuf]) -> Vec<(PathBuf, bool)> {
    let mut files = Vec::new();
    if let Ok(d) = data_dir() {
        let p = d.join("GHOST.md");
        files.push((p.clone(), p.is_file()));
    }
    for f in folders {
        match INSTRUCTION_FILES.iter().map(|n| f.join(n)).find(|p| p.is_file()) {
            Some(p) => files.push((p, true)),
            None => files.push((f.join("GHOST.md"), false)),
        }
    }
    files
}

pub fn instructions_json(folders: &[PathBuf]) -> Json {
    Json::Arr(
        instruction_files(folders)
            .iter()
            .enumerate()
            .map(|(i, (p, exists))| obj(vec![("path", str(&p.display().to_string())), ("exists", Json::Bool(*exists)), ("scope", str(if i == 0 { "global" } else { "folder" }))]))
            .collect(),
    )
}

/// A file's text without its <!-- comments --> (notes for whoever edits it, not for the model).
pub(super) fn without_comments(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("<!--") {
        out.push_str(&rest[..at]);
        rest = rest[at..].find("-->").map_or("", |end| &rest[at + end + 3..]);
    }
    out.push_str(rest);
    out
}

/// What the system prompt says from the instruction files, or nothing.
fn instructions_prompt(folders: &[PathBuf]) -> String {
    let mut parts = Vec::new();
    let mut left = INSTRUCTIONS_MAX;
    for (path, exists) in instruction_files(folders) {
        let text = if exists { std::fs::read_to_string(&path).unwrap_or_default() } else { String::new() };
        let text = without_comments(text.trim_start_matches('\u{feff}'));
        let text = text.trim();
        if text.is_empty() || left < 200 {
            continue;
        }
        let text = crate::tools::clip(text, left);
        left = left.saturating_sub(text.chars().count());
        parts.push(format!("From {}:\n{text}", path.display()));
    }
    if parts.is_empty() {
        return String::new();
    }
    format!("The user's instructions, from their GHOST.md files. Follow them:\n\n{}", parts.join("\n\n"))
}

// ---- runs: how many are going (the API's requests too), when the last one ended, and the chats they hold
// (two runs in one chat would each save their own copy of it)

static RUNNING: AtomicUsize = AtomicUsize::new(0);
static LAST_END: AtomicU64 = AtomicU64::new(0); // milliseconds since 1970

fn busy_chats() -> &'static Mutex<HashSet<String>> {
    static B: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    B.get_or_init(Default::default)
}

/// A run going (or a request to /v1): counted while it lives, and holding its chat once it has one.
pub(super) struct Going {
    chat: Option<String>,
}

impl Going {
    pub(super) fn new() -> Going {
        RUNNING.fetch_add(1, Ordering::SeqCst);
        Going { chat: None }
    }

    /// Takes the chat, waiting while another run holds it; false if `still_wanted` says to give up.
    fn hold(&mut self, id: &str, mut still_wanted: impl FnMut() -> bool) -> bool {
        loop {
            if busy_chats().lock().unwrap().insert(id.to_string()) {
                self.chat = Some(id.to_string());
                return true;
            }
            if !still_wanted() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
}

impl Drop for Going {
    fn drop(&mut self) {
        if let Some(id) = &self.chat {
            busy_chats().lock().unwrap().remove(id);
        }
        LAST_END.store(now_ms() as u64, Ordering::SeqCst);
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Milliseconds since the last run ended, or None while one is going (scheduled runs wait for quiet).
pub fn idle_ms() -> Option<u64> {
    (RUNNING.load(Ordering::SeqCst) == 0).then(|| (now_ms() as u64).saturating_sub(LAST_END.load(Ordering::SeqCst)))
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

pub(super) fn now_ms() -> f64 {
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
            let unread = Json::Bool(c.get("unread").and_then(Json::as_bool).unwrap_or(false));
            let summary = obj(vec![("id", c.get("id").cloned().unwrap_or(Json::Null)), ("title", c.get("title").cloned().unwrap_or(Json::Null)), ("updated", num(at)), ("unread", unread)]);
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

/// The user has seen what a scheduled run added to this chat. Left alone while a run holds the chat (it
/// saves the chat when it ends, unread again).
pub fn mark_read(id: &str) -> Result<(), String> {
    let mut going = Going::new();
    if !going.hold(id, || false) {
        return Ok(());
    }
    let mut chat = Chat::load(id)?;
    if chat.unread {
        chat.unread = false;
        chat.save_as_is()?;
    }
    Ok(())
}

/// A chat in memory: its id, title and messages (OpenAI-style, with a few extra fields the clients read).
struct Chat {
    id: String,
    title: String,
    created: f64,
    updated: f64,
    messages: Vec<Json>,
    tokens: f64, // how much of the context the last round used (the clients show it)
    tasks: Json, // the model's task list (update_tasks), as [{"text","status"}]
    unread: bool, // a scheduled run wrote to it and the user has not opened it since
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
        Chat { id: format!("{:x}-{rnd:04x}", t as u64), title: String::new(), created: t, updated: t, messages: Vec::new(), tokens: 0.0, tasks: Json::Arr(Vec::new()), unread: false }
    }

    fn load(id: &str) -> Result<Chat, String> {
        let j = load_chat(id)?;
        Ok(Chat {
            id: id.to_string(),
            title: j.get("title").and_then(Json::as_str).unwrap_or("").to_string(),
            created: j.get("created").and_then(Json::as_f64).unwrap_or(0.0),
            updated: j.get("updated").and_then(Json::as_f64).unwrap_or(0.0),
            messages: j.get("messages").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default(),
            tokens: j.get("tokens").and_then(Json::as_f64).unwrap_or(0.0),
            tasks: j.get("tasks").cloned().unwrap_or(Json::Arr(Vec::new())),
            unread: j.get("unread").and_then(Json::as_bool).unwrap_or(false),
        })
    }

    fn save(&mut self) -> Result<(), String> {
        self.updated = now_ms();
        self.save_as_is()
    }

    /// Saved without counting as a change (the chat list is newest first, and reading a chat is not news).
    fn save_as_is(&self) -> Result<(), String> {
        let j = obj(vec![
            ("id", str(&self.id)),
            ("title", str(&self.title)),
            ("created", num(self.created)),
            ("updated", num(self.updated)),
            ("tokens", num(self.tokens)),
            ("tasks", self.tasks.clone()),
            ("unread", Json::Bool(self.unread)),
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
    goal: bool,           // keep going until the task list is done (the goal loop)
    instructions: String, // the GHOST.md files as the reply started
    memory: String, // the remembered facts, likewise
    skills: String, // the installed skills, likewise
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
            goal: b.get("keep_going").and_then(Json::as_bool).unwrap_or(false),
            instructions: instructions_prompt(&shared.harness.lock().unwrap().folders.clone()),
            memory: memory_prompt(),
            skills: super::skills::prompt(),
        }
    }
}

/// Where a run's events go: the client that asked for it, or nobody (a scheduled run).
enum Out<'a> {
    Client(&'a mut TcpStream),
    Away,
}

impl Out<'_> {
    fn away(&self) -> bool {
        matches!(self, Out::Away)
    }
}

/// One event to the client; Err when it has gone away.
fn emit(out: &mut Out, ev: Json) -> Result<(), String> {
    match out {
        Out::Client(conn) => sse(conn, &ev.to_string()).map_err(|_| "the client went away".to_string()),
        Out::Away => Ok(()),
    }
}

/// A comment line, to notice a client that went away while the run waits on the user.
fn ping(out: &mut Out) -> bool {
    let Out::Client(conn) = out else { return true };
    let c = ": waiting\n\n";
    write!(conn, "{:x}\r\n{c}\r\n", c.len()).and_then(|_| conn.flush()).is_ok()
}

/// POST /api/agent: runs the agent for one reply, streaming events (see the module comment).
pub fn run(conn: &mut TcpStream, body: &Json, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(), String> {
    let req = Request::parse(body, shared);
    let mut going = Going::new();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    let mut out = Out::Client(conn);
    // a scheduled task answering in this chat: this message goes after its reply
    if let Some(id) = &req.chat {
        if !going.hold(id, || ping(&mut out)) {
            return Err("the client went away".into());
        }
    }
    let result = (|| {
        let mut chat = match &req.chat {
            Some(id) => Chat::load(id)?,
            None => Chat::new(),
        };
        chat.unread = false;
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
                let last_user = chat.messages.iter().rposition(|m| m.get("role").and_then(Json::as_str) == Some("user") && !is_auto(m)).ok_or("this chat has no message to answer")?;
                chat.messages.truncate(last_user + 1);
            }
        }
        chat.save()?;
        let first = req.message.is_some() && chat.messages.iter().filter(|m| m.get("role").and_then(Json::as_str) == Some("user")).count() == 1;
        reply(&mut out, &req, &mut chat, first, work, shared, loaded)
    })();
    if let Err(e) = result {
        let _ = emit(&mut out, obj(vec![("type", str("error")), ("message", str(&e))]));
    }
    let _ = sse(conn, "[DONE]");
    let _ = conn.write_all(b"0\r\n\r\n");
    Ok(())
}

/// One reply in a chat whose message is in place: the model's rounds, then "done" (or "error"), and for a
/// chat's first reply a title. Err only when the chat cannot be saved.
#[allow(clippy::too_many_arguments)]
fn reply(out: &mut Out, req: &Request, chat: &mut Chat, first: bool, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(), String> {
    let checkpoint = format!("{}-{}", chat.id, now_ms() as u64); // this reply's file changes, for undo
    emit(out, obj(vec![("type", str("chat")), ("id", str(&chat.id)), ("title", str(&chat.title)), ("checkpoint", str(&checkpoint))]))?;
    let result = rounds(out, req, chat, &checkpoint, work, shared, loaded);
    chat.save()?;
    let named = first && result.is_ok();
    // the model answered with only its current step left open: that step is done
    if matches!(result, Ok("stop")) && close_last_task(chat) {
        chat.save()?;
        let _ = emit(out, obj(vec![("type", str("tasks")), ("tasks", chat.tasks.clone())]));
    }
    let end = match &result {
        Ok(finish) => obj(vec![("type", str("done")), ("finish", str(finish))]),
        Err(e) => obj(vec![("type", str("error")), ("message", str(e))]),
    };
    let _ = emit(out, end);
    // a chat's first reply: the model names it, after "done" so nobody waits for it
    if named {
        if let Some(title) = name_chat(req, chat, work, shared, loaded) {
            chat.title = title;
            chat.save()?;
            let _ = emit(out, obj(vec![("type", str("title")), ("id", str(&chat.id)), ("title", str(&chat.title))]));
        }
    }
    if out.away() {
        result.map(|_| ())
    } else {
        Ok(())
    }
}

/// What a scheduled run is: the message sent to the model (`text`) and the line the clients show for it.
pub struct Wakeup<'a> {
    pub chat: Option<&'a str>, // the chat to answer in (None, or one that is gone: a new one, titled `title`)
    pub title: &'a str,
    pub text: &'a str,
    pub shown: &'a str,
    pub groups: &'a [String],
    pub thinking: bool,
}

/// A reply nobody is watching (a scheduled task or the heartbeat), saved in its chat and marked unread
/// unless `keep` says the reply is not worth the user's time (then the exchange is taken out of the chat).
/// Returns the chat's id, the reply's text, and whether it was kept.
pub fn run_away(w: &Wakeup, keep: impl Fn(&str) -> bool, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<(String, String, bool), String> {
    let mut going = Going::new();
    let existing = w.chat.filter(|id| load_chat(id).is_ok());
    let mut chat = Chat::new();
    let id = existing.map_or_else(|| chat.id.clone(), String::from);
    if !going.hold(&id, || false) {
        return Err("a reply is going in that chat".into());
    }
    match existing {
        Some(id) => chat = Chat::load(id)?, // as it is now that it is held
        None => chat.title = w.title.to_string(),
    }
    let req = Request {
        chat: Some(chat.id.clone()),
        message: Some(w.text.to_string()),
        system: String::new(),
        thinking: w.thinking,
        temperature: shared.defaults.0,
        max_tokens: shared.defaults.3.max(4096),
        groups: w.groups.to_vec(),
        goal: true, // nobody is there to say "go on"
        instructions: instructions_prompt(&shared.harness.lock().unwrap().folders.clone()),
        memory: memory_prompt(),
        skills: super::skills::prompt(),
    };
    let before = chat.messages.len();
    chat.messages.push(obj(vec![("role", str("user")), ("content", str(w.text)), ("shown", str(w.shown))]));
    chat.save()?;
    reply(&mut Out::Away, &req, &mut chat, false, work, shared, loaded)?;
    let text = chat.messages[before..].iter().rev().find(|m| role(m) == "assistant").and_then(|m| m.get("content").and_then(Json::as_str)).unwrap_or("").to_string();
    let kept = keep(&text);
    if kept {
        chat.unread = true;
    } else {
        chat.messages.truncate(before);
    }
    chat.save()?;
    if chat.messages.is_empty() {
        let _ = delete_chat(&chat.id); // made for this run, and nothing of it kept
    }
    Ok((chat.id.clone(), text, kept))
}

/// The prompt for these messages: the client's system text plus where tools may work, then the chat.
fn prompt_for(req: &Request, history: &[Json], tools: &[Json], thinking: bool, shared: &Shared, loaded: &Loaded) -> Result<String, String> {
    let folders = shared.harness.lock().unwrap().folders.clone();
    let system = [req.system.trim().to_string(), req.instructions.clone(), req.memory.clone(), if tools.is_empty() { String::new() } else { tool_prompt(&folders, tools.iter().any(|t| tool_name(t) == "run_command")) }, if tools.is_empty() { String::new() } else { req.skills.clone() }].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    let mut messages = Vec::new();
    if !system.is_empty() {
        messages.push(obj(vec![("role", str("system")), ("content", str(&system))]));
    }
    messages.extend(history.iter().cloned());
    let mut fields = vec![("messages", Json::Arr(messages)), ("chat_template_kwargs", obj(vec![("enable_thinking", Json::Bool(thinking))]))];
    if !tools.is_empty() {
        fields.push(("tools", Json::Arr(tools.to_vec())));
    }
    let body = obj(fields);
    if loaded.glm { chat_prompt_glm(&body, shared.think_default, loaded.thinking) } else { chat_prompt(&body, loaded.default_system.as_deref(), shared.think_default, loaded.thinking) }
}

/// A short title for a chat after its first reply, written by the model; None if it gave nothing usable.
/// The question goes after the chat itself, so the prompt starts as the reply's did and the server's cache
/// already holds it: only the question is read, and the next message finds the chat still cached.
fn name_chat(req: &Request, chat: &Chat, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Option<String> {
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let mut history = for_model(&chat.messages);
    let ask = "Give this chat a title of 3 to 6 words that says what it is about. Reply with the title only: no quotes, no full stop, no tool calls.";
    history.push(obj(vec![("role", str("user")), ("content", str(ask))]));
    let prompt = prompt_for(req, &history, &tools, false, shared, loaded).ok()?;
    let (tx, events) = mpsc::channel();
    let job = Job { model: loaded.id.clone(), kind: Kind::Chat, prompt, max_tokens: Some(24), temperature: 0.0, top_k: shared.defaults.1, top_p: shared.defaults.2, seed: 1, stop: Vec::new(), lead: "", events: tx };
    work.send(Work::Job(job)).ok()?;
    let mut out = String::new();
    for ev in events {
        match ev {
            Event::Text(t) => out.push_str(&t),
            Event::Done { .. } => break,
            Event::Error(..) => return None,
        }
    }
    // the title alone: no thinking, quotes, markdown or "Title:" around it
    let out = out.rsplit("</think>").next().unwrap_or(&out);
    let line = out.lines().map(str::trim).find(|l| !l.is_empty())?;
    let line = line.trim_start_matches("Title:").trim_start_matches("title:");
    let title = line.trim_matches(|c: char| c == '"' || c == '\'' || c == '*' || c == '#' || c == '`' || c.is_whitespace()).trim_end_matches('.');
    if title.is_empty() || title.contains('<') || title.split_whitespace().count() > 10 {
        return None;
    }
    Some(title.chars().take(80).collect())
}

fn tool_name(t: &Json) -> &str {
    t.get("function").and_then(|f| f.get("name")).and_then(Json::as_str).unwrap_or("")
}

/// The model's rounds for one reply; Ok is how it finished ("stop", "length", or "rounds" at MAX_ROUNDS).
/// A reply about to end with code it changed and nothing run since is sent back, once, to check it; with
/// the goal loop on, one about to end with steps of the task list it keeps in this reply still open is sent
/// back to them.
fn rounds(out: &mut Out, req: &Request, chat: &mut Chat, checkpoint: &str, work: &mpsc::Sender<Work>, shared: &Shared, loaded: &Loaded) -> Result<&'static str, String> {
    let reply_start = chat.messages.len();
    let tools = if req.groups.is_empty() { Vec::new() } else { tools_for(&req.groups) };
    let mut changed = false;
    let can_run = tools.iter().any(|t| tool_name(t) == "run_command");
    let (mut unchecked, mut nudged) = (false, false); // code changed and nothing run since; the nudge sent
    let plans = tools.iter().any(|t| tool_name(t) == "update_tasks");
    let mut planned = false; // the task list was written in this reply (an old one is not this reply's goal)
    let (mut goal_nudges, mut done_then) = (0, 0); // goal nudges in a row, and the steps done at the last one
    let asked = chat.messages[..reply_start].iter().rposition(|m| m.get("role").and_then(Json::as_str) == Some("user") && !is_auto(m));
    for round in 0..MAX_ROUNDS {
        let prompt = fitted_prompt(req, chat, asked.filter(|_| plans), &tools, shared, loaded)?;
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
        let take = |parts: Vec<Part>, out: &mut Out, text: &mut String, calls: &mut Vec<(String, String, String)>| -> Result<(), String> {
            for part in parts {
                match part {
                    Part::Text(t) => {
                        text.push_str(&t);
                        emit(out, obj(vec![("type", str("text")), ("text", str(&t))]))?;
                    }
                    Part::Call { name, arguments } => {
                        let id = format!("call_{}_{}", round, calls.len());
                        emit(out, obj(vec![("type", str("tool_call")), ("id", str(&id)), ("name", str(&name)), ("arguments", str(&arguments))]))?;
                        calls.push((id, name, arguments));
                    }
                    // a call it could not read: kept as a call, so its result tells the model to send it again
                    Part::Bad(raw) => {
                        let id = format!("call_{}_{}", round, calls.len());
                        // what it wrote as a string (the prompt shows calls inside tags of their own)
                        let written = raw.trim_start_matches(ToolCalls::OPEN).trim_end_matches(ToolCalls::CLOSE).trim();
                        let arguments = obj(vec![("text", str(written))]).to_string();
                        emit(out, obj(vec![("type", str("tool_call")), ("id", str(&id)), ("name", str(BAD_CALL)), ("arguments", str(&arguments))]))?;
                        calls.push((id, BAD_CALL.to_string(), arguments));
                    }
                }
            }
            Ok(())
        };
        for ev in events {
            match ev {
                // the client going away drops `events`, which stops the generation
                Event::Text(t) => take(if tools.is_empty() { vec![Part::Text(t)] } else { split.push(&t) }, out, &mut text, &mut calls)?,
                Event::Done { finish: f, prompt_tokens, completion_tokens } => {
                    finish = f;
                    usage = (prompt_tokens, completion_tokens);
                    break;
                }
                Event::Error(_, e) => return Err(e),
            }
        }
        take(split.finish(), out, &mut text, &mut calls)?;
        chat.tokens = (usage.0 + usage.1) as f64;
        emit(out, obj(vec![("type", str("context")), ("prompt_tokens", num(usage.0 as f64)), ("completion_tokens", num(usage.1 as f64)), ("context_length", num(loaded.ctx as f64))]))?;

        // the round as the chat keeps it: what the model wrote, and its calls in OpenAI's form
        let mut msg = vec![("role", str("assistant")), ("content", str(text.trim())), ("checkpoint", str(checkpoint))];
        if !calls.is_empty() {
            let tc = calls.iter().map(|(id, name, args)| obj(vec![("id", str(id)), ("type", str("function")), ("function", obj(vec![("name", str(name)), ("arguments", str(args))]))])).collect();
            msg.push(("tool_calls", Json::Arr(tc)));
        }
        chat.messages.push(obj(msg));
        if calls.is_empty() && unchecked && can_run && !nudged && round + 1 < MAX_ROUNDS {
            nudged = true;
            chat.messages.push(obj(vec![("role", str("user")), ("content", str(NUDGE)), ("auto", Json::Bool(true))]));
            emit(out, obj(vec![("type", str("nudge")), ("kind", str("check")), ("text", str(NUDGE))]))?;
            mark_changed(chat, reply_start, changed);
            chat.save()?;
            continue;
        }
        // the goal loop: steps still open go back to the model, while it keeps getting steps done
        let open = if calls.is_empty() && planned && req.goal { open_tasks(chat) } else { Vec::new() };
        if !open.is_empty() && round + 1 < MAX_ROUNDS {
            let done = chat.tasks.as_array().map_or(0, |t| t.iter().filter(|t| t.get("status").and_then(Json::as_str) == Some("done")).count());
            if done > done_then {
                goal_nudges = 0;
            }
            if goal_nudges < GOAL_NUDGES {
                (goal_nudges, done_then) = (goal_nudges + 1, done);
                let say = goal_nudge(&open);
                chat.messages.push(obj(vec![("role", str("user")), ("content", str(&say)), ("auto", Json::Bool(true)), ("goal", Json::Bool(true))]));
                emit(out, obj(vec![("type", str("nudge")), ("kind", str("goal")), ("text", str(&say))]))?;
                mark_changed(chat, reply_start, changed);
                chat.save()?;
                continue;
            }
        }
        if calls.is_empty() {
            mark_changed(chat, reply_start, changed);
            return Ok(if finish == "length" { "length" } else { "stop" });
        }

        for (id, name, arguments) in calls {
            planned |= name == "update_tasks";
            let (result, denied) = match name.as_str() {
                "update_tasks" => (update_tasks(out, chat, &arguments)?, false),
                BAD_CALL => (BAD_CALL_RESULT.to_string(), false),
                _ => call(out, &id, &name, &arguments, chat, reply_start, checkpoint, req, shared, loaded)?,
            };
            if !denied && approval_group(&name) == Some("edit") && !result.starts_with("Error") {
                changed = true;
                unchecked |= Json::parse(&arguments).ok().and_then(|a| a.get("path").and_then(Json::as_str).map(is_code)).unwrap_or(false);
            }
            if !denied && approval_group(&name) == Some("cmd") {
                unchecked = false;
            }
            emit(out, obj(vec![("type", str("tool_result")), ("id", str(&id)), ("name", str(&name)), ("result", str(&result)), ("denied", Json::Bool(denied))]))?;
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

/// The update_tasks tool: the chat's new task list, sent to the client as a tasks event. Its result for the model.
fn update_tasks(out: &mut Out, chat: &mut Chat, arguments: &str) -> Result<String, String> {
    let args = Json::parse(arguments).unwrap_or(Json::Null);
    let Some(list) = args.get("tasks").and_then(Json::as_array) else {
        return Ok("Error: give tasks, a list of {\"text\", \"status\"}".into());
    };
    let tasks: Vec<Json> = list
        .iter()
        .filter_map(|t| {
            let text = t.get("text").and_then(Json::as_str)?.trim();
            let status = match t.get("status").and_then(Json::as_str).unwrap_or("pending") {
                s @ ("in_progress" | "done") => s,
                _ => "pending",
            };
            (!text.is_empty()).then(|| obj(vec![("text", str(text)), ("status", str(status))]))
        })
        .collect();
    let done = tasks.iter().filter(|t| t.get("status").and_then(Json::as_str) == Some("done")).count();
    let total = tasks.len();
    chat.tasks = Json::Arr(tasks);
    emit(out, obj(vec![("type", str("tasks")), ("tasks", chat.tasks.clone())]))?;
    Ok(format!("Task list updated: {done} of {total} done."))
}

/// Marks the task list's one in_progress task done when every other task is done; whether it did.
fn close_last_task(chat: &mut Chat) -> bool {
    let Json::Arr(tasks) = &mut chat.tasks else { return false };
    let status = |t: &Json| t.get("status").and_then(Json::as_str).unwrap_or("").to_string();
    let open: Vec<usize> = (0..tasks.len()).filter(|&i| status(&tasks[i]) != "done").collect();
    let [i] = open[..] else { return false };
    if status(&tasks[i]) != "in_progress" {
        return false;
    }
    let text = tasks[i].get("text").and_then(Json::as_str).unwrap_or("").to_string();
    tasks[i] = obj(vec![("text", str(&text)), ("status", str("done"))]);
    true
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
fn call(out: &mut Out, id: &str, name: &str, arguments: &str, chat: &Chat, reply_start: usize, checkpoint: &str, req: &Request, shared: &Shared, loaded: &Loaded) -> Result<(String, bool), String> {
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
            // a scheduled run has nobody to ask: only what the user always allows runs in it
            if out.away() {
                return Ok((format!("Not done: this needs the user's OK, and this is a scheduled run with nobody here to give it (they can allow {group} tools always). Do what you can without it, and say in your reply what needs their OK."), true));
            }
            let (tx, rx) = mpsc::channel();
            pending().lock().unwrap().insert(id.to_string(), tx);
            emit(out, obj(vec![("type", str("approval")), ("id", str(id)), ("name", str(name)), ("arguments", str(arguments)), ("group", str(group))]))?;
            let decision = loop {
                match rx.recv_timeout(Duration::from_secs(2)) {
                    Ok(d) => break d,
                    Err(mpsc::RecvTimeoutError::Timeout) if ping(out) => continue,
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
    if name == "use_skill" || name == "install_skill" {
        let args = Json::parse(arguments).unwrap_or(Json::Null);
        let s = |k: &str| args.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        return Ok((if name == "use_skill" { super::skills::use_skill(&s("name")) } else { super::skills::install_tool(&s("source"), &s("skill")) }, false));
    }
    if name == "search_chats" || name == "read_chat" {
        let args = Json::parse(arguments).unwrap_or(Json::Null);
        let s = |k: &str| args.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        return Ok((if name == "search_chats" { search_chats(&s("query"), &chat.id) } else { read_chat(&s("id")) }, false));
    }
    if matches!(name, "schedule_task" | "list_schedules" | "cancel_schedule") {
        let args = Json::parse(arguments).unwrap_or(Json::Null);
        let always = shared.harness.lock().unwrap().always.clone();
        return Ok((super::schedule::tool(name, &args, &chat.id, &req.groups, req.thinking, &always), false));
    }
    if name.starts_with("mcp__") {
        let args = if arguments.trim().is_empty() { Json::Obj(Vec::new()) } else { Json::parse(arguments).unwrap_or(Json::Null) };
        return Ok((super::mcp::call(name, &args), false));
    }
    if name == "remember" || name == "forget" {
        return Ok((memory_tool(name, &Json::parse(arguments).unwrap_or(Json::Null)), false));
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
