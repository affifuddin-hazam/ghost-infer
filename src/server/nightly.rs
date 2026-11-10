//! The nightly LoRA run (scripts/nightly/run.py): what the chats taught, trained into an adapter for the model
//! that learns. The script does the work; the server starts it once a night, when nobody has used the agent for
//! a while, hands it the tool token (it unloads the models for the training and loads them back), and reports
//! its state. Settings are the script's own `%LOCALAPPDATA%\ghost-infer\nightly\nightly.json`: `enabled`,
//! `hour` (local, 0-23), `python` and `script`, which the script writes on its first run.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::{agent, Shared};
use crate::json::{obj, str, Json};

/// The agent quiet this long before a night's run starts.
const QUIET_MS: u64 = 30 * 60 * 1000;
static RUNNING: AtomicBool = AtomicBool::new(false);

fn home() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("nightly"))
}

fn read(name: &str) -> Json {
    home().and_then(|h| std::fs::read_to_string(h.join(name)).ok()).and_then(|t| Json::parse(&t).ok()).unwrap_or(Json::Null)
}

/// Starts the thread that looks once a minute whether tonight's run is due.
pub fn start(shared: Arc<Shared>) {
    std::thread::spawn(move || {
        let mut last_day = String::new();
        loop {
            std::thread::sleep(Duration::from_secs(60));
            let cfg = read("nightly.json");
            if cfg.get("enabled").and_then(Json::as_bool) != Some(true) || RUNNING.load(Ordering::SeqCst) {
                continue;
            }
            let (day, hour) = super::schedule::local_day_hour();
            let at = cfg.get("hour").and_then(Json::as_f64).unwrap_or(3.0) as u32;
            // its hour or the three after it, once a day, after a quiet half hour
            let ran_today = last_day == day || read("state.json").get("last").and_then(|l| l.get("at")).and_then(Json::as_str).is_some_and(|s| s.starts_with(&day));
            if ran_today || !(at..at + 4).contains(&hour) || agent::idle_ms().is_none_or(|ms| ms < QUIET_MS) {
                continue;
            }
            last_day = day;
            if let Err(e) = launch(&shared) {
                eprintln!("[nightly: {e}]");
            }
        }
    });
}

/// Starts run.py now (the scheduler, or POST /api/nightly/run); its output goes to nightly\server-run.log.
pub fn launch(shared: &Shared) -> Result<(), String> {
    let cfg = read("nightly.json");
    let field = |k: &str| cfg.get(k).and_then(Json::as_str).map(str::to_string).filter(|s| !s.is_empty());
    let (Some(python), Some(script)) = (field("python"), field("script")) else {
        return Err("nightly.json names no python and script yet: run scripts/nightly/run.py once by hand".into());
    };
    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err("a nightly run is going on".into());
    }
    let log = home().map(|h| h.join("server-run.log")).ok_or("LOCALAPPDATA is not set")?;
    let out = std::fs::File::create(&log).map_err(|e| format!("cannot write {}: {e}", log.display()))?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    let mut command = Command::new(&python);
    command.arg("-I").arg(&script).env("GHOST_INFER_TOKEN", &shared.token).stdin(Stdio::null()).stdout(out).stderr(err);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // no console window
    }
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            RUNNING.store(false, Ordering::SeqCst);
            return Err(format!("cannot start {python} {script}: {e}"));
        }
    };
    eprintln!("[nightly: started {script}; its log is {}]", log.display());
    std::thread::spawn(move || {
        let status = child.wait();
        RUNNING.store(false, Ordering::SeqCst);
        eprintln!("[nightly: finished ({})]", status.map(|s| s.to_string()).unwrap_or_else(|e| e.to_string()));
    });
    Ok(())
}

/// Changes the settings a client may change: {"enabled"?: bool, "hour"?: 0-23}. The rest of nightly.json (the
/// script's own settings) is kept as it is.
pub fn set(b: &Json) -> Result<Json, String> {
    let path = home().ok_or("LOCALAPPDATA is not set")?.join("nightly.json");
    let mut fields: Vec<(String, Json)> = match read("nightly.json") {
        Json::Obj(f) => f,
        _ => Vec::new(),
    };
    let mut put = |k: &str, v: Json| match fields.iter_mut().find(|(n, _)| n == k) {
        Some(f) => f.1 = v,
        None => fields.push((k.to_string(), v)),
    };
    if let Some(v) = b.get("enabled") {
        put("enabled", Json::Bool(v.as_bool().ok_or("enabled is true or false")?));
    }
    if let Some(v) = b.get("hour") {
        let h = v.as_f64().filter(|h| h.fract() == 0.0 && (0.0..=23.0).contains(h)).ok_or("hour is 0 to 23")?;
        put("hour", Json::Num(h));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, Json::Obj(fields).to_string()).map_err(|e| format!("cannot save {}: {e}", path.display()))?;
    Ok(status())
}

fn jsonl(name: &str) -> Vec<Json> {
    home().and_then(|h| std::fs::read_to_string(h.join(name)).ok()).map(|t| t.lines().filter_map(|l| Json::parse(l.trim()).ok()).collect()).unwrap_or_default()
}

fn ms_of(t: std::io::Result<std::time::SystemTime>) -> f64 {
    t.ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as f64).unwrap_or(0.0)
}

/// What the model has learned: every fact it knows now (the newest version of each, not retired or forgotten),
/// with the title of the chat it came from, and every night, newest first, with its numbers and the facts it
/// read that night (a run reads the chats first, so a fact belongs to the run that started before it).
pub fn learned() -> Json {
    let rows = jsonl("facts.jsonl");
    let mut latest: Vec<Json> = Vec::new();
    for r in &rows {
        let name = r.get("name").and_then(Json::as_str).unwrap_or("");
        match latest.iter_mut().find(|f| f.get("name").and_then(Json::as_str) == Some(name)) {
            Some(f) => *f = r.clone(),
            None => latest.push(r.clone()),
        }
    }
    let chats = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("ghost-infer").join("chats"));
    let title = |id: &str| -> String {
        chats.as_ref().and_then(|c| std::fs::read_to_string(c.join(format!("{id}.json"))).ok()).and_then(|t| Json::parse(&t).ok()).and_then(|j| j.get("title").and_then(Json::as_str).map(String::from)).unwrap_or_default()
    };
    let text = |r: &Json, k: &str| r.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    let mut facts: Vec<Json> = latest
        .iter()
        .filter(|f| !text(f, "fact").is_empty())
        .map(|f| {
            obj(vec![
                ("name", str(&text(f, "name"))),
                ("type", str(&text(f, "type"))),
                ("fact", str(&text(f, "fact"))),
                ("chat", str(&text(f, "chat"))),
                ("chat_title", str(&title(&text(f, "chat")))),
                ("at", Json::Num(f.get("at").and_then(Json::as_f64).unwrap_or(0.0))),
            ])
        })
        .collect();
    facts.sort_by(|a, b| b.get("at").and_then(Json::as_f64).unwrap_or(0.0).total_cmp(&a.get("at").and_then(Json::as_f64).unwrap_or(0.0)));

    // the nights: each run's folder (made when it started) and its result
    let mut runs: Vec<(String, f64, Json)> = home()
        .and_then(|h| std::fs::read_dir(h.join("runs")).ok())
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| {
                    let started = std::fs::metadata(e.path()).map(|m| ms_of(m.created())).unwrap_or(0.0);
                    let result = std::fs::read_to_string(e.path().join("result.json")).ok().and_then(|t| Json::parse(&t).ok()).unwrap_or(Json::Null);
                    (e.file_name().to_string_lossy().into_owned(), started, result)
                })
                .collect()
        })
        .unwrap_or_default();
    runs.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut nights = Vec::new();
    for (i, (stamp, started, result)) in runs.iter().enumerate() {
        let until = runs.get(i + 1).map(|r| r.1).unwrap_or(f64::MAX);
        let at = |r: &Json| r.get("at").and_then(Json::as_f64).unwrap_or(0.0);
        let read: Vec<&Json> = rows.iter().filter(|r| (*started..until).contains(&at(r)) && !text(r, "fact").is_empty()).collect();
        let checks = result.get("tool_checks").and_then(Json::as_array).map(|c| c.to_vec()).unwrap_or_default();
        let passed = checks.iter().filter(|c| c.get("base").is_some() && c.get("base") == c.get("adapter")).count();
        let mut fields = vec![
            ("stamp", str(stamp)),
            ("started", Json::Num(*started)),
            ("learned", Json::Arr(read.iter().map(|r| obj(vec![("name", str(&text(r, "name"))), ("fact", str(&text(r, "fact")))])).collect())),
            ("installed", Json::Bool(result.get("ok").and_then(Json::as_bool) == Some(true))),
            ("tools", if checks.is_empty() { Json::Null } else { obj(vec![("passed", Json::Num(passed as f64)), ("of", Json::Num(checks.len() as f64))]) }),
        ];
        for k in ["reason", "facts", "fact_loss", "general_loss", "minutes", "tools_ok"] {
            if let Some(v) = result.get(k) {
                fields.push((k, v.clone()));
            }
        }
        nights.push(obj(fields));
    }
    nights.reverse();
    obj(vec![("facts", Json::Arr(facts)), ("nights", Json::Arr(nights))])
}

/// {"name"}: the fact is retired as forgotten (its text kept as "was", so a later read of its chat does not
/// learn it again), and the next night trains without it.
pub fn forget(b: &Json) -> Result<Json, String> {
    let name = b.get("name").and_then(Json::as_str).ok_or("send {\"name\": ...}")?;
    let latest = jsonl("facts.jsonl").into_iter().filter(|r| r.get("name").and_then(Json::as_str) == Some(name)).last().ok_or("no such fact")?;
    let was = latest.get("fact").and_then(Json::as_str).filter(|f| !f.is_empty()).ok_or("that fact is already forgotten")?;
    let now = ms_of(Ok(std::time::SystemTime::now()));
    let row = obj(vec![
        ("name", str(name)),
        ("type", latest.get("type").cloned().unwrap_or(Json::Null)),
        ("fact", str("")),
        ("forgotten", Json::Bool(true)),
        ("was", str(was)),
        ("chat", latest.get("chat").cloned().unwrap_or(Json::Null)),
        ("at", Json::Num(now)),
        ("v", Json::Num(latest.get("v").and_then(Json::as_f64).unwrap_or(0.0) + 1.0)),
    ]);
    let path = home().ok_or("LOCALAPPDATA is not set")?.join("facts.jsonl");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().append(true).create(true).open(&path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    writeln!(f, "{}", row.to_string()).map_err(|e| e.to_string())?;
    Ok(learned())
}

/// The settings, whether a run is going on, and the last run's result.
pub fn status() -> Json {
    let st = read("state.json");
    obj(vec![
        ("running", Json::Bool(RUNNING.load(Ordering::SeqCst))),
        ("settings", read("nightly.json")),
        ("last", st.get("last").cloned().unwrap_or(Json::Null)),
        ("home", str(&home().map(|h| h.display().to_string()).unwrap_or_default())),
    ])
}
