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
