//! A model download for the app's first run: one GGUF file from Hugging Face into the models folder, through
//! Windows' own curl.exe (no TLS library here). It goes to `<name>.gguf.part` and takes its name once whole,
//! so the model list never shows half a file, and a download that stops resumes from that part.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use crate::json::{num, obj, str, Json};

/// The download running or last finished.
struct Job {
    file: String,
    part: PathBuf,
    total: u64,
    state: &'static str, // "downloading", "done", "failed" or "cancelled"
    error: String,
    child: Option<Child>,
}

static JOB: Mutex<Option<Job>> = Mutex::new(None);

/// Starts downloading `url` (a Hugging Face file) to `dir/file`; `total` is its size in bytes if known, to
/// check it and show progress.
pub fn start(url: &str, file: &str, total: u64, dir: &Path) -> Result<(), String> {
    if !url.starts_with("https://huggingface.co/") {
        return Err("models download from https://huggingface.co/ only".into());
    }
    if !file.to_ascii_lowercase().ends_with(".gguf") || file.contains(['/', '\\', ':']) || file.starts_with('.') {
        return Err(format!("not a model file name: {file}"));
    }
    let mut job = JOB.lock().unwrap();
    if let Some(j) = job.as_ref().filter(|j| j.state == "downloading") {
        return Err(format!("{} is downloading; wait for it or cancel it", j.file));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
    let target = dir.join(file);
    if target.exists() {
        *job = Some(Job { file: file.into(), part: target, total, state: "done", error: String::new(), child: None });
        return Ok(());
    }
    let part = dir.join(format!("{file}.part"));
    use std::os::windows::process::CommandExt;
    let child = Command::new("curl.exe")
        // follow the redirect to the file's host, fail on an HTTP error, retry a dropped connection, and go
        // on from the part already there
        .args(["-L", "--fail", "--silent", "--show-error", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o"])
        .arg(&part)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .creation_flags(0x0800_0000) // no console window
        .spawn()
        .map_err(|e| format!("cannot run curl.exe: {e}"))?;
    *job = Some(Job { file: file.into(), part: part.clone(), total, state: "downloading", error: String::new(), child: Some(child) });
    drop(job);
    std::thread::spawn(move || watch(part, target));
    Ok(())
}

/// Waits for curl, then gives the part its name (if it is whole) or records why it stopped.
fn watch(part: PathBuf, target: PathBuf) {
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let mut guard = JOB.lock().unwrap();
        let Some(job) = guard.as_mut().filter(|j| j.part == part && j.state == "downloading") else { return };
        let Some(child) = job.child.as_mut() else { return };
        let Ok(Some(status)) = child.try_wait() else { continue };
        let mut why = String::new();
        if let Some(mut err) = job.child.take().and_then(|mut c| c.stderr.take()) {
            use std::io::Read;
            let _ = err.read_to_string(&mut why);
        }
        let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if !status.success() {
            job.state = "failed";
            job.error = format!("the download stopped: {}", why.trim().trim_start_matches("curl: "));
        } else if job.total > 0 && got != job.total {
            job.state = "failed";
            job.error = format!("the file came out at {got} bytes, not {}", job.total);
        } else if let Err(e) = std::fs::rename(&part, &target) {
            job.state = "failed";
            job.error = format!("cannot name {}: {e}", target.display());
        } else {
            job.state = "done";
            job.part = target.clone();
        }
        return;
    }
}

/// Stops the running download; its part stays, so starting it again goes on from there.
pub fn cancel() {
    let mut guard = JOB.lock().unwrap();
    if let Some(job) = guard.as_mut().filter(|j| j.state == "downloading") {
        if let Some(mut c) = job.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        job.state = "cancelled";
    }
}

/// {"file", "bytes", "total", "state", "error"?} for the download running or last finished, or null.
pub fn status() -> Json {
    let guard = JOB.lock().unwrap();
    let Some(job) = guard.as_ref() else { return Json::Null };
    let bytes = std::fs::metadata(&job.part).map(|m| m.len()).unwrap_or(0);
    let mut fields = vec![("file", str(&job.file)), ("bytes", num(bytes as f64)), ("total", num(job.total as f64)), ("state", str(job.state))];
    if !job.error.is_empty() {
        fields.push(("error", str(&job.error)));
    }
    obj(fields)
}
