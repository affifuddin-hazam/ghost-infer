//! `ghost-infer monitor [host:port]`: a live view of a running engine's generation speed in the terminal. It
//! asks the server's /api/speed (and /api/progress while a prompt is read) four times a second and draws the
//! speed now in big digits, a scrolling graph of the last half minute or so, and the reply's average and peak.
//! The speed now is measured here, from how many tokens came between two looks (the server's own figure is the
//! reply's average so far). Ctrl+C quits.

use crate::json::Json;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

const STEP: Duration = Duration::from_millis(250);
const ROWS: usize = 12; // graph height in lines (8 levels each)

const GREEN: &str = "\x1b[38;5;48m";
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// One GET to the engine; its JSON body, or None when it does not answer.
fn get(addr: &str, path: &str) -> Option<Json> {
    let sock = addr.to_socket_addrs().ok()?.next()?;
    let mut s = TcpStream::connect_timeout(&sock, Duration::from_millis(500)).ok()?;
    s.set_read_timeout(Some(Duration::from_millis(800))).ok()?;
    write!(s, "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").ok()?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf);
    Json::parse(text.split_once("\r\n\r\n")?.1).ok()
}

/// Big digits, 5 lines high (full blocks), for the speed now.
fn big(text: &str) -> [String; 5] {
    let glyph = |c: char| -> [&str; 5] {
        match c {
            '0' => ["███", "█ █", "█ █", "█ █", "███"],
            '1' => [" █ ", "██ ", " █ ", " █ ", "███"],
            '2' => ["███", "  █", "███", "█  ", "███"],
            '3' => ["███", "  █", "███", "  █", "███"],
            '4' => ["█ █", "█ █", "███", "  █", "  █"],
            '5' => ["███", "█  ", "███", "  █", "███"],
            '6' => ["███", "█  ", "███", "█ █", "███"],
            '7' => ["███", "  █", "  █", "  █", "  █"],
            '8' => ["███", "█ █", "███", "█ █", "███"],
            '9' => ["███", "█ █", "███", "  █", "███"],
            '.' => [" ", " ", " ", " ", "█"],
            _ => ["   ", "   ", "███", "   ", "   "], // "-"
        }
    };
    let mut out: [String; 5] = Default::default();
    for c in text.chars() {
        for (row, part) in out.iter_mut().zip(glyph(c)) {
            row.push_str(part);
            row.push(' ');
        }
    }
    out
}

/// "Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q3_K_M" -> "Qwen3.5 35B A3B Claude 4.6 Opus Q3_K_M"
fn short_model(id: &str) -> String {
    id.replace("-Reasoning-Distilled", "").replace(".i1-", " ").replace('-', " ")
}

/// Pads `s` with spaces to `width` characters (not bytes: the digits are multi-byte blocks).
fn pad(s: &str, width: usize) -> String {
    let n = s.chars().count();
    format!("{s}{}", " ".repeat(width.saturating_sub(n)))
}

pub fn run(addr: &str) -> Result<(), String> {
    os::setup();
    let mut hist: VecDeque<f64> = VecDeque::new();
    let (mut prev_tokens, mut prev_gen, mut prev_at) = (0f64, false, Instant::now());
    let (mut now_rate, mut peak) = (0f64, 0f64);
    let mut out = std::io::stdout();
    loop {
        let t0 = Instant::now();
        let speed = get(addr, "/api/speed");
        let (w, _) = os::size();
        let cols = w.saturating_sub(9).clamp(20, 400);
        let mut f = String::from("\x1b[H");
        let mut line = |s: &str| {
            f.push_str(s);
            f.push_str("\x1b[K\n");
        };
        let Some(s) = speed else {
            line(&format!("{BOLD} GHOST-INFER{RESET}  {DIM}live speed{RESET}"));
            line("");
            line(&format!(" {DIM}waiting for the engine at {addr} (start-engine.bat)...{RESET}"));
            f.push_str("\x1b[J");
            let _ = out.write_all(f.as_bytes());
            let _ = out.flush();
            std::thread::sleep(Duration::from_millis(500));
            continue;
        };
        let n = |k: &str| s.get(k).and_then(Json::as_f64).unwrap_or(0.0);
        let generating = s.get("generating").and_then(Json::as_bool).unwrap_or(false);
        let reading = s.get("reading_prompt").and_then(Json::as_bool).unwrap_or(false);
        let tokens = n("tokens");
        // the speed now: tokens since the last look over the time since it (a new reply starts its count at 0)
        let dt = prev_at.elapsed().as_secs_f64().max(1e-3);
        let inst = if generating && prev_gen && tokens >= prev_tokens { (tokens - prev_tokens) / dt } else { 0.0 };
        if generating && !prev_gen {
            peak = 0.0;
        }
        now_rate = if !generating || inst == 0.0 && !prev_gen { 0.0 } else if now_rate == 0.0 { inst } else { 0.6 * now_rate + 0.4 * inst };
        if generating {
            peak = peak.max(now_rate);
        }
        (prev_tokens, prev_gen, prev_at) = (tokens, generating, Instant::now());
        hist.push_back(if generating { now_rate } else { 0.0 });
        while hist.len() > 400 {
            hist.pop_front();
        }

        // header
        let model = s.get("model").and_then(Json::as_str).map(short_model).unwrap_or_else(|| "no model loaded".into());
        line(&format!("{BOLD} GHOST-INFER{RESET}  {DIM}live speed{RESET}   {CYAN}{model}{RESET}  {DIM}on RTX 2060 SUPER 8 GB{RESET}"));
        line("");
        // the speed now in big digits, its state beside it
        let (shown, color, label) = if generating {
            (now_rate, GREEN, format!("{GREEN}{BOLD}tok/s{RESET}  {DIM}writing{RESET}"))
        } else if reading {
            (0.0, DIM, format!("{CYAN}reading the prompt{RESET}"))
        } else {
            (n("last_tok_s"), DIM, format!("{BOLD}tok/s{RESET}  {DIM}last reply, average{RESET}"))
        };
        let digits = big(&if shown > 0.0 { format!("{shown:.1}") } else { "--".into() });
        for (i, row) in digits.iter().enumerate() {
            let side = match i {
                1 => label.clone(),
                3 if generating => format!("{DIM}reply avg{RESET} {:.1}  {DIM}peak{RESET} {:.1}  {DIM}tokens{RESET} {}", n("tok_s"), peak, tokens as u64),
                3 if reading => match get(addr, "/api/progress") {
                    Some(p) => {
                        let (done, total) = (p.get("done").and_then(Json::as_f64).unwrap_or(0.0), p.get("total").and_then(Json::as_f64).unwrap_or(0.0));
                        if total > 0.0 {
                            let bar = 24;
                            let fill = (((done / total) * bar as f64).round() as usize).min(bar);
                            format!("{CYAN}{}{DIM}{}{RESET} {}/{} tokens", "█".repeat(fill), "░".repeat(bar - fill), done as u64, total as u64)
                        } else {
                            String::new()
                        }
                    }
                    None => String::new(),
                },
                // a finished reply keeps its numbers: the average is the big one, its peak and size beside it
                3 if n("last_tok_s") > 0.0 => format!("{DIM}reply avg{RESET} {:.1}  {DIM}peak{RESET} {:.1}  {DIM}tokens{RESET} {}  {DIM}prompt{RESET} {}", n("last_tok_s"), peak, tokens as u64, n("prompt_tokens") as u64),
                3 => String::new(),
                _ => String::new(),
            };
            line(&format!("  {color}{BOLD}{}{RESET}  {side}", pad(row, 22)));
        }
        line("");

        // the graph: a line through the last `cols` looks, right-aligned, drawn with box-drawing characters (every
        // console font has them; the eighth blocks are missing from Consolas and show as boxes). Row 0 is the floor.
        let top = hist.iter().cloned().fold(100f64, f64::max);
        let ymax = (top / 25.0).ceil() * 25.0;
        let skip = hist.len().saturating_sub(cols);
        let shown: Vec<f64> = hist.iter().skip(skip).cloned().collect();
        let level = |v: f64| ((v / ymax * (ROWS - 1) as f64).round() as usize).min(ROWS - 1);
        let mut grid = vec![vec![' '; cols]; ROWS];
        let x0 = cols - shown.len();
        for (i, &v) in shown.iter().enumerate() {
            let (x, y0) = (x0 + i, level(v));
            let y1 = shown.get(i + 1).map_or(y0, |&w| level(w));
            if y0 == y1 {
                grid[y0][x] = '─';
            } else {
                let (lo, hi) = (y0.min(y1), y0.max(y1));
                for row in grid.iter_mut().take(hi).skip(lo + 1) {
                    row[x] = '│';
                }
                // up: turn up from the left, then right at the top; down: the other way round
                (grid[y0][x], grid[y1][x]) = if y1 > y0 { ('╯', '╭') } else { ('╮', '╰') };
            }
        }
        for r in (0..ROWS).rev() {
            let label = if r == ROWS - 1 {
                format!("{ymax:>5.0}")
            } else if r == (ROWS - 1) / 2 {
                format!("{:>5.0}", ymax * r as f64 / (ROWS - 1) as f64)
            } else if r == 0 {
                "    0".into()
            } else {
                "     ".into()
            };
            let body: String = grid[r].iter().collect();
            line(&format!("{DIM}{label} ┤{RESET}{GREEN}{BOLD}{body}{RESET}"));
        }
        line(&format!("{DIM}      └{}{RESET}", "─".repeat(cols)));
        let span = format!("-{:.0}s", cols as f64 * STEP.as_secs_f64());
        line(&format!("{DIM}       {span}{}now{RESET}", " ".repeat(cols.saturating_sub(span.len() + 3))));
        line("");
        line(&format!("{DIM} Ctrl+C to quit{RESET}"));
        f.push_str("\x1b[J");
        let _ = out.write_all(f.as_bytes());
        let _ = out.flush();
        if let Some(rest) = STEP.checked_sub(t0.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct Info {
        size: [i16; 2],
        cursor: [i16; 2],
        attr: u16,
        window: [i16; 4], // left, top, right, bottom
        max: [i16; 2],
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(n: u32) -> *mut c_void;
        fn GetConsoleMode(h: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: *mut c_void, mode: u32) -> i32;
        fn SetConsoleOutputCP(cp: u32) -> i32;
        fn GetConsoleScreenBufferInfo(h: *mut c_void, info: *mut Info) -> i32;
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }
    const STD_OUTPUT: u32 = -11i32 as u32;

    unsafe extern "system" fn on_ctrl(_: u32) -> i32 {
        print!("\x1b[0m\x1b[?25h\n");
        std::process::exit(0)
    }

    /// ANSI escapes and UTF-8 on, the cursor hidden (shown again on Ctrl+C), the screen cleared.
    pub fn setup() {
        unsafe {
            let h = GetStdHandle(STD_OUTPUT);
            let mut mode = 0;
            if GetConsoleMode(h, &mut mode) != 0 {
                SetConsoleMode(h, mode | 0x0004); // ENABLE_VIRTUAL_TERMINAL_PROCESSING
            }
            SetConsoleOutputCP(65001);
            SetConsoleCtrlHandler(Some(on_ctrl), 1);
        }
        print!("\x1b[2J\x1b[?25l");
    }

    /// The window's columns and lines (80 x 24 when unknown).
    pub fn size() -> (usize, usize) {
        let mut i = Info::default();
        if unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT), &mut i) } != 0 {
            ((i.window[2] - i.window[0] + 1).max(20) as usize, (i.window[3] - i.window[1] + 1).max(10) as usize)
        } else {
            (80, 24)
        }
    }
}

#[cfg(not(windows))]
mod os {
    pub fn setup() {
        print!("\x1b[2J\x1b[?25l");
    }
    pub fn size() -> (usize, usize) {
        let w = std::env::var("COLUMNS").ok().and_then(|v| v.parse().ok()).unwrap_or(80);
        (w, 24)
    }
}
