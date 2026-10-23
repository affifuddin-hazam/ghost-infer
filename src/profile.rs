//! Opt-in per-operation timing (BETTERLLAMA_PROFILE=1), cheap enough to leave compiled in.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const OPS: [&str; 8] = ["embed", "norm+quant", "qkv matmul", "rope+cache", "attention", "out matmul", "ffn matmul", "logits"];

static TOTALS: [AtomicU64; OPS.len()] = [const { AtomicU64::new(0) }; OPS.len()];
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enabled() -> bool {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| ENABLED.store(std::env::var_os("BETTERLLAMA_PROFILE").is_some(), Ordering::Relaxed));
    ENABLED.load(Ordering::Relaxed)
}

/// Times the closure into bucket `op` when profiling is on.
#[inline]
pub fn time<T>(op: usize, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    TOTALS[op].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

pub fn reset() {
    for t in &TOTALS {
        t.store(0, Ordering::Relaxed);
    }
}

pub fn report(tokens: usize) {
    if !enabled() {
        return;
    }
    let total: u64 = TOTALS.iter().map(|t| t.load(Ordering::Relaxed)).sum();
    eprintln!("per-token time by operation ({tokens} tokens):");
    for (name, t) in OPS.iter().zip(&TOTALS) {
        let ns = t.load(Ordering::Relaxed);
        eprintln!("  {name:<12} {:>8.3} ms  {:>5.1}%", ns as f64 / 1e6 / tokens as f64, 100.0 * ns as f64 / total.max(1) as f64);
    }
    eprintln!("  {:<12} {:>8.3} ms", "total", total as f64 / 1e6 / tokens as f64);
}
