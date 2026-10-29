//! A small pool of spinning worker threads for short, frequent parallel work: one token's CPU experts, a
//! fraction of a millisecond every layer. Rayon hands such work out through a queue and puts idle workers
//! to sleep, which costs tens of microseconds per call; here the caller writes the job, bumps a counter
//! that the workers are spinning on, does its own share and spins until every worker is done. Workers that
//! see no work for a while sleep until the next job wakes them.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// How long an idle worker spins before it sleeps: longer than the gap between two layers' work.
const SPIN: Duration = Duration::from_millis(2);

type Job = dyn Fn(usize, usize) + Sync;

struct Shared {
    job: UnsafeCell<*const Job>, // the running job (valid until every worker is done with it)
    epoch: AtomicU64,            // bumped once per job
    pending: AtomicUsize,        // workers still running the current job
    sleeping: Vec<AtomicBool>,
    threads: OnceLock<Vec<Thread>>,
}

// SAFETY: `job` is written only by the caller of `run`, before the epoch bump that hands it to the
// workers, and read by them only after they see that bump; `run` returns once they are all done with it.
unsafe impl Sync for Shared {}

pub struct Pool {
    shared: &'static Shared,
    n: usize,
}

/// The pool, with as many threads (the caller's included) as rayon's.
pub fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(rayon::current_num_threads().max(1)))
}

impl Pool {
    fn new(n: usize) -> Pool {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            job: UnsafeCell::new({
                let idle: &'static Job = &|_, _| {};
                idle as *const Job
            }),
            epoch: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            sleeping: (1..n).map(|_| AtomicBool::new(false)).collect(),
            threads: OnceLock::new(),
        }));
        let threads = (1..n)
            .map(|i| thread::Builder::new().name(format!("ghost-pool-{i}")).spawn(move || worker(shared, i, n)).expect("pool thread").thread().clone())
            .collect();
        let _ = shared.threads.set(threads);
        Pool { shared, n }
    }

    /// Runs `f(i, n)` for every thread i of n (the caller is thread 0); returns when all have finished.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        let s = self.shared;
        if self.n == 1 {
            return f(0, 1);
        }
        // SAFETY: the workers use the job only until they decrement `pending`, and this waits for that
        unsafe { *s.job.get() = std::mem::transmute::<*const (dyn Fn(usize, usize) + Sync + '_), *const Job>(f) };
        s.pending.store(self.n - 1, Ordering::SeqCst);
        s.epoch.fetch_add(1, Ordering::SeqCst);
        for (sleeping, t) in s.sleeping.iter().zip(s.threads.get().unwrap()) {
            if sleeping.swap(false, Ordering::SeqCst) {
                t.unpark();
            }
        }
        f(0, self.n);
        while s.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

fn worker(s: &'static Shared, i: usize, n: usize) {
    let mut seen = 0u64;
    loop {
        let mut since = Instant::now();
        let mut spins = 0u32;
        loop {
            let e = s.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            spins = spins.wrapping_add(1);
            if spins % 512 == 0 && since.elapsed() > SPIN {
                // announce the sleep, then look once more: a job posted in between either is seen here
                // or finds the flag set and wakes this thread
                s.sleeping[i - 1].store(true, Ordering::SeqCst);
                if s.epoch.load(Ordering::SeqCst) == seen {
                    thread::park();
                }
                s.sleeping[i - 1].store(false, Ordering::SeqCst);
                since = Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: see `Shared`
        let job = unsafe { &**s.job.get() };
        job(i, n);
        s.pending.fetch_sub(1, Ordering::Release);
    }
}
