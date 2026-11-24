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
        // a thread that runs pool work does its share on core 0, off the workers' cores: on a worker's other
        // hyper-thread both would run at half speed, and the pass waits for the slowest
        CALLER_PINNED.with(|p| {
            if !p.get() {
                p.set(true);
                pin(0, self.n);
            }
        });
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

/// The workers run below normal priority: they spin between a token's layers, and at normal priority that
/// spinning would hold whole cores against the programs the user is working in (and against this
/// process's own expert uploads). GHOST_INFER_POOL_PRIORITY=normal keeps them at normal priority.
fn lower_priority() {
    if std::env::var("GHOST_INFER_POOL_PRIORITY").is_ok_and(|v| v == "normal") {
        return;
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadPriority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
    }
    #[cfg(unix)]
    {
        // PRIO_PROCESS with who=0 nicens the calling thread on Linux (threads are
        // tasks with their own TID). +1 matches Windows' BELOW_NORMAL closely: a
        // single nudge down so the workers yield to interactive processes without
        // being preempted while they spin between layers.
        const PRIO_PROCESS: i32 = 0;
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        }
        unsafe { setpriority(PRIO_PROCESS, 0, 1) };
    }
}

/// Worker i on its own physical core (logical CPU 2i: Windows numbers a core's two hyper-threads next to each
/// other), so two workers never share one: a parallel pass is as slow as its slowest thread. Only when the
/// pool has at most one thread per core; GHOST_INFER_POOL_PIN=0 leaves placement to the OS.
fn pin(i: usize, n: usize) {
    if std::env::var("GHOST_INFER_POOL_PIN").is_ok_and(|v| v == "0") {
        return;
    }
    let logical = thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    if 2 * n > logical {
        return;
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
        }
        if 2 * i >= usize::BITS as usize {
            return;
        }
        // the caller (thread 0) on core 0, both its hyper-threads allowed; workers on cores 1..n
        let mask = if i == 0 { 3usize } else { 1usize << (2 * i) };
        unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
    }
    #[cfg(unix)]
    {
        // Linux picks logical CPU ids however the firmware wants, not the "siblings next
        // to each other" that Windows assumes. Read /sys/devices/system/cpu/.../topology
        // to get a physical-core -> sibling-list map: the caller on all siblings of core
        // 0, each worker on the first sibling of core i.
        let cores = physical_cores();
        let Some(siblings) = cores.get(i) else { return };
        let cpus: &[usize] = if i == 0 { siblings } else { &siblings[..1] };
        unix::set_affinity(cpus);
    }
}

#[cfg(unix)]
mod unix {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    #[link(name = "pthread")]
    unsafe extern "C" {
        fn pthread_self() -> *mut c_void;
        fn pthread_setaffinity_np(thread: *mut c_void, size: usize, set: *const CpuSet) -> i32;
    }

    // glibc's cpu_set_t is 128 bytes (1024 CPU bits). Oversized here is fine; the kernel
    // reads what we pass, nothing more. Any extra bits are zeroed and simply ignored.
    #[repr(C)]
    pub(super) struct CpuSet {
        pub(super) bits: [u64; 16],
    }

    /// Pin the calling thread to the given logical CPU ids.
    pub(super) fn set_affinity(cpus: &[usize]) {
        let mut set = CpuSet { bits: [0; 16] };
        for &c in cpus {
            if c < 64 * set.bits.len() {
                set.bits[c / 64] |= 1u64 << (c % 64);
            }
        }
        let _ = unsafe { pthread_setaffinity_np(pthread_self(), std::mem::size_of::<CpuSet>(), &set) };
    }

    /// First numeric id in a cpu-list file content like "0,8", "0-1,4-5" or "0".
    fn parse_cpu_list(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in s.trim().split(',') {
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    out.extend(a..=b);
                }
            } else if let Ok(a) = part.parse::<usize>() {
                out.push(a);
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// For each physical core on this machine, the list of logical CPU ids that share
    /// it (SMT siblings), in the order the kernel reports them. Cached on first call.
    pub(super) fn physical_cores() -> &'static [Vec<usize>] {
        static CORES: OnceLock<Vec<Vec<usize>>> = OnceLock::new();
        CORES.get_or_init(|| {
            let mut seen: Vec<Vec<usize>> = Vec::new();
            for cpu in 0..4096 {
                let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
                let Ok(s) = std::fs::read_to_string(&path) else { break };
                let siblings = parse_cpu_list(&s);
                if siblings.is_empty() {
                    continue;
                }
                if !seen.iter().any(|existing| existing == &siblings) {
                    seen.push(siblings);
                }
            }
            seen
        })
    }
}

#[cfg(unix)]
pub fn physical_cores() -> &'static [Vec<usize>] {
    unix::physical_cores()
}

thread_local! {
    static CALLER_PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn worker(s: &'static Shared, i: usize, n: usize) {
    lower_priority();
    pin(i, n);
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
