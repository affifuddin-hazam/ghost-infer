//! Drafting by lookup: guesses of the next few tokens copied from earlier in the context, for the model to
//! check in one batch (speculative decoding without a draft model). Agents copy a lot: a file they read
//! comes back as an edit, a tool call repeats with one argument changed, code is quoted in a review.
//! Wherever the last tokens appeared before, what followed them then is a good guess for what follows now.
//!
//! A batch costs more than a one-token step, so guesses that miss make decoding slower. `Pace` times both
//! and drafts only as many guesses as are likely to pay (SuffixDecoding sizes drafts by how they fared);
//! when rounds lose anyway, it stops drafting for a while, then tries again (Crane's fallback).

use std::collections::HashMap;

use crate::cuda::VERIFY_MAX_BATCH;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Tokens a match is extended backwards to, to choose between candidates and size the draft.
const MAX_MATCH: usize = 24;
/// Most recent earlier occurrences of the last two tokens looked at per draft.
const MAX_CANDIDATES: usize = 48;

static OFF: AtomicBool = AtomicBool::new(false);

/// --no-lookup: decode one token at a time (also GHOST_INFER_NO_LOOKUP=1).
pub fn set_off() {
    OFF.store(true, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    !OFF.load(Ordering::Relaxed) && !*ENV.get_or_init(|| std::env::var_os("GHOST_INFER_NO_LOOKUP").is_some())
}

/// The token sequence so far, and where each pair of adjacent tokens occurs in it.
#[derive(Default)]
pub struct Lookup {
    toks: Vec<u32>,
    /// pair (a, b) -> every i with toks[i - 1] == a and toks[i] == b, in increasing order
    at: HashMap<u64, Vec<u32>>,
}

fn pair(a: u32, b: u32) -> u64 {
    (a as u64) << 32 | b as u64
}

impl Lookup {
    /// Makes the sequence `hist`, keeping what it shares with the current one.
    pub fn sync(&mut self, hist: &[u32]) {
        let same = self.toks.iter().zip(hist).take_while(|(a, b)| a == b).count();
        self.truncate(same);
        for &t in &hist[same..] {
            self.push(t);
        }
    }

    pub fn truncate(&mut self, len: usize) {
        while self.toks.len() > len {
            let i = self.toks.len() - 1;
            if i > 0 {
                let key = pair(self.toks[i - 1], self.toks[i]);
                if let Some(v) = self.at.get_mut(&key) {
                    v.pop(); // i is the last entry: positions go in increasing order
                    if v.is_empty() {
                        self.at.remove(&key);
                    }
                }
            }
            self.toks.pop();
        }
    }

    pub fn push(&mut self, t: u32) {
        if let Some(&prev) = self.toks.last() {
            self.at.entry(pair(prev, t)).or_default().push(self.toks.len() as u32);
        }
        self.toks.push(t);
    }

    /// The longest earlier match of the sequence's end (at least its last two tokens; the most recent of
    /// equally long ones): its length, and up to `max` tokens that followed it.
    pub fn draft(&self, max: usize) -> Option<(usize, &[u32])> {
        let n = self.toks.len();
        if n < 3 || max == 0 {
            return None;
        }
        let list = self.at.get(&pair(self.toks[n - 2], self.toks[n - 1]))?;
        let mut best: Option<(usize, usize)> = None; // (match length, position of its last token)
        for &p in list.iter().rev().filter(|&&p| (p as usize) < n - 1).take(MAX_CANDIDATES) {
            let p = p as usize;
            // p >= 1 and the pair matches: extend the match backwards from p - 2 and n - 3
            let len = 2 + (0..MAX_MATCH - 2).take_while(|&j| j + 2 <= p && self.toks[p - 2 - j] == self.toks[n - 3 - j]).count();
            if best.is_none_or(|(l, _)| len > l) {
                best = Some((len, p));
                if len == MAX_MATCH {
                    break;
                }
            }
        }
        let (len, p) = best?;
        Some((len, &self.toks[p + 1..(p + 1 + max).min(n)]))
    }
}

/// When drafting pays, and how much. A round of n tokens costs more than a plain step (on an RTX 2060 Super
/// about 1.35 steps for 4 tokens, 1.5 for 8, 3.5 for 32), so a guess is worth checking only when it is likely
/// to be kept.
/// Pace times plain steps and rounds of each size (kept across replies: they depend on the hardware), and
/// counts how often guesses after matches of each length were kept in this reply (the latest rounds weigh
/// most, so it follows what the model writes now: a copy, then new text); it drafts the number of guesses
/// with the most tokens per second expected, or none when no number beats a plain step. Counts fade back
/// to a hopeful guess, so a kind of match that stopped paying is tried again now and then. And as a safety
/// net, while recent rounds made fewer tokens per second than plain steps would have, drafting stops for a
/// stretch of plain steps (doubling while it keeps losing).
pub struct Pace {
    step_secs: f64,                      // a plain step's time (moving average); 0 until one is timed
    cost: [f64; VERIFY_MAX_BATCH + 1],   // a round of n tokens' time (moving average); 0 until timed
    kept: [f64; BUCKETS],                // guesses kept after matches of each length (fading) ...
    reached: [f64; BUCKETS],             // ... of those checked with every guess before them kept
    made: f64,                           // tokens made by recent rounds (a decaying sum) ...
    spent: f64,                          // ... and the seconds they took
    rounds_seen: usize,
    rest: usize,    // plain steps left before drafting again
    backoff: usize, // the next rest
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
}

/// Match lengths are counted in buckets: 2, 3, 4-5, 6-9, 10-15, 16 and more.
const BUCKETS: usize = 6;
fn bucket(matched: usize) -> usize {
    match matched {
        0..=2 => 0,
        3 => 1,
        4..=5 => 2,
        6..=9 => 3,
        10..=15 => 4,
        _ => 5,
    }
}
/// How likely a guess is kept before anything is known (hopeful, so every bucket gets tried), and how many
/// guesses that hope counts as.
const PRIOR: [f64; BUCKETS] = [0.5, 0.5, 0.55, 0.6, 0.7, 0.8];
const PRIOR_WEIGHT: f64 = 2.0;
/// Per token generated, what is left of the counts; per round after the same length of match, what is left of
/// that bucket's earlier ones.
const FADE: f64 = 0.998;
const RECENT: f64 = 0.8;
/// Expected tokens per second must beat plain steps by this much to draft.
const MARGIN: f64 = 1.05;
/// A round of n tokens in plain steps, until rounds are timed (and the shape used between timed sizes).
fn usual_cost(n: usize) -> f64 {
    // the verify kernels: nearly flat to 8 tokens, then about 0.09 steps a token (RTX 2060 Super, Qwen3 4B / 8B)
    if n <= 8 { 1.15 + 0.045 * (n - 1) as f64 } else { 1.47 + 0.09 * (n - 8) as f64 }
}

const FIRST_REST: usize = 16;
const MAX_REST: usize = 512;
/// Rounds judged together, and how quickly older ones fade from the judgement.
const JUDGE_AFTER: usize = 4;
const DECAY: f64 = 0.85;

impl Default for Pace {
    fn default() -> Pace {
        Pace {
            step_secs: 0.0,
            cost: [0.0; VERIFY_MAX_BATCH + 1],
            kept: [0.0; BUCKETS],
            reached: [0.0; BUCKETS],
            made: 0.0,
            spent: 0.0,
            rounds_seen: 0,
            rest: 0,
            backoff: FIRST_REST,
            rounds: 0,
            proposed: 0,
            accepted: 0,
        }
    }
}

impl Pace {
    /// A new reply: what was kept in the last one, and whether its rounds lost, say little about this one
    /// (the timings stay: they depend on the hardware).
    pub fn new_reply(&mut self) {
        self.kept = [0.0; BUCKETS];
        self.reached = [0.0; BUCKETS];
        self.made = 0.0;
        self.spent = 0.0;
        self.rounds_seen = 0;
        self.rest = 0;
        self.backoff = FIRST_REST;
    }

    /// Whether the next step may be a draft round at all.
    pub fn drafting(&self) -> bool {
        self.rest == 0 && self.step_secs > 0.0
    }

    /// How many of `avail` guesses after a `matched`-token match to check: the number with the most tokens
    /// per second expected, 0 when none beats a plain step.
    pub fn guesses(&self, matched: usize, avail: usize) -> usize {
        if !self.drafting() {
            return 0;
        }
        let b = bucket(matched);
        let q = (self.kept[b] + PRIOR[b] * PRIOR_WEIGHT) / (self.reached[b] + PRIOR_WEIGHT);
        let (mut best, mut best_gain) = (0, MARGIN);
        let (mut made, mut p) = (1.0, 1.0); // tokens a round of k guesses makes on average: 1 + q + ... + q^k
        for k in 1..=avail.min(VERIFY_MAX_BATCH - 1) {
            p *= q;
            made += p;
            let gain = made * self.step_secs / self.cost_of(k + 1);
            if gain > best_gain {
                (best, best_gain) = (k, gain);
            }
        }
        best
    }

    /// A round of n tokens' time: as timed, else scaled from the nearest size timed, else from a plain step.
    fn cost_of(&self, n: usize) -> f64 {
        if self.cost[n] > 0.0 {
            return self.cost[n];
        }
        match (2..=VERIFY_MAX_BATCH).filter(|&m| self.cost[m] > 0.0).min_by_key(|&m| m.abs_diff(n)) {
            Some(m) => self.cost[m] * usual_cost(n) / usual_cost(m),
            None => self.step_secs * usual_cost(n),
        }
    }

    fn fade(&mut self, tokens: usize) {
        let f = FADE.powi(tokens as i32);
        for b in 0..BUCKETS {
            self.kept[b] *= f;
            self.reached[b] *= f;
        }
    }

    /// A plain one-token step took `secs`.
    pub fn step(&mut self, secs: f64) {
        self.step_secs = if self.step_secs == 0.0 { secs } else { 0.9 * self.step_secs + 0.1 * secs };
        self.rest = self.rest.saturating_sub(1);
        self.fade(1);
    }

    /// A round checked `proposed` guesses that followed a `matched`-token match in `secs`, kept `accepted`
    /// of them and so made accepted + 1 tokens.
    pub fn round(&mut self, secs: f64, matched: usize, proposed: usize, accepted: usize) {
        self.rounds += 1;
        self.proposed += proposed;
        self.accepted += accepted;
        let c = &mut self.cost[proposed + 1];
        *c = if *c == 0.0 { secs } else { 0.7 * *c + 0.3 * secs };
        self.fade(accepted + 1);
        let b = bucket(matched);
        self.kept[b] = RECENT * self.kept[b] + accepted as f64;
        self.reached[b] = RECENT * self.reached[b] + (accepted + 1).min(proposed) as f64;

        self.made = DECAY * self.made + (accepted + 1) as f64;
        self.spent = DECAY * self.spent + secs;
        self.rounds_seen += 1;
        if self.rounds_seen < JUDGE_AFTER {
            return;
        }
        let gain = self.made * self.step_secs / self.spent;
        if gain < 1.0 {
            self.rest = self.backoff;
            self.backoff = (2 * self.backoff).min(MAX_REST);
            self.made = 0.0;
            self.spent = 0.0;
            self.rounds_seen = 0;
        } else if gain > 1.2 {
            self.backoff = FIRST_REST;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(toks: &[u32]) -> Lookup {
        let mut l = Lookup::default();
        l.sync(toks);
        l
    }

    #[test]
    fn drafts_what_followed_the_longest_match() {
        // ... 1 2 3 4 5 ... 9 2 3 7 ... 1 2 3: the match "1 2 3" beats the more recent "2 3"
        let l = lookup(&[1, 2, 3, 4, 5, 6, 9, 2, 3, 7, 8, 1, 2, 3]);
        assert_eq!(l.draft(7), Some((3, &[4, 5, 6, 9, 2, 3, 7][..])));
        assert_eq!(l.draft(2), Some((3, &[4, 5][..])));
        // equally long: the most recent
        let l = lookup(&[2, 3, 4, 2, 3, 5, 2, 3]);
        assert_eq!(l.draft(7), Some((2, &[5, 2, 3][..])));
        // no earlier "3 1"
        assert_eq!(lookup(&[1, 2, 3, 4, 3, 1]).draft(7), None);
    }

    #[test]
    fn a_draft_stops_at_the_end_of_the_sequence() {
        // a repeating pattern: the guess runs into the newest tokens
        let l = lookup(&[5, 6, 5, 6, 5, 6]);
        assert_eq!(l.draft(7), Some((4, &[5, 6][..])));
    }

    #[test]
    fn sync_and_truncate_keep_the_index_right() {
        let mut l = lookup(&[1, 2, 3, 4, 1, 2]);
        assert_eq!(l.draft(7), Some((2, &[3, 4, 1, 2][..])));
        l.sync(&[1, 2, 9, 9, 1, 2]); // the tail changed
        assert_eq!(l.draft(7), Some((2, &[9, 9, 1, 2][..])));
        l.truncate(2);
        assert!(l.at.values().all(|v| v.iter().all(|&p| (p as usize) < 2)));
        l.sync(&[1, 2, 7, 1, 2]);
        assert_eq!(l.draft(7), Some((2, &[7, 1, 2][..])));
        assert_eq!(l.at.len(), 3);
    }

    #[test]
    fn pace_drafts_what_pays() {
        let mut p = Pace::default();
        // rounds as the verify kernels time them: 1.15 steps of 10 ms, and 0.05 more per guess
        let secs = |k: usize| 0.0115 + 0.0005 * k as f64;
        assert_eq!(p.guesses(20, 7), 0); // nothing timed yet
        p.step(0.010);
        // a copy in progress: rounds that keep every guess
        for _ in 0..20 {
            p.round(secs(7), 20, 7, 7);
        }
        assert_eq!(p.guesses(20, 7), 7);
        assert_eq!(p.guesses(20, 3), 3);
        // two-token matches get a try, and stop being checked when their guesses miss ...
        let mut tries = 0;
        while p.guesses(2, 7) > 0 {
            let k = p.guesses(2, 7);
            p.round(secs(k), 2, k, 0);
            tries += 1;
        }
        assert!((1..=3).contains(&tries), "{tries}");
        assert_eq!(p.guesses(20, 7), 7);
        // ... until enough tokens later, when the misses have faded
        let mut steps = 0;
        while p.guesses(2, 7) == 0 {
            p.step(0.010);
            steps += 1;
        }
        assert!((20..3000).contains(&steps), "{steps}");
        // a reply that copies, then writes new text: the long matches stop being drafted within a few misses
        let mut misses = 0;
        while p.guesses(20, 7) > 0 {
            let k = p.guesses(20, 7);
            p.round(secs(k), 20, k, 0);
            misses += 1;
        }
        assert!((1..=20).contains(&misses), "{misses}"); // cheap misses: a few more tries before it stops
        // and the next reply starts hopeful again
        p.new_reply();
        assert!(p.guesses(2, 7) > 0 && p.guesses(20, 7) > 0);
    }

    #[test]
    fn pace_rests_when_rounds_lose_and_comes_back() {
        let mut p = Pace::default();
        p.step(0.010);
        // rounds of 12 ms that keep nothing: slower than 10 ms steps
        for _ in 0..JUDGE_AFTER {
            p.round(0.012, 20, 2, 0);
        }
        assert!(!p.drafting());
        for _ in 0..FIRST_REST {
            p.step(0.010);
        }
        assert!(p.drafting());
        assert_eq!((p.rounds, p.proposed, p.accepted), (4, 8, 0));
    }
}
