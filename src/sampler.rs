//! Next-token selection: greedy, or temperature + top-k + top-p sampling.

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler { temperature, top_k, top_p, rng: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        ((self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Temperature 0: always the most likely token.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let (cand, mass) = self.distribution(logits);
        let mut r = self.next_f32() * mass;
        for &(id, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        cand.last().map(|c| c.0).unwrap_or_else(|| argmax(logits))
    }

    /// Whether to keep `draft`, a token proposed for the position these logits score: Ok, or Err with the
    /// token to take instead. Kept with the probability p(draft) that sampling gives it; otherwise the token
    /// is drawn from the distribution with `draft` left out. Either way the result is distributed exactly as
    /// `sample` would pick (speculative sampling with a draft that is sure of its one token).
    pub fn accept(&mut self, logits: &[f32], draft: u32) -> Result<(), u32> {
        if self.temperature <= 0.0 {
            let best = argmax(logits);
            return if best == draft { Ok(()) } else { Err(best) };
        }
        let (cand, mass) = self.distribution(logits);
        let p = cand.iter().find(|c| c.0 == draft).map_or(0.0, |c| c.1);
        if self.next_f32() * mass < p {
            return Ok(());
        }
        let mut r = self.next_f32() * (mass - p);
        let mut fallback = None;
        for &(id, q) in cand.iter().filter(|c| c.0 != draft) {
            fallback = Some(id);
            r -= q;
            if r <= 0.0 {
                return Err(id);
            }
        }
        // rounding left a sliver over: the last token, or the draft when it was the only candidate
        fallback.map_or(Ok(()), Err)
    }

    /// The tokens sampling can pick, most likely first, with their weights (temperature, then top-k and
    /// top-p), and the sum of the weights.
    fn distribution(&self, logits: &[f32]) -> (Vec<(u32, f32)>, f32) {
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        let k = if self.top_k == 0 { cand.len() } else { self.top_k.min(cand.len()) };
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max = cand[0].1;
        let mut total = 0.0;
        for c in cand.iter_mut() {
            c.1 = ((c.1 - max) / self.temperature).exp();
            total += c.1;
        }
        let mut cum = 0.0;
        let mut cut = cand.len();
        for (i, c) in cand.iter_mut().enumerate() {
            c.1 /= total;
            cum += c.1;
            if cum >= self.top_p {
                cut = i + 1;
                break;
            }
        }
        cand.truncate(cut);
        let mass: f32 = cand.iter().map(|c| c.1).sum();
        (cand, mass)
    }
}

/// The greedy pick: the first maximum (NaN never wins).
fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drafting a fixed token and taking `accept`'s answer picks tokens with the same frequencies as `sample`.
    #[test]
    fn accepting_a_draft_keeps_the_distribution() {
        let logits = [2.0f32, 1.5, 1.0, 0.2, -1.0, -3.0];
        let runs = 200_000;
        let freq = |pick: &mut dyn FnMut(&mut Sampler) -> u32| {
            let mut s = Sampler::new(0.8, 5, 0.95, 7);
            let mut n = [0usize; 6];
            for _ in 0..runs {
                n[pick(&mut s) as usize] += 1;
            }
            n.map(|c| c as f64 / runs as f64)
        };
        let want = freq(&mut |s| s.sample(&logits));
        // a likely draft, an unlikely one, and one top-k or top-p leaves out
        for draft in [0u32, 3, 5] {
            let got = freq(&mut |s| match s.accept(&logits, draft) {
                Ok(()) => draft,
                Err(t) => t,
            });
            for (w, g) in want.iter().zip(&got) {
                assert!((w - g).abs() < 0.006, "draft {draft}: {want:?} vs {got:?}");
            }
        }
        assert_eq!(want[5], 0.0);
    }

    #[test]
    fn greedy_accepts_only_the_best() {
        let mut s = Sampler::new(0.0, 40, 0.9, 1);
        assert_eq!(s.accept(&[0.1, 3.0, 2.0], 1), Ok(()));
        assert_eq!(s.accept(&[0.1, 3.0, 2.0], 2), Err(1));
    }
}
