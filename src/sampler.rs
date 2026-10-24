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
        let argmax = || logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        if self.temperature <= 0.0 {
            return argmax();
        }
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

        let mut r = self.next_f32() * mass;
        for &(id, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        cand.last().map(|c| c.0).unwrap_or_else(argmax)
    }
}
