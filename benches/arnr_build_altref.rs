//! Round 409 — the ARNR temporal-filter altref builder in isolation.
//!
//! [`build_arnr_altref`] synthesises the noise-reduced anchor picture the
//! §9.7 ALTREF slot transports: per 16×16 block of the center frame it
//! runs a whole-pel three-round refinement search (±15 px, step
//! 8 → 4 → 2 → 1) against every other frame in the window, drops blocks
//! whose best SAD stays above the occlusion cutoff, and blends the
//! surviving pixels with a difference-driven weight. Every encoded
//! altref pays this cost once per window, yet the filter had no
//! criterion harness at all — it was only ever exercised through the
//! two-pass / altref integration tests, which assert quality, not
//! wall time.
//!
//! Three workloads bracket the real cost:
//!
//! * `arnr_5f_128x128_static_noise` — a five-frame window of the same
//!   textured scene under independent per-frame LCG noise (the content
//!   ARNR exists for). Motion search converges near the zero MV, every
//!   block survives the cutoff, and the per-pixel blend runs over the
//!   whole frame: this is the steady-state denoise shape.
//! * `arnr_3f_128x128_translating` — a three-frame window translating
//!   four pixels per frame with the same noise. The refinement search
//!   has to walk to a genuine offset before the blend, so the SAD
//!   probe count (and its per-candidate block fetches) dominates.
//! * `arnr_5f_128x128_strength0` — the strength-0 pass-through floor:
//!   plane copies only, no search, no blend. The delta between this
//!   and the static-noise row is the whole filter's marginal cost.
//!
//! Inputs are synthesised in-bench (no committed fixtures) from a
//! deterministic LCG so successive runs measure the identical work.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{build_arnr_altref, ArnrConfig, I420Frame};

const WIDTH: usize = 128;
const HEIGHT: usize = 128;

/// Deterministic LCG noise source (same recurrence the arnr unit tests
/// use), so the bench inputs are fixed across runs and machines.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 33
    }

    fn noise(&mut self, amp: i32) -> i32 {
        (self.next() % (2 * amp as u64 + 1)) as i32 - amp
    }
}

/// A clean textured luma scene plus flat-ish chroma ramps.
fn clean_scene() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (WIDTH, HEIGHT);
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = (64 + ((r * 3 + c * 2) & 0x7f)) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (110 + (c * 24 / cw)) as u8;
            v[r * cw + c] = (126 + (r * 24 / ch)) as u8;
        }
    }
    (y, u, v)
}

/// Add ±amp LCG noise to a plane (seeded per frame).
fn noisy(plane: &[u8], seed: u64, amp: i32) -> Vec<u8> {
    let mut rng = Lcg(seed);
    plane
        .iter()
        .map(|&p| (p as i32 + rng.noise(amp)).clamp(0, 255) as u8)
        .collect()
}

/// Translate a luma plane `dx` pixels to the right (edge-clamped).
fn translated(plane: &[u8], dx: i32) -> Vec<u8> {
    let (w, h) = (WIDTH, HEIGHT);
    let mut out = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let sc = (c as i32 - dx).clamp(0, w as i32 - 1) as usize;
            out[r * w + c] = plane[r * w + sc];
        }
    }
    out
}

/// Owned plane storage for a bench window; `I420Frame` borrows from it.
struct Window {
    planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

impl Window {
    fn frames(&self) -> Vec<I420Frame<'_>> {
        self.planes
            .iter()
            .map(|(y, u, v)| I420Frame::packed(WIDTH as u32, HEIGHT as u32, y, u, v))
            .collect()
    }
}

/// Five same-scene frames under independent noise — the denoise shape.
fn static_noise_window() -> Window {
    let (cy, cu, cv) = clean_scene();
    Window {
        planes: (0..5u64)
            .map(|i| {
                (
                    noisy(&cy, 1000 + i, 6),
                    noisy(&cu, 2000 + i, 3),
                    noisy(&cv, 3000 + i, 3),
                )
            })
            .collect(),
    }
}

/// Three frames translating four pixels per step, plus noise — the
/// motion-search shape.
fn translating_window() -> Window {
    let (cy, cu, cv) = clean_scene();
    Window {
        planes: (0..3i32)
            .map(|i| {
                let shifted = translated(&cy, (i - 1) * 4);
                (
                    noisy(&shifted, 4000 + i as u64, 6),
                    noisy(&cu, 5000 + i as u64, 3),
                    noisy(&cv, 6000 + i as u64, 3),
                )
            })
            .collect(),
    }
}

fn bench_arnr(c: &mut Criterion) {
    let mut g = c.benchmark_group("arnr_build_altref");
    g.throughput(Throughput::Elements((WIDTH * HEIGHT) as u64));
    g.sample_size(20);

    let static_window = static_noise_window();
    let translating = translating_window();

    // Steady-state denoise: 5-frame window, center 2, default strength.
    {
        let frames = static_window.frames();
        let cfg = ArnrConfig::default();
        g.bench_function("arnr_5f_128x128_static_noise", |b| {
            b.iter(|| build_arnr_altref(black_box(&frames), black_box(2), black_box(&cfg)).unwrap())
        });
    }

    // Motion-heavy: 3-frame translating window, center 1.
    {
        let frames = translating.frames();
        let cfg = ArnrConfig::default();
        g.bench_function("arnr_3f_128x128_translating", |b| {
            b.iter(|| build_arnr_altref(black_box(&frames), black_box(1), black_box(&cfg)).unwrap())
        });
    }

    // Strength-0 pass-through floor: plane copies only.
    {
        let frames = static_window.frames();
        let cfg = ArnrConfig::new(0);
        g.bench_function("arnr_5f_128x128_strength0", |b| {
            b.iter(|| build_arnr_altref(black_box(&frames), black_box(2), black_box(&cfg)).unwrap())
        });
    }

    g.finish();
}

criterion_group!(benches, bench_arnr);
criterion_main!(benches);
