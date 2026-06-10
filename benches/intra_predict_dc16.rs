//! Micro-bench of the §12 intra-prediction kernels.
//!
//! `predict_y16x16_dc` is the cheapest 16×16 luma predictor (a single
//! row+column average plus a 256-pixel fill) and is selected on the
//! majority of flat-region macroblocks; making it visible at bench
//! resolution gives the per-MB intra-pick overhead a clean A/B baseline.
//! `predict_y16x16_v` / `predict_y16x16_h` are paired sibling benches so
//! a future memcpy-style optimisation is comparable.
//! `predict_y16x16_tm` (round 268) covers the only §12.2 mode with
//! per-pixel arithmetic (`clamp255(L_r + A_c - P)` over all 256 cells)
//! — the A/B anchor for the `simd` feature's TM row kernel.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::intra_predict::{
    predict_y16x16_dc, predict_y16x16_h, predict_y16x16_tm, predict_y16x16_v,
};

fn bench_predict_dc16(c: &mut Criterion) {
    let above = [128u8; 16];
    let left = [129u8; 16];
    let mut out = [0u8; 256];
    let mut g = c.benchmark_group("intra_predict_dc16");
    g.bench_function("predict_y16x16_dc", |b| {
        b.iter(|| {
            predict_y16x16_dc(
                black_box(&mut out),
                Some(black_box(&above)),
                Some(black_box(&left)),
            );
            black_box(out[0])
        });
    });
    g.bench_function("predict_y16x16_v", |b| {
        b.iter(|| {
            predict_y16x16_v(black_box(&mut out), black_box(&above));
            black_box(out[0])
        });
    });
    g.bench_function("predict_y16x16_h", |b| {
        b.iter(|| {
            predict_y16x16_h(black_box(&mut out), black_box(&left));
            black_box(out[0])
        });
    });
    // Non-flat neighbours so the per-pixel §12.2 TM arithmetic (and
    // its clamp) can't be constant-folded away.
    let above_ramp: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
    let left_ramp: [u8; 16] = core::array::from_fn(|i| 255 - (i * 13) as u8);
    g.bench_function("predict_y16x16_tm", |b| {
        b.iter(|| {
            predict_y16x16_tm(
                black_box(&mut out),
                black_box(&above_ramp),
                black_box(&left_ramp),
                black_box(77),
            );
            black_box(out[0])
        });
    });
    g.finish();
}

criterion_group!(benches, bench_predict_dc16);
criterion_main!(benches);
