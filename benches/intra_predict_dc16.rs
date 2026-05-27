//! Micro-bench of the §12 intra-prediction kernels.
//!
//! `predict_y16x16_dc` is the cheapest 16×16 luma predictor (a single
//! row+column average plus a 256-pixel fill) and is selected on the
//! majority of flat-region macroblocks; making it visible at bench
//! resolution gives the per-MB intra-pick overhead a clean A/B baseline.
//! `predict_y16x16_v` / `predict_y16x16_h` are paired sibling benches so
//! a future memcpy-style optimisation is comparable.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::intra_predict::{predict_y16x16_dc, predict_y16x16_h, predict_y16x16_v};

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
    g.finish();
}

criterion_group!(benches, bench_predict_dc16);
criterion_main!(benches);
