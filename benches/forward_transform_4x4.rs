//! Micro-bench of the §14.3 forward WHT + §14.4 forward DCT 4×4
//! encoder primitives — `oxideav_vp8::forward_wht_4x4` and
//! `oxideav_vp8::forward_dct_4x4`.
//!
//! The forward transforms are the encoder partners of the §14.3 /
//! §14.4 inverse primitives benched in `inverse_transform_4x4.rs`,
//! but they fire on *every* coded residual block during encode — once
//! per Y / U / V 4×4 sub-block (24 calls per MB for the forward DCT
//! plus one call per MB for the Y2 forward WHT on any MB that emits
//! a luma DC plane). The inverse partners then re-fire on decode +
//! every reference-frame reconstruction pass, but for a one-shot
//! encode the forward path is the heavier of the two.
//!
//! No bench has covered the forward primitives directly. The
//! round-170 `keyframe_encode` whole-frame bench wraps them inside
//! the §11 intra picker + §13 token emit + §15 loop-filter cascade
//! and so cannot attribute a wall-time delta to a forward-transform
//! rewrite — this micro-bench gives a stable A/B target for future
//! `core::simd` / unroll work parallel to the round-170 / round-180
//! inverse-side SIMD rewrites.
//!
//! Input is the same representative 4×4 block the inverse micro-bench
//! uses so successive measurements on the forward and inverse passes
//! compare directly.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::{forward_dct_4x4, forward_wht_4x4};

/// Representative pre-quant 4×4 residual block — a low-frequency
/// DC-heavy pattern with two non-trivial AC coefficients. Sized so
/// the forward-pass 16-bit intermediates stay in i32 range with
/// comfortable headroom (matches `inverse_transform_4x4.rs` so the
/// two micro-benches share an input).
const SAMPLE_INPUT: [i16; 16] = [
    320, -64, 16, -4, //
    -48, 32, -16, 8, //
    24, -12, 8, -4, //
    -8, 4, -2, 1,
];

fn bench_forward_dct_4x4(c: &mut Criterion) {
    let input = SAMPLE_INPUT;
    let mut out = [0i16; 16];
    let mut g = c.benchmark_group("forward_transform_4x4");
    g.bench_function("forward_dct_4x4", |b| {
        b.iter(|| {
            forward_dct_4x4(black_box(&input), black_box(&mut out));
            black_box(out)
        });
    });
    g.finish();
}

fn bench_forward_wht_4x4(c: &mut Criterion) {
    let input = SAMPLE_INPUT;
    let mut out = [0i16; 16];
    let mut g = c.benchmark_group("forward_transform_4x4");
    g.bench_function("forward_wht_4x4", |b| {
        b.iter(|| {
            forward_wht_4x4(black_box(&input), black_box(&mut out));
            black_box(out)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_forward_dct_4x4, bench_forward_wht_4x4);
criterion_main!(benches);
