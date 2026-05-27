//! Micro-bench of the §14.3 inverse WHT + §14.4 inverse DCT 4×4
//! primitives — `oxideav_vp8::inverse_wht_4x4` and
//! `oxideav_vp8::inverse_dct_4x4`. These are the cheapest hot blocks in
//! the decode path (the 16 Y + 4 U + 4 V sub-blocks per MB each call
//! `inverse_dct_4x4`; the per-MB Y2 plane calls `inverse_wht_4x4`) so
//! making them visible at criterion's micro-bench resolution gives a
//! clean A/B target for SIMD / unrolled rewrites.
//!
//! The input is a fixed quantised 4×4 block (a representative high-AC
//! pattern) so successive measurements compare apples-to-apples.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::{inverse_dct_4x4, inverse_wht_4x4};

/// Representative quantised 4×4 block — a low-frequency DC-heavy
/// pattern with two non-trivial AC coefficients, sized so the
/// inverse-DCT temporary stays in i32 range with comfortable headroom.
const SAMPLE_INPUT: [i16; 16] = [
    320, -64, 16, -4, //
    -48, 32, -16, 8, //
    24, -12, 8, -4, //
    -8, 4, -2, 1,
];

fn bench_inverse_dct_4x4(c: &mut Criterion) {
    let input = SAMPLE_INPUT;
    let mut out = [0i16; 16];
    let mut g = c.benchmark_group("inverse_transform_4x4");
    g.bench_function("inverse_dct_4x4", |b| {
        b.iter(|| {
            inverse_dct_4x4(black_box(&input), black_box(&mut out));
            black_box(out)
        });
    });
    g.finish();
}

fn bench_inverse_wht_4x4(c: &mut Criterion) {
    let input = SAMPLE_INPUT;
    let mut out = [0i16; 16];
    let mut g = c.benchmark_group("inverse_transform_4x4");
    g.bench_function("inverse_wht_4x4", |b| {
        b.iter(|| {
            inverse_wht_4x4(black_box(&input), black_box(&mut out));
            black_box(out)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_inverse_dct_4x4, bench_inverse_wht_4x4);
criterion_main!(benches);
