//! Micro-bench of the §15.3 `MBfilter` wide inter-macroblock loop
//! filter primitive — the encoder's heaviest deblock kernel and the
//! partner of the round-170 `loop_filter_normal` bench (which covers
//! only `subblock_filter` + `simple_segment`).
//!
//! `mb_filter` fires on every macroblock-to-macroblock edge — once per
//! left edge and once per top edge of every non-skipped macroblock,
//! i.e. up to two times per MB on luma plus the two chroma analogues
//! (RFC 6386 §15.1, page 87). For a 320×240 frame that is 4×300 = 1 200
//! invocations at minimum, each updating six of the eight straddling
//! pixels with three decaying adjustments (`27/128`, `18/128`, `9/128`
//! of the edge difference per §15.3 page 92). The kernel is therefore
//! noticeably heavier than `subblock_filter`: the wide branch executes
//! three `clamp_s8` + `s2u` pairs versus one inner adjust in
//! `subblock_filter`'s wide branch.
//!
//! The round-170 `loop_filter_normal` bench picked the subblock kernel
//! because it dominates by call count (3 internal edges × 4 sub-MB rows
//! × 2 orientations = 24 invocations per MB on luma alone). But the
//! wall-time picture is two-pronged: `subblock_filter` wins on count,
//! `mb_filter` wins on per-call cost. Future SIMD / unroll work on the
//! deblock path needs an A/B target for *both*.
//!
//! This micro-bench supplies that second pole, plus a head-to-head
//! number against `subblock_filter` on the same 8-pixel segment so a
//! reader can read off the per-call cost ratio directly, and a leaf
//! number for `common_adjust` (the inner §15.2 4-pixel core that both
//! filters call into and that `simple_segment` calls directly).
//!
//! ## Input shape
//!
//! Both `mb_filter` and `subblock_filter` are gated by §15.3
//! `filter_yes` (the four interior-difference + edge metric clamp). To
//! make the bench measure the *adjustment* arithmetic — the part a
//! SIMD rewrite would target — and not the gate falling through, the
//! input is a low-variance 8-pixel ramp `[120, 122, 124, 126, 130, 132,
//! 134, 136]` (matches `loop_filter_normal.rs` so the two benches share
//! an input) with limits set so `filter_yes` accepts and `hev` reports
//! *low* variance — driving `mb_filter` into its three-decaying-step
//! wide branch and `subblock_filter` into its two-pixel-extended branch.
//! A second `mb_filter` variant with `hev_threshold = 0` forces the
//! `hev=true` path so the simple-inner-window fallback is also benched.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::loop_filter::{common_adjust, mb_filter, subblock_filter};

/// Representative low-variance 8-pixel segment — the §15.3
/// `[p3, p2, p1, p0 | q0, q1, q2, q3]` layout. The 4-step interior
/// ramp keeps `filter_yes` enabled at `interior_limit = 4`, the
/// `|p0 - q0| * 2 + |p1 - q1| / 2 = 9` edge metric keeps the gate
/// passing at `edge_limit = 16`, and `|p1 - q1| = 8` exceeds the
/// `hev_threshold = 4` so the high-edge-variance variant fires the
/// outer-tap fallback branch, while `hev_threshold = 16` selects the
/// wide-decaying-adjustment branch.
const SAMPLE_SEG: [u8; 8] = [120, 122, 124, 126, 130, 132, 134, 136];

fn bench_mb_filter(c: &mut Criterion) {
    let mut g = c.benchmark_group("loop_filter_mb_edge");

    // Wide branch — `hev` reports low variance so the three decaying
    // adjustments (27/18/9 weights) all fire.
    g.bench_function("mb_filter_wide", |b| {
        b.iter(|| {
            let mut seg = SAMPLE_SEG;
            // hev_threshold = 16 (so |p1 - q1| = 8 <= 16, low-variance),
            // interior_limit = 4, edge_limit = 16.
            mb_filter(16, 4, 16, black_box(&mut seg), 0);
            black_box(seg)
        });
    });

    // High-variance branch — `hev` reports high variance so the inner
    // 4-pixel `common_adjust` (with outer taps) fires instead. This is
    // the same code path `subblock_filter` takes when hev is true, but
    // without the half-amount p1/q1 carry.
    g.bench_function("mb_filter_hev", |b| {
        b.iter(|| {
            let mut seg = SAMPLE_SEG;
            // hev_threshold = 0 (so anything trips high-variance),
            // interior_limit = 4, edge_limit = 16.
            mb_filter(0, 4, 16, black_box(&mut seg), 0);
            black_box(seg)
        });
    });

    g.finish();
}

fn bench_subblock_filter_partner(c: &mut Criterion) {
    let mut g = c.benchmark_group("loop_filter_mb_edge");

    // Head-to-head partner: same input, same low-variance gate, so the
    // reader can read off the per-call ratio between the wide
    // MB-edge filter and the narrower sub-block filter directly. The
    // round-170 `loop_filter_normal` bench measures `subblock_filter`
    // at `hev_threshold = 4` (high-variance branch); this entry covers
    // the *low*-variance branch (the §15.3 spec's "wider"
    // sub-block variant: `q1 -= a/2`, `p1 += a/2`) so both branches
    // have a number on file.
    g.bench_function("subblock_filter_low_variance", |b| {
        b.iter(|| {
            let mut seg = SAMPLE_SEG;
            subblock_filter(16, 4, 16, black_box(&mut seg), 0);
            black_box(seg)
        });
    });

    g.finish();
}

fn bench_common_adjust(c: &mut Criterion) {
    let mut g = c.benchmark_group("loop_filter_mb_edge");

    // The §15.2 `common_adjust` inner 4-pixel core that every
    // adjustment branch funnels into. `simple_segment` calls it
    // directly; `subblock_filter` and `mb_filter` both call it on the
    // inner `[p1, p0, q0, q1]` window when their gates trip.
    // Benching the leaf in isolation gives future work a baseline for
    // the `clamp_s8` chain (3 calls) + the `s2u` pair that dominate
    // each invocation.
    g.bench_function("common_adjust_outer_taps", |b| {
        b.iter(|| {
            // Inner window only — pass a 4-pixel slice starting at the
            // p1 position of the wider segment. `common_adjust` reads
            // 4 bytes and writes 2.
            let mut win = [122u8, 126, 130, 132];
            common_adjust(true, black_box(&mut win), 0);
            black_box(win)
        });
    });

    g.bench_function("common_adjust_no_outer", |b| {
        b.iter(|| {
            let mut win = [122u8, 126, 130, 132];
            common_adjust(false, black_box(&mut win), 0);
            black_box(win)
        });
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_mb_filter,
    bench_subblock_filter_partner,
    bench_common_adjust,
);
criterion_main!(benches);
