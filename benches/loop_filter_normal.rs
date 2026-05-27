//! Micro-bench of the §15.3 normal-subblock loop filter primitive.
//!
//! `subblock_filter` is the per-8-pixel-segment normal-filter call that
//! every internal subblock edge of every non-skipped MB fires for. A
//! representative input segment is constructed with a low-edge-variance
//! pattern so the filter actually runs (the §15.3 `filter_yes` gate
//! passes and the §15.3 `hev` test trips into the wide adjustment).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::loop_filter::{simple_segment, subblock_filter};

fn bench_subblock_filter(c: &mut Criterion) {
    // [p3, p2, p1, p0 | q0, q1, q2, q3] — a low-variance ramp the
    // §15.3 `filter_yes` test accepts.
    let seg = [120u8, 122, 124, 126, 130, 132, 134, 136];
    let mut g = c.benchmark_group("loop_filter_normal");
    g.bench_function("subblock_filter_4_4", |b| {
        b.iter(|| {
            let mut window = seg;
            // hev_threshold = 4, interior_limit = 4, edge_limit = 4.
            // The ramp above passes both filter_yes and hev=false so
            // the wide adjustment branch runs.
            subblock_filter(4, 4, 4, black_box(&mut window), 0);
            black_box(window)
        });
    });
    g.bench_function("simple_segment_4", |b| {
        b.iter(|| {
            let mut window = seg;
            simple_segment(4, black_box(&mut window), 2);
            black_box(window)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_subblock_filter);
criterion_main!(benches);
