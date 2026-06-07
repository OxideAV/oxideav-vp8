//! Micro-bench of the §17.1 / §18.3 encoder-side luma motion search —
//! `oxideav_vp8::motion_search::small_diamond_search_luma`,
//! `half_pixel_refine_luma`, and `quarter_pixel_refine_luma`.
//!
//! The three functions are the encoder's per-MB motion-vector picker
//! laid out as a quarter-pixel descent ladder: the whole-pixel diamond
//! collapses the search window to one whole-pixel grid point, the
//! half-pixel refine then probes the 8-neighbour half-pixel ring around
//! that point, and the quarter-pixel refine finally probes the 8
//! quarter-pixel offsets around the half-pixel pick. Every encoded
//! inter MB walks the full ladder (P-frames + the §17 ALT-REF / GOLDEN
//! probes the `Vp8TwoPassEncoder` issues per scene-cut analysis) so the
//! ladder is one of the hottest §18 paths in the encoder.
//!
//! No existing bench attributes wall-time to the three stages: the
//! round-170 `inter_encode_short_clip` bench wraps them inside the §11
//! mode picker + §13 token emit + §15 loop-filter cascade and so cannot
//! isolate a delta from a future descent-shape rewrite (hex, full,
//! SIMD-fanout SAD), and the round-170 `motion_comp_subpel_luma` bench
//! sits one layer *below* (the §18.3 sixtap kernel only). This bench
//! gives a stable A/B target for the §17 search-shape layer the round-194
//! / round-220 rate-control + forward-transform sweeps stopped short of.
//!
//! Inputs are deterministic 64×64 luma planes with a mixed-frequency
//! gradient pattern — the same shape `motion_comp_subpel_luma.rs` uses
//! — so the per-MB SAD landscape is non-trivial and the descent makes a
//! non-zero number of probes per stage. The MB is placed well inside
//! the plane so the §20.14 edge-replication clamp stays cold.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::motion_search::{
    half_pixel_refine_luma, quarter_pixel_refine_luma, small_diamond_search_luma, LumaRef,
};
use oxideav_vp8::motion_vector::Mv;

/// Build an 8-bit luma plane with a deterministic mixed-frequency
/// gradient. The pattern is intentionally non-monotonic so the
/// per-candidate 16×16 SAD landscape has more than one local minimum
/// and the small-diamond descent makes a measurable number of probes.
fn make_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut buf = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let mix = (r as u32 * 7)
                .wrapping_add(c as u32 * 11)
                .wrapping_add(seed.wrapping_mul(13))
                ^ (r as u32 * c as u32);
            buf[r * w + c] = mix as u8;
        }
    }
    buf
}

/// Lift the 16×16 sub-rect at `(blk_x, blk_y)` of `plane` into a
/// dense `[u8; 256]` source block (the input shape the motion-search
/// API consumes — `src_y` is always packed row-major, not strided).
fn source_block(plane: &[u8], stride: usize, blk_x: usize, blk_y: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    for r in 0..16 {
        let row_off = (blk_y + r) * stride + blk_x;
        out[r * 16..r * 16 + 16].copy_from_slice(&plane[row_off..row_off + 16]);
    }
    out
}

fn bench_small_diamond(c: &mut Criterion) {
    let w = 64;
    let h = 64;
    // The source frame and the reference frame share a gradient shape
    // but use distinct seeds so the SAD landscape is non-degenerate
    // (a self-referent search would collapse to MV = (0, 0) on the
    // first probe).
    let src_plane = make_plane(w, h, 1);
    let ref_plane = make_plane(w, h, 2);
    let stride = w;

    // Place the MB at (col, row) = (1, 1) — i.e. (blk_x, blk_y) = (16, 16) —
    // so the search window stays clear of the §20.14 boundary clamp.
    let mb_col = 1;
    let mb_row = 1;
    let src_y = source_block(&src_plane, stride, mb_col * 16, mb_row * 16);

    let reference = LumaRef {
        plane: &ref_plane,
        stride,
        width: w,
        height: h,
    };

    let center = Mv { row: 0, col: 0 };

    let mut g = c.benchmark_group("motion_search_descent");
    g.bench_function("small_diamond_search_luma_iters_8", |b| {
        b.iter(|| {
            let r = small_diamond_search_luma(
                black_box(reference),
                black_box(mb_col),
                black_box(mb_row),
                black_box(&src_y),
                black_box(center),
                /* max_iters = */ 8,
            );
            black_box(r)
        });
    });
    g.finish();
}

fn bench_half_pixel_refine(c: &mut Criterion) {
    let w = 64;
    let h = 64;
    let src_plane = make_plane(w, h, 1);
    let ref_plane = make_plane(w, h, 2);
    let stride = w;

    let mb_col = 1;
    let mb_row = 1;
    let src_y = source_block(&src_plane, stride, mb_col * 16, mb_row * 16);

    let reference = LumaRef {
        plane: &ref_plane,
        stride,
        width: w,
        height: h,
    };

    // Pre-resolve the whole-pixel center so the half-pixel bench
    // measures the refine step in isolation (not refine + a fresh
    // diamond descent on every iteration).
    let whole_pixel_center =
        small_diamond_search_luma(reference, mb_col, mb_row, &src_y, Mv { row: 0, col: 0 }, 8).mv;

    let mut g = c.benchmark_group("motion_search_descent");
    g.bench_function("half_pixel_refine_luma_8_offsets", |b| {
        b.iter(|| {
            let r = half_pixel_refine_luma(
                black_box(reference),
                black_box(mb_col),
                black_box(mb_row),
                black_box(&src_y),
                black_box(whole_pixel_center),
            );
            black_box(r)
        });
    });
    g.finish();
}

fn bench_quarter_pixel_refine(c: &mut Criterion) {
    let w = 64;
    let h = 64;
    let src_plane = make_plane(w, h, 1);
    let ref_plane = make_plane(w, h, 2);
    let stride = w;

    let mb_col = 1;
    let mb_row = 1;
    let src_y = source_block(&src_plane, stride, mb_col * 16, mb_row * 16);

    let reference = LumaRef {
        plane: &ref_plane,
        stride,
        width: w,
        height: h,
    };

    // Run whole- + half-pixel stages once outside the timed loop so
    // the quarter-pixel bench attributes wall-time only to the
    // quarter-pixel ring.
    let whole_pixel_center =
        small_diamond_search_luma(reference, mb_col, mb_row, &src_y, Mv { row: 0, col: 0 }, 8).mv;
    let half_pixel_center =
        half_pixel_refine_luma(reference, mb_col, mb_row, &src_y, whole_pixel_center).mv;

    let mut g = c.benchmark_group("motion_search_descent");
    g.bench_function("quarter_pixel_refine_luma_8_offsets", |b| {
        b.iter(|| {
            let r = quarter_pixel_refine_luma(
                black_box(reference),
                black_box(mb_col),
                black_box(mb_row),
                black_box(&src_y),
                black_box(half_pixel_center),
            );
            black_box(r)
        });
    });
    g.finish();
}

fn bench_full_descent(c: &mut Criterion) {
    // Composite of all three stages — the wall-time number a caller
    // pays for the entire §17 / §18.3 luma MV pick on one inter MB.
    // Useful as the headline "per-MB motion-search cost" number that
    // future work needs to move.
    let w = 64;
    let h = 64;
    let src_plane = make_plane(w, h, 1);
    let ref_plane = make_plane(w, h, 2);
    let stride = w;

    let mb_col = 1;
    let mb_row = 1;
    let src_y = source_block(&src_plane, stride, mb_col * 16, mb_row * 16);

    let reference = LumaRef {
        plane: &ref_plane,
        stride,
        width: w,
        height: h,
    };

    let center = Mv { row: 0, col: 0 };

    let mut g = c.benchmark_group("motion_search_descent");
    g.bench_function("full_descent_whole_half_quarter", |b| {
        b.iter(|| {
            let whole =
                small_diamond_search_luma(black_box(reference), mb_col, mb_row, &src_y, center, 8);
            let half = half_pixel_refine_luma(reference, mb_col, mb_row, &src_y, whole.mv);
            let quarter = quarter_pixel_refine_luma(reference, mb_col, mb_row, &src_y, half.mv);
            black_box(quarter)
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_small_diamond,
    bench_half_pixel_refine,
    bench_quarter_pixel_refine,
    bench_full_descent,
);
criterion_main!(benches);
