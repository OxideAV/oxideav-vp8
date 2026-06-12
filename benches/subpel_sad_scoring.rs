//! Round 285 — encoder-side per-candidate SAD scoring micro-bench
//! (`motion_search::mb_luma_sad_at_mv` + `motion_comp::fetch_luma_mb_halo`).
//!
//! The round-281 post-change profile put the sub-pixel
//! `mb_luma_sad_at_mv` leg at #1 encoder self-time (2103 of ~9276
//! samples) with `fetch_luma_mb_halo` at 590 — the §17 half-/quarter-
//! pixel refinements score 16 sub-pixel candidates per searched MB, and
//! each candidate currently materialises the full 16×16 `sixtap_mb_luma`
//! output before `block_sad_16x16` reads it back. The standing
//! BENCHMARKS candidate ("sub-pixel SAD without patch materialisation"
//! — fuse the SAD into the vertical convolution pass row by row)
//! explicitly asks for a micro-bench first; this is that harness.
//!
//! Decomposition story (per single candidate, stacking against the
//! existing suites):
//!
//! * `mb_luma_sad_at_mv_half_pel` / `_quarter_pel` — the full scoring
//!   call in its two §17 refinement shapes (both run the same 21×21
//!   halo fetch + whole-MB §18.3 two-pass + 256-byte SAD; the fraction
//!   pair only selects tap sets).
//! * `mb_luma_sad_at_mv_whole_pel` — the §18.3 "simply copied" shape
//!   (round-281 fused fetch-and-SAD fast path), the cheap fraction of
//!   the 17-candidate ladder.
//! * `fetch_luma_mb_halo_21x21_in_bounds` — the fetch share alone, so
//!   the halo / convolution / SAD split is readable next to the
//!   existing `motion_comp_subpel_luma/mb_luma_batched_16x16` (fetch +
//!   convolution) and `motion_search_descent/block_sad_16x16_single_pair`
//!   (SAD leaf) rows. A future fused row-SAD lands as a delta on the
//!   `_half_pel` row at constant `fetch` + `batched_16x16` rows.
//!
//! Inputs are the same deterministic 64×64 mixed-frequency planes the
//! `motion_search_descent` bench drives, MB at (1, 1) so the §20.14
//! border clamp stays cold (the profiled mid-frame case).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::motion_comp::fetch_luma_mb_halo;
use oxideav_vp8::motion_search::{mb_luma_sad_at_mv, LumaRef};
use oxideav_vp8::motion_vector::Mv;

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

fn source_block(plane: &[u8], stride: usize, blk_x: usize, blk_y: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    for r in 0..16 {
        let row_off = (blk_y + r) * stride + blk_x;
        out[r * 16..r * 16 + 16].copy_from_slice(&plane[row_off..row_off + 16]);
    }
    out
}

fn bench_subpel_sad_scoring(c: &mut Criterion) {
    let w = 64;
    let h = 64;
    let src_plane = make_plane(w, h, 1);
    let ref_plane = make_plane(w, h, 2);
    let stride = w;

    let (mb_col, mb_row) = (1usize, 1usize);
    let src_y = source_block(&src_plane, stride, mb_col * 16, mb_row * 16);

    let reference = LumaRef {
        plane: &ref_plane,
        stride,
        width: w,
        height: h,
    };

    // §17 quarter-pixel units: half-pel = multiples of 2 with a set bit 1,
    // quarter-pel = odd components, whole-pel = multiples of 4.
    let cases: [(&str, Mv); 3] = [
        ("mb_luma_sad_at_mv_half_pel", Mv { row: 2, col: -2 }),
        ("mb_luma_sad_at_mv_quarter_pel", Mv { row: 1, col: 3 }),
        ("mb_luma_sad_at_mv_whole_pel", Mv { row: 4, col: -4 }),
    ];

    let mut g = c.benchmark_group("subpel_sad_scoring");
    for (name, mv) in cases {
        g.bench_function(name, |b| {
            b.iter(|| {
                let sad = mb_luma_sad_at_mv(
                    black_box(reference),
                    black_box(mb_col),
                    black_box(mb_row),
                    black_box(&src_y),
                    black_box(mv),
                );
                black_box(sad)
            });
        });
    }

    // The fetch share alone — eighth-pixel MV with sub-pixel fractions
    // (the doubled half-pel candidate above) at the same interior MB.
    let mv_eighth = Mv { row: 4, col: -4 };
    g.bench_function("fetch_luma_mb_halo_21x21_in_bounds", |b| {
        b.iter(|| {
            let halo = fetch_luma_mb_halo(
                black_box(&ref_plane),
                stride,
                w,
                h,
                black_box(mb_col * 16),
                black_box(mb_row * 16),
                black_box(mv_eighth),
            );
            // Opaque reference: forces the whole 21×21 halo to be
            // materialised without paying an extra 441-byte copy into
            // criterion's sink (which would dominate this cheap fetch).
            black_box(&halo)[0]
        });
    });
    g.finish();
}

criterion_group!(benches, bench_subpel_sad_scoring);
criterion_main!(benches);
