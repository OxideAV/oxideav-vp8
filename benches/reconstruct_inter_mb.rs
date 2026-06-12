//! Round 285 — per-MB micro-bench of the §14.2 / §18 inter-macroblock
//! reconstruction orchestrator (`motion_comp::reconstruct_inter_mb`).
//!
//! The round-283 post-change profile ranked `reconstruct_inter_mb` the
//! #1 self-time symbol on the inter decode bench under nightly + `simd`
//! (2242 of ~9237 samples, ≈ 24 %), yet the function had no bench of its
//! own: `inter_decode_short_clip` wraps it inside header parse + token
//! decode + loop filter, and `motion_comp_subpel_luma` only covers the
//! §18.3 prediction kernels *below* it. This harness times the whole
//! per-MB pipeline — §18.1 MV doubling/averaging, §18.2/§18.3 prediction
//! (batched whole-pixel fetch or six-tap synthesis), §14.3 Y2 WHT + DC
//! seeding, 24 × §14.4 inverse DCT, and the §14.5 add-residue clamp —
//! in the three shapes that partition the decoder's inter workload:
//!
//! * `subpel_full_residue` — both-axis sub-pixel MV + all 24 sub-blocks
//!   coded (the most expensive shape; six-tap synthesis + full residue).
//! * `whole_pixel_full_residue` — whole-pixel MV (the §18.3 "simply
//!   copied" batched fetch) + full residue: isolates the residue-add
//!   share from the interpolation share.
//! * `subpel_skip` — sub-pixel MV with `mb_skip_coeff` (prediction-only;
//!   the §11.1 skip short-circuit) — the lower bound of the pipeline.
//!
//! The reference planes are deterministic 4×4-MB (64×64 luma) planes and
//! the benched MB sits at (1, 1), so the in-bounds fast paths engage —
//! the dominant mid-frame case the profile measures.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::motion_comp::{filter_set_for_version, reconstruct_inter_mb, ReferencePlanes};
use oxideav_vp8::motion_vector::Mv;

const MB_COLS: usize = 4;
const MB_ROWS: usize = 4;

/// Deterministic mixed-frequency plane (same generator shape as the
/// `motion_search_descent` bench, so numbers stack against that suite).
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

/// Deterministic dequantized-magnitude coefficient block: a strong DC
/// plus a low-frequency AC skirt, the shape §14.1 dequantization
/// produces from small tokens at mid quantisers. Every one of the 24
/// residual sub-blocks (and Y2) gets a distinct variant via `seed`.
fn coeff_block(seed: i16) -> [i16; 16] {
    let mut out = [0i16; 16];
    out[0] = 40 + (seed % 7) * 9; // DC
    out[1] = -12 + seed % 5;
    out[4] = 17 - seed % 3;
    out[5] = 6 + seed % 4;
    out[2] = -(4 + seed % 2);
    out
}

struct Fixture {
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
    y2: [i16; 16],
    y: [[i16; 16]; 16],
    u: [[i16; 16]; 4],
    v: [[i16; 16]; 4],
}

impl Fixture {
    fn new() -> Self {
        let lw = MB_COLS * 16;
        let lh = MB_ROWS * 16;
        let cw = MB_COLS * 8;
        let ch = MB_ROWS * 8;
        Fixture {
            y_plane: make_plane(lw, lh, 2),
            u_plane: make_plane(cw, ch, 3),
            v_plane: make_plane(cw, ch, 4),
            y2: coeff_block(1),
            y: core::array::from_fn(|i| coeff_block(2 + i as i16)),
            u: core::array::from_fn(|i| coeff_block(20 + i as i16)),
            v: core::array::from_fn(|i| coeff_block(30 + i as i16)),
        }
    }

    fn reference(&self) -> ReferencePlanes<'_> {
        ReferencePlanes {
            y: &self.y_plane,
            u: &self.u_plane,
            v: &self.v_plane,
            y_stride: MB_COLS * 16,
            uv_stride: MB_COLS * 8,
            mb_cols: MB_COLS,
            mb_rows: MB_ROWS,
        }
    }
}

fn bench_reconstruct_inter_mb(c: &mut Criterion) {
    let fx = Fixture::new();
    let filters = filter_set_for_version(0).taps();
    // The benched MB: (col, row) = (1, 1) — interior, in-bounds fast paths.
    let (mb_col, mb_row) = (1usize, 1usize);

    // Quarter-pixel MVs (§17 units): (3, 5) has non-zero eighth-pixel
    // fractions on both axes after the §18.1 doubling → full 2-D six-tap;
    // (4, -8) is whole-pixel → the batched contiguous fetch.
    let subpel_mv = Mv { row: 3, col: 5 };
    let whole_mv = Mv { row: 4, col: -8 };

    let cases: [(&str, Mv, bool); 3] = [
        ("subpel_full_residue", subpel_mv, false),
        ("whole_pixel_full_residue", whole_mv, false),
        ("subpel_skip", subpel_mv, true),
    ];

    let mut g = c.benchmark_group("reconstruct_inter_mb");
    for (name, mv, skip) in cases {
        g.bench_function(name, |b| {
            b.iter(|| {
                let out = reconstruct_inter_mb(
                    black_box(&fx.reference()),
                    black_box(mb_col),
                    black_box(mb_row),
                    black_box(mv),
                    /* full_pixel = */ false,
                    filters,
                    skip,
                    &fx.y2,
                    &fx.y,
                    &fx.u,
                    &fx.v,
                );
                // Opaque reference: forces the 384-byte ReconstructedMb to
                // be materialised without an extra copy into criterion's
                // sink.
                black_box(&out).y[0]
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_reconstruct_inter_mb);
criterion_main!(benches);
