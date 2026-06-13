//! Micro-bench of the §12.3 4×4 B_PRED sub-block intra predictor.
//!
//! The 16×16 / 8×8 whole-block predictors (`intra_predict_dc16`) cover
//! the flat-region modes, but a B_PRED macroblock instead splits its
//! luma plane into sixteen independent 4×4 sub-blocks, each predicted by
//! one of the ten §12.3 directional modes (`predict_b4x4`). On detailed
//! keyframe content B_PRED is the dominant luma intra mode, and a single
//! B_PRED macroblock invokes `predict_b4x4` sixteen times — so this is
//! the per-MB intra cost the whole-frame keyframe-decode bench folds in
//! but never isolates.
//!
//! Two layers are measured:
//!
//! * Each of the ten modes individually, so a future optimisation of any
//!   one directional kernel (the diagonal `Vr` / `Vl` / `Hd` modes do
//!   ~16 `avg3p` / `avg2p` calls each; `Dc` is the cheap one) has a
//!   clean per-mode A/B anchor.
//! * `bpred_mb_16_subblocks` — sixteen `predict_b4x4` calls cycling
//!   through all ten modes, the realistic per-macroblock B_PRED decode
//!   unit (mode count and call count match a real 16×16 luma block).
//!
//! Inputs are synthesised in-bench from a non-flat ramp so the per-pixel
//! `avg3` / `clamp255` arithmetic of the directional modes cannot be
//! constant-folded away. No fixture files; no behaviour change.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::intra_predict::predict_b4x4;
use oxideav_vp8::IntraBmode;

/// The full §12.3 sub-block mode set, in spec order.
const ALL_MODES: [IntraBmode; 10] = [
    IntraBmode::Dc,
    IntraBmode::Tm,
    IntraBmode::Ve,
    IntraBmode::He,
    IntraBmode::Ld,
    IntraBmode::Rd,
    IntraBmode::Vr,
    IntraBmode::Vl,
    IntraBmode::Hd,
    IntraBmode::Hu,
];

/// Human label for each mode (group-function names).
fn mode_label(m: IntraBmode) -> &'static str {
    match m {
        IntraBmode::Dc => "b_dc",
        IntraBmode::Tm => "b_tm",
        IntraBmode::Ve => "b_ve",
        IntraBmode::He => "b_he",
        IntraBmode::Ld => "b_ld",
        IntraBmode::Rd => "b_rd",
        IntraBmode::Vr => "b_vr",
        IntraBmode::Vl => "b_vl",
        IntraBmode::Hd => "b_hd",
        IntraBmode::Hu => "b_hu",
    }
}

fn bench_predict_b4x4(c: &mut Criterion) {
    // Non-flat neighbour context: an `above[0..8]` ramp (the upper four
    // pixels feed the right-edge diagonal modes), a `left[0..4]` ramp,
    // and a distinct top-left pixel `p`. Picked so no directional
    // kernel collapses to a constant.
    let above: [u8; 8] = core::array::from_fn(|i| (20 + i * 23) as u8);
    let left: [u8; 4] = core::array::from_fn(|i| (200 - i * 31) as u8);
    let p: u8 = 137;

    let mut out = [0u8; 16];

    let mut g = c.benchmark_group("intra_predict_b4x4");

    // Per-mode micro-benches.
    for &mode in ALL_MODES.iter() {
        g.bench_function(mode_label(mode), |b| {
            b.iter(|| {
                predict_b4x4(
                    black_box(&mut out),
                    black_box(mode),
                    black_box(&above),
                    black_box(&left),
                    black_box(p),
                );
                black_box(out[0])
            });
        });
    }

    // Full B_PRED luma macroblock: sixteen sub-block predictions cycling
    // through the ten modes (the realistic per-MB call count). Each
    // sub-block gets a slightly rotated neighbour context so successive
    // calls don't all hit the same cached arithmetic.
    g.bench_function("bpred_mb_16_subblocks", |b| {
        b.iter(|| {
            for sb in 0..16usize {
                let mode = ALL_MODES[sb % ALL_MODES.len()];
                let above_sb: [u8; 8] = core::array::from_fn(|i| above[i].wrapping_add(sb as u8));
                let left_sb: [u8; 4] = core::array::from_fn(|i| left[i].wrapping_add(sb as u8));
                predict_b4x4(
                    black_box(&mut out),
                    black_box(mode),
                    black_box(&above_sb),
                    black_box(&left_sb),
                    black_box(p.wrapping_add(sb as u8)),
                );
                black_box(out[0]);
            }
        });
    });

    g.finish();
}

criterion_group!(benches, bench_predict_b4x4);
criterion_main!(benches);
