//! Round 298 — the §14.1 dequantization layer in isolation.
//!
//! RFC 6386 §14.1 multiplies every decoded (quantized) coefficient by one
//! of six dequantization factors chosen by plane (Y1 / Y2 / chroma) and
//! position (DC = coefficient 0, AC = coefficients 1..=15). Two layers run
//! on the decode path and neither had an isolated criterion harness — both
//! were only ever measured folded inside the whole-frame decode benches
//! (`keyframe_decode`, `inter_decode_short_clip`, `token_decode`):
//!
//! * **factor derivation** — `MbDequantFactors::from_quant_indices`
//!   (per-frame, no segmentation) and `MbDequantFactors::for_segment` (the
//!   §10 per-segment override). Both run the §20.4 `dequant_init` body: six
//!   `dc_qlookup` / `ac_qlookup` table reads through `clamp_qindex`, the
//!   `*2` Y2-DC and `*155/100` Y2-AC scalings, and the `>132` chroma-DC /
//!   `<8` Y2-AC clamps. `from_quant_indices` fires once per frame;
//!   `for_segment` once per active segment.
//!
//! * **the apply** — `MbDequantFactors::dequantize`, the per-MB hot loop.
//!   It scales all twenty-five 4×4 blocks of one macroblock (the Y2 block,
//!   the sixteen Y sub-blocks, the four U and four V chroma sub-blocks):
//!   400 coefficient×factor multiplies per non-skip MB, computed in `i32`
//!   and stored back as `i16` (§14.1 page 76). This fires on every coded
//!   macroblock and is the SIMD/unroll A/B target — `dequant_block` already
//!   carries an optional `core::simd` path, so a stable per-MB baseline at
//!   criterion resolution was overdue.
//!
//! Inputs are synthesised in-bench (no committed fixtures). The apply is
//! benched at two coefficient densities — a sparse block (DC + a few low-
//! frequency AC, the common post-quantisation residual shape) and a dense
//! block (every lane non-zero, the worst case) — so the per-coefficient
//! cost and the loop overhead are both visible.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::coded_header::QuantIndices;
use oxideav_vp8::frame::MbCoeffs;
use oxideav_vp8::MbDequantFactors;

/// A mid-range §9.6 quantiser header with every per-plane delta present —
/// the worst case for `from_quant_indices` (all six factors take the full
/// derivation path, none short-circuit a `None` delta to 0).
fn qi_full() -> QuantIndices {
    QuantIndices {
        y_ac_qi: 64,
        y_dc_delta: Some(1),
        y2_dc_delta: Some(-2),
        y2_ac_delta: Some(15),
        uv_dc_delta: Some(-15),
        uv_ac_delta: Some(0),
    }
}

/// Representative §14.1 dequant factors for a mid-range quantiser, used to
/// drive the apply benches independently of the derivation cost.
fn sample_factors() -> MbDequantFactors {
    MbDequantFactors::from_quant_indices(&qi_full())
}

/// A sparse raster-order 4×4 block: a DC plus three low-frequency AC
/// coefficients, zero elsewhere — the common shape a quantised residual
/// takes after the high-frequency lanes round to zero.
const SPARSE_BLOCK: [i16; 16] = [
    7, -3, 1, 0, //
    2, 0, 0, 0, //
    -1, 0, 0, 0, //
    0, 0, 0, 0,
];

/// A dense raster-order 4×4 block: every lane non-zero — the worst case
/// for the per-coefficient multiply (no early-out on a run of zeros).
const DENSE_BLOCK: [i16; 16] = [
    40, -8, 4, -2, //
    -6, 5, -3, 1, //
    3, -2, 1, -1, //
    -1, 1, 2, -2,
];

/// Build an `MbCoeffs` whose twenty-five blocks are all the given block —
/// the apply visits every plane (Y2, sixteen Y, four U, four V).
fn coeffs_filled(block: [i16; 16]) -> MbCoeffs {
    MbCoeffs {
        y2: block,
        y: [block; 16],
        u: [block; 4],
        v: [block; 4],
    }
}

fn bench_factor_derivation(c: &mut Criterion) {
    let mut g = c.benchmark_group("dequantize_mb");

    // Per-frame factor derivation (no segmentation override).
    let qi = qi_full();
    g.bench_function("from_quant_indices", |b| {
        b.iter(|| MbDequantFactors::from_quant_indices(black_box(&qi)));
    });

    // §10 per-segment override, delta mode (base = yac_qi + segment_quant).
    g.bench_function("for_segment_delta", |b| {
        b.iter(|| MbDequantFactors::for_segment(black_box(&qi), black_box(10), black_box(false)));
    });

    // §10 per-segment override, absolute mode (base = segment_quant).
    g.bench_function("for_segment_absolute", |b| {
        b.iter(|| MbDequantFactors::for_segment(black_box(&qi), black_box(70), black_box(true)));
    });

    g.finish();
}

fn bench_dequantize_apply(c: &mut Criterion) {
    let factors = sample_factors();
    let mut g = c.benchmark_group("dequantize_mb");

    // The per-MB apply over a sparse-residual macroblock (the common case).
    let sparse = coeffs_filled(SPARSE_BLOCK);
    g.bench_function("dequantize_sparse_mb", |b| {
        b.iter(|| {
            let mut coeffs = sparse;
            black_box(&factors).dequantize(black_box(&mut coeffs));
            black_box(coeffs)
        });
    });

    // The per-MB apply over a fully-dense macroblock (the worst case).
    let dense = coeffs_filled(DENSE_BLOCK);
    g.bench_function("dequantize_dense_mb", |b| {
        b.iter(|| {
            let mut coeffs = dense;
            black_box(&factors).dequantize(black_box(&mut coeffs));
            black_box(coeffs)
        });
    });

    g.finish();
}

criterion_group!(benches, bench_factor_derivation, bench_dequantize_apply);
criterion_main!(benches);
