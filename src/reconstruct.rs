//! Per-macroblock reconstruction orchestration (RFC 6386 §14.2).
//!
//! This module is the glue that ties together the bitstream pieces
//! already landed in earlier rounds:
//!
//! * §13 token decode (`dct_tokens` / `decode_block`) → caller produces
//!   raw 4×4 sub-block coefficient arrays;
//! * §14.1 dequantization (`inverse_transform::dequant_block`) → caller
//!   multiplies each sub-block's coefficients by the DC / AC factors;
//! * §14.3 inverse WHT (`inverse_transform::inverse_wht_4x4`) → applied
//!   here when the Y2 block is present;
//! * §14.4 inverse DCT (`inverse_transform::inverse_dct_4x4`) → applied
//!   here per Y / U / V sub-block;
//! * §11 mode decode (`macroblock::parse_key_frame_macroblock_modes`)
//!   selects which §12 intra-prediction kernel applies;
//! * §12 intra-prediction kernels (`intra_predict::predict_y16x16` /
//!   `predict_uv8x8`) → applied here;
//! * §14.5 summation (`inverse_transform::add_residue_4x4`) → applied
//!   here to fold the per-sub-block residue back into the predicted
//!   pixels.
//!
//! ## Scope (single-MB orchestrator)
//!
//! [`decode_keyframe_mb_non_bpred`] reconstructs one macroblock whose
//! 16×16 luma mode is one of the four §12.3-16×16 modes
//! ([`IntraYMode::Dc`] / [`IntraYMode::V`] / [`IntraYMode::H`] /
//! [`IntraYMode::Tm`]). It does **not** drive `B_PRED` macroblocks —
//! those need a per-sub-block intra-prediction walker that re-uses each
//! sub-block's own reconstructed pixels as the next sub-block's
//! `above` / `left`, which is a follow-up integration step. It also
//! does not walk the raster of macroblocks across a frame; that walker
//! sits on top of this primitive and is the next layer up.
//!
//! ## Inputs
//!
//! The caller supplies pre-dequantized 16-bit coefficient arrays for
//! the 25th (Y2) block, the sixteen Y sub-blocks, and the eight chroma
//! sub-blocks. The dequant step is the caller's responsibility for
//! two reasons:
//!
//! 1. Y1 dequant factors (`Y1DequantFactors`) are already exposed by
//!    [`inverse_transform`]; pulling them in here would force a choice
//!    on the caller about quantiser-index plumbing that the §14.2
//!    orchestration step itself does not care about.
//! 2. The Y2 / chroma dequant scaling and clamping that RFC 6386 §14.1
//!    defers to the reference decoder `dixie.c` source is currently a
//!    documented spec gap (filed against `docs/video/vp8/`) — keeping
//!    that gap **out** of this orchestrator's signature means this
//!    module lands today without depending on the gap being filled.
//!    When the gap is filled, a follow-up convenience entry point that
//!    accepts raw tokens + a quantiser-index bundle can wrap this
//!    function.
//!
//! The caller also supplies the neighbouring pixel context that the
//! §12 kernels read (the 16-pixel `above` / `left` strips and the
//! top-left pixel for Y, the 8-pixel pair for each chroma plane).
//!
//! ## §14.2 procedure (transcribed)
//!
//! 1. If the Y2 block exists (Y mode ≠ `B_PRED` and ≠ `SPLITMV`):
//!    inverse-WHT the Y2 coefficients and seed each Y sub-block's
//!    coefficient 0 with the matching WHT-output element. This is the
//!    §14.2 first-paragraph rule: *"the element of the result at row
//!    `i`, column `j` is used as the 0th coefficient of the Y subblock
//!    at position `(i, j)`, that is, the Y subblock whose index is
//!    `(i * 4) + j`."*
//! 2. Inverse-DCT all sixteen Y sub-blocks and all eight chroma
//!    sub-blocks (§14.2 second paragraph: *"All 24 of these inversions
//!    are independent of each other."*)
//! 3. Apply the §12 intra-prediction kernel selected by the §11 mode
//!    record, producing the prediction buffer.
//! 4. Sum the prediction and the residue with `clamp255` (§14.5).
//!
//! ## `mb_skip_coeff`
//!
//! When the §11 mode record's `mb_skip_coeff` is true, the entire
//! residue is zero by definition — token decode is skipped, the Y2
//! block is absent, and the reconstructed pixels equal the prediction
//! exactly. This orchestrator honours that short-circuit and avoids
//! the wasted WHT / DCT / summation cycles for skip MBs.

use crate::intra_predict::{predict_uv8x8, predict_y16x16};
use crate::inverse_transform::{add_residue_4x4, inverse_dct_4x4, inverse_wht_4x4};
use crate::macroblock::{IntraUvMode, IntraYMode};

/// Pixel context surrounding a macroblock, supplying the neighbours
/// the §12 intra-prediction kernels read.
///
/// All fields are `Option`: `None` means the corresponding strip lies
/// off the top / left edge of the frame and the §12 kernels' default
/// substitution rules apply (the kernels fill from
/// [`crate::intra_predict::DEFAULT_ABOVE_PIXEL`] / `DEFAULT_LEFT_PIXEL`
/// / `DEFAULT_TOPLEFT_DC` as the spec specifies).
#[derive(Debug, Clone, Copy, Default)]
pub struct MbNeighbors {
    /// Sixteen luma pixels immediately above the MB (row `-1`,
    /// columns `0..=15`).
    pub y_above: Option<[u8; 16]>,
    /// Sixteen luma pixels immediately to the left of the MB
    /// (column `-1`, rows `0..=15`).
    pub y_left: Option<[u8; 16]>,
    /// Luma pixel at `(-1, -1)`. Only `TM_PRED` reads it; the
    /// §12 kernels substitute [`crate::intra_predict::DEFAULT_TOPLEFT_DC`]
    /// when `None`.
    pub y_topleft: Option<u8>,
    /// Eight chroma-U pixels immediately above the MB.
    pub u_above: Option<[u8; 8]>,
    /// Eight chroma-U pixels immediately to the left of the MB.
    pub u_left: Option<[u8; 8]>,
    /// Chroma-U pixel at `(-1, -1)`.
    pub u_topleft: Option<u8>,
    /// Eight chroma-V pixels immediately above the MB.
    pub v_above: Option<[u8; 8]>,
    /// Eight chroma-V pixels immediately to the left of the MB.
    pub v_left: Option<[u8; 8]>,
    /// Chroma-V pixel at `(-1, -1)`.
    pub v_topleft: Option<u8>,
}

/// Reconstructed pixel output for one macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconstructedMb {
    /// 16×16 luma pixels in row-major (raster) order.
    pub y: [u8; 256],
    /// 8×8 U chroma pixels in row-major order.
    pub u: [u8; 64],
    /// 8×8 V chroma pixels in row-major order.
    pub v: [u8; 64],
}

impl Default for ReconstructedMb {
    fn default() -> Self {
        ReconstructedMb {
            y: [0u8; 256],
            u: [0u8; 64],
            v: [0u8; 64],
        }
    }
}

/// Errors surfaced by the reconstruction orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructError {
    /// Called with `y_mode == IntraYMode::B`. The current orchestrator
    /// handles only the four 16×16 luma modes (`DC_PRED` / `V_PRED` /
    /// `H_PRED` / `TM_PRED`). A future `decode_keyframe_mb_bpred`
    /// variant will sit beside this one to handle `B_PRED`, including
    /// the per-sub-block neighbour-evolution walk §12.3 specifies.
    BPredNotSupported,
}

impl core::fmt::Display for ReconstructError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReconstructError::BPredNotSupported => f.write_str(
                "vp8 reconstruct: B_PRED MBs require the per-sub-block intra walker — \
                 use decode_keyframe_mb_non_bpred only for the four 16x16 luma modes",
            ),
        }
    }
}

impl std::error::Error for ReconstructError {}

/// Copy a 4×4 block at sub-block position `(by, bx)` (in 4-pixel
/// units) out of a row-major plane of stride `plane_stride` pixels.
#[inline]
fn extract_4x4(plane: &[u8], plane_stride: usize, by: usize, bx: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    let y0 = by * 4;
    let x0 = bx * 4;
    for r in 0..4 {
        let src = (y0 + r) * plane_stride + x0;
        let dst = r * 4;
        out[dst..dst + 4].copy_from_slice(&plane[src..src + 4]);
    }
    out
}

/// Splat a 4×4 block back into a row-major plane.
#[inline]
fn insert_4x4(plane: &mut [u8], plane_stride: usize, by: usize, bx: usize, src: &[u8; 16]) {
    let y0 = by * 4;
    let x0 = bx * 4;
    for r in 0..4 {
        let dst = (y0 + r) * plane_stride + x0;
        let s = r * 4;
        plane[dst..dst + 4].copy_from_slice(&src[s..s + 4]);
    }
}

/// Reconstruct one macroblock whose Y mode is one of the four 16×16
/// modes (`DC_PRED` / `V_PRED` / `H_PRED` / `TM_PRED`), per RFC 6386
/// §14.2.
///
/// # Inputs
///
/// * `y_mode` — the §11.2 16×16 luma intra mode. Must not be
///   [`IntraYMode::B`]; passing it returns
///   [`ReconstructError::BPredNotSupported`].
/// * `uv_mode` — the §11.4 8×8 chroma intra mode (always one of four
///   variants; chroma never has a `B_PRED` analogue).
/// * `mb_skip_coeff` — the §11.1 skip flag. When `true`, the entire
///   residue is zero and the four coefficient arrays are ignored; the
///   reconstructed pixels equal the prediction.
/// * `neighbors` — the surrounding pixel context the §12 kernels read
///   (see [`MbNeighbors`]).
/// * `y2_coeffs_dequant` — the dequantized Y2 (25th-block) WHT
///   coefficients in row-major order. Inverse-WHT is applied here and
///   each element of the result seeds the DC of one Y sub-block per
///   §14.2.
/// * `y_coeffs_dequant` — the sixteen dequantized Y sub-block DCT
///   coefficient arrays in raster sub-block order (sub-block index
///   `i * 4 + j` for row `i`, column `j` of the 4×4 sub-block grid).
///   The DC slot of each entry is overwritten with the WHT result
///   before the inverse DCT is applied.
/// * `u_coeffs_dequant` / `v_coeffs_dequant` — the eight chroma
///   sub-block coefficient arrays, four U then four V, in raster
///   sub-block order (`i * 2 + j` for the 2×2 chroma grid).
///
/// # Output
///
/// A fully reconstructed macroblock — the predictor-plus-residue sum
/// with `clamp255` saturation, before loop filtering.
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §14.2 input;
                                     // bundling them into a struct would obscure the spec mapping.
pub fn decode_keyframe_mb_non_bpred(
    y_mode: IntraYMode,
    uv_mode: IntraUvMode,
    mb_skip_coeff: bool,
    neighbors: &MbNeighbors,
    y2_coeffs_dequant: &[i16; 16],
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> Result<ReconstructedMb, ReconstructError> {
    if y_mode == IntraYMode::B {
        return Err(ReconstructError::BPredNotSupported);
    }

    let mut out = ReconstructedMb::default();

    // Step 3 (done first because we always need it, even on skip MBs):
    // intra prediction into the output buffers.
    let topleft_y = neighbors.y_topleft.unwrap_or(0); // unused except by TM
    predict_y16x16(
        &mut out.y,
        y_mode,
        neighbors.y_above.as_ref(),
        neighbors.y_left.as_ref(),
        topleft_y,
    )
    // predict_y16x16 returns Option<()> only to distinguish the B_PRED
    // case; we've rejected B above, so this is always Some.
    .expect("y_mode rejected B_PRED at function entry");

    let topleft_u = neighbors.u_topleft.unwrap_or(0);
    predict_uv8x8(
        &mut out.u,
        uv_mode,
        neighbors.u_above.as_ref(),
        neighbors.u_left.as_ref(),
        topleft_u,
    );
    let topleft_v = neighbors.v_topleft.unwrap_or(0);
    predict_uv8x8(
        &mut out.v,
        uv_mode,
        neighbors.v_above.as_ref(),
        neighbors.v_left.as_ref(),
        topleft_v,
    );

    if mb_skip_coeff {
        // §11.1: skip flag → zero residue → reconstructed pixels equal
        // prediction. No transform work needed; out already holds the
        // prediction.
        return Ok(out);
    }

    // Step 1 (§14.2 first paragraph): if Y2 exists, inverse-WHT and
    // seed each Y sub-block's DC. For all four 16×16 modes the Y2
    // block exists (it's absent only for B_PRED and SPLITMV).
    let mut y2_residue = [0i16; 16];
    inverse_wht_4x4(y2_coeffs_dequant, &mut y2_residue);

    // Copy the Y coefficients so we can splice in the WHT seeds without
    // mutating the caller's input.
    let mut y_coeffs = *y_coeffs_dequant;
    for i in 0..4 {
        for j in 0..4 {
            // §14.2: y2_residue[i*4+j] seeds Y sub-block (i,j)'s
            // coefficient 0.
            y_coeffs[i * 4 + j][0] = y2_residue[i * 4 + j];
        }
    }

    // Step 2 (§14.2 second paragraph): inverse-DCT all 16 Y +
    // 8 chroma sub-blocks. Step 4 (§14.5): add to prediction with
    // clamp255 and store back into the output buffer.
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&y_coeffs[idx], &mut residue);
            let pred = extract_4x4(&out.y, 16, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.y, 16, i, j, &summed);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&u_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.u, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.u, 8, i, j, &summed);

            let mut residue = [0i16; 16];
            inverse_dct_4x4(&v_coeffs_dequant[idx], &mut residue);
            let pred = extract_4x4(&out.v, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(&mut out.v, 8, i, j, &summed);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_predict::{
        predict_uv8x8, predict_y16x16, DEFAULT_ABOVE_PIXEL, DEFAULT_LEFT_PIXEL, DEFAULT_TOPLEFT_DC,
    };

    /// Convenience type alias for the four coefficient bundles
    /// `zero_coeffs` returns (Y2, sixteen Y, four U, four V).
    type CoeffBundle = ([i16; 16], [[i16; 16]; 16], [[i16; 16]; 4], [[i16; 16]; 4]);

    /// All-zero coefficients in every block.
    fn zero_coeffs() -> CoeffBundle {
        (
            [0i16; 16],
            [[0i16; 16]; 16],
            [[0i16; 16]; 4],
            [[0i16; 16]; 4],
        )
    }

    #[test]
    fn bpred_yields_error() {
        let (y2, y, u, v) = zero_coeffs();
        let res = decode_keyframe_mb_non_bpred(
            IntraYMode::B,
            IntraUvMode::Dc,
            false,
            &MbNeighbors::default(),
            &y2,
            &y,
            &u,
            &v,
        );
        assert_eq!(res, Err(ReconstructError::BPredNotSupported));
    }

    #[test]
    fn skip_mb_equals_prediction_dc_mode_top_left_corner() {
        // A top-left corner MB with skip=true and DC_PRED in both
        // planes. Per §12.2 page 53, DC_PRED with no neighbours uses
        // 128 for both Y and chroma (DEFAULT_TOPLEFT_DC). With skip=
        // true the residue is zero, so every reconstructed pixel must
        // be 128.
        let (y2, y, u, v) = zero_coeffs();
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::Dc,
            IntraUvMode::Dc,
            true,
            &MbNeighbors::default(),
            &y2,
            &y,
            &u,
            &v,
        )
        .expect("non-bpred path");
        assert!(out.y.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
        assert!(out.u.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
        assert!(out.v.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
    }

    #[test]
    fn skip_mb_matches_standalone_intra_predict_v_mode() {
        // V_PRED in luma + V_PRED in chroma with known `above` strips.
        // skip=true → output must match a standalone predict_*_v call.
        let y_above = [50u8; 16];
        let u_above = [60u8; 8];
        let v_above = [70u8; 8];

        let neighbors = MbNeighbors {
            y_above: Some(y_above),
            u_above: Some(u_above),
            v_above: Some(v_above),
            ..MbNeighbors::default()
        };
        let (y2, y, u, v) = zero_coeffs();
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::V,
            IntraUvMode::V,
            true,
            &neighbors,
            &y2,
            &y,
            &u,
            &v,
        )
        .expect("non-bpred path");

        let mut expect_y = [0u8; 256];
        predict_y16x16(&mut expect_y, IntraYMode::V, Some(&y_above), None, 0).unwrap();
        let mut expect_u = [0u8; 64];
        predict_uv8x8(&mut expect_u, IntraUvMode::V, Some(&u_above), None, 0);
        let mut expect_v = [0u8; 64];
        predict_uv8x8(&mut expect_v, IntraUvMode::V, Some(&v_above), None, 0);

        assert_eq!(out.y, expect_y);
        assert_eq!(out.u, expect_u);
        assert_eq!(out.v, expect_v);
    }

    #[test]
    fn zero_residue_non_skip_also_equals_prediction() {
        // With skip=false but all coefficients zero, the WHT and the
        // sixteen DCT inversions all yield zero residues, so the
        // reconstructed pixels still equal the prediction — but this
        // time the §14.2 orchestration path is exercised (all the
        // transforms run, all the adds run; they just contribute 0).
        let y_above = [80u8; 16];
        let y_left = [90u8; 16];
        let u_above = [100u8; 8];
        let u_left = [110u8; 8];
        let v_above = [120u8; 8];
        let v_left = [130u8; 8];
        let neighbors = MbNeighbors {
            y_above: Some(y_above),
            y_left: Some(y_left),
            y_topleft: Some(85),
            u_above: Some(u_above),
            u_left: Some(u_left),
            u_topleft: Some(105),
            v_above: Some(v_above),
            v_left: Some(v_left),
            v_topleft: Some(125),
        };
        let (y2, y, u, v) = zero_coeffs();
        let out_skip = decode_keyframe_mb_non_bpred(
            IntraYMode::Tm,
            IntraUvMode::Tm,
            true,
            &neighbors,
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();
        let out_run = decode_keyframe_mb_non_bpred(
            IntraYMode::Tm,
            IntraUvMode::Tm,
            false,
            &neighbors,
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();
        assert_eq!(out_skip.y, out_run.y);
        assert_eq!(out_skip.u, out_run.u);
        assert_eq!(out_skip.v, out_run.v);
    }

    #[test]
    fn y2_dc_only_seeds_y_subblock_dc_and_lifts_pixels() {
        // Put a single non-zero DC in Y2 and verify the §14.2 seeding
        // contract: the WHT of an all-zero-except-DC=8 block is the
        // flat array `[1, 1, 1, 1, ..., 1]` (16 ones) — see the
        // existing `wht_dc_only_matches_general_path` test in
        // inverse_transform.rs. Each of the sixteen Y sub-blocks then
        // gets coefficient[0] = 1 spliced in. The inverse DCT of an
        // all-zero-except-DC=1 4×4 block is, per §14.4's two-pass
        // arithmetic with `(x + 4) >> 3` rounding, all zero (1*8 = 8,
        // (8+4)>>3 = 1, then again (1+4)>>3 = 0 — actually let's not
        // rely on a hand derivation here; we verify it empirically).
        let mut y2 = [0i16; 16];
        y2[0] = 8;
        let (_, y, u, v) = zero_coeffs();

        // Top-left corner MB with DC_PRED everywhere — every prediction
        // pixel = 128, all chroma residue = 0.
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::Dc,
            IntraUvMode::Dc,
            false,
            &MbNeighbors::default(),
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();

        // Empirically derive the expected per-pixel residue: WHT
        // distributes DC=8 as `1` into every Y sub-block's slot 0,
        // then inverse_dct_4x4 with input[0]=1 (and the rest zero) is
        // computed and added to the 128 prediction.
        let mut seed = [0i16; 16];
        seed[0] = 1;
        let mut residue4x4 = [0i16; 16];
        inverse_dct_4x4(&seed, &mut residue4x4);
        let pred4x4 = [DEFAULT_TOPLEFT_DC; 16];
        let mut expect_sub = [0u8; 16];
        add_residue_4x4(&pred4x4, &residue4x4, &mut expect_sub);

        // Every Y sub-block must equal `expect_sub` since they all see
        // the same prediction value (128) and the same WHT-seed (1).
        for i in 0..4 {
            for j in 0..4 {
                let got = extract_4x4(&out.y, 16, i, j);
                assert_eq!(
                    got, expect_sub,
                    "sub-block ({i},{j}) mismatch — y2 DC distribution broken"
                );
            }
        }

        // Chroma residue was zero everywhere → chroma equals prediction.
        assert!(out.u.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
        assert!(out.v.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
    }

    #[test]
    fn y2_off_diagonal_seeds_distinct_subblocks_per_142_index_rule() {
        // Drive a Y2 input that, after the inverse WHT, produces a
        // distinguishable residue at row 0 column 0 versus row 1
        // column 0 versus row 0 column 1. Per §14.2 these distinct
        // values must seed sub-block (0,0), sub-block (1,0)=index 4
        // and sub-block (0,1)=index 1 respectively — proving the
        // `i * 4 + j` index rule.
        //
        // From the existing `wht_hand_derived_two_value_input` test
        // in inverse_transform.rs, with input[0]=8 and input[4]=8:
        //   WHT output = [2,2,2,2, 2,2,2,2, 0,0,0,0, 0,0,0,0]
        // So sub-blocks (0,0)/(0,1)/(0,2)/(0,3) and (1,0)/(1,1)/(1,2)
        // /(1,3) get DC seed 2, while (2,*)/(3,*) get DC seed 0.
        let mut y2 = [0i16; 16];
        y2[0] = 8;
        y2[4] = 8;
        let (_, y, u, v) = zero_coeffs();

        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::Dc,
            IntraUvMode::Dc,
            false,
            &MbNeighbors::default(),
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();

        let mut seed_lit = [0i16; 16];
        seed_lit[0] = 2;
        let mut residue_lit = [0i16; 16];
        inverse_dct_4x4(&seed_lit, &mut residue_lit);
        let pred = [DEFAULT_TOPLEFT_DC; 16];
        let mut expect_lit = [0u8; 16];
        add_residue_4x4(&pred, &residue_lit, &mut expect_lit);

        let mut seed_zero = [0i16; 16];
        seed_zero[0] = 0;
        let mut residue_zero = [0i16; 16];
        inverse_dct_4x4(&seed_zero, &mut residue_zero);
        let mut expect_zero = [0u8; 16];
        add_residue_4x4(&pred, &residue_zero, &mut expect_zero);

        // Rows 0..2 are "lit" (DC=2); rows 2..4 are "zero" (DC=0).
        for i in 0..4 {
            for j in 0..4 {
                let got = extract_4x4(&out.y, 16, i, j);
                let expected = if i < 2 { expect_lit } else { expect_zero };
                assert_eq!(
                    got, expected,
                    "sub-block ({i},{j}) wrong — i*4+j index rule violated"
                );
            }
        }
    }

    #[test]
    fn neighbour_defaults_drive_v_mode_when_above_absent() {
        // V_PRED with no `above` neighbour: §12 substitutes
        // DEFAULT_ABOVE_PIXEL = 127. Verify the orchestrator forwards
        // the substitution by checking the luma output equals 127
        // everywhere (skip MB, zero residue).
        let (y2, y, u, v) = zero_coeffs();
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::V,
            IntraUvMode::V,
            true,
            &MbNeighbors::default(),
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();
        assert!(out.y.iter().all(|&p| p == DEFAULT_ABOVE_PIXEL));
        assert!(out.u.iter().all(|&p| p == DEFAULT_ABOVE_PIXEL));
    }

    #[test]
    fn neighbour_defaults_drive_h_mode_when_left_absent() {
        let (y2, y, u, v) = zero_coeffs();
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::H,
            IntraUvMode::H,
            true,
            &MbNeighbors::default(),
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();
        assert!(out.y.iter().all(|&p| p == DEFAULT_LEFT_PIXEL));
        assert!(out.v.iter().all(|&p| p == DEFAULT_LEFT_PIXEL));
    }

    #[test]
    fn residue_summation_saturates_high_per_145() {
        // Drive an MB where the Y residue plus a normal-range V_PRED
        // prediction overflows past 255, and verify the §14.5
        // clamp255 step is applied (instead of wrapping at u8).
        //
        // §14.2 overwrites every Y sub-block's coefficient[0] with
        // the Y2 WHT result, so we drive the saturation through Y2.
        // A Y2 input of `[8000, 0, ...]` produces a WHT output that
        // distributes large values into every Y sub-block's DC; the
        // per-sub-block inverse DCT then yields per-pixel residues
        // well over +1000.
        let mut y2 = [0i16; 16];
        y2[0] = 8000;
        let y = [[0i16; 16]; 16];
        let (_, _, u, v) = zero_coeffs();

        let y_above = [200u8; 16];
        let neighbors = MbNeighbors {
            y_above: Some(y_above),
            ..MbNeighbors::default()
        };
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::V,
            IntraUvMode::Dc,
            false,
            &neighbors,
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();

        // Every luma pixel must clamp to 255 (residue + 200 ≫ 255 in
        // every sub-block) — proves clamp255 is wired into the
        // summation step rather than wrapping at u8.
        assert!(
            out.y.iter().all(|&p| p == 255),
            "expected 255 saturation everywhere in Y plane; got first pixel {}",
            out.y[0]
        );
    }

    #[test]
    fn residue_summation_saturates_low_per_145() {
        // Symmetric to the high-saturation test: negative Y2 input,
        // small prediction, every pixel must clamp to 0.
        let mut y2 = [0i16; 16];
        y2[0] = -8000;
        let y = [[0i16; 16]; 16];
        let (_, _, u, v) = zero_coeffs();

        let y_above = [40u8; 16];
        let neighbors = MbNeighbors {
            y_above: Some(y_above),
            ..MbNeighbors::default()
        };
        let out = decode_keyframe_mb_non_bpred(
            IntraYMode::V,
            IntraUvMode::Dc,
            false,
            &neighbors,
            &y2,
            &y,
            &u,
            &v,
        )
        .unwrap();

        assert!(
            out.y.iter().all(|&p| p == 0),
            "expected 0 saturation everywhere in Y plane; got first pixel {}",
            out.y[0]
        );
    }

    #[test]
    fn extract_and_insert_4x4_roundtrip() {
        // Tiny helper test: extract + insert is a no-op (modulo unrelated
        // bytes), guarding against an off-by-one in the orchestrator's
        // address math.
        let mut plane = [0u8; 256];
        for (i, slot) in plane.iter_mut().enumerate() {
            *slot = i as u8;
        }
        // Save sub-block (2,1) (rows 8..12, cols 4..8).
        let got = extract_4x4(&plane, 16, 2, 1);
        // Expected: row r, col c → byte at (8+r)*16 + (4+c).
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(got[r * 4 + c], ((8 + r) * 16 + (4 + c)) as u8);
            }
        }
        // Zero it out, then re-insert and verify the plane is restored.
        let zeros = [0u8; 16];
        insert_4x4(&mut plane, 16, 2, 1, &zeros);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(plane[(8 + r) * 16 + (4 + c)], 0);
            }
        }
        insert_4x4(&mut plane, 16, 2, 1, &got);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    plane[(8 + r) * 16 + (4 + c)],
                    ((8 + r) * 16 + (4 + c)) as u8
                );
            }
        }
        // And bytes outside the patched sub-block are untouched.
        assert_eq!(plane[0], 0);
        assert_eq!(plane[255], 255);
    }
}
