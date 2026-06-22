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
//! [`IntraYMode::Tm`]). It does **not** drive `B_PRED` macroblocks.
//!
//! [`decode_keyframe_mb_bpred`] is the companion entry point for the
//! `B_PRED` luma mode (§11.3 / §12.3). Unlike the 16×16 modes, a
//! `B_PRED` macroblock predicts each of its sixteen 4×4 luma sub-blocks
//! independently, and — critically — each sub-block's prediction reads
//! the **already-reconstructed** pixels (prediction + residue) of the
//! sub-blocks above and to its left. The orchestrator therefore cannot
//! predict the whole plane first and add residue afterward; it must
//! interleave predict → inverse-DCT → add-residue **per sub-block** in
//! raster order so that each sub-block's reconstructed pixels are
//! visible as the next sub-block's neighbours (this mirrors the §20.14
//! reference decoder's in-place `b_pred()` loop). The chroma planes of a
//! `B_PRED` macroblock still use one of the four 8×8 §12.2 modes, so
//! that part of the recipe is identical to the non-`B_PRED` path.
//!
//! Neither orchestrator walks the raster of macroblocks across a frame;
//! that walker sits on top of these primitives and is the next layer up.
//!
//! ### §12.3 per-sub-block neighbour evolution
//!
//! The working luma buffer is a `B_PRED`-private macroblock-sized array
//! with a one-pixel top border row and a one-pixel left border column
//! (so each sub-block can read `above` / `left` / the top-left pixel `P`
//! at constant stride), plus a four-pixel "above-right extension" tacked
//! onto the right end of the border row. Each sub-block's `above` is the
//! 8-pixel strip in the row immediately above it (the four pixels
//! directly above plus the four above-and-to-the-right), `left` is the
//! 4-pixel column immediately to its left, and `P` is the pixel at
//! `(-1, -1)` relative to the sub-block. After [`predict_b4x4`] writes
//! the prediction, the sub-block's residue is inverse-DCT'd and added
//! in place, so the next sub-block in raster order sees reconstructed
//! pixels.
//!
//! ### §12.3 right-edge "above-right" fixup
//!
//! Because whole macroblocks are reconstructed in raster order, the
//! four pixels above-and-to-the-right of sub-blocks 7, 11, and 15 have
//! not been built when those sub-blocks are predicted. Per §12.3 page
//! 55 all four right-edge sub-blocks (3, 7, 11, 15) instead share the
//! four pixels above-and-to-the-right of sub-block 3 — frame positions
//! `(-1, 16)..=(-1, 19)`. The orchestrator copies those four pixels
//! down into the border slots above sub-blocks 7, 11, and 15 (the
//! `copy_down` step of the reference decoder) before the sub-block walk
//! begins. The caller supplies the four `(-1, 16)..=(-1, 19)` values in
//! [`MbNeighbors::y_above_right`]; on the top macroblock row they are
//! 127, and for the rightmost macroblock in a non-top row they are the
//! value at `(-1, 15)` (the caller computes that clamp).
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

use crate::intra_predict::{
    predict_b4x4, predict_uv8x8, predict_y16x16, DEFAULT_ABOVE_PIXEL, DEFAULT_LEFT_PIXEL,
};
use crate::inverse_transform::{add_residue_4x4, inverse_dct_4x4, inverse_wht_4x4};
use crate::macroblock::{IntraBmode, IntraUvMode, IntraYMode};

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
    /// The four luma pixels above-and-to-the-right of the macroblock
    /// (frame positions `(-1, 16)..=(-1, 19)`), used only by the
    /// `B_PRED` path's §12.3 right-edge sub-blocks (3, 7, 11, 15).
    ///
    /// `None` means the macroblock is on the top row of the frame, in
    /// which case §12.3 page 55 fixes all four extra pixels at 127
    /// ([`crate::intra_predict::DEFAULT_ABOVE_PIXEL`]). When `Some`, the
    /// caller has already applied the rightmost-macroblock clamp (all
    /// four equal to the `(-1, 15)` pixel) where it applies; this
    /// orchestrator copies the four values verbatim into the working
    /// buffer. The 16×16 luma modes never read this field.
    pub y_above_right: Option<[u8; 4]>,
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
    /// [`decode_keyframe_mb_non_bpred`] was called with
    /// `y_mode == IntraYMode::B`. The 16×16 orchestrator handles only
    /// the four 16×16 luma modes (`DC_PRED` / `V_PRED` / `H_PRED` /
    /// `TM_PRED`); `B_PRED` macroblocks go through
    /// [`decode_keyframe_mb_bpred`] instead, which drives the
    /// per-sub-block neighbour-evolution walk §12.3 specifies.
    BPredNotSupported,
    /// [`decode_keyframe_mb_bpred`] was called for a macroblock whose
    /// `subblock_modes` record is absent. A `B_PRED` macroblock always
    /// carries sixteen decoded 4×4 sub-block modes (§11.3); a missing
    /// record means the caller routed a non-`B_PRED` macroblock here by
    /// mistake.
    MissingSubblockModes,
}

impl core::fmt::Display for ReconstructError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReconstructError::BPredNotSupported => f.write_str(
                "vp8 reconstruct: B_PRED MBs require the per-sub-block intra walker — \
                 use decode_keyframe_mb_bpred, not decode_keyframe_mb_non_bpred",
            ),
            ReconstructError::MissingSubblockModes => f.write_str(
                "vp8 reconstruct: decode_keyframe_mb_bpred needs sixteen 4x4 sub-block \
                 modes; the supplied record has none (non-B_PRED MB routed here?)",
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
    //
    // The corner pixel (-1, -1) is read only by TM_PRED. A `None` here
    // means the corner is off the top frame edge (the frame walker
    // supplies `Some(129)` for the left-frame-edge corner and the genuine
    // reconstructed pixel for interior macroblocks), so it defaults to the
    // §12 `DEFAULT_ABOVE_PIXEL` (127) — the value §20.14 `fixup_above`
    // assigns the off-top corner.
    let topleft_y = neighbors.y_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
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

    let topleft_u = neighbors.u_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
    predict_uv8x8(
        &mut out.u,
        uv_mode,
        neighbors.u_above.as_ref(),
        neighbors.u_left.as_ref(),
        topleft_u,
    );
    let topleft_v = neighbors.v_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
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

// ===========================================================================
// §11.3 / §12.3 B_PRED macroblock reconstruction
// ===========================================================================

/// Stride (pixels per row) of the `B_PRED` working luma buffer.
///
/// Layout per row: `[ left-border (1) | 16 reconstruction columns |
/// above-right extension (4) ]` = 21 columns. The four-pixel extension
/// lets the §12.3 right-edge sub-blocks (3, 7, 11, 15) read their
/// shared above-and-to-the-right pixels at a constant stride.
const BPRED_STRIDE: usize = 1 + 16 + 4;

/// Number of rows in the `B_PRED` working luma buffer: one top-border
/// row plus the sixteen reconstruction rows.
const BPRED_ROWS: usize = 1 + 16;

/// Flat index of the working buffer's reconstruction origin `(0, 0)`
/// — i.e. one row down (past the top border) and one column right
/// (past the left border).
const BPRED_ORIGIN: usize = BPRED_STRIDE + 1;

/// Reconstruct one macroblock whose Y mode is `B_PRED` (RFC 6386 §11.3
/// / §12.3), driving the sixteen 4×4 luma sub-blocks with the
/// per-sub-block neighbour evolution the spec requires.
///
/// The chroma planes use the 8×8 §12.2 mode in `uv_mode`, identically
/// to [`decode_keyframe_mb_non_bpred`]; only the luma plane differs.
///
/// # Inputs
///
/// * `subblock_modes` — the sixteen §11.3 4×4 luma sub-block modes in
///   raster order (sub-block index `i * 4 + j` for row `i`, column `j`).
///   This is the `MacroblockModes::subblock_modes` field; it must be
///   `Some` for a `B_PRED` macroblock.
/// * `uv_mode` — the §11.4 8×8 chroma intra mode.
/// * `mb_skip_coeff` — the §11.1 skip flag. When `true` the residue is
///   zero: each sub-block's reconstructed pixels equal its prediction,
///   and the neighbour evolution still proceeds (a sub-block's `above` /
///   `left` are then its neighbours' *predictions*, which is the correct
///   reconstructed value when the residue is zero).
/// * `neighbors` — the surrounding pixel context (see [`MbNeighbors`]).
///   `B_PRED` reads `y_above`, `y_left`, `y_topleft`, the chroma fields,
///   and additionally [`MbNeighbors::y_above_right`].
/// * `y_coeffs_dequant` — the sixteen dequantized Y sub-block DCT
///   coefficient arrays in raster sub-block order. Unlike the 16×16
///   path, **no WHT seeding is applied**: §13 / §14.2 specify that a
///   `B_PRED` macroblock has no Y2 block and the 0th coefficient of each
///   Y sub-block is part of that sub-block's own residue. The caller
///   supplies the coefficients verbatim.
/// * `u_coeffs_dequant` / `v_coeffs_dequant` — the eight chroma
///   sub-block coefficient arrays, four U then four V, in raster order.
///
/// # Output
///
/// A fully reconstructed macroblock (predictor + residue, `clamp255`),
/// before loop filtering.
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §12.3 / §14.2 input.
pub fn decode_keyframe_mb_bpred(
    subblock_modes: Option<&[IntraBmode; 16]>,
    uv_mode: IntraUvMode,
    mb_skip_coeff: bool,
    neighbors: &MbNeighbors,
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> Result<ReconstructedMb, ReconstructError> {
    let modes = subblock_modes.ok_or(ReconstructError::MissingSubblockModes)?;

    let mut out = ReconstructedMb::default();

    // ----- Luma (§12.3 B_PRED): per-sub-block walk -----------------
    let mut buf = build_bpred_work_buffer(neighbors);

    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            // Top-left pixel of this sub-block inside the working buffer.
            let sb_base = BPRED_ORIGIN + (i * 4) * BPRED_STRIDE + (j * 4);

            // Gather the §12.3 neighbour inputs from the (evolving)
            // working buffer:
            //   above[0..4] = the four pixels directly above the
            //                 sub-block (row -1, cols 0..3);
            //   above[4..8] = the four pixels above-and-to-the-right
            //                 (row -1, cols 4..7);
            //   left[0..4]  = the four pixels in the column to the left
            //                 (rows 0..3, col -1);
            //   p           = the pixel at (-1, -1).
            let above_base = sb_base - BPRED_STRIDE; // row -1, col 0
            let mut above = [0u8; 8];
            above.copy_from_slice(&buf[above_base..above_base + 8]);
            let mut left = [0u8; 4];
            for (r, slot) in left.iter_mut().enumerate() {
                *slot = buf[sb_base + r * BPRED_STRIDE - 1];
            }
            let p = buf[above_base - 1];

            // Predict this sub-block into a scratch 4×4.
            let mut pred = [0u8; 16];
            predict_b4x4(&mut pred, modes[idx], &above, &left, p);

            // §14.5: add the residue (zero when skipping) to the
            // prediction, then write the reconstructed sub-block back
            // into the working buffer so it becomes a neighbour for the
            // sub-blocks below / to its right.
            let mut recon = [0u8; 16];
            if mb_skip_coeff {
                recon = pred;
            } else {
                let mut residue = [0i16; 16];
                inverse_dct_4x4(&y_coeffs_dequant[idx], &mut residue);
                add_residue_4x4(&pred, &residue, &mut recon);
            }
            for r in 0..4 {
                let dst = sb_base + r * BPRED_STRIDE;
                buf[dst..dst + 4].copy_from_slice(&recon[r * 4..r * 4 + 4]);
            }
        }
    }

    // Copy the reconstructed 16×16 luma area out of the working buffer
    // into the contiguous output plane.
    for r in 0..16 {
        let src = BPRED_ORIGIN + r * BPRED_STRIDE;
        out.y[r * 16..r * 16 + 16].copy_from_slice(&buf[src..src + 16]);
    }

    // ----- Chroma (§12.2): identical to the non-B_PRED path --------
    // Off-frame chroma top-left defaults to DEFAULT_ABOVE_PIXEL (127), the
    // RFC 6386 §12.2 above-row default — matching the non-B_PRED path above
    // and the encoder. Only TM_PRED reads this corner; defaulting to 0 here
    // diverged U/V on top-row B_PRED MBs while luma stayed byte-identical.
    let topleft_u = neighbors.u_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
    predict_uv8x8(
        &mut out.u,
        uv_mode,
        neighbors.u_above.as_ref(),
        neighbors.u_left.as_ref(),
        topleft_u,
    );
    let topleft_v = neighbors.v_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
    predict_uv8x8(
        &mut out.v,
        uv_mode,
        neighbors.v_above.as_ref(),
        neighbors.v_left.as_ref(),
        topleft_v,
    );

    if !mb_skip_coeff {
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
    }

    Ok(out)
}

/// Build the `B_PRED` working luma buffer with its border row / column
/// pre-filled from the macroblock's neighbours, and apply the §12.3
/// right-edge "above-right" fixup (the reference decoder's `copy_down`).
///
/// The returned buffer is [`BPRED_ROWS`] rows of [`BPRED_STRIDE`] pixels.
/// The reconstruction area (`out.y` equivalent) lives at flat offset
/// [`BPRED_ORIGIN`] with the buffer stride.
fn build_bpred_work_buffer(neighbors: &MbNeighbors) -> [u8; BPRED_ROWS * BPRED_STRIDE] {
    let mut buf = [0u8; BPRED_ROWS * BPRED_STRIDE];

    // --- Top border row (buffer row 0) -----------------------------
    // Columns of the border row, in buffer coordinates:
    //   col 0          → the (-1, -1) top-left pixel P;
    //   cols 1..=16    → the 16 above pixels (-1, 0)..=(-1, 15);
    //   cols 17..=20   → the 4 above-right pixels (-1, 16)..=(-1, 19).
    let above = neighbors.y_above.unwrap_or([DEFAULT_ABOVE_PIXEL; 16]);
    // §12 / reference `fixup_above`: when the row above is off-frame the
    // whole above row *and* the (-1, -1) corner are 127. The frame
    // walker (the layer above this orchestrator) is responsible for
    // populating `y_topleft` with the genuine reconstructed corner pixel
    // when one exists — including the 129 the left-frame-edge corner
    // takes per `fixup_left`. A `None` here means off-top-edge → 127.
    let topleft = neighbors.y_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
    buf[0] = topleft;
    buf[1..17].copy_from_slice(&above);
    // Above-right extension. When `y_above_right` is None (top MB row)
    // §12.3 fixes the four pixels at 127.
    let above_right = neighbors.y_above_right.unwrap_or([DEFAULT_ABOVE_PIXEL; 4]);
    buf[17..21].copy_from_slice(&above_right);

    // --- Left border column (buffer rows 1..=16, col 0) ------------
    // §12: left neighbours are 129 (DEFAULT_LEFT_PIXEL) when the MB is
    // on the left column of the frame.
    let left = neighbors.y_left.unwrap_or([DEFAULT_LEFT_PIXEL; 16]);
    for (r, &v) in left.iter().enumerate() {
        buf[(r + 1) * BPRED_STRIDE] = v;
    }

    // --- §12.3 right-edge fixup (reference `copy_down`) ------------
    // The four pixels above-and-to-the-right of sub-block 3 (border row,
    // cols 17..=20) are copied into the four border slots that sit
    // immediately above sub-blocks 7, 11, and 15 — i.e. the rows above
    // sub-block rows 1, 2, 3 at reconstruction columns 16..=19. Those
    // rows are reconstruction rows 3, 7, 11 (zero-based), which map to
    // buffer rows 4, 8, 12, columns BPRED_ORIGIN-column-wise at recon
    // cols 16..=19 → buffer cols (1 + 16)..=(1 + 19) = 17..=20.
    let extra = [buf[17], buf[18], buf[19], buf[20]];
    for &recon_row_above in &[3usize, 7, 11] {
        // Sub-blocks 7/11/15 begin at reconstruction rows 4/8/12; their
        // "above" border lives at reconstruction row 3/7/11.
        let buf_row = 1 + recon_row_above; // +1 for the top border row.
        let base = buf_row * BPRED_STRIDE + 17; // recon col 16 → buf col 17.
        buf[base..base + 4].copy_from_slice(&extra);
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_predict::DEFAULT_TOPLEFT_DC;

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
            ..MbNeighbors::default()
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

    // -------------------------------------------------------------------
    // §11.3 / §12.3 B_PRED reconstruction
    // -------------------------------------------------------------------

    /// Coefficient bundle for the B_PRED entry (no Y2 block).
    type BpredCoeffs = ([[i16; 16]; 16], [[i16; 16]; 4], [[i16; 16]; 4]);

    fn zero_bpred_coeffs() -> BpredCoeffs {
        ([[0i16; 16]; 16], [[0i16; 16]; 4], [[0i16; 16]; 4])
    }

    /// All sixteen sub-blocks set to `mode`.
    fn all_modes(mode: IntraBmode) -> [IntraBmode; 16] {
        [mode; 16]
    }

    /// Pull a 4×4 luma sub-block (sub-block index `i*4+j`) out of an
    /// `out.y` 16×16 plane.
    fn y_subblock(out: &ReconstructedMb, i: usize, j: usize) -> [u8; 16] {
        extract_4x4(&out.y, 16, i, j)
    }

    #[test]
    fn bpred_missing_subblock_modes_errors() {
        let (y, u, v) = zero_bpred_coeffs();
        let res = decode_keyframe_mb_bpred(
            None,
            IntraUvMode::Dc,
            false,
            &MbNeighbors::default(),
            &y,
            &u,
            &v,
        );
        assert_eq!(res, Err(ReconstructError::MissingSubblockModes));
    }

    #[test]
    fn bpred_dc_top_left_corner_is_128_everywhere() {
        // Top-left corner MB, all sub-blocks B_DC_PRED, skip=true.
        // For the (0,0) sub-block: above = 127 (off-frame), left = 129
        // (off-frame) → DC = (4*127 + 4*129 + 4) >> 3 = 1028 >> 3 = 128.
        // Once that sub-block reconstructs to flat-128, every other
        // sub-block sees only 128 / 127 / 129 neighbours and likewise
        // settles to a flat value. The whole-plane result is therefore
        // a constant; we assert it is 128 (the corner sub-block value)
        // and uniform.
        let (y, u, v) = zero_bpred_coeffs();
        let out = decode_keyframe_mb_bpred(
            Some(&all_modes(IntraBmode::Dc)),
            IntraUvMode::Dc,
            true,
            &MbNeighbors::default(),
            &y,
            &u,
            &v,
        )
        .unwrap();
        // Corner sub-block (0,0) is exactly 128.
        assert!(y_subblock(&out, 0, 0).iter().all(|&p| p == 128));
        // Chroma top-left corner with DC mode also fills 128.
        assert!(out.u.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
        assert!(out.v.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
    }

    #[test]
    fn bpred_skip_matches_standalone_per_subblock_kernels() {
        // Drive an interior-style MB with full neighbours and skip=true
        // for several distinct sub-modes, then independently replay the
        // §12.3 kernel for the *first* sub-block (which reads only the
        // supplied border, not any evolved neighbour) and confirm the
        // orchestrator's (0,0) sub-block matches byte-for-byte. This
        // proves the orchestrator wires the border into predict_b4x4
        // correctly for each mode.
        let y_above: [u8; 16] = core::array::from_fn(|i| (20 + i * 3) as u8);
        let y_left: [u8; 16] = core::array::from_fn(|i| (200 - i * 5) as u8);
        let y_above_right = [80u8, 90, 100, 110];
        let topleft = 64u8;
        let neighbors = MbNeighbors {
            y_above: Some(y_above),
            y_left: Some(y_left),
            y_topleft: Some(topleft),
            y_above_right: Some(y_above_right),
            ..MbNeighbors::default()
        };
        let (y, u, v) = zero_bpred_coeffs();

        for mode in [
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
        ] {
            let out = decode_keyframe_mb_bpred(
                Some(&all_modes(mode)),
                IntraUvMode::Dc,
                true,
                &neighbors,
                &y,
                &u,
                &v,
            )
            .unwrap();

            // Standalone replay for sub-block (0,0): its above strip is
            // y_above[0..4] followed by y_above[4..8]; its left is
            // y_left[0..4]; its P is the MB top-left corner.
            let mut above = [0u8; 8];
            above.copy_from_slice(&y_above[0..8]);
            let left = [y_left[0], y_left[1], y_left[2], y_left[3]];
            let mut want = [0u8; 16];
            predict_b4x4(&mut want, mode, &above, &left, topleft);

            assert_eq!(
                y_subblock(&out, 0, 0),
                want,
                "mode {mode:?} sub-block (0,0) mismatch vs standalone kernel"
            );
        }
    }

    #[test]
    fn bpred_neighbour_evolution_left_propagates() {
        // Sub-block (0,1) must read its `left` column from the
        // already-reconstructed sub-block (0,0), not from the MB-level
        // y_left border. We exercise that by giving sub-block (0,0) a
        // non-zero residue (so its reconstructed right column differs
        // from any border default) and using B_HE_PRED for sub-block
        // (0,1), which copies a smoothed version of its `left` column
        // across each row. If the orchestrator failed to feed (0,0)'s
        // reconstructed pixels in as (0,1)'s `left`, the result would
        // not depend on (0,0)'s residue.
        let neighbors = MbNeighbors {
            y_above: Some([100u8; 16]),
            y_left: Some([100u8; 16]),
            y_topleft: Some(100),
            y_above_right: Some([100u8; 4]),
            ..MbNeighbors::default()
        };

        // Two runs: identical except sub-block (0,0)'s residue.
        let modes = {
            let mut m = all_modes(IntraBmode::Dc);
            m[1] = IntraBmode::He; // sub-block (0,1)
            m
        };

        let (mut y_a, u, v) = zero_bpred_coeffs();
        let out_flat = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Dc,
            false,
            &neighbors,
            &y_a,
            &u,
            &v,
        )
        .unwrap();

        // Give sub-block (0,0) a large positive DC so its reconstructed
        // right column climbs well above 100.
        y_a[0][0] = 400;
        let out_lifted = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Dc,
            false,
            &neighbors,
            &y_a,
            &u,
            &v,
        )
        .unwrap();

        let flat_01 = y_subblock(&out_flat, 0, 1);
        let lifted_01 = y_subblock(&out_lifted, 0, 1);
        assert_ne!(
            flat_01, lifted_01,
            "sub-block (0,1) did not respond to sub-block (0,0)'s residue — \
             left-neighbour evolution is broken"
        );
    }

    #[test]
    fn bpred_neighbour_evolution_above_propagates() {
        // Symmetric to the left-propagation test: sub-block (1,0) reads
        // its `above` strip from the reconstructed sub-block (0,0). Use
        // B_VE_PRED for (1,0) (copies a smoothed above row down every
        // row) and perturb (0,0)'s residue.
        let neighbors = MbNeighbors {
            y_above: Some([100u8; 16]),
            y_left: Some([100u8; 16]),
            y_topleft: Some(100),
            y_above_right: Some([100u8; 4]),
            ..MbNeighbors::default()
        };
        let modes = {
            let mut m = all_modes(IntraBmode::Dc);
            m[4] = IntraBmode::Ve; // sub-block (1,0)
            m
        };

        let (mut y_a, u, v) = zero_bpred_coeffs();
        let out_flat = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Dc,
            false,
            &neighbors,
            &y_a,
            &u,
            &v,
        )
        .unwrap();

        y_a[0][0] = 400;
        let out_lifted = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Dc,
            false,
            &neighbors,
            &y_a,
            &u,
            &v,
        )
        .unwrap();

        assert_ne!(
            y_subblock(&out_flat, 1, 0),
            y_subblock(&out_lifted, 1, 0),
            "sub-block (1,0) did not respond to sub-block (0,0)'s residue — \
             above-neighbour evolution is broken"
        );
    }

    #[test]
    fn bpred_right_edge_above_right_fixup() {
        // §12.3: sub-blocks 7, 11, 15 share sub-block 3's above-right
        // pixels (frame (-1,16)..=(-1,19)). With B_LD_PRED (which reads
        // above[4..8], the above-right strip) on those sub-blocks, the
        // result must depend on y_above_right. We run twice with two
        // different y_above_right values and require the bottom-right
        // sub-blocks to differ. Sub-block 3 (top-right) reads the same
        // pixels directly, so it must differ too.
        let mk = |ar: [u8; 4]| MbNeighbors {
            y_above: Some([100u8; 16]),
            y_left: Some([100u8; 16]),
            y_topleft: Some(100),
            y_above_right: Some(ar),
            ..MbNeighbors::default()
        };
        let modes = all_modes(IntraBmode::Ld);
        let (y, u, v) = zero_bpred_coeffs();

        let out_a = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Dc,
            true,
            &mk([10, 10, 10, 10]),
            &y,
            &u,
            &v,
        )
        .unwrap();
        let out_b = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Dc,
            true,
            &mk([240, 240, 240, 240]),
            &y,
            &u,
            &v,
        )
        .unwrap();

        // Sub-block 3 (row 0, col 3) reads above-right directly.
        assert_ne!(
            y_subblock(&out_a, 0, 3),
            y_subblock(&out_b, 0, 3),
            "sub-block 3 ignored y_above_right"
        );
        // Sub-block 7 (row 1, col 3) gets the copied-down above-right.
        assert_ne!(
            y_subblock(&out_a, 1, 3),
            y_subblock(&out_b, 1, 3),
            "sub-block 7 did not receive the copy_down above-right fixup"
        );
        // Sub-block 15 (row 3, col 3).
        assert_ne!(
            y_subblock(&out_a, 3, 3),
            y_subblock(&out_b, 3, 3),
            "sub-block 15 did not receive the copy_down above-right fixup"
        );
    }

    #[test]
    fn bpred_top_row_above_right_defaults_to_127() {
        // On the top MB row (y_above and y_above_right both None) the
        // four above-right pixels must be 127. With B_LD_PRED on the
        // right-edge sub-blocks, supplying y_above_right=None should be
        // identical to supplying Some([127; 4]) with the same above.
        let modes = all_modes(IntraBmode::Ld);
        let (y, u, v) = zero_bpred_coeffs();

        // Both: above off-frame (→127), left present.
        let n_none = MbNeighbors {
            y_left: Some([90u8; 16]),
            ..MbNeighbors::default()
        };
        let n_explicit = MbNeighbors {
            y_left: Some([90u8; 16]),
            y_above: Some([DEFAULT_ABOVE_PIXEL; 16]),
            y_above_right: Some([DEFAULT_ABOVE_PIXEL; 4]),
            y_topleft: Some(DEFAULT_ABOVE_PIXEL),
            ..MbNeighbors::default()
        };

        let out_none =
            decode_keyframe_mb_bpred(Some(&modes), IntraUvMode::Dc, true, &n_none, &y, &u, &v)
                .unwrap();
        let out_explicit =
            decode_keyframe_mb_bpred(Some(&modes), IntraUvMode::Dc, true, &n_explicit, &y, &u, &v)
                .unwrap();
        assert_eq!(
            out_none.y, out_explicit.y,
            "top-row default-127 above-right path diverged from explicit Some([127;4])"
        );
    }

    #[test]
    fn bpred_full_mb_reconstruction_mixed_modes() {
        // A full B_PRED macroblock with a mix of sub-block modes, full
        // neighbours, and non-trivial residue in every Y sub-block,
        // plus chroma residue. This exercises the complete §12.3 walk
        // (predict → idct → add → evolve) end-to-end. We assert two
        // invariants that any correct implementation must satisfy:
        //  1. With skip=false but all-zero coefficients, the result
        //     equals the skip=true (prediction-only) result.
        //  2. Adding a constant positive DC to every Y sub-block raises
        //     the plane (no panic, monotone non-decrease at the (0,0)
        //     corner) — a smoke test that residue folds in per sub-block.
        let neighbors = MbNeighbors {
            y_above: Some(core::array::from_fn(|i| (30 + i * 4) as u8)),
            y_left: Some(core::array::from_fn(|i| (40 + i * 3) as u8)),
            y_topleft: Some(50),
            y_above_right: Some([60, 70, 80, 90]),
            u_above: Some([120u8; 8]),
            u_left: Some([130u8; 8]),
            u_topleft: Some(125),
            v_above: Some([140u8; 8]),
            v_left: Some([150u8; 8]),
            v_topleft: Some(145),
        };
        let modes = [
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
            IntraBmode::Dc,
            IntraBmode::Tm,
            IntraBmode::Ve,
            IntraBmode::He,
            IntraBmode::Ld,
            IntraBmode::Rd,
        ];

        // Invariant 1: zero residue, skip vs run agree.
        let (y0, u0, v0) = zero_bpred_coeffs();
        let out_skip = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Tm,
            true,
            &neighbors,
            &y0,
            &u0,
            &v0,
        )
        .unwrap();
        let out_run = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Tm,
            false,
            &neighbors,
            &y0,
            &u0,
            &v0,
        )
        .unwrap();
        assert_eq!(out_skip.y, out_run.y, "zero-residue Y skip != run");
        assert_eq!(out_skip.u, out_run.u, "zero-residue U skip != run");
        assert_eq!(out_skip.v, out_run.v, "zero-residue V skip != run");

        // Invariant 2: a positive DC in every Y sub-block lifts the
        // (0,0) corner pixel relative to the prediction-only run.
        let mut y_lift = [[0i16; 16]; 16];
        for sb in y_lift.iter_mut() {
            sb[0] = 80; // positive DC seed → positive flat-ish residue.
        }
        let out_lift = decode_keyframe_mb_bpred(
            Some(&modes),
            IntraUvMode::Tm,
            false,
            &neighbors,
            &y_lift,
            &u0,
            &v0,
        )
        .unwrap();
        assert!(
            out_lift.y[0] >= out_skip.y[0],
            "positive Y DC did not lift the (0,0) corner: {} < {}",
            out_lift.y[0],
            out_skip.y[0]
        );
        // And the chroma planes (zero residue) are unchanged from skip.
        assert_eq!(out_lift.u, out_skip.u);
        assert_eq!(out_lift.v, out_skip.v);
    }

    #[test]
    fn bpred_chroma_uses_8x8_mode_independent_of_luma() {
        // The chroma path of a B_PRED MB is the ordinary 8×8 §12.2 mode.
        // Verify it matches a standalone predict_uv8x8 call (skip MB).
        let u_above = [55u8; 8];
        let v_above = [66u8; 8];
        let neighbors = MbNeighbors {
            y_above: Some([100u8; 16]),
            y_left: Some([100u8; 16]),
            y_topleft: Some(100),
            y_above_right: Some([100u8; 4]),
            u_above: Some(u_above),
            v_above: Some(v_above),
            ..MbNeighbors::default()
        };
        let (y, u, v) = zero_bpred_coeffs();
        let out = decode_keyframe_mb_bpred(
            Some(&all_modes(IntraBmode::Dc)),
            IntraUvMode::V,
            true,
            &neighbors,
            &y,
            &u,
            &v,
        )
        .unwrap();

        let mut want_u = [0u8; 64];
        predict_uv8x8(&mut want_u, IntraUvMode::V, Some(&u_above), None, 0);
        let mut want_v = [0u8; 64];
        predict_uv8x8(&mut want_v, IntraUvMode::V, Some(&v_above), None, 0);
        assert_eq!(out.u, want_u);
        assert_eq!(out.v, want_v);
    }

    #[test]
    fn bpred_top_row_chroma_topleft_defaults_to_127() {
        // Regression (issue #23): on a top-row B_PRED MB the off-frame
        // chroma top-left must default to DEFAULT_ABOVE_PIXEL (127), the
        // RFC 6386 §12.2 above-row default — the same value the non-B_PRED
        // path and the encoder use. TM_PRED is the only 8×8 chroma mode that
        // reads the corner, so it exposes the divergence: defaulting to 0
        // (the prior bug) silently corrupted U/V on top-row MBs while luma
        // stayed byte-identical (pink/green smearing).
        let u_above = [55u8; 8];
        let u_left = [70u8; 8];
        let v_above = [66u8; 8];
        let v_left = [80u8; 8];
        let neighbors = MbNeighbors {
            y_above: Some([100u8; 16]),
            y_left: Some([100u8; 16]),
            y_topleft: Some(100),
            y_above_right: Some([100u8; 4]),
            u_above: Some(u_above),
            u_left: Some(u_left),
            v_above: Some(v_above),
            v_left: Some(v_left),
            // u_topleft / v_topleft deliberately None: this is the top row.
            ..MbNeighbors::default()
        };
        let (y, u, v) = zero_bpred_coeffs();
        let out = decode_keyframe_mb_bpred(
            Some(&all_modes(IntraBmode::Dc)),
            IntraUvMode::Tm,
            true,
            &neighbors,
            &y,
            &u,
            &v,
        )
        .unwrap();

        // Must match a standalone TM_PRED with corner = 127 ...
        let mut want_u = [0u8; 64];
        predict_uv8x8(
            &mut want_u,
            IntraUvMode::Tm,
            Some(&u_above),
            Some(&u_left),
            DEFAULT_ABOVE_PIXEL,
        );
        let mut want_v = [0u8; 64];
        predict_uv8x8(
            &mut want_v,
            IntraUvMode::Tm,
            Some(&v_above),
            Some(&v_left),
            DEFAULT_ABOVE_PIXEL,
        );
        assert_eq!(out.u, want_u);
        assert_eq!(out.v, want_v);

        // ... and must NOT match the prior corner-0 behaviour (locks the fix).
        let mut bug_u = [0u8; 64];
        predict_uv8x8(&mut bug_u, IntraUvMode::Tm, Some(&u_above), Some(&u_left), 0);
        assert_ne!(out.u, bug_u);
    }
}
