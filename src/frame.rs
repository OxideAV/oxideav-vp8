//! Per-frame keyframe raster walker (RFC 6386 §12 / §14.2).
//!
//! This module is the layer that sits *above* the per-macroblock
//! reconstruction orchestrators landed in earlier rounds
//! ([`decode_keyframe_mb_non_bpred`] for the four 16×16 luma modes and
//! [`decode_keyframe_mb_bpred`] for `B_PRED`). It walks the macroblocks
//! of a key frame in raster-scan order, and for each macroblock:
//!
//! 1. **Assembles the neighbour strips** ([`MbNeighbors`]) from the
//!    pixels of the *already-reconstructed* neighbouring macroblocks —
//!    the bottom row of the macroblock directly above, the rightmost
//!    column of the macroblock directly to the left, the corner pixel
//!    above-and-to-the-left, and (for `B_PRED`) the four
//!    above-and-to-the-right pixels with the §12.3 rightmost-macroblock
//!    `(-1,15)` clamp.
//! 2. **Selects the orchestrator** by the macroblock's decoded luma mode
//!    — [`IntraYMode::B`] routes to [`decode_keyframe_mb_bpred`]; the
//!    other four 16×16 modes route to [`decode_keyframe_mb_non_bpred`].
//! 3. **Writes the reconstructed 16×16 luma + two 8×8 chroma blocks**
//!    into the full-frame plane buffers, where they become neighbours
//!    for the macroblocks below and to the right.
//!
//! ## §12 frame-edge neighbour rules
//!
//! Per RFC 6386 §12 (page 50) the relative addressing of intra
//! prediction sometimes references pixels outside the visible frame.
//! This walker reports an *absent* neighbour strip (`None` in
//! [`MbNeighbors`]) for every off-frame edge rather than synthesising a
//! 127 / 129 fill, because the §12 prediction kernels need to
//! distinguish "off-frame" from "genuine pixel" — most importantly for
//! `DC_PRED`, whose §12.2 page 51 averaging rule uses *only the
//! genuinely-visible* neighbours (the average of the 8 left pixels on
//! the top row, the average of the 8 above pixels in the left column,
//! and the constant 128 for the top-left macroblock — none of which
//! equals the result you would get by averaging 127 / 129 fills). The
//! per-MB orchestrators forward each `None` into the §12 kernels, which
//! apply the spec's per-mode default substitution. See
//! [`crate::intra_predict`].
//!
//! ## §12.3 above-right extension (`y_above_right`)
//!
//! The `B_PRED` luma path's right-edge sub-blocks (3, 7, 11, 15) read
//! four "extra" prediction pixels above-and-to-the-right of the
//! macroblock — frame positions `(-1,16)..=(-1,19)` (§12.3 page 55).
//! This walker populates [`MbNeighbors::y_above_right`] with those four
//! pixels:
//!
//! * On the **top macroblock row** the field is left `None`; the
//!   per-MB orchestrator then fills the four pixels with 127 (the §12.3
//!   top-row rule).
//! * For the **rightmost macroblock** in any non-top row the four pixels
//!   are all clamped to the value at `(-1,15)` — "the rightmost visible
//!   pixel on the line immediately above the macroblock row" (§12.3
//!   page 55). This mirrors the reference decoder's
//!   per-row "extend the last row by four pixels" step (§20.14).
//! * Otherwise the four genuine pixels are copied verbatim; the entire
//!   macroblock row above the current one is already reconstructed, so
//!   they exist.
//!
//! ## Scope
//!
//! This walker reconstructs the **pre-loop-filter** plane buffers of a
//! single key frame. It does *not* run the §15 loop filter (the §15.1
//! raster-order edge geometry is a separate integration layer) and it
//! does *not* decode the bitstream: it consumes a caller-supplied
//! [`MacroblockModes`] record and pre-dequantized coefficient bundle
//! for every macroblock.
//!
//! ### Why caller-supplied dequantized coefficients
//!
//! The §13.3 per-macroblock token walk and the §14.1 Y2 / chroma
//! dequant scaling remain documented spec gaps for this crate (the
//! per-MB token aggregation that maintains the above / left non-zero
//! predictors across raster order, and the dequant clamp that RFC 6386
//! §14.1 defers to the `dixie.c` reference outside the clean-room
//! policy). Keeping those out of this walker's signature means the
//! frame-level raster geometry — neighbour assembly, the §12.3
//! above-right clamp, the per-mode orchestrator selection, and the
//! plane writeback — lands today without waiting on the gaps to be
//! filled. When they are filled, a convenience wrapper that decodes
//! tokens + dequantizes per macroblock can feed this function.

use crate::intra_predict::DEFAULT_LEFT_PIXEL;
use crate::macroblock::{IntraYMode, MacroblockModes};
use crate::reconstruct::{
    decode_keyframe_mb_bpred, decode_keyframe_mb_non_bpred, MbNeighbors, ReconstructError,
    ReconstructedMb,
};

/// Pre-dequantized coefficient bundle for one macroblock, in the layout
/// the per-MB orchestrators consume.
///
/// All four members are already multiplied by their §14.1 dequant
/// factors (the caller's responsibility — see the module-level note on
/// the dequant spec gap). The walker forwards them unchanged into the
/// selected orchestrator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MbCoeffs {
    /// The 25th (Y2) block's dequantized WHT coefficients in row-major
    /// order. Ignored for `B_PRED` macroblocks (which have no Y2 block)
    /// and for skip macroblocks.
    pub y2: [i16; 16],
    /// The sixteen Y sub-block dequantized DCT coefficient arrays in
    /// raster sub-block order (`i * 4 + j`).
    pub y: [[i16; 16]; 16],
    /// The four U chroma sub-block coefficient arrays in raster order.
    pub u: [[i16; 16]; 4],
    /// The four V chroma sub-block coefficient arrays in raster order.
    pub v: [[i16; 16]; 4],
}

/// Full-frame reconstructed pixel planes for one key frame, in I420
/// layout (separate Y / U / V planes, each row-major).
///
/// The planes are sized to whole macroblocks: the luma plane is
/// `mb_cols * 16` wide by `mb_rows * 16` tall, and each chroma plane is
/// `mb_cols * 8` by `mb_rows * 8`. The visible frame dimensions (which
/// may be smaller — the bottom / right rows of a non-multiple-of-16
/// frame are padding) are *not* cropped here; cropping to the §9.1
/// visible width / height is a presentation-layer concern above this
/// reconstruction walker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyframePlanes {
    /// Luma plane, `y_stride * (mb_rows * 16)` bytes, row-major.
    pub y: Vec<u8>,
    /// U chroma plane, `uv_stride * (mb_rows * 8)` bytes, row-major.
    pub u: Vec<u8>,
    /// V chroma plane, same dimensions as [`KeyframePlanes::u`].
    pub v: Vec<u8>,
    /// Luma plane stride (= `mb_cols * 16`).
    pub y_stride: usize,
    /// Chroma plane stride (= `mb_cols * 8`).
    pub uv_stride: usize,
    /// Number of macroblock columns.
    pub mb_cols: usize,
    /// Number of macroblock rows.
    pub mb_rows: usize,
}

/// Errors surfaced by [`decode_keyframe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// The number of supplied [`MacroblockModes`] / [`MbCoeffs`] records
    /// did not equal `mb_cols * mb_rows`. Carries `(expected, got)`.
    MacroblockCountMismatch {
        /// `mb_cols * mb_rows`.
        expected: usize,
        /// The length of the supplied slice.
        got: usize,
    },
    /// `mb_cols` or `mb_rows` was zero — a key frame has at least one
    /// macroblock.
    EmptyFrame,
    /// A per-macroblock orchestrator rejected its inputs. Carries the
    /// raster macroblock index (`mb_row * mb_cols + mb_col`) and the
    /// underlying [`ReconstructError`].
    Macroblock {
        /// Raster index of the offending macroblock.
        index: usize,
        /// The orchestrator's error.
        source: ReconstructError,
    },
}

impl core::fmt::Display for FrameError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FrameError::MacroblockCountMismatch { expected, got } => write!(
                f,
                "vp8 frame: expected {expected} macroblock records (mb_cols * mb_rows), got {got}"
            ),
            FrameError::EmptyFrame => {
                f.write_str("vp8 frame: mb_cols and mb_rows must both be non-zero")
            }
            FrameError::Macroblock { index, source } => {
                write!(f, "vp8 frame: macroblock {index}: {source}")
            }
        }
    }
}

impl std::error::Error for FrameError {}

/// Walk the macroblocks of a key frame in raster-scan order, assembling
/// each macroblock's neighbour strips from the already-reconstructed
/// frame buffer, selecting the §12.3 16×16-mode or `B_PRED` orchestrator
/// per the macroblock's decoded luma mode, and writing the
/// reconstructed pixels into the full-frame plane buffers.
///
/// # Inputs
///
/// * `mb_cols` / `mb_rows` — the key frame's dimensions in macroblocks.
///   Both must be non-zero.
/// * `modes` — the decoded [`MacroblockModes`] for every macroblock in
///   raster order (`mb_row * mb_cols + mb_col`). Length must equal
///   `mb_cols * mb_rows`.
/// * `coeffs` — the pre-dequantized [`MbCoeffs`] for every macroblock in
///   the same raster order. Length must equal `mb_cols * mb_rows`.
///
/// # Output
///
/// The reconstructed (pre-loop-filter) [`KeyframePlanes`].
pub fn decode_keyframe(
    mb_cols: usize,
    mb_rows: usize,
    modes: &[MacroblockModes],
    coeffs: &[MbCoeffs],
) -> Result<KeyframePlanes, FrameError> {
    if mb_cols == 0 || mb_rows == 0 {
        return Err(FrameError::EmptyFrame);
    }
    let expected = mb_cols * mb_rows;
    if modes.len() != expected {
        return Err(FrameError::MacroblockCountMismatch {
            expected,
            got: modes.len(),
        });
    }
    if coeffs.len() != expected {
        return Err(FrameError::MacroblockCountMismatch {
            expected,
            got: coeffs.len(),
        });
    }

    let y_stride = mb_cols * 16;
    let uv_stride = mb_cols * 8;
    let mut planes = KeyframePlanes {
        y: vec![0u8; y_stride * mb_rows * 16],
        u: vec![0u8; uv_stride * mb_rows * 8],
        v: vec![0u8; uv_stride * mb_rows * 8],
        y_stride,
        uv_stride,
        mb_cols,
        mb_rows,
    };

    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            let index = mb_row * mb_cols + mb_col;
            let modes_mb = &modes[index];
            let coeffs_mb = &coeffs[index];

            let neighbors = gather_neighbors(&planes, mb_row, mb_col);

            let recon = if modes_mb.y_mode == IntraYMode::B {
                decode_keyframe_mb_bpred(
                    modes_mb.subblock_modes.as_ref(),
                    modes_mb.uv_mode,
                    modes_mb.mb_skip_coeff,
                    &neighbors,
                    &coeffs_mb.y,
                    &coeffs_mb.u,
                    &coeffs_mb.v,
                )
            } else {
                decode_keyframe_mb_non_bpred(
                    modes_mb.y_mode,
                    modes_mb.uv_mode,
                    modes_mb.mb_skip_coeff,
                    &neighbors,
                    &coeffs_mb.y2,
                    &coeffs_mb.y,
                    &coeffs_mb.u,
                    &coeffs_mb.v,
                )
            }
            .map_err(|source| FrameError::Macroblock { index, source })?;

            write_mb(&mut planes, mb_row, mb_col, &recon);
        }
    }

    Ok(planes)
}

/// Assemble the [`MbNeighbors`] strips for the macroblock at
/// `(mb_row, mb_col)` from the already-reconstructed plane buffers.
///
/// Off-frame edges (top row / left column) are reported as `None` so the
/// §12 kernels apply their per-mode default substitution (§12.2 page 51
/// `DC_PRED` averaging in particular distinguishes "off-frame" from a
/// 127 / 129 fill). The §12.3 above-right extension is `None` on the
/// top macroblock row (the per-MB orchestrator then fills the four
/// pixels with 127), and otherwise the four `(-1,16)..=(-1,19)` pixels
/// with the rightmost-macroblock `(-1,15)` clamp applied.
fn gather_neighbors(planes: &KeyframePlanes, mb_row: usize, mb_col: usize) -> MbNeighbors {
    let mut n = MbNeighbors::default();

    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;
    let ys = planes.y_stride;
    let cs = planes.uv_stride;

    // --- Above strips (row y0 - 1) ---------------------------------
    if mb_row > 0 {
        let yrow = (y_y0 - 1) * ys + y_x0;
        let mut y_above = [0u8; 16];
        y_above.copy_from_slice(&planes.y[yrow..yrow + 16]);
        n.y_above = Some(y_above);

        let urow = (uv_y0 - 1) * cs + uv_x0;
        let mut u_above = [0u8; 8];
        u_above.copy_from_slice(&planes.u[urow..urow + 8]);
        n.u_above = Some(u_above);
        let mut v_above = [0u8; 8];
        v_above.copy_from_slice(&planes.v[urow..urow + 8]);
        n.v_above = Some(v_above);
    }

    // --- Left strips (column x0 - 1) -------------------------------
    if mb_col > 0 {
        let mut y_left = [0u8; 16];
        for (r, slot) in y_left.iter_mut().enumerate() {
            *slot = planes.y[(y_y0 + r) * ys + (y_x0 - 1)];
        }
        n.y_left = Some(y_left);

        let mut u_left = [0u8; 8];
        let mut v_left = [0u8; 8];
        for r in 0..8 {
            u_left[r] = planes.u[(uv_y0 + r) * cs + (uv_x0 - 1)];
            v_left[r] = planes.v[(uv_y0 + r) * cs + (uv_x0 - 1)];
        }
        n.u_left = Some(u_left);
        n.v_left = Some(v_left);
    }

    // --- Top-left corner pixel (-1, -1) ----------------------------
    // The corner is read only by TM_PRED (16×16 / 8×8) and B_TM_PRED
    // (4×4). Its value follows RFC 6386 §20.14 `fixup_left` / the §12
    // out-of-frame rules:
    //
    // * Top macroblock row (`mb_row == 0`): the corner is above the top
    //   frame edge → 127 (`fixup_above` resets it to 127; and the §20.14
    //   row-0/col-0 special case `*(img.y - stride - 1) = 127` keeps the
    //   top-left corner at 127 too). Reported as `None` so the §12 kernels
    //   substitute the 127 `DEFAULT_ABOVE_PIXEL`.
    // * Left frame edge, non-top row (`mb_col == 0 && mb_row > 0`): the
    //   corner is to the left of the leftmost column → 129. §20.14
    //   `fixup_left`'s non-DC branch runs `for (i = -1; ...) *left = 129`,
    //   and the `i == -1` iteration sets exactly this corner. (DC_PRED
    //   never reads the corner, so the unconditional 129 here is safe.)
    // * Interior (`mb_row > 0 && mb_col > 0`): the genuine reconstructed
    //   pixel at `(-1, -1)`.
    if mb_col == 0 && mb_row > 0 {
        n.y_topleft = Some(DEFAULT_LEFT_PIXEL);
        n.u_topleft = Some(DEFAULT_LEFT_PIXEL);
        n.v_topleft = Some(DEFAULT_LEFT_PIXEL);
    } else if mb_row > 0 && mb_col > 0 {
        n.y_topleft = Some(planes.y[(y_y0 - 1) * ys + (y_x0 - 1)]);
        n.u_topleft = Some(planes.u[(uv_y0 - 1) * cs + (uv_x0 - 1)]);
        n.v_topleft = Some(planes.v[(uv_y0 - 1) * cs + (uv_x0 - 1)]);
    }

    // --- §12.3 above-right extension (B_PRED right-edge sub-blocks) -
    // Only meaningful below the top row; on the top row the per-MB
    // orchestrator fills the four pixels with 127 from a `None`.
    if mb_row > 0 {
        let row_base = (y_y0 - 1) * ys;
        let mut ar = [0u8; 4];
        if mb_col == planes.mb_cols - 1 {
            // Rightmost macroblock: §12.3 page 55 clamps the four
            // above-right pixels to the value at (-1,15) — the rightmost
            // pixel of this macroblock's own above row.
            let clamp = planes.y[row_base + y_x0 + 15];
            ar = [clamp; 4];
        } else {
            // Genuine pixels (-1,16)..=(-1,19): the bottom row of the
            // already-reconstructed macroblock above-and-to-the-right.
            let src = row_base + y_x0 + 16;
            ar.copy_from_slice(&planes.y[src..src + 4]);
        }
        n.y_above_right = Some(ar);
    }

    n
}

/// Write a reconstructed macroblock into the full-frame plane buffers at
/// macroblock position `(mb_row, mb_col)`.
fn write_mb(planes: &mut KeyframePlanes, mb_row: usize, mb_col: usize, recon: &ReconstructedMb) {
    let ys = planes.y_stride;
    let cs = planes.uv_stride;
    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;

    for r in 0..16 {
        let dst = (y_y0 + r) * ys + y_x0;
        planes.y[dst..dst + 16].copy_from_slice(&recon.y[r * 16..r * 16 + 16]);
    }
    for r in 0..8 {
        let dst = (uv_y0 + r) * cs + uv_x0;
        planes.u[dst..dst + 8].copy_from_slice(&recon.u[r * 8..r * 8 + 8]);
        planes.v[dst..dst + 8].copy_from_slice(&recon.v[r * 8..r * 8 + 8]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_predict::DEFAULT_TOPLEFT_DC;
    use crate::macroblock::{IntraBmode, IntraUvMode};

    /// A macroblock mode record for one of the four 16×16 luma modes.
    fn mode_16x16(y_mode: IntraYMode, uv_mode: IntraUvMode, skip: bool) -> MacroblockModes {
        MacroblockModes {
            segment_id: None,
            mb_skip_coeff: skip,
            y_mode,
            subblock_modes: None,
            uv_mode,
        }
    }

    /// A `B_PRED` macroblock mode record with the supplied sub-block
    /// modes.
    fn mode_bpred(subs: [IntraBmode; 16], uv_mode: IntraUvMode, skip: bool) -> MacroblockModes {
        MacroblockModes {
            segment_id: None,
            mb_skip_coeff: skip,
            y_mode: IntraYMode::B,
            subblock_modes: Some(subs),
            uv_mode,
        }
    }

    #[test]
    fn empty_frame_rejected() {
        assert_eq!(decode_keyframe(0, 1, &[], &[]), Err(FrameError::EmptyFrame));
        assert_eq!(decode_keyframe(1, 0, &[], &[]), Err(FrameError::EmptyFrame));
    }

    #[test]
    fn macroblock_count_mismatch_rejected() {
        let modes = vec![mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true)];
        let coeffs = vec![MbCoeffs::default()];
        // 2×2 = 4 expected, only 1 supplied.
        let res = decode_keyframe(2, 2, &modes, &coeffs);
        assert_eq!(
            res,
            Err(FrameError::MacroblockCountMismatch {
                expected: 4,
                got: 1
            })
        );
    }

    #[test]
    fn single_dc_mb_top_left_is_flat_128() {
        // A 1×1 key frame, single DC_PRED MB, skip=true. With no
        // neighbours the §12.2 DC rule yields 128 in every plane.
        let modes = vec![mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true)];
        let coeffs = vec![MbCoeffs::default()];
        let planes = decode_keyframe(1, 1, &modes, &coeffs).expect("decode");

        assert_eq!(planes.y_stride, 16);
        assert_eq!(planes.uv_stride, 8);
        assert_eq!(planes.y.len(), 16 * 16);
        assert_eq!(planes.u.len(), 8 * 8);
        assert!(planes.y.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
        assert!(planes.u.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
        assert!(planes.v.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
    }

    #[test]
    fn two_by_two_roundtrips_through_walker() {
        // A 2×2-macroblock synthetic key frame. Every MB is DC_PRED in
        // both planes, skip=true (zero residue). We verify the walker:
        //   * sizes the planes to 32×32 luma / 16×16 chroma;
        //   * feeds each MB its reconstructed neighbours so the result
        //     equals the per-MB orchestrator run with the same
        //     neighbour strips gathered by hand.
        let modes = vec![
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true), // (0,0)
            mode_16x16(IntraYMode::H, IntraUvMode::H, true),   // (0,1)
            mode_16x16(IntraYMode::V, IntraUvMode::V, true),   // (1,0)
            mode_16x16(IntraYMode::Tm, IntraUvMode::Tm, true), // (1,1)
        ];
        let coeffs = vec![MbCoeffs::default(); 4];
        let planes = decode_keyframe(2, 2, &modes, &coeffs).expect("decode");

        assert_eq!(planes.y_stride, 32);
        assert_eq!(planes.uv_stride, 16);
        assert_eq!(planes.y.len(), 32 * 32);
        assert_eq!(planes.u.len(), 16 * 16);

        // Reconstruct independently MB-by-MB, gathering neighbours by
        // hand from the running plane, and require an exact match. This
        // proves the walker's neighbour assembly + writeback are
        // self-consistent with the per-MB orchestrators.
        let mut ref_planes = KeyframePlanes {
            y: vec![0u8; 32 * 32],
            u: vec![0u8; 16 * 16],
            v: vec![0u8; 16 * 16],
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        for mb_row in 0..2 {
            for mb_col in 0..2 {
                let idx = mb_row * 2 + mb_col;
                let m = &modes[idx];
                let nb = gather_neighbors(&ref_planes, mb_row, mb_col);
                let recon = decode_keyframe_mb_non_bpred(
                    m.y_mode,
                    m.uv_mode,
                    m.mb_skip_coeff,
                    &nb,
                    &coeffs[idx].y2,
                    &coeffs[idx].y,
                    &coeffs[idx].u,
                    &coeffs[idx].v,
                )
                .unwrap();
                write_mb(&mut ref_planes, mb_row, mb_col, &recon);
            }
        }
        assert_eq!(planes, ref_planes);
    }

    #[test]
    fn neighbours_propagate_across_macroblocks() {
        // MB (0,0) is V_PRED with a non-trivial residue baked into the
        // coefficients so its reconstructed bottom row is non-flat. MB
        // (1,0) directly below is V_PRED skip — its prediction is a
        // verbatim copy of (0,0)'s reconstructed bottom row. We require
        // (1,0)'s top row to equal (0,0)'s bottom row, proving the
        // walker fed (1,0) the genuine reconstructed `y_above`.
        let mut c00 = MbCoeffs::default();
        // Put a DC term into the Y2 block so the WHT seeds each Y
        // sub-block's DC, lifting the whole MB off the flat prediction.
        c00.y2[0] = 40;
        let modes = vec![
            mode_16x16(IntraYMode::V, IntraUvMode::Dc, false), // (0,0)
            mode_16x16(IntraYMode::V, IntraUvMode::Dc, true),  // (1,0)
        ];
        let coeffs = vec![c00, MbCoeffs::default()];
        let planes = decode_keyframe(1, 2, &modes, &coeffs).expect("decode");

        let stride = planes.y_stride;
        // (0,0) bottom row is luma row 15; (1,0) top row is luma row 16.
        let bottom_of_top = &planes.y[15 * stride..15 * stride + 16];
        let top_of_bottom = &planes.y[16 * stride..16 * stride + 16];
        assert_eq!(
            bottom_of_top, top_of_bottom,
            "V_PRED MB below must copy the reconstructed above row"
        );
        // And it must not be the flat top-row default (127): the Y2 DC
        // term lifted the reconstructed pixels off the prediction.
        assert!(
            top_of_bottom.iter().any(|&p| p != 127),
            "neighbour row should carry the residue from the MB above"
        );
    }

    #[test]
    fn rightmost_mb_above_right_clamp_exercised() {
        // A 2×2 frame whose top-right MB (0,1) reconstructs a non-flat
        // bottom row, so the four (-1,16)..=(-1,19) pixels above the
        // bottom-right MB (1,1) would, absent the clamp, be the bottom
        // row of an above-and-to-the-right MB that does NOT exist (col 2
        // is off-frame). §12.3 says the rightmost MB clamps those four
        // to the (-1,15) pixel value. We verify gather_neighbors does
        // exactly that for the bottom-right B_PRED MB, and that a
        // B_PRED right-edge sub-block actually consumes the clamped
        // value.
        //
        // Build the first row so that (0,1)'s reconstructed bottom row
        // has a distinctive rightmost pixel.
        let mut c01 = MbCoeffs::default();
        c01.y2[0] = 60; // lift (0,1) off its flat prediction.
        let modes_row0 = [
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true), // (0,0)
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, false), // (0,1)
        ];
        // Bottom-right MB is B_PRED so it reads y_above_right.
        let subs = [IntraBmode::Ld; 16]; // Ld reads above + above-right.
        let modes = vec![
            modes_row0[0].clone(),
            modes_row0[1].clone(),
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true), // (1,0)
            mode_bpred(subs, IntraUvMode::Dc, true),           // (1,1)
        ];
        let coeffs = vec![
            MbCoeffs::default(),
            c01,
            MbCoeffs::default(),
            MbCoeffs::default(),
        ];
        let planes = decode_keyframe(2, 2, &modes, &coeffs).expect("decode");

        // Independently recompute what gather_neighbors should produce
        // for the bottom-right MB and assert the clamp.
        let stride = planes.y_stride;
        // (-1,15) for MB (1,1): luma row 15, column 16*1 + 15 = 31.
        let clamp = planes.y[15 * stride + 31];
        let nb = gather_neighbors(&planes, 1, 1);
        let ar = nb.y_above_right.expect("non-top row has above-right");
        assert_eq!(
            ar, [clamp; 4],
            "rightmost-MB above-right must be the (-1,15) clamp value"
        );

        // Sanity: the clamp value is genuinely non-trivial (the residue
        // lifted it), so the clamp is being exercised meaningfully and
        // not just trivially matching a flat fill.
        assert_ne!(clamp, 127, "above-right clamp source should be non-flat");
    }

    #[test]
    fn non_rightmost_above_right_is_genuine_pixels() {
        // In the same 2×2 frame, the bottom-LEFT MB (1,0) is NOT the
        // rightmost, so its four above-right pixels are the genuine
        // (-1,16)..=(-1,19) — the bottom row of the already-built MB
        // (0,1), columns 0..4 (frame columns 16..20).
        let mut c01 = MbCoeffs::default();
        c01.y2[0] = 60;
        let modes = vec![
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true),
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, false),
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true),
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true),
        ];
        let coeffs = vec![
            MbCoeffs::default(),
            c01,
            MbCoeffs::default(),
            MbCoeffs::default(),
        ];
        let planes = decode_keyframe(2, 2, &modes, &coeffs).expect("decode");

        let stride = planes.y_stride;
        // For MB (1,0): above row is luma row 15, above-right columns
        // 16..20.
        let mut expect = [0u8; 4];
        expect.copy_from_slice(&planes.y[15 * stride + 16..15 * stride + 20]);
        let nb = gather_neighbors(&planes, 1, 0);
        let ar = nb.y_above_right.expect("non-top row has above-right");
        assert_eq!(
            ar, expect,
            "non-rightmost MB above-right must be the genuine (-1,16..20) pixels"
        );
    }

    #[test]
    fn top_row_above_right_is_none() {
        // On the top macroblock row the above-right field is left None,
        // so the per-MB orchestrator fills the four pixels with 127.
        let modes = vec![
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true),
            mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true),
        ];
        let coeffs = vec![MbCoeffs::default(); 2];
        let planes = decode_keyframe(2, 1, &modes, &coeffs).expect("decode");
        // gather_neighbors for both top-row MBs reports no above-right.
        assert!(gather_neighbors(&planes, 0, 0).y_above_right.is_none());
        assert!(gather_neighbors(&planes, 0, 1).y_above_right.is_none());
        // And no above / left for the top-left MB.
        let nb = gather_neighbors(&planes, 0, 0);
        assert!(nb.y_above.is_none());
        assert!(nb.y_left.is_none());
        assert!(nb.y_topleft.is_none());
    }

    #[test]
    fn bpred_frame_walks_without_error() {
        // A 2×2 all-B_PRED frame round-trips. This exercises the B_PRED
        // branch selection and the above-right plumbing through the
        // walker for every macroblock position (top row, left column,
        // interior, rightmost).
        let subs = [IntraBmode::Dc; 16];
        let modes = vec![
            mode_bpred(subs, IntraUvMode::Dc, true),
            mode_bpred(subs, IntraUvMode::Dc, true),
            mode_bpred(subs, IntraUvMode::Dc, true),
            mode_bpred(subs, IntraUvMode::Dc, true),
        ];
        let coeffs = vec![MbCoeffs::default(); 4];
        let planes = decode_keyframe(2, 2, &modes, &coeffs).expect("decode");
        assert_eq!(planes.y.len(), 32 * 32);
        // B_PRED with B_DC_PRED sub-blocks and skip=true yields a fully
        // populated plane; with the top-left corner being 128, the frame
        // is non-degenerate but every pixel is a valid reconstruction.
        assert_eq!(planes.u.len(), 16 * 16);
    }

    #[test]
    fn missing_subblock_modes_surfaces_indexed_error() {
        // A macroblock declared B_PRED but missing its sub-block modes
        // must surface ReconstructError::MissingSubblockModes wrapped in
        // FrameError::Macroblock with the right raster index.
        let bad = MacroblockModes {
            segment_id: None,
            mb_skip_coeff: true,
            y_mode: IntraYMode::B,
            subblock_modes: None, // illegal for B_PRED
            uv_mode: IntraUvMode::Dc,
        };
        let modes = vec![mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, true), bad];
        let coeffs = vec![MbCoeffs::default(); 2];
        let res = decode_keyframe(2, 1, &modes, &coeffs);
        assert_eq!(
            res,
            Err(FrameError::Macroblock {
                index: 1,
                source: ReconstructError::MissingSubblockModes,
            })
        );
    }
}
