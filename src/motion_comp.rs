//! Interframe motion compensation — RFC 6386 §16.2 reference selection,
//! §18.1 motion-vector adjustment, and §18.2 whole-pixel prediction.
//!
//! This module is the first inter-frame *prediction* slice: it consumes
//! the §17 motion vectors decoded by [`crate::motion_vector`] and turns a
//! reference frame plus a per-macroblock motion vector into a predicted
//! macroblock, then folds in the §14 dequantized residual to complete
//! inter-MB reconstruction. It sits one layer below the §16.3
//! near/nearest/best census (which produces the actual per-MB vector a
//! NEWMV offset is added to) and the §16.4 SPLITMV walk; those choose
//! *which* vector applies, this applies it.
//!
//! ## Scope this round — whole-pixel motion compensation
//!
//! VP8 motion vectors carry a fractional (sub-pixel) part: §18.3 sixtap
//! / bilinear interpolation synthesises the missing samples. That
//! interpolation is large and is deferred to the next round. What lands
//! here is the *whole-pixel* path — the case where, after the §18.1
//! adjustments, both fractional components are zero, so §18.3 page 115
//! says "the prediction subblock is simply copied". This is the
//! `filter_block` special case `mx | my == 0` in the §20.14 reference
//! decoder, where `filter_block` returns the reference pointer unchanged
//! and `recon_1_block` adds the residue straight onto the copied pixels.
//!
//! Concretely this module provides:
//!
//! * [`RefFrame`] + [`select_ref_frame`] — the §16.2 `prob_last` /
//!   `prob_gf` reference-frame selector.
//! * [`ReferencePlanes`] — a borrow of one reference frame's I420 planes
//!   for prediction fetch.
//! * [`stored_luma_mv`] / [`chroma_mv`] / [`apply_full_pixel`] — the
//!   §18.1 motion-vector adjustments (stored-luma doubling, the chroma
//!   `avg()` averaging, and the version-3 full-pel truncation).
//! * [`whole_pixel_fraction_is_zero`] — the §18.3 whole-pixel test
//!   applied to an already-§18.1-adjusted (eighth-pixel) vector.
//! * [`fetch_block_whole_pixel`] — the §20.14 `build_mc_border`
//!   edge-replicated 4×4 block fetch, specialised to whole-pixel offsets
//!   (no six-tap support pixels needed).
//! * [`predict_inter_mb_whole_pixel`] — the §18.2 whole-MB prediction
//!   buffer for a non-SPLITMV macroblock (one vector for all sixteen Y
//!   sub-blocks, the averaged chroma vector for the eight chroma
//!   sub-blocks).
//! * [`reconstruct_inter_mb_whole_pixel`] — prediction + §14 dequantized
//!   residual, producing a [`ReconstructedMb`].
//!
//! ## What is deferred (next inter-prediction rounds)
//!
//! * §18.3 sub-pixel interpolation (sixtap + bilinear). When a vector's
//!   §18.1-adjusted fraction is non-zero, [`predict_inter_mb_whole_pixel`]
//!   and [`reconstruct_inter_mb_whole_pixel`] return
//!   [`MotionCompError::SubPixelNotSupported`] rather than guess.
//! * §16.4 SPLITMV (per-sub-block vectors): this round handles the four
//!   whole-MB inter modes (`mv_nearest` / `mv_near` / `mv_zero` /
//!   `mv_new`), all of which apply one vector to the whole MB.
//! * §16.3 `vp8_find_near_mvs` census: this module takes the *resolved*
//!   per-MB vector as an input; deriving it from neighbours is a separate
//!   slice.

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use crate::inverse_transform::{add_residue_4x4, inverse_dct_4x4, inverse_wht_4x4};
use crate::motion_vector::Mv;
use crate::reconstruct::ReconstructedMb;

/// The reference frame a macroblock predicts from — RFC 6386 §16.2.
///
/// On an interframe, after the intra/inter discriminator selects
/// inter-prediction, a `Bool(prob_last)` then (conditionally) a
/// `Bool(prob_gf)` choose between the three stored reference frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefFrame {
    /// The previous (last) decoded frame — `prob_last` reads 0.
    Last,
    /// The golden frame — `prob_last` reads 1, `prob_gf` reads 0.
    Golden,
    /// The altref frame — `prob_last` reads 1, `prob_gf` reads 1.
    AltRef,
}

/// Select the reference frame for an inter-predicted macroblock — RFC
/// 6386 §16.2.
///
/// "The next datum is then another bool, `B(prob_last)`, selecting the
/// reference frame. If 0, the reference frame is the previous frame (the
/// last frame); if 1, another bool (`prob_gf`) selects the reference
/// frame between the golden frame (0) and the altref frame (1)."
///
/// `prob_last` and `prob_gf` come from field J of the frame header (the
/// §9.10 `prob_last` / `prob_gf` already parsed by [`crate::coded_header`]).
pub fn select_ref_frame(
    dec: &mut BoolDecoder<'_>,
    prob_last: u8,
    prob_gf: u8,
) -> Result<RefFrame, BoolDecoderError> {
    if !dec.read_bool(prob_last)? {
        Ok(RefFrame::Last)
    } else if !dec.read_bool(prob_gf)? {
        Ok(RefFrame::Golden)
    } else {
        Ok(RefFrame::AltRef)
    }
}

/// A borrow of one reference frame's reconstructed I420 planes, in the
/// whole-macroblock layout produced by [`crate::frame::KeyframePlanes`].
///
/// The planes are sized to whole macroblocks: the luma plane is
/// `mb_cols * 16` wide by `mb_rows * 16` tall, each chroma plane half
/// that in both dimensions. The §18.2 prediction fetch reads from these
/// planes with the §20.14 `build_mc_border` edge replication when a
/// motion vector points outside the plane.
#[derive(Debug, Clone, Copy)]
pub struct ReferencePlanes<'a> {
    /// Luma plane, row-major, `y_stride * (mb_rows * 16)` bytes.
    pub y: &'a [u8],
    /// U chroma plane, row-major, `uv_stride * (mb_rows * 8)` bytes.
    pub u: &'a [u8],
    /// V chroma plane, same dimensions as [`ReferencePlanes::u`].
    pub v: &'a [u8],
    /// Luma plane stride (= `mb_cols * 16`).
    pub y_stride: usize,
    /// Chroma plane stride (= `mb_cols * 8`).
    pub uv_stride: usize,
    /// Number of macroblock columns.
    pub mb_cols: usize,
    /// Number of macroblock rows.
    pub mb_rows: usize,
}

/// Errors surfaced by the motion-compensation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionCompError {
    /// The §18.1-adjusted motion vector has a non-zero fractional part,
    /// so §18.3 sub-pixel interpolation is required — which this round
    /// does not implement. The whole-pixel path covers only vectors
    /// whose eighth-pixel fraction is `(0, 0)`.
    SubPixelNotSupported,
}

impl core::fmt::Display for MotionCompError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MotionCompError::SubPixelNotSupported => f.write_str(
                "vp8 motion-comp: sub-pixel interpolation (§18.3) not yet implemented; \
                 only whole-pixel motion vectors are supported this round",
            ),
        }
    }
}

impl std::error::Error for MotionCompError {}

/// Apply the §18.1 stored-luma doubling to a decoded luma motion vector.
///
/// RFC 6386 §18.1: "the synthetic pixel calculation ... uses [1/8-pixel]
/// resolution for the luma subblocks as well. In accordance, the stored
/// luma motion vectors are all doubled, each component of each luma
/// vector becoming an even integer in the range -2046 to +2046,
/// inclusive."
///
/// The §17 decode produces quarter-pixel luma components; doubling moves
/// them to the eighth-pixel resolution chroma uses, so a single
/// fractional-extraction rule (`& 7`) serves both planes.
#[inline]
pub fn stored_luma_mv(mv: Mv) -> Mv {
    Mv {
        row: mv.row.wrapping_mul(2),
        col: mv.col.wrapping_mul(2),
    }
}

/// The §18.1 `avg()` four-vector averaging primitive (one component).
///
/// RFC 6386 §18.1:
///
/// ```text
/// int avg(int c1, int c2, int c3, int c4) {
///     int s = c1 + c2 + c3 + c4;
///     return s >= 0 ? (s + 4) >> 3 : -((-s + 4) >> 3);
/// }
/// ```
///
/// The shift divides by 8 (not 4) because chroma pixels have twice the
/// diameter of luma pixels; the negative-number handling is explicit
/// because C right shifts of negatives are not well-defined.
#[inline]
fn avg(c1: i32, c2: i32, c3: i32, c4: i32) -> i32 {
    let s = c1 + c2 + c3 + c4;
    if s >= 0 {
        (s + 4) >> 3
    } else {
        -((-s + 4) >> 3)
    }
}

/// Compute the chroma motion vector for a whole (non-SPLITMV)
/// macroblock — RFC 6386 §18.1.
///
/// For a whole-MB vector all sixteen luma sub-blocks share `luma_mv`
/// (already §18.1-doubled to eighth-pixel resolution), so each chroma
/// vector is `avg(v, v, v, v)` of that one vector. The §20.14 reference
/// decoder writes this as the closed form
/// `(c + 1 + (c >> 31) * 2) / 2` per component, which equals
/// `avg(c, c, c, c)`; we use [`avg`] directly so the single §18.1
/// primitive backs both the whole-MB and (future) SPLITMV chroma paths.
///
/// `luma_mv` must already be the §18.1 stored (doubled) luma vector —
/// see [`stored_luma_mv`].
#[inline]
pub fn chroma_mv(luma_mv: Mv) -> Mv {
    let row = luma_mv.row as i32;
    let col = luma_mv.col as i32;
    Mv {
        row: avg(row, row, row, row) as i16,
        col: avg(col, col, col, col) as i16,
    }
}

/// Truncate the fractional part of a vector for the version-3 full-pel
/// chroma profile — RFC 6386 §18.1.
///
/// "if the version number in the frame tag specifies only full-pel
/// chroma motion vectors, then the fractional parts of both components
/// of the vector are truncated to zero":
///
/// ```text
/// x = x & (~7);
/// y = y & (~7);
/// ```
///
/// Applied to an eighth-pixel-resolution vector (the §18.1 stored luma
/// vector or the [`chroma_mv`] average).
#[inline]
pub fn apply_full_pixel(mv: Mv) -> Mv {
    Mv {
        row: mv.row & !7,
        col: mv.col & !7,
    }
}

/// Test whether an already-§18.1-adjusted (eighth-pixel) vector has a
/// zero fractional part — i.e. it is a whole-pixel vector and §18.3 page
/// 115 "the prediction subblock is simply copied" applies.
///
/// This mirrors the §20.14 `filter_block` test `mx = mv.x & 7;
/// my = mv.y & 7; if (mx | my) interpolate; else copy`.
#[inline]
pub fn whole_pixel_fraction_is_zero(mv: Mv) -> bool {
    (mv.row & 7) == 0 && (mv.col & 7) == 0
}

/// Fetch a 4×4 whole-pixel prediction block from a reference plane with
/// the §20.14 `build_mc_border` edge-replication rule.
///
/// `mv` is the §18.1-adjusted (eighth-pixel) vector for this sub-block;
/// its fractional part must be zero (whole-pixel — see
/// [`whole_pixel_fraction_is_zero`]). The integer pixel offset is
/// `mv >> 3` per component (sign-propagating right shift, §18.2 "rounding
/// up or left by right-shifting each component 3 bits with sign
/// propagation").
///
/// `(blk_x, blk_y)` is the sub-block's top-left position in the plane in
/// pixels (before the motion offset). `plane` / `stride` describe the
/// reference plane; `w` / `h` are its width / height in pixels. Pixels
/// the source position would read outside `[0, w) × [0, h)` are clamped
/// to the nearest edge pixel — the whole-pixel specialisation of
/// `build_mc_border` (which, for an integer fetch, just replicates the
/// border row / column).
pub fn fetch_block_whole_pixel(
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    blk_x: usize,
    blk_y: usize,
    mv: Mv,
) -> [u8; 16] {
    // §18.2: integer pixel offset is the eighth-pixel vector shifted
    // right 3 with sign propagation.
    let off_x = (mv.col >> 3) as isize;
    let off_y = (mv.row >> 3) as isize;
    let src_x0 = blk_x as isize + off_x;
    let src_y0 = blk_y as isize + off_y;

    let w = w as isize;
    let h = h as isize;
    let mut out = [0u8; 16];
    for r in 0..4 {
        // §20.14 build_mc_border: rows past the top / bottom edge read
        // the first / last in-bounds row (edge replication).
        let sy = (src_y0 + r as isize).clamp(0, h - 1);
        for c in 0..4 {
            // Columns past the left / right edge read the first / last
            // in-bounds column.
            let sx = (src_x0 + c as isize).clamp(0, w - 1);
            out[r * 4 + c] = plane[sy as usize * stride + sx as usize];
        }
    }
    out
}

/// Build the §18.2 whole-pixel prediction buffer for a non-SPLITMV
/// inter-predicted macroblock.
///
/// The same `luma_mv` (raw §17-decoded, quarter-pixel) applies to all
/// sixteen Y sub-blocks; the chroma vector is the §18.1 average of that
/// one vector. `full_pixel` is the version-3 full-pel-chroma flag
/// (`frame_hdr.version == 3`).
///
/// Returns the predicted macroblock (no residue) as a
/// [`ReconstructedMb`], or [`MotionCompError::SubPixelNotSupported`] if
/// the §18.1-adjusted luma or chroma vector has a non-zero fractional
/// part (sub-pixel interpolation is a future round).
///
/// `(mb_col, mb_row)` is the macroblock position; the reference planes
/// supply the dimensions.
pub fn predict_inter_mb_whole_pixel(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    luma_mv: Mv,
    full_pixel: bool,
) -> Result<ReconstructedMb, MotionCompError> {
    // §18.1: double the quarter-pixel luma vector to eighth-pixel
    // resolution.
    let mut ymv = stored_luma_mv(luma_mv);
    // §18.1 chroma averaging (avg() of the single repeated vector).
    let mut uvmv = chroma_mv(ymv);
    if full_pixel {
        // §18.1 version-3 full-pel-chroma truncation.
        ymv = apply_full_pixel(ymv);
        uvmv = apply_full_pixel(uvmv);
    }

    // §18.3 whole-pixel gate: both fractions must be zero.
    if !whole_pixel_fraction_is_zero(ymv) || !whole_pixel_fraction_is_zero(uvmv) {
        return Err(MotionCompError::SubPixelNotSupported);
    }

    let lw = reference.mb_cols * 16;
    let lh = reference.mb_rows * 16;
    let cw = reference.mb_cols * 8;
    let ch = reference.mb_rows * 8;

    let mut out = ReconstructedMb::default();

    // Luma: sixteen 4×4 sub-blocks, each fetched with the same ymv.
    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    for sb in 0..4 {
        for sc in 0..4 {
            let blk = fetch_block_whole_pixel(
                reference.y,
                reference.y_stride,
                lw,
                lh,
                y_x0 + sc * 4,
                y_y0 + sb * 4,
                ymv,
            );
            // Write into the 16×16 row-major output.
            for r in 0..4 {
                let dst = (sb * 4 + r) * 16 + sc * 4;
                out.y[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
            }
        }
    }

    // Chroma: four 4×4 sub-blocks per plane, each fetched with uvmv.
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;
    for sb in 0..2 {
        for sc in 0..2 {
            let ublk = fetch_block_whole_pixel(
                reference.u,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
            );
            let vblk = fetch_block_whole_pixel(
                reference.v,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 8 + sc * 4;
                out.u[dst..dst + 4].copy_from_slice(&ublk[r * 4..r * 4 + 4]);
                out.v[dst..dst + 4].copy_from_slice(&vblk[r * 4..r * 4 + 4]);
            }
        }
    }

    Ok(out)
}

/// Extract a 4×4 sub-block at `(sub_row, sub_col)` (4-pixel units) from a
/// row-major plane of `stride` pixels.
#[inline]
fn extract_4x4(plane: &[u8], stride: usize, sub_row: usize, sub_col: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    let y0 = sub_row * 4;
    let x0 = sub_col * 4;
    for r in 0..4 {
        let src = (y0 + r) * stride + x0;
        out[r * 4..r * 4 + 4].copy_from_slice(&plane[src..src + 4]);
    }
    out
}

/// Splat a 4×4 sub-block back into a row-major plane.
#[inline]
fn insert_4x4(plane: &mut [u8], stride: usize, sub_row: usize, sub_col: usize, src: &[u8; 16]) {
    let y0 = sub_row * 4;
    let x0 = sub_col * 4;
    for r in 0..4 {
        let dst = (y0 + r) * stride + x0;
        plane[dst..dst + 4].copy_from_slice(&src[r * 4..r * 4 + 4]);
    }
}

/// Reconstruct one whole-pixel inter-predicted (non-SPLITMV)
/// macroblock — RFC 6386 §16.2 / §18 prediction + §14 residue.
///
/// This is the inter analogue of [`crate::reconstruct::decode_keyframe_mb_non_bpred`]:
///
/// 1. Build the §18.2 whole-pixel prediction buffer
///    ([`predict_inter_mb_whole_pixel`]).
/// 2. If `mb_skip_coeff`, the residue is zero and the prediction is the
///    reconstruction (§11.1 skip short-circuit).
/// 3. Otherwise inverse-WHT the Y2 block (an inter non-SPLITMV MB has a
///    Y2 block, §14.2), seed each Y sub-block DC, inverse-DCT all 24
///    sub-blocks, and add the residue with `clamp255` (§14.5).
///
/// The coefficient arrays are pre-dequantized (the caller's
/// responsibility, matching the keyframe path).
///
/// Returns [`MotionCompError::SubPixelNotSupported`] when the vector is
/// not whole-pixel.
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §14.2/§18 input.
pub fn reconstruct_inter_mb_whole_pixel(
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    luma_mv: Mv,
    full_pixel: bool,
    mb_skip_coeff: bool,
    y2_coeffs_dequant: &[i16; 16],
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> Result<ReconstructedMb, MotionCompError> {
    let mut out = predict_inter_mb_whole_pixel(reference, mb_col, mb_row, luma_mv, full_pixel)?;

    if mb_skip_coeff {
        return Ok(out);
    }

    // §14.2: inverse-WHT Y2 and seed each Y sub-block's DC.
    let mut y2_residue = [0i16; 16];
    inverse_wht_4x4(y2_coeffs_dequant, &mut y2_residue);
    let mut y_coeffs = *y_coeffs_dequant;
    for i in 0..4 {
        for j in 0..4 {
            y_coeffs[i * 4 + j][0] = y2_residue[i * 4 + j];
        }
    }

    // Luma: inverse-DCT + add residue per sub-block.
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

    // Chroma: inverse-DCT + add residue per sub-block, U then V.
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

    /// Minimal test-side VP8 boolean encoder, mirroring the proven
    /// encoder in `motion_vector::tests` / `bool_decoder::tests`. Used to
    /// build bitstreams the §16.2 reference selector decodes back.
    struct BoolEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl BoolEncoder {
        fn new() -> Self {
            BoolEncoder {
                out: Vec::new(),
                range: 255,
                bottom: 0,
                bit_count: 24,
            }
        }

        fn write_bool(&mut self, prob: u8, val: bool) {
            let split = 1 + (((self.range - 1) * prob as u32) >> 8);
            if val {
                self.bottom = self.bottom.wrapping_add(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.range <<= 1;
                if (self.bottom >> 31) & 1 == 1 {
                    let mut i = self.out.len();
                    while i > 0 {
                        i -= 1;
                        if self.out[i] == 255 {
                            self.out[i] = 0;
                        } else {
                            self.out[i] = self.out[i].wrapping_add(1);
                            break;
                        }
                    }
                }
                self.bottom <<= 1;
                self.bit_count -= 1;
                if self.bit_count == 0 {
                    let byte = (self.bottom >> 24) as u8;
                    self.out.push(byte);
                    self.bottom &= (1 << 24) - 1;
                    self.bit_count = 8;
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            let c = self.bit_count;
            let v = self.bottom;
            if v & (1u32 << (32 - c)) != 0 {
                let mut i = self.out.len();
                while i > 0 {
                    i -= 1;
                    if self.out[i] == 255 {
                        self.out[i] = 0;
                    } else {
                        self.out[i] = self.out[i].wrapping_add(1);
                        break;
                    }
                }
            }
            let mut v = v;
            v <<= c & 7;
            let mut c_shift = c >> 3;
            while c_shift > 0 {
                v <<= 8;
                c_shift -= 1;
            }
            for _ in 0..4 {
                self.out.push((v >> 24) as u8);
                v <<= 8;
            }
            while self.out.len() < 2 {
                self.out.push(0);
            }
            self.out
        }
    }

    // ----- §16.2 reference frame selection ---------------------------

    #[test]
    fn ref_frame_last_reads_single_zero() {
        // prob_last bit 0 → Last; prob_gf is never read.
        let mut enc = BoolEncoder::new();
        enc.write_bool(128, false); // prob_last = 0
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 128, 128).unwrap(),
            RefFrame::Last
        );
    }

    #[test]
    fn ref_frame_golden_reads_one_then_zero() {
        let mut enc = BoolEncoder::new();
        enc.write_bool(128, true); // prob_last = 1
        enc.write_bool(128, false); // prob_gf = 0 → Golden
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 128, 128).unwrap(),
            RefFrame::Golden
        );
    }

    #[test]
    fn ref_frame_altref_reads_one_then_one() {
        let mut enc = BoolEncoder::new();
        enc.write_bool(128, true); // prob_last = 1
        enc.write_bool(128, true); // prob_gf = 1 → AltRef
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 128, 128).unwrap(),
            RefFrame::AltRef
        );
    }

    #[test]
    fn ref_frame_uses_distinct_probs() {
        // Encode Golden against asymmetric probs; decode with the same
        // probs must recover Golden. This proves prob_last and prob_gf
        // are wired to the right read order.
        let mut enc = BoolEncoder::new();
        enc.write_bool(30, true); // prob_last = 30
        enc.write_bool(200, false); // prob_gf = 200
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(
            select_ref_frame(&mut dec, 30, 200).unwrap(),
            RefFrame::Golden
        );
    }

    // ----- §18.1 motion-vector adjustments ---------------------------

    #[test]
    fn stored_luma_doubles_each_component() {
        // §18.1: quarter-pixel luma vector doubled to eighth-pixel.
        assert_eq!(
            stored_luma_mv(Mv { row: 3, col: -5 }),
            Mv { row: 6, col: -10 }
        );
        // Extremes: ±1023 quarter-pel → ±2046 eighth-pel (the §18.1
        // stated range).
        assert_eq!(
            stored_luma_mv(Mv {
                row: 1023,
                col: -1023
            }),
            Mv {
                row: 2046,
                col: -2046
            }
        );
    }

    #[test]
    fn avg_matches_spec_formula() {
        // Positive rounding: (s + 4) >> 3.
        assert_eq!(avg(8, 8, 8, 8), (32 + 4) >> 3); // = 4
        assert_eq!(avg(1, 1, 1, 1), (4 + 4) >> 3); // = 1
        assert_eq!(avg(0, 0, 0, 0), 0);
        // Negative rounding: -((-s + 4) >> 3) — symmetric to positive.
        assert_eq!(avg(-8, -8, -8, -8), -((32 + 4) >> 3)); // = -4
        assert_eq!(avg(-1, -1, -1, -1), -((4 + 4) >> 3)); // = -1
                                                          // Mixed sign, sum exactly zero.
        assert_eq!(avg(5, -5, 3, -3), 0);
    }

    #[test]
    fn chroma_mv_of_repeated_vector_is_quarter() {
        // For a whole-MB vector, chroma_mv = avg(v,v,v,v). With the
        // eighth-pixel doubled luma vector v=6, avg(6,6,6,6) = (24+4)>>3
        // = 3.
        let ymv = stored_luma_mv(Mv { row: 3, col: -5 }); // (6, -10)
        let uvmv = chroma_mv(ymv);
        assert_eq!(uvmv.row, ((6 * 4 + 4) >> 3) as i16); // 3
        assert_eq!(uvmv.col, -(((10 * 4 + 4) >> 3) as i16)); // -5
    }

    #[test]
    fn chroma_mv_matches_reference_closed_form() {
        // §20.14 writes the whole-MB chroma vector as
        // (c + 1 + (c >> 31) * 2) / 2 per component. Cross-check our
        // avg()-based chroma_mv against that closed form across a spread
        // of eighth-pixel luma components.
        fn ref_form(c: i32) -> i32 {
            (c + 1 + (c >> 31) * 2) / 2
        }
        for &c in &[0i32, 2, 6, 10, 14, 100, -2, -6, -10, -14, -100, 2046, -2046] {
            let mv = Mv {
                row: c as i16,
                col: c as i16,
            };
            let got = chroma_mv(mv);
            assert_eq!(got.row as i32, ref_form(c), "row c={c}");
            assert_eq!(got.col as i32, ref_form(c), "col c={c}");
        }
    }

    #[test]
    fn full_pixel_truncation_clears_low_three_bits() {
        // §18.1 version-3: x &= ~7, y &= ~7.
        assert_eq!(
            apply_full_pixel(Mv { row: 13, col: 9 }),
            Mv { row: 8, col: 8 }
        );
        assert_eq!(
            apply_full_pixel(Mv { row: 16, col: 24 }),
            Mv { row: 16, col: 24 }
        );
        // Negative two's-complement & ~7 rounds toward negative infinity.
        assert_eq!(apply_full_pixel(Mv { row: -1, col: -9 }).row, -8);
        assert_eq!(apply_full_pixel(Mv { row: -1, col: -9 }).col, -16);
    }

    #[test]
    fn whole_pixel_test_detects_fraction() {
        assert!(whole_pixel_fraction_is_zero(Mv { row: 16, col: -8 }));
        assert!(whole_pixel_fraction_is_zero(Mv { row: 0, col: 0 }));
        assert!(!whole_pixel_fraction_is_zero(Mv { row: 1, col: 0 }));
        assert!(!whole_pixel_fraction_is_zero(Mv { row: 0, col: 3 }));
        assert!(!whole_pixel_fraction_is_zero(Mv { row: -1, col: 0 }));
    }

    // ----- §20.14 build_mc_border whole-pixel fetch ------------------

    /// A 8×8 single-plane ramp where pixel (r, c) = r * 8 + c.
    fn ramp_plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn fetch_zero_mv_copies_in_place() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Sub-block at (0,0), zero vector → the top-left 4×4.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: 0, col: 0 });
        let mut expect = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                expect[r * 4 + c] = plane[r * w + c];
            }
        }
        assert_eq!(blk, expect);
    }

    #[test]
    fn fetch_whole_pixel_offset_shifts_origin() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Eighth-pixel vector (8, 16) → integer offset (1, 2): origin
        // moves down 1 row, right 2 cols.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: 8, col: 16 });
        let mut expect = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                expect[r * 4 + c] = plane[(1 + r) * w + (2 + c)];
            }
        }
        assert_eq!(blk, expect);
    }

    #[test]
    fn fetch_left_edge_replicates_first_column() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Negative col offset large enough to push all 4 columns past
        // the left edge: every fetched pixel is column 0 of its row.
        // mv.col = -64 → off_x = -8; sub-block at x=0 → src_x0 = -8,
        // all columns clamp to 0.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: 0, col: -64 });
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(blk[r * 4 + c], plane[r * w], "row {r} col {c}");
            }
        }
    }

    #[test]
    fn fetch_top_edge_replicates_first_row() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // mv.row = -64 → off_y = -8; all rows clamp to row 0.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 0, 0, Mv { row: -64, col: 0 });
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(blk[r * 4 + c], plane[c], "row {r} col {c}");
            }
        }
    }

    #[test]
    fn fetch_bottom_right_corner_replicates() {
        let w = 8;
        let h = 8;
        let plane = ramp_plane(w, h);
        // Push the fetch fully past the bottom-right corner: every pixel
        // becomes the (h-1, w-1) corner pixel.
        let blk = fetch_block_whole_pixel(&plane, w, w, h, 4, 4, Mv { row: 64, col: 64 });
        let corner = plane[(h - 1) * w + (w - 1)];
        for px in blk {
            assert_eq!(px, corner);
        }
    }

    // ----- §18.2 whole-MB prediction ---------------------------------

    /// Build a reference frame whose planes are filled with a deterministic
    /// per-position value, big enough for a 2×2 macroblock grid.
    fn build_reference(mb_cols: usize, mb_rows: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let lw = mb_cols * 16;
        let lh = mb_rows * 16;
        let cw = mb_cols * 8;
        let ch = mb_rows * 8;
        let y = (0..lw * lh).map(|i| (i % 251) as u8).collect();
        let u = (0..cw * ch).map(|i| ((i * 3 + 1) % 251) as u8).collect();
        let v = (0..cw * ch).map(|i| ((i * 7 + 2) % 251) as u8).collect();
        (y, u, v)
    }

    #[test]
    fn predict_zero_mv_copies_corresponding_mb() {
        // A zero motion vector predicts the current MB from the matching
        // MB in the reference frame (mv_zero, §16.3).
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        // Predict MB (1, 1) with a zero vector.
        let pred =
            predict_inter_mb_whole_pixel(&reference, 1, 1, Mv { row: 0, col: 0 }, false).unwrap();

        // The luma block must equal reference MB (1,1) verbatim.
        for r in 0..16 {
            for c in 0..16 {
                let ref_px = y[(16 + r) * 32 + (16 + c)];
                assert_eq!(pred.y[r * 16 + c], ref_px, "luma ({r},{c})");
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                let ru = u[(8 + r) * 16 + (8 + c)];
                let rv = v[(8 + r) * 16 + (8 + c)];
                assert_eq!(pred.u[r * 8 + c], ru, "u ({r},{c})");
                assert_eq!(pred.v[r * 8 + c], rv, "v ({r},{c})");
            }
        }
    }

    #[test]
    fn predict_whole_pixel_offset_shifts_source() {
        // Pick a luma vector whose §18.1-doubled value AND its chroma
        // average are both whole-pixel, so the prediction is a pure copy
        // with a non-trivial integer offset (no §18.3 interpolation).
        //
        // luma quarter-pel (8, 16) → doubled (16, 32): both eighth-pel
        // fractions zero → luma integer offset (16>>3, 32>>3) = (2, 4).
        // chroma row avg(16,16,16,16) = (64+4)>>3 = 8 → fraction zero →
        // chroma row offset 8>>3 = 1; chroma col avg(32..) = (128+4)>>3 =
        // 16 → fraction zero → chroma col offset 16>>3 = 2. So both
        // planes copy in place at shifted origins. Use a 3×3 grid so the
        // shifted reads stay in-bounds and predict the centre MB (1,1).
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let mv = Mv { row: 8, col: 16 };
        let pred = predict_inter_mb_whole_pixel(&reference, 1, 1, mv, false).unwrap();

        // Luma: MB (1,1) origin (16, 16) + offset (2, 4).
        for r in 0..16 {
            for c in 0..16 {
                let ref_px = y[(16 + 2 + r) * 48 + (16 + 4 + c)];
                assert_eq!(pred.y[r * 16 + c], ref_px, "luma ({r},{c})");
            }
        }
        // Chroma: MB (1,1) origin (8, 8) + offset (1, 2).
        for r in 0..8 {
            for c in 0..8 {
                let ru = u[(8 + 1 + r) * 24 + (8 + 2 + c)];
                let rv = v[(8 + 1 + r) * 24 + (8 + 2 + c)];
                assert_eq!(pred.u[r * 8 + c], ru, "u ({r},{c})");
                assert_eq!(pred.v[r * 8 + c], rv, "v ({r},{c})");
            }
        }
    }

    #[test]
    fn predict_sub_pixel_vector_rejected() {
        // A vector whose doubled luma fraction is non-zero must be
        // refused (sub-pel interpolation is a future round). row=1
        // quarter-pel → 2 eighth-pel → fraction 2, non-zero.
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        assert_eq!(
            predict_inter_mb_whole_pixel(&reference, 0, 0, Mv { row: 1, col: 0 }, false),
            Err(MotionCompError::SubPixelNotSupported)
        );
    }

    #[test]
    fn predict_chroma_sub_pixel_vector_rejected() {
        // A luma vector that is whole-pixel but whose chroma average is
        // sub-pixel must also be refused. luma quarter-pel (2, 0) →
        // doubled (4, 0): luma fraction 4 — non-zero, so it's already
        // refused at luma. Use (4, 0) quarter → doubled (8, 0): luma
        // whole-pixel; chroma avg(8..)=4 → fraction 4, sub-pixel.
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        assert_eq!(
            predict_inter_mb_whole_pixel(&reference, 0, 0, Mv { row: 4, col: 0 }, false),
            Err(MotionCompError::SubPixelNotSupported)
        );
    }

    #[test]
    fn predict_full_pixel_version_truncates_to_whole() {
        // With full_pixel set, a vector that would be chroma-sub-pixel is
        // truncated to whole-pixel and accepted. luma quarter (4, 0) →
        // doubled (8, 0); chroma avg(8..) = 4 (sub-pixel); full_pixel
        // truncates both to whole-pixel → (8&~7, 0) = (8, 0) luma,
        // (4&~7, 0) = (0, 0) chroma. Accepted.
        //
        // Use a 2×2 grid and predict MB (0,0) so the luma row-1 offset
        // stays in-bounds; cross-check luma against the same
        // edge-clamped fetch the implementation uses, so the test is
        // robust to any boundary behaviour.
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        let pred =
            predict_inter_mb_whole_pixel(&reference, 0, 0, Mv { row: 4, col: 0 }, true).unwrap();
        // Luma offset (8>>3, 0) = (1, 0): row shifted down by 1. Compare
        // against fetch_block_whole_pixel for each 4×4 sub-block.
        for sb in 0..4 {
            for sc in 0..4 {
                let blk =
                    fetch_block_whole_pixel(&y, 32, 32, 32, sc * 4, sb * 4, Mv { row: 8, col: 0 });
                for r in 0..4 {
                    for c in 0..4 {
                        let pr = sb * 4 + r;
                        let pc = sc * 4 + c;
                        assert_eq!(pred.y[pr * 16 + pc], blk[r * 4 + c], "luma ({pr},{pc})");
                    }
                }
            }
        }
        // Chroma offset (0, 0): copied in place from MB (0,0).
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(pred.u[r * 8 + c], u[r * 16 + c], "u ({r},{c})");
            }
        }
    }

    // ----- §16.2 / §18 / §14 inter reconstruction --------------------

    #[test]
    fn reconstruct_skip_equals_prediction() {
        // mb_skip_coeff → zero residue → reconstruction == prediction.
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        let mv = Mv { row: 0, col: 0 };
        let pred = predict_inter_mb_whole_pixel(&reference, 1, 0, mv, false).unwrap();
        let recon = reconstruct_inter_mb_whole_pixel(
            &reference,
            1,
            0,
            mv,
            false,
            true, // skip
            &[0i16; 16],
            &[[0i16; 16]; 16],
            &[[0i16; 16]; 4],
            &[[0i16; 16]; 4],
        )
        .unwrap();
        assert_eq!(recon, pred);
    }

    #[test]
    fn reconstruct_adds_dc_residue() {
        // A pure Y2 DC term lifts every Y sub-block's DC uniformly; the
        // reconstruction must equal prediction + a constant on luma.
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        let mv = Mv { row: 0, col: 0 };
        let pred = predict_inter_mb_whole_pixel(&reference, 0, 0, mv, false).unwrap();

        // A Y2 block with a single DC term. The inverse WHT of a
        // DC-only block distributes (dc + 3) >> 3 to every output; that
        // becomes each Y sub-block's DC, and the inverse DCT of a
        // DC-only block distributes (dc + 4) >> 3 to every pixel.
        let mut y2 = [0i16; 16];
        y2[0] = 64; // WHT: each sub-block DC = (64 + 3) >> 3 ... see below
        let recon = reconstruct_inter_mb_whole_pixel(
            &reference,
            0,
            0,
            mv,
            false,
            false,
            &y2,
            &[[0i16; 16]; 16],
            &[[0i16; 16]; 4],
            &[[0i16; 16]; 4],
        )
        .unwrap();

        // Compute the expected luma delta independently via the same
        // public transform primitives.
        let mut y2_residue = [0i16; 16];
        inverse_wht_4x4(&y2, &mut y2_residue);
        // Every Y sub-block gets DC = y2_residue[idx]; with a single DC
        // input the WHT spreads it evenly, so all sixteen are equal.
        let sub_dc = y2_residue[0];
        let mut residue = [0i16; 16];
        let mut coeffs = [0i16; 16];
        coeffs[0] = sub_dc;
        inverse_dct_4x4(&coeffs, &mut residue);
        let delta = residue[0];

        // Chroma has no residue → equals prediction.
        assert_eq!(recon.u, pred.u);
        assert_eq!(recon.v, pred.v);
        // Luma: each pixel is clamp255(pred + delta).
        for i in 0..256 {
            let expect = (pred.y[i] as i32 + delta as i32).clamp(0, 255) as u8;
            assert_eq!(recon.y[i], expect, "luma px {i}");
        }
        // Sanity: the residue is genuinely non-zero so this isn't a
        // trivial pass.
        assert_ne!(delta, 0);
    }

    #[test]
    fn reconstruct_sub_pixel_vector_rejected() {
        let (y, u, v) = build_reference(1, 1);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 16,
            uv_stride: 8,
            mb_cols: 1,
            mb_rows: 1,
        };
        assert_eq!(
            reconstruct_inter_mb_whole_pixel(
                &reference,
                0,
                0,
                Mv { row: 1, col: 0 },
                false,
                true,
                &[0i16; 16],
                &[[0i16; 16]; 16],
                &[[0i16; 16]; 4],
                &[[0i16; 16]; 4],
            ),
            Err(MotionCompError::SubPixelNotSupported)
        );
    }
}
