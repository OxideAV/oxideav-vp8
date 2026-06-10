//! VP8 §14.1 dequantization scaling and the bitstream→dequant wrapper.
//!
//! RFC 6386 §14.1 (page 76) multiplies every decoded (quantized)
//! coefficient by one of **six** dequantization factors, chosen by plane
//! (Y1, Y2, chroma) and position (DC = coefficient 0, AC = coefficients
//! 1..=15). The two base lookup tables `dc_qlookup[128]` / `ac_qlookup[128]`
//! live in [`crate::inverse_transform`] (transcribed verbatim from §14.1
//! page 77 / §20.3 `dequant_data.h`). §14.1 page 77 states that the Y1 DC
//! and Y1 AC factors are *"directly used"* from those tables, but the Y2
//! and chroma factors *"undergo either scaling or clamping before the
//! multiplies. Details ... can be found in related lookup functions in
//! dixie.c (Section 20.4)."*
//!
//! Section 20.4 (`dixie.c`) is part of RFC 6386, so the scaling / clamping
//! rules it gives are in-spec. The reference `dequant_init` (§20.4 page
//! 144) computes the six factors as:
//!
//! ```text
//! q = quant_hdr->q_index;                        // = yac_qi (§9.6)
//! if (seg->enabled)
//!     q = seg->abs ? seg->quant_idx[i]
//!                  : q + seg->quant_idx[i];       // §10 per-segment override
//! Y1.dc = dc_q(q + y1_dc_delta_q);
//! Y1.ac = ac_q(q);
//! UV.dc = dc_q(q + uv_dc_delta_q);   if (UV.dc > 132) UV.dc = 132;
//! UV.ac = ac_q(q + uv_ac_delta_q);
//! Y2.dc = dc_q(q + y2_dc_delta_q) * 2;
//! Y2.ac = ac_q(q + y2_ac_delta_q) * 155 / 100;   if (Y2.ac < 8) Y2.ac = 8;
//! ```
//!
//! where `dc_q(q) = dc_qlookup[clamp_q(q)]`, `ac_q(q) = ac_qlookup[clamp_q(q)]`,
//! and `clamp_q` saturates the index into `0..=127` (§20.4 page 143).
//!
//! This module turns those rules into:
//!
//! * [`MbDequantFactors`] — the six factors for one macroblock's segment.
//! * [`MbDequantFactors::from_quant_indices`] — the frame-level (no
//!   segmentation override) derivation from a [`QuantIndices`].
//! * [`MbDequantFactors::for_segment`] — the §10 per-segment override
//!   (delta or absolute) on top of the frame-level baseline.
//! * [`MbDequantFactors::dequantize`] — apply the factors to a raw
//!   [`MbCoeffs`] in place (Y2 DC/AC, the sixteen Y sub-blocks, the four U
//!   and four V chroma sub-blocks), each coefficient's product computed in
//!   `i32` and stored back as `i16` (§14.1 page 76 *"computed and stored
//!   using 16-bit signed integers"*).
//! * [`decode_and_dequantize_mb`] — the bitstream→dequant wrapper that
//!   feeds [`decode_keyframe`]: it calls [`decode_mb_coeffs`] to recover
//!   the raw quantized [`MbCoeffs`] from the token partition, then applies
//!   the supplied [`MbDequantFactors`], producing the pre-dequantized
//!   [`MbCoeffs`] that the per-macroblock reconstruction orchestrators (and
//!   thus [`decode_keyframe`]) consume.
//!
//! [`QuantIndices`]: crate::coded_header::QuantIndices
//! [`decode_keyframe`]: crate::frame::decode_keyframe
//! [`decode_mb_coeffs`]: crate::dct_tokens::decode_mb_coeffs

use crate::coded_header::QuantIndices;
use crate::dct_tokens::{decode_mb_coeffs, CoeffProbs, MbCoeffError, MbEntropyCtx};
use crate::frame::MbCoeffs;
use crate::inverse_transform::{clamp_qindex, AC_QLOOKUP, DC_QLOOKUP};

/// `dc_q(q)` from RFC 6386 §20.4: `dc_qlookup[clamp_q(q)]`.
#[inline]
fn dc_q(q: i32) -> i32 {
    DC_QLOOKUP[clamp_qindex(q)] as i32
}

/// `ac_q(q)` from RFC 6386 §20.4: `ac_qlookup[clamp_q(q)]`.
#[inline]
fn ac_q(q: i32) -> i32 {
    AC_QLOOKUP[clamp_qindex(q)] as i32
}

/// The §14.1 maximum chroma-DC factor: `if (UV.dc > 132) UV.dc = 132`
/// (RFC 6386 §20.4 page 144).
pub const UV_DC_MAX: i16 = 132;

/// The §14.1 minimum Y2-AC factor: `if (Y2.ac < 8) Y2.ac = 8`
/// (RFC 6386 §20.4 page 144).
pub const Y2_AC_MIN: i16 = 8;

/// The six §14.1 dequantization factors for one macroblock's segment, in
/// the layout RFC 6386 §20.4 `dequant_init` produces.
///
/// All six are stored as `i16` because §14.1 page 76 specifies the
/// multiplies are *"computed and stored using 16-bit signed integers"*.
/// They never overflow `i16`: the largest base lookup value is
/// `ac_qlookup[127] = 284`, and `284 * 155 / 100 = 440` for Y2-AC,
/// `157 * 2 = 314` for Y2-DC — both well within `i16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbDequantFactors {
    /// Y1 DC factor: `dc_q(q + ydc_delta)`.
    pub y1_dc: i16,
    /// Y1 AC factor: `ac_q(q)`.
    pub y1_ac: i16,
    /// Y2 DC factor: `dc_q(q + y2dc_delta) * 2`.
    pub y2_dc: i16,
    /// Y2 AC factor: `ac_q(q + y2ac_delta) * 155 / 100`, floored at 8.
    pub y2_ac: i16,
    /// Chroma DC factor: `dc_q(q + uvdc_delta)`, capped at 132.
    pub uv_dc: i16,
    /// Chroma AC factor: `ac_q(q + uvac_delta)`.
    pub uv_ac: i16,
}

impl MbDequantFactors {
    /// Compute the six factors from a single base quantizer index `q` and
    /// the five §9.6 deltas, applying the §14.1 / §20.4 scaling and
    /// clamping rules. This is the shared core for both the frame-level and
    /// per-segment paths; `q` is the (already segment-adjusted) base index.
    ///
    /// The `*155/100` Y2-AC scaling and the `*2` Y2-DC scaling are computed
    /// in `i32` (the spec C source uses plain `int`); the products are then
    /// clamped (`< 8 -> 8` for Y2-AC, `> 132 -> 132` for chroma-DC) and
    /// stored as `i16`.
    pub fn from_base_and_deltas(
        q: i32,
        ydc_delta: i32,
        y2dc_delta: i32,
        y2ac_delta: i32,
        uvdc_delta: i32,
        uvac_delta: i32,
    ) -> Self {
        let y1_dc = dc_q(q + ydc_delta);
        let y1_ac = ac_q(q);

        // §20.4: dc_q(q + y2_dc_delta_q) * 2.
        let y2_dc = dc_q(q + y2dc_delta) * 2;
        // §20.4: ac_q(q + y2_ac_delta_q) * 155 / 100, then floor at 8.
        let mut y2_ac = ac_q(q + y2ac_delta) * 155 / 100;
        if y2_ac < Y2_AC_MIN as i32 {
            y2_ac = Y2_AC_MIN as i32;
        }

        // §20.4: dc_q(q + uv_dc_delta_q), then cap at 132.
        let mut uv_dc = dc_q(q + uvdc_delta);
        if uv_dc > UV_DC_MAX as i32 {
            uv_dc = UV_DC_MAX as i32;
        }
        let uv_ac = ac_q(q + uvac_delta);

        MbDequantFactors {
            y1_dc: y1_dc as i16,
            y1_ac: y1_ac as i16,
            y2_dc: y2_dc as i16,
            y2_ac: y2_ac as i16,
            uv_dc: uv_dc as i16,
            uv_ac: uv_ac as i16,
        }
    }

    /// Frame-level dequant factors (no §10 per-segment override) derived
    /// from a parsed [`QuantIndices`]. The base index is `q = yac_qi`
    /// (§9.6), and each `None` delta contributes `0` (§9.6: a delta is read
    /// only when its flag is true; otherwise it is 0).
    ///
    /// [`QuantIndices`]: crate::coded_header::QuantIndices
    pub fn from_quant_indices(qi: &QuantIndices) -> Self {
        Self::from_base_and_deltas(
            qi.y_ac_qi as i32,
            qi.y_dc_delta.unwrap_or(0) as i32,
            qi.y2_dc_delta.unwrap_or(0) as i32,
            qi.y2_ac_delta.unwrap_or(0) as i32,
            qi.uv_dc_delta.unwrap_or(0) as i32,
            qi.uv_ac_delta.unwrap_or(0) as i32,
        )
    }

    /// Per-segment dequant factors with the §10 quantizer override applied
    /// on top of the frame-level [`QuantIndices`] baseline.
    ///
    /// Per RFC 6386 §20.4 `dequant_init`, when segmentation is enabled the
    /// base index for a segment is:
    ///
    /// * **absolute** (`segment_feature_mode_absolute == true`): the
    ///   segment's quantizer value `segment_quant` replaces `yac_qi`
    ///   entirely (`q = segment_quant`);
    /// * **delta** (`false`): the segment value is *added* to the baseline
    ///   (`q = yac_qi + segment_quant`).
    ///
    /// The five §9.6 frame-level deltas (Y-DC, Y2-DC, Y2-AC, UV-DC, UV-AC)
    /// are then applied to that segment base exactly as in the frame-level
    /// path — they are *not* themselves segment-adjusted (§20.4 applies the
    /// segment override only to `q`, then adds the per-plane deltas).
    ///
    /// `segment_quant` is the segment's `quantizer_update` value (§9.3 /
    /// §10); a segment whose `quantizer_update` bit was 0 carries `0` here
    /// (in delta mode that leaves the baseline unchanged; in absolute mode
    /// it pins the base index to 0).
    pub fn for_segment(qi: &QuantIndices, segment_quant: i32, absolute: bool) -> Self {
        let q = if absolute {
            segment_quant
        } else {
            qi.y_ac_qi as i32 + segment_quant
        };
        Self::from_base_and_deltas(
            q,
            qi.y_dc_delta.unwrap_or(0) as i32,
            qi.y2_dc_delta.unwrap_or(0) as i32,
            qi.y2_ac_delta.unwrap_or(0) as i32,
            qi.uv_dc_delta.unwrap_or(0) as i32,
            qi.uv_ac_delta.unwrap_or(0) as i32,
        )
    }

    /// Apply these factors to a raw (quantized) [`MbCoeffs`] in place,
    /// turning it into the pre-dequantized bundle the §14 inverse
    /// transforms / reconstruction orchestrators consume.
    ///
    /// Each block's coefficient 0 (DC) is multiplied by the plane's DC
    /// factor and coefficients 1..=15 (AC) by the plane's AC factor, with
    /// the product computed in `i32` and stored back as `i16` (§14.1 page
    /// 76). The blocks are assumed to already be in raster (natural) order
    /// — which [`decode_mb_coeffs`] guarantees — so coefficient 0 is the DC
    /// in every block regardless of plane.
    ///
    /// For a `B_PRED` / `SPLITMV` macroblock there is no Y2 block; the
    /// caller leaves `coeffs.y2` at zero and the Y plane carries its own DC
    /// (so the Y2 factors simply scale a zero block — harmless). For all
    /// non-`B_PRED` macroblocks the Y2 block is dequantized with the Y2
    /// factors and the sixteen Y sub-blocks with the Y1 factors.
    ///
    /// [`decode_mb_coeffs`]: crate::dct_tokens::decode_mb_coeffs
    pub fn dequantize(&self, coeffs: &mut MbCoeffs) {
        dequant_block(&mut coeffs.y2, self.y2_dc, self.y2_ac);
        for blk in coeffs.y.iter_mut() {
            dequant_block(blk, self.y1_dc, self.y1_ac);
        }
        for blk in coeffs.u.iter_mut() {
            dequant_block(blk, self.uv_dc, self.uv_ac);
        }
        for blk in coeffs.v.iter_mut() {
            dequant_block(blk, self.uv_dc, self.uv_ac);
        }
    }
}

/// Dequantize one raster-order 4×4 block in place: coefficient 0 (DC) is
/// multiplied by `dc_factor`, coefficients 1..=15 (AC) by `ac_factor`. The
/// product is computed in `i32` and stored back as `i16` (§14.1 page 76).
///
/// Dispatch: SIMD path on nightly + `simd`, scalar otherwise. The SIMD
/// path is byte-exact against the scalar listing on every test fixture
/// (`dequant_block_simd_matches_scalar_on_stress_inputs`).
#[inline]
fn dequant_block(block: &mut [i16; 16], dc_factor: i16, ac_factor: i16) {
    #[cfg(feature = "simd")]
    {
        dequant_block_simd(block, dc_factor, ac_factor);
    }
    #[cfg(not(feature = "simd"))]
    {
        dequant_block_scalar(block, dc_factor, ac_factor);
    }
}

/// Scalar §14.1 dequantize — the multiply written out longhand: lane 0
/// scaled by `dc_factor`, lanes 1..=15 by `ac_factor`. Each product is
/// formed in `i32` and truncated back to `i16`, matching the §14.1 page
/// 76 *"computed and stored using 16-bit signed integers"* convention.
///
/// The public [`dequant_block`] dispatches here on stable builds (and on
/// nightly without the `simd` feature); the `simd` feature swaps in
/// [`dequant_block_simd`], which is itself byte-exact against this
/// implementation (`dequant_block_simd_matches_scalar_on_stress_inputs`).
#[allow(dead_code)] // Used by `dequant_block` only on the !simd path.
#[inline]
fn dequant_block_scalar(block: &mut [i16; 16], dc_factor: i16, ac_factor: i16) {
    block[0] = (block[0] as i32 * dc_factor as i32) as i16;
    for c in block.iter_mut().skip(1) {
        *c = (*c as i32 * ac_factor as i32) as i16;
    }
}

/// SIMD §14.1 dequantize — `core::simd::Simd<i32, 16>` rewrite of
/// [`dequant_block_scalar`].
///
/// The whole 4×4 block is sixteen independent coefficient×factor
/// multiplies with no cross-lane dependency, so it maps directly onto a
/// single 16-wide vector: widen the `i16` coefficients to `i32`, multiply
/// lane-wise by a per-lane factor vector (`dc_factor` in lane 0,
/// `ac_factor` in lanes 1..=15), then truncate each product back to `i16`.
///
/// The widening + truncation reproduce the scalar `as i32` / `as i16`
/// casts exactly: `Simd::<i16,16>::cast::<i32>()` is the lane-wise
/// sign-extending widen, lane-wise `*` on `Simd<i32, 16>` is wrapping i32
/// multiply, and `cast::<i16>()` truncates each lane to its low 16 bits
/// (the `as i16` conversion of an `i32`). With every lane independent there
/// is no reassociation question — the SIMD path computes the identical
/// sixteen products as the scalar loop. The byte-exactness is enforced on
/// every fixture by `dequant_block_simd_matches_scalar_on_stress_inputs`.
#[cfg(feature = "simd")]
#[inline]
fn dequant_block_simd(block: &mut [i16; 16], dc_factor: i16, ac_factor: i16) {
    use core::simd::num::SimdInt;
    use core::simd::Simd;

    // Per-lane factor vector: dc_factor for the DC coefficient (lane 0),
    // ac_factor for the fifteen AC coefficients (lanes 1..=15).
    let mut factors = [ac_factor as i32; 16];
    factors[0] = dc_factor as i32;
    let factor_v: Simd<i32, 16> = Simd::from_array(factors);

    // Widen the i16 coefficients to i32 (sign-extending), multiply
    // lane-wise (wrapping i32), then truncate each product back to i16.
    let coeff_v: Simd<i32, 16> = Simd::<i16, 16>::from_array(*block).cast::<i32>();
    let product: Simd<i32, 16> = coeff_v * factor_v;
    *block = product.cast::<i16>().to_array();
}

/// Bitstream→dequant wrapper: decode one macroblock's raw quantized
/// coefficient tokens (§13.3) and apply the §14.1 dequant factors,
/// producing the pre-dequantized [`MbCoeffs`] that [`decode_keyframe`]
/// consumes.
///
/// This completes the keyframe decode chain for one macroblock:
/// bitstream → [`decode_mb_coeffs`] (raw quantized + raster reorder) →
/// [`MbDequantFactors::dequantize`] (§14.1 scaling) → ready for the §14.2
/// inverse-transform / reconstruction step inside [`decode_keyframe`].
///
/// # Inputs
///
/// * `dec` — the §13.2 boolean decoder positioned at this macroblock's
///   residual record (forwarded to [`decode_mb_coeffs`]).
/// * `has_y2` — whether this macroblock carries a Y2 block (`true` for the
///   four 16×16 luma modes; `false` for `B_PRED` / `SPLITMV`).
/// * `mb_skip_coeff` — the §11.1 / §13.1 skip flag.
/// * `coeff_probs` — the resolved coefficient probabilities.
/// * `factors` — the §14.1 dequant factors for this macroblock's segment
///   (from [`MbDequantFactors::from_quant_indices`] or
///   [`MbDequantFactors::for_segment`]).
/// * `above` / `left` — the macroblock's non-zero predictor contexts
///   (forwarded to [`decode_mb_coeffs`] and updated in place).
///
/// # Output
///
/// The macroblock's **dequantized** [`MbCoeffs`], in raster order, ready
/// for the §14 inverse transforms. A skip macroblock yields all-zero
/// coefficients (dequantizing zeros is a no-op).
///
/// [`decode_keyframe`]: crate::frame::decode_keyframe
/// [`decode_mb_coeffs`]: crate::dct_tokens::decode_mb_coeffs
#[allow(clippy::too_many_arguments)]
pub fn decode_and_dequantize_mb(
    dec: &mut crate::bool_decoder::BoolDecoder<'_>,
    has_y2: bool,
    mb_skip_coeff: bool,
    coeff_probs: &CoeffProbs,
    factors: &MbDequantFactors,
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> Result<MbCoeffs, MbCoeffError> {
    let mut coeffs = decode_mb_coeffs(dec, has_y2, mb_skip_coeff, coeff_probs, above, left)?;
    factors.dequantize(&mut coeffs);
    Ok(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coded_header::QuantIndices;

    /// A QuantIndices with a chosen base and all deltas absent (0).
    fn qi_base(yac: u8) -> QuantIndices {
        QuantIndices {
            y_ac_qi: yac,
            y_dc_delta: None,
            y2_dc_delta: None,
            y2_ac_delta: None,
            uv_dc_delta: None,
            uv_ac_delta: None,
        }
    }

    #[test]
    fn y1_factors_use_dc_and_ac_lookups_directly() {
        // §14.1 page 77: Y1 DC uses dc_qlookup[q + ydc_delta], Y1 AC uses
        // ac_qlookup[q]. With q = 40 and ydc_delta = +3:
        let qi = QuantIndices {
            y_ac_qi: 40,
            y_dc_delta: Some(3),
            ..qi_base(40)
        };
        let f = MbDequantFactors::from_quant_indices(&qi);
        assert_eq!(f.y1_dc, DC_QLOOKUP[43]);
        assert_eq!(f.y1_ac, AC_QLOOKUP[40]);
    }

    #[test]
    fn y2_dc_is_doubled() {
        // §20.4: Y2.dc = dc_q(q + y2dc_delta) * 2. q = 30, no delta.
        let qi = qi_base(30);
        let f = MbDequantFactors::from_quant_indices(&qi);
        assert_eq!(f.y2_dc, DC_QLOOKUP[30] * 2);
        // With a delta the index shifts before the doubling.
        let qi2 = QuantIndices {
            y2_dc_delta: Some(5),
            ..qi_base(30)
        };
        let f2 = MbDequantFactors::from_quant_indices(&qi2);
        assert_eq!(f2.y2_dc, DC_QLOOKUP[35] * 2);
    }

    #[test]
    fn y2_ac_is_scaled_by_155_over_100_and_floored_at_8() {
        // §20.4: Y2.ac = ac_q(q + y2ac_delta) * 155 / 100, then >= 8.
        // Pick a q where the base ac value is large enough that the floor
        // does not bite. q = 60 -> ac_qlookup[60] = 70; 70*155/100 = 108.
        let qi = qi_base(60);
        let f = MbDequantFactors::from_quant_indices(&qi);
        let expect = (AC_QLOOKUP[60] as i32 * 155 / 100) as i16;
        assert_eq!(f.y2_ac, expect);
        assert_eq!(AC_QLOOKUP[60], 70);
        assert_eq!(f.y2_ac, 108);

        // The smallest possible base (q = 0 -> ac_qlookup[0] = 4):
        // 4 * 155 / 100 = 620 / 100 = 6 (integer) -> floored up to 8.
        let qi0 = qi_base(0);
        let f0 = MbDequantFactors::from_quant_indices(&qi0);
        assert_eq!(AC_QLOOKUP[0], 4);
        assert_eq!(4 * 155 / 100, 6);
        assert_eq!(f0.y2_ac, 8, "the <8 floor must lift 6 up to 8");
    }

    #[test]
    fn y2_ac_integer_division_truncates() {
        // q = 5 -> ac_qlookup[5] = 9; 9 * 155 = 1395; 1395 / 100 = 13
        // (integer truncation, not 13.95 rounded). This is large enough to
        // skip the >=8 floor, so it isolates the truncation behaviour.
        let qi = qi_base(5);
        let f = MbDequantFactors::from_quant_indices(&qi);
        assert_eq!(AC_QLOOKUP[5], 9);
        assert_eq!(9 * 155 / 100, 13);
        assert_eq!(f.y2_ac, 13);
    }

    #[test]
    fn uv_dc_is_clamped_to_132() {
        // §20.4: UV.dc = dc_q(q + uvdc_delta), capped at 132. dc_qlookup
        // hits 132 at index 117 (...130, 132, 134...). Index 118 = 134
        // would exceed 132, so the cap applies from there on.
        assert_eq!(DC_QLOOKUP[117], 132);
        assert!(DC_QLOOKUP[118] > 132);
        let f_at = MbDequantFactors::from_quant_indices(&qi_base(117));
        assert_eq!(f_at.uv_dc, 132);
        let f_over = MbDequantFactors::from_quant_indices(&qi_base(120));
        assert_eq!(f_over.uv_dc, UV_DC_MAX, "uv_dc must be capped at 132");
        // Below the cap it passes through unscaled.
        let f_below = MbDequantFactors::from_quant_indices(&qi_base(50));
        assert_eq!(f_below.uv_dc, DC_QLOOKUP[50]);
        assert!(f_below.uv_dc < 132);
    }

    #[test]
    fn uv_ac_uses_ac_lookup_with_its_own_delta() {
        let qi = QuantIndices {
            uv_ac_delta: Some(-4),
            ..qi_base(50)
        };
        let f = MbDequantFactors::from_quant_indices(&qi);
        assert_eq!(f.uv_ac, AC_QLOOKUP[46]);
    }

    #[test]
    fn qindex_clamps_at_both_ends_through_factors() {
        // A huge negative Y-DC delta saturates the dc index to 0.
        let qi = QuantIndices {
            y_dc_delta: Some(-127),
            ..qi_base(10)
        };
        let f = MbDequantFactors::from_quant_indices(&qi);
        assert_eq!(f.y1_dc, DC_QLOOKUP[0]);
        // A base above 127 (only reachable via segmentation) saturates the
        // ac index to 127.
        let f_hi = MbDequantFactors::from_base_and_deltas(200, 0, 0, 0, 0, 0);
        assert_eq!(f_hi.y1_ac, AC_QLOOKUP[127]);
    }

    #[test]
    fn segment_delta_mode_adds_to_baseline() {
        // §20.4 delta mode: q = yac_qi + segment_quant.
        let qi = qi_base(40);
        let f = MbDequantFactors::for_segment(&qi, 10, false);
        // base index = 50; Y1 AC = ac_qlookup[50].
        assert_eq!(f.y1_ac, AC_QLOOKUP[50]);
        // A negative segment delta subtracts.
        let f_neg = MbDequantFactors::for_segment(&qi, -15, false);
        assert_eq!(f_neg.y1_ac, AC_QLOOKUP[25]);
    }

    #[test]
    fn segment_absolute_mode_replaces_baseline() {
        // §20.4 absolute mode: q = segment_quant (yac_qi ignored for q).
        let qi = qi_base(40);
        let f = MbDequantFactors::for_segment(&qi, 70, true);
        assert_eq!(f.y1_ac, AC_QLOOKUP[70]);
        // The frame-level base (40) must NOT contribute in absolute mode.
        assert_ne!(f.y1_ac, AC_QLOOKUP[110]);
    }

    #[test]
    fn segment_keeps_perplane_deltas() {
        // The five §9.6 per-plane deltas still apply on top of the segment
        // base. Absolute segment q = 60, y2_dc_delta = +5: Y2.dc must use
        // index 65.
        let qi = QuantIndices {
            y2_dc_delta: Some(5),
            ..qi_base(40)
        };
        let f = MbDequantFactors::for_segment(&qi, 60, true);
        assert_eq!(f.y2_dc, DC_QLOOKUP[65] * 2);
    }

    #[test]
    fn dequantize_scales_each_plane_correctly() {
        let mut c = MbCoeffs::default();
        // Y2: DC and one AC.
        c.y2[0] = 2;
        c.y2[1] = 3;
        // Y sub-block 0: DC and one AC.
        c.y[0][0] = 4;
        c.y[0][5] = -2;
        // U / V sub-blocks.
        c.u[0][0] = 6;
        c.u[0][2] = 5;
        c.v[3][0] = -3;
        c.v[3][15] = 2;

        let f = MbDequantFactors {
            y1_dc: 10,
            y1_ac: 7,
            y2_dc: 100,
            y2_ac: 50,
            uv_dc: 20,
            uv_ac: 9,
        };
        f.dequantize(&mut c);

        assert_eq!(c.y2[0], 2 * 100);
        assert_eq!(c.y2[1], 3 * 50);
        assert_eq!(c.y[0][0], 4 * 10);
        assert_eq!(c.y[0][5], -2 * 7);
        assert_eq!(c.u[0][0], 6 * 20);
        assert_eq!(c.u[0][2], 5 * 9);
        assert_eq!(c.v[3][0], -3 * 20);
        assert_eq!(c.v[3][15], 2 * 9);
        // Untouched coefficients stay zero (0 * anything = 0).
        assert_eq!(c.y[1][0], 0);
        assert_eq!(c.u[1][1], 0);
    }

    #[test]
    fn dequantize_is_noop_on_zero_block() {
        let mut c = MbCoeffs::default();
        let f = MbDequantFactors::from_quant_indices(&qi_base(50));
        f.dequantize(&mut c);
        assert_eq!(c.y2, [0i16; 16]);
        assert!(c.y.iter().all(|b| *b == [0i16; 16]));
        assert!(c.u.iter().all(|b| *b == [0i16; 16]));
        assert!(c.v.iter().all(|b| *b == [0i16; 16]));
    }

    #[test]
    fn factors_match_dixie_reference_arithmetic_for_a_sample_q() {
        // Independent re-derivation of the §20.4 dequant_init body for
        // q = 64 with the worked deltas (y_dc=+1, y2_dc=-2, y2_ac=+15,
        // uv_dc=-15, uv_ac=0) from the §9.6 header test vector.
        let qi = QuantIndices {
            y_ac_qi: 64,
            y_dc_delta: Some(1),
            y2_dc_delta: Some(-2),
            y2_ac_delta: Some(15),
            uv_dc_delta: Some(-15),
            uv_ac_delta: Some(0),
        };
        let f = MbDequantFactors::from_quant_indices(&qi);

        let q = 64i32;
        let exp_y1_dc = DC_QLOOKUP[(q + 1) as usize];
        let exp_y1_ac = AC_QLOOKUP[q as usize];
        let exp_y2_dc = DC_QLOOKUP[(q - 2) as usize] * 2;
        let mut exp_y2_ac = AC_QLOOKUP[(q + 15) as usize] as i32 * 155 / 100;
        if exp_y2_ac < 8 {
            exp_y2_ac = 8;
        }
        let mut exp_uv_dc = DC_QLOOKUP[(q - 15) as usize] as i32;
        if exp_uv_dc > 132 {
            exp_uv_dc = 132;
        }
        let exp_uv_ac = AC_QLOOKUP[q as usize];

        assert_eq!(f.y1_dc, exp_y1_dc);
        assert_eq!(f.y1_ac, exp_y1_ac);
        assert_eq!(f.y2_dc, exp_y2_dc);
        assert_eq!(f.y2_ac, exp_y2_ac as i16);
        assert_eq!(f.uv_dc, exp_uv_dc as i16);
        assert_eq!(f.uv_ac, exp_uv_ac);
    }

    // -----------------------------------------------------------------
    // Wired-chain tests: bitstream → dequant → reconstruct → pixels
    // -----------------------------------------------------------------

    use crate::bool_decoder::BoolDecoder;
    use crate::dct_tokens::{decode_mb_coeffs, merge_default_token_probs, DEFAULT_COEFF_PROBS};
    use crate::frame::{decode_keyframe, MbCoeffs};
    use crate::intra_predict::DEFAULT_TOPLEFT_DC;
    use crate::macroblock::{IntraUvMode, IntraYMode, MacroblockModes};

    /// A 16×16-mode macroblock mode record.
    fn mode_16x16(y_mode: IntraYMode, uv_mode: IntraUvMode, skip: bool) -> MacroblockModes {
        MacroblockModes {
            segment_id: None,
            mb_skip_coeff: skip,
            y_mode,
            subblock_modes: None,
            uv_mode,
        }
    }

    #[test]
    fn wrapper_matches_decode_then_dequantize_for_skip_mb() {
        // The bitstream→dequant wrapper must equal decode_mb_coeffs +
        // dequantize. A skip macroblock reads no tokens, so two empty
        // partitions suffice to drive a real BoolDecoder through both
        // paths; the result is all-zero coefficients (dequant of zeros).
        let probs = merge_default_token_probs(&[[[[None; 11]; 3]; 8]; 4]);
        // The default table directly, just to exercise both.
        assert_eq!(probs, DEFAULT_COEFF_PROBS);
        let factors = MbDequantFactors::from_quant_indices(&qi_base(40));

        // Path A: the wrapper.
        let buf_a = [0u8; 4];
        let mut dec_a = BoolDecoder::init(&buf_a).unwrap();
        let mut a_above = MbEntropyCtx::default();
        let mut a_left = MbEntropyCtx::default();
        let via_wrapper = super::decode_and_dequantize_mb(
            &mut dec_a,
            true,
            true, // skip
            &probs,
            &factors,
            &mut a_above,
            &mut a_left,
        )
        .unwrap();

        // Path B: decode then dequantize by hand.
        let buf_b = [0u8; 4];
        let mut dec_b = BoolDecoder::init(&buf_b).unwrap();
        let mut b_above = MbEntropyCtx::default();
        let mut b_left = MbEntropyCtx::default();
        let mut manual =
            decode_mb_coeffs(&mut dec_b, true, true, &probs, &mut b_above, &mut b_left).unwrap();
        factors.dequantize(&mut manual);

        assert_eq!(via_wrapper, manual);
        // Both predictor contexts must have evolved identically (skip-reset).
        assert_eq!(a_above, b_above);
        assert_eq!(a_left, b_left);
        // A skip MB yields all-zero dequantized coefficients.
        assert_eq!(via_wrapper.y2, [0i16; 16]);
    }

    #[test]
    fn full_keyframe_decode_through_wired_chain() {
        // End-to-end: a 1×1 key frame whose single non-B_PRED macroblock
        // carries a Y2 DC coefficient. We run the wired chain
        // (raw MbCoeffs -> MbDequantFactors::dequantize -> decode_keyframe)
        // and confirm the §14.1 Y2 DC scaling propagates into the pixels.
        //
        // The MB is DC_PRED in both planes with no neighbours, so the §12.2
        // prediction is a flat 128. The Y2 block holds a single DC; after
        // dequant (Y2.dc = dc_q(q) * 2) the inverse WHT seeds each Y
        // sub-block's DC, the inverse DCT spreads it, and §14.5 adds it to
        // the 128 prediction. With the same raw coefficient but a LARGER
        // quantizer, the reconstructed luma must move further from 128 —
        // proving the dequant factor is actually being applied.
        let raw_dc = 3i16;

        let recon_for = |yac: u8| -> Vec<u8> {
            let factors = MbDequantFactors::from_quant_indices(&qi_base(yac));
            let mut coeffs = MbCoeffs::default();
            coeffs.y2[0] = raw_dc; // raw quantized Y2 DC (raster pos 0)
            factors.dequantize(&mut coeffs);
            let modes = vec![mode_16x16(IntraYMode::Dc, IntraUvMode::Dc, false)];
            let planes = decode_keyframe(1, 1, &modes, &[coeffs]).expect("decode");
            // Chroma had no residue → flat 128.
            assert!(planes.u.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
            assert!(planes.v.iter().all(|&p| p == DEFAULT_TOPLEFT_DC));
            planes.y
        };

        // Small quantizer (q small → small dc_q factor).
        let y_small = recon_for(8);
        // Larger quantizer (q large → larger dc_q factor).
        let y_large = recon_for(100);

        // Both must differ from the flat 128 prediction (the Y2 DC lifted
        // them), and the larger quantizer must lift them further.
        let dev = |plane: &[u8]| -> i32 {
            plane
                .iter()
                .map(|&p| (p as i32 - DEFAULT_TOPLEFT_DC as i32).abs())
                .sum()
        };
        let d_small = dev(&y_small);
        let d_large = dev(&y_large);
        assert!(d_small > 0, "Y2 DC residue must move luma off flat 128");
        assert!(
            d_large > d_small,
            "a larger dequant factor must move luma further from 128 \
             (small q deviation {d_small}, large q deviation {d_large})"
        );

        // Cross-check the exact factor effect: the dequantized Y2 DC equals
        // dc_q(q) * 2 * raw_dc for each q, so the larger one is strictly
        // bigger.
        let f_small = MbDequantFactors::from_quant_indices(&qi_base(8));
        let f_large = MbDequantFactors::from_quant_indices(&qi_base(100));
        assert!(
            (f_large.y2_dc as i32 * raw_dc as i32) > (f_small.y2_dc as i32 * raw_dc as i32),
            "larger quantizer must yield a larger dequantized Y2 DC"
        );
    }

    // -----------------------------------------------------------------
    // SIMD vs scalar byte-exact equivalence test (round-267 dequant SIMD).
    //
    // Mirrors the transform-side `*_simd_matches_scalar_on_stress_inputs`
    // pairs. On stable / no `simd` feature this exercises the scalar path
    // twice (the public `dequant_block` dispatch and the directly-named
    // `_scalar` fn) — harmless; keeps CI green. On nightly + `simd` it's
    // the primary safety net: `dequant_block` dispatches to the SIMD path
    // while `dequant_block_scalar` stays reachable as the fallback, so the
    // assertion compares the two bit-for-bit, including the i16-overflow
    // truncation cases that distinguish `cast::<i16>()` from saturation.
    // -----------------------------------------------------------------

    /// (coefficient block, dc_factor, ac_factor) stress fixtures covering
    /// all-zero, DC-only, single-AC at each lane, mixed patterns, and the
    /// i16-overflow cases that force the `as i16` / `cast::<i16>()`
    /// wrap-around path. The factors span the real §14.1 range
    /// (`dc_qlookup[0]=4` .. Y2-AC `440`) plus deliberate overflow drivers.
    fn dequant_stress_inputs() -> Vec<([i16; 16], i16, i16)> {
        let mut v: Vec<([i16; 16], i16, i16)> = Vec::new();
        // All-zero block (dequant of zeros is a no-op).
        v.push(([0i16; 16], 4, 8));
        // DC-only, every real factor pairing.
        for &(dc_f, ac_f) in &[(4i16, 8i16), (157, 284), (314, 440), (132, 20)] {
            let mut b = [0i16; 16];
            b[0] = 17;
            v.push((b, dc_f, ac_f));
        }
        // Single non-zero AC at each lane (1..=15).
        for pos in 1..16 {
            let mut b = [0i16; 16];
            b[pos] = 23;
            v.push((b, 17, 29));
        }
        // Mixed sign / magnitude block at a mid-range factor.
        v.push((
            [
                40, -8, 4, -2, //
                -6, 5, -3, 1, //
                3, -2, 1, -1, //
                -1, 1, 0, 2,
            ],
            41,
            37,
        ));
        // i16-overflow drivers: large coefficient × large factor wraps the
        // i32 product when narrowed to i16 — the case where saturating and
        // truncating casts diverge. Both lanes must agree on `as i16` wrap.
        v.push(([i16::MAX; 16], 440, 440));
        v.push(([i16::MIN; 16], 440, 440));
        v.push((
            [
                30000, -30000, 20000, -20000, //
                12345, -23456, 32000, -32000, //
                500, -500, 9999, -9999, //
                32767, -32768, 16384, -16384,
            ],
            314,
            440,
        ));
        v
    }

    #[test]
    fn dequant_block_simd_matches_scalar_on_stress_inputs() {
        for (idx, &(block, dc_f, ac_f)) in dequant_stress_inputs().iter().enumerate() {
            // `dequant_block` is the public dispatcher: SIMD on nightly +
            // `simd`, scalar otherwise. Compare it against the directly-
            // named scalar fallback on the identical input.
            let mut via_dispatch = block;
            dequant_block(&mut via_dispatch, dc_f, ac_f);

            let mut via_scalar = block;
            super::dequant_block_scalar(&mut via_scalar, dc_f, ac_f);

            assert_eq!(
                via_dispatch, via_scalar,
                "dequant_block (dispatch) diverged from scalar on stress \
                 input #{idx} (dc_factor {dc_f}, ac_factor {ac_f})"
            );
        }
    }
}
