//! VP8 loop filter per RFC 6386 §15.
//!
//! The loop filter is the last stage of frame reconstruction (§15 page
//! 84): after the predictor and residue have been summed for every
//! macroblock (§14), the filter is applied to the edges between adjacent
//! macroblocks and the edges between adjacent subblocks, reducing the
//! blocking artifacts that arise from the semi-independent coding of
//! macroblocks and their subblocks. Because filtered output feeds the
//! prediction of subsequent frames (§15 page 84, *"the results of loop
//! filtering are used in the prediction of subsequent frames"*), a
//! conforming decoder must reproduce it bit-for-bit.
//!
//! This module implements the parts of §15 that the RFC specifies
//! **completely in its body**: the per-segment filtering arithmetic
//! (§15.2 simple filter, §15.3 normal filter) and the derivation of the
//! threshold/limit control parameters from `loop_filter_level` and
//! `sharpness_level` (§15.4). Every filter routine operates on a single
//! "segment" — the small run of pixels (length 2, 4, 6, or 8)
//! symmetrically straddling one edge position — exactly as the spec's
//! reference routines do, so the routines are agnostic to whether the
//! edge is horizontal or vertical: the caller gathers the segment into
//! a contiguous array and writes it back.
//!
//! # What is in scope (§15 body)
//!
//! * **§15.2 helpers** — the saturating clamp `c` (`clamp_s8`), the
//!   signed/unsigned pixel conversions `u2s` / `s2u`, and the
//!   `common_adjust` core adjustment shared by both filter types.
//! * **§15.2 simple filter** — `simple_segment`, the 4-pixel filter
//!   gated by an `edge_limit`. The spec notes the simple filter only
//!   applies to luma edges; chroma edges are left unfiltered (§15.2
//!   page 88).
//! * **§15.3 normal filter** — the `filter_yes` enable test, the `hev`
//!   high-edge-variance test, `subblock_filter` (the inter-subblock
//!   variant), and `mb_filter` (the wider inter-macroblock variant
//!   spelled `MBfilter` in the RFC).
//! * **§15.4 control parameters** — [`LoopFilterParams::derive`], which
//!   computes `interior_limit`, `hev_threshold`, `mbedge_limit`, and
//!   `sub_bedge_limit` from a per-macroblock `loop_filter_level`, the
//!   frame `sharpness_level`, and the key-frame flag.
//!
//! # What is NOT in this module (out of scope for §15 standalone)
//!
//! * The §15.1 macroblock-by-macroblock filter *geometry* — walking the
//!   reconstructed frame in raster order, gathering the 16 luma / 8
//!   chroma segments straddling each of the four edges per macroblock,
//!   honouring the order of the four filtering steps, and applying the
//!   §15.1 page 86 skip rule (steps 2 and 4 skipped when the macroblock
//!   is neither `B_PRED` nor `SPLITMV` and has no coded coefficient).
//!   That orchestration belongs to the per-macroblock reconstruction
//!   walk (the integration round); this module provides the per-segment
//!   primitives it calls.
//! * The §9.4 / §10 derivation of the per-macroblock `loop_filter_level`
//!   itself (segment override + the reference-frame / prediction-mode
//!   deltas). [`LoopFilterParams::derive`] takes the already-resolved
//!   level as input; computing it from the [`crate::Vp8CodedHeader`]
//!   fields is the integration round's job.

/// The §15.4 control parameters for one filtering pass over a
/// macroblock, derived from a resolved (post-override) per-macroblock
/// `loop_filter_level`, the frame-constant `sharpness_level`, and the
/// key-frame flag.
///
/// All four fields are unsigned 8-bit limits as the spec defines them.
/// The two edge limits differ for inter-macroblock vs. inter-subblock
/// edges; `interior_limit` and `hev_threshold` are the same for all
/// edge types within the macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterParams {
    /// §15.4 `interior_limit` — limit on differences between adjacent
    /// interior pixels of a segment (used only by the normal filter).
    pub interior_limit: u8,
    /// §15.4 `hev_threshold` — the high-edge-variance threshold (used
    /// only by the normal filter).
    pub hev_threshold: u8,
    /// §15.4 `mbedge_limit` — the edge limit applied to inter-macroblock
    /// edges (used by both filters).
    pub mbedge_limit: u8,
    /// §15.4 `sub_bedge_limit` — the edge limit applied to
    /// inter-subblock edges (used by both filters).
    pub sub_bedge_limit: u8,
}

impl LoopFilterParams {
    /// Derive the §15.4 control parameters.
    ///
    /// `loop_filter_level` is the resolved per-macroblock level (§9.4 /
    /// §10 overrides already applied by the caller), in the range
    /// `0..=63`. `sharpness_level` is the frame-constant 3-bit value
    /// (`0..=7`). `key_frame` selects the §15.4 page 96 key-frame vs.
    /// interframe `hev_threshold` ladder.
    ///
    /// Note (§15 page 84): the caller must *skip* filtering entirely
    /// when `loop_filter_level == 0`; this function still returns a
    /// (degenerate) parameter set in that case rather than panicking, so
    /// the skip decision stays the caller's responsibility.
    pub fn derive(loop_filter_level: u8, sharpness_level: u8, key_frame: bool) -> Self {
        // §15.4: interior_limit derivation.
        let mut interior_limit = loop_filter_level;
        if sharpness_level != 0 {
            interior_limit >>= if sharpness_level > 4 { 2 } else { 1 };
            let cap = 9u8.saturating_sub(sharpness_level);
            if interior_limit > cap {
                interior_limit = cap;
            }
        }
        if interior_limit == 0 {
            interior_limit = 1;
        }

        // §15.4: hev_threshold derivation (key frame vs. interframe).
        let hev_threshold = if key_frame {
            if loop_filter_level >= 40 {
                2
            } else if loop_filter_level >= 15 {
                1
            } else {
                0
            }
        } else if loop_filter_level >= 40 {
            3
        } else if loop_filter_level >= 20 {
            2
        } else if loop_filter_level >= 15 {
            1
        } else {
            0
        };

        // §15.4: edge limits. Both arithmetics are performed in a wider
        // type and saturated into u8 — for the maximum legal inputs
        // (`loop_filter_level = 63`, `interior_limit = 63`) the
        // mbedge_limit reaches `(63 + 2) * 2 + 63 = 193`, well inside
        // u8, but saturating is defensive against malformed levels.
        let mbedge_limit =
            (((loop_filter_level as u16 + 2) * 2) + interior_limit as u16).min(255) as u8;
        let sub_bedge_limit =
            ((loop_filter_level as u16 * 2) + interior_limit as u16).min(255) as u8;

        LoopFilterParams {
            interior_limit,
            hev_threshold,
            mbedge_limit,
            sub_bedge_limit,
        }
    }
}

/// §15.2 `c` — clamp an integer to the signed 8-bit range
/// `-128..=127`.
///
/// The spec's `int8 c(int v)` returns `v < -128 ? -128 : (v < 128 ? v :
/// 127)`.
#[inline]
pub fn clamp_s8(v: i32) -> i8 {
    if v < -128 {
        -128
    } else if v < 128 {
        v as i8
    } else {
        127
    }
}

/// §15.2 `u2s` — convert an unsigned pixel `0..=255` to a signed 8-bit
/// number (`v - 128`), interpreted as `i8`.
#[inline]
pub fn u2s(v: u8) -> i8 {
    (v as i16 - 128) as i8
}

/// §15.2 `s2u` — clamp a signed value with [`clamp_s8`], then convert
/// back to an unsigned pixel by adding 128.
#[inline]
pub fn s2u(v: i32) -> u8 {
    (clamp_s8(v) as i16 + 128) as u8
}

/// §15.2 `abs` for pixel differences (`-255..=255`).
#[inline]
fn abs_i(v: i32) -> i32 {
    v.abs()
}

/// §15.2 `common_adjust` — the core edge adjustment shared by both
/// filter types.
///
/// Operates in-place on a 4-pixel window `[p1, p0, q0, q1]` (indices
/// `0..=3` of `seg`), where `p0`/`q0` straddle the edge. When
/// `use_outer_taps` is true the outer taps `p1`/`q1` participate (a
/// 4-tap filter); when false they do not (a 2-tap filter). Returns the
/// signed adjustment `a` applied to `q0` (the subblock filter uses it).
///
/// The spec passes the four pixels as separate pointers `P1, P0, Q0,
/// Q1`; here they are the first four entries of the shared segment
/// buffer so the normal-filter routines can index further out.
#[inline]
pub fn common_adjust(use_outer_taps: bool, seg: &mut [u8], base: usize) -> i32 {
    let p1 = u2s(seg[base]) as i32;
    let p0 = u2s(seg[base + 1]) as i32;
    let q0 = u2s(seg[base + 2]) as i32;
    let q1 = u2s(seg[base + 3]) as i32;

    // §15.2: a = c((use_outer_taps? c(p1 - q1) : 0) + 3*(q0 - p0)).
    let outer = if use_outer_taps {
        clamp_s8(p1 - q1) as i32
    } else {
        0
    };
    let a = clamp_s8(outer + 3 * (q0 - p0)) as i32;

    // §15.2: b balances the rounding of a/8 when the fractional part is
    // exactly 1/2; b = c(a + 3) >> 3.
    let b = (clamp_s8(a + 3) as i32) >> 3;

    // §15.2: a = c(a + 4) >> 3 — divide by 8 rounding up at the half.
    let a = (clamp_s8(a + 4) as i32) >> 3;

    // §15.2: Q0 = s2u(q0 - a); P0 = s2u(p0 + b).
    seg[base + 2] = s2u(q0 - a);
    seg[base + 1] = s2u(p0 + b);

    a
}

/// §15.2 `simple_segment` — the simple loop filter applied to one
/// 4-pixel window `[p1, p0, q0, q1]` at `seg[base..base + 4]`.
///
/// Does nothing if the §15.2 edge metric `abs(p0 - q0)*2 + abs(p1 -
/// q1)/2` exceeds `edge_limit`; otherwise runs [`common_adjust`] with
/// outer taps.
#[inline]
pub fn simple_segment(edge_limit: u8, seg: &mut [u8], base: usize) {
    let p1 = seg[base] as i32;
    let p0 = seg[base + 1] as i32;
    let q0 = seg[base + 2] as i32;
    let q1 = seg[base + 3] as i32;
    if (abs_i(p0 - q0) * 2 + abs_i(p1 - q1) / 2) <= edge_limit as i32 {
        common_adjust(true, seg, base);
    }
}

/// §15.3 `filter_yes` — the normal-filter enable test over an 8-pixel
/// segment `p3 p2 p1 p0 | q0 q1 q2 q3` (signed values).
///
/// Filtering proceeds only if the edge metric is within `e` AND every
/// adjacent interior difference is within `i`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn filter_yes(
    i: u8,
    e: u8,
    p3: i32,
    p2: i32,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    q3: i32,
) -> bool {
    let i = i as i32;
    let e = e as i32;
    (abs_i(p0 - q0) * 2 + abs_i(p1 - q1) / 2) <= e
        && abs_i(p3 - p2) <= i
        && abs_i(p2 - p1) <= i
        && abs_i(p1 - p0) <= i
        && abs_i(q3 - q2) <= i
        && abs_i(q2 - q1) <= i
        && abs_i(q1 - q0) <= i
}

/// §15.3 `hev` — the high-edge-variance test over the 4 pixels nearest
/// the edge (signed values).
#[inline]
fn hev(threshold: u8, p1: i32, p0: i32, q0: i32, q1: i32) -> bool {
    let t = threshold as i32;
    abs_i(p1 - p0) > t || abs_i(q1 - q0) > t
}

/// §15.3 `subblock_filter` — the normal inter-subblock filter over an
/// 8-pixel segment `[p3, p2, p1, p0, q0, q1, q2, q3]` at
/// `seg[base..base + 8]`.
///
/// If [`filter_yes`] passes, [`common_adjust`] is applied to the inner
/// 4-pixel window with the outer taps gated by [`hev`]; when edge
/// variance is *low* the two pixels one step in (`p1`, `q1`) are
/// additionally adjusted by roughly half the edge adjustment.
#[inline]
pub fn subblock_filter(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    seg: &mut [u8],
    base: usize,
) {
    let p3 = u2s(seg[base]) as i32;
    let p2 = u2s(seg[base + 1]) as i32;
    let p1 = u2s(seg[base + 2]) as i32;
    let p0 = u2s(seg[base + 3]) as i32;
    let q0 = u2s(seg[base + 4]) as i32;
    let q1 = u2s(seg[base + 5]) as i32;
    let q2 = u2s(seg[base + 6]) as i32;
    let q3 = u2s(seg[base + 7]) as i32;

    if filter_yes(interior_limit, edge_limit, p3, p2, p1, p0, q0, q1, q2, q3) {
        let hv = hev(hev_threshold, p1, p0, q0, q1);
        // common_adjust operates on the inner window p1 p0 q0 q1 =
        // seg[base + 2 ..= base + 5].
        let a = (common_adjust(hv, seg, base + 2) + 1) >> 1;
        if !hv {
            // §15.3: Q1 = s2u(q1 - a); P1 = s2u(p1 + a).
            seg[base + 5] = s2u(q1 - a);
            seg[base + 2] = s2u(p1 + a);
        }
    }
}

/// §15.3 `MBfilter` — the normal inter-macroblock filter over an
/// 8-pixel segment `[p3, p2, p1, p0, q0, q1, q2, q3]` at
/// `seg[base..base + 8]`.
///
/// If [`filter_yes`] passes and edge variance is *low*, the wider
/// adjustment is applied: six pixels symmetric about the edge are
/// modified with decaying magnitude (3/7, 2/7, 1/7 of the edge
/// difference). If edge variance is *high*, the simple [`common_adjust`]
/// (outer taps) is used instead.
#[inline]
pub fn mb_filter(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    seg: &mut [u8],
    base: usize,
) {
    let p3 = u2s(seg[base]) as i32;
    let p2 = u2s(seg[base + 1]) as i32;
    let p1 = u2s(seg[base + 2]) as i32;
    let p0 = u2s(seg[base + 3]) as i32;
    let q0 = u2s(seg[base + 4]) as i32;
    let q1 = u2s(seg[base + 5]) as i32;
    let q2 = u2s(seg[base + 6]) as i32;
    let q3 = u2s(seg[base + 7]) as i32;

    if filter_yes(interior_limit, edge_limit, p3, p2, p1, p0, q0, q1, q2, q3) {
        if !hev(hev_threshold, p1, p0, q0, q1) {
            // §15.3: w = c(c(p1 - q1) + 3*(q0 - p0)).
            let w = clamp_s8(clamp_s8(p1 - q1) as i32 + 3 * (q0 - p0)) as i32;

            // §15.3: a = c((27*w + 63) >> 7); Q0 -= a; P0 += a.
            let a = clamp_s8((27 * w + 63) >> 7) as i32;
            seg[base + 4] = s2u(q0 - a);
            seg[base + 3] = s2u(p0 + a);

            // §15.3: a = c((18*w + 63) >> 7); Q1 -= a; P1 += a.
            let a = clamp_s8((18 * w + 63) >> 7) as i32;
            seg[base + 5] = s2u(q1 - a);
            seg[base + 2] = s2u(p1 + a);

            // §15.3: a = c((9*w + 63) >> 7); Q2 -= a; P2 += a.
            let a = clamp_s8((9 * w + 63) >> 7) as i32;
            seg[base + 6] = s2u(q2 - a);
            seg[base + 1] = s2u(p2 + a);
        } else {
            // §15.3: if hev, do the simple filter on the inner window.
            common_adjust(true, seg, base + 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- §15.2 helpers -----------------------------------------------

    #[test]
    fn clamp_s8_saturates_both_ends() {
        assert_eq!(clamp_s8(-200), -128);
        assert_eq!(clamp_s8(-129), -128);
        assert_eq!(clamp_s8(-128), -128);
        assert_eq!(clamp_s8(-1), -1);
        assert_eq!(clamp_s8(0), 0);
        assert_eq!(clamp_s8(127), 127);
        assert_eq!(clamp_s8(128), 127);
        assert_eq!(clamp_s8(500), 127);
    }

    #[test]
    fn u2s_s2u_round_trip_in_range() {
        // u2s maps 0..=255 -> -128..=127; s2u inverts within the signed
        // range. For any pixel value, s2u(u2s(v) as i32) == v.
        for v in 0u8..=255 {
            assert_eq!(s2u(u2s(v) as i32), v, "v = {v}");
        }
    }

    #[test]
    fn u2s_known_points() {
        assert_eq!(u2s(0), -128);
        assert_eq!(u2s(128), 0);
        assert_eq!(u2s(255), 127);
    }

    #[test]
    fn s2u_clamps_out_of_range() {
        // s2u clamps the signed argument to -128..=127 before +128.
        assert_eq!(s2u(-200), 0); // clamp to -128 -> 0
        assert_eq!(s2u(200), 255); // clamp to 127 -> 255
        assert_eq!(s2u(0), 128);
    }

    // ----- §15.4 control parameters ------------------------------------

    #[test]
    fn derive_interior_limit_no_sharpness() {
        // sharpness 0: interior_limit == loop_filter_level (but never 0).
        let p = LoopFilterParams::derive(30, 0, true);
        assert_eq!(p.interior_limit, 30);
        let p0 = LoopFilterParams::derive(0, 0, true);
        assert_eq!(p0.interior_limit, 1, "interior_limit floored to 1");
    }

    #[test]
    fn derive_interior_limit_low_sharpness_shifts_by_one() {
        // sharpness in 1..=4 shifts right by 1, then caps at 9 - sharpness.
        // level 20, sharpness 2: 20 >> 1 = 10, cap = 9 - 2 = 7 -> 7.
        let p = LoopFilterParams::derive(20, 2, true);
        assert_eq!(p.interior_limit, 7);
        // level 10, sharpness 1: 10 >> 1 = 5, cap = 8 -> 5 (uncapped).
        let p2 = LoopFilterParams::derive(10, 1, true);
        assert_eq!(p2.interior_limit, 5);
    }

    #[test]
    fn derive_interior_limit_high_sharpness_shifts_by_two() {
        // sharpness > 4 shifts right by 2, then caps at 9 - sharpness.
        // level 40, sharpness 7: 40 >> 2 = 10, cap = 9 - 7 = 2 -> 2.
        let p = LoopFilterParams::derive(40, 7, true);
        assert_eq!(p.interior_limit, 2);
        // level 8, sharpness 5: 8 >> 2 = 2, cap = 9 - 5 = 4 -> 2 (uncapped).
        let p2 = LoopFilterParams::derive(8, 5, true);
        assert_eq!(p2.interior_limit, 2);
    }

    #[test]
    fn derive_hev_threshold_key_frame_ladder() {
        assert_eq!(LoopFilterParams::derive(45, 0, true).hev_threshold, 2);
        assert_eq!(LoopFilterParams::derive(40, 0, true).hev_threshold, 2);
        assert_eq!(LoopFilterParams::derive(39, 0, true).hev_threshold, 1);
        assert_eq!(LoopFilterParams::derive(15, 0, true).hev_threshold, 1);
        assert_eq!(LoopFilterParams::derive(14, 0, true).hev_threshold, 0);
    }

    #[test]
    fn derive_hev_threshold_interframe_ladder() {
        assert_eq!(LoopFilterParams::derive(45, 0, false).hev_threshold, 3);
        assert_eq!(LoopFilterParams::derive(40, 0, false).hev_threshold, 3);
        assert_eq!(LoopFilterParams::derive(39, 0, false).hev_threshold, 2);
        assert_eq!(LoopFilterParams::derive(20, 0, false).hev_threshold, 2);
        assert_eq!(LoopFilterParams::derive(19, 0, false).hev_threshold, 1);
        assert_eq!(LoopFilterParams::derive(15, 0, false).hev_threshold, 1);
        assert_eq!(LoopFilterParams::derive(14, 0, false).hev_threshold, 0);
    }

    #[test]
    fn derive_edge_limits_formula() {
        // level 30, sharpness 0 -> interior_limit 30.
        // mbedge = (30 + 2) * 2 + 30 = 94; sub_bedge = 30 * 2 + 30 = 90.
        let p = LoopFilterParams::derive(30, 0, true);
        assert_eq!(p.interior_limit, 30);
        assert_eq!(p.mbedge_limit, 94);
        assert_eq!(p.sub_bedge_limit, 90);
    }

    #[test]
    fn derive_edge_limits_max_level_fits_u8() {
        // level 63, sharpness 0 -> interior_limit 63.
        // mbedge = (63 + 2) * 2 + 63 = 193; sub_bedge = 126 + 63 = 189.
        let p = LoopFilterParams::derive(63, 0, true);
        assert_eq!(p.mbedge_limit, 193);
        assert_eq!(p.sub_bedge_limit, 189);
    }

    // ----- §15.2 simple filter -----------------------------------------

    #[test]
    fn simple_segment_skips_when_metric_exceeds_limit() {
        // p1 p0 | q0 q1 with a large p0/q0 jump; tiny limit disables.
        let mut seg = [10u8, 10, 200, 200];
        let before = seg;
        simple_segment(1, &mut seg, 0);
        assert_eq!(seg, before, "filter must not run when metric > limit");
    }

    #[test]
    fn simple_segment_adjusts_small_step() {
        // A modest edge difference under a generous limit gets shaved.
        let mut seg = [100u8, 100, 116, 116];
        simple_segment(255, &mut seg, 0);
        // p0 should rise toward q0, q0 should fall toward p0; the
        // adjustment preserves their ordering and pulls them together.
        assert!(seg[1] >= 100, "p0 moved toward q0");
        assert!(seg[2] <= 116, "q0 moved toward p0");
        assert!(seg[2] - seg[1] < 16, "gap reduced");
        // Outer taps untouched by the simple filter.
        assert_eq!(seg[0], 100);
        assert_eq!(seg[3], 116);
    }

    #[test]
    fn common_adjust_hand_derived_no_outer_taps() {
        // Re-derive the spec arithmetic for a known input with
        // use_outer_taps = false. p1=100 p0=100 q0=108 q1=108.
        // u2s: p0 = -28, q0 = -20. a = c(0 + 3*(-20 - -28)) = c(24) = 24.
        // b = c(24 + 3) >> 3 = 27 >> 3 = 3.
        // a = c(24 + 4) >> 3 = 28 >> 3 = 3.
        // Q0 = s2u(q0 - a) = s2u(-20 - 3) = s2u(-23) = 105.
        // P0 = s2u(p0 + b) = s2u(-28 + 3) = s2u(-25) = 103.
        let mut seg = [100u8, 100, 108, 108];
        let ret = common_adjust(false, &mut seg, 0);
        assert_eq!(ret, 3, "returned a");
        assert_eq!(seg[1], 103, "P0");
        assert_eq!(seg[2], 105, "Q0");
        assert_eq!(seg[0], 100, "P1 untouched");
        assert_eq!(seg[3], 108, "Q1 untouched");
    }

    #[test]
    fn common_adjust_outer_taps_changes_result() {
        // Same edge but with outer taps engaged the outer term shifts a.
        // p1=90 q1=120: outer = c(u2s(90) - u2s(120)) = c(-38 - -8)
        // = c(-30) = -30. p0=100 q0=108: a = c(-30 + 3*8) = c(-6) = -6.
        // b = c(-6 + 3) >> 3 = (-3) >> 3 = -1.
        // a = c(-6 + 4) >> 3 = (-2) >> 3 = -1.
        // Q0 = s2u(-20 - -1) = s2u(-19) = 109.
        // P0 = s2u(-28 + -1) = s2u(-29) = 99.
        let mut seg = [90u8, 100, 108, 120];
        let ret = common_adjust(true, &mut seg, 0);
        assert_eq!(ret, -1, "returned a");
        assert_eq!(seg[1], 99, "P0");
        assert_eq!(seg[2], 109, "Q0");
    }

    // ----- §15.3 normal filter -----------------------------------------

    #[test]
    fn subblock_filter_skips_when_interior_diff_too_large() {
        // A large p3/p2 step exceeds interior_limit -> filter_yes false.
        let mut seg = [10u8, 200, 100, 100, 108, 108, 108, 108];
        let before = seg;
        subblock_filter(0, 5, 255, &mut seg, 0);
        assert_eq!(seg, before, "interior difference disables filter");
    }

    #[test]
    fn subblock_filter_low_hev_adjusts_inner_pixels() {
        // Smooth interior, modest edge step, hev_threshold high so the
        // edge is treated as low-variance -> p1/q1 also get adjusted.
        let mut seg = [100u8, 100, 100, 100, 110, 110, 110, 110];
        let before = seg;
        subblock_filter(50, 100, 255, &mut seg, 0);
        // p0 (idx 3) and q0 (idx 4) move together.
        assert!(seg[3] > before[3], "p0 raised");
        assert!(seg[4] < before[4], "q0 lowered");
        // Low hev path also nudges p1 (idx 2) and q1 (idx 5).
        assert!(seg[2] >= before[2], "p1 nudged up");
        assert!(seg[5] <= before[5], "q1 nudged down");
    }

    #[test]
    fn subblock_filter_high_hev_leaves_inner_pixels() {
        // hev_threshold 0 with a steep p1->p0 inner gradient -> high edge
        // variance (|p1 - p0| = 10 > 0), so the inner p1/q1 adjustment is
        // skipped and only common_adjust runs on p0/q0.
        let mut seg = [100u8, 100, 100, 110, 120, 120, 120, 120];
        let before = seg;
        subblock_filter(0, 100, 255, &mut seg, 0);
        // p1/q1 (idx 2/5) untouched in the high-variance branch.
        assert_eq!(seg[2], before[2], "p1 untouched under hev");
        assert_eq!(seg[5], before[5], "q1 untouched under hev");
        // Edge pixels (idx 3/4) still adjusted by common_adjust.
        assert!(seg[3] > before[3]);
        assert!(seg[4] < before[4]);
    }

    #[test]
    fn mb_filter_skips_when_disabled() {
        let mut seg = [10u8, 200, 100, 100, 108, 108, 108, 108];
        let before = seg;
        mb_filter(0, 5, 255, &mut seg, 0);
        assert_eq!(seg, before, "interior difference disables MB filter");
    }

    #[test]
    fn mb_filter_low_hev_modifies_six_pixels() {
        // Low-variance edge: the wider filter touches p2,p1,p0,q0,q1,q2.
        let mut seg = [100u8, 100, 100, 100, 112, 112, 112, 112];
        let before = seg;
        mb_filter(50, 100, 255, &mut seg, 0);
        // Six interior pixels (idx 1..=6) all move; p3/q3 untouched.
        assert!(seg[3] > before[3], "p0");
        assert!(seg[4] < before[4], "q0");
        assert!(seg[2] >= before[2], "p1");
        assert!(seg[5] <= before[5], "q1");
        assert!(seg[1] >= before[1], "p2");
        assert!(seg[6] <= before[6], "q2");
        assert_eq!(seg[0], before[0], "p3 untouched");
        assert_eq!(seg[7], before[7], "q3 untouched");
    }

    #[test]
    fn mb_filter_high_hev_falls_back_to_common_adjust() {
        // High-variance edge (steep p1->p0 inner gradient, |p1 - p0| = 10
        // > 0 against hev_threshold 0): only the inner 4-pixel window is
        // adjusted (common_adjust with outer taps); p2/q2 stay put.
        let mut seg = [100u8, 100, 100, 110, 120, 120, 120, 120];
        let before = seg;
        mb_filter(0, 100, 255, &mut seg, 0);
        assert_eq!(seg[1], before[1], "p2 untouched under hev");
        assert_eq!(seg[6], before[6], "q2 untouched under hev");
        // Inner edge pixels (idx 3/4) adjusted by common_adjust.
        assert!(seg[3] > before[3]);
        assert!(seg[4] < before[4]);
    }

    #[test]
    fn mb_filter_low_hev_hand_derived() {
        // Re-derive the §15.3 MBfilter arithmetic for an exact match.
        // Segment (unsigned): 100 100 100 100 | 116 116 116 116.
        // signed: p3..p0 = -28; q0..q3 = -12.
        // hev: |p1 - p0| = 0, |q1 - q0| = 0 -> 0 > threshold(50)? no.
        //   So low-variance branch.
        // w = c(c(p1 - q1) + 3*(q0 - p0))
        //   = c(c(-28 - -12) + 3*(-12 - -28))
        //   = c(c(-16) + 3*16) = c(-16 + 48) = c(32) = 32.
        // a0 = c((27*32 + 63) >> 7) = c((864 + 63) >> 7) = c(927 >> 7)
        //    = c(7) = 7.  Q0 = s2u(-12 - 7) = s2u(-19) = 109;
        //    P0 = s2u(-28 + 7) = s2u(-21) = 107.
        // a1 = c((18*32 + 63) >> 7) = c((576 + 63) >> 7) = c(639 >> 7)
        //    = c(4) = 4.  Q1 = s2u(-12 - 4) = 112; P1 = s2u(-28 + 4) = 104.
        // a2 = c((9*32 + 63) >> 7) = c((288 + 63) >> 7) = c(351 >> 7)
        //    = c(2) = 2.  Q2 = s2u(-12 - 2) = 114; P2 = s2u(-28 + 2) = 102.
        let mut seg = [100u8, 100, 100, 100, 116, 116, 116, 116];
        mb_filter(50, 100, 255, &mut seg, 0);
        assert_eq!(seg, [100, 102, 104, 107, 109, 112, 114, 116]);
    }

    #[test]
    fn filter_routines_respect_base_offset() {
        // The same filtering applied at a non-zero base must leave the
        // surrounding buffer untouched and operate on the window.
        let mut seg = vec![7u8; 12];
        seg[2..10].copy_from_slice(&[100, 100, 100, 100, 116, 116, 116, 116]);
        mb_filter(50, 100, 255, &mut seg, 2);
        // Guard bytes intact.
        assert_eq!(&seg[0..2], &[7, 7]);
        assert_eq!(&seg[10..12], &[7, 7]);
        // Window matches the hand-derived result above.
        assert_eq!(&seg[2..10], &[100, 102, 104, 107, 109, 112, 114, 116]);
    }
}
