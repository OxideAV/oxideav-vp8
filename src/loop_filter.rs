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
//! # §15.1 frame geometry (also in this module)
//!
//! * The §15.1 macroblock-by-macroblock filter *geometry* —
//!   [`filter_frame`] walks the reconstructed [`crate::KeyframePlanes`]
//!   in raster order, gathers the 16 luma / 8 chroma segments straddling
//!   each of the four edges per macroblock, honours the order of the four
//!   filtering steps, and applies the §15.1 page 86 skip rule (steps 2
//!   and 4 skipped when the macroblock is neither `B_PRED` nor `SPLITMV`
//!   and has no coded coefficient) plus the §15 page 84 level-0 whole-MB
//!   skip. It calls the per-segment primitives above.
//! * The §9.4 / §10 derivation of the per-macroblock `loop_filter_level`
//!   itself — [`calculate_mb_filter_level`] implements the §20.6
//!   `calculate_filter_parameters` body (segment override + the
//!   reference-frame / prediction-mode deltas).
//!   [`LoopFilterParams::derive`] still takes an already-resolved level
//!   for callers wanting just the §15.4 limit derivation;
//!   [`FrameFilterConfig::keyframe`] resolves the level inputs from a
//!   [`crate::Vp8CodedHeader`].

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

// ===================================================================
// §15.1 / §15.2 frame loop-filter geometry
// ===================================================================
//
// This section implements the §15.1 "Filter Geometry and Overall
// Procedure" — the per-frame pass that walks the reconstructed plane
// buffers in raster-scan order and applies the §15.2 simple or §15.3
// normal filter kernels above to the edges between adjacent macroblocks
// and the edges between adjacent subblocks.
//
// The arithmetic that derives the per-macroblock filter level from the
// base `loop_filter_level`, the §10 per-segment override, and the §9.4
// reference / mode deltas is the §20.6 `calculate_filter_parameters`
// body of the RFC's reference-decoder annex (`dixie.c`, part of RFC 6386
// itself). The four-step per-macroblock edge iteration — left
// inter-MB edge, three internal vertical subblock edges, top inter-MB
// edge, three internal horizontal subblock edges, with chroma analogues
// — is the §20.6 `filter_row_normal` / `filter_row_simple` geometry.
//
// `LoopFilterParams::derive` above already returns the §15.4 limits in
// the form the kernels consume: its `mbedge_limit` / `sub_bedge_limit`
// equal the `2 * E + I` disabling metric the §20.6 `normal_threshold`
// (and the §20.6 simple `mb_limit` / `b_limit`) compute, so the
// geometry pass passes those precomputed values straight into
// `mb_filter` / `subblock_filter` / `simple_segment` as their
// `edge_limit` argument.

use crate::frame::{KeyframePlanes, MbCoeffs};
use crate::macroblock::{IntraYMode, MacroblockModes};

/// Number of reference-frame loop-filter deltas (`MAX_REF_LF_DELTAS`,
/// RFC 6386 §9.4) — one per reference-frame slot.
pub const MAX_REF_LF_DELTAS: usize = 4;
/// Number of prediction-mode loop-filter deltas (`MAX_MODE_LF_DELTAS`,
/// RFC 6386 §9.4).
pub const MAX_MODE_LF_DELTAS: usize = 4;
/// Number of segments (`MAX_MB_SEGMENTS`, RFC 6386 §10).
pub const MAX_MB_SEGMENTS: usize = 4;

/// Frame-constant configuration controlling the §15 loop-filter pass.
///
/// Holds the resolved (post-header) state the §20.6
/// `calculate_filter_parameters` body reads: the base
/// `loop_filter_level` and `sharpness_level`, the §10 per-segment level
/// override config, and the §9.4 reference / mode delta config. Build it
/// with [`FrameFilterConfig::keyframe`] from a parsed
/// [`crate::Vp8CodedHeader`], or construct directly for unit testing.
///
/// On a key frame every macroblock predicts from the current frame, so
/// of the four §9.4 mode deltas only `mode_delta[0]` (the `B_PRED`
/// delta) ever applies, and the reference delta is always the
/// current-frame slot. The geometry pass below therefore models exactly
/// those two contributions; the remaining inter-frame mode/reference
/// deltas are reserved for the §16 inter-prediction round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameFilterConfig {
    /// `filter_type` — `false` selects the §15.3 normal filter, `true`
    /// the §15.2 simple filter.
    pub simple: bool,
    /// Whether the current frame is a key frame (selects the §15.4
    /// key-frame `hev_threshold` ladder).
    pub key_frame: bool,
    /// Base frame `loop_filter_level` (`0..=63`, RFC 6386 §9.4).
    pub loop_filter_level: u8,
    /// Frame `sharpness_level` (`0..=7`, RFC 6386 §9.4).
    pub sharpness_level: u8,
    /// `segmentation_enabled` (RFC 6386 §9.3) — gates the §10
    /// per-segment loop-filter override.
    pub segmentation_enabled: bool,
    /// `segment_feature_mode` (RFC 6386 §9.3 4.a): `true` = absolute
    /// (the per-segment value *replaces* the base level), `false` =
    /// delta (it is *added* to the base level). Only consulted when
    /// `segmentation_enabled`.
    pub segment_abs: bool,
    /// The four §10 per-segment loop-filter levels (absolute values or
    /// signed deltas per `segment_abs`). Only consulted when
    /// `segmentation_enabled`.
    pub segment_lf_level: [i16; MAX_MB_SEGMENTS],
    /// `loop_filter_adj_enable` (RFC 6386 §9.4) — gates the per-MB
    /// reference / mode delta adjustment.
    pub delta_enabled: bool,
    /// The reference-frame delta applied for the current (intra) frame —
    /// `ref_delta[CURRENT_FRAME]` in §20.6. Only consulted when
    /// `delta_enabled`.
    pub ref_delta_current: i16,
    /// The `B_PRED` mode delta — `mode_delta[0]` in §20.6. Only
    /// consulted when `delta_enabled` and the macroblock is `B_PRED`.
    pub bpred_mode_delta: i16,
}

impl FrameFilterConfig {
    /// Build the key-frame configuration from a parsed boolean-coded
    /// frame header.
    ///
    /// Resolves the §10 per-segment loop-filter levels from the header's
    /// `update_segmentation.loop_filter_update` (a `None` entry means the
    /// segment carries level 0 — the §9.3 "value is 0" path) and the
    /// §9.4 reference / mode deltas from `mb_lf_adjustments`. Because a
    /// key frame has no prior persisted delta state (the §9.4 deltas are
    /// "updated for the current frame" and key frames begin a fresh
    /// sequence), a `None` update entry resolves to 0.
    ///
    /// The reference delta used is the current-frame slot
    /// (`ref_frame_delta_update[0]`, the `CURRENT_FRAME` reference per
    /// §20.6's `ref_delta[CURRENT_FRAME]`) and the only mode delta that
    /// applies on a key frame is the `B_PRED` slot
    /// (`mb_mode_delta_update[0]`, `mode_delta[0]` in §20.6).
    pub fn keyframe(header: &crate::Vp8CodedHeader) -> Self {
        let mut segment_lf_level = [0i16; MAX_MB_SEGMENTS];
        let mut segment_abs = false;
        if header.segmentation_enabled {
            if let Some(seg) = header.update_segmentation {
                segment_abs = seg.segment_feature_mode_absolute;
                for (dst, src) in segment_lf_level
                    .iter_mut()
                    .zip(seg.loop_filter_update.iter())
                {
                    *dst = src.unwrap_or(0);
                }
            }
        }

        let lf = &header.mb_lf_adjustments;
        let (ref_delta_current, bpred_mode_delta) = if lf.loop_filter_adj_enable {
            (
                lf.ref_frame_delta_update[0].unwrap_or(0),
                lf.mb_mode_delta_update[0].unwrap_or(0),
            )
        } else {
            (0, 0)
        };

        FrameFilterConfig {
            simple: header.filter_type,
            key_frame: true,
            loop_filter_level: header.loop_filter_level,
            sharpness_level: header.sharpness_level,
            segmentation_enabled: header.segmentation_enabled,
            segment_abs,
            segment_lf_level,
            delta_enabled: lf.loop_filter_adj_enable,
            ref_delta_current,
            bpred_mode_delta,
        }
    }
}

/// Resolve the per-macroblock loop-filter level per the §20.6
/// `calculate_filter_parameters` body: start from the frame base,
/// apply the §10 per-segment override (delta or absolute), clamp to
/// `0..=63`, apply the §9.4 reference + mode deltas, then clamp again.
///
/// `segment_id` is the macroblock's §10 segment (`None` resolves to
/// segment 0, the §10 default when the map was not updated). `y_mode`
/// selects the §9.4 `B_PRED` mode delta. Returns the resolved level in
/// `0..=63`; the caller skips filtering entirely when it is 0 (RFC 6386
/// §15 page 84).
pub fn calculate_mb_filter_level(
    config: &FrameFilterConfig,
    segment_id: Option<u8>,
    y_mode: IntraYMode,
) -> u8 {
    // §20.6: filter_level = loopfilter_hdr.level.
    let mut level = config.loop_filter_level as i32;

    // §20.6 + §10: per-segment override.
    if config.segmentation_enabled {
        let seg = segment_id.unwrap_or(0) as usize;
        let seg = seg.min(MAX_MB_SEGMENTS - 1);
        if config.segment_abs {
            level = config.segment_lf_level[seg] as i32;
        } else {
            level += config.segment_lf_level[seg] as i32;
        }
    }

    // §20.6: clamp to 0..=63 after the segment override.
    level = level.clamp(0, 63);

    // §20.6 + §9.4: reference / mode deltas. On a key frame every MB is
    // CURRENT_FRAME, so the reference delta always applies and only the
    // B_PRED mode delta participates.
    if config.delta_enabled {
        level += config.ref_delta_current as i32;
        if y_mode == IntraYMode::B {
            level += config.bpred_mode_delta as i32;
        }
    }

    // §20.6: clamp to 0..=63 after the deltas.
    level.clamp(0, 63) as u8
}

/// Whether a macroblock has any coded (non-zero) DCT coefficient.
///
/// The §20.6 step-2 / step-4 skip rule (RFC 6386 §15.1 page 86) is
/// "neither `B_PRED` nor `SPLITMV` *and* no DCT coefficient coded for
/// the whole macroblock". The §20.6 annex notes this conditional is
/// "actually dependent on the number of coefficients decoded, not the
/// skip flag as coded in the bitstream" — so we examine the
/// dequantized coefficient bundle directly rather than trusting
/// `mb_skip_coeff`.
fn mb_has_coeffs(coeffs: &MbCoeffs) -> bool {
    coeffs.y2.iter().any(|&c| c != 0)
        || coeffs.y.iter().any(|b| b.iter().any(|&c| c != 0))
        || coeffs.u.iter().any(|b| b.iter().any(|&c| c != 0))
        || coeffs.v.iter().any(|b| b.iter().any(|&c| c != 0))
}

/// Apply the normal §15.3 filter to one vertical edge of a plane.
///
/// `x` is the column at the edge (`p`-side is `x - 1`, `q`-side is `x`);
/// `y0` is the first row of the edge, `len` the number of rows (16 for
/// luma, 8 for chroma). `mb_edge` selects the wide [`mb_filter`] for
/// inter-macroblock edges vs. the [`subblock_filter`] for inter-subblock
/// edges.
#[allow(clippy::too_many_arguments)]
fn filter_v_edge_normal(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y0: usize,
    len: usize,
    params: &LoopFilterParams,
    mb_edge: bool,
) {
    let edge_limit = if mb_edge {
        params.mbedge_limit
    } else {
        params.sub_bedge_limit
    };
    let mut seg = [0u8; 8];
    for r in 0..len {
        let row = (y0 + r) * stride;
        // Gather the 8-pixel segment p3 p2 p1 p0 | q0 q1 q2 q3 centred on
        // the vertical edge at column `x` (p-side is to the left).
        for (k, slot) in seg.iter_mut().enumerate() {
            *slot = plane[row + x - 4 + k];
        }
        if mb_edge {
            mb_filter(
                params.hev_threshold,
                params.interior_limit,
                edge_limit,
                &mut seg,
                0,
            );
        } else {
            subblock_filter(
                params.hev_threshold,
                params.interior_limit,
                edge_limit,
                &mut seg,
                0,
            );
        }
        for (k, &val) in seg.iter().enumerate() {
            plane[row + x - 4 + k] = val;
        }
    }
}

/// Apply the normal §15.3 filter to one horizontal edge of a plane.
///
/// `y` is the row at the edge (`p`-side is `y - 1`, `q`-side is `y`);
/// `x0` is the first column, `len` the number of columns.
#[allow(clippy::too_many_arguments)]
fn filter_h_edge_normal(
    plane: &mut [u8],
    stride: usize,
    y: usize,
    x0: usize,
    len: usize,
    params: &LoopFilterParams,
    mb_edge: bool,
) {
    let edge_limit = if mb_edge {
        params.mbedge_limit
    } else {
        params.sub_bedge_limit
    };
    let mut seg = [0u8; 8];
    for c in 0..len {
        let col = x0 + c;
        // Gather the 8-pixel segment straddling the horizontal edge at
        // row `y` (p-side is above).
        for (k, slot) in seg.iter_mut().enumerate() {
            *slot = plane[(y - 4 + k) * stride + col];
        }
        if mb_edge {
            mb_filter(
                params.hev_threshold,
                params.interior_limit,
                edge_limit,
                &mut seg,
                0,
            );
        } else {
            subblock_filter(
                params.hev_threshold,
                params.interior_limit,
                edge_limit,
                &mut seg,
                0,
            );
        }
        for (k, &val) in seg.iter().enumerate() {
            plane[(y - 4 + k) * stride + col] = val;
        }
    }
}

/// Apply the §15.2 simple filter to one vertical edge (luma only).
///
/// The simple filter examines a 4-pixel window `p1 p0 | q0 q1`; the
/// `edge_limit` is the precomputed `mbedge_limit` / `sub_bedge_limit`.
fn filter_v_edge_simple(plane: &mut [u8], stride: usize, x: usize, y0: usize, edge_limit: u8) {
    let mut seg = [0u8; 4];
    for r in 0..16 {
        let row = (y0 + r) * stride;
        for (k, slot) in seg.iter_mut().enumerate() {
            *slot = plane[row + x - 2 + k];
        }
        simple_segment(edge_limit, &mut seg, 0);
        for (k, &val) in seg.iter().enumerate() {
            plane[row + x - 2 + k] = val;
        }
    }
}

/// Apply the §15.2 simple filter to one horizontal edge (luma only).
fn filter_h_edge_simple(plane: &mut [u8], stride: usize, y: usize, x0: usize, edge_limit: u8) {
    let mut seg = [0u8; 4];
    for c in 0..16 {
        let col = x0 + c;
        for (k, slot) in seg.iter_mut().enumerate() {
            *slot = plane[(y - 2 + k) * stride + col];
        }
        simple_segment(edge_limit, &mut seg, 0);
        for (k, &val) in seg.iter().enumerate() {
            plane[(y - 2 + k) * stride + col] = val;
        }
    }
}

/// Run the §15 loop filter over a reconstructed key frame in place.
///
/// This is the §15.1 frame-geometry pass: it walks the macroblocks of
/// `planes` in raster-scan order and, for each, applies the four §15.1
/// filtering steps using the kernels above —
///
/// 1. the left (vertical) inter-macroblock edge (skipped on the leftmost
///    column),
/// 2. the three internal vertical subblock edges,
/// 3. the top (horizontal) inter-macroblock edge (skipped on the topmost
///    row),
/// 4. the three internal horizontal subblock edges,
///
/// with the chroma analogues for the normal filter (the §15.2 simple
/// filter touches only luma). Steps 2 and 4 are skipped for a macroblock
/// whose coding mode is neither `B_PRED` nor `SPLITMV` *and* which has no
/// coded coefficient (RFC 6386 §15.1 page 86). The whole macroblock is
/// skipped when its resolved §20.6 filter level is 0 (RFC 6386 §15
/// page 84). `modes` and `coeffs` must each have `mb_cols * mb_rows`
/// entries in the same raster order as [`crate::decode_keyframe`].
///
/// The §20.6 ordering "must be respected" (RFC 6386 §15.1 page 86): many
/// pixels straddle two or more edges and are filtered more than once, so
/// the raster MB order and the within-MB step order are both load-bearing.
pub fn filter_frame(
    planes: &mut KeyframePlanes,
    modes: &[MacroblockModes],
    coeffs: &[MbCoeffs],
    config: &FrameFilterConfig,
) {
    let mb_cols = planes.mb_cols;
    let mb_rows = planes.mb_rows;
    if mb_cols == 0 || mb_rows == 0 {
        return;
    }
    let expected = mb_cols * mb_rows;
    if modes.len() != expected || coeffs.len() != expected {
        return;
    }

    let ys = planes.y_stride;
    let cs = planes.uv_stride;

    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            let index = mb_row * mb_cols + mb_col;
            let mb = &modes[index];

            let level = calculate_mb_filter_level(config, mb.segment_id, mb.y_mode);
            // §15 page 84: skip the macroblock entirely when level is 0.
            if level == 0 {
                continue;
            }
            let params = LoopFilterParams::derive(level, config.sharpness_level, config.key_frame);

            // §15.1 page 86: steps 2 and 4 run when the MB is B_PRED /
            // SPLITMV (no SPLITMV on key frames) OR has any coded coeff.
            let filter_subblocks = mb.y_mode == IntraYMode::B || mb_has_coeffs(&coeffs[index]);

            let y_x0 = mb_col * 16;
            let y_y0 = mb_row * 16;
            let uv_x0 = mb_col * 8;
            let uv_y0 = mb_row * 8;

            if config.simple {
                // §20.6 filter_row_simple — luma only.
                // Step 1: left inter-MB edge.
                if mb_col > 0 {
                    filter_v_edge_simple(&mut planes.y, ys, y_x0, y_y0, params.mbedge_limit);
                }
                // Step 2: internal vertical subblock edges (1/4, 1/2, 3/4).
                if filter_subblocks {
                    for q in 1..4 {
                        filter_v_edge_simple(
                            &mut planes.y,
                            ys,
                            y_x0 + 4 * q,
                            y_y0,
                            params.sub_bedge_limit,
                        );
                    }
                }
                // Step 3: top inter-MB edge.
                if mb_row > 0 {
                    filter_h_edge_simple(&mut planes.y, ys, y_y0, y_x0, params.mbedge_limit);
                }
                // Step 4: internal horizontal subblock edges.
                if filter_subblocks {
                    for q in 1..4 {
                        filter_h_edge_simple(
                            &mut planes.y,
                            ys,
                            y_y0 + 4 * q,
                            y_x0,
                            params.sub_bedge_limit,
                        );
                    }
                }
            } else {
                // §20.6 filter_row_normal — luma + both chroma planes.
                // Step 1: left inter-MB edges.
                if mb_col > 0 {
                    filter_v_edge_normal(&mut planes.y, ys, y_x0, y_y0, 16, &params, true);
                    filter_v_edge_normal(&mut planes.u, cs, uv_x0, uv_y0, 8, &params, true);
                    filter_v_edge_normal(&mut planes.v, cs, uv_x0, uv_y0, 8, &params, true);
                }
                // Step 2: internal vertical subblock edges. Luma has
                // three (1/4, 1/2, 3/4); chroma has one (centre, 1/2).
                if filter_subblocks {
                    for q in 1..4 {
                        filter_v_edge_normal(
                            &mut planes.y,
                            ys,
                            y_x0 + 4 * q,
                            y_y0,
                            16,
                            &params,
                            false,
                        );
                    }
                    filter_v_edge_normal(&mut planes.u, cs, uv_x0 + 4, uv_y0, 8, &params, false);
                    filter_v_edge_normal(&mut planes.v, cs, uv_x0 + 4, uv_y0, 8, &params, false);
                }
                // Step 3: top inter-MB edges.
                if mb_row > 0 {
                    filter_h_edge_normal(&mut planes.y, ys, y_y0, y_x0, 16, &params, true);
                    filter_h_edge_normal(&mut planes.u, cs, uv_y0, uv_x0, 8, &params, true);
                    filter_h_edge_normal(&mut planes.v, cs, uv_y0, uv_x0, 8, &params, true);
                }
                // Step 4: internal horizontal subblock edges.
                if filter_subblocks {
                    for q in 1..4 {
                        filter_h_edge_normal(
                            &mut planes.y,
                            ys,
                            y_y0 + 4 * q,
                            y_x0,
                            16,
                            &params,
                            false,
                        );
                    }
                    filter_h_edge_normal(&mut planes.u, cs, uv_y0 + 4, uv_x0, 8, &params, false);
                    filter_h_edge_normal(&mut planes.v, cs, uv_y0 + 4, uv_x0, 8, &params, false);
                }
            }
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

    // ----- §15.1 / §20.6 per-MB filter-level derivation ----------------
    // KeyframePlanes, MbCoeffs, IntraYMode and MacroblockModes come in via
    // `use super::*`; IntraUvMode is only needed by the test helpers.
    use crate::macroblock::IntraUvMode;

    fn base_config() -> FrameFilterConfig {
        FrameFilterConfig {
            simple: false,
            key_frame: true,
            loop_filter_level: 20,
            sharpness_level: 0,
            segmentation_enabled: false,
            segment_abs: false,
            segment_lf_level: [0; MAX_MB_SEGMENTS],
            delta_enabled: false,
            ref_delta_current: 0,
            bpred_mode_delta: 0,
        }
    }

    fn flat_planes(value: u8, mb_cols: usize, mb_rows: usize) -> KeyframePlanes {
        let y_stride = mb_cols * 16;
        let uv_stride = mb_cols * 8;
        KeyframePlanes {
            y: vec![value; y_stride * mb_rows * 16],
            u: vec![value; uv_stride * mb_rows * 8],
            v: vec![value; uv_stride * mb_rows * 8],
            y_stride,
            uv_stride,
            mb_cols,
            mb_rows,
        }
    }

    fn mode(y_mode: IntraYMode, segment_id: Option<u8>) -> MacroblockModes {
        MacroblockModes {
            segment_id,
            mb_skip_coeff: true,
            y_mode,
            subblock_modes: None,
            uv_mode: IntraUvMode::Dc,
        }
    }

    #[test]
    fn mb_filter_level_no_overrides_is_base() {
        let c = base_config();
        assert_eq!(calculate_mb_filter_level(&c, None, IntraYMode::Dc), 20);
        // segment_id ignored when segmentation disabled.
        assert_eq!(calculate_mb_filter_level(&c, Some(3), IntraYMode::Dc), 20);
    }

    #[test]
    fn mb_filter_level_segment_delta_adds() {
        let mut c = base_config();
        c.segmentation_enabled = true;
        c.segment_abs = false;
        c.segment_lf_level = [5, -8, 0, 0];
        // segment 0: 20 + 5 = 25.
        assert_eq!(calculate_mb_filter_level(&c, Some(0), IntraYMode::Dc), 25);
        // segment 1: 20 + (-8) = 12.
        assert_eq!(calculate_mb_filter_level(&c, Some(1), IntraYMode::Dc), 12);
    }

    #[test]
    fn mb_filter_level_segment_absolute_replaces() {
        let mut c = base_config();
        c.segmentation_enabled = true;
        c.segment_abs = true;
        c.segment_lf_level = [40, 0, 63, 7];
        // absolute mode replaces the base entirely.
        assert_eq!(calculate_mb_filter_level(&c, Some(0), IntraYMode::Dc), 40);
        assert_eq!(calculate_mb_filter_level(&c, Some(2), IntraYMode::Dc), 63);
        // absolute 0 -> level 0 (caller then skips the MB).
        assert_eq!(calculate_mb_filter_level(&c, Some(1), IntraYMode::Dc), 0);
    }

    #[test]
    fn mb_filter_level_segment_clamps_before_deltas() {
        // §20.6 clamps to 0..=63 after the segment override and again
        // after the deltas. A segment delta that overshoots saturates.
        let mut c = base_config();
        c.segmentation_enabled = true;
        c.segment_abs = false;
        c.segment_lf_level = [60, -40, 0, 0];
        // 20 + 60 = 80 -> clamp 63.
        assert_eq!(calculate_mb_filter_level(&c, Some(0), IntraYMode::Dc), 63);
        // 20 - 40 = -20 -> clamp 0.
        assert_eq!(calculate_mb_filter_level(&c, Some(1), IntraYMode::Dc), 0);
    }

    #[test]
    fn mb_filter_level_ref_and_bpred_mode_deltas() {
        let mut c = base_config();
        c.delta_enabled = true;
        c.ref_delta_current = 4;
        c.bpred_mode_delta = -3;
        // Non-B_PRED: only the reference delta applies. 20 + 4 = 24.
        assert_eq!(calculate_mb_filter_level(&c, None, IntraYMode::Dc), 24);
        // B_PRED: ref + mode. 20 + 4 - 3 = 21.
        assert_eq!(calculate_mb_filter_level(&c, None, IntraYMode::B), 21);
    }

    #[test]
    fn mb_filter_level_deltas_clamp_at_ends() {
        let mut c = base_config();
        c.loop_filter_level = 62;
        c.delta_enabled = true;
        c.ref_delta_current = 10; // 62 + 10 = 72 -> clamp 63.
        assert_eq!(calculate_mb_filter_level(&c, None, IntraYMode::Dc), 63);
        c.loop_filter_level = 2;
        c.ref_delta_current = -10; // 2 - 10 = -8 -> clamp 0.
        assert_eq!(calculate_mb_filter_level(&c, None, IntraYMode::Dc), 0);
    }

    #[test]
    fn mb_filter_level_deltas_disabled_when_flag_off() {
        let mut c = base_config();
        c.delta_enabled = false;
        c.ref_delta_current = 30;
        c.bpred_mode_delta = 30;
        // Both deltas ignored.
        assert_eq!(calculate_mb_filter_level(&c, None, IntraYMode::B), 20);
    }

    // ----- §15.1 frame geometry ----------------------------------------

    #[test]
    fn filter_frame_level_zero_is_noop() {
        let mut c = base_config();
        c.loop_filter_level = 0;
        // Two MBs with a sharp boundary; level 0 must leave them alone.
        let mut planes = flat_planes(100, 2, 1);
        for px in planes.y[16..32].iter_mut() {
            *px = 200;
        }
        let before = planes.clone();
        let modes = vec![mode(IntraYMode::Dc, None), mode(IntraYMode::Dc, None)];
        let coeffs = vec![MbCoeffs::default(); 2];
        filter_frame(&mut planes, &modes, &coeffs, &c);
        assert_eq!(planes, before, "level 0 must skip filtering entirely");
    }

    #[test]
    fn filter_frame_normal_mb_edge_hand_derived() {
        // A 2×1 luma frame: MB(0,0) flat 100, MB(0,1) flat 110. Skip
        // coeffs so the subblock steps are skipped, isolating MB(0,1)'s
        // left inter-MB vertical edge (frame column 16). The normal MB
        // filter's wide low-variance branch rewrites p2,p1,p0,q0,q1,q2
        // (columns 13..=18) to the hand-derived values; everything else
        // is untouched.
        //
        // level 20, sharpness 0, key frame:
        //   interior_limit = 20, mbedge_limit = (20+2)*2+20 = 64,
        //   hev_threshold = 1 (level >= 15).
        // signed p* = -28, q* = -18; |p1-p0| = |q1-q0| = 0 -> low var.
        // w = c(c(-28 - -18) + 3*(-18 - -28)) = c(c(-10) + 30) = c(20) = 20.
        // a0 = c((27*20+63)>>7) = c(4) = 4: P0=104, Q0=106.
        // a1 = c((18*20+63)>>7) = c(3) = 3: P1=105, Q1=107.
        // a2 = c((9*20+63)>>7)  = c(1) = 1: P2=103, Q2=109.
        let c = base_config();
        let mut planes = flat_planes(100, 2, 1);
        for r in 0..16 {
            for px in planes.y[r * 32 + 16..r * 32 + 32].iter_mut() {
                *px = 110;
            }
        }
        let before = planes.clone();
        let modes = vec![mode(IntraYMode::Dc, None), mode(IntraYMode::Dc, None)];
        let coeffs = vec![MbCoeffs::default(); 2]; // no coeffs -> steps 2/4 skipped
        filter_frame(&mut planes, &modes, &coeffs, &c);

        // The six pixels straddling column 16 on every row are rewritten.
        // Columns 13,14,15 = P2,P1,P0; columns 16,17,18 = Q0,Q1,Q2.
        for r in 0..16 {
            let row = r * 32;
            assert_eq!(planes.y[row + 13], 101, "P2 row {r}");
            assert_eq!(planes.y[row + 14], 103, "P1 row {r}");
            assert_eq!(planes.y[row + 15], 104, "P0 row {r}");
            assert_eq!(planes.y[row + 16], 106, "Q0 row {r}");
            assert_eq!(planes.y[row + 17], 107, "Q1 row {r}");
            assert_eq!(planes.y[row + 18], 109, "Q2 row {r}");
            // Columns away from the edge are untouched.
            assert_eq!(planes.y[row + 12], before.y[row + 12]);
            assert_eq!(planes.y[row + 19], before.y[row + 19]);
            assert_eq!(planes.y[row], 100);
            assert_eq!(planes.y[row + 31], 110);
        }
        // Chroma planes are flat (both MBs same value) -> unchanged.
        assert_eq!(planes.u, before.u);
        assert_eq!(planes.v, before.v);
    }

    #[test]
    fn filter_frame_leftmost_mb_skips_left_edge() {
        // A single-MB-wide frame stacked 2 rows: MB(0,0) flat 100,
        // MB(1,0) flat 120. There is no left inter-MB edge for either MB
        // (both are on the leftmost column), so step 1 is skipped; the
        // top inter-MB edge of MB(1,0) (frame row 16) is filtered.
        let c = base_config();
        let mut planes = flat_planes(100, 1, 2);
        for px in planes.y[16 * 16..32 * 16].iter_mut() {
            *px = 120;
        }
        // Also flatten chroma rows: MB(1,0) chroma 120 to exercise the
        // horizontal chroma edge.
        for px in planes.u[8 * 8..16 * 8].iter_mut() {
            *px = 120;
        }
        for px in planes.v[8 * 8..16 * 8].iter_mut() {
            *px = 120;
        }
        let before = planes.clone();
        let modes = vec![mode(IntraYMode::Dc, None), mode(IntraYMode::Dc, None)];
        let coeffs = vec![MbCoeffs::default(); 2];
        filter_frame(&mut planes, &modes, &coeffs, &c);

        // No vertical edge is ever filtered (both MBs are leftmost and
        // skip coeffs disable internal subblock edges), so every row's
        // far-from-the-horizontal-edge pixels are untouched: rows 0..=12
        // and 19..=31 stay exactly at their original flat fills. The
        // horizontal MB edge at luma row 16 rewrites rows 13..=18.
        for r in 0..13 {
            for col in 0..16 {
                assert_eq!(
                    planes.y[r * 16 + col],
                    before.y[r * 16 + col],
                    "row {r} col {col} (above horizontal edge influence) untouched"
                );
            }
        }
        for r in 19..32 {
            for col in 0..16 {
                assert_eq!(
                    planes.y[r * 16 + col],
                    before.y[r * 16 + col],
                    "row {r} col {col} (below horizontal edge influence) untouched"
                );
            }
        }
        // The horizontal MB edge at luma row 16 changed the pixels above
        // and below it (including column 0 — horizontal edges span the
        // full MB width, so the leftmost column IS a p/q here).
        assert_ne!(planes.y[15 * 16], before.y[15 * 16], "P-side col0 moved");
        assert_ne!(planes.y[16 * 16], before.y[16 * 16], "Q-side col0 moved");
        // Chroma horizontal edge at chroma row 8 also moved.
        assert_ne!(planes.u[7 * 8 + 3], before.u[7 * 8 + 3]);
        assert_ne!(planes.v[8 * 8 + 3], before.v[8 * 8 + 3]);
    }

    #[test]
    fn filter_frame_simple_touches_only_luma() {
        // Simple filter: a 2×1 frame with a luma step. Only luma is
        // touched; chroma (even if it had a step) is left alone.
        let mut c = base_config();
        c.simple = true;
        let mut planes = flat_planes(100, 2, 1);
        for r in 0..16 {
            for px in planes.y[r * 32 + 16..r * 32 + 32].iter_mut() {
                *px = 110;
            }
        }
        // Introduce a chroma step too — the simple filter must ignore it.
        for r in 0..8 {
            for px in planes.u[r * 16 + 8..r * 16 + 16].iter_mut() {
                *px = 130;
            }
        }
        let before = planes.clone();
        let modes = vec![mode(IntraYMode::Dc, None), mode(IntraYMode::Dc, None)];
        let coeffs = vec![MbCoeffs::default(); 2];
        filter_frame(&mut planes, &modes, &coeffs, &c);

        // Luma changed at the MB edge (column 16).
        assert_ne!(planes.y[15], before.y[15], "luma P0 moved");
        assert_ne!(planes.y[16], before.y[16], "luma Q0 moved");
        // Chroma untouched by the simple filter.
        assert_eq!(planes.u, before.u, "simple filter must not touch U");
        assert_eq!(planes.v, before.v, "simple filter must not touch V");
    }

    #[test]
    fn filter_frame_subblock_steps_gated_by_coeffs() {
        // A single non-skip MB (mode DC) with a synthetic internal luma
        // ramp. With no coded coefficients the internal subblock edges
        // must be skipped (the §15.1 page-86 rule); with a coefficient
        // present they run and modify the interior.
        let c = base_config();
        // Build a luma plane with a step at internal vertical edge x=8
        // (the 1/2 subblock edge), flat elsewhere within the MB.
        let mut planes = flat_planes(100, 1, 1);
        for r in 0..16 {
            for col in 8..16 {
                planes.y[r * 16 + col] = 112;
            }
        }
        let modes = vec![mode(IntraYMode::Dc, None)];

        // No coefficients -> steps 2/4 skipped -> the internal edge at
        // column 8 is NOT filtered.
        let mut p_skip = planes.clone();
        filter_frame(&mut p_skip, &modes, &[MbCoeffs::default()], &c);
        assert_eq!(
            p_skip, planes,
            "no coeffs: internal subblock edges must be skipped (top-left MB has no MB edges either)"
        );

        // A single coded coefficient -> steps 2/4 run -> column 8 edge
        // gets filtered.
        let mut coeffs = MbCoeffs::default();
        coeffs.y[0][1] = 7; // any non-zero coefficient flags coded data
        let mut p_run = planes.clone();
        filter_frame(&mut p_run, &modes, &[coeffs], &c);
        assert_ne!(
            p_run, planes,
            "coded coeff: internal subblock edge must be filtered"
        );
        // The change is localized around column 8.
        assert_ne!(p_run.y[7], planes.y[7], "P0 of x=8 edge moved");
        assert_ne!(p_run.y[8], planes.y[8], "Q0 of x=8 edge moved");
    }

    #[test]
    fn filter_frame_bpred_runs_subblocks_without_coeffs() {
        // A B_PRED MB triggers the subblock steps even with no coded
        // coefficients (mode B_PRED satisfies the §15.1 page-86 rule).
        let c = base_config();
        let mut planes = flat_planes(100, 1, 1);
        for r in 0..16 {
            for col in 8..16 {
                planes.y[r * 16 + col] = 112;
            }
        }
        let modes = vec![mode(IntraYMode::B, None)];
        let mut p = planes.clone();
        filter_frame(&mut p, &modes, &[MbCoeffs::default()], &c);
        assert_ne!(
            p, planes,
            "B_PRED MB must filter internal subblock edges even when skip"
        );
    }

    #[test]
    fn filter_frame_config_keyframe_resolves_header() {
        // FrameFilterConfig::keyframe pulls the level inputs out of a
        // parsed header. Build a minimal header by hand and check the
        // resolved config.
        use crate::coded_header::{MbLfAdjustments, UpdateSegmentation};

        let mut header = minimal_header();
        header.filter_type = true; // simple
        header.loop_filter_level = 31;
        header.sharpness_level = 5;
        header.segmentation_enabled = true;
        header.update_segmentation = Some(UpdateSegmentation {
            update_mb_segmentation_map: true,
            update_segment_feature_data: true,
            segment_feature_mode_absolute: true,
            quantizer_update: [None; 4],
            loop_filter_update: [Some(10), None, Some(-4), Some(63)],
            segment_prob: [None; 3],
        });
        header.mb_lf_adjustments = MbLfAdjustments {
            loop_filter_adj_enable: true,
            mode_ref_lf_delta_update: true,
            ref_frame_delta_update: [Some(2), None, None, None],
            mb_mode_delta_update: [Some(-5), None, None, None],
        };

        let cfg = FrameFilterConfig::keyframe(&header);
        assert!(cfg.simple);
        assert!(cfg.key_frame);
        assert_eq!(cfg.loop_filter_level, 31);
        assert_eq!(cfg.sharpness_level, 5);
        assert!(cfg.segmentation_enabled);
        assert!(cfg.segment_abs);
        assert_eq!(cfg.segment_lf_level, [10, 0, -4, 63]);
        assert!(cfg.delta_enabled);
        assert_eq!(cfg.ref_delta_current, 2);
        assert_eq!(cfg.bpred_mode_delta, -5);
    }

    #[test]
    fn filter_frame_config_keyframe_no_segmentation() {
        let header = minimal_header();
        let cfg = FrameFilterConfig::keyframe(&header);
        assert!(!cfg.segmentation_enabled);
        assert_eq!(cfg.segment_lf_level, [0; MAX_MB_SEGMENTS]);
        assert!(!cfg.delta_enabled);
        assert_eq!(cfg.ref_delta_current, 0);
        assert_eq!(cfg.bpred_mode_delta, 0);
    }

    /// A minimal `Vp8CodedHeader` with the loop-filter knobs zeroed,
    /// used by the `FrameFilterConfig::keyframe` resolution tests.
    fn minimal_header() -> crate::Vp8CodedHeader {
        use crate::coded_header::{MbLfAdjustments, QuantIndices};
        crate::Vp8CodedHeader {
            color_space: Some(false),
            clamping_type: Some(false),
            segmentation_enabled: false,
            update_segmentation: None,
            filter_type: false,
            loop_filter_level: 0,
            sharpness_level: 0,
            mb_lf_adjustments: MbLfAdjustments {
                loop_filter_adj_enable: false,
                mode_ref_lf_delta_update: false,
                ref_frame_delta_update: [None; 4],
                mb_mode_delta_update: [None; 4],
            },
            log2_nbr_of_dct_partitions: 0,
            nbr_of_dct_partitions: 1,
            quant_indices: QuantIndices {
                y_ac_qi: 0,
                y_dc_delta: None,
                y2_dc_delta: None,
                y2_ac_delta: None,
                uv_dc_delta: None,
                uv_ac_delta: None,
            },
            refresh_entropy_probs: false,
            refresh_golden_frame: None,
            refresh_alternate_frame: None,
            copy_buffer_to_golden: None,
            copy_buffer_to_alternate: None,
            sign_bias_golden: None,
            sign_bias_alternate: None,
            refresh_last: None,
            token_prob_updates: [[[[None; 11]; 3]; 8]; 4],
            mb_no_skip_coeff: false,
            prob_skip_false: None,
            prob_intra: None,
            prob_last: None,
            prob_gf: None,
            intra_y_mode_prob_update: None,
            intra_uv_mode_prob_update: None,
            mv_prob_update: None,
        }
    }
}
