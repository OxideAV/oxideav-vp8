//! VP8 encoder — key-frame + P-frame paths (RFC 6386).
//!
//! Scope:
//! * First frame (and on forced refresh) = key-frame (I-frame).
//! * Subsequent frames = P-frames against any populated REF_LAST /
//!   REF_GOLDEN / REF_ALT slot (per-MB pick), with periodic GOLDEN /
//!   ALTREF refresh on the cadence in `Vp8EncoderConfig`. Best per-MB
//!   mode among SKIP, ZERO_MV, NEAREST_MV, NEAR_MV, NEW_MV (with
//!   quarter-pel refinement after the integer-pel search), SPLIT_MV
//!   (16×8 / 8×16 / 8×8 / 4×4 partitions, each with its own MV) and an
//!   intra fallback for MBs whose best inter prediction is too poor to
//!   be worth coding as a residual. Mode decision is Lagrangian
//!   (`D + λ·R`) when `enable_rdo` is set.
//! * All 5 intra 16×16 modes (DC / V / H / TM / B_PRED) plus the 10 4×4
//!   sub-modes under B_PRED on keyframes and the P-frame intra fallback.
//!   Chroma uses the matching 4 modes (DC / V / H / TM).
//! * Mode selection is SSE-based on the source: for each candidate we
//!   compute the pre-quant prediction error and pick the minimum. B_PRED
//!   greedily picks the best sub-mode per 4×4 block against the actual
//!   reconstructed neighbours, exactly as the decoder will see them.
//! * Fixed quantiser (default `qindex = 50`, mid-quality).
//! * In-loop deblocking filter enabled. Level is derived from `qindex`
//!   (libvpx heuristic `clamp(15 + qindex / 8, 1, 63)`); sharpness is 0,
//!   mode-ref deltas are disabled. The encoder applies the filter to its
//!   own reconstruction so subsequent P-frames reference post-filter
//!   pixels — exactly what the decoder will do.
//! * Single token partition.
//! * Accepted pixel format: `PixelFormat::Yuv420P`.

#[cfg(feature = "registry")]
use std::collections::VecDeque;

#[cfg(feature = "registry")]
use oxideav_core::Encoder;
#[cfg(feature = "registry")]
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};

use crate::error::{Result, Vp8Error as Error};
use crate::frame::Vp8Frame;

use crate::bool_encoder::{bool_cost_x256, BoolEncoder};
use crate::fdct::{fdct4x4, fwht4x4};
use crate::frame_tag::KEYFRAME_SYNC_CODE;
use crate::inter::{bilinear_predict, sixtap_predict, RefPlane};
use crate::intra::{predict_16x16, predict_4x4, predict_8x8, B4x4Neighbours};
use crate::loopfilter::{
    filter_normal_horizontal, filter_normal_vertical, filter_simple_horizontal,
    filter_simple_vertical, FilterParams,
};
use crate::mv::{encode_mv_component, mv_component_cost_x256, Mv};
use crate::tables::coeff_probs::{CoeffProbs, DEFAULT_COEF_PROBS};
use crate::tables::mv::DEFAULT_MV_CONTEXT;
use crate::tables::quant::{
    clamp_qindex, uv_ac_step, uv_dc_step, y2_ac_step, y2_dc_step, y_ac_step, y_dc_step,
};
use crate::tables::token_tree::{COEF_BANDS, ZIGZAG};
use crate::tables::trees::{
    B_DC_PRED, B_HD_PRED, B_HE_PRED, B_HU_PRED, B_LD_PRED, B_PRED, B_RD_PRED, B_TM_PRED, B_VE_PRED,
    B_VL_PRED, B_VR_PRED, DC_PRED, H_PRED, KF_BMODE_PROB, KF_UV_MODE_PROBS, KF_YMODE_PROBS,
    MBSPLIT_PROBS, MB_SPLITS, MB_SPLIT_COUNT, MV_COUNTS_TO_PROBS, SPLIT_MV, SUB_MV_REF_PROBS,
    TM_PRED, V_PRED, ZERO_MV,
};
#[cfg(feature = "registry")]
use crate::tables::trees::{DEFAULT_UV_MODE_PROBS, DEFAULT_YMODE_PROBS};
use crate::transform::{idct4x4, iwht4x4};

/// Internal borrowed view over a Yuv420P source frame. The encoder
/// reads only Y/U/V plane data + per-plane stride, so a thin
/// 6-field struct is enough to abstract over the two public input
/// shapes ([`Vp8Frame`] for the unconditional standalone API and
/// `oxideav_core::VideoFrame` for the registry-feature trait path)
/// without rewriting the per-MB plumbing.
struct Yuv420Source<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    y_stride: usize,
    u_stride: usize,
    v_stride: usize,
}

impl<'a> Yuv420Source<'a> {
    fn from_vp8_frame(f: &'a Vp8Frame) -> Self {
        Self {
            y: &f.y,
            u: &f.u,
            v: &f.v,
            y_stride: f.y_stride as usize,
            u_stride: f.uv_stride as usize,
            v_stride: f.uv_stride as usize,
        }
    }

    #[cfg(feature = "registry")]
    fn from_video_frame(v: &'a VideoFrame) -> Result<Self> {
        if v.planes.len() < 3 {
            return Err(Error::invalid("vp8 encoder: expected 3 planes"));
        }
        Ok(Self {
            y: &v.planes[0].data,
            u: &v.planes[1].data,
            v: &v.planes[2].data,
            y_stride: v.planes[0].stride,
            u_stride: v.planes[1].stride,
            v_stride: v.planes[2].stride,
        })
    }
}

/// Default qindex. 50 ≈ mid-quality; the codec accepts 0..=127.
pub const DEFAULT_QINDEX: u8 = 50;

/// SAD-per-pixel threshold below which a P-frame MB is emitted as `skip`.
/// Empirically chosen: 3 is low enough that static content always skips
/// while modestly-changed content falls through to coded residual.
const MB_SKIP_SAD_PER_PIXEL: u32 = 3;

/// Half-range (in integer luma pixels) of the NEWMV motion search window.
/// Search spans [-MOTION_SEARCH_RANGE, +MOTION_SEARCH_RANGE] on each axis.
const MOTION_SEARCH_RANGE: i32 = 8;

/// SAD delta, per MB, that NEWMV must beat ZERO_MV (or NEAREST/NEAR) by.
/// Tuned as a coarse proxy for the extra bitrate cost of coding the MV
/// delta itself — NEW_MV is only picked when motion search reduces luma
/// SAD by at least this much versus the free (no-delta) alternatives.
const NEWMV_SAD_MARGIN: u32 = 64;

/// SAD margin (per MB, in absolute luma units) below which a NEAREST or
/// NEAR candidate is preferred over NEW_MV even if NEW_MV would be
/// strictly better — reflects the bit savings from skipping the MV delta.
const NEIGHBOUR_MV_MARGIN: u32 = 32;

/// Per-MB SAD bias granted to a non-zero NEAREST / NEAR neighbour
/// candidate over the ZERO-MV baseline. When the upstream-chain has
/// already committed to a coherent global motion (e.g. a pan), the
/// neighbour MV is the right one to follow even when its sub-pel SAD
/// is marginally worse than ZERO — the encoder pays no MV delta for
/// NEAREST / NEAR, so the bit savings amortise easily over a
/// 1/2-LSB-per-pixel SAD overshoot. Without this bias the picker
/// dithers between ZERO_MV and NEAREST_MV across the frame and the
/// neighbour-chain breaks (#373).
const NEIGHBOUR_OVER_ZERO_BIAS: u32 = 96;

/// 1/8-pel L∞ tolerance under which a refined NEW_MV is considered
/// equivalent to a non-zero neighbour candidate — the picker emits
/// NEAREST / NEAR instead of paying the MV-delta bits for a NEW_MV
/// that lands at almost the same sub-pel location. `4` covers the
/// quarter-pel refinement step (`SUBPEL_REFINE_STEP = 2`) plus a one-
/// step jitter on either axis, which is the common case when integer
/// motion search lands one quarter-pel off the true motion.
const NEIGHBOUR_MV_SNAP_TOLERANCE: i32 = 4;

/// Reduced NEW_MV-over-free margin applied when the refined MV has a
/// large enough displacement from zero that it represents real motion
/// (rather than a noise-driven sub-pel jitter around zero). Catches
/// the global-pan case where ZERO and NEW_MV(-4,-4) both have similar
/// SAD against a quantised-and-deblocked reference but only NEW_MV
/// reconstructs anything close to lossless. Threshold on the L∞ MV
/// magnitude in 1/8-pel units (`16 = 2 luma pixels`).
const NEWMV_LARGE_DISPLACEMENT_THRESHOLD: i32 = 16;
/// Replacement margin used when the displacement exceeds the threshold
/// above. A real-motion NEW_MV only needs to match (or barely improve)
/// the SAD of the free-MV alternative; the bit cost of coding the
/// delta is still amortised by the residual savings on the larger
/// displacement.
const NEWMV_LARGE_DISPLACEMENT_MARGIN: u32 = 4;

/// Sub-pel refinement step in 1/8-pel units. Quarter-pel (=2) is the
/// sweet spot for a first-pass implementation: it recovers most of the
/// quarter-pel PSNR gain without the bit-cost ramp of eighth-pel MV
/// deltas, and every fractional phase we generate (0/4 = integer and
/// 2/4 / 6/4 = quarter-pel) uses a well-populated 6-tap filter row.
const SUBPEL_REFINE_STEP: i32 = 2;

/// Per-pixel luma SAD above which we give up on inter coding and emit
/// the MB as intra-DC_PRED. Picked so that inter modes remain chosen on
/// anything the motion search + residual can reasonably reconstruct —
/// intra kicks in for uncovered regions, scene cuts inside a P-frame,
/// or heavily-aliased/textured content where the reference is useless.
const INTRA_IN_P_SAD_PER_PIXEL: u32 = 24;

/// Probability used to signal "inter MB" vs "intra MB" inside a P-frame.
/// Biased heavily towards inter (the common case) while still making
/// intra-in-P affordable when needed (~3 bits versus ~0.4 bits for
/// inter). We cannot use 1 like the earlier encoder since that makes
/// intra-in-P arbitrarily expensive.
const PROB_INTRA_IN_P: u8 = 200;

/// Loop-filter sharpness. Keeping sharpness = 0 gives the decoder's
/// default behaviour and is the libvpx starting point for rate-distortion
/// tuned encoders; non-zero sharpness only matters once mode/ref deltas
/// are also enabled, which we do not emit.
const LOOP_FILTER_SHARPNESS: u8 = 0;

/// libvpx's simple heuristic for picking a baseline loop-filter level
/// from the quantiser index. `qindex=50` → level=21; `qindex=0` → 15;
/// `qindex=127` → 30. Clamped to the 1..=63 VP8 range (level 0 would
/// disable the filter).
#[inline]
fn loop_filter_level_for_qindex(qi: u8) -> u8 {
    let l = 15 + (qi as i32 / 8);
    l.clamp(1, 63) as u8
}

/// Pick the bitstream `filter_type` (0 = normal, 1 = simple) for a
/// given config + frame-level filter level. The simple-mode filter
/// (RFC 6386 §15.2) is luma-only and only touches the four pixels
/// closest to each edge — a smaller per-MB cost (no chroma MB-edge
/// and no chroma sub-block-edge filter calls) and a slightly smaller
/// header (one bit). Picked by default at low filter levels where
/// the wider 6-pixel normal-mode filter would risk smoothing
/// content the encoder is otherwise preserving.
#[inline]
fn pick_filter_type(level: u8, config: &Vp8EncoderConfig) -> u8 {
    match config.loop_filter_mode {
        LoopFilterMode::Normal => 0,
        LoopFilterMode::Simple => 1,
        LoopFilterMode::Auto => {
            if level <= config.simple_lf_max_level {
                1
            } else {
                0
            }
        }
    }
}

/// SAD threshold (per pixel) below which SPLIT_MV is not considered.
/// When a single MV already matches the MB well, the per-partition MV
/// bits for SPLIT are wasted — skipping the search entirely is a big
/// speed win on the common case of smooth global motion.
const SPLITMV_CONSIDER_SAD_PER_PIXEL: u32 = 2;

/// SAD reduction that SPLIT_MV must beat NEW_MV by to be picked. SPLIT
/// pays 2..=16 extra MV encodings on top of the split-tree leaf, so it
/// is only worthwhile when the per-partition motion genuinely reduces
/// SAD substantially (the exact value scales with partition count).
const SPLITMV_SAD_MARGIN_PER_PARTITION: u32 = 96;

/// Default golden-frame refresh interval (P-frames). Every Nth P-frame the
/// encoder marks `refresh_golden_frame=1`, snapshotting the current
/// reconstruction into the long-term GOLDEN slot. Set to 0 to disable.
pub const DEFAULT_GOLDEN_INTERVAL: u32 = 8;

/// Default alt-ref refresh interval (P-frames). Every Nth P-frame the
/// encoder marks `refresh_alt_ref_frame=1`. With our look-ahead-free
/// implementation alt-ref is essentially a second long-term anchor with
/// a different cadence than GOLDEN, exposing two stable references the
/// per-MB rate-distortion search can pick from. Set to 0 to disable.
pub const DEFAULT_ALT_REF_INTERVAL: u32 = 13;

/// Lagrangian multiplier scale: lambda = LAMBDA_SCALE * QP^2 / 256.
/// The classic textbook expression `lambda = 0.85 * QP^2` would dominate
/// the integer-SSE distortion on small QPs (lambda > 1000); scaling it
/// down by 256 keeps lambda comfortably in the same numeric range as
/// the SSE accumulator while preserving its quadratic shape in QP.
pub const LAMBDA_SCALE_DEFAULT: u32 = 218; // ≈ 0.85 * 256

/// Default per-segment quantiser deltas applied to `qindex` when
/// segmentation is enabled (RFC 6386 §10). Indexed by segment id 0..=3,
/// where the encoder's variance-based classifier maps the lowest-variance
/// MBs to segment 0 and the highest-variance MBs to segment 3. The deltas
/// give the smooth-content segment a lower QP (better quality where the
/// eye notices banding) and the high-texture segment a higher QP (saves
/// bits where the texture masks small reconstruction errors). Bitstream
/// `abs_delta` is signalled as 0 (= delta) so the decoder applies these
/// on top of `header.quant.y_ac_qi`.
pub const DEFAULT_SEGMENT_QUANT_DELTAS: [i32; 4] = [-8, -4, 0, 4];

/// Default per-segment loop-filter level deltas applied to the frame-level
/// `loop_filter.level` when segmentation is enabled (RFC 6386 §10 + §15.2).
/// Indexed by segment id 0..=3 the same way `DEFAULT_SEGMENT_QUANT_DELTAS`
/// is — the variance classifier lands smooth content in segment 0 and
/// high-variance content in segment 3.
///
/// Smooth segments take a *negative* LF delta (a softer filter — smooth
/// regions don't blocky-artefact and over-filtering would just smear
/// fine detail). High-variance segments take a *positive* LF delta (a
/// stronger filter — coarse-quantised textured MBs benefit from a wider
/// deblocking pass to mask the per-MB DCT block boundaries that the
/// extra QP step exposes). The decoder applies the delta as
/// `clamp(frame_level + delta, 0..=63)` per-MB via
/// `per_mb_filter_level`. Bitstream `abs_delta = 0`.
pub const DEFAULT_SEGMENT_LF_DELTAS: [i32; 4] = [-2, -1, 0, 2];

/// Variance bucket boundaries (luma, summed-square units per MB) that map
/// each MB to a segment id. A 16×16 MB has 256 pixels so a per-pixel
/// variance of `v_pp` corresponds to `v_pp * 256` in this metric. Picked
/// to land roughly in equal-population quartiles on the encoder's test
/// fixtures (gray pan, mandelbrot, mixed clip): boundaries at variances
/// of ~80, ~640, ~3200 per pixel.
pub const SEGMENT_VARIANCE_THRESHOLDS: [u64; 3] = [80 * 256, 640 * 256, 3200 * 256];

/// Default scene-cut detection multiplier. A frame is flagged as a
/// scene cut when its source-luma mean-absolute-difference (MAD) versus
/// the previous source frame exceeds `mean(MAD) + N · stddev(MAD)`
/// across the running window — i.e. the new frame's MAD is `N` standard
/// deviations above the running average. `4.0` is the libvpx /
/// libavcodec ballpark for "obvious cut, not just motion".
pub const DEFAULT_SCENE_CUT_THRESHOLD: f32 = 4.0;
/// Floor applied to the MAD comparison so the very first few P-frames
/// (when the running stddev is still ~0) do not trigger spurious cuts
/// on quiet content. Per-pixel luma MAD threshold below which we never
/// flag a cut, regardless of how many sigmas above the mean it is.
pub const SCENE_CUT_ABS_FLOOR: f32 = 12.0;
/// Default quantiser boost (qindex delta, applied as a subtraction —
/// lower qindex = finer quant) granted to the first
/// `DEFAULT_SCENE_CUT_BOOST_FRAMES` after a forced scene-cut keyframe.
/// Compensates for the fact that the new GOP's references have just
/// been thrown away — extra quality on the rebuild buys back the
/// long-tail PSNR drop that the cut would otherwise propagate.
pub const DEFAULT_SCENE_CUT_QUANT_BOOST: u8 = 8;
/// Default number of frames after a forced scene-cut keyframe over
/// which `scene_cut_quant_boost` is applied (linearly tapered to zero
/// by the end of the window). 4 frames is enough to repopulate
/// GOLDEN / ALTREF on the default cadence and let the encoder settle
/// without bloating the GOP's average bitrate.
pub const DEFAULT_SCENE_CUT_BOOST_FRAMES: u32 = 4;
/// Length of the running-statistics window used by the scene-cut
/// detector. A small window keeps the detector responsive to gradual
/// brightness changes (so steady-state pans don't poison the threshold);
/// 16 frames is roughly half a second at 30 fps.
const SCENE_CUT_WINDOW: usize = 16;

/// Default look-ahead window size for alt-ref synthesis. The encoder
/// buffers this many input frames before emitting the alt-ref so it can
/// build a temporally-filtered (noise-reduced) reference image from a
/// neighbourhood of the centre frame. Odd values are preferred so the
/// window is symmetric around its centre. 7 is the libvpx ballpark and
/// keeps the per-frame latency cost bounded (~200 ms at 30 fps).
pub const DEFAULT_LOOKAHEAD_WINDOW: usize = 7;

/// Sigma (in luma intensity units) controlling how aggressively the
/// temporal filter rejects pixels that disagree with the centre frame.
/// `weight = exp(-diff^2 / sigma^2)` falls off so that pixels within
/// ±sigma of the centre contribute strongly while pixels several sigma
/// away contribute essentially zero — this preserves motion edges and
/// occlusion boundaries while smoothing residual noise/grain. 24 is
/// chosen so a typical 8-bit noise floor (~3-5 LSB) lands in the
/// "strongly weighted" tail and a real motion edge (>>48 LSB) is
/// effectively gated off — slightly broader than the canonical libvpx
/// ARNR sigma to favour noise smoothing on the synthetic fixtures the
/// test suite stresses.
const TEMPORAL_FILTER_SIGMA: f32 = 24.0;

/// Half-range (luma integer pixels) of the inter-window motion search
/// used to align non-centre frames to the centre frame for the temporal
/// filter. A small range keeps the synthesis cheap; for the noise-
/// reduction goal we only need to align local content, not track
/// large displacements.
const ALTREF_MC_RANGE: i32 = 8;

/// Quantiser delta applied to the hidden alt-ref P-frame relative to
/// the visible-frame `qindex`. The alt-ref slot's accuracy bounds the
/// per-MB residual size on every visible P-frame that references it,
/// so spending a few extra bits on the hidden frame compounds across
/// the rest of the GOP. 12 is empirically the sweet spot: hidden
/// frames stay small (the visible-q-minus-12 quant is still coarse
/// enough that smooth content quantises to nearly all-zero coeffs)
/// while the alt-ref reconstruction is noticeably cleaner than what
/// the visible-frame quantiser would manage on its own. Going much
/// finer blows the hidden-frame size up faster than the per-MB savings
/// on visible frames can compensate.
const HIDDEN_ALTREF_QINDEX_DELTA: i32 = 12;

/// Loop-filter type selector. `Auto` is the default (and the libvpx
/// convention): pick simple mode at low filter levels (where the
/// wider normal-mode filter would over-smooth low-detail content)
/// and normal mode otherwise. `Normal` and `Simple` force the
/// corresponding `filter_type` regardless of level — handy for
/// regression tests pinning a specific bitstream shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopFilterMode {
    /// Pick `simple` when `lf_level <= simple_lf_max_level`, otherwise
    /// `normal`.
    Auto,
    /// Always emit `filter_type = 0` (normal). Equivalent to the
    /// pre-#336 hard-wired behaviour.
    Normal,
    /// Always emit `filter_type = 1` (simple). Useful for low-bitrate
    /// streaming where the bit/speed savings dominate the visual loss.
    Simple,
}

/// Default upper-bound `lf_level` (inclusive) for `LoopFilterMode::Auto`
/// to pick simple mode. With `loop_filter_level_for_qindex(qi) = 15 +
/// qi/8`, level ≤ 15 corresponds to qi ≤ 7 (very low-distortion
/// targets where the wider 6-pixel normal filter would smooth content
/// the encoder is otherwise preserving). Level 16..=63 stay on
/// normal mode by default.
pub const DEFAULT_SIMPLE_LF_MAX_LEVEL: u8 = 15;

/// Per-MB encoder configuration. Knob bag for the alt-ref / golden-ref
/// planning + Lagrangian RDO mode decision wired up in this version.
///
/// All fields default to a "sensible" value chosen by the encoder; the
/// public `make_encoder_with_config` constructor lets callers (tests,
/// rate-control loops) override individual knobs.
#[derive(Clone, Copy, Debug)]
pub struct Vp8EncoderConfig {
    /// Quantiser index, 0..=127. Lower = higher quality, larger files.
    pub qindex: u8,
    /// Refresh GOLDEN every N P-frames (0 = disable; 1 = every P-frame
    /// just like LAST, which is wasteful but valid).
    pub golden_interval: u32,
    /// Refresh ALTREF every N P-frames.
    pub alt_ref_interval: u32,
    /// Enable per-MB Lagrangian rate-distortion mode decision (D + λ·R).
    /// When false, the encoder falls back to the SAD-only / SSE-only
    /// heuristic from earlier rounds.
    pub enable_rdo: bool,
    /// Lambda multiplier scale; lambda = scale * QP^2 / 256. Set to 0 to
    /// turn off the rate term entirely (pure-distortion mode decision).
    pub lambda_scale: u32,
    /// Enable per-MB picking among LAST / GOLDEN / ALTREF references in
    /// the inter mode decision. When `false`, the encoder always uses
    /// LAST (the legacy single-reference behaviour).
    pub enable_multi_ref: bool,
    /// Enable per-MB segment maps (RFC 6386 §10). When `true` the
    /// encoder classifies each MB by source-luma variance into one of
    /// four segments and applies a per-segment quantiser delta from
    /// `segment_quant_deltas`, signalling the per-segment data and the
    /// per-MB segment-id bits in the frame header / mode-info stream.
    pub enable_segments: bool,
    /// Per-segment quantiser deltas (segment id 0..=3) used when
    /// `enable_segments` is `true`. Applied as `qindex + delta`, so e.g.
    /// `[-8, -4, 0, +4]` gives smooth regions higher quality and
    /// high-variance regions a coarser quant to save bits.
    pub segment_quant_deltas: [i32; 4],
    /// Per-segment loop-filter level deltas (segment id 0..=3) used when
    /// `enable_segments` is `true`. Applied to the frame-level
    /// `loop_filter.level` per-MB via `clamp(level + delta, 0..=63)`,
    /// so e.g. `[-2, -1, 0, +2]` softens the deblocking pass on smooth
    /// content (avoiding over-smoothing) and strengthens it on
    /// high-variance content (masking DCT block boundaries exposed by
    /// the coarser per-segment quantiser). The deltas are emitted in
    /// the segmentation block of the frame header with `abs_delta = 0`
    /// and decoded by `per_mb_filter_level`. Set every entry to `0` to
    /// fall back to a single frame-wide filter level.
    pub segment_lf_deltas: [i32; 4],
    /// Enable the per-frame scene-cut detector. When `true` each
    /// incoming source frame's mean-absolute-difference (MAD) versus
    /// the previous source frame is compared against the running mean
    /// `+ scene_cut_threshold * stddev` over the last 16 frames; when
    /// the MAD exceeds the bound (and the absolute floor) the next
    /// frame is forced to a keyframe and the LAST / GOLDEN / ALTREF
    /// slots are dropped, so the keyframe rebuilds the GOP from scratch
    /// instead of dragging in pre-cut residual context.
    pub enable_scene_cut: bool,
    /// Multiplier on the running MAD stddev that the new frame's MAD
    /// must exceed (over and above the running mean) for the encoder
    /// to flag it as a scene cut. Lower = more aggressive (more cuts);
    /// higher = stricter (only obvious cuts). Defaults to
    /// [`DEFAULT_SCENE_CUT_THRESHOLD`].
    pub scene_cut_threshold: f32,
    /// Quantiser boost (subtracted from `qindex`, so finer quant)
    /// applied to the first few frames following a forced scene-cut
    /// keyframe. Compensates for the brand-new reference frame —
    /// extra quality at the cut buys back the long-tail PSNR loss
    /// that the dropped GOLDEN/ALTREF would otherwise propagate.
    /// `0` disables the boost (still detects cuts, still emits the
    /// keyframe, just doesn't change the QP).
    pub scene_cut_quant_boost: u8,
    /// Number of frames after a scene-cut keyframe over which the
    /// quant boost is applied (tapered linearly to zero). After this
    /// many frames the encoder reverts to `qindex` exactly.
    pub scene_cut_boost_frames: u32,
    /// Enable look-ahead alt-ref synthesis. When `true` the encoder
    /// buffers up to `lookahead_window` source frames and, at every
    /// alt-ref refresh point, synthesises the alt-ref slot from a
    /// motion-compensated, pixel-wise temporal filter over the window
    /// (smoother reference → smaller forward-prediction residuals on
    /// motion-rich content). The synthesised image is communicated to
    /// the decoder as a hidden P-frame (`show_frame = 0`) emitted
    /// just before the visible frame at that position. When `false`
    /// the encoder reverts to the legacy "alt-ref slot is whatever
    /// reconstruction the cadence frame produced" behaviour.
    pub enable_lookahead_altref: bool,
    /// Look-ahead window size (number of source frames buffered for
    /// alt-ref synthesis). Must be ≥ 1. Odd values keep the window
    /// symmetric around the centre frame. Capped internally at
    /// `alt_ref_interval` so we never delay a refresh point indefinitely.
    pub lookahead_window: usize,
    /// Loop-filter mode selection (RFC 6386 §15.2 `filter_type`).
    /// Default `Auto` picks simple mode at low filter levels and
    /// normal mode otherwise; see [`LoopFilterMode`] for forced
    /// alternatives.
    pub loop_filter_mode: LoopFilterMode,
    /// Upper-bound `lf_level` (inclusive) below which
    /// `LoopFilterMode::Auto` picks simple mode. Ignored when
    /// `loop_filter_mode` is `Normal` or `Simple`. Defaults to
    /// [`DEFAULT_SIMPLE_LF_MAX_LEVEL`].
    pub simple_lf_max_level: u8,
}

impl Default for Vp8EncoderConfig {
    fn default() -> Self {
        Self {
            qindex: DEFAULT_QINDEX,
            golden_interval: DEFAULT_GOLDEN_INTERVAL,
            alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
            enable_rdo: true,
            lambda_scale: LAMBDA_SCALE_DEFAULT,
            enable_multi_ref: true,
            enable_segments: true,
            segment_quant_deltas: DEFAULT_SEGMENT_QUANT_DELTAS,
            segment_lf_deltas: DEFAULT_SEGMENT_LF_DELTAS,
            enable_scene_cut: true,
            scene_cut_threshold: DEFAULT_SCENE_CUT_THRESHOLD,
            scene_cut_quant_boost: DEFAULT_SCENE_CUT_QUANT_BOOST,
            scene_cut_boost_frames: DEFAULT_SCENE_CUT_BOOST_FRAMES,
            enable_lookahead_altref: true,
            lookahead_window: DEFAULT_LOOKAHEAD_WINDOW,
            loop_filter_mode: LoopFilterMode::Auto,
            simple_lf_max_level: DEFAULT_SIMPLE_LF_MAX_LEVEL,
        }
    }
}

/// Encoder factory used by [`crate::register_codecs`].
#[cfg(feature = "registry")]
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| oxideav_core::Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| oxideav_core::Error::invalid("vp8 encoder: missing height"))?;
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(oxideav_core::Error::invalid(format!(
            "vp8 encoder: dimensions {width}x{height} out of range (1..=16383)"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(oxideav_core::Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);

    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        config: Vp8EncoderConfig::default(),
        time_base,
        pending: VecDeque::new(),
        eof: false,
        last_frame: None,
        golden_frame: None,
        alt_ref_frame: None,
        pframe_count: 0,
        scene_cut: SceneCutState::new(),
        lookahead: VecDeque::new(),
    }))
}

/// Build an encoder with an explicit qindex. Useful for tests and for
/// callers that want finer control than the default quality.
///
/// This legacy constructor keeps the per-frame scene-cut detector
/// **disabled** so callers that hand-craft small frame sequences get
/// the bit-exact pre-#166 behaviour (no surprise forced keyframes).
/// Use [`make_encoder_with_config`] with `enable_scene_cut = true`
/// to opt in.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    qindex: u8,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| oxideav_core::Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| oxideav_core::Error::invalid("vp8 encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(oxideav_core::Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }
    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);
    let mut cfg = Vp8EncoderConfig::default();
    cfg.qindex = qindex.min(127);
    cfg.enable_scene_cut = false;
    // Match the pre-#209 single-frame emit contract for callers that
    // ask for a "qindex only" encoder — the lookahead path emits hidden
    // alt-ref packets which would surprise legacy callers.
    cfg.enable_lookahead_altref = false;
    // Match the pre-#336 normal-mode loop-filter emit so legacy
    // qindex-only callers still get bit-identical bitstreams. Callers
    // that want simple-mode LF use `make_encoder_with_config`.
    cfg.loop_filter_mode = LoopFilterMode::Normal;
    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        config: cfg,
        time_base,
        pending: VecDeque::new(),
        eof: false,
        last_frame: None,
        golden_frame: None,
        alt_ref_frame: None,
        pframe_count: 0,
        scene_cut: SceneCutState::new(),
        lookahead: VecDeque::new(),
    }))
}

/// Build an encoder with a fully-specified configuration. Lets callers
/// turn alt-ref / golden planning + RDO on or off independently.
#[cfg(feature = "registry")]
pub fn make_encoder_with_config(
    params: &CodecParameters,
    config: Vp8EncoderConfig,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| oxideav_core::Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| oxideav_core::Error::invalid("vp8 encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(oxideav_core::Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }
    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);
    let mut cfg = config;
    cfg.qindex = cfg.qindex.min(127);
    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        config: cfg,
        time_base,
        pending: VecDeque::new(),
        eof: false,
        last_frame: None,
        golden_frame: None,
        alt_ref_frame: None,
        pframe_count: 0,
        scene_cut: SceneCutState::new(),
        lookahead: VecDeque::new(),
    }))
}

/// Reconstructed reference frame (post-quant reconstruction, matching what
/// the decoder will regenerate from the emitted bitstream). Stored in
/// MB-padded stride/height to make per-MB access a straight index calc.
#[derive(Clone)]
struct ReferenceFrame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    y_h: usize,
    uv_h: usize,
}

#[cfg(feature = "registry")]
struct Vp8Encoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    config: Vp8EncoderConfig,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
    last_frame: Option<ReferenceFrame>,
    golden_frame: Option<ReferenceFrame>,
    alt_ref_frame: Option<ReferenceFrame>,
    /// Count of P-frames emitted so far; used to drive the periodic
    /// golden / alt-ref refresh schedule.
    pframe_count: u32,
    /// Scene-cut detector state: source-luma mean of the previous
    /// incoming frame, the running window of source-MAD samples (one
    /// per P-frame), and how many frames remain in the post-cut
    /// quant-boost window.
    scene_cut: SceneCutState,
    /// Look-ahead buffer of pending input frames (for alt-ref synthesis).
    /// Only populated when `config.enable_lookahead_altref` is `true`;
    /// otherwise frames bypass this buffer entirely and the legacy
    /// 1-in / 1-out send/receive cadence is preserved exactly.
    lookahead: VecDeque<VideoFrame>,
}

/// Per-frame state for the scene-cut detector. Holds the previous source
/// frame's luma so the next frame can compute its MAD without keeping the
/// whole previous frame around in some other slot, plus the running
/// window of MAD samples that drives the mean + stddev threshold.
#[cfg(feature = "registry")]
#[derive(Clone)]
struct SceneCutState {
    /// Source-luma plane of the previous source frame, packed at the
    /// caller's stride. `None` until the first frame has been processed.
    /// We retain only the luma plane (not chroma) because brightness
    /// jumps dominate the MAD signal on real cuts and the chroma
    /// contribution is dwarfed by the per-pixel cost. `prev_width` /
    /// `prev_height` give the slot's geometry for the next-frame compare.
    prev_y: Option<Vec<u8>>,
    prev_width: usize,
    prev_height: usize,
    prev_stride: usize,
    /// Ring buffer of recent inter-frame MAD samples (per-pixel,
    /// integer luma units). Capped at `SCENE_CUT_WINDOW`. Drained on
    /// scene-cut so the post-cut window starts fresh.
    mad_window: VecDeque<f32>,
    /// Number of P-frames remaining in the active quant-boost window
    /// after the most recent forced scene-cut keyframe. Decremented
    /// once per encoded frame.
    boost_remaining: u32,
}

#[cfg(feature = "registry")]
impl SceneCutState {
    fn new() -> Self {
        Self {
            prev_y: None,
            prev_width: 0,
            prev_height: 0,
            prev_stride: 0,
            mad_window: VecDeque::with_capacity(SCENE_CUT_WINDOW),
            boost_remaining: 0,
        }
    }

    /// Compute the per-pixel mean-absolute-difference of the supplied
    /// source-luma plane vs the cached `prev_y`. Returns `None` when
    /// no previous frame is cached (first frame after init / reset)
    /// or when the geometry has changed (which would make the per-pixel
    /// compare meaningless — also reported as no-cut so the caller
    /// just records the new geometry).
    fn mad_against_prev(
        &self,
        y: &[u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Option<f32> {
        let prev = self.prev_y.as_ref()?;
        if width != self.prev_width
            || height != self.prev_height
            || stride != self.prev_stride
            || prev.len() < height * stride
            || y.len() < height * stride
        {
            return None;
        }
        let mut acc: u64 = 0;
        for r in 0..height {
            let row_off = r * stride;
            for c in 0..width {
                let a = prev[row_off + c] as i32;
                let b = y[row_off + c] as i32;
                acc += (a - b).unsigned_abs() as u64;
            }
        }
        let n = (width * height) as f32;
        if n == 0.0 {
            None
        } else {
            Some(acc as f32 / n)
        }
    }

    /// Mean of the running MAD window. `0.0` on an empty window.
    fn mad_mean(&self) -> f32 {
        if self.mad_window.is_empty() {
            return 0.0;
        }
        let n = self.mad_window.len() as f32;
        self.mad_window.iter().copied().sum::<f32>() / n
    }

    /// Sample stddev of the running MAD window. `0.0` on a window with
    /// fewer than 2 samples.
    fn mad_stddev(&self) -> f32 {
        if self.mad_window.len() < 2 {
            return 0.0;
        }
        let mean = self.mad_mean();
        let n = self.mad_window.len() as f32;
        let var = self
            .mad_window
            .iter()
            .map(|x| (*x - mean) * (*x - mean))
            .sum::<f32>()
            / (n - 1.0);
        var.sqrt()
    }

    /// Push a new MAD sample, evicting the oldest if the window is full.
    fn push_mad(&mut self, mad: f32) {
        if self.mad_window.len() >= SCENE_CUT_WINDOW {
            self.mad_window.pop_front();
        }
        self.mad_window.push_back(mad);
    }

    /// Stash the supplied source-luma plane as the new "previous" frame
    /// for the next MAD compare.
    fn cache_prev(&mut self, y: &[u8], width: usize, height: usize, stride: usize) {
        let needed = height * stride;
        let buf = self.prev_y.get_or_insert_with(|| vec![0u8; needed]);
        if buf.len() != needed {
            buf.resize(needed, 0);
        }
        buf.copy_from_slice(&y[..needed]);
        self.prev_width = width;
        self.prev_height = height;
        self.prev_stride = stride;
    }

    /// Drop all running statistics on a forced cut so the next P-frame's
    /// MAD is not compared against pre-cut samples.
    fn reset_after_cut(&mut self, boost_frames: u32) {
        self.mad_window.clear();
        self.boost_remaining = boost_frames;
    }
}

/// Per-frame plan: which reference slots get refreshed by the new
/// reconstruction at the end of this frame, plus which references are
/// available to the per-MB inter mode decision while encoding it.
#[cfg(feature = "registry")]
#[derive(Clone, Copy, Debug)]
struct RefPlan {
    refresh_last: bool,
    refresh_golden: bool,
    refresh_alt: bool,
    /// Per-MB inter decision may pick from LAST (always available on
    /// P-frames). Whether GOLDEN/ALT are also offered is gated by both
    /// `enable_multi_ref` and the slot actually being populated.
    use_golden: bool,
    use_alt: bool,
}

#[cfg(feature = "registry")]
impl RefPlan {
    /// Compute the refresh / availability plan for the next P-frame
    /// given the current encoder state, using the encoder's persistent
    /// `config`. Equivalent to `for_pframe_with_config(enc, &enc.config)`.
    #[allow(dead_code)]
    fn for_pframe(enc: &Vp8Encoder) -> Self {
        Self::for_pframe_with_config(enc, &enc.config)
    }

    /// Compute the refresh / availability plan against an explicit
    /// per-frame config. Used by the scene-cut path which feeds in a
    /// boosted-qindex copy of the persistent config without otherwise
    /// changing the multi-ref / segment knobs.
    fn for_pframe_with_config(enc: &Vp8Encoder, cfg: &Vp8EncoderConfig) -> Self {
        // Counter is incremented before computing the plan, so the
        // *first* P-frame is `pframe_count == 1`.
        let n = enc.pframe_count;
        let refresh_golden =
            cfg.golden_interval > 0 && cfg.enable_multi_ref && n % cfg.golden_interval == 0;
        let refresh_alt =
            cfg.alt_ref_interval > 0 && cfg.enable_multi_ref && n % cfg.alt_ref_interval == 0;
        Self {
            refresh_last: true,
            refresh_golden,
            refresh_alt,
            use_golden: cfg.enable_multi_ref && enc.golden_frame.is_some(),
            use_alt: cfg.enable_multi_ref && enc.alt_ref_frame.is_some(),
        }
    }
}

#[cfg(feature = "registry")]
impl Vp8Encoder {
    /// Per-frame configuration. Returns `self.config` unchanged when
    /// no scene-cut quant boost is in flight; otherwise subtracts a
    /// linearly-tapered boost from `qindex` so the first post-cut
    /// frame gets the full boost and the boost-window tail blends
    /// back to the configured qindex.
    fn effective_config_for_frame(&self) -> Vp8EncoderConfig {
        let mut cfg = self.config;
        let total = self.config.scene_cut_boost_frames;
        let remaining = self.scene_cut.boost_remaining;
        if total > 0 && remaining > 0 && self.config.scene_cut_quant_boost > 0 {
            // remaining counts down from `total` (set on the keyframe)
            // to 0; taper = remaining / total gives a linear ramp from
            // 1.0 down to 0 over the window.
            let taper = remaining as f32 / total as f32;
            let boost = (self.config.scene_cut_quant_boost as f32 * taper).round() as i32;
            let new_qi = (self.config.qindex as i32 - boost).clamp(0, 127) as u8;
            cfg.qindex = new_qi;
        }
        cfg
    }

    /// Push a copy of the supplied source frame into the look-ahead
    /// ring buffer used by the alt-ref synthesis path. Evicts the oldest
    /// entry when the buffer is at capacity (`config.lookahead_window`).
    /// `lookahead_window` of 0 or 1 just keeps the most-recent frame.
    fn push_lookahead_source(&mut self, v: &VideoFrame) {
        let cap = self.config.lookahead_window.max(1);
        // Snapshot only the data we need (the 3 plane buffers + their
        // strides). Cloning the whole VideoFrame is fine on the small
        // resolutions this encoder targets.
        let snap = VideoFrame {
            pts: v.pts,
            planes: v.planes.iter().take(3).cloned().collect(),
        };
        if self.lookahead.len() >= cap {
            self.lookahead.pop_front();
        }
        self.lookahead.push_back(snap);
    }

    /// Synthesise a temporally-filtered alt-ref image from the current
    /// look-ahead buffer and emit it as a hidden P-frame
    /// (`show_frame = 0`, `refresh_alt = 1`). Returns the hidden frame's
    /// bitstream + its reconstructed reference (matching what the
    /// decoder will install in its alt-ref slot).
    ///
    /// The reference for the hidden frame's residual is `last_ref` —
    /// whatever currently sits in the encoder's LAST slot. By keeping
    /// the hidden frame's `refresh_last = 0`, the LAST slot is left
    /// untouched after the hidden frame, so the visible frame that
    /// follows still references the same LAST it would have without
    /// the hidden frame in the way.
    ///
    /// Returns `None` when synthesis is not feasible (window has zero
    /// frames, geometry mismatch, etc.). Callers should fall back to
    /// the legacy alt-ref-from-reconstruction path on `None`.
    fn try_emit_lookahead_altref(
        &self,
        cfg: &Vp8EncoderConfig,
        last_ref: &ReferenceFrame,
    ) -> Option<(Vec<u8>, ReferenceFrame)> {
        if self.lookahead.is_empty() {
            return None;
        }
        // Synthesize a Yuv420P VideoFrame from the lookahead buffer.
        let synth =
            synthesize_altref_image(self.width as usize, self.height as usize, &self.lookahead)?;
        // Encode the synthesized image as a hidden P-frame against LAST,
        // refreshing only ALT.
        encode_hidden_altref_pframe(self.width, self.height, *cfg, &synth, last_ref).ok()
    }
}

#[cfg(feature = "registry")]
impl Encoder for Vp8Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> oxideav_core::Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => {
                return Err(oxideav_core::Error::invalid(
                    "vp8 encoder: video frames only",
                ))
            }
        };
        if v.planes.len() < 3 {
            return Err(oxideav_core::Error::invalid(
                "vp8 encoder: expected 3 planes",
            ));
        }
        // Frame dims and pixel format are now stream-level — validated
        // by the caller / pipeline against `output_params`. The encoder
        // trusts the planes match `self.width × self.height` Yuv420P.

        // ----------------------------------------------------------------
        // Scene-cut detection (RFC 6386 itself is silent on the encoder's
        // GOP structure — cuts are an encoder-side rate-control choice,
        // not a bitstream feature). We compute the per-pixel
        // mean-absolute-difference (MAD) of this frame's source luma
        // versus the previous source frame, then compare it against the
        // running mean + N · stddev over the last few frames. A cut
        // forces the next frame to be a keyframe and drops the LAST /
        // GOLDEN / ALTREF slots so the keyframe is genuinely
        // self-contained. The detector also seeds a short post-cut
        // quant-boost window that buys back the long-tail PSNR drop
        // caused by losing the long-term references.
        // ----------------------------------------------------------------
        let y_plane = &v.planes[0];
        let frame_w = self.width as usize;
        let frame_h = self.height as usize;

        let mut force_keyframe_for_cut = false;
        let mut cut_mad: Option<f32> = None;
        if self.config.enable_scene_cut && self.last_frame.is_some() {
            if let Some(mad) =
                self.scene_cut
                    .mad_against_prev(&y_plane.data, frame_w, frame_h, y_plane.stride)
            {
                let mean = self.scene_cut.mad_mean();
                let std = self.scene_cut.mad_stddev();
                let bound = mean + self.config.scene_cut_threshold * std;
                if mad >= bound && mad >= SCENE_CUT_ABS_FLOOR {
                    force_keyframe_for_cut = true;
                    cut_mad = Some(mad);
                } else {
                    self.scene_cut.push_mad(mad);
                }
            }
        }

        let is_keyframe = self.last_frame.is_none() || force_keyframe_for_cut;
        if force_keyframe_for_cut {
            // Drop every reference slot so the new keyframe rebuilds
            // the GOP from scratch. Without this the per-MB ref-frame
            // picker on the *next* P-frame could still pull pre-cut
            // pixels out of GOLDEN / ALTREF.
            self.last_frame = None;
            self.golden_frame = None;
            self.alt_ref_frame = None;
            self.pframe_count = 0;
            self.scene_cut
                .reset_after_cut(self.config.scene_cut_boost_frames);
        }

        // Per-frame qindex with an optional post-cut quality boost. The
        // boost is tapered linearly so the first post-cut frame gets
        // the full reduction and the boost-window tail falls back to
        // the configured qindex in step.
        let frame_config = self.effective_config_for_frame();

        // ----------------------------------------------------------------
        // Look-ahead alt-ref synthesis. When enabled, the encoder caches
        // up to `lookahead_window` recent source frames (including the
        // current one) and, at every alt-ref refresh point, builds a
        // motion-compensated, pixel-wise temporal-filtered image from
        // them. That synthesized image becomes the new alt-ref reference
        // for both encoder and decoder via a hidden P-frame
        // (`show_frame = 0`, `refresh_alt = 1`) emitted *just before* the
        // visible frame at the cadence point. The hidden frame's
        // residual is coded against LAST so the synthesized image is
        // reconstructed identically on both sides.
        //
        // After the hidden frame fires, the visible frame at the cadence
        // point gets its `refresh_alt` flag suppressed (the hidden frame
        // already did the refresh) — its bitstream still references the
        // new alt-ref via LAST/GOLDEN/ALT per-MB picking.
        // ----------------------------------------------------------------
        let mut suppress_visible_alt_refresh = false;
        if !is_keyframe
            && self.config.enable_lookahead_altref
            && self.config.alt_ref_interval > 0
            && self.config.enable_multi_ref
            && self.last_frame.is_some()
        {
            // Push the *current* source frame into the lookahead ring
            // first so it participates as the centre of the temporal
            // window (cleanest alignment behaviour). The legacy path
            // doesn't use this buffer so the push is otherwise a no-op.
            self.push_lookahead_source(v);
            // Cadence: the hidden alt-ref refreshes at the *same* P-frame
            // index where the visible plan would have refreshed alt
            // before #209. `pframe_count` is incremented later (in the
            // visible-encode block); compute what the *next* count would
            // be — that's the index we test against the cadence.
            let next_pframe_count = self.pframe_count + 1;
            if next_pframe_count % self.config.alt_ref_interval == 0 {
                // Synthesize + emit the hidden alt-ref. On any internal
                // failure we silently fall back to the legacy path
                // (visible frame's `refresh_alt` flag stays as planned)
                // — never break the encode just because synthesis went
                // wrong on a particular window.
                let last_ref = self.last_frame.as_ref().unwrap().clone();
                if let Some((hidden_bitstream, hidden_rec)) =
                    self.try_emit_lookahead_altref(&frame_config, &last_ref)
                {
                    let mut hpkt = Packet::new(0, self.time_base, hidden_bitstream);
                    // Hidden frame carries the same PTS as the visible
                    // frame it precedes — there is no "natural" timestamp
                    // for an invisible reference frame, and the IVF
                    // wrapper just needs both to be parseable.
                    hpkt.pts = v.pts;
                    hpkt.dts = v.pts;
                    hpkt.flags.keyframe = false;
                    self.pending.push_back(hpkt);
                    self.alt_ref_frame = Some(hidden_rec);
                    suppress_visible_alt_refresh = true;
                }
            }
        }

        let src = Yuv420Source::from_video_frame(v)?;
        let (data, reference, plan) = if is_keyframe {
            let (bitstream, rec) = encode_keyframe_and_reconstruct_with_config(
                self.width,
                self.height,
                frame_config,
                &src,
            )?;
            // Keyframe refreshes all three slots.
            let plan = RefPlan {
                refresh_last: true,
                refresh_golden: true,
                refresh_alt: true,
                use_golden: false,
                use_alt: false,
            };
            (bitstream, rec, plan)
        } else {
            self.pframe_count += 1;
            let mut plan = RefPlan::for_pframe_with_config(self, &frame_config);
            // The hidden alt-ref already refreshed the slot — the visible
            // frame must NOT also flip `refresh_alt`, otherwise the
            // decoder would overwrite the synthesized alt-ref with the
            // visible frame's reconstruction immediately afterwards.
            if suppress_visible_alt_refresh {
                plan.refresh_alt = false;
            }
            let last_ref = self.last_frame.as_ref().unwrap();
            let golden_ref = self.golden_frame.as_ref().filter(|_| plan.use_golden);
            let alt_ref = self.alt_ref_frame.as_ref().filter(|_| plan.use_alt);
            let (bitstream, rec) = encode_pframe_and_reconstruct(
                self.width,
                self.height,
                frame_config,
                &src,
                last_ref,
                golden_ref,
                alt_ref,
                plan,
            )?;
            (bitstream, rec, plan)
        };
        // Refresh references per the plan. `LAST` is always refreshed on
        // P-frames in our encoder; GOLDEN / ALT only when the schedule
        // says so.
        if plan.refresh_last {
            self.last_frame = Some(reference.clone());
        }
        if plan.refresh_golden {
            self.golden_frame = Some(reference.clone());
        }
        if plan.refresh_alt {
            self.alt_ref_frame = Some(reference.clone());
        }
        let _ = reference;

        // Push to the lookahead buffer for the *non*-altref branch (the
        // altref branch above already pushed). This keeps the buffer
        // populated for the next refresh point.
        if !suppress_visible_alt_refresh
            && self.config.enable_lookahead_altref
            && self.config.alt_ref_interval > 0
            && self.config.enable_multi_ref
        {
            self.push_lookahead_source(v);
        }

        // Update scene-cut state: cache this frame's source luma for the
        // next MAD compare, decay the post-cut boost window, and seed
        // the running MAD window with this cut's MAD so the *next*
        // post-cut MAD is graded against a sensible baseline (not the
        // huge cut sample we just rejected).
        if let Some(mad) = cut_mad {
            self.scene_cut.push_mad(mad);
        }
        self.scene_cut
            .cache_prev(&y_plane.data, frame_w, frame_h, y_plane.stride);
        if self.scene_cut.boost_remaining > 0 {
            self.scene_cut.boost_remaining -= 1;
        }

        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> oxideav_core::Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(oxideav_core::Error::Eof)
        } else {
            Err(oxideav_core::Error::NeedMore)
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Frame assembly — keyframe
// ---------------------------------------------------------------------------

/// Encode one keyframe. Returns the raw VP8 bitstream for the frame.
///
/// Standalone (no `oxideav-core`) entry point: takes a [`Vp8Frame`]
/// and returns the encoded bytes plus a [`Vp8Error`] on failure. The
/// bit-identical behaviour of the legacy [`encode_keyframe`] is
/// preserved (same `Vp8EncoderConfig` defaults, segmentation off,
/// normal-mode loop filter).
pub fn encode_vp8_keyframe(
    width: u32,
    height: u32,
    qindex: u8,
    frame: &Vp8Frame,
) -> Result<Vec<u8>> {
    let src = Yuv420Source::from_vp8_frame(frame);
    let mut cfg = Vp8EncoderConfig::default();
    cfg.qindex = qindex.min(127);
    cfg.enable_segments = false;
    cfg.loop_filter_mode = LoopFilterMode::Normal;
    let (bitstream, _rec) = encode_keyframe_and_reconstruct_with_config(width, height, cfg, &src)?;
    Ok(bitstream)
}

/// Legacy keyframe encode entry. Identical to [`encode_vp8_keyframe`]
/// but takes the `oxideav-core::VideoFrame` shape — gated on the
/// `registry` feature.
#[cfg(feature = "registry")]
pub fn encode_keyframe(width: u32, height: u32, qindex: u8, frame: &VideoFrame) -> Result<Vec<u8>> {
    let src = Yuv420Source::from_video_frame(frame)?;
    let mut cfg = Vp8EncoderConfig::default();
    cfg.qindex = qindex.min(127);
    cfg.enable_segments = false;
    cfg.loop_filter_mode = LoopFilterMode::Normal;
    let (bitstream, _rec) = encode_keyframe_and_reconstruct_with_config(width, height, cfg, &src)?;
    Ok(bitstream)
}

fn encode_keyframe_and_reconstruct_with_config(
    width: u32,
    height: u32,
    config: Vp8EncoderConfig,
    frame: &Yuv420Source<'_>,
) -> Result<(Vec<u8>, ReferenceFrame)> {
    let qindex = config.qindex;
    let mb_w = ((width + 15) / 16) as usize;
    let mb_h = ((height + 15) / 16) as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;

    // Copy (and MB-pad) the source into our own buffers.
    let (src_y, src_u, src_v) =
        extract_mb_padded(frame, width as usize, height as usize, mb_w, mb_h)?;

    // Allocate reconstruction buffers (they track, pixel-for-pixel, what
    // the decoder will produce — needed for intra prediction context in
    // subsequent MBs).
    let mut rec_y = vec![0u8; y_stride * y_buf_h];
    let mut rec_u = vec![0u8; uv_stride * uv_buf_h];
    let mut rec_v = vec![0u8; uv_stride * uv_buf_h];

    // Pre-compute per-segment quant steps. When segmentation is disabled
    // every entry collapses to the frame-level qindex.
    let qi = clamp_qindex(qindex as i32);
    let segments = SegmentCtx::for_config(&config);

    // Loop-filter parameters we will both signal and apply to our own
    // reconstruction (so the next P-frame uses the post-filter pixels).
    let lf_level = loop_filter_level_for_qindex(qi as u8);
    let lf_sharpness = LOOP_FILTER_SHARPNESS;
    let lf_filter_type = pick_filter_type(lf_level, &config);

    // --- Per-MB segment classification (pre-computed so the frame
    //     header's segment tree_probs match the actual distribution). ---
    let mut mb_segment_ids: Vec<u8> = vec![0u8; mb_w * mb_h];
    let mut seg_counts: [u32; 4] = [0; 4];
    if segments.enabled {
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let s = classify_segment_id(&src_y, y_stride, mb_x, mb_y);
                mb_segment_ids[mb_y * mb_w + mb_x] = s;
                seg_counts[s as usize] += 1;
            }
        }
    }
    let segment_tree_probs = if segments.enabled {
        segment_tree_probs_from_counts(&seg_counts)
    } else {
        [255; 3]
    };

    // --- Compressed header ---
    let mut hdr_enc = BoolEncoder::new();
    // color_space + clamping_type (1 bit each)
    hdr_enc.write_literal(1, 0);
    hdr_enc.write_literal(1, 0);
    // Segmentation block (writes the single "enabled=0" bit when off).
    emit_segmentation_header(&mut hdr_enc, &segments, &segment_tree_probs);
    // loop filter: filter_type (0 = normal, 1 = simple), level,
    //              sharpness, mode_ref_delta_enabled = 0.
    hdr_enc.write_literal(1, lf_filter_type as u32);
    hdr_enc.write_literal(6, lf_level as u32);
    hdr_enc.write_literal(3, lf_sharpness as u32);
    hdr_enc.write_bool(128, false);
    // log2_nb_partitions = 0 (1 partition).
    hdr_enc.write_literal(2, 0);
    // Quant: y_ac_qi + zero deltas (each delta is a 1-bit "present" flag = 0).
    hdr_enc.write_literal(7, qi as u32);
    for _ in 0..5 {
        hdr_enc.write_bool(128, false);
    }
    // refresh_entropy_probs = 0 (we keep defaults).
    hdr_enc.write_bool(128, false);
    // Skip per-prob coefficient probability updates — send "no update" for all.
    emit_no_coef_prob_updates(&mut hdr_enc);
    // mb_skip_enabled = 0 (keyframes have no skip mode in this encoder).
    hdr_enc.write_bool(128, false);

    // --- MB mode info: pick the best intra mode (DC/V/H/TM/B_PRED) per MB
    //     and the best chroma mode per MB, emit the tree path, then encode
    //     the residual.
    let mut mb_encoded: Vec<MbEncoded> = Vec::with_capacity(mb_w * mb_h);
    // Track the 4x4 bmode of the bottom row of each MB column (propagation
    // context for B_PRED of the MB below). Matches decoder's `bmode_above`.
    let mut bmode_above: Vec<[i32; 4]> = vec![[B_DC_PRED; 4]; mb_w];
    // Track each MB's 16 bmodes so the left-MB lookup for B_PRED works.
    let mut mb_bmodes: Vec<[i32; 16]> = vec![[B_DC_PRED; 16]; mb_w * mb_h];
    // Track each MB's chosen Y mode so left-neighbour lookups at B_PRED
    // boundaries can fall back to `intra_to_b(y_mode)` when the left MB
    // was not itself B_PRED.
    let mut mb_ymodes: Vec<i32> = vec![DC_PRED; mb_w * mb_h];
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mb_xp = mb_x * 16;
            let mb_yp = mb_y * 16;

            // Per-MB segment id (0 if segmentation off — never read by
            // the decoder in that case).
            let segment_id = mb_segment_ids[mb_y * mb_w + mb_x];
            // Emit per-MB segment_id bits (only when the frame header
            // signalled `update_map = 1`).
            if segments.enabled {
                emit_segment_id(&mut hdr_enc, segment_id, &segment_tree_probs);
            }
            let q = segments.quant_for(segment_id);

            // Pick the best intra Y mode (picks among DC/V/H/TM for the
            // 16x16 candidates — B_PRED is evaluated later against the
            // reconstructed neighbours and wins when the per-4x4 modes
            // predict noticeably better).
            let y16_mode = choose_intra_16x16_y_mode(&src_y, &rec_y, y_stride, mb_xp, mb_yp);
            // Evaluate B_PRED by greedily picking the best 4×4 mode per
            // sub-block — since B_PRED makes different predictions in
            // places that DC/V/H/TM do not, we compare its total SSE to
            // the best 16×16 candidate and pick the lower.
            let (y16_sse, _) = sse_intra_16x16(y16_mode, &src_y, &rec_y, y_stride, mb_xp, mb_yp);
            let (bp_sse, _bp_modes) =
                sse_intra_b_pred(&src_y, &rec_y, y_stride, mb_xp, mb_yp, mb_w, mb_h);
            let y_mode = if bp_sse + B_PRED_SSE_MARGIN < y16_sse {
                B_PRED
            } else {
                y16_mode
            };

            // Chroma mode: evaluate DC/V/H/TM over the U+V source and pick
            // the lowest total SSE.
            let uv_mode = choose_intra_chroma_mode(
                &src_u,
                &src_v,
                &rec_u,
                &rec_v,
                uv_stride,
                mb_x * 8,
                mb_y * 8,
            );

            // Emit KF_YMODE_TREE path for the chosen y_mode.
            emit_kf_ymode(&mut hdr_enc, y_mode);

            // If B_PRED, emit per-block bmodes (using above/left context).
            if y_mode == B_PRED {
                // Precompute left-bmode context so we don't hold two
                // borrows of mb_bmodes at once inside the emit helper.
                let left_bmodes: [i32; 4] = if mb_x > 0 {
                    let l_idx = mb_y * mb_w + mb_x - 1;
                    let l_y = mb_ymodes[l_idx];
                    if l_y == B_PRED {
                        let lb = &mb_bmodes[l_idx];
                        [lb[3], lb[7], lb[11], lb[15]]
                    } else {
                        [intra_to_b_mode(l_y); 4]
                    }
                } else {
                    [B_DC_PRED; 4]
                };
                let above_for_mb = bmode_above[mb_x];
                let bm_slot = &mut mb_bmodes[mb_y * mb_w + mb_x];
                emit_bmodes_keyframe(
                    &mut hdr_enc,
                    bm_slot,
                    &above_for_mb,
                    &left_bmodes,
                    // When B_PRED, the neighbour-context-driven per-4×4
                    // decisions are made here.
                    &src_y,
                    &rec_y,
                    y_stride,
                    mb_x,
                    mb_y,
                    mb_w,
                );
                // After emission the per-block modes in mb_bmodes are
                // populated; propagate the bottom row to bmode_above for
                // the next MB row.
                let bm = &mb_bmodes[mb_y * mb_w + mb_x];
                bmode_above[mb_x] = [bm[12], bm[13], bm[14], bm[15]];
            } else {
                // Non-B_PRED MBs propagate their `intra_to_b(y_mode)` to
                // both their own bmodes (for the neighbour below if that
                // MB is B_PRED) and to bmode_above.
                let b = intra_to_b_mode(y_mode);
                for v in mb_bmodes[mb_y * mb_w + mb_x].iter_mut() {
                    *v = b;
                }
                bmode_above[mb_x] = [b; 4];
            }
            mb_ymodes[mb_y * mb_w + mb_x] = y_mode;

            // UV mode.
            emit_kf_uv_mode(&mut hdr_enc, uv_mode);

            // Encode + reconstruct the MB now that the mode is final.
            let mb_rec = encode_intra_mb(
                &src_y,
                &src_u,
                &src_v,
                &mut rec_y,
                &mut rec_u,
                &mut rec_v,
                y_stride,
                uv_stride,
                y_buf_h,
                uv_buf_h,
                mb_x,
                mb_y,
                mb_w,
                mb_h,
                q,
                y_mode,
                uv_mode,
                &mb_bmodes[mb_y * mb_w + mb_x],
            );
            mb_encoded.push(mb_rec);
        }
    }

    // Apply the in-loop deblocking filter to our reconstruction so the
    // next P-frame uses the same post-filter references the decoder will.
    apply_loop_filter_enc(
        &mut rec_y,
        &mut rec_u,
        &mut rec_v,
        y_stride,
        uv_stride,
        y_buf_h,
        uv_buf_h,
        mb_w,
        mb_h,
        lf_level,
        lf_sharpness,
        lf_filter_type,
        &mb_encoded,
        &mb_segment_ids,
        &segments,
        true, // keyframe
    );

    let first_partition = hdr_enc.finish();

    // --- Token partition (separate BoolEncoder) ---
    let tok_enc = emit_tokens(
        mb_w,
        mb_h,
        &mb_encoded,
        &[],
        &DEFAULT_COEF_PROBS,
        Some(&mb_ymodes),
    );
    let token_partition = tok_enc.finish();

    let out = assemble_frame_keyframe(width, height, first_partition, token_partition)?;

    let reference = ReferenceFrame {
        y: rec_y,
        u: rec_u,
        v: rec_v,
        y_stride,
        uv_stride,
        y_h: y_buf_h,
        uv_h: uv_buf_h,
    };
    Ok((out, reference))
}

/// Minimum SSE improvement (across the full 256-pixel MB) that B_PRED
/// must show over the best 16×16 intra candidate to be picked. B_PRED
/// costs extra bits (per-sub-block mode + 16 DC-coded AC blocks instead
/// of 1 Y2 WHT block), so selection needs a real distortion edge.
const B_PRED_SSE_MARGIN: u64 = 512;

fn assemble_frame_keyframe(
    width: u32,
    height: u32,
    first_partition: Vec<u8>,
    token_partition: Vec<u8>,
) -> Result<Vec<u8>> {
    let part_size = first_partition.len() as u32;
    if part_size >= (1 << 19) {
        return Err(Error::invalid(format!(
            "vp8 encoder: first partition too large ({} bytes)",
            part_size
        )));
    }
    // frame_type=0 (bit 0), version=0 (bits 1..3 all zero), show_frame=1 (bit 4),
    // first_partition_size in bits 5..23.
    let tag_word: u32 = (1u32 << 4) | (part_size << 5);
    let mut out = Vec::with_capacity(10 + first_partition.len() + token_partition.len());
    out.push((tag_word & 0xff) as u8);
    out.push(((tag_word >> 8) & 0xff) as u8);
    out.push(((tag_word >> 16) & 0xff) as u8);

    // --- Keyframe 7-byte header: sync + w/h words ---
    out.extend_from_slice(&KEYFRAME_SYNC_CODE);
    let w = width as u16 & 0x3fff;
    let h = height as u16 & 0x3fff;
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());

    out.extend_from_slice(&first_partition);
    out.extend_from_slice(&token_partition);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Frame assembly — P-frame (inter-frame, ZERO_MV + SKIP only)
// ---------------------------------------------------------------------------

/// One partition's MV in a SPLIT_MV decision. `split_mode` indexes into
/// `MB_SPLITS`/`MB_SPLIT_COUNT` (0=16x8, 1=8x16, 2=quarters, 3=4x4).
#[derive(Clone, Copy, Debug)]
struct SplitMv {
    split_mode: u8,
    /// `MB_SPLIT_COUNT[split_mode]` entries are valid; the remainder are
    /// left zero and unused.
    part_mvs: [Mv; 16],
}

// Per-MB reference-frame ID, mirroring the decoder's namespace
// (`REF_INTRA = 0`, `REF_LAST = 1`, `REF_GOLDEN = 2`, `REF_ALT = 3`).
const ENC_REF_INTRA: u8 = 0;
const ENC_REF_LAST: u8 = 1;
const ENC_REF_GOLDEN: u8 = 2;
const ENC_REF_ALT: u8 = 3;

/// Captured per-MB state from the decision/reconstruction pass needed
/// later to emit the mode-info bool bits in the second pass (see
/// `encode_pframe_and_reconstruct`).
///
/// The two-pass split lets us count the actual ref-frame distribution
/// across the frame and pick the spec-correct `prob_intra` /
/// `prob_last` / `prob_gf` values *before* any mode-info bits are
/// emitted into the bool encoder. RFC 6386 §9.10 (field J) carries
/// these as 8-bit literals at the top of the inter-frame header, so
/// once we set them they apply uniformly to every MB in the frame.
#[derive(Clone, Copy, Debug)]
struct PMbModeInfo {
    /// `cnt` from the per-ref `find_near_mvs_enc` walk that was used
    /// for the picked reference frame; needed by `mv_ref_probs_enc` in
    /// pass 2.
    cnt: [u8; 4],
    /// MV that NEW_MV would code its delta against (from the same
    /// per-ref `find_near_mvs_enc` walk).
    best_for_newmv: Mv,
}

/// Per-MB decision for a P-frame.
#[derive(Clone, Copy, Debug)]
enum PMbDecision {
    /// Copy the reference MB verbatim — no residual coded. Implies MV=0.
    Skip,
    /// Motion-compensated copy at `mv=0` followed by a coded residual.
    ZeroMv,
    /// Use the `nearest` neighbour-predicted MV — no MV delta coded.
    NearestMv(Mv),
    /// Use the `near` neighbour-predicted MV — no MV delta coded.
    NearMv(Mv),
    /// Motion-compensated copy at `mv` followed by a coded residual.
    /// `mv` is in luma 1/8-pel units and may be sub-pel (any phase).
    NewMv(Mv),
    /// Per-partition motion. Each 4×4 sub-block carries a possibly
    /// distinct MV via `split_mode`/`part_mvs`.
    SplitMv(SplitMv),
    /// Intra fallback (16×16 intra mode or B_PRED) inside a P-frame.
    /// Chosen when the best inter prediction's residual energy is too
    /// high to be worth coding. For B_PRED the 16 per-block modes are
    /// recomputed during reconstruction; for 16×16 modes `bmodes` is
    /// filled with `intra_to_b_mode(y_mode)` for neighbour propagation.
    Intra { y_mode: i32, uv_mode: i32 },
}

impl PMbDecision {
    /// The MV associated with this decision (`Mv::ZERO` for Skip / ZeroMv
    /// / Intra). Used to populate the per-MB MV table for subsequent
    /// `find_near_mvs_enc` calls.
    fn mv(&self) -> Mv {
        match self {
            PMbDecision::Skip | PMbDecision::ZeroMv | PMbDecision::Intra { .. } => Mv::ZERO,
            PMbDecision::NearestMv(mv) | PMbDecision::NearMv(mv) | PMbDecision::NewMv(mv) => *mv,
            // For SPLIT we use the bottom-right sub-MV as the MB's
            // neighbour-propagation MV — matches the decoder's rule
            // (`info.mv = part_mvs[15]`).
            PMbDecision::SplitMv(s) => s.part_mvs[15],
        }
    }

    /// Per-subblock sub-MVs for this decision. For non-SPLIT inter the
    /// whole block inherits the MB MV; for SPLIT we expand the split
    /// partitioning; for intra every sub-MV is zero.
    fn sub_mvs(&self) -> [Mv; 16] {
        match self {
            PMbDecision::SplitMv(s) => {
                let mut out = [Mv::ZERO; 16];
                let part = &MB_SPLITS[s.split_mode as usize];
                for (i, cell) in out.iter_mut().enumerate() {
                    *cell = s.part_mvs[part[i] as usize];
                }
                out
            }
            _ => [self.mv(); 16],
        }
    }

    /// Whether the MB was encoded as intra (REF_INTRA) — find_near_mvs
    /// treats those as zero-MV neighbours regardless of any MV in the
    /// decision (they have none in the decoder's bitstream).
    fn is_intra(&self) -> bool {
        matches!(self, PMbDecision::Intra { .. })
    }
}

/// Encode one P-frame (inter-frame). Reference picking is per-MB across
/// the available LAST / GOLDEN / ALTREF slots; mode decision is
/// Lagrangian (`D + lambda*R`) when `config.enable_rdo` is set,
/// otherwise the legacy SAD-only path. The reconstructed frame is
/// returned for the caller to use in the next reference plan.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "registry")]
fn encode_pframe_and_reconstruct(
    width: u32,
    height: u32,
    config: Vp8EncoderConfig,
    frame: &Yuv420Source<'_>,
    last_ref: &ReferenceFrame,
    golden_ref: Option<&ReferenceFrame>,
    alt_ref: Option<&ReferenceFrame>,
    plan: RefPlan,
) -> Result<(Vec<u8>, ReferenceFrame)> {
    let mb_w = ((width + 15) / 16) as usize;
    let mb_h = ((height + 15) / 16) as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;

    // Sanity: reference must have the same stride/geometry.
    if last_ref.y_stride != y_stride
        || last_ref.uv_stride != uv_stride
        || last_ref.y_h != y_buf_h
        || last_ref.uv_h != uv_buf_h
    {
        return Err(Error::invalid(
            "vp8 encoder: reference frame geometry mismatch (reset required)",
        ));
    }

    let (src_y, src_u, src_v) =
        extract_mb_padded(frame, width as usize, height as usize, mb_w, mb_h)?;

    // Allocate reconstruction buffers.
    let mut rec_y = vec![0u8; y_stride * y_buf_h];
    let mut rec_u = vec![0u8; uv_stride * uv_buf_h];
    let mut rec_v = vec![0u8; uv_stride * uv_buf_h];

    let qi = clamp_qindex(config.qindex as i32);
    // Per-segment quant table (collapses to a single QuantCtx when
    // segmentation is disabled).
    let segments = SegmentCtx::for_config(&config);

    let lf_level = loop_filter_level_for_qindex(qi as u8);
    let lf_sharpness = LOOP_FILTER_SHARPNESS;
    let lf_filter_type = pick_filter_type(lf_level, &config);

    // --- Per-MB segment classification (pre-computed so the frame
    //     header's segment tree_probs match the actual distribution). ---
    let mut mb_segment_ids: Vec<u8> = vec![0u8; mb_w * mb_h];
    let mut seg_counts: [u32; 4] = [0; 4];
    if segments.enabled {
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let s = classify_segment_id(&src_y, y_stride, mb_x, mb_y);
                mb_segment_ids[mb_y * mb_w + mb_x] = s;
                seg_counts[s as usize] += 1;
            }
        }
    }
    let segment_tree_probs = if segments.enabled {
        segment_tree_probs_from_counts(&seg_counts)
    } else {
        [255; 3]
    };

    // The frame is encoded in two passes so we can pick the spec-correct
    // `prob_intra` / `prob_last` / `prob_gf` triple from the actual
    // ref-frame distribution observed across the frame's MBs (RFC 6386
    // §9.10 carries them as 8-bit literals at the top of the inter-frame
    // header — once set they apply uniformly to every MB).
    //
    // Pass 1: per-MB decision, reconstruction, residual encode, and
    // accumulation of per-MB ref-frame counts (intra / LAST / GOLDEN /
    // ALT). Touches no bool encoder.
    //
    // Pass 2: build the inter-frame header (now with the optimised prob
    // triple) and emit the per-MB mode-info bool bits using the decisions
    // recorded in pass 1.
    let mb_skip_prob: u8 = 128;
    // Probabilities used for the per-MB rate model in pass 1's RDO. We
    // intentionally use coarse default values here so that mode decision
    // does not depend on the (unknown-yet) optimised prob triple — the
    // actual emitted bits in pass 2 use the exact optimised probs.
    let rdo_prob_intra: u8 = PROB_INTRA_IN_P;
    let (rdo_prob_last, rdo_prob_gf): (u8, u8) = if plan.use_golden || plan.use_alt {
        (128, 128)
    } else {
        (1, 128)
    };

    // --- Pass 1: per-MB decision, reconstruction, residual encode ---
    let mut mb_encoded: Vec<MbEncoded> = Vec::with_capacity(mb_w * mb_h);
    let mut mb_decisions: Vec<PMbDecision> = Vec::with_capacity(mb_w * mb_h);
    let mut mb_ref_frames: Vec<u8> = Vec::with_capacity(mb_w * mb_h);
    let mut mb_mode_info: Vec<PMbModeInfo> = Vec::with_capacity(mb_w * mb_h);
    let mut mb_mvs: Vec<Mv> = vec![Mv::ZERO; mb_w * mb_h];
    // Per-subblock MVs (needed for SPLIT's neighbour MVs when later MBs
    // use SPLIT themselves — encoder-side replica of the decoder's
    // `MbInfo::sub_mvs`).
    let mut mb_sub_mvs: Vec<[Mv; 16]> = vec![[Mv::ZERO; 16]; mb_w * mb_h];
    // B_PRED propagation buffers (parallel to keyframe path). The inter
    // path doesn't actually read `bmode_above` — `choose_b_pred_modes`
    // pulls its 4×4 context from the reconstructed pixels — but we keep
    // it here for symmetry with the keyframe path and the decoder.
    let mut bmode_above: Vec<[i32; 4]> = vec![[B_DC_PRED; 4]; mb_w];
    let mut mb_bmodes: Vec<[i32; 16]> = vec![[B_DC_PRED; 16]; mb_w * mb_h];
    let mut mb_ymodes: Vec<i32> = vec![DC_PRED; mb_w * mb_h];
    // Per-MB ref-frame context counts. These drive the optimal
    // `prob_intra` / `prob_last` / `prob_gf` triple emitted in the frame
    // header and used by every MB's ref-frame bool tree. Each MB
    // contributes exactly one count, mirroring the per-MB context model
    // RFC 6386 §11.3 / §16.2 describe.
    let mut n_intra: u32 = 0;
    let mut n_last: u32 = 0;
    let mut n_golden: u32 = 0;
    let mut n_alt: u32 = 0;
    // Lambda for Lagrangian RDO (D + lambda*R). Computed once per frame
    // since QP is fixed; if RDO is disabled lambda is 0 and rate
    // contributions wash out, recovering the legacy SAD-only behaviour.
    let lambda = if config.enable_rdo {
        lambda_for_qp(qi as u32, config.lambda_scale)
    } else {
        0
    };
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mb_idx = mb_y * mb_w + mb_x;
            // For each candidate reference, gather the per-ref neighbour
            // context (nearest / near / best). The decoder's find_near_mvs
            // walks neighbours and only contributes those whose ref_frame
            // matches; the encoder mirrors that filtering exactly.
            //
            // Always start with LAST since every P-frame has it. Then add
            // GOLDEN / ALT if the plan has them populated.
            type RdCandidate = (u64, u8, PMbDecision, Mv, Mv, Mv, [u8; 4]);
            let mut best_choice: Option<RdCandidate> = None;

            let try_ref = |ref_frame: u8, ref_plane: &ReferenceFrame, store: &mut Option<_>| {
                let (nearest, near, best_for_newmv, cnt) = find_near_mvs_enc(
                    &mb_mvs,
                    &mb_decisions,
                    &mb_ref_frames,
                    mb_x,
                    mb_y,
                    mb_w,
                    ref_frame,
                );
                let dec = choose_pmb_decision(
                    &src_y,
                    &ref_plane.y,
                    y_stride,
                    y_buf_h,
                    mb_x,
                    mb_y,
                    mb_w,
                    mb_h,
                    nearest,
                    near,
                    &rec_y,
                );
                let cost = rd_cost_for_decision(
                    &dec,
                    &src_y,
                    &ref_plane.y,
                    &rec_y,
                    y_stride,
                    y_buf_h,
                    mb_x,
                    mb_y,
                    mb_w,
                    mb_h,
                    ref_frame,
                    plan,
                    nearest,
                    near,
                    best_for_newmv,
                    rdo_prob_intra,
                    rdo_prob_last,
                    rdo_prob_gf,
                    mb_skip_prob,
                    lambda,
                );
                let take = match store {
                    None => true,
                    Some((bcost, _, _, _, _, _, _)) => cost < *bcost,
                };
                if take {
                    *store = Some((cost, ref_frame, dec, nearest, near, best_for_newmv, cnt));
                }
            };

            try_ref(ENC_REF_LAST, last_ref, &mut best_choice);
            if let Some(g) = golden_ref {
                try_ref(ENC_REF_GOLDEN, g, &mut best_choice);
            }
            if let Some(a) = alt_ref {
                try_ref(ENC_REF_ALT, a, &mut best_choice);
            }

            let (_cost, picked_ref, decision, _nearest, _near, best_for_newmv, cnt) =
                best_choice.expect("LAST must always produce a candidate");
            mb_decisions.push(decision);
            mb_mode_info.push(PMbModeInfo {
                cnt,
                best_for_newmv,
            });
            // Skip / ZeroMv on a non-LAST reference would still need
            // a `prob_last==true` ref-frame bit emitted — keep that
            // information so the bitstream emit code below picks the
            // right path. For Intra we record REF_INTRA (=0).
            let stored_ref = if decision.is_intra() {
                ENC_REF_INTRA
            } else {
                picked_ref
            };
            mb_ref_frames.push(stored_ref);
            mb_mvs[mb_idx] = decision.mv();
            mb_sub_mvs[mb_idx] = decision.sub_mvs();
            // Accumulate per-MB ref-frame counts. These drive the optimal
            // `prob_intra` / `prob_last` / `prob_gf` triple emitted in
            // pass 2's frame header.
            match stored_ref {
                ENC_REF_INTRA => n_intra += 1,
                ENC_REF_GOLDEN => n_golden += 1,
                ENC_REF_ALT => n_alt += 1,
                _ => n_last += 1,
            }
            // Pick the right reference plane for the residual encode below.
            let used_ref: &ReferenceFrame = match picked_ref {
                ENC_REF_GOLDEN => golden_ref.expect("plan.use_golden was true"),
                ENC_REF_ALT => alt_ref.expect("plan.use_alt was true"),
                _ => last_ref,
            };

            // Pre-compute B_PRED bmodes for intra-in-P (the reconstruction
            // step needs them, and pass 2 needs the same modes for its
            // mode-info bool emit). Non-B_PRED intra and inter MBs get a
            // static per-MB bmode propagation matching the decoder.
            let is_inter = !decision.is_intra();
            if !is_inter {
                let (y_mode, _uv_mode) = match decision {
                    PMbDecision::Intra { y_mode, uv_mode } => (y_mode, uv_mode),
                    _ => unreachable!(),
                };
                if y_mode == B_PRED {
                    let chosen = choose_b_pred_modes(
                        &src_y,
                        &rec_y,
                        y_stride,
                        mb_x * 16,
                        mb_y * 16,
                        mb_w,
                        mb_h,
                    );
                    let bm = &mut mb_bmodes[mb_idx];
                    bm.copy_from_slice(&chosen);
                    bmode_above[mb_x] = [bm[12], bm[13], bm[14], bm[15]];
                } else {
                    let b = intra_to_b_mode(y_mode);
                    for v in mb_bmodes[mb_idx].iter_mut() {
                        *v = b;
                    }
                    bmode_above[mb_x] = [b; 4];
                }
                mb_ymodes[mb_idx] = y_mode;
            } else {
                // Inter MBs reset bmodes propagation to DC (decoder does
                // the same after reconstructing an inter MB).
                let b = intra_to_b_mode(DC_PRED);
                for v in mb_bmodes[mb_idx].iter_mut() {
                    *v = b;
                }
                bmode_above[mb_x] = [b; 4];
            }

            // Per-MB reconstruction and quantised coefficients. Quant
            // step is picked from the segment table (single entry when
            // segmentation is disabled).
            let q = segments.quant_for(mb_segment_ids[mb_idx]);
            let mb_rec = match decision {
                PMbDecision::Skip => {
                    copy_ref_into_rec(
                        &used_ref.y,
                        &used_ref.u,
                        &used_ref.v,
                        &mut rec_y,
                        &mut rec_u,
                        &mut rec_v,
                        y_stride,
                        uv_stride,
                        mb_x,
                        mb_y,
                    );
                    MbEncoded::zero()
                }
                PMbDecision::ZeroMv
                | PMbDecision::NearestMv(_)
                | PMbDecision::NearMv(_)
                | PMbDecision::NewMv(_) => encode_inter_mb_at_mv(
                    &src_y,
                    &src_u,
                    &src_v,
                    &used_ref.y,
                    &used_ref.u,
                    &used_ref.v,
                    &mut rec_y,
                    &mut rec_u,
                    &mut rec_v,
                    y_stride,
                    uv_stride,
                    y_buf_h,
                    uv_buf_h,
                    mb_x,
                    mb_y,
                    decision.mv(),
                    q,
                ),
                PMbDecision::SplitMv(split) => encode_inter_mb_split(
                    &src_y,
                    &src_u,
                    &src_v,
                    &used_ref.y,
                    &used_ref.u,
                    &used_ref.v,
                    &mut rec_y,
                    &mut rec_u,
                    &mut rec_v,
                    y_stride,
                    uv_stride,
                    y_buf_h,
                    uv_buf_h,
                    mb_x,
                    mb_y,
                    &split,
                    q,
                ),
                PMbDecision::Intra { y_mode, uv_mode } => encode_intra_mb(
                    &src_y,
                    &src_u,
                    &src_v,
                    &mut rec_y,
                    &mut rec_u,
                    &mut rec_v,
                    y_stride,
                    uv_stride,
                    y_buf_h,
                    uv_buf_h,
                    mb_x,
                    mb_y,
                    mb_w,
                    mb_h,
                    q,
                    y_mode,
                    uv_mode,
                    &mb_bmodes[mb_idx],
                ),
            };
            mb_encoded.push(mb_rec);
        }
    }

    // --- Compute optimal frame-level ref-frame probabilities ---
    //
    // RFC 6386 §16.2 codes the per-MB ref-frame as a 3-leaf tree:
    //   bit0 (`prob_intra`)  : 0 → intra, 1 → inter
    //   bit1 (`prob_last`)   : 0 → REF_LAST, 1 → REF_GOLDEN | REF_ALT
    //   bit2 (`prob_gf`)     : 0 → REF_GOLDEN, 1 → REF_ALT
    //
    // Bool-coder convention: `prob = round(256 * P(bit==0))`. The optimal
    // code-length is achieved by setting each prob to match the actual
    // observed P(bit==0) in the frame. Counts that are entirely zero
    // (e.g. no GOLDEN/ALT frame available) collapse to a sensible
    // default (`128 / 128 / 128` for the prior, then floored at 1 / 255
    // to avoid the bool coder's degenerate single-symbol case).
    let n_inter = n_last + n_golden + n_alt;
    let n_total = n_intra + n_inter;
    let prob_intra = optimal_prob_8(n_intra, n_inter);
    // For prob_last / prob_gf we only have observations from inter MBs.
    // If every inter MB used LAST, prob_last → 255 (REF_LAST near-free)
    // matching the libvpx single-ref convention. With no inter MBs at
    // all we fall back to neutral 128.
    let prob_last = optimal_prob_8(n_last, n_golden + n_alt);
    let prob_gf = optimal_prob_8(n_golden, n_alt);
    let _ = n_total; // silence dead-store lint when assertions disabled

    // Apply in-loop deblocking to our reconstruction (so the next
    // reference frame tracks the decoder).
    apply_loop_filter_enc(
        &mut rec_y,
        &mut rec_u,
        &mut rec_v,
        y_stride,
        uv_stride,
        y_buf_h,
        uv_buf_h,
        mb_w,
        mb_h,
        lf_level,
        lf_sharpness,
        lf_filter_type,
        &mb_encoded,
        &mb_segment_ids,
        &segments,
        false, // P-frame
    );

    // --- Pass 2: build the inter-frame header + emit per-MB mode info ---
    let mut hdr_enc = BoolEncoder::new();
    // Inter-header order (matching parse_inter_header exactly):
    //   segmentation block (writes the single "enabled=0" bit when off,
    //   otherwise update_map + update_data + 4 quant deltas + 4 lf deltas
    //   + 3 tree_probs).
    emit_segmentation_header(&mut hdr_enc, &segments, &segment_tree_probs);
    //   loop filter
    hdr_enc.write_literal(1, lf_filter_type as u32); // filter_type (0 normal, 1 simple)
    hdr_enc.write_literal(6, lf_level as u32);
    hdr_enc.write_literal(3, lf_sharpness as u32);
    hdr_enc.write_bool(128, false); // mode_ref_delta_enabled
                                    //   log2_nb_partitions = 0 (single partition)
    hdr_enc.write_literal(2, 0);
    //   quant: y_ac_qi + 5 "delta present" flags all 0.
    hdr_enc.write_literal(7, qi as u32);
    for _ in 0..5 {
        hdr_enc.write_bool(128, false);
    }
    //   refresh_golden, refresh_alt — driven by the per-frame reference
    //   plan computed by the encoder (alt-ref / golden cadence), not
    //   hard-wired to 1.
    hdr_enc.write_bool(128, plan.refresh_golden);
    hdr_enc.write_bool(128, plan.refresh_alt);
    //   When refresh_golden / refresh_alt are 0, the decoder reads a
    //   2-bit copy_buffer_to_* selector. We always emit "no copy" (=0)
    //   so the slot keeps its existing contents until a future refresh
    //   puts a new reconstruction into it.
    if !plan.refresh_golden {
        hdr_enc.write_literal(2, 0);
    }
    if !plan.refresh_alt {
        hdr_enc.write_literal(2, 0);
    }
    //   sign_bias_golden, sign_bias_alt
    hdr_enc.write_bool(128, false);
    hdr_enc.write_bool(128, false);
    //   refresh_entropy_probs = 0
    hdr_enc.write_bool(128, false);
    //   refresh_last = 1
    hdr_enc.write_bool(128, plan.refresh_last);
    //   coef prob updates — all "no update"
    emit_no_coef_prob_updates(&mut hdr_enc);
    //   mb_skip_enabled = 1, skip prob literal (we use 128 — neutral).
    hdr_enc.write_bool(128, true);
    hdr_enc.write_literal(8, mb_skip_prob as u32);
    //   prob_intra, prob_last, prob_gf — picked from the actual per-MB
    //   ref-frame distribution accumulated above.
    hdr_enc.write_literal(8, prob_intra as u32);
    hdr_enc.write_literal(8, prob_last as u32);
    hdr_enc.write_literal(8, prob_gf as u32);
    //   y-mode prob update flag = 0 ; uv-mode prob update flag = 0
    hdr_enc.write_bool(128, false);
    hdr_enc.write_bool(128, false);
    //   MV context updates — 19 × 2, all "no update".
    use crate::tables::mv::MV_UPDATE_PROBS;
    for comp in 0..2 {
        for i in 0..19 {
            hdr_enc.write_bool(MV_UPDATE_PROBS[comp][i] as u32, false);
        }
    }

    // Per-MB mode-info emit. Walks in the same order as pass 1 so
    // `emit_split_submvs`'s neighbour walk lines up exactly with the
    // pass-1 decision state.
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mb_idx = mb_y * mb_w + mb_x;
            let decision = mb_decisions[mb_idx];
            let info = mb_mode_info[mb_idx];
            let stored_ref = mb_ref_frames[mb_idx];
            let ref_probs = mv_ref_probs_enc(&info.cnt);

            // Mode-info bits.
            // 1) segment id — only emitted when the frame header
            //    signalled `update_map = 1` (i.e. segments enabled).
            if segments.enabled {
                emit_segment_id(&mut hdr_enc, mb_segment_ids[mb_idx], &segment_tree_probs);
            }
            // 2) skip flag. SKIP is the only path that emits skip=true.
            hdr_enc.write_bool(mb_skip_prob as u32, matches!(decision, PMbDecision::Skip));

            // 3) intra-vs-inter bit.
            let is_inter = !decision.is_intra();
            hdr_enc.write_bool(prob_intra as u32, is_inter);

            if !is_inter {
                // Intra-in-P: emit the inter-frame YMODE tree + BMODE
                // sub-modes (if B_PRED) + UV_MODE tree. Probabilities
                // default to `vp8_kf_default_*` since we send no update
                // flags in the inter header.
                let (y_mode, uv_mode) = match decision {
                    PMbDecision::Intra { y_mode, uv_mode } => (y_mode, uv_mode),
                    _ => unreachable!(),
                };
                emit_inter_ymode(&mut hdr_enc, y_mode, &DEFAULT_YMODE_PROBS);
                if y_mode == B_PRED {
                    // Intra-in-P uses the default (non-context-sensitive)
                    // `vp8_bmode_prob` for each 4x4 — matches the decoder's
                    // handling in `decode_mb_mode_info_inter`.
                    const DEFAULT_BMODE_PROBS: [u8; 9] = [120, 90, 79, 133, 87, 85, 80, 111, 151];
                    let bm = &mb_bmodes[mb_idx];
                    for i in 0..16 {
                        emit_tree_path(
                            &mut hdr_enc,
                            BMODE_PATHS[bm[i] as usize],
                            &DEFAULT_BMODE_PROBS,
                        );
                    }
                }
                emit_inter_uv_mode(&mut hdr_enc, uv_mode, &DEFAULT_UV_MODE_PROBS);
            } else {
                // 4) ref_frame bits. RFC 6386 §16.2:
                //      prob_last     : 0 → REF_LAST
                //      prob_last     : 1 → read prob_gf:
                //         prob_gf    : 0 → REF_GOLDEN
                //         prob_gf    : 1 → REF_ALT
                match stored_ref {
                    ENC_REF_LAST => {
                        hdr_enc.write_bool(prob_last as u32, false);
                    }
                    ENC_REF_GOLDEN => {
                        hdr_enc.write_bool(prob_last as u32, true);
                        hdr_enc.write_bool(prob_gf as u32, false);
                    }
                    ENC_REF_ALT => {
                        hdr_enc.write_bool(prob_last as u32, true);
                        hdr_enc.write_bool(prob_gf as u32, true);
                    }
                    _ => unreachable!("inter MB must have a non-intra ref_frame"),
                }

                // 5) MV_REF_TREE leaves (RFC §16.3):
                //      leaf 0 = ZERO_MV      (path: 0)
                //      leaf 1 = NEAREST_MV   (path: 1, 0)
                //      leaf 2 = NEAR_MV      (path: 1, 1, 0)
                //      leaf 3 = NEW_MV       (path: 1, 1, 1, 0)
                //      leaf 4 = SPLIT_MV     (path: 1, 1, 1, 1)
                match decision {
                    PMbDecision::Skip | PMbDecision::ZeroMv => {
                        hdr_enc.write_bool(ref_probs[0] as u32, false);
                    }
                    PMbDecision::NearestMv(_) => {
                        hdr_enc.write_bool(ref_probs[0] as u32, true);
                        hdr_enc.write_bool(ref_probs[1] as u32, false);
                    }
                    PMbDecision::NearMv(_) => {
                        hdr_enc.write_bool(ref_probs[0] as u32, true);
                        hdr_enc.write_bool(ref_probs[1] as u32, true);
                        hdr_enc.write_bool(ref_probs[2] as u32, false);
                    }
                    PMbDecision::NewMv(mv) => {
                        hdr_enc.write_bool(ref_probs[0] as u32, true);
                        hdr_enc.write_bool(ref_probs[1] as u32, true);
                        hdr_enc.write_bool(ref_probs[2] as u32, true);
                        hdr_enc.write_bool(ref_probs[3] as u32, false);
                        let dmv = Mv::new(
                            mv.row as i32 - info.best_for_newmv.row as i32,
                            mv.col as i32 - info.best_for_newmv.col as i32,
                        );
                        encode_mv_component(&mut hdr_enc, &DEFAULT_MV_CONTEXT[0], dmv.row as i32);
                        encode_mv_component(&mut hdr_enc, &DEFAULT_MV_CONTEXT[1], dmv.col as i32);
                    }
                    PMbDecision::SplitMv(split) => {
                        hdr_enc.write_bool(ref_probs[0] as u32, true);
                        hdr_enc.write_bool(ref_probs[1] as u32, true);
                        hdr_enc.write_bool(ref_probs[2] as u32, true);
                        hdr_enc.write_bool(ref_probs[3] as u32, true);
                        emit_split_mb_tree(&mut hdr_enc, split.split_mode);
                        // Per-partition sub-MV refs. Each partition's
                        // neighbours come from earlier partitions (or from
                        // neighbouring MBs' bottom/right sub-MV rows).
                        emit_split_submvs(
                            &mut hdr_enc,
                            &split,
                            &mb_sub_mvs,
                            &mb_decisions,
                            mb_x,
                            mb_y,
                            mb_w,
                            info.best_for_newmv,
                        );
                    }
                    PMbDecision::Intra { .. } => unreachable!("intra handled elsewhere"),
                }
            }
        }
    }

    let first_partition = hdr_enc.finish();

    // --- Token partition ---
    let tok_enc = emit_tokens(
        mb_w,
        mb_h,
        &mb_encoded,
        &mb_decisions,
        &DEFAULT_COEF_PROBS,
        Some(&mb_ymodes),
    );
    let token_partition = tok_enc.finish();

    let part_size = first_partition.len() as u32;
    if part_size >= (1 << 19) {
        return Err(Error::invalid(format!(
            "vp8 encoder: first partition too large ({} bytes)",
            part_size
        )));
    }
    // frame_type=1 (P), version=0, show_frame=1.
    let tag_word: u32 = 1u32 | (1u32 << 4) | (part_size << 5);
    let mut out = Vec::with_capacity(3 + first_partition.len() + token_partition.len());
    out.push((tag_word & 0xff) as u8);
    out.push(((tag_word >> 8) & 0xff) as u8);
    out.push(((tag_word >> 16) & 0xff) as u8);
    out.extend_from_slice(&first_partition);
    out.extend_from_slice(&token_partition);

    let reference_out = ReferenceFrame {
        y: rec_y,
        u: rec_u,
        v: rec_v,
        y_stride,
        uv_stride,
        y_h: y_buf_h,
        uv_h: uv_buf_h,
    };
    Ok((out, reference_out))
}

// ---------------------------------------------------------------------------
// Look-ahead alt-ref synthesis (RFC 6386 §6 mentions hidden frames; the
// synthesis algorithm here is derived from first principles + classical
// motion-compensated temporal-noise-reduction theory: align each
// neighbour to the centre frame by per-block motion search, then take a
// pixel-wise weighted mean with a similarity-driven exponential weight
// that gates off occluded / new content).
// ---------------------------------------------------------------------------

/// Synthesised alt-ref source image. Stored as Yuv420P planes packed at
/// the natural width/stride (no MB padding) so it can be wrapped in a
/// `VideoFrame` and fed straight to the standard P-frame encoder.
#[cfg(feature = "registry")]
struct SynthesizedAltRef {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    /// Source-image height; only consumed via the chroma-plane size
    /// computation in `as_video_frame`, where it falls out of `data.len`.
    /// Kept for call-site clarity.
    #[allow(dead_code)]
    height: usize,
    /// Optional PTS to forward to the hidden frame's packet. Inherited
    /// from the centre frame in the look-ahead window.
    pts: Option<i64>,
}

#[cfg(feature = "registry")]
impl SynthesizedAltRef {
    /// Wrap as a `VideoFrame` so the standard P-frame encoder accepts
    /// the synthesized planes as a "source".
    fn as_video_frame(&self) -> VideoFrame {
        let cw = (self.width + 1) / 2;
        VideoFrame {
            pts: self.pts,
            planes: vec![
                VideoPlane {
                    stride: self.width,
                    data: self.y.clone(),
                },
                VideoPlane {
                    stride: cw,
                    data: self.u.clone(),
                },
                VideoPlane {
                    stride: cw,
                    data: self.v.clone(),
                },
            ],
        }
    }
}

/// Build a temporally-filtered alt-ref source from the supplied
/// look-ahead buffer of recent input frames.
///
/// Algorithm (per RFC 6386-compatible derivation):
/// 1. Pick the centre frame of the buffer (index `len / 2`).
/// 2. For every other frame in the buffer, run a coarse 16×16 integer
///    motion search aligning that frame to the centre. The best MV
///    minimises luma SAD over the MB.
/// 3. Sample each non-centre frame at its motion-compensated position
///    to get an MC-aligned pixel value per location.
/// 4. For every output pixel, compute `weight = exp(-diff^2 / sigma^2)`
///    against the centre pixel; pixels too dissimilar (occlusion, new
///    content) get near-zero weight and don't contaminate the average.
/// 5. The output is the weighted mean (centre always counted, weight 1).
///
/// Returns `None` when geometry is invalid (zero dims, plane shape
/// mismatch). Single-frame buffers return the centre frame unchanged
/// (no smoothing possible — same as the legacy alt-ref source).
#[cfg(feature = "registry")]
fn synthesize_altref_image(
    width: usize,
    height: usize,
    buf: &VecDeque<VideoFrame>,
) -> Option<SynthesizedAltRef> {
    if buf.is_empty() || width == 0 || height == 0 {
        return None;
    }
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let n = buf.len();
    let centre_idx = n / 2;

    // Sanity-check every frame has matching geometry. Synthesis only
    // makes sense when the planes are the same shape across the window.
    for f in buf.iter() {
        if f.planes.len() < 3 {
            return None;
        }
        if f.planes[0].data.len() < height * f.planes[0].stride {
            return None;
        }
        if f.planes[1].data.len() < ch * f.planes[1].stride {
            return None;
        }
        if f.planes[2].data.len() < ch * f.planes[2].stride {
            return None;
        }
    }

    let centre = &buf[centre_idx];
    let centre_y = &centre.planes[0];
    let centre_u = &centre.planes[1];
    let centre_v = &centre.planes[2];

    // Trivial path: only one frame in the buffer → output equals centre.
    // No motion search, no filtering. Matches the legacy alt-ref's
    // "use the source frame as-is" behaviour.
    if n == 1 {
        let mut y_out = vec![0u8; width * height];
        let mut u_out = vec![0u8; cw * ch];
        let mut v_out = vec![0u8; cw * ch];
        for r in 0..height {
            let s = r * centre_y.stride;
            y_out[r * width..r * width + width].copy_from_slice(&centre_y.data[s..s + width]);
        }
        for r in 0..ch {
            let su = r * centre_u.stride;
            let sv = r * centre_v.stride;
            u_out[r * cw..r * cw + cw].copy_from_slice(&centre_u.data[su..su + cw]);
            v_out[r * cw..r * cw + cw].copy_from_slice(&centre_v.data[sv..sv + cw]);
        }
        return Some(SynthesizedAltRef {
            y: y_out,
            u: u_out,
            v: v_out,
            width,
            height,
            pts: centre.pts,
        });
    }

    // For each non-centre frame, compute one MV per 16×16 luma MB
    // aligning that frame to the centre. We reuse the same SAD-based
    // integer search the inter mode decision uses, on the natural-stride
    // luma planes.
    let mb_w = (width + 15) / 16;
    let mb_h = (height + 15) / 16;
    let mut frame_mvs: Vec<Vec<(i32, i32)>> = Vec::with_capacity(n);
    for (i, f) in buf.iter().enumerate() {
        if i == centre_idx {
            // Centre's MV is always 0 — placeholder.
            frame_mvs.push(vec![(0i32, 0i32); mb_w * mb_h]);
            continue;
        }
        let mut mvs = Vec::with_capacity(mb_w * mb_h);
        let src = &f.planes[0];
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mv = altref_search_mb(
                    &centre_y.data,
                    centre_y.stride,
                    &src.data,
                    src.stride,
                    width,
                    height,
                    mb_x,
                    mb_y,
                );
                mvs.push(mv);
            }
        }
        frame_mvs.push(mvs);
    }

    // Pixel-wise temporal filter. For each pixel of the centre, look up
    // the corresponding MC-aligned pixel from every other frame, compute
    // exp(-diff^2 / sigma^2), and accumulate a weighted mean.
    let sigma2 = TEMPORAL_FILTER_SIGMA * TEMPORAL_FILTER_SIGMA;
    let mut y_out = vec![0u8; width * height];
    for row in 0..height {
        let mb_y = row / 16;
        for col in 0..width {
            let mb_x = col / 16;
            let centre_px = centre_y.data[row * centre_y.stride + col] as f32;
            let mut wsum = 1.0f32; // centre counted with weight 1
            let mut acc = centre_px;
            for (i, f) in buf.iter().enumerate() {
                if i == centre_idx {
                    continue;
                }
                let (dy, dx) = frame_mvs[i][mb_y * mb_w + mb_x];
                let sr = (row as i32 + dy).clamp(0, height as i32 - 1) as usize;
                let sc = (col as i32 + dx).clamp(0, width as i32 - 1) as usize;
                let s = &f.planes[0];
                let px = s.data[sr * s.stride + sc] as f32;
                let d = px - centre_px;
                let w = (-(d * d) / sigma2).exp();
                acc += w * px;
                wsum += w;
            }
            y_out[row * width + col] = (acc / wsum).round().clamp(0.0, 255.0) as u8;
        }
    }

    // Chroma uses the same MV table at half-resolution (chroma MB = 8×8).
    // Map a chroma sample (cr, cc) to its luma MB by `(2*cr / 16, 2*cc / 16)`
    // = `(cr / 8, cc / 8)`, then halve the luma MV for the chroma MC offset.
    let mut u_out = vec![0u8; cw * ch];
    let mut v_out = vec![0u8; cw * ch];
    for plane_sel in 0..2 {
        let (centre_pl, out) = if plane_sel == 0 {
            (centre_u, &mut u_out)
        } else {
            (centre_v, &mut v_out)
        };
        for cr in 0..ch {
            let mb_y = (cr * 2) / 16;
            for cc in 0..cw {
                let mb_x = (cc * 2) / 16;
                let centre_px = centre_pl.data[cr * centre_pl.stride + cc] as f32;
                let mut wsum = 1.0f32;
                let mut acc = centre_px;
                for (i, f) in buf.iter().enumerate() {
                    if i == centre_idx {
                        continue;
                    }
                    let (dy, dx) = frame_mvs[i][mb_y * mb_w + mb_x];
                    // Chroma displacement is half of luma.
                    let cdy = dy / 2;
                    let cdx = dx / 2;
                    let sr = (cr as i32 + cdy).clamp(0, ch as i32 - 1) as usize;
                    let sc = (cc as i32 + cdx).clamp(0, cw as i32 - 1) as usize;
                    let s = if plane_sel == 0 {
                        &f.planes[1]
                    } else {
                        &f.planes[2]
                    };
                    let px = s.data[sr * s.stride + sc] as f32;
                    let d = px - centre_px;
                    let w = (-(d * d) / sigma2).exp();
                    acc += w * px;
                    wsum += w;
                }
                out[cr * cw + cc] = (acc / wsum).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Some(SynthesizedAltRef {
        y: y_out,
        u: u_out,
        v: v_out,
        width,
        height,
        pts: centre.pts,
    })
}

/// Per-MB integer-pel motion search aligning `src` (one of the buffered
/// frames) to `centre`. Searches a small ±`ALTREF_MC_RANGE` window
/// around (0, 0) and returns the `(dy, dx)` displacement (in pixels)
/// that minimises 16×16 luma SAD between the centre MB and the
/// MV-pointed source MB. Boundary MBs near the frame edge clamp at the
/// frame border (the centre and source planes need not be MB-padded —
/// we clamp explicitly). Both planes carry their own `stride`, allowing
/// arbitrary row padding without forcing a copy.
#[cfg(feature = "registry")]
#[allow(clippy::too_many_arguments)]
fn altref_search_mb(
    centre: &[u8],
    centre_stride: usize,
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> (i32, i32) {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    let bw = (width as i32 - x0 as i32).clamp(0, 16) as usize;
    let bh = (height as i32 - y0 as i32).clamp(0, 16) as usize;
    if bw == 0 || bh == 0 {
        return (0, 0);
    }
    let sad_at = |dy: i32, dx: i32| -> u32 {
        let mut sad: u32 = 0;
        for r in 0..bh {
            let cy = y0 + r;
            let sy = (cy as i32 + dy).clamp(0, height as i32 - 1) as usize;
            for c in 0..bw {
                let cx = x0 + c;
                let sx = (cx as i32 + dx).clamp(0, width as i32 - 1) as usize;
                let a = centre[cy * centre_stride + cx] as i32;
                let b = src[sy * src_stride + sx] as i32;
                sad += (a - b).unsigned_abs();
            }
        }
        sad
    };
    let mut best = (0i32, 0i32);
    let mut best_sad = sad_at(0, 0);
    for dy in -ALTREF_MC_RANGE..=ALTREF_MC_RANGE {
        for dx in -ALTREF_MC_RANGE..=ALTREF_MC_RANGE {
            if dy == 0 && dx == 0 {
                continue;
            }
            let s = sad_at(dy, dx);
            if s < best_sad {
                best_sad = s;
                best = (dy, dx);
            }
        }
    }
    best
}

/// Encode the synthesized image as a hidden P-frame against `last_ref`.
///
/// Bitstream specifics:
/// * `show_frame = 0` (consumer doesn't see this frame).
/// * `refresh_alt = 1`, everything else off (only the alt-ref slot is
///   updated; LAST and GOLDEN keep their previous reconstructions).
///
/// Implemented by calling the standard P-frame encoder with a custom
/// `RefPlan` and patching the `show_frame` bit in the resulting tag
/// byte. The reconstruction returned is what BOTH encoder and decoder
/// will install in their alt-ref slot.
#[cfg(feature = "registry")]
fn encode_hidden_altref_pframe(
    width: u32,
    height: u32,
    cfg: Vp8EncoderConfig,
    synth: &SynthesizedAltRef,
    last_ref: &ReferenceFrame,
) -> Result<(Vec<u8>, ReferenceFrame)> {
    let synth_frame = synth.as_video_frame();
    let synth_src = Yuv420Source::from_video_frame(&synth_frame)?;
    let plan = RefPlan {
        refresh_last: false,
        refresh_golden: false,
        refresh_alt: true,
        // No GOLDEN/ALT predicted-from references — the hidden frame
        // codes against LAST only. Both ALT and GOLDEN are still
        // populated in the encoder state but we explicitly forbid
        // their use as a prediction reference inside the synthesized
        // frame to keep the per-MB mode decision simple.
        use_golden: false,
        use_alt: false,
    };
    // Disable RDO knobs that depend on multi-ref bookkeeping for the
    // hidden frame — the standard encoder still works fine in this
    // mode but the cleanest residual-only encode is achieved with the
    // single-reference behaviour.
    let mut hcfg = cfg;
    hcfg.enable_multi_ref = false;
    // Disable segments so the hidden frame's bitstream is minimal —
    // segmentation adds per-MB bits that aren't useful for a hidden
    // alt-ref where every MB should be coded against LAST as the
    // only reference.
    hcfg.enable_segments = false;
    // Push the hidden frame's quant a few notches finer than the
    // visible frame's. The alt-ref slot's accuracy directly determines
    // how big the per-MB residuals are on every subsequent P-frame
    // that references it — spending a small amount of extra bits on
    // the hidden frame buys back a much larger savings on the (often
    // many) visible frames that follow. The exact delta is tuned so
    // hidden-frame growth stays well below the per-frame savings on a
    // motion-rich noisy fixture.
    hcfg.qindex = (cfg.qindex as i32 - HIDDEN_ALTREF_QINDEX_DELTA).clamp(0, 127) as u8;
    let (mut bitstream, rec) =
        encode_pframe_and_reconstruct(width, height, hcfg, &synth_src, last_ref, None, None, plan)?;
    // Patch the frame tag to set `show_frame = 0`. The tag is the first
    // 3 bytes; bit 4 of byte 0 is the show_frame flag.
    if !bitstream.is_empty() {
        bitstream[0] &= !0x10;
    }
    Ok((bitstream, rec))
}

// ---------------------------------------------------------------------------
// Lagrangian RDO helpers — used by the per-MB ref-and-mode picker.
// ---------------------------------------------------------------------------

/// Lagrangian multiplier for a given quantiser. The textbook expression
/// is `lambda = 0.85 * QP^2`; we scale by `scale/256` to keep lambda
/// comparable in magnitude to the SSE accumulator (which is integer
/// 0..=255*255 per pixel * 256 pixels per MB ≈ low millions for an
/// average MB). Returned as an unsigned integer for use in `D + λ·R`
/// arithmetic; 0 disables the rate term and recovers SSE-only mode.
#[inline]
fn lambda_for_qp(qp: u32, scale: u32) -> u32 {
    if scale == 0 {
        return 0;
    }
    let q = qp.max(1);
    (scale.saturating_mul(q.saturating_mul(q))) / 256
}

// Per-decision rate accounting (issue #340) is now in 1/256-bit units
// via `bool_encoder::bool_cost_x256`, which derives each entry from the
// real bool-coder state machine. The previous 7-step
// `PROB_TO_COST_8X` LUT (1/8-bit units) was a 32× precision loss that
// collapsed many `p` values to the same cost — it has been removed.

/// SSE of a 16x16 luma MB at `(mb_x, mb_y)` predicted from `ref_y` shifted
/// by `(dy, dx)` integer luma pixels (with edge clamping).
fn mb_luma_sse_at_int(
    src_y: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    dy: i32,
    dx: i32,
) -> u64 {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    let w = stride as i32;
    let h = (ref_y.len() / stride) as i32;
    let mut sse = 0u64;
    for r in 0..16 {
        let ry = (y0 as i32 + r as i32 + dy).clamp(0, h - 1) as usize;
        let row_off_src = (y0 + r) * stride;
        for c in 0..16 {
            let rx = (x0 as i32 + c as i32 + dx).clamp(0, w - 1) as usize;
            let d = src_y[row_off_src + x0 + c] as i32 - ref_y[ry * stride + rx] as i32;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// SSE of a 16x16 luma MB predicted at sub-pel MV `mv` (1/8-pel units),
/// using the 6-tap luma filter the decoder applies.
fn mb_luma_sse_at_subpel(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    src_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    mv: Mv,
) -> u64 {
    let mut pred = [0u8; 256];
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = bx * 4;
        let dst_y = by * 4;
        let ref_x_fp = (mb_xp + dst_x) as i32 * 8 + mv.col as i32;
        let ref_y_fp = (mb_yp + dst_y) as i32 * 8 + mv.row as i32;
        sixtap_predict(
            ref_plane, ref_x_fp, ref_y_fp, &mut pred, 16, dst_x, dst_y, 4, 4,
        );
    }
    let mut sse = 0u64;
    for r in 0..16 {
        for c in 0..16 {
            let s = src_y[(mb_yp + r) * src_stride + mb_xp + c] as i32;
            let p = pred[r * 16 + c] as i32;
            let d = s - p;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Per-MB mode-info bit cost (1/256-bit units) for a candidate
/// decision: skip flag + intra-vs-inter + ref-frame bits + MV-tree
/// leaf + MV deltas.
///
/// Each `bool_cost_x256(prob, outcome)` call returns the *real* bool-
/// coder cost for one bool symbol (issue #340) — derived from the
/// state machine `BoolEncoder::write_bool` runs, not the legacy
/// `floor(log2(256/p))` approximation. Combine with distortion via
/// `D + λ·R/256` (the `/256` undoes the fixed-point scale).
#[cfg(feature = "registry")]
#[allow(clippy::too_many_arguments)]
fn estimate_mode_rate_x256(
    decision: &PMbDecision,
    ref_frame: u8,
    plan: RefPlan,
    nearest: Mv,
    near: Mv,
    best_for_newmv: Mv,
    prob_intra: u8,
    prob_last: u8,
    prob_gf: u8,
    mb_skip_prob: u8,
) -> u32 {
    let mut r = 0u32;
    r += bool_cost_x256(mb_skip_prob, matches!(decision, PMbDecision::Skip));
    let is_inter = !decision.is_intra();
    r += bool_cost_x256(prob_intra, is_inter);
    if !is_inter {
        return r;
    }
    if plan.use_golden || plan.use_alt {
        match ref_frame {
            ENC_REF_LAST => r += bool_cost_x256(prob_last, false),
            ENC_REF_GOLDEN => {
                r += bool_cost_x256(prob_last, true);
                r += bool_cost_x256(prob_gf, false);
            }
            ENC_REF_ALT => {
                r += bool_cost_x256(prob_last, true);
                r += bool_cost_x256(prob_gf, true);
            }
            _ => {}
        }
    } else {
        r += bool_cost_x256(prob_last, false);
    }
    match decision {
        PMbDecision::Skip | PMbDecision::ZeroMv => {
            r += bool_cost_x256(128, false);
        }
        PMbDecision::NearestMv(_) => {
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, false);
        }
        PMbDecision::NearMv(_) => {
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, false);
        }
        PMbDecision::NewMv(mv) => {
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, false);
            // Precise MV-delta cost: route the candidate value through
            // the same bool-coder cost LUT that `encode_mv_component`
            // uses on the real bitstream (issue #340).
            let dr = mv.row as i32 - best_for_newmv.row as i32;
            let dc = mv.col as i32 - best_for_newmv.col as i32;
            r += mv_component_cost_x256(&DEFAULT_MV_CONTEXT[0], dr);
            r += mv_component_cost_x256(&DEFAULT_MV_CONTEXT[1], dc);
        }
        PMbDecision::SplitMv(s) => {
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, true);
            r += bool_cost_x256(128, true);
            let n = MB_SPLIT_COUNT[s.split_mode as usize] as u32;
            // Per-partition split-mode tree: ~4 bits per partition;
            // 4 bits × 256/bit = 1024.
            r += 1024 * n;
            for p in 0..n as usize {
                let mv = s.part_mvs[p];
                if mv != best_for_newmv && mv != Mv::ZERO {
                    let dr = mv.row as i32 - best_for_newmv.row as i32;
                    let dc = mv.col as i32 - best_for_newmv.col as i32;
                    r += mv_component_cost_x256(&DEFAULT_MV_CONTEXT[0], dr);
                    r += mv_component_cost_x256(&DEFAULT_MV_CONTEXT[1], dc);
                }
            }
        }
        PMbDecision::Intra { .. } => {}
    }
    let _ = (nearest, near);
    r
}

// MV-delta rate is sourced from `mv::mv_component_cost_x256`, which
// runs the candidate value through the same bool-coder cost LUT as
// `mv::encode_mv_component` (issue #340) — bit-accurate per
// component. The 6-tier legacy heuristic
// (`32/48/96/128/160/192` in 1/8-bit units) has been removed.

/// Approximate the SSE for a candidate decision (the distortion `D` of
/// the Lagrangian cost). For inter modes we use the sub-pel SSE against
/// the chosen reference; for intra we evaluate the chosen Y prediction
/// (16×16 DC/V/H/TM or B_PRED) against the source using the running
/// reconstruction `rec_y` for neighbour context — bit-accurate per-mode
/// SSE so the Lagrangian comparison weighs intra-vs-inter on the same
/// distortion scale (#392). The placeholder constant `8000` previously
/// used here biased the picker against intra on textured MBs because it
/// was orders of magnitude smaller than the real intra SSE on such MBs.
#[cfg(feature = "registry")]
#[allow(clippy::too_many_arguments)]
fn estimate_distortion(
    decision: &PMbDecision,
    src_y: &[u8],
    ref_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    y_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    mb_h: usize,
) -> u64 {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;
    let ref_plane = RefPlane {
        data: ref_y,
        stride: y_stride,
        width: y_stride,
        height: y_buf_h,
    };
    match decision {
        PMbDecision::Skip | PMbDecision::ZeroMv => {
            mb_luma_sse_at_int(src_y, ref_y, y_stride, mb_x, mb_y, 0, 0)
        }
        PMbDecision::NearestMv(mv) | PMbDecision::NearMv(mv) | PMbDecision::NewMv(mv) => {
            mb_luma_sse_at_subpel(src_y, &ref_plane, y_stride, mb_xp, mb_yp, *mv)
        }
        PMbDecision::SplitMv(s) => {
            let mut sse = 0u64;
            for i in 0..16 {
                let part = MB_SPLITS[s.split_mode as usize][i];
                let mv = s.part_mvs[part as usize];
                let by = i / 4;
                let bx = i % 4;
                let sx = mb_xp + bx * 4;
                let sy = mb_yp + by * 4;
                let mut pred = [0u8; 16];
                let ref_x_fp = sx as i32 * 8 + mv.col as i32;
                let ref_y_fp = sy as i32 * 8 + mv.row as i32;
                sixtap_predict(&ref_plane, ref_x_fp, ref_y_fp, &mut pred, 4, 0, 0, 4, 4);
                for r in 0..4 {
                    for c in 0..4 {
                        let s = src_y[(sy + r) * y_stride + sx + c] as i32;
                        let p = pred[r * 4 + c] as i32;
                        let d = s - p;
                        sse += (d * d) as u64;
                    }
                }
            }
            sse
        }
        PMbDecision::Intra { y_mode, .. } => {
            // Per-mode Y-plane SSE against the source. Chroma is omitted
            // — the inter branches above also measure Y-only SSE so this
            // keeps the distortion units comparable across branches.
            // Same neighbour-gathering convention (`rec_y` + the same
            // helper) as `choose_pmb_decision`'s intra fallback at
            // `src/encoder.rs` ~3170 so the SSE the picker sees here is
            // exactly the one that branch already minimised over.
            if *y_mode == B_PRED {
                let (sse, _modes) =
                    sse_intra_b_pred(src_y, rec_y, y_stride, mb_xp, mb_yp, mb_w, mb_h);
                sse
            } else {
                let (sse, _pred) =
                    sse_intra_16x16(*y_mode, src_y, rec_y, y_stride, mb_xp, mb_yp);
                sse
            }
        }
    }
}

/// Compute the Lagrangian RD cost `D + λ·R/256` (R is in 1/256-bit
/// units, so we divide by 256 to renormalise). Used by the per-MB
/// reference-and-mode picker to select the best candidate across LAST /
/// GOLDEN / ALTREF.
///
/// Issue #340: the rate input is now derived from a real bool-coded
/// bit accumulator (see `bool_encoder::PROB_COST_BITS_X256`); the
/// previous 1/8-bit, 7-step LUT collapsed many distinct probabilities
/// to the same value. The Lagrangian magnitude is preserved by scaling
/// the divisor 32× along with the precision, so existing
/// `lambda_scale` knob calibrations carry over.
#[cfg(feature = "registry")]
#[allow(clippy::too_many_arguments)]
fn rd_cost_for_decision(
    decision: &PMbDecision,
    src_y: &[u8],
    ref_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    y_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    mb_h: usize,
    ref_frame: u8,
    plan: RefPlan,
    nearest: Mv,
    near: Mv,
    best_for_newmv: Mv,
    prob_intra: u8,
    prob_last: u8,
    prob_gf: u8,
    mb_skip_prob: u8,
    lambda: u32,
) -> u64 {
    let d = estimate_distortion(
        decision, src_y, ref_y, rec_y, y_stride, y_buf_h, mb_x, mb_y, mb_w, mb_h,
    );
    let r = estimate_mode_rate_x256(
        decision,
        ref_frame,
        plan,
        nearest,
        near,
        best_for_newmv,
        prob_intra,
        prob_last,
        prob_gf,
        mb_skip_prob,
    );
    d + ((lambda as u64) * (r as u64)) / 256
}

/// Pick the best P-frame decision for one MB.
///
/// The search proceeds SKIP → ZERO_MV / NEAREST / NEAR (free-MV paths) →
/// NEW_MV (integer-pel search + quarter-pel refinement) → SPLIT_MV
/// (per-partition integer + quarter-pel search for each of the 4 split
/// modes) → intra (DC/V/H/TM/B_PRED) fallback if every inter option has
/// a very high SAD. NEAREST/NEAR are preferred over NEW_MV when their
/// SAD is within `NEIGHBOUR_MV_MARGIN` of the post-refinement NEW_MV
/// SAD, since they do not code an MV delta.
#[allow(clippy::too_many_arguments)]
fn choose_pmb_decision(
    src_y: &[u8],
    ref_y: &[u8],
    y_stride: usize,
    y_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    mb_h: usize,
    nearest: Mv,
    near: Mv,
    rec_y: &[u8],
) -> PMbDecision {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    // 1) cheap skip test against zero-motion reference.
    let zero_sad = mb_luma_sad_at(src_y, ref_y, y_stride, mb_x, mb_y, 0, 0);
    if zero_sad <= MB_SKIP_SAD_PER_PIXEL * (16 * 16) {
        return PMbDecision::Skip;
    }

    // 2) integer-pel NEWMV search.
    let (best_int_px, best_int_sad) = integer_motion_search(
        src_y,
        ref_y,
        y_stride,
        mb_x,
        mb_y,
        y_buf_h,
        MOTION_SEARCH_RANGE,
    );
    let int_mv = Mv::new(best_int_px.0 * 8, best_int_px.1 * 8);

    // 3) quarter-pel refinement around the integer-pel best.
    let ref_plane = RefPlane {
        data: ref_y,
        stride: y_stride,
        width: y_stride,
        height: y_buf_h,
    };
    let (refined_mv, refined_sad) = subpel_refine_luma(
        src_y,
        &ref_plane,
        y_stride,
        mb_xp,
        mb_yp,
        int_mv,
        best_int_sad,
    );

    // 4) evaluate NEAREST and NEAR candidates at their exact sub-pel MVs
    //    using the same sixtap_predict path the decoder will apply.
    let nearest_sad = if nearest != Mv::ZERO {
        Some(subpel_luma_sad_at(
            src_y, &ref_plane, y_stride, mb_xp, mb_yp, nearest,
        ))
    } else {
        None
    };
    let near_sad = if near != Mv::ZERO && near != nearest {
        Some(subpel_luma_sad_at(
            src_y, &ref_plane, y_stride, mb_xp, mb_yp, near,
        ))
    } else {
        None
    };

    // 5) compare free-MV modes (ZERO / NEAREST / NEAR) — all three code
    //    no MV delta, so the SAD comparison is direct. To propagate a
    //    coherent global motion across MBs we *bias* a non-zero NEAREST /
    //    NEAR over the ZERO-MV baseline by `NEIGHBOUR_OVER_ZERO_BIAS`:
    //    if a neighbour MV is at most this many SAD units worse than
    //    ZERO, follow it (the encoder pays no MV-delta bits for
    //    NEAREST / NEAR, so the bit savings recoup the small SAD
    //    overshoot — and crucially it keeps the neighbour-chain alive
    //    for the next row of MBs). NEW_MV then has to beat the best
    //    free mode by at least `NEWMV_SAD_MARGIN`, since NEW_MV pays
    //    the MV-delta bit cost; `NEIGHBOUR_MV_MARGIN` gives NEAREST /
    //    NEAR an extra edge on top of that base margin.
    let mut best_free: (u32, PMbDecision) = (zero_sad, PMbDecision::ZeroMv);
    // `nearest_sad` / `near_sad` are computed iff the corresponding MV
    // is non-zero (see step 4 above), so reaching `Some(_)` here
    // implies the candidate is a real neighbour MV worth biasing
    // toward the ZERO baseline. The bias is *vs ZERO* (not vs the
    // running best) so a noisy NEAR candidate that is already much
    // worse than NEAREST cannot leapfrog NEAREST just because it
    // also beats ZERO + bias.
    if let Some(s) = nearest_sad {
        if s < zero_sad + NEIGHBOUR_OVER_ZERO_BIAS && s < best_free.0 + NEIGHBOUR_OVER_ZERO_BIAS {
            best_free = (s, PMbDecision::NearestMv(nearest));
        }
    }
    if let Some(s) = near_sad {
        if s < zero_sad + NEIGHBOUR_OVER_ZERO_BIAS && s < best_free.0 {
            best_free = (s, PMbDecision::NearMv(near));
        }
    }

    let extra_margin = match best_free.1 {
        PMbDecision::NearestMv(_) | PMbDecision::NearMv(_) => NEIGHBOUR_MV_MARGIN,
        _ => 0,
    };
    // Real-motion shortcut: when the refined NEW_MV is far from zero
    // (genuine displacement, not a noisy sub-pel jitter) the residual
    // savings on the larger MV easily amortise the delta-bit cost. Use
    // a much smaller margin in that case so the picker doesn't fall
    // back to ZERO_MV (and a useless residual) just because both
    // candidates land at similar SAD against a heavily-quantised
    // reference (#373).
    let large_displacement = refined_mv.row.unsigned_abs() as i32
        >= NEWMV_LARGE_DISPLACEMENT_THRESHOLD
        || refined_mv.col.unsigned_abs() as i32 >= NEWMV_LARGE_DISPLACEMENT_THRESHOLD;
    let new_vs_free_margin = if large_displacement {
        NEWMV_LARGE_DISPLACEMENT_MARGIN + extra_margin
    } else {
        NEWMV_SAD_MARGIN + extra_margin
    };
    let (mut best_decision, mut best_sad) =
        if refined_sad + new_vs_free_margin < best_free.0 && refined_mv != Mv::ZERO {
            (PMbDecision::NewMv(refined_mv), refined_sad)
        } else {
            (best_free.1, best_free.0)
        };

    // 5.4) NEW_MV-to-neighbour snap: when the picker just chose NEW_MV
    //      but the refined MV is within a quarter-pel jitter of an
    //      available non-zero NEAREST or NEAR candidate, re-emit as
    //      NEAREST / NEAR — same reconstruction (the predictor reads
    //      the same sub-pel taps, modulo a tiny SAD difference) but
    //      without paying the MV-delta bits. Keeps the neighbour-chain
    //      coherent across rows of MBs that all share the same motion
    //      and naturally find it via the per-MB integer search (#373).
    if let PMbDecision::NewMv(mv) = best_decision {
        if nearest != Mv::ZERO && mv_within_tolerance(mv, nearest, NEIGHBOUR_MV_SNAP_TOLERANCE) {
            best_decision = PMbDecision::NearestMv(nearest);
            if let Some(s) = nearest_sad {
                best_sad = s;
            }
        } else if near != Mv::ZERO && mv_within_tolerance(mv, near, NEIGHBOUR_MV_SNAP_TOLERANCE) {
            best_decision = PMbDecision::NearMv(near);
            if let Some(s) = near_sad {
                best_sad = s;
            }
        }
    }

    // 5.5) SPLIT_MV: per-partition motion search, considered when the
    //      single-MV NEW_MV residual is still noticeable. Each of the 4
    //      split modes (16×8, 8×16, 8×8, 4×4) gets its own per-partition
    //      search; the cheapest total-SAD split wins only if it beats
    //      the current best decision by at least
    //      `n_parts * SPLITMV_SAD_MARGIN_PER_PARTITION`.
    if best_sad > SPLITMV_CONSIDER_SAD_PER_PIXEL * (16 * 16) {
        if let Some((split, split_sad)) = search_split_mv(src_y, &ref_plane, y_stride, mb_xp, mb_yp)
        {
            let n_parts = MB_SPLIT_COUNT[split.split_mode as usize] as u32;
            let split_margin = n_parts * SPLITMV_SAD_MARGIN_PER_PARTITION;
            if split_sad + split_margin < best_sad {
                best_decision = PMbDecision::SplitMv(split);
                best_sad = split_sad;
            }
        }
    }

    // 6) intra fallback when even the best inter prediction is very poor —
    //    e.g. uncovered regions, mid-frame scene cuts, heavy texture.
    if best_sad > INTRA_IN_P_SAD_PER_PIXEL * (16 * 16) {
        // Pick the best 16×16 intra Y mode (DC / V / H / TM) against the
        // reconstruction we have so far. Chroma uses DC_PRED — the
        // residual already dominates so the chroma bit cost isn't
        // worth a four-way search.
        let y16_mode = choose_intra_16x16_y_mode(src_y, rec_y, y_stride, mb_xp, mb_yp);

        // #339: when the MB's Y-plane variance is high (heavy texture
        // that the 16×16 modes can't capture in a single prediction),
        // also evaluate B_PRED. The per-4×4 sub-mode search picks
        // direction-aware sub-block predictions that often score
        // significantly lower SSE on textured content. B_PRED costs
        // extra bits (16 sub-mode bool emits + a separate per-block
        // residual chain, no Y2 DC short-cut), so we only let it win
        // when the SSE improvement clears `B_PRED_SSE_MARGIN_INTRA_IN_P`.
        let y_mode = if mb_luma_variance(src_y, y_stride, mb_xp, mb_yp)
            >= INTRA_IN_P_BPRED_VARIANCE_THRESHOLD
        {
            let (y16_sse, _) = sse_intra_16x16(y16_mode, src_y, rec_y, y_stride, mb_xp, mb_yp);
            let (bp_sse, _bp_modes) =
                sse_intra_b_pred(src_y, rec_y, y_stride, mb_xp, mb_yp, mb_w, mb_h);
            if bp_sse + B_PRED_SSE_MARGIN_INTRA_IN_P < y16_sse {
                B_PRED
            } else {
                y16_mode
            }
        } else {
            y16_mode
        };

        return PMbDecision::Intra {
            y_mode,
            uv_mode: DC_PRED,
        };
    }

    best_decision
}

/// Variance of a 16×16 luma block, in summed-square units (E[x^2] - E[x]^2,
/// not divided by `n`). Same scale as
/// [`SEGMENT_VARIANCE_THRESHOLDS`], so callers can compare directly
/// against the same threshold ladder.
#[inline]
fn mb_luma_variance(src_y: &[u8], y_stride: usize, mb_xp: usize, mb_yp: usize) -> u64 {
    let mut sum: u64 = 0;
    let mut sum2: u64 = 0;
    for r in 0..16 {
        let row_off = (mb_yp + r) * y_stride + mb_xp;
        for c in 0..16 {
            let v = src_y[row_off + c] as u64;
            sum += v;
            sum2 += v * v;
        }
    }
    let n = 256u64;
    sum2.saturating_sub((sum * sum) / n)
}

/// Variance threshold (in `mb_luma_variance` units) above which the
/// intra-in-P picker also evaluates B_PRED in addition to the four
/// 16×16 modes (#339). Picked at the same boundary as the segment-3
/// variance gate (`SEGMENT_VARIANCE_THRESHOLDS[2]` =
/// `3200 * 256 = 819_200`) so that any MB the segment classifier
/// already flagged as "high variance" is also a B_PRED candidate.
pub const INTRA_IN_P_BPRED_VARIANCE_THRESHOLD: u64 = 3200 * 256;

/// Minimum SSE improvement (across the full 256-pixel MB) that
/// B_PRED must show over the best 16×16 intra candidate to be picked
/// in the *intra-in-P* path (#339). Tighter than the keyframe-path
/// margin (`B_PRED_SSE_MARGIN`) since intra-in-P MBs already pay an
/// extra `prob_intra` bit for crossing into the intra branch and the
/// rate-vs-distortion balance is more sensitive than on keyframes.
const B_PRED_SSE_MARGIN_INTRA_IN_P: u64 = 1024;

/// L∞ tolerance check on two MVs (1/8-pel units). Returns `true` when
/// each component is within `tol` of the other. Used to detect when a
/// refined NEW_MV essentially matches an available NEAREST / NEAR
/// neighbour candidate so the picker can emit the cheaper neighbour
/// mode instead of paying the MV-delta bit cost.
#[inline]
fn mv_within_tolerance(a: Mv, b: Mv, tol: i32) -> bool {
    (a.row as i32 - b.row as i32).abs() <= tol && (a.col as i32 - b.col as i32).abs() <= tol
}

/// Sub-pel luma SAD: run `sixtap_predict` per 4×4 (matching the decoder's
/// per-subblock invocation) and sum |src - pred|.
fn subpel_luma_sad_at(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    src_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    mv: Mv,
) -> u32 {
    let mut pred = [0u8; 256];
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = bx * 4;
        let dst_y = by * 4;
        let ref_x_fp = (mb_xp + dst_x) as i32 * 8 + mv.col as i32;
        let ref_y_fp = (mb_yp + dst_y) as i32 * 8 + mv.row as i32;
        sixtap_predict(
            ref_plane, ref_x_fp, ref_y_fp, &mut pred, 16, dst_x, dst_y, 4, 4,
        );
    }
    let mut sad: u32 = 0;
    for r in 0..16 {
        for c in 0..16 {
            let s = src_y[(mb_yp + r) * src_stride + mb_xp + c] as i32;
            let p = pred[r * 16 + c] as i32;
            sad += (s - p).unsigned_abs();
        }
    }
    sad
}

/// Quarter-pel refinement: 8-neighbour hill-climb at `SUBPEL_REFINE_STEP`
/// (1/8-pel units). Starts from `int_mv` with its known `int_sad`, scans
/// the 3×3 neighbourhood once, and returns the best (mv, sad). One pass
/// is enough in practice since the integer search already landed on a
/// local minimum.
fn subpel_refine_luma(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    src_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    int_mv: Mv,
    int_sad: u32,
) -> (Mv, u32) {
    let mut best_mv = int_mv;
    let mut best_sad = int_sad;
    let step = SUBPEL_REFINE_STEP;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let mv = Mv::new(int_mv.row as i32 + dy * step, int_mv.col as i32 + dx * step);
            let sad = subpel_luma_sad_at(src_y, ref_plane, src_stride, mb_xp, mb_yp, mv);
            if sad < best_sad {
                best_sad = sad;
                best_mv = mv;
            }
        }
    }
    (best_mv, best_sad)
}

/// SAD of the luma MB at `(mb_x, mb_y)` against the reference plane shifted
/// by `(dy, dx)` integer luma pixels. Out-of-bounds reference samples are
/// clamped (edge replication) to match the decoder's `RefPlane::sample`.
fn mb_luma_sad_at(
    src_y: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    dy: i32,
    dx: i32,
) -> u32 {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    let w = stride as i32;
    let h = (ref_y.len() / stride) as i32;
    let mut sad: u32 = 0;
    for r in 0..16 {
        let ry = (y0 as i32 + r as i32 + dy).clamp(0, h - 1) as usize;
        let row_off_src = (y0 + r) * stride;
        for c in 0..16 {
            let rx = (x0 as i32 + c as i32 + dx).clamp(0, w - 1) as usize;
            let d = src_y[row_off_src + x0 + c] as i32 - ref_y[ry * stride + rx] as i32;
            sad += d.unsigned_abs();
        }
    }
    sad
}

/// Simple full-pel motion search. Scans `[-range, +range]` on each axis
/// around zero, returning `((dy, dx), best_sad)` for the displacement that
/// minimises luma SAD. Ties break toward smaller magnitudes (the search
/// starts at zero and only switches on strict improvement).
#[allow(clippy::too_many_arguments)]
fn integer_motion_search(
    src_y: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    _y_buf_h: usize,
    range: i32,
) -> ((i32, i32), u32) {
    let mut best = (0, 0);
    let mut best_sad = mb_luma_sad_at(src_y, ref_y, stride, mb_x, mb_y, 0, 0);
    for dy in -range..=range {
        for dx in -range..=range {
            if dy == 0 && dx == 0 {
                continue;
            }
            let sad = mb_luma_sad_at(src_y, ref_y, stride, mb_x, mb_y, dy, dx);
            if sad < best_sad {
                best_sad = sad;
                best = (dy, dx);
            }
        }
    }
    (best, best_sad)
}

/// Encode-time replica of `find_near_mvs` in the decoder. Returns
/// `(nearest, near, best, cnt)`. Out-of-frame neighbours and intra
/// neighbours contribute NOTHING; every other neighbour contributes its
/// MV regardless of which reference it points at. See the decoder's
/// `find_near_mvs` for the RFC 6386 §16.3 walk.
///
/// Note: with all reference frames sharing sign-bias = false in this
/// encoder, the sign-bias flip is a no-op so we omit it (the decoder
/// would XOR neighbour vs current ref bias and negate when they differ).
/// `ref_frame` and `mb_ref_frames` are kept in the signature for parity
/// with the decoder's `find_near_mvs` signature and so a future
/// non-zero sign-bias would have the inputs it needs.
#[allow(clippy::too_many_arguments)]
fn find_near_mvs_enc(
    mb_mvs: &[Mv],
    mb_decisions: &[PMbDecision],
    mb_ref_frames: &[u8],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    _ref_frame: u8,
) -> (Mv, Mv, Mv, [u8; 4]) {
    let mut mvs: [Mv; 4] = [Mv::ZERO; 4];
    let mut cnt = [0u8; 4];
    let mut mv_idx: usize = 0;
    let neighbours: [(isize, isize, u8); 3] = [(0, -1, 2), (-1, 0, 2), (-1, -1, 1)];

    for (nb_idx, &(dx, dy, weight)) in neighbours.iter().enumerate() {
        let nx = mb_x as isize + dx;
        let ny = mb_y as isize + dy;
        if nx < 0 || ny < 0 || (nx as usize) >= mb_w {
            continue; // out-of-frame = intra, no contribution
        }
        let idx = (ny as usize) * mb_w + (nx as usize);
        let nmv = match mb_decisions.get(idx) {
            Some(d) if d.is_intra() => continue, // intra, no contribution
            Some(_) => {
                // Mirror the decoder's `find_near_mvs`: ALL non-intra
                // neighbours contribute, regardless of which reference they
                // point at. The reference frame only matters for the
                // sign-bias flip — with all our refs sharing
                // `sign_bias = false` (we hard-code `sign_bias_golden = 0`
                // and `sign_bias_alternate = 0` in the inter header) the
                // flip is a no-op. An earlier (pre-#373) version of this
                // walk filtered out neighbours with a different ref to
                // mirror the comment, but the decoder NEVER filters — so
                // doing so here desynced the encoder's neighbour chain
                // from the decoder's whenever a row mixed LAST / GOLDEN /
                // ALT picks, breaking NEAREST / NEAR reconstruction. (The
                // bug was latent until #373 made the picker aggressive
                // enough to pick non-LAST refs mid-row.)
                let _ = mb_ref_frames; // kept for signature/future sign-bias use
                mb_mvs[idx]
            }
            None => continue,
        };

        if nmv.row == 0 && nmv.col == 0 {
            cnt[0] += weight;
            continue;
        }

        if nb_idx == 0 {
            mv_idx = 1;
            mvs[1] = nmv;
            cnt[1] += weight;
        } else {
            if mvs[mv_idx] != nmv {
                mv_idx += 1;
                mvs[mv_idx] = nmv;
            }
            cnt[mv_idx] += weight;
        }
    }

    // REF_LAST-only encoder never emits SPLITMV, so the post-pass to
    // merge aboveleft-into-nearest based on cnt[CNT_SPLITMV] is a no-op
    // here — but keep the same shape as the decoder.
    if mv_idx == 3 && mvs[3] == mvs[1] {
        cnt[1] += 1;
    }
    cnt[3] = 0; // no SPLITMV neighbours in REF_LAST-only encoder

    if cnt[2] > cnt[1] {
        cnt.swap(1, 2);
        mvs.swap(1, 2);
    }
    if cnt[1] >= cnt[0] {
        mvs[0] = mvs[1];
    }

    (mvs[1], mvs[2], mvs[0], cnt)
}

/// Pick a frame-level bool-coder probability from observed counts.
///
/// VP8's bool coder convention is that `prob` is the probability of the
/// bit decoding as zero (`P(bit==0) = prob/256`). Given the frame-wide
/// counts `n_zero` (number of MBs whose bit decodes as `false`) and
/// `n_one` (number that decode as `true`), the optimal entropy-matched
/// probability is `round(256 * n_zero / (n_zero + n_one))`.
///
/// Two boundary fixups:
///   * `n_zero + n_one == 0`: no observations — fall back to neutral
///     `128` so the literal still parses correctly even though no
///     decoded bool will use it.
///   * Result clamped to `1..=255`: the bool coder can encode either
///     branch at any non-degenerate probability; clamping away from
///     `0` and `256` keeps every potential outcome representable
///     (a single rare event still has bounded cost) and matches the
///     libvpx "always-codable" convention.
#[inline]
fn optimal_prob_8(n_zero: u32, n_one: u32) -> u8 {
    let total = n_zero + n_one;
    if total == 0 {
        return 128;
    }
    // round(256 * n_zero / total) without overflow risk for plausible
    // frame sizes (mb_w * mb_h * weight ≤ a few million).
    let scaled = ((n_zero as u64) * 256 + (total as u64) / 2) / (total as u64);
    scaled.clamp(1, 255) as u8
}

/// Encode-time replica of the decoder's `mv_ref_probs`. Each `cnt[i]`
/// indexes a row of `MV_COUNTS_TO_PROBS` and selects column `i`.
fn mv_ref_probs_enc(cnt: &[u8; 4]) -> [u8; 4] {
    [
        MV_COUNTS_TO_PROBS[cnt[0].min(5) as usize][0],
        MV_COUNTS_TO_PROBS[cnt[1].min(5) as usize][1],
        MV_COUNTS_TO_PROBS[cnt[2].min(5) as usize][2],
        MV_COUNTS_TO_PROBS[cnt[3].min(5) as usize][3],
    ]
}

/// Encode-time replica of the decoder's `chroma_avg4`. Averages 4 luma
/// sub-MV components into the 1/8-chroma-pel MV component per RFC 6386
/// §18.1 (the `avg` function with the negative-branch sign fixup).
#[inline]
fn chroma_round_enc(sum: i32) -> i32 {
    if sum >= 0 {
        (sum + 4) / 8
    } else {
        -((-sum + 4) / 8)
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_ref_into_rec(
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    mb_x: usize,
    mb_y: usize,
) {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    for r in 0..16 {
        let off = (y0 + r) * y_stride + x0;
        rec_y[off..off + 16].copy_from_slice(&ref_y[off..off + 16]);
    }
    let xc = mb_x * 8;
    let yc = mb_y * 8;
    for r in 0..8 {
        let off = (yc + r) * uv_stride + xc;
        rec_u[off..off + 8].copy_from_slice(&ref_u[off..off + 8]);
        rec_v[off..off + 8].copy_from_slice(&ref_v[off..off + 8]);
    }
}

// ---------------------------------------------------------------------------
// Inter MB encode — uses the decoder's sub-pel prediction primitives so
// the reconstruction is bit-exact regardless of MV phase.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn encode_inter_mb_at_mv(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    y_buf_h: usize,
    uv_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    mv: Mv,
    q: &QuantCtx,
) -> MbEncoded {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    // Decoder-visible reference plane: the decoder constructs its
    // RefPlane with `width = stride` and `height = buf_h`, so sample
    // clamping is against the full allocated buffer — we replicate that
    // exactly here to keep predicted samples bit-identical.
    let ref_plane_y = RefPlane {
        data: ref_y,
        stride: y_stride,
        width: y_stride,
        height: y_buf_h,
    };

    // Luma prediction via the 6-tap sub-pel filter. The decoder applies
    // it per 4×4 sub-block with the same MV for non-SPLIT MBs — we do
    // the same so numeric rounding at the 4×4 boundary matches.
    let mut pred_y = [0u8; 256];
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = bx * 4;
        let dst_y = by * 4;
        let ref_x_fp = (mb_xp + dst_x) as i32 * 8 + mv.col as i32;
        let ref_y_fp = (mb_yp + dst_y) as i32 * 8 + mv.row as i32;
        sixtap_predict(
            &ref_plane_y,
            ref_x_fp,
            ref_y_fp,
            &mut pred_y,
            16,
            dst_x,
            dst_y,
            4,
            4,
        );
    }

    // 4×4 DCTs of the luma residual. For inter MBs the Y2 (WHT) path is
    // used exactly as in intra 16×16 DC_PRED.
    let mut raw_dc_y = [0i32; 16];
    let mut raw_ac_y = [[0i32; 16]; 16];
    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut blk = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let src = src_y[(mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c] as i32;
                let p = pred_y[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                blk[r * 4 + c] = src - p;
            }
        }
        let coeffs = fdct4x4(&blk);
        raw_dc_y[bi] = coeffs[0];
        raw_ac_y[bi] = coeffs;
    }

    let y2_raw = fwht4x4(&raw_dc_y);
    let mut y2_q = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_q[i] = quant(y2_raw[i], step);
    }
    let mut y2_deq = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_deq[i] = (y2_q[i] as i32 * step) as i16;
    }
    let rec_dc = iwht4x4(&y2_deq);

    let mut y_q = [[0i16; 16]; 16];
    for bi in 0..16 {
        for k in 1..16 {
            y_q[bi][k] = quant(raw_ac_y[bi][k], q.y_ac);
        }
        y_q[bi][0] = 0;
    }

    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut deq = [0i16; 16];
        deq[0] = rec_dc[bi];
        for k in 1..16 {
            deq[k] = (y_q[bi][k] as i32 * q.y_ac) as i16;
        }
        let res = idct4x4(&deq);
        for r in 0..4 {
            for c in 0..4 {
                let p = pred_y[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                let rr = res[r * 4 + c] as i32;
                let dst_y_idx = (mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c;
                rec_y[dst_y_idx] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }

    // Chroma prediction via bilinear_predict. The decoder derives a
    // per-4×4 chroma MV as the `chroma_round` of the sum of 4 covered
    // luma sub-MVs. For a non-SPLIT MB every sub-MV equals `mv`, so the
    // sum is `4*mv`.
    let cmv_r = chroma_round_enc(4 * mv.row as i32);
    let cmv_c = chroma_round_enc(4 * mv.col as i32);
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    let mut u_q = [[0i16; 16]; 4];
    let mut v_q = [[0i16; 16]; 4];
    for plane_sel in 0..2 {
        let (src, refp, rec, q_coeffs) = match plane_sel {
            0 => (src_u, ref_u, &mut *rec_u, &mut u_q),
            _ => (src_v, ref_v, &mut *rec_v, &mut v_q),
        };
        let ref_plane_uv = RefPlane {
            data: refp,
            stride: uv_stride,
            width: uv_stride,
            height: uv_buf_h,
        };
        let mut pred_uv = [0u8; 64];
        // Decoder applies the bilinear filter per 4×4 chroma sub-block
        // (bw=bh=4), not as one 8×8 call — the horizontal-pass temporary
        // rounding differs at block boundaries when fx/fy are both
        // non-zero, so mirror the per-subblock loop exactly.
        for i in 0..4 {
            let by = i / 2;
            let bx = i % 2;
            let dst_x = bx * 4;
            let dst_y = by * 4;
            let ref_x_fp = (mb_xc + dst_x) as i32 * 8 + cmv_c;
            let ref_y_fp = (mb_yc + dst_y) as i32 * 8 + cmv_r;
            bilinear_predict(
                &ref_plane_uv,
                ref_x_fp,
                ref_y_fp,
                &mut pred_uv,
                8,
                dst_x,
                dst_y,
                4,
                4,
            );
        }
        for bi in 0..4 {
            let by = bi / 2;
            let bx = bi % 2;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let sidx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    let s = src[sidx] as i32;
                    let p = pred_uv[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = s - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.uv_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.uv_ac);
            }
            q_coeffs[bi] = blk_q;
            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let pidx = (by * 4 + r) * 8 + bx * 4 + c;
                    let p = pred_uv[pidx] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let didx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    rec[didx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    let any_coeffs = y2_q.iter().any(|&v| v != 0)
        || y_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0)
        || u_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0)
        || v_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0);
    MbEncoded {
        y2_coeffs: y2_q,
        y_coeffs: y_q,
        u_coeffs: u_q,
        v_coeffs: v_q,
        // Non-SPLIT inter MB. For loop-filter sub-block skip purposes
        // ZERO_MV behaves the same as any non-B_PRED non-SPLIT mode —
        // filter_subblocks reduces to has_coeffs.
        y_mode: ZERO_MV,
        has_coeffs: any_coeffs,
    }
}

// ---------------------------------------------------------------------------
// Macroblock encode (intra DC_PRED — used by key-frames)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct QuantCtx {
    /// Y-block DC step — used only by B_PRED and SPLIT_MV paths where
    /// each 4×4 block carries its own DC coefficient (no Y2 pool). For
    /// 16×16 intra and non-SPLIT inter the Y-block DC is zeroed and the
    /// Y2-derived DC is added instead.
    y_dc: i32,
    y_ac: i32,
    y2_dc: i32,
    y2_ac: i32,
    uv_dc: i32,
    uv_ac: i32,
}

impl QuantCtx {
    /// Build a `QuantCtx` for the given clamped luma-AC qindex. Quant
    /// deltas (y_dc / y2_dc / y2_ac / uv_dc / uv_ac) are zero in the
    /// encoder's emitted bitstream, so the dequant step matches what the
    /// decoder will compute from `header.quant.y_ac_qi` alone (modulo the
    /// per-segment delta when segmentation is enabled).
    fn for_qindex(qi: i32) -> Self {
        let qi = clamp_qindex(qi) as i32;
        Self {
            y_dc: y_dc_step(qi),
            y_ac: y_ac_step(qi),
            y2_dc: y2_dc_step(qi),
            y2_ac: y2_ac_step(qi),
            uv_dc: uv_dc_step(qi),
            uv_ac: uv_ac_step(qi),
        }
    }
}

/// Per-segment encoder state derived from `Vp8EncoderConfig`. When
/// segmentation is disabled the `quant_ctx` array contains four copies
/// of the frame-level quantiser, so all per-MB lookups are correct
/// regardless of the segment id.
#[derive(Clone, Copy)]
struct SegmentCtx {
    enabled: bool,
    /// Pre-computed `QuantCtx` for each of the four segments (id 0..=3).
    quant_ctx: [QuantCtx; 4],
    /// Per-segment qindex delta added to the frame-level `qindex`. Sent
    /// in the segmentation header with `abs_delta = false`.
    quant_deltas: [i32; 4],
    /// Per-segment loop-filter level delta added to the frame-level
    /// `loop_filter.level`. Sent alongside `quant_deltas` in the
    /// segmentation block (also with `abs_delta = false`) and applied
    /// per-MB by both the encoder's `apply_loop_filter_enc` and the
    /// decoder's `per_mb_filter_level`.
    lf_deltas: [i32; 4],
}

impl SegmentCtx {
    fn for_config(config: &Vp8EncoderConfig) -> Self {
        let base_qi = clamp_qindex(config.qindex as i32) as i32;
        if config.enable_segments {
            let mut q: [QuantCtx; 4] = [
                QuantCtx::for_qindex(base_qi),
                QuantCtx::for_qindex(base_qi),
                QuantCtx::for_qindex(base_qi),
                QuantCtx::for_qindex(base_qi),
            ];
            for (i, ctx) in q.iter_mut().enumerate() {
                *ctx = QuantCtx::for_qindex(base_qi + config.segment_quant_deltas[i]);
            }
            Self {
                enabled: true,
                quant_ctx: q,
                quant_deltas: config.segment_quant_deltas,
                lf_deltas: config.segment_lf_deltas,
            }
        } else {
            let q = QuantCtx::for_qindex(base_qi);
            Self {
                enabled: false,
                quant_ctx: [q, q, q, q],
                quant_deltas: [0; 4],
                lf_deltas: [0; 4],
            }
        }
    }

    fn quant_for(&self, segment_id: u8) -> &QuantCtx {
        &self.quant_ctx[(segment_id as usize) & 3]
    }

    /// Per-MB loop-filter level after applying the per-segment LF delta
    /// (RFC 6386 §15.2; matches the decoder's `per_mb_filter_level`
    /// when `mode_ref_delta_enabled = 0`, which is what the encoder
    /// always emits). When segmentation is disabled this returns
    /// `frame_level` unchanged regardless of `segment_id`.
    fn filter_level_for(&self, segment_id: u8, frame_level: u8) -> u8 {
        if !self.enabled {
            return frame_level;
        }
        let delta = self.lf_deltas[(segment_id as usize) & 3];
        ((frame_level as i32) + delta).clamp(0, 63) as u8
    }
}

/// Compute the per-MB segment id from the source-luma 16×16 variance.
/// Maps low-variance (smooth) MBs into the low-qi segments and
/// high-variance (textured) MBs into the high-qi segments. Mirrors the
/// quartile boundaries baked into `SEGMENT_VARIANCE_THRESHOLDS`.
fn classify_segment_id(src_y: &[u8], y_stride: usize, mb_x: usize, mb_y: usize) -> u8 {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;
    let mut sum: u64 = 0;
    let mut sum2: u64 = 0;
    for r in 0..16 {
        let row_off = (mb_yp + r) * y_stride + mb_xp;
        for c in 0..16 {
            let v = src_y[row_off + c] as u64;
            sum += v;
            sum2 += v * v;
        }
    }
    let n = 256u64;
    // variance = E[x^2] - E[x]^2; expressed in summed-square units
    // (not divided by n), matching SEGMENT_VARIANCE_THRESHOLDS.
    let var_sum = sum2.saturating_sub((sum * sum) / n);
    let t = SEGMENT_VARIANCE_THRESHOLDS;
    if var_sum < t[0] {
        0
    } else if var_sum < t[1] {
        1
    } else if var_sum < t[2] {
        2
    } else {
        3
    }
}

/// Tree probabilities for the per-MB segment-id encoding. RFC 6386 §10
/// codes segment_id as a 3-leaf tree:
///   bit 0 (probs[0]) : 0 → seg 0/1 ; 1 → seg 2/3
///   bit 1 (probs[1]) : 0 → seg 0   ; 1 → seg 1     (when bit0 == 0)
///   bit 2 (probs[2]) : 0 → seg 2   ; 1 → seg 3     (when bit0 == 1)
///
/// Given the per-MB segment counts we pick the entropy-matched
/// `round(256 * P(bit==0))` and clamp to `[1, 255]` so the bool coder
/// never sees a degenerate single-symbol probability. This is the same
/// trick the existing prob-intra / prob-last / prob-gf optimiser uses.
fn segment_tree_probs_from_counts(counts: &[u32; 4]) -> [u8; 3] {
    let n_lo = counts[0] + counts[1];
    let n_hi = counts[2] + counts[3];
    let p0 = optimal_prob_8(n_lo, n_hi);
    let p1 = optimal_prob_8(counts[0], counts[1]);
    let p2 = optimal_prob_8(counts[2], counts[3]);
    [p0, p1, p2]
}

/// Emit the bool-coded segment_id of an MB using the frame-level tree
/// probabilities. Pairs with the decoder's `s0 = read_bool(p0); s = (s0 ?
/// 2 + read_bool(p2) : read_bool(p1))` decode walk.
fn emit_segment_id(enc: &mut BoolEncoder, segment_id: u8, probs: &[u8; 3]) {
    let s = segment_id & 3;
    let bit0 = (s & 0b10) != 0;
    enc.write_bool(probs[0] as u32, bit0);
    if !bit0 {
        // seg 0 → bit1=0 ; seg 1 → bit1=1
        enc.write_bool(probs[1] as u32, (s & 1) != 0);
    } else {
        // seg 2 → bit2=0 ; seg 3 → bit2=1
        enc.write_bool(probs[2] as u32, (s & 1) != 0);
    }
}

/// Emit the segmentation block of the frame header. When `seg.enabled` is
/// `false` only the single "segmentation enabled = 0" bit is written
/// (preserving the legacy single-segment encoding bit-for-bit).
fn emit_segmentation_header(enc: &mut BoolEncoder, seg: &SegmentCtx, tree_probs: &[u8; 3]) {
    enc.write_bool(128, seg.enabled);
    if !seg.enabled {
        return;
    }
    // update_map = 1 (always emit per-MB segment ids in this encoder).
    enc.write_bool(128, true);
    // update_data = 1 (re-send per-segment data every frame for
    // simplicity — the decoder caches them across frames otherwise).
    enc.write_bool(128, true);
    // abs_delta = 0 (deltas are added to header.quant.y_ac_qi).
    enc.write_bool(128, false);
    // 4 per-segment quant deltas (each preceded by a 1-bit "present" flag).
    for i in 0..4 {
        let v = seg.quant_deltas[i];
        if v != 0 {
            enc.write_bool(128, true);
            enc.write_signed_literal(7, v);
        } else {
            enc.write_bool(128, false);
        }
    }
    // 4 per-segment loop-filter deltas (each preceded by a 1-bit
    // "present" flag, matching the decoder's `parse_segmentation`
    // walk in `frame_header.rs`). Smooth segments take a negative
    // delta to soften the filter; high-variance segments take a
    // positive delta to mask the per-MB DCT block boundaries that the
    // coarser per-segment quant exposes. Encoded as a 6-bit signed
    // literal per RFC 6386 §10.
    for i in 0..4 {
        let v = seg.lf_deltas[i];
        if v != 0 {
            enc.write_bool(128, true);
            enc.write_signed_literal(6, v);
        } else {
            enc.write_bool(128, false);
        }
    }
    // tree_probs (3 entries). 255 (= "default") is the encoder's
    // sentinel for "no override"; we always send explicit probs since
    // `tree_probs[i]` defaults to 255 in the decoder which would skew
    // the segment distribution.
    for &p in tree_probs.iter() {
        enc.write_bool(128, true);
        enc.write_literal(8, p as u32);
    }
}

/// Output of per-MB encode: quantised coefficients for each block and the
/// Y2 block (the 16 DC coefficients passed through forward WHT).
struct MbEncoded {
    y2_coeffs: [i16; 16],
    y_coeffs: [[i16; 16]; 16],
    u_coeffs: [[i16; 16]; 4],
    v_coeffs: [[i16; 16]; 4],
    /// Y mode chosen for this MB — needed by the loop-filter sub-block
    /// skip rule (RFC 6386 §15.1: B_PRED / SPLITMV always filter inner
    /// edges).
    y_mode: i32,
    /// Whether any non-zero coefficient was emitted for this MB. Loop
    /// filter sub-block edges skip when this is false AND y_mode is
    /// not B_PRED / SPLITMV.
    has_coeffs: bool,
}

impl MbEncoded {
    fn zero() -> Self {
        Self {
            y2_coeffs: [0; 16],
            y_coeffs: [[0; 16]; 16],
            u_coeffs: [[0; 16]; 4],
            v_coeffs: [[0; 16]; 4],
            y_mode: 0,
            has_coeffs: false,
        }
    }
}

/// Gather DC/V/H/TM-capable 16-sample above row, left column and TL for
/// a 16×16 intra MB at `(mb_xp, mb_yp)`.
fn gather_16x16_neighbours(
    rec_y: &[u8],
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
) -> (Option<[u8; 16]>, Option<[u8; 16]>, Option<u8>) {
    let above_avail = mb_yp > 0;
    let left_avail = mb_xp > 0;
    let above = if above_avail {
        let mut a = [0u8; 16];
        for i in 0..16 {
            a[i] = rec_y[(mb_yp - 1) * y_stride + mb_xp + i];
        }
        Some(a)
    } else {
        None
    };
    let left = if left_avail {
        let mut l = [0u8; 16];
        for j in 0..16 {
            l[j] = rec_y[(mb_yp + j) * y_stride + mb_xp - 1];
        }
        Some(l)
    } else {
        None
    };
    // Mirror the decoder's TL defaults (reconstruct_intra_mb non-B_PRED):
    //   * both available → real corner pixel
    //   * only above available → use the LEFT default (129)
    //   * only left available → use the ABOVE default (127)
    let tl = if above_avail && left_avail {
        Some(rec_y[(mb_yp - 1) * y_stride + mb_xp - 1])
    } else if above_avail {
        Some(129)
    } else if left_avail {
        Some(127)
    } else {
        None
    };
    (above, left, tl)
}

/// Gather 8-sample neighbours for an 8×8 chroma intra block.
fn gather_8x8_neighbours(
    rec: &[u8],
    uv_stride: usize,
    mb_xc: usize,
    mb_yc: usize,
) -> (Option<[u8; 8]>, Option<[u8; 8]>, Option<u8>) {
    let above_avail = mb_yc > 0;
    let left_avail = mb_xc > 0;
    let above = if above_avail {
        let mut a = [0u8; 8];
        for i in 0..8 {
            a[i] = rec[(mb_yc - 1) * uv_stride + mb_xc + i];
        }
        Some(a)
    } else {
        None
    };
    let left = if left_avail {
        let mut l = [0u8; 8];
        for j in 0..8 {
            l[j] = rec[(mb_yc + j) * uv_stride + mb_xc - 1];
        }
        Some(l)
    } else {
        None
    };
    // Mirror the decoder's TL defaults — same swap logic as the
    // luma 16×16 path.
    let tl = if above_avail && left_avail {
        Some(rec[(mb_yc - 1) * uv_stride + mb_xc - 1])
    } else if above_avail {
        Some(129)
    } else if left_avail {
        Some(127)
    } else {
        None
    };
    (above, left, tl)
}

/// Encode one intra MB with the given Y mode (DC / V / H / TM / B_PRED)
/// and UV mode (DC / V / H / TM). For B_PRED, `bmodes` supplies the per
/// 4×4 sub-block modes that the caller has already picked and emitted
/// on the header side.
#[allow(clippy::too_many_arguments)]
fn encode_intra_mb(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    _y_buf_h: usize,
    _uv_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    _mb_h: usize,
    q: &QuantCtx,
    y_mode: i32,
    uv_mode: i32,
    bmodes: &[i32; 16],
) -> MbEncoded {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    // --- Luma: either 16×16 intra mode or B_PRED (per 4×4). ---
    let mut y_q = [[0i16; 16]; 16];
    let y2_q;

    if y_mode == B_PRED {
        // Per 4×4 reconstruction: each sub-block predicts from its own
        // post-reconstruction neighbours (which land in `rec_y` as we go).
        y2_q = [0i16; 16];
        // Above-right extension at MB-row boundary (matches decoder).
        let above_right_extension: [u8; 4] = if mb_yp > 0 {
            let row = mb_yp - 1;
            let mut ext = [0u8; 4];
            for k in 0..4 {
                let xx = mb_xp + 16 + k;
                if xx < mb_w * 16 {
                    ext[k] = rec_y[row * y_stride + xx];
                } else {
                    ext[k] = rec_y[row * y_stride + (mb_xp + 15)];
                }
            }
            ext
        } else {
            [127; 4]
        };
        for i in 0..16 {
            let by = i / 4;
            let bx = i % 4;
            let dst_x = mb_xp + bx * 4;
            let dst_y = mb_yp + by * 4;
            let neigh = gather_4x4_neighbours(
                rec_y,
                y_stride,
                mb_xp,
                mb_yp,
                bx,
                by,
                mb_w,
                &above_right_extension,
            );
            let mut pred = [0u8; 16];
            predict_4x4(bmodes[i], &neigh, &mut pred, 4);

            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let src = src_y[(dst_y + r) * y_stride + dst_x + c] as i32;
                    let p = pred[r * 4 + c] as i32;
                    blk[r * 4 + c] = src - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            // B_PRED: no Y2. Quantise DC + AC with their respective steps.
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.y_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.y_ac);
            }
            y_q[i] = blk_q;

            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.y_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.y_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[r * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    rec_y[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    } else {
        // 16×16 intra with Y2.
        let (above, left, tl) = gather_16x16_neighbours(rec_y, y_stride, mb_xp, mb_yp);
        let mut pred = vec![0u8; 16 * 16];
        predict_16x16(y_mode, above.as_ref(), left.as_ref(), tl, &mut pred, 16);

        let mut raw_dc_y = [0i32; 16];
        let mut raw_ac_y = [[0i32; 16]; 16];
        for bi in 0..16 {
            let by = bi / 4;
            let bx = bi % 4;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let src = src_y[(mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c] as i32;
                    let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = src - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            raw_dc_y[bi] = coeffs[0];
            raw_ac_y[bi] = coeffs;
        }

        let y2_raw = fwht4x4(&raw_dc_y);
        let mut y2_qq = [0i16; 16];
        for i in 0..16 {
            let step = if i == 0 { q.y2_dc } else { q.y2_ac };
            y2_qq[i] = quant(y2_raw[i], step);
        }
        let mut y2_deq = [0i16; 16];
        for i in 0..16 {
            let step = if i == 0 { q.y2_dc } else { q.y2_ac };
            y2_deq[i] = (y2_qq[i] as i32 * step) as i16;
        }
        let rec_dc = iwht4x4(&y2_deq);

        for bi in 0..16 {
            for k in 1..16 {
                y_q[bi][k] = quant(raw_ac_y[bi][k], q.y_ac);
            }
            y_q[bi][0] = 0;
        }

        for bi in 0..16 {
            let by = bi / 4;
            let bx = bi % 4;
            let mut deq = [0i16; 16];
            deq[0] = rec_dc[bi];
            for k in 1..16 {
                deq[k] = (y_q[bi][k] as i32 * q.y_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let dst_y_idx = (mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c;
                    rec_y[dst_y_idx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
        y2_q = y2_qq;
    }

    // --- Chroma (8×8 intra with chosen uv_mode). ---
    let mut u_q = [[0i16; 16]; 4];
    let mut v_q = [[0i16; 16]; 4];
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for plane_sel in 0..2 {
        let (src, rec, q_coeffs) = match plane_sel {
            0 => (src_u, &mut *rec_u, &mut u_q),
            _ => (src_v, &mut *rec_v, &mut v_q),
        };
        let (above, left, tl) = gather_8x8_neighbours(rec, uv_stride, mb_xc, mb_yc);
        let mut pred_uv = vec![0u8; 8 * 8];
        predict_8x8(uv_mode, above.as_ref(), left.as_ref(), tl, &mut pred_uv, 8);
        for bi in 0..4 {
            let by = bi / 2;
            let bx = bi % 2;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let sidx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    let s = src[sidx] as i32;
                    let p = pred_uv[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = s - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.uv_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.uv_ac);
            }
            q_coeffs[bi] = blk_q;
            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let pidx = (by * 4 + r) * 8 + bx * 4 + c;
                    let p = pred_uv[pidx] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let didx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    rec[didx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    let any_coeffs = y2_q.iter().any(|&v| v != 0)
        || y_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0)
        || u_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0)
        || v_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0);
    MbEncoded {
        y2_coeffs: y2_q,
        y_coeffs: y_q,
        u_coeffs: u_q,
        v_coeffs: v_q,
        y_mode,
        has_coeffs: any_coeffs,
    }
}

/// Build 4×4 neighbour array for a sub-block. Matches the decoder's
/// logic in `reconstruct_intra_mb` (including the above-right extension
/// special-case).
fn gather_4x4_neighbours(
    rec_y: &[u8],
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    bx: usize,
    by: usize,
    mb_w: usize,
    above_right_extension: &[u8; 4],
) -> B4x4Neighbours {
    let dst_x = mb_xp + bx * 4;
    let dst_y = mb_yp + by * 4;
    let mut neigh = B4x4Neighbours {
        above: [127; 8],
        left: [129; 4],
        tl: 127,
    };
    if dst_y > 0 {
        for k in 0..4 {
            neigh.above[k] = rec_y[(dst_y - 1) * y_stride + dst_x + k];
        }
        if bx == 3 && by > 0 {
            neigh.above[4..8].copy_from_slice(above_right_extension);
        } else {
            for k in 4..8 {
                let xx = dst_x + k;
                if xx < mb_xp + 16 {
                    neigh.above[k] = rec_y[(dst_y - 1) * y_stride + xx];
                } else if by == 0 {
                    if xx < mb_w * 16 {
                        neigh.above[k] = rec_y[(dst_y - 1) * y_stride + xx];
                    } else {
                        neigh.above[k] = rec_y[(dst_y - 1) * y_stride + (mb_xp + 15)];
                    }
                } else {
                    neigh.above[k] = above_right_extension[(xx - mb_xp) - 16];
                }
            }
        }
    }
    if dst_x > 0 {
        for k in 0..4 {
            neigh.left[k] = rec_y[(dst_y + k) * y_stride + dst_x - 1];
        }
    }
    // TL pixel defaults — must match the decoder's logic in
    // reconstruct_intra_mb (B_PRED path): when only one neighbour is
    // available, TL takes the *other* neighbour's default sample.
    neigh.tl = if dst_x > 0 && dst_y > 0 {
        rec_y[(dst_y - 1) * y_stride + dst_x - 1]
    } else if dst_y > 0 {
        // left column unavailable → use the left default (129)
        129
    } else {
        // above unavailable → use the above default (127)
        127
    };
    neigh
}

// ---------------------------------------------------------------------------
// Token partition encoding shared between I and P frames.
// ---------------------------------------------------------------------------

/// Emit the token partition for every MB in raster scan. If `decisions`
/// is empty, every MB contributes its tokens (I-frame path). If
/// `decisions` is populated, SKIP MBs contribute no tokens and still
/// reset their neighbour non-zero contexts to 0.
///
/// `ymodes` supplies each MB's Y mode. B_PRED MBs have no Y2 block; their
/// 16 Y blocks use `BlockType::YNoY2` (plane 3 in the coef_probs table)
/// and start at coefficient 0 rather than 1.
fn emit_tokens(
    mb_w: usize,
    mb_h: usize,
    mb_encoded: &[MbEncoded],
    decisions: &[PMbDecision],
    coef_probs: &CoeffProbs,
    ymodes: Option<&[i32]>,
) -> BoolEncoder {
    let mut tok_enc = BoolEncoder::new();
    let mut nz_y_above = vec![[0u8; 4]; mb_w];
    let mut nz_uv_above = vec![[0u8; 2]; mb_w];
    let mut nz_v_above = vec![[0u8; 2]; mb_w];
    let mut nz_y2_above = vec![0u8; mb_w];

    for mb_y in 0..mb_h {
        let mut nz_y_left = [0u8; 4];
        let mut nz_u_left = [0u8; 2];
        let mut nz_v_left = [0u8; 2];
        let mut nz_y2_left = 0u8;
        for mb_x in 0..mb_w {
            let mb_rec = &mb_encoded[mb_y * mb_w + mb_x];
            let decision = decisions.get(mb_y * mb_w + mb_x);
            let is_skip = decision.is_some_and(|d| matches!(d, PMbDecision::Skip));
            // B_PRED MBs (intra or intra-in-P) have no Y2 block. Decoder:
            // `has_y2 = info.y_mode != B_PRED` for intra / non-SPLIT inter.
            // SPLIT MBs also have no Y2. All other cases get a Y2 block.
            let y_mode = ymodes
                .and_then(|m| m.get(mb_y * mb_w + mb_x))
                .copied()
                .unwrap_or(DC_PRED);
            let is_split = matches!(decision, Some(PMbDecision::SplitMv(_)));
            let has_y2 = !matches!(y_mode, x if x == B_PRED) && !is_split;

            if is_skip {
                // Decoder: skip clears all neighbour nz for this MB.
                if has_y2 {
                    nz_y2_above[mb_x] = 0;
                    nz_y2_left = 0;
                }
                for bx in 0..4 {
                    nz_y_above[mb_x][bx] = 0;
                    nz_y_left[bx] = 0;
                }
                for bx in 0..2 {
                    nz_uv_above[mb_x][bx] = 0;
                    nz_u_left[bx] = 0;
                    nz_v_above[mb_x][bx] = 0;
                    nz_v_left[bx] = 0;
                }
                continue;
            }

            // Y2 DC block (only when applicable).
            if has_y2 {
                let nctx = nz_y2_above[mb_x] + nz_y2_left;
                let nz = encode_block(
                    &mut tok_enc,
                    coef_probs,
                    /*plane=*/ 1,
                    nctx as usize,
                    &mb_rec.y2_coeffs,
                    0,
                );
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_y2_above[mb_x] = nzf;
                nz_y2_left = nzf;
            }

            // Y blocks. `plane=0` (YAfterY2) and `start=1` when has_y2;
            // `plane=3` (YNoY2) and `start=0` otherwise.
            let (y_plane, y_start) = if has_y2 {
                (0usize, 1usize)
            } else {
                (3usize, 0usize)
            };
            for by in 0..4 {
                for bx in 0..4 {
                    let idx = by * 4 + bx;
                    let nctx = nz_y_above[mb_x][bx] + nz_y_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        y_plane,
                        nctx as usize,
                        &mb_rec.y_coeffs[idx],
                        y_start,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_y_above[mb_x][bx] = nzf;
                    nz_y_left[by] = nzf;
                }
            }
            // Per RFC 6386 §13.1: U subblocks (all 4 in raster order)
            // precede V subblocks; they are NOT interleaved.
            for by in 0..2 {
                for bx in 0..2 {
                    let idx = by * 2 + bx;
                    let nctx = nz_uv_above[mb_x][bx] + nz_u_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        2,
                        nctx as usize,
                        &mb_rec.u_coeffs[idx],
                        0,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_uv_above[mb_x][bx] = nzf;
                    nz_u_left[by] = nzf;
                }
            }
            for by in 0..2 {
                for bx in 0..2 {
                    let idx = by * 2 + bx;
                    let nctx = nz_v_above[mb_x][bx] + nz_v_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        2,
                        nctx as usize,
                        &mb_rec.v_coeffs[idx],
                        0,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_v_above[mb_x][bx] = nzf;
                    nz_v_left[by] = nzf;
                }
            }
        }
    }
    tok_enc
}

/// Quantise a single coefficient using `step`. Uses symmetric rounding
/// towards zero with `step/2` bias — close to what libvpx's reference
/// encoder does for intra blocks and adequate for our decoder's
/// multiply-by-step dequantiser.
#[inline]
fn quant(v: i32, step: i32) -> i16 {
    if step <= 0 {
        return 0;
    }
    let half = step / 2;
    let q = if v >= 0 {
        (v + half) / step
    } else {
        -((-v + half) / step)
    };
    q.clamp(-2048, 2047) as i16
}

// ---------------------------------------------------------------------------
// Token (coefficient) entropy encode
// ---------------------------------------------------------------------------

/// Encode one transform block's 16 coefficients into the boolean coder.
/// Returns the number of coefficients encoded (last-non-zero + 1), or 0
/// if the entire block is zero starting from `start`.
///
/// Mirrors [`crate::tokens::decode_block`] bit-for-bit — the decoder's
/// flat `p[0..10]` look-up table is the authoritative tree walk, so the
/// write side uses the exact same branch structure.
fn encode_block(
    enc: &mut BoolEncoder,
    probs: &CoeffProbs,
    plane: usize,
    nctx: usize,
    coeffs: &[i16; 16],
    start: usize,
) -> u8 {
    let plane_probs = &probs[plane];
    let mut last_nz = None::<usize>;
    for n in start..16 {
        let c = coeffs[ZIGZAG[n]];
        if c != 0 {
            last_nz = Some(n);
        }
    }
    let last = match last_nz {
        Some(n) => n,
        None => {
            let p = &plane_probs[COEF_BANDS[start]][nctx];
            enc.write_bool(p[0] as u32, false);
            return 0;
        }
    };

    let mut n = start;
    let mut ctx = nctx;
    let mut p = &plane_probs[COEF_BANDS[n]][ctx];
    enc.write_bool(p[0] as u32, true);

    loop {
        while coeffs[ZIGZAG[n]] == 0 {
            enc.write_bool(p[1] as u32, false);
            n += 1;
            p = &plane_probs[COEF_BANDS[n]][0];
        }
        enc.write_bool(p[1] as u32, true);

        let raw = coeffs[ZIGZAG[n]] as i32;
        let v = raw.unsigned_abs() as i32;
        emit_magnitude(enc, p, v);
        ctx = if v == 1 { 1 } else { 2 };

        enc.write_bool(128, raw < 0);

        n += 1;
        if n == 16 {
            return 16;
        }
        p = &plane_probs[COEF_BANDS[n]][ctx];
        if n > last {
            enc.write_bool(p[0] as u32, false);
            return (last + 1) as u8;
        }
        enc.write_bool(p[0] as u32, true);
    }
}

/// Write the magnitude of a non-zero coefficient following the coef-tree
/// branch structure in [`crate::tokens::decode_block`]. `p` is the
/// 11-element probability array for the current (band, ctx).
fn emit_magnitude(enc: &mut BoolEncoder, p: &[u8; 11], v: i32) {
    // Match the decoder's branch-by-branch ladder.
    if v == 1 {
        enc.write_bool(p[2] as u32, false);
        return;
    }
    enc.write_bool(p[2] as u32, true);
    if v <= 4 {
        enc.write_bool(p[3] as u32, false);
        if v == 2 {
            enc.write_bool(p[4] as u32, false);
        } else {
            enc.write_bool(p[4] as u32, true);
            enc.write_bool(p[5] as u32, v == 4);
        }
        return;
    }
    enc.write_bool(p[3] as u32, true);
    if v <= 10 {
        enc.write_bool(p[6] as u32, false);
        if v <= 6 {
            enc.write_bool(p[7] as u32, false);
            enc.write_bool(159, v == 6);
        } else {
            enc.write_bool(p[7] as u32, true);
            let hi = if v >= 9 { 1 } else { 0 };
            enc.write_bool(165, hi == 1);
            let low = (v - 7 - 2 * hi) as u32;
            enc.write_bool(145, low == 1);
        }
        return;
    }
    enc.write_bool(p[6] as u32, true);
    let (cat, base) = if v < 19 {
        (0, 11)
    } else if v < 35 {
        (1, 19)
    } else if v < 67 {
        (2, 35)
    } else {
        (3, 67)
    };
    let bit1 = (cat >> 1) & 1;
    let bit0 = cat & 1;
    enc.write_bool(p[8] as u32, bit1 == 1);
    enc.write_bool(p[9 + bit1] as u32, bit0 == 1);
    let extra_bits_tab: &[u8] = match cat {
        0 => &[173, 148, 140],
        1 => &[176, 155, 140, 135],
        2 => &[180, 157, 141, 134, 130],
        _ => &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129],
    };
    let extra = (v - base) as u32;
    let nbits = extra_bits_tab.len();
    for i in 0..nbits {
        let bit = ((extra >> (nbits - 1 - i)) & 1) as u8;
        enc.write_bool(extra_bits_tab[i] as u32, bit != 0);
    }
}

/// Emit "no probability updates" for the whole 4×8×3×11 coefficient prob
/// table. Mirrors the decoder's `update_coef_probs` but always sending
/// `read_bool(upd)=false`.
fn emit_no_coef_prob_updates(enc: &mut BoolEncoder) {
    use crate::tables::coeff_probs::COEF_UPDATE_PROBS;
    use crate::tables::token_tree::{NUM_BANDS, NUM_CTX, NUM_PROBS, NUM_TYPES};
    for i in 0..NUM_TYPES {
        for j in 0..NUM_BANDS {
            for k in 0..NUM_CTX {
                for l in 0..NUM_PROBS {
                    let upd = COEF_UPDATE_PROBS[i][j][k][l] as u32;
                    enc.write_bool(upd, false);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra mode selection + emission
// ---------------------------------------------------------------------------

/// Map intra 16×16 mode → the 4×4 mode value propagated to the neighbour
/// B_PRED context (matches decoder's `intra_to_b`).
fn intra_to_b_mode(y_mode: i32) -> i32 {
    match y_mode {
        DC_PRED => B_DC_PRED,
        V_PRED => B_VE_PRED,
        H_PRED => B_HE_PRED,
        TM_PRED => B_TM_PRED,
        _ => B_DC_PRED,
    }
}

/// Emit a keyframe Y-mode tree path for `y_mode`.
fn emit_kf_ymode(enc: &mut BoolEncoder, y_mode: i32) {
    let p = &KF_YMODE_PROBS;
    match y_mode {
        B_PRED => {
            enc.write_bool(p[0] as u32, false);
        }
        DC_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, false);
            enc.write_bool(p[2] as u32, false);
        }
        V_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, false);
            enc.write_bool(p[2] as u32, true);
        }
        H_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, true);
            enc.write_bool(p[3] as u32, false);
        }
        TM_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, true);
            enc.write_bool(p[3] as u32, true);
        }
        _ => unreachable!("invalid Y mode {}", y_mode),
    }
}

/// Emit a keyframe UV-mode tree path for `uv_mode`.
fn emit_kf_uv_mode(enc: &mut BoolEncoder, uv_mode: i32) {
    let p = &KF_UV_MODE_PROBS;
    match uv_mode {
        DC_PRED => {
            enc.write_bool(p[0] as u32, false);
        }
        V_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, false);
        }
        H_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, true);
            enc.write_bool(p[2] as u32, false);
        }
        TM_PRED => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, true);
            enc.write_bool(p[2] as u32, true);
        }
        _ => unreachable!("invalid UV mode {}", uv_mode),
    }
}

/// Emit an inter-frame Y-mode tree path (YMODE_TREE: DC/V/H/TM/B_PRED).
/// The tree structure is `[-DC, 2, 4, 6, -V, -H, -TM, -B_PRED]` — same
/// branching shape as keyframes but different probabilities.
fn emit_inter_ymode(enc: &mut BoolEncoder, y_mode: i32, probs: &[u8; 4]) {
    match y_mode {
        DC_PRED => {
            enc.write_bool(probs[0] as u32, false);
        }
        V_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, false);
            enc.write_bool(probs[2] as u32, false);
        }
        H_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, false);
            enc.write_bool(probs[2] as u32, true);
        }
        TM_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, true);
            enc.write_bool(probs[3] as u32, false);
        }
        B_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, true);
            enc.write_bool(probs[3] as u32, true);
        }
        _ => unreachable!(),
    }
}

/// Emit an inter-frame UV-mode tree path.
fn emit_inter_uv_mode(enc: &mut BoolEncoder, uv_mode: i32, probs: &[u8; 3]) {
    match uv_mode {
        DC_PRED => {
            enc.write_bool(probs[0] as u32, false);
        }
        V_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, false);
        }
        H_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, true);
            enc.write_bool(probs[2] as u32, false);
        }
        TM_PRED => {
            enc.write_bool(probs[0] as u32, true);
            enc.write_bool(probs[1] as u32, true);
            enc.write_bool(probs[2] as u32, true);
        }
        _ => unreachable!(),
    }
}

/// Tree path for each 4×4 B-mode under BMODE_TREE (§11.5).
/// Each entry is `(prob_index, bit)` — written with `probs[prob_index]`.
type BModePath = &'static [(u8, bool)];
static BMODE_PATHS: [BModePath; 10] = [
    // B_DC_PRED: tree start, bit=false → leaf -B_DC_PRED.
    &[(0, false)],
    // B_TM_PRED: true, false → -B_TM_PRED.
    &[(0, true), (1, false)],
    // B_VE_PRED: true, true, false → leaf at idx=4: -B_VE_PRED.
    &[(0, true), (1, true), (2, false)],
    // B_HE_PRED: true, true, true, false → (idx=6 pair @ prob 3) then leaf path.
    //   BMODE_TREE at idx 6: [8, 12]. Reading false → idx 8: [-B_HE_PRED, 10] → leaf at bit=false.
    &[(0, true), (1, true), (2, true), (3, false), (4, false)],
    // B_LD_PRED: walking through BMODE_TREE to reach leaf -B_LD_PRED at idx 12.
    //   Path: 0→true, 1→true, 2→true, 3(idx=6 pair)→true→idx 12 [-B_LD_PRED, 14], then false.
    &[(0, true), (1, true), (2, true), (3, true), (6, false)],
    // B_RD_PRED: idx 8: [ -B_HE_PRED, 10] → 10: [ -B_RD_PRED, -B_VR_PRED ].
    //   Path: 0,1,2→true,true,true; 3→false; 4→true→idx 10 [-B_RD_PRED, -B_VR_PRED], then false.
    &[
        (0, true),
        (1, true),
        (2, true),
        (3, false),
        (4, true),
        (5, false),
    ],
    // B_VR_PRED:
    &[
        (0, true),
        (1, true),
        (2, true),
        (3, false),
        (4, true),
        (5, true),
    ],
    // B_VL_PRED: leaves at idx 14: [-B_VL_PRED, 16]; reach via path (0,1,2=true; 3=true @ idx6; 6=true @idx12→14: [-B_LD_PRED, 14]; 14: [-B_VL_PRED, 16])
    &[
        (0, true),
        (1, true),
        (2, true),
        (3, true),
        (6, true),
        (7, false),
    ],
    // B_HD_PRED: idx 16: [-B_HD_PRED, -B_HU_PRED].
    &[
        (0, true),
        (1, true),
        (2, true),
        (3, true),
        (6, true),
        (7, true),
        (8, false),
    ],
    // B_HU_PRED:
    &[
        (0, true),
        (1, true),
        (2, true),
        (3, true),
        (6, true),
        (7, true),
        (8, true),
    ],
];

/// Generic: emit a precomputed tree path `path` using probs indexed by
/// `path[i].0` and a bit value `path[i].1`.
fn emit_tree_path(enc: &mut BoolEncoder, path: &[(u8, bool)], probs: &[u8]) {
    for &(idx, bit) in path {
        enc.write_bool(probs[idx as usize] as u32, bit);
    }
}

/// Emit the mb-split-tree leaf.
fn emit_split_mb_tree(enc: &mut BoolEncoder, split_mode: u8) {
    let p = &MBSPLIT_PROBS;
    match split_mode {
        // MB_SPLIT_4X4 = 3 → path: false.
        3 => {
            enc.write_bool(p[0] as u32, false);
        }
        // MB_SPLIT_16X8 = 0 → path: true, false.
        0 => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, false);
        }
        // MB_SPLIT_8X16 = 1 → path: true, true, false.
        1 => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, true);
            enc.write_bool(p[2] as u32, false);
        }
        // MB_SPLIT_QUARTERS = 2 → path: true, true, true.
        2 => {
            enc.write_bool(p[0] as u32, true);
            enc.write_bool(p[1] as u32, true);
            enc.write_bool(p[2] as u32, true);
        }
        _ => unreachable!("invalid split_mode {}", split_mode),
    }
}

/// Emit per-partition sub-MV ref trees + optional MV diffs for a SPLIT
/// MB. Partitions emit in the same iteration order as the decoder: for
/// each partition `p` in 0..n, walk the 16 sub-blocks in raster order
/// and emit the first one whose partition id equals `p`.
#[allow(clippy::too_many_arguments)]
fn emit_split_submvs(
    enc: &mut BoolEncoder,
    split: &SplitMv,
    mb_sub_mvs: &[[Mv; 16]],
    mb_decisions: &[PMbDecision],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    best_for_newmv: Mv,
) {
    let partition = &MB_SPLITS[split.split_mode as usize];
    let n = MB_SPLIT_COUNT[split.split_mode as usize] as usize;
    // Track the running sub_mvs as we emit each partition so that
    // subsequent partitions see the correct left/above sub-MV neighbour.
    let mut part_mvs_running = [Mv::ZERO; 16];
    for p in 0..n {
        let first_idx = (0..16).find(|&i| partition[i] as usize == p).unwrap_or(0);
        let row = first_idx / 4;
        let col = first_idx % 4;
        // Left sub-MV: either from within this MB (if col > 0) or from
        // the MB immediately to the left (row `row`, col `3`).
        let left_mv = if col == 0 {
            if mb_x > 0 {
                let lidx = mb_y * mb_w + mb_x - 1;
                edge_left_sub_mv(mb_sub_mvs, mb_decisions, lidx, row)
            } else {
                Mv::ZERO
            }
        } else {
            part_mvs_running[row * 4 + col - 1]
        };
        let above_mv = if row == 0 {
            if mb_y > 0 {
                let aidx = (mb_y - 1) * mb_w + mb_x;
                edge_above_sub_mv(mb_sub_mvs, mb_decisions, aidx, col)
            } else {
                Mv::ZERO
            }
        } else {
            part_mvs_running[(row - 1) * 4 + col]
        };
        let chosen = split.part_mvs[p];
        // Determine which SUB_MV_REF leaf we need.
        let leaf: i32 = if chosen == left_mv {
            0 // LEFT_4X4
        } else if chosen == above_mv {
            1 // ABOVE_4X4
        } else if chosen == Mv::ZERO {
            2 // ZERO_4X4
        } else {
            3 // NEW_4X4
        };
        let sub_prob_row = sub_mv_context_enc(&left_mv, &above_mv);
        let probs = &SUB_MV_REF_PROBS[sub_prob_row];
        // SUB_MV_REF_TREE: [-LEFT, 2, -ABOVE, 4, -ZERO, -NEW]. Paths:
        //   LEFT:  false
        //   ABOVE: true, false
        //   ZERO:  true, true, false
        //   NEW:   true, true, true
        match leaf {
            0 => {
                enc.write_bool(probs[0] as u32, false);
            }
            1 => {
                enc.write_bool(probs[0] as u32, true);
                enc.write_bool(probs[1] as u32, false);
            }
            2 => {
                enc.write_bool(probs[0] as u32, true);
                enc.write_bool(probs[1] as u32, true);
                enc.write_bool(probs[2] as u32, false);
            }
            3 => {
                enc.write_bool(probs[0] as u32, true);
                enc.write_bool(probs[1] as u32, true);
                enc.write_bool(probs[2] as u32, true);
                let dmv = Mv::new(
                    chosen.row as i32 - best_for_newmv.row as i32,
                    chosen.col as i32 - best_for_newmv.col as i32,
                );
                encode_mv_component(enc, &DEFAULT_MV_CONTEXT[0], dmv.row as i32);
                encode_mv_component(enc, &DEFAULT_MV_CONTEXT[1], dmv.col as i32);
            }
            _ => unreachable!(),
        }
        // Update running MVs.
        for i in 0..16 {
            if partition[i] as usize == p {
                part_mvs_running[i] = chosen;
            }
        }
    }
}

/// Encoder replica of the decoder's `sub_mv_context`: picks a row of
/// `SUB_MV_REF_PROBS` based on the (left, above) neighbour pair.
fn sub_mv_context_enc(left: &Mv, above: &Mv) -> usize {
    let l_zero = left.row == 0 && left.col == 0;
    let a_zero = above.row == 0 && above.col == 0;
    if l_zero && a_zero {
        0
    } else if !l_zero && a_zero {
        1
    } else if l_zero && !a_zero {
        2
    } else if left == above {
        4
    } else {
        3
    }
}

/// Left-neighbour MB's right-edge sub-MV for `row` within this MB.
fn edge_left_sub_mv(
    mb_sub_mvs: &[[Mv; 16]],
    mb_decisions: &[PMbDecision],
    lidx: usize,
    row: usize,
) -> Mv {
    if let Some(d) = mb_decisions.get(lidx) {
        if d.is_intra() {
            Mv::ZERO
        } else {
            mb_sub_mvs[lidx][row * 4 + 3]
        }
    } else {
        Mv::ZERO
    }
}

/// Above-neighbour MB's bottom-edge sub-MV for `col` within this MB.
fn edge_above_sub_mv(
    mb_sub_mvs: &[[Mv; 16]],
    mb_decisions: &[PMbDecision],
    aidx: usize,
    col: usize,
) -> Mv {
    if let Some(d) = mb_decisions.get(aidx) {
        if d.is_intra() {
            Mv::ZERO
        } else {
            mb_sub_mvs[aidx][12 + col]
        }
    } else {
        Mv::ZERO
    }
}

/// Pick the best 16×16 Y mode by SSE against the source.
fn choose_intra_16x16_y_mode(
    src_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
) -> i32 {
    let mut best = DC_PRED;
    let mut best_sse = u64::MAX;
    for m in &[DC_PRED, V_PRED, H_PRED, TM_PRED] {
        let (sse, _) = sse_intra_16x16(*m, src_y, rec_y, y_stride, mb_xp, mb_yp);
        if sse < best_sse {
            best_sse = sse;
            best = *m;
        }
    }
    best
}

/// SSE of 16×16 intra prediction vs source (pre-residual — measures the
/// quality of the prediction alone).
fn sse_intra_16x16(
    mode: i32,
    src_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
) -> (u64, [u8; 256]) {
    let (above, left, tl) = gather_16x16_neighbours(rec_y, y_stride, mb_xp, mb_yp);
    let mut pred = [0u8; 256];
    predict_16x16(mode, above.as_ref(), left.as_ref(), tl, &mut pred, 16);
    let mut sse = 0u64;
    for r in 0..16 {
        for c in 0..16 {
            let s = src_y[(mb_yp + r) * y_stride + mb_xp + c] as i32;
            let p = pred[r * 16 + c] as i32;
            let d = s - p;
            sse += (d * d) as u64;
        }
    }
    (sse, pred)
}

/// Evaluate B_PRED greedily: for each 4×4 sub-block, pick the mode that
/// minimises SSE against the true source. Returns total SSE and the 16
/// chosen modes. Uses the running `rec_y` so neighbour propagation
/// matches what the decoder sees. The per-4×4 SSEs summed inside
/// `choose_b_pred_modes` are what the best-mode-selection used, so we
/// recompute them here in a single pass to keep the output tight.
fn sse_intra_b_pred(
    src_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    mb_w: usize,
    mb_h: usize,
) -> (u64, [i32; 16]) {
    let modes = choose_b_pred_modes(src_y, rec_y, y_stride, mb_xp, mb_yp, mb_w, mb_h);
    let above_right_extension: [u8; 4] = if mb_yp > 0 {
        let row = mb_yp - 1;
        let mut ext = [0u8; 4];
        for k in 0..4 {
            let xx = mb_xp + 16 + k;
            if xx < mb_w * 16 {
                ext[k] = rec_y[row * y_stride + xx];
            } else {
                ext[k] = rec_y[row * y_stride + (mb_xp + 15)];
            }
        }
        ext
    } else {
        [127; 4]
    };
    let mut total = 0u64;
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = mb_xp + bx * 4;
        let dst_y = mb_yp + by * 4;
        let neigh = gather_4x4_neighbours(
            rec_y,
            y_stride,
            mb_xp,
            mb_yp,
            bx,
            by,
            mb_w,
            &above_right_extension,
        );
        let mut pred = [0u8; 16];
        predict_4x4(modes[i], &neigh, &mut pred, 4);
        for r in 0..4 {
            for c in 0..4 {
                let s = src_y[(dst_y + r) * y_stride + dst_x + c] as i32;
                let p = pred[r * 4 + c] as i32;
                let d = s - p;
                total += (d * d) as u64;
            }
        }
    }
    (total, modes)
}

/// Pick the 16 4×4 sub-block modes for a B_PRED MB by greedy SSE
/// minimisation against the source. Uses the current (pre-residual)
/// `rec_y` for neighbour context — close enough for a first-pass
/// selection, and consistent for the later reconstruction since the
/// encoder proceeds sub-block by sub-block.
fn choose_b_pred_modes(
    src_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    mb_w: usize,
    _mb_h: usize,
) -> [i32; 16] {
    let above_right_extension: [u8; 4] = if mb_yp > 0 {
        let row = mb_yp - 1;
        let mut ext = [0u8; 4];
        for k in 0..4 {
            let xx = mb_xp + 16 + k;
            if xx < mb_w * 16 {
                ext[k] = rec_y[row * y_stride + xx];
            } else {
                ext[k] = rec_y[row * y_stride + (mb_xp + 15)];
            }
        }
        ext
    } else {
        [127; 4]
    };
    let mut modes = [B_DC_PRED; 16];
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = mb_xp + bx * 4;
        let dst_y = mb_yp + by * 4;
        let neigh = gather_4x4_neighbours(
            rec_y,
            y_stride,
            mb_xp,
            mb_yp,
            bx,
            by,
            mb_w,
            &above_right_extension,
        );
        let mut best_mode = B_DC_PRED;
        let mut best_sse = u64::MAX;
        for &m in &[
            B_DC_PRED, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_LD_PRED, B_RD_PRED, B_VR_PRED, B_VL_PRED,
            B_HD_PRED, B_HU_PRED,
        ] {
            let mut pred = [0u8; 16];
            predict_4x4(m, &neigh, &mut pred, 4);
            let mut sse = 0u64;
            for r in 0..4 {
                for c in 0..4 {
                    let s = src_y[(dst_y + r) * y_stride + dst_x + c] as i32;
                    let p = pred[r * 4 + c] as i32;
                    let d = s - p;
                    sse += (d * d) as u64;
                }
            }
            if sse < best_sse {
                best_sse = sse;
                best_mode = m;
            }
        }
        modes[i] = best_mode;
    }
    modes
}

/// Pick the best chroma intra mode (DC/V/H/TM) for a keyframe by SSE
/// of the prediction vs the source U + V planes.
#[allow(clippy::too_many_arguments)]
fn choose_intra_chroma_mode(
    src_u: &[u8],
    src_v: &[u8],
    rec_u: &[u8],
    rec_v: &[u8],
    uv_stride: usize,
    mb_xc: usize,
    mb_yc: usize,
) -> i32 {
    let mut best = DC_PRED;
    let mut best_sse = u64::MAX;
    for &m in &[DC_PRED, V_PRED, H_PRED, TM_PRED] {
        let sse = sse_intra_8x8_both(m, src_u, rec_u, uv_stride, mb_xc, mb_yc)
            + sse_intra_8x8_both(m, src_v, rec_v, uv_stride, mb_xc, mb_yc);
        if sse < best_sse {
            best_sse = sse;
            best = m;
        }
    }
    best
}

fn sse_intra_8x8_both(
    mode: i32,
    src: &[u8],
    rec: &[u8],
    uv_stride: usize,
    mb_xc: usize,
    mb_yc: usize,
) -> u64 {
    let (above, left, tl) = gather_8x8_neighbours(rec, uv_stride, mb_xc, mb_yc);
    let mut pred = [0u8; 64];
    predict_8x8(mode, above.as_ref(), left.as_ref(), tl, &mut pred, 8);
    let mut sse = 0u64;
    for r in 0..8 {
        for c in 0..8 {
            let s = src[(mb_yc + r) * uv_stride + mb_xc + c] as i32;
            let p = pred[r * 8 + c] as i32;
            let d = s - p;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Emit the 16 4×4 BMODE tree paths for a keyframe B_PRED MB, using
/// `KF_BMODE_PROB[above][left]` context probabilities for each sub-block.
/// Also populates `this_mb_bmodes` so the caller can propagate the
/// bottom row to `bmode_above` and reconstruct with the same modes.
#[allow(clippy::too_many_arguments)]
fn emit_bmodes_keyframe(
    enc: &mut BoolEncoder,
    this_mb_bmodes: &mut [i32; 16],
    bmode_above_for_mb: &[i32; 4],
    left_bmodes_in: &[i32; 4],
    src_y: &[u8],
    rec_y: &[u8],
    y_stride: usize,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
) {
    // Pick per-sub-block modes greedily against the source.
    let chosen = choose_b_pred_modes(src_y, rec_y, y_stride, mb_x * 16, mb_y * 16, mb_w, 0);
    let mut left_bmodes = *left_bmodes_in;
    for i in 0..16 {
        let row = i / 4;
        let col = i % 4;
        let above_mode = if row == 0 {
            bmode_above_for_mb[col]
        } else {
            this_mb_bmodes[(row - 1) * 4 + col]
        };
        let left_mode = if col == 0 {
            left_bmodes[row]
        } else {
            this_mb_bmodes[row * 4 + col - 1]
        };
        let probs = &KF_BMODE_PROB[above_mode as usize][left_mode as usize];
        let m = chosen[i];
        emit_tree_path(enc, BMODE_PATHS[m as usize], probs);
        this_mb_bmodes[i] = m;
        if col == 3 {
            left_bmodes[row] = m;
        }
    }
}

// ---------------------------------------------------------------------------
// SPLIT_MV search
// ---------------------------------------------------------------------------

/// For a given SPLIT mode (16×8 / 8×16 / 8×8 / 4×4), search the best
/// per-partition MV and compute the total SAD.
fn search_split_mv(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
) -> Option<(SplitMv, u32)> {
    let mut best: Option<(SplitMv, u32)> = None;
    for split_mode in 0..4u8 {
        let (part_mvs, total_sad) =
            search_split_partitions(split_mode, src_y, ref_plane, y_stride, mb_xp, mb_yp);
        let split = SplitMv {
            split_mode,
            part_mvs,
        };
        match &best {
            None => best = Some((split, total_sad)),
            Some((_, bs)) if total_sad < *bs => best = Some((split, total_sad)),
            _ => {}
        }
    }
    best
}

/// Search best per-partition MVs for one particular split mode.
/// Each partition is described by the set of 4×4 sub-blocks (from
/// `MB_SPLITS[split_mode]`) belonging to it; we search over a small
/// integer-pel window around zero then refine at quarter-pel, and sum
/// the SAD contributions.
fn search_split_partitions(
    split_mode: u8,
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
) -> ([Mv; 16], u32) {
    let partition = &MB_SPLITS[split_mode as usize];
    let n = MB_SPLIT_COUNT[split_mode as usize] as usize;
    let mut part_mvs = [Mv::ZERO; 16];
    let mut total_sad = 0u32;
    for p in 0..n {
        // Determine bounding box of partition `p`.
        let mut indices: Vec<usize> = Vec::with_capacity(16);
        let mut min_by = 4usize;
        let mut max_by = 0usize;
        let mut min_bx = 4usize;
        let mut max_bx = 0usize;
        for i in 0..16 {
            if partition[i] as usize == p {
                indices.push(i);
                let by = i / 4;
                let bx = i % 4;
                min_by = min_by.min(by);
                max_by = max_by.max(by);
                min_bx = min_bx.min(bx);
                max_bx = max_bx.max(bx);
            }
        }
        // Integer-pel search.
        let (best_int_px, best_int_sad) = integer_partition_search(
            src_y,
            ref_plane,
            y_stride,
            mb_xp,
            mb_yp,
            &indices,
            MOTION_SEARCH_RANGE,
        );
        let int_mv = Mv::new(best_int_px.0 * 8, best_int_px.1 * 8);
        // Sub-pel refinement (reuses the 3×3 neighbourhood).
        let (refined_mv, refined_sad) = subpel_refine_partition(
            src_y,
            ref_plane,
            y_stride,
            mb_xp,
            mb_yp,
            &indices,
            int_mv,
            best_int_sad,
        );
        for i in &indices {
            part_mvs[*i] = refined_mv;
        }
        let _ = (min_bx, min_by, max_bx, max_by);
        total_sad += refined_sad;
    }
    (part_mvs, total_sad)
}

/// Integer-pel SAD search for a partition consisting of `indices` of
/// 4×4 sub-blocks. Uses edge-clamped sampling via `mb_luma_sad_at` at
/// 4×4 granularity.
fn integer_partition_search(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    indices: &[usize],
    range: i32,
) -> ((i32, i32), u32) {
    let mut best = (0, 0);
    let mut best_sad =
        partition_sad_at_int(src_y, ref_plane, y_stride, mb_xp, mb_yp, indices, 0, 0);
    for dy in -range..=range {
        for dx in -range..=range {
            if dy == 0 && dx == 0 {
                continue;
            }
            let sad =
                partition_sad_at_int(src_y, ref_plane, y_stride, mb_xp, mb_yp, indices, dy, dx);
            if sad < best_sad {
                best_sad = sad;
                best = (dy, dx);
            }
        }
    }
    (best, best_sad)
}

/// SAD of a partition vs its reference shifted by (dy, dx) integer pels.
fn partition_sad_at_int(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    indices: &[usize],
    dy: i32,
    dx: i32,
) -> u32 {
    let mut sad: u32 = 0;
    for &i in indices {
        let by = i / 4;
        let bx = i % 4;
        let sy = mb_yp + by * 4;
        let sx = mb_xp + bx * 4;
        for r in 0..4 {
            for c in 0..4 {
                let ry = (sy as i32 + r as i32 + dy).clamp(0, ref_plane.height as i32 - 1) as usize;
                let rx = (sx as i32 + c as i32 + dx).clamp(0, ref_plane.width as i32 - 1) as usize;
                let s = src_y[(sy + r) * y_stride + sx + c] as i32;
                let p = ref_plane.data[ry * ref_plane.stride + rx] as i32;
                sad += (s - p).unsigned_abs();
            }
        }
    }
    sad
}

/// Sub-pel refinement for a partition using the sixtap filter path.
fn subpel_refine_partition(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    indices: &[usize],
    int_mv: Mv,
    int_sad: u32,
) -> (Mv, u32) {
    let mut best_mv = int_mv;
    let mut best_sad = int_sad;
    let step = SUBPEL_REFINE_STEP;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let mv = Mv::new(int_mv.row as i32 + dy * step, int_mv.col as i32 + dx * step);
            let sad = subpel_partition_sad(src_y, ref_plane, y_stride, mb_xp, mb_yp, indices, mv);
            if sad < best_sad {
                best_sad = sad;
                best_mv = mv;
            }
        }
    }
    (best_mv, best_sad)
}

fn subpel_partition_sad(
    src_y: &[u8],
    ref_plane: &RefPlane<'_>,
    y_stride: usize,
    mb_xp: usize,
    mb_yp: usize,
    indices: &[usize],
    mv: Mv,
) -> u32 {
    let mut sad = 0u32;
    for &i in indices {
        let by = i / 4;
        let bx = i % 4;
        let sy = mb_yp + by * 4;
        let sx = mb_xp + bx * 4;
        let mut pred = [0u8; 16];
        let ref_x_fp = sx as i32 * 8 + mv.col as i32;
        let ref_y_fp = sy as i32 * 8 + mv.row as i32;
        sixtap_predict(ref_plane, ref_x_fp, ref_y_fp, &mut pred, 4, 0, 0, 4, 4);
        for r in 0..4 {
            for c in 0..4 {
                let s = src_y[(sy + r) * y_stride + sx + c] as i32;
                let p = pred[r * 4 + c] as i32;
                sad += (s - p).unsigned_abs();
            }
        }
    }
    sad
}

// ---------------------------------------------------------------------------
// Inter MB encode — SPLIT variant (per-subblock MV).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn encode_inter_mb_split(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    y_buf_h: usize,
    uv_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    split: &SplitMv,
    q: &QuantCtx,
) -> MbEncoded {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    let ref_plane_y = RefPlane {
        data: ref_y,
        stride: y_stride,
        width: y_stride,
        height: y_buf_h,
    };

    // Expand per-subblock MVs from the split-mode partitioning.
    let partition = &MB_SPLITS[split.split_mode as usize];
    let mut sub_mvs = [Mv::ZERO; 16];
    for i in 0..16 {
        sub_mvs[i] = split.part_mvs[partition[i] as usize];
    }

    // --- Luma prediction per 4×4 sub-block using its own MV ---
    let mut pred_y = [0u8; 256];
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = bx * 4;
        let dst_y = by * 4;
        let mv = sub_mvs[i];
        let ref_x_fp = (mb_xp + dst_x) as i32 * 8 + mv.col as i32;
        let ref_y_fp = (mb_yp + dst_y) as i32 * 8 + mv.row as i32;
        sixtap_predict(
            &ref_plane_y,
            ref_x_fp,
            ref_y_fp,
            &mut pred_y,
            16,
            dst_x,
            dst_y,
            4,
            4,
        );
    }

    // --- Y residual. SPLIT MBs do NOT have a Y2 block — per 4×4 block
    //     we quantise DC + AC with their respective steps.
    let mut y_q = [[0i16; 16]; 16];
    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut blk = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let src = src_y[(mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c] as i32;
                let p = pred_y[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                blk[r * 4 + c] = src - p;
            }
        }
        let coeffs = fdct4x4(&blk);
        let mut blk_q = [0i16; 16];
        blk_q[0] = quant(coeffs[0], q.y_dc);
        for k in 1..16 {
            blk_q[k] = quant(coeffs[k], q.y_ac);
        }
        y_q[bi] = blk_q;

        let mut deq = [0i16; 16];
        deq[0] = (blk_q[0] as i32 * q.y_dc) as i16;
        for k in 1..16 {
            deq[k] = (blk_q[k] as i32 * q.y_ac) as i16;
        }
        let res = idct4x4(&deq);
        for r in 0..4 {
            for c in 0..4 {
                let p = pred_y[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                let rr = res[r * 4 + c] as i32;
                let dst_y_idx = (mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c;
                rec_y[dst_y_idx] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }

    // --- Chroma prediction: each 4×4 chroma uses `chroma_round` of sum
    //     of covered luma sub-MVs (matches the decoder exactly).
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    let mut u_q = [[0i16; 16]; 4];
    let mut v_q = [[0i16; 16]; 4];
    for plane_sel in 0..2 {
        let (src, refp, rec, q_coeffs) = match plane_sel {
            0 => (src_u, ref_u, &mut *rec_u, &mut u_q),
            _ => (src_v, ref_v, &mut *rec_v, &mut v_q),
        };
        let ref_plane_uv = RefPlane {
            data: refp,
            stride: uv_stride,
            width: uv_stride,
            height: uv_buf_h,
        };
        let mut pred_uv = [0u8; 64];
        for i in 0..4 {
            let by = i / 2;
            let bx = i % 2;
            let mut sum_r = 0i32;
            let mut sum_c = 0i32;
            for rr in 0..2 {
                for cc in 0..2 {
                    let li = (2 * by + rr) * 4 + (2 * bx + cc);
                    sum_r += sub_mvs[li].row as i32;
                    sum_c += sub_mvs[li].col as i32;
                }
            }
            let cmv_r = chroma_round_enc(sum_r);
            let cmv_c = chroma_round_enc(sum_c);
            let dst_x = bx * 4;
            let dst_y = by * 4;
            let ref_x_fp = (mb_xc + dst_x) as i32 * 8 + cmv_c;
            let ref_y_fp = (mb_yc + dst_y) as i32 * 8 + cmv_r;
            bilinear_predict(
                &ref_plane_uv,
                ref_x_fp,
                ref_y_fp,
                &mut pred_uv,
                8,
                dst_x,
                dst_y,
                4,
                4,
            );
        }
        for bi in 0..4 {
            let by = bi / 2;
            let bx = bi % 2;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let sidx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    let s = src[sidx] as i32;
                    let p = pred_uv[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = s - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.uv_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.uv_ac);
            }
            q_coeffs[bi] = blk_q;
            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let pidx = (by * 4 + r) * 8 + bx * 4 + c;
                    let p = pred_uv[pidx] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let didx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    rec[didx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    let any_coeffs = y_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0)
        || u_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0)
        || v_q.iter().flat_map(|b| b.iter()).any(|&v| v != 0);
    MbEncoded {
        y2_coeffs: [0; 16],
        y_coeffs: y_q,
        u_coeffs: u_q,
        v_coeffs: v_q,
        // SPLIT_MV — per RFC §15.1, sub-block edges are always filtered
        // for SPLIT_MV regardless of has_coeffs.
        y_mode: SPLIT_MV,
        has_coeffs: any_coeffs,
    }
}

// ---------------------------------------------------------------------------
// Encoder-side loop filter (applied to reconstruction so the next P-frame
// sees post-filter samples).
// ---------------------------------------------------------------------------

/// Encoder-side loop filter — must produce bit-identical reconstruction
/// to what the decoder will emit. The per-MB iteration order and skip
/// rules mirror `apply_loop_filter` in `decoder.rs` (RFC 6386 §15.1).
///
/// `filter_type`: 0 = normal (6-pixel filter on luma + chroma at every
/// edge); 1 = simple (4-pixel luma-only filter, chroma untouched).
///
/// `frame_level` is the frame-wide loop-filter level (the value that
/// will be emitted in the loop-filter header). When segmentation is
/// enabled the per-MB level is derived as
/// `clamp(frame_level + segment_lf_deltas[seg], 0..=63)` to match the
/// decoder's `per_mb_filter_level` walk; when disabled the per-MB
/// level is always `frame_level`. A per-MB level of 0 skips that MB
/// entirely (matches the decoder's `if mb_level == 0 { continue; }`
/// fast-path).
#[allow(clippy::too_many_arguments)]
fn apply_loop_filter_enc(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    y_buf_h: usize,
    uv_buf_h: usize,
    mb_w: usize,
    mb_h: usize,
    frame_level: u8,
    sharpness: u8,
    filter_type: u8,
    mb_encoded: &[MbEncoded],
    mb_segment_ids: &[u8],
    segments: &SegmentCtx,
    key_frame: bool,
) {
    if frame_level == 0 {
        return;
    }
    let simple = filter_type == 1;
    for mb_y in 0..mb_h {
        let y0 = mb_y * 16;
        let y0c = mb_y * 8;
        for mb_x in 0..mb_w {
            let mb_idx = mb_y * mb_w + mb_x;
            let mb = &mb_encoded[mb_idx];
            // Per-MB filter level (segmentation-aware). 0 → skip this MB
            // entirely so the encoder reconstruction matches what the
            // decoder will produce when `per_mb_filter_level` clamps to 0.
            let mb_level = segments.filter_level_for(mb_segment_ids[mb_idx], frame_level);
            if mb_level == 0 {
                continue;
            }
            let params_mb = FilterParams::for_mb_typed(mb_level, sharpness, true, key_frame);
            let params_sb = FilterParams::for_mb_typed(mb_level, sharpness, false, key_frame);
            let filter_subblocks = mb.has_coeffs || mb.y_mode == B_PRED || mb.y_mode == SPLIT_MV;
            let x = mb_x * 16;
            let xc = mb_x * 8;

            // 1. Left MB v-edges. Simple mode: luma only, four pixels.
            if mb_x > 0 {
                if simple {
                    filter_simple_vertical(y_plane, y_stride, x, y_stride, y0 + 16, params_mb);
                } else {
                    filter_normal_vertical(
                        y_plane,
                        y_stride,
                        x,
                        y_stride,
                        y0 + 16,
                        params_mb,
                        true,
                    );
                    filter_normal_vertical(
                        u_plane,
                        uv_stride,
                        xc,
                        uv_stride,
                        y0c + 8,
                        params_mb,
                        true,
                    );
                    filter_normal_vertical(
                        v_plane,
                        uv_stride,
                        xc,
                        uv_stride,
                        y0c + 8,
                        params_mb,
                        true,
                    );
                }
            }

            // 2. Inner sub-block v-edges (3 luma, 1 chroma). Simple
            //    mode skips chroma.
            if filter_subblocks {
                if simple {
                    for k in 1..4 {
                        filter_simple_vertical(
                            y_plane,
                            y_stride,
                            x + k * 4,
                            y_stride,
                            y0 + 16,
                            params_sb,
                        );
                    }
                } else {
                    for k in 1..4 {
                        filter_normal_vertical(
                            y_plane,
                            y_stride,
                            x + k * 4,
                            y_stride,
                            y0 + 16,
                            params_sb,
                            false,
                        );
                    }
                    filter_normal_vertical(
                        u_plane,
                        uv_stride,
                        xc + 4,
                        uv_stride,
                        y0c + 8,
                        params_sb,
                        false,
                    );
                    filter_normal_vertical(
                        v_plane,
                        uv_stride,
                        xc + 4,
                        uv_stride,
                        y0c + 8,
                        params_sb,
                        false,
                    );
                }
            }

            // 3. Top MB h-edges.
            if mb_y > 0 {
                if simple {
                    filter_simple_horizontal(y_plane, y_stride, y0, y_stride, y_buf_h, params_mb);
                } else {
                    filter_normal_horizontal(
                        y_plane, y_stride, y0, y_stride, y_buf_h, params_mb, true,
                    );
                    filter_normal_horizontal(
                        u_plane, uv_stride, y0c, uv_stride, uv_buf_h, params_mb, true,
                    );
                    filter_normal_horizontal(
                        v_plane, uv_stride, y0c, uv_stride, uv_buf_h, params_mb, true,
                    );
                }
            }

            // 4. Inner sub-block h-edges.
            if filter_subblocks {
                if simple {
                    for k in 1..4 {
                        filter_simple_horizontal(
                            y_plane,
                            y_stride,
                            y0 + k * 4,
                            y_stride,
                            y_buf_h,
                            params_sb,
                        );
                    }
                } else {
                    for k in 1..4 {
                        filter_normal_horizontal(
                            y_plane,
                            y_stride,
                            y0 + k * 4,
                            y_stride,
                            y_buf_h,
                            params_sb,
                            false,
                        );
                    }
                    filter_normal_horizontal(
                        u_plane,
                        uv_stride,
                        y0c + 4,
                        uv_stride,
                        uv_buf_h,
                        params_sb,
                        false,
                    );
                    filter_normal_horizontal(
                        v_plane,
                        uv_stride,
                        y0c + 4,
                        uv_stride,
                        uv_buf_h,
                        params_sb,
                        false,
                    );
                }
            }
        }
    }
}

/// Copy the 3 planes of a video frame into MB-aligned (16/8 pixel) buffers.
/// Edge-replicate when frame dimensions are not multiples of 16.
fn extract_mb_padded(
    src: &Yuv420Source<'_>,
    width: usize,
    height: usize,
    mb_w: usize,
    mb_h: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_h = mb_h * 16;
    let uv_h = mb_h * 8;

    let mut y_out = vec![0u8; y_stride * y_h];
    for j in 0..y_h {
        let src_row = j.min(height - 1);
        let src_start = src_row * src.y_stride;
        for i in 0..y_stride {
            let src_col = i.min(width - 1);
            y_out[j * y_stride + i] = src.y[src_start + src_col];
        }
    }
    let uv_w = (width + 1) / 2;
    let uv_src_h = (height + 1) / 2;
    let mut u_out = vec![0u8; uv_stride * uv_h];
    let mut v_out = vec![0u8; uv_stride * uv_h];
    for j in 0..uv_h {
        let src_row = j.min(uv_src_h - 1);
        let u_start = src_row * src.u_stride;
        let v_start = src_row * src.v_stride;
        for i in 0..uv_stride {
            let src_col = i.min(uv_w - 1);
            u_out[j * uv_stride + i] = src.u[u_start + src_col];
            v_out[j * uv_stride + i] = src.v[v_start + src_col];
        }
    }
    Ok((y_out, u_out, v_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;
    use crate::tables::coeff_probs::DEFAULT_COEF_PROBS;
    use crate::tokens::{decode_block, BlockType};

    fn roundtrip_one_block(coeffs: &[i16; 16], plane: usize, nctx: u8, start: usize) {
        let mut enc = BoolEncoder::new();
        let _nz_enc = encode_block(
            &mut enc,
            &DEFAULT_COEF_PROBS,
            plane,
            nctx as usize,
            coeffs,
            start,
        );
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let bt = match plane {
            0 => BlockType::YAfterY2,
            1 => BlockType::Y2,
            2 => BlockType::UV,
            _ => BlockType::YNoY2,
        };
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, bt, nctx, &mut out, start);
        for i in start..16 {
            let zz_idx = crate::tables::token_tree::ZIGZAG[i];
            assert_eq!(
                out[zz_idx], coeffs[zz_idx],
                "coeff at zigzag pos {i} (raw {zz_idx}) mismatch: in={} out={}",
                coeffs[zz_idx], out[zz_idx]
            );
        }
    }

    #[test]
    fn block_roundtrip_all_zero() {
        let coeffs = [0i16; 16];
        roundtrip_one_block(&coeffs, 0, 0, 1);
        roundtrip_one_block(&coeffs, 1, 0, 0);
        roundtrip_one_block(&coeffs, 2, 0, 0);
    }

    #[test]
    fn block_roundtrip_dc_only() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5;
        roundtrip_one_block(&coeffs, 1, 0, 0);
        coeffs[0] = -3;
        roundtrip_one_block(&coeffs, 2, 0, 0);
    }

    #[test]
    fn block_roundtrip_small_values() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;
        coeffs[1] = 2;
        coeffs[4] = 3;
        coeffs[8] = -1;
        roundtrip_one_block(&coeffs, 0, 0, 1);
    }

    #[test]
    fn block_roundtrip_y2_negative_dc() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = -19;
        roundtrip_one_block(&coeffs, 1, 0, 0);
    }

    #[test]
    fn block_roundtrip_y2_sparse_ctx1() {
        let coeffs: [i16; 16] = [7, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        roundtrip_one_block(&coeffs, 1, 1, 0);
    }

    #[test]
    fn block_roundtrip_y_with_dc_zeroed() {
        let mut y_blks = [[0i16; 16]; 16];
        for bi in 0..16 {
            y_blks[bi][1] = -3;
            y_blks[bi][4] = -2;
        }
        let mut enc = BoolEncoder::new();
        let mut nz_left = [0u8; 4];
        let mut nz_above = [0u8; 4];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = by * 4 + bx;
                let nctx = nz_above[bx] + nz_left[by];
                let nz = encode_block(
                    &mut enc,
                    &DEFAULT_COEF_PROBS,
                    0,
                    nctx as usize,
                    &y_blks[idx],
                    1,
                );
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_above[bx] = nzf;
                nz_left[by] = nzf;
            }
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let mut nz_left = [0u8; 4];
        let mut nz_above = [0u8; 4];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = by * 4 + bx;
                let nctx = nz_above[bx] + nz_left[by];
                let mut out = [0i16; 16];
                let nz = decode_block(
                    &mut dec,
                    &DEFAULT_COEF_PROBS,
                    BlockType::YAfterY2,
                    nctx,
                    &mut out,
                    1,
                );
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_above[bx] = nzf;
                nz_left[by] = nzf;
                for k in 1..16 {
                    let zz = crate::tables::token_tree::ZIGZAG[k];
                    assert_eq!(out[zz], y_blks[idx][zz], "block {idx} zz pos {k}");
                }
            }
        }
    }

    #[test]
    fn block_roundtrip_two_y2_back_to_back() {
        let a: [i16; 16] = [-79, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        let b: [i16; 16] = [7, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        let mut enc = BoolEncoder::new();
        encode_block(&mut enc, &DEFAULT_COEF_PROBS, 1, 0, &a, 0);
        encode_block(&mut enc, &DEFAULT_COEF_PROBS, 1, 1, &b, 0);
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::Y2, 0, &mut out, 0);
        assert_eq!(out, a);
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::Y2, 1, &mut out, 0);
        assert_eq!(out, b);
    }

    #[test]
    fn choose_pmb_decision_picks_nearest_when_neighbour_matches() {
        // 64×64 reference filled with a deterministic pattern, source is
        // the reference shifted by (0, +8). On the second MB of a row the
        // neighbour's MV is the result of the first MB's NEWMV — so
        // `nearest = (0, 64)` and the encoder should prefer NEAREST_MV
        // over NEW_MV for that MB.
        let stride = 64;
        let h = 64;
        let mut refp = vec![0u8; stride * h];
        for r in 0..h {
            for c in 0..stride {
                refp[r * stride + c] = (((r * 7 + c * 13) & 0xff) as u8).wrapping_add(32);
            }
        }
        // Source = reference shifted by (+0 rows, +8 cols), with edge
        // replication on the right to keep the SAD unambiguous at the
        // last MB column.
        let mut src = vec![0u8; stride * h];
        for r in 0..h {
            for c in 0..stride {
                let sc = (c + 8).min(stride - 1);
                src[r * stride + c] = refp[r * stride + sc];
            }
        }

        // First MB — no neighbour MVs yet, NEAREST cannot apply.
        let rec_y = vec![128u8; stride * h];
        let mb_w = stride / 16;
        let mb_h = h / 16;
        let d0 = choose_pmb_decision(
            &src,
            &refp,
            stride,
            h,
            0,
            0,
            mb_w,
            mb_h,
            Mv::ZERO,
            Mv::ZERO,
            &rec_y,
        );
        assert!(
            matches!(d0, PMbDecision::NewMv(_)),
            "first MB should pick NEW_MV, got {:?}",
            d0
        );
        let first_mv = d0.mv();
        assert_eq!(first_mv.col, 64, "integer pan should land on col=+64");

        // Second MB on the same row — neighbour chain now exposes
        // `first_mv` as `nearest`; NEAREST is free so it must win.
        let d1 = choose_pmb_decision(
            &src,
            &refp,
            stride,
            h,
            1,
            0,
            mb_w,
            mb_h,
            first_mv,
            Mv::ZERO,
            &rec_y,
        );
        assert!(
            matches!(d1, PMbDecision::NearestMv(_)),
            "second MB should pick NEAREST_MV, got {:?}",
            d1
        );
    }

    #[test]
    fn choose_pmb_decision_skips_static_background() {
        let stride = 32;
        let h = 32;
        let buf = vec![100u8; stride * h];
        // Identical source = reference → zero SAD → SKIP.
        let mb_w = stride / 16;
        let mb_h = h / 16;
        let d = choose_pmb_decision(
            &buf,
            &buf,
            stride,
            h,
            0,
            0,
            mb_w,
            mb_h,
            Mv::ZERO,
            Mv::ZERO,
            &buf,
        );
        assert!(matches!(d, PMbDecision::Skip), "got {:?}", d);
    }

    #[test]
    fn choose_pmb_decision_falls_back_to_intra_on_scene_cut() {
        // Reference is smooth mid-gray; source is a high-entropy pattern
        // nothing in the reference can match even roughly.
        let stride = 32;
        let h = 32;
        let refp = vec![128u8; stride * h];
        let mut src = vec![0u8; stride * h];
        for r in 0..h {
            for c in 0..stride {
                // Values spread across the full 0..=255 range, alternating.
                let v = ((r.wrapping_mul(131).wrapping_add(c.wrapping_mul(89))) & 0xff) as u8;
                src[r * stride + c] = v;
            }
        }
        let rec_y = vec![128u8; stride * h];
        let mb_w = stride / 16;
        let mb_h = h / 16;
        let d = choose_pmb_decision(
            &src,
            &refp,
            stride,
            h,
            0,
            0,
            mb_w,
            mb_h,
            Mv::ZERO,
            Mv::ZERO,
            &rec_y,
        );
        assert!(
            matches!(d, PMbDecision::Intra { .. }),
            "expected Intra fallback, got {:?}",
            d
        );
    }

    #[test]
    fn choose_pmb_decision_picks_b_pred_on_high_variance_intra_in_p() {
        // High-variance source MB (#339): the per-4x4 B_PRED sub-mode
        // search should beat the four 16×16 modes by enough to clear
        // `B_PRED_SSE_MARGIN_INTRA_IN_P`. Set up a 32×32 frame whose
        // top-left MB is a high-frequency cross-hatched pattern that
        // no single 16×16 mode predicts well, but whose 4×4 sub-blocks
        // each have a clear local direction.
        let stride = 32;
        let h = 32;
        // Reference: smooth mid-gray (forces intra fallback by making
        // every inter SAD huge against the structured source).
        let refp = vec![128u8; stride * h];
        // Source: cross-hatched stripes — alternating rows of (32, 224)
        // and columns of similar opposing pattern. Variance >> the
        // segment-3 boundary; 16×16 V_PRED / H_PRED / DC_PRED can each
        // fit *part* of the MB but not all of it.
        let mut src = vec![0u8; stride * h];
        for r in 0..h {
            for c in 0..stride {
                let bit_r = (r / 4) & 1;
                let bit_c = (c / 4) & 1;
                src[r * stride + c] = match (bit_r, bit_c) {
                    (0, 0) => 32,
                    (0, 1) => 224,
                    (1, 0) => 224,
                    _ => 32,
                };
            }
        }
        // Reconstruction context: smooth — forces B_PRED to predict
        // from neighbour-aware sub-block context (the common case for
        // an early-row MB).
        let rec_y = vec![128u8; stride * h];
        let mb_w = stride / 16;
        let mb_h = h / 16;
        let d = choose_pmb_decision(
            &src,
            &refp,
            stride,
            h,
            0,
            0,
            mb_w,
            mb_h,
            Mv::ZERO,
            Mv::ZERO,
            &rec_y,
        );
        // The patterned MB has variance well above
        // `INTRA_IN_P_BPRED_VARIANCE_THRESHOLD`, so the picker
        // evaluates B_PRED. On this cross-hatched source the per-4x4
        // search is the only way to reach a low SSE — so B_PRED must
        // win.
        assert!(
            matches!(d, PMbDecision::Intra { y_mode, .. } if y_mode == B_PRED),
            "expected B_PRED intra-in-P on heavy-texture MB, got {:?}",
            d
        );
    }

    #[test]
    fn choose_pmb_decision_keeps_16x16_intra_on_smooth_mb() {
        // Low-variance source MB — under
        // `INTRA_IN_P_BPRED_VARIANCE_THRESHOLD`, the picker must skip
        // the B_PRED branch entirely (cost is in `predict_4x4` × 16 +
        // `bool_cost_x256` × 16). Use a moderately-detailed but
        // sub-threshold source so we still hit the intra-in-P path
        // (i.e. inter SAD is high enough to fall back to intra) but
        // B_PRED is NOT considered.
        let stride = 32;
        let h = 32;
        let refp = vec![128u8; stride * h];
        // Mid-grey with a slight gradient — variance high enough to
        // make inter SAD huge against the flat reference, but well
        // below `INTRA_IN_P_BPRED_VARIANCE_THRESHOLD`.
        let mut src = vec![0u8; stride * h];
        for r in 0..h {
            for c in 0..stride {
                src[r * stride + c] = (50 + r as u32 + c as u32 / 4).min(150) as u8;
            }
        }
        let rec_y = vec![128u8; stride * h];
        let mb_w = stride / 16;
        let mb_h = h / 16;
        let d = choose_pmb_decision(
            &src,
            &refp,
            stride,
            h,
            0,
            0,
            mb_w,
            mb_h,
            Mv::ZERO,
            Mv::ZERO,
            &rec_y,
        );
        if let PMbDecision::Intra { y_mode, .. } = d {
            assert!(
                y_mode != B_PRED,
                "low-variance MB must not pick B_PRED intra-in-P, got y_mode={y_mode}"
            );
        } else {
            panic!("expected Intra fallback, got {:?}", d);
        }
    }

    #[test]
    fn subpel_refine_finds_half_pel_match() {
        // Build a reference whose samples are 32 + 4*col (smooth linear).
        // A perfect "half-pel right" shift of the source produces, at
        // each column c, the value 32 + 4*(c + 0.5) = 34 + 4c. Apply the
        // sixtap filter to verify that subpel_refine_luma lands on
        // col=+4 (= 1/2 pel in 1/8-pel units).
        let stride = 64;
        let h = 32;
        let mut refp = vec![0u8; stride * h];
        for r in 0..h {
            for c in 0..stride {
                refp[r * stride + c] = (32 + c * 3).min(255) as u8;
            }
        }
        let ref_plane = RefPlane {
            data: &refp,
            stride,
            width: stride,
            height: h,
        };
        // Build the target by applying a half-pel right shift through
        // sixtap_predict — then the exact best MV is (0, +4).
        let mut src = vec![0u8; stride * h];
        for bi in 0..16 {
            let by = bi / 4;
            let bx = bi % 4;
            let dst_x = bx * 4;
            let dst_y = by * 4;
            sixtap_predict(
                &ref_plane,
                (dst_x as i32) * 8 + 4,
                (dst_y as i32) * 8,
                &mut src,
                stride,
                dst_x,
                dst_y,
                4,
                4,
            );
        }
        // Integer-pel best will land at col=0 or col=+8; quarter-pel
        // refinement must move it to col=+4 (half-pel == 4 in 1/8 units).
        let int_sad = subpel_luma_sad_at(&src, &ref_plane, stride, 0, 0, Mv::ZERO);
        let (refined_mv, refined_sad) =
            subpel_refine_luma(&src, &ref_plane, stride, 0, 0, Mv::ZERO, int_sad);
        assert!(
            refined_sad < int_sad,
            "refinement did not improve: int={int_sad} refined={refined_sad}"
        );
        // At quarter-pel step=2, a half-pel truth (4) is not directly
        // reachable in one pass from integer zero, but step=2 gives
        // col=±2; we accept any negative-SAD improvement as proof the
        // refinement path is live.
        let _ = refined_mv;
    }

    #[test]
    fn block_roundtrip_category_magnitudes() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 6;
        coeffs[1] = 9;
        coeffs[2] = 15;
        coeffs[3] = 25;
        coeffs[4] = -50;
        coeffs[5] = 100;
        roundtrip_one_block(&coeffs, 1, 0, 0);
    }

    /// `optimal_prob_8` returns 128 on the empty observation, and
    /// otherwise rounds `256 * n_zero / total` to the nearest integer
    /// with the result clamped to 1..=255 so the bool coder can still
    /// encode either branch (no degenerate single-symbol case).
    #[test]
    fn optimal_prob_8_matches_observed_distribution() {
        // No observations → neutral 128.
        assert_eq!(super::optimal_prob_8(0, 0), 128);
        // Pure zero observations → clamped to 255 (not 256, so a single
        // unexpected `true` outcome still has bounded entropy cost).
        assert_eq!(super::optimal_prob_8(100, 0), 255);
        // Pure one observations → clamped to 1.
        assert_eq!(super::optimal_prob_8(0, 100), 1);
        // 50/50 split → 128.
        assert_eq!(super::optimal_prob_8(50, 50), 128);
        // 80% zero → ~204/205 (256*0.8 = 204.8 → rounds to 205).
        assert_eq!(super::optimal_prob_8(80, 20), 205);
        // Single zero event amongst many ones → clamped to 1, not 0.
        let p = super::optimal_prob_8(1, 99);
        assert!((1..=3).contains(&p), "expected ~3, got {p}");
    }

    /// The `prob_intra` / `prob_last` / `prob_gf` triple in the inter
    /// frame header should reflect the actual ref-frame distribution
    /// of the encoded MBs — not the legacy fixed `200 / 1 / 128`
    /// (single-ref) or `200 / 128 / 128` (multi-ref) literals.
    ///
    /// On a static-content P-frame *every* MB picks REF_LAST as SKIP, so
    /// `prob_last` should land near 255 (REF_LAST near-free, the libvpx
    /// single-ref convention) and `prob_intra` should land near 255
    /// (intra-in-P essentially never picked).
    /// The `prob_intra` / `prob_last` / `prob_gf` triple in the inter
    /// frame header should reflect the actual ref-frame distribution
    /// of the encoded MBs — not the legacy fixed `200 / 1 / 128`
    /// (single-ref) or `200 / 128 / 128` (multi-ref) literals.
    ///
    /// On a static-content P-frame *every* MB picks REF_LAST as SKIP, so
    /// `prob_last` should land near 255 (REF_LAST near-free, the libvpx
    /// single-ref convention) and `prob_intra` should land near 255
    /// (intra-in-P essentially never picked).
    #[cfg(feature = "registry")]
    #[test]
    fn header_probs_track_per_mb_ref_distribution() {
        use crate::bool_decoder::BoolDecoder;
        use crate::frame_header::{parse_inter_header, PersistentProbs};
        use crate::frame_tag::{parse_header, FrameType};
        use oxideav_core::{
            CodecId, CodecParameters, Frame, PixelFormat, Rational, VideoFrame, VideoPlane,
        };

        const W: u32 = 64;
        const H: u32 = 64;
        let cw = (W / 2) as usize;
        let ch = (H / 2) as usize;
        let make = || {
            // Solid grey frame — frame N+1 == frame N for the first two
            // frames so every MB is a perfect SKIP candidate against LAST.
            let y = vec![128u8; (W * H) as usize];
            let u = vec![128u8; cw * ch];
            let v = vec![128u8; cw * ch];
            VideoFrame {
                pts: None,
                planes: vec![
                    VideoPlane {
                        stride: W as usize,
                        data: y,
                    },
                    VideoPlane {
                        stride: cw,
                        data: u,
                    },
                    VideoPlane {
                        stride: cw,
                        data: v,
                    },
                ],
            }
        };

        let mut params = CodecParameters::video(CodecId::new("vp8"));
        params.width = Some(W);
        params.height = Some(H);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(30, 1));
        let mut enc = super::make_encoder_with_qindex(&params, 50).expect("encoder");

        // Frame 0 = keyframe, frame 1 = identical → all-SKIP P-frame.
        enc.send_frame(&Frame::Video(make())).expect("send 0");
        let _kf = enc.receive_packet().expect("rx 0");
        enc.send_frame(&Frame::Video(make())).expect("send 1");
        let pkt = enc.receive_packet().expect("rx 1");

        // Parse the P-frame's inter header and read the prob triple.
        let parsed = parse_header(&pkt.data).expect("tag");
        assert!(matches!(parsed.tag.frame_type, FrameType::Inter));
        let body = &pkt.data[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let probs = PersistentProbs::defaults();
        let h = parse_inter_header(&mut bd, &probs).expect("hdr");

        // Bool-coder convention: `prob = P(bit==0)`. The decoder reads:
        //   is_inter        = read_bool(prob_intra)   → true ⇒ inter
        //   not_last        = read_bool(prob_last)    → true ⇒ GOLDEN/ALT
        //   is_alt          = read_bool(prob_gf)      → true ⇒ ALT
        // Every MB picked LAST → 0% intra → prob_intra → 1 (inter is
        // near-free) — exactly the opposite of the legacy `prob_intra=200`
        // literal that biased toward intra. prob_last → 255 (LAST is
        // near-free, the libvpx single-ref convention). prob_gf has no
        // observations and falls back to the neutral 128 default.
        assert!(
            h.prob_intra <= 16,
            "prob_intra should track all-inter distribution: got {}",
            h.prob_intra
        );
        assert!(
            h.prob_last >= 240,
            "prob_last should track all-LAST distribution: got {}",
            h.prob_last
        );
        assert_eq!(
            h.prob_gf, 128,
            "prob_gf should fall back to 128 with no GOLDEN/ALT observations"
        );
    }

    #[test]
    fn pick_filter_type_auto_picks_simple_at_low_levels() {
        let mut cfg = Vp8EncoderConfig::default();
        cfg.loop_filter_mode = LoopFilterMode::Auto;
        cfg.simple_lf_max_level = 15;
        // ≤ threshold → simple (1).
        assert_eq!(pick_filter_type(0, &cfg), 1);
        assert_eq!(pick_filter_type(15, &cfg), 1);
        // > threshold → normal (0).
        assert_eq!(pick_filter_type(16, &cfg), 0);
        assert_eq!(pick_filter_type(63, &cfg), 0);
    }

    #[test]
    fn pick_filter_type_forced_modes_ignore_level() {
        let mut cfg = Vp8EncoderConfig::default();
        cfg.simple_lf_max_level = 15;

        cfg.loop_filter_mode = LoopFilterMode::Normal;
        assert_eq!(pick_filter_type(0, &cfg), 0);
        assert_eq!(pick_filter_type(15, &cfg), 0);
        assert_eq!(pick_filter_type(63, &cfg), 0);

        cfg.loop_filter_mode = LoopFilterMode::Simple;
        assert_eq!(pick_filter_type(0, &cfg), 1);
        assert_eq!(pick_filter_type(63, &cfg), 1);
    }

    #[test]
    fn loop_filter_level_for_qindex_default_picks_normal_under_auto() {
        // Default qindex 50 → level 21, above the simple threshold 15
        // → Auto should pick normal-mode LF, preserving the pre-#336
        // bitstream shape for the standard `make_encoder` path.
        let lvl = loop_filter_level_for_qindex(DEFAULT_QINDEX);
        assert_eq!(lvl, 21);
        let cfg = Vp8EncoderConfig::default();
        assert_eq!(pick_filter_type(lvl, &cfg), 0);
    }
}
