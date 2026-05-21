//! VP8 boolean-coded frame header (RFC 6386 §19.2).
//!
//! Where `frame_header.rs` parses the uncompressed three-byte frame
//! tag (and key-frame start code + width/height/scale) directly out
//! of the input bytes, **this** module reads through the
//! [`BoolDecoder`](crate::bool_decoder::BoolDecoder) over the
//! `first_partition_size`-byte control partition that immediately
//! follows the uncompressed header. The whole §19.2 table is
//! probability-128 fixed-width literals (no actual entropy coding of
//! the values themselves; the bool decoder simply consumes bits at
//! the arithmetic-coder's natural rate).
//!
//! # Scope of this round
//!
//! Round 3 covers the table from the top of §19.2 through
//! `mb_no_skip_coeff` / `prob_skip_false` — every field whose presence
//! in the stream is determined by the frame's key/inter flag alone and
//! one or two enable bits along the way. Specifically:
//!
//! 1. If `key_frame`: `color_space` (L1) and `clamping_type` (L1).
//! 2. `segmentation_enabled` (L1). When set, the
//!    `update_segmentation()` block follows; this round records the
//!    *shape* of that block via the [`UpdateSegmentation`] struct (the
//!    enable bits and the 4-segment / 4-loop-filter delta arrays).
//! 3. `filter_type` (L1), `loop_filter_level` (L6), `sharpness_level` (L3).
//! 4. `mb_lf_adjustments()` — `loop_filter_adj_enable` (L1) and, when
//!    set, `mode_ref_lf_delta_update` (L1) plus the eight `ref_frame`/
//!    `mb_mode` delta-update flag + 6-bit magnitude + sign triples.
//! 5. `log2_nbr_of_dct_partitions` (L2). The decoded
//!    `nbr_of_dct_partitions` = `1 << log2_nbr_of_dct_partitions` is
//!    surfaced for caller convenience (1, 2, 4, or 8).
//! 6. `quant_indices()` — `y_ac_qi` (L7) plus the five 5-bit
//!    `present?` / `delta(L4)` / `sign(L1)` deltas for ydc, y2dc, y2ac,
//!    uvdc, uvac.
//! 7. `refresh_entropy_probs` (L1) on every frame.
//! 8. Inter frames only: `refresh_golden_frame` (L1),
//!    `refresh_alternate_frame` (L1), and — when the corresponding
//!    refresh flag is 0 — `copy_buffer_to_golden` / `_to_alternate`
//!    (L2 each), plus `sign_bias_golden` (L1), `sign_bias_alternate`
//!    (L1), and `refresh_last` (L1). Key frames force refresh of all
//!    three buffers per §9.7 / §9.8 and so the stream omits these
//!    bits.
//! 9. `token_prob_update()` — the `[4][8][3][11]` DCT-coefficient
//!    context probability sweep (§19.2). Each entry is an
//!    `update_flag` (L1) plus an optional 8-bit probability. This
//!    block has to be consumed in order to reach the
//!    `mb_no_skip_coeff` bit that follows it, so it is decoded here
//!    even though the resulting probabilities are not yet used.
//! 10. `mb_no_skip_coeff` (L1) and, when set, `prob_skip_false` (L8).
//!
//! What this round deliberately does **not** decode:
//!
//! * `prob_intra` / `prob_last` / `prob_gf` and the intra-mode /
//!   intra-chroma probability updates (inter-frame only).
//! * `mv_prob_update()` — motion-vector context probability updates
//!   (inter-frame only).
//!
//! Those land in later rounds together with the macroblock-decode work
//! that consumes them.
//!
//! # Reference
//!
//! RFC 6386 §19.2 (syntax table), §9.2–§9.10 (semantics of the
//! individual fields), and §7 (the underlying boolean decoder).

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use core::fmt;

/// Errors surfaced by [`Vp8CodedHeader::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodedHeaderError {
    /// The boolean decoder ran out of input partway through the
    /// header. Wraps the underlying [`BoolDecoderError`] which carries
    /// the specific reason (input shorter than two bytes for `init`,
    /// or end-of-stream during renormalisation).
    BoolDecoder(BoolDecoderError),
}

impl fmt::Display for CodedHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodedHeaderError::BoolDecoder(inner) => {
                write!(f, "vp8 coded header: {inner}")
            }
        }
    }
}

impl std::error::Error for CodedHeaderError {}

impl From<BoolDecoderError> for CodedHeaderError {
    fn from(value: BoolDecoderError) -> Self {
        CodedHeaderError::BoolDecoder(value)
    }
}

/// `update_segmentation()` sub-block of §19.2.
///
/// Present in [`Vp8CodedHeader`] only when `segmentation_enabled` is
/// `true` in the same frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateSegmentation {
    /// L(1) — whether the per-MB segmentation map is updated for the
    /// current frame.
    pub update_mb_segmentation_map: bool,
    /// L(1) — whether the segment-feature data block follows.
    pub update_segment_feature_data: bool,
    /// L(1) — `segment_feature_mode`: `false` = delta, `true` =
    /// absolute (RFC 6386 §9.3 4.a). Only meaningful when
    /// `update_segment_feature_data` is `true`.
    pub segment_feature_mode_absolute: bool,
    /// Per-segment quantizer deltas. `None` for segments whose
    /// `quantizer_update` bit was 0; otherwise the signed delta value
    /// reconstructed from L(7) magnitude + L(1) sign.
    pub quantizer_update: [Option<i16>; 4],
    /// Per-segment loop-filter deltas. `None` for segments whose
    /// `loop_filter_update` bit was 0; otherwise the signed delta
    /// reconstructed from L(6) magnitude + L(1) sign.
    pub loop_filter_update: [Option<i16>; 4],
    /// L(1) × 3 — segmentation-map branch probabilities, present only
    /// when `update_mb_segmentation_map` is `true`. `None` for the
    /// entries whose `segment_prob_update` flag was 0 (the decoder is
    /// expected to fall back to 255 per §9.3 item 5); otherwise the
    /// L(8) value from the bitstream.
    pub segment_prob: [Option<u8>; 3],
}

/// `mb_lf_adjustments()` sub-block of §19.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbLfAdjustments {
    /// `loop_filter_adj_enable` — top-level toggle for per-macroblock
    /// loop-filter adjustments (RFC 6386 §9.4).
    pub loop_filter_adj_enable: bool,
    /// `mode_ref_lf_delta_update` — only present when
    /// `loop_filter_adj_enable` is set.
    pub mode_ref_lf_delta_update: bool,
    /// Per-reference-frame deltas (`MAX_REF_LF_DELTAS == 4`).
    /// Reconstructed as L(6) magnitude + L(1) sign for each
    /// flag-true entry; `None` for entries whose
    /// `ref_frame_delta_update_flag` was 0.
    pub ref_frame_delta_update: [Option<i16>; 4],
    /// Per-prediction-mode deltas (`MAX_MODE_LF_DELTAS == 4`). Same
    /// L(6) + L(1) reconstruction as `ref_frame_delta_update`.
    pub mb_mode_delta_update: [Option<i16>; 4],
}

/// `quant_indices()` sub-block of §19.2 / §9.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantIndices {
    /// L(7) `y_ac_qi` — baseline dequantisation index always present.
    pub y_ac_qi: u8,
    /// Signed delta for Y DC. `None` when the `y_dc_delta_present`
    /// flag was 0.
    pub y_dc_delta: Option<i8>,
    /// Signed delta for Y2 DC.
    pub y2_dc_delta: Option<i8>,
    /// Signed delta for Y2 AC.
    pub y2_ac_delta: Option<i8>,
    /// Signed delta for chroma DC.
    pub uv_dc_delta: Option<i8>,
    /// Signed delta for chroma AC.
    pub uv_ac_delta: Option<i8>,
}

/// `token_prob_update()` block (§19.2). The four nested dimensions
/// are `[block_type][band][prev_token_class][token_position]`. Each
/// entry is `Some(prob)` when the encoder transmitted a new
/// probability and `None` when the bitstream is signalling
/// "inherit the default".
pub type TokenProbUpdates = [[[[Option<u8>; 11]; 3]; 8]; 4];

/// Decoded prefix of the boolean-coded frame header — every field up
/// to and including `prob_skip_false`.
///
/// Caller responsibility:
///
/// * Construct via [`Vp8CodedHeader::parse`], passing the parsed
///   uncompressed header's `key_frame` flag and the slice
///   `&bytes[header_bytes_consumed..][..first_partition_size]`.
/// * The bool decoder cursor inside `parse` is advanced past every
///   field listed at the module level (including the
///   `token_prob_update()` sweep); subsequent rounds wiring up
///   `prob_intra`, `prob_last`, `prob_gf` and `mv_prob_update()`
///   re-`init` the bool decoder at the same partition start and
///   replay this prefix to reach their entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp8CodedHeader {
    /// `color_space` (key frame only). Per §9.2 only value `0`
    /// (BT.601-like) is defined; we surface the raw bit so audits can
    /// confirm.
    pub color_space: Option<bool>,
    /// `clamping_type` (key frame only). Per §9.2: `0` requires
    /// clamping; `1` declares pre-clamped pixels.
    pub clamping_type: Option<bool>,
    /// `segmentation_enabled` — frame-level segmentation toggle.
    pub segmentation_enabled: bool,
    /// `update_segmentation()` block. `None` when
    /// `segmentation_enabled` is `false`.
    pub update_segmentation: Option<UpdateSegmentation>,
    /// `filter_type` — `false` = normal, `true` = simple loop filter.
    pub filter_type: bool,
    /// `loop_filter_level` (6 bits, `0..=63`).
    pub loop_filter_level: u8,
    /// `sharpness_level` (3 bits, `0..=7`).
    pub sharpness_level: u8,
    /// `mb_lf_adjustments()` block.
    pub mb_lf_adjustments: MbLfAdjustments,
    /// `log2_nbr_of_dct_partitions` (2 bits) per §9.5.
    pub log2_nbr_of_dct_partitions: u8,
    /// `1 << log2_nbr_of_dct_partitions` — 1, 2, 4, or 8.
    pub nbr_of_dct_partitions: u8,
    /// `quant_indices()` block.
    pub quant_indices: QuantIndices,
    /// `refresh_entropy_probs` — true means token-probability updates
    /// in this frame persist to the next frame's defaults.
    pub refresh_entropy_probs: bool,
    /// `refresh_golden_frame` (inter only). Key frames implicitly
    /// refresh the golden buffer per §9.7, so this is `None` on key
    /// frames.
    pub refresh_golden_frame: Option<bool>,
    /// `refresh_alternate_frame` (inter only). Same key-frame
    /// behaviour as `refresh_golden_frame`.
    pub refresh_alternate_frame: Option<bool>,
    /// `copy_buffer_to_golden` (inter only). 0 = no copy, 1 = copy
    /// last_frame, 2 = copy alt_ref_frame. Present in the stream
    /// only when `refresh_golden_frame` is 0; otherwise `None`.
    pub copy_buffer_to_golden: Option<u8>,
    /// `copy_buffer_to_alternate` (inter only). 0 = no copy, 1 =
    /// copy last_frame, 2 = copy golden_frame. Present in the
    /// stream only when `refresh_alternate_frame` is 0; otherwise
    /// `None`.
    pub copy_buffer_to_alternate: Option<u8>,
    /// `sign_bias_golden` (inter only). Controls MV sign for the
    /// golden reference.
    pub sign_bias_golden: Option<bool>,
    /// `sign_bias_alternate` (inter only).
    pub sign_bias_alternate: Option<bool>,
    /// `refresh_last` (inter only). Key frames implicitly refresh
    /// the last buffer per §9.8, so this is `None` on key frames.
    pub refresh_last: Option<bool>,
    /// `token_prob_update()` block (§19.2). The 4×8×3×11 entries
    /// each carry either `None` (the corresponding
    /// `coeff_prob_update_flag` was 0) or the L(8) replacement
    /// probability. The block has to be parsed in order to reach
    /// the `mb_no_skip_coeff` bit that follows it.
    pub token_prob_updates: TokenProbUpdates,
    /// `mb_no_skip_coeff` — frame-level enable for per-MB
    /// `mb_skip_coeff` flag.
    pub mb_no_skip_coeff: bool,
    /// `prob_skip_false` — probability used to decode the per-MB
    /// `mb_skip_coeff` flag. `None` when `mb_no_skip_coeff` is
    /// `false` (in which case `mb_skip_coeff` is forced to 0 for all
    /// MBs per §9.10/§9.11).
    pub prob_skip_false: Option<u8>,
}

impl Vp8CodedHeader {
    /// Parse the boolean-coded frame header from the start of the
    /// control partition.
    ///
    /// `partition` is the `first_partition_size` bytes the caller
    /// extracted from the input slice using
    /// [`Vp8FrameHeader::header_bytes_consumed`](crate::frame_header::Vp8FrameHeader)
    /// and the parsed `first_partition_size`. `key_frame` is the
    /// same field off the uncompressed header.
    pub fn parse(partition: &[u8], key_frame: bool) -> Result<Self, CodedHeaderError> {
        let mut dec = BoolDecoder::init(partition)?;

        // §19.2 row 1–2: key-frame-only color_space / clamping_type.
        let (color_space, clamping_type) = if key_frame {
            let cs = dec.read_bool(128)?;
            let ct = dec.read_bool(128)?;
            (Some(cs), Some(ct))
        } else {
            (None, None)
        };

        // §19.2 row 3 + §9.3: segmentation toggle, optional sub-block.
        let segmentation_enabled = dec.read_bool(128)?;
        let update_segmentation = if segmentation_enabled {
            Some(parse_update_segmentation(&mut dec)?)
        } else {
            None
        };

        // §19.2 + §9.4 — loop filter knobs.
        let filter_type = dec.read_bool(128)?;
        let loop_filter_level = dec.read_literal(6)? as u8;
        let sharpness_level = dec.read_literal(3)? as u8;

        // mb_lf_adjustments() sub-block.
        let mb_lf_adjustments = parse_mb_lf_adjustments(&mut dec)?;

        // §19.2 + §9.5 — DCT partition count.
        let log2_nbr_of_dct_partitions = dec.read_literal(2)? as u8;
        let nbr_of_dct_partitions = 1u8 << log2_nbr_of_dct_partitions;

        // quant_indices() sub-block (§9.6).
        let quant_indices = parse_quant_indices(&mut dec)?;

        // §19.2 — refresh / sign-bias bits. Key vs inter is asymmetric:
        // key frames have only refresh_entropy_probs in the stream and
        // implicitly refresh golden / alternate / last; inter frames
        // carry refresh_golden / refresh_alternate / (copy bufs) /
        // sign_bias_* / refresh_entropy_probs / refresh_last.
        let (
            refresh_golden_frame,
            refresh_alternate_frame,
            copy_buffer_to_golden,
            copy_buffer_to_alternate,
            sign_bias_golden,
            sign_bias_alternate,
            refresh_entropy_probs,
            refresh_last,
        ) = if key_frame {
            let refresh_entropy_probs = dec.read_bool(128)?;
            (
                None,
                None,
                None,
                None,
                None,
                None,
                refresh_entropy_probs,
                None,
            )
        } else {
            let refresh_golden_frame = dec.read_bool(128)?;
            let refresh_alternate_frame = dec.read_bool(128)?;
            let copy_buffer_to_golden = if !refresh_golden_frame {
                Some(dec.read_literal(2)? as u8)
            } else {
                None
            };
            let copy_buffer_to_alternate = if !refresh_alternate_frame {
                Some(dec.read_literal(2)? as u8)
            } else {
                None
            };
            let sign_bias_golden = dec.read_bool(128)?;
            let sign_bias_alternate = dec.read_bool(128)?;
            let refresh_entropy_probs = dec.read_bool(128)?;
            let refresh_last = dec.read_bool(128)?;
            (
                Some(refresh_golden_frame),
                Some(refresh_alternate_frame),
                copy_buffer_to_golden,
                copy_buffer_to_alternate,
                Some(sign_bias_golden),
                Some(sign_bias_alternate),
                refresh_entropy_probs,
                Some(refresh_last),
            )
        };

        // token_prob_update() — must be consumed in order to reach
        // mb_no_skip_coeff. The probabilities themselves will be
        // consumed by the macroblock decoder in a later round.
        let token_prob_updates = parse_token_prob_update(&mut dec)?;

        // mb_no_skip_coeff and (conditionally) prob_skip_false.
        let mb_no_skip_coeff = dec.read_bool(128)?;
        let prob_skip_false = if mb_no_skip_coeff {
            Some(dec.read_literal(8)? as u8)
        } else {
            None
        };

        Ok(Vp8CodedHeader {
            color_space,
            clamping_type,
            segmentation_enabled,
            update_segmentation,
            filter_type,
            loop_filter_level,
            sharpness_level,
            mb_lf_adjustments,
            log2_nbr_of_dct_partitions,
            nbr_of_dct_partitions,
            quant_indices,
            refresh_entropy_probs,
            refresh_golden_frame,
            refresh_alternate_frame,
            copy_buffer_to_golden,
            copy_buffer_to_alternate,
            sign_bias_golden,
            sign_bias_alternate,
            refresh_last,
            token_prob_updates,
            mb_no_skip_coeff,
            prob_skip_false,
        })
    }
}

/// `coeff_update_probs[i][j][k][t]` — the fixed-probability table that
/// each `coeff_prob_update_flag` in §19.2's `token_prob_update()` is
/// read against. Transcribed verbatim from RFC 6386 §13.4 (block under
/// "The (constant) update probabilities are as follows"). Note that
/// `coeff_prob_update_flag` is NOT read at probability 128 — using the
/// flat 128 would consume far more bits than the partition holds for
/// frames that elect not to update any probabilities.
const COEFF_UPDATE_PROBS: [[[[u8; 11]; 3]; 8]; 4] = [
    [
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255],
            [249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255],
            [234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255],
            [250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255],
            [254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
    [
        [
            [217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255],
            [234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255],
        ],
        [
            [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255],
            [250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
    [
        [
            [186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255],
            [234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255],
            [251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255],
        ],
        [
            [255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
    [
        [
            [248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255],
            [248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255],
            [246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255],
            [252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255],
            [248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255],
            [253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255],
            [252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255],
            [250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
        [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        ],
    ],
];

fn parse_token_prob_update(
    dec: &mut BoolDecoder<'_>,
) -> Result<TokenProbUpdates, BoolDecoderError> {
    let mut out: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    for (i, plane) in out.iter_mut().enumerate() {
        for (j, band) in plane.iter_mut().enumerate() {
            for (k, ctx) in band.iter_mut().enumerate() {
                for (t, slot) in ctx.iter_mut().enumerate() {
                    // Each update flag is read at the position-specific
                    // probability from §13.4's COEFF_UPDATE_PROBS table —
                    // NOT a flat 128.
                    let present = dec.read_bool(COEFF_UPDATE_PROBS[i][j][k][t])?;
                    *slot = if present {
                        Some(dec.read_literal(8)? as u8)
                    } else {
                        None
                    };
                }
            }
        }
    }
    Ok(out)
}

/// Read a magnitude-`n`-and-sign delta as defined throughout §19.2.
///
/// Returns the reconstructed signed value (`-((1<<n) - 1) ..=
/// (1<<n) - 1`). The boolean decoder reads `n` flag bits for the
/// magnitude (MSB-first) followed by a single sign bit; sign 1 is
/// negative.
fn read_signed_delta(dec: &mut BoolDecoder<'_>, n: u32) -> Result<i16, BoolDecoderError> {
    let magnitude = dec.read_literal(n)? as i16;
    let sign = dec.read_bool(128)?;
    Ok(if sign { -magnitude } else { magnitude })
}

fn parse_update_segmentation(
    dec: &mut BoolDecoder<'_>,
) -> Result<UpdateSegmentation, BoolDecoderError> {
    let update_mb_segmentation_map = dec.read_bool(128)?;
    let update_segment_feature_data = dec.read_bool(128)?;

    let mut segment_feature_mode_absolute = false;
    let mut quantizer_update: [Option<i16>; 4] = [None; 4];
    let mut loop_filter_update: [Option<i16>; 4] = [None; 4];

    if update_segment_feature_data {
        segment_feature_mode_absolute = dec.read_bool(128)?;
        for slot in &mut quantizer_update {
            let present = dec.read_bool(128)?;
            *slot = if present {
                Some(read_signed_delta(dec, 7)?)
            } else {
                None
            };
        }
        for slot in &mut loop_filter_update {
            let present = dec.read_bool(128)?;
            *slot = if present {
                Some(read_signed_delta(dec, 6)?)
            } else {
                None
            };
        }
    }

    let mut segment_prob: [Option<u8>; 3] = [None; 3];
    if update_mb_segmentation_map {
        for slot in &mut segment_prob {
            let present = dec.read_bool(128)?;
            *slot = if present {
                Some(dec.read_literal(8)? as u8)
            } else {
                None
            };
        }
    }

    Ok(UpdateSegmentation {
        update_mb_segmentation_map,
        update_segment_feature_data,
        segment_feature_mode_absolute,
        quantizer_update,
        loop_filter_update,
        segment_prob,
    })
}

fn parse_mb_lf_adjustments(dec: &mut BoolDecoder<'_>) -> Result<MbLfAdjustments, BoolDecoderError> {
    let loop_filter_adj_enable = dec.read_bool(128)?;
    let mut mode_ref_lf_delta_update = false;
    let mut ref_frame_delta_update: [Option<i16>; 4] = [None; 4];
    let mut mb_mode_delta_update: [Option<i16>; 4] = [None; 4];

    if loop_filter_adj_enable {
        mode_ref_lf_delta_update = dec.read_bool(128)?;
        if mode_ref_lf_delta_update {
            for slot in &mut ref_frame_delta_update {
                let present = dec.read_bool(128)?;
                *slot = if present {
                    Some(read_signed_delta(dec, 6)?)
                } else {
                    None
                };
            }
            for slot in &mut mb_mode_delta_update {
                let present = dec.read_bool(128)?;
                *slot = if present {
                    Some(read_signed_delta(dec, 6)?)
                } else {
                    None
                };
            }
        }
    }

    Ok(MbLfAdjustments {
        loop_filter_adj_enable,
        mode_ref_lf_delta_update,
        ref_frame_delta_update,
        mb_mode_delta_update,
    })
}

fn parse_quant_indices(dec: &mut BoolDecoder<'_>) -> Result<QuantIndices, BoolDecoderError> {
    let y_ac_qi = dec.read_literal(7)? as u8;
    let mut deltas: [Option<i8>; 5] = [None; 5];
    for slot in &mut deltas {
        let present = dec.read_bool(128)?;
        *slot = if present {
            // §9.6: L(4) magnitude + L(1) sign. Range is
            // -15..=15, so i8 is plenty.
            let mag = dec.read_literal(4)? as i8;
            let sign = dec.read_bool(128)?;
            Some(if sign { -mag } else { mag })
        } else {
            None
        };
    }

    Ok(QuantIndices {
        y_ac_qi,
        y_dc_delta: deltas[0],
        y2_dc_delta: deltas[1],
        y2_ac_delta: deltas[2],
        uv_dc_delta: deltas[3],
        uv_ac_delta: deltas[4],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // In-test boolean encoder mirroring RFC 6386 §7.2 — copy of the
    // helper in `bool_decoder::tests`, kept module-local so the
    // production crate exports nothing that resembles an encoder.
    // This is the only practical way to build ground-truth payloads
    // for the round-trip tests below.
    struct TestEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl TestEncoder {
        fn new() -> Self {
            TestEncoder {
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

        fn write_literal(&mut self, value: u32, num_bits: u32) {
            for i in (0..num_bits).rev() {
                let bit = ((value >> i) & 1) == 1;
                self.write_bool(128, bit);
            }
        }

        /// Convenience for the tests: emit a token_prob_update()
        /// sub-block whose every update_flag is 0 (no probability
        /// updates). The flag at position [i][j][k][t] is coded at
        /// `COEFF_UPDATE_PROBS[i][j][k][t]` per §13.4 — most are 255,
        /// so a `false` flag consumes only a fraction of a bit.
        fn write_empty_token_prob_update(&mut self) {
            for plane in &COEFF_UPDATE_PROBS {
                for band in plane {
                    for ctx in band {
                        for &prob in ctx {
                            self.write_bool(prob, false);
                        }
                    }
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            let c = self.bit_count;
            let mut v = self.bottom;
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
            self.out
        }
    }

    /// Encode the minimal key-frame coded header — every optional
    /// sub-block omitted — and round-trip it through `parse`.
    fn build_minimal_key_frame() -> Vec<u8> {
        let mut enc = TestEncoder::new();
        enc.write_bool(128, false); // color_space = 0
        enc.write_bool(128, false); // clamping_type = 0
        enc.write_bool(128, false); // segmentation_enabled = false
        enc.write_bool(128, false); // filter_type = false
        enc.write_literal(0, 6); // loop_filter_level = 0
        enc.write_literal(0, 3); // sharpness_level = 0
        enc.write_bool(128, false); // loop_filter_adj_enable = false
        enc.write_literal(0, 2); // log2_nbr_of_dct_partitions = 0
        enc.write_literal(4, 7); // y_ac_qi = 4
        for _ in 0..5 {
            enc.write_bool(128, false); // each delta-present flag = 0
        }
        enc.write_bool(128, true); // refresh_entropy_probs = true
        enc.write_empty_token_prob_update();
        enc.write_bool(128, false); // mb_no_skip_coeff = false
        enc.finish()
    }

    #[test]
    fn minimal_key_frame_round_trips() {
        let buf = build_minimal_key_frame();
        let hdr = Vp8CodedHeader::parse(&buf, true).unwrap();
        assert_eq!(hdr.color_space, Some(false));
        assert_eq!(hdr.clamping_type, Some(false));
        assert!(!hdr.segmentation_enabled);
        assert!(hdr.update_segmentation.is_none());
        assert!(!hdr.filter_type);
        assert_eq!(hdr.loop_filter_level, 0);
        assert_eq!(hdr.sharpness_level, 0);
        assert!(!hdr.mb_lf_adjustments.loop_filter_adj_enable);
        assert!(!hdr.mb_lf_adjustments.mode_ref_lf_delta_update);
        assert_eq!(hdr.log2_nbr_of_dct_partitions, 0);
        assert_eq!(hdr.nbr_of_dct_partitions, 1);
        assert_eq!(hdr.quant_indices.y_ac_qi, 4);
        assert_eq!(hdr.quant_indices.y_dc_delta, None);
        assert_eq!(hdr.quant_indices.uv_ac_delta, None);
        assert!(hdr.refresh_entropy_probs);
        assert_eq!(hdr.refresh_golden_frame, None);
        assert_eq!(hdr.refresh_alternate_frame, None);
        assert_eq!(hdr.refresh_last, None);
        assert!(!hdr.mb_no_skip_coeff);
        assert_eq!(hdr.prob_skip_false, None);
    }

    #[test]
    fn key_frame_filter_and_partition_fields_round_trip() {
        let mut enc = TestEncoder::new();
        enc.write_bool(128, false); // color_space
        enc.write_bool(128, true); // clamping_type
        enc.write_bool(128, false); // segmentation_enabled
        enc.write_bool(128, true); // filter_type = simple
        enc.write_literal(63, 6); // loop_filter_level = 63 (6-bit max)
        enc.write_literal(7, 3); // sharpness_level = 7 (3-bit max)
        enc.write_bool(128, false); // loop_filter_adj_enable = false
        enc.write_literal(3, 2); // log2_nbr_of_dct_partitions = 3 → 8 parts
        enc.write_literal(127, 7); // y_ac_qi = 127 (7-bit max)
        for _ in 0..5 {
            enc.write_bool(128, false);
        }
        enc.write_bool(128, false); // refresh_entropy_probs
        enc.write_empty_token_prob_update();
        enc.write_bool(128, true); // mb_no_skip_coeff
        enc.write_literal(0xab, 8); // prob_skip_false = 0xab
        let buf = enc.finish();

        let hdr = Vp8CodedHeader::parse(&buf, true).unwrap();
        assert_eq!(hdr.clamping_type, Some(true));
        assert!(hdr.filter_type);
        assert_eq!(hdr.loop_filter_level, 63);
        assert_eq!(hdr.sharpness_level, 7);
        assert_eq!(hdr.log2_nbr_of_dct_partitions, 3);
        assert_eq!(hdr.nbr_of_dct_partitions, 8);
        assert_eq!(hdr.quant_indices.y_ac_qi, 127);
        assert!(!hdr.refresh_entropy_probs);
        assert!(hdr.mb_no_skip_coeff);
        assert_eq!(hdr.prob_skip_false, Some(0xab));
    }

    #[test]
    fn mb_lf_adjustments_full_block_round_trips() {
        // loop_filter_adj_enable = 1, mode_ref_lf_delta_update = 1,
        // with the eight delta-update flags carrying a mix of
        // present-with-value and absent entries that exercise the
        // sign-bit encoding.
        let mut enc = TestEncoder::new();
        enc.write_bool(128, false); // color_space
        enc.write_bool(128, false); // clamping_type
        enc.write_bool(128, false); // segmentation_enabled
        enc.write_bool(128, false); // filter_type
        enc.write_literal(0, 6); // lf_level
        enc.write_literal(0, 3); // sharpness
        enc.write_bool(128, true); // loop_filter_adj_enable = true
        enc.write_bool(128, true); // mode_ref_lf_delta_update = true
                                   // ref_frame deltas: (+1, none, -2, +5)
        enc.write_bool(128, true); // present
        enc.write_literal(1, 6);
        enc.write_bool(128, false); // sign +
        enc.write_bool(128, false); // not present
        enc.write_bool(128, true);
        enc.write_literal(2, 6);
        enc.write_bool(128, true); // sign -
        enc.write_bool(128, true);
        enc.write_literal(5, 6);
        enc.write_bool(128, false);
        // mb_mode deltas: (none, +3, -1, none)
        enc.write_bool(128, false);
        enc.write_bool(128, true);
        enc.write_literal(3, 6);
        enc.write_bool(128, false);
        enc.write_bool(128, true);
        enc.write_literal(1, 6);
        enc.write_bool(128, true);
        enc.write_bool(128, false);
        enc.write_literal(0, 2); // log2_dct_parts
        enc.write_literal(0, 7); // y_ac_qi
        for _ in 0..5 {
            enc.write_bool(128, false);
        }
        enc.write_bool(128, false); // refresh_entropy_probs
        enc.write_empty_token_prob_update();
        enc.write_bool(128, false); // mb_no_skip_coeff
        let buf = enc.finish();

        let hdr = Vp8CodedHeader::parse(&buf, true).unwrap();
        let lf = hdr.mb_lf_adjustments;
        assert!(lf.loop_filter_adj_enable);
        assert!(lf.mode_ref_lf_delta_update);
        assert_eq!(
            lf.ref_frame_delta_update,
            [Some(1), None, Some(-2), Some(5)]
        );
        assert_eq!(lf.mb_mode_delta_update, [None, Some(3), Some(-1), None]);
    }

    #[test]
    fn update_segmentation_block_round_trips() {
        // segmentation_enabled = true, both sub-updates exercised:
        // - quantizer deltas: (+10, none, none, -20)
        // - lf deltas:        (none, +5, -3, none)
        // - segment_prob:     (255, none, 0)
        // - segment_feature_mode_absolute = true
        let mut enc = TestEncoder::new();
        enc.write_bool(128, false); // color_space
        enc.write_bool(128, false); // clamping_type
        enc.write_bool(128, true); // segmentation_enabled = true
                                   // update_segmentation():
        enc.write_bool(128, true); // update_mb_segmentation_map = 1
        enc.write_bool(128, true); // update_segment_feature_data = 1
        enc.write_bool(128, true); // segment_feature_mode = 1 (absolute)
                                   // quantizer_update[0..4]:
        enc.write_bool(128, true);
        enc.write_literal(10, 7);
        enc.write_bool(128, false); // +10
        enc.write_bool(128, false); // absent
        enc.write_bool(128, false); // absent
        enc.write_bool(128, true);
        enc.write_literal(20, 7);
        enc.write_bool(128, true); // -20
                                   // loop_filter_update[0..4]:
        enc.write_bool(128, false);
        enc.write_bool(128, true);
        enc.write_literal(5, 6);
        enc.write_bool(128, false); // +5
        enc.write_bool(128, true);
        enc.write_literal(3, 6);
        enc.write_bool(128, true); // -3
        enc.write_bool(128, false);
        // segment_prob[0..3] (update_mb_segmentation_map=1 so this fires):
        enc.write_bool(128, true);
        enc.write_literal(255, 8);
        enc.write_bool(128, false);
        enc.write_bool(128, true);
        enc.write_literal(0, 8);

        // Remaining required fields after update_segmentation():
        enc.write_bool(128, false); // filter_type
        enc.write_literal(0, 6); // lf_level
        enc.write_literal(0, 3); // sharpness
        enc.write_bool(128, false); // lf_adj_enable
        enc.write_literal(0, 2); // log2_dct_parts
        enc.write_literal(0, 7); // y_ac_qi
        for _ in 0..5 {
            enc.write_bool(128, false);
        }
        enc.write_bool(128, false); // refresh_entropy_probs
        enc.write_empty_token_prob_update();
        enc.write_bool(128, false); // mb_no_skip_coeff
        let buf = enc.finish();

        let hdr = Vp8CodedHeader::parse(&buf, true).unwrap();
        assert!(hdr.segmentation_enabled);
        let seg = hdr.update_segmentation.unwrap();
        assert!(seg.update_mb_segmentation_map);
        assert!(seg.update_segment_feature_data);
        assert!(seg.segment_feature_mode_absolute);
        assert_eq!(seg.quantizer_update, [Some(10), None, None, Some(-20)]);
        assert_eq!(seg.loop_filter_update, [None, Some(5), Some(-3), None]);
        assert_eq!(seg.segment_prob, [Some(255), None, Some(0)]);
    }

    #[test]
    fn quant_indices_all_deltas_present() {
        // y_ac_qi = 64, deltas = (+1, -2, +15, -15, +0)
        let mut enc = TestEncoder::new();
        enc.write_bool(128, false); // color_space
        enc.write_bool(128, false); // clamping_type
        enc.write_bool(128, false); // segmentation_enabled
        enc.write_bool(128, false); // filter_type
        enc.write_literal(0, 6); // lf_level
        enc.write_literal(0, 3); // sharpness
        enc.write_bool(128, false); // lf_adj_enable
        enc.write_literal(0, 2); // log2_dct_parts
        enc.write_literal(64, 7); // y_ac_qi
                                  // y_dc_delta = +1
        enc.write_bool(128, true);
        enc.write_literal(1, 4);
        enc.write_bool(128, false);
        // y2_dc_delta = -2
        enc.write_bool(128, true);
        enc.write_literal(2, 4);
        enc.write_bool(128, true);
        // y2_ac_delta = +15
        enc.write_bool(128, true);
        enc.write_literal(15, 4);
        enc.write_bool(128, false);
        // uv_dc_delta = -15
        enc.write_bool(128, true);
        enc.write_literal(15, 4);
        enc.write_bool(128, true);
        // uv_ac_delta = +0
        enc.write_bool(128, true);
        enc.write_literal(0, 4);
        enc.write_bool(128, false);

        enc.write_bool(128, false); // refresh_entropy_probs
        enc.write_empty_token_prob_update();
        enc.write_bool(128, false); // mb_no_skip_coeff
        let buf = enc.finish();

        let hdr = Vp8CodedHeader::parse(&buf, true).unwrap();
        assert_eq!(hdr.quant_indices.y_ac_qi, 64);
        assert_eq!(hdr.quant_indices.y_dc_delta, Some(1));
        assert_eq!(hdr.quant_indices.y2_dc_delta, Some(-2));
        assert_eq!(hdr.quant_indices.y2_ac_delta, Some(15));
        assert_eq!(hdr.quant_indices.uv_dc_delta, Some(-15));
        assert_eq!(hdr.quant_indices.uv_ac_delta, Some(0));
    }

    #[test]
    fn interframe_refresh_block_full_path() {
        // refresh_golden_frame = 0 → copy_buffer_to_golden present
        // refresh_alternate_frame = 1 → no copy_buffer_to_alternate
        // sign_bias_golden = 1, sign_bias_alternate = 0
        // refresh_entropy_probs = 0
        // refresh_last = 1
        let mut enc = TestEncoder::new();
        // Inter frame: no color_space / clamping_type.
        enc.write_bool(128, false); // segmentation_enabled
        enc.write_bool(128, false); // filter_type
        enc.write_literal(10, 6); // lf_level
        enc.write_literal(2, 3); // sharpness
        enc.write_bool(128, false); // lf_adj_enable
        enc.write_literal(1, 2); // log2_dct_parts = 1 → 2 parts
        enc.write_literal(8, 7); // y_ac_qi
        for _ in 0..5 {
            enc.write_bool(128, false);
        }
        // Inter-frame refresh ladder:
        enc.write_bool(128, false); // refresh_golden_frame = 0
        enc.write_bool(128, true); // refresh_alternate_frame = 1
        enc.write_literal(2, 2); // copy_buffer_to_golden = 2 (alt → gold)
                                 // (no copy_buffer_to_alternate because refresh_alt = 1)
        enc.write_bool(128, true); // sign_bias_golden
        enc.write_bool(128, false); // sign_bias_alternate
        enc.write_bool(128, false); // refresh_entropy_probs
        enc.write_bool(128, true); // refresh_last
        enc.write_empty_token_prob_update();
        enc.write_bool(128, true); // mb_no_skip_coeff
        enc.write_literal(200, 8); // prob_skip_false
        let buf = enc.finish();

        let hdr = Vp8CodedHeader::parse(&buf, false).unwrap();
        assert_eq!(hdr.color_space, None);
        assert_eq!(hdr.clamping_type, None);
        assert_eq!(hdr.loop_filter_level, 10);
        assert_eq!(hdr.sharpness_level, 2);
        assert_eq!(hdr.nbr_of_dct_partitions, 2);
        assert_eq!(hdr.refresh_golden_frame, Some(false));
        assert_eq!(hdr.refresh_alternate_frame, Some(true));
        assert_eq!(hdr.copy_buffer_to_golden, Some(2));
        assert_eq!(hdr.copy_buffer_to_alternate, None);
        assert_eq!(hdr.sign_bias_golden, Some(true));
        assert_eq!(hdr.sign_bias_alternate, Some(false));
        assert!(!hdr.refresh_entropy_probs);
        assert_eq!(hdr.refresh_last, Some(true));
        assert!(hdr.mb_no_skip_coeff);
        assert_eq!(hdr.prob_skip_false, Some(200));
    }

    #[test]
    fn mb_no_skip_coeff_false_omits_prob_skip_false() {
        let buf = build_minimal_key_frame();
        let hdr = Vp8CodedHeader::parse(&buf, true).unwrap();
        assert!(!hdr.mb_no_skip_coeff);
        assert_eq!(hdr.prob_skip_false, None);
    }

    #[test]
    fn input_too_short_surfaces_bool_decoder_error() {
        // Empty input → BoolDecoder::init rejects it.
        let err = Vp8CodedHeader::parse(&[], true).unwrap_err();
        assert!(matches!(
            err,
            CodedHeaderError::BoolDecoder(BoolDecoderError::InputTooShort)
        ));
        // One byte is also too short.
        let err = Vp8CodedHeader::parse(&[0x00], true).unwrap_err();
        assert!(matches!(
            err,
            CodedHeaderError::BoolDecoder(BoolDecoderError::InputTooShort)
        ));
    }

    #[test]
    fn parses_real_tiny_fixture_partition() {
        // Re-use the 16x16 key frame from docs/video/vp8/fixtures/
        // tiny-i-only-16x16/input.ivf. The IVF frame payload starts
        // at file offset 0x2c. The 10-byte uncompressed header sits at
        // 0x2c..0x36 (parsed in frame_header.rs tests); the bool-coded
        // partition is the next first_partition_size=23 bytes —
        // i.e. file offset 0x36..0x4d.
        //
        // Expected values per docs/video/vp8/fixtures/
        // tiny-i-only-16x16/trace.txt line 1:
        //   filter_simple=0, filter_level=1, filter_sharpness=0,
        //   lf_delta_enabled=1, num_partitions=1, seg_enabled=0,
        //   mbskip_enabled=1, prob_skip=255, qi_y_ac=4,
        //   all qi_*_d=0, update_probs=1.
        let partition: [u8; 23] = [
            0x00, 0x47, 0x08, 0x85, 0x85, 0x88, 0x85, 0x84, 0x88, 0x02, 0x02, 0x02, 0x75, 0xaa,
            0x03, 0xf8, 0x03, 0xfa, 0x02, 0x0d, 0x4d, 0x18, 0x00,
        ];
        let hdr = Vp8CodedHeader::parse(&partition, true).unwrap();
        // color_space and clamping_type are not in the trace listing —
        // §9.2 only defines color_space=0; clamping_type may be either
        // 0 or 1. We just confirm both bits decoded (i.e. didn't
        // collapse to None on this key frame).
        assert!(hdr.color_space.is_some());
        assert!(hdr.clamping_type.is_some());
        assert!(!hdr.segmentation_enabled);
        assert!(!hdr.filter_type, "filter_simple=0 per trace");
        assert_eq!(hdr.loop_filter_level, 1, "filter_level=1 per trace");
        assert_eq!(hdr.sharpness_level, 0, "filter_sharpness=0 per trace");
        assert!(
            hdr.mb_lf_adjustments.loop_filter_adj_enable,
            "lf_delta_enabled=1 per trace"
        );
        assert_eq!(
            hdr.log2_nbr_of_dct_partitions, 0,
            "num_partitions=1 per trace → log2=0"
        );
        assert_eq!(hdr.nbr_of_dct_partitions, 1);
        assert_eq!(hdr.quant_indices.y_ac_qi, 4, "qi_y_ac=4 per trace");
        assert_eq!(hdr.quant_indices.y_dc_delta, None, "qi_y_dc_d=0 per trace");
        assert_eq!(hdr.quant_indices.y2_dc_delta, None);
        assert_eq!(hdr.quant_indices.y2_ac_delta, None);
        assert_eq!(hdr.quant_indices.uv_dc_delta, None);
        assert_eq!(hdr.quant_indices.uv_ac_delta, None);
        assert!(hdr.refresh_entropy_probs, "update_probs=1 per trace");
        assert!(hdr.mb_no_skip_coeff, "mbskip_enabled=1 per trace");
        assert_eq!(hdr.prob_skip_false, Some(255), "prob_skip=255 per trace");
    }
}
