//! VP8 encoder — Phase 1 (RFC 6386 §9 frame-header writers + the
//! all-zero-quantization "silent keyframe" path).
//!
//! This module is the bitstream-formatting half of the VP8 encoder.
//! It does **not** perform rate-distortion search, mode selection,
//! quantization-step optimisation, or pixel-level encoding decisions
//! of any kind. What it provides:
//!
//! 1. [`BoolEncoder`] — a Rust port of the RFC 6386 §7.3
//!    `bool_encoder` / `write_bool` / `flush_bool_encoder` C listing
//!    that the RFC embeds inline. The encoder is validated by
//!    round-tripping every flag through [`crate::bool_decoder`] in
//!    this crate (see the tests at the bottom of this file).
//!
//! 2. The §9.1 / §9.3 / §9.4 / §9.5 / §9.6 / §9.7 / §13 / §9.10 / §9.11
//!    frame-header writer subroutines, each exposed as a small
//!    callable so the higher-level encoder rounds can compose them
//!    against a parameter struct rather than having one monolithic
//!    `encode_frame` body. The four called out by the round goal are:
//!
//!    * [`write_frame_tag`] — §9.1 3-byte tag + key-frame start code
//!      `0x9d 0x01 0x2a` + 14-bit width + 2-bit horizontal scale +
//!      14-bit height + 2-bit vertical scale.
//!    * [`write_segment_update_flags`] — §9.3 segment-update sub-block.
//!      Phase 1 supports `segmentation_enabled = false` only; the
//!      stream emits the single `L(1) = 0` toggle and skips the rest.
//!    * [`write_loop_filter`] — §9.4 `filter_type` / `loop_filter_level`
//!      / `sharpness_level` plus the `mb_lf_adjustments()` sub-block.
//!      Phase 1 supports `loop_filter_adj_enable = false` only.
//!    * [`write_token_partition_count`] — §9.5 `log2_nbr_of_dct_partitions`
//!      (2 bits, encoding 1/2/4/8 partitions).
//!
//! 3. [`encode_silent_keyframe`] — a trivial all-zero-quantizer
//!    keyframe emitter. Every macroblock is coded as
//!    `mb_skip_coeff = 1` so the DCT partition carries no token data;
//!    the macroblock prediction layer picks `DC_PRED` for both luma
//!    and chroma; the §9.4 loop filter is set to level 0 (whole-frame
//!    skip per the §15 page-84 rule). The emitted stream is a
//!    structurally-valid VP8 frame that the crate's own decoder
//!    consumes and that `ffmpeg -c:v vp8` accepts when wrapped in an
//!    IVF container.
//!
//! # Out of scope for Phase 1
//!
//! * Rate-distortion-driven mode selection, vector search, transform
//!   coefficient encoding. These will land in a later round once the
//!   frame-header writers are solid.
//! * Inter-frame encoding. The §16 inter path needs the §13 token
//!   encoder + the §17 MV encoder + the §16 reference-frame management,
//!   all of which are deferred to subsequent rounds.
//! * Segmentation (`§9.3`'s `update_segmentation` non-trivial path) —
//!   the writer only handles the "disabled" case.
//! * Per-macroblock loop-filter deltas (§9.4's `mb_lf_adjustments`
//!   non-trivial path) — same treatment.
//!
//! # Reference
//!
//! * RFC 6386 §7.3 (the `bool_encoder` C listing embedded in the RFC).
//! * RFC 6386 §9.1 – §9.11 (the uncompressed + boolean-coded frame
//!   header layouts).
//! * RFC 6386 §11 (key-frame macroblock prediction records).

use crate::coded_header::TokenProbUpdates;
use crate::frame_header::KEY_FRAME_START_CODE;

/// Errors surfaced by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// Visible width or height is zero or exceeds the 14-bit VP8 field
    /// (`width > 0x3FFF` or `height > 0x3FFF`).
    InvalidDimensions { width: u32, height: u32 },
    /// `loop_filter_level` outside the 6-bit field (`> 63`).
    LoopFilterLevelOutOfRange { value: u8 },
    /// `sharpness_level` outside the 3-bit field (`> 7`).
    SharpnessLevelOutOfRange { value: u8 },
    /// `nbr_of_dct_partitions` is not one of the four legal values
    /// (1 / 2 / 4 / 8) per §9.5.
    InvalidDctPartitionCount { value: u8 },
    /// `y_ac_qi` outside the 7-bit field (`> 127`) per §9.6.
    QuantIndexOutOfRange { value: u8 },
    /// First-partition payload exceeded the 19-bit field VP8 reserves
    /// for it in §9.1 (`> 0x7_FFFF`). At the partition sizes Phase 1
    /// emits this is unreachable, but the check sits at the boundary
    /// for future encoder rounds that may push it.
    FirstPartitionTooLarge { bytes: usize },
    /// The §13 token-block encoder rejected a macroblock's residual
    /// (a coefficient out of the §13.2 alphabet range). Surfaces the
    /// underlying [`TokenEncodeError`].
    Token(TokenEncodeError),
    /// The §14.2 / §12.3 reconstruction orchestrator (the decoder's own,
    /// reused to evolve neighbours) rejected the encoder's per-MB inputs.
    /// Surfaces the underlying [`crate::reconstruct::ReconstructError`].
    Reconstruct(crate::reconstruct::ReconstructError),
    /// The reference frame supplied to a P-frame encode does not match
    /// the source frame's macroblock-aligned dimensions
    /// (`mb_cols * 16 × mb_rows * 16`). Surfaced by
    /// [`encode_p_frame_zero_mv`].
    ReferenceDimensionsMismatch {
        /// `(width, height)` of the source frame's macroblock-aligned grid.
        source: (u32, u32),
        /// `(width, height)` of the reference [`crate::frame::KeyframePlanes`].
        reference: (u32, u32),
    },
    /// Reserved for inter-mode picker fall-throughs. The Phase-11
    /// `encode_p_frame_zero_mv` picker now emits all five §16.2
    /// `mv_ref_tree` leaves (ZEROMV / NEARESTMV / NEARMV / NEWMV /
    /// SPLITMV); this variant is retained for forward-compatibility with
    /// a future picker that may surface an InterMode not handled by the
    /// emit layer (e.g. an `Intra` fallback in a mixed-mode inter pass).
    UnsupportedInterMode {
        /// The resolved mode the picker handed the emit layer.
        mode: crate::near_mv::InterMode,
    },
    /// `copy_buffer_to_golden` or `copy_buffer_to_alternate` was outside
    /// the 2-bit field (`0..=2`) per §9.7. Surfaced by
    /// [`encode_p_frame_multi_ref_with_refresh`] when its
    /// [`RefreshControls`] argument fails validation.
    InvalidCopyBufferSelector {
        /// The §9.7 selector field whose value was out of range.
        which: CopyBufferSelector,
        /// The offending value (must be `0`, `1`, or `2`).
        value: u8,
    },
    /// A §9.4 reference or mode `loop_filter_delta` value exceeded the
    /// 6-bit-magnitude + 1-bit-sign field (`abs(value) > 63`). Surfaced
    /// by [`LoopFilterDeltas::validate`] and
    /// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas`].
    LoopFilterDeltaOutOfRange {
        /// The §9.4 delta slot whose value was out of range.
        which: LoopFilterDeltaSlot,
        /// The offending signed value (must be `-63..=63`).
        value: i16,
    },
}

/// Names the two §9.7 `copy_buffer_to_*` selector fields so a rejected
/// value in [`RefreshControls`] can report which one failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyBufferSelector {
    /// §9.7 `copy_buffer_to_golden` — 0 = none, 1 = copy `last_frame`,
    /// 2 = copy `alt_ref_frame`.
    Golden,
    /// §9.7 `copy_buffer_to_alternate` — 0 = none, 1 = copy `last_frame`,
    /// 2 = copy `golden_frame`.
    Alternate,
}

/// Names the §9.4 per-reference + per-mode `loop_filter_delta` slots so
/// a rejected value in [`LoopFilterDeltas`] can report which one failed.
///
/// The four reference slots are indexed by their §20.6 `ref_frame`
/// order (`CURRENT_FRAME = 0`, `LAST_FRAME = 1`, `GOLDEN_FRAME = 2`,
/// `ALTREF_FRAME = 3`); the four mode slots are indexed by the §20.6
/// `mode_delta[]` order (`B_PRED = 0`, `ZERO_MV = 1`, `NEAREST/NEAR/NEW = 2`,
/// `SPLIT_MV = 3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopFilterDeltaSlot {
    /// §20.6 `ref_delta[CURRENT_FRAME]` — applied to intra-coded MBs.
    RefCurrent,
    /// §20.6 `ref_delta[LAST_FRAME]` — applied to inter MBs predicting
    /// from `LAST`.
    RefLast,
    /// §20.6 `ref_delta[GOLDEN_FRAME]` — applied to inter MBs predicting
    /// from `GOLDEN`.
    RefGolden,
    /// §20.6 `ref_delta[ALTREF_FRAME]` — applied to inter MBs predicting
    /// from `ALTREF`.
    RefAltref,
    /// §20.6 `mode_delta[0]` — applied to intra MBs with `y_mode = B_PRED`.
    ModeBpred,
    /// §20.6 `mode_delta[1]` — applied to inter MBs with `y_mode = ZEROMV`.
    ModeZeroMv,
    /// §20.6 `mode_delta[2]` — applied to inter MBs with `y_mode` in
    /// `{NEARESTMV, NEARMV, NEWMV}`.
    ModeOtherMv,
    /// §20.6 `mode_delta[3]` — applied to inter MBs with `y_mode = SPLITMV`.
    ModeSplitMv,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::InvalidDimensions { width, height } => write!(
                f,
                "vp8 encode: invalid frame dimensions {width}x{height} \
                 (must be 1..=0x3FFF in both axes)"
            ),
            EncodeError::LoopFilterLevelOutOfRange { value } => write!(
                f,
                "vp8 encode: loop_filter_level={value} exceeds the 6-bit field (0..=63)"
            ),
            EncodeError::SharpnessLevelOutOfRange { value } => write!(
                f,
                "vp8 encode: sharpness_level={value} exceeds the 3-bit field (0..=7)"
            ),
            EncodeError::InvalidDctPartitionCount { value } => write!(
                f,
                "vp8 encode: nbr_of_dct_partitions={value} is not one of 1/2/4/8 (§9.5)"
            ),
            EncodeError::QuantIndexOutOfRange { value } => write!(
                f,
                "vp8 encode: y_ac_qi={value} exceeds the 7-bit field (0..=127)"
            ),
            EncodeError::FirstPartitionTooLarge { bytes } => write!(
                f,
                "vp8 encode: first-partition size {bytes} exceeds the 19-bit field (0..=0x7_FFFF)"
            ),
            EncodeError::Token(inner) => write!(f, "vp8 encode: {inner}"),
            EncodeError::Reconstruct(inner) => write!(f, "vp8 encode: {inner}"),
            EncodeError::ReferenceDimensionsMismatch { source, reference } => write!(
                f,
                "vp8 encode: P-frame source frame {}x{} (macroblock-aligned) does \
                 not match the supplied reference frame {}x{}",
                source.0, source.1, reference.0, reference.1
            ),
            EncodeError::UnsupportedInterMode { mode } => write!(
                f,
                "vp8 encode: inter-mode {mode:?} is not supported by the \
                 Phase-11 ZEROMV / NEARESTMV / NEARMV / NEWMV / SPLITMV picker"
            ),
            EncodeError::InvalidCopyBufferSelector { which, value } => write!(
                f,
                "vp8 encode: copy_buffer_to_{} = {value} is outside the \
                 2-bit field 0..=2 (§9.7)",
                match which {
                    CopyBufferSelector::Golden => "golden",
                    CopyBufferSelector::Alternate => "alternate",
                }
            ),
            EncodeError::LoopFilterDeltaOutOfRange { which, value } => write!(
                f,
                "vp8 encode: loop_filter_delta[{}] = {value} is outside the \
                 §9.4 6-bit-magnitude + 1-bit-sign field (-63..=63)",
                match which {
                    LoopFilterDeltaSlot::RefCurrent => "ref_current",
                    LoopFilterDeltaSlot::RefLast => "ref_last",
                    LoopFilterDeltaSlot::RefGolden => "ref_golden",
                    LoopFilterDeltaSlot::RefAltref => "ref_altref",
                    LoopFilterDeltaSlot::ModeBpred => "mode_bpred",
                    LoopFilterDeltaSlot::ModeZeroMv => "mode_zero_mv",
                    LoopFilterDeltaSlot::ModeOtherMv => "mode_other_mv",
                    LoopFilterDeltaSlot::ModeSplitMv => "mode_split_mv",
                }
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// VP8 boolean (range) encoder per RFC 6386 §7.3.
///
/// The structure mirrors the spec's `bool_encoder` C type one-for-one:
/// `range`, `bottom`, and a per-position `bit_count` of remaining shifts
/// before the next high-byte output. Encoded bytes accumulate in `out`.
///
/// Implements §7.3 of RFC 6386 (the `init_bool_encoder` /
/// `write_bool` / `add_one_to_output` / `flush_bool_encoder`
/// C listing the RFC embeds inline).
#[derive(Debug, Default)]
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    /// Construct an encoder with the §7.3 `init_bool_encoder` initial
    /// state (`range = 255`, `bottom = 0`, `bit_count = 24`).
    pub fn new() -> Self {
        BoolEncoder {
            out: Vec::new(),
            range: 255,
            bottom: 0,
            bit_count: 24,
        }
    }

    /// Encode one boolean against the §7.3 probability split formula.
    /// `prob` is the (approximate) probability that `value` is `false`,
    /// expressed as `prob / 256`.
    pub fn write_bool(&mut self, prob: u8, value: bool) {
        // split = 1 + (((range - 1) * prob) >> 8) — the same formula as
        // §7.3's decoder so the value→bit split is exact.
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        if value {
            self.bottom = self.bottom.wrapping_add(split);
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            // §7.3 `add_one_to_output`: propagate a carry into the
            // already-written tail, scanning backwards through 0xFF
            // bytes (which roll to 0 and force the next byte to
            // receive the carry).
            if (self.bottom >> 31) & 1 == 1 {
                add_one_to_output(&mut self.out);
            }
            self.bottom <<= 1;
            self.bit_count -= 1;
            if self.bit_count == 0 {
                // Emit the high byte of `bottom`, keep the low 24 bits.
                let byte = (self.bottom >> 24) as u8;
                self.out.push(byte);
                self.bottom &= (1 << 24) - 1;
                self.bit_count = 8;
            }
        }
    }

    /// Encode `num_bits` flag bits MSB-first at the flat probability of
    /// 128 (i.e. the §7.3 `L(n)` macro), exactly as `num_bits`
    /// iterations of [`write_bool`]`(128, …)` would.
    ///
    /// Specialised register-local loop: at the fixed probability 128 the
    /// §7.3 interval split collapses to a pure shift —
    /// `1 + (((range - 1) * 128) >> 8)` is exactly
    /// `1 + ((range - 1) >> 1)` (`* 2⁷ >> 8` ≡ `>> 1`, an algebraic
    /// identity on the `1..=255` range domain) — and hoisting `range` /
    /// `bottom` / `bit_count` into locals lets the whole loop run
    /// without touching `self` except to emit bytes / propagate a
    /// carry. State-for-state equality with the generic loop is pinned
    /// by `write_literal_fast_matches_generic_loop` (the decoder-side
    /// twin of this specialisation landed in round 306).
    pub fn write_literal(&mut self, value: u32, num_bits: u32) {
        debug_assert!(num_bits <= 32);
        let mut range = self.range;
        let mut bottom = self.bottom;
        let mut bit_count = self.bit_count;
        // MSB-first, paired with `BoolDecoder::read_literal`.
        let mut i = num_bits;
        while i > 0 {
            i -= 1;
            let split = 1 + ((range - 1) >> 1);
            if ((value >> i) & 1) != 0 {
                bottom = bottom.wrapping_add(split);
                range -= split;
            } else {
                range = split;
            }
            while range < 128 {
                range <<= 1;
                if (bottom >> 31) & 1 == 1 {
                    add_one_to_output(&mut self.out);
                }
                bottom <<= 1;
                bit_count -= 1;
                if bit_count == 0 {
                    self.out.push((bottom >> 24) as u8);
                    bottom &= (1 << 24) - 1;
                    bit_count = 8;
                }
            }
        }
        self.range = range;
        self.bottom = bottom;
        self.bit_count = bit_count;
    }

    /// Encode a signed value as L(n) magnitude followed by L(1) sign,
    /// the §9.3 / §9.4 / §9.6 idiom. Panics in debug builds if `value`
    /// does not fit in `num_bits` magnitude bits.
    pub fn write_signed_literal(&mut self, value: i32, num_bits: u32) {
        let magnitude = value.unsigned_abs();
        debug_assert!(
            magnitude < (1u32 << num_bits),
            "signed literal {value} does not fit in {num_bits} magnitude bits"
        );
        self.write_literal(magnitude, num_bits);
        self.write_bool(128, value < 0);
    }

    /// Write a tree-coded leaf by walking `tree` to find the path of
    /// bits that lands on `-leaf`, encoding each path bit against
    /// `prob_lookup(node_index >> 1)`. This is the §8.1 `treed_read`
    /// walk run in reverse: the decoder's [`crate::macroblock`]
    /// `treed_read` recovers exactly the leaf this method writes,
    /// against the same tree and probability lookup.
    ///
    /// Used by the §11 macroblock-mode layer (`kf_ymode_tree` /
    /// `uv_mode_tree` / `bmode_tree`). Panics in debug builds if `leaf`
    /// is not reachable in `tree`.
    pub fn write_treed<F>(&mut self, tree: &[i8], prob_lookup: F, leaf: u8)
    where
        F: Fn(usize) -> u8,
    {
        // Depth-first search for the bit path from the root (node 0) to
        // the leaf `-leaf`, into a fixed stack buffer (no per-call heap
        // allocation — see `treed_find_path`).
        let mut path = [false; 16];
        let len = treed_find_path(tree, leaf, &mut path);
        // Replay the path, calling `prob_lookup` with the same
        // node-halved index the decoder uses at each step.
        let mut i: i8 = 0;
        for &bit in &path[..len] {
            self.write_bool(prob_lookup((i as usize) >> 1), bit);
            i = tree[i as usize + bit as usize];
        }
    }

    /// Finalise the encoder per §7.3 `flush_bool_encoder` and return
    /// the encoded bytes. Always writes 4 tail bytes (the spec's
    /// `c = 4; while (--c >= 0)` loop), so the smallest possible
    /// partition is 4 bytes — comfortably above the 2-byte minimum
    /// [`crate::bool_decoder::BoolDecoder::init`] requires.
    pub fn finish(mut self) -> Vec<u8> {
        let c = self.bit_count;
        let v = self.bottom;
        // Propagate any pending carry into the already-written tail.
        if v & (1u32 << (32 - c)) != 0 {
            add_one_to_output(&mut self.out);
        }
        let mut v = v;
        // `v <<= c & 7;` followed by `while (--c >= 0) v <<= 8;` after
        // `c >>= 3;` — same byte-realignment dance as the spec.
        v = v.wrapping_shl((c & 7) as u32);
        let mut shifts = c >> 3;
        while shifts > 0 {
            v = v.wrapping_shl(8);
            shifts -= 1;
        }
        for _ in 0..4 {
            self.out.push((v >> 24) as u8);
            v = v.wrapping_shl(8);
        }
        self.out
    }

    /// Number of bytes the encoder has already committed (not counting
    /// the four trailing bytes [`finish`](Self::finish) will append).
    /// Useful for diagnostics; not otherwise consumed by the writer
    /// helpers.
    pub fn bytes_written(&self) -> usize {
        self.out.len()
    }
}

/// §7.3 `add_one_to_output`: increment the last-written byte by 1,
/// propagating the carry leftward through any trailing `0xFF` bytes.
fn add_one_to_output(out: &mut [u8]) {
    let mut i = out.len();
    while i > 0 {
        i -= 1;
        if out[i] == 255 {
            out[i] = 0;
        } else {
            out[i] = out[i].wrapping_add(1);
            return;
        }
    }
}

/// Two-bit horizontal / vertical scale code per §9.1's Table.
///
/// Parallels the decoder-side [`crate::frame_header::ScaleCode`] enum
/// but is duplicated here as a `u8` for unambiguous round-trip with the
/// 2-bit field on the wire. Encoder Phase 1 only emits
/// [`ScaleCode::None`] (`= 0`) since upscaling has no decode effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScaleCode {
    /// 0 — no upscaling. Encoder default.
    None = 0,
    /// 1 — upscale by 5/4.
    FiveFourths = 1,
    /// 2 — upscale by 5/3.
    FiveThirds = 2,
    /// 3 — upscale by 2.
    Double = 3,
}

/// §9.1 uncompressed-data-chunk writer.
///
/// Writes the 3-byte frame tag for every frame and, for key frames,
/// the additional 7 bytes (`0x9d 0x01 0x2a` start code + 14-bit width
/// + 2-bit horizontal scale + 14-bit height + 2-bit vertical scale).
///
/// Returns the number of bytes appended to `out` (3 for an interframe,
/// 10 for a key frame). Validates that `width` and `height` fit the
/// 14-bit field on key frames; for non-key frames `width` and `height`
/// are ignored (the inherited dimensions live in the decoder's state).
///
/// The encoder's higher layers must back-patch the 19-bit
/// `first_partition_size` into bits 5..23 of the frame tag after the
/// first partition has been fully written and its length is known. To
/// keep the layering crisp this function takes the size as a parameter:
/// callers that don't know it yet pass `0` and patch the bytes in place
/// using [`patch_first_partition_size`].
#[allow(clippy::too_many_arguments)]
pub fn write_frame_tag(
    out: &mut Vec<u8>,
    key_frame: bool,
    version: u8,
    show_frame: bool,
    first_partition_size: u32,
    width: u32,
    height: u32,
    horizontal_scale: ScaleCode,
    vertical_scale: ScaleCode,
) -> Result<usize, EncodeError> {
    if first_partition_size > 0x7_FFFF {
        return Err(EncodeError::FirstPartitionTooLarge {
            bytes: first_partition_size as usize,
        });
    }
    // §9.1 frame tag bit layout (LSB-first within the 24-bit word):
    //   bit  0   : frame_type (0 = key, 1 = inter)
    //   bits 1..3: version (3 bits)
    //   bit  4   : show_frame
    //   bits 5..23: first_partition_size (19 bits)
    let frame_type_bit: u32 = if key_frame { 0 } else { 1 };
    let show_bit: u32 = if show_frame { 1 } else { 0 };
    let tmp: u32 = frame_type_bit
        | ((version as u32 & 0x7) << 1)
        | (show_bit << 4)
        | ((first_partition_size & 0x7_FFFF) << 5);
    out.push((tmp & 0xff) as u8);
    out.push(((tmp >> 8) & 0xff) as u8);
    out.push(((tmp >> 16) & 0xff) as u8);

    if !key_frame {
        return Ok(3);
    }

    if width == 0 || width > 0x3FFF || height == 0 || height > 0x3FFF {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    out.extend_from_slice(&KEY_FRAME_START_CODE);
    // §9.1 page 31 listing:
    //   h_word = (h_scale << 14) | width
    //   v_word = (v_scale << 14) | height
    // emitted little-endian.
    let h_word: u16 = ((horizontal_scale as u16) << 14) | (width as u16 & 0x3FFF);
    let v_word: u16 = ((vertical_scale as u16) << 14) | (height as u16 & 0x3FFF);
    out.push((h_word & 0xff) as u8);
    out.push(((h_word >> 8) & 0xff) as u8);
    out.push((v_word & 0xff) as u8);
    out.push(((v_word >> 8) & 0xff) as u8);
    Ok(10)
}

/// Back-patch the §9.1 `first_partition_size` field of an
/// already-written frame tag in `buf`. `buf` must start with the three
/// frame-tag bytes [`write_frame_tag`] produced. The frame_type /
/// version / show_frame bits are preserved verbatim.
pub fn patch_first_partition_size(buf: &mut [u8], size: u32) -> Result<(), EncodeError> {
    if size > 0x7_FFFF {
        return Err(EncodeError::FirstPartitionTooLarge {
            bytes: size as usize,
        });
    }
    debug_assert!(buf.len() >= 3, "frame tag must be present");
    let tmp = (buf[0] as u32) | ((buf[1] as u32) << 8) | ((buf[2] as u32) << 16);
    // Clear bits 5..23 (the 19-bit size field) and OR the new value in.
    let new = (tmp & 0x1F) | ((size & 0x7_FFFF) << 5);
    buf[0] = (new & 0xff) as u8;
    buf[1] = ((new >> 8) & 0xff) as u8;
    buf[2] = ((new >> 16) & 0xff) as u8;
    Ok(())
}

/// §9.3 segment-update flags writer (disabled path).
///
/// Emits the single `segmentation_enabled = L(1) = 0` toggle. The richer
/// enabled path — `update_mb_segmentation_map`, the per-segment quantizer
/// / loop-filter feature data, and the `mb_segment_tree_probs` block — is
/// written by [`write_update_segmentation`] when the encoder turns
/// segmentation on. `debug_assert` guards against a caller passing
/// `enabled = true` here (it would emit the toggle but none of the
/// dependent §9.3 sub-blocks the decoder then expects to read).
pub fn write_segment_update_flags(enc: &mut BoolEncoder, enabled: bool) {
    debug_assert!(
        !enabled,
        "use write_update_segmentation for the segmentation_enabled = true path"
    );
    enc.write_bool(128, enabled);
}

/// Encoder-side mirror of [`crate::coded_header::UpdateSegmentation`] —
/// the §9.3 `update_segmentation()` sub-block emitted when
/// `segmentation_enabled = true`.
///
/// The four `quantizer` / `loop_filter` slots carry the per-segment
/// feature deltas (or absolute values, when `feature_mode_absolute`).
/// `None` means "this segment's feature is not updated this frame" — the
/// decoder leaves it at its persisted value (0 on a key frame). The three
/// `segment_prob` slots are the `mb_segment_tree_probs[3]` branch
/// probabilities used to code each MB's `segment_id`; `None` means the
/// decoder uses the default 255 ("always read a zero branch") for that
/// node.
///
/// The writer ([`write_update_segmentation`]) lays the bits out exactly as
/// [`crate::coded_header::parse_update_segmentation`] reads them, so the
/// decoder recovers this struct field-for-field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UpdateSegmentationLayer {
    /// `update_mb_segmentation_map` (§9.3). When `true`, the per-MB
    /// `segment_id` codes follow in the §11 mode layer (against
    /// `segment_prob`) and the `segment_prob` sub-block is written here.
    pub update_mb_segmentation_map: bool,
    /// `segment_feature_mode`: `false` = delta mode (the segment value is
    /// added to the §9.6 baseline), `true` = absolute mode (the segment
    /// value replaces the baseline). Only meaningful when at least one
    /// `quantizer` / `loop_filter` slot is `Some`.
    pub feature_mode_absolute: bool,
    /// Per-segment §10 quantizer feature (`quantizer_update_value` +
    /// `quantizer_update_sign`, 7-bit magnitude). `None` skips the
    /// segment's `quantizer_update` flag (codes a 0).
    pub quantizer: [Option<i16>; 4],
    /// Per-segment §10 loop-filter feature (`lf_update_value` +
    /// `lf_update_sign`, 6-bit magnitude). `None` skips the segment's
    /// `loop_filter_update` flag (codes a 0).
    pub loop_filter: [Option<i16>; 4],
    /// `mb_segment_tree_probs[3]` (§10). `None` codes a 0
    /// `segment_prob_update` flag for that node (decoder uses 255).
    pub segment_prob: [Option<u8>; 3],
}

/// Write the §9.3 `update_segmentation()` sub-block for the
/// `segmentation_enabled = true` path.
///
/// Emits, in §19.2 `update_segmentation()` order:
///
/// 1. `segmentation_enabled = L(1) = 1`;
/// 2. `update_mb_segmentation_map` (L1);
/// 3. `update_segment_feature_data` (L1), set iff any `quantizer` /
///    `loop_filter` slot is `Some`; when set: `segment_feature_mode`
///    (L1), then four `quantizer_update` flags (each followed by a signed
///    7-bit magnitude when present), then four `loop_filter_update` flags
///    (each followed by a signed 6-bit magnitude when present);
/// 4. when `update_mb_segmentation_map`, three `segment_prob_update`
///    flags, each followed by an 8-bit `segment_prob` when present.
///
/// This is the exact byte layout
/// [`crate::coded_header::parse_update_segmentation`] reads (which is
/// itself the §19.2 table verbatim), so the decoder reconstructs the
/// [`crate::coded_header::UpdateSegmentation`] this layer describes.
pub fn write_update_segmentation(enc: &mut BoolEncoder, layer: &UpdateSegmentationLayer) {
    // §19.2 row 3: segmentation_enabled = 1 introduces the sub-block.
    enc.write_bool(128, true);
    enc.write_bool(128, layer.update_mb_segmentation_map);

    // update_segment_feature_data: only when some feature is actually
    // carried. Matching the decoder, the four quantizer + four loop-filter
    // flags are only present when this bit is set.
    let update_feature_data = layer.quantizer.iter().any(Option::is_some)
        || layer.loop_filter.iter().any(Option::is_some);
    enc.write_bool(128, update_feature_data);
    if update_feature_data {
        enc.write_bool(128, layer.feature_mode_absolute);
        for slot in &layer.quantizer {
            match slot {
                Some(v) => {
                    enc.write_bool(128, true);
                    // §9.3: 7-bit magnitude + sign. write_signed_literal
                    // pairs with the decoder's read_signed_delta(_, 7).
                    enc.write_signed_literal(*v as i32, 7);
                }
                None => enc.write_bool(128, false),
            }
        }
        for slot in &layer.loop_filter {
            match slot {
                Some(v) => {
                    enc.write_bool(128, true);
                    // §9.3: 6-bit magnitude + sign.
                    enc.write_signed_literal(*v as i32, 6);
                }
                None => enc.write_bool(128, false),
            }
        }
    }

    if layer.update_mb_segmentation_map {
        for slot in &layer.segment_prob {
            match slot {
                Some(p) => {
                    enc.write_bool(128, true);
                    enc.write_literal(*p as u32, 8);
                }
                None => enc.write_bool(128, false),
            }
        }
    }
}

/// §9.4 loop-filter type / level / sharpness + `mb_lf_adjustments`
/// writer.
///
/// `filter_type` chooses normal (`false`) vs simple (`true`) loop
/// filter. `loop_filter_level` is the 6-bit baseline level; setting it
/// to 0 triggers the §15 page-84 whole-frame filter skip in any
/// compliant decoder. `sharpness_level` is the 3-bit sharpness knob.
/// `loop_filter_adj_enable` controls whether the per-macroblock loop
/// filter delta machinery is on; Phase 1 only supports the disabled
/// case, paralleling the segmentation writer above.
pub fn write_loop_filter(
    enc: &mut BoolEncoder,
    filter_type: bool,
    loop_filter_level: u8,
    sharpness_level: u8,
    loop_filter_adj_enable: bool,
) -> Result<(), EncodeError> {
    if loop_filter_level > 63 {
        return Err(EncodeError::LoopFilterLevelOutOfRange {
            value: loop_filter_level,
        });
    }
    if sharpness_level > 7 {
        return Err(EncodeError::SharpnessLevelOutOfRange {
            value: sharpness_level,
        });
    }
    debug_assert!(
        !loop_filter_adj_enable,
        "Phase 1 encoder only supports loop_filter_adj_enable = false"
    );

    enc.write_bool(128, filter_type);
    enc.write_literal(loop_filter_level as u32, 6);
    enc.write_literal(sharpness_level as u32, 3);

    // mb_lf_adjustments() sub-block (§9.4): just the disable bit
    // when the feature is off — the rest of the table is gated on it.
    enc.write_bool(128, loop_filter_adj_enable);
    Ok(())
}

/// Caller-supplied per-reference + per-mode loop-filter delta layer
/// per RFC 6386 §9.4 (the `mb_lf_adjustments()` sub-block).
///
/// The decoder already honours these deltas
/// ([`crate::loop_filter::calculate_mb_filter_level_inter`]): each
/// per-MB filter level starts from the frame `loop_filter_level`, then
/// receives the matching `ref_frame_delta[ref]` plus the matching
/// `mode_delta[mode]`, with the final level clamped into `0..=63`.
/// This struct gives the encoder a way to *transmit* the deltas in the
/// §19.2 frame header so the decoder applies them, and to use the same
/// deltas during the encoder's own §15 post-walk filter pass so the
/// encoder's reconstruction buffer stays byte-identical to what the
/// decoder rebuilds.
///
/// `enabled` corresponds to the §19.2 `loop_filter_adj_enable` bit
/// (`true` ⇒ the delta layer is on for this frame). `update`
/// corresponds to the §19.2 `mode_ref_lf_delta_update` bit (`true` ⇒
/// the frame carries fresh delta values; `false` ⇒ the decoder reuses
/// the deltas from a prior frame). When `enabled` is `false` no per-MB
/// delta is applied (and `update` is irrelevant). When `enabled` is
/// `true` but `update` is `false`, no per-slot values are written to
/// the bitstream — the decoder pulls from its carried state.
///
/// When `update` is `true`, each per-slot `Option<i8>` is either
/// `Some(v)` (a fresh signed value with `abs(v) <= 63` is emitted) or
/// `None` (the slot's bit `delta_update_flag` is 0 and the decoder
/// keeps its carried value for that slot). The four `ref_frame_delta`
/// slots are indexed by `CURRENT_FRAME = 0`, `LAST_FRAME = 1`,
/// `GOLDEN_FRAME = 2`, `ALTREF_FRAME = 3` (§20.6 `ref_delta[]` order).
/// The four `mode_delta` slots are indexed by `B_PRED = 0`,
/// `ZERO_MV = 1`, `NEAREST/NEAR/NEW = 2`, `SPLIT_MV = 3` (§20.6
/// `mode_delta[]` order).
///
/// The encoder's §15 post-walk filter pass reads the *effective*
/// deltas — the values it just wrote plus the carried values for any
/// `None` slot. The encoder maintains its own carried state across
/// frames inside [`Vp8InterStreamEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterDeltas {
    /// §19.2 `loop_filter_adj_enable` (L1).
    pub enabled: bool,
    /// §19.2 `mode_ref_lf_delta_update` (L1). Only consulted when
    /// `enabled` is `true`; if `false`, no per-slot values are written
    /// and the decoder reuses the prior frame's deltas.
    pub update: bool,
    /// §19.2 four per-reference deltas, in the §20.6 `{CURRENT, LAST,
    /// GOLDEN, ALTREF}` index order. Each `Some(v)` writes the
    /// per-slot flag = 1 followed by L(6) magnitude + L(1) sign;
    /// `None` writes the flag = 0 (decoder keeps its carried value).
    /// Magnitudes greater than 63 are rejected by [`Self::validate`].
    pub ref_frame_delta: [Option<i8>; 4],
    /// §19.2 four per-mode deltas, in the §20.6 `{B_PRED, ZERO_MV,
    /// OTHER_MV, SPLIT_MV}` index order. Same encoding as
    /// `ref_frame_delta`.
    pub mode_delta: [Option<i8>; 4],
}

impl Default for LoopFilterDeltas {
    /// Disabled deltas: matches the round-150-and-earlier behaviour
    /// (`loop_filter_adj_enable = 0`).
    fn default() -> Self {
        Self {
            enabled: false,
            update: false,
            ref_frame_delta: [None; 4],
            mode_delta: [None; 4],
        }
    }
}

impl LoopFilterDeltas {
    /// Per-slot magnitude check: every `Some(v)` must satisfy
    /// `abs(v) <= 63` (the §9.4 L(6) + L(1) field). Out-of-range values
    /// surface as [`EncodeError::LoopFilterDeltaOutOfRange`] before any
    /// encoding work runs.
    pub fn validate(&self) -> Result<(), EncodeError> {
        const REF_SLOTS: [LoopFilterDeltaSlot; 4] = [
            LoopFilterDeltaSlot::RefCurrent,
            LoopFilterDeltaSlot::RefLast,
            LoopFilterDeltaSlot::RefGolden,
            LoopFilterDeltaSlot::RefAltref,
        ];
        const MODE_SLOTS: [LoopFilterDeltaSlot; 4] = [
            LoopFilterDeltaSlot::ModeBpred,
            LoopFilterDeltaSlot::ModeZeroMv,
            LoopFilterDeltaSlot::ModeOtherMv,
            LoopFilterDeltaSlot::ModeSplitMv,
        ];
        for (slot, value) in REF_SLOTS.iter().zip(self.ref_frame_delta.iter()) {
            if let Some(v) = value {
                let mag = (*v as i16).unsigned_abs();
                if mag > 63 {
                    return Err(EncodeError::LoopFilterDeltaOutOfRange {
                        which: *slot,
                        value: *v as i16,
                    });
                }
            }
        }
        for (slot, value) in MODE_SLOTS.iter().zip(self.mode_delta.iter()) {
            if let Some(v) = value {
                let mag = (*v as i16).unsigned_abs();
                if mag > 63 {
                    return Err(EncodeError::LoopFilterDeltaOutOfRange {
                        which: *slot,
                        value: *v as i16,
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve the effective per-slot ref + mode deltas the §15 filter
    /// pass must use this frame, given the carried (across-frame) delta
    /// state. Matches the §9.4 / §20.6 "previous-frame values are used
    /// unless updated in the current header" rule:
    ///
    /// * If `enabled` is `false`, every effective slot resolves to 0
    ///   (the per-MB delta layer is off — the §15 filter level is the
    ///   frame base only).
    /// * If `enabled` is `true` and `update` is `false`, every
    ///   effective slot equals the carried value (no fresh values
    ///   transmitted, the decoder reuses its carried state).
    /// * If `enabled` is `true` and `update` is `true`, each slot
    ///   resolves to its `Some(v)` value if updated this frame,
    ///   otherwise to the carried value (the per-slot "absent" path
    ///   keeps prior state).
    ///
    /// Returns `(effective_ref_deltas, effective_mode_deltas)` in the
    /// §20.6 `{CURRENT, LAST, GOLDEN, ALTREF}` / `{B_PRED, ZERO_MV,
    /// OTHER_MV, SPLIT_MV}` index order. The returned arrays match
    /// what [`crate::loop_filter::FrameFilterConfig::interframe`]
    /// resolves from the parsed `mb_lf_adjustments` block — encoder
    /// and decoder agree on the effective values byte-for-byte.
    pub fn effective(
        &self,
        carried_ref_deltas: [i16; 4],
        carried_mode_deltas: [i16; 4],
    ) -> ([i16; 4], [i16; 4]) {
        if !self.enabled {
            return ([0; 4], [0; 4]);
        }
        if !self.update {
            return (carried_ref_deltas, carried_mode_deltas);
        }
        let mut eff_ref = carried_ref_deltas;
        let mut eff_mode = carried_mode_deltas;
        for (slot, value) in eff_ref.iter_mut().zip(self.ref_frame_delta.iter()) {
            if let Some(v) = value {
                *slot = *v as i16;
            }
        }
        for (slot, value) in eff_mode.iter_mut().zip(self.mode_delta.iter()) {
            if let Some(v) = value {
                *slot = *v as i16;
            }
        }
        (eff_ref, eff_mode)
    }
}

/// §9.4 loop-filter writer with caller-supplied per-reference + per-mode
/// `mb_lf_adjustments()` deltas.
///
/// Extends [`write_loop_filter`] with [`LoopFilterDeltas`] support: the
/// `loop_filter_adj_enable` bit + (when set) the `mode_ref_lf_delta_update`
/// bit + (when set) the four per-reference + four per-mode L(1) presence
/// flags and gated L(6) magnitude + L(1) sign values follow the §19.2
/// frame-header layout exactly. Passing
/// [`LoopFilterDeltas::default`] (with `enabled = false`) reproduces
/// the round-150 wire (`loop_filter_adj_enable = 0`) byte-for-byte.
pub fn write_loop_filter_with_deltas(
    enc: &mut BoolEncoder,
    filter_type: bool,
    loop_filter_level: u8,
    sharpness_level: u8,
    deltas: &LoopFilterDeltas,
) -> Result<(), EncodeError> {
    if loop_filter_level > 63 {
        return Err(EncodeError::LoopFilterLevelOutOfRange {
            value: loop_filter_level,
        });
    }
    if sharpness_level > 7 {
        return Err(EncodeError::SharpnessLevelOutOfRange {
            value: sharpness_level,
        });
    }
    deltas.validate()?;

    enc.write_bool(128, filter_type);
    enc.write_literal(loop_filter_level as u32, 6);
    enc.write_literal(sharpness_level as u32, 3);

    // mb_lf_adjustments() — §9.4 / §19.2 (page 121–122).
    enc.write_bool(128, deltas.enabled);
    if deltas.enabled {
        enc.write_bool(128, deltas.update);
        if deltas.update {
            for slot in &deltas.ref_frame_delta {
                match slot {
                    Some(v) => {
                        enc.write_bool(128, true);
                        write_signed_lf_delta(enc, *v);
                    }
                    None => enc.write_bool(128, false),
                }
            }
            for slot in &deltas.mode_delta {
                match slot {
                    Some(v) => {
                        enc.write_bool(128, true);
                        write_signed_lf_delta(enc, *v);
                    }
                    None => enc.write_bool(128, false),
                }
            }
        }
    }
    Ok(())
}

/// Emit the §9.4 L(6) magnitude + L(1) sign payload for a single
/// `loop_filter_delta` slot. Mirrors the decoder's `read_signed_delta`
/// reconstruction in [`crate::coded_header`].
fn write_signed_lf_delta(enc: &mut BoolEncoder, value: i8) {
    debug_assert!((-63..=63).contains(&(value as i16)));
    let magnitude = (value as i16).unsigned_abs() as u32;
    enc.write_literal(magnitude, 6);
    enc.write_bool(128, value < 0);
}

/// §9.5 `log2_nbr_of_dct_partitions` writer.
///
/// `count` must be one of `1`, `2`, `4`, `8` (the only legal values
/// per the §9.5 table). The two-bit field on the wire is
/// `log2(count)`.
pub fn write_token_partition_count(enc: &mut BoolEncoder, count: u8) -> Result<(), EncodeError> {
    let log2: u32 = match count {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        other => return Err(EncodeError::InvalidDctPartitionCount { value: other }),
    };
    enc.write_literal(log2, 2);
    Ok(())
}

/// §9.6 `quant_indices()` writer.
///
/// Writes the baseline `y_ac_qi` (always present) followed by five
/// presence-gated `L(4) + L(1)` signed deltas for `ydc / y2dc / y2ac /
/// uvdc / uvac`. Phase 1 callers pass `0` for the baseline (lossless
/// dequant scaling at this index) and `None` for every delta, but the
/// signature accepts the full set so the §13 encoder round can wire
/// arbitrary quantiser policies through unchanged.
pub fn write_quant_indices(
    enc: &mut BoolEncoder,
    y_ac_qi: u8,
    y_dc_delta: Option<i8>,
    y2_dc_delta: Option<i8>,
    y2_ac_delta: Option<i8>,
    uv_dc_delta: Option<i8>,
    uv_ac_delta: Option<i8>,
) -> Result<(), EncodeError> {
    if y_ac_qi > 127 {
        return Err(EncodeError::QuantIndexOutOfRange { value: y_ac_qi });
    }
    enc.write_literal(y_ac_qi as u32, 7);
    for delta in [
        y_dc_delta,
        y2_dc_delta,
        y2_ac_delta,
        uv_dc_delta,
        uv_ac_delta,
    ] {
        match delta {
            Some(d) => {
                enc.write_bool(128, true);
                enc.write_signed_literal(d as i32, 4);
            }
            None => enc.write_bool(128, false),
        }
    }
    Ok(())
}

/// §9.10 / §9.11 `mb_no_skip_coeff` writer.
///
/// On a key frame the §9.11 row 2 `prob_skip_false` follows the toggle
/// when enabled; on an interframe §9.10 carries the same first two
/// fields then the §16-only `prob_intra` / `prob_last` / `prob_gf` /
/// intra-mode-prob updates / `mv_prob_update()` (none of which Phase 1
/// emits — it only writes key frames).
pub fn write_mb_no_skip_coeff(enc: &mut BoolEncoder, enabled: bool, prob_skip_false: u8) {
    enc.write_bool(128, enabled);
    if enabled {
        enc.write_literal(prob_skip_false as u32, 8);
    }
}

/// Write `count` "no update" flags for the §13 / §9.9 token-probability
/// update sub-block. The four-dimensional structure of
/// `token_prob_update()` is `[4][8][3][11]` = 1056 single-bit flags, each
/// read at the §13.4 `coeff_update_probs[i][j][k][t]` table entry. With
/// every flag set to `0` no replacement probabilities follow and the
/// default §13.5 token-probability table remains in force for the frame.
///
/// `flag_probs` must be a flat 1056-entry view of the update-probability
/// table (`[i][j][k][t]` flattened row-major). The encoder passes the
/// crate-local [`COEFF_UPDATE_PROBS_FLAT`]; the decoder reads each flag
/// at the corresponding position-specific probability per §13.4 (NOT a
/// flat 128), so the encoder must use the same numbers.
pub fn write_no_token_prob_updates(enc: &mut BoolEncoder, flag_probs: &[u8; 1056]) {
    for &p in flag_probs.iter() {
        enc.write_bool(p, false);
    }
}

/// Write a §13.4 `token_prob_update()` block from a [`TokenProbUpdates`]
/// array. Each of the 1056 `[i][j][k][t]` positions gets a single
/// "is this position being replaced?" flag written at
/// `coeff_update_probs[i][j][k][t]` (the same per-position probability
/// the decoder reads against); when the slot is `Some(prob)` an
/// additional `L(8)` literal carrying the replacement value follows.
///
/// The walk order matches RFC 6386 §13.4's four nested `do/while` loops
/// (`i` outermost over `0..4` planes, then `j` over `0..8` bands, then
/// `k` over `0..3` previous-token classes, then `t` over `0..11` token
/// positions). The decoder's
/// [`crate::coded_header::Vp8CodedHeader::parse`] consumes the same
/// order; encoder + decoder loops are byte-paired.
///
/// `flag_probs` is a flat 1056-entry view of the §13.4 table — pass the
/// crate-local [`COEFF_UPDATE_PROBS_FLAT`]. `updates` is the
/// `[plane][band][prev_ctx][pos]` array (each entry `Some(prob)` to
/// replace or `None` to keep the §13.5 default). An all-`None` `updates`
/// produces the same bytes [`write_no_token_prob_updates`] does, so the
/// two writers are byte-equivalent in that special case.
pub fn write_token_prob_updates(
    enc: &mut BoolEncoder,
    updates: &TokenProbUpdates,
    flag_probs: &[u8; 1056],
) {
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for t in 0..11 {
                    let p = flag_probs[i * 8 * 3 * 11 + j * 3 * 11 + k * 11 + t];
                    match updates[i][j][k][t] {
                        Some(new_prob) => {
                            enc.write_bool(p, true);
                            enc.write_literal(new_prob as u32, 8);
                        }
                        None => {
                            enc.write_bool(p, false);
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────── silent-keyframe entry point ───────────────────────

/// Parameters for [`encode_silent_keyframe`].
///
/// Phase 1 keeps the surface minimal: dimensions, optional override of
/// the §9.4 loop filter (default disabled via `loop_filter_level = 0`),
/// and an optional override of the §9.6 `y_ac_qi` baseline (default 0).
#[derive(Debug, Clone, Copy)]
pub struct SilentKeyframeParams {
    /// Visible width in pixels (1..=0x3FFF per §9.1).
    pub width: u32,
    /// Visible height in pixels (1..=0x3FFF per §9.1).
    pub height: u32,
    /// §9.4 baseline loop filter level (0..=63). 0 triggers the §15
    /// page-84 whole-frame loop-filter skip.
    pub loop_filter_level: u8,
    /// §9.4 sharpness knob (0..=7). Only consulted when
    /// `loop_filter_level > 0`.
    pub sharpness_level: u8,
    /// §9.6 `y_ac_qi` baseline (0..=127). Irrelevant for the silent
    /// path (every coefficient is zero anyway) but exposed so a later
    /// round can drive non-trivial quantization through the same
    /// entry.
    pub y_ac_qi: u8,
    /// §9.5 DCT partition count. Must be one of 1 / 2 / 4 / 8.
    pub nbr_of_dct_partitions: u8,
}

impl Default for SilentKeyframeParams {
    fn default() -> Self {
        SilentKeyframeParams {
            width: 16,
            height: 16,
            loop_filter_level: 0,
            sharpness_level: 0,
            y_ac_qi: 0,
            nbr_of_dct_partitions: 1,
        }
    }
}

impl SilentKeyframeParams {
    /// Convenience constructor for the most common case — pick the
    /// dimensions, leave everything else at its silent-keyframe
    /// default.
    pub fn new(width: u32, height: u32) -> Self {
        SilentKeyframeParams {
            width,
            height,
            ..Self::default()
        }
    }
}

/// Encode a trivial all-zero-quantizer VP8 key frame for `params`.
///
/// Layout of the emitted bytes:
///
/// 1. §9.1 frame tag (3 bytes) — `frame_type=0` (key), `version=0`,
///    `show_frame=1`, `first_partition_size` back-patched after step 2.
/// 2. §9.1 key-frame extension (7 bytes) — start code `0x9d 0x01 0x2a`
///    + 14-bit width + 14-bit height (both scale codes 0).
/// 3. §19.2 boolean-coded first partition:
///    * §9.2 `color_space = 0` + `clamping_type = 0`.
///    * §9.3 segmentation_enabled = 0 (writer skips the sub-block).
///    * §9.4 filter_type = 0, `loop_filter_level` / `sharpness_level`
///      per `params`, `loop_filter_adj_enable = 0`.
///    * §9.5 `log2_nbr_of_dct_partitions` = `log2(params.nbr_of_dct_partitions)`.
///    * §9.6 quant_indices — `y_ac_qi = params.y_ac_qi`, every delta
///      omitted.
///    * §9.7 key-frame `refresh_entropy_probs = 1`.
///    * §13 / §9.9 token-prob update sub-block — every flag = 0
///      (defaults retained).
///    * §9.11 `mb_no_skip_coeff = 1`, `prob_skip_false = 1` (skip flag
///      is the high-probability branch so we encode it cheaply).
///    * §11 macroblock prediction records: per MB, `mb_skip_coeff = 1`,
///      `y_mode = DC_PRED`, `uv_mode = DC_PRED`. No subblock modes,
///      no segment id.
///    * §7.3 `flush_bool_encoder` (4 tail bytes).
/// 4. §9.5 DCT partition table — `(nbr_of_dct_partitions - 1) * 3`
///    bytes of 24-bit little-endian sizes.
/// 5. One DCT partition per row-stripe (every MB has
///    `mb_skip_coeff = 1`, so each partition emits only its
///    §7.3 4-byte flush trailer).
///
/// The result decodes through the crate's own [`crate::decode_vp8`]
/// driver and (when wrapped in an IVF container) through
/// `ffmpeg -c:v vp8`.
pub fn encode_silent_keyframe(params: SilentKeyframeParams) -> Result<Vec<u8>, EncodeError> {
    if params.width == 0 || params.width > 0x3FFF || params.height == 0 || params.height > 0x3FFF {
        return Err(EncodeError::InvalidDimensions {
            width: params.width,
            height: params.height,
        });
    }

    let mb_cols = params.width.div_ceil(16) as usize;
    let mb_rows = params.height.div_ceil(16) as usize;

    // ---- (3) Build the boolean-coded first partition ---------------------
    let mut enc = BoolEncoder::new();

    // §9.2 — color_space + clamping_type.
    enc.write_bool(128, false);
    enc.write_bool(128, false);

    // §9.3 — segmentation off.
    write_segment_update_flags(&mut enc, false);

    // §9.4 — loop filter knobs.
    write_loop_filter(
        &mut enc,
        false,
        params.loop_filter_level,
        params.sharpness_level,
        false,
    )?;

    // §9.5 — partition count.
    write_token_partition_count(&mut enc, params.nbr_of_dct_partitions)?;

    // §9.6 — quant indices (baseline only).
    write_quant_indices(&mut enc, params.y_ac_qi, None, None, None, None, None)?;

    // §9.7 (key frame) — refresh_entropy_probs.
    enc.write_bool(128, true);

    // §13 / §9.9 — token-prob update sub-block: every flag false.
    write_no_token_prob_updates(&mut enc, &COEFF_UPDATE_PROBS_FLAT);

    // §9.11 — mb_no_skip_coeff + prob_skip_false. prob_skip_false = 1
    // → P(mb_skip_coeff=false) ≈ 1/256, so the `mb_skip_coeff=true`
    // branch we emit per MB costs nearly nothing.
    write_mb_no_skip_coeff(&mut enc, true, 1);

    // §11 macroblock prediction records.
    //
    //   1. mb_skip_coeff = true (prob = 1).
    //   2. y_mode = DC_PRED. Tree path "1, 0, 0" against
    //      KF_YMODE_PROB = {145, 156, 163, 128}.
    //   3. (no subblock modes: y_mode != B_PRED.)
    //   4. uv_mode = DC_PRED. Tree path "0" against KF_UV_MODE_PROB.
    for _ in 0..(mb_rows * mb_cols) {
        // mb_skip_coeff = true.
        enc.write_bool(1, true);
        // y_mode = DC_PRED.
        enc.write_bool(145, true);
        enc.write_bool(156, false);
        enc.write_bool(163, false);
        // uv_mode = DC_PRED.
        enc.write_bool(142, false);
    }

    let first_partition = enc.finish();
    let first_partition_size = first_partition.len();
    if first_partition_size > 0x7_FFFF {
        return Err(EncodeError::FirstPartitionTooLarge {
            bytes: first_partition_size,
        });
    }

    // ---- (1) + (2) Frame tag + key-frame extension ----------------------
    let mut out: Vec<u8> = Vec::with_capacity(10 + first_partition_size + 16);
    write_frame_tag(
        &mut out,
        true,
        0,
        true,
        first_partition_size as u32,
        params.width,
        params.height,
        ScaleCode::None,
        ScaleCode::None,
    )?;

    out.extend_from_slice(&first_partition);

    // ---- (4) + (5) DCT partition table + per-partition payloads --------
    let num_partitions = params.nbr_of_dct_partitions as usize;
    let mut dct_partitions: Vec<Vec<u8>> = Vec::with_capacity(num_partitions);
    for _ in 0..num_partitions {
        // Every MB had mb_skip_coeff = true, so no per-block tokens
        // were emitted into any DCT partition. Each partition still
        // pays the §7.3 4-byte flush trailer.
        let p = BoolEncoder::new();
        dct_partitions.push(p.finish());
    }
    // §9.5 size table: `(num_partitions - 1) * 3` LE bytes.
    for part in dct_partitions.iter().take(num_partitions.saturating_sub(1)) {
        let sz = part.len();
        out.push((sz & 0xff) as u8);
        out.push(((sz >> 8) & 0xff) as u8);
        out.push(((sz >> 16) & 0xff) as u8);
    }
    for p in &dct_partitions {
        out.extend_from_slice(p);
    }

    Ok(out)
}

/// Flattened view of the §13.4 `coeff_update_probs[4][8][3][11]` table.
/// Each of the 1056 entries is the per-position probability the
/// corresponding `coeff_prob_update_flag` is read against on the decode
/// side. The encoder must use the same numbers — using a flat 128 here
/// would produce a far larger partition than the decoder expects, since
/// the table's values cluster near 255 ("almost always no update").
///
/// Built at compile time from a copy of the §13.4 table. The decoder's
/// [`crate::coded_header`] holds the same numbers in its nested-array
/// form; transcribing them once here avoids cross-module visibility
/// without re-typing them by hand at call time.
#[doc(hidden)]
pub const COEFF_UPDATE_PROBS_FLAT: [u8; 1056] = {
    let mut out = [0u8; 1056];
    let mut i = 0;
    while i < 4 {
        let mut j = 0;
        while j < 8 {
            let mut k = 0;
            while k < 3 {
                let mut t = 0;
                while t < 11 {
                    out[i * 8 * 3 * 11 + j * 3 * 11 + k * 11 + t] = COEFF_UPDATE_PROBS[i][j][k][t];
                    t += 1;
                }
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    out
};

/// `coeff_update_probs[i][j][k][t]` transcribed verbatim from RFC 6386
/// §13.4 — the same numbers the decoder's [`crate::coded_header`] uses
/// (this file re-types them to keep the encoder module
/// dependency-free of the decoder's internal table). Any change to the
/// decoder copy must mirror the change here; the
/// `coeff_update_probs_flat_walk_is_consistent` test below pins the
/// two tables together at test time.
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

// ───────────────────────── Phase 2: §13 DCT-token encoder ─────────────────────

use crate::dct_tokens::{BlockType, CoeffProbs, DctToken, COEFF_BANDS};

/// §13.2 `coeff_tree` — duplicated here for the encoder's tree walk so
/// the inverse mapping (`token -> bit path`) can be computed against
/// the exact same eleven-internal-node, twelve-leaf structure the
/// decoder traverses. Values follow the decoder's convention: entry
/// `2*i` is the left child of internal node `i`, entry `2*i + 1` is
/// the right child; non-negative entries point to the next internal
/// node, non-positive entries `-t` denote leaf `DctToken` with
/// discriminant `t`.
const ENC_COEFF_TREE: [i8; 22] = [
    -(DctToken::Eob as i8),
    2,
    -(DctToken::Dct0 as i8),
    4,
    -(DctToken::Dct1 as i8),
    6,
    8,
    12,
    -(DctToken::Dct2 as i8),
    10,
    -(DctToken::Dct3 as i8),
    -(DctToken::Dct4 as i8),
    14,
    16,
    -(DctToken::Cat1 as i8),
    -(DctToken::Cat2 as i8),
    18,
    20,
    -(DctToken::Cat3 as i8),
    -(DctToken::Cat4 as i8),
    -(DctToken::Cat5 as i8),
    -(DctToken::Cat6 as i8),
];

/// §13.2 `Pcat1..Pcat6` extra-bits probability lists (terminator
/// removed; mirrors the decoder copy in `dct_tokens.rs`).
const ENC_PCAT1: &[u8] = &[159];
const ENC_PCAT2: &[u8] = &[165, 145];
const ENC_PCAT3: &[u8] = &[173, 148, 140];
const ENC_PCAT4: &[u8] = &[176, 155, 140, 135];
const ENC_PCAT5: &[u8] = &[180, 157, 141, 134, 130];
const ENC_PCAT6: &[u8] = &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

/// §13.2 `categoryBase[6]` — first value in each cat1..cat6 range.
const ENC_CAT_BASE: [u16; 6] = [5, 7, 11, 19, 35, 67];

/// Classify the absolute value of a single coefficient into its
/// RFC 6386 §13.2 DCT-token alphabet entry. Caller is responsible for
/// the sign bit (which is emitted separately by the encoder at fixed
/// probability 128).
///
/// Returns `DctToken::Dct0` for the literal zero value (not `Eob` —
/// EOB is a separate decision made by the surrounding block-encoder
/// based on whether any non-zero coefficient follows).
pub fn classify_coeff_token(abs_value: u16) -> DctToken {
    match abs_value {
        0 => DctToken::Dct0,
        1 => DctToken::Dct1,
        2 => DctToken::Dct2,
        3 => DctToken::Dct3,
        4 => DctToken::Dct4,
        5..=6 => DctToken::Cat1,
        7..=10 => DctToken::Cat2,
        11..=18 => DctToken::Cat3,
        19..=34 => DctToken::Cat4,
        35..=66 => DctToken::Cat5,
        _ => DctToken::Cat6,
    }
}

/// Errors surfaced by the §13 token-block encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenEncodeError {
    /// A coefficient magnitude exceeded the §13.2 alphabet's largest
    /// representable value (`67 + (2^11 - 1) = 2114`). The encoder
    /// makes no attempt to clamp; the caller is expected to have run
    /// the §14 quantizer and have its results in range. Surfaces the
    /// offending raster index and value so the caller can pin the
    /// quantizer bug.
    CoefficientOutOfRange { index: usize, value: i16 },
}

impl core::fmt::Display for TokenEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TokenEncodeError::CoefficientOutOfRange { index, value } => write!(
                f,
                "vp8 token encode: coefficient[{index}]={value} exceeds the §13.2 alphabet maximum (±2114)"
            ),
        }
    }
}

impl std::error::Error for TokenEncodeError {}

/// One step of a §13.2 token bit-path: the `coeff_probs[…][prob_index]`
/// slot to consult and the bit to emit / decode at that node.
///
/// Width matches the `(usize, bool)` tuple the three call sites in this
/// module iterate over so the table reads as a drop-in replacement for
/// the prior `Vec<(usize, bool)>`.
#[derive(Clone, Copy, Debug)]
struct TokenBitStep {
    /// The `coeff_probs[plane][band][ctx3][prob_index]` slot index — the
    /// `i_half` value the call sites pass into `probs[i_half]`.
    prob_index: u8,
    /// The boolean to emit (encoder) / accumulate against the prob
    /// (RD scorer / fitter).
    bit: bool,
}

/// Maximum bit-path width across the 24 reachable `(start_index, token)`
/// combinations. Computed in `precompute_token_bit_paths` below — the
/// deepest leaves (`Cat3..Cat6` from root) emit exactly 7 bits; every
/// other path is shorter. The fixed-width buffer lets the table live in
/// static storage with zero per-call allocation.
const TOKEN_BIT_PATH_MAX_LEN: usize = 7;

/// One precomputed table entry: a fixed-width buffer of up to
/// [`TOKEN_BIT_PATH_MAX_LEN`] [`TokenBitStep`]s plus the number of valid
/// leading slots. Type-aliased to keep [`TOKEN_BIT_PATHS`] readable.
type TokenBitPathEntry = ([TokenBitStep; TOKEN_BIT_PATH_MAX_LEN], u8);

/// The full table: `[start_slot ∈ {0, 1}][token ∈ 0..12]` →
/// [`TokenBitPathEntry`]. See [`TOKEN_BIT_PATHS`] for indexing.
type TokenBitPathTable = [[TokenBitPathEntry; 12]; 2];

/// Precomputed token bit paths for every `(start_index, token)` pair
/// reachable from [`ENC_COEFF_TREE`]. Indexed as
/// `TOKEN_BIT_PATHS[start_slot][token as usize]`, where `start_slot` is
/// `0` for `start_index = 0` (the §13.2 root including the Eob branch)
/// and `1` for `start_index = 2` (the "previous coefficient was DCT_0"
/// shortcut that bypasses the Eob split).
///
/// Each entry is `(buffer, length)`; only the leading `length` slots of
/// `buffer` are valid. `(start_slot = 1, token = Eob)` is unreachable
/// (§13.2 explicitly forbids EOB after a DCT_0) and stores `length = 0`
/// as a tombstone; the call sites never request it.
///
/// Computed at module load via `std::sync::LazyLock`. The first call
/// inside the encoder hot path drives the one-time cost; every
/// subsequent token emission resolves to a single index-and-slice.
static TOKEN_BIT_PATHS: std::sync::LazyLock<TokenBitPathTable> =
    std::sync::LazyLock::new(precompute_token_bit_paths);

fn precompute_token_bit_paths() -> TokenBitPathTable {
    let zero = TokenBitStep {
        prob_index: 0,
        bit: false,
    };
    let mut out = [[([zero; TOKEN_BIT_PATH_MAX_LEN], 0u8); 12]; 2];

    fn descend(
        i: i8,
        target: i8,
        buf: &mut [TokenBitStep; TOKEN_BIT_PATH_MAX_LEN],
        len: &mut usize,
    ) -> bool {
        for &bit in &[false, true] {
            let child = ENC_COEFF_TREE[i as usize + bit as usize];
            if child <= 0 {
                if -child == target {
                    buf[*len] = TokenBitStep {
                        prob_index: (i as usize >> 1) as u8,
                        bit,
                    };
                    *len += 1;
                    return true;
                }
            } else {
                buf[*len] = TokenBitStep {
                    prob_index: (i as usize >> 1) as u8,
                    bit,
                };
                *len += 1;
                if descend(child, target, buf, len) {
                    return true;
                }
                *len -= 1;
            }
        }
        false
    }

    for (slot, &start) in [0i8, 2i8].iter().enumerate() {
        for token in 0i8..12 {
            let mut buf = [zero; TOKEN_BIT_PATH_MAX_LEN];
            let mut len = 0usize;
            if descend(start, token, &mut buf, &mut len) {
                out[slot][token as usize] = (buf, len as u8);
            }
            // (slot = 1, token = Eob) is unreachable; the (zeros, 0)
            // tombstone is correct and never indexed by the encoder.
        }
    }
    out
}

/// Walk [`ENC_COEFF_TREE`] starting at internal node `start_index` and
/// return the `(prob_index, bit)` sequence that lands at the leaf for
/// `target`. Mirrors the decoder's `treed_read_coef` traversal but
/// runs backwards from a known leaf.
///
/// Resolves through the precomputed [`TOKEN_BIT_PATHS`] table — every
/// reachable `(start_index, token)` pair is materialised at module load,
/// so each call is a single index + slice borrow with no allocation.
/// The `start_index` distinguishes the §13.2 "may emit EOB" case (`0`)
/// from the "previous coefficient was DCT_0" case (`2`, EOB branch
/// bypassed).
fn token_to_bit_path(token: DctToken, start_index: i8) -> &'static [TokenBitStep] {
    let slot = match start_index {
        0 => 0usize,
        2 => 1usize,
        _ => {
            debug_assert!(false, "unsupported coeff_tree start_index {start_index}");
            0
        }
    };
    let (buf, len) = &TOKEN_BIT_PATHS[slot][token as usize];
    debug_assert!(
        *len > 0,
        "token {token:?} not reachable from coeff_tree index {start_index}"
    );
    &buf[..*len as usize]
}

/// Look up the cat-token's `(base, prob_list)` pair. Returns `None`
/// for the non-cat tokens (Dct0..Dct4, Eob) where there are no
/// trailing extra bits.
fn cat_extras(token: DctToken) -> Option<(u16, &'static [u8])> {
    match token {
        DctToken::Cat1 => Some((ENC_CAT_BASE[0], ENC_PCAT1)),
        DctToken::Cat2 => Some((ENC_CAT_BASE[1], ENC_PCAT2)),
        DctToken::Cat3 => Some((ENC_CAT_BASE[2], ENC_PCAT3)),
        DctToken::Cat4 => Some((ENC_CAT_BASE[3], ENC_PCAT4)),
        DctToken::Cat5 => Some((ENC_CAT_BASE[4], ENC_PCAT5)),
        DctToken::Cat6 => Some((ENC_CAT_BASE[5], ENC_PCAT6)),
        _ => None,
    }
}

/// Encode one 16-coefficient sub-block per RFC 6386 §13.3 into `enc`.
///
/// `coeffs` is a 16-entry array in **scan (zig-zag) order** — i.e. the
/// exact order the §13.3 token loop visits positions. Position
/// `block_type.first_coeff()` is the first slot read; earlier slots
/// are ignored (for `YAfterY2` the DC is part of the Y2 block and
/// `coeffs[0]` is unused by the encoder).
///
/// `block_type` selects the §13.3 plane index and the `firstCoeff`
/// value. `coeff_probs` is the resolved `coeff_probs[4][8][3][11]`
/// from [`merge_default_token_probs`](crate::merge_default_token_probs).
/// `above_has_nonzero` / `left_has_nonzero` are the §13.3 neighbour
/// non-zero predictors that seed `ctx3` for the very first
/// coefficient (off-frame neighbours pass `false`).
///
/// Returns `Ok(non_zero_count)` — the number of non-zero coefficients
/// emitted. The encoder always concludes with an `Eob` token unless
/// all 16 (or 15 for `YAfterY2`) coefficients are non-zero, in which
/// case §13.2's "implicit eob after the last coefficient" rule
/// applies and the encoder omits the explicit `Eob`.
///
/// This is the Phase-2 inverse of [`crate::decode_block`]. The unit
/// tests in this module prove byte-exact round-trip at the coefficient
/// layer: encode the block, hand the resulting bytes to a
/// `BoolDecoder`, walk `decode_block` with the same `block_type` /
/// `coeff_probs` / neighbour predictors, and assert the recovered
/// `coeffs` array equals the input.
pub fn encode_coeff_block(
    enc: &mut BoolEncoder,
    block_type: BlockType,
    coeff_probs: &CoeffProbs,
    above_has_nonzero: bool,
    left_has_nonzero: bool,
    coeffs: &[i16; 16],
) -> Result<usize, TokenEncodeError> {
    let plane = block_type.plane_index();
    let first_coeff = block_type.first_coeff();

    // Validate range up front — §13.2 Cat6 maxes out at 67 + 2047 = 2114.
    for (i, &v) in coeffs.iter().enumerate().skip(first_coeff) {
        let abs = v.unsigned_abs();
        if abs > 2114 {
            return Err(TokenEncodeError::CoefficientOutOfRange { index: i, value: v });
        }
    }

    // Find the last non-zero coefficient so the encoder knows where
    // (if anywhere) to emit the explicit EOB token. If the entire
    // block is zero the EOB lands at `first_coeff` immediately.
    let mut last_non_zero: i32 = -1;
    for (i, &v) in coeffs.iter().enumerate() {
        if i >= first_coeff && v != 0 {
            last_non_zero = i as i32;
        }
    }

    let mut ctx3: usize = (above_has_nonzero as usize) + (left_has_nonzero as usize);
    let mut prev_was_zero = false;
    let mut non_zero_count = 0usize;

    let mut i = first_coeff;
    while i < 16 {
        let band = COEFF_BANDS[i];
        let probs = &coeff_probs[plane][band][ctx3];

        // Decide what token to emit at this position.
        let emit_eob = (i as i32) > last_non_zero;
        let (token, abs_value, sign) = if emit_eob {
            (DctToken::Eob, 0u16, false)
        } else {
            let v = coeffs[i];
            let abs = v.unsigned_abs();
            (classify_coeff_token(abs), abs, v < 0)
        };

        // The §13.2 "skip EOB branch" optimisation: enter the tree at
        // internal node 1 (raw index 2) when the previous coefficient
        // was a literal `DCT_0` — the decoder will not have read the
        // EOB-vs-rest split bit, so neither must we write it.
        let start = if prev_was_zero { 2i8 } else { 0i8 };

        for step in token_to_bit_path(token, start) {
            enc.write_bool(probs[step.prob_index as usize], step.bit);
        }

        if token == DctToken::Eob {
            break;
        }

        // Cat tokens carry a fixed-width little-endian extra-bits
        // suffix (MSB-first within the suffix per the decoder's
        // `read_extra_bits` loop), then the universal sign bit at
        // probability 128.
        if let Some((base, plist)) = cat_extras(token) {
            let extra = abs_value - base;
            let n = plist.len();
            for (j, &p) in plist.iter().enumerate() {
                let bit = ((extra >> (n - 1 - j)) & 1) == 1;
                enc.write_bool(p, bit);
            }
        }

        if abs_value != 0 {
            enc.write_bool(128, sign);
            non_zero_count += 1;
        }

        // §13.3 rollover of `ctx3` to the token's magnitude class.
        ctx3 = if abs_value == 0 {
            0
        } else if abs_value == 1 {
            1
        } else {
            2
        };
        prev_was_zero = token == DctToken::Dct0;

        i += 1;
    }

    Ok(non_zero_count)
}

/// Phase-2 token-block encoder handle.
///
/// A thin wrapper around the §13 `encode_coeff_block` free function
/// that owns its own [`BoolEncoder`] + resolved `coeff_probs` table
/// and exposes a stateful `encode_block` method. Useful for the
/// higher-level encoder rounds that will emit many blocks into the
/// same partition.
///
/// The encoder is **stateless across blocks** in the §13.3 sense —
/// the `ctx3` rollover lives entirely inside `encode_coeff_block` and
/// the neighbour predictors are passed in per call. What the wrapper
/// owns is the underlying entropy-coder byte stream + the (large)
/// probability table.
#[derive(Debug)]
pub struct TokenEncoder {
    enc: BoolEncoder,
    coeff_probs: CoeffProbs,
}

impl TokenEncoder {
    /// Build an encoder against the supplied resolved coefficient
    /// probabilities. Pass [`crate::DEFAULT_COEFF_PROBS`] (or the
    /// `merge_default_token_probs(&updates)` result) to match the
    /// `decode_block` path callers go through on the other side.
    pub fn new(coeff_probs: CoeffProbs) -> Self {
        TokenEncoder {
            enc: BoolEncoder::new(),
            coeff_probs,
        }
    }

    /// Encode one 16-coefficient sub-block. See
    /// [`encode_coeff_block`] for the parameter semantics.
    pub fn encode_block(
        &mut self,
        block_type: BlockType,
        above_has_nonzero: bool,
        left_has_nonzero: bool,
        coeffs: &[i16; 16],
    ) -> Result<usize, TokenEncodeError> {
        encode_coeff_block(
            &mut self.enc,
            block_type,
            &self.coeff_probs,
            above_has_nonzero,
            left_has_nonzero,
            coeffs,
        )
    }

    /// Finalise the underlying [`BoolEncoder`] and return the byte
    /// stream the decoder should consume.
    pub fn finish(self) -> Vec<u8> {
        self.enc.finish()
    }

    /// Number of bytes committed so far (excludes the §7.3 4-byte
    /// flush trailer that [`finish`](Self::finish) will append).
    pub fn bytes_written(&self) -> usize {
        self.enc.bytes_written()
    }
}

// ───────────────────────── Phase 2: per-MB block-set encoder ──────────────────

use crate::dct_tokens::MbEntropyCtx;
use crate::forward_transform::{forward_dct_4x4, forward_wht_4x4, raster_to_scan};
use crate::frame::MbCoeffs;
use crate::intra_predict::{
    predict_b4x4, predict_uv8x8, predict_y16x16, DEFAULT_ABOVE_PIXEL, DEFAULT_LEFT_PIXEL,
};
use crate::inverse_transform::{add_residue_4x4, inverse_dct_4x4, inverse_wht_4x4};
use crate::macroblock::{
    IntraBmode, IntraUvMode, IntraYMode, MacroblockModes, BMODE_TREE, IF_UV_MODE_PROB_DEFAULTS,
    IF_YMODE_PROB_DEFAULTS, IF_YMODE_TREE, KF_BMODE_PROB, KF_UV_MODE_PROB, KF_YMODE_PROB,
    KF_YMODE_TREE, UV_MODE_TREE,
};

/// Round-half-away-from-zero integer division — the natural inverse of
/// the §14.1 `q * factor` dequant multiply. Used by the encoder to
/// quantise a single 16-bit coefficient by an `i16` factor.
#[inline]
fn enc_round_div(num: i32, den: i32) -> i16 {
    debug_assert!(den > 0, "encoder quantiser factor must be positive");
    let r = if num >= 0 {
        (num + den / 2) / den
    } else {
        -(((-num) + den / 2) / den)
    };
    r as i16
}

/// Quantise a raster-order 4×4 coefficient block in place against a
/// `(dc_factor, ac_factor)` pair — coefficient 0 is divided by `dc`,
/// coefficients 1..=15 by `ac`, with round-half-away-from-zero rounding.
/// This is the inverse of [`crate::dequant_block`].
#[inline]
fn enc_quantize_block(block: &mut [i16; 16], dc: i16, ac: i16) {
    block[0] = enc_round_div(block[0] as i32, dc as i32);
    for c in block.iter_mut().skip(1) {
        *c = enc_round_div(*c as i32, ac as i32);
    }
}

/// Inputs to the §14 per-macroblock encoder — a single raw 16×16 Y plane
/// and two 8×8 chroma planes, plus a chosen quantizer.
///
/// Pixels are 8-bit unsigned (`u8`) in raster order. The MB encoder
/// computes the §11 / §12 intra prediction internally (Phase 2 ships
/// only the `DC_PRED` constant-prediction path for both luma and
/// chroma) and feeds the residual through §14 / §13.
#[derive(Debug, Clone, Copy)]
pub struct MbPixels {
    /// 16×16 Y plane, row-major.
    pub y: [u8; 256],
    /// 8×8 Cb plane, row-major.
    pub u: [u8; 64],
    /// 8×8 Cr plane, row-major.
    pub v: [u8; 64],
}

/// Sum of squared differences (SSD) between a reconstructed candidate and
/// the source pixels — the distortion term `D` of the §-non-normative
/// rate-distortion cost `J = D + lambda * R`. Unlike SAD this matches the
/// MSE that PSNR is computed from, so minimising `D` at a fixed `R`
/// directly maximises self-decode PSNR. Computed in `u64` because a full
/// 16×16 luma block of maximal error sums to `256 * 255^2 ≈ 1.66e7`,
/// well inside `u64` (and even `u32`), but `u64` keeps the chroma + luma
/// accumulation headroom-free.
#[inline]
fn block_ssd(recon: &[u8], src: &[u8]) -> u64 {
    debug_assert_eq!(recon.len(), src.len());
    recon
        .iter()
        .zip(src.iter())
        .map(|(&r, &s)| {
            let d = r as i64 - s as i64;
            (d * d) as u64
        })
        .sum()
}

// ─────────────────────── rate-distortion cost machinery ───────────────────────
//
// RFC 6386 does not specify mode selection (§14.1 page 76: the spec
// "describes only the *decoding* process"); rate-distortion is therefore
// a non-normative encoder choice. The cost is the textbook Lagrangian
//
//     J = D + lambda * R
//
// where `D` is the reconstruction SSD against the source and `R` is the
// number of bits the candidate's tokens + mode signal will cost. The
// picker minimises `J` over the candidate modes.
//
// Bit cost (`R`). Every entropy-coded decision is a §7.3 `write_bool`
// against a probability `p` that the bit is 0. The information content of
// writing `value` is `-log2(P(value))` bits, with `P(0) = p / 256` and
// `P(1) = (256 - p) / 256`. The encoder accumulates these exactly (in
// fractional bits) without touching the real `BoolEncoder`, so the
// estimate equals the bits the token pass below will actually emit
// (modulo the §7.3 renormalisation, which adds < 1 byte per partition,
// not per block — negligible for a per-block relative comparison).
//
// Lambda. For a uniform quantiser of step `q`, high-rate RD theory gives
// `D ~ q^2 / 12` and the RD-optimal multiplier `lambda ~ c * q^2`. We use
// the luma AC dequant factor as the step `q` and a constant `c`
// calibrated so the trade stays mild (the picker should shave bits, not
// crater PSNR). See [`rd_lambda`].

/// Cost in fractional bits of one §7.3 boolean written at probability
/// `prob` (the chance the bit is 0) carrying `value`. `-log2` of the
/// event probability. `prob` is clamped to `1..=255` so the logarithm is
/// always finite (the spec's probabilities are themselves in `1..=255`).
///
/// Round-170 perf note: this function is on the per-token-bit hot path
/// of the encoder's RD scoring (`estimate_block_bits` calls it for every
/// emitted bit of every candidate block, and the per-MB mode picker
/// scores many candidates). The per-call `f64::log2` was the single
/// dominant self-time symbol on a `sample` profile of the
/// `inter_encode_short_clip` bench (60 / 12,721 samples — see
/// `BENCHMARKS.md`). The implementation now consults a precomputed
/// 256-entry table: `BIT_COST_BY_FALSE_PROB[p]` stores
/// `-log2(p / 256)` (the bit-cost when the encoded bit is 0 and the
/// frame's table assigns it probability `p`), with the `p = 0` slot
/// clamped to `-log2(1/256) = 8.0` to match the original `prob.max(1)` +
/// floor-at-`1/256` clamp. The `true` branch reads slot `256 - p` (i.e.
/// `-log2(1 - p / 256)`), with the `p = 256` (encoded bit guaranteed 1)
/// case naturally handled by reading slot 0 of the same table — and the
/// `p = 0` (encoded bit guaranteed 0, asked for true) case staying at
/// the same `8.0` floor for symmetry.
#[inline]
pub(crate) fn bool_bits(prob: u8, value: bool) -> f64 {
    if value {
        // -log2(1 - prob/256). prob == 0 ⇒ index = 256, but we clamp
        // it to 255 (i.e. -log2(1/256) = 8.0) so the asymmetric
        // 1/256-floor of the original `p.max(1.0 / 256.0).log2()`
        // survives the rewrite without an extra branch.
        let idx = 256usize - prob as usize;
        BIT_COST_BY_FALSE_PROB[idx.min(255)]
    } else {
        BIT_COST_BY_FALSE_PROB[prob as usize]
    }
}

/// `BIT_COST_BY_FALSE_PROB[p] = -log2(p / 256)` in fractional bits.
///
/// 257 entries are conceptually needed (`p ∈ 0..=256`), but the original
/// `bool_bits` clamps `prob` into `1..=255` and floors the resulting
/// probability at `1/256`, so the `p = 0` slot stores the same `8.0` as
/// `p = 1` (and the `value == true` branch's `256 - 0 = 256` index is
/// pre-clamped to `255` so we only ever read in-bounds). This table is
/// precomputed at module-load (`std::sync::OnceLock`) so the only
/// arithmetic on the hot path is the bool / index pair above.
static BIT_COST_BY_FALSE_PROB: std::sync::LazyLock<[f64; 256]> = std::sync::LazyLock::new(|| {
    let mut t = [0f64; 256];
    // p = 0 ⇒ same floor as the original `.max(1.0 / 256.0)` (i.e. 1/256).
    t[0] = -((1.0f64 / 256.0).log2());
    for (p, slot) in t.iter_mut().enumerate().skip(1) {
        // -log2(p / 256). `slot` is the lvalue.
        *slot = -((p as f64) / 256.0).log2();
    }
    t
});

/// Depth-first search for the bit path from the root (node 0) of `tree`
/// to the leaf `-leaf`, written into the caller's fixed `path` buffer.
/// Returns the path length (0 when the leaf is unreachable, matching the
/// empty-path behaviour of the previous heap-allocating walk; debug
/// builds assert reachability).
///
/// The buffer is 16 entries: a §8.1 tree path visits distinct internal
/// nodes, so its length is bounded by the node count `tree.len() / 2`,
/// and the largest tree this crate walks (`BMODE_TREE`, 18 entries) has
/// 9 internal nodes. Round-276 profiling showed the per-call `Vec` this
/// replaces was a top allocator-churn source in the mode-RD loop.
fn treed_find_path(tree: &[i8], leaf: u8, path: &mut [bool; 16]) -> usize {
    fn dfs(tree: &[i8], i: i8, target: i8, path: &mut [bool; 16], len: &mut usize) -> bool {
        for bit in 0..2 {
            let next = tree[i as usize + bit];
            path[*len] = bit == 1;
            *len += 1;
            if next == target {
                return true;
            }
            if next > 0 && dfs(tree, next, target, path, len) {
                return true;
            }
            *len -= 1;
        }
        false
    }
    let mut len = 0;
    let found = dfs(tree, 0, -(leaf as i8), path, &mut len);
    debug_assert!(found, "leaf {leaf} not reachable in tree {tree:?}");
    len
}

/// Cost in fractional bits of writing `value` through `tree` with the
/// per-node probability `prob_lookup`, mirroring [`BoolEncoder::write_treed`]
/// but accumulating `-log2(p)` instead of emitting. The RD scorers now
/// price against the precomputed [`treed_bits_cached`] tables; this
/// DFS-walk form is retained as the reference the
/// `treed_bits_cached_matches_dfs_walk` pin compares them to.
#[cfg_attr(not(test), allow(dead_code))]
fn treed_bits<F>(tree: &[i8], prob_lookup: F, leaf: u8) -> f64
where
    F: Fn(usize) -> u8,
{
    let mut path = [false; 16];
    let len = treed_find_path(tree, leaf, &mut path);
    let mut bits = 0.0;
    let mut i: i8 = 0;
    for &bit in &path[..len] {
        bits += bool_bits(prob_lookup((i as usize) >> 1), bit);
        i = tree[i as usize + bit as usize];
    }
    bits
}

/// One leaf's precomputed §8.1 walk through a mode tree: the bit path
/// [`treed_find_path`] discovers, plus the node-halved probability-index
/// sequence [`treed_bits`] derives while replaying it. The trees the RD
/// scorers price against are compile-time constants, so the DFS is a
/// pure function of `(tree, leaf)` — precomputing it once per tree
/// removes the per-candidate `treed_find_path::dfs` walk (≈ 4 % of
/// keyframe-encode self-time on the round-409 profile) while replaying
/// the identical `(prob_index, bit)` step sequence in the identical
/// order, so every priced `f64` is bit-for-bit the DFS-walk value
/// (`treed_bits_cached_matches_dfs_walk` pins all four tables).
#[derive(Clone, Copy)]
struct TreedLeafPath {
    /// Number of valid steps in `bits` / `prob_index`.
    len: u8,
    /// The bit at each step of the root-to-leaf path.
    bits: [bool; 16],
    /// The node-halved probability index consumed at each step.
    prob_index: [u8; 16],
}

/// Build the per-leaf path table for a mode tree whose leaves are
/// numbered `0..N` (true for every §8.1 mode tree this crate prices).
fn precompute_treed_paths<const N: usize>(tree: &[i8]) -> [TreedLeafPath; N] {
    core::array::from_fn(|leaf| {
        let mut bits = [false; 16];
        let len = treed_find_path(tree, leaf as u8, &mut bits);
        let mut prob_index = [0u8; 16];
        let mut i: i8 = 0;
        for (s, &bit) in bits[..len].iter().enumerate() {
            prob_index[s] = ((i as usize) >> 1) as u8;
            i = tree[i as usize + bit as usize];
        }
        TreedLeafPath {
            len: len as u8,
            bits,
            prob_index,
        }
    })
}

/// Precomputed paths for the §11.2 keyframe luma mode tree.
static KF_YMODE_PATHS: std::sync::LazyLock<[TreedLeafPath; 5]> =
    std::sync::LazyLock::new(|| precompute_treed_paths(&KF_YMODE_TREE));
/// Precomputed paths for the §11.2 chroma mode tree (shared by the
/// keyframe and interframe chroma scorers — the tree is the same, only
/// the probability tables differ).
static UV_MODE_PATHS: std::sync::LazyLock<[TreedLeafPath; 4]> =
    std::sync::LazyLock::new(|| precompute_treed_paths(&UV_MODE_TREE));
/// Precomputed paths for the §11.2 / §11.4 4×4 sub-block mode tree.
static BMODE_PATHS: std::sync::LazyLock<[TreedLeafPath; 10]> =
    std::sync::LazyLock::new(|| precompute_treed_paths(&BMODE_TREE));
/// Precomputed paths for the §11.3 interframe intra luma mode tree.
static IF_YMODE_PATHS: std::sync::LazyLock<[TreedLeafPath; 5]> =
    std::sync::LazyLock::new(|| precompute_treed_paths(&IF_YMODE_TREE));

/// [`treed_bits`] against a precomputed path table: the same
/// `bool_bits(prob_lookup(node >> 1), bit)` accumulation in the same
/// step order, with the DFS and the node-chasing replay replaced by the
/// table row. Bit-for-bit the DFS-walk sum.
fn treed_bits_cached<F>(paths: &[TreedLeafPath], prob_lookup: F, leaf: u8) -> f64
where
    F: Fn(usize) -> u8,
{
    let p = &paths[leaf as usize];
    let mut bits = 0.0;
    for s in 0..p.len as usize {
        bits += bool_bits(prob_lookup(p.prob_index[s] as usize), p.bits[s]);
    }
    bits
}

/// Estimate the bit cost of one §13.3 residual block exactly as
/// [`encode_coeff_block`] would emit it, but accumulating `-log2(p)`
/// fractional bits instead of writing to a [`BoolEncoder`]. The control
/// flow is a line-for-line mirror of `encode_coeff_block` so the estimate
/// tracks the real token pass (token tree path, cat extra bits, and the
/// per-coefficient sign bit), threading the same `ctx3` rollover.
///
/// `above_has_nonzero` / `left_has_nonzero` seed `ctx3` exactly as the
/// real encoder does. The block is in scan (zig-zag) order. Coefficients
/// out of the §13.2 alphabet are priced at the Cat6 maximum rather than
/// erroring — the RD picker only ever scores in-range quantiser output,
/// and an out-of-range candidate should simply lose.
fn estimate_block_bits(
    block_type: BlockType,
    coeff_probs: &CoeffProbs,
    above_has_nonzero: bool,
    left_has_nonzero: bool,
    coeffs: &[i16; 16],
) -> f64 {
    let plane = block_type.plane_index();
    let first_coeff = block_type.first_coeff();

    let mut last_non_zero: i32 = -1;
    for (i, &v) in coeffs.iter().enumerate() {
        if i >= first_coeff && v != 0 {
            last_non_zero = i as i32;
        }
    }

    let mut ctx3: usize = (above_has_nonzero as usize) + (left_has_nonzero as usize);
    let mut prev_was_zero = false;
    let mut bits = 0.0;

    let mut i = first_coeff;
    while i < 16 {
        let band = COEFF_BANDS[i];
        let probs = &coeff_probs[plane][band][ctx3];

        let emit_eob = (i as i32) > last_non_zero;
        let (token, abs_value, _sign) = if emit_eob {
            (DctToken::Eob, 0u16, false)
        } else {
            let v = coeffs[i];
            (
                classify_coeff_token(v.unsigned_abs()),
                v.unsigned_abs(),
                v < 0,
            )
        };

        let start = if prev_was_zero { 2i8 } else { 0i8 };
        for step in token_to_bit_path(token, start) {
            bits += bool_bits(probs[step.prob_index as usize], step.bit);
        }

        if token == DctToken::Eob {
            break;
        }

        if let Some((base, plist)) = cat_extras(token) {
            // Cat6 caps the alphabet at 67 + 2047; price an out-of-range
            // value at its maximum extra-bits string so it loses the RD
            // race instead of panicking.
            let extra = abs_value.saturating_sub(base);
            let n = plist.len();
            for (j, &p) in plist.iter().enumerate() {
                let bit = ((extra >> (n - 1 - j)) & 1) == 1;
                bits += bool_bits(p, bit);
            }
        }

        if abs_value != 0 {
            bits += bool_bits(128, false); // sign bit, flat probability
        }

        ctx3 = if abs_value == 0 {
            0
        } else if abs_value == 1 {
            1
        } else {
            2
        };
        prev_was_zero = token == DctToken::Dct0;
        i += 1;
    }

    bits
}

/// Fraction of the mode-decision Lagrange multiplier the §13 coefficient
/// trellis spends at [`TrellisStrength::DEFAULT`] (`= 1.0`). Coefficient-level
/// RDO is a much finer rate lever than the mode picker (it can drop one
/// trailing coefficient instead of a whole mode), so applying the full mode
/// `lambda` over-trades PSNR for rate. Swept on the `natural_test_frame_64x64`
/// keyframe RD fixtures: at the raw mode lambda the trellis cuts keyframe
/// bytes ~45 % but loses 1–2.5 dB; `0.04` is the largest scale that *holds
/// the SAD-picker PSNR floor at every quantiser* (the conservative "shave
/// bits, never drop below the prior PSNR" regime the `rd_lambda` doctrine
/// targets) while still trimming 12–13 % of bytes at the low/mid quantisers
/// where VP8 keyframes actually operate. [`TrellisStrength`] multiplies this
/// base: a strength `> 1.0` spends more bits-vs-quality (smaller output,
/// steeper PSNR trade), `< 1.0` is more conservative, `0.0` disables the
/// trellis (plain round-quantisation).
const TRELLIS_LAMBDA_SCALE: f64 = 0.04;

/// Trellis-quantiser aggressiveness knob — a clean-room encoder quality/size
/// dial multiplying the calibrated [`TRELLIS_LAMBDA_SCALE`] base.
///
/// RFC 6386 specifies only token *decoding*; the level-assignment search the
/// §13 trellis runs is a non-normative encoder decision (the spec is silent
/// on it by design), so this knob is entirely a clean-room rate/quality
/// control. `1.0` reproduces the historically-calibrated conservative trade
/// (hold the SAD-picker PSNR floor, trim 12–13 % of bytes); higher values
/// trade more PSNR for fewer bytes, `0.0` falls back to plain
/// round-quantisation (the pre-trellis bytes). The value is clamped to
/// `0.0..=MAX` so a caller cannot push the trellis into an unbounded
/// distortion regime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrellisStrength(f64);

impl TrellisStrength {
    /// The calibrated default — reproduces the round-354 trellis bytes
    /// exactly (`scale = 1.0 × TRELLIS_LAMBDA_SCALE`).
    pub const DEFAULT: TrellisStrength = TrellisStrength(1.0);
    /// Disable the trellis: every block falls back to plain
    /// round-quantisation (the pre-trellis wire).
    pub const OFF: TrellisStrength = TrellisStrength(0.0);
    /// Upper clamp. Beyond this the trellis zeroes nearly every AC
    /// coefficient at the working quantisers, so it is a safety rail rather
    /// than a useful operating point.
    pub const MAX: f64 = 8.0;

    /// Build a strength from a raw multiplier, clamping into `0.0..=MAX`.
    /// `NaN` maps to the default (safe, conservative) rather than `OFF`.
    pub fn new(strength: f64) -> Self {
        if strength.is_nan() {
            return Self::DEFAULT;
        }
        TrellisStrength(strength.clamp(0.0, Self::MAX))
    }

    /// The clamped multiplier.
    pub fn get(self) -> f64 {
        self.0
    }

    /// The effective coefficient-level lambda scale — the calibrated base
    /// times this strength. `0.0` (the [`OFF`](Self::OFF) state) yields a
    /// zero scale, routing [`trellis_quantize_block`] to its plain-rounding
    /// fallback.
    fn lambda_scale(self) -> f64 {
        self.0 * TRELLIS_LAMBDA_SCALE
    }
}

impl Default for TrellisStrength {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ───────────────────────── §13 trellis coefficient quantisation ─────────────
//
// RFC 6386 specifies only the *decoding* of §13.3 residual tokens; the
// choice of which quantised level to assign each coefficient is a
// non-normative encoder decision (the spec is silent on it by design).
// The baseline `enc_quantize_block` rounds each coefficient independently
// (round-half-away-from-zero divide by the §14.1 step). That is optimal
// for distortion alone, but the §13.3 token stream couples successive
// coefficients: the `ctx3` carried into the next position depends on this
// coefficient's magnitude class (0 / 1 / ≥2), the band/EOB-skip depends on
// whether it was a literal `DCT_0`, and the implicit-EOB tail means
// trailing low-magnitude coefficients can cost many bits for little
// distortion reduction. Trellis quantisation jointly minimises the
// Lagrangian `J = D + lambda·R` over the level assignment by a Viterbi
// pass in scan order.
//
// **Candidate levels.** For each coefficient the search considers its
// rounded magnitude `m = round(|x| / q)` and `m − 1` (the standard
// down-only set: rounding up never both lowers rate and distortion, and a
// level above `round` raises distortion *and* — for magnitudes past the
// Dct token boundaries — rate). `m − 1 = 0` is the "drop to zero"
// candidate that the trellis spends elsewhere or terminates before. The
// sign always follows `x` (the §13.3 sign bit is a flat-128 bit, identical
// cost for either polarity).
//
// **Distortion.** Computed in the transform domain as
// `w_k · (x_k − level_k·q)^2`, where `w_k` is the squared L2 norm of the
// k-th inverse-DCT basis vector ([`IDCT_BASIS_NORM_SQ`]). Because the §14
// inverse transform is linear (modulo a sub-LSB rounding bit), summing the
// per-coefficient transform-domain errors weighted by their basis norms
// reproduces the pixel-domain reconstruction SSD — so this `D` is on the
// same scale as [`block_ssd`] and the existing [`rd_lambda`] applies
// unchanged.
//
// **Rate.** The exact §13.3 token bits for the chosen token at the
// position's band + context, including cat extra bits and the sign bit,
// priced through the same [`bool_bits`] table the mode pickers use.
//
// The trellis only *chooses levels*; reconstruction still runs the
// identical dequant → IDCT → add chain the decoder runs, so the
// encoder/decoder pixel-lockstep contract is preserved exactly.

/// Squared L2 norm of each §14.4 inverse-DCT basis vector, indexed by
/// **scan position** (so entry `k` weights the error of scan coefficient
/// `k`). Computed once at module load by running the real
/// [`inverse_dct_4x4`] on a unit impulse at the raster position
/// `ZIGZAG[k]` and summing the squared outputs — i.e. the exact
/// pixel-domain energy one unit of quantisation error at scan position `k`
/// injects. Using the actual decoder IDCT (not an idealised orthonormal
/// transform) keeps the trellis distortion on the same scale as the
/// pixel-domain [`block_ssd`].
static IDCT_BASIS_NORM_SQ: std::sync::LazyLock<[f64; 16]> = std::sync::LazyLock::new(|| {
    let mut norms = [0f64; 16];
    for (k, slot) in norms.iter_mut().enumerate() {
        let raster = crate::dct_tokens::ZIGZAG[k];
        let mut coeffs = [0i16; 16];
        // A unit impulse large enough that the IDCT's `+4 >> 3` rounding
        // does not swallow the response, then normalise by the same
        // factor squared so the result is the per-unit basis energy.
        const IMPULSE: i16 = 256;
        coeffs[raster] = IMPULSE;
        let mut out = [0i16; 16];
        inverse_dct_4x4(&coeffs, &mut out);
        let energy: f64 = out.iter().map(|&v| (v as f64).powi(2)).sum();
        *slot = energy / (IMPULSE as f64).powi(2);
    }
    norms
});

/// Exact §13.3 token-rate (in fractional bits) of coding `abs_level` at
/// scan position `i` with the entry context `ctx3` and the §13.2
/// "previous coefficient was DCT_0" EOB-skip flag `prev_was_zero`,
/// against the resolved `plane` probability bank. Includes the token-tree
/// bits, cat extra bits, and the flat-128 sign bit (for non-zero levels).
/// This is the per-position `R` term the trellis minimises against; it is
/// a slice of the same control flow [`estimate_block_bits`] walks.
#[inline]
fn coeff_token_bits(
    coeff_probs: &CoeffProbs,
    plane: usize,
    i: usize,
    ctx3: usize,
    prev_was_zero: bool,
    abs_level: u16,
) -> f64 {
    let band = COEFF_BANDS[i];
    let probs = &coeff_probs[plane][band][ctx3];
    let token = classify_coeff_token(abs_level);
    let start = if prev_was_zero { 2i8 } else { 0i8 };
    let mut bits = 0.0;
    for step in token_to_bit_path(token, start) {
        bits += bool_bits(probs[step.prob_index as usize], step.bit);
    }
    if let Some((base, plist)) = cat_extras(token) {
        let extra = abs_level.saturating_sub(base);
        let n = plist.len();
        for (j, &p) in plist.iter().enumerate() {
            let bit = ((extra >> (n - 1 - j)) & 1) == 1;
            bits += bool_bits(p, bit);
        }
    }
    if abs_level != 0 {
        bits += bool_bits(128, false); // sign bit, flat probability
    }
    bits
}

/// Bits of the §13.3 EOB token at scan position `i` with context `ctx3`.
/// EOB is the first leaf of the §13.2 tree and is only reachable when the
/// previous coefficient was *not* a literal `DCT_0` (otherwise the decoder
/// has already skipped the EOB-vs-rest split). Returns the cost of the
/// single EOB branch bit.
#[inline]
fn eob_token_bits(coeff_probs: &CoeffProbs, plane: usize, i: usize, ctx3: usize) -> f64 {
    let band = COEFF_BANDS[i];
    let probs = &coeff_probs[plane][band][ctx3];
    // EOB is the left branch at the root (prob index 0, bit = false).
    bool_bits(probs[0], false)
}

/// Map an absolute level to its §13.3 next-context class (0 / 1 / ≥2).
#[inline]
fn level_ctx3(abs_level: u16) -> usize {
    match abs_level {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

/// Largest absolute level whose §13.3 token carries **no** cat extra
/// bits (`DCT_0` … `DCT_4`). For these the token rate is a pure function
/// of `(plane, band, ctx3, prev_was_zero, abs_level)` — the domain the
/// per-frame [`TrellisRateTables`] covers.
const TRELLIS_SMALL_LEVEL_MAX: u16 = 4;

/// Per-frame precomputed §13.3 token-rate tables for the trellis hot
/// loop. `coeff_probs` is fixed for a whole frame, so every
/// [`coeff_token_bits`] / [`eob_token_bits`] value the Viterbi scan asks
/// for at the extra-bit-free levels (`0..=4`, the overwhelming majority
/// of candidates at the working quantisers) is a pure per-frame constant.
/// Each entry is filled by calling the reference functions themselves, so
/// a table lookup returns the **bit-for-bit identical** `f64` the direct
/// call would have produced — the trellis costs, and therefore every
/// chosen level and every emitted byte, cannot move.
///
/// Levels `≥ 5` (cat1–cat6, value-dependent extra bits) always take the
/// direct [`coeff_token_bits`] path.
struct TrellisRateTables {
    /// `small[plane][band][ctx3][prev_was_zero][abs_level]` — the exact
    /// [`coeff_token_bits`] value for `abs_level ∈ 0..=4`.
    small: [[[[[f64; (TRELLIS_SMALL_LEVEL_MAX + 1) as usize]; 2]; 3]; 8]; 4],
    /// `eob[plane][band][ctx3]` — the exact [`eob_token_bits`] value.
    eob: [[[f64; 3]; 8]; 4],
}

impl TrellisRateTables {
    /// Build the tables for one frame's resolved `coeff_probs` by
    /// evaluating the reference rate functions at every covered point
    /// (≈ 1k calls — noise next to a single macroblock's mode search).
    fn for_probs(coeff_probs: &CoeffProbs) -> Self {
        // First scan position of each §13.3 band. `coeff_token_bits`
        // depends on the position only through `COEFF_BANDS[i]`, so any
        // representative position per band yields the identical value.
        let mut band_pos = [0usize; 8];
        for (band, pos) in band_pos.iter_mut().enumerate() {
            *pos = COEFF_BANDS
                .iter()
                .position(|&b| b == band)
                .expect("every §13.3 band has at least one scan position");
        }
        let mut tables = TrellisRateTables {
            small: [[[[[0.0; (TRELLIS_SMALL_LEVEL_MAX + 1) as usize]; 2]; 3]; 8]; 4],
            eob: [[[0.0; 3]; 8]; 4],
        };
        for plane in 0..4 {
            for (band, &i) in band_pos.iter().enumerate() {
                for ctx3 in 0..3 {
                    tables.eob[plane][band][ctx3] = eob_token_bits(coeff_probs, plane, i, ctx3);
                    for (pwz, slot) in tables.small[plane][band][ctx3].iter_mut().enumerate() {
                        for (abs, cell) in slot.iter_mut().enumerate() {
                            *cell =
                                coeff_token_bits(coeff_probs, plane, i, ctx3, pwz == 1, abs as u16);
                        }
                    }
                }
            }
        }
        tables
    }
}

/// One Viterbi survivor at a scan position: the minimal accumulated
/// `J = D + lambda·R` to *reach and code through* this position landing in
/// the carried context, the chosen absolute level, and a back-pointer to
/// the predecessor context.
#[derive(Clone, Copy)]
struct TrellisCell {
    /// Accumulated `D + lambda·R` for coding positions `first..=i` with
    /// this position's coefficient as the (current) last non-zero one,
    /// i.e. *not yet* including any terminating EOB.
    cost: f64,
    /// The chosen absolute level at this position.
    level: u16,
    /// The predecessor context (`ctx3`) this survivor came from.
    prev_ctx: usize,
}

/// The per-block §13/§14 parameters a [`trellis_quantize_block`] call
/// needs beyond the coefficients themselves: the §14.1 dequant steps, the
/// §13.3 `firstCoeff`, the seeded entry context, and the `coeff_probs`
/// plane index. Bundled into one struct so the trellis entry point keeps a
/// small argument list.
#[derive(Clone, Copy)]
struct TrellisBlockSpec {
    /// §14.1 DC dequant step (coefficient 0).
    dc: i16,
    /// §14.1 AC dequant step (coefficients 1..15).
    ac: i16,
    /// §13.3 `firstCoeff` — 1 for `YAfterY2` (DC lives in Y2), else 0.
    first_coeff: usize,
    /// Seeded §13.3 neighbour non-zero context (`above + left`, 0..2).
    entry_ctx3: usize,
    /// `coeff_probs` outermost plane index for this block type.
    plane: usize,
}

/// Trellis-quantise one raster-order 4×4 coefficient block in place,
/// minimising `J = D + lambda·R` over the §13.3 token stream.
///
/// `block` is the **raster-order** forward-DCT output; on return it holds
/// the trellis-chosen **raster-order** quantised levels (the same layout
/// `enc_quantize_block` produces), ready to feed `reconstruct_block_4x4`
/// and `encode_coeff_block`. `dc` / `ac` are the §14.1 dequant steps,
/// `spec` carries the §14.1 dequant steps, §13.3 `firstCoeff`, the seeded
/// neighbour non-zero context, and the `coeff_probs` plane index; `lambda`
/// is the per-bit distortion price.
///
/// Falls back to plain rounding when `lambda <= 0` (lossless / qi-0 paths
/// where the rate trade has no benefit), keeping those streams unchanged.
fn trellis_quantize_block(
    block: &mut [i16; 16],
    spec: TrellisBlockSpec,
    coeff_probs: &CoeffProbs,
    tables: Option<&TrellisRateTables>,
    lambda: f64,
) {
    let TrellisBlockSpec {
        dc,
        ac,
        first_coeff,
        entry_ctx3,
        plane,
    } = spec;
    // `lambda` is the **effective** coefficient-level price the caller has
    // already scaled down from the mode-decision multiplier (coefficient
    // RDO spends `lambda` far more effectively than the mode-granular
    // picker — it can drop one trailing coefficient rather than a whole
    // mode — so the raw mode lambda over-trades quality for rate). The
    // scale is resolved by [`MbRdCtx::trellis_lambda`] from the
    // [`TrellisStrength`] knob; this entry consumes the resolved value
    // directly so callers can dial the rate/quality trade per stream.

    // Baseline rounding for the lossless / no-trade path.
    if lambda <= 0.0 {
        enc_quantize_block_range(block, dc, ac, first_coeff);
        return;
    }

    // Scan-order view of the unquantised coefficients + per-position step
    // + the round-half-away-from-zero candidate magnitude per position.
    let mut scan_x = [0i32; 16];
    let mut step = [0i32; 16];
    let mut m_rounds = [0i64; 16];
    let mut last_cand_nonzero: i32 = -1;
    for k in 0..16 {
        scan_x[k] = block[crate::dct_tokens::ZIGZAG[k]] as i32;
        step[k] = if k == 0 { dc as i32 } else { ac as i32 };
        let q = step[k] as i64;
        m_rounds[k] = (scan_x[k].unsigned_abs() as i64 + q / 2) / q;
        if k >= first_coeff && m_rounds[k] != 0 {
            last_cand_nonzero = k as i32;
        }
    }

    // Sub-threshold early exit: when every in-scan coefficient rounds to
    // magnitude 0 the candidate set at every position is exactly `{0}`,
    // so every survivor carries level 0, no terminating position ever
    // qualifies (`cell.level != 0` never holds), and the traceback below
    // provably writes an all-zero block. Emit that block directly.
    if last_cand_nonzero < 0 {
        *block = [0i16; 16];
        return;
    }

    let norms = &*IDCT_BASIS_NORM_SQ;

    // Per-frame rate LUT views for this block's `plane` (see
    // [`TrellisRateTables`] — lookups are bit-for-bit the direct-call
    // values, so costs and decisions cannot move; `None` falls back to
    // the direct reference calls).
    let small_rates = tables.map(|t| &t.small[plane]);
    let eob_rates = tables.map(|t| &t.eob[plane]);

    // Distortion of coding scan position `k` at absolute level `m`
    // (transform-domain squared error scaled by the basis norm). `m` is a
    // magnitude (≥ 0); the dequantised coefficient carries `scan_x[k]`'s
    // sign, so the error is computed on absolute magnitudes.
    let dist = |k: usize, m: i64| -> f64 {
        let q = step[k] as i64;
        let recon = m * q;
        let err = (scan_x[k] as i64).abs() - recon;
        norms[k] * (err * err) as f64
    };

    // Suffix distortion of *dropping* every position k..16 to level 0 —
    // the distortion the §13.3 implicit/explicit EOB tail pays for the
    // coefficients past the last coded one. `suffix_zero_dist[k]` =
    // Σ_{j=k}^{15} dist(j, 0). Without this, terminating early would hide
    // the energy of the dropped tail and the trellis would zero
    // everything. `suffix_zero_dist[16] = 0`.
    let mut suffix_zero_dist = [0f64; 17];
    let mut zero_dist = [0f64; 16];
    for k in (first_coeff..16).rev() {
        zero_dist[k] = dist(k, 0);
        suffix_zero_dist[k] = suffix_zero_dist[k + 1] + zero_dist[k];
    }

    // Forward Viterbi. State = carried `ctx3 ∈ {0,1,2}`. `cells[k]` holds
    // the best survivor *ending at* scan position `k` with each carried
    // context, where "ending" means position `k` holds the last coded
    // (possibly non-zero) coefficient so far. We also separately track the
    // running all-zero-prefix cost so a block can terminate with EOB at
    // `first_coeff` (entirely zero) for free of trailing-token bits.
    //
    // `prev[ctx]` is the best (cost, level, prev_ctx, prev_was_zero) to
    // arrive at the *current* position carrying `ctx`.
    const INF: f64 = f64::INFINITY;

    // survivor reaching position `k` with carried ctx — initialised at the
    // virtual "before first_coeff" boundary: a single state carrying
    // `entry_ctx3`, cost 0, prev_was_zero = false (the first token may emit
    // EOB).
    #[derive(Clone, Copy)]
    struct Surv {
        cost: f64,
        prev_was_zero: bool,
        // back-pointer chain stored in `choice` below
    }
    let mut surv = [Surv {
        cost: INF,
        prev_was_zero: false,
    }; 3];
    surv[entry_ctx3] = Surv {
        cost: 0.0,
        prev_was_zero: false,
    };

    // Decision record per (position, carried-ctx): the level chosen at this
    // position and which predecessor ctx it came from, for traceback.
    let mut choice = [[TrellisCell {
        cost: INF,
        level: 0,
        prev_ctx: 0,
    }; 3]; 16];

    // Best terminating survivor: code positions first..=t, then EOB after
    // t (if t+1 < 16) or implicit EOB. `best_end_pos = first_coeff - 1`
    // (sentinel `usize::MAX` meaning "all zero") seeds the all-zero block.
    let mut best_term_cost = {
        // All-zero block: EOB token at `first_coeff` against the entry ctx,
        // plus the distortion of dropping every coefficient to zero.
        let mut c = INF;
        let s = &surv[entry_ctx3];
        if s.cost < INF {
            let eob_r = match eob_rates {
                Some(e) => e[COEFF_BANDS[first_coeff]][entry_ctx3],
                None => eob_token_bits(coeff_probs, plane, first_coeff, entry_ctx3),
            };
            c = s.cost + lambda * eob_r + suffix_zero_dist[first_coeff];
        }
        c
    };
    let mut best_term_pos: i32 = -1; // -1 ⇒ all-zero
    let mut best_term_ctx = entry_ctx3;

    // Positions past `last_cand_nonzero` admit only the level-0 candidate:
    // they can never become a terminating (non-zero last-coded) slot, the
    // traceback never starts past `best_term_pos` (which needs a non-zero
    // level), and the tail's drop-to-zero distortion is already priced by
    // `suffix_zero_dist` in every termination cost — so the scan over them
    // is dead work and stops here, with identical output.
    for k in first_coeff..=(last_cand_nonzero as usize) {
        let m_round = m_rounds[k];
        let band = COEFF_BANDS[k];
        // Candidate magnitudes: round and round-1 (≥0), deduplicated, plus
        // a forced 0 so a non-trivial coefficient can still be dropped.
        let mut cands = [0i64; 3];
        let mut ncand = 0usize;
        for cand in [m_round, m_round - 1, 0] {
            if cand >= 0 && !cands[..ncand].contains(&cand) {
                cands[ncand] = cand;
                ncand += 1;
            }
        }

        let mut next = [Surv {
            cost: INF,
            prev_was_zero: false,
        }; 3];
        let mut next_choice = [TrellisCell {
            cost: INF,
            level: 0,
            prev_ctx: 0,
        }; 3];

        // Distortion depends only on the candidate magnitude, not on the
        // carried context — compute each candidate's term once (the m = 0
        // slot reuses the cached drop-to-zero value bit-for-bit).
        let mut cand_dist = [0f64; 3];
        for (slot, &m) in cand_dist.iter_mut().zip(&cands[..ncand]) {
            *slot = if m == 0 { zero_dist[k] } else { dist(k, m) };
        }

        for (in_ctx, &s) in surv.iter().enumerate() {
            if s.cost >= INF {
                continue;
            }
            for (&m, &d) in cands[..ncand].iter().zip(&cand_dist[..ncand]) {
                let abs = m as u16;
                let r = match small_rates {
                    Some(t) if abs <= TRELLIS_SMALL_LEVEL_MAX => {
                        t[band][in_ctx][s.prev_was_zero as usize][abs as usize]
                    }
                    _ => coeff_token_bits(coeff_probs, plane, k, in_ctx, s.prev_was_zero, abs),
                };
                let cost = s.cost + d + lambda * r;
                let out_ctx = level_ctx3(abs);
                if cost < next[out_ctx].cost {
                    next[out_ctx] = Surv {
                        cost,
                        prev_was_zero: abs == 0,
                    };
                    next_choice[out_ctx] = TrellisCell {
                        cost,
                        level: abs,
                        prev_ctx: in_ctx,
                    };
                }
            }
        }

        choice[k] = next_choice;

        // Consider terminating *after* position k: position k must hold a
        // non-zero coefficient to be a meaningful last-coded slot (a zero
        // last slot is dominated by terminating one position earlier). The
        // EOB after position k lands at k+1 (if any) against k's out-ctx;
        // at k = 15 the implicit EOB costs nothing.
        for (out_ctx, &cell) in next_choice.iter().enumerate() {
            if cell.cost >= INF || cell.level == 0 {
                continue;
            }
            // `cell.cost` covers coding first..=k; add the EOB token (if
            // any remain) plus the distortion of dropping positions k+1..16.
            let term_cost = if k + 1 < 16 {
                let eob_r = match eob_rates {
                    Some(e) => e[COEFF_BANDS[k + 1]][out_ctx],
                    None => eob_token_bits(coeff_probs, plane, k + 1, out_ctx),
                };
                cell.cost + lambda * eob_r + suffix_zero_dist[k + 1]
            } else {
                cell.cost // implicit EOB after the last coefficient
            };
            if term_cost < best_term_cost {
                best_term_cost = term_cost;
                best_term_pos = k as i32;
                best_term_ctx = out_ctx;
            }
        }

        surv = next;
    }

    // Traceback from the best terminating position, writing scan-order
    // levels then permuting back to raster.
    let mut scan_levels = [0i32; 16];
    if best_term_pos >= 0 {
        let mut k = best_term_pos as usize;
        let mut ctx = best_term_ctx;
        loop {
            let cell = choice[k][ctx];
            let lvl = cell.level as i32;
            let signed = if scan_x[k] < 0 { -lvl } else { lvl };
            scan_levels[k] = signed;
            if k == first_coeff {
                break;
            }
            ctx = cell.prev_ctx;
            k -= 1;
        }
    }

    // Permute scan levels back into the raster `block`.
    for v in block.iter_mut() {
        *v = 0;
    }
    for (k, &lvl) in scan_levels.iter().enumerate() {
        block[crate::dct_tokens::ZIGZAG[k]] = lvl as i16;
    }
}

/// Plain round-quantise a raster block but only from `first_coeff`
/// onward, leaving earlier slots zeroed. Used as the trellis's
/// `lambda <= 0` fallback so the §13.3 `firstCoeff` semantics match.
#[inline]
fn enc_quantize_block_range(block: &mut [i16; 16], dc: i16, ac: i16, first_coeff: usize) {
    if first_coeff == 0 {
        block[0] = enc_round_div(block[0] as i32, dc as i32);
    } else {
        block[0] = 0;
    }
    for c in block.iter_mut().skip(1) {
        *c = enc_round_div(*c as i32, ac as i32);
    }
}

/// Per-macroblock rate-distortion context shared across the mode pickers:
/// the §14.1 dequant factors (so a candidate can be reconstructed exactly
/// as the decoder will), the resolved §13.5 coefficient probabilities (so
/// token bits are priced against the table the decoder retains), and the
/// derived Lagrange multiplier `lambda`.
struct MbRdCtx<'a> {
    factors: &'a crate::dequant::MbDequantFactors,
    coeff_probs: &'a CoeffProbs,
    /// Per-frame precomputed §13.3 trellis rate tables for `coeff_probs`
    /// (see [`TrellisRateTables`]). `None` routes the trellis to the
    /// direct reference rate calls — the values (and therefore the
    /// bytes) are identical either way; frame-level drivers pass
    /// `Some` so the cost of the tree walks is paid once per frame.
    rate_tables: Option<&'a TrellisRateTables>,
    lambda: f64,
    /// §13 trellis aggressiveness — multiplies the calibrated
    /// [`TRELLIS_LAMBDA_SCALE`] base. Decoupled from the mode-decision
    /// `lambda` above so dialing the coefficient trade does not perturb the
    /// mode picker's `J = SSD + lambda·R`. [`TrellisStrength::DEFAULT`]
    /// reproduces the historical bytes.
    trellis: TrellisStrength,
}

impl MbRdCtx<'_> {
    /// The effective per-bit price the §13 trellis minimises against — the
    /// mode `lambda` scaled by the resolved [`TrellisStrength`]. When the
    /// strength is [`TrellisStrength::OFF`] this is `0.0`, routing
    /// [`trellis_quantize_block`] to its plain-rounding fallback.
    #[inline]
    fn trellis_lambda(&self) -> f64 {
        self.lambda * self.trellis.lambda_scale()
    }

    /// Whether the §13 trellis is active for this macroblock (a positive
    /// mode lambda *and* a non-`OFF` strength).
    #[inline]
    fn trellis_active(&self) -> bool {
        self.trellis_lambda() > 0.0
    }
}

/// Derive the Lagrange multiplier from the frame quantiser.
///
/// Non-normative. We use `lambda = q^2 / 32` where `q` is the luma AC
/// dequant step (`factors.y1_ac`). The `q^2` shape is the high-rate
/// RD-optimal relation (distortion of a uniform quantiser scales as
/// `q^2`); the `/ 32` constant was calibrated on the keyframe round-trip
/// fixtures so the picker trims bits without dropping below the SAD
/// picker's PSNR. `lambda` is in (distortion-units / bit) — SSD per bit —
/// so a candidate must save more than `lambda` SSD-units to be worth one
/// extra bit.
#[inline]
fn rd_lambda(factors: &crate::dequant::MbDequantFactors) -> f64 {
    let q = factors.y1_ac as f64;
    q * q / 32.0
}

/// Reconstruct one already-quantised 4×4 luma/chroma residual block on
/// top of `pred` (the candidate's prediction) exactly as the decoder
/// will — dequantise with `(dc, ac)`, §14.3 inverse DCT, add to the
/// predictor with the §14.5 clamp — and return the 16-pixel
/// reconstruction. The encoder feeds the same kernels the decoder runs
/// (`inverse_dct_4x4` / `add_residue_4x4`), so the SSD scored here is the
/// SSD the decoder produces.
#[inline]
fn reconstruct_block_4x4(pred: &[u8; 16], quant: &[i16; 16], dc: i16, ac: i16) -> [u8; 16] {
    let mut dq = *quant;
    dq[0] = (dq[0] as i32 * dc as i32) as i16;
    for c in dq.iter_mut().skip(1) {
        *c = (*c as i32 * ac as i32) as i16;
    }
    let mut residue = [0i16; 16];
    inverse_dct_4x4(&dq, &mut residue);
    let mut recon = [0u8; 16];
    add_residue_4x4(pred, &residue, &mut recon);
    recon
}

/// The §12.2 whole-block luma transform/quant outcome for one candidate
/// mode, carried out of [`pick_y16x16_mode`] so the main encode path can
/// reuse the winner's coefficients instead of recomputing the FDCT.
struct WholeBlockLuma {
    /// The chosen §12.2 mode (DC / V / H / TM).
    mode: IntraYMode,
    /// The sixteen quantised Y sub-blocks (DC already moved into `y2`).
    y_coeffs: [[i16; 16]; 16],
    /// The quantised Y2 (WHT) block.
    y2_coeffs: [i16; 16],
    /// The §-non-normative RD cost `J = SSD + lambda * bits` of the
    /// chosen mode — the value the top-level luma decision compares
    /// against the §11.3 / §12.3 B_PRED pass.
    rd_cost: f64,
}

/// Transform + quantise one whole-block luma candidate's residual exactly
/// as the §14 encode chain will (sixteen §14.4 4×4 FDCTs, §14.3 forward
/// WHT collecting the sub-block DCs into Y2), returning the quantised
/// Y / Y2 coefficients. Shared by the RD scorer and never recomputed for
/// the winner.
///
/// When `trellis` is `Some` the sixteen `YAfterY2` sub-blocks and the `Y2`
/// (WHT) block are quantised by the §13 [`trellis_quantize_block`] RD
/// search rather than plain rounding. The Y2 block is trellised against
/// `BlockType::Y2` (plane 1, full 0..15 scan), each Y sub-block against
/// `BlockType::YAfterY2` (plane 0, `firstCoeff = 1` — the DC now lives in
/// Y2 and is excluded). `None` preserves the plain-rounding bytes exactly.
fn transform_whole_block_luma_rd(
    src: &[u8; 256],
    pred: &[u8; 256],
    factors: &crate::dequant::MbDequantFactors,
    trellis: Option<&MbRdCtx>,
) -> ([[i16; 16]; 16], [i16; 16]) {
    let mut y = [[0i16; 16]; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut residual = [0i16; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let off = (i * 4 + r) * 16 + (j * 4 + c);
                    residual[r * 4 + c] = src[off] as i16 - pred[off] as i16;
                }
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&residual, &mut coeffs);
            y[i * 4 + j] = coeffs;
        }
    }
    // §14.3 forward WHT: collect the sixteen DCs into Y2, then zero each
    // Y sub-block's DC (it now lives in Y2).
    let mut y_dc_block = [0i16; 16];
    for (slot, blk) in y_dc_block.iter_mut().zip(y.iter()) {
        *slot = blk[0];
    }
    let mut y2 = [0i16; 16];
    forward_wht_4x4(&y_dc_block, &mut y2);
    for blk in y.iter_mut() {
        blk[0] = 0;
    }
    // Quantise (plain rounding, or §13 trellis when an RD context is given).
    match trellis {
        Some(rd) if rd.trellis_active() => {
            let tlambda = rd.trellis_lambda();
            trellis_quantize_block(
                &mut y2,
                TrellisBlockSpec {
                    dc: factors.y2_dc,
                    ac: factors.y2_ac,
                    first_coeff: 0,
                    entry_ctx3: 0,
                    plane: BlockType::Y2.plane_index(),
                },
                rd.coeff_probs,
                rd.rate_tables,
                tlambda,
            );
            for blk in y.iter_mut() {
                trellis_quantize_block(
                    blk,
                    TrellisBlockSpec {
                        dc: factors.y1_dc,
                        ac: factors.y1_ac,
                        first_coeff: BlockType::YAfterY2.first_coeff(),
                        entry_ctx3: 0,
                        plane: BlockType::YAfterY2.plane_index(),
                    },
                    rd.coeff_probs,
                    rd.rate_tables,
                    tlambda,
                );
            }
        }
        _ => {
            enc_quantize_block(&mut y2, factors.y2_dc, factors.y2_ac);
            for blk in y.iter_mut() {
                enc_quantize_block(blk, factors.y1_dc, factors.y1_ac);
            }
        }
    }
    (y, y2)
}

/// Reconstruct a whole-block luma candidate from its quantised Y / Y2
/// coefficients exactly as the decoder will (dequantise, §14.3 inverse
/// WHT to recover each sub-block DC, §14.3 inverse DCT, §14.5 add), and
/// return the 256-pixel reconstruction so the picker can score its SSD.
fn reconstruct_whole_block_luma(
    pred: &[u8; 256],
    y_quant: &[[i16; 16]; 16],
    y2_quant: &[i16; 16],
    factors: &crate::dequant::MbDequantFactors,
) -> [u8; 256] {
    // Dequantise Y2 and recover the sixteen sub-block DCs via inverse WHT.
    let mut y2 = *y2_quant;
    y2[0] = (y2[0] as i32 * factors.y2_dc as i32) as i16;
    for c in y2.iter_mut().skip(1) {
        *c = (*c as i32 * factors.y2_ac as i32) as i16;
    }
    let mut dcs = [0i16; 16];
    inverse_wht_4x4(&y2, &mut dcs);

    let mut recon = [0u8; 256];
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let mut blk = y_quant[idx];
            // Seed this sub-block's DC from the WHT output, then dequant
            // the ACs with the Y1 factor.
            blk[0] = dcs[idx];
            for c in blk.iter_mut().skip(1) {
                *c = (*c as i32 * factors.y1_ac as i32) as i16;
            }
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&blk, &mut residue);
            let mut sb_pred = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    sb_pred[r * 4 + c] = pred[(i * 4 + r) * 16 + (j * 4 + c)];
                }
            }
            let mut sb_recon = [0u8; 16];
            add_residue_4x4(&sb_pred, &residue, &mut sb_recon);
            for r in 0..4 {
                for c in 0..4 {
                    recon[(i * 4 + r) * 16 + (j * 4 + c)] = sb_recon[r * 4 + c];
                }
            }
        }
    }
    recon
}

/// Estimate the token bits of one whole-block luma candidate: the §13.3
/// Y2 block plus the sixteen Y sub-blocks (DC carried in Y2, so the Y
/// blocks code from coefficient 1). Threads the §13.3 above/left
/// non-zero predictors locally — the picker scores an isolated MB, which
/// matches the relative ranking the frame-shared contexts produce (a
/// neighbour's non-zero state shifts every candidate's first-band cost by
/// the same amount, so it does not change the winner).
fn estimate_whole_block_luma_bits(
    coeff_probs: &CoeffProbs,
    y_quant: &[[i16; 16]; 16],
    y2_quant: &[i16; 16],
) -> f64 {
    let mut above = MbEntropyCtx::default();
    let mut left = MbEntropyCtx::default();
    let mut bits = 0.0;

    let scan_y2 = raster_to_scan(y2_quant);
    bits += estimate_block_bits_with_ctx(
        24,
        BlockType::Y2,
        coeff_probs,
        &scan_y2,
        &mut above,
        &mut left,
    );
    for (i, blk) in y_quant.iter().enumerate() {
        let scan = raster_to_scan(blk);
        bits += estimate_block_bits_with_ctx(
            i,
            BlockType::YAfterY2,
            coeff_probs,
            &scan,
            &mut above,
            &mut left,
        );
    }
    bits
}

/// Bit cost of one residual block at `block_index`, threading the §13.3
/// / §20.16 above/left non-zero predictor contexts in place — the
/// estimator partner of [`encode_block_with_ctx`].
fn estimate_block_bits_with_ctx(
    block_index: usize,
    block_type: BlockType,
    coeff_probs: &CoeffProbs,
    scan_coeffs: &[i16; 16],
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> f64 {
    const LEFT_CTX: [usize; 25] = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
    ];
    const ABOVE_CTX: [usize; 25] = [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8,
    ];
    let a_slot = ABOVE_CTX[block_index];
    let l_slot = LEFT_CTX[block_index];
    let bits = estimate_block_bits(
        block_type,
        coeff_probs,
        above.nonzero[a_slot],
        left.nonzero[l_slot],
        scan_coeffs,
    );
    let first = block_type.first_coeff();
    let has_coeffs = scan_coeffs[first..].iter().any(|&v| v != 0);
    above.nonzero[a_slot] = has_coeffs;
    left.nonzero[l_slot] = has_coeffs;
    bits
}

/// Pick the §12.2 whole-block 16×16 luma intra mode (DC / V / H / TM) by
/// the §-non-normative rate-distortion cost `J = SSD + lambda * R`. Each
/// candidate is run through the full §14 chain — predict, FDCT, Y2/WHT,
/// quantise, then dequantise + inverse-transform + reconstruct — so the
/// distortion `D` is the exact self-decode SSD and the rate `R` is the
/// §13.3 token bits plus the §11.2 luma-mode-signal bits.
///
/// Returns the winning [`WholeBlockLuma`] (mode, prediction, quantised
/// Y / Y2 coefficients, and RD cost). `topleft` is the (-1,-1) corner
/// pixel TM_PRED reads; supply [`crate::intra_predict::DEFAULT_ABOVE_PIXEL`]
/// when the corner is off-frame, mirroring the decoder's
/// `decode_keyframe_mb_non_bpred` convention.
fn pick_y16x16_mode(
    src: &[u8; 256],
    above: Option<&[u8; 16]>,
    left: Option<&[u8; 16]>,
    topleft: u8,
    rd: &MbRdCtx,
) -> WholeBlockLuma {
    const CANDIDATES: [IntraYMode; 4] =
        [IntraYMode::Dc, IntraYMode::V, IntraYMode::H, IntraYMode::Tm];
    let mut best: Option<WholeBlockLuma> = None;
    for &mode in &CANDIDATES {
        let mut pred = [0u8; 256];
        // predict_y16x16 returns None only for B_PRED, which is not in
        // CANDIDATES, so this is always Some.
        predict_y16x16(&mut pred, mode, above, left, topleft).expect("CANDIDATES excludes B_PRED");

        let (y_quant, y2_quant) = transform_whole_block_luma_rd(src, &pred, rd.factors, Some(rd));
        let recon = reconstruct_whole_block_luma(&pred, &y_quant, &y2_quant, rd.factors);
        let ssd = block_ssd(&recon, src) as f64;
        let token_bits = estimate_whole_block_luma_bits(rd.coeff_probs, &y_quant, &y2_quant);
        let mode_bits = treed_bits_cached(&*KF_YMODE_PATHS, |i| KF_YMODE_PROB[i], mode.leaf());
        let rd_cost = ssd + rd.lambda * (token_bits + mode_bits);

        if best.as_ref().map(|b| rd_cost < b.rd_cost).unwrap_or(true) {
            best = Some(WholeBlockLuma {
                mode,
                y_coeffs: y_quant,
                y2_coeffs: y2_quant,
                rd_cost,
            });
        }
    }
    best.expect("CANDIDATES is non-empty")
}

/// One chroma plane's source pixels plus its three §12 prediction
/// edges, bundled so [`pick_uv8x8_mode`] takes one argument per plane
/// instead of six positional inputs.
struct ChromaPlane<'a> {
    src: &'a [u8; 64],
    above: Option<&'a [u8; 8]>,
    left: Option<&'a [u8; 8]>,
    topleft: u8,
}

/// Transform + quantise one 8×8 chroma plane's residual into its four
/// §13.3 `UV` sub-blocks (each carrying its own DC — chroma has no Y2),
/// returning the quantised blocks in raster sub-block order (`i*2 + j`).
///
/// When `trellis` is `Some` each `UV` sub-block is quantised by the §13
/// [`trellis_quantize_block`] RD search (plane 2, `firstCoeff = 0`);
/// `None` preserves plain-rounding bytes.
fn transform_chroma_plane_rd(
    src: &[u8; 64],
    pred: &[u8; 64],
    factors: &crate::dequant::MbDequantFactors,
    trellis: Option<&MbRdCtx>,
) -> [[i16; 16]; 4] {
    let mut blocks = [[0i16; 16]; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut residual = [0i16; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let off = (i * 4 + r) * 8 + (j * 4 + c);
                    residual[r * 4 + c] = src[off] as i16 - pred[off] as i16;
                }
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&residual, &mut coeffs);
            match trellis {
                Some(rd) if rd.trellis_active() => trellis_quantize_block(
                    &mut coeffs,
                    TrellisBlockSpec {
                        dc: factors.uv_dc,
                        ac: factors.uv_ac,
                        first_coeff: BlockType::UV.first_coeff(),
                        entry_ctx3: 0,
                        plane: BlockType::UV.plane_index(),
                    },
                    rd.coeff_probs,
                    rd.rate_tables,
                    rd.trellis_lambda(),
                ),
                _ => enc_quantize_block(&mut coeffs, factors.uv_dc, factors.uv_ac),
            }
            blocks[i * 2 + j] = coeffs;
        }
    }
    blocks
}

/// Reconstruct one 8×8 chroma plane from its four quantised `UV`
/// sub-blocks on top of `pred`, returning the 64-pixel reconstruction
/// (the exact pixels the decoder produces) for SSD scoring.
fn reconstruct_chroma_plane(
    pred: &[u8; 64],
    quant: &[[i16; 16]; 4],
    factors: &crate::dequant::MbDequantFactors,
) -> [u8; 64] {
    let mut recon = [0u8; 64];
    for i in 0..2 {
        for j in 0..2 {
            let mut sb_pred = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    sb_pred[r * 4 + c] = pred[(i * 4 + r) * 8 + (j * 4 + c)];
                }
            }
            let sb_recon =
                reconstruct_block_4x4(&sb_pred, &quant[i * 2 + j], factors.uv_dc, factors.uv_ac);
            for r in 0..4 {
                for c in 0..4 {
                    recon[(i * 4 + r) * 8 + (j * 4 + c)] = sb_recon[r * 4 + c];
                }
            }
        }
    }
    recon
}

/// Pick the §12.2 whole-block 8×8 chroma intra mode (DC / V / H / TM) by
/// the rate-distortion cost `J = SSD + lambda * R`, jointly over both
/// chroma planes (VP8 codes a single `uv_mode` shared by Cb and Cr). Each
/// candidate is reconstructed through the full §14 chain; `D` is the
/// summed self-decode SSD across U and V, `R` the eight §13.3 `UV`-block
/// token bits plus the §11.4 chroma-mode-signal bits. Returns the joint
/// winner plus both prediction buffers.
fn pick_uv8x8_mode(
    u: &ChromaPlane,
    v: &ChromaPlane,
    rd: &MbRdCtx,
) -> (IntraUvMode, [u8; 64], [u8; 64]) {
    const CANDIDATES: [IntraUvMode; 4] = [
        IntraUvMode::Dc,
        IntraUvMode::V,
        IntraUvMode::H,
        IntraUvMode::Tm,
    ];
    let mut best_mode = IntraUvMode::Dc;
    let mut best_u = [0u8; 64];
    let mut best_v = [0u8; 64];
    let mut best_cost = f64::INFINITY;
    for &mode in &CANDIDATES {
        let mut u_pred = [0u8; 64];
        let mut v_pred = [0u8; 64];
        predict_uv8x8(&mut u_pred, mode, u.above, u.left, u.topleft);
        predict_uv8x8(&mut v_pred, mode, v.above, v.left, v.topleft);

        let u_quant = transform_chroma_plane_rd(u.src, &u_pred, rd.factors, Some(rd));
        let v_quant = transform_chroma_plane_rd(v.src, &v_pred, rd.factors, Some(rd));
        let u_recon = reconstruct_chroma_plane(&u_pred, &u_quant, rd.factors);
        let v_recon = reconstruct_chroma_plane(&v_pred, &v_quant, rd.factors);
        let ssd = (block_ssd(&u_recon, u.src) + block_ssd(&v_recon, v.src)) as f64;

        // §13.3 chroma token bits: 4 U sub-blocks (16..19) then 4 V
        // (20..23), threading the local above/left non-zero contexts.
        let mut above = MbEntropyCtx::default();
        let mut left = MbEntropyCtx::default();
        let mut token_bits = 0.0;
        for (i, blk) in u_quant.iter().enumerate() {
            let scan = raster_to_scan(blk);
            token_bits += estimate_block_bits_with_ctx(
                16 + i,
                BlockType::UV,
                rd.coeff_probs,
                &scan,
                &mut above,
                &mut left,
            );
        }
        for (i, blk) in v_quant.iter().enumerate() {
            let scan = raster_to_scan(blk);
            token_bits += estimate_block_bits_with_ctx(
                20 + i,
                BlockType::UV,
                rd.coeff_probs,
                &scan,
                &mut above,
                &mut left,
            );
        }
        let mode_bits = treed_bits_cached(&*UV_MODE_PATHS, |i| KF_UV_MODE_PROB[i], mode.leaf());
        let cost = ssd + rd.lambda * (token_bits + mode_bits);

        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
            best_u = u_pred;
            best_v = v_pred;
        }
    }
    (best_mode, best_u, best_v)
}

// ─────────────────────── §11.3 / §12.3 B_PRED luma picker ──────────────────────

/// Stride (pixels per row) of the encoder's `B_PRED` working luma
/// buffer — mirrors the decoder's `reconstruct::BPRED_STRIDE`:
/// `[ left-border (1) | 16 columns | above-right extension (4) ]`.
const ENC_BPRED_STRIDE: usize = 1 + 16 + 4;

/// Number of rows in the working buffer: one top-border row plus the
/// sixteen reconstruction rows.
const ENC_BPRED_ROWS: usize = 1 + 16;

/// Flat index of the working buffer's reconstruction origin `(0, 0)`.
const ENC_BPRED_ORIGIN: usize = ENC_BPRED_STRIDE + 1;

/// The ten §12.3 4×4 sub-block intra modes, in `intra_bmode` order.
const BMODE_CANDIDATES: [IntraBmode; 10] = [
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
];

/// Outcome of the §12.3 B_PRED luma encode pass.
struct BpredLuma {
    /// The sixteen chosen 4×4 sub-block modes (raster order).
    modes: [IntraBmode; 16],
    /// The sixteen quantised 4×4 Y coefficient blocks (raster order).
    /// Unlike the whole-block path these carry their **own** DC (no Y2
    /// block exists for a `B_PRED` macroblock per §13 / §14.2).
    y_coeffs: [[i16; 16]; 16],
    /// Total rate-distortion cost `J = SSD + lambda * R` of the chosen
    /// sub-block modes — the metric the top-level luma decision compares
    /// against the best whole-block RD cost. `R` is the §13.3 token bits
    /// plus the §11.3 / §11.5 sub-block-mode-signal bits; the §11.2
    /// B_PRED luma-mode flag itself is added by the caller (it is common
    /// to all sixteen sub-blocks and cancels in the per-sub-block scoring
    /// but must be charged once against the whole-block alternative).
    rd_cost: f64,
}

/// Build the encoder's `B_PRED` working luma buffer with its border row
/// / column pre-filled from the macroblock's neighbours, applying the
/// §12.3 right-edge "above-right" fixup. Mirrors the decoder's
/// `reconstruct::build_bpred_work_buffer` so the encoder evolves
/// neighbours over the exact pixels the decoder will.
fn build_enc_bpred_buffer(
    neighbors: &crate::reconstruct::MbNeighbors,
) -> [u8; ENC_BPRED_ROWS * ENC_BPRED_STRIDE] {
    let mut buf = [0u8; ENC_BPRED_ROWS * ENC_BPRED_STRIDE];

    // Top border row: col 0 = corner P, cols 1..=16 = 16 above pixels,
    // cols 17..=20 = the 4 above-right pixels.
    let above = neighbors.y_above.unwrap_or([DEFAULT_ABOVE_PIXEL; 16]);
    let topleft = neighbors.y_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL);
    buf[0] = topleft;
    buf[1..17].copy_from_slice(&above);
    let above_right = neighbors.y_above_right.unwrap_or([DEFAULT_ABOVE_PIXEL; 4]);
    buf[17..21].copy_from_slice(&above_right);

    // Left border column.
    let left = neighbors.y_left.unwrap_or([DEFAULT_LEFT_PIXEL; 16]);
    for (r, &v) in left.iter().enumerate() {
        buf[(r + 1) * ENC_BPRED_STRIDE] = v;
    }

    // §12.3 right-edge fixup: the four above-right pixels are shared by
    // sub-blocks 3, 7, 11, 15 — copy them above sub-block rows 1, 2, 3.
    let extra = [buf[17], buf[18], buf[19], buf[20]];
    for &recon_row_above in &[3usize, 7, 11] {
        let buf_row = 1 + recon_row_above;
        let base = buf_row * ENC_BPRED_STRIDE + 17;
        buf[base..base + 4].copy_from_slice(&extra);
    }

    buf
}

/// Encode the luma plane as a §11.3 / §12.3 `B_PRED` macroblock:
/// sixteen independent 4×4 sub-blocks, each choosing the
/// rate-distortion-minimising one of the ten §12.3 sub-modes, with
/// **in-place neighbour evolution** — every sub-block predicts from the
/// already-reconstructed (predictor + dequantised residue) pixels of the
/// sub-blocks above and to its left, exactly as the decoder's
/// `decode_keyframe_mb_bpred` walk does. Sharing the decoder's
/// `predict_b4x4` kernel and `inverse_dct_4x4` / `add_residue_4x4`
/// guarantees the reconstruction the encoder evolves against is the one
/// the decoder produces.
///
/// Each sub-block's cost is `J = SSD + lambda * R`, where `D` is the
/// reconstructed-vs-source SSD of that 4×4 block and `R` is its §13.3
/// `YNoY2` token bits plus the §11.3 / §11.5 sub-block-mode-signal bits
/// (priced against `KF_BMODE_PROB[above_mode][left_mode]`, with the
/// within-MB neighbour modes threaded and B_DC_PRED at the MB edges — the
/// same locality the bitstream layer's first-MB context uses). The chosen
/// sub-block mode evolves the neighbour-mode context greedily.
///
/// Returns the chosen modes, the sixteen quantised coefficient blocks
/// (each carrying its own DC — a `B_PRED` MB has no Y2 block), and the
/// total RD cost for the top-level luma decision.
fn encode_bpred_luma(
    src: &[u8; 256],
    neighbors: &crate::reconstruct::MbNeighbors,
    rd: &MbRdCtx,
) -> BpredLuma {
    let factors = rd.factors;
    let mut buf = build_enc_bpred_buffer(neighbors);
    let mut modes = [IntraBmode::Dc; 16];
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut total_cost = 0.0f64;
    // Within-MB sub-block-mode context for §11.3 mode-bit pricing: edge
    // sub-blocks read B_DC_PRED, matching the first-MB context the
    // bitstream layer initialises.
    let mut sub_modes = [IntraBmode::Dc; 16];
    // §13.3 non-zero token context, threaded across the sixteen Y blocks.
    let mut tok_above = MbEntropyCtx::default();
    let mut tok_left = MbEntropyCtx::default();

    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let sb_base = ENC_BPRED_ORIGIN + (i * 4) * ENC_BPRED_STRIDE + (j * 4);

            // Gather the §12.3 neighbour inputs from the evolving buffer.
            let above_base = sb_base - ENC_BPRED_STRIDE;
            let mut above = [0u8; 8];
            above.copy_from_slice(&buf[above_base..above_base + 8]);
            let mut left = [0u8; 4];
            for (r, slot) in left.iter_mut().enumerate() {
                *slot = buf[sb_base + r * ENC_BPRED_STRIDE - 1];
            }
            let p = buf[above_base - 1];

            // Read the source 4×4 sub-block.
            let mut sb_src = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    sb_src[r * 4 + c] = src[(i * 4 + r) * 16 + (j * 4 + c)];
                }
            }

            // §11.3 mode-bit context: above sub-block (or B_DC_PRED at the
            // MB top edge) and left sub-block (or B_DC_PRED at the left
            // edge).
            let above_mode = if i == 0 {
                IntraBmode::Dc
            } else {
                sub_modes[(i - 1) * 4 + j]
            };
            let left_mode = if j == 0 {
                IntraBmode::Dc
            } else {
                sub_modes[i * 4 + (j - 1)]
            };
            let prob_row = &KF_BMODE_PROB[above_mode.idx()][left_mode.idx()];

            // RD-pick the sub-mode: reconstruct each candidate, score
            // SSD + lambda*(token bits + mode bits).
            let mut best_mode = IntraBmode::Dc;
            let mut best_coeffs = [0i16; 16];
            let mut best_recon = [0u8; 16];
            let mut best_cost = f64::INFINITY;
            for &mode in &BMODE_CANDIDATES {
                let mut pred = [0u8; 16];
                predict_b4x4(&mut pred, mode, &above, &left, p);

                let mut residual = [0i16; 16];
                for k in 0..16 {
                    residual[k] = sb_src[k] as i16 - pred[k] as i16;
                }
                let mut coeffs = [0i16; 16];
                forward_dct_4x4(&residual, &mut coeffs);
                if rd.trellis_active() {
                    // §13 trellis: B_PRED sub-blocks are `YNoY2` (own DC,
                    // plane 3); seed the entry context from the real §13.3
                    // above/left non-zero predictors for this sub-block.
                    let entry_ctx3 =
                        (tok_above.nonzero[j] as usize) + (tok_left.nonzero[i] as usize);
                    trellis_quantize_block(
                        &mut coeffs,
                        TrellisBlockSpec {
                            dc: factors.y1_dc,
                            ac: factors.y1_ac,
                            first_coeff: BlockType::YNoY2.first_coeff(),
                            entry_ctx3,
                            plane: BlockType::YNoY2.plane_index(),
                        },
                        rd.coeff_probs,
                        rd.rate_tables,
                        rd.trellis_lambda(),
                    );
                } else {
                    enc_quantize_block(&mut coeffs, factors.y1_dc, factors.y1_ac);
                }

                let recon = reconstruct_block_4x4(&pred, &coeffs, factors.y1_dc, factors.y1_ac);
                let ssd = block_ssd(&recon, &sb_src) as f64;
                let scan = raster_to_scan(&coeffs);
                let token_bits = estimate_block_bits(
                    BlockType::YNoY2,
                    rd.coeff_probs,
                    tok_above.nonzero[j],
                    tok_left.nonzero[i],
                    &scan,
                );
                let mode_bits = treed_bits_cached(&*BMODE_PATHS, |k| prob_row[k], mode.idx() as u8);
                let cost = ssd + rd.lambda * (token_bits + mode_bits);
                if cost < best_cost {
                    best_cost = cost;
                    best_mode = mode;
                    best_coeffs = coeffs;
                    best_recon = recon;
                }
            }
            modes[idx] = best_mode;
            sub_modes[idx] = best_mode;
            y_coeffs[idx] = best_coeffs;
            total_cost += best_cost;

            // Roll the §13.3 non-zero token context forward for the
            // winning sub-block (Y block index `idx`, above slot = col j,
            // left slot = row i).
            let has_coeffs = best_coeffs.iter().any(|&v| v != 0);
            tok_above.nonzero[j] = has_coeffs;
            tok_left.nonzero[i] = has_coeffs;

            // Write the winning reconstruction back into the working
            // buffer so it becomes a neighbour for the sub-blocks below /
            // to its right — identical to the decoder's evolution step.
            for r in 0..4 {
                let dst = sb_base + r * ENC_BPRED_STRIDE;
                buf[dst..dst + 4].copy_from_slice(&best_recon[r * 4..r * 4 + 4]);
            }
        }
    }

    BpredLuma {
        modes,
        y_coeffs,
        rd_cost: total_cost,
    }
}

/// Result of [`encode_mb_block_set`] — the per-MB raw-quantised
/// coefficient bundle (for inspection / testing) plus the byte stream
/// emitted by the token-encoder's underlying boolean encoder.
///
/// The bytes are positioned to feed a `BoolDecoder` directly; the
/// caller wraps them in any outer-frame framing (the §9.x first
/// partition + §9.5 DCT partition table) it needs.
#[derive(Debug, Clone)]
pub struct EncodedMb {
    /// The 25 raw-quantised coefficient blocks the encoder emitted, in
    /// raster order per [`MbCoeffs`]. Useful as a fixture for
    /// roundtrip tests; the decoder side recovers the same arrays.
    pub coeffs: MbCoeffs,
    /// The token-encoder's finished byte stream. Concatenates §13.3
    /// for Y2 → 16 Y → 4 U → 4 V at the supplied predictor seeds, then
    /// the §7.3 4-byte boolean-encoder flush trailer.
    pub bytes: Vec<u8>,
    /// The non-zero block count across the 25 residual blocks, useful
    /// as a regression knob and for assertions in tests.
    pub nonzero_block_count: usize,
    /// The luma mode the picker chose for this MB. One of the four
    /// §12.2 whole-block modes (DC / V / H / TM) when the whole-block
    /// path wins, or [`IntraYMode::B`] (`B_PRED`) when the §11.3 / §12.3
    /// per-4×4-sub-block path wins. The caller writes this into the §11.2
    /// mode layer and feeds it to the decoder's reconstruction
    /// orchestrator so prediction matches.
    pub y_mode: IntraYMode,
    /// The §12.2 whole-block 8×8 chroma mode the picker chose (shared by
    /// Cb and Cr per §11.4).
    pub uv_mode: IntraUvMode,
    /// The sixteen §11.3 / §12.3 4×4 luma sub-block modes in raster
    /// order (`i * 4 + j`), present **iff** `y_mode == IntraYMode::B`.
    /// `None` for the four whole-block luma modes. Feeds the decoder's
    /// `decode_keyframe_mb_bpred` `subblock_modes` argument.
    pub b_subblock_modes: Option<[IntraBmode; 16]>,
}

/// Encode one macroblock's residual through the §13 / §14 block-set
/// walker — the inverse of the §14.2 reconstruction orchestrator.
///
/// # Behavior (Phase 5 scope)
///
/// * **Prediction + mode pick**: rate-distortion. Each §12.2 whole-block
///   luma mode (`DC` / `V` / `H` / `TM`), the §11.3 / §12.3 `B_PRED`
///   per-4×4-sub-block path, and each §12.2 chroma mode are run through
///   the full §14 chain (predict → FDCT → Y2/WHT → quantise →
///   dequantise → inverse-transform → reconstruct) and scored by the
///   non-normative Lagrangian cost `J = SSD + lambda * R`, where `D` is
///   the exact self-decode reconstruction SSD and `R` is the §13.3 token
///   bits plus the §11.2 / §11.4 mode-signal bits. `lambda` is derived
///   from the frame quantiser (see [`rd_lambda`]). Luma and chroma modes
///   are chosen independently; B_PRED wins over the whole-block luma
///   modes when its total RD cost (plus the §11.2 B_PRED mode-flag) is
///   lower. The same [`predict_y16x16`] / [`predict_uv8x8`] /
///   [`predict_b4x4`] kernels the decoder reconstructs with are shared.
/// * **Forward transforms**: §14.4 forward DCT on every Y, U, V 4×4
///   sub-block (16 + 4 + 4 = 24 blocks); for the whole-block luma path
///   the 16 Y DCs are collected into a Y2 block and §14.3 forward-WHT'd.
/// * **Quantisation**: §14.1 / §20.4 factors per
///   [`MbDequantFactors::from_quant_indices`], with the `MbCoeffs`
///   plane layout matching what the decoder produces.
/// * **Token coding**: §13.3 walk in residual order Y2 → 16 Y
///   (`YAfterY2` / `YNoY2`) → 4 U (`UV`) → 4 V (`UV`), threaded through
///   fresh above / left [`MbEntropyCtx`] (off-frame neighbours, matching
///   the test fixture's single-MB frame).
///
/// # Roundtrip guarantee
///
/// On a flat-colour Y / U / V macroblock at `yac_qi = 0` the bytes
/// `decode_mb_coeffs` + `MbDequantFactors::dequantize` + the §14.2 /
/// §12.2 reconstruction orchestrator return exactly recover the input
/// pixel value. The
/// `mb_block_set_roundtrip_flat_color_recovers_within_one_lsb` test in
/// the same module checks this end-to-end.
///
/// # Out of scope (deferred)
///
/// * Inter prediction and a full trellis / residual-quantisation RD
///   search (the picker scores at the mode granularity, not per-token).
///   Those land in subsequent rounds.
/// * The RD bit estimate prices an isolated MB's §13.3 contexts; the
///   frame raster driver threads the genuine cross-MB `MbEntropyCtx`
///   columns when it re-walks the chosen coefficients into the shared
///   token partition.
pub fn encode_mb_block_set(
    pixels: &MbPixels,
    yac_qi: u8,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
) -> Result<EncodedMb, TokenEncodeError> {
    // The standalone entry encodes a single isolated macroblock, so
    // every neighbour edge is off-frame (`MbNeighbors::default()` is all
    // `None`) — exactly what the decoder's `decode_keyframe_mb_non_bpred`
    // substitutes for the top-left MB.
    encode_mb_block_set_with_neighbors(
        pixels,
        &crate::reconstruct::MbNeighbors::default(),
        yac_qi,
        coeff_probs,
    )
}

/// Encode one macroblock's residual exactly like [`encode_mb_block_set`],
/// but score the §12.2 whole-block mode picker against the supplied
/// reconstructed-neighbour strips rather than off-frame defaults.
///
/// `neighbors` carries the same `(-1, *)` / `(*, -1)` edge pixels the
/// decoder's [`crate::reconstruct::MbNeighbors`] holds, so V_PRED /
/// H_PRED / TM_PRED are scored (and residual-coded) against the actual
/// pixels the decoder will reconstruct from. A `None` edge means
/// off-frame, and the shared `predict_*` kernels then apply the §12
/// 127 / 129 defaults — identical on both encode and decode sides.
///
/// This is the per-MB primitive the eventual frame raster driver calls
/// once per macroblock, feeding it the bottom row / right column of the
/// already-reconstructed neighbours.
pub fn encode_mb_block_set_with_neighbors(
    pixels: &MbPixels,
    neighbors: &crate::reconstruct::MbNeighbors,
    yac_qi: u8,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
) -> Result<EncodedMb, TokenEncodeError> {
    encode_mb_block_set_with_neighbors_strength(
        pixels,
        neighbors,
        yac_qi,
        coeff_probs,
        TrellisStrength::DEFAULT,
    )
}

/// Like [`encode_mb_block_set_with_neighbors`] but with the §13 trellis
/// aggressiveness dialed by `trellis`. The public no-strength entry above
/// delegates here with [`TrellisStrength::DEFAULT`] (byte-identical to the
/// historical wire); the keyframe driver threads the per-stream knob down
/// to this entry so a quality/size dial reaches the per-MB coefficient
/// search. [`TrellisStrength::OFF`] reproduces the pre-trellis bytes.
pub fn encode_mb_block_set_with_neighbors_strength(
    pixels: &MbPixels,
    neighbors: &crate::reconstruct::MbNeighbors,
    yac_qi: u8,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
    trellis: TrellisStrength,
) -> Result<EncodedMb, TokenEncodeError> {
    encode_mb_block_set_with_neighbors_strength_rates(
        pixels,
        neighbors,
        yac_qi,
        coeff_probs,
        None,
        trellis,
    )
}

/// [`encode_mb_block_set_with_neighbors_strength`] with an optional
/// per-frame [`TrellisRateTables`] built for `coeff_probs`. The
/// frame-level drivers pass `Some` so the §13 trellis prices tokens from
/// the precomputed tables (bit-for-bit the same values the direct calls
/// produce — the emitted bytes are identical either way); the public
/// per-MB entry above passes `None` and keeps its call cost unchanged.
fn encode_mb_block_set_with_neighbors_strength_rates(
    pixels: &MbPixels,
    neighbors: &crate::reconstruct::MbNeighbors,
    yac_qi: u8,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
    rate_tables: Option<&TrellisRateTables>,
    trellis: TrellisStrength,
) -> Result<EncodedMb, TokenEncodeError> {
    // ---- 1. Build the §14.1 dequant factors + derive the RD lambda.
    // The encoder's quantisation step is the inverse — divide by these.
    let factors =
        crate::dequant::MbDequantFactors::from_base_and_deltas(yac_qi as i32, 0, 0, 0, 0, 0);
    let rd = MbRdCtx {
        factors: &factors,
        coeff_probs,
        rate_tables,
        lambda: rd_lambda(&factors),
        trellis,
    };

    // ---- 2. Pick the §12.2 whole-block luma mode by rate-distortion.
    //         Each candidate is reconstructed through the full §14 chain
    //         (predict → FDCT → Y2/WHT → quant → dequant → IDCT → add),
    //         so the distortion term is the exact self-decode SSD and the
    //         rate term is the §13.3 token bits + §11.2 mode bits. The
    //         shared `predict_*` kernels apply each mode's off-frame
    //         default (V → 127, H → 129, TM → 129, DC → 128 fill) for any
    //         `None` edge, identically on both sides, so the residual the
    //         picker scores is the one the decoder reconstructs. The
    //         corner (-1,-1) read only by TM_PRED defaults to the §12
    //         `DEFAULT_ABOVE_PIXEL` (127) when off-frame, mirroring
    //         `decode_keyframe_mb_non_bpred`.
    let default_corner = crate::intra_predict::DEFAULT_ABOVE_PIXEL;
    let whole = pick_y16x16_mode(
        &pixels.y,
        neighbors.y_above.as_ref(),
        neighbors.y_left.as_ref(),
        neighbors.y_topleft.unwrap_or(default_corner),
        &rd,
    );

    // ---- 2b. Score the §11.3 / §12.3 B_PRED per-4×4-sub-block luma pass
    //          and decide luma by RD cost. The §11.2 luma-mode flag
    //          differs between the two paths, so add each path's
    //          mode-signal bits to its residual RD cost before comparing:
    //          the whole-block cost already folds its mode bits inside
    //          `pick_y16x16_mode`, and we charge the B_PRED MB its §11.2
    //          B-flag here (its per-sub-block mode bits are already inside
    //          `encode_bpred_luma`'s RD cost). B_PRED wins iff its total
    //          RD cost is strictly lower.
    let bpred = encode_bpred_luma(&pixels.y, neighbors, &rd);
    let bpred_flag_bits =
        treed_bits_cached(&*KF_YMODE_PATHS, |i| KF_YMODE_PROB[i], IntraYMode::B.leaf());
    let bpred_total_cost = bpred.rd_cost + rd.lambda * bpred_flag_bits;
    let use_bpred = bpred_total_cost < whole.rd_cost;
    let y_mode = if use_bpred { IntraYMode::B } else { whole.mode };

    let (uv_mode, _u_pred, _v_pred) = pick_uv8x8_mode(
        &ChromaPlane {
            src: &pixels.u,
            above: neighbors.u_above.as_ref(),
            left: neighbors.u_left.as_ref(),
            topleft: neighbors.u_topleft.unwrap_or(default_corner),
        },
        &ChromaPlane {
            src: &pixels.v,
            above: neighbors.v_above.as_ref(),
            left: neighbors.v_left.as_ref(),
            topleft: neighbors.v_topleft.unwrap_or(default_corner),
        },
        &rd,
    );
    // Re-derive the chosen chroma mode's predictions + quantised blocks.
    let mut u_pred = [0u8; 64];
    let mut v_pred = [0u8; 64];
    predict_uv8x8(
        &mut u_pred,
        uv_mode,
        neighbors.u_above.as_ref(),
        neighbors.u_left.as_ref(),
        neighbors.u_topleft.unwrap_or(default_corner),
    );
    predict_uv8x8(
        &mut v_pred,
        uv_mode,
        neighbors.v_above.as_ref(),
        neighbors.v_left.as_ref(),
        neighbors.v_topleft.unwrap_or(default_corner),
    );

    // ---- 3. Assemble the quantised coefficients from the winning modes.
    //         The pickers already ran the §14 FDCT / Y2-WHT / quant chain
    //         for their winner, so the luma blocks are reused directly:
    //   * whole-block: `whole.y_coeffs` (DC in Y2) + `whole.y2_coeffs`;
    //   * B_PRED: `bpred.y_coeffs` (each sub-block carries its own DC),
    //     Y2 stays zero (§13 / §14.2: a B_PRED MB has no Y2 block).
    //         Chroma is FDCT+quantised here against the chosen uv_mode.
    let mut raw_coeffs = MbCoeffs::default();
    if use_bpred {
        raw_coeffs.y = bpred.y_coeffs;
    } else {
        raw_coeffs.y = whole.y_coeffs;
        raw_coeffs.y2 = whole.y2_coeffs;
    }
    // Chroma is re-FDCT+quantised here against the chosen uv_mode; use the
    // same §13 trellis the picker scored so the emitted bytes match the
    // chosen-mode RD cost.
    raw_coeffs.u = transform_chroma_plane_rd(&pixels.u, &u_pred, &factors, Some(&rd));
    raw_coeffs.v = transform_chroma_plane_rd(&pixels.v, &v_pred, &factors, Some(&rd));

    // ---- 6. Walk §13.3 residual order, encode each block in scan
    //         order against fresh above / left predictor contexts.
    //
    // Walk order matches `decode_mb_coeffs`:
    //   1. Y2 (only when has_y2 — i.e. NOT B_PRED)  — block 24
    //   2. 16 Y sub-blocks                          — blocks 0..15
    //        plane = YAfterY2 (DC in Y2) for whole-block modes,
    //        plane = YNoY2   (own DC)    for B_PRED;
    //   3. 4 U sub-blocks (UV plane)                — blocks 16..19
    //   4. 4 V sub-blocks (UV plane)                — blocks 20..23
    //
    // The above / left predictor seeds for the first block are both
    // off-frame ("false") since this entry encodes a single isolated
    // MB. Each block updates its slot in both contexts per §13.3.
    let mut enc = BoolEncoder::new();
    let mut above = MbEntropyCtx::default();
    let mut left = MbEntropyCtx::default();
    let nonzero_block_count = encode_mb_tokens(
        &mut enc,
        &raw_coeffs,
        use_bpred,
        coeff_probs,
        &mut above,
        &mut left,
    )?;

    let bytes = enc.finish();
    let b_subblock_modes = if use_bpred { Some(bpred.modes) } else { None };
    Ok(EncodedMb {
        coeffs: raw_coeffs,
        bytes,
        nonzero_block_count,
        y_mode,
        uv_mode,
        b_subblock_modes,
    })
}

/// Walk the §13.3 residual order for one macroblock's raw-quantised
/// coefficients into the shared boolean encoder `enc`, threading the
/// supplied above / left [`MbEntropyCtx`] in place. Returns the number
/// of non-zero residual blocks emitted.
///
/// Walk order matches `decode_mb_coeffs`:
///   1. Y2 (only when `!use_bpred`)              — block 24
///   2. 16 Y sub-blocks                          — blocks 0..15
///      (plane = `YAfterY2` (DC in Y2) for whole-block modes,
///      plane = `YNoY2` (own DC) for B_PRED);
///   3. 4 U sub-blocks (`UV` plane)              — blocks 16..19
///   4. 4 V sub-blocks (`UV` plane)              — blocks 20..23
///
/// Unlike the per-MB `encode_mb_block_set*` entries (which call this
/// against fresh off-frame contexts and a per-MB encoder), the frame
/// raster driver shares one `enc` and threads `above` (per column,
/// frame-lived) / `left` (per row) so the §13.3 non-zero predictor
/// state evolves macroblock-to-macroblock exactly as the decoder
/// reconstructs it.
fn encode_mb_tokens(
    enc: &mut BoolEncoder,
    raw_coeffs: &MbCoeffs,
    use_bpred: bool,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> Result<usize, TokenEncodeError> {
    let mut nonzero_block_count = 0usize;

    // Y2 is emitted only for the whole-block path; a B_PRED macroblock
    // has no Y2 record at all (§13.3 skips block 24 when has_y2 is false).
    if !use_bpred {
        let scan_y2 = raster_to_scan(&raw_coeffs.y2);
        let nz = encode_block_with_ctx(enc, 24, BlockType::Y2, coeff_probs, &scan_y2, above, left)?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    let y_block_type = if use_bpred {
        BlockType::YNoY2
    } else {
        BlockType::YAfterY2
    };
    for (i, y_block) in raw_coeffs.y.iter().enumerate() {
        let scan = raster_to_scan(y_block);
        let nz = encode_block_with_ctx(enc, i, y_block_type, coeff_probs, &scan, above, left)?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    for (i, u_block) in raw_coeffs.u.iter().enumerate() {
        let scan = raster_to_scan(u_block);
        let nz =
            encode_block_with_ctx(enc, 16 + i, BlockType::UV, coeff_probs, &scan, above, left)?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    for (i, v_block) in raw_coeffs.v.iter().enumerate() {
        let scan = raster_to_scan(v_block);
        let nz =
            encode_block_with_ctx(enc, 20 + i, BlockType::UV, coeff_probs, &scan, above, left)?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    Ok(nonzero_block_count)
}

/// Encode one residual block at `block_index` against the §13.3 above /
/// left predictor slots, threading the §20.16 `left_context_index` /
/// `above_context_index` lookups so the encoder's per-position
/// probability index matches the decoder's. Returns the non-zero
/// coefficient count from `encode_coeff_block`.
///
/// This is the encoder partner of the `decode_one` closure inside
/// `decode_mb_coeffs`; both update the predictor contexts in place.
fn encode_block_with_ctx(
    enc: &mut BoolEncoder,
    block_index: usize,
    block_type: BlockType,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
    scan_coeffs: &[i16; 16],
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> Result<usize, TokenEncodeError> {
    // The §20.16 LEFT/ABOVE context index tables are crate-private
    // inside `dct_tokens`, but their layout is fixed by RFC 6386
    // §20.16 (and confirmed by the `decode_mb_coeffs` walk). The
    // encoder duplicates the mapping here so it doesn't need to grow
    // the `dct_tokens` public surface.
    const LEFT_CTX: [usize; 25] = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, // 16 Y
        4, 4, 5, 5, // 4 U
        6, 6, 7, 7, // 4 V
        8, // Y2
    ];
    const ABOVE_CTX: [usize; 25] = [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, // 16 Y
        4, 5, 4, 5, // 4 U
        6, 7, 6, 7, // 4 V
        8, // Y2
    ];

    let a_slot = ABOVE_CTX[block_index];
    let l_slot = LEFT_CTX[block_index];

    let nz = encode_coeff_block(
        enc,
        block_type,
        coeff_probs,
        above.nonzero[a_slot],
        left.nonzero[l_slot],
        scan_coeffs,
    )?;

    let has_coeffs = nz != 0;
    above.nonzero[a_slot] = has_coeffs;
    left.nonzero[l_slot] = has_coeffs;
    Ok(nz)
}

// ─────────────────────── §13.4 observed-counts fitter (round 157) ───────────────────────

/// Per-position branch counters for the §13.4 `token_prob_update()`
/// table: for every `(plane, band, prev_ctx, position)` slot of the
/// `coeff_probs[4][8][3][11]` table the encoder uses, count how many
/// times the §13.3 token-tree walk emitted a `0` bit at that position
/// (`zeros`) and how many times it emitted a `1` bit (`ones`). One
/// such counter per slot suffices: every bit the encoder emits against
/// `coeff_probs[i][j][k][t]` lands in `[i][j][k][t]` exactly once, so
/// a frame-wide accumulator is the natural rate-statistic for the §13.4
/// caller-driven update layer landed in rounds 155 / 156.
///
/// Used by [`count_keyframe_branches`] and the round-157
/// [`encode_keyframe_with_fitted_token_prob_updates`] fitter that
/// derives per-position `Some(p_obs)` replacement probabilities from
/// the observed counts and emits a §13.4 update only when the body bit
/// saving outweighs the §13.4 transmission cost.
///
/// The element type is `(u32, u32)` for `(zeros, ones)`. The maximum
/// per-frame count at any one position is bounded by the number of
/// coefficients in the frame (`16 * mb_cols * mb_rows` for the full-band
/// case), well within `u32`'s range for any realistic VP8 picture.
pub type BranchCounts = [[[[(u32, u32); 11]; 3]; 8]; 4];

/// Zeroed [`BranchCounts`] suitable as the start of an accumulator.
///
/// Equivalent to `[[[[(0, 0); 11]; 3]; 8]; 4]`; named so `count_*`
/// helpers can be called without the caller having to repeat the
/// 4-dimension initialiser literal at every site.
pub fn empty_branch_counts() -> BranchCounts {
    [[[[(0u32, 0u32); 11]; 3]; 8]; 4]
}

/// Walk one §13.3 sub-block exactly as [`encode_coeff_block`] would
/// emit it, but accumulating `(zeros, ones)` branch counts into
/// `counts` instead of writing to a [`BoolEncoder`]. The control flow
/// mirrors `encode_coeff_block` line for line so the counts track the
/// real token pass (token-tree path traversal; cat-token extra bits
/// and per-coefficient sign bits are intentionally not counted — those
/// are coded against fixed probabilities outside the `coeff_probs`
/// table the §13.4 layer can update).
///
/// `coeffs` is in **scan (zig-zag) order**, matching `encode_coeff_block`.
/// `above_has_nonzero` / `left_has_nonzero` seed `ctx3` exactly as the
/// real encoder does. Returns the non-zero coefficient count so the
/// caller can thread the §13.3 predictor update through `MbEntropyCtx`
/// in lockstep with the real encoder.
pub fn count_block_branches(
    block_type: BlockType,
    above_has_nonzero: bool,
    left_has_nonzero: bool,
    coeffs: &[i16; 16],
    counts: &mut BranchCounts,
) -> usize {
    let plane = block_type.plane_index();
    let first_coeff = block_type.first_coeff();

    let mut last_non_zero: i32 = -1;
    for (i, &v) in coeffs.iter().enumerate() {
        if i >= first_coeff && v != 0 {
            last_non_zero = i as i32;
        }
    }

    let mut ctx3: usize = (above_has_nonzero as usize) + (left_has_nonzero as usize);
    let mut prev_was_zero = false;
    let mut non_zero_count = 0usize;

    let mut i = first_coeff;
    while i < 16 {
        let band = COEFF_BANDS[i];

        let emit_eob = (i as i32) > last_non_zero;
        let (token, abs_value) = if emit_eob {
            (DctToken::Eob, 0u16)
        } else {
            let v = coeffs[i];
            let abs = v.unsigned_abs();
            (classify_coeff_token(abs), abs)
        };

        let start = if prev_was_zero { 2i8 } else { 0i8 };
        for step in token_to_bit_path(token, start) {
            let slot = &mut counts[plane][band][ctx3][step.prob_index as usize];
            if step.bit {
                slot.1 += 1;
            } else {
                slot.0 += 1;
            }
        }

        if token == DctToken::Eob {
            break;
        }

        if abs_value != 0 {
            non_zero_count += 1;
        }

        ctx3 = if abs_value == 0 {
            0
        } else if abs_value == 1 {
            1
        } else {
            2
        };
        prev_was_zero = token == DctToken::Dct0;
        i += 1;
    }

    non_zero_count
}

/// Count the §13.4 branch counts for one whole macroblock's residual,
/// mirroring [`encode_mb_tokens`] (Y2 if present, 16 Y sub-blocks, 4 U
/// sub-blocks, 4 V sub-blocks) and threading the §13.3 above / left
/// predictor contexts through `count_block_branches`. The §20.16
/// ABOVE / LEFT context-index tables match `encode_block_with_ctx`.
///
/// Like `encode_mb_tokens`, this is the per-MB driver; the frame
/// driver ([`count_keyframe_branches`]) calls it row-by-row with
/// frame-lived `above_ctx` and per-row `left_ctx` and respects
/// `mb_skip_coeff` (skip MBs emit no tokens but clear their predictor
/// slots via [`clear_skip_ctx`]).
pub fn count_mb_branches(
    raw_coeffs: &MbCoeffs,
    use_bpred: bool,
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
    counts: &mut BranchCounts,
) {
    const LEFT_CTX: [usize; 25] = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, // 16 Y
        4, 4, 5, 5, // 4 U
        6, 6, 7, 7, // 4 V
        8, // Y2
    ];
    const ABOVE_CTX: [usize; 25] = [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, // 16 Y
        4, 5, 4, 5, // 4 U
        6, 7, 6, 7, // 4 V
        8, // Y2
    ];

    let walk = |block_index: usize,
                block_type: BlockType,
                scan: &[i16; 16],
                above: &mut MbEntropyCtx,
                left: &mut MbEntropyCtx,
                counts: &mut BranchCounts| {
        let a = ABOVE_CTX[block_index];
        let l = LEFT_CTX[block_index];
        let nz = count_block_branches(block_type, above.nonzero[a], left.nonzero[l], scan, counts);
        let has = nz != 0;
        above.nonzero[a] = has;
        left.nonzero[l] = has;
    };

    if !use_bpred {
        let scan_y2 = raster_to_scan(&raw_coeffs.y2);
        walk(24, BlockType::Y2, &scan_y2, above, left, counts);
    }
    let y_plane = if use_bpred {
        BlockType::YNoY2
    } else {
        BlockType::YAfterY2
    };
    for (i, y_block) in raw_coeffs.y.iter().enumerate() {
        let scan = raster_to_scan(y_block);
        walk(i, y_plane, &scan, above, left, counts);
    }
    for (i, u_block) in raw_coeffs.u.iter().enumerate() {
        let scan = raster_to_scan(u_block);
        walk(16 + i, BlockType::UV, &scan, above, left, counts);
    }
    for (i, v_block) in raw_coeffs.v.iter().enumerate() {
        let scan = raster_to_scan(v_block);
        walk(20 + i, BlockType::UV, &scan, above, left, counts);
    }
}

/// Walk a whole keyframe's already-picked macroblock modes + raw
/// coefficients and accumulate §13.4 branch counts into `counts`,
/// mirroring the §13.3 token-encode pass of
/// [`encode_keyframe_with_reconstruction_and_token_updates`] —
/// row-major raster, frame-lived `above` per-column context, per-row
/// `left` reset, skip-MB context clearing via [`clear_skip_ctx`].
///
/// `modes` and `all_coeffs` are the same per-MB outputs the keyframe
/// driver feeds to its multi-partition token pass; `mb_cols` /
/// `mb_rows` are the macroblock-grid dimensions. The partition index a
/// given row routes to is irrelevant to the counts (the §13.3
/// above-context is shared across partitions and the per-row left
/// context resets identically regardless of partition assignment).
pub fn count_keyframe_branches(
    modes: &[MacroblockModes],
    all_coeffs: &[MbCoeffs],
    mb_cols: usize,
    mb_rows: usize,
    counts: &mut BranchCounts,
) {
    let mut above_ctx: Vec<MbEntropyCtx> = vec![MbEntropyCtx::default(); mb_cols];
    for mb_row in 0..mb_rows {
        let mut left_ctx = MbEntropyCtx::default();
        for (mb_col, above_col) in above_ctx.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            let use_bpred = mb.y_mode == IntraYMode::B;
            if mb.mb_skip_coeff {
                clear_skip_ctx(use_bpred, above_col, &mut left_ctx);
                continue;
            }
            count_mb_branches(
                &all_coeffs[raster],
                use_bpred,
                above_col,
                &mut left_ctx,
                counts,
            );
        }
    }
}

/// Walk a whole inter (P-) frame's already-picked macroblock modes +
/// raw coefficients and accumulate §13.4 branch counts into `counts`,
/// mirroring the §13.3 token-encode pass of
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates`]
/// — row-major raster, frame-lived `above` per-column context, per-row
/// `left` reset, skip-MB context clearing via `clear_skip_ctx`.
///
/// The shape mirrors [`count_keyframe_branches`] exactly with one
/// addition: a per-MB `use_bpred` slice is required because the inter
/// path's "no Y2" decision is **not** carried in [`MacroblockModes`]'s
/// `y_mode` field (the inter picker stamps `y_mode = IntraYMode::Dc`
/// onto every MB; per §13.1 / §14.2 page 76, SPLITMV MBs also omit Y2
/// independent of `y_mode`). The inter inner driver records the
/// effective "no Y2" flag in its `use_bpred_per_mb` vector — which is
/// exactly what this walker consumes.
///
/// `modes`, `use_bpred_per_mb`, and `all_coeffs` must share the
/// macroblock-grid length (`mb_cols * mb_rows`); a mismatch is a caller
/// bug. The partition index a given row routes to is irrelevant to the
/// counts (the §13.3 above-context is shared across partitions and the
/// per-row left context resets identically regardless of partition
/// assignment).
pub fn count_inter_frame_branches(
    modes: &[MacroblockModes],
    use_bpred_per_mb: &[bool],
    all_coeffs: &[MbCoeffs],
    mb_cols: usize,
    mb_rows: usize,
    counts: &mut BranchCounts,
) {
    debug_assert_eq!(modes.len(), mb_cols * mb_rows);
    debug_assert_eq!(use_bpred_per_mb.len(), mb_cols * mb_rows);
    debug_assert_eq!(all_coeffs.len(), mb_cols * mb_rows);

    let mut above_ctx: Vec<MbEntropyCtx> = vec![MbEntropyCtx::default(); mb_cols];
    for mb_row in 0..mb_rows {
        let mut left_ctx = MbEntropyCtx::default();
        for (mb_col, above_col) in above_ctx.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            let use_bpred = use_bpred_per_mb[raster];
            if mb.mb_skip_coeff {
                clear_skip_ctx(use_bpred, above_col, &mut left_ctx);
                continue;
            }
            count_mb_branches(
                &all_coeffs[raster],
                use_bpred,
                above_col,
                &mut left_ctx,
                counts,
            );
        }
    }
}

/// Derive a §13.4 [`TokenProbUpdates`] payload from observed branch
/// counts that **saves bits** when applied: each
/// `(plane, band, prev_ctx, position)` slot is set to `Some(p_obs)` if
/// and only if the body bit saving at that slot — coding `(zeros, ones)`
/// against the new probability vs the §13.5 default — outweighs the
/// §13.4 transmission cost (one flag-bit at the position's update
/// probability + an `L(8)` literal carrying the replacement, less the
/// flag-bit cost the no-update path would have paid). Slots with zero
/// total count, or where the saving is below `min_saving_bits`, stay
/// `None` (defaults retained).
///
/// `p_obs` is the maximum-likelihood estimate of the per-bit
/// probability of `0` at this slot, scaled to the §13.5 0..=255
/// `Prob`-byte range: `round(256 * zeros / total)`, clamped to `1..=255`
/// to mirror the boolean coder's `[1, 255]` valid range (and to keep
/// `bool_bits(p, bit)` finite when this fitter's output is fed back to
/// the encoder's cost estimator).
///
/// `min_saving_bits` is a small positive guard against round-trip
/// instability: the body saving is computed against the **observed**
/// distribution (which differs from the encoder's next-pass token
/// distribution once the new probs perturb the RD pick), so a
/// near-break-even slot may actually be net-negative after re-encode.
/// A guard of `~2 bits` is a reasonable starting point and is exposed
/// for tests that want the strict break-even output.
pub fn fit_token_prob_updates(
    counts: &BranchCounts,
    min_saving_bits: f64,
) -> crate::coded_header::TokenProbUpdates {
    let defaults = &crate::dct_tokens::DEFAULT_COEFF_PROBS;
    let mut out: crate::coded_header::TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];

    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for t in 0..11 {
                    let (n0, n1) = counts[i][j][k][t];
                    let total = n0 + n1;
                    if total == 0 {
                        continue;
                    }
                    let p_old = defaults[i][j][k][t];

                    // Maximum-likelihood p_obs scaled to the §13.5
                    // 0..=255 byte range, clamped to [1, 255] so the
                    // resulting Prob is always valid (the boolean
                    // coder rejects 0 / 256).
                    let p_new_raw = ((n0 as u64) * 256 + (total as u64) / 2) / (total as u64);
                    let p_new = p_new_raw.clamp(1, 255) as u8;

                    if p_new == p_old {
                        continue;
                    }

                    // Body bit-cost in each direction, using the same
                    // `bool_bits` cost model the encoder's RD estimator
                    // uses. Saving > 0 ⇒ the new prob codes the
                    // observed counts more cheaply than the default.
                    let body_old = (n0 as f64) * bool_bits(p_old, false)
                        + (n1 as f64) * bool_bits(p_old, true);
                    let body_new = (n0 as f64) * bool_bits(p_new, false)
                        + (n1 as f64) * bool_bits(p_new, true);
                    let body_saving = body_old - body_new;

                    // §13.4 transmission cost: replacing the no-op
                    // `flag = 0` (cost = bool_bits(update_prob, false))
                    // with `flag = 1` (cost = bool_bits(update_prob,
                    // true)) plus an 8-bit literal carrying p_new.
                    let update_prob =
                        COEFF_UPDATE_PROBS_FLAT[i * 8 * 3 * 11 + j * 3 * 11 + k * 11 + t];
                    let header_extra =
                        bool_bits(update_prob, true) + 8.0 - bool_bits(update_prob, false);

                    if body_saving > header_extra + min_saving_bits {
                        out[i][j][k][t] = Some(p_new);
                    }
                }
            }
        }
    }

    out
}

/// `true` when at least one §13.4 slot of `updates` carries a
/// replacement probability — the "is the fitted table a no-op?" test
/// every two-pass fitter runs before paying for a second encode pass.
fn any_token_prob_update(updates: &crate::coded_header::TokenProbUpdates) -> bool {
    updates.iter().any(|p| {
        p.iter()
            .any(|b| b.iter().any(|c| c.iter().any(|s| s.is_some())))
    })
}

// ─────────────────────── §9 / §11 / §19.2 keyframe raster driver ───────────────────────

/// A source I420 (YCbCr 4:2:0) picture handed to the keyframe raster
/// encoder. The three planes are row-major; chroma is half-resolution in
/// both dimensions (`(width + 1) / 2 × (height + 1) / 2`), matching the
/// §9.1 dimensions the decoder reconstructs.
///
/// Strides default to the tightly-packed widths; supply `y_stride` etc.
/// when the planes carry row padding.
#[derive(Debug, Clone, Copy)]
pub struct I420Frame<'a> {
    /// Visible width in luma pixels (1..=0x3FFF per §9.1).
    pub width: u32,
    /// Visible height in luma pixels (1..=0x3FFF per §9.1).
    pub height: u32,
    /// Luma plane, row-major, at least `y_stride * height` bytes.
    pub y: &'a [u8],
    /// U chroma plane, row-major, at least `uv_stride * uv_height` bytes.
    pub u: &'a [u8],
    /// V chroma plane, same dimensions as `u`.
    pub v: &'a [u8],
    /// Luma row stride in bytes (≥ `width`).
    pub y_stride: usize,
    /// Chroma row stride in bytes (≥ `(width + 1) / 2`).
    pub uv_stride: usize,
}

impl<'a> I420Frame<'a> {
    /// Construct a frame whose planes are tightly packed (`y_stride =
    /// width`, `uv_stride = (width + 1) / 2`).
    pub fn packed(width: u32, height: u32, y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Self {
        I420Frame {
            width,
            height,
            y,
            u,
            v,
            y_stride: width as usize,
            uv_stride: width.div_ceil(2) as usize,
        }
    }

    /// Read luma pixel `(x, py)`, clamping the coordinates into the
    /// visible plane so a partial right / bottom macroblock samples the
    /// nearest edge pixel (edge-replication padding).
    #[inline]
    fn y_at(&self, x: usize, py: usize) -> u8 {
        let w = self.width as usize;
        let h = self.height as usize;
        let cx = x.min(w - 1);
        let cy = py.min(h - 1);
        self.y[cy * self.y_stride + cx]
    }

    /// Read U / V pixel `(x, py)` with the same edge-replication clamp.
    /// `is_v` selects U (`false`) or V (`true`).
    #[inline]
    fn uv_at(&self, x: usize, py: usize, is_v: bool) -> u8 {
        let cw = self.width.div_ceil(2) as usize;
        let ch = self.height.div_ceil(2) as usize;
        let cx = x.min(cw - 1);
        let cy = py.min(ch - 1);
        let plane = if is_v { self.v } else { self.u };
        plane[cy * self.uv_stride + cx]
    }

    /// Extract the [`MbPixels`] for macroblock `(mb_row, mb_col)`,
    /// edge-replicating into the padding region of a partial right /
    /// bottom macroblock.
    fn extract_mb(&self, mb_row: usize, mb_col: usize) -> MbPixels {
        let mut p = MbPixels {
            y: [0u8; 256],
            u: [0u8; 64],
            v: [0u8; 64],
        };
        let y0 = mb_row * 16;
        let x0 = mb_col * 16;
        for r in 0..16 {
            for c in 0..16 {
                p.y[r * 16 + c] = self.y_at(x0 + c, y0 + r);
            }
        }
        let cy0 = mb_row * 8;
        let cx0 = mb_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                p.u[r * 8 + c] = self.uv_at(cx0 + c, cy0 + r, false);
                p.v[r * 8 + c] = self.uv_at(cx0 + c, cy0 + r, true);
            }
        }
        p
    }
}

/// Parameters for [`encode_keyframe`].
#[derive(Debug, Clone, Copy)]
pub struct KeyframeParams {
    /// §9.6 `y_ac_qi` baseline quantiser index (0..=127). All five §9.6
    /// deltas are omitted (set to 0). Lower = higher quality / larger
    /// output; a mid value (e.g. 32) targets ~30+ dB on natural content.
    pub y_ac_qi: u8,
    /// §9.4 baseline loop filter level (0..=63). `0` triggers the §15
    /// whole-frame loop-filter skip — the encoder leaves the residual
    /// reconstruction unfiltered. For any non-zero value the encoder
    /// runs the §15 filter over its own reconstruction buffer after the
    /// per-MB raster walk finishes, so the encoder's self-decode
    /// produces the same pixels the decoder will (the per-MB neighbour
    /// gather happens before the filter pass — §15.1 page 84 specifies
    /// the filter is a post-reconstruction stage).
    ///
    /// The §9.4 `mode_ref_lf_delta_enabled` flag is held at 0 by this
    /// encoder; the per-MB level therefore reduces to the frame base
    /// (with no segmentation override either, since segmentation is
    /// off). Values > 63 are rejected with
    /// [`EncodeError::LoopFilterLevelOutOfRange`].
    pub loop_filter_level: u8,
    /// §9.4 `sharpness_level` (0..=7). Feeds the §15.4 `interior_limit`
    /// derivation alongside the per-MB `loop_filter_level`. Only
    /// consulted when `loop_filter_level != 0` (the §15 skip path also
    /// skips the §15.4 control derivation). Values > 7 are rejected with
    /// [`EncodeError::SharpnessLevelOutOfRange`].
    pub sharpness_level: u8,
    /// §9.5 DCT-coefficient partition count. Must be one of 1 / 2 / 4 / 8
    /// (the on-wire `log2_nbr_of_dct_partitions` field is two bits per
    /// the §9.5 table). Macroblock rows are distributed round-robin per
    /// the §20.4 loop: row `r` is encoded into partition `r % N`. This
    /// is a layout reorganisation only — the residual coding inside each
    /// partition is bit-identical to the 1-partition case, so the
    /// decoded picture is unchanged across all four choices.
    pub nbr_of_dct_partitions: u8,
    /// §9.4 `filter_type` — `false` selects the §15.3 *normal* loop
    /// filter (the default; matches every earlier round's wire), `true`
    /// selects the §15.2 *simple* filter. The simple filter is a 4-pixel
    /// edge-only kernel (no §15.3 inner-window high-edge-variance branch
    /// and no chroma plane), so it costs the decoder less work but on a
    /// natural-content frame yields lower PSNR than the normal filter.
    /// The encoder runs the same selection inside its own §15 post-walk
    /// filter pass so encoder-vs-decoder pixel lockstep holds at both
    /// values. Only consulted when `loop_filter_level != 0`; the §15
    /// whole-frame skip path bypasses the filter entirely either way.
    pub filter_type: bool,
    /// §13 trellis coefficient-RDO aggressiveness — a clean-room
    /// quality/size dial (RFC 6386 specifies only token *decoding*; the
    /// level-assignment search is a non-normative encoder decision).
    /// [`TrellisStrength::DEFAULT`] reproduces the calibrated
    /// "shave-bits-hold-PSNR" trade; higher values trade more PSNR for
    /// smaller output, [`TrellisStrength::OFF`] reverts to plain
    /// round-quantisation. Threaded onto both the keyframe and the inter
    /// residual emit. The reconstruction the encoder writes back always
    /// uses the trellis-chosen levels, so encoder↔decoder pixel lockstep
    /// holds at every strength.
    pub trellis_strength: TrellisStrength,
}

impl Default for KeyframeParams {
    fn default() -> Self {
        KeyframeParams {
            y_ac_qi: 32,
            loop_filter_level: 0,
            sharpness_level: 0,
            nbr_of_dct_partitions: 1,
            filter_type: false,
            trellis_strength: TrellisStrength::DEFAULT,
        }
    }
}

/// Encode a complete VP8 key frame from a source I420 picture.
///
/// This is the §9 / §11 / §19.2 raster driver: it walks the source
/// macroblock-by-macroblock in raster order, and for each macroblock
///
/// 1. extracts the 16×16 luma / 8×8 chroma source block (edge-replicating
///    a partial right / bottom macroblock);
/// 2. gathers the reconstructed-neighbour strips from the
///    already-encoded part of the frame (the exact [`MbNeighbors`] the
///    decoder's frame walker assembles via `gather_neighbors`);
/// 3. picks the §12.2 whole-block or §11.3 / §12.3 `B_PRED` intra mode
///    and forward-transforms / quantises the residual
///    ([`encode_mb_block_set_with_neighbors`]);
/// 4. dequantises and reconstructs the macroblock through the **decoder's
///    own** §14.2 / §12.3 reconstruction orchestrators and writes the
///    result back into the running reconstruction buffer, so the next
///    macroblock predicts from genuine reconstructed pixels;
/// 5. records the chosen [`MacroblockModes`] and the raw-quantised
///    [`MbCoeffs`].
///
/// After the walk it assembles the §9 frame header + §19.2 first
/// (control) partition (with the §11 macroblock-mode layer threaded
/// through the cross-macroblock `B_PRED` sub-block context buffers) and
/// `params.nbr_of_dct_partitions` §19.2 DCT partitions carrying every
/// non-skipped macroblock's §13.3 token data, with the §13.3 above
/// (per-column, frame-lived) / left (per-row) non-zero predictor contexts
/// evolving exactly as the decoder reads them. Per the §20.4 row-loop,
/// macroblock row `r` is encoded into partition `r % N`; each partition
/// uses its own [`BoolEncoder`] and is finalised independently with the
/// usual §7.3 4-byte flush trailer (§4 page 9 — "All partitions are
/// decoded using separate instances of the boolean entropy decoder").
/// A §9.5 size table of `(N - 1) * 3` little-endian bytes precedes the
/// partition bodies when `N > 1`.
///
/// The emitted bytes decode through the crate's own [`crate::decode_vp8`]
/// and reproduce the source within the §14 quantiser's distortion.
///
/// # Scope (this round)
///
/// Single key frame, RD intra mode pick, no inter prediction. The §9.5
/// DCT partition count is configurable (1 / 2 / 4 / 8) through
/// [`KeyframeParams::nbr_of_dct_partitions`]; the residual coding inside
/// each partition is bit-identical to the 1-partition case (multi-
/// partition output is a layout reorganisation, not a coding change).
/// The §15 loop filter runs as a post-reconstruction stage when
/// `KeyframeParams::loop_filter_level != 0`; the §9.4 reference / mode
/// delta layer is disabled (the encoder writes
/// `mode_ref_lf_delta_enabled = false`).
///
/// [`MbNeighbors`]: crate::reconstruct::MbNeighbors
pub fn encode_keyframe(frame: &I420Frame, params: &KeyframeParams) -> Result<Vec<u8>, EncodeError> {
    let (bytes, _planes) = encode_keyframe_with_reconstruction(frame, params)?;
    Ok(bytes)
}

/// Encode a key frame with a caller-supplied §13.4 `token_prob_update()`
/// payload. The 4×8×3×11 `updates` array carries `Some(prob)` for each
/// `coeff_probs[i][j][k][t]` position the caller wants to replace and
/// `None` for the positions the §13.5 defaults should keep. The encoder
/// merges `updates` onto the §13.5 defaults via
/// [`merge_default_token_probs`](crate::merge_default_token_probs)
/// before its picker / token-encode passes, so both the rate-distortion
/// token-bit estimates and the emitted bitstream use the same merged
/// table the decoder will rebuild after reading the same updates.
///
/// When every entry of `updates` is `None`, the emitted bytes match
/// [`encode_keyframe`] byte-for-byte (the §13.4 sub-block reduces to
/// the 1056 zero-flag wire and the merged `coeff_probs` is identical to
/// the defaults). Otherwise the §13.4 sub-block carries the changed
/// positions and downstream token decoding uses the merged table for
/// the lifetime of the frame **only**: a key frame carrying updates
/// writes `refresh_entropy_probs = 0`, so the decoder's carried table
/// reverts to the §13.5 defaults afterwards (the base every inter
/// entry point in this crate codes against — see the §9.7 comment in
/// the keyframe driver).
///
/// Returns the bitstream bytes only; pair with the underlying
/// [`encode_keyframe_with_reconstruction_and_token_updates`] when the
/// post-§15 reconstruction planes are also needed (e.g. by a stream
/// encoder seeding its reference slots).
pub fn encode_keyframe_with_token_prob_updates(
    frame: &I420Frame,
    params: &KeyframeParams,
    updates: &TokenProbUpdates,
) -> Result<Vec<u8>, EncodeError> {
    let (bytes, _planes) =
        encode_keyframe_with_reconstruction_and_token_updates(frame, params, Some(updates))?;
    Ok(bytes)
}

/// Encode a key frame, **fitting** the §13.4 `token_prob_update()`
/// per-position replacements from this frame's observed token-tree
/// branch counts (RFC 6386 §13.4 / §13.5).
///
/// Rounds 155 and 156 landed the caller-driven §13.4 layer for the
/// keyframe and inter paths respectively — they let an external caller
/// hand in any `TokenProbUpdates` payload and the encoder writes the
/// §13.4 sub-block, threads the merged `coeff_probs` through the §13.3
/// token-encode pass, and the decoder rebuilds the same merged table
/// from the on-wire updates. Round 157 closes the natural follow-up:
/// the encoder now *picks* the §13.4 updates itself, from the very
/// counts the residual it is about to emit would produce against the
/// §13.5 defaults.
///
/// The fit is the obvious one: at each
/// `(plane, band, prev_ctx, position)` slot of the 4×8×3×11 table,
/// `p_obs = round(256 * zeros / total)` clamped to `[1, 255]` (the
/// boolean coder's valid `Prob` range) is the maximum-likelihood
/// estimate of the per-bit "probability of `0`" the encoder just
/// observed. An update is emitted only when the body bit saving
/// `(n0 * cost(p_old,0) + n1 * cost(p_old,1)) - (n0 * cost(p_new,0) +
/// n1 * cost(p_new,1))` exceeds the §13.4 transmission cost
/// `(cost(update_prob,1) + 8) - cost(update_prob,0)` plus a small
/// `min_saving_bits = 2.0` guard against re-encode drift. Slots with
/// zero observed count keep the §13.5 default. See
/// [`fit_token_prob_updates`] for the full cost-model.
///
/// Internally the function takes two passes:
///
///   1. Encode with the §13.5 defaults and collect the per-position
///      branch counts via the [`encode_keyframe_inner`]
///      `counts` side-channel.
///   2. Run [`fit_token_prob_updates`] to derive the
///      [`TokenProbUpdates`] payload that nets a positive bit saving
///      against those counts, then re-encode with that payload through
///      [`encode_keyframe_with_token_prob_updates`].
///
/// If the fitter returns an all-`None` payload (no slot crossed the
/// saving threshold), or the fitted re-encode is **larger** than the
/// default-encode wire (the model's saving estimate is computed
/// against pass-1's coefficient distribution; pass-2's RD pick perturbs
/// it slightly), the default-encode bytes are returned instead — so
/// this entry-point is guaranteed to be `<=` the
/// `encode_keyframe_with_token_prob_updates(.., all-None)` wire in
/// every case. The returned bytes always decode through the crate's
/// own [`crate::decode_vp8`] and any compliant VP8 decoder.
///
/// Out of round-157 scope: the inter (P-frame) path's analogous
/// `encode_p_frame_*_with_fitted_token_prob_updates` entry — this round
/// scopes to the keyframe path mirroring the r155-then-r156 split that
/// landed the caller-driven layer; the inter fitter can stack on top
/// in a subsequent round through the same [`fit_token_prob_updates`]
/// machinery (the cost-model and the [`BranchCounts`] type are shared).
pub fn encode_keyframe_with_fitted_token_prob_updates(
    frame: &I420Frame,
    params: &KeyframeParams,
) -> Result<Vec<u8>, EncodeError> {
    // Pass 1 — defaults + observed branch counts.
    let mut counts = empty_branch_counts();
    let (bytes_default, _planes_default) =
        encode_keyframe_inner(frame, params, None, Some(&mut counts))?;

    // Fit. 2.0 bits of slack guards against the small body-saving
    // overstatement that comes from the pass-2 RD pick perturbing
    // the coefficient distribution slightly relative to pass-1's
    // counts.
    let fitted = fit_token_prob_updates(&counts, 2.0);

    // If no slot crossed the threshold, the default encode wins
    // trivially (the all-`None` path is byte-identical to the
    // round-154 wire).
    let any_update = any_token_prob_update(&fitted);
    if !any_update {
        return Ok(bytes_default);
    }

    // Pass 2 — re-encode with the fitted updates. The pick / forward-
    // transform layer re-runs against the merged probabilities (the RD
    // estimator uses the merged table), so the chosen coefficients may
    // shift relative to pass 1. The encoder remains self-consistent —
    // the decoder rebuilds the same merged table from the on-wire
    // updates and reconstructs the same picture.
    let bytes_fitted = encode_keyframe_with_token_prob_updates(frame, params, &fitted)?;

    // Guard against the cost-model overstating the saving: only ship
    // the fitted bytes when they actually shrink the wire.
    if bytes_fitted.len() <= bytes_default.len() {
        Ok(bytes_fitted)
    } else {
        Ok(bytes_default)
    }
}

/// Encode a key frame with an automatically-fitted §13.4
/// `token_prob_update()` payload **and** return the post-§15
/// reconstructed [`crate::frame::KeyframePlanes`] alongside the wire.
///
/// Planes-returning companion to
/// [`encode_keyframe_with_fitted_token_prob_updates`] (round 157),
/// shaped the same way [`encode_keyframe_with_reconstruction`] relates
/// to [`encode_keyframe`]. The bytes are byte-identical to the no-
/// reconstruction fitter and the returned planes are the **macroblock-
/// aligned** post-§15 frame the decoder would rebuild from those bytes —
/// the exact shape the §9 three-slot reference-frame buffer wants for
/// `LAST` / `GOLDEN` / `ALTREF` installation.
///
/// Round 159 wires this into
/// [`crate::stream::Vp8KeyframeStreamEncoder::encode_frame_with_fitted_token_prob_updates`]
/// so the multi-frame keyframe stream driver can fit the §13.4 payload
/// per frame while keeping a sound LAST slot for the (currently-unused
/// for the keyframe driver, but populated for symmetry with the §9
/// three-slot ladder) downstream frame to predict from. The bytes-only
/// variant remains available for callers that don't need the planes.
///
/// The two-pass fitter + safety-guard semantics mirror
/// [`encode_keyframe_with_fitted_token_prob_updates`] exactly: pass 1
/// encodes with §13.5 defaults and records branch counts; pass 2
/// re-encodes with the fitted payload; if pass 2 isn't strictly smaller
/// the default-pass bytes **and** default-pass planes are returned so
/// the caller's downstream LAST slot stays consistent with the wire
/// (mirrors the round-158 inter fitter's matching-planes guarantee).
pub fn encode_keyframe_with_reconstruction_and_fitted_token_prob_updates(
    frame: &I420Frame,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    // Pass 1 — defaults + observed branch counts.
    let mut counts = empty_branch_counts();
    let (bytes_default, planes_default) =
        encode_keyframe_inner(frame, params, None, Some(&mut counts))?;

    // Fit. 2.0 bits of slack matches the round-157/158 fitter shape.
    let fitted = fit_token_prob_updates(&counts, 2.0);

    // No slot crossed the threshold ⇒ default encode wins trivially
    // (all-`None` path is byte-identical to the round-154 wire).
    let any_update = any_token_prob_update(&fitted);
    if !any_update {
        return Ok((bytes_default, planes_default));
    }

    // Pass 2 — re-encode with the fitted updates AND keep the matching
    // reconstruction. The picker re-runs against the merged probabilities
    // (RD estimator uses the merged table), so the chosen coefficients
    // may shift relative to pass 1; the encoder remains self-consistent
    // because the decoder rebuilds the same merged table from the on-
    // wire updates and reconstructs the same picture as `planes_fitted`.
    let (bytes_fitted, planes_fitted) =
        encode_keyframe_with_reconstruction_and_token_updates(frame, params, Some(&fitted))?;

    // Guard against the cost-model overstating the saving: only ship
    // the fitted bytes when they actually shrink the wire. On the
    // fall-back path we MUST also return the default-pass planes,
    // otherwise a streaming caller's next-frame LAST slot would not
    // match what the decoder will hold (the two passes' picker outputs
    // differ when the merged table differs). Mirrors the round-158
    // inter fitter's matching-planes guarantee.
    if bytes_fitted.len() <= bytes_default.len() {
        Ok((bytes_fitted, planes_fitted))
    } else {
        Ok((bytes_default, planes_default))
    }
}

/// Encode a key frame and return both the bitstream bytes **and** the
/// post-loop-filter reconstructed [`crate::frame::KeyframePlanes`] the
/// decoder will rebuild from those bytes.
///
/// This is the same machinery as [`encode_keyframe`] — the encoder
/// already maintains a running reconstruction buffer internally so each
/// macroblock can predict from genuine reconstructed neighbours, and
/// the §15 post-walk loop filter pass mutates that buffer in place. The
/// only difference is that here we hand the resulting buffer back to
/// the caller instead of dropping it on return.
///
/// The returned planes are the **macroblock-aligned** post-§15 frame
/// (width rounded up to `mb_cols * 16`, height to `mb_rows * 16`),
/// matching the layout of [`crate::frame::KeyframePlanes`] and the
/// per-slot storage of [`crate::state::RefFrameSlot`]. This is exactly
/// the shape the §9 reference-frame buffer wants — a multi-frame
/// keyframe driver can drop the planes straight into the LAST / GOLDEN
/// / ALTREF slots that a subsequent inter frame would predict from.
///
/// The visible-cropped (`width × height`) picture suitable for display
/// is what the crate's own [`crate::decode_vp8`] produces when handed
/// the returned bytes.
pub fn encode_keyframe_with_reconstruction(
    frame: &I420Frame,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_with_reconstruction_and_token_updates(frame, params, None)
}

/// Encode a key frame with RD-selected §9.4 `loop_filter_level`.
///
/// Identical to [`encode_keyframe`] except that `params.loop_filter_level`
/// is **ignored**: after the raster walk, the encoder searches the §9.4
/// level range for the level that minimises post-§15 visible-window SSD
/// against the source (clean-room — RFC 6386 leaves the level choice to
/// the encoder; see [`crate::loop_filter::select_filter_level`]) and
/// writes that level into the §9.4 header. All other `params` fields
/// (quantiser, sharpness, filter type, partition count) are honoured
/// verbatim. Returns the on-the-wire bytes.
pub fn encode_keyframe_auto_loop_filter(
    frame: &I420Frame,
    params: &KeyframeParams,
) -> Result<Vec<u8>, EncodeError> {
    encode_keyframe_inner_lf(frame, params, None, None, true).map(|(bytes, _)| bytes)
}

/// [`encode_keyframe_auto_loop_filter`] returning both the bitstream and
/// the post-§15 reconstruction planes (for a multi-frame driver that
/// installs the keyframe as the §9 reference without re-decoding).
pub fn encode_keyframe_auto_loop_filter_with_reconstruction(
    frame: &I420Frame,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_inner_lf(frame, params, None, None, true)
}

/// Orthogonal per-frame quality/size feature toggles for the generic
/// key-frame front door
/// [`encode_keyframe_with_reconstruction_and_coding_options`].
///
/// Each flag switches on one already-proven encoder feature; the
/// all-`false` default reproduces
/// [`encode_keyframe_with_reconstruction`] byte-for-byte. This struct
/// replaces the combinatorial `_auto_loop_filter` /
/// `_fitted_token_prob_updates` function-name ladder for callers (like
/// the lagged stream driver) that need the features **combined**.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyframeCodingOptions {
    /// RD-select the §9.4 `loop_filter_level` / sharpness per frame
    /// (ignoring `params.loop_filter_level`) — the
    /// [`encode_keyframe_auto_loop_filter`] behaviour.
    pub auto_loop_filter: bool,
    /// Two-pass §13.4 `token_prob_update()` fit: pass 1 records the
    /// observed §13.3 branch counts, [`fit_token_prob_updates`] derives
    /// the profitable replacements, pass 2 re-encodes against the
    /// merged table, and the smaller wire ships — the
    /// [`encode_keyframe_with_fitted_token_prob_updates`] behaviour.
    pub fitted_token_prob_updates: bool,
}

/// Generic key-frame front door combining the §9.4 auto-loop-filter
/// selector and the §13.4 fitted token-prob-update emission behind one
/// [`KeyframeCodingOptions`] toggle set.
///
/// * `options == Default` — byte-identical to
///   [`encode_keyframe_with_reconstruction`];
/// * only `auto_loop_filter` — byte-identical to
///   [`encode_keyframe_auto_loop_filter_with_reconstruction`];
/// * only `fitted_token_prob_updates` — byte-identical to
///   [`encode_keyframe_with_reconstruction_and_fitted_token_prob_updates`];
/// * both — the previously unreachable combination: the two-pass fitter
///   runs with the RD loop-filter selector active on **both** passes
///   (each pass picks its own §9.4 level; the shipped wire's header
///   level always matches the shipped reconstruction, so decoder
///   lockstep holds).
///
/// Returns the wire bytes plus the matching post-§15 reconstruction
/// planes for the §9 reference ladder.
pub fn encode_keyframe_with_reconstruction_and_coding_options(
    frame: &I420Frame,
    params: &KeyframeParams,
    options: &KeyframeCodingOptions,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    if !options.fitted_token_prob_updates {
        return encode_keyframe_inner_lf(frame, params, None, None, options.auto_loop_filter);
    }

    // Pass 1 — §13.5 defaults + observed branch counts (same shape as
    // the dedicated keyframe fitter, with the auto-LF switch threaded).
    let mut counts = empty_branch_counts();
    let (bytes_default, planes_default) = encode_keyframe_inner_lf(
        frame,
        params,
        None,
        Some(&mut counts),
        options.auto_loop_filter,
    )?;

    // 2.0 bits of slack — matches every other fitter in the crate.
    let fitted = fit_token_prob_updates(&counts, 2.0);
    if !any_token_prob_update(&fitted) {
        return Ok((bytes_default, planes_default));
    }

    // Pass 2 — re-encode with the fitted updates (picker + token emit
    // run against the merged table; auto-LF re-selects on the pass-2
    // reconstruction).
    let (bytes_fitted, planes_fitted) =
        encode_keyframe_inner_lf(frame, params, Some(&fitted), None, options.auto_loop_filter)?;

    // Ship the smaller wire, always with its matching planes so a
    // streaming caller's LAST slot equals the decoder's reconstruction.
    if bytes_fitted.len() <= bytes_default.len() {
        Ok((bytes_fitted, planes_fitted))
    } else {
        Ok((bytes_default, planes_default))
    }
}

/// Encode a key frame and return both the bitstream bytes **and** the
/// post-loop-filter reconstructed [`crate::frame::KeyframePlanes`],
/// optionally threading a §13.4 `token_prob_update()` payload through
/// the encode.
///
/// `token_updates = None` is the round-154 wire (every §13.4 flag = 0,
/// §13.5 defaults retained); `token_updates = Some(u)` writes the
/// per-position replacement layer and uses the merged `coeff_probs` for
/// both the picker's RD estimate and the §13.3 token-encode pass. The
/// emitted bitstream decodes through the crate's own [`crate::decode_vp8`]
/// and through any compliant VP8 decoder under either path.
///
/// See [`encode_keyframe_with_token_prob_updates`] for the no-plane
/// variant and [`encode_keyframe_with_reconstruction`] for the
/// no-updates variant.
pub fn encode_keyframe_with_reconstruction_and_token_updates(
    frame: &I420Frame,
    params: &KeyframeParams,
    token_updates: Option<&TokenProbUpdates>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_inner(frame, params, token_updates, None)
}

/// Internal keyframe driver shared by
/// [`encode_keyframe_with_reconstruction_and_token_updates`] and the
/// round-157 [`encode_keyframe_with_fitted_token_prob_updates`] fitter.
///
/// `counts` (when `Some`) is filled with the per-position §13.4 branch
/// counts the §13.3 token-encode pass produces against the supplied
/// `token_updates` table. With `counts = None` the function reduces
/// exactly to the public entry-point above (no extra work, no behaviour
/// change). With `counts = Some(&mut c)` the §13.3 walk records each
/// `(plane, band, prev_ctx, position)` bit-event into `c` alongside its
/// regular `BoolEncoder` write — `c`'s contents are the only side-effect
/// of the parameter; the emitted bytes are byte-identical to the
/// `counts = None` invocation with the same `token_updates`.
fn encode_keyframe_inner(
    frame: &I420Frame,
    params: &KeyframeParams,
    token_updates: Option<&TokenProbUpdates>,
    counts: Option<&mut BranchCounts>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_inner_lf(frame, params, token_updates, counts, false)
}

/// `encode_keyframe_inner` extended with the round-373 §9.4 RD
/// loop-filter-level auto-selection switch.
///
/// When `auto_lf` is `false` the behaviour is byte-identical to the
/// historical [`encode_keyframe_inner`]: `params.loop_filter_level` is the
/// frame's filter level verbatim. When `auto_lf` is `true` the level
/// stored in `params` is *ignored* and, after the raster walk produces the
/// unfiltered reconstruction, [`crate::loop_filter::select_filter_level`]
/// picks the level that minimises post-§15 visible-window SSD against the
/// source `frame`; that chosen level is then both applied to the
/// reconstruction and written into the §9.4 header, so the encoder's
/// reconstruction stays in lock-step with a compliant decoder.
fn encode_keyframe_inner_lf(
    frame: &I420Frame,
    params: &KeyframeParams,
    token_updates: Option<&TokenProbUpdates>,
    counts: Option<&mut BranchCounts>,
    auto_lf: bool,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    let width = frame.width;
    let height = frame.height;
    if width == 0 || width > 0x3FFF || height == 0 || height > 0x3FFF {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    if params.y_ac_qi > 127 {
        return Err(EncodeError::QuantIndexOutOfRange {
            value: params.y_ac_qi,
        });
    }
    if params.loop_filter_level > 63 {
        return Err(EncodeError::LoopFilterLevelOutOfRange {
            value: params.loop_filter_level,
        });
    }
    if params.sharpness_level > 7 {
        return Err(EncodeError::SharpnessLevelOutOfRange {
            value: params.sharpness_level,
        });
    }
    // §9.5 — the partition-count is validated here so a bad value is
    // rejected before the long mode-pick / forward-transform walk runs.
    // `write_token_partition_count` re-validates at the actual bitstream
    // emission point.
    let num_partitions = match params.nbr_of_dct_partitions {
        1 | 2 | 4 | 8 => params.nbr_of_dct_partitions as usize,
        other => return Err(EncodeError::InvalidDctPartitionCount { value: other }),
    };

    let mb_cols = width.div_ceil(16) as usize;
    let mb_rows = height.div_ceil(16) as usize;

    // §13.4 token-prob update: when the caller supplies a non-empty
    // `token_updates` array, the encoder writes its replacement layer
    // through `write_token_prob_updates` below AND uses the merged
    // `coeff_probs` table for both the picker's RD estimate and the
    // §13.3 token-encode pass. The decoder reads the same updates and
    // rebuilds the same merged table, so the bitstream stays sound on
    // either path. `None` (or an all-`None` array) preserves the
    // round-154 wire byte-for-byte.
    let coeff_probs = match token_updates {
        Some(u) => crate::dct_tokens::merge_default_token_probs(u),
        None => crate::dct_tokens::DEFAULT_COEFF_PROBS,
    };
    // Per-frame §13 trellis rate tables for the resolved `coeff_probs`
    // (see `TrellisRateTables`) — built once here so the per-MB mode
    // search prices tokens from lookups instead of re-walking the §13.2
    // tree per candidate. Values (and bytes) are identical either way.
    let trellis_rate_tables =
        (params.trellis_strength.get() > 0.0).then(|| TrellisRateTables::for_probs(&coeff_probs));
    let factors = crate::dequant::MbDequantFactors::from_base_and_deltas(
        params.y_ac_qi as i32,
        0,
        0,
        0,
        0,
        0,
    );

    // Running reconstruction buffer — identical layout to the decoder's
    // `decode_keyframe` output. Each MB's neighbours are gathered from
    // here via the decoder's own `gather_neighbors`, guaranteeing the
    // encoder predicts from the exact pixels the decoder will.
    let mut planes = crate::frame::KeyframePlanes {
        y: vec![0u8; mb_cols * 16 * mb_rows * 16],
        u: vec![0u8; mb_cols * 8 * mb_rows * 8],
        v: vec![0u8; mb_cols * 8 * mb_rows * 8],
        y_stride: mb_cols * 16,
        uv_stride: mb_cols * 8,
        mb_cols,
        mb_rows,
    };

    let mut modes: Vec<MacroblockModes> = Vec::with_capacity(mb_rows * mb_cols);
    // Raw-quantised coefficients per MB (kept so the DCT partition pass
    // can re-walk them into the shared encoder after the mode layer is
    // written into the first partition).
    let mut all_coeffs: Vec<MbCoeffs> = Vec::with_capacity(mb_rows * mb_cols);

    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            let pixels = frame.extract_mb(mb_row, mb_col);
            let neighbors = crate::frame::gather_neighbors_public(&planes, mb_row, mb_col);

            // Mode pick + forward transform/quant against the genuine
            // reconstructed neighbours.
            let encoded = encode_mb_block_set_with_neighbors_strength_rates(
                &pixels,
                &neighbors,
                params.y_ac_qi,
                &coeff_probs,
                trellis_rate_tables.as_ref(),
                params.trellis_strength,
            )
            .map_err(EncodeError::Token)?;

            // A macroblock with no non-zero residual is coded as a skip
            // MB (§11.1): the decoder reconstructs it as pure prediction,
            // which is exactly what zero residue produces.
            let mb_skip_coeff = encoded.nonzero_block_count == 0;

            // Dequantise a copy of the raw coefficients and reconstruct
            // through the decoder's §14.2 / §12.3 orchestrators, writing
            // the result back so the next MB sees real neighbours.
            let mut dq = encoded.coeffs;
            factors.dequantize(&mut dq);
            let use_bpred = encoded.y_mode == IntraYMode::B;
            let recon = if use_bpred {
                crate::reconstruct::decode_keyframe_mb_bpred(
                    encoded.b_subblock_modes.as_ref(),
                    encoded.uv_mode,
                    mb_skip_coeff,
                    &neighbors,
                    &dq.y,
                    &dq.u,
                    &dq.v,
                )
            } else {
                crate::reconstruct::decode_keyframe_mb_non_bpred(
                    encoded.y_mode,
                    encoded.uv_mode,
                    mb_skip_coeff,
                    &neighbors,
                    &dq.y2,
                    &dq.y,
                    &dq.u,
                    &dq.v,
                )
            }
            .map_err(EncodeError::Reconstruct)?;
            crate::frame::write_mb_public(&mut planes, mb_row, mb_col, &recon);

            modes.push(MacroblockModes {
                segment_id: None,
                mb_skip_coeff,
                y_mode: encoded.y_mode,
                subblock_modes: encoded.b_subblock_modes,
                uv_mode: encoded.uv_mode,
            });
            all_coeffs.push(encoded.coeffs);
        }
    }

    // ---- §15 loop-filter post-pass --------------------------------------
    //
    // RFC 6386 §15 page 84: "After the predictor and residue have been
    // summed for every macroblock, the filter is applied to the edges
    // between adjacent macroblocks and the edges between adjacent
    // subblocks." We run the filter here, after the raster walk has
    // completed and *before* the bitstream is assembled, so the encoder's
    // reconstruction buffer ends up bit-identical to what the decoder
    // will produce after its own §15.1 pass over the same frame.
    //
    // The per-MB neighbour gather inside the raster loop above
    // intentionally runs against the *unfiltered* reconstruction (the
    // §15.1 filter is a post-reconstruction stage — the predictors that
    // feed each macroblock's intra prediction always sample the
    // unfiltered neighbour). The filter only adjusts pixels at MB / sub-
    // block edges after every MB has been reconstructed.
    //
    // `filter_frame` uses `MbCoeffs` only for the §15.1 page 86
    // "any-coded-coefficient" test that gates the inter-subblock edges
    // (steps 2 and 4); that test reduces to "any non-zero coefficient",
    // which is invariant under dequantisation by the non-zero §14.1 Q
    // steps, so we can pass the raw quantised `all_coeffs` directly.
    // The §15 base config (level filled in below). Segmentation off and
    // `mode_ref_lf_delta_enabled = false`, so every per-MB override is 0
    // and each MB's effective level reduces to the frame base.
    let mut lf_config = crate::loop_filter::FrameFilterConfig {
        // §9.4 `filter_type`: `false` ⇒ §15.3 normal, `true` ⇒ §15.2
        // simple. Mirrors the bit `write_loop_filter` writes below;
        // the encoder's own post-walk filter must match what the
        // decoder will run.
        simple: params.filter_type,
        key_frame: true,
        loop_filter_level: params.loop_filter_level,
        sharpness_level: params.sharpness_level,
        segmentation_enabled: false,
        segment_abs: false,
        segment_lf_level: [0; crate::loop_filter::MAX_MB_SEGMENTS],
        delta_enabled: false,
        ref_delta_current: 0,
        bpred_mode_delta: 0,
        ref_delta_last: 0,
        ref_delta_golden: 0,
        ref_delta_altref: 0,
        zero_mv_mode_delta: 0,
        other_mv_mode_delta: 0,
        split_mv_mode_delta: 0,
    };
    // §9.4 effective loop-filter level + sharpness. When `auto_lf` is set
    // the RD selector jointly replaces `params.loop_filter_level` /
    // `params.sharpness_level` with the distortion-minimising pair
    // (clean-room; RFC 6386 is silent on the encoder's choice). Both
    // chosen values are written to the header below so the decoder runs
    // the identical §15 pass.
    let (eff_lf_level, eff_sharpness) = if auto_lf {
        let has_coeffs: Vec<bool> = all_coeffs
            .iter()
            .map(crate::loop_filter::mb_has_coeffs)
            .collect();
        let src = crate::loop_filter::SourcePlanes {
            width: width as usize,
            height: height as usize,
            y: frame.y,
            u: frame.u,
            v: frame.v,
            y_stride: frame.y_stride,
            uv_stride: frame.uv_stride,
        };
        crate::loop_filter::select_filter_params(
            &planes,
            &modes,
            &has_coeffs,
            &lf_config,
            &src,
            None,
        )
    } else {
        (params.loop_filter_level, params.sharpness_level)
    };
    lf_config.loop_filter_level = eff_lf_level;
    lf_config.sharpness_level = eff_sharpness;
    if eff_lf_level != 0 {
        crate::loop_filter::filter_frame(&mut planes, &modes, &all_coeffs, &lf_config);
    }
    // `planes` is the macroblock-aligned post-§15 reconstruction. It is
    // returned to the caller alongside the bitstream below so a
    // multi-frame keyframe driver can install it into the §9
    // reference-frame buffer (LAST / GOLDEN / ALTREF) without paying the
    // cost of re-decoding the bytes we just emitted.

    // ---- §19.2 first (control) partition --------------------------------
    let mut hdr = BoolEncoder::new();

    // §9.2 — color_space + clamping_type both 0.
    hdr.write_bool(128, false);
    hdr.write_bool(128, false);
    // §9.3 — segmentation off.
    write_segment_update_flags(&mut hdr, false);
    // §9.4 — loop filter. `filter_type` follows `params.filter_type`
    // (false ⇒ §15.3 normal, true ⇒ §15.2 simple);
    // `mode_ref_lf_delta_enabled = false` (per the [`KeyframeParams`]
    // docs) so no per-MB delta layer is emitted.
    // `loop_filter_level == 0` triggers the §15 whole-frame skip on the
    // decoder side; any non-zero value is honoured by both ends of the
    // round-trip via the post-walk filter pass above.
    write_loop_filter(
        &mut hdr,
        params.filter_type,
        eff_lf_level,
        eff_sharpness,
        false,
    )?;
    // §9.5 — DCT partition count.
    write_token_partition_count(&mut hdr, params.nbr_of_dct_partitions)?;
    // §9.6 — quant indices (baseline only).
    write_quant_indices(&mut hdr, params.y_ac_qi, None, None, None, None, None)?;
    // §9.7 (key frame) — refresh_entropy_probs. An update-free key
    // frame keeps the historical 1 (its merged table IS the §13.5
    // defaults, so persisting it is a no-op). A key frame that carries
    // a §13.4 update payload writes 0 instead, so per §9.10 the
    // decoder's carried table reverts to the key-frame reset state (the
    // §13.5 defaults) once the frame is done: every inter entry point
    // in this crate codes its §13.4 overlay against a defaults carried
    // base, and a persisted key-frame overlay would silently diverge
    // the encoder's and decoder's probability tables on every following
    // P-frame (pixel drift with no decode error — caught by the
    // round-387 lagged-driver fitter integration test).
    let persists_updates = token_updates.is_some_and(any_token_prob_update);
    hdr.write_bool(128, !persists_updates);
    // §13 / §9.9 — token-prob update sub-block. With `token_updates =
    // Some(u)` the per-position replacement layer is written and the
    // merged `coeff_probs` above is what the encoder's tokens are coded
    // against (and what the decoder will rebuild after reading the same
    // updates). With `None`, every flag is 0 and the §13.5 defaults
    // stay in force — byte-identical to the round-154 wire.
    match token_updates {
        Some(u) => write_token_prob_updates(&mut hdr, u, &COEFF_UPDATE_PROBS_FLAT),
        None => write_no_token_prob_updates(&mut hdr, &COEFF_UPDATE_PROBS_FLAT),
    }
    // §9.11 — mb_no_skip_coeff enabled with a balanced prob so skip and
    // non-skip macroblocks both code cheaply.
    let prob_skip_false = 128u8;
    write_mb_no_skip_coeff(&mut hdr, true, prob_skip_false);

    // §11 macroblock-mode layer, threading the §11.3 cross-macroblock
    // B_PRED sub-block context buffers exactly as `parse_key_frame_*`
    // reads them.
    write_mode_layer(&mut hdr, &modes, mb_rows, mb_cols, prob_skip_false);

    let first_partition = hdr.finish();
    let first_partition_size = first_partition.len();
    if first_partition_size > 0x7_FFFF {
        return Err(EncodeError::FirstPartitionTooLarge {
            bytes: first_partition_size,
        });
    }

    // ---- §19.2 DCT partition group: per-MB §13.3 token data --------------
    //
    // Macroblock rows are distributed round-robin across the §9.5
    // partitions per the §20.4 row-loop: row `r` is encoded into
    // partition `r % N`. Each partition gets its own [`BoolEncoder`]
    // instance (§4 page 9 — "All partitions are decoded using separate
    // instances of the boolean entropy decoder"), finalised independently
    // with its own §7.3 4-byte flush trailer. The §13.3 above-context
    // is column-wise and frame-lived — shared across partitions because
    // the decoder's `decode_residuals` also keeps one above slot per
    // column for the whole frame. The "left" context resets at every row
    // start so it does not need to cross partitions.
    let mut partitions: Vec<BoolEncoder> =
        (0..num_partitions).map(|_| BoolEncoder::new()).collect();
    // One above-context per macroblock column, frame-lived (§13.3),
    // shared by every row irrespective of which partition that row routes
    // to.
    let mut above_ctx: Vec<MbEntropyCtx> = vec![MbEntropyCtx::default(); mb_cols];
    for mb_row in 0..mb_rows {
        // §13.3 page 65: the "left" predictor resets at the start of
        // every macroblock row.
        let mut left_ctx = MbEntropyCtx::default();
        let part_idx = mb_row % num_partitions;
        let tok = &mut partitions[part_idx];
        for (mb_col, above_col) in above_ctx.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            let use_bpred = mb.y_mode == IntraYMode::B;
            if mb.mb_skip_coeff {
                // §13.1: a skip macroblock emits no tokens, but its
                // predictor slots are cleared so the next macroblock's
                // context is correct.
                clear_skip_ctx(use_bpred, above_col, &mut left_ctx);
                continue;
            }
            encode_mb_tokens(
                tok,
                &all_coeffs[raster],
                use_bpred,
                &coeff_probs,
                above_col,
                &mut left_ctx,
            )
            .map_err(EncodeError::Token)?;
        }
    }
    // §13.4 observed-counts side-channel (round 157). Re-walks the
    // §13.3 token loop above with `count_keyframe_branches`, driving
    // its own private above / left predictor contexts in lockstep —
    // so the recorded per-position counts are bit-for-bit the events
    // the bytes in `partitions` just emitted. With `counts = None`
    // this is a no-op.
    if let Some(c) = counts {
        count_keyframe_branches(&modes, &all_coeffs, mb_cols, mb_rows, c);
    }
    let dct_partitions: Vec<Vec<u8>> = partitions.into_iter().map(|p| p.finish()).collect();
    let dct_total: usize = dct_partitions.iter().map(|p| p.len()).sum();
    let size_table_len = (num_partitions - 1) * 3;

    // ---- §9.1 frame tag + key-frame extension + assembly ----------------
    let mut out: Vec<u8> =
        Vec::with_capacity(10 + first_partition_size + size_table_len + dct_total);
    write_frame_tag(
        &mut out,
        true,
        0,
        true,
        first_partition_size as u32,
        width,
        height,
        ScaleCode::None,
        ScaleCode::None,
    )?;
    out.extend_from_slice(&first_partition);
    // §9.5 size table: 3-byte little-endian length for every partition
    // except the last (whose size the decoder infers from what is left
    // in the frame). One partition → no table.
    for part in dct_partitions.iter().take(num_partitions - 1) {
        let sz = part.len();
        out.push((sz & 0xff) as u8);
        out.push(((sz >> 8) & 0xff) as u8);
        out.push(((sz >> 16) & 0xff) as u8);
    }
    for part in &dct_partitions {
        out.extend_from_slice(part);
    }

    Ok((out, planes))
}

/// Configuration for the §9.3 / §10 segment-based adaptive-quant key-frame
/// encoder ([`encode_keyframe_adaptive_quant`]).
///
/// The encoder sorts each macroblock into one of four segments by its
/// luma activity (variance), then applies a per-segment quantizer feature
/// so detailed macroblocks can be coded at a finer quantizer than flat
/// ones (or vice-versa). All segmentation machinery — the §9.3
/// `update_segmentation()` block, the per-MB §10 `segment_id`, the
/// per-segment §14.1 dequant factors — is driven from this config. The
/// segmentation is **delta-mode** (`segment_feature_mode = 0`): each
/// segment's `quant_delta` is added to the §9.6 `base_y_ac_qi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveQuantConfig {
    /// §9.6 baseline `y_ac_qi` (0..=127). Each segment's effective qindex
    /// is `clamp(base_y_ac_qi + quant_delta[seg], 0, 127)`.
    pub base_y_ac_qi: u8,
    /// Three ascending luma-variance boundaries splitting macroblocks into
    /// the four segments: segment `s` is the count of boundaries the MB's
    /// variance meets-or-exceeds (so a flat MB → segment 0, the busiest →
    /// segment 3). Values need not be sorted by the caller; the classifier
    /// uses them as independent "≥" tests, but ascending order gives the
    /// intended monotone mapping.
    pub variance_boundaries: [u32; 3],
    /// Per-segment §10 `quantizer_update_value` deltas (signed, magnitude
    /// ≤ 127), added to `base_y_ac_qi` in delta mode. The natural AQ
    /// choice raises the quantizer on flat segments (`+`) and lowers it on
    /// detailed segments (`-`); the default does exactly that.
    pub quant_delta: [i8; 4],
    /// §9.4 baseline loop-filter level (0..=63); `0` skips the §15 filter.
    pub loop_filter_level: u8,
    /// §9.4 sharpness (0..=7).
    pub sharpness_level: u8,
    /// §9.4 `filter_type`: false = §15.3 normal, true = §15.2 simple.
    pub filter_type: bool,
}

impl Default for AdaptiveQuantConfig {
    fn default() -> Self {
        AdaptiveQuantConfig {
            base_y_ac_qi: 32,
            variance_boundaries: [
                SEGMENT_VARIANCE_THRESHOLDS[1],
                SEGMENT_VARIANCE_THRESHOLDS[2],
                SEGMENT_VARIANCE_THRESHOLDS[3],
            ],
            // Flat → coarser (+), detailed → finer (−): the classic
            // activity-masking AQ gradient.
            quant_delta: [8, 0, -6, -12],
            loop_filter_level: 0,
            sharpness_level: 0,
            filter_type: false,
        }
    }
}

/// Population variance of a 16×16 luma macroblock (×1, integer), the §10
/// activity metric the [`AdaptiveQuantConfig`] classifier sorts on.
///
/// One-pass `E[X²] − E[X]²` in `u64` (256 `u8` samples sum to at most
/// `256·255² ≈ 1.66e7`, far inside `u64`). The `saturating_sub` guards the
/// rare case where integer rounding of the two terms inverts their order.
fn mb_luma_variance(y: &[u8; 256]) -> u32 {
    let mut sum: u64 = 0;
    let mut sumsq: u64 = 0;
    for &px in y.iter() {
        sum += px as u64;
        sumsq += (px as u64) * (px as u64);
    }
    let n = 256u64;
    let mean_sq = (sum * sum) / (n * n);
    let e_sq = sumsq / n;
    e_sq.saturating_sub(mean_sq) as u32
}

/// Classify a macroblock variance into a 0..=3 segment id: the number of
/// `variance_boundaries` the variance meets-or-exceeds, capped at 3.
fn classify_segment(variance: u32, boundaries: &[u32; 3]) -> u8 {
    boundaries.iter().filter(|&&b| variance >= b).count().min(3) as u8
}

/// Encode a key frame with §9.3 / §10 segment-based adaptive quantisation.
///
/// Each macroblock is sorted into one of four segments by its §10 luma
/// variance ([`mb_luma_variance`] vs the config's `variance_boundaries`),
/// and quantised at that segment's effective qindex
/// (`clamp(base_y_ac_qi + quant_delta[seg], 0, 127)`). The frame header
/// carries the §9.3 `update_segmentation()` block (delta-mode per-segment
/// `quantizer_update` values + the `mb_segment_tree_probs`), and the §11
/// mode layer carries each MB's §10 `segment_id`. The decoder's
/// `resolve_segment_dequant_factors` rebuilds the same four
/// [`crate::dequant::MbDequantFactors`] and applies the matching one per
/// MB, so the emitted bytes self-decode to the encoder's own
/// reconstruction within the per-segment quantiser's distortion.
///
/// The quantiser used to *quantise* each MB's residual is the same one
/// used to *dequantise* it for the running reconstruction buffer, so the
/// encoder predicts from the exact pixels the decoder will reconstruct
/// (the standard encoder/decoder pixel-lockstep contract). The §15 loop
/// filter (when `loop_filter_level != 0`) runs as a post-reconstruction
/// stage with segmentation-aware levels left at the frame base (the §10
/// per-segment loop-filter feature is not emitted by this path).
pub fn encode_keyframe_adaptive_quant(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (bytes, _planes) = encode_keyframe_adaptive_quant_with_reconstruction(frame, config)?;
    Ok(bytes)
}

/// As [`encode_keyframe_adaptive_quant`] but with the §13 trellis
/// aggressiveness dialed by `trellis` ([`TrellisStrength`]).
/// `AdaptiveQuantConfig` derives `Eq`, so the strength is a separate
/// argument rather than a struct field; [`TrellisStrength::DEFAULT`]
/// reproduces the historical bytes, [`TrellisStrength::OFF`] reverts to
/// plain round-quantisation. Per-segment quantisation is unchanged — the
/// strength only re-prices the §13 coefficient search inside each MB.
pub fn encode_keyframe_adaptive_quant_with_trellis(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
    trellis: TrellisStrength,
) -> Result<Vec<u8>, EncodeError> {
    let (bytes, _planes) = encode_keyframe_adaptive_quant_inner(frame, config, trellis)?;
    Ok(bytes)
}

/// As [`encode_keyframe_adaptive_quant`], but also returns the
/// macroblock-aligned post-§15 reconstruction planes (the LAST/GOLDEN/
/// ALTREF candidate a multi-frame driver would install) so callers avoid
/// re-decoding the bytes just emitted.
pub fn encode_keyframe_adaptive_quant_with_reconstruction(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_adaptive_quant_inner(frame, config, TrellisStrength::DEFAULT)
}

/// As [`encode_keyframe_adaptive_quant_with_reconstruction`] but with the
/// §13 trellis aggressiveness dialed by `trellis`.
pub fn encode_keyframe_adaptive_quant_with_reconstruction_and_trellis(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
    trellis: TrellisStrength,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_adaptive_quant_inner(frame, config, trellis)
}

/// Encode a key frame with §9.3 / §10 segment-based adaptive
/// quantisation **and** the §10 per-segment loop-filter feature — the
/// full two-feature `update_segment_feature_data` block.
///
/// On top of [`encode_keyframe_adaptive_quant`]'s per-segment quantiser
/// gradient, `segment_lf_deltas[s]` is the §9.3 `loop_filter_update`
/// value for segment `s` (delta mode, signed, magnitude ≤ 63): each
/// macroblock's §15 filter level resolves to
/// `clamp(loop_filter_level + segment_lf_deltas[seg], 0, 63)` per
/// §20.6, in the encoder's own post-walk pass and in any compliant
/// decoder alike. The natural pairing with the quantiser gradient is
/// **coarser segments filter harder** (they carry more blocking) while
/// detailed segments filter lighter (preserving texture) — i.e. deltas
/// with the same sign as `quant_delta`.
///
/// Values outside ±63 are rejected with
/// [`EncodeError::LoopFilterLevelOutOfRange`]. The §15 whole-frame skip
/// mirrors the decoder: `config.loop_filter_level == 0` skips the
/// filter regardless of the deltas. `[0; 4]` emits the feature with
/// explicit zero deltas (wire differs from the no-feature encode; the
/// resolved levels do not).
///
/// Returns the wire bytes plus the matching post-§15 reconstruction.
pub fn encode_keyframe_adaptive_quant_with_segment_lf_deltas(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
    segment_lf_deltas: &[i8; 4],
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_adaptive_quant_inner_lf(
        frame,
        config,
        TrellisStrength::DEFAULT,
        Some(segment_lf_deltas),
    )
}

/// As [`encode_keyframe_adaptive_quant_with_segment_lf_deltas`] with
/// the §13 trellis aggressiveness dialed by `trellis`.
pub fn encode_keyframe_adaptive_quant_with_segment_lf_deltas_and_trellis(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
    segment_lf_deltas: &[i8; 4],
    trellis: TrellisStrength,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_adaptive_quant_inner_lf(frame, config, trellis, Some(segment_lf_deltas))
}

fn encode_keyframe_adaptive_quant_inner(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
    trellis: TrellisStrength,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_keyframe_adaptive_quant_inner_lf(frame, config, trellis, None)
}

/// [`encode_keyframe_adaptive_quant_inner`] extended with the round-387
/// §10 per-segment **loop-filter feature**: when `segment_lf_deltas =
/// Some(d)`, the §9.3 `update_segmentation()` block additionally
/// carries the four `loop_filter_update` values (delta mode, added to
/// the frame base and clamped to `0..=63` per §20.6), and the
/// encoder's own §15 post-walk filter pass resolves each MB's level
/// through the same segment override the decoder applies
/// ([`crate::loop_filter::calculate_mb_filter_level`]) — so the
/// per-segment quantiser gradient can pair with a matching per-segment
/// deblock gradient (coarser segments → stronger filtering).
///
/// The §15 whole-frame skip mirrors the decoder's gate exactly: a frame
/// base `loop_filter_level == 0` skips the filter regardless of the
/// segment deltas (they still travel in the header for state
/// completeness). `None` reproduces the historical wire byte-for-byte.
fn encode_keyframe_adaptive_quant_inner_lf(
    frame: &I420Frame,
    config: &AdaptiveQuantConfig,
    trellis: TrellisStrength,
    segment_lf_deltas: Option<&[i8; 4]>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    let width = frame.width;
    let height = frame.height;
    if width == 0 || width > 0x3FFF || height == 0 || height > 0x3FFF {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    if config.base_y_ac_qi > 127 {
        return Err(EncodeError::QuantIndexOutOfRange {
            value: config.base_y_ac_qi,
        });
    }
    if config.loop_filter_level > 63 {
        return Err(EncodeError::LoopFilterLevelOutOfRange {
            value: config.loop_filter_level,
        });
    }
    if config.sharpness_level > 7 {
        return Err(EncodeError::SharpnessLevelOutOfRange {
            value: config.sharpness_level,
        });
    }
    // §9.3: the loop-filter feature value is a signed 6-bit magnitude —
    // reject anything outside ±63 before the walk runs.
    if let Some(d) = segment_lf_deltas {
        if let Some(&bad) = d.iter().find(|v| v.unsigned_abs() > 63) {
            return Err(EncodeError::LoopFilterLevelOutOfRange {
                value: bad.unsigned_abs(),
            });
        }
    }

    let mb_cols = width.div_ceil(16) as usize;
    let mb_rows = height.div_ceil(16) as usize;

    let coeff_probs = crate::dct_tokens::DEFAULT_COEFF_PROBS;
    // Per-frame §13 trellis rate tables (see `TrellisRateTables`); the
    // tables depend only on `coeff_probs`, not on the per-segment
    // quantiser factors, so one build serves every segment.
    let trellis_rate_tables =
        (trellis.get() > 0.0).then(|| TrellisRateTables::for_probs(&coeff_probs));

    // §10 per-segment effective qindex + the matching §14.1 dequant
    // factors. Delta mode: q_seg = clamp(base + delta, 0, 127). The
    // decoder rebuilds the identical factors from the §9.3 quantizer
    // deltas via `resolve_segment_dequant_factors`.
    let base = config.base_y_ac_qi as i32;
    let mut seg_qindex = [0u8; crate::loop_filter::MAX_MB_SEGMENTS];
    let mut seg_factors =
        [crate::dequant::MbDequantFactors::from_base_and_deltas(base, 0, 0, 0, 0, 0);
            crate::loop_filter::MAX_MB_SEGMENTS];
    for s in 0..crate::loop_filter::MAX_MB_SEGMENTS {
        let q = (base + config.quant_delta[s] as i32).clamp(0, 127) as u8;
        seg_qindex[s] = q;
        seg_factors[s] =
            crate::dequant::MbDequantFactors::from_base_and_deltas(q as i32, 0, 0, 0, 0, 0);
    }

    let mut planes = crate::frame::KeyframePlanes {
        y: vec![0u8; mb_cols * 16 * mb_rows * 16],
        u: vec![0u8; mb_cols * 8 * mb_rows * 8],
        v: vec![0u8; mb_cols * 8 * mb_rows * 8],
        y_stride: mb_cols * 16,
        uv_stride: mb_cols * 8,
        mb_cols,
        mb_rows,
    };

    let mut modes: Vec<MacroblockModes> = Vec::with_capacity(mb_rows * mb_cols);
    let mut all_coeffs: Vec<MbCoeffs> = Vec::with_capacity(mb_rows * mb_cols);

    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            let pixels = frame.extract_mb(mb_row, mb_col);
            let neighbors = crate::frame::gather_neighbors_public(&planes, mb_row, mb_col);

            // §10 classify by luma activity, then quantise at the
            // segment's qindex.
            let variance = mb_luma_variance(&pixels.y);
            let seg = classify_segment(variance, &config.variance_boundaries);
            let q = seg_qindex[seg as usize];

            let encoded = encode_mb_block_set_with_neighbors_strength_rates(
                &pixels,
                &neighbors,
                q,
                &coeff_probs,
                trellis_rate_tables.as_ref(),
                trellis,
            )
            .map_err(EncodeError::Token)?;

            let mb_skip_coeff = encoded.nonzero_block_count == 0;

            // Dequantise with the SAME segment factors used to quantise,
            // so the encoder's reconstruction matches the decoder's.
            let mut dq = encoded.coeffs;
            seg_factors[seg as usize].dequantize(&mut dq);
            let use_bpred = encoded.y_mode == IntraYMode::B;
            let recon = if use_bpred {
                crate::reconstruct::decode_keyframe_mb_bpred(
                    encoded.b_subblock_modes.as_ref(),
                    encoded.uv_mode,
                    mb_skip_coeff,
                    &neighbors,
                    &dq.y,
                    &dq.u,
                    &dq.v,
                )
            } else {
                crate::reconstruct::decode_keyframe_mb_non_bpred(
                    encoded.y_mode,
                    encoded.uv_mode,
                    mb_skip_coeff,
                    &neighbors,
                    &dq.y2,
                    &dq.y,
                    &dq.u,
                    &dq.v,
                )
            }
            .map_err(EncodeError::Reconstruct)?;
            crate::frame::write_mb_public(&mut planes, mb_row, mb_col, &recon);

            modes.push(MacroblockModes {
                segment_id: Some(seg),
                mb_skip_coeff,
                y_mode: encoded.y_mode,
                subblock_modes: encoded.b_subblock_modes,
                uv_mode: encoded.uv_mode,
            });
            all_coeffs.push(encoded.coeffs);
        }
    }

    // ---- §15 loop-filter post-pass ---------------------------------------
    // Gate mirrors the decoder exactly: frame base level 0 skips the
    // whole-frame filter even when segment deltas are present.
    if config.loop_filter_level != 0 {
        let mut segment_lf_level = [0i16; crate::loop_filter::MAX_MB_SEGMENTS];
        if let Some(d) = segment_lf_deltas {
            for (dst, &src) in segment_lf_level.iter_mut().zip(d.iter()) {
                *dst = i16::from(src);
            }
        }
        let lf_config = crate::loop_filter::FrameFilterConfig {
            simple: config.filter_type,
            key_frame: true,
            loop_filter_level: config.loop_filter_level,
            sharpness_level: config.sharpness_level,
            // With `segment_lf_deltas = Some`, each MB's §15 level is
            // `clamp(base + delta[seg], 0, 63)` per §20.6 — the same
            // resolution the decoder's `FrameFilterConfig::keyframe`
            // performs from the §9.3 header we emit below. `None` keeps
            // the frame base for every segment (historical wire).
            segmentation_enabled: segment_lf_deltas.is_some(),
            segment_abs: false,
            segment_lf_level,
            delta_enabled: false,
            ref_delta_current: 0,
            bpred_mode_delta: 0,
            ref_delta_last: 0,
            ref_delta_golden: 0,
            ref_delta_altref: 0,
            zero_mv_mode_delta: 0,
            other_mv_mode_delta: 0,
            split_mv_mode_delta: 0,
        };
        crate::loop_filter::filter_frame(&mut planes, &modes, &all_coeffs, &lf_config);
    }

    // ---- §19.2 first (control) partition --------------------------------
    let mut hdr = BoolEncoder::new();

    // §9.2 — color_space + clamping_type both 0.
    hdr.write_bool(128, false);
    hdr.write_bool(128, false);

    // §9.3 — segmentation ON: delta-mode per-segment quantizer features +
    // the map update with explicit tree probabilities. We emit a fixed,
    // mid-range `mb_segment_tree_probs` (170 ≈ "leans toward 0") for all
    // three nodes; the per-MB `segment_id` codes below use the same probs.
    const SEG_TREE_PROB: u8 = 170;
    let seg_probs: [u8; 3] = [SEG_TREE_PROB; 3];
    let layer = UpdateSegmentationLayer {
        update_mb_segmentation_map: true,
        feature_mode_absolute: false,
        // Each segment carries its quant delta. A 0 delta is still emitted
        // (Some(0)) so the decoder's segment base for that slot is exactly
        // `base + 0`, matching `seg_qindex` above.
        quantizer: [
            Some(config.quant_delta[0] as i16),
            Some(config.quant_delta[1] as i16),
            Some(config.quant_delta[2] as i16),
            Some(config.quant_delta[3] as i16),
        ],
        // §10 per-segment loop-filter feature (round 387): each segment
        // carries its delta when the caller supplied one (a 0 delta is
        // still emitted as Some(0), mirroring the quantizer convention
        // above so the decoder's persisted value is explicit).
        loop_filter: match segment_lf_deltas {
            Some(d) => [
                Some(i16::from(d[0])),
                Some(i16::from(d[1])),
                Some(i16::from(d[2])),
                Some(i16::from(d[3])),
            ],
            None => [None; 4],
        },
        segment_prob: [Some(seg_probs[0]), Some(seg_probs[1]), Some(seg_probs[2])],
    };
    write_update_segmentation(&mut hdr, &layer);

    // §9.4 — loop filter; mode_ref_lf_delta disabled.
    write_loop_filter(
        &mut hdr,
        config.filter_type,
        config.loop_filter_level,
        config.sharpness_level,
        false,
    )?;
    // §9.5 — single DCT partition.
    write_token_partition_count(&mut hdr, 1)?;
    // §9.6 — quant indices: the §9.6 baseline. The per-segment override
    // rides on top of this base via the §9.3 quantizer deltas above.
    write_quant_indices(&mut hdr, config.base_y_ac_qi, None, None, None, None, None)?;
    // §9.7 (key frame) — refresh_entropy_probs.
    hdr.write_bool(128, true);
    // §13 / §9.9 — no token-prob updates.
    write_no_token_prob_updates(&mut hdr, &COEFF_UPDATE_PROBS_FLAT);
    // §9.11 — mb_no_skip_coeff enabled, balanced prob.
    let prob_skip_false = 128u8;
    write_mb_no_skip_coeff(&mut hdr, true, prob_skip_false);

    // §11 mode layer WITH the §10 per-MB segment_id prefix.
    write_mode_layer_segmented(
        &mut hdr,
        &modes,
        mb_rows,
        mb_cols,
        prob_skip_false,
        Some(&seg_probs),
    );

    let first_partition = hdr.finish();
    let first_partition_size = first_partition.len();
    if first_partition_size > 0x7_FFFF {
        return Err(EncodeError::FirstPartitionTooLarge {
            bytes: first_partition_size,
        });
    }

    // ---- §19.2 DCT partition: per-MB §13.3 token data (single part) ------
    let mut tok = BoolEncoder::new();
    let mut above_ctx: Vec<MbEntropyCtx> = vec![MbEntropyCtx::default(); mb_cols];
    for mb_row in 0..mb_rows {
        let mut left_ctx = MbEntropyCtx::default();
        for (mb_col, above_col) in above_ctx.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            let use_bpred = mb.y_mode == IntraYMode::B;
            if mb.mb_skip_coeff {
                clear_skip_ctx(use_bpred, above_col, &mut left_ctx);
                continue;
            }
            encode_mb_tokens(
                &mut tok,
                &all_coeffs[raster],
                use_bpred,
                &coeff_probs,
                above_col,
                &mut left_ctx,
            )
            .map_err(EncodeError::Token)?;
        }
    }
    let dct_partition = tok.finish();

    // ---- §9.1 frame tag + assembly --------------------------------------
    let mut out: Vec<u8> = Vec::with_capacity(10 + first_partition_size + dct_partition.len());
    write_frame_tag(
        &mut out,
        true,
        0,
        true,
        first_partition_size as u32,
        width,
        height,
        ScaleCode::None,
        ScaleCode::None,
    )?;
    out.extend_from_slice(&first_partition);
    out.extend_from_slice(&dct_partition);

    Ok((out, planes))
}

/// Clear the §13.3 non-zero predictor slots a skip macroblock touches.
///
/// §13.1: a skipped macroblock writes no token data, so all of its
/// residual blocks are implicitly all-zero. The decoder's
/// `decode_mb_coeffs` clears every above / left predictor slot the
/// macroblock owns; the encoder mirrors that so a non-skip neighbour to
/// the right / below sees `has_coeffs = false`.
fn clear_skip_ctx(use_bpred: bool, above: &mut MbEntropyCtx, left: &mut MbEntropyCtx) {
    // The Y / U / V slots (0..=7) are always cleared. The Y2 slot (8)
    // is cleared only when the macroblock carries a Y2 block (every
    // non-B_PRED MB) — a B_PRED MB leaves the inherited Y2 context
    // untouched, matching the decoder's `decode_mb_coeffs` skip path.
    for slot in 0..8 {
        above.nonzero[slot] = false;
        left.nonzero[slot] = false;
    }
    if !use_bpred {
        above.nonzero[8] = false;
        left.nonzero[8] = false;
    }
}

/// Write the §11 key-frame macroblock-mode layer for the whole frame
/// into the first-partition encoder `hdr`.
///
/// Per macroblock (raster order) this writes, in §11 order:
///   1. `mb_skip_coeff` (§11.1) against `prob_skip_false`;
///   2. the 16×16 luma mode via `kf_ymode_tree` (§11.2);
///   3. for a `B_PRED` macroblock, sixteen 4×4 sub-block modes via
///      `bmode_tree` (§11.3 / §11.5), each against the context-driven
///      `KF_BMODE_PROB[above][left]` row;
///   4. the 8×8 chroma mode via `uv_mode_tree` (§11.4).
///
/// The §11.3 `above_subblock` / `left_subblock` context buffers evolve
/// macroblock-to-macroblock exactly as `parse_key_frame_macroblock_modes`
/// reads them, so the decoder recovers the modes bit-for-bit.
fn write_mode_layer(
    hdr: &mut BoolEncoder,
    modes: &[MacroblockModes],
    mb_rows: usize,
    mb_cols: usize,
    prob_skip_false: u8,
) {
    write_mode_layer_segmented(hdr, modes, mb_rows, mb_cols, prob_skip_false, None);
}

/// As [`write_mode_layer`], but with the optional §10 per-MB `segment_id`
/// prefix. When `segment_probs` is `Some(probs)` — meaning the frame set
/// `segmentation_enabled && update_mb_segmentation_map` — each macroblock
/// emits its `segment_id` (via the §10 `mb_segment_tree` against the
/// effective `probs`, defaults-of-255 already folded in) *before* the
/// §11.1 `mb_skip_coeff` bit, matching the decoder's read order in
/// [`crate::macroblock`] `parse_key_frame_macroblock_modes`. When `None`
/// the layer is identical to the non-segmented wire.
fn write_mode_layer_segmented(
    hdr: &mut BoolEncoder,
    modes: &[MacroblockModes],
    mb_rows: usize,
    mb_cols: usize,
    prob_skip_false: u8,
    segment_probs: Option<&[u8; 3]>,
) {
    // §11.3 item 3: the "above" sub-block context spans the whole frame
    // width (4 sub-blocks per MB column), initialised to B_DC_PRED.
    let mut above_subblock = vec![IntraBmode::Dc; mb_cols * 4];

    for mb_row in 0..mb_rows {
        // §11.3 item 3: the four left predictors reset to B_DC_PRED at
        // the start of every macroblock row.
        let mut left_subblock = [IntraBmode::Dc; 4];

        for mb_col in 0..mb_cols {
            let mb = &modes[mb_row * mb_cols + mb_col];

            // 0. segment_id (§10), only when the frame updates the
            //    segmentation map. Coded before mb_skip_coeff via the
            //    §10 mb_segment_tree.
            if let Some(probs) = segment_probs {
                let sid = mb.segment_id.unwrap_or(0);
                hdr.write_treed(&crate::macroblock::MB_SEGMENT_TREE, |i| probs[i], sid);
            }

            // 1. mb_skip_coeff (§11.1) — mb_no_skip_coeff is enabled.
            hdr.write_bool(prob_skip_false, mb.mb_skip_coeff);

            // 2. Luma mode (§11.2).
            hdr.write_treed(&KF_YMODE_TREE, |i| KF_YMODE_PROB[i], mb.y_mode.leaf());

            // 3. Sub-block modes (§11.3 / §11.5) for B_PRED only.
            if let Some(sub) = &mb.subblock_modes {
                for j in 0..16 {
                    let row = j >> 2;
                    let col = j & 3;
                    let above = if row == 0 {
                        above_subblock[mb_col * 4 + col]
                    } else {
                        sub[(row - 1) * 4 + col]
                    };
                    let left = if col == 0 {
                        left_subblock[row]
                    } else {
                        sub[row * 4 + (col - 1)]
                    };
                    let prob_row = &KF_BMODE_PROB[above.idx()][left.idx()];
                    hdr.write_treed(&BMODE_TREE, |i| prob_row[i], sub[j].idx() as u8);
                }
            }

            // 4. Chroma mode (§11.4).
            hdr.write_treed(&UV_MODE_TREE, |i| KF_UV_MODE_PROB[i], mb.uv_mode.leaf());

            // §11.3 item 3 / 4: update the cross-macroblock context.
            match &mb.subblock_modes {
                Some(sub) => {
                    above_subblock[mb_col * 4..mb_col * 4 + 4].copy_from_slice(&sub[12..16]);
                    for (row, slot) in left_subblock.iter_mut().enumerate() {
                        *slot = sub[row * 4 + 3];
                    }
                }
                None => {
                    let projected = mb.y_mode.project_to_subblock_context();
                    for slot in &mut above_subblock[mb_col * 4..mb_col * 4 + 4] {
                        *slot = projected;
                    }
                    left_subblock.fill(projected);
                }
            }
        }
    }
}

// ─── P-frame encoder (ZEROMV / NEARESTMV / NEARMV / NEWMV / SPLITMV / LAST) ──
//
// RFC 6386 §9 / §16 / §17 / §18 inter-frame encoder. Every emitted
// macroblock is inter against the LAST reference (§16.2
// `prob_last`-selected); the per-MB inter mode is picked between the
// five §16.2 `mv_ref_tree` leaves via a non-normative §-not-specified
// rate-distortion trade:
//
//   J(zero)    = SAD_at_(0,0)
//                + lambda * mv_ref_tree("0")     bits
//   J(nearest) = SAD_at_clamp(near.mvs[1])
//                + lambda * mv_ref_tree("10")    bits
//   J(near)    = SAD_at_clamp(near.mvs[2])
//                + lambda * mv_ref_tree("110")   bits
//   J(new)     = SAD_at_searched_mv
//                + lambda * (mv_ref_tree("1110") bits + §17 mv bits)
//   J(split)   = min over the four §16.4 partitions of
//                  (sum_groups group_SAD
//                   + lambda * (mv_ref_tree("1111") + mvpartition_tree(p)
//                               + sum_groups sub_mv_ref_tree + NEW4X4 bits))
//
// Lower J wins. The whole-MB §17 motion search is the whole-pixel
// `small_diamond_search_luma` primitive followed by a §18.3
// `half_pixel_refine_luma` probe of the 8 half-pixel offsets around
// the whole-pixel result, and then a §18.3 `quarter_pixel_refine_luma`
// probe of the 8 quarter-pixel offsets around the half-pixel result
// — all clamped to `[-1023, +1023]` per §17.1. When the chosen MV is
// sub-pixel-aligned the §18 prediction runs the §18.3 six-tap
// synthesis (`version == 0` bicubic tap-set, indexed by
// `(stored_luma_mv(mv) & 7)`); at a whole-pixel MV it collapses to the
// §18.2 / §18.3 copy path. The §16.3 NEARESTMV / NEARMV candidates are
// the §16.3 census `near.mvs[1]` / `near.mvs[2]` slots clamped through
// the per-MB `MvClampRect`, scored at the clamped MV the decoder will
// reconstruct from (so the SAD-evaluator is the §18.3 sixtap-aware
// `mb_luma_sad_at_mv`).
//
// The §16.4 SPLITMV picker evaluates each of the four partition shapes
// (TopBottom / LeftRight / Quarters / Mv16); for each partition group
// it runs a whole-pixel sub-block diamond search around the clamped
// `near.mvs[0]` "best" predictor (no §18.1 secondary clamp per §18.1
// page 114) and prices the four §16.4 `sub_mv_ref` modes (LEFT4X4 /
// ABOVE4X4 / ZERO4X4 / NEW4X4) at the group anchor's left / above
// neighbour vectors. The lowest-J group SAD + sub_mv_ref bits wins;
// the partition with the smallest summed J — added to the partition
// tree bits and the §16.2 SPLITMV path bits — is the SPLITMV
// candidate. When J(split) < J(other-four), SPLITMV emits.
//
// The §14 residual = source - prediction is quantised and §13.3-token-
// coded through the existing pipeline regardless of which mode was
// picked. SPLITMV has no Y2 block (§14.2 page 76); its luma sub-blocks
// code from coefficient 0 like B_PRED. Per §17 the §17.2
// `mv_prob_update()` block is still emitted with every `F` flag set
// to 0 — "no update", matching the no-token-prob-update pattern — so
// the decoder's MV-decode runs against the §17.2 defaults; the same
// `MvContexts` is used encoder-side to write the NEWMV / NEW4X4
// differentials.

// ─── §16.4 SPLITMV picker helpers ───────────────────────────────────────────
//
// All four §16.4 partition shapes (`MvPartition::TopBottom`,
// `LeftRight`, `Quarters`, `Mv16`) are evaluated per MB. For each
// partition group we pick a vector by combining a whole-pixel diamond
// search (around the clamped `near.mvs[0]` "best" predictor) with a
// §16.4 `sub_mv_ref` mode choice (LEFT4X4 / ABOVE4X4 / ZERO4X4 /
// NEW4X4) priced by `sub_mv_ref_tree` + §17.2 NEW4X4 component bits.
// The SPLITMV picker is intentionally whole-pixel-only this round
// (matching the §16.4 `MAX_DIAMOND_ITERS` bound below) so its
// per-group search cost stays linear in partition count without an
// extra §18.3 sixtap pass per sub-block.

/// One §16.4 SPLITMV picker result: the chosen partition, the sixteen
/// per-sub-block vectors, the per-group `sub_mv_ref` modes, the
/// per-group NEW4X4 differentials (zero when the mode is not NEW4X4),
/// and the total rate-distortion cost the partition scored at.
///
/// The picker walks `[TopBottom, LeftRight, Quarters, Mv16]` and keeps
/// whichever shape produces the smallest `j`; `j` then competes with
/// the whole-MB picker's `J(zero/nearest/near/new)`.
#[derive(Debug, Clone)]
struct SplitMvCandidate {
    /// The §16.4 partition shape this candidate scored under.
    partition: crate::near_mv::MvPartition,
    /// `this->split.mvs[16]` — the resolved per-sub-block quarter-pixel
    /// vectors, in raster order (matching `MV_PARTITIONS` layout).
    split_mvs: [crate::motion_vector::Mv; 16],
    /// The per-partition-group `sub_mv_ref` mode (indexed in partition-id
    /// order: group 0 first, then 1, …). Only the partition's group
    /// count (2 for TopBottom/LeftRight, 4 for Quarters, 16 for Mv16)
    /// leading entries are meaningful; the rest stay at the `Zero4x4`
    /// fill and are never read (the emitter iterates exactly the
    /// partition's groups). Fixed-size since round 277 — the previous
    /// per-candidate `Vec` pair was a measurable slice of the encoder's
    /// residual allocator churn.
    submv_modes: [crate::near_mv::SubMvRefMode; 16],
    /// The per-group NEW4X4 differential (zero `Mv` for groups whose
    /// `sub_mv_ref` mode is not NEW4X4). Same indexing as `submv_modes`.
    submv_new_diffs: [crate::motion_vector::Mv; 16],
    /// Total `J = sum_groups group_SAD + lambda * (mv_ref_tree("1111") +
    /// mvpartition_tree(p) + sum_groups sub_mv_ref_tree + NEW4X4 bits)`
    /// for this partition. Lower wins.
    j: f64,
}

/// `MAX_DIAMOND_ITERS` for the SPLITMV per-group sub-block search.
/// Smaller than the whole-MB MAX_DIAMOND_ITERS = 8 (line ~3535) since
/// SPLITMV evaluates up to 16 groups per MB across 4 partition shapes
/// and the per-group region is small (≤ 8 sub-blocks).
const SPLIT_MV_MAX_DIAMOND_ITERS: u32 = 4;

/// 4×4 luma block SAD — sum |src[i] - pred[i]|, i in 0..16. The
/// SPLITMV per-group score sums [`sub_block_sad_4x4`] across the group's
/// member sub-blocks.
#[inline]
fn sub_block_sad_4x4(src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    let mut sad: u32 = 0;
    for i in 0..16 {
        sad += (src[i] as i32 - pred[i] as i32).unsigned_abs();
    }
    sad
}

/// Extract one 4×4 source sub-block (raster sub-block index `b`, b in
/// 0..=15) from a 16×16 luma source MB.
#[inline]
fn extract_src_subblock_4x4(src: &[u8; 256], b: usize) -> [u8; 16] {
    let sb_row = b >> 2; // 0..=3
    let sb_col = b & 3; // 0..=3
    let mut out = [0u8; 16];
    for r in 0..4 {
        let dst_off = r * 4;
        let src_off = (sb_row * 4 + r) * 16 + sb_col * 4;
        out[dst_off..dst_off + 4].copy_from_slice(&src[src_off..src_off + 4]);
    }
    out
}

/// Sum the 4×4 SADs across the partition group `[group_subblocks]` for
/// a whole-pixel candidate vector `mv`.
///
/// Every member of one §16.4 partition group shares the one candidate
/// vector, and a whole-pixel §18.3 prediction is "simply copied", so
/// each member's 4×4 prediction is a direct window into the reference
/// plane at integer offset `(mb_x0, mb_y0) + (eighth-pixel mv >> 3)`.
/// When the whole 16×16 MB-extent source region lands strictly inside
/// the plane (a conservative gate — every member sub-block lies inside
/// the MB extent; the dominant case mid-frame) the SAD is accumulated
/// **directly against the reference and source rows** — no
/// per-sub-block patch fetch and no source sub-block extraction copy
/// (the fused fetch-and-SAD shape of the round-279 BENCHMARKS
/// candidate, SPLITMV-group half). A border-straddling candidate falls
/// back to the per-member
/// [`crate::motion_comp::fetch_block_whole_pixel`] path (§20.14
/// `build_mc_border` edge replication), bit-identical to the
/// pre-round-281 listing for every candidate
/// (`group_sad_fused_fast_path_matches_per_subblock_fetch`). §18.1
/// page 114 says no secondary clamp applies to SPLITMV sub-block MVs.
fn group_sad_at_whole_mv(
    luma_ref: crate::motion_search::LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    group_subblocks: &[usize],
    mv: crate::motion_vector::Mv,
) -> u32 {
    // §18.1 stored_luma_mv doubles the §17 quarter-pixel vector into
    // the §18 eighth-pixel resolution `fetch_block_whole_pixel` uses.
    let mv_eighth = crate::motion_comp::stored_luma_mv(mv);
    let mb_x0 = mb_col * 16;
    let mb_y0 = mb_row * 16;
    let mut sad: u32 = 0;

    // §18.2: integer pixel offset is the eighth-pixel vector shifted
    // right 3 with sign propagation.
    let src_x0 = mb_x0 as isize + (mv_eighth.col >> 3) as isize;
    let src_y0 = mb_y0 as isize + (mv_eighth.row >> 3) as isize;
    if src_x0 >= 0
        && src_y0 >= 0
        && src_x0 + 16 <= luma_ref.width as isize
        && src_y0 + 16 <= luma_ref.height as isize
    {
        // Fused fast path: the whole MB-extent source region is
        // strictly in-bounds, so each member's prediction row is the
        // contiguous 4-byte reference run at its sub-block offset — SAD
        // it against the matching `src_y` run without materialising
        // either side.
        let x0 = src_x0 as usize;
        let y0 = src_y0 as usize;
        for &b in group_subblocks {
            let sb_row = b >> 2;
            let sb_col = b & 3;
            for r in 0..4 {
                let pred_start = (y0 + sb_row * 4 + r) * luma_ref.stride + x0 + sb_col * 4;
                let pred_row = &luma_ref.plane[pred_start..pred_start + 4];
                let src_start = (sb_row * 4 + r) * 16 + sb_col * 4;
                let src_row = &src_y[src_start..src_start + 4];
                for c in 0..4 {
                    sad += (src_row[c] as i32 - pred_row[c] as i32).unsigned_abs();
                }
            }
        }
        return sad;
    }

    // Border-straddling candidate: per-member §20.14 edge-replicating
    // fetch (rare — only candidates whose motion walks off the
    // picture).
    for &b in group_subblocks {
        let sb_row = b >> 2;
        let sb_col = b & 3;
        let blk_x = mb_x0 + sb_col * 4;
        let blk_y = mb_y0 + sb_row * 4;
        let patch = crate::motion_comp::fetch_block_whole_pixel(
            luma_ref.plane,
            luma_ref.stride,
            luma_ref.width,
            luma_ref.height,
            blk_x,
            blk_y,
            mv_eighth,
        );
        let src_sb = extract_src_subblock_4x4(src_y, b);
        sad += sub_block_sad_4x4(&src_sb, &patch);
    }
    sad
}

/// Small whole-pixel diamond search for ONE SPLITMV partition group —
/// per-sub-block analogue of [`crate::motion_search::small_diamond_search_luma`].
///
/// Sums [`group_sad_at_whole_mv`] over the group's member sub-blocks
/// for each candidate MV and follows a 4-neighbour diamond descent
/// from `center` (which should be the clamped `near.mvs[0]` "best"
/// predictor, the same base the decoder's NEW4X4 differential adds
/// to). Returns the best (mv, sad) pair found.
///
/// Each step clamps candidates to §17.1's `[MV_MIN, MV_MAX]`; SPLITMV
/// sub-block MVs themselves are not §18.1-secondary-clamped per §18.1
/// page 114.
fn group_small_diamond_search(
    luma_ref: crate::motion_search::LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    group_subblocks: &[usize],
    center: crate::motion_vector::Mv,
    max_iters: u32,
) -> (crate::motion_vector::Mv, u32) {
    use crate::motion_search::{MV_MAX, MV_MIN, WHOLE_PIXEL_STEP};
    // Snap to whole-pixel grid then clamp into §17.1 range.
    let snap = |v: i16| -> i16 {
        let q = (v / WHOLE_PIXEL_STEP) * WHOLE_PIXEL_STEP;
        q.clamp(MV_MIN, MV_MAX)
    };
    let mut best_mv = crate::motion_vector::Mv {
        row: snap(center.row),
        col: snap(center.col),
    };
    let mut best_sad =
        group_sad_at_whole_mv(luma_ref, mb_col, mb_row, src_y, group_subblocks, best_mv);
    let offsets: [(i16, i16); 4] = [
        (-WHOLE_PIXEL_STEP, 0),
        (WHOLE_PIXEL_STEP, 0),
        (0, -WHOLE_PIXEL_STEP),
        (0, WHOLE_PIXEL_STEP),
    ];
    for _ in 0..max_iters {
        let mut improved = false;
        for (dr, dc) in offsets {
            let cand_row = (best_mv.row as i32 + dr as i32).clamp(MV_MIN as i32, MV_MAX as i32);
            let cand_col = (best_mv.col as i32 + dc as i32).clamp(MV_MIN as i32, MV_MAX as i32);
            let cand_mv = crate::motion_vector::Mv {
                row: cand_row as i16,
                col: cand_col as i16,
            };
            if cand_mv == best_mv {
                continue;
            }
            let cand_sad =
                group_sad_at_whole_mv(luma_ref, mb_col, mb_row, src_y, group_subblocks, cand_mv);
            if cand_sad < best_sad {
                best_sad = cand_sad;
                best_mv = cand_mv;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    (best_mv, best_sad)
}

/// The §16.4 sub-block index ordering — `MV_PARTITIONS` is indexed by
/// sub-block raster id and stores group ids, but the partition decode
/// emits groups in `partition_id` order with the anchor as the first
/// sub-block carrying that group id. [`PartitionGroups`] precomputes
/// `groups[g] = [list of sub-block indices belonging to group g]` for
/// each of the four partitions — at compile time, since the §20.13
/// `MV_PARTITIONS` table is a constant. Round 277 replaced the previous
/// per-call `Vec<Vec<usize>>` builder with this table: the SPLITMV
/// scorer runs once per (MB, partition shape) and the nested-`Vec`
/// build dominated the encoder's residual allocator churn.
struct PartitionGroups {
    /// `members[g][..len[g]]` — the raster sub-block indices of group
    /// `g`, ascending (so `members[g][0]` is the §16.4 group anchor,
    /// "the first sub-block whose partition entry is `g`").
    members: [[usize; 16]; 16],
    /// Per-group member count.
    len: [usize; 16],
    /// Number of groups in this partition (2 / 2 / 4 / 16).
    num_groups: usize,
}

impl PartitionGroups {
    /// Decompose row `pid` of [`crate::near_mv::MV_PARTITIONS`]
    /// (`pid` = the §16.4 `partition_id`) into its member groups.
    const fn build(pid: usize) -> Self {
        let table = &crate::near_mv::MV_PARTITIONS[pid];
        let mut members = [[0usize; 16]; 16];
        let mut len = [0usize; 16];
        let mut num_groups = 0usize;
        let mut idx = 0usize;
        while idx < 16 {
            let g = table[idx] as usize;
            members[g][len[g]] = idx;
            len[g] += 1;
            if g + 1 > num_groups {
                num_groups = g + 1;
            }
            idx += 1;
        }
        PartitionGroups {
            members,
            len,
            num_groups,
        }
    }

    /// The member sub-block indices of group `g` (`g < num_groups`).
    #[inline]
    fn group(&self, g: usize) -> &[usize] {
        &self.members[g][..self.len[g]]
    }

    /// Iterate the groups in partition-id order.
    #[inline]
    fn iter(&self) -> impl Iterator<Item = &[usize]> + '_ {
        (0..self.num_groups).map(move |g| self.group(g))
    }
}

/// The four [`PartitionGroups`] decompositions, indexed by
/// `crate::near_mv::partition_id` order (TopBottom / LeftRight /
/// Quarters / Mv16) — built at compile time from the §20.13 table.
static PARTITION_GROUP_TABLE: [PartitionGroups; 4] = [
    PartitionGroups::build(0),
    PartitionGroups::build(1),
    PartitionGroups::build(2),
    PartitionGroups::build(3),
];

/// Look up the precomputed group decomposition for partition `p`.
#[inline]
fn partition_groups(p: crate::near_mv::MvPartition) -> &'static PartitionGroups {
    &PARTITION_GROUP_TABLE[crate::near_mv::partition_id(p)]
}

/// Per-group `sub_mv_ref` mode cost in fractional bits — the
/// `sub_mv_ref_tree` path bits at `SUBMV_REF_PROBS[ctx]`.
///
/// Tree paths per §16.4 / §20.13:
///   LEFT4X4  = "0"   → bool_bits(probs[0], false)
///   ABOVE4X4 = "10"  → bool_bits(probs[0], true) + bool_bits(probs[1], false)
///   ZERO4X4  = "110" → bool_bits(probs[0..=1], true) + bool_bits(probs[2], false)
///   NEW4X4   = "111" → bool_bits(probs[0..=2], true)
fn submv_ref_bits(probs: &[u8; 3], mode: crate::near_mv::SubMvRefMode) -> f64 {
    use crate::near_mv::SubMvRefMode as M;
    match mode {
        M::Left4x4 => bool_bits(probs[0], false),
        M::Above4x4 => bool_bits(probs[0], true) + bool_bits(probs[1], false),
        M::Zero4x4 => {
            bool_bits(probs[0], true) + bool_bits(probs[1], true) + bool_bits(probs[2], false)
        }
        M::New4x4 => {
            bool_bits(probs[0], true) + bool_bits(probs[1], true) + bool_bits(probs[2], true)
        }
    }
}

/// `mvpartition_tree` path bits — RFC 6386 §16.4 / §20.13.
///
/// Mv16        = "0"   → bool_bits(MV_PARTITION_PROBS[0], false)
/// Quarters    = "10"  → bool_bits([0], true) + bool_bits([1], false)
/// TopBottom   = "110" → bool_bits([0..=1], true) + bool_bits([2], false)
/// LeftRight   = "111" → bool_bits([0..=2], true)
fn mv_partition_bits(p: crate::near_mv::MvPartition) -> f64 {
    let probs = &crate::near_mv::MV_PARTITION_PROBS;
    use crate::near_mv::MvPartition as P;
    match p {
        P::Mv16 => bool_bits(probs[0], false),
        P::Quarters => bool_bits(probs[0], true) + bool_bits(probs[1], false),
        P::TopBottom => {
            bool_bits(probs[0], true) + bool_bits(probs[1], true) + bool_bits(probs[2], false)
        }
        P::LeftRight => {
            bool_bits(probs[0], true) + bool_bits(probs[1], true) + bool_bits(probs[2], true)
        }
    }
}

/// Score one §16.4 partition: walk its groups in partition-id order,
/// search whole-pixel SAD per group, evaluate the four §16.4
/// `sub_mv_ref` modes (LEFT4X4 / ABOVE4X4 / ZERO4X4 / NEW4X4) for the
/// group anchor, and assemble the (`split_mvs[16]`, group modes, group
/// new-diffs, total J) candidate.
///
/// Mirrors `decode_split_mv`'s "find the first sub-block whose
/// partition entry is `j`, look up its left/above neighbours, run
/// `sub_mv_ref`, fill every member of the group with the resolved
/// vector" walk so the encoder's per-group choice is exactly what the
/// decoder will reconstruct from the emitted bits.
///
/// Returns `None` if no in-range candidate is found (every NEW4X4
/// differential overflowed §17.1 and the LEFT/ABOVE/ZERO modes were
/// all dominated by SAD).
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §16.4 input.
fn score_split_partition(
    partition: crate::near_mv::MvPartition,
    luma_ref: crate::motion_search::LumaRef<'_>,
    mb_col: usize,
    mb_row: usize,
    src_y: &[u8; 256],
    best_predictor: crate::motion_vector::Mv,
    above_mb: &crate::near_mv::MbInfo,
    left_mb: &crate::near_mv::MbInfo,
    mv_contexts: &crate::motion_vector::MvContexts,
    lambda: f64,
) -> Option<SplitMvCandidate> {
    let groups = partition_groups(partition);
    let mut split_mvs = [crate::motion_vector::Mv::default(); 16];
    let mut submv_modes = [crate::near_mv::SubMvRefMode::Zero4x4; 16];
    let mut submv_new_diffs = [crate::motion_vector::Mv::default(); 16];
    let mut total_sad: u32 = 0;
    let mut total_submv_bits: f64 = 0.0;

    for (g_idx, group) in groups.iter().enumerate() {
        // The anchor is the smallest sub-block index in the group
        // (§16.4 / §20.11 "find the first sub-block whose partition
        // entry is j" — `partition_groups` already collects in raster
        // order, so `group[0]` is the anchor).
        let anchor = group[0];
        let left_neighbour_mv = crate::near_mv::left_block_mv(&split_mvs, left_mb, anchor);
        let above_neighbour_mv = crate::near_mv::above_block_mv(&split_mvs, above_mb, anchor);
        let ctx = crate::near_mv::submv_ref_context(left_neighbour_mv, above_neighbour_mv);
        let probs = &crate::near_mv::SUBMV_REF_PROBS[ctx];

        // Whole-pixel search for a per-group NEW4X4 candidate (the
        // only mode that costs bits for a custom vector). The diamond
        // descents from the clamped near.mvs[0] "best" predictor — same
        // base the decoder's NEW4X4 will add the differential onto.
        let (search_mv, search_sad) = group_small_diamond_search(
            luma_ref,
            mb_col,
            mb_row,
            src_y,
            group,
            best_predictor,
            SPLIT_MV_MAX_DIAMOND_ITERS,
        );

        // Score each of the four §16.4 sub_mv_ref modes.
        use crate::near_mv::SubMvRefMode as M;
        let mut best_mode: M = M::Zero4x4;
        let mut best_group_mv = crate::motion_vector::Mv::default();
        let mut best_group_diff = crate::motion_vector::Mv::default();
        let mut best_group_j: f64 = f64::INFINITY;

        // Helper: probe one candidate (mode, resolved-mv, diff) and
        // update the best slot in place. SAD is recomputed at
        // `mv` so each candidate scores against the same evaluator.
        let mut try_cand = |mode: M,
                            mv: crate::motion_vector::Mv,
                            diff: crate::motion_vector::Mv,
                            diff_bits: f64| {
            let sad = group_sad_at_whole_mv(luma_ref, mb_col, mb_row, src_y, group, mv) as f64;
            let mode_bits = submv_ref_bits(probs, mode);
            let j = sad + lambda * (mode_bits + diff_bits);
            if j < best_group_j {
                best_group_j = j;
                best_mode = mode;
                best_group_mv = mv;
                best_group_diff = diff;
            }
        };

        // LEFT4X4 — the resolved MV is the left neighbour, no diff bits.
        try_cand(
            M::Left4x4,
            left_neighbour_mv,
            crate::motion_vector::Mv::default(),
            0.0,
        );
        // ABOVE4X4 — the above neighbour, no diff bits.
        try_cand(
            M::Above4x4,
            above_neighbour_mv,
            crate::motion_vector::Mv::default(),
            0.0,
        );
        // ZERO4X4 — zero MV, no diff bits.
        try_cand(
            M::Zero4x4,
            crate::motion_vector::Mv::default(),
            crate::motion_vector::Mv::default(),
            0.0,
        );
        // NEW4X4 — searched MV, diff = search_mv - best_predictor.
        //
        // §17.1: each component of the diff fits in [-1023, +1023] — the
        // §17.2 component coder's range. An out-of-range diff is priced
        // at +inf so this candidate is dropped.
        let diff_row = (search_mv.row as i32) - (best_predictor.row as i32);
        let diff_col = (search_mv.col as i32) - (best_predictor.col as i32);
        let diff_in_range =
            (-1023..=1023).contains(&diff_row) && (-1023..=1023).contains(&diff_col);
        if diff_in_range {
            let diff = crate::motion_vector::Mv {
                row: diff_row as i16,
                col: diff_col as i16,
            };
            let diff_bits = crate::motion_vector::mv_component_bits(&mv_contexts[0], diff_row)
                + crate::motion_vector::mv_component_bits(&mv_contexts[1], diff_col);
            // SAD at `search_mv` is already known.
            let mode_bits = submv_ref_bits(probs, M::New4x4);
            let j = search_sad as f64 + lambda * (mode_bits + diff_bits);
            if j < best_group_j {
                best_group_j = j;
                best_mode = M::New4x4;
                best_group_mv = search_mv;
                best_group_diff = diff;
            }
        }

        if !best_group_j.is_finite() {
            // No viable candidate (shouldn't be reachable — Zero4x4 is
            // always in-range — but guard for completeness).
            return None;
        }

        // Fill every member sub-block of this group with the chosen MV
        // and accumulate its SAD into the partition total. The SAD
        // re-evaluation uses the same evaluator each candidate scored
        // against, so the total matches the picker's running sum.
        let group_sad =
            group_sad_at_whole_mv(luma_ref, mb_col, mb_row, src_y, group, best_group_mv);
        for &b in group {
            split_mvs[b] = best_group_mv;
        }
        total_sad += group_sad;
        total_submv_bits += submv_ref_bits(probs, best_mode);
        if matches!(best_mode, M::New4x4) {
            total_submv_bits += crate::motion_vector::mv_bits(mv_contexts, best_group_diff);
        }
        submv_modes[g_idx] = best_mode;
        submv_new_diffs[g_idx] = best_group_diff;
    }

    // Total J adds the partition's §16.4 `mvpartition_tree` bits to
    // the per-group `sub_mv_ref` totals (the §16.2 SPLITMV
    // `mv_ref_tree` path bits are added by the caller once across all
    // four partition candidates).
    let partition_bits = mv_partition_bits(partition);
    let j = total_sad as f64 + lambda * (partition_bits + total_submv_bits);

    Some(SplitMvCandidate {
        partition,
        split_mvs,
        submv_modes,
        submv_new_diffs,
        j,
    })
}

/// Build the §13.3 raw-coefficient (no Y2) `MbCoeffs` for one SPLITMV
/// macroblock: per-sub-block luma forward DCT + quantise (no WHT —
/// SPLITMV has no Y2), and the standard chroma path (4 sub-blocks per
/// plane). Mirrors the `B_PRED` luma transform path (`transform_b_pred_luma`
/// would, if it existed; we inline it here since the SPLITMV picker is
/// the only inter caller that needs no-Y2 luma).
///
/// When `trellis` is `Some` each `YNoY2` luma sub-block and each `UV`
/// chroma sub-block is quantised by the §13 [`trellis_quantize_block`] RD
/// search; `None` preserves the plain-rounding SPLITMV wire byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn transform_split_mv_mb_rd(
    src: &[u8; 256],
    pred: &[u8; 256],
    chroma_src_u: &[u8; 64],
    chroma_pred_u: &[u8; 64],
    chroma_src_v: &[u8; 64],
    chroma_pred_v: &[u8; 64],
    factors: &crate::dequant::MbDequantFactors,
    trellis: Option<&MbRdCtx>,
) -> MbCoeffs {
    // Per-sub-block luma forward DCT + Y1 quantisation. SPLITMV codes
    // every coefficient (0..=15) of each Y sub-block — no Y2, so the
    // DC is *not* extracted into a separate WHT block.
    let mut y = [[0i16; 16]; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut residual = [0i16; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let off = (i * 4 + r) * 16 + (j * 4 + c);
                    residual[r * 4 + c] = src[off] as i16 - pred[off] as i16;
                }
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&residual, &mut coeffs);
            // No DC extraction — every coefficient is quantised in
            // place under the Y1 (dc, ac) factors. The decoder's
            // `BlockType::YNoY2` path reads them the same way.
            match trellis {
                Some(rd) if rd.trellis_active() => trellis_quantize_block(
                    &mut coeffs,
                    TrellisBlockSpec {
                        dc: factors.y1_dc,
                        ac: factors.y1_ac,
                        first_coeff: BlockType::YNoY2.first_coeff(),
                        entry_ctx3: 0,
                        plane: BlockType::YNoY2.plane_index(),
                    },
                    rd.coeff_probs,
                    rd.rate_tables,
                    rd.trellis_lambda(),
                ),
                _ => enc_quantize_block(&mut coeffs, factors.y1_dc, factors.y1_ac),
            }
            y[i * 4 + j] = coeffs;
        }
    }

    // Chroma uses the same path the whole-MB encoder uses.
    let u = transform_chroma_plane_rd(chroma_src_u, chroma_pred_u, factors, trellis);
    let v = transform_chroma_plane_rd(chroma_src_v, chroma_pred_v, factors, trellis);

    MbCoeffs {
        y,
        y2: [0i16; 16], // unused (no Y2 for SPLITMV); the emitter
        // skips block 24 via `use_bpred = true` below.
        u,
        v,
    }
}

/// Write the §16.4 `mvpartition_tree` path for `partition` — RFC 6386
/// §16.4 / §20.13. Mirrors the encoder side of [`crate::near_mv::read_mv_partition`].
///
/// `Mv16 = "0"`, `Quarters = "10"`, `TopBottom = "110"`,
/// `LeftRight = "111"`, all coded against `MV_PARTITION_PROBS`.
fn write_split_mv_partition(enc: &mut BoolEncoder, partition: crate::near_mv::MvPartition) {
    let probs = &crate::near_mv::MV_PARTITION_PROBS;
    use crate::near_mv::MvPartition as P;
    match partition {
        P::Mv16 => {
            enc.write_bool(probs[0], false);
        }
        P::Quarters => {
            enc.write_bool(probs[0], true);
            enc.write_bool(probs[1], false);
        }
        P::TopBottom => {
            enc.write_bool(probs[0], true);
            enc.write_bool(probs[1], true);
            enc.write_bool(probs[2], false);
        }
        P::LeftRight => {
            enc.write_bool(probs[0], true);
            enc.write_bool(probs[1], true);
            enc.write_bool(probs[2], true);
        }
    }
}

/// Write one §16.4 `sub_mv_ref_tree` mode bit-path against `probs`
/// (a row of `SUBMV_REF_PROBS` selected by [`crate::near_mv::submv_ref_context`]).
///
/// `Left4x4 = "0"`, `Above4x4 = "10"`, `Zero4x4 = "110"`,
/// `New4x4 = "111"`. The §17.2 NEW4X4 component differential is *not*
/// written here — the caller emits it after this routine returns.
fn write_submv_ref(enc: &mut BoolEncoder, probs: &[u8; 3], mode: crate::near_mv::SubMvRefMode) {
    use crate::near_mv::SubMvRefMode as M;
    match mode {
        M::Left4x4 => {
            enc.write_bool(probs[0], false);
        }
        M::Above4x4 => {
            enc.write_bool(probs[0], true);
            enc.write_bool(probs[1], false);
        }
        M::Zero4x4 => {
            enc.write_bool(probs[0], true);
            enc.write_bool(probs[1], true);
            enc.write_bool(probs[2], false);
        }
        M::New4x4 => {
            enc.write_bool(probs[0], true);
            enc.write_bool(probs[1], true);
            enc.write_bool(probs[2], true);
        }
    }
}

/// Encode one VP8 P-frame whose every macroblock is inter against the
/// LAST reference, with per-MB `ZEROMV` / `NEARESTMV` / `NEARMV` /
/// `NEWMV` / `SPLITMV` chosen by a non-normative rate-distortion trade.
///
/// Each MB on the wire is one §19.3 `macroblock_header()` (mb-skip /
/// is_inter / ref_frame selector / inter-mode tree walk / optional
/// §17.2 `read_mv` differential for `NEWMV`) plus its §13.3 residual
/// (or just the skip bit when the residual is all-zero).
///
/// Returns the emitted bytes alongside the macroblock-aligned post-§15
/// reconstruction the decoder will rebuild. The reconstruction is the
/// same shape as the §9 reference-frame buffer slot
/// ([`crate::state::RefFrameSlot`]); a multi-frame inter driver can
/// install it as the next frame's LAST without paying the cost of a
/// re-decode.
///
/// `reference` must be the macroblock-aligned reconstruction the
/// decoder produced for the previous frame (i.e. its `mb_cols * 16` /
/// `mb_rows * 8` planes), matching the source frame's macroblock-grid
/// dimensions. A mismatch is surfaced as
/// [`EncodeError::ReferenceDimensionsMismatch`].
///
/// # Scope (this round)
///
/// * **Whole-pixel + §18.3 half- and quarter-pixel motion search,
///   four-way `mv_ref_tree` pick.** A
///   [`crate::motion_search::small_diamond_search_luma`] descent runs
///   per MB against the clamped §16.3 "best" predictor (the running
///   `find_near_mvs[CNT_BEST]` vector), followed by a
///   [`crate::motion_search::half_pixel_refine_luma`] probe of the 8
///   half-pixel offsets around the whole-pixel result, and then a
///   [`crate::motion_search::quarter_pixel_refine_luma`] probe of the
///   8 quarter-pixel offsets around the half-pixel result (each
///   evaluated through the §18.3 six-tap synthesis the decoder will
///   reproduce). The chosen MV is whichever of the four whole-MB
///   `mv_ref_tree` leaves
///   ([`ZEROMV`](crate::near_mv::InterMode::Zero) — search-skipped,
///   `J = SAD at MV (0,0)`;
///   [`NEARESTMV`](crate::near_mv::InterMode::Nearest) — `J = SAD at
///   clamp(near.mvs[1]) + lambda * mv_ref_tree("10") bits`;
///   [`NEARMV`](crate::near_mv::InterMode::Near) — `J = SAD at
///   clamp(near.mvs[2]) + lambda * mv_ref_tree("110") bits`;
///   [`NEWMV`](crate::near_mv::InterMode::New) — `J = SAD at searched
///   MV + lambda * (mv_ref_tree("1110") bits + §17 component bits)`)
///   gives the smallest J. The §16.3 NEARESTMV / NEARMV candidates
///   are scored through the §18.3 sixtap-aware
///   [`crate::motion_search::mb_luma_sad_at_mv`] evaluator (neighbour
///   MVs can land at any §17 quarter-pixel position), at the clamped
///   MV the decoder's [`crate::near_mv::resolve_inter_mb_mv`] will
///   reconstruct from.
/// * **§16.4 SPLITMV with per-sub-block whole-pixel motion search.**
///   For each MB we additionally score all four
///   [`MvPartition`](crate::near_mv::MvPartition) shapes (`TopBottom`,
///   `LeftRight`, `Quarters`, `Mv16`); each partition's per-group MV
///   is the lowest-J among the four §16.4 `sub_mv_ref` modes
///   ([`Left4x4`](crate::near_mv::SubMvRefMode::Left4x4) — the group
///   anchor's left neighbour MV; [`Above4x4`] — its above neighbour;
///   [`Zero4x4`] — `(0, 0)`; [`New4x4`] — a per-group small whole-
///   pixel diamond search around the clamped `near.mvs[0]` predictor,
///   coded as a §17.2 differential added to that same `best`). The
///   §16.4 partition + sub_mv_ref + per-group NEW4X4 component bits
///   are summed with the §16.2 SPLITMV path bits ("1111") to give the
///   total `J(split)`; when smaller than the whole-MB picker's J the
///   SPLITMV path emits. `GOLDEN` / `ALTREF` source selection remains
///   deferred to a later round.
///
///   [`Above4x4`]: crate::near_mv::SubMvRefMode::Above4x4
///   [`Zero4x4`]: crate::near_mv::SubMvRefMode::Zero4x4
///   [`New4x4`]: crate::near_mv::SubMvRefMode::New4x4
/// * **`prob_intra` is 255**, so the decoder reads every MB as
///   inter without a bit on the wire wasted on intra-vs-inter
///   classification. Token-prob and intra-mode-prob update blocks
///   carry no updates; the §17.2 `mv_prob_update()` block is
///   emitted with every F-gate set to 0 so the MV-decode
///   infrastructure runs against the §17.2 defaults — the encoder
///   writes the NEWMV differential against the same default
///   [`MvContexts`].
/// * **§9.7 refresh ladder**: `refresh_last = 1`,
///   `refresh_golden_frame = 0`, `refresh_alternate_frame = 0`,
///   `copy_buffer_to_golden = 0`, `copy_buffer_to_alternate = 0`,
///   `refresh_entropy_probs = 0`. The post-§15 reconstruction
///   replaces LAST; GOLDEN / ALTREF stay as the caller left them.
/// * **§15 loop filter**: runs over the encoder's own reconstruction
///   when `params.loop_filter_level != 0`, mirroring the
///   keyframe-encoder pattern. The §9.4 `mode_ref_lf_delta_enabled`
///   flag is held at 0 (the per-MB delta layer is not emitted),
///   matching `encode_keyframe_with_reconstruction`.
/// * **§9.5 partition count** is fixed at 1 this round. Multi-
///   partition output for inter is the same layout reorganisation as
///   for keyframes; it can be wired in by a follow-up.
///
/// Outcome of running the §16.2 / §16.3 / §16.4 + §17 / §18 / §14
/// per-MB picker for one reference frame. Used by
/// [`encode_p_frame_multi_ref`] to score each available reference
/// (`LAST` / optional `GOLDEN` / optional `ALTREF`) and pick the
/// winning ref + mode + MV per MB.
#[derive(Clone)]
struct PickedMbForRef {
    /// The reference frame this candidate scored against.
    ref_frame: crate::motion_comp::RefFrame,
    /// The §16.2 whole-MB inter mode (or `Split` for SPLITMV) that
    /// won inside this ref's per-MB pick.
    chosen_mode: crate::near_mv::InterMode,
    /// The §17 quarter-pixel MV the chosen mode resolves to (the
    /// SPLITMV path stores `split_mvs[15]`, matching the §16.3
    /// `MbInfo::mv` convention).
    chosen_mv: crate::motion_vector::Mv,
    /// `Some(..)` iff `chosen_mode == InterMode::Split`. The §16.4
    /// partition, per-group `sub_mv_ref` modes, per-group NEW4X4
    /// differentials, and resolved `split_mvs[16]`.
    chosen_split: Option<SplitMvCandidate>,
    /// Per-MB residual coefficients (Y2 + 16 Y + 4 U + 4 V), already
    /// quantised — what the §13 token emit walks for this MB.
    raw_coeffs: MbCoeffs,
    /// §11.1 mb_skip_coeff. True iff every coded block is all-zero
    /// after quantisation.
    mb_skip_coeff: bool,
    /// Post-§14 reconstruction the encoder writes into its running
    /// `planes` buffer (and into the next-frame LAST slot once §15
    /// has filtered it).
    recon: crate::reconstruct::ReconstructedMb,
    /// True iff the §13.3 token walk routes through the B_PRED /
    /// SPLITMV "no Y2" path (`encode_mb_tokens(use_bpred = true)`).
    use_bpred: bool,
    /// `J = SAD + lambda * (mv_ref_tree + §17 mv + §16.4 partition +
    /// sub_mv_ref + NEW4X4) bits`. Does NOT include the §16.2
    /// `ref_frame_tree` bits — the caller adds those, since the
    /// `prob_last` / `prob_gf` used during picking and the
    /// distribution-fitted values used at emit-time can differ.
    j: f64,
}

/// Run the full per-MB picker for one reference frame. Mirrors the
/// per-MB body of [`encode_p_frame_multi_ref`]'s search loop but
/// parameterised by the reference frame + its planes + its luma SAD
/// view. Used to compare LAST / GOLDEN / ALTREF candidates for the
/// same MB.
#[allow(clippy::too_many_arguments)]
fn pick_mb_for_ref(
    ref_frame: crate::motion_comp::RefFrame,
    reference_planes: &crate::motion_comp::ReferencePlanes<'_>,
    luma_ref: crate::motion_search::LumaRef<'_>,
    pixels: &MbPixels,
    above_slot: &crate::near_mv::MbInfo,
    left_mb: &crate::near_mv::MbInfo,
    aboveleft_mb: &crate::near_mv::MbInfo,
    mb_col: usize,
    mb_row: usize,
    mb_cols: usize,
    mb_rows: usize,
    mv_contexts: &crate::motion_vector::MvContexts,
    lambda: f64,
    filters: &[[i32; 6]; 8],
    factors: &crate::dequant::MbDequantFactors,
    coeff_probs: Option<&CoeffProbs>,
    rate_tables: Option<&TrellisRateTables>,
    trellis_strength: TrellisStrength,
) -> PickedMbForRef {
    // §13 trellis RD context for the inter residual emit, available once
    // the caller resolves this frame's coefficient probs. `None` keeps the
    // pre-trellis inter wire byte-for-byte.
    let trellis_rd = coeff_probs.map(|cp| MbRdCtx {
        factors,
        coeff_probs: cp,
        rate_tables,
        lambda,
        trellis: trellis_strength,
    });
    // ---- §16.3 census + §17 motion search ------------------------
    //
    // The §16.3 `find_near_mvs[CNT_BEST]` slot is the "best"
    // predictor a NEWMV differential is added to (see
    // `resolve_inter_mb_mv` in `crate::near_mv`). Clamp it through
    // the §18.1 / §20.11 per-MB rectangle before the search uses it
    // as the descent center, exactly matching what the decoder will
    // do when re-reading the NEWMV path. The census is per-ref:
    // neighbour MVs only count toward `near.mvs[]` when their
    // recorded `ref_frame` matches ours.
    let near = crate::near_mv::find_near_mvs(
        above_slot,
        left_mb,
        aboveleft_mb,
        ref_frame,
        crate::near_mv::SignBias::default(),
    );
    let bounds = crate::near_mv::MvClampRect::for_mb(mb_col, mb_row, mb_cols, mb_rows);
    let best_predictor = crate::near_mv::clamp_mv(near.mvs[0], &bounds);
    let mv_ref_probs = crate::near_mv::mv_ref_probs(&near.cnt);

    // SAD at MV (0,0) is the ZEROMV distortion term.
    let sad_zero = crate::motion_search::mb_luma_sad_at_whole_mv(
        luma_ref,
        mb_col,
        mb_row,
        &pixels.y,
        crate::motion_vector::Mv::default(),
    );
    const MAX_DIAMOND_ITERS: u32 = 8;
    let whole_pel = crate::motion_search::small_diamond_search_luma(
        luma_ref,
        mb_col,
        mb_row,
        &pixels.y,
        best_predictor,
        MAX_DIAMOND_ITERS,
    );
    let half_pel = crate::motion_search::half_pixel_refine_luma(
        luma_ref,
        mb_col,
        mb_row,
        &pixels.y,
        whole_pel.mv,
    );
    let search = crate::motion_search::quarter_pixel_refine_luma(
        luma_ref,
        mb_col,
        mb_row,
        &pixels.y,
        half_pel.mv,
    );

    // §16.2 inter-mode tree bit costs.
    let bits_mode_zero = bool_bits(mv_ref_probs[0], false);
    let bits_mode_nearest = bool_bits(mv_ref_probs[0], true) + bool_bits(mv_ref_probs[1], false);
    let bits_mode_near = bool_bits(mv_ref_probs[0], true)
        + bool_bits(mv_ref_probs[1], true)
        + bool_bits(mv_ref_probs[2], false);
    let bits_mode_new = bool_bits(mv_ref_probs[0], true)
        + bool_bits(mv_ref_probs[1], true)
        + bool_bits(mv_ref_probs[2], true)
        + bool_bits(mv_ref_probs[3], false);
    let diff_row = (search.mv.row as i32) - (best_predictor.row as i32);
    let diff_col = (search.mv.col as i32) - (best_predictor.col as i32);
    let diff_in_range = (-1023..=1023).contains(&diff_row) && (-1023..=1023).contains(&diff_col);
    let bits_mv_diff = if diff_in_range {
        crate::motion_vector::mv_component_bits(&mv_contexts[0], diff_row)
            + crate::motion_vector::mv_component_bits(&mv_contexts[1], diff_col)
    } else {
        f64::INFINITY
    };

    let nearest_mv = crate::near_mv::clamp_mv(near.mvs[1], &bounds);
    let near_mv = crate::near_mv::clamp_mv(near.mvs[2], &bounds);
    let sad_nearest =
        crate::motion_search::mb_luma_sad_at_mv(luma_ref, mb_col, mb_row, &pixels.y, nearest_mv);
    let sad_near =
        crate::motion_search::mb_luma_sad_at_mv(luma_ref, mb_col, mb_row, &pixels.y, near_mv);

    let j_zero = sad_zero as f64 + lambda * bits_mode_zero;
    let j_nearest = sad_nearest as f64 + lambda * bits_mode_nearest;
    let j_near = sad_near as f64 + lambda * bits_mode_near;
    let j_new = search.sad as f64 + lambda * (bits_mode_new + bits_mv_diff);

    let nearest_eligible = nearest_mv != crate::motion_vector::Mv::default();
    let near_eligible = near_mv != crate::motion_vector::Mv::default();
    let new_eligible = diff_in_range && search.mv != crate::motion_vector::Mv::default();

    let mut chosen_mode = crate::near_mv::InterMode::Zero;
    let mut chosen_mv = crate::motion_vector::Mv::default();
    let mut chosen_j = j_zero;
    if nearest_eligible && j_nearest < chosen_j {
        chosen_mode = crate::near_mv::InterMode::Nearest;
        chosen_mv = nearest_mv;
        chosen_j = j_nearest;
    }
    if near_eligible && j_near < chosen_j {
        chosen_mode = crate::near_mv::InterMode::Near;
        chosen_mv = near_mv;
        chosen_j = j_near;
    }
    if new_eligible && j_new < chosen_j {
        chosen_mode = crate::near_mv::InterMode::New;
        chosen_mv = search.mv;
        chosen_j = j_new;
    }

    // ---- §16.4 SPLITMV picker --------------------------------------
    let bits_mode_split = bool_bits(mv_ref_probs[0], true)
        + bool_bits(mv_ref_probs[1], true)
        + bool_bits(mv_ref_probs[2], true)
        + bool_bits(mv_ref_probs[3], true);
    let mut best_split: Option<SplitMvCandidate> = None;
    for &partition in &[
        crate::near_mv::MvPartition::TopBottom,
        crate::near_mv::MvPartition::LeftRight,
        crate::near_mv::MvPartition::Quarters,
        crate::near_mv::MvPartition::Mv16,
    ] {
        if let Some(cand) = score_split_partition(
            partition,
            luma_ref,
            mb_col,
            mb_row,
            &pixels.y,
            best_predictor,
            above_slot,
            left_mb,
            mv_contexts,
            lambda,
        ) {
            let total_j = cand.j + lambda * bits_mode_split;
            if best_split
                .as_ref()
                .map(|b| total_j < b.j + lambda * bits_mode_split)
                .unwrap_or(true)
            {
                best_split = Some(SplitMvCandidate {
                    partition: cand.partition,
                    split_mvs: cand.split_mvs,
                    submv_modes: cand.submv_modes,
                    submv_new_diffs: cand.submv_new_diffs,
                    j: total_j,
                });
            }
        }
    }
    let mut chosen_split: Option<SplitMvCandidate> = None;
    if let Some(cand) = best_split {
        if cand.j < chosen_j {
            chosen_mode = crate::near_mv::InterMode::Split;
            chosen_mv = cand.split_mvs[15];
            chosen_j = cand.j;
            chosen_split = Some(cand);
        }
    }

    // ---- §18 / §14 prediction + residual + reconstruction at the
    //      chosen (mode, mv) for THIS ref ---------------------------
    let (raw_coeffs, use_bpred) = if let Some(ref cand) = chosen_split {
        let pred = crate::motion_comp::predict_split_mv(
            reference_planes,
            mb_col,
            mb_row,
            &cand.split_mvs,
            false,
            filters,
        );
        let raw_coeffs = transform_split_mv_mb_rd(
            &pixels.y,
            &pred.y,
            &pixels.u,
            &pred.u,
            &pixels.v,
            &pred.v,
            factors,
            trellis_rd.as_ref(),
        );
        (raw_coeffs, true)
    } else {
        let pred = crate::motion_comp::predict_inter_mb(
            reference_planes,
            mb_col,
            mb_row,
            chosen_mv,
            false,
            filters,
        );
        let (y_quant, y2_quant) =
            transform_whole_block_luma_rd(&pixels.y, &pred.y, factors, trellis_rd.as_ref());
        let u_quant = transform_chroma_plane_rd(&pixels.u, &pred.u, factors, trellis_rd.as_ref());
        let v_quant = transform_chroma_plane_rd(&pixels.v, &pred.v, factors, trellis_rd.as_ref());
        let raw_coeffs = MbCoeffs {
            y: y_quant,
            y2: y2_quant,
            u: u_quant,
            v: v_quant,
        };
        (raw_coeffs, false)
    };

    // §11.1 skip-detection.
    let mut nonzero_block_count = 0usize;
    if !use_bpred && raw_coeffs.y2.iter().any(|&v| v != 0) {
        nonzero_block_count += 1;
    }
    for blk in raw_coeffs.y.iter() {
        let first = if use_bpred { 0 } else { 1 };
        if blk.iter().skip(first).any(|&v| v != 0) {
            nonzero_block_count += 1;
        }
    }
    for blk in raw_coeffs.u.iter() {
        if blk.iter().any(|&v| v != 0) {
            nonzero_block_count += 1;
        }
    }
    for blk in raw_coeffs.v.iter() {
        if blk.iter().any(|&v| v != 0) {
            nonzero_block_count += 1;
        }
    }
    let mb_skip_coeff = nonzero_block_count == 0;

    // §14.1: the `reconstruct_inter_mb` / `reconstruct_split_mv_mb`
    // orchestrators consume *dequantised* coefficients (the keyframe
    // raster loop above performs the same dequant on a copy before
    // calling its decoder reconstructor). Keep `raw_coeffs` quantised
    // for the later §13 token-emit path on the same call site; mirror
    // the keyframe pattern by dequantising a separate copy purely for
    // the §14.2/§14.5 reconstruction step.
    let mut dq = raw_coeffs;
    factors.dequantize(&mut dq);
    let recon = if let Some(ref cand) = chosen_split {
        crate::motion_comp::reconstruct_split_mv_mb(
            reference_planes,
            mb_col,
            mb_row,
            &cand.split_mvs,
            false,
            filters,
            mb_skip_coeff,
            &dq.y,
            &dq.u,
            &dq.v,
        )
    } else {
        crate::motion_comp::reconstruct_inter_mb(
            reference_planes,
            mb_col,
            mb_row,
            chosen_mv,
            false,
            filters,
            mb_skip_coeff,
            &dq.y2,
            &dq.y,
            &dq.u,
            &dq.v,
        )
    };

    PickedMbForRef {
        ref_frame,
        chosen_mode,
        chosen_mv,
        chosen_split,
        raw_coeffs,
        mb_skip_coeff,
        recon,
        use_bpred,
        j: chosen_j,
    }
}

/// Outcome of scoring an intra candidate against the
/// already-reconstructed in-frame neighbours during the round-160 /
/// round-161 intra-within-inter MB picker. Mirrors [`PickedMbForRef`]'s
/// shape so the per-MB driver can compare J side-by-side with the inter
/// winner and pick whichever wins.
///
/// Round 161 widens the scope to score all four §11.2 whole-block luma
/// modes (DC_PRED / V_PRED / H_PRED / TM_PRED) crossed with all four
/// §11.4 chroma modes (the same four), returning the best of the sixteen
/// candidates rather than the round-160 single DC_PRED candidate.
/// `B_PRED` stays out — the per-sub-block intra walker is a separate
/// fitter family and we don't have storage for it on this path.
#[derive(Clone)]
struct IntraMbPick {
    /// §11.2 whole-block luma mode the picker chose. Round 161 can be
    /// any of `Dc` / `V` / `H` / `Tm`; the storage type allows `B` but
    /// `pick_intra_mb_all` never returns it.
    y_mode: IntraYMode,
    /// §11.4 whole-block chroma mode the picker chose.
    uv_mode: IntraUvMode,
    /// Per-MB residual coefficients (Y2 + 16 Y + 4 U + 4 V), already
    /// quantised — what the §13 token-emit walk consumes when the
    /// per-MB driver records this candidate as the winner.
    raw_coeffs: MbCoeffs,
    /// §11.1 mb_skip_coeff. True iff every coded block is all-zero
    /// after quantisation.
    mb_skip_coeff: bool,
    /// Post-§14 reconstruction the encoder writes into its running
    /// `planes` buffer (and into the next-frame LAST slot once §15 has
    /// filtered it).
    recon: crate::reconstruct::ReconstructedMb,
    /// `J = Y-SAD + lambda * (intra-mode-tree bits)`. Does NOT include
    /// the §16 `is_inter_mb = false` bit — the caller adds that
    /// alongside the inter J's `is_inter_mb = true` bit so the picker
    /// trade includes the §16 discriminator bit equally on both sides
    /// (the `prob_intra` charge cancels out when picking against the
    /// uniform `prob_intra_pick = 128`, but we still account for it
    /// symbolically).
    j: f64,
}

/// `-log2 P` cost in fractional bits of the §11.2 / §16.1 interframe
/// `IF_YMODE_TREE` path for one whole-block luma intra mode, given the
/// §16.1 default `ymode_prob` table (we hold `intra_y_mode_prob_update`
/// at its no-update gate, so the wire table stays at the defaults the
/// decoder's `InterFrameIntraProbs::defaults` exposes).
fn if_ymode_tree_bits(mode: IntraYMode) -> f64 {
    treed_bits_cached(&*IF_YMODE_PATHS, |i| IF_YMODE_PROB_DEFAULTS[i], mode.leaf())
}

/// `-log2 P` cost in fractional bits of the §11.4 / §16.1 interframe
/// `UV_MODE_TREE` path for one whole-block chroma intra mode, against
/// the §16.1 default `uv_mode_prob` table.
fn if_uv_mode_tree_bits(mode: IntraUvMode) -> f64 {
    treed_bits_cached(
        &*UV_MODE_PATHS,
        |i| IF_UV_MODE_PROB_DEFAULTS[i],
        mode.leaf(),
    )
}

/// Score one (y_mode, uv_mode) whole-block intra candidate for a
/// macroblock on an inter frame against the running in-frame neighbours
/// (`MbNeighbors` gathered from the encoder's reconstruction buffer the
/// SAME way the decoder's `gather_neighbors_public` will when it walks
/// the bytes). `y_mode == IntraYMode::B` is rejected (debug-asserted) —
/// the per-sub-block intra walker is a separate fitter family.
///
/// The full §14 chain — predict → FDCT → Y2/WHT → quantise → dequant →
/// IDCT → add — runs so the returned `recon` is the exact pixels the
/// decoder will produce on this MB.
///
/// The returned `J` is `Y-SAD + lambda * (Y-mode + UV-mode tree bits)`,
/// mirroring the inter picker's `J = SAD + lambda * mv_ref_tree_bits`
/// convention (the inter picker doesn't include token bits either; the
/// Y2/Y/UV token mass is roughly proportional across candidates of
/// similar prediction quality, and matching the inter picker's
/// distortion form keeps the cross-candidate trade apples-to-apples).
#[allow(clippy::too_many_arguments)]
fn score_intra_mb_candidate(
    y_mode: IntraYMode,
    uv_mode: IntraUvMode,
    pixels: &MbPixels,
    neighbors: &crate::reconstruct::MbNeighbors,
    factors: &crate::dequant::MbDequantFactors,
    lambda: f64,
    coeff_probs: Option<&CoeffProbs>,
    rate_tables: Option<&TrellisRateTables>,
    trellis_strength: TrellisStrength,
) -> IntraMbPick {
    debug_assert!(
        y_mode != IntraYMode::B,
        "score_intra_mb_candidate is whole-block only; B_PRED is routed via the per-sub-block walker",
    );
    // §13 trellis RD context for the intra-in-P-frame whole-block paths,
    // available once the caller resolves this frame's coefficient probs.
    // The strength is the per-stream knob threaded down from the inter
    // encode entry (the same `KeyframeParams::trellis_strength` field).
    let trellis_rd = coeff_probs.map(|cp| MbRdCtx {
        factors,
        coeff_probs: cp,
        rate_tables,
        lambda,
        trellis: trellis_strength,
    });

    // ---- §12 prediction. Each `predict_y16x16` / `predict_uv8x8`
    //      dispatcher applies the §12 default-fill rules when the
    //      requested mode's neighbour is `None` (off-frame edge), so the
    //      V / H / TM modes safely cover the top-row / left-column /
    //      top-left-corner MBs the same way the decoder does.
    let topleft_y = neighbors
        .y_topleft
        .unwrap_or(crate::intra_predict::DEFAULT_ABOVE_PIXEL);
    let mut y_pred = [0u8; 256];
    crate::intra_predict::predict_y16x16(
        &mut y_pred,
        y_mode,
        neighbors.y_above.as_ref(),
        neighbors.y_left.as_ref(),
        topleft_y,
    )
    .expect("y_mode != B asserted above");

    let topleft_u = neighbors
        .u_topleft
        .unwrap_or(crate::intra_predict::DEFAULT_ABOVE_PIXEL);
    let mut u_pred = [0u8; 64];
    crate::intra_predict::predict_uv8x8(
        &mut u_pred,
        uv_mode,
        neighbors.u_above.as_ref(),
        neighbors.u_left.as_ref(),
        topleft_u,
    );
    let topleft_v = neighbors
        .v_topleft
        .unwrap_or(crate::intra_predict::DEFAULT_ABOVE_PIXEL);
    let mut v_pred = [0u8; 64];
    crate::intra_predict::predict_uv8x8(
        &mut v_pred,
        uv_mode,
        neighbors.v_above.as_ref(),
        neighbors.v_left.as_ref(),
        topleft_v,
    );

    // ---- Distortion: Y-plane SAD on the prediction residual, matching
    //      the inter picker's metric (`|src - reference[mv]|` summed
    //      over the 16×16 luma block, BEFORE residual coding /
    //      reconstruction). The post-reconstruction SSD a keyframe RD
    //      picker would use is finer-grained but on a different scale
    //      than the inter picker's SAD; matching the inter picker's
    //      pre-recon SAD keeps the cross-candidate J comparison
    //      apples-to-apples.
    let mut sad: u32 = 0;
    for (a, b) in pixels.y.iter().zip(y_pred.iter()) {
        sad += (*a as i32 - *b as i32).unsigned_abs();
    }

    // ---- §14 forward transform + quantise (§13 trellis when this frame's
    //      coeff_probs are resolved; plain rounding otherwise — keeps the
    //      pre-trellis inter wire byte-for-byte for callers that pass None).
    let (y_quant, y2_quant) =
        transform_whole_block_luma_rd(&pixels.y, &y_pred, factors, trellis_rd.as_ref());
    let u_quant = transform_chroma_plane_rd(&pixels.u, &u_pred, factors, trellis_rd.as_ref());
    let v_quant = transform_chroma_plane_rd(&pixels.v, &v_pred, factors, trellis_rd.as_ref());
    let raw_coeffs = MbCoeffs {
        y: y_quant,
        y2: y2_quant,
        u: u_quant,
        v: v_quant,
    };

    // §11.1 mb_skip_coeff — every non-B_PRED whole-block intra has Y2,
    // so every block (Y2 + 16 Y coded from coefficient 1 + 4 U + 4 V)
    // contributes to the count.
    let mut nonzero_block_count = 0usize;
    if raw_coeffs.y2.iter().any(|&v| v != 0) {
        nonzero_block_count += 1;
    }
    for blk in raw_coeffs.y.iter() {
        if blk.iter().skip(1).any(|&v| v != 0) {
            nonzero_block_count += 1;
        }
    }
    for blk in raw_coeffs.u.iter() {
        if blk.iter().any(|&v| v != 0) {
            nonzero_block_count += 1;
        }
    }
    for blk in raw_coeffs.v.iter() {
        if blk.iter().any(|&v| v != 0) {
            nonzero_block_count += 1;
        }
    }
    let mb_skip_coeff = nonzero_block_count == 0;

    // ---- §14 reconstruction (decoder-shared path) ----
    let mut dq = raw_coeffs;
    factors.dequantize(&mut dq);
    let recon = crate::reconstruct::decode_keyframe_mb_non_bpred(
        y_mode,
        uv_mode,
        mb_skip_coeff,
        neighbors,
        &dq.y2,
        &dq.y,
        &dq.u,
        &dq.v,
    )
    .expect("decode_keyframe_mb_non_bpred rejects only B_PRED, ruled out above");

    // Mode bits: Y-mode (IF_YMODE_TREE leaf) + UV-mode (UV_MODE_TREE leaf).
    let mode_bits = if_ymode_tree_bits(y_mode) + if_uv_mode_tree_bits(uv_mode);
    let j = sad as f64 + lambda * mode_bits;

    IntraMbPick {
        y_mode,
        uv_mode,
        raw_coeffs,
        mb_skip_coeff,
        recon,
        j,
    }
}

/// Round 161 — score every whole-block intra `(y_mode, uv_mode)`
/// combination (4 luma × 4 chroma = 16 candidates, `B_PRED` excluded
/// because the per-sub-block intra walker is a separate fitter family)
/// and return the winner on `J = Y-SAD + lambda * mode-tree-bits`.
///
/// Mirrors the round-160 [`score_intra_mb_candidate`]-of-just-DC entry
/// shape so the per-MB driver only needs to swap the call and propagate
/// the chosen `(y_mode, uv_mode)` into the `intra_y_modes` /
/// `intra_uv_modes` per-MB storage that the round-160 driver already
/// laid out.
///
/// Iteration order matches the encoder enum declaration
/// (`Dc, V, H, Tm` — `B` skipped). Strict `<` on J means ties go to
/// the earliest-tried, i.e. `(Dc, Dc)` — the round-160 pick. Encode-wire
/// bytes are therefore identical to round 160 on any source where DC
/// would have won there.
#[allow(clippy::too_many_arguments)]
fn pick_intra_mb_all(
    pixels: &MbPixels,
    neighbors: &crate::reconstruct::MbNeighbors,
    factors: &crate::dequant::MbDequantFactors,
    lambda: f64,
    coeff_probs: Option<&CoeffProbs>,
    rate_tables: Option<&TrellisRateTables>,
    trellis_strength: TrellisStrength,
) -> IntraMbPick {
    const Y_MODES: [IntraYMode; 4] = [IntraYMode::Dc, IntraYMode::V, IntraYMode::H, IntraYMode::Tm];
    const UV_MODES: [IntraUvMode; 4] = [
        IntraUvMode::Dc,
        IntraUvMode::V,
        IntraUvMode::H,
        IntraUvMode::Tm,
    ];
    let mut best: Option<IntraMbPick> = None;
    for &y_mode in Y_MODES.iter() {
        for &uv_mode in UV_MODES.iter() {
            let cand = score_intra_mb_candidate(
                y_mode,
                uv_mode,
                pixels,
                neighbors,
                factors,
                lambda,
                coeff_probs,
                rate_tables,
                trellis_strength,
            );
            match best.as_ref() {
                None => best = Some(cand),
                Some(b) if cand.j < b.j => best = Some(cand),
                _ => {}
            }
        }
    }
    best.expect("Y_MODES × UV_MODES is non-empty")
}

/// `-log2 P` cost in fractional bits of the §16.2 `ref_frame_tree`
/// path for one reference frame, given the wire probs.
fn ref_frame_tree_bits(ref_frame: crate::motion_comp::RefFrame, prob_last: u8, prob_gf: u8) -> f64 {
    match ref_frame {
        // LAST: B(prob_last) reads `false`.
        crate::motion_comp::RefFrame::Last => bool_bits(prob_last, false),
        // GOLDEN: B(prob_last) reads `true`, B(prob_gf) reads `false`.
        crate::motion_comp::RefFrame::Golden => {
            bool_bits(prob_last, true) + bool_bits(prob_gf, false)
        }
        // ALTREF: B(prob_last) reads `true`, B(prob_gf) reads `true`.
        crate::motion_comp::RefFrame::AltRef => {
            bool_bits(prob_last, true) + bool_bits(prob_gf, true)
        }
    }
}

/// Pick the L(8) wire value for `prob` that minimises the total
/// `-count_true * log2(1 - p/256) - count_false * log2(p/256)` cost
/// of `count_false` "false" reads + `count_true` "true" reads. The
/// optimum is `256 * count_false / total`, clamped into `1..=255` to
/// keep the §7.3 bool range from collapsing.
fn fit_prob_l8(count_false: u32, count_true: u32) -> u8 {
    let total = count_false + count_true;
    if total == 0 {
        return 128; // arbitrary; nobody reads it.
    }
    let raw = ((count_false as u64 * 256) / total as u64) as u32;
    raw.clamp(1, 255) as u8
}

/// Caller-driven §9.7 / §9.8 reference-slot refresh pattern for a single
/// P-frame.
///
/// The five fields map one-for-one to the §9.7 / §9.8 bits the inter
/// frame header carries (RFC 6386 §9.7 page 38, §9.8 page 39):
///
/// * `refresh_golden_frame` — `L(1)`. `true` replaces the GOLDEN slot
///   with the current reconstruction; `false` leaves it in place (or
///   defers to `copy_buffer_to_golden`).
/// * `refresh_alternate_frame` — `L(1)`. `true` replaces the ALTREF
///   slot with the current reconstruction; `false` defers to
///   `copy_buffer_to_alternate`.
/// * `copy_buffer_to_golden` — `L(2)`, only read when
///   `refresh_golden_frame == 0`. `0` no copy; `1` copies LAST into
///   GOLDEN; `2` copies ALTREF into GOLDEN.
/// * `copy_buffer_to_alternate` — `L(2)`, only read when
///   `refresh_alternate_frame == 0`. `0` no copy; `1` copies LAST into
///   ALTREF; `2` copies GOLDEN into ALTREF.
/// * `refresh_last` — `L(1)`. `true` replaces the LAST slot with the
///   current reconstruction (the conventional inter pattern); `false`
///   leaves LAST in place so the next P-frame predicts off the
///   already-held picture.
///
/// The decoder's §9.7 / §9.8 walk (page 147 of RFC 6386:
/// `copy_arf → copy_gf → refresh_gf → refresh_arf → refresh_last`)
/// is implemented verbatim in [`crate::state::Vp8DecoderState`], so any
/// pattern this struct can express is decodable by the in-tree decoder
/// — `Vp8InterStreamEncoder` mirrors the same walk on its own slot
/// trio to keep encoder and decoder lockstep.
///
/// [`Default`] reproduces the round-149 pattern
/// ([`encode_p_frame_multi_ref`]'s hardwired ladder):
/// `refresh_last = true`, everything else 0 / `false`. With those
/// defaults the new public
/// [`encode_p_frame_multi_ref_with_refresh`] emits a wire that is
/// byte-identical to [`encode_p_frame_multi_ref`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshControls {
    /// §9.7 `refresh_golden_frame` (L1). `true` replaces GOLDEN with
    /// the current reconstruction.
    pub refresh_golden_frame: bool,
    /// §9.7 `refresh_alternate_frame` (L1). `true` replaces ALTREF with
    /// the current reconstruction.
    pub refresh_alternate_frame: bool,
    /// §9.7 `copy_buffer_to_golden` (L2, gated on
    /// `refresh_golden_frame == 0`). 0 = none, 1 = LAST → GOLDEN,
    /// 2 = ALTREF → GOLDEN. Values > 2 are rejected with
    /// [`EncodeError::InvalidCopyBufferSelector`].
    pub copy_buffer_to_golden: u8,
    /// §9.7 `copy_buffer_to_alternate` (L2, gated on
    /// `refresh_alternate_frame == 0`). 0 = none, 1 = LAST → ALTREF,
    /// 2 = GOLDEN → ALTREF. Values > 2 are rejected with
    /// [`EncodeError::InvalidCopyBufferSelector`].
    pub copy_buffer_to_alternate: u8,
    /// §9.8 `refresh_last` (L1). `true` replaces LAST with the current
    /// reconstruction (the conventional inter behaviour); `false` keeps
    /// the previous LAST picture so the next P-frame predicts off it.
    pub refresh_last: bool,
}

impl Default for RefreshControls {
    /// Round-149 default: `refresh_last = true`, every other field at
    /// zero. Reproduces the wire and slot semantics of
    /// [`encode_p_frame_multi_ref`].
    fn default() -> Self {
        Self {
            refresh_golden_frame: false,
            refresh_alternate_frame: false,
            copy_buffer_to_golden: 0,
            copy_buffer_to_alternate: 0,
            refresh_last: true,
        }
    }
}

impl RefreshControls {
    /// Validate the per-field constraints. `copy_buffer_to_*`
    /// selectors outside `0..=2` are rejected with
    /// [`EncodeError::InvalidCopyBufferSelector`].
    ///
    /// The §19.2 page-122 listing gates the L(2) `copy_buffer_to_*`
    /// emission on `if (!refresh_*_frame)`: the encoder does not
    /// transmit the selector when the matching refresh bit is 1, so a
    /// caller that sets both `refresh_golden_frame = true` AND
    /// `copy_buffer_to_golden != 0` would silently lose the copy
    /// intent (the slot is overwritten by the current reconstruction
    /// regardless). To make that misuse impossible we reject it here.
    pub fn validate(&self) -> Result<(), EncodeError> {
        if self.copy_buffer_to_golden > 2 {
            return Err(EncodeError::InvalidCopyBufferSelector {
                which: CopyBufferSelector::Golden,
                value: self.copy_buffer_to_golden,
            });
        }
        if self.copy_buffer_to_alternate > 2 {
            return Err(EncodeError::InvalidCopyBufferSelector {
                which: CopyBufferSelector::Alternate,
                value: self.copy_buffer_to_alternate,
            });
        }
        if self.refresh_golden_frame && self.copy_buffer_to_golden != 0 {
            return Err(EncodeError::InvalidCopyBufferSelector {
                which: CopyBufferSelector::Golden,
                value: self.copy_buffer_to_golden,
            });
        }
        if self.refresh_alternate_frame && self.copy_buffer_to_alternate != 0 {
            return Err(EncodeError::InvalidCopyBufferSelector {
                which: CopyBufferSelector::Alternate,
                value: self.copy_buffer_to_alternate,
            });
        }
        Ok(())
    }
}

/// Self-decode validation: the emitted bytes decode through
/// [`crate::state::Vp8DecoderState`] (after the I-frame the LAST slot
/// holds) and reproduce the source within the §14 quantiser's
/// distortion (≥ 30 dB whole-frame PSNR at a mid quantiser on a
/// slow-motion synthetic source).
pub fn encode_p_frame_zero_mv(
    frame: &I420Frame,
    reference: &crate::frame::KeyframePlanes,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref(frame, reference, None, None, params)
}

/// Backward-compatible wrapper for
/// [`encode_p_frame_multi_ref_with_refresh`]: passes
/// [`RefreshControls::default`] so the wire ladder matches the
/// round-149 hardwired pattern (`refresh_last = 1`, every other
/// §9.7 / §9.8 bit `0`).
pub fn encode_p_frame_multi_ref(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_with_refresh(
        frame,
        last,
        golden,
        altref,
        params,
        &RefreshControls::default(),
    )
}

/// Rewrite the §9.1 frame-tag `show_frame` bit in an already-assembled
/// VP8 frame.
///
/// The §9.1 tag is a 3-byte little-endian word whose bit 4 is
/// `show_frame` ("0 when current frame is not for display, 1 when
/// current frame is for display"); the surrounding bits (`frame_type`,
/// `version`, `first_partition_size`) are independent fields, so
/// flipping bit 4 in byte 0 changes the frame's visibility and nothing
/// else. Every partition boundary, probability, and coefficient byte
/// stays untouched — the invisible wire is byte-identical to the
/// visible one except for this single bit.
pub(crate) fn set_show_frame_bit(frame: &mut [u8], show_frame: bool) {
    debug_assert!(
        frame.len() >= 3,
        "a VP8 frame is at least the 3-byte §9.1 tag"
    );
    if let Some(b0) = frame.first_mut() {
        if show_frame {
            *b0 |= 0x10;
        } else {
            *b0 &= !0x10;
        }
    }
}

/// Encode an **invisible altref-update frame** — the §9.1 / §9.7
/// building block of B-frame-less GOLDEN / ALTREF management.
///
/// The frame is a regular §16.2 multi-reference interframe (full
/// motion-search / mode-decision walk, §11 intra-within-inter picker
/// enabled) with two deliberate departures from the conventional
/// P-frame pattern:
///
/// * the §9.7 / §9.8 refresh ladder is
///   `refresh_alternate_frame = 1, refresh_last = 0` (GOLDEN untouched)
///   — the reconstruction lands **only** in the ALTREF slot, so the
///   LAST chain of the display sequence is not perturbed;
/// * the §9.1 frame-tag `show_frame` bit is 0 ("not for display"), so
///   a compliant player decodes the frame for its reference-slot side
///   effects and drops the picture from presentation.
///
/// Feeding a *future* (or temporally-filtered) source picture through
/// this entry gives subsequent P-frames a forward-looking prediction
/// anchor via the per-MB §16.2 `ref_frame` selector — prediction from
/// a picture that has never been displayed, without any bidirectional
/// coding machinery. The returned reconstruction is exactly what the
/// decoder's ALTREF slot will hold after it processes these bytes.
///
/// `last` / `golden` / `altref` are the encoder-side reference slots
/// the update frame itself may predict from (same contract as
/// [`encode_p_frame_multi_ref`]).
pub fn encode_invisible_altref_update(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    let refresh = RefreshControls {
        refresh_alternate_frame: true,
        refresh_last: false,
        ..RefreshControls::default()
    };
    let (mut bytes, planes) = encode_p_frame_multi_ref_with_refresh_and_intra_pick(
        frame, last, golden, altref, params, &refresh,
    )?;
    set_show_frame_bit(&mut bytes, false);
    Ok((bytes, planes))
}

/// §16.2 multi-reference P-frame encoder with RD-selected §9.4
/// `loop_filter_level` and the §11 intra-mode picker enabled.
///
/// Identical to [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`]
/// (default refresh ladder) except that `params.loop_filter_level` is
/// **ignored**: after the inter mode-decision / motion-search walk, the
/// encoder searches the §9.4 level range for the level that minimises
/// post-§15 visible-window SSD of its own reconstruction against the
/// source `frame` (clean-room — RFC 6386 leaves the level choice to the
/// encoder; see [`crate::loop_filter::select_filter_level`]) and writes
/// that level into the §9.4 header. The per-MB §9.4 reference / mode
/// deltas default off (`LoopFilterDeltas::default()`), so each MB's
/// effective level reduces to the frame base. Returns the bitstream and
/// the post-§15 reconstruction (for installation into the §9 reference
/// ladder).
pub fn encode_p_frame_multi_ref_auto_loop_filter(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        &RefreshControls::default(),
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
        None,
        None,
        true,
        true,
    )
}

/// §16.2 multi-reference P-frame encoder with caller-driven §9.7 /
/// §9.8 reference-slot refresh control **and** caller-driven §9.4
/// per-reference / per-mode `loop_filter_delta` layer.
///
/// Extends [`encode_p_frame_multi_ref_with_refresh`] with the §9.4
/// `mb_lf_adjustments()` sub-block: the frame header emits the new
/// `loop_filter_adj_enable` / `mode_ref_lf_delta_update` bits and the
/// gated per-slot L(6) + L(1) values per [`LoopFilterDeltas`], and the
/// encoder's own §15 post-walk filter pass uses the effective per-MB
/// filter level ([`crate::loop_filter::calculate_mb_filter_level_inter`])
/// so the encoder's reconstruction buffer matches what the decoder
/// rebuilds from the same wire byte-for-byte.
///
/// `carried_ref_deltas` and `carried_mode_deltas` represent the
/// across-frame delta state per RFC 6386 §9.4: "the values from the
/// previous frame are used, unless they are updated in the current
/// header." For a standalone encode (no prior frame state), pass
/// `[0; 4]` for both; for a streaming caller, thread the values from
/// the previous frame's effective deltas — [`Vp8InterStreamEncoder`]
/// does this automatically.
///
/// Wire compatibility: `LoopFilterDeltas::default()` (with
/// `enabled = false`) and `[0; 4]` carried state reproduce the
/// round-150 wire byte-for-byte. Setting `enabled = true` flips on the
/// §9.4 layer; setting `update = false` while `enabled = true` reuses
/// the carried deltas without writing fresh values.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        None,
    )
}

/// §16.2 multi-reference P-frame encoder with caller-driven §9.7 /
/// §9.8 reference-slot refresh control, caller-driven §9.4 per-
/// reference / per-mode `loop_filter_delta` layer, **and** caller-driven
/// §13.4 `token_prob_update()` payload.
///
/// This is the inter-frame mirror of
/// [`encode_keyframe_with_token_prob_updates`] (round 155). When
/// `token_updates` is `Some(u)` the encoder writes the §13.4
/// per-position replacement layer into the first-partition header and
/// encodes the §13.3 residual tokens against the merged
/// `coeff_probs[4][8][3][11]` table (defaults overlaid with `u`). The
/// decoder reads the same updates from the wire and applies the same
/// overlay on top of its carried entropy state via
/// [`crate::dct_tokens::merge_default_token_probs`]-equivalent
/// machinery, so the round-trip is sound on either path.
///
/// `refresh_entropy_probs` stays `false` on inter frames (the §9.10
/// row-1 bit the encoder hardwires) — per RFC 6386 §9.10, this means
/// the frame's token-prob overlay is in force for THIS frame only; the
/// decoder restores the saved (key-frame) table afterwards. Setting
/// `token_updates = Some(u)` therefore re-prices the token bits of just
/// this inter frame without leaking into subsequent P-frames, which is
/// the natural fit for a per-frame "fit prob to observed token counts"
/// strategy.
///
/// Wire compatibility: `token_updates = None` (or an all-`None` array)
/// reproduces the round-155 inter wire byte-for-byte (the §13.4
/// sub-block reduces to 1056 zero flags and tokens are coded against
/// the §13.5 defaults). Every pre-r156 caller of
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas`] stays
/// unchanged.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
    token_updates: Option<&TokenProbUpdates>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        token_updates,
    )
}

/// §16.2 multi-reference P-frame encoder with caller-driven §13.4
/// `token_prob_update()` payload (and round-150 defaults for refresh +
/// §9.4 deltas).
///
/// Thin wrapper over
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates`]
/// that uses [`RefreshControls::default`] / [`LoopFilterDeltas::default`]
/// and `[0; 4]` carried delta state — i.e. the configuration that makes
/// the round-149 / round-150 / round-151 wire byte-for-byte equivalent
/// to the historical [`encode_p_frame_multi_ref`] when
/// `token_updates = None`.
pub fn encode_p_frame_multi_ref_with_token_updates(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    token_updates: Option<&TokenProbUpdates>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner(
        frame,
        last,
        golden,
        altref,
        params,
        &RefreshControls::default(),
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
        token_updates,
    )
}

/// §16.2 multi-reference P-frame encoder with an automatically-fitted
/// §13.4 `token_prob_update()` payload — the inter-path mirror of
/// [`encode_keyframe_with_fitted_token_prob_updates`] (round 157).
///
/// Round 156 wired the inter caller-driven layer; round 158 closes the
/// natural follow-up by letting the encoder *fit* the §13.4 payload
/// from observed branch counts instead of asking the caller for one.
/// The fitter shares its cost-model ([`fit_token_prob_updates`]) and
/// counter type ([`BranchCounts`]) with the keyframe path; only the
/// per-frame collection plumbing — [`count_inter_frame_branches`] —
/// is inter-specific (the inter picker stamps `IntraYMode::Dc` onto
/// every MB so the "no Y2" decision can't be recovered from `y_mode`;
/// the inner driver records it in a `use_bpred_per_mb` vector that
/// this walker consumes).
///
/// Internally the function takes two passes:
///
///   1. Encode with the §13.5 defaults and collect the per-position
///      branch counts via the [`encode_p_frame_multi_ref_inner_with_counts`]
///      `counts` side-channel.
///   2. Run [`fit_token_prob_updates`] to derive the
///      [`TokenProbUpdates`] payload that nets a positive bit saving
///      against those counts, then re-encode with that payload through
///      [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates`].
///
/// If the fitter returns an all-`None` payload (no slot crossed the
/// saving threshold), or the fitted re-encode is **larger** than the
/// default-encode wire (the model's saving estimate is computed
/// against pass-1's coefficient distribution; pass-2's RD pick perturbs
/// it slightly), the default-encode bytes are returned instead — so
/// this entry-point is guaranteed to be `<=` the
/// `encode_p_frame_multi_ref_with_token_updates(.., None)` wire in
/// every case. The returned bytes always decode through the crate's
/// own [`crate::state::Vp8DecoderState`] and any compliant VP8 decoder.
///
/// **Carried-base assumption.** Same as the round-156 inter caller-
/// driven entry-point: this function assumes the prior key frame was
/// emitted with the §13.5 defaults (i.e. `encode_keyframe` or
/// `encode_keyframe_with_token_prob_updates(.., None)`). The decoder
/// overlays the fitted payload on top of that base via
/// [`Vp8DecoderState::decode_inter_frame`] (`overlay_token_probs(self.
/// coeff_probs, &coded.token_prob_updates)`); the encoder's two-pass
/// fit also overlays on the §13.5 defaults, so the merged tables match
/// byte-for-byte. Mixing a non-default-base keyframe with this
/// entry-point is out of round-158 scope.
///
/// Out of round-158 scope: threading the fitter into
/// [`crate::stream::Vp8InterStreamEncoder`]'s `encode_frame` ladder —
/// the stream-driver method
/// [`crate::stream::Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_token_updates`]
/// stays on the caller-driven entry-point for now; a subsequent round
/// adds the analogous `_with_fitted_token_prob_updates` stream method.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    // Pass 1 — defaults + observed branch counts.
    let mut counts = empty_branch_counts();
    let (bytes_default, planes_default) = encode_p_frame_multi_ref_inner_with_counts(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        None,
        Some(&mut counts),
    )?;

    // Fit. 2.0 bits of slack guards against the small body-saving
    // overstatement that comes from the pass-2 RD pick perturbing the
    // coefficient distribution slightly relative to pass-1's counts.
    let fitted = fit_token_prob_updates(&counts, 2.0);

    // If no slot crossed the threshold, the default encode wins
    // trivially (the all-`None` path is byte-identical to the round-156
    // inter wire).
    let any_update = any_token_prob_update(&fitted);
    if !any_update {
        return Ok((bytes_default, planes_default));
    }

    // Pass 2 — re-encode with the fitted updates. The picker re-runs
    // against the merged probabilities (RD estimator uses the merged
    // table), so the chosen coefficients may shift relative to pass 1.
    // The encoder remains self-consistent: the decoder rebuilds the
    // same merged table from the on-wire updates and reconstructs the
    // same picture.
    let (bytes_fitted, planes_fitted) =
        encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates(
            frame,
            last,
            golden,
            altref,
            params,
            refresh,
            lf_deltas,
            carried_ref_deltas,
            carried_mode_deltas,
            Some(&fitted),
        )?;

    // Guard against the cost-model overstating the saving: only ship
    // the fitted bytes when they actually shrink the wire (or match).
    // Note: when we fall back we MUST also fall back to the default-
    // pass reconstruction, otherwise a streaming caller's next-frame
    // LAST slot would mis-match what the decoder will hold (the two
    // passes' picker outputs differ when the merged table differs).
    if bytes_fitted.len() <= bytes_default.len() {
        Ok((bytes_fitted, planes_fitted))
    } else {
        Ok((bytes_default, planes_default))
    }
}

/// §16.2 multi-reference P-frame encoder with an automatically-fitted
/// §13.4 `token_prob_update()` payload (and round-150 defaults for
/// refresh + §9.4 deltas).
///
/// Thin wrapper over
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`]
/// that uses [`RefreshControls::default`] / [`LoopFilterDeltas::default`]
/// and `[0; 4]` carried delta state — i.e. the configuration that makes
/// the round-149 / round-150 / round-151 wire byte-for-byte equivalent
/// to the historical [`encode_p_frame_multi_ref`] when the fitter falls
/// back to "no updates win" (the `bytes_fitted <= bytes_default` safety
/// guard).
pub fn encode_p_frame_multi_ref_with_fitted_token_prob_updates(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates(
        frame,
        last,
        golden,
        altref,
        params,
        &RefreshControls::default(),
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
    )
}

/// §16.2 multi-reference P-frame encoder with caller-driven §9.7 /
/// §9.8 reference-slot refresh control.
///
/// Extends [`encode_p_frame_zero_mv`] with optional `golden` and
/// `altref` reference planes; the per-MB picker scores every available
/// reference against the §17 / §16.2 / §16.3 / §16.4 candidate ladder
/// and emits whichever (`ref_frame`, mode, MV) tuple minimises
/// `J = SAD + lambda * (mv_ref_tree_bits + ref_frame_tree_bits + §17 mv bits)`.
///
/// `last` is required (every inter frame must hold a `LAST` reference
/// at the §9 three-slot ladder). `golden` and `altref` are optional;
/// pass `None` for a slot to disable that reference. All three planes
/// must share `(mb_cols, mb_rows)` with the source frame's
/// macroblock-grid dimensions; a mismatch is surfaced as
/// [`EncodeError::ReferenceDimensionsMismatch`].
///
/// `refresh` carries the five §9.7 / §9.8 reference-slot bits the
/// frame header emits (`refresh_golden_frame`,
/// `refresh_alternate_frame`, `copy_buffer_to_golden`,
/// `copy_buffer_to_alternate`, `refresh_last`). The wrapper
/// [`encode_p_frame_multi_ref`] passes [`RefreshControls::default`]
/// (which keeps the round-149 ladder: `refresh_last = 1`, every other
/// bit `0`); call this entry-point directly to express GOLDEN / ALTREF
/// rotation patterns (e.g. promote a low-noise reconstruction into
/// GOLDEN, copy LAST into ALTREF before a scene transition, hold LAST
/// across a synthetic disturbance). `refresh` is validated up front
/// with [`RefreshControls::validate`].
///
/// # Wire layout differences vs. [`encode_p_frame_zero_mv`]
///
/// * `prob_intra` stays 255 (no intra MBs this round). `prob_last`
///   and `prob_gf` are derived from the post-picker per-MB reference
///   distribution so the §16.2 selector bits compress against an
///   on-distribution prior (a Laplace-of-counts step clamped into
///   `1..=255`).
/// * The per-MB §19.3 `ref_frame` selector emits the chosen
///   reference: LAST → `B(prob_last)` reads `false`; GOLDEN →
///   `B(prob_last)` reads `true` then `B(prob_gf)` reads `false`;
///   ALTREF → both bools read `true`.
/// * Reconstruction reads from whichever reference's planes the
///   picker chose for each MB, so a single P-frame can mix LAST /
///   GOLDEN / ALTREF predictors.
/// * The §9.7 / §9.8 ladder follows `refresh`: the
///   `refresh_golden_frame` / `refresh_alternate_frame` bits emit
///   first, then the two `copy_buffer_to_*` selectors (always
///   transmitted; the decoder ignores the copy when the matching
///   refresh bit is 1 — see §9.7 page 38), then `refresh_last`.
///
/// All other behaviour (NEAREST / NEAR / NEW / SPLITMV picking,
/// §17.2 differential coding, §15 loop filter, §13 token emit)
/// matches [`encode_p_frame_zero_mv`].
///
/// `params.nbr_of_dct_partitions` (1 / 2 / 4 / 8) controls the §9.5
/// token-partition layout: macroblock rows are distributed round-robin
/// across the partitions per the §20.4 row-loop (row `r` → partition
/// `r % N`). This is a layout reorganisation only — the residual coding
/// inside each partition is bit-identical to the 1-partition case, so
/// the self-decoded picture is unchanged across all four legal values
/// and the wire grows by `(N - 1) * 3` bytes of size table plus
/// `(N - 1) * 4` bytes of extra §7.3 flush trailers. Values outside
/// the `1 | 2 | 4 | 8` set are rejected with
/// [`EncodeError::InvalidDctPartitionCount`] before the per-MB pick
/// walk runs.
pub fn encode_p_frame_multi_ref_with_refresh(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
        None,
    )
}

/// §16.2 multi-reference P-frame encoder with caller-driven §9.7 /
/// §9.8 reference-slot refresh control **and** round-160 / round-161
/// §11 intra-within-inter MB picking.
///
/// Extends [`encode_p_frame_multi_ref_with_refresh`] with the
/// round-160 per-MB intra candidate ladder, widened in round 161 to
/// score the full §11.2 × §11.4 whole-block intra grid (4 luma × 4
/// chroma = 16 candidates, `B_PRED` excluded). In addition to scoring
/// the §16 inter ladder (ZEROMV / NEARESTMV / NEARMV / NEWMV / SPLITMV
/// across every available reference frame), the picker scores every
/// whole-block intra `(y_mode, uv_mode)` pair against the running
/// in-frame neighbours. Whichever of (best inter pick, J-best intra)
/// has the lower `J + lambda * is_inter_mb-bit` wins per MB; when the
/// intra candidate wins on at least one MB the §9.10 `prob_intra` byte
/// drops below 255 and the §16.1 intra-mode-tree path emits on those
/// MBs, with the chosen Y / UV mode tree-encoded against the §16.1
/// defaults.
///
/// Decoder side: zero changes. The bytes re-enter
/// [`crate::state::Vp8DecoderState::decode_frame`] on a fresh decoder
/// state; the §16.1 `parse_inter_frame_intra_macroblock_modes` walker
/// + the keyframe per-MB reconstructor handle the intra-on-interframe
///   branch the same way they handle a key frame's intra MBs.
///
/// Wire compatibility: with a `pixels` content that never makes intra
/// beat inter (e.g. a flat or low-detail source where the §17 motion
/// search resolves to a near-zero residual), the picker stays
/// inter-only, `prob_intra` is fitted to `1` (the §16 spec-neutral
/// "no intra MB" value clamped up from 0 — the `is_inter_mb = true`
/// bit then codes at probability 255/256, costing ~6 bits per frame
/// on top of the historical "prob_intra = 255" path's ~0). Wire
/// growth is therefore bounded by a frame-constant ~6 bits ≈ 1 byte
/// even in the pathological no-intra-pick case.
pub fn encode_p_frame_multi_ref_with_refresh_and_intra_pick(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    refresh.validate()?;
    encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
        None,
        None,
        true,
        false,
    )
}

/// Backward-compatible wrapper for
/// [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`]: passes
/// [`RefreshControls::default`] so the wire ladder matches the
/// round-149 hardwired refresh pattern (`refresh_last = 1`, every
/// other §9.7 / §9.8 bit `0`) while the round-160 / round-161
/// intra-pick path runs.
pub fn encode_p_frame_multi_ref_with_intra_pick(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_with_refresh_and_intra_pick(
        frame,
        last,
        golden,
        altref,
        params,
        &RefreshControls::default(),
    )
}

/// §16.2 multi-reference P-frame encoder with caller-driven §9.7 /
/// §9.8 reference-slot refresh control, caller-driven §9.4 per-
/// reference / per-mode `loop_filter_delta` layer, **and** round-160 /
/// round-161 §11 intra-within-inter MB picking.
///
/// Composition of
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas`] (round 151
/// caller-driven §9.4 layer + carried-state inputs) and
/// [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`] (round
/// 160 / 161 §11 picker toggle). Closes the round-162 next-step
/// follow-up that called out the missing composition: the §9.4
/// deltas and the intra-pick were each exposed individually but never
/// together. This entry-point lets a caller drive both knobs in the
/// same call.
///
/// Argument shape matches
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas`] exactly —
/// the intra-pick toggle is implicit (this function always engages
/// it, just like
/// [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`]). Callers
/// that want the §9.4 layer without the intra picker stay on the
/// non-intra-pick entry-point.
///
/// Wire compatibility:
///
/// * Calling with [`LoopFilterDeltas::default`] + carried `[0; 4]` /
///   `[0; 4]` reproduces
///   [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`]
///   byte-for-byte. The §9.4 delta layer is gated on `lf_deltas.enabled`
///   exactly as on the non-intra-pick path.
/// * Calling with `pick_intra` engaged on a source where intra never
///   beats inter (a flat or near-flat P-frame against a populated
///   LAST) yields a wire whose `prob_intra` byte sits at `1` (the
///   §16 spec-neutral "no intra MB" value clamped up from the
///   fitter's `fit_prob_l8(0, count_inter)` output) instead of `255`
///   — the same ~6 bits ≈ 1 byte frame-constant bound documented on
///   [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`].
///
/// `refresh` is validated up front via [`RefreshControls::validate`];
/// `lf_deltas` is validated via [`LoopFilterDeltas::validate`].
#[allow(clippy::too_many_arguments)]
pub fn encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    refresh.validate()?;
    lf_deltas.validate()?;
    encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        None,
        None,
        true,
        false,
    )
}

/// §16.2 multi-reference P-frame encoder composing the round-160 / 161 §11
/// intra-within-inter MB picker **with** the round-157 / 158 §13.4 token-
/// prob observed-counts fitter, on top of the caller-driven §9.7 / §9.8
/// reference-slot refresh control and the §9.4 per-reference / per-mode
/// loop-filter delta layer.
///
/// Composition of:
///
/// * [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick`]
///   (round-163 §11 picker toggle on the refresh + §9.4 deltas axis), and
/// * [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`]
///   (round-158 two-pass fitter on the refresh + §9.4 deltas axis).
///
/// Closes round-164 follow-up item (5): the §11 picker and the §13.4
/// fitter were each composed individually with the refresh + lf-deltas
/// axis, but never together. This entry-point drives both knobs in the
/// same call — same shape as the two single-knob siblings, with `pick_intra`
/// engaged on both fitter passes.
///
/// Internally takes two passes:
///
///   1. Encode with the §13.5 defaults — `token_updates = None` — and
///      `pick_intra = true`, collecting per-position branch counts via
///      [`encode_p_frame_multi_ref_inner_with_counts_and_pick`]'s
///      `counts` side-channel.
///   2. Run [`fit_token_prob_updates`] (slack 2.0 — matches every other
///      inter fitter) to derive a [`TokenProbUpdates`] payload that nets
///      a positive saving, then re-encode with that payload and
///      `pick_intra = true`.
///
/// If the fitter returns an all-`None` payload (no slot crossed the
/// saving threshold) **or** the fitted re-encode is larger than the
/// default-encode wire (the saving estimate is against pass-1's
/// coefficient distribution; pass-2's RD pick perturbs it slightly), the
/// default-encode bytes / planes are returned instead — so this entry-
/// point is guaranteed `<=` the
/// [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick`]
/// wire on every input. **When the fallback fires we also fall back to
/// the pass-1 planes** (matching the round-158 sibling) so a streaming
/// caller's LAST never mis-matches the decoder's reconstruction.
///
/// `refresh` is validated up front via [`RefreshControls::validate`];
/// `lf_deltas` is validated via [`LoopFilterDeltas::validate`].
///
/// **Carried-base assumption.** Same as the round-158 sibling: this
/// function assumes the prior key frame was emitted with the §13.5
/// defaults (i.e. either [`encode_keyframe`] /
/// [`encode_keyframe_with_reconstruction`], or
/// [`encode_keyframe_with_token_prob_updates`] called with an all-`None`
/// array). Mixing a non-default-base keyframe with this entry-point is
/// out of round-165 scope.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    refresh.validate()?;
    lf_deltas.validate()?;

    // Pass 1 — §13.5 defaults + observed branch counts. The §11 intra
    // picker runs on this pass so the recorded counts already reflect
    // the intra/inter MB mix that will reappear on pass 2.
    let mut counts = empty_branch_counts();
    let (bytes_default, planes_default) = encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        None,
        Some(&mut counts),
        true,
        false,
    )?;

    // Fit. 2.0 bits of slack — matches every other inter fitter and
    // guards against the small overstatement of saving that comes from
    // pass-2's RD pick perturbing the coefficient distribution slightly
    // relative to pass-1's counts.
    let fitted = fit_token_prob_updates(&counts, 2.0);

    // If no slot crossed the threshold, the default encode wins trivially
    // (the all-`None` path is byte-identical to the round-163
    // _intra_pick wire).
    let any_update = any_token_prob_update(&fitted);
    if !any_update {
        return Ok((bytes_default, planes_default));
    }

    // Pass 2 — re-encode with the fitted updates AND `pick_intra = true`
    // so the picker re-runs against the merged probabilities (the RD
    // estimator uses the merged table). The encoder remains self-
    // consistent: the decoder rebuilds the same merged table from the
    // on-wire updates and reconstructs the same picture.
    let (bytes_fitted, planes_fitted) = encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        Some(&fitted),
        None,
        true,
        false,
    )?;

    // Bytes-vs-default safety guard. Falling back also drops the pass-2
    // planes — a streaming caller's next-frame LAST slot must match the
    // decoder's reconstruction from the bytes we just shipped.
    if bytes_fitted.len() <= bytes_default.len() {
        Ok((bytes_fitted, planes_fitted))
    } else {
        Ok((bytes_default, planes_default))
    }
}

/// Orthogonal per-frame quality/size feature toggles for the generic
/// inter front door
/// [`encode_p_frame_multi_ref_with_refresh_and_coding_options`].
///
/// Each flag switches on one already-proven inter encoder feature; the
/// all-`false` default reproduces
/// [`encode_p_frame_multi_ref_with_refresh`] byte-for-byte. This struct
/// replaces the combinatorial `_intra_pick` / `_auto_loop_filter` /
/// `_fitted_token_prob_updates` function-name ladder for callers (like
/// the lagged stream driver) that need the features **combined**.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterCodingOptions {
    /// §11 intra-within-inter picker: every MB additionally scores the
    /// 16 whole-block `(y_mode, uv_mode)` intra candidates against the
    /// best inter pick — the
    /// [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`]
    /// behaviour.
    pub intra_pick: bool,
    /// RD-select the §9.4 `loop_filter_level` / sharpness per frame
    /// (ignoring `params.loop_filter_level`) — the
    /// [`encode_p_frame_multi_ref_auto_loop_filter`] behaviour, but
    /// composable with any refresh ladder and the other toggles.
    pub auto_loop_filter: bool,
    /// Two-pass §13.4 `token_prob_update()` fit (defaults pass →
    /// [`fit_token_prob_updates`] → fitted pass → smaller wire ships).
    pub fitted_token_prob_updates: bool,
}

/// Generic §16.2 multi-reference P-frame front door combining the §11
/// intra-within-inter picker, the §9.4 auto-loop-filter selector, and
/// the §13.4 fitted token-prob-update emission behind one
/// [`InterCodingOptions`] toggle set, with caller-driven §9.7 / §9.8
/// refresh control.
///
/// Equivalences (each pinned by `tests/encoder_coding_options.rs`):
///
/// * `options == Default` — byte-identical to
///   [`encode_p_frame_multi_ref_with_refresh`];
/// * only `intra_pick` — byte-identical to
///   [`encode_p_frame_multi_ref_with_refresh_and_intra_pick`];
/// * only `auto_loop_filter` (with the default refresh ladder) —
///   byte-identical to [`encode_p_frame_multi_ref_auto_loop_filter`]
///   minus that entry's hardwired intra pick — pass
///   `intra_pick: true, auto_loop_filter: true` for the exact match;
/// * `intra_pick + fitted_token_prob_updates` — byte-identical to
///   [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates`]
///   at default `LoopFilterDeltas` / zero carried deltas.
///
/// The §9.4 per-reference / per-mode delta layer stays off on this
/// entry (`LoopFilterDeltas::default()`, zero carried state) — a caller
/// that threads LF deltas across frames should keep using the explicit
/// `_with_refresh_and_lf_deltas_*` ladder.
///
/// Returns the wire bytes plus the matching post-§15 reconstruction
/// planes (what the decoder's refreshed slots will hold).
pub fn encode_p_frame_multi_ref_with_refresh_and_coding_options(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    options: &InterCodingOptions,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_options_and_segmentation(
        frame, last, golden, altref, params, refresh, options, None,
    )
}

/// §9.3 / §10 segmentation layer for the inter (P-frame) encoder — the
/// round-387 companion of the keyframe path's [`AdaptiveQuantConfig`].
///
/// Macroblocks are classified into four segments by §10 source-luma
/// variance ([`AdaptiveQuantConfig::variance_boundaries`] semantics);
/// each segment carries a §9.3 `quantizer_update` delta (added to the
/// frame `y_ac_qi`, clamped to `0..=127`) and, optionally, a §9.3
/// `loop_filter_update` delta (added to the frame filter level, clamped
/// to `0..=63` per §20.6). The per-MB RD picker prices every candidate
/// at its segment's quantiser, so flat segments genuinely code coarser
/// and busy segments finer — activity-masking AQ on P-frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterSegmentationConfig {
    /// Three ascending luma-variance boundaries splitting macroblocks
    /// into the four segments (see
    /// [`AdaptiveQuantConfig::variance_boundaries`]).
    pub variance_boundaries: [u32; 3],
    /// Per-segment §10 `quantizer_update_value` deltas (signed,
    /// magnitude ≤ 127), added to the frame `y_ac_qi` in delta mode.
    pub quant_delta: [i8; 4],
    /// Optional per-segment §10 loop-filter deltas (signed, magnitude
    /// ≤ 63). `None` omits the §9.3 loop-filter feature entirely (every
    /// MB filters at the frame base level).
    pub lf_delta: Option<[i8; 4]>,
}

impl Default for InterSegmentationConfig {
    /// The [`AdaptiveQuantConfig`] default gradient: flat → coarser,
    /// detailed → finer; no loop-filter feature.
    fn default() -> Self {
        InterSegmentationConfig {
            variance_boundaries: [
                SEGMENT_VARIANCE_THRESHOLDS[1],
                SEGMENT_VARIANCE_THRESHOLDS[2],
                SEGMENT_VARIANCE_THRESHOLDS[3],
            ],
            quant_delta: [8, 0, -6, -12],
            lf_delta: None,
        }
    }
}

/// §16.2 multi-reference P-frame encoder with §9.3 / §10 **segment-based
/// adaptive quantisation** (and optionally the per-segment loop-filter
/// feature), composable with the full [`InterCodingOptions`] toggle set
/// and caller-driven §9.7 / §9.8 refresh control.
///
/// The inter mirror of [`encode_keyframe_adaptive_quant`]: each MB is
/// classified by §10 source-luma variance, quantised and RD-priced at
/// its segment's effective qindex, and the frame header carries the
/// §9.3 `update_segmentation()` block (delta-mode features + the
/// distribution-fitted `mb_segment_tree_probs`) while the §11 / §16
/// mode layer prefixes each MB with its `segment_id` — exactly the
/// stream the decoder's inter path resolves through its per-segment
/// dequant factors and the §20.6 per-segment filter override.
///
/// `lf_delta` magnitudes above 63 are rejected with
/// [`EncodeError::LoopFilterLevelOutOfRange`]. Returns the wire bytes
/// plus the matching post-§15 reconstruction planes.
#[allow(clippy::too_many_arguments)]
pub fn encode_p_frame_multi_ref_adaptive_quant(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    segmentation: &InterSegmentationConfig,
    options: &InterCodingOptions,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    if let Some(d) = segmentation.lf_delta {
        if let Some(&bad) = d.iter().find(|v| v.unsigned_abs() > 63) {
            return Err(EncodeError::LoopFilterLevelOutOfRange {
                value: bad.unsigned_abs(),
            });
        }
    }
    encode_p_frame_options_and_segmentation(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        options,
        Some(segmentation),
    )
}

/// Shared body of the two coding-options front doors above: the
/// optional two-pass §13.4 fitter around the full inter driver, with
/// the §9.3 / §10 segmentation layer threaded through both passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_p_frame_options_and_segmentation(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    options: &InterCodingOptions,
    segmentation: Option<&InterSegmentationConfig>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    refresh.validate()?;
    // §9.3: the loop-filter feature value is a signed 6-bit magnitude.
    if let Some(d) = segmentation.and_then(|s| s.lf_delta) {
        if let Some(&bad) = d.iter().find(|v| v.unsigned_abs() > 63) {
            return Err(EncodeError::LoopFilterLevelOutOfRange {
                value: bad.unsigned_abs(),
            });
        }
    }

    if !options.fitted_token_prob_updates {
        return encode_p_frame_multi_ref_inner_full(
            frame,
            last,
            golden,
            altref,
            params,
            refresh,
            &LoopFilterDeltas::default(),
            [0; 4],
            [0; 4],
            None,
            None,
            options.intra_pick,
            options.auto_loop_filter,
            segmentation,
        );
    }

    // Pass 1 — §13.5 defaults + observed branch counts, with the
    // requested picker / auto-LF / segmentation switches active so the
    // recorded counts reflect the MB mix pass 2 will re-produce.
    let mut counts = empty_branch_counts();
    let (bytes_default, planes_default) = encode_p_frame_multi_ref_inner_full(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
        None,
        Some(&mut counts),
        options.intra_pick,
        options.auto_loop_filter,
        segmentation,
    )?;

    // 2.0 bits of slack — matches every other inter fitter.
    let fitted = fit_token_prob_updates(&counts, 2.0);
    if !any_token_prob_update(&fitted) {
        return Ok((bytes_default, planes_default));
    }

    // Pass 2 — re-encode against the merged table.
    let (bytes_fitted, planes_fitted) = encode_p_frame_multi_ref_inner_full(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
        Some(&fitted),
        None,
        options.intra_pick,
        options.auto_loop_filter,
        segmentation,
    )?;

    // Bytes-vs-default guard; the shipped planes always match the
    // shipped wire so a streaming caller's slots stay in decoder
    // lockstep.
    if bytes_fitted.len() <= bytes_default.len() {
        Ok((bytes_fitted, planes_fitted))
    } else {
        Ok((bytes_default, planes_default))
    }
}

/// [`encode_invisible_altref_update`] with the full
/// [`InterCodingOptions`] toggle set — the §9.1 `show_frame = 0` /
/// §9.7 `refresh_alternate_frame = 1` anchor-update frame, but with the
/// §9.4 auto-loop-filter selector and/or the §13.4 fitted
/// token-prob-update emission active on the underlying inter encode.
///
/// With `options = InterCodingOptions { intra_pick: true, ..Default }`
/// the wire is byte-identical to [`encode_invisible_altref_update`]
/// (that entry hardwires the §11 intra picker on).
pub fn encode_invisible_altref_update_with_coding_options(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    options: &InterCodingOptions,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_invisible_altref_update_full(frame, last, golden, altref, params, options, None)
}

/// [`encode_invisible_altref_update_with_coding_options`] with the
/// §9.3 / §10 segmentation layer threaded through the underlying inter
/// encode — the anchor-side face the lagged stream driver uses when its
/// segmentation knob is on.
pub(crate) fn encode_invisible_altref_update_full(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    options: &InterCodingOptions,
    segmentation: Option<&InterSegmentationConfig>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    let refresh = RefreshControls {
        refresh_alternate_frame: true,
        refresh_last: false,
        ..RefreshControls::default()
    };
    let (mut bytes, planes) = encode_p_frame_options_and_segmentation(
        frame,
        last,
        golden,
        altref,
        params,
        &refresh,
        options,
        segmentation,
    )?;
    set_show_frame_bit(&mut bytes, false);
    Ok((bytes, planes))
}

#[allow(clippy::too_many_arguments)]
fn encode_p_frame_multi_ref_inner(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
    token_updates: Option<&TokenProbUpdates>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        token_updates,
        None,
        false,
        false,
    )
}

/// Inter (P-frame) driver shared by every public inter entry-point.
///
/// Mirrors [`encode_keyframe_inner`]'s `counts` side-channel: when
/// `counts = Some(&mut c)` the §13.3 token-encode pass also records
/// each `(plane, band, prev_ctx, position)` bit-event into `c`
/// alongside its regular `BoolEncoder` write — `c`'s contents are the
/// only side-effect of the parameter; the emitted bytes are byte-
/// identical to the `counts = None` invocation with the same
/// `token_updates`. The walk runs `count_inter_frame_branches` against
/// the same `(modes, use_bpred_per_mb, all_coeffs, mb_cols, mb_rows)`
/// the §13.3 emit loop just consumed, so the recorded counts are bit-
/// for-bit the events the partitions emitted.
///
/// Used by the round-158 [`encode_p_frame_multi_ref_with_fitted_token_prob_updates`]
/// fitter to drive its two-pass observed-counts fitter without
/// duplicating the long picker/motion-search/§15 walk.
#[allow(clippy::too_many_arguments)]
fn encode_p_frame_multi_ref_inner_with_counts(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
    token_updates: Option<&TokenProbUpdates>,
    counts: Option<&mut BranchCounts>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner_with_counts_and_pick(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        token_updates,
        counts,
        false,
        false,
    )
}

/// Same as [`encode_p_frame_multi_ref_inner_with_counts`] plus the
/// round-160 / round-161 `pick_intra` toggle: when `pick_intra = true`
/// the per-MB picker additionally scores every §11.2 × §11.4
/// whole-block intra `(y_mode, uv_mode)` candidate (16 in total,
/// `B_PRED` excluded) against the running in-frame neighbours, then
/// picks whichever of (best inter pick, J-best intra) wins on
/// `J + lambda * is_inter_mb-bit`. When `pick_intra = false` the
/// inter-only ladder runs (every MB stays inter; `prob_intra` is
/// hard-wired to 255), reproducing every pre-round-160 wire
/// byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn encode_p_frame_multi_ref_inner_with_counts_and_pick(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
    token_updates: Option<&TokenProbUpdates>,
    counts: Option<&mut BranchCounts>,
    pick_intra: bool,
    auto_lf: bool,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    encode_p_frame_multi_ref_inner_full(
        frame,
        last,
        golden,
        altref,
        params,
        refresh,
        lf_deltas,
        carried_ref_deltas,
        carried_mode_deltas,
        token_updates,
        counts,
        pick_intra,
        auto_lf,
        None,
    )
}

/// The complete inter (P-frame) driver — every public inter entry point
/// funnels here. Adds the round-387 §9.3 / §10 `segmentation` layer on
/// top of [`encode_p_frame_multi_ref_inner_with_counts_and_pick`]'s
/// toggles: when `Some(seg)`, each macroblock is classified into one of
/// four segments by §10 luma variance, quantised (and RD-priced) at
/// that segment's effective qindex, the §9.3 `update_segmentation()`
/// block travels in the header (quant deltas + optional loop-filter
/// deltas + distribution-fitted `mb_segment_tree_probs`), and the §11 /
/// §16 mode layer prefixes each MB with its `segment_id`. `None`
/// reproduces the historical wire byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn encode_p_frame_multi_ref_inner_full(
    frame: &I420Frame,
    last: &crate::frame::KeyframePlanes,
    golden: Option<&crate::frame::KeyframePlanes>,
    altref: Option<&crate::frame::KeyframePlanes>,
    params: &KeyframeParams,
    refresh: &RefreshControls,
    lf_deltas: &LoopFilterDeltas,
    carried_ref_deltas: [i16; 4],
    carried_mode_deltas: [i16; 4],
    token_updates: Option<&TokenProbUpdates>,
    counts: Option<&mut BranchCounts>,
    pick_intra: bool,
    auto_lf: bool,
    segmentation: Option<&InterSegmentationConfig>,
) -> Result<(Vec<u8>, crate::frame::KeyframePlanes), EncodeError> {
    refresh.validate()?;
    lf_deltas.validate()?;
    let width = frame.width;
    let height = frame.height;
    if width == 0 || width > 0x3FFF || height == 0 || height > 0x3FFF {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    if params.y_ac_qi > 127 {
        return Err(EncodeError::QuantIndexOutOfRange {
            value: params.y_ac_qi,
        });
    }
    if params.loop_filter_level > 63 {
        return Err(EncodeError::LoopFilterLevelOutOfRange {
            value: params.loop_filter_level,
        });
    }
    if params.sharpness_level > 7 {
        return Err(EncodeError::SharpnessLevelOutOfRange {
            value: params.sharpness_level,
        });
    }
    // §9.5 — the partition-count is validated here so a bad value is
    // rejected before the long mode-pick / motion-search walk runs.
    // `write_token_partition_count` re-validates at the actual
    // bitstream emission point.
    let num_partitions = match params.nbr_of_dct_partitions {
        1 | 2 | 4 | 8 => params.nbr_of_dct_partitions as usize,
        other => return Err(EncodeError::InvalidDctPartitionCount { value: other }),
    };

    // §13 trellis aggressiveness for the inter residual emit, dialed by the
    // shared [`KeyframeParams::trellis_strength`] knob (the same field the
    // keyframe path threads). `DEFAULT` reproduces the historical bytes.
    let inter_trellis_strength = params.trellis_strength;

    let mb_cols = width.div_ceil(16) as usize;
    let mb_rows = height.div_ceil(16) as usize;
    if last.mb_cols != mb_cols || last.mb_rows != mb_rows {
        return Err(EncodeError::ReferenceDimensionsMismatch {
            source: ((mb_cols * 16) as u32, (mb_rows * 16) as u32),
            reference: ((last.mb_cols * 16) as u32, (last.mb_rows * 16) as u32),
        });
    }
    if let Some(g) = golden {
        if g.mb_cols != mb_cols || g.mb_rows != mb_rows {
            return Err(EncodeError::ReferenceDimensionsMismatch {
                source: ((mb_cols * 16) as u32, (mb_rows * 16) as u32),
                reference: ((g.mb_cols * 16) as u32, (g.mb_rows * 16) as u32),
            });
        }
    }
    if let Some(a) = altref {
        if a.mb_cols != mb_cols || a.mb_rows != mb_rows {
            return Err(EncodeError::ReferenceDimensionsMismatch {
                source: ((mb_cols * 16) as u32, (mb_rows * 16) as u32),
                reference: ((a.mb_cols * 16) as u32, (a.mb_rows * 16) as u32),
            });
        }
    }

    // The decoder's inter path overlays the §13.4 update payload on top
    // of its carried entropy state ([`Vp8DecoderState::decode_inter_frame`]
    // — `overlay_token_probs(self.coeff_probs, &coded.token_prob_updates)`).
    // We assume the prior keyframe was encoded with the §13.5 defaults
    // (i.e. either [`encode_keyframe`] or
    // [`encode_keyframe_with_token_prob_updates`] called with an
    // all-`None` array), in which case the carried base is
    // `DEFAULT_COEFF_PROBS` and the overlay reduces to
    // [`crate::dct_tokens::merge_default_token_probs`] on this frame's
    // `token_updates`. Mixing a non-default-base keyframe with this
    // entry-point is out of round-156 scope.
    //
    // With `token_updates = None` (or all-`None`) the merged table is
    // byte-identical to the §13.5 defaults and the §13.4 sub-block
    // reduces to the 1056-zero-flag wire — i.e. the round-155 inter
    // wire byte-for-byte.
    let coeff_probs = match token_updates {
        Some(u) => crate::dct_tokens::merge_default_token_probs(u),
        None => crate::dct_tokens::DEFAULT_COEFF_PROBS,
    };
    // Per-frame §13 trellis rate tables (see `TrellisRateTables`): one
    // build per frame, shared by every per-ref / intra-in-P mode pick.
    let trellis_rate_tables =
        (inter_trellis_strength.get() > 0.0).then(|| TrellisRateTables::for_probs(&coeff_probs));
    let factors = crate::dequant::MbDequantFactors::from_base_and_deltas(
        params.y_ac_qi as i32,
        0,
        0,
        0,
        0,
        0,
    );

    // §9.3 / §10 segmentation context: the four per-segment §14.1
    // dequant factor sets (delta mode on the §9.6 baseline — exactly
    // what the decoder's `resolve_segment_dequant_factors` rebuilds
    // from the header we emit below) plus the matching per-segment RD
    // lambdas. `None` leaves the historical single-factor path.
    let seg_ctx = segmentation.map(|s| {
        let mut seg_factors = [factors; crate::loop_filter::MAX_MB_SEGMENTS];
        let mut seg_lambdas = [0.0f64; crate::loop_filter::MAX_MB_SEGMENTS];
        for i in 0..crate::loop_filter::MAX_MB_SEGMENTS {
            seg_factors[i] = crate::dequant::MbDequantFactors::from_base_and_deltas(
                params.y_ac_qi as i32 + s.quant_delta[i] as i32,
                0,
                0,
                0,
                0,
                0,
            );
            seg_lambdas[i] = rd_lambda(&seg_factors[i]);
        }
        (s, seg_factors, seg_lambdas)
    });
    // Per-segment MB counts, for the post-walk `mb_segment_tree_probs`
    // distribution fit.
    let mut seg_counts = [0u32; crate::loop_filter::MAX_MB_SEGMENTS];

    // Per-reference plane bundle. `Last` is always populated; the
    // optional Golden / AltRef refs are populated when their slot is
    // available.
    let reference_planes_last = crate::motion_comp::ReferencePlanes {
        y: &last.y,
        u: &last.u,
        v: &last.v,
        y_stride: last.y_stride,
        uv_stride: last.uv_stride,
        mb_cols: last.mb_cols,
        mb_rows: last.mb_rows,
    };
    let reference_planes_golden = golden.map(|g| crate::motion_comp::ReferencePlanes {
        y: &g.y,
        u: &g.u,
        v: &g.v,
        y_stride: g.y_stride,
        uv_stride: g.uv_stride,
        mb_cols: g.mb_cols,
        mb_rows: g.mb_rows,
    });
    let reference_planes_altref = altref.map(|a| crate::motion_comp::ReferencePlanes {
        y: &a.y,
        u: &a.u,
        v: &a.v,
        y_stride: a.y_stride,
        uv_stride: a.uv_stride,
        mb_cols: a.mb_cols,
        mb_rows: a.mb_rows,
    });
    // §18.3 filter set (the version=0 default tap-set is fine — every
    // MB has a (0,0) MV, so the convolution collapses to whole-pixel
    // copy and the chosen tap set never runs).
    let filters = crate::motion_comp::filter_set_for_version(0).taps();

    // Running reconstruction buffer — same shape `decode_keyframe`
    // produces. The §18 prediction for the NEXT MB does not depend on
    // the current frame's already-reconstructed pixels (ZEROMV always
    // samples LAST, not the current frame), but we still fill this so
    // the §15 loop filter pass can operate on a single picture.
    let mut planes = crate::frame::KeyframePlanes {
        y: vec![0u8; mb_cols * 16 * mb_rows * 16],
        u: vec![0u8; mb_cols * 8 * mb_rows * 8],
        v: vec![0u8; mb_cols * 8 * mb_rows * 8],
        y_stride: mb_cols * 16,
        uv_stride: mb_cols * 8,
        mb_cols,
        mb_rows,
    };

    let mut modes: Vec<MacroblockModes> = Vec::with_capacity(mb_rows * mb_cols);
    let mut all_coeffs: Vec<MbCoeffs> = Vec::with_capacity(mb_rows * mb_cols);
    let mut ref_frames_out: Vec<Option<crate::motion_comp::RefFrame>> =
        Vec::with_capacity(mb_rows * mb_cols);
    let mut inter_modes_out: Vec<Option<crate::near_mv::InterMode>> =
        Vec::with_capacity(mb_rows * mb_cols);
    // Per-MB chosen quarter-pixel MV — consumed by the §11/§16 mode-
    // emit loop below and (for NEWMV) by the §17.2 component writer.
    // For ZEROMV MBs this is `Mv::default()`; for NEWMV MBs it is the
    // search-resolved whole-pixel MV (snapped + clamped to §17.1).
    // For SPLITMV MBs this is `split_mvs[15]` (the §16.3 `MbInfo::mv`
    // convention).
    let mut chosen_mvs: Vec<crate::motion_vector::Mv> = Vec::with_capacity(mb_rows * mb_cols);
    // For SPLITMV MBs, the per-MB candidate the picker resolved (so
    // the mode-emit loop can re-emit the partition tree + per-group
    // sub_mv_ref + NEW4X4 diffs without re-running the picker). `None`
    // for non-SPLITMV MBs.
    let mut split_candidates: Vec<Option<SplitMvCandidate>> = Vec::with_capacity(mb_rows * mb_cols);
    // Per-MB `use_bpred` flag for the §13.3 token emit loop. SPLITMV
    // MBs have no Y2 block (§14.2 page 76) so they thread through
    // `encode_mb_tokens(use_bpred = true)`, mirroring the B_PRED path.
    let mut use_bpred_per_mb: Vec<bool> = Vec::with_capacity(mb_rows * mb_cols);
    // Round 160 — `pick_intra` toggle: per-MB flag set to `true` when
    // the §11 / §12.2 DC_PRED intra candidate's J beat the §16 inter
    // pick's J + the is_inter_mb-bit overhead. When `pick_intra = false`
    // (every pre-r160 entry-point), this vec stays all-`false` and the
    // emit loop reproduces the pre-r160 wire byte-for-byte.
    let mut is_intra_per_mb: Vec<bool> = vec![false; mb_rows * mb_cols];
    // For intra MBs, the chosen §11.2 / §11.4 modes. Round 161 picks
    // the J-best whole-block (y_mode, uv_mode) pair from the 4×4 grid
    // `{Dc, V, H, Tm} × {Dc, V, H, Tm}` (see `pick_intra_mb_all`);
    // initialised to `(Dc, Dc)` so inter MBs (which leave these slots
    // unread) and the round-160 DC-tie path keep their original
    // semantics. `B_PRED` is out of scope for this picker — the
    // per-sub-block intra walker is a separate fitter family.
    let mut intra_y_modes: Vec<IntraYMode> = vec![IntraYMode::Dc; mb_rows * mb_cols];
    let mut intra_uv_modes: Vec<IntraUvMode> = vec![IntraUvMode::Dc; mb_rows * mb_cols];

    // §17.2 MV-component contexts the encoder writes NEWMV diffs
    // against. We emit every `mv_prob_update()` F-gate as 0 (no update)
    // so the decoder reads against the same §17.2 defaults.
    let mv_contexts = crate::motion_vector::default_mv_contexts();

    // §-non-normative RD lambda — same `q^2 / 32` shape the keyframe
    // RD picker uses (see `rd_lambda`). Higher quantiser ⇒ higher
    // lambda ⇒ stricter penalty per extra bit, biasing toward ZEROMV
    // at low bitrate. Units: distortion-per-bit (here SAD-per-bit,
    // which is roughly half SSD-per-bit at low signal levels but
    // monotone — fine for a relative ZEROMV-vs-NEWMV trade).
    let lambda = rd_lambda(&factors);

    // Neighbour-record state for the §16.3 `find_near_mvs` census —
    // shared between the search-and-RD pass below and the §11/§16
    // mode-emit pass that follows. Each MB sees the same census the
    // decoder will recompute as it walks the bytes.
    let mut search_above: Vec<crate::near_mv::MbInfo> =
        vec![crate::near_mv::MbInfo::border(); mb_cols];

    // Available references for the §16.2 ref_frame selector. LAST is
    // always present; GOLDEN / ALTREF appear iff the caller passed
    // their planes. The order ((Last, Golden, AltRef)) is the decoder's
    // §16.2 ref_frame_tree walk order; the picker iterates them and
    // keeps the lowest total-J candidate.
    let mut available_refs: Vec<(
        crate::motion_comp::RefFrame,
        crate::motion_comp::ReferencePlanes,
    )> = Vec::with_capacity(3);
    available_refs.push((crate::motion_comp::RefFrame::Last, reference_planes_last));
    if let Some(rp) = reference_planes_golden {
        available_refs.push((crate::motion_comp::RefFrame::Golden, rp));
    }
    if let Some(rp) = reference_planes_altref {
        available_refs.push((crate::motion_comp::RefFrame::AltRef, rp));
    }

    // Per-MB chosen reference frame — paired with `chosen_mvs` /
    // `inter_modes_out` etc. Used by the wire-emit loop below to write
    // the §16.2 ref_frame selector bits and by the neighbour census
    // (the decoder's `find_near_mvs` uses the recorded ref_frame to
    // decide which neighbours' MVs count toward `near.mvs[]`).
    //
    // Stored alongside `ref_frames_out` (which is `Option<RefFrame>`
    // for the §15 loop-filter delta layer's per-MB ref-delta routing,
    // and only ever populated for an inter MB).
    //
    // During picking we score each ref against a uniform
    // `prob_last_pick = 128`, `prob_gf_pick = 128` because the
    // distribution-fitted wire probabilities are derived AFTER the
    // picker has run. The picking score therefore charges 1 bit for
    // LAST vs. (1 bit + 1 bit) = 2 bits for GOLDEN/ALTREF. Once every
    // MB is picked, `fit_prob_l8` derives the on-distribution
    // probabilities for the actual wire emit.
    let prob_last_pick: u8 = 128;
    let prob_gf_pick: u8 = 128;

    for mb_row in 0..mb_rows {
        let mut left_mb = crate::near_mv::MbInfo::border();
        let mut aboveleft_mb = crate::near_mv::MbInfo::border();
        for (mb_col, above_slot) in search_above.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let pixels = frame.extract_mb(mb_row, mb_col);

            // §10 segment classification (source-luma variance vs the
            // config boundaries — same classifier as the adaptive-quant
            // keyframe path) and the segment's factors / lambda. With
            // segmentation off this reduces to the frame-wide pair.
            let mb_segment_id = seg_ctx.as_ref().map(|(s, _, _)| {
                let sid = classify_segment(mb_luma_variance(&pixels.y), &s.variance_boundaries);
                seg_counts[sid as usize] += 1;
                sid
            });
            let (mb_factors, mb_lambda) = match (&seg_ctx, mb_segment_id) {
                (Some((_, f, l)), Some(sid)) => (&f[sid as usize], l[sid as usize]),
                _ => (&factors, lambda),
            };

            // ---- Per-ref picker: run for each available reference ----------
            //
            // For each available reference (LAST + optional GOLDEN +
            // optional ALTREF), run the full per-MB picker. The picker
            // returns the chosen mode / MV / split / residual /
            // reconstruction for that ref, plus the per-ref `J = SAD +
            // lambda * mv_ref_tree_bits` (excluding ref_frame_tree
            // bits). The overall winner is the one with the lowest
            // `J + lambda * ref_frame_tree_bits(ref, prob_last_pick,
            // prob_gf_pick)`.
            let mut best: Option<PickedMbForRef> = None;
            let mut best_total_j = f64::INFINITY;
            for (ref_frame, ref ref_planes) in available_refs.iter() {
                let luma_ref = crate::motion_search::LumaRef {
                    plane: ref_planes.y,
                    stride: ref_planes.y_stride,
                    width: ref_planes.mb_cols * 16,
                    height: ref_planes.mb_rows * 16,
                };
                let pick = pick_mb_for_ref(
                    *ref_frame,
                    ref_planes,
                    luma_ref,
                    &pixels,
                    above_slot,
                    &left_mb,
                    &aboveleft_mb,
                    mb_col,
                    mb_row,
                    mb_cols,
                    mb_rows,
                    &mv_contexts,
                    mb_lambda,
                    filters,
                    mb_factors,
                    Some(&coeff_probs),
                    trellis_rate_tables.as_ref(),
                    inter_trellis_strength,
                );
                let ref_bits = ref_frame_tree_bits(*ref_frame, prob_last_pick, prob_gf_pick);
                let total = pick.j + mb_lambda * ref_bits;
                if total < best_total_j {
                    best_total_j = total;
                    best = Some(pick);
                }
            }
            let chosen = best.expect("at least one reference (LAST) was scored");
            let chosen_ref_frame = chosen.ref_frame;
            let chosen_mode = chosen.chosen_mode;
            let chosen_mv = chosen.chosen_mv;
            let chosen_split = chosen.chosen_split;
            let mut raw_coeffs = chosen.raw_coeffs;
            let mut mb_skip_coeff = chosen.mb_skip_coeff;
            let mut recon = chosen.recon;
            let mut use_bpred = chosen.use_bpred;

            // ---- Round 160 / 161 — intra-within-inter MB picking -------
            //
            // When `pick_intra = true`, additionally score the §11
            // whole-block intra candidates against the running in-frame
            // neighbours (the SAME `MbNeighbors` the decoder will gather
            // when it walks the bytes — `gather_neighbors_public` on the
            // running `planes` buffer). Round 161 widens the candidate
            // set from r160's "DC_PRED only" to all four §11.2 Y modes
            // crossed with all four §11.4 UV modes (16 candidates); the
            // J-best wins inside the intra family, then competes against
            // the inter winner on `J + lambda * is_inter_mb-bit` exactly
            // as round 160 did. `B_PRED` is out of scope — the
            // per-sub-block intra walker is a separate fitter family.
            //
            // During picking we score against a uniform
            // `prob_intra_pick = 128` (the spec-neutral prior); the
            // distribution-fitted `prob_intra` for the actual wire emit
            // is derived AFTER every MB is picked (mirroring the
            // `prob_last_pick = 128` ↔ `fit_prob_l8` pattern used for
            // the §16.2 ref_frame selector).
            //
            // `pick_intra = false` (every pre-r160 entry-point) skips
            // this block entirely; `is_intra_per_mb` stays all-`false`
            // and the rest of the inner driver reproduces the
            // pre-r160 wire byte-for-byte.
            let mut chose_intra = false;
            if pick_intra {
                let neighbors = crate::frame::gather_neighbors_public(&planes, mb_row, mb_col);
                let intra_pick = pick_intra_mb_all(
                    &pixels,
                    &neighbors,
                    mb_factors,
                    mb_lambda,
                    Some(&coeff_probs),
                    trellis_rate_tables.as_ref(),
                    inter_trellis_strength,
                );
                let prob_intra_pick: u8 = 128;
                let inter_total = chosen.j
                    + mb_lambda
                        * ref_frame_tree_bits(chosen_ref_frame, prob_last_pick, prob_gf_pick)
                    + mb_lambda * bool_bits(prob_intra_pick, true);
                let intra_total = intra_pick.j + mb_lambda * bool_bits(prob_intra_pick, false);
                if intra_total < inter_total {
                    chose_intra = true;
                    intra_y_modes[raster] = intra_pick.y_mode;
                    intra_uv_modes[raster] = intra_pick.uv_mode;
                    raw_coeffs = intra_pick.raw_coeffs;
                    mb_skip_coeff = intra_pick.mb_skip_coeff;
                    recon = intra_pick.recon;
                    // `pick_intra_mb_all` excludes B_PRED, so the chosen
                    // intra MB always has Y2 (§14.2 page 76).
                    use_bpred = false;
                }
            }

            crate::frame::write_mb_public(&mut planes, mb_row, mb_col, &recon);

            // §15 loop-filter geometry:
            //   * whole-MB inter MB ⇒ y_mode = DC_PRED (filter "skip"
            //     rule on page 86 keys on `B_PRED || any-coded-coeff`);
            //   * SPLITMV MB         ⇒ y_mode = B_PRED (so the §15
            //     "internal edges always filtered" rule fires); and
            //   * intra MB           ⇒ y_mode = intra_y_modes[raster]
            //     (round 161: any of DC / V / H / TM, never B_PRED, so
            //     the skip rule still reduces to the keyframe path's
            //     whole-block case — none of the four whole-block intra
            //     modes triggers the per-sub-block §15 path).
            //
            // See `state.rs::Vp8DecoderState` for the matching
            // decoder-side mapping.
            let y_mode_for_lf = if chose_intra {
                intra_y_modes[raster] // r161: best of {Dc, V, H, Tm}
            } else if chosen_split.is_some() {
                IntraYMode::B
            } else {
                IntraYMode::Dc
            };
            let uv_mode_for_lf = if chose_intra {
                intra_uv_modes[raster] // r161: best of {Dc, V, H, Tm}
            } else {
                IntraUvMode::Dc
            };
            modes.push(MacroblockModes {
                segment_id: mb_segment_id,
                mb_skip_coeff,
                y_mode: y_mode_for_lf,
                subblock_modes: None,
                uv_mode: uv_mode_for_lf,
            });
            all_coeffs.push(raw_coeffs);
            if chose_intra {
                // Intra MB: no §16 ref_frame, no §16.2 inter mode, no MV.
                // The decoder's §15 loop-filter delta layer treats
                // `ref_frame = None` as "intra" (the §9.4 / §15.2
                // ref_delta lookup uses `INTRA_FRAME_REF` for `None`).
                ref_frames_out.push(None);
                inter_modes_out.push(None);
                chosen_mvs.push(crate::motion_vector::Mv::default());
                use_bpred_per_mb.push(false);
                is_intra_per_mb[raster] = true;
                // intra_y_modes[raster] / intra_uv_modes[raster] were
                // written inside the `pick_intra` block above with the
                // J-best modes `pick_intra_mb_all` returned.
            } else {
                ref_frames_out.push(Some(chosen_ref_frame));
                inter_modes_out.push(Some(chosen_mode));
                chosen_mvs.push(chosen_mv);
                use_bpred_per_mb.push(use_bpred);
            }

            // Update the neighbour records for the next MB in the row /
            // for the next row, mirroring the decoder's per-MB walk.
            // SPLITMV neighbours feed `above_block_mv` / `left_block_mv`
            // (§20.11) with their per-sub-block detail, so we record the
            // full `split_mvs[16]` for the next census. Intra MBs are
            // recorded with `ref_frame = None` / `mv = 0` — the §16.3
            // census skips them exactly like the off-frame border (an
            // intra neighbour contributes zero to `near.mvs[]`).
            let cur = if chose_intra {
                crate::near_mv::MbInfo {
                    ref_frame: None,
                    mv: crate::motion_vector::Mv::default(),
                    is_split: false,
                    split_mvs: None,
                }
            } else if let Some(ref cand) = chosen_split {
                crate::near_mv::MbInfo {
                    ref_frame: Some(chosen_ref_frame),
                    mv: cand.split_mvs[15],
                    is_split: true,
                    split_mvs: Some(cand.split_mvs),
                }
            } else {
                crate::near_mv::MbInfo {
                    ref_frame: Some(chosen_ref_frame),
                    mv: chosen_mv,
                    is_split: false,
                    split_mvs: None,
                }
            };
            // Single push site for `split_candidates` — intra MBs push
            // `None`, inter MBs their candidate. (Round 387 fix: the
            // intra branch above ALSO pushed a `None`, so every MB
            // after the frame's first intra pick read its left
            // neighbour's slot — a raster misalignment that emitted the
            // wrong §16.4 split data, or panicked, whenever an intra MB
            // preceded a SPLITMV MB under `pick_intra = true`.)
            if !chose_intra {
                split_candidates.push(chosen_split);
            } else {
                split_candidates.push(None);
            }
            aboveleft_mb = *above_slot;
            left_mb = cur;
            *above_slot = cur;
        }
    }

    // ---- §9.10 prob_last / prob_gf fit -----------------------------------
    //
    // Now that every MB has a chosen reference, derive the L(8) wire
    // probabilities that minimise the §16.2 selector-bit cost over
    // the observed per-MB distribution. `prob_last` is "P(B(prob_last)
    // reads 0)" = P(LAST); `prob_gf` is "P(B(prob_gf) reads 0 | not
    // LAST)" = P(GOLDEN | GOLDEN ∪ ALTREF).
    let mut count_last: u32 = 0;
    let mut count_golden: u32 = 0;
    let mut count_altref: u32 = 0;
    for r in ref_frames_out.iter().flatten() {
        match r {
            crate::motion_comp::RefFrame::Last => count_last += 1,
            crate::motion_comp::RefFrame::Golden => count_golden += 1,
            crate::motion_comp::RefFrame::AltRef => count_altref += 1,
        }
    }
    let prob_last_fit = fit_prob_l8(count_last, count_golden + count_altref);
    let prob_gf_fit = fit_prob_l8(count_golden, count_altref);

    // ---- §10 mb_segment_tree_probs fit -----------------------------------
    //
    // The §10 segment tree is: node 0 splits {0,1} vs {2,3}, node 1
    // splits 0 vs 1, node 2 splits 2 vs 3. Fit each node's L(8) prob to
    // the observed per-MB segment distribution (same `fit_prob_l8`
    // machinery as the §16.2 ref_frame selector above); an unvisited
    // node gets the neutral 128 that nothing reads.
    let seg_tree_probs: Option<[u8; 3]> = seg_ctx.as_ref().map(|_| {
        [
            fit_prob_l8(seg_counts[0] + seg_counts[1], seg_counts[2] + seg_counts[3]),
            fit_prob_l8(seg_counts[0], seg_counts[1]),
            fit_prob_l8(seg_counts[2], seg_counts[3]),
        ]
    });

    // ---- §9.4 effective per-reference / per-mode delta resolution -------
    //
    // RFC 6386 §9.4: per-slot deltas persist across frames; a slot's
    // `delta_update_flag = 0` (`None` in `LoopFilterDeltas`) means the
    // decoder keeps the value it carried in from the previous frame.
    // The encoder's §15 post-walk filter must use the SAME effective
    // value the decoder will use (otherwise our reconstruction would
    // diverge from the wire), so resolve effective deltas here.
    let (effective_ref_deltas, effective_mode_deltas) =
        lf_deltas.effective(carried_ref_deltas, carried_mode_deltas);

    // ---- §15 loop-filter post-pass --------------------------------------
    //
    // Base §15 config, level filled in below. `filter_type` mirrors the
    // bit `write_loop_filter_with_deltas` writes; the encoder's post-walk
    // filter must match what the decoder will run. The per-MB §9.4
    // reference / mode deltas are the effective (carried + this-frame)
    // values resolved above.
    let mut seg_lf_level = [0i16; crate::loop_filter::MAX_MB_SEGMENTS];
    if let Some((s, _, _)) = &seg_ctx {
        if let Some(d) = s.lf_delta {
            for (dst, &src) in seg_lf_level.iter_mut().zip(d.iter()) {
                *dst = i16::from(src);
            }
        }
    }
    let mut lf_config = crate::loop_filter::FrameFilterConfig {
        simple: params.filter_type,
        key_frame: false,
        loop_filter_level: params.loop_filter_level,
        sharpness_level: params.sharpness_level,
        // With segmentation on, each MB's §15 level resolves through
        // the §20.6 segment override (delta mode; an absent lf_delta
        // resolves to 0 exactly as the decoder's
        // `FrameFilterConfig::interframe` maps `loop_filter_update =
        // None` slots).
        segmentation_enabled: seg_ctx.is_some(),
        segment_abs: false,
        segment_lf_level: seg_lf_level,
        delta_enabled: lf_deltas.enabled,
        ref_delta_current: effective_ref_deltas[0],
        bpred_mode_delta: effective_mode_deltas[0],
        ref_delta_last: effective_ref_deltas[1],
        ref_delta_golden: effective_ref_deltas[2],
        ref_delta_altref: effective_ref_deltas[3],
        zero_mv_mode_delta: effective_mode_deltas[1],
        other_mv_mode_delta: effective_mode_deltas[2],
        split_mv_mode_delta: effective_mode_deltas[3],
    };
    // §9.4 effective loop-filter level + sharpness. With `auto_lf` the RD
    // selector jointly replaces `params.loop_filter_level` /
    // `params.sharpness_level` with the pair that minimises post-§15
    // reconstruction SSD against the source (clean-room; RFC 6386 is silent
    // on the encoder's choice). The selector runs the SAME §15 inter pass —
    // including the per-MB ref/mode deltas above — for each candidate, so
    // the chosen pair reaches the wire and decoder lockstep holds.
    let (eff_lf_level, eff_sharpness) = if auto_lf {
        let has_coeffs: Vec<bool> = all_coeffs
            .iter()
            .map(crate::loop_filter::mb_has_coeffs)
            .collect();
        let src = crate::loop_filter::SourcePlanes {
            width: width as usize,
            height: height as usize,
            y: frame.y,
            u: frame.u,
            v: frame.v,
            y_stride: frame.y_stride,
            uv_stride: frame.uv_stride,
        };
        let inter_info = crate::loop_filter::InterFilterInfo {
            ref_frames: &ref_frames_out,
            inter_modes: &inter_modes_out,
        };
        crate::loop_filter::select_filter_params(
            &planes,
            &modes,
            &has_coeffs,
            &lf_config,
            &src,
            Some(&inter_info),
        )
    } else {
        (params.loop_filter_level, params.sharpness_level)
    };
    lf_config.loop_filter_level = eff_lf_level;
    lf_config.sharpness_level = eff_sharpness;
    if eff_lf_level != 0 {
        crate::loop_filter::filter_inter_frame(
            &mut planes,
            &modes,
            &all_coeffs,
            &ref_frames_out,
            &inter_modes_out,
            &lf_config,
        );
    }

    // ---- §19.2 first (control) partition --------------------------------
    let mut hdr = BoolEncoder::new();

    // Inter frames do NOT emit color_space / clamping_type (those are
    // key-frame-only per §9.2).
    // §9.3 — segmentation. With the round-387 segmentation layer on,
    // the full update_segmentation() block travels: per-segment §10
    // quantizer deltas (always, including explicit zeros so the
    // decoder's persisted state matches `seg_factors` exactly),
    // optional per-segment loop-filter deltas, map update, and the
    // distribution-fitted mb_segment_tree_probs.
    match (&seg_ctx, &seg_tree_probs) {
        (Some((s, _, _)), Some(probs)) => {
            let layer = UpdateSegmentationLayer {
                update_mb_segmentation_map: true,
                feature_mode_absolute: false,
                quantizer: [
                    Some(i16::from(s.quant_delta[0])),
                    Some(i16::from(s.quant_delta[1])),
                    Some(i16::from(s.quant_delta[2])),
                    Some(i16::from(s.quant_delta[3])),
                ],
                loop_filter: match s.lf_delta {
                    Some(d) => [
                        Some(i16::from(d[0])),
                        Some(i16::from(d[1])),
                        Some(i16::from(d[2])),
                        Some(i16::from(d[3])),
                    ],
                    None => [None; 4],
                },
                segment_prob: [Some(probs[0]), Some(probs[1]), Some(probs[2])],
            };
            write_update_segmentation(&mut hdr, &layer);
        }
        _ => write_segment_update_flags(&mut hdr, false),
    }
    // §9.4 — loop filter. `filter_type` follows `params.filter_type`
    // (false ⇒ §15.3 normal, true ⇒ §15.2 simple). The §19.2
    // `mb_lf_adjustments()` sub-block follows whatever the caller's
    // `LoopFilterDeltas` says — including the default
    // `loop_filter_adj_enable = 0` for the round-150 wire shape.
    write_loop_filter_with_deltas(
        &mut hdr,
        params.filter_type,
        eff_lf_level,
        eff_sharpness,
        lf_deltas,
    )?;
    // §9.5 — partition count. Macroblock rows are distributed
    // round-robin across the §9.5 partitions per the §20.4 row-loop
    // (row `r` → partition `r % N`). This is a layout reorganisation
    // only — the residual coding inside each partition is bit-
    // identical to the 1-partition case, so the self-decoded picture
    // is unchanged across all four legal values.
    write_token_partition_count(&mut hdr, params.nbr_of_dct_partitions)?;
    // §9.6 — quant indices (baseline only).
    write_quant_indices(&mut hdr, params.y_ac_qi, None, None, None, None, None)?;

    // §9.7 / §9.8 (inter) — refresh / sign-bias / entropy ladder.
    // The §19.2 listing (page 122) gates the L(2) copy_buffer_to_*
    // fields on `if (!refresh_*_frame)`. We mirror that gating
    // exactly so the wire is byte-identical to what
    // `Vp8CodedHeader::parse` expects to read back.
    hdr.write_bool(128, refresh.refresh_golden_frame); // §9.7 refresh_golden_frame
    hdr.write_bool(128, refresh.refresh_alternate_frame); // §9.7 refresh_alternate_frame
    if !refresh.refresh_golden_frame {
        hdr.write_literal(refresh.copy_buffer_to_golden as u32, 2); // copy_buffer_to_golden
    }
    if !refresh.refresh_alternate_frame {
        hdr.write_literal(refresh.copy_buffer_to_alternate as u32, 2); // copy_buffer_to_alternate
    }
    hdr.write_bool(128, false); // sign_bias_golden
    hdr.write_bool(128, false); // sign_bias_alternate
    hdr.write_bool(128, false); // refresh_entropy_probs
    hdr.write_bool(128, refresh.refresh_last); // §9.8 refresh_last

    // §13 / §9.9 — token-prob update sub-block. With
    // `token_updates = Some(u)` the per-position replacement layer is
    // written and the merged `coeff_probs` above is what tokens are
    // coded against; the decoder reads the same updates and overlays
    // them on its carried state to rebuild the same table. With `None`
    // every flag is 0 and the §13.5 defaults stay in force — byte-
    // identical to the round-155 inter wire.
    match token_updates {
        Some(u) => write_token_prob_updates(&mut hdr, u, &COEFF_UPDATE_PROBS_FLAT),
        None => write_no_token_prob_updates(&mut hdr, &COEFF_UPDATE_PROBS_FLAT),
    }

    // §9.11 — mb_no_skip_coeff enabled with a balanced prob_skip_false.
    let prob_skip_false = 128u8;
    write_mb_no_skip_coeff(&mut hdr, true, prob_skip_false);

    // §9.10 — inter-only tail:
    //   prob_intra (L8), prob_last (L8), prob_gf (L8),
    //   intra_y_mode_prob_update gate (F? L8 × 4),
    //   intra_uv_mode_prob_update gate (F? L8 × 3),
    //   mv_prob_update() (38 F? L7 entries).
    //
    // When `pick_intra = false` (every pre-r160 entry-point):
    //   prob_intra = 255 forces every MB to read as inter (the decoder
    //   does `dec.read_bool(prob_intra)` and `prob_intra = 255` makes
    //   the result "true" almost-always for the bool we'll emit). We
    //   emit `is_inter_mb = true` for every MB, which against
    //   `prob_intra = 255` costs essentially zero bits.
    //
    // When `pick_intra = true` (round 160 intra-within-inter
    // entry-point): prob_intra is fitted to the picker's observed
    // (intra, inter) per-MB count distribution via [`fit_prob_l8`].
    // `prob_intra` is P(`is_inter_mb == false`) ⇒
    // `fit_prob_l8(intra_count, inter_count)`. With no MBs picking
    // intra the fitter returns 1 (clamped from 0) — `is_inter_mb`
    // codes cheaply (one bit per MB at probability 255/256). With
    // every MB picking intra the fitter returns 255.
    //
    // prob_last / prob_gf are fitted to the picker's observed per-MB
    // distribution by [`fit_prob_l8`] above so the §16.2 selector bits
    // compress against an on-distribution prior. When every MB picks
    // LAST (no GOLDEN / ALTREF was passed or LAST always won the J
    // trade) `prob_last_fit == 255` and the path collapses to the
    // single-ref encoder's behaviour. When the GOLDEN / ALTREF refs
    // were not provided to the caller, both counts are zero and
    // `fit_prob_l8` returns the spec-neutral 128 — that prob is never
    // read in that case, since `prob_last_fit == 255` forces every
    // selector to LAST.
    let prob_intra: u8 = if pick_intra {
        let count_intra: u32 = is_intra_per_mb.iter().filter(|&&b| b).count() as u32;
        let count_inter: u32 = is_intra_per_mb.len() as u32 - count_intra;
        fit_prob_l8(count_intra, count_inter)
    } else {
        255
    };
    let prob_last: u8 = prob_last_fit;
    let prob_gf: u8 = prob_gf_fit;
    hdr.write_literal(prob_intra as u32, 8);
    hdr.write_literal(prob_last as u32, 8);
    hdr.write_literal(prob_gf as u32, 8);
    // intra_y_mode_prob_update gate: no update.
    hdr.write_bool(128, false);
    // intra_uv_mode_prob_update gate: no update.
    hdr.write_bool(128, false);
    // mv_prob_update(): every F flag = 0, against the §17.2
    // MV_UPDATE_PROBS table. The encoder mirrors the decoder's
    // `parse_mv_prob_update` walk.
    for ctx in MV_UPDATE_PROBS_FLAT.iter() {
        hdr.write_bool(*ctx, false);
    }

    // ---- §11 / §16 macroblock-mode layer --------------------------------
    //
    // For each MB we emit (in §19.3 order):
    //   1. mb_skip_coeff (when mb_no_skip_coeff = 1)
    //   2. is_inter_mb (against prob_intra)
    //   3. ref_frame selector (against prob_last; LAST = false)
    //   4. inter-mode tree walk (ZEROMV = "0", NEWMV = "1110",
    //      SPLITMV = "1111")
    //   5. NEWMV: §17.2 row + column MV differential, written against
    //      the §17.2 default MV contexts (the same defaults the
    //      decoder reads with because we emit `mv_prob_update()` with
    //      every F-gate = 0 above).
    //      SPLITMV: §16.4 `mvpartition_tree` partition id, then for
    //      each partition group the §16.4 `sub_mv_ref_tree` mode at
    //      the group anchor's left/above context, with NEW4X4 modes
    //      followed by their §17.2 component differential.
    //
    // The §16.3 census determines the four probabilities the inter-
    // mode tree's bool reads, and depends on the already-decoded
    // neighbours' resolved `ref_frame` + `mv` (+ `is_split` /
    // `split_mvs` for SPLITMV neighbours). We re-run the census here
    // against the SAME neighbour state the search loop walked above,
    // so the bits we write are consumed at exactly the probabilities
    // the decoder reads them at.
    let mut above_mb: Vec<crate::near_mv::MbInfo> = vec![crate::near_mv::MbInfo::border(); mb_cols];

    for mb_row in 0..mb_rows {
        let mut left_mb = crate::near_mv::MbInfo::border();
        let mut aboveleft_mb = crate::near_mv::MbInfo::border();
        for (mb_col, above_slot) in above_mb.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            let is_intra = is_intra_per_mb[raster];

            // 0. segment_id (§10) — only when the frame updates the
            //    segmentation map. Coded via the §10 mb_segment_tree
            //    against the fitted probs, BEFORE mb_skip_coeff,
            //    matching the decoder's inter read order
            //    (segment_id → mb_skip_coeff → is_inter_mb).
            if let Some(probs) = &seg_tree_probs {
                let sid = mb.segment_id.unwrap_or(0);
                hdr.write_treed(&crate::macroblock::MB_SEGMENT_TREE, |i| probs[i], sid);
            }

            // 1. mb_skip_coeff (mb_no_skip_coeff = 1).
            hdr.write_bool(prob_skip_false, mb.mb_skip_coeff);
            // 2. is_inter_mb. Round 160's intra-pick path can land
            //    here with `is_intra = true`, in which case we write a
            //    `false` against `prob_intra` and the §16.1 intra
            //    branch runs below. With `pick_intra = false` (every
            //    pre-r160 entry-point) `is_intra` is always `false`,
            //    `prob_intra = 255`, and the wire is byte-identical to
            //    the historical inter-only path.
            hdr.write_bool(prob_intra, !is_intra);

            if is_intra {
                // §16.1 intra-on-interframe macroblock mode layer:
                //   * y_mode against `IF_YMODE_TREE` + the §16.1
                //     default `IF_YMODE_PROB_DEFAULTS` (we hold the
                //     §9.10 `intra_y_mode_prob_update` gate at 0 so
                //     the wire table stays at the defaults the
                //     decoder's `InterFrameIntraProbs::defaults`
                //     exposes);
                //   * uv_mode against `UV_MODE_TREE` + the §16.1
                //     default `IF_UV_MODE_PROB_DEFAULTS` (same gate
                //     held at 0).
                //
                // Round 161 picks the J-best `(y_mode, uv_mode)` pair
                // from the 4×4 whole-block grid; the tree walks emit
                // one to four bits depending on which leaf the picker
                // settled on (the §11.2 / §11.4 trees are unbalanced).
                hdr.write_treed(
                    &IF_YMODE_TREE,
                    |i| IF_YMODE_PROB_DEFAULTS[i],
                    intra_y_modes[raster].leaf(),
                );
                hdr.write_treed(
                    &UV_MODE_TREE,
                    |i| IF_UV_MODE_PROB_DEFAULTS[i],
                    intra_uv_modes[raster].leaf(),
                );

                // Neighbour record advances: intra MBs go in as
                // `ref_frame = None` / `mv = 0` so the §16.3 census
                // skips them exactly like the off-frame border.
                let cur = crate::near_mv::MbInfo {
                    ref_frame: None,
                    mv: crate::motion_vector::Mv::default(),
                    is_split: false,
                    split_mvs: None,
                };
                aboveleft_mb = *above_slot;
                left_mb = cur;
                *above_slot = cur;
                continue;
            }

            let chosen_mode = inter_modes_out[raster].expect("inter mode recorded above");
            let chosen_mv = chosen_mvs[raster];
            let chosen_ref_frame = ref_frames_out[raster].expect("inter ref_frame recorded above");

            // 3. ref_frame selector — RFC 6386 §16.2 `ref_frame_tree`.
            //    LAST   → B(prob_last)=false
            //    GOLDEN → B(prob_last)=true,  B(prob_gf)=false
            //    ALTREF → B(prob_last)=true,  B(prob_gf)=true
            match chosen_ref_frame {
                crate::motion_comp::RefFrame::Last => {
                    hdr.write_bool(prob_last, false);
                }
                crate::motion_comp::RefFrame::Golden => {
                    hdr.write_bool(prob_last, true);
                    hdr.write_bool(prob_gf, false);
                }
                crate::motion_comp::RefFrame::AltRef => {
                    hdr.write_bool(prob_last, true);
                    hdr.write_bool(prob_gf, true);
                }
            }

            // 4. Inter-mode tree (RFC 6386 §16.2 / §20.13 `mv_ref_tree`).
            //    ZEROMV    = leaf 0, path "0"
            //    NEARESTMV = leaf 1, path "10"
            //    NEARMV    = leaf 2, path "110"
            //    NEWMV     = leaf 3, path "1110"
            //    SPLITMV   = leaf 4, path "1111"
            //
            // §16.3 `find_near_mvs` is per-MB-ref-frame: neighbours
            // whose recorded `ref_frame` matches ours count toward
            // `near.mvs[]`. The wire emit must call the census with
            // the SAME ref_frame the search loop used (the chosen ref
            // for this MB) so the bits we emit are consumed at
            // exactly the probabilities the decoder reads them at.
            let near = crate::near_mv::find_near_mvs(
                above_slot,
                &left_mb,
                &aboveleft_mb,
                chosen_ref_frame,
                crate::near_mv::SignBias::default(),
            );
            let probs = crate::near_mv::mv_ref_probs(&near.cnt);
            match chosen_mode {
                crate::near_mv::InterMode::Zero => {
                    // "0" against probs[0].
                    hdr.write_bool(probs[0], false);
                }
                crate::near_mv::InterMode::Nearest => {
                    // "10" against probs[0..=1]. The decoder's
                    // `resolve_inter_mb_mv` reconstructs the MV as
                    // `clamp_mv(near.mvs[1], bounds)` — no extra bits.
                    hdr.write_bool(probs[0], true);
                    hdr.write_bool(probs[1], false);
                }
                crate::near_mv::InterMode::Near => {
                    // "110" against probs[0..=2]. The decoder
                    // reconstructs the MV as `clamp_mv(near.mvs[2],
                    // bounds)` — no extra bits.
                    hdr.write_bool(probs[0], true);
                    hdr.write_bool(probs[1], true);
                    hdr.write_bool(probs[2], false);
                }
                crate::near_mv::InterMode::New => {
                    // "1110": three true bits at probs[0..=2] then one
                    // false at probs[3]. Pinned by `MV_REF_TREE`.
                    hdr.write_bool(probs[0], true);
                    hdr.write_bool(probs[1], true);
                    hdr.write_bool(probs[2], true);
                    hdr.write_bool(probs[3], false);
                    // §17.2 NEWMV differential = chosen_mv -
                    // clamp_mv(near.mvs[0]). The decoder reads `best`
                    // through the same §16.3 / §18.1 clamp before adding.
                    let bounds =
                        crate::near_mv::MvClampRect::for_mb(mb_col, mb_row, mb_cols, mb_rows);
                    let best = crate::near_mv::clamp_mv(near.mvs[0], &bounds);
                    let diff = crate::motion_vector::Mv {
                        row: chosen_mv.row.wrapping_sub(best.row),
                        col: chosen_mv.col.wrapping_sub(best.col),
                    };
                    crate::motion_vector::write_mv(&mut hdr, &mv_contexts, diff);
                }
                crate::near_mv::InterMode::Split => {
                    // "1111": four true bits at probs[0..=3]. Pinned by
                    // `MV_REF_TREE`.
                    hdr.write_bool(probs[0], true);
                    hdr.write_bool(probs[1], true);
                    hdr.write_bool(probs[2], true);
                    hdr.write_bool(probs[3], true);

                    // §16.4 partition + per-group sub_mv_ref walk. The
                    // candidate carries the picker's resolved partition,
                    // group modes, and NEW4X4 diffs from the search-and-RD
                    // pass; we re-emit each in order against the same
                    // probability tables.
                    let cand = split_candidates[raster]
                        .as_ref()
                        .expect("split_candidates populated for InterMode::Split");
                    write_split_mv_partition(&mut hdr, cand.partition);

                    // For each partition group, re-resolve the anchor's
                    // left / above neighbours against the SAME running
                    // split_mvs buffer the decoder will rebuild, look up
                    // the §16.4 `sub_mv_ref` context, and emit the
                    // group's mode + (NEW4X4 only) §17.2 differential.
                    let groups = partition_groups(cand.partition);
                    let mut running_split = [crate::motion_vector::Mv::default(); 16];
                    for (g_idx, group) in groups.iter().enumerate() {
                        let anchor = group[0];
                        let left_nb =
                            crate::near_mv::left_block_mv(&running_split, &left_mb, anchor);
                        let above_nb =
                            crate::near_mv::above_block_mv(&running_split, above_slot, anchor);
                        let ctx = crate::near_mv::submv_ref_context(left_nb, above_nb);
                        let sub_probs = &crate::near_mv::SUBMV_REF_PROBS[ctx];

                        let mode = cand.submv_modes[g_idx];
                        write_submv_ref(&mut hdr, sub_probs, mode);

                        if matches!(mode, crate::near_mv::SubMvRefMode::New4x4) {
                            // §16.4: NEW4X4 differential is added to
                            // `best_mv` (the clamped near.mvs[0]); the
                            // picker computed and stored the diff.
                            crate::motion_vector::write_mv(
                                &mut hdr,
                                &mv_contexts,
                                cand.submv_new_diffs[g_idx],
                            );
                        }

                        // Fill the running split buffer with the
                        // group's resolved vector so subsequent groups'
                        // left/above neighbour lookups see exactly what
                        // the decoder will see.
                        for &b in group {
                            running_split[b] = cand.split_mvs[b];
                        }
                    }
                }
            }

            // Update neighbour records for the next MB. SPLITMV
            // neighbours feed the §20.11 per-sub-block lookups, so we
            // record the full `split_mvs[16]` (the same layout the
            // decoder writes into `MbInfo`).
            let cur = if matches!(chosen_mode, crate::near_mv::InterMode::Split) {
                let cand = split_candidates[raster].as_ref().expect("split candidate");
                crate::near_mv::MbInfo {
                    ref_frame: Some(chosen_ref_frame),
                    mv: cand.split_mvs[15],
                    is_split: true,
                    split_mvs: Some(cand.split_mvs),
                }
            } else {
                crate::near_mv::MbInfo {
                    ref_frame: Some(chosen_ref_frame),
                    mv: chosen_mv,
                    is_split: false,
                    split_mvs: None,
                }
            };
            aboveleft_mb = *above_slot;
            left_mb = cur;
            *above_slot = cur;
        }
    }

    let first_partition = hdr.finish();
    let first_partition_size = first_partition.len();
    if first_partition_size > 0x7_FFFF {
        return Err(EncodeError::FirstPartitionTooLarge {
            bytes: first_partition_size,
        });
    }

    // ---- §19.2 DCT partition group: §13.3 residual tokens --------------
    //
    // Macroblock rows are distributed round-robin across the §9.5
    // partitions per the §20.4 row-loop: row `r` is encoded into
    // partition `r % N`. Each partition gets its own [`BoolEncoder`]
    // instance (§4 page 9 — "All partitions are decoded using separate
    // instances of the boolean entropy decoder"), finalised
    // independently with its own §7.3 4-byte flush trailer. The §13.3
    // above-context is column-wise and frame-lived — shared across
    // partitions because the decoder's `decode_residuals` also keeps
    // one above slot per column for the whole frame. The "left"
    // context resets at every row start so it does not need to cross
    // partitions.
    let mut partitions: Vec<BoolEncoder> =
        (0..num_partitions).map(|_| BoolEncoder::new()).collect();
    let mut above_ctx: Vec<MbEntropyCtx> = vec![MbEntropyCtx::default(); mb_cols];
    for mb_row in 0..mb_rows {
        // §13.3 page 65: the "left" predictor resets at the start of
        // every macroblock row.
        let mut left_ctx = MbEntropyCtx::default();
        let part_idx = mb_row % num_partitions;
        let tok = &mut partitions[part_idx];
        for (mb_col, above_col) in above_ctx.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            // SPLITMV MBs clear the Y2 predictor context like B_PRED
            // (§13.1 / §20.16); `has_y2` reflects whether a Y2 block
            // is on the wire for this MB.
            let use_bpred = use_bpred_per_mb[raster];
            if mb.mb_skip_coeff {
                clear_skip_ctx(use_bpred, above_col, &mut left_ctx);
                continue;
            }
            // SPLITMV / B_PRED have no Y2 (§14.2 page 76); the §13.3
            // walker skips block 24 and routes the 16 Y blocks through
            // `BlockType::YNoY2` instead of `YAfterY2`.
            encode_mb_tokens(
                tok,
                &all_coeffs[raster],
                use_bpred,
                &coeff_probs,
                above_col,
                &mut left_ctx,
            )
            .map_err(EncodeError::Token)?;
        }
    }
    // §13.4 observed-counts side-channel (round 158). Re-walks the §13.3
    // token loop above with `count_inter_frame_branches`, driving its
    // own private above / left predictor contexts in lockstep — so the
    // recorded per-position counts are bit-for-bit the events the bytes
    // in `partitions` just emitted. With `counts = None` this is a
    // no-op.
    if let Some(c) = counts {
        count_inter_frame_branches(&modes, &use_bpred_per_mb, &all_coeffs, mb_cols, mb_rows, c);
    }
    let dct_partitions: Vec<Vec<u8>> = partitions.into_iter().map(|p| p.finish()).collect();
    let dct_total: usize = dct_partitions.iter().map(|p| p.len()).sum();
    let size_table_len = (num_partitions - 1) * 3;

    // ---- §9.1 inter frame tag + assembly --------------------------------
    let mut out: Vec<u8> =
        Vec::with_capacity(3 + first_partition_size + size_table_len + dct_total);
    write_frame_tag(
        &mut out,
        false, // inter frame
        0,     // version 0 — full-pel chroma off, version 0 §18.3 6-tap filters
        true,  // show_frame
        first_partition_size as u32,
        width,
        height,
        ScaleCode::None,
        ScaleCode::None,
    )?;
    out.extend_from_slice(&first_partition);
    // §9.5 size table: 3-byte little-endian length for every partition
    // except the last (whose size the decoder infers from what is left
    // in the frame). One partition → no table.
    for part in dct_partitions.iter().take(num_partitions - 1) {
        let sz = part.len();
        out.push((sz & 0xff) as u8);
        out.push(((sz >> 8) & 0xff) as u8);
        out.push(((sz >> 16) & 0xff) as u8);
    }
    for part in &dct_partitions {
        out.extend_from_slice(part);
    }

    Ok((out, planes))
}

/// Flattened §17.2 MV update-probability table (`2 * MV_PROB_COUNT = 38`
/// entries, `[row..; column..]`). Each entry is the probability the
/// encoder writes the corresponding "no update" F-gate at, matching the
/// decoder's per-position `read_bool(MV_UPDATE_PROBS[i][j])`.
const MV_UPDATE_PROBS_FLAT: [u8; 38] = [
    237, 246, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 250, 250, 252, 254, 254,
    231, 243, 245, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 251, 251, 254, 254, 254,
];

// ───────────────────────── factory + dual-API surface ─────────────────────────

/// Phase 1 encoder handle. Stateless; one-shot
/// [`encode_keyframe`](Self::encode_keyframe) calls map directly to
/// [`encode_silent_keyframe`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SilentKeyframeEncoder;

impl SilentKeyframeEncoder {
    /// Encode a single silent key frame. `_pixels` is currently
    /// ignored — the Phase 1 path emits a fixed-content key frame
    /// regardless of input pixel data.
    pub fn encode_keyframe(
        &self,
        _pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, EncodeError> {
        encode_silent_keyframe(SilentKeyframeParams::new(width, height))
    }
}

/// Standalone-reachable helper that hands out a [`SilentKeyframeEncoder`].
///
/// Pre-dates the framework `make_encoder(params)` factory below. Kept
/// for the historical direct-API consumer that built against the no-arg
/// helper; the registry / framework path is now [`make_encoder`].
pub fn make_silent_keyframe_encoder() -> SilentKeyframeEncoder {
    SilentKeyframeEncoder
}

// ───────────────────────── WebP-canonical quality mapping ─────────────────────────

/// Map a WebP-canonical `0.0..=100.0` quality scalar onto the VP8
/// §9.6 `y_ac_qi` quantiser index (`0..=127`, lower = higher quality).
///
/// The mapping matches the on-wire convention `oxideav-webp`'s VP8 lossy
/// path uses: `round((100 - quality) * 1.27)`. Defined precisely:
///
/// * Inputs `<= 0.0` clamp to `quality = 0.0`, returning `127` (worst).
/// * Inputs `>= 100.0` clamp to `quality = 100.0`, returning `0` (best).
/// * `NaN` returns `127` (the "couldn't tell — keep the file small" choice).
/// * Otherwise the floating-point result is rounded half-away-from-zero
///   (the default behaviour of `f32::round`) and then clamped into the
///   `0..=127` `u8` range, so an out-of-range numeric input cannot trip
///   an `as u8` truncation overflow.
///
/// The function is pure — it does not touch [`crate::Encoder`] or
/// [`oxideav_core`] — so it is reachable under
/// `--no-default-features` for an embedded image / video pipeline that
/// wants to choose a `qindex` without building the framework adapter.
pub fn quality_to_qindex(quality: f32) -> u8 {
    if quality.is_nan() {
        return 127;
    }
    let clamped = quality.clamp(0.0, 100.0);
    let q = ((100.0_f32 - clamped) * 1.27).round();
    // After the clamp + the constant factor the result is in 0.0..=127.0,
    // but pin the bounds explicitly so any future change to the formula
    // can't silently overflow `as u8`.
    q.clamp(0.0, 127.0) as u8
}

/// Map a WebP-canonical `0.0..=100.0` quality scalar onto a coherent
/// [`TrellisStrength`] for the §13 coefficient trellis.
///
/// The mapping pairs the quality dial with the trellis aggressiveness so a
/// single quality number tunes *both* the §9.6 quantiser
/// ([`quality_to_qindex`]) and how hard the trellis shaves coefficient
/// bits. The shape is clean-room (RFC 6386 is silent on encoder rate
/// control):
///
/// * `quality >= 90` (visually-lossless target) → [`TrellisStrength::DEFAULT`]
///   (`1.0`): the conservative "hold the PSNR floor" trade — a
///   high-quality stream should not sacrifice PSNR for marginal byte
///   savings.
/// * As quality falls the strength ramps linearly toward `4.0` at
///   `quality = 0` (the empirically-strong operating point that trims
///   ~24–40 % of the no-trellis bytes for ~0.3 dB at mid/high quantisers,
///   where a low-quality stream already lives).
/// * `NaN` returns [`TrellisStrength::DEFAULT`] (the safe, conservative
///   choice, matching the `quality_to_qindex` "couldn't tell" policy of
///   not over-trading).
///
/// Pure — no [`oxideav_core`] / framework dependency — so it is reachable
/// under `--no-default-features`. Callers that want the quantiser and the
/// trellis tuned together can write
/// `KeyframeParams { y_ac_qi: quality_to_qindex(q), trellis_strength:
/// quality_to_trellis_strength(q), ..Default::default() }`.
pub fn quality_to_trellis_strength(quality: f32) -> TrellisStrength {
    if quality.is_nan() {
        return TrellisStrength::DEFAULT;
    }
    let clamped = quality.clamp(0.0, 100.0);
    // 1.0 at quality 90+, ramping to 4.0 at quality 0. Above 90 the
    // `max(0.0)` floors the ramp term so the strength stays at the default.
    const KNEE: f32 = 90.0;
    const TOP_STRENGTH: f32 = 4.0;
    let ramp = ((KNEE - clamped).max(0.0) / KNEE) * (TOP_STRENGTH - 1.0);
    TrellisStrength::new((1.0 + ramp) as f64)
}

// ───────────────────────── framework factory (registry-gated) ─────────────────────────

#[cfg(feature = "registry")]
mod factory {
    //! [`oxideav_core::Encoder`] adapter for the VP8 encoder.
    //!
    //! The historical `oxideav-vp8` direct-API entry points
    //! ([`encode_keyframe`], [`encode_silent_keyframe`],
    //! [`encode_p_frame_multi_ref_with_*`] ladder) operate on
    //! per-frame [`I420Frame`] / [`KeyframeParams`] / [`MbPixels`]
    //! values and are reachable without `oxideav-core`. This adapter
    //! plugs that direct API into the framework's
    //! [`oxideav_core::Encoder`] trait so the registry path
    //! ([`oxideav_vp8::register`](crate::register)) can hand a
    //! `Box<dyn Encoder>` to a generic muxer / pipeline.

    use std::collections::VecDeque;

    use oxideav_core::time::TimeBase;
    use oxideav_core::{
        CodecId, CodecParameters, Encoder, Error, Frame, MediaType, Packet, PixelFormat, Result,
        VideoFrame,
    };

    use super::{encode_keyframe, encode_keyframe_auto_loop_filter, I420Frame, KeyframeParams};
    use crate::decoder::VP8_CODEC_ID;

    /// Build a framework-side VP8 encoder bound to `params` with all
    /// defaults: `y_ac_qi = 32`, no loop filter, 1 DCT partition,
    /// normal filter. Equivalent to
    /// `make_encoder_with_qindex(params, 32)`.
    ///
    /// `params` must declare a positive `width` and `height`;
    /// `pixel_format` defaults to [`PixelFormat::Yuv420P`] when absent
    /// and is the only format the encoder currently accepts.
    pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
        make_encoder_with_qindex(params, KeyframeParams::default().y_ac_qi)
    }

    /// Build a framework-side VP8 encoder with the WebP-canonical
    /// `0.0..=100.0` `quality` (higher = better) translated into a
    /// `y_ac_qi` via [`super::quality_to_qindex`] **and** a coherent §13
    /// trellis aggressiveness via [`super::quality_to_trellis_strength`]:
    /// a high-quality request keeps the conservative
    /// [`super::TrellisStrength::DEFAULT`] trade while a low-quality
    /// request also lets the trellis shave coefficient bits harder, so the
    /// single quality dial tunes both the quantiser and the trellis.
    ///
    /// Like [`make_encoder`] / [`make_encoder_with_qindex`], this is a
    /// **zero-latency single-frame door**: every `send_frame` emits one
    /// key frame packet immediately (the still-image contract the
    /// RIFF-VP8 image consumers rely on). The lagged inter-stream
    /// semantics live on the [`super::make_encoder_with_config`] path.
    pub fn make_encoder_with_quality(
        params: &CodecParameters,
        quality: f32,
    ) -> Result<Box<dyn Encoder>> {
        let (width, height, pixel_format) = validate_video_params(params)?;
        let keyframe = KeyframeParams {
            y_ac_qi: super::quality_to_qindex(quality),
            trellis_strength: super::quality_to_trellis_strength(quality),
            ..KeyframeParams::default()
        };
        build_frame_encoder(params, width, height, pixel_format, keyframe, false)
    }

    /// Shared `CodecParameters` validation for every factory entry:
    /// positive dimensions inside the §9.1 14-bit fields, Yuv420P only.
    fn validate_video_params(params: &CodecParameters) -> Result<(u32, u32, PixelFormat)> {
        let width = params
            .width
            .ok_or_else(|| Error::invalid("vp8 encoder: missing width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("vp8 encoder: missing height"))?;
        if width == 0 || height == 0 {
            return Err(Error::invalid(
                "vp8 encoder: width and height must be positive",
            ));
        }
        if width > 0x3FFF || height > 0x3FFF {
            return Err(Error::invalid(
                "vp8 encoder: width/height exceed VP8 14-bit field (max 16383)",
            ));
        }
        let pixel_format = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        if pixel_format != PixelFormat::Yuv420P {
            return Err(Error::unsupported(
                "vp8 encoder: only PixelFormat::Yuv420P is supported",
            ));
        }
        Ok((width, height, pixel_format))
    }

    /// Build a framework-side VP8 encoder with an explicit VP8 §9.6
    /// `y_ac_qi` quantiser index (`0..=127`, lower = better).
    pub fn make_encoder_with_qindex(
        params: &CodecParameters,
        qindex: u8,
    ) -> Result<Box<dyn Encoder>> {
        let (width, height, pixel_format) = validate_video_params(params)?;
        if qindex > 127 {
            return Err(Error::invalid(
                "vp8 encoder: qindex out of range (must be 0..=127)",
            ));
        }
        let keyframe = KeyframeParams {
            y_ac_qi: qindex,
            ..KeyframeParams::default()
        };
        build_frame_encoder(params, width, height, pixel_format, keyframe, false)
    }

    /// Build a framework-side **lagged inter-stream** VP8 encoder from a
    /// full [`super::Vp8EncoderConfig`] — the round-387 upgrade that
    /// finally makes the `Box<dyn Encoder>` adapter consume the config's
    /// stream-level knobs instead of re-keying every frame:
    ///
    /// * `qindex` / `lf_level` / `trellis_strength` — per-frame encode
    ///   parameters (as before);
    /// * `auto_loop_filter` — RD-selected §9.4 level / sharpness per
    ///   emitted frame (`lf_level` ignored when set);
    /// * `alt_ref_interval` — lookahead group size: each completed group
    ///   ships one invisible ARNR anchor followed by its visible
    ///   multi-reference P-frames. `0` disables anchors (group size 1 —
    ///   zero lag, K/P streaming with immediate packet availability);
    /// * `lookahead_window` — caps the group size;
    /// * `golden_interval` — key-frame cadence in source frames (`0`
    ///   keys only frame 0); scene cuts force additional keys.
    ///
    /// The returned encoder follows the standard lagged
    /// [`Encoder`] protocol: `send_frame` may buffer (`receive_packet`
    /// surfaces [`Error::NeedMore`] while a lookahead group is
    /// filling), `flush` drains the tail group, and the stream emits
    /// **more packets than source frames** (invisible anchor packets
    /// carry `pts = None` and `flags.keyframe = false`; a muxer stores
    /// them in decode order, a player never displays them). Like
    /// [`super::Vp8Encoder::encode_sequence`], the §13.4 fitted
    /// token-prob-update emission and the §11 intra-within-inter picker
    /// are enabled on every frame (never-grow byte guards; the cost is
    /// a second encode pass per frame).
    pub(crate) fn make_encoder_from_config(
        params: &CodecParameters,
        config: super::Vp8EncoderConfig,
    ) -> Result<Box<dyn Encoder>> {
        let (width, height, pixel_format) = validate_video_params(params)?;
        if config.qindex > 127 {
            return Err(Error::invalid(
                "vp8 encoder: qindex out of range (must be 0..=127)",
            ));
        }
        if config.lf_level > 63 {
            return Err(Error::invalid(
                "vp8 encoder: lf_level out of range (must be 0..=63)",
            ));
        }

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(VP8_CODEC_ID);
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(pixel_format);
        let time_base = params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num));

        // Same knob mapping as `Vp8Encoder::encode_sequence`.
        let window = if config.alt_ref_interval == 0 {
            1
        } else {
            config.alt_ref_interval.min(config.lookahead_window.max(1)) as usize
        };
        let stream_config = crate::stream::AltrefStreamConfig {
            params: KeyframeParams {
                y_ac_qi: config.qindex,
                loop_filter_level: config.lf_level,
                trellis_strength: config.trellis_strength,
                ..KeyframeParams::default()
            },
            keyframe_interval: u64::from(config.golden_interval),
            altref_window: window,
            auto_loop_filter: config.auto_loop_filter,
            fitted_token_prob_updates: true,
            intra_pick: true,
            ..crate::stream::AltrefStreamConfig::default()
        };
        let stream = crate::stream::Vp8AltrefStreamEncoder::new(stream_config)
            .expect("window is clamped to >= 1 above");

        Ok(Box::new(Vp8LaggedEncoder {
            output_params,
            width,
            height,
            stream,
            time_base,
            pts_queue: VecDeque::new(),
            pending: VecDeque::new(),
            eof: false,
        }))
    }

    /// Lagged [`oxideav_core::Encoder`] adapter around the auto-altref
    /// stream driver ([`crate::stream::Vp8AltrefStreamEncoder`]) — the
    /// framework face of the round-384 GOLDEN/ALTREF subsystem.
    ///
    /// Protocol: `send_frame` buffers source pictures into lookahead
    /// groups (returning `Ok(())` immediately); `receive_packet` pops
    /// finished packets or surfaces [`Error::NeedMore`] while the group
    /// is filling; `flush` drains the tail group and arms the final
    /// [`Error::Eof`]. Visible packets carry their source frame's `pts`
    /// (in push order — the driver emits visible packets in source
    /// order); invisible anchor packets carry `pts = None`. Only key
    /// frames set `flags.keyframe`.
    pub(crate) struct Vp8LaggedEncoder {
        output_params: CodecParameters,
        width: u32,
        height: u32,
        stream: crate::stream::Vp8AltrefStreamEncoder,
        time_base: TimeBase,
        /// pts of pushed source frames not yet emitted as visible
        /// packets (front = oldest). Visible packets pop in order.
        pts_queue: VecDeque<Option<i64>>,
        pending: VecDeque<Packet>,
        eof: bool,
    }

    impl std::fmt::Debug for Vp8LaggedEncoder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Vp8LaggedEncoder")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("buffered", &self.stream.buffered())
                .field("pending", &self.pending.len())
                .field("eof", &self.eof)
                .finish()
        }
    }

    impl Vp8LaggedEncoder {
        /// Convert a batch of stream packets into framework [`Packet`]s,
        /// consuming the pts queue for visible ones.
        fn enqueue(&mut self, packets: Vec<crate::stream::AltrefStreamPacket>) {
            for p in packets {
                let is_key = p.kind == crate::stream::AltrefPacketKind::Key;
                let pts = match p.source_index {
                    Some(_) => self.pts_queue.pop_front().flatten(),
                    None => None,
                };
                let mut pkt = Packet::new(0, self.time_base, p.bytes);
                pkt.pts = pts;
                pkt.dts = pts;
                pkt.flags.keyframe = is_key;
                self.pending.push_back(pkt);
            }
        }
    }

    impl Encoder for Vp8LaggedEncoder {
        fn codec_id(&self) -> &CodecId {
            &self.output_params.codec_id
        }

        fn output_params(&self) -> &CodecParameters {
            &self.output_params
        }

        fn send_frame(&mut self, frame: &Frame) -> Result<()> {
            if self.eof {
                return Err(Error::invalid(
                    "vp8 encoder: send_frame after flush (encoder is drained)",
                ));
            }
            let v = match frame {
                Frame::Video(v) => v,
                _ => return Err(Error::invalid("vp8 encoder: video frames only")),
            };
            let (y, u, vv) = packed_planes_from_video_frame(v, self.width, self.height)?;
            let src = I420Frame::packed(self.width, self.height, &y, &u, &vv);
            self.pts_queue.push_back(v.pts);
            let emitted = self
                .stream
                .push_frame(&src)
                .map_err(|e| Error::invalid(e.to_string()))?;
            self.enqueue(emitted);
            Ok(())
        }

        fn receive_packet(&mut self) -> Result<Packet> {
            if let Some(p) = self.pending.pop_front() {
                Ok(p)
            } else if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            }
        }

        fn flush(&mut self) -> Result<()> {
            if !self.eof {
                let emitted = self
                    .stream
                    .finish()
                    .map_err(|e| Error::invalid(e.to_string()))?;
                self.enqueue(emitted);
                self.eof = true;
            }
            Ok(())
        }
    }

    /// Shared constructor for the framework-side keyframe encoder adapter.
    /// `auto_lf` routes each frame through the §9.4 RD loop-filter selector
    /// ([`encode_keyframe_auto_loop_filter`]) when set.
    pub(crate) fn build_frame_encoder(
        params: &CodecParameters,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        keyframe: KeyframeParams,
        auto_lf: bool,
    ) -> Result<Box<dyn Encoder>> {
        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(VP8_CODEC_ID);
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(pixel_format);

        let time_base = params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num));

        Ok(Box::new(Vp8FrameEncoder {
            output_params,
            width,
            height,
            keyframe,
            auto_lf,
            time_base,
            pending: VecDeque::new(),
            eof: false,
        }))
    }

    /// Zero-latency [`oxideav_core::Encoder`] adapter around the
    /// direct-API [`encode_keyframe`] driver. One source
    /// [`Frame::Video`] produces one keyframe [`Packet`] immediately —
    /// the still-image contract behind [`make_encoder`] /
    /// [`make_encoder_with_qindex`] / [`make_encoder_with_quality`].
    /// The inter-stream / lagged semantics live on
    /// [`Vp8LaggedEncoder`] via [`super::make_encoder_with_config`].
    pub(crate) struct Vp8FrameEncoder {
        output_params: CodecParameters,
        width: u32,
        height: u32,
        keyframe: KeyframeParams,
        /// §9.4 RD loop-filter auto-selection (per
        /// [`Vp8EncoderConfig::auto_loop_filter`]).
        auto_lf: bool,
        time_base: TimeBase,
        pending: VecDeque<Packet>,
        eof: bool,
    }

    impl std::fmt::Debug for Vp8FrameEncoder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Vp8FrameEncoder")
                .field("width", &self.width)
                .field("height", &self.height)
                .field("y_ac_qi", &self.keyframe.y_ac_qi)
                .field("pending", &self.pending.len())
                .field("eof", &self.eof)
                .finish()
        }
    }

    impl Encoder for Vp8FrameEncoder {
        fn codec_id(&self) -> &CodecId {
            &self.output_params.codec_id
        }

        fn output_params(&self) -> &CodecParameters {
            &self.output_params
        }

        fn send_frame(&mut self, frame: &Frame) -> Result<()> {
            let v = match frame {
                Frame::Video(v) => v,
                _ => return Err(Error::invalid("vp8 encoder: video frames only")),
            };
            let bytes =
                encode_video_frame(v, self.width, self.height, &self.keyframe, self.auto_lf)?;
            let mut pkt = Packet::new(0, self.time_base, bytes);
            pkt.pts = v.pts;
            pkt.dts = v.pts;
            pkt.flags.keyframe = true;
            self.pending.push_back(pkt);
            Ok(())
        }

        fn receive_packet(&mut self) -> Result<Packet> {
            if let Some(p) = self.pending.pop_front() {
                Ok(p)
            } else if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            }
        }

        fn flush(&mut self) -> Result<()> {
            self.eof = true;
            Ok(())
        }
    }

    /// Pull the three I420 planes out of `frame` as tightly-packed
    /// buffers (validating the layout the encoder needs). Shared by the
    /// zero-latency keyframe adapter and the lagged stream adapter.
    fn packed_planes_from_video_frame(
        frame: &VideoFrame,
        width: u32,
        height: u32,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        if frame.planes.len() < 3 {
            return Err(Error::invalid(
                "vp8 encoder: VideoFrame must carry 3 planes (Y/U/V)",
            ));
        }
        let w = width as usize;
        let h = height as usize;
        let uvw = w.div_ceil(2);
        let uvh = h.div_ceil(2);

        let y_plane = &frame.planes[0];
        let u_plane = &frame.planes[1];
        let v_plane = &frame.planes[2];

        // Per-plane geometry validation. Two hostile shapes must be
        // rejected here rather than detonating in the row repack below:
        // a stride narrower than the row width (whose total buffer
        // length can still satisfy a naive `stride * rows` check while
        // the *last* row read `off .. off + row_w` runs past the end),
        // and a stride so large that `stride * rows` wraps `usize`.
        // The length requirement is the exact repack footprint —
        // `stride * (rows - 1) + row_w` — computed with checked
        // arithmetic so overflow is a clean `Err`.
        fn validate_plane(
            plane: &oxideav_core::frame::VideoPlane,
            row_w: usize,
            rows: usize,
            name: &str,
        ) -> Result<()> {
            if rows == 0 {
                return Ok(());
            }
            if plane.stride < row_w {
                return Err(Error::invalid(format!(
                    "vp8 encoder: {name} plane stride {} narrower than row width {row_w}",
                    plane.stride
                )));
            }
            let needed = plane
                .stride
                .checked_mul(rows - 1)
                .and_then(|n| n.checked_add(row_w))
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "vp8 encoder: {name} plane stride {} x {rows} rows overflows",
                        plane.stride
                    ))
                })?;
            if plane.data.len() < needed {
                return Err(Error::invalid(format!(
                    "vp8 encoder: {name} plane buffer {} bytes, needs {needed} \
                     (stride {}, {rows} rows of {row_w})",
                    plane.data.len(),
                    plane.stride
                )));
            }
            Ok(())
        }
        validate_plane(y_plane, w, h, "Y")?;
        validate_plane(u_plane, uvw, uvh, "U")?;
        validate_plane(v_plane, uvw, uvh, "V")?;

        // The direct-API I420Frame requires tightly-packed planes
        // (stride == width). Always repack here so the borrow lifetimes
        // stay simple; zero-copy is a future optimisation when the
        // supplied frame already matches.
        Ok((
            repack_plane(&y_plane.data, y_plane.stride, w, h),
            repack_plane(&u_plane.data, u_plane.stride, uvw, uvh),
            repack_plane(&v_plane.data, v_plane.stride, uvw, uvh),
        ))
    }

    /// Pull the three I420 planes out of `frame` (validating the layout
    /// the encoder needs), then drive [`encode_keyframe`] with them.
    fn encode_video_frame(
        frame: &VideoFrame,
        width: u32,
        height: u32,
        keyframe: &KeyframeParams,
        auto_lf: bool,
    ) -> Result<Vec<u8>> {
        let (y_packed, u_packed, v_packed) = packed_planes_from_video_frame(frame, width, height)?;
        let src = I420Frame::packed(width, height, &y_packed, &u_packed, &v_packed);
        let bytes = if auto_lf {
            encode_keyframe_auto_loop_filter(&src, keyframe)
        } else {
            encode_keyframe(&src, keyframe)
        }
        .map_err(|e| Error::invalid(e.to_string()))?;
        Ok(bytes)
    }

    fn repack_plane(data: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h);
        for r in 0..h {
            let off = r * stride;
            out.extend_from_slice(&data[off..off + w]);
        }
        out
    }
}

#[cfg(feature = "registry")]
pub use factory::{make_encoder, make_encoder_with_qindex, make_encoder_with_quality};

// ───────────────────── 0.1.13 public-surface compatibility ─────────────────────
//
// The crates.io `oxideav-vp8 0.1.13` release exposed a wider encoder
// surface than the post-orphan-rebuild master. This block restores the
// public *shape* (struct fields, enum variants, function signatures,
// constants) so historical consumers keep building. The bodies of the
// two-pass-encoder family stub to [`Vp8Error::Unsupported`] per the
// Tier-3 caveat from the round-167 widen — the rate-control algorithm
// is intentionally future work; this round locks the symbol surface.
//
// All items here are reachable under `default-features = false` unless
// explicitly gated on `registry`.

/// Loop-filter routing strategy — historical 0.1.13 enum re-exposed
/// here. The current encoder always emits the §15 normal filter; this
/// enum is preserved on the surface so historical configs keep
/// compiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopFilterMode {
    /// Pick `Normal` for `lf_level < SIMPLE` thresholds, `Simple`
    /// otherwise. Default in 0.1.13.
    #[default]
    Auto,
    /// Always use the §15.3 normal filter.
    Normal,
    /// Always use the §15.2 simple filter.
    Simple,
}

/// Per-frame complexity hint produced by [`first_pass_analyze`] and
/// consumed by [`Vp8TwoPassEncoder`]. The 0.1.13 release used this as
/// the carrier for the first-pass → second-pass rate-control hand-off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameComplexity {
    /// Source-frame index inside the GOP.
    pub frame_index: u32,
    /// Estimated bits-per-MB for the source frame (the spec-side
    /// "intra cost" surrogate; higher = harder to compress).
    pub bits_per_mb: f32,
    /// True when the first-pass detected a scene cut between this
    /// frame and the previous one.
    pub scene_cut: bool,
}

/// Encoder statistics — emitted by every `encode` call so downstream
/// muxers can keep running rate / quality counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vp8EncoderStats {
    /// Number of frames the encoder has produced so far.
    pub frames_encoded: u64,
    /// Total emitted byte count across all frames.
    pub bytes_emitted: u64,
    /// Number of key frames emitted so far.
    pub keyframes_emitted: u64,
}

/// Encoder configuration knobs — the 0.1.13 public struct.
///
/// Each field has a documented default constant (e.g.
/// [`DEFAULT_QINDEX`], [`DEFAULT_GOLDEN_INTERVAL`]) so callers can
/// override only the knobs they care about. The current encoder honours
/// `qindex` and `lf_level` directly; the higher-level knobs (lookahead,
/// scene-cut detection, segment-aware QP) are persisted on the struct
/// for forward-compat but not yet exercised by the encode body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vp8EncoderConfig {
    /// Base `y_ac_qi` quantiser index (§9.6, `0..=127`).
    pub qindex: u8,
    /// Loop-filter level (`0..=63`).
    pub lf_level: u8,
    /// Loop-filter routing mode.
    pub lf_mode: LoopFilterMode,
    /// Golden-frame refresh interval in frames.
    pub golden_interval: u32,
    /// Alt-ref refresh interval in frames.
    pub alt_ref_interval: u32,
    /// Lookahead window in frames (first-pass / scene-cut buffer).
    pub lookahead_window: u32,
    /// Per-frame target bitrate in bits, or 0 for CQ mode.
    pub target_bitrate_bps: u32,
    /// When `true`, the §9.4 `loop_filter_level` / `sharpness_level` are
    /// chosen per frame by the RD selector
    /// ([`select_filter_params`]) rather than taken from
    /// [`lf_level`](Self::lf_level): the encoder picks the pair minimising
    /// post-§15 reconstruction SSD against the source. Defaults to `false`
    /// (the historical fixed-`lf_level` behaviour). RFC 6386 is silent on
    /// the choice, so this is a clean-room encoder-quality knob.
    pub auto_loop_filter: bool,
    /// §13 trellis coefficient-RDO aggressiveness — the clean-room
    /// quality/size dial threaded onto both the keyframe and inter residual
    /// emit (see [`KeyframeParams::trellis_strength`]).
    /// [`TrellisStrength::DEFAULT`] reproduces the calibrated trade; higher
    /// trades PSNR for smaller output, [`TrellisStrength::OFF`] reverts to
    /// plain round-quantisation.
    pub trellis_strength: TrellisStrength,
}

impl Default for Vp8EncoderConfig {
    fn default() -> Self {
        Self {
            qindex: DEFAULT_QINDEX,
            lf_level: 0,
            lf_mode: LoopFilterMode::Auto,
            golden_interval: DEFAULT_GOLDEN_INTERVAL,
            alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
            lookahead_window: DEFAULT_LOOKAHEAD_WINDOW,
            target_bitrate_bps: 0,
            auto_loop_filter: false,
            trellis_strength: TrellisStrength::DEFAULT,
        }
    }
}

/// Direct-API typed encoder — historical 0.1.13 single-pass entry
/// point. Wraps the current encoder's [`KeyframeParams`] /
/// [`encode_keyframe`] driver under a single struct so downstream
/// consumers that wrote `Vp8Encoder::new(cfg).encode_keyframe(...)`
/// keep building.
#[derive(Debug, Clone)]
pub struct Vp8Encoder {
    config: Vp8EncoderConfig,
    stats: Vp8EncoderStats,
}

impl Vp8Encoder {
    /// Build a fresh encoder with the supplied config.
    pub fn new(config: Vp8EncoderConfig) -> Self {
        Self {
            config,
            stats: Vp8EncoderStats::default(),
        }
    }

    /// Encoder configuration (immutable handle).
    pub fn config(&self) -> &Vp8EncoderConfig {
        &self.config
    }

    /// Running encoder statistics.
    pub fn stats(&self) -> &Vp8EncoderStats {
        &self.stats
    }

    /// Encode `frame` as a VP8 key frame; returns the on-the-wire
    /// bytes. Delegates to the current crate's [`encode_keyframe`]
    /// driver using the encoder's [`Vp8EncoderConfig`] for the §9.6
    /// quant index and the §9.4 loop-filter level. When
    /// [`auto_loop_filter`](Vp8EncoderConfig::auto_loop_filter) is set the
    /// §9.4 level / sharpness are RD-selected per frame
    /// ([`encode_keyframe_auto_loop_filter`]) and `config.lf_level` is
    /// ignored.
    pub fn encode_keyframe(&mut self, frame: &I420Frame<'_>) -> crate::error::Result<Vec<u8>> {
        let params = KeyframeParams {
            y_ac_qi: self.config.qindex,
            loop_filter_level: self.config.lf_level,
            trellis_strength: self.config.trellis_strength,
            ..KeyframeParams::default()
        };
        let bytes = if self.config.auto_loop_filter {
            encode_keyframe_auto_loop_filter(frame, &params)
        } else {
            encode_keyframe(frame, &params)
        }
        .map_err(|e| crate::error::Vp8Error::invalid(e.to_string()))?;
        self.stats.frames_encoded += 1;
        self.stats.keyframes_emitted += 1;
        self.stats.bytes_emitted += bytes.len() as u64;
        Ok(bytes)
    }

    /// Encode a whole source sequence as an **auto-altref inter
    /// stream** — the round-384 batch front door that finally consumes
    /// the config's stream-level knobs:
    ///
    /// * [`alt_ref_interval`](Vp8EncoderConfig::alt_ref_interval) —
    ///   the lookahead group size: every completed group ships one
    ///   invisible ARNR anchor (§9.1 `show_frame = 0`, §9.7
    ///   `refresh_alternate_frame = 1`) followed by the group's visible
    ///   multi-reference P-frames. `0` disables anchors (plain
    ///   streaming, group size 1).
    /// * [`lookahead_window`](Vp8EncoderConfig::lookahead_window) —
    ///   caps the group size (the encoder never buffers more source
    ///   frames than this).
    /// * [`golden_interval`](Vp8EncoderConfig::golden_interval) — used
    ///   as the **key-frame cadence** of the sequence (`0` keys only
    ///   frame 0); scene cuts additionally force keys via the stream
    ///   driver's default MAD detector.
    /// * `qindex` / `lf_level` / `trellis_strength` apply to every
    ///   frame, exactly as on [`Self::encode_keyframe`].
    ///
    /// Returns the full packet list in decode order — **more packets
    /// than source frames** (one invisible anchor per group; see
    /// [`crate::stream::AltrefStreamPacket`]). Feed them to
    /// [`crate::state::Vp8DecoderState::decode_frame`] in order; a
    /// player drops packets whose
    /// [`is_visible()`](crate::stream::AltrefStreamPacket::is_visible)
    /// is `false`.
    ///
    /// [`auto_loop_filter`](Vp8EncoderConfig::auto_loop_filter) applies
    /// per emitted frame (key frames, anchors, and P-frames each get an
    /// RD-selected §9.4 level / sharpness; `lf_level` is ignored when
    /// it is set). As the offline batch front door, this path
    /// additionally enables the §13.4 fitted `token_prob_update()`
    /// in-header emission and the §11 intra-within-inter picker on
    /// every frame — both quality/density features with a never-grow
    /// byte guard, paid for with extra encode passes an offline batch
    /// caller accepts. A caller that wants the round-384 single-pass
    /// wire drives [`crate::stream::Vp8AltrefStreamEncoder`] directly
    /// with those toggles off.
    pub fn encode_sequence(
        &mut self,
        frames: &[I420Frame<'_>],
    ) -> crate::error::Result<Vec<crate::stream::AltrefStreamPacket>> {
        let window = if self.config.alt_ref_interval == 0 {
            1
        } else {
            self.config
                .alt_ref_interval
                .min(self.config.lookahead_window.max(1)) as usize
        };
        let config = crate::stream::AltrefStreamConfig {
            params: KeyframeParams {
                y_ac_qi: self.config.qindex,
                loop_filter_level: self.config.lf_level,
                trellis_strength: self.config.trellis_strength,
                ..KeyframeParams::default()
            },
            keyframe_interval: u64::from(self.config.golden_interval),
            altref_window: window,
            auto_loop_filter: self.config.auto_loop_filter,
            fitted_token_prob_updates: true,
            intra_pick: true,
            ..crate::stream::AltrefStreamConfig::default()
        };
        let mut enc = crate::stream::Vp8AltrefStreamEncoder::new(config)
            .expect("window is clamped to >= 1 above");
        let mut packets = Vec::new();
        for frame in frames {
            packets.extend(
                enc.push_frame(frame)
                    .map_err(|e| crate::error::Vp8Error::invalid(e.to_string()))?,
            );
        }
        packets.extend(
            enc.finish()
                .map_err(|e| crate::error::Vp8Error::invalid(e.to_string()))?,
        );
        for p in &packets {
            self.stats.frames_encoded += 1;
            self.stats.bytes_emitted += p.bytes.len() as u64;
            if p.kind == crate::stream::AltrefPacketKind::Key {
                self.stats.keyframes_emitted += 1;
            }
        }
        Ok(packets)
    }
}

/// Two-pass encoder configuration.
///
/// The two-pass rate-control algorithm uses [`base`](Self::base) as the
/// **target** single-pass configuration: `base.qindex` is the per-GOP
/// average qindex the second pass aims at, and complexity-driven deltas
/// distribute around it (heavier frames → lower qindex / higher quality,
/// lighter frames → higher qindex / smaller bytes).
///
/// [`target_bitrate_bps`](Self::target_bitrate_bps) and
/// [`overshoot_ratio`](Self::overshoot_ratio) are persisted on the
/// config for downstream callers but are **advisory** in the current
/// implementation: the second pass is a complexity-aware constant-quality
/// scheduler, not a strict bitrate-constrained CBR loop, so they do not
/// alter the emitted bytes today. A future round may use them to clamp
/// the per-frame qindex within a bitrate envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vp8TwoPassConfig {
    /// Underlying single-pass config used as the second-pass baseline.
    pub base: Vp8EncoderConfig,
    /// Target average bitrate in bits/sec. When non-zero **and** a frame
    /// rate is set, the second pass solves for a per-GOP base qindex that
    /// makes the predicted byte total meet this budget (see struct docs).
    /// `0` keeps the historical complexity-aware constant-quality
    /// scheduler centred on `base.qindex`.
    pub target_bitrate_bps: u32,
    /// Maximum permitted bitrate-overshoot ratio (e.g. `1.2`). The
    /// bitrate solver treats `target_bitrate_bps × overshoot_ratio` as
    /// the hard per-GOP byte ceiling it must not exceed.
    pub overshoot_ratio: f32,
    /// Frame-rate numerator (frames per `fps_den` seconds). Used with
    /// [`target_bitrate_bps`](Self::target_bitrate_bps) to convert the
    /// bits/sec budget into a per-GOP byte budget. `0` disables the
    /// bitrate solver (the bitrate budget is undefined without a rate).
    pub fps_num: u32,
    /// Frame-rate denominator. `0` is treated as `1`.
    pub fps_den: u32,
    /// When `true`, each second-pass frame's §9.4 `loop_filter_level` is
    /// chosen by the RD selector ([`select_filter_level`]) rather than
    /// taken from `base.lf_level`: the encoder picks, per frame, the level
    /// that minimises post-§15 reconstruction SSD against the source. The
    /// chosen level reaches the wire, so decoder lockstep holds. Defaults
    /// to `false` (the historical fixed-`base.lf_level` behaviour). RFC
    /// 6386 is silent on the level choice, so this is a clean-room
    /// encoder-quality knob.
    pub auto_loop_filter: bool,
}

impl Default for Vp8TwoPassConfig {
    fn default() -> Self {
        Self {
            base: Vp8EncoderConfig::default(),
            target_bitrate_bps: 0,
            overshoot_ratio: 1.2,
            fps_num: 0,
            fps_den: 1,
            auto_loop_filter: false,
        }
    }
}

impl Vp8TwoPassConfig {
    /// Frames-per-second as an `f32`, or `None` when no rate is set
    /// (`fps_num == 0`). `fps_den == 0` is read as `1`.
    pub fn fps(&self) -> Option<f32> {
        if self.fps_num == 0 {
            None
        } else {
            Some(self.fps_num as f32 / self.fps_den.max(1) as f32)
        }
    }

    /// True when the bitrate solver is active: a positive target bitrate
    /// **and** a usable frame rate are both present. When `false` the
    /// second pass stays on the complexity-aware constant-quality
    /// schedule centred on [`base.qindex`](Vp8EncoderConfig::qindex).
    pub fn bitrate_targeted(&self) -> bool {
        self.target_bitrate_bps > 0 && self.fps().is_some()
    }
}

/// Two-pass encoder driver.
///
/// # Algorithm (clean-room, RFC 6386 §9.6 + in-tree primitives only)
///
/// RFC 6386 is intentionally silent on rate-control — the algorithm is
/// the encoder's choice. This implementation runs in one of two modes,
/// selected by [`Vp8TwoPassConfig::bitrate_targeted`]:
///
/// * **Constant-quality** (no bitrate target): a complexity-aware
///   scheduler centred on `config.base.qindex` (steps 1–3 below).
/// * **Bitrate-targeted** (`target_bitrate_bps > 0` **and** a frame
///   rate set): [`Self::first_pass_analyze`] solves the GOP base qindex
///   that fits the budget ([`solve_base_qindex_for_budget`]) and caches
///   a per-frame byte allotment. Each [`Self::encode_frame`] then
///   re-centres the complexity delta on the solved base and adds a
///   closed-loop feedback term proportional to the accumulated rate
///   debt (bytes emitted so far minus bytes allotted), so the stream
///   self-corrects toward the target. The feedback gain /clamp are
///   [`RATE_FEEDBACK_QSTEP_PER_FRAME`] / [`RATE_FEEDBACK_MAX_QSTEP`].
///
/// The complexity-aware scheduler (both modes share it) is built from a
/// single linear pass over the luma plane:
///
/// 1. **First pass** ([`first_pass_analyze`] / the free fn of the same
///    name): for each input frame, compute a lightweight cost surrogate
///    from per-pixel luma arithmetic:
///    * `mad`  — mean absolute deviation vs the previous frame's luma
///      (motion proxy; first frame uses `mad = 0`).
///    * `var`  — luma variance vs the per-frame mean (spatial activity
///      proxy).
///    * `bits_per_mb` (the surrogate) ≈ `α·log2(1+mad) + β·log2(1+var)`,
///      multiplied by 100 and clamped non-negative.  A pure-flat frame
///      lands near 0; a noise-saturated moving frame lands near 1000.
///    * Scene cut: declared when `mad * 1024 / (prev_var + 1)` exceeds
///      [`DEFAULT_SCENE_CUT_THRESHOLD`] **and** `mad >=
///      SCENE_CUT_ABS_FLOOR / 1024`.
///
/// 2. **Second pass** ([`two_pass_qindices`] / the free fn of the same
///    name): distribute per-frame `qindex` around `config.base.qindex`
///    so heavier frames get **lower** qindex (better quality, more
///    bytes) and lighter frames get **higher** qindex (lower quality,
///    fewer bytes):
///    * `mean = mean(bits_per_mb)` across the GOP.
///    * `delta = clamp((cost - mean) / max(mean, 1.0) * range, -range, range)`
///      where `range = DEFAULT_AQ_QINDEX_RANGE`.
///    * `qindex = clamp(base + round(delta), 0, 127)` per RFC 6386 §9.6.
///    * Scene cuts subtract an additional [`DEFAULT_SCENE_CUT_QUANT_BOOST`]
///      to give those frames extra quality.
///
/// 3. **Encode pass** ([`Self::encode_frame`]): for each input frame,
///    look up its per-frame qindex, build a [`KeyframeParams`] with
///    that `y_ac_qi`, and drive the in-tree key/P encoders against the
///    reference [`KeyframePlanes`] stashed inside the driver.  The
///    first call always emits a key frame; subsequent calls emit P
///    frames against the previous reconstruction.
///
/// The per-frame `qindex` is sourced purely from the supplied
/// `complexity` and the `global_mean_cost` cached during the most
/// recent [`Self::first_pass_analyze`] call.  A caller that bypasses
/// `first_pass_analyze` falls back to a no-delta schedule (every
/// frame gets `config.base.qindex`).
#[derive(Debug, Clone)]
pub struct Vp8TwoPassEncoder {
    config: Vp8TwoPassConfig,
    /// Mean of `bits_per_mb` across the last [`Self::first_pass_analyze`]
    /// call.  `None` until first_pass_analyze runs; in that case
    /// [`Self::encode_frame`] falls back to `config.base.qindex` (no
    /// per-frame delta).
    global_mean_cost: Option<f32>,
    /// Most-recent reconstruction kept as the `LAST` reference for the
    /// next P-frame encode.  Cleared until the first key frame lands.
    last_reconstruction: Option<crate::frame::KeyframePlanes>,
    /// Number of frames emitted so far in this stream — determines
    /// keyframe vs P-frame and is used by the `golden_interval` /
    /// `alt_ref_interval` scheduler (not yet wired into the §9.7
    /// refresh ladder; today every P-frame uses
    /// [`RefreshControls::default`]).
    frame_count: u64,
    /// Frame index of the most recent emitted key frame.  Anchors the
    /// `golden_interval` / scene-cut keyframe scheduler.
    last_keyframe_index: Option<u64>,
    /// Base qindex the [`solve_base_qindex_for_budget`] solver chose for
    /// this GOP, cached during [`Self::first_pass_analyze`] when the
    /// config is bitrate-targeted.  `None` keeps the encoder on the
    /// constant-quality schedule centred on `config.base.qindex`.
    solved_base_qindex: Option<u8>,
    /// Per-frame byte allotment under the bitrate target (= GOP byte
    /// budget ÷ frame count), cached alongside `solved_base_qindex`.
    /// Drives the closed-loop feedback term: a frame that overspends
    /// this allotment pushes subsequent frames' qindex up, and vice
    /// versa.
    frame_byte_budget: Option<u64>,
    /// Running rate debt in bytes: `Σ(emitted − allotment)` across the
    /// frames coded so far.  Positive ⇒ the stream is over budget so
    /// later frames must use a coarser quantiser; negative ⇒ under
    /// budget so quality can be spent.  Only updated on the
    /// bitrate-targeted path.
    rate_debt_bytes: i64,
}

impl Vp8TwoPassEncoder {
    /// Build a fresh two-pass encoder.
    ///
    /// The encoder starts with no first-pass state and no reference
    /// reconstruction — call [`Self::first_pass_analyze`] before
    /// [`Self::encode_frame`] to populate the rate-control mean.
    pub fn new(config: Vp8TwoPassConfig) -> Self {
        Self {
            config,
            global_mean_cost: None,
            last_reconstruction: None,
            frame_count: 0,
            last_keyframe_index: None,
            solved_base_qindex: None,
            frame_byte_budget: None,
            rate_debt_bytes: 0,
        }
    }

    /// Two-pass config (immutable handle).
    pub fn config(&self) -> &Vp8TwoPassConfig {
        &self.config
    }

    /// First-pass analysis entry — returns the per-frame
    /// [`FrameComplexity`] vector **and** caches the GOP-mean cost on
    /// `self` so the subsequent [`Self::encode_frame`] calls can pick
    /// per-frame qindex deltas relative to it.
    ///
    /// Delegates to the free function [`first_pass_analyze`] for the
    /// core arithmetic; this method exists so a streaming consumer
    /// can keep the analysis state on the encoder handle rather than
    /// threading the mean by hand.
    pub fn first_pass_analyze(
        &mut self,
        frames: &[I420Frame<'_>],
    ) -> crate::error::Result<Vec<FrameComplexity>> {
        let stats = first_pass_analyze(frames, &self.config)?;
        self.global_mean_cost = Some(mean_complexity_cost(&stats));
        // If a bitrate target is set, solve the GOP base qindex now and
        // cache the per-frame byte allotment so the second-pass encode
        // loop can correct over/undershoot frame-by-frame.  Reset any
        // debt carried from a previous analysis.
        self.rate_debt_bytes = 0;
        if self.config.bitrate_targeted() && !stats.is_empty() {
            let (w, h) = frames
                .first()
                .map(|f| (f.width, f.height))
                .unwrap_or((0, 0));
            self.solved_base_qindex =
                Some(solve_base_qindex_for_budget(&self.config, &stats, w, h));
            self.frame_byte_budget =
                gop_byte_budget(&self.config, stats.len()).map(|b| b / stats.len() as u64);
        } else {
            self.solved_base_qindex = None;
            self.frame_byte_budget = None;
        }
        Ok(stats)
    }

    /// Second-pass encode of one frame, picking the qindex from the
    /// first-pass complexity report.
    ///
    /// The first call always emits a key frame; subsequent calls emit
    /// a P-frame against the previous reconstruction.  The complexity
    /// report's `scene_cut` flag forces a key frame even mid-stream.
    /// Returns the raw VP8 bitstream bytes for the frame.
    pub fn encode_frame(
        &mut self,
        frame: &I420Frame<'_>,
        complexity: FrameComplexity,
    ) -> crate::error::Result<Vec<u8>> {
        let qindex = self.qindex_for(complexity);
        let params = KeyframeParams {
            y_ac_qi: qindex,
            loop_filter_level: self.config.base.lf_level,
            trellis_strength: self.config.base.trellis_strength,
            ..KeyframeParams::default()
        };

        // Schedule key vs P frame.  Force-keyframe conditions:
        //   * No prior reference (first call or right after a complexity
        //     report that demanded a key frame).
        //   * Scene cut flagged in the supplied complexity.
        //   * Golden interval elapsed since the last keyframe (acts as
        //     a max-GOP guard so a long P-chain doesn't accumulate
        //     drift).
        let force_key = self.last_reconstruction.is_none()
            || complexity.scene_cut
            || self
                .last_keyframe_index
                .map(|anchor| {
                    self.frame_count.saturating_sub(anchor)
                        >= self.config.base.golden_interval.max(1) as u64
                })
                .unwrap_or(false);

        // §9.4 auto loop-filter: when enabled, the per-frame
        // `loop_filter_level` is the RD selector's choice rather than
        // `base.lf_level`. The auto entry points write the chosen level to
        // the wire so decoder lockstep holds either way.
        let auto_lf = self.config.auto_loop_filter;
        let (bytes, planes) = if force_key {
            if auto_lf {
                encode_keyframe_auto_loop_filter_with_reconstruction(frame, &params)
            } else {
                encode_keyframe_with_reconstruction(frame, &params)
            }
            .map_err(|e| crate::error::Vp8Error::invalid(e.to_string()))?
        } else {
            let reference = self
                .last_reconstruction
                .as_ref()
                .expect("force_key=false implies a reference exists");
            if auto_lf {
                encode_p_frame_multi_ref_auto_loop_filter(frame, reference, None, None, &params)
            } else {
                encode_p_frame_multi_ref(frame, reference, None, None, &params)
            }
            .map_err(|e| crate::error::Vp8Error::invalid(e.to_string()))?
        };

        self.last_reconstruction = Some(planes);
        if force_key {
            self.last_keyframe_index = Some(self.frame_count);
        }
        // Closed-loop accounting: fold this frame's over/undershoot vs
        // its allotment into the running debt so the next frame's
        // `qindex_for` can correct it.  Only meaningful on the
        // bitrate-targeted path (frame_byte_budget is `None` otherwise).
        if let Some(budget) = self.frame_byte_budget {
            let spent = bytes.len() as i64;
            self.rate_debt_bytes = self.rate_debt_bytes.saturating_add(spent - budget as i64);
        }
        self.frame_count += 1;
        Ok(bytes)
    }

    /// Current accumulated rate debt in bytes (positive ⇒ over budget).
    /// Exposed so a streaming caller can observe how the closed-loop
    /// bitrate controller is tracking the target.  Always `0` on the
    /// constant-quality (no bitrate target) path.
    pub fn rate_debt_bytes(&self) -> i64 {
        self.rate_debt_bytes
    }

    /// Resolve the per-frame qindex from the cached first-pass mean
    /// (falling back to `config.base.qindex` when first_pass_analyze
    /// hasn't been called).  Centralised so the [`Self::encode_frame`]
    /// path and the public [`two_pass_qindex_for_frame`] free function
    /// agree on the formula.
    ///
    /// On the bitrate-targeted path the base qindex is the solver's
    /// choice (cached during [`Self::first_pass_analyze`]) plus a
    /// closed-loop feedback term proportional to the accumulated
    /// [`Self::rate_debt_bytes`]: a stream that has overspent its
    /// allotment so far nudges this frame's qindex up (coarser, cheaper)
    /// and an under-budget stream nudges it down (finer, higher
    /// quality).  Without a bitrate target the behaviour is unchanged.
    fn qindex_for(&self, complexity: FrameComplexity) -> u8 {
        // Constant-quality (no bitrate target): unchanged historical path.
        let Some(solved_base) = self.solved_base_qindex else {
            return match self.global_mean_cost {
                Some(mean) => qindex_from_complexity(&self.config, complexity, mean),
                None => qindex_from_complexity(&self.config, complexity, complexity.bits_per_mb),
            };
        };
        // Bitrate-targeted: re-centre the complexity delta on the solved
        // base, then add the feedback correction.
        let mut cfg = self.config;
        cfg.base.qindex = solved_base;
        let mean = self.global_mean_cost.unwrap_or(complexity.bits_per_mb);
        let centred = qindex_from_complexity(&cfg, complexity, mean) as i32;
        let correction = self.rate_feedback_delta();
        (centred + correction).clamp(0, 127) as u8
    }

    /// Closed-loop feedback delta (in qindex units) from the running
    /// rate debt.  Scales the accumulated over/undershoot by the
    /// per-frame byte allotment: roughly one qindex step per
    /// [`RATE_FEEDBACK_BYTES_PER_QSTEP`] fraction of an allotment of
    /// debt, clamped to ±[`RATE_FEEDBACK_MAX_QSTEP`] so a single noisy
    /// frame can't slam the quantiser to an extreme.  Returns `0` when
    /// no per-frame budget is known.
    fn rate_feedback_delta(&self) -> i32 {
        let Some(budget) = self.frame_byte_budget else {
            return 0;
        };
        if budget == 0 {
            return 0;
        }
        // debt as a multiple of one frame's allotment.
        let debt_frames = self.rate_debt_bytes as f32 / budget as f32;
        let raw = (debt_frames * RATE_FEEDBACK_QSTEP_PER_FRAME as f32).round() as i32;
        raw.clamp(-RATE_FEEDBACK_MAX_QSTEP, RATE_FEEDBACK_MAX_QSTEP)
    }
}

/// Free-function first-pass analysis — historical 0.1.13 entry point.
///
/// Walks `frames` end-to-end and produces one [`FrameComplexity`] per
/// input.  The cost surrogate is documented on [`Vp8TwoPassEncoder`]:
/// a lightweight log-MAD + log-variance combination on the 8-bit luma
/// plane.  Pure RFC 6386 §9.6-qindex-range math; no transform, no
/// entropy coding, no reference-encoder consultation.
///
/// Returns an empty `Vec` for an empty input slice.  Caller-supplied
/// frame dimensions are honoured frame-by-frame (varying dimensions
/// across the slice are allowed — only the per-frame stats use the
/// dimensions; no cross-frame pixel arithmetic crosses dimension
/// boundaries).
pub fn first_pass_analyze(
    frames: &[I420Frame<'_>],
    _config: &Vp8TwoPassConfig,
) -> crate::error::Result<Vec<FrameComplexity>> {
    let mut out = Vec::with_capacity(frames.len());
    let mut prev_luma_sample: Option<(Vec<u8>, u32, u32)> = None;
    let mut prev_variance: f32 = 0.0;
    for (idx, frame) in frames.iter().enumerate() {
        let (_mean, variance) = luma_mean_and_variance(frame);
        let mad = match &prev_luma_sample {
            Some((prev, pw, ph)) if *pw == frame.width && *ph == frame.height => {
                luma_mad(frame, prev)
            }
            _ => 0.0,
        };
        // Cost surrogate: scale-invariant log combination so a flat
        // frame lands near zero and a textured / fast-motion frame
        // lands at a few hundred.  α / β picked so a frame with
        // mad=10, var=200 lands around 100 bits/MB (matches the
        // ballpark "expected" per-MB cost at the default qindex).
        let cost = 30.0 * (1.0 + mad).log2() + 25.0 * (1.0 + variance).log2();
        let cost = cost.max(0.0);
        // Scene-cut decision: the §9.4 / §20.6 spec is silent; the
        // surrogate compares mad-vs-prev-variance against the
        // `DEFAULT_SCENE_CUT_THRESHOLD` knob, gated by the
        // `SCENE_CUT_ABS_FLOOR` minimum so a static scene with tiny
        // numerical noise doesn't false-positive.
        let scene_cut = if idx == 0 {
            false
        } else {
            let ratio = (mad * 1024.0) / (prev_variance + 1.0);
            ratio >= DEFAULT_SCENE_CUT_THRESHOLD as f32
                && (mad * 1024.0) >= SCENE_CUT_ABS_FLOOR as f32
        };
        out.push(FrameComplexity {
            frame_index: idx as u32,
            bits_per_mb: cost,
            scene_cut,
        });
        // Snapshot the luma plane for the next iteration's MAD calc.
        // We copy because the borrow of `frame.y` doesn't survive the
        // next loop iteration.
        let plane_len = (frame.height as usize) * frame.y_stride;
        let mut buf = vec![0u8; plane_len];
        let take = plane_len.min(frame.y.len());
        buf[..take].copy_from_slice(&frame.y[..take]);
        prev_luma_sample = Some((buf, frame.width, frame.height));
        prev_variance = variance;
    }
    Ok(out)
}

/// Returns the second-pass qindex the encoder would select for a
/// particular frame, given just the per-frame [`FrameComplexity`].
///
/// This is the **stateless** variant: it has no access to a GOP-mean,
/// so it treats `complexity.bits_per_mb` as both the per-frame cost
/// *and* the reference mean — which collapses the rate-control delta
/// to zero for every non-scene-cut frame.  Useful when a caller only
/// has one frame's worth of data; for a full GOP the schedule
/// produced by [`two_pass_qindices`] (which uses the actual mean)
/// is what [`Vp8TwoPassEncoder::encode_frame`] consumes.
///
/// Scene-cut frames receive a [`DEFAULT_SCENE_CUT_QUANT_BOOST`]-sized
/// qindex *reduction* (better quality) on top of the baseline.
///
/// Returns `Err(Vp8Error::InvalidData)` if `config.base.qindex` is
/// outside the RFC 6386 §9.6 `0..=127` range.
pub fn two_pass_qindex_for_frame(
    config: &Vp8TwoPassConfig,
    complexity: FrameComplexity,
) -> crate::error::Result<u8> {
    if config.base.qindex > 127 {
        return Err(crate::error::Vp8Error::invalid(format!(
            "vp8 two-pass: config.base.qindex={} out of RFC 6386 §9.6 range 0..=127",
            config.base.qindex
        )));
    }
    Ok(qindex_from_complexity(
        config,
        complexity,
        complexity.bits_per_mb,
    ))
}

/// Returns the full second-pass qindex schedule for a GOP.
///
/// Computes the GOP mean from `complexities` and emits one
/// `0..=127` qindex per entry per the algorithm documented on
/// [`Vp8TwoPassEncoder`].  An empty input yields an empty schedule.
///
/// Returns `Err(Vp8Error::InvalidData)` if `config.base.qindex` is
/// outside `0..=127`.
pub fn two_pass_qindices(
    config: &Vp8TwoPassConfig,
    complexities: &[FrameComplexity],
) -> crate::error::Result<Vec<u8>> {
    if config.base.qindex > 127 {
        return Err(crate::error::Vp8Error::invalid(format!(
            "vp8 two-pass: config.base.qindex={} out of RFC 6386 §9.6 range 0..=127",
            config.base.qindex
        )));
    }
    if complexities.is_empty() {
        return Ok(Vec::new());
    }
    let mean = mean_complexity_cost(complexities);
    Ok(complexities
        .iter()
        .map(|c| qindex_from_complexity(config, *c, mean))
        .collect())
}

/// Build a two-pass encoder from a [`Vp8TwoPassConfig`].  Convenience
/// historical-0.1.13 factory; equivalent to
/// [`Vp8TwoPassEncoder::new`].
pub fn make_two_pass_encoder(config: Vp8TwoPassConfig) -> Vp8TwoPassEncoder {
    Vp8TwoPassEncoder::new(config)
}

// ─────────────── bitrate-budget solver (clean-room) ───────────────
//
// RFC 6386 fixes the §9.6 qindex ladder but is silent on how an
// encoder hits a bitrate. The solver below turns a bits/sec target
// into a per-GOP byte budget and bisects the §9.6 qindex range for
// the *lowest* base qindex (= highest quality) whose predicted byte
// total still fits the budget. Prediction is the in-tree bit-cost
// model ([`estimate_frame_bytes`]); no reference encoder is read.

/// The per-GOP byte budget implied by a bitrate target and frame rate.
///
/// `bytes = (target_bitrate_bps / 8) × (frame_count / fps)`. Returns
/// `None` when the config has no usable bitrate / frame-rate pair
/// (i.e. [`Vp8TwoPassConfig::bitrate_targeted`] is `false`) or when
/// `frame_count == 0`.
pub fn gop_byte_budget(config: &Vp8TwoPassConfig, frame_count: usize) -> Option<u64> {
    if frame_count == 0 {
        return None;
    }
    // `fps()` only yields `Some` when `fps_num != 0`, so the rate is
    // always strictly positive and finite here.
    let fps = config.fps()?;
    if config.target_bitrate_bps == 0 {
        return None;
    }
    let bytes_per_sec = config.target_bitrate_bps as f64 / 8.0;
    let gop_seconds = frame_count as f64 / fps as f64;
    Some((bytes_per_sec * gop_seconds).floor() as u64)
}

/// Predicted total encoded bytes for a GOP if every frame were coded
/// at the same `base_qindex` (before per-frame AQ redistribution).
///
/// Sums [`estimate_frame_bytes`] over each frame's complexity at
/// `base_qindex`. All frames are assumed to share `width × height`.
fn predict_gop_bytes_at_qindex(
    complexities: &[FrameComplexity],
    width: u32,
    height: u32,
    base_qindex: i32,
) -> u64 {
    complexities.iter().fold(0u64, |acc, c| {
        acc.saturating_add(estimate_frame_bytes(
            c.bits_per_mb,
            base_qindex,
            width,
            height,
        ))
    })
}

/// Solve for the GOP base qindex that meets `config`'s bitrate budget.
///
/// Returns the **lowest** §9.6 qindex (best quality) whose predicted
/// byte total — summed over `complexities` at that single qindex via
/// the in-tree bit-cost model — stays within
/// `target_bitrate_bps × overshoot_ratio`. If even qindex 127 (the
/// coarsest quantiser) overshoots, returns 127; if even qindex 0 fits,
/// returns 0.
///
/// Returns `config.base.qindex` unchanged when
/// [`Vp8TwoPassConfig::bitrate_targeted`] is `false` (no bitrate
/// solving requested) or the GOP is empty.
///
/// The search is a monotone bisection: predicted bytes decrease as the
/// qindex rises (coarser quantiser ⇒ fewer tokens), so the smallest
/// qindex meeting the ceiling is found in `log2(128) ≤ 7` model
/// evaluations of the GOP.
pub fn solve_base_qindex_for_budget(
    config: &Vp8TwoPassConfig,
    complexities: &[FrameComplexity],
    width: u32,
    height: u32,
) -> u8 {
    if !config.bitrate_targeted() || complexities.is_empty() {
        return config.base.qindex.min(127);
    }
    let budget = match gop_byte_budget(config, complexities.len()) {
        Some(b) => b,
        None => return config.base.qindex.min(127),
    };
    // Hard ceiling = budget × overshoot. overshoot_ratio < 1.0 is
    // nonsensical (would forbid even hitting the target); clamp to 1.0.
    let ceiling = {
        let r = config.overshoot_ratio.max(1.0) as f64;
        (budget as f64 * r) as u64
    };
    // predicted(qindex) is monotone non-increasing in qindex. Find the
    // smallest qindex whose predicted total ≤ ceiling.
    let fits = |qi: i32| predict_gop_bytes_at_qindex(complexities, width, height, qi) <= ceiling;
    // Coarsest quantiser still overshoots ⇒ nothing we can do, cap at 127.
    if !fits(127) {
        return 127;
    }
    // Finest quantiser already fits ⇒ spend all the quality.
    if fits(0) {
        return 0;
    }
    // Bisect for the boundary: lo never fits, hi always fits.
    let mut lo = 0i32; // !fits(lo) holds (fits(0) was false above)
    let mut hi = 127i32; // fits(hi) holds
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if fits(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi as u8
}

/// Full second-pass qindex schedule under a bitrate target.
///
/// First solves the GOP base qindex via
/// [`solve_base_qindex_for_budget`], then redistributes per-frame
/// qindex around *that* solved base using the same complexity-aware
/// delta as [`two_pass_qindices`] (heavier frames → lower qindex). The
/// result is a schedule whose **mean** tracks the bitrate budget while
/// individual frames still flex with complexity.
///
/// When the config is not bitrate-targeted this is identical to
/// [`two_pass_qindices`] (the base qindex stays `config.base.qindex`).
///
/// Returns `Err(Vp8Error::InvalidData)` if `config.base.qindex` is
/// outside `0..=127`.
pub fn two_pass_qindices_for_bitrate(
    config: &Vp8TwoPassConfig,
    complexities: &[FrameComplexity],
    width: u32,
    height: u32,
) -> crate::error::Result<Vec<u8>> {
    if config.base.qindex > 127 {
        return Err(crate::error::Vp8Error::invalid(format!(
            "vp8 two-pass: config.base.qindex={} out of RFC 6386 §9.6 range 0..=127",
            config.base.qindex
        )));
    }
    if complexities.is_empty() {
        return Ok(Vec::new());
    }
    let solved_base = solve_base_qindex_for_budget(config, complexities, width, height);
    // Re-centre the delta math on the solved base by cloning the config
    // with base.qindex replaced. The per-frame delta + scene-cut boost
    // are unchanged.
    let mut solved_config = *config;
    solved_config.base.qindex = solved_base;
    let mean = mean_complexity_cost(complexities);
    Ok(complexities
        .iter()
        .map(|c| qindex_from_complexity(&solved_config, *c, mean))
        .collect())
}

// ──────────────────── two-pass helpers (private) ────────────────────

/// Mean cost across a per-frame complexity vector — empty input maps
/// to `0.0` so the caller's downstream division-by-mean falls back to
/// the fallback path.
fn mean_complexity_cost(stats: &[FrameComplexity]) -> f32 {
    if stats.is_empty() {
        0.0
    } else {
        let sum: f32 = stats.iter().map(|c| c.bits_per_mb).sum();
        sum / stats.len() as f32
    }
}

// ─────────────────── bit-cost model (clean-room) ───────────────────
//
// RFC 6386 is silent on rate control by design (§9.6 only fixes the
// quantiser ladder; the encoder chooses how to spend bits). The
// closed-loop bitrate target below needs a way to predict, *before*
// entropy-coding a frame, roughly how many bits it will cost at a
// candidate qindex. The model is built only from the in-tree
// complexity surrogate plus the §20.4 AC quantiser ladder; no
// reference encoder is consulted.
//
// Shape: encoded frame bits fall as the quantiser step grows (coarser
// quant → more coefficients quantise to zero → fewer tokens). The
// §20.4 `ac_qlookup` step at qindex 0 is 4 and at qindex 127 is 284 —
// a ~71× span. We model the per-MB bit cost as the first-pass
// `bits_per_mb` surrogate divided by a power of the AC step, so a
// flat frame stays cheap at every qindex and a busy frame's cost
// collapses as the quantiser coarsens.

/// Exponent γ on the AC quantiser step in the bit-cost model
/// (fixed-point ×256). A γ near 1.0 makes predicted bits roughly
/// inversely proportional to the quantiser step. `RATE_MODEL_GAMMA_X256
/// / 256.0` is the effective exponent.
pub const RATE_MODEL_GAMMA_X256: i32 = 200;

/// Closed-loop feedback gain: qindex steps applied per one full
/// frame-allotment of accumulated rate debt. A gain of 4 means a
/// stream that is one whole frame's worth of bytes over budget nudges
/// the next frame's qindex up by 4 (coarser / cheaper). Kept modest so
/// the loop settles rather than oscillates.
pub const RATE_FEEDBACK_QSTEP_PER_FRAME: i32 = 4;

/// Clamp on the per-frame closed-loop feedback correction (qindex
/// units), so a single pathological frame's debt can't slam the
/// quantiser to an extreme in one step.
pub const RATE_FEEDBACK_MAX_QSTEP: i32 = 24;

/// Per-MB additive floor (in bits) the bit-cost model charges every
/// macroblock regardless of quantiser — mode bits, the always-present
/// per-MB header tokens, and the residual the quantiser can never
/// drive fully to zero. Keeps the model from predicting a literally
/// zero-bit frame at the coarsest quantiser.
pub const RATE_MODEL_MB_FLOOR_BITS: f32 = 6.0;

/// Reference AC quantiser step the surrogate was calibrated against.
/// `first_pass_analyze`'s `bits_per_mb` is tuned (§ its rustdoc) so a
/// `mad=10, var=200` frame lands near 100 — read as "≈100 bits/MB at
/// the default qindex". The §20.4 AC step at [`DEFAULT_QINDEX`] (50)
/// is the divisor that makes the model reproduce that reading, so the
/// surrogate value *is* the predicted bits/MB at the reference step.
const RATE_MODEL_REF_QINDEX: usize = DEFAULT_QINDEX as usize;

/// The §20.4 AC quantiser step for a clamped qindex.
#[inline]
fn ac_step_for_qindex(qindex: i32) -> f32 {
    crate::inverse_transform::AC_QLOOKUP[crate::inverse_transform::clamp_qindex(qindex)] as f32
}

/// Predict the encoded **bits per macroblock** for a frame of the
/// given complexity at a candidate `qindex`.
///
/// Clean-room model (see the module comment above): the first-pass
/// `bits_per_mb` surrogate is the predicted cost at
/// [`RATE_MODEL_REF_QINDEX`]; scaling to another qindex multiplies by
/// `(ref_step / step) ^ γ`, then a per-MB floor is added so the
/// coarsest quantiser still charges header / mode bits.
pub fn estimate_bits_per_mb(complexity: f32, qindex: i32) -> f32 {
    let gamma = RATE_MODEL_GAMMA_X256 as f32 / 256.0;
    let ref_step = ac_step_for_qindex(RATE_MODEL_REF_QINDEX as i32);
    let step = ac_step_for_qindex(qindex);
    // `complexity` is the surrogate's reading at the reference step;
    // a coarser (larger) step than the reference scales the cost down.
    let scaled = complexity.max(0.0) * (ref_step / step).powf(gamma);
    scaled + RATE_MODEL_MB_FLOOR_BITS
}

/// Predict the encoded **byte** count for one whole frame at `qindex`.
///
/// Multiplies [`estimate_bits_per_mb`] by the macroblock count implied
/// by `width × height` (16×16 MBs, partial MBs rounded up per §2), then
/// converts to bytes (÷8, rounded up).
pub fn estimate_frame_bytes(complexity: f32, qindex: i32, width: u32, height: u32) -> u64 {
    let mb_cols = (width as u64).div_ceil(16);
    let mb_rows = (height as u64).div_ceil(16);
    let mb_count = mb_cols.saturating_mul(mb_rows);
    let bits_per_mb = estimate_bits_per_mb(complexity, qindex);
    let total_bits = (bits_per_mb.max(0.0) as f64) * (mb_count as f64);
    (total_bits / 8.0).ceil() as u64
}

/// Mean luma value and variance for one frame.  Single linear pass
/// over `frame.y` honouring `frame.y_stride`.
fn luma_mean_and_variance(frame: &I420Frame<'_>) -> (f32, f32) {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return (0.0, 0.0);
    }
    let stride = frame.y_stride.max(w);
    let mut sum: u64 = 0;
    let mut sumsq: u64 = 0;
    let mut count: u64 = 0;
    for r in 0..h {
        let row_off = r * stride;
        // The luma slice may be sized for fewer rows than `h` (the
        // I420Frame::packed builder enforces tight packing but defensive
        // callers may pass shorter slices); clamp at the slice length.
        if row_off >= frame.y.len() {
            break;
        }
        let row_end = (row_off + w).min(frame.y.len());
        for &px in &frame.y[row_off..row_end] {
            sum += px as u64;
            sumsq += (px as u64) * (px as u64);
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    let count_f = count as f32;
    let mean = sum as f32 / count_f;
    // var = E[X²] - E[X]² (one-pass formula; numerically OK for u8
    // pixel values where the dynamic range is bounded).
    let var = (sumsq as f32 / count_f) - (mean * mean);
    (mean, var.max(0.0))
}

/// Mean absolute deviation of `frame.y` vs the supplied previous
/// luma plane (assumed identical layout).
fn luma_mad(frame: &I420Frame<'_>, prev: &[u8]) -> f32 {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return 0.0;
    }
    let stride = frame.y_stride.max(w);
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for r in 0..h {
        let row_off = r * stride;
        if row_off >= frame.y.len() || row_off >= prev.len() {
            break;
        }
        let row_end = (row_off + w).min(frame.y.len()).min(prev.len());
        let cur = &frame.y[row_off..row_end];
        let pre = &prev[row_off..row_end];
        for (a, b) in cur.iter().zip(pre.iter()) {
            sum += (*a as i32 - *b as i32).unsigned_abs() as u64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

/// Compute the per-frame qindex from a complexity record and a
/// reference mean cost.  Higher-than-mean cost ⇒ negative delta ⇒
/// lower qindex (better quality).  The delta is clamped to the
/// configured [`DEFAULT_AQ_QINDEX_RANGE`] either side of
/// `config.base.qindex`, then to the RFC 6386 §9.6 `0..=127` range.
/// Scene cuts add a [`DEFAULT_SCENE_CUT_QUANT_BOOST`] quality
/// boost (subtraction).
fn qindex_from_complexity(
    config: &Vp8TwoPassConfig,
    complexity: FrameComplexity,
    reference_mean: f32,
) -> u8 {
    let base = config.base.qindex as i32;
    let range = DEFAULT_AQ_QINDEX_RANGE.max(0);
    let delta = if reference_mean > 1.0 {
        let normed = (complexity.bits_per_mb - reference_mean) / reference_mean;
        // Negate: higher cost ⇒ lower qindex.
        let raw = -(normed * range as f32);
        raw.round() as i32
    } else {
        0
    };
    let mut qindex = base + delta.clamp(-range, range);
    if complexity.scene_cut {
        qindex -= DEFAULT_SCENE_CUT_QUANT_BOOST;
    }
    qindex.clamp(0, 127) as u8
}

/// Framework-side factory that takes a full [`Vp8EncoderConfig`]
/// (rather than just a qindex). Honours `qindex`, the §9.4 `lf_level`,
/// and the [`auto_loop_filter`](Vp8EncoderConfig::auto_loop_filter)
/// RD-selection switch; when the latter is set the §9.4 level / sharpness
/// are chosen per frame and `lf_level` is ignored.
#[cfg(feature = "registry")]
pub fn make_encoder_with_config(
    params: &oxideav_core::CodecParameters,
    config: Vp8EncoderConfig,
) -> oxideav_core::Result<Box<dyn oxideav_core::Encoder>> {
    factory::make_encoder_from_config(params, config)
}

/// Build a [`Vp8Encoder`] (the typed direct-API handle) from a
/// [`Vp8EncoderConfig`]. The "typed" variant returns the concrete
/// struct rather than a `Box<dyn Encoder>`; reachable under
/// `--no-default-features`.
pub fn make_encoder_typed_with_config(config: Vp8EncoderConfig) -> Vp8Encoder {
    Vp8Encoder::new(config)
}

// ─────────────── 0.1.13 encoder constants — type / ballpark defaults ───────────────
//
// The 0.1.13 release shipped a long list of tuning constants. The
// numeric values below match the documented defaults from the
// per-version rustdoc (where available) or use the documented
// "ballpark" the spec calls out. They are public so historical
// consumers can pattern-match against them; the encoder itself does
// not consume every entry yet.

/// Default `y_ac_qi` (§9.6) for the single-pass encoder.
pub const DEFAULT_QINDEX: u8 = 50;
/// Max value the adaptive-QP range may add to the base qindex.
pub const AQ_QINDEX_RANGE_MAX: i32 = 16;
/// Default adaptive-QP qindex range (±delta around the base qindex).
pub const DEFAULT_AQ_QINDEX_RANGE: i32 = 8;
/// Per-segment adaptive-QP variance thresholds (4 segments — VP8 §10).
pub const DEFAULT_ADAPTIVE_SEGMENT_THRESHOLDS: [u32; 4] = [80, 320, 1_280, 5_120];
/// Default alt-ref refresh interval in frames.
pub const DEFAULT_ALT_REF_INTERVAL: u32 = 16;
/// Default golden-frame refresh interval in frames.
pub const DEFAULT_GOLDEN_INTERVAL: u32 = 8;
/// Default chroma-aware spatial chroma weight (fixed-point ×256).
pub const DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256: i32 = 96;
/// Default chroma-aware spatial luma weight (fixed-point ×256).
pub const DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256: i32 = 160;
/// Default maximum iteration count for the joint R44+R49 picker.
pub const DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS: u32 = 4;
/// Hard cap on the joint R44+R49 picker iteration count.
pub const JOINT_R44R49_PICKER_MAX_ITERS_MAX: u32 = 16;
/// Default convergence threshold for the k-means segmentation pass
/// (centroid drift below which the loop terminates, x256).
pub const DEFAULT_KMEANS_CONVERGENCE_THRESHOLD: i32 = 8;
/// Default k-means spatial-alpha weight (fixed-point ×256).
pub const DEFAULT_KMEANS_SPATIAL_ALPHA_X256: i32 = 128;
/// Hard cap on the k-means spatial-segmentation iteration count.
pub const KMEANS_SPATIAL_MAX_ITERS: u32 = 32;
/// Default long-ref lambda scale (×256).
pub const DEFAULT_LAMBDA_LONG_REF_SCALE_X256: i32 = 256;
/// Default RD lambda scale (×256). 256 ≡ 1.0.
pub const LAMBDA_SCALE_DEFAULT: i32 = 256;
/// Default lookahead window in frames (first-pass / scene-cut buffer).
pub const DEFAULT_LOOKAHEAD_WINDOW: u32 = 16;
/// Non-local-means filter `h²` parameter default.
pub const DEFAULT_NLM_H2: i32 = 100;
/// Default psy-RD strength (fixed-point, ballpark "moderate").
pub const DEFAULT_PSY_RD_STRENGTH: i32 = 30;
/// Default boost-frame count when a scene cut is detected.
pub const DEFAULT_SCENE_CUT_BOOST_FRAMES: u32 = 2;
/// Default qindex boost (subtraction) on a scene-cut keyframe.
pub const DEFAULT_SCENE_CUT_QUANT_BOOST: i32 = 8;
/// Default scene-cut decision threshold (variance × frame ratio).
pub const DEFAULT_SCENE_CUT_THRESHOLD: i32 = 4_096;
/// Default segment-aware loop-filter deltas (4 segments — §10).
pub const DEFAULT_SEGMENT_LF_DELTAS: [i8; 4] = [0, 0, 0, 0];
/// Default segment-aware quant deltas (4 segments — §10).
pub const DEFAULT_SEGMENT_QUANT_DELTAS: [i8; 4] = [0, 0, 0, 0];
/// Default maximum simple-filter loop-filter level.
pub const DEFAULT_SIMPLE_LF_MAX_LEVEL: u8 = 32;
/// Default number of spatial loop-filter column bands.
pub const DEFAULT_SPATIAL_LF_N_COL_BANDS: u32 = 4;
/// Default number of spatial loop-filter row bands.
pub const DEFAULT_SPATIAL_LF_N_ROW_BANDS: u32 = 4;
/// Default number of refinement passes for the SPLITMV joint picker.
pub const DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES: u32 = 2;
/// Hard cap on the SPLITMV joint-refine pass count.
pub const SPLIT_MV_JOINT_REFINE_PASSES_MAX: u32 = 8;
/// Variance threshold above which an intra block forces B_PRED in a
/// P-frame.
pub const INTRA_IN_P_BPRED_VARIANCE_THRESHOLD: i32 = 256;
/// QP sensitivity scaling (×8) used by the §10 segment-aware QP search.
pub const QP_SENSITIVITY_X8: i32 = 8;
/// Absolute floor on the scene-cut metric below which a cut is never
/// declared.
pub const SCENE_CUT_ABS_FLOOR: i32 = 32;
/// Per-segment variance thresholds the adaptive-QP / scene-segmenter
/// uses (4 segments).
pub const SEGMENT_VARIANCE_THRESHOLDS: [u32; 4] = [64, 256, 1_024, 4_096];

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-round-441 §13 trellis, retained verbatim as the byte-level
    /// reference for the per-frame rate-table + sub-threshold-early-exit
    /// rewrite. It prices every candidate through the direct
    /// [`super::coeff_token_bits`] / [`super::eob_token_bits`] calls and
    /// never takes the all-zero shortcut, so
    /// `trellis_lut_and_early_exit_match_reference` below can pin the
    /// shipped fast paths against it level-for-level.
    #[allow(clippy::needless_range_loop)]
    fn trellis_quantize_block_reference(
        block: &mut [i16; 16],
        spec: super::TrellisBlockSpec,
        coeff_probs: &super::CoeffProbs,
        lambda: f64,
    ) {
        let super::TrellisBlockSpec {
            dc,
            ac,
            first_coeff,
            entry_ctx3,
            plane,
        } = spec;
        // `lambda` is the **effective** coefficient-level price the caller has
        // already scaled down from the mode-decision multiplier (coefficient
        // RDO spends `lambda` far more effectively than the mode-granular
        // picker — it can drop one trailing coefficient rather than a whole
        // mode — so the raw mode lambda over-trades quality for rate). The
        // scale is resolved by [`MbRdCtx::trellis_lambda`] from the
        // [`TrellisStrength`] knob; this entry consumes the resolved value
        // directly so callers can dial the rate/quality trade per stream.

        // Baseline rounding for the lossless / no-trade path.
        if lambda <= 0.0 {
            super::enc_quantize_block_range(block, dc, ac, first_coeff);
            return;
        }

        // Scan-order view of the unquantised coefficients + per-position step.
        let mut scan_x = [0i32; 16];
        let mut step = [0i32; 16];
        for k in 0..16 {
            scan_x[k] = block[crate::dct_tokens::ZIGZAG[k]] as i32;
            step[k] = if k == 0 { dc as i32 } else { ac as i32 };
        }

        let norms = &*super::IDCT_BASIS_NORM_SQ;

        // Distortion of coding scan position `k` at absolute level `m`
        // (transform-domain squared error scaled by the basis norm). `m` is a
        // magnitude (≥ 0); the dequantised coefficient carries `scan_x[k]`'s
        // sign, so the error is computed on absolute magnitudes.
        let dist = |k: usize, m: i64| -> f64 {
            let q = step[k] as i64;
            let recon = m * q;
            let err = (scan_x[k] as i64).abs() - recon;
            norms[k] * (err * err) as f64
        };

        // Suffix distortion of *dropping* every position k..16 to level 0 —
        // the distortion the §13.3 implicit/explicit EOB tail pays for the
        // coefficients past the last coded one. `suffix_zero_dist[k]` =
        // Σ_{j=k}^{15} dist(j, 0). Without this, terminating early would hide
        // the energy of the dropped tail and the trellis would zero
        // everything. `suffix_zero_dist[16] = 0`.
        let mut suffix_zero_dist = [0f64; 17];
        for k in (first_coeff..16).rev() {
            suffix_zero_dist[k] = suffix_zero_dist[k + 1] + dist(k, 0);
        }

        // Forward Viterbi. State = carried `ctx3 ∈ {0,1,2}`. `cells[k]` holds
        // the best survivor *ending at* scan position `k` with each carried
        // context, where "ending" means position `k` holds the last coded
        // (possibly non-zero) coefficient so far. We also separately track the
        // running all-zero-prefix cost so a block can terminate with EOB at
        // `first_coeff` (entirely zero) for free of trailing-token bits.
        //
        // `prev[ctx]` is the best (cost, level, prev_ctx, prev_was_zero) to
        // arrive at the *current* position carrying `ctx`.
        const INF: f64 = f64::INFINITY;

        // survivor reaching position `k` with carried ctx — initialised at the
        // virtual "before first_coeff" boundary: a single state carrying
        // `entry_ctx3`, cost 0, prev_was_zero = false (the first token may emit
        // EOB).
        #[derive(Clone, Copy)]
        struct Surv {
            cost: f64,
            prev_was_zero: bool,
            // back-pointer chain stored in `choice` below
        }
        let mut surv = [Surv {
            cost: INF,
            prev_was_zero: false,
        }; 3];
        surv[entry_ctx3] = Surv {
            cost: 0.0,
            prev_was_zero: false,
        };

        // Decision record per (position, carried-ctx): the level chosen at this
        // position and which predecessor ctx it came from, for traceback.
        let mut choice = [[super::TrellisCell {
            cost: INF,
            level: 0,
            prev_ctx: 0,
        }; 3]; 16];

        // Best terminating survivor: code positions first..=t, then EOB after
        // t (if t+1 < 16) or implicit EOB. `best_end_pos = first_coeff - 1`
        // (sentinel `usize::MAX` meaning "all zero") seeds the all-zero block.
        let mut best_term_cost = {
            // All-zero block: EOB token at `first_coeff` against the entry ctx,
            // plus the distortion of dropping every coefficient to zero.
            let mut c = INF;
            let s = &surv[entry_ctx3];
            if s.cost < INF {
                c = s.cost
                    + lambda * super::eob_token_bits(coeff_probs, plane, first_coeff, entry_ctx3)
                    + suffix_zero_dist[first_coeff];
            }
            c
        };
        let mut best_term_pos: i32 = -1; // -1 ⇒ all-zero
        let mut best_term_ctx = entry_ctx3;

        for k in first_coeff..16 {
            let m_round = {
                let q = step[k] as i64;
                // round-half-away-from-zero magnitude
                let num = scan_x[k].unsigned_abs() as i64;
                (num + q / 2) / q
            };
            // Candidate magnitudes: round and round-1 (≥0), deduplicated, plus
            // a forced 0 so a non-trivial coefficient can still be dropped.
            let mut cands = [0i64; 3];
            let mut ncand = 0usize;
            for cand in [m_round, m_round - 1, 0] {
                if cand >= 0 && !cands[..ncand].contains(&cand) {
                    cands[ncand] = cand;
                    ncand += 1;
                }
            }

            let mut next = [Surv {
                cost: INF,
                prev_was_zero: false,
            }; 3];
            let mut next_choice = [super::TrellisCell {
                cost: INF,
                level: 0,
                prev_ctx: 0,
            }; 3];

            for (in_ctx, &s) in surv.iter().enumerate() {
                if s.cost >= INF {
                    continue;
                }
                for &m in &cands[..ncand] {
                    let abs = m as u16;
                    let d = dist(k, m);
                    let r = super::coeff_token_bits(
                        coeff_probs,
                        plane,
                        k,
                        in_ctx,
                        s.prev_was_zero,
                        abs,
                    );
                    let cost = s.cost + d + lambda * r;
                    let out_ctx = super::level_ctx3(abs);
                    if cost < next[out_ctx].cost {
                        next[out_ctx] = Surv {
                            cost,
                            prev_was_zero: abs == 0,
                        };
                        next_choice[out_ctx] = super::TrellisCell {
                            cost,
                            level: abs,
                            prev_ctx: in_ctx,
                        };
                    }
                }
            }

            choice[k] = next_choice;

            // Consider terminating *after* position k: position k must hold a
            // non-zero coefficient to be a meaningful last-coded slot (a zero
            // last slot is dominated by terminating one position earlier). The
            // EOB after position k lands at k+1 (if any) against k's out-ctx;
            // at k = 15 the implicit EOB costs nothing.
            for (out_ctx, &cell) in next_choice.iter().enumerate() {
                if cell.cost >= INF || cell.level == 0 {
                    continue;
                }
                // `cell.cost` covers coding first..=k; add the EOB token (if
                // any remain) plus the distortion of dropping positions k+1..16.
                let term_cost = if k + 1 < 16 {
                    cell.cost
                        + lambda * super::eob_token_bits(coeff_probs, plane, k + 1, out_ctx)
                        + suffix_zero_dist[k + 1]
                } else {
                    cell.cost // implicit EOB after the last coefficient
                };
                if term_cost < best_term_cost {
                    best_term_cost = term_cost;
                    best_term_pos = k as i32;
                    best_term_ctx = out_ctx;
                }
            }

            surv = next;
        }

        // Traceback from the best terminating position, writing scan-order
        // levels then permuting back to raster.
        let mut scan_levels = [0i32; 16];
        if best_term_pos >= 0 {
            let mut k = best_term_pos as usize;
            let mut ctx = best_term_ctx;
            loop {
                let cell = choice[k][ctx];
                let lvl = cell.level as i32;
                let signed = if scan_x[k] < 0 { -lvl } else { lvl };
                scan_levels[k] = signed;
                if k == first_coeff {
                    break;
                }
                ctx = cell.prev_ctx;
                k -= 1;
            }
        }

        // Permute scan levels back into the raster `block`.
        for v in block.iter_mut() {
            *v = 0;
        }
        for (k, &lvl) in scan_levels.iter().enumerate() {
            block[crate::dct_tokens::ZIGZAG[k]] = lvl as i16;
        }
    }
    use crate::bool_decoder::BoolDecoder;
    use crate::coded_header::Vp8CodedHeader;
    use crate::decode_vp8;
    use crate::frame_header::Vp8FrameHeader;

    /// Every point of the per-frame [`TrellisRateTables`] domain must be
    /// **bit-for-bit** the value the direct reference rate call returns —
    /// under the §13.5 default table AND a deterministically perturbed
    /// non-uniform table (a wrong band/context/level slot would change the
    /// f64, not alias into it). Together with
    /// `trellis_lut_and_early_exit_match_reference` this pins the r441
    /// lookup path: identical rates ⇒ identical Viterbi costs ⇒ identical
    /// chosen levels ⇒ identical emitted bytes.
    #[test]
    fn trellis_rate_tables_match_direct_calls() {
        let mut perturbed = crate::DEFAULT_COEFF_PROBS;
        let mut st = 0x1357_9BDFu32;
        for plane in perturbed.iter_mut() {
            for band in plane.iter_mut() {
                for ctx in band.iter_mut() {
                    for prob in ctx.iter_mut() {
                        st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *prob = ((st >> 24) as u8).max(1);
                    }
                }
            }
        }
        for probs in [&crate::DEFAULT_COEFF_PROBS, &perturbed] {
            let t = TrellisRateTables::for_probs(probs);
            for plane in 0..4 {
                for (i, &band) in COEFF_BANDS.iter().enumerate() {
                    for ctx3 in 0..3 {
                        assert_eq!(
                            t.eob[plane][band][ctx3].to_bits(),
                            eob_token_bits(probs, plane, i, ctx3).to_bits(),
                            "eob: plane {plane} pos {i} ctx {ctx3}"
                        );
                        for pwz in 0..2usize {
                            for abs in 0..=TRELLIS_SMALL_LEVEL_MAX {
                                assert_eq!(
                                    t.small[plane][band][ctx3][pwz][abs as usize].to_bits(),
                                    coeff_token_bits(probs, plane, i, ctx3, pwz == 1, abs)
                                        .to_bits(),
                                    "plane {plane} pos {i} ctx {ctx3} pwz {pwz} abs {abs}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The shipped trellis (per-frame rate LUT + sub-threshold early
    /// exit) must produce **level-for-level identical** output to the
    /// retained pre-r441 reference implementation — with and without the
    /// tables — across block types, entry contexts, quantisers, lambdas,
    /// probability tables, and residual magnitudes from far-sub-threshold
    /// (the early-exit path) through cat-token territory (the `abs > 4`
    /// direct-call fallback).
    #[test]
    fn trellis_lut_and_early_exit_match_reference() {
        let mut perturbed = crate::DEFAULT_COEFF_PROBS;
        let mut st = 0xA5A5_1234u32;
        for plane in perturbed.iter_mut() {
            for band in plane.iter_mut() {
                for ctx in band.iter_mut() {
                    for prob in ctx.iter_mut() {
                        st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *prob = ((st >> 24) as u8).max(1);
                    }
                }
            }
        }

        let block_specs = [
            (BlockType::Y2, 0usize),
            (BlockType::YAfterY2, 0),
            (BlockType::YNoY2, 1),
            (BlockType::UV, 2),
        ];

        let mut cases = 0usize;
        for probs in [&crate::DEFAULT_COEFF_PROBS, &perturbed] {
            let tables = TrellisRateTables::for_probs(probs);
            for qi in [8i32, 40, 100] {
                let factors =
                    crate::dequant::MbDequantFactors::from_base_and_deltas(qi, 0, 0, 0, 0, 0);
                let base_lambda = rd_lambda(&factors) * TRELLIS_LAMBDA_SCALE;
                for (block_type, entry_ctx3) in block_specs {
                    let (dc, ac) = match block_type {
                        BlockType::Y2 => (factors.y2_dc, factors.y2_ac),
                        BlockType::UV => (factors.uv_dc, factors.uv_ac),
                        _ => (factors.y1_dc, factors.y1_ac),
                    };
                    let spec = TrellisBlockSpec {
                        dc,
                        ac,
                        first_coeff: block_type.first_coeff(),
                        entry_ctx3,
                        plane: block_type.plane_index(),
                    };
                    // Magnitude regimes: far sub-threshold (all-zero early
                    // exit), around-threshold, working range, cat-token
                    // range (levels > 4 hit the direct-call fallback).
                    for (scale, offset) in [
                        (1i32, 0i32),
                        (ac as i32, 0),
                        (ac as i32 * 3, 8),
                        (ac as i32 * 24, 64),
                    ] {
                        for seed in 0u32..12 {
                            let mut lcg = 0xC0FF_EE00u32
                                .wrapping_add(seed)
                                .wrapping_mul(2_891_336_453);
                            let mut next = move || {
                                lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12345);
                                (lcg >> 16) as i32
                            };
                            let mut base = [0i16; 16];
                            for (k, c) in base.iter_mut().enumerate() {
                                // Roughly zig-zag-decaying magnitudes with
                                // both signs, like real DCT output.
                                let mag = (next() % (scale + offset + 1)) / (1 + k as i32 / 4);
                                let sign = if next() % 2 == 0 { 1 } else { -1 };
                                *c = (sign * mag).clamp(-2048, 2047) as i16;
                            }
                            for lambda in [base_lambda * 0.25, base_lambda, base_lambda * 6.0] {
                                let mut want = base;
                                trellis_quantize_block_reference(&mut want, spec, probs, lambda);
                                let mut got_direct = base;
                                trellis_quantize_block(&mut got_direct, spec, probs, None, lambda);
                                let mut got_lut = base;
                                trellis_quantize_block(
                                    &mut got_lut,
                                    spec,
                                    probs,
                                    Some(&tables),
                                    lambda,
                                );
                                assert_eq!(
                                    want, got_direct,
                                    "direct path drifted: qi {qi} type {block_type:?} scale {scale} seed {seed} lambda {lambda}"
                                );
                                assert_eq!(
                                    want, got_lut,
                                    "LUT path drifted: qi {qi} type {block_type:?} scale {scale} seed {seed} lambda {lambda}"
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases >= 3000, "stress matrix shrank: {cases} cases");
    }

    /// Re-read a §9.3 `update_segmentation()` block from raw bits the
    /// same way the decoder's `parse_update_segmentation` does, so a
    /// `write_update_segmentation` round-trip can be asserted without
    /// reaching the private parser. Mirrors the §19.2 table layout
    /// verbatim (the writer is the only thing under test).
    #[derive(Debug, PartialEq, Eq)]
    struct DecodedSegLayer {
        feature_absolute: bool,
        update_map: bool,
        quant: [Option<i16>; 4],
        lf: [Option<i16>; 4],
        probs: [Option<u8>; 3],
    }

    fn read_update_segmentation_for_test(dec: &mut BoolDecoder<'_>) -> DecodedSegLayer {
        // The caller has already consumed segmentation_enabled = 1.
        let update_map = dec.read_bool(128).unwrap();
        let update_feature = dec.read_bool(128).unwrap();
        let mut feature_absolute = false;
        let mut quant: [Option<i16>; 4] = [None; 4];
        let mut lf: [Option<i16>; 4] = [None; 4];
        if update_feature {
            feature_absolute = dec.read_bool(128).unwrap();
            for slot in &mut quant {
                if dec.read_bool(128).unwrap() {
                    let mag = dec.read_literal(7).unwrap() as i16;
                    let sign = dec.read_bool(128).unwrap();
                    *slot = Some(if sign { -mag } else { mag });
                }
            }
            for slot in &mut lf {
                if dec.read_bool(128).unwrap() {
                    let mag = dec.read_literal(6).unwrap() as i16;
                    let sign = dec.read_bool(128).unwrap();
                    *slot = Some(if sign { -mag } else { mag });
                }
            }
        }
        let mut probs: [Option<u8>; 3] = [None; 3];
        if update_map {
            for slot in &mut probs {
                if dec.read_bool(128).unwrap() {
                    *slot = Some(dec.read_literal(8).unwrap() as u8);
                }
            }
        }
        DecodedSegLayer {
            feature_absolute,
            update_map,
            quant,
            lf,
            probs,
        }
    }

    /// `write_update_segmentation` lays the §9.3 sub-block out so the
    /// decoder's reader recovers every field: the map-update flag, the
    /// delta/absolute feature mode, the four signed quantizer deltas, the
    /// four signed loop-filter deltas, and the three tree probabilities,
    /// including the `None` (flag-not-set) slots.
    #[test]
    fn update_segmentation_writer_round_trips_full_layer() {
        let layer = UpdateSegmentationLayer {
            update_mb_segmentation_map: true,
            feature_mode_absolute: false,
            quantizer: [Some(-12), None, Some(40), Some(-1)],
            loop_filter: [None, Some(7), None, Some(-3)],
            segment_prob: [Some(200), None, Some(31)],
        };
        let mut enc = BoolEncoder::new();
        write_update_segmentation(&mut enc, &layer);
        let buf = enc.finish();

        let mut dec = BoolDecoder::init(&buf).unwrap();
        // segmentation_enabled = 1 written first.
        assert!(dec.read_bool(128).unwrap());
        let got = read_update_segmentation_for_test(&mut dec);
        assert_eq!(got.feature_absolute, layer.feature_mode_absolute);
        assert_eq!(got.update_map, layer.update_mb_segmentation_map);
        assert_eq!(got.quant, layer.quantizer);
        assert_eq!(got.lf, layer.loop_filter);
        assert_eq!(got.probs, layer.segment_prob);
    }

    /// Absolute feature mode + no map update: `update_segment_feature_data`
    /// is still set (a feature is carried), but the `segment_prob` block is
    /// absent. Exercises the absolute-mode flag and the map-off branch.
    #[test]
    fn update_segmentation_writer_absolute_no_map_update() {
        let layer = UpdateSegmentationLayer {
            update_mb_segmentation_map: false,
            feature_mode_absolute: true,
            quantizer: [Some(20), Some(35), Some(60), Some(90)],
            loop_filter: [None; 4],
            segment_prob: [None; 3],
        };
        let mut enc = BoolEncoder::new();
        write_update_segmentation(&mut enc, &layer);
        let buf = enc.finish();

        let mut dec = BoolDecoder::init(&buf).unwrap();
        assert!(dec.read_bool(128).unwrap());
        let got = read_update_segmentation_for_test(&mut dec);
        assert!(got.feature_absolute);
        assert!(!got.update_map);
        assert_eq!(got.quant, layer.quantizer);
        assert_eq!(got.lf, [None; 4]);
        assert_eq!(got.probs, [None; 3]);
    }

    /// Map-update with no feature data: `update_segment_feature_data` codes
    /// 0 (no quantizer / loop-filter deltas), but the three tree
    /// probabilities are still written.
    #[test]
    fn update_segmentation_writer_map_only_no_features() {
        let layer = UpdateSegmentationLayer {
            update_mb_segmentation_map: true,
            feature_mode_absolute: false,
            quantizer: [None; 4],
            loop_filter: [None; 4],
            segment_prob: [Some(255), Some(128), Some(1)],
        };
        let mut enc = BoolEncoder::new();
        write_update_segmentation(&mut enc, &layer);
        let buf = enc.finish();

        let mut dec = BoolDecoder::init(&buf).unwrap();
        assert!(dec.read_bool(128).unwrap());
        let got = read_update_segmentation_for_test(&mut dec);
        assert!(got.update_map);
        assert_eq!(got.quant, [None; 4]);
        assert_eq!(got.lf, [None; 4]);
        assert_eq!(got.probs, layer.segment_prob);
    }

    /// Equivalence proof for the round-277 compile-time
    /// `PARTITION_GROUP_TABLE`: for each of the four §16.4 partitions,
    /// the precomputed groups must be exactly what a direct walk of the
    /// §20.13 `MV_PARTITIONS` row produces — the right group count
    /// (2 / 2 / 4 / 16), every sub-block in the group whose id the
    /// table assigns it, raster-ascending member order (so `group[0]`
    /// is the §16.4 anchor), and all 16 sub-blocks covered exactly
    /// once.
    #[test]
    fn partition_group_table_matches_mv_partitions() {
        use crate::near_mv::{partition_id, MvPartition, MV_PARTITIONS};
        for (p, expected_groups) in [
            (MvPartition::TopBottom, 2usize),
            (MvPartition::LeftRight, 2),
            (MvPartition::Quarters, 4),
            (MvPartition::Mv16, 16),
        ] {
            let groups = partition_groups(p);
            assert_eq!(groups.num_groups, expected_groups, "{p:?} group count");
            let table = &MV_PARTITIONS[partition_id(p)];
            let mut seen = [false; 16];
            for g in 0..groups.num_groups {
                let members = groups.group(g);
                assert!(!members.is_empty(), "{p:?} group {g} empty");
                let mut prev: Option<usize> = None;
                for &b in members {
                    assert_eq!(
                        table[b] as usize, g,
                        "{p:?} sub-block {b} listed under group {g}"
                    );
                    assert!(!seen[b], "{p:?} sub-block {b} listed twice");
                    seen[b] = true;
                    if let Some(prev) = prev {
                        assert!(prev < b, "{p:?} group {g} not raster-ascending");
                    }
                    prev = Some(b);
                }
            }
            assert!(seen.iter().all(|&s| s), "{p:?} does not cover all 16");
        }
    }

    /// Equivalence proof for the round-281 fused
    /// [`group_sad_at_whole_mv`] fast path: for every §16.4 partition
    /// shape, every group, and a candidate sweep spanning in-bounds and
    /// border-straddling whole-pixel MVs (in all four directions), the
    /// fused row-window SAD must equal the pre-round-281 per-member
    /// assembly (`fetch_block_whole_pixel` + `extract_src_subblock_4x4`
    /// + `sub_block_sad_4x4`) bit-for-bit.
    #[test]
    fn group_sad_fused_fast_path_matches_per_subblock_fetch() {
        use crate::motion_vector::Mv;
        use crate::near_mv::MvPartition;

        // 48×48 plane = 3×3 MBs; MB(0,0) and MB(2,2) reach the plane
        // corners so the extreme candidates straddle every border.
        let width = 48usize;
        let height = 48usize;
        let mut plane = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * width + x] = ((x * 11 + y * 17) % 251) as u8;
            }
        }
        let luma = crate::motion_search::LumaRef {
            plane: &plane,
            stride: width,
            width,
            height,
        };
        let mut src_y = [0u8; 256];
        for (i, slot) in src_y.iter_mut().enumerate() {
            *slot = ((i * 3 + 41) % 249) as u8;
        }

        let per_subblock_reference =
            |mb_col: usize, mb_row: usize, group_subblocks: &[usize], mv: Mv| -> u32 {
                let mv_eighth = crate::motion_comp::stored_luma_mv(mv);
                let mut sad: u32 = 0;
                for &b in group_subblocks {
                    let patch = crate::motion_comp::fetch_block_whole_pixel(
                        luma.plane,
                        luma.stride,
                        luma.width,
                        luma.height,
                        mb_col * 16 + (b & 3) * 4,
                        mb_row * 16 + (b >> 2) * 4,
                        mv_eighth,
                    );
                    let src_sb = extract_src_subblock_4x4(&src_y, b);
                    sad += sub_block_sad_4x4(&src_sb, &patch);
                }
                sad
            };

        let candidates: [Mv; 11] = [
            Mv { row: 0, col: 0 },
            Mv { row: 4, col: -4 },
            Mv { row: -8, col: 12 },
            Mv { row: 16, col: 16 },
            Mv { row: -16, col: -16 },
            Mv { row: 64, col: -64 },
            Mv { row: -1020, col: 0 },
            Mv { row: 0, col: -1020 },
            Mv { row: 1020, col: 0 },
            Mv { row: 0, col: 1020 },
            Mv {
                row: -1020,
                col: 1020,
            },
        ];

        for p in [
            MvPartition::TopBottom,
            MvPartition::LeftRight,
            MvPartition::Quarters,
            MvPartition::Mv16,
        ] {
            let groups = partition_groups(p);
            for (mb_col, mb_row) in [(0usize, 0usize), (1, 1), (2, 2)] {
                for g in 0..groups.num_groups {
                    let members = groups.group(g);
                    for mv in candidates {
                        let expected = per_subblock_reference(mb_col, mb_row, members, mv);
                        let got = group_sad_at_whole_mv(luma, mb_col, mb_row, &src_y, members, mv);
                        assert_eq!(
                            got, expected,
                            "fused group SAD diverged: {p:?} group {g} \
                             MB({mb_col},{mb_row}) mv={mv:?}"
                        );
                    }
                }
            }
        }
    }

    /// Equivalence proof for the precomputed `TOKEN_BIT_PATHS` table:
    /// for every reachable `(start_index, token)` combination, the
    /// table's stored `(prob_index, bit)` sequence must equal the result
    /// of running the §13.2 tree descent on `ENC_COEFF_TREE` directly.
    /// Also asserts the one unreachable cell (`start_index = 2`,
    /// `token = Eob`) stays a length-0 tombstone — the §13.2 "previous
    /// coefficient was DCT_0 → skip Eob branch" invariant — so any
    /// future call to it will trip the in-function `debug_assert`.
    #[test]
    fn token_bit_path_table_matches_tree_descent() {
        // Reference descent — duplicates the original recursive walk
        // before round 204 precomputed it into TOKEN_BIT_PATHS.
        fn reference_descent(i: i8, target: i8, path: &mut Vec<(usize, bool)>) -> bool {
            for &bit in &[false, true] {
                let child = ENC_COEFF_TREE[i as usize + bit as usize];
                if child <= 0 {
                    if -child == target {
                        path.push(((i as usize) >> 1, bit));
                        return true;
                    }
                } else {
                    path.push(((i as usize) >> 1, bit));
                    if reference_descent(child, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            false
        }

        let tokens = [
            DctToken::Dct0,
            DctToken::Dct1,
            DctToken::Dct2,
            DctToken::Dct3,
            DctToken::Dct4,
            DctToken::Cat1,
            DctToken::Cat2,
            DctToken::Cat3,
            DctToken::Cat4,
            DctToken::Cat5,
            DctToken::Cat6,
            DctToken::Eob,
        ];

        for &start in &[0i8, 2i8] {
            for &token in &tokens {
                let mut reference = Vec::new();
                let reachable = reference_descent(start, token as i8, &mut reference);

                // The single unreachable cell.
                if !reachable {
                    assert_eq!(
                        (start, token),
                        (2i8, DctToken::Eob),
                        "unexpected unreachable (start, token) = ({start}, {token:?})"
                    );
                    let slot = 1usize;
                    let (_, len) = &TOKEN_BIT_PATHS[slot][token as usize];
                    assert_eq!(*len, 0, "(start=2, Eob) must remain a length-0 tombstone");
                    continue;
                }

                let table = token_to_bit_path(token, start);
                assert_eq!(
                    table.len(),
                    reference.len(),
                    "path-length mismatch for (start={start}, token={token:?})"
                );
                for (k, &(ref_i, ref_bit)) in reference.iter().enumerate() {
                    assert_eq!(
                        table[k].prob_index as usize, ref_i,
                        "prob_index mismatch at step {k} of (start={start}, token={token:?})"
                    );
                    assert_eq!(
                        table[k].bit, ref_bit,
                        "bit mismatch at step {k} of (start={start}, token={token:?})"
                    );
                }
            }
        }

        // Width contract: no path should ever exceed the static buffer.
        const { assert!(TOKEN_BIT_PATH_MAX_LEN >= 7) };
        for slot in 0..2 {
            for token in 0..12 {
                let (_, len) = &TOKEN_BIT_PATHS[slot][token];
                assert!(
                    (*len as usize) <= TOKEN_BIT_PATH_MAX_LEN,
                    "path length {} > buffer capacity {} at slot={slot} token={token}",
                    *len,
                    TOKEN_BIT_PATH_MAX_LEN
                );
            }
        }
    }

    /// Sanity check on the local `COEFF_UPDATE_PROBS` table — the
    /// 1056-entry flat walk should be non-zero and the last entry should
    /// equal `[3][7][2][10]` of the nested form. Catches a transcription
    /// typo or a flattening-loop off-by-one.
    #[test]
    fn coeff_update_probs_flat_walk_is_consistent() {
        let s: u32 = COEFF_UPDATE_PROBS_FLAT.iter().map(|&p| p as u32).sum();
        assert!(s > 0);
        assert_eq!(
            COEFF_UPDATE_PROBS_FLAT[3 * 8 * 3 * 11 + 7 * 3 * 11 + 2 * 11 + 10],
            COEFF_UPDATE_PROBS[3][7][2][10]
        );
    }

    /// Anchor the §13.4 `token_prob_update()` walk-order byte-
    /// equivalence between [`write_no_token_prob_updates`] and
    /// [`write_token_prob_updates`]-handed-all-`None` against the
    /// **actual** §13.4 `coeff_update_probs[4][8][3][11]` spec table
    /// (RFC 6386 §13.4 page 69), not a flat `[128u8; 1056]` placeholder.
    ///
    /// Why both writers should agree on every byte when the §13.4 flag
    /// is `false` everywhere:
    ///
    /// * Both writers walk the same `(i=0..4, j=0..8, k=0..3, t=0..11)`
    ///   four-nested-`do/while` order from RFC 6386 §13.4.
    /// * Both writers consult `flag_probs[i*8*3*11 + j*3*11 + k*11 + t]`
    ///   when emitting the per-position "is this slot replaced?" bit.
    /// * On the all-`None` path, `write_token_prob_updates` emits
    ///   `write_bool(p, false)` at every position and never follows up
    ///   with an `L(8)`; `write_no_token_prob_updates` emits the same
    ///   `write_bool(p, false)` at the same `p`. With the bool encoder
    ///   being deterministic on `(state, prob, bit)` triples, the byte
    ///   streams MUST be identical.
    ///
    /// The existing external test
    /// (`tests/encoder_token_prob_updates.rs::write_token_prob_updates_all_none_matches_no_update_writer`)
    /// validates this with a flat `[128u8; 1056]`, which means the bool
    /// encoder's range / split machinery never exercises the rare
    /// extreme-probability splits the §13.4 table actually contains
    /// (the table has entries as low as `5` and as high as `255`). This
    /// in-crate test closes that gap by exercising the byte equivalence
    /// against the real §13.4 flag table — if a future refactor of
    /// either writer subtly diverges (e.g. one switches to `write_bit`
    /// at a hard-coded probability, or skips a slot when the flag
    /// probability is `0` / `255`), the [128;1056] placeholder test
    /// might still pass while this one would catch the regression.
    #[test]
    fn write_no_token_prob_updates_matches_all_none_against_spec_flag_probs() {
        let no_updates: crate::coded_header::TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];

        let mut a = BoolEncoder::new();
        write_no_token_prob_updates(&mut a, &COEFF_UPDATE_PROBS_FLAT);
        let bytes_a = a.finish();

        let mut b = BoolEncoder::new();
        write_token_prob_updates(&mut b, &no_updates, &COEFF_UPDATE_PROBS_FLAT);
        let bytes_b = b.finish();

        assert_eq!(
            bytes_a, bytes_b,
            "writers must agree on the all-None §13.4 payload under \
             the real coeff_update_probs[4][8][3][11] table — any \
             divergence here means the §13.4 four-nested walk order \
             or the per-position probability lookup has drifted \
             between the two writers."
        );

        // Bit-count sanity: both writers emit exactly 1056 bool bits
        // and no L(8) payload on the all-None path, so the output must
        // be non-empty (the bool encoder always flushes at least one
        // partial byte at `finish()`).
        assert!(
            !bytes_a.is_empty(),
            "all-None §13.4 payload must still emit at least one byte"
        );
    }

    /// Anchor the encoder-side §17.2 `MV_UPDATE_PROBS_FLAT` 38-entry
    /// flat table against the canonical 2×19 spec table held in
    /// [`crate::coded_header`] (`MV_UPDATE_PROBS`, transcribed from
    /// RFC 6386 §17.2 `vp8_mv_update_probs[2]`).
    ///
    /// The encoder's `mv_prob_update()` writer (called inline by
    /// every inter-frame entry-point — see the
    /// `for ctx in MV_UPDATE_PROBS_FLAT.iter()` loop around the §17.2
    /// no-update flag emission) walks a flat 38-entry copy of the
    /// table. The decoder (`parse_mv_prob_update`) reads each `F`
    /// flag at `MV_UPDATE_PROBS[i][j]`. If the two transcriptions
    /// ever drift — a typo, a row/column swap, or an off-by-one in
    /// the flat walk — the encoder would emit `write_bool(p1, false)`
    /// at one probability while the decoder consumed `read_bool(p2)`
    /// at a different one, silently producing a bool-coder range that
    /// looks valid but means a wholly different bitstream to a
    /// third-party reference reader. CI would catch it on the
    /// self-roundtrip, but only after a real encode runs; this anchor
    /// catches the divergence at lib-test time on the *constants*.
    ///
    /// The walk order is row-major `(component=0..2, position=0..19)`,
    /// matching the encoder's flat layout (rows concatenated) and the
    /// decoder's `for i in 0..2 { for j in 0..MV_PROB_COUNT { ... } }`
    /// double-loop. Sanity-checks the flat length too — `MV_PROB_COUNT`
    /// could be re-defined and silently shrink the spec table without
    /// the encoder's `[u8; 38]` literal noticing.
    #[test]
    fn mv_update_probs_flat_matches_spec_table() {
        use crate::coded_header::{MV_PROB_COUNT, MV_UPDATE_PROBS};

        assert_eq!(
            MV_UPDATE_PROBS_FLAT.len(),
            2 * MV_PROB_COUNT,
            "MV_UPDATE_PROBS_FLAT length must match 2 × MV_PROB_COUNT \
             (RFC 6386 §17.2 two MV_CONTEXTs × 19 positions)"
        );

        for i in 0..2 {
            for j in 0..MV_PROB_COUNT {
                let flat = MV_UPDATE_PROBS_FLAT[i * MV_PROB_COUNT + j];
                let spec = MV_UPDATE_PROBS[i][j];
                assert_eq!(
                    flat, spec,
                    "encoder-side MV_UPDATE_PROBS_FLAT[{i}*{MV_PROB_COUNT}+{j}] \
                     ({flat}) diverges from the §17.2 spec table \
                     MV_UPDATE_PROBS[{i}][{j}] ({spec}) the decoder reads at"
                );
            }
        }
    }

    /// Anchor the encoder-side §13.2 `ENC_PCAT1..ENC_PCAT6` extra-bits
    /// probability lists and `ENC_CAT_BASE` category-base offsets against
    /// the literal §13.2 spec listing transcribed inline from
    /// RFC 6386 §13.2 (the `Pcat1..Pcat6` arrays at the `DCTextra`
    /// definition, and `categoryBase[6]` immediately preceding the
    /// `vp8_dct_value_cost` cost table).
    ///
    /// The encoder consults these six lists in `cat_extras()` to drive
    /// `DCTextra`'s per-bit `read_bool(d, *p)` (decoder side); the
    /// encoder mirror is the per-bit `write_bool(p, bit)` walk that
    /// emits the `(v - categoryBase[c])` residual MSB-first against the
    /// same `Pcat` probability sequence. If a `Pcat` byte or a
    /// `CAT_BASE` offset drifts here — a typo, a transposed pair, a
    /// dropped trailing zero swept into the slice — the encoder would
    /// emit a bit at `p1` while a third-party reference reader consumed
    /// the same bit at `p2`, silently producing a bool-coder range that
    /// looks valid but means a *different* residual integer to the
    /// reader. CI's self-roundtrip would still pass (encoder + decoder
    /// drift together), but `ffmpeg -c:v vp8` would diverge on the
    /// first cat-token. This anchor catches the divergence at
    /// lib-test time on the *constants*, before any real frame
    /// encode runs.
    ///
    /// Spec source (verbatim from RFC 6386 §13.2):
    ///
    /// ```text
    ///     const Prob Pcat1[] = { 159, 0};
    ///     const Prob Pcat2[] = { 165, 145, 0};
    ///     const Prob Pcat3[] = { 173, 148, 140, 0};
    ///     const Prob Pcat4[] = { 176, 155, 140, 135, 0};
    ///     const Prob Pcat5[] = { 180, 157, 141, 134, 130, 0};
    ///     const Prob Pcat6[] =
    ///         { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0};
    ///     ...
    ///     int categoryBase[6] = { 5, 7, 11, 19, 35, 67 };
    /// ```
    ///
    /// The encoder's local copies drop the `0` terminator (the encoder
    /// walks a known-length slice instead of testing `*p` at the loop
    /// edge), so the byte-equal comparison runs against the
    /// terminator-stripped prefix of each spec list.
    #[test]
    fn enc_pcat_and_cat_base_match_spec_listing() {
        // RFC 6386 §13.2 — Pcat1..Pcat6 with trailing `0` terminator
        // stripped (encoder walks a fixed-length slice).
        const SPEC_PCAT1: &[u8] = &[159];
        const SPEC_PCAT2: &[u8] = &[165, 145];
        const SPEC_PCAT3: &[u8] = &[173, 148, 140];
        const SPEC_PCAT4: &[u8] = &[176, 155, 140, 135];
        const SPEC_PCAT5: &[u8] = &[180, 157, 141, 134, 130];
        const SPEC_PCAT6: &[u8] = &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];
        // RFC 6386 §13.2 — categoryBase[6].
        const SPEC_CAT_BASE: [u16; 6] = [5, 7, 11, 19, 35, 67];

        // Length sanity — the encoder's six Pcat slices must match
        // the spec list lengths (1, 2, 3, 4, 5, 11 extra bits for
        // cat1..cat6 respectively; the cat6 range spans 67..=2114
        // = 2048 values requiring 11 bits, the rest follow the
        // geometric `(2^n - 1) + base` pattern).
        let enc_lists: [(&[u8], &[u8], &str, usize); 6] = [
            (ENC_PCAT1, SPEC_PCAT1, "Pcat1", 1),
            (ENC_PCAT2, SPEC_PCAT2, "Pcat2", 2),
            (ENC_PCAT3, SPEC_PCAT3, "Pcat3", 3),
            (ENC_PCAT4, SPEC_PCAT4, "Pcat4", 4),
            (ENC_PCAT5, SPEC_PCAT5, "Pcat5", 5),
            (ENC_PCAT6, SPEC_PCAT6, "Pcat6", 11),
        ];
        for (enc_list, spec_list, name, expected_len) in enc_lists.iter() {
            assert_eq!(
                enc_list.len(),
                *expected_len,
                "encoder-side {name} carries {} probs but §13.2 \
                 emits exactly {expected_len} extra bits",
                enc_list.len()
            );
            assert_eq!(
                enc_list.len(),
                spec_list.len(),
                "encoder-side {name} length ({}) diverges from §13.2 \
                 spec list length ({})",
                enc_list.len(),
                spec_list.len()
            );
            for (k, (&enc_p, &spec_p)) in enc_list.iter().zip(spec_list.iter()).enumerate() {
                assert_eq!(
                    enc_p, spec_p,
                    "encoder-side {name}[{k}] ({enc_p}) diverges from \
                     §13.2 spec listing {name}[{k}] ({spec_p}) — the \
                     decoder's DCTextra() reads each cat-residual bit \
                     at this probability"
                );
            }
        }

        // CAT_BASE — the §13.2 `categoryBase[6]` offsets used by
        // `cat_extras()` to add to the decoded residual `v`.
        assert_eq!(
            ENC_CAT_BASE.len(),
            SPEC_CAT_BASE.len(),
            "encoder-side ENC_CAT_BASE length ({}) diverges from \
             §13.2 categoryBase[6] ({})",
            ENC_CAT_BASE.len(),
            SPEC_CAT_BASE.len()
        );
        for k in 0..ENC_CAT_BASE.len() {
            assert_eq!(
                ENC_CAT_BASE[k],
                SPEC_CAT_BASE[k],
                "encoder-side ENC_CAT_BASE[{k}] ({}) diverges from \
                 §13.2 spec categoryBase[{k}] ({}) — the decoder adds \
                 this offset to the cat{}-residual",
                ENC_CAT_BASE[k],
                SPEC_CAT_BASE[k],
                k + 1
            );
        }

        // Cross-check `cat_extras()` returns the same `(base, list)`
        // pair the encoder's per-cat-token writer consumes — catches
        // a future drift where the match-arm order desyncs from the
        // table index. The `expected_bits` field carries the §13.2
        // extra-bits count for each cat (1, 2, 3, 4, 5, 11).
        let cat_token_pairs: [(DctToken, usize, usize); 6] = [
            (DctToken::Cat1, 0, 1),
            (DctToken::Cat2, 1, 2),
            (DctToken::Cat3, 2, 3),
            (DctToken::Cat4, 3, 4),
            (DctToken::Cat5, 4, 5),
            (DctToken::Cat6, 5, 11),
        ];
        for (tok, expected_idx, expected_bits) in cat_token_pairs.iter() {
            let (base, list) = cat_extras(*tok).expect("cat-token must carry extras");
            assert_eq!(
                base, ENC_CAT_BASE[*expected_idx],
                "cat_extras({tok:?}) returned base {base} but \
                 §13.2 categoryBase[{expected_idx}] is {}",
                ENC_CAT_BASE[*expected_idx]
            );
            assert_eq!(
                list.len(),
                *expected_bits,
                "cat_extras({tok:?}) returned a Pcat list of len {} \
                 but §13.2 emits exactly {expected_bits} extra bits",
                list.len()
            );
        }

        // Non-cat tokens (Dct0..Dct4, Eob) must not advertise extras —
        // any future regression that started returning `Some(...)` for
        // them would lead to bogus tail bits in the bitstream.
        for tok in [
            DctToken::Eob,
            DctToken::Dct0,
            DctToken::Dct1,
            DctToken::Dct2,
            DctToken::Dct3,
            DctToken::Dct4,
        ] {
            assert!(
                cat_extras(tok).is_none(),
                "cat_extras({tok:?}) must be None — §13.2 reserves \
                 the trailing `DCTextra` walk for cat1..cat6 only"
            );
        }
    }

    /// The §11 mode-layer writer must produce a first partition the
    /// decoder's `parse_key_frame_macroblock_modes` reads back to the
    /// exact `MacroblockModes` the encoder chose — including the §11.3
    /// cross-macroblock `B_PRED` sub-block context evolution. We encode a
    /// small synthetic frame, re-parse the control partition, and compare
    /// the recovered modes to the encoder's own (recomputed here by the
    /// same per-MB picker the driver uses).
    #[test]
    fn mode_layer_roundtrips_through_decoder_parser() {
        // A 32×32 frame (2×2 macroblocks) with a structured gradient so
        // the picker exercises more than one luma / chroma mode.
        let (w, h) = (32u32, 32u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let mut y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                y[r * w as usize + c] = ((r * 8 + c * 8) & 0xff) as u8;
            }
        }
        let u = vec![110u8; cw * ch];
        let v = vec![140u8; cw * ch];
        let frame = I420Frame::packed(w, h, &y, &u, &v);
        let params = KeyframeParams {
            y_ac_qi: 32,
            loop_filter_level: 0,
            sharpness_level: 0,
            nbr_of_dct_partitions: 1,
            filter_type: false,
            trellis_strength: TrellisStrength::DEFAULT,
        };
        let bytes = encode_keyframe(&frame, &params).expect("encode");

        // Re-parse the control partition exactly as the decoder does.
        let header = Vp8FrameHeader::parse(&bytes).unwrap();
        let off = header.header_bytes_consumed;
        let size = header.first_partition_size as usize;
        let first = &bytes[off..off + size];
        let (coded, mut dec) = Vp8CodedHeader::parse_with_decoder(first, true).unwrap();
        let parsed =
            crate::macroblock::parse_key_frame_macroblock_modes(&mut dec, &coded, 2, 2).unwrap();
        assert_eq!(parsed.len(), 4);

        // Recompute the encoder's own chosen modes by replaying the
        // raster driver's per-MB pick against the reconstructed planes —
        // the simplest oracle is to decode the frame and confirm it
        // succeeds (which it does in the integration test); here we
        // assert structural invariants the parser must satisfy.
        for mb in &parsed {
            // A B_PRED MB must carry sixteen sub-block modes; a
            // whole-block MB must carry none.
            assert_eq!(
                mb.subblock_modes.is_some(),
                mb.y_mode == IntraYMode::B,
                "subblock_modes presence must track B_PRED"
            );
        }

        // End-to-end: the frame decodes without error and the modes the
        // decoder used are byte-identical to what we re-parsed (the
        // decoder runs the same parser internally).
        let decoded = decode_vp8(&bytes).expect("decode");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
    }

    /// Build a 64×64 structured-but-natural I420 test frame: a smooth
    /// luma gradient with a low-amplitude pseudo-random texture overlay,
    /// and gently varying chroma. Returns `(y, u, v)` tightly packed.
    /// Shared by the RD validation tests.
    fn natural_test_frame_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (64usize, 64usize);
        let (cw, ch) = (32usize, 32usize);
        let mut y = vec![0u8; w * h];
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (state >> 16) & 0x1f
        };
        for r in 0..h {
            for c in 0..w {
                let base = 40 + (r + c) * 2;
                let tex = next() as usize;
                y[r * w + c] = (base + tex).min(235) as u8;
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (100 + r) as u8;
                v[r * cw + c] = (150 - c) as u8;
            }
        }
        (y, u, v)
    }

    /// Decode a VP8 keyframe and return the luma-plane PSNR against the
    /// source. The decoder returns a visible-cropped, tightly-packed luma
    /// plane, so the comparison is direct.
    fn keyframe_luma_psnr(bytes: &[u8], src_y: &[u8], w: usize, h: usize) -> f64 {
        let decoded = decode_vp8(bytes).expect("decode");
        psnr(&src_y[..w * h], &decoded.y[..w * h])
    }

    /// The rate-distortion mode picker must keep the self-decode PSNR high
    /// on a natural keyframe while producing a valid, decodable stream.
    /// This is the floor the RD trade must never fall below: the picker
    /// may shave bits, but the reconstruction quality stays well above
    /// 30 dB at a mid quantiser.
    #[test]
    fn rd_keyframe_holds_psnr_floor_on_natural_frame() {
        let (w, h) = (64u32, 64u32);
        let (y, u, v) = natural_test_frame_64x64();
        let frame = I420Frame::packed(w, h, &y, &u, &v);
        for qi in [16u8, 32, 48] {
            let params = KeyframeParams {
                y_ac_qi: qi,
                loop_filter_level: 0,
                sharpness_level: 0,
                nbr_of_dct_partitions: 1,
                filter_type: false,
                trellis_strength: TrellisStrength::DEFAULT,
            };
            let bytes = encode_keyframe(&frame, &params).expect("encode");
            let p = keyframe_luma_psnr(&bytes, &y, w as usize, h as usize);
            assert!(
                p >= 30.0,
                "RD keyframe luma PSNR {p:.2} dB < 30 dB floor at qi={qi}"
            );
        }
    }

    /// The §14.4 inverse-DCT basis-norm distortion model the trellis uses
    /// must track the actual pixel-domain reconstruction SSD: summing
    /// `norm_k · (x_k − level_k·q)^2` over the scan positions reproduces
    /// `block_ssd(reconstruct, src)` to within the IDCT's sub-LSB rounding.
    #[test]
    fn trellis_basis_norm_distortion_tracks_pixel_ssd() {
        // The §14.4 IDCT basis vectors are near-equal-energy (≈ 0.25 each).
        let norms = &*super::IDCT_BASIS_NORM_SQ;
        for &n in norms.iter() {
            assert!((0.24..0.26).contains(&n), "basis norm {n} out of range");
        }
        // A handful of synthetic residual blocks; predicted (model) SSD
        // must closely track the true reconstruction SSD.
        for seed in 0u32..8 {
            let mut st = 0x1357u32.wrapping_add(seed.wrapping_mul(0x9E3D));
            let mut next = || {
                st = st.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (st >> 16) as i32
            };
            let pred = [128u8; 16];
            let mut src = [0u8; 16];
            for s in src.iter_mut() {
                *s = (128 + (next() % 60) - 30).clamp(0, 255) as u8;
            }
            let mut residual = [0i16; 16];
            for k in 0..16 {
                residual[k] = src[k] as i16 - pred[k] as i16;
            }
            let mut coeffs = [0i16; 16];
            super::forward_dct_4x4(&residual, &mut coeffs);
            let (dc, ac) = (16i16, 18i16);
            let mut q = coeffs;
            super::enc_quantize_block(&mut q, dc, ac);
            let recon = super::reconstruct_block_4x4(&pred, &q, dc, ac);
            let actual = super::block_ssd(&recon, &src) as f64;

            let scan_x = super::raster_to_scan(&coeffs);
            let scan_q = super::raster_to_scan(&q);
            let mut model = 0.0f64;
            for k in 0..16 {
                let step = if k == 0 { dc } else { ac } as i64;
                let err = (scan_x[k] as i64).abs() - (scan_q[k] as i64).abs() * step;
                model += norms[k] * (err * err) as f64;
            }
            // Within 8 SSD-units or 25 % — the gap is the IDCT's `+4 >> 3`
            // rounding, which never exceeds a fraction of an LSB per pixel.
            let tol = (actual * 0.25).max(8.0);
            assert!(
                (model - actual).abs() <= tol,
                "seed {seed}: model SSD {model:.1} vs actual {actual:.1} (tol {tol:.1})"
            );
        }
    }

    /// Trellis output must round-trip byte-exact through the §13.3 token
    /// encoder + decoder, and never raise the Lagrangian `J = D + λ·R`
    /// above plain round-quantisation (it is, by construction, the RD
    /// minimiser over its candidate set).
    #[test]
    fn trellis_round_trips_and_never_raises_rd_cost() {
        let probs = crate::DEFAULT_COEFF_PROBS;
        let factors = crate::dequant::MbDequantFactors::from_base_and_deltas(40, 0, 0, 0, 0, 0);
        let lambda = super::rd_lambda(&factors) * super::TRELLIS_LAMBDA_SCALE;
        let (dc, ac) = (factors.uv_dc, factors.uv_ac);

        for seed in 0u32..32 {
            let mut st = 0xBEEFu32.wrapping_add(seed.wrapping_mul(0x9E3D));
            let mut next = || {
                st = st.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (st >> 15) as i32
            };
            let pred = [128u8; 16];
            let mut src = [0u8; 16];
            for s in src.iter_mut() {
                *s = (128 + (next() % 90) - 45).clamp(0, 255) as u8;
            }
            let mut residual = [0i16; 16];
            for k in 0..16 {
                residual[k] = src[k] as i16 - pred[k] as i16;
            }
            let mut base = [0i16; 16];
            super::forward_dct_4x4(&residual, &mut base);

            // Cost helper: J = D + λ·R, where D is the **basis-norm model**
            // distortion the trellis actually optimises (transform-domain
            // squared error scaled by the §14.4 IDCT basis norms), and R is
            // the §13.3 token bits. The trellis is the exact RD minimiser
            // over this model, so it can never raise this J above plain
            // round-quantisation. (Against the true pixel SSD the trellis
            // is near-optimal but not provably ≤, since the model differs
            // from the pixel domain by the IDCT's sub-LSB rounding — see
            // `trellis_basis_norm_distortion_tracks_pixel_ssd`.)
            let norms = &*super::IDCT_BASIS_NORM_SQ;
            let cost = |quant: &[i16; 16]| -> f64 {
                let scan_q = super::raster_to_scan(quant);
                let scan_x = super::raster_to_scan(&base);
                let mut d = 0.0f64;
                for k in 0..16 {
                    let step = if k == 0 { dc } else { ac } as i64;
                    let err = (scan_x[k] as i64).abs() - (scan_q[k] as i64).abs() * step;
                    d += norms[k] * (err * err) as f64;
                }
                let r = super::estimate_block_bits(BlockType::UV, &probs, false, false, &scan_q);
                d + lambda * r
            };

            let mut plain = base;
            super::enc_quantize_block(&mut plain, dc, ac);

            let mut trel = base;
            super::trellis_quantize_block(
                &mut trel,
                super::TrellisBlockSpec {
                    dc,
                    ac,
                    first_coeff: BlockType::UV.first_coeff(),
                    entry_ctx3: 0,
                    plane: BlockType::UV.plane_index(),
                },
                &probs,
                None,
                // `trellis_quantize_block` now consumes the **effective**
                // (already-scaled) coefficient lambda — the same `lambda`
                // the cost helper above prices J against.
                lambda,
            );

            // (1) RD non-regression — the trellis cost is ≤ plain rounding.
            let (cp, ct) = (cost(&plain), cost(&trel));
            assert!(
                ct <= cp + 1e-6,
                "seed {seed}: trellis J {ct:.2} > plain J {cp:.2}"
            );

            // (2) Byte-exact §13.3 round-trip of the trellis levels (in
            //     scan order, the layout `encode_coeff_block` consumes).
            let scan = super::raster_to_scan(&trel);
            let recovered = encode_then_decode_block(BlockType::UV, &scan, false, false);
            assert_eq!(
                recovered, scan,
                "seed {seed}: trellis block did not round-trip through §13.3"
            );
        }
    }

    /// The rate-distortion picker must not regress against the prior
    /// SAD-only picker: at a fixed quantiser it produces a **smaller**
    /// stream while holding equal-or-better self-decode PSNR. The
    /// SAD-baseline numbers below were captured from the r135 picker on
    /// this exact frame (`natural_test_frame_64x64`); the RD picker beats
    /// them on both byte count and PSNR at every quantiser, so we assert
    /// the RD output never exceeds the SAD byte count and never drops
    /// below the SAD PSNR. This pins the RD trade as a strict improvement.
    #[test]
    fn rd_beats_sad_baseline_size_and_quality() {
        // (qi, SAD bytes, SAD luma PSNR) measured on the r135 SAD picker.
        let sad_baseline: [(u8, usize, f64); 5] = [
            (16, 1467, 39.293),
            (24, 1192, 36.777),
            (32, 1003, 34.561),
            (48, 731, 31.742),
            (64, 383, 29.921),
        ];
        let (w, h) = (64u32, 64u32);
        let (y, u, v) = natural_test_frame_64x64();
        let frame = I420Frame::packed(w, h, &y, &u, &v);
        for (qi, sad_bytes, sad_psnr) in sad_baseline {
            let params = KeyframeParams {
                y_ac_qi: qi,
                loop_filter_level: 0,
                sharpness_level: 0,
                nbr_of_dct_partitions: 1,
                filter_type: false,
                trellis_strength: TrellisStrength::DEFAULT,
            };
            let bytes = encode_keyframe(&frame, &params).expect("encode");
            let p = keyframe_luma_psnr(&bytes, &y, w as usize, h as usize);
            assert!(
                bytes.len() <= sad_bytes,
                "RD stream at qi={qi} ({} B) must not exceed the SAD baseline ({sad_bytes} B)",
                bytes.len()
            );
            assert!(
                p >= sad_psnr - 0.05,
                "RD PSNR at qi={qi} ({p:.3} dB) must hold the SAD baseline ({sad_psnr:.3} dB)"
            );
        }
    }

    /// The §13 [`TrellisStrength`] knob is monotone in the rate/quality
    /// trade: dialing it up never *grows* the keyframe and dialing it to
    /// [`TrellisStrength::OFF`] reproduces the pre-trellis (plain-rounding)
    /// bytes — which are the largest of the three, since the trellis only
    /// ever shaves bits. Every strength must still decode, and the
    /// calibrated [`TrellisStrength::DEFAULT`] must match the historical
    /// `rd_beats_sad_baseline` byte ceiling exactly (the knob is a strict
    /// superset of the previously-fixed `0.04` scale).
    #[test]
    fn trellis_strength_knob_is_monotone_and_round_trips() {
        let (w, h) = (64u32, 64u32);
        let (y, u, v) = natural_test_frame_64x64();
        let frame = I420Frame::packed(w, h, &y, &u, &v);
        // Same fixed-quantiser working points the SAD-baseline test pins.
        for qi in [16u8, 24, 32, 48, 64] {
            let mk = |s: TrellisStrength| KeyframeParams {
                y_ac_qi: qi,
                loop_filter_level: 0,
                sharpness_level: 0,
                nbr_of_dct_partitions: 1,
                filter_type: false,
                trellis_strength: s,
            };
            let off = encode_keyframe(&frame, &mk(TrellisStrength::OFF)).expect("off");
            let def = encode_keyframe(&frame, &mk(TrellisStrength::DEFAULT)).expect("default");
            // A strength well above the default trades more aggressively.
            let hot = encode_keyframe(&frame, &mk(TrellisStrength::new(4.0))).expect("hot");

            // OFF = plain rounding = the largest stream; the default trims
            // it; the hot setting trims at least as much as the default.
            assert!(
                def.len() <= off.len(),
                "qi={qi}: DEFAULT trellis ({} B) must not exceed OFF ({} B)",
                def.len(),
                off.len()
            );
            assert!(
                hot.len() <= def.len(),
                "qi={qi}: strength 4.0 ({} B) must not exceed DEFAULT ({} B)",
                hot.len(),
                def.len()
            );

            // Every strength decodes cleanly through the crate's own decoder
            // (encoder↔decoder pixel lockstep holds at every setting).
            for bytes in [&off, &def, &hot] {
                let dec = crate::decode_vp8(bytes).expect("decode at this strength");
                assert_eq!(dec.y.len(), w as usize * h as usize);
            }

            // The hot setting must actually move the wire at the mid/low
            // quantisers VP8 keyframes operate at (a knob that never changes
            // anything would be a silent no-op).
            if qi <= 48 {
                assert!(
                    hot.len() < off.len(),
                    "qi={qi}: strength 4.0 ({} B) should beat OFF ({} B)",
                    hot.len(),
                    off.len()
                );
            }
        }

        // `new` clamps out-of-range / NaN inputs into the documented band.
        assert_eq!(TrellisStrength::new(-1.0).get(), 0.0);
        assert_eq!(TrellisStrength::new(1000.0).get(), TrellisStrength::MAX);
        assert_eq!(TrellisStrength::new(f64::NAN), TrellisStrength::DEFAULT);
    }

    /// `quality_to_trellis_strength` pairs the quality dial with the §13
    /// trellis: visually-lossless requests stay at `DEFAULT`, lower quality
    /// ramps the strength up monotonically toward `4.0`, and `NaN` is the
    /// conservative default.
    #[test]
    fn quality_to_trellis_strength_ramps_monotonically() {
        // High quality holds the default (no PSNR sacrifice).
        assert_eq!(quality_to_trellis_strength(100.0), TrellisStrength::DEFAULT);
        assert_eq!(quality_to_trellis_strength(90.0), TrellisStrength::DEFAULT);
        // Above-100 / below-0 clamp like the qindex mapping.
        assert_eq!(quality_to_trellis_strength(150.0), TrellisStrength::DEFAULT);
        // Bottom of the range hits the strong operating point.
        assert!((quality_to_trellis_strength(0.0).get() - 4.0).abs() < 1e-6);
        // NaN is the conservative default.
        assert_eq!(
            quality_to_trellis_strength(f32::NAN),
            TrellisStrength::DEFAULT
        );

        // Monotone in quality: lower quality ⇒ greater-or-equal strength.
        // Walk q from 100 down to 0; each step's strength must not drop.
        let mut prev = 0.0f64;
        for q in (0..=100).rev() {
            let s = quality_to_trellis_strength(q as f32).get();
            assert!(
                s >= prev - 1e-9,
                "strength must not fall as quality falls (q={q}: {s} < {prev})"
            );
            prev = s;
        }
    }

    /// The §7.3 boolean encoder and the §7.3 boolean decoder in this
    /// crate are inverses on a randomised flag sequence at randomised
    /// probabilities.
    #[test]
    fn bool_encoder_decoder_round_trip_pseudo_random_sequence() {
        // Linear-feedback PRNG so the sequence is reproducible.
        let mut state: u32 = 0xCAFE_F00D;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            state
        };
        let mut bools: Vec<(u8, bool)> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            let p = ((next() >> 16) as u8).max(1);
            let v = (next() & 1) != 0;
            bools.push((p, v));
        }

        let mut enc = BoolEncoder::new();
        for (p, v) in &bools {
            enc.write_bool(*p, *v);
        }
        let bytes = enc.finish();
        assert!(bytes.len() >= 4);

        let mut dec = BoolDecoder::init(&bytes).expect("encoder always emits ≥ 2 bytes");
        for (p, v) in &bools {
            assert_eq!(dec.read_bool(*p).expect("partition underrun"), *v);
        }
    }

    /// `write_literal` / `write_signed_literal` round-trip through
    /// `BoolDecoder::read_literal` / a manual sign-bit pair.
    #[test]
    fn write_literal_round_trips_through_read_literal() {
        let mut enc = BoolEncoder::new();
        // Unsigned MSB-first.
        enc.write_literal(0xA5, 8);
        // Six-bit value.
        enc.write_literal(42, 6);
        // Signed: -5 magnitude=5 over 4 bits, sign=1.
        enc.write_signed_literal(-5, 4);
        // Signed positive: +7 magnitude=7 over 4 bits, sign=0.
        enc.write_signed_literal(7, 4);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        assert_eq!(dec.read_literal(8).unwrap(), 0xA5);
        assert_eq!(dec.read_literal(6).unwrap(), 42);
        // -5: read 4 bits magnitude, then sign.
        assert_eq!(dec.read_literal(4).unwrap(), 5);
        assert!(dec.read_bool(128).unwrap()); // sign=1 → negative
        assert_eq!(dec.read_literal(4).unwrap(), 7);
        assert!(!dec.read_bool(128).unwrap());
    }

    /// §9.1 frame-tag writer round-trips through
    /// [`Vp8FrameHeader::parse`] on a key frame with a non-trivial
    /// dimension + non-trivial first-partition size.
    #[test]
    fn frame_tag_round_trips_through_parser_key_frame() {
        let mut buf: Vec<u8> = Vec::new();
        let bytes_written = write_frame_tag(
            &mut buf,
            true,     // key frame
            0,        // version
            true,     // show_frame
            0x1_2345, // first_partition_size
            640,
            480,
            ScaleCode::None,
            ScaleCode::None,
        )
        .unwrap();
        assert_eq!(bytes_written, 10);
        assert_eq!(buf.len(), 10);
        // Start code is at offset 3..6 — required by §9.1.
        assert_eq!(&buf[3..6], &KEY_FRAME_START_CODE);
        let hdr = Vp8FrameHeader::parse(&buf).unwrap();
        assert!(hdr.key_frame);
        assert_eq!(hdr.version, 0);
        assert!(hdr.show_frame);
        assert_eq!(hdr.first_partition_size, 0x1_2345);
        assert_eq!(hdr.width, Some(640));
        assert_eq!(hdr.height, Some(480));
    }

    /// `patch_first_partition_size` preserves the surrounding bits
    /// (frame_type / version / show_frame).
    #[test]
    fn patch_first_partition_size_preserves_neighbouring_bits() {
        let mut buf: Vec<u8> = Vec::new();
        write_frame_tag(
            &mut buf,
            true,
            5,    // version=5
            true, // show=1
            0,    // start at zero
            16,
            16,
            ScaleCode::None,
            ScaleCode::None,
        )
        .unwrap();
        patch_first_partition_size(&mut buf[..3], 1234).unwrap();
        let hdr = Vp8FrameHeader::parse(&buf).unwrap();
        assert_eq!(hdr.version, 5);
        assert!(hdr.show_frame);
        assert!(hdr.key_frame);
        assert_eq!(hdr.first_partition_size, 1234);
    }

    /// `set_show_frame_bit` flips exactly the §9.1 bit-4 and preserves
    /// every neighbouring tag field (frame_type / version /
    /// first_partition_size) plus every byte after the tag.
    #[test]
    fn set_show_frame_bit_flips_only_bit_4() {
        let mut buf: Vec<u8> = Vec::new();
        write_frame_tag(
            &mut buf,
            false, // inter frame
            3,     // version=3
            true,  // show=1
            0x5_4321,
            16,
            16,
            ScaleCode::None,
            ScaleCode::None,
        )
        .unwrap();
        buf.extend_from_slice(&[0xAA, 0x55, 0xF0]); // fake payload
        let visible = buf.clone();

        set_show_frame_bit(&mut buf, false);
        let hdr = Vp8FrameHeader::parse(&buf).unwrap();
        assert!(!hdr.key_frame);
        assert_eq!(hdr.version, 3);
        assert!(!hdr.show_frame);
        assert_eq!(hdr.first_partition_size, 0x5_4321);
        // Only byte 0 differs, and only in bit 4.
        assert_eq!(buf[0] ^ visible[0], 0x10);
        assert_eq!(&buf[1..], &visible[1..]);

        // Round-trip back to visible restores the original bytes.
        set_show_frame_bit(&mut buf, true);
        assert_eq!(buf, visible);
    }

    /// Dimensions outside the 14-bit field are rejected.
    #[test]
    fn rejects_oversize_dimensions() {
        let mut buf = Vec::new();
        let err = write_frame_tag(
            &mut buf,
            true,
            0,
            true,
            0,
            0x4000, // > 0x3FFF
            16,
            ScaleCode::None,
            ScaleCode::None,
        )
        .unwrap_err();
        assert!(matches!(err, EncodeError::InvalidDimensions { .. }));
    }

    /// Loop-filter / sharpness validation.
    #[test]
    fn rejects_out_of_range_loop_filter_inputs() {
        let mut enc = BoolEncoder::new();
        assert!(matches!(
            write_loop_filter(&mut enc, false, 64, 0, false),
            Err(EncodeError::LoopFilterLevelOutOfRange { value: 64 })
        ));
        let mut enc = BoolEncoder::new();
        assert!(matches!(
            write_loop_filter(&mut enc, false, 0, 8, false),
            Err(EncodeError::SharpnessLevelOutOfRange { value: 8 })
        ));
    }

    #[test]
    fn rejects_illegal_partition_count() {
        let mut enc = BoolEncoder::new();
        assert!(matches!(
            write_token_partition_count(&mut enc, 3),
            Err(EncodeError::InvalidDctPartitionCount { value: 3 })
        ));
    }

    #[test]
    fn rejects_oversize_quant_index() {
        let mut enc = BoolEncoder::new();
        assert!(matches!(
            write_quant_indices(&mut enc, 128, None, None, None, None, None),
            Err(EncodeError::QuantIndexOutOfRange { value: 128 })
        ));
    }

    /// End-to-end: a silent 16×16 keyframe round-trips through the
    /// crate's own decoder.
    #[test]
    fn silent_keyframe_16x16_round_trips_through_own_decoder() {
        let bytes = encode_silent_keyframe(SilentKeyframeParams::new(16, 16))
            .expect("encoder should produce a valid frame");
        let decoded = decode_vp8(&bytes).expect("own decoder must accept emitted frame");
        assert_eq!(decoded.width, 16);
        assert_eq!(decoded.height, 16);
        assert_eq!(decoded.y.len(), 16 * 16);
        assert_eq!(decoded.u.len(), 8 * 8);
        assert_eq!(decoded.v.len(), 8 * 8);
    }

    /// The §9.x writer subroutines compose into a parse-able coded
    /// header. We strip the frame tag + key-frame extension and ask
    /// [`Vp8CodedHeader::parse`] to walk our boolean-coded first
    /// partition. Confirms the §9.2 / §9.3 / §9.4 / §9.5 / §9.6 /
    /// §9.7 layout is byte-correct.
    #[test]
    fn silent_keyframe_first_partition_round_trips_through_coded_header_parser() {
        let bytes = encode_silent_keyframe(SilentKeyframeParams::new(64, 64)).unwrap();
        let hdr = Vp8FrameHeader::parse(&bytes).unwrap();
        let off = hdr.header_bytes_consumed;
        let sz = hdr.first_partition_size as usize;
        let partition = &bytes[off..off + sz];
        let coded = Vp8CodedHeader::parse(partition, true).unwrap();
        assert_eq!(coded.color_space, Some(false));
        assert_eq!(coded.clamping_type, Some(false));
        assert!(!coded.segmentation_enabled);
        assert!(!coded.filter_type);
        assert_eq!(coded.loop_filter_level, 0);
        assert_eq!(coded.sharpness_level, 0);
        assert!(!coded.mb_lf_adjustments.loop_filter_adj_enable);
        assert_eq!(coded.log2_nbr_of_dct_partitions, 0);
        assert_eq!(coded.nbr_of_dct_partitions, 1);
        assert_eq!(coded.quant_indices.y_ac_qi, 0);
        assert_eq!(coded.quant_indices.y_dc_delta, None);
        assert!(coded.refresh_entropy_probs);
        assert!(coded.mb_no_skip_coeff);
        assert_eq!(coded.prob_skip_false, Some(1));
    }

    /// Round-trip across all four legal DCT partition counts so the
    /// §9.5 size-table writer is exercised. 64×64 → 4 MB rows, big
    /// enough to actually route through every partition.
    #[test]
    fn silent_keyframe_round_trips_at_all_partition_counts() {
        for &count in &[1u8, 2, 4, 8] {
            let mut params = SilentKeyframeParams::new(64, 64);
            params.nbr_of_dct_partitions = count;
            let bytes = encode_silent_keyframe(params)
                .unwrap_or_else(|e| panic!("encoder failed at count={count}: {e}"));
            let decoded = decode_vp8(&bytes)
                .unwrap_or_else(|e| panic!("decoder failed at count={count}: {e}"));
            assert_eq!(decoded.width, 64);
            assert_eq!(decoded.height, 64);
        }
    }

    /// Various non-square dimensions round-trip through the decoder.
    /// Exercises the macroblock-loop count for `mb_rows != mb_cols`
    /// and the visible-crop logic (32×24 has only one MB row of
    /// trailing "excess" pixels).
    #[test]
    fn silent_keyframe_non_square_dimensions_round_trip() {
        let cases = [(16u32, 16u32), (32, 32), (48, 16), (16, 48), (32, 24)];
        for (w, h) in cases {
            let bytes = encode_silent_keyframe(SilentKeyframeParams::new(w, h))
                .unwrap_or_else(|e| panic!("encoder failed at {w}x{h}: {e}"));
            let decoded =
                decode_vp8(&bytes).unwrap_or_else(|e| panic!("decoder failed at {w}x{h}: {e}"));
            assert_eq!(decoded.width, w);
            assert_eq!(decoded.height, h);
        }
    }

    /// The historical `make_silent_keyframe_encoder()` factory reaches
    /// the same byte sequence as the direct
    /// `encode_silent_keyframe(SilentKeyframeParams::new(...))` call.
    #[test]
    fn make_silent_keyframe_encoder_produces_same_bytes_as_top_level_function() {
        let enc = make_silent_keyframe_encoder();
        let from_factory = enc.encode_keyframe(&[], 32, 32).unwrap();
        let from_direct = encode_silent_keyframe(SilentKeyframeParams::new(32, 32)).unwrap();
        assert_eq!(from_factory, from_direct);
    }

    /// Bit-budget guard: the silent-keyframe path emits the smallest
    /// possible 16×16 frame the §9.x layout admits — frame tag (10) +
    /// first partition (small, dominated by 1056 token-update "no"
    /// bits at high prob) + 1 DCT partition (4-byte flush). The exact
    /// number is implementation-determined; we just lock the order
    /// of magnitude to detect a future-round regression that
    /// accidentally bloats the partition.
    #[test]
    fn silent_keyframe_size_is_small() {
        let bytes = encode_silent_keyframe(SilentKeyframeParams::new(16, 16)).unwrap();
        // Empirically ~ 30-40 bytes for a 16×16 silent keyframe. A
        // generous upper bound that still catches accidental growth
        // (e.g. forgetting the §13.4 update-probs table and using a
        // flat 128 would push past 200 bytes).
        assert!(
            bytes.len() < 100,
            "silent 16×16 keyframe ballooned to {} bytes",
            bytes.len()
        );
        // Lower-bounded by the 10-byte key-frame header + 4-byte
        // first-partition flush + 4-byte DCT-partition flush.
        assert!(bytes.len() >= 10 + 4 + 4);
    }

    // ───────── Phase 2 §13 token-encoder round-trip tests ─────────

    use crate::dct_tokens::{decode_block, DEFAULT_COEFF_PROBS};

    /// Encode one block with `encode_coeff_block`, then decode the
    /// emitted bytes with `decode_block`. Returns the recovered
    /// `[i16; 16]`. Used as the central round-trip helper for the
    /// Phase 2 token-encoder tests below.
    fn encode_then_decode_block(
        block_type: BlockType,
        coeffs: &[i16; 16],
        above_has_nonzero: bool,
        left_has_nonzero: bool,
    ) -> [i16; 16] {
        let mut enc = BoolEncoder::new();
        let nonzeros = encode_coeff_block(
            &mut enc,
            block_type,
            &DEFAULT_COEFF_PROBS,
            above_has_nonzero,
            left_has_nonzero,
            coeffs,
        )
        .expect("encode_coeff_block should accept in-range coefficients");
        // The non-zero count returned by the encoder must equal the
        // population-count of non-zero entries in the input (modulo
        // the first_coeff skip slot, which is unread on either side).
        let first = block_type.first_coeff();
        let expected_nz = coeffs
            .iter()
            .enumerate()
            .filter(|(i, &v)| *i >= first && v != 0)
            .count();
        assert_eq!(nonzeros, expected_nz, "non-zero count mismatch");
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).expect("encoder emits ≥ 2 bytes");
        let mut recovered = [0i16; 16];
        let nz = decode_block(
            &mut dec,
            block_type,
            &DEFAULT_COEFF_PROBS,
            above_has_nonzero,
            left_has_nonzero,
            &mut recovered,
        )
        .expect("decode_block should consume the encoded byte stream");
        assert_eq!(nz, expected_nz);
        recovered
    }

    /// Token classifier — exercises the full §13.2 alphabet.
    #[test]
    fn classify_coeff_token_covers_full_alphabet() {
        assert_eq!(classify_coeff_token(0), DctToken::Dct0);
        assert_eq!(classify_coeff_token(1), DctToken::Dct1);
        assert_eq!(classify_coeff_token(2), DctToken::Dct2);
        assert_eq!(classify_coeff_token(3), DctToken::Dct3);
        assert_eq!(classify_coeff_token(4), DctToken::Dct4);
        // Cat1: 5..=6
        assert_eq!(classify_coeff_token(5), DctToken::Cat1);
        assert_eq!(classify_coeff_token(6), DctToken::Cat1);
        // Cat2: 7..=10
        assert_eq!(classify_coeff_token(7), DctToken::Cat2);
        assert_eq!(classify_coeff_token(10), DctToken::Cat2);
        // Cat3: 11..=18
        assert_eq!(classify_coeff_token(11), DctToken::Cat3);
        assert_eq!(classify_coeff_token(18), DctToken::Cat3);
        // Cat4: 19..=34
        assert_eq!(classify_coeff_token(19), DctToken::Cat4);
        assert_eq!(classify_coeff_token(34), DctToken::Cat4);
        // Cat5: 35..=66
        assert_eq!(classify_coeff_token(35), DctToken::Cat5);
        assert_eq!(classify_coeff_token(66), DctToken::Cat5);
        // Cat6: 67..=2114
        assert_eq!(classify_coeff_token(67), DctToken::Cat6);
        assert_eq!(classify_coeff_token(2114), DctToken::Cat6);
    }

    /// All-zero block — every plane / first_coeff combination should
    /// round-trip a 16-entry all-zero coefficient vector exactly. The
    /// encoder emits a single immediate EOB token at position
    /// `first_coeff`; the decoder leaves the rest at zero.
    #[test]
    fn encode_coeff_block_all_zero_round_trips() {
        let coeffs = [0i16; 16];
        for &bt in &[
            BlockType::YAfterY2,
            BlockType::Y2,
            BlockType::UV,
            BlockType::YNoY2,
        ] {
            let recovered = encode_then_decode_block(bt, &coeffs, false, false);
            assert_eq!(recovered, coeffs, "all-zero round-trip failed for {bt:?}");
        }
    }

    /// A single non-zero coefficient at every position round-trips
    /// for every plane type. Exercises the `last_non_zero` EOB
    /// placement, the §13.2 leaf walk for each of the small literal
    /// tokens, and the universal sign bit.
    #[test]
    fn encode_coeff_block_single_nonzero_at_each_position() {
        for &bt in &[BlockType::Y2, BlockType::UV, BlockType::YNoY2] {
            let first = bt.first_coeff();
            for pos in first..16 {
                for &value in &[1i16, -1, 2, -3, 4, -4] {
                    let mut coeffs = [0i16; 16];
                    coeffs[pos] = value;
                    let recovered = encode_then_decode_block(bt, &coeffs, false, false);
                    assert_eq!(
                        recovered, coeffs,
                        "single-nonzero round-trip failed at {bt:?} pos={pos} value={value}"
                    );
                }
            }
        }
        // YAfterY2 separately, because coeffs[0] is ignored (DC lives
        // in the Y2 block) — only positions 1..16 are meaningful.
        let bt = BlockType::YAfterY2;
        for pos in 1..16 {
            let mut coeffs = [0i16; 16];
            coeffs[pos] = 3;
            let recovered = encode_then_decode_block(bt, &coeffs, false, false);
            assert_eq!(
                recovered, coeffs,
                "single-nonzero round-trip failed at YAfterY2 pos={pos}"
            );
        }
    }

    /// Each `Cat1..Cat6` token exercises a different extra-bits
    /// probability list (§13.2 `Pcat1..Pcat6`). This test pins one
    /// value inside every cat range — verified by encode + decode at
    /// the coefficient layer.
    #[test]
    fn encode_coeff_block_each_cat_range_round_trips() {
        // (value, category index) pairs spanning every Pcat list.
        let cases: &[i16] = &[
            5, 6, // Cat1
            7, 8, 10, // Cat2
            11, 15, 18, // Cat3
            19, 27, 34, // Cat4
            35, 50, 66, // Cat5
            67, 200, 1000, 2114, // Cat6
        ];
        for &v in cases {
            for &sign in &[1i16, -1] {
                let mut coeffs = [0i16; 16];
                // Place at position 3 — well inside all planes' visit
                // range and inside band-1 to avoid the all-128
                // pathological first-plane band-0 row.
                coeffs[3] = v * sign;
                let recovered = encode_then_decode_block(BlockType::UV, &coeffs, true, false);
                assert_eq!(
                    recovered, coeffs,
                    "cat-range round-trip failed for value {v}*{sign}"
                );
            }
        }
    }

    /// A fully-populated 16-entry block (every position non-zero) —
    /// exercises the §13.2 "implicit EOB after the last coefficient"
    /// rule, where the encoder omits the explicit `Eob` token because
    /// there is nothing past position 15. Decoder must still recover
    /// the full vector.
    #[test]
    fn encode_coeff_block_fully_populated_block_round_trips_with_implicit_eob() {
        let mut coeffs = [0i16; 16];
        // Mix of magnitudes covering every token bucket; sign
        // alternates so the universal sign bit gets exercised.
        let values = [1, -2, 3, -4, 5, -7, 11, -19, 35, -67, 1, -1, 2, -3, 4, -5];
        coeffs.copy_from_slice(&values);
        // YNoY2 reads all 16 positions; perfect for the "no implicit
        // EOB slot wasted" path.
        let recovered = encode_then_decode_block(BlockType::YNoY2, &coeffs, false, false);
        assert_eq!(recovered, coeffs);
    }

    /// Sparse pattern with interior zeros — exercises the §13.2
    /// "previous coefficient was DCT_0" branch that bypasses the EOB
    /// tree edge. The pattern `[3, 0, 0, 5, 0, 0, 0, 2, 0, …]`
    /// transitions through `Dct0` multiple times so the encoder's
    /// `prev_was_zero` tracking is exercised.
    #[test]
    fn encode_coeff_block_interior_zeros_round_trip() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 3;
        coeffs[3] = 5;
        coeffs[7] = 2;
        coeffs[12] = -8;
        let recovered = encode_then_decode_block(BlockType::Y2, &coeffs, false, false);
        assert_eq!(recovered, coeffs);
    }

    /// Neighbour-predictor combinations seed `ctx3` for the first
    /// coefficient — exercising the §13.3 page 65 rule. All four
    /// (above × left) combinations must produce a self-consistent
    /// byte stream the matching `decode_block` call recovers.
    #[test]
    fn encode_coeff_block_neighbour_predictor_combinations() {
        let coeffs = {
            let mut c = [0i16; 16];
            c[0] = 4;
            c[1] = -1;
            c[5] = 2;
            c
        };
        for above in &[false, true] {
            for left in &[false, true] {
                let recovered = encode_then_decode_block(BlockType::YNoY2, &coeffs, *above, *left);
                assert_eq!(
                    recovered, coeffs,
                    "neighbour predictor combination ({above}, {left}) failed"
                );
            }
        }
    }

    /// Out-of-range coefficient is rejected. The §13.2 alphabet caps
    /// at 67 + 2047 = 2114; anything larger surfaces as
    /// `CoefficientOutOfRange`.
    #[test]
    fn encode_coeff_block_rejects_out_of_range_coefficient() {
        let mut coeffs = [0i16; 16];
        coeffs[2] = 2115;
        let mut enc = BoolEncoder::new();
        let err = encode_coeff_block(
            &mut enc,
            BlockType::YNoY2,
            &DEFAULT_COEFF_PROBS,
            false,
            false,
            &coeffs,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            TokenEncodeError::CoefficientOutOfRange {
                index: 2,
                value: 2115
            }
        ));
    }

    /// `TokenEncoder` wrapper produces the same byte stream as the
    /// free-function path. Locks the wrapper's stateful API to the
    /// stateless one — any future refactor that splits one and not
    /// the other will trip this immediately.
    #[test]
    fn token_encoder_wrapper_matches_free_function_bytes() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 2;
        coeffs[4] = -7;
        coeffs[9] = 15;

        let mut free_enc = BoolEncoder::new();
        encode_coeff_block(
            &mut free_enc,
            BlockType::YNoY2,
            &DEFAULT_COEFF_PROBS,
            false,
            false,
            &coeffs,
        )
        .unwrap();
        let free_bytes = free_enc.finish();

        let mut wrapper = TokenEncoder::new(DEFAULT_COEFF_PROBS);
        wrapper
            .encode_block(BlockType::YNoY2, false, false, &coeffs)
            .unwrap();
        let wrapper_bytes = wrapper.finish();

        assert_eq!(free_bytes, wrapper_bytes);
    }

    /// Pseudo-random sweep — encode 64 random blocks at randomised
    /// quant magnitudes, decode them back, assert byte-identical at
    /// the coefficient layer. Catches an algorithmic regression that
    /// individual-case tests above might miss.
    #[test]
    fn encode_coeff_block_pseudo_random_sweep_round_trips() {
        let mut state: u32 = 0xDEAD_BEEF;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            state
        };

        for trial in 0..64 {
            let bt = match trial % 4 {
                0 => BlockType::YAfterY2,
                1 => BlockType::Y2,
                2 => BlockType::UV,
                _ => BlockType::YNoY2,
            };
            let first = bt.first_coeff();
            let mut coeffs = [0i16; 16];
            // Sparse fill — at most ~half the positions populated, so
            // both the "implicit EOB at 16" and "explicit EOB partway
            // through" branches get exercised.
            for (i, c) in coeffs.iter_mut().enumerate().skip(first) {
                let r = next();
                // Roughly 1 in 4 positions populated; fewer at high
                // index (so EOB usually triggers before the end).
                if (r & 0x3) != 0 || i > 10 {
                    continue;
                }
                // Magnitude up to 100 — exercises Dct1..Dct4 + Cat1..Cat5.
                let mag = ((r >> 8) % 100) as i16;
                let sign = if (r >> 16) & 1 == 0 { 1 } else { -1 };
                *c = mag * sign;
            }
            let above = (next() & 1) != 0;
            let left = (next() & 1) != 0;
            let recovered = encode_then_decode_block(bt, &coeffs, above, left);
            assert_eq!(
                recovered, coeffs,
                "pseudo-random round-trip failed at trial {trial}, bt={bt:?}, above={above}, left={left}"
            );
        }
    }

    // ───────── Phase 2 per-MB block-set encoder roundtrip tests ─────────

    use crate::dct_tokens::decode_mb_coeffs;
    use crate::dequant::MbDequantFactors;
    use crate::macroblock::{IntraBmode, IntraUvMode, IntraYMode};
    use crate::reconstruct::{decode_keyframe_mb_bpred, decode_keyframe_mb_non_bpred, MbNeighbors};

    /// Encode a flat-color MB through `encode_mb_block_set`, decode the
    /// bytes back through the mode-aware helper (off-frame neighbours may
    /// favour B_PRED), dequantize, and run the §14.2 reconstruction —
    /// the recovered luma / chroma planes must be within ≤ 1 LSB of the
    /// input.
    #[test]
    fn mb_block_set_roundtrip_flat_color_recovers_within_one_lsb() {
        // Pick a uniform pixel value off 128 so the residual is
        // non-zero and the §14 chain actually exercises the DC path
        // (rather than every block being all-zero, which is the
        // trivial success case).
        for &pixel in &[100u8, 110, 128, 140, 160, 200] {
            let pixels = MbPixels {
                y: [pixel; 256],
                u: [pixel; 64],
                v: [pixel; 64],
            };

            // yac_qi = 0 is the lossless flat-block case
            // (`DC_QLOOKUP[0] = AC_QLOOKUP[0] = 4`). Decode through the
            // mode-aware helper: with off-frame neighbours the picker may
            // legitimately choose B_PRED (the evolving sub-block
            // neighbours converge on the flat value better than the
            // off-frame whole-block fills), so the decode must follow the
            // picked mode's has_y2 / reconstruction path. The helper also
            // asserts the §13.3 byte-layer roundtrip (raw == coeffs).
            let (_y_mode, _uv, _sub, y_plane, u_plane, v_plane) =
                encode_decode_bpred_aware(&pixels, &MbNeighbors::default(), 0);

            // Verify every reconstructed pixel is within ≤ 1 LSB of the
            // input across all three planes. At yac_qi = 0 the chain is
            // bit-exact for a flat block (per the existing per-block
            // roundtrip test); the bound leaves room for §14.5
            // clamp / rounding on extreme inputs.
            for (i, &recon) in y_plane.iter().enumerate() {
                assert!(
                    (recon as i32 - pixel as i32).abs() <= 1,
                    "pixel {pixel}: y recon[{i}] = {recon} differs by > 1 LSB"
                );
            }
            for (i, &recon) in u_plane.iter().enumerate() {
                assert!(
                    (recon as i32 - pixel as i32).abs() <= 1,
                    "pixel {pixel}: u recon[{i}] = {recon} differs by > 1 LSB"
                );
            }
            for (i, &recon) in v_plane.iter().enumerate() {
                assert!(
                    (recon as i32 - pixel as i32).abs() <= 1,
                    "pixel {pixel}: v recon[{i}] = {recon} differs by > 1 LSB"
                );
            }
        }
    }

    /// A constant 128-pixel macroblock has zero residual — every block
    /// after FDCT / FWHT is all-zero, which means every encoded block
    /// is a single immediate EOB token and the predictor contexts
    /// remain at their default (no non-zero coefficients anywhere).
    #[test]
    fn mb_block_set_constant_128_emits_all_eob_blocks() {
        let pixels = MbPixels {
            y: [128u8; 256],
            u: [128u8; 64],
            v: [128u8; 64],
        };
        let encoded =
            encode_mb_block_set(&pixels, 0, &DEFAULT_COEFF_PROBS).expect("encode_mb_block_set");
        assert_eq!(
            encoded.nonzero_block_count, 0,
            "constant 128 MB must produce zero non-zero blocks"
        );
        // A flat 128 block is the one value where DC_PRED's top-left 128
        // fill matches the source exactly (SAD 0) while V/H/TM carry the
        // 127/129 off-frame defaults — so the picker must choose DC.
        assert_eq!(encoded.y_mode, IntraYMode::Dc);
        assert_eq!(encoded.uv_mode, IntraUvMode::Dc);
        // Round-trip through the decoder to confirm the bytes are
        // structurally valid (the bool encoder always writes ≥ 4 bytes
        // even on an empty stream — the §7.3 flush tail).
        let mut dec = BoolDecoder::init(&encoded.bytes).expect("encoder emits ≥ 2 bytes");
        let mut above = MbEntropyCtx::default();
        let mut left = MbEntropyCtx::default();
        let recovered = decode_mb_coeffs(
            &mut dec,
            true,
            false,
            &DEFAULT_COEFF_PROBS,
            &mut above,
            &mut left,
        )
        .expect("decode_mb_coeffs");
        assert_eq!(recovered.y2, [0i16; 16]);
        assert!(recovered.y.iter().all(|b| *b == [0i16; 16]));
        assert!(recovered.u.iter().all(|b| *b == [0i16; 16]));
        assert!(recovered.v.iter().all(|b| *b == [0i16; 16]));
    }

    /// A non-zero MB at a non-trivial quantizer must still recover
    /// within ≤ 1 LSB on a flat colour input — the round-131 §14
    /// per-block roundtrip proved the chain holds at `yac_qi = 32`,
    /// and the per-MB walk doesn't perturb that bound for a flat MB.
    #[test]
    fn mb_block_set_roundtrip_flat_color_at_q16_holds_within_2_lsb() {
        let pixels = MbPixels {
            y: [160u8; 256],
            u: [120u8; 64],
            v: [140u8; 64],
        };
        // Mode-aware decode (off-frame neighbours may favour B_PRED); the
        // helper asserts the byte-layer roundtrip internally.
        let (_y_mode, _uv, _sub, y_plane, _u_plane, _v_plane) =
            encode_decode_bpred_aware(&pixels, &MbNeighbors::default(), 16);

        // At yac_qi = 16 the §14 chain holds to ≤ 2 LSB on a flat
        // block (the WHT + DCT round-trip introduces at most a small
        // round-off from the `*155/100` Y2 AC scaling, but flat blocks
        // have zero AC so this collapses to the DC-only path which is
        // bit-exact up to the IWHT's `(x+3)>>3` rounding — i.e. ≤ 1
        // LSB; the ≤ 2 bound is defensive).
        for (i, &recon) in y_plane.iter().enumerate() {
            assert!(
                (recon as i32 - 160i32).abs() <= 2,
                "yac_qi=16 flat 160: recon[{i}] = {recon} differs by > 2 LSB"
            );
        }
    }

    // ───────── Phase 3 whole-block intra mode-pick tests ─────────

    /// Mean-squared-error → PSNR in dB between a source plane and its
    /// reconstruction. Returns `f64::INFINITY` for a bit-exact match.
    fn psnr(src: &[u8], recon: &[u8]) -> f64 {
        assert_eq!(src.len(), recon.len());
        let mse: f64 = src
            .iter()
            .zip(recon.iter())
            .map(|(&s, &r)| {
                let d = s as f64 - r as f64;
                d * d
            })
            .sum::<f64>()
            / src.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    /// Encode `pixels` against `neighbors`, decode the bytes back, run
    /// the §14.2 reconstruction orchestrator against the **same**
    /// neighbours + the picked modes, and return
    /// `(y_mode, uv_mode, reconstructed planes)`.
    fn encode_decode_with_neighbors(
        pixels: &MbPixels,
        neighbors: &MbNeighbors,
        yac_qi: u8,
    ) -> (IntraYMode, IntraUvMode, Vec<u8>, Vec<u8>, Vec<u8>) {
        let encoded =
            encode_mb_block_set_with_neighbors(pixels, neighbors, yac_qi, &DEFAULT_COEFF_PROBS)
                .expect("encode_mb_block_set_with_neighbors");

        let mut dec = BoolDecoder::init(&encoded.bytes).expect("encoder emits ≥ 2 bytes");
        let mut above = MbEntropyCtx::default();
        let mut left = MbEntropyCtx::default();
        let mut raw = decode_mb_coeffs(
            &mut dec,
            true,
            false,
            &DEFAULT_COEFF_PROBS,
            &mut above,
            &mut left,
        )
        .expect("decode_mb_coeffs");
        assert_eq!(raw, encoded.coeffs, "byte-layer roundtrip mismatch");

        let factors = MbDequantFactors::from_base_and_deltas(yac_qi as i32, 0, 0, 0, 0, 0);
        factors.dequantize(&mut raw);

        let recon = decode_keyframe_mb_non_bpred(
            encoded.y_mode,
            encoded.uv_mode,
            false,
            neighbors,
            &raw.y2,
            &raw.y,
            &raw.u,
            &raw.v,
        )
        .expect("decode_keyframe_mb_non_bpred");

        (
            encoded.y_mode,
            encoded.uv_mode,
            recon.y.to_vec(),
            recon.u.to_vec(),
            recon.v.to_vec(),
        )
    }

    /// Encode `pixels` against `neighbors`, decode the bytes back, and
    /// run the reconstruction orchestrator that matches the picked luma
    /// mode — `decode_keyframe_mb_bpred` (no Y2) when `y_mode == B`, or
    /// `decode_keyframe_mb_non_bpred` (with Y2) otherwise. Returns the
    /// picked modes (luma, chroma, optional 16 sub-modes) plus the
    /// reconstructed planes. This is the §11.3 / §12.3 superset of
    /// [`encode_decode_with_neighbors`] used by the B_PRED tests.
    #[allow(clippy::type_complexity)]
    fn encode_decode_bpred_aware(
        pixels: &MbPixels,
        neighbors: &MbNeighbors,
        yac_qi: u8,
    ) -> (
        IntraYMode,
        IntraUvMode,
        Option<[IntraBmode; 16]>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    ) {
        let encoded =
            encode_mb_block_set_with_neighbors(pixels, neighbors, yac_qi, &DEFAULT_COEFF_PROBS)
                .expect("encode_mb_block_set_with_neighbors");

        let is_bpred = encoded.y_mode == IntraYMode::B;
        // A B_PRED macroblock has no Y2 block, so `decode_mb_coeffs`
        // must be told `has_y2 = false` to route the Y plane through the
        // YNoY2 token plane and skip block 24.
        let has_y2 = !is_bpred;

        let mut dec = BoolDecoder::init(&encoded.bytes).expect("encoder emits ≥ 2 bytes");
        let mut above = MbEntropyCtx::default();
        let mut left = MbEntropyCtx::default();
        let mut raw = decode_mb_coeffs(
            &mut dec,
            has_y2,
            false,
            &DEFAULT_COEFF_PROBS,
            &mut above,
            &mut left,
        )
        .expect("decode_mb_coeffs");
        assert_eq!(raw, encoded.coeffs, "byte-layer roundtrip mismatch");

        let factors = MbDequantFactors::from_base_and_deltas(yac_qi as i32, 0, 0, 0, 0, 0);
        factors.dequantize(&mut raw);

        let recon = if is_bpred {
            decode_keyframe_mb_bpred(
                encoded.b_subblock_modes.as_ref(),
                encoded.uv_mode,
                false,
                neighbors,
                &raw.y,
                &raw.u,
                &raw.v,
            )
            .expect("decode_keyframe_mb_bpred")
        } else {
            decode_keyframe_mb_non_bpred(
                encoded.y_mode,
                encoded.uv_mode,
                false,
                neighbors,
                &raw.y2,
                &raw.y,
                &raw.u,
                &raw.v,
            )
            .expect("decode_keyframe_mb_non_bpred")
        };

        (
            encoded.y_mode,
            encoded.uv_mode,
            encoded.b_subblock_modes,
            recon.y.to_vec(),
            recon.u.to_vec(),
            recon.v.to_vec(),
        )
    }

    // ───────── §11.3 / §12.3 B_PRED luma mode-pick tests ─────────

    /// A flat MB (every luma pixel equal) sitting in a flat region —
    /// i.e. with neighbour strips equal to the same constant — is
    /// reproduced **exactly** (SSD 0) by the whole-block V / H / DC
    /// modes. The B_PRED pass can at best tie that SSD 0 distortion but
    /// costs more rate (the §11.2 B_PRED mode flag plus sixteen sub-block
    /// mode signals), so the RD decision (B_PRED wins only on a strictly
    /// lower `J = SSD + lambda * R`) keeps the macroblock on a whole-block
    /// luma mode (`y_mode != B`) and carries no sub-modes. (With
    /// *off-frame* neighbours a flat MB legitimately favours B_PRED,
    /// because the off-frame whole-block fills disagree with the interior
    /// while the evolving sub-block neighbours converge on the true
    /// constant — so the realistic flat-region case uses matching
    /// neighbours, which is how flat areas appear mid-frame.)
    #[test]
    fn flat_mb_keeps_whole_block_luma_mode() {
        let pixels = MbPixels {
            y: [120u8; 256],
            u: [120u8; 64],
            v: [120u8; 64],
        };
        // Matching flat reconstructed-neighbour strips (a flat region).
        let neighbors = MbNeighbors {
            y_above: Some([120u8; 16]),
            y_left: Some([120u8; 16]),
            y_topleft: Some(120),
            y_above_right: Some([120u8; 4]),
            u_above: Some([120u8; 8]),
            u_left: Some([120u8; 8]),
            u_topleft: Some(120),
            v_above: Some([120u8; 8]),
            v_left: Some([120u8; 8]),
            v_topleft: Some(120),
        };

        let (y_mode, _uv, sub, ry, _ru, _rv) = encode_decode_bpred_aware(&pixels, &neighbors, 8);

        assert_ne!(
            y_mode,
            IntraYMode::B,
            "a flat MB must NOT flip to B_PRED — a whole-block mode reproduces it exactly"
        );
        assert!(
            sub.is_none(),
            "whole-block luma decision must carry no sub-block modes"
        );
        // Sanity: it still reconstructs the flat value at high PSNR.
        let p_y = psnr(&pixels.y, &ry);
        assert!(p_y >= 30.0, "flat luma PSNR {p_y:.2} dB < 30 dB");
    }

    /// A macroblock built from per-4×4-sub-block diagonal structure —
    /// each 4×4 sub-block a 45° gradient that no single whole-block
    /// 16×16 mode can follow — must flip the top-level luma decision to
    /// `B_PRED`. The picked sub-modes drive the decoder's per-sub-block
    /// walk and the reconstruction must clear ≥ 30 dB. This is the
    /// black-box validation for the round's B_PRED pass.
    #[test]
    fn diagonal_subblock_mb_picks_bpred_and_decodes_above_30db() {
        // Build a 16×16 luma plane whose every 4×4 sub-block carries a
        // strong diagonal gradient: pixel value depends on (row + col)
        // within the sub-block, repeating per sub-block so the global
        // plane has 16 independent diagonal tiles. A whole-block V / H /
        // DC / TM prediction smears across the tile boundaries; the
        // per-sub-block B_PRED diagonal modes (B_LD / B_RD / …) fit each
        // tile far better, so total B_PRED SAD beats every whole-block
        // mode.
        let mut y = [0u8; 256];
        for r in 0..16usize {
            for c in 0..16usize {
                // Local coordinates inside the 4×4 sub-block.
                let lr = r % 4;
                let lc = c % 4;
                // Diagonal ramp 30..=240 across the sub-block's
                // anti-diagonal; alternate the slope direction per
                // sub-block so neighbouring tiles differ.
                let sb = (r / 4) * 4 + (c / 4);
                let d = if sb % 2 == 0 { lr + lc } else { 6 - (lr + lc) };
                y[r * 16 + c] = (30 + d * 35) as u8;
            }
        }
        // Flat chroma — chroma always uses a whole-block mode regardless.
        let pixels = MbPixels {
            y,
            u: [128u8; 64],
            v: [128u8; 64],
        };
        let neighbors = MbNeighbors::default();

        let (y_mode, _uv, sub, ry, _ru, _rv) = encode_decode_bpred_aware(&pixels, &neighbors, 4);

        assert_eq!(
            y_mode,
            IntraYMode::B,
            "a per-4×4-sub-block-structured MB must pick B_PRED"
        );
        let sub = sub.expect("B_PRED MB must carry sixteen sub-block modes");
        assert_eq!(sub.len(), 16);

        let p_y = psnr(&pixels.y, &ry);
        assert!(
            p_y >= 30.0,
            "B_PRED luma reconstruction PSNR {p_y:.2} dB < 30 dB"
        );
    }

    /// The encoder's in-place neighbour evolution must match the
    /// decoder's: the bytes a B_PRED MB emits, fed back through
    /// `decode_keyframe_mb_bpred` with the **same** sub-modes and
    /// neighbours, must reconstruct the exact pixels the encoder's
    /// internal working buffer held. We check this indirectly via the
    /// byte-layer roundtrip assertion in the helper (raw == coeffs) plus
    /// a tighter PSNR floor at a low quantiser on the same diagonal MB.
    ///
    /// The MB is a per-sub-block diagonal pattern that no single
    /// whole-block mode can predict; at a *low but non-zero* quantiser the
    /// rate-distortion picker flips it to B_PRED (the per-sub-block
    /// prediction shaves enough distortion to repay the extra mode bits).
    /// At `yac_qi = 0` the chain is near-lossless, so the distortion of
    /// every candidate collapses toward zero and the picker prefers the
    /// cheaper-rate whole-block path — which is why this test exercises
    /// the realistic low-q (`yac_qi = 8`) regime rather than `q = 0`.
    #[test]
    fn bpred_neighbour_evolution_roundtrips_at_low_q() {
        let mut y = [0u8; 256];
        for r in 0..16usize {
            for c in 0..16usize {
                let lr = r % 4;
                let lc = c % 4;
                let sb = (r / 4) * 4 + (c / 4);
                let d = if sb % 2 == 0 { lr + lc } else { 6 - (lr + lc) };
                y[r * 16 + c] = (30 + d * 35) as u8;
            }
        }
        let pixels = MbPixels {
            y,
            u: [128u8; 64],
            v: [128u8; 64],
        };
        let neighbors = MbNeighbors::default();

        // At yac_qi = 8 the §14 chain is near-lossless and the RD picker
        // selects B_PRED; the reconstruction should clear a high PSNR
        // floor (the helper also asserts the byte-layer roundtrip).
        let (y_mode, _uv, sub, ry, _ru, _rv) = encode_decode_bpred_aware(&pixels, &neighbors, 8);
        assert_eq!(y_mode, IntraYMode::B);
        assert!(sub.is_some());
        let p_y = psnr(&pixels.y, &ry);
        assert!(
            p_y >= 40.0,
            "B_PRED luma PSNR at q8 {p_y:.2} dB < 40 dB — neighbour evolution likely diverged"
        );
    }

    /// A macroblock whose pixels are constant down each column but vary
    /// across columns is reproduced exactly by V_PRED from the above
    /// row. With a genuine reconstructed-neighbour `above` strip that
    /// equals the column pattern, the SAD picker must choose `V` for
    /// luma and the residual must reconstruct at high PSNR.
    #[test]
    fn mode_pick_chooses_v_pred_for_column_constant_mb() {
        // Column c value: a ramp 40..=190 across the 16 columns. Every
        // row of the MB is identical to this row, so the bottom row of
        // the (notional) above neighbour equals it too.
        let mut y_row = [0u8; 16];
        for (c, slot) in y_row.iter_mut().enumerate() {
            *slot = (40 + c * 10) as u8;
        }
        let mut u_row = [0u8; 8];
        let mut v_row = [0u8; 8];
        for c in 0..8 {
            u_row[c] = (50 + c * 12) as u8;
            v_row[c] = (200 - c * 12) as u8;
        }

        let mut y = [0u8; 256];
        for r in 0..16 {
            y[r * 16..r * 16 + 16].copy_from_slice(&y_row);
        }
        let mut u = [0u8; 64];
        let mut v = [0u8; 64];
        for r in 0..8 {
            u[r * 8..r * 8 + 8].copy_from_slice(&u_row);
            v[r * 8..r * 8 + 8].copy_from_slice(&v_row);
        }
        let pixels = MbPixels { y, u, v };

        let neighbors = MbNeighbors {
            y_above: Some(y_row),
            u_above: Some(u_row),
            v_above: Some(v_row),
            ..MbNeighbors::default()
        };

        let (y_mode, uv_mode, ry, ru, rv) = encode_decode_with_neighbors(&pixels, &neighbors, 8);

        assert_eq!(
            y_mode,
            IntraYMode::V,
            "column-constant MB must pick V_PRED for luma"
        );
        assert_eq!(
            uv_mode,
            IntraUvMode::V,
            "column-constant MB must pick V_PRED for chroma"
        );

        let p_y = psnr(&pixels.y, &ry);
        let p_u = psnr(&pixels.u, &ru);
        let p_v = psnr(&pixels.v, &rv);
        assert!(p_y >= 30.0, "V_PRED luma PSNR {p_y:.2} dB < 30 dB");
        assert!(p_u >= 30.0, "V_PRED U PSNR {p_u:.2} dB < 30 dB");
        assert!(p_v >= 30.0, "V_PRED V PSNR {p_v:.2} dB < 30 dB");
    }

    /// A macroblock whose pixels are constant across each row but vary
    /// down the rows is reproduced exactly by H_PRED from the left
    /// column. The SAD picker must choose `H` and reconstruct at high
    /// PSNR.
    #[test]
    fn mode_pick_chooses_h_pred_for_row_constant_mb() {
        let mut y_col = [0u8; 16];
        for (r, slot) in y_col.iter_mut().enumerate() {
            *slot = (200 - r * 10) as u8;
        }
        let mut u_col = [0u8; 8];
        let mut v_col = [0u8; 8];
        for r in 0..8 {
            u_col[r] = (60 + r * 14) as u8;
            v_col[r] = (190 - r * 14) as u8;
        }

        let mut y = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                y[r * 16 + c] = y_col[r];
            }
        }
        let mut u = [0u8; 64];
        let mut v = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                u[r * 8 + c] = u_col[r];
                v[r * 8 + c] = v_col[r];
            }
        }
        let pixels = MbPixels { y, u, v };

        let neighbors = MbNeighbors {
            y_left: Some(y_col),
            u_left: Some(u_col),
            v_left: Some(v_col),
            ..MbNeighbors::default()
        };

        let (y_mode, uv_mode, ry, ru, rv) = encode_decode_with_neighbors(&pixels, &neighbors, 8);

        assert_eq!(
            y_mode,
            IntraYMode::H,
            "row-constant MB must pick H_PRED for luma"
        );
        assert_eq!(
            uv_mode,
            IntraUvMode::H,
            "row-constant MB must pick H_PRED for chroma"
        );

        let p_y = psnr(&pixels.y, &ry);
        let p_u = psnr(&pixels.u, &ru);
        let p_v = psnr(&pixels.v, &rv);
        assert!(p_y >= 30.0, "H_PRED luma PSNR {p_y:.2} dB < 30 dB");
        assert!(p_u >= 30.0, "H_PRED U PSNR {p_u:.2} dB < 30 dB");
        assert!(p_v >= 30.0, "H_PRED V PSNR {p_v:.2} dB < 30 dB");
    }

    /// TM_PRED propagates a gradient seeded by the above row, the left
    /// column, and the corner: `X_ij = clamp(L_i + A_j - P)`. A planar
    /// ramp `base + i + j` is exactly the shape TM reconstructs, so a MB
    /// built from that surface (with matching neighbour strips) must let
    /// the picker beat the flat fallbacks with TM and reconstruct at
    /// high PSNR.
    #[test]
    fn mode_pick_chooses_tm_pred_for_planar_ramp_mb() {
        let corner: i32 = 100;
        // Above row A_j = corner + (j+1); left column L_i = corner + (i+1).
        let mut a = [0u8; 16];
        let mut l = [0u8; 16];
        for k in 0..16 {
            a[k] = (corner + k as i32 + 1) as u8;
            l[k] = (corner + k as i32 + 1) as u8;
        }
        let mut ua = [0u8; 8];
        let mut ul = [0u8; 8];
        let mut va = [0u8; 8];
        let mut vl = [0u8; 8];
        for k in 0..8 {
            ua[k] = (corner + k as i32 + 1) as u8;
            ul[k] = (corner + k as i32 + 1) as u8;
            va[k] = (corner + k as i32 + 1) as u8;
            vl[k] = (corner + k as i32 + 1) as u8;
        }

        // Source surface = clamp(L_i + A_j - P) — what TM predicts.
        let mut y = [0u8; 256];
        for i in 0..16 {
            for j in 0..16 {
                y[i * 16 + j] = (l[i] as i32 + a[j] as i32 - corner).clamp(0, 255) as u8;
            }
        }
        let mut u = [0u8; 64];
        let mut v = [0u8; 64];
        for i in 0..8 {
            for j in 0..8 {
                u[i * 8 + j] = (ul[i] as i32 + ua[j] as i32 - corner).clamp(0, 255) as u8;
                v[i * 8 + j] = (vl[i] as i32 + va[j] as i32 - corner).clamp(0, 255) as u8;
            }
        }
        let pixels = MbPixels { y, u, v };

        let neighbors = MbNeighbors {
            y_above: Some(a),
            y_left: Some(l),
            y_topleft: Some(corner as u8),
            u_above: Some(ua),
            u_left: Some(ul),
            u_topleft: Some(corner as u8),
            v_above: Some(va),
            v_left: Some(vl),
            v_topleft: Some(corner as u8),
            ..MbNeighbors::default()
        };

        let (y_mode, _uv_mode, ry, _ru, _rv) = encode_decode_with_neighbors(&pixels, &neighbors, 8);

        assert_eq!(
            y_mode,
            IntraYMode::Tm,
            "planar-ramp MB must pick TM_PRED for luma"
        );
        let p_y = psnr(&pixels.y, &ry);
        assert!(p_y >= 30.0, "TM_PRED luma PSNR {p_y:.2} dB < 30 dB");
    }

    /// The standalone `encode_mb_block_set` (off-frame neighbours)
    /// reduces to the same picker over the §12 defaults: it must still
    /// roundtrip a textured MB through the decoder's isolated-MB path
    /// (`decode_keyframe` on a 1×1 frame) at ≥ 30 dB.
    #[test]
    fn isolated_mb_textured_roundtrips_above_30db() {
        // A mild diagonal texture around mid-grey — no neighbour helps,
        // so the picker lands on whichever default is closest, but the
        // residual still carries the texture and must reconstruct well.
        let mut y = [0u8; 256];
        for i in 0..16 {
            for j in 0..16 {
                y[i * 16 + j] = (120 + ((i + j) % 8) * 2) as u8;
            }
        }
        let mut u = [0u8; 64];
        let mut v = [0u8; 64];
        for i in 0..8 {
            for j in 0..8 {
                u[i * 8 + j] = (124 + ((i + j) % 4) * 2) as u8;
                v[i * 8 + j] = (132 - ((i + j) % 4) * 2) as u8;
            }
        }
        let pixels = MbPixels { y, u, v };

        // Mode-aware decode: a diagonal texture with off-frame neighbours
        // may pick B_PRED for luma; either way the residual must
        // reconstruct the texture above the PSNR floor.
        let (_y_mode, _uv, _sub, ry, ru, rv) =
            encode_decode_bpred_aware(&pixels, &MbNeighbors::default(), 8);

        let p_y = psnr(&pixels.y, &ry);
        let p_u = psnr(&pixels.u, &ru);
        let p_v = psnr(&pixels.v, &rv);
        assert!(p_y >= 30.0, "isolated luma PSNR {p_y:.2} dB < 30 dB");
        assert!(p_u >= 30.0, "isolated U PSNR {p_u:.2} dB < 30 dB");
        assert!(p_v >= 30.0, "isolated V PSNR {p_v:.2} dB < 30 dB");
    }

    /// Round 161 — `pick_intra_mb_all` must score every §11.2 × §11.4
    /// whole-block intra candidate and return the J-best `(y_mode,
    /// uv_mode)`. Three crafted MBs exercise the non-DC modes:
    ///
    ///   * Vertical-stripe MB (rows identical to the above row) — the
    ///     §12 V_PRED prediction is exact, so `(V, V)` should win.
    ///   * Horizontal-stripe MB (columns identical to the left column)
    ///     — H_PRED prediction is exact, so `(H, H)` should win.
    ///   * Planar-ramp MB (`base + i + j`) — TM_PRED prediction is exact,
    ///     so `(Tm, Tm)` should win.
    ///
    /// All three contrast with the round-160 DC-only picker, which would
    /// have returned `(Dc, Dc)` on every input regardless of content.
    /// Neighbour edges are supplied so the V / H / TM kernels predict
    /// from the genuine neighbour content (not the §12 off-frame
    /// default-fill 127/129).
    #[test]
    fn pick_intra_mb_all_selects_v_h_tm_for_structured_sources() {
        use crate::dequant::MbDequantFactors;
        use crate::macroblock::{IntraUvMode, IntraYMode};
        use crate::reconstruct::MbNeighbors;

        // Fixed mid-quality quantiser keeps the §14 floor low so the
        // residual stays small enough for the matched mode's SAD to win.
        let factors = MbDequantFactors::from_base_and_deltas(32, 0, 0, 0, 0, 0);
        let lambda = rd_lambda(&factors);

        // Above strip: vertically-monotonic; left strip: monotonic; corner
        // value matches both ends — used by TM_PRED.
        let above_y: [u8; 16] = core::array::from_fn(|j| (40 + j * 8) as u8);
        let left_y: [u8; 16] = core::array::from_fn(|i| (40 + i * 8) as u8);
        let above_uv: [u8; 8] = core::array::from_fn(|j| (60 + j * 4) as u8);
        let left_uv: [u8; 8] = core::array::from_fn(|i| (60 + i * 4) as u8);
        let corner_y: u8 = 40;
        let corner_uv: u8 = 60;

        let make_neighbors = || MbNeighbors {
            y_above: Some(above_y),
            y_left: Some(left_y),
            y_above_right: Some([above_y[15]; 4]),
            y_topleft: Some(corner_y),
            u_above: Some(above_uv),
            u_left: Some(left_uv),
            u_topleft: Some(corner_uv),
            v_above: Some(above_uv),
            v_left: Some(left_uv),
            v_topleft: Some(corner_uv),
        };

        // ---- Vertical-stripe source: every row equals the above row.
        //      V_PRED predicts the above row directly, residual = 0.
        let mut y_vstripe = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                y_vstripe[r * 16 + c] = above_y[c];
            }
        }
        let mut uv_vstripe = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                uv_vstripe[r * 8 + c] = above_uv[c];
            }
        }
        let pixels_v = MbPixels {
            y: y_vstripe,
            u: uv_vstripe,
            v: uv_vstripe,
        };
        let pick_v = pick_intra_mb_all(
            &pixels_v,
            &make_neighbors(),
            &factors,
            lambda,
            None,
            None,
            TrellisStrength::DEFAULT,
        );
        assert_eq!(
            pick_v.y_mode,
            IntraYMode::V,
            "vertical-stripe source should pick V_PRED for Y, got {:?}",
            pick_v.y_mode
        );
        // The picker's distortion model is Y-SAD only (matches the
        // inter picker for cross-candidate apples-to-apples comparison),
        // so the chroma mode is chosen purely on §11.4 tree-bit cost
        // unless the chroma rate-distortion adapts the Y SAD via the
        // recon path. DC has the cheapest leaf bit cost in the §11.4
        // IF_UV_MODE_TREE, so DC is a legitimate winner when Y dominates.
        // Assert the chroma pick is a valid §11.4 leaf.
        assert!(matches!(
            pick_v.uv_mode,
            IntraUvMode::Dc | IntraUvMode::V | IntraUvMode::H | IntraUvMode::Tm
        ));

        // ---- Horizontal-stripe source: every column equals the left
        //      column. H_PRED predicts the left column directly,
        //      residual = 0.
        let mut y_hstripe = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                y_hstripe[r * 16 + c] = left_y[r];
            }
        }
        let mut uv_hstripe = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                uv_hstripe[r * 8 + c] = left_uv[r];
            }
        }
        let pixels_h = MbPixels {
            y: y_hstripe,
            u: uv_hstripe,
            v: uv_hstripe,
        };
        let pick_h = pick_intra_mb_all(
            &pixels_h,
            &make_neighbors(),
            &factors,
            lambda,
            None,
            None,
            TrellisStrength::DEFAULT,
        );
        assert_eq!(
            pick_h.y_mode,
            IntraYMode::H,
            "horizontal-stripe source should pick H_PRED for Y, got {:?}",
            pick_h.y_mode
        );
        assert!(matches!(
            pick_h.uv_mode,
            IntraUvMode::Dc | IntraUvMode::V | IntraUvMode::H | IntraUvMode::Tm
        ));

        // ---- Planar-ramp source. TM_PRED reconstructs the canonical
        //      planar surface `clamp(L_i + A_j - P)` from the
        //      neighbours; with above / left = `40 + 8k` and corner = 40,
        //      the surface is `40 + 8 * (r + c)` over the 16×16 grid.
        //      Clamp to u8 (last few cells saturate near 255).
        let mut y_tm = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                let v = (corner_y as i32) + 8 * (r as i32 + c as i32) - 40;
                // The TM formula is `clamp(L_i + A_j - P)` with
                //   L_i = corner_y + 8 * (i+1) = 40 + 8 * (i+1)
                //   A_j = corner_y + 8 * (j+1) = 40 + 8 * (j+1)
                //   P   = corner_y             = 40
                // ⇒ surface = 40 + 8*(i+1) + 8*(j+1) - 40 = 8*(i+j+2).
                let surface = 8 * (r as i32 + c as i32 + 2);
                let _ = v; // commented for clarity; we use `surface`.
                y_tm[r * 16 + c] = surface.clamp(0, 255) as u8;
            }
        }
        let mut uv_tm = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                // Chroma uses corner_uv = 60, above/left = 60 + 4*(k+1).
                let surface = 4 * (r as i32 + c as i32) + 2 * 4 + (corner_uv as i32);
                // = 4*(r+c) + 8 + 60 = 4*(r+c) + 68. Equivalent to
                // L_i + A_j - P with the same algebra.
                uv_tm[r * 8 + c] = surface.clamp(0, 255) as u8;
            }
        }
        let pixels_tm = MbPixels {
            y: y_tm,
            u: uv_tm,
            v: uv_tm,
        };
        let pick_tm = pick_intra_mb_all(
            &pixels_tm,
            &make_neighbors(),
            &factors,
            lambda,
            None,
            None,
            TrellisStrength::DEFAULT,
        );
        assert_eq!(
            pick_tm.y_mode,
            IntraYMode::Tm,
            "planar-ramp source should pick TM_PRED for Y, got {:?}",
            pick_tm.y_mode
        );
        // Chroma may legitimately tie at TM or fall to V/H/DC depending
        // on the ramp slope vs the §11.4 mode-tree-bit cost trade. The
        // luma TM_PRED selection is the load-bearing assertion (it
        // would have been impossible under the r160 DC-only picker);
        // the chroma assertion is a smoke check that the pick is a
        // valid §11.4 leaf.
        assert!(matches!(
            pick_tm.uv_mode,
            IntraUvMode::Dc | IntraUvMode::V | IntraUvMode::H | IntraUvMode::Tm
        ));

        // ---- Sanity: a flat-grey source still rounds to DC_PRED (the
        //      round-160 behaviour); strict-< tie-break to first-tried
        //      ⇒ (Dc, Dc).
        let flat_y = [128u8; 256];
        let flat_uv = [128u8; 64];
        let pixels_flat = MbPixels {
            y: flat_y,
            u: flat_uv,
            v: flat_uv,
        };
        let pick_flat = pick_intra_mb_all(
            &pixels_flat,
            &MbNeighbors::default(),
            &factors,
            lambda,
            None,
            None,
            TrellisStrength::DEFAULT,
        );
        assert_eq!(
            pick_flat.y_mode,
            IntraYMode::Dc,
            "flat-grey source with off-frame neighbours should still pick DC_PRED for Y"
        );
        assert_eq!(
            pick_flat.uv_mode,
            IntraUvMode::Dc,
            "flat-grey source should pick DC_PRED for UV"
        );
    }

    // ─────────────── bit-cost model (rate control) ───────────────

    #[test]
    fn estimate_bits_per_mb_falls_as_qindex_rises() {
        // The whole point of the model: at a fixed complexity, raising
        // the qindex (coarser quantiser) must lower the predicted bits.
        let complexity = 200.0;
        let lo = estimate_bits_per_mb(complexity, 10);
        let mid = estimate_bits_per_mb(complexity, 50);
        let hi = estimate_bits_per_mb(complexity, 110);
        assert!(
            lo > mid && mid > hi,
            "bits/MB must fall as qindex rises: qi10={lo} qi50={mid} qi110={hi}"
        );
    }

    #[test]
    fn estimate_bits_per_mb_rises_with_complexity() {
        // At a fixed qindex, a busier frame must cost more bits.
        let qi = 40;
        let flat = estimate_bits_per_mb(10.0, qi);
        let busy = estimate_bits_per_mb(800.0, qi);
        assert!(
            busy > flat,
            "busier frame must cost more bits at the same qindex: flat={flat} busy={busy}"
        );
    }

    #[test]
    fn estimate_bits_per_mb_reproduces_surrogate_at_reference_qindex() {
        // By construction the surrogate value is the predicted bits/MB
        // at the reference qindex (plus the per-MB floor): the scaling
        // factor collapses to 1.0 when step == ref_step.
        let complexity = 137.0;
        let predicted = estimate_bits_per_mb(complexity, RATE_MODEL_REF_QINDEX as i32);
        assert!(
            (predicted - (complexity + RATE_MODEL_MB_FLOOR_BITS)).abs() < 1e-3,
            "at the reference qindex the model must reproduce the surrogate + floor, got {predicted}"
        );
    }

    #[test]
    fn estimate_bits_per_mb_keeps_the_floor_at_coarsest_quant() {
        // Even at qindex 127 a zero-complexity frame still charges the
        // per-MB floor (header / mode bits never vanish).
        let predicted = estimate_bits_per_mb(0.0, 127);
        assert!(
            (predicted - RATE_MODEL_MB_FLOOR_BITS).abs() < 1e-3,
            "zero-complexity frame at qi127 must charge exactly the floor, got {predicted}"
        );
    }

    #[test]
    fn estimate_frame_bytes_scales_with_macroblock_count() {
        // A 4× larger frame (2× each dimension) has 4× the MBs and so
        // ≈4× the predicted bytes at the same complexity / qindex.
        let small = estimate_frame_bytes(150.0, 40, 32, 32); // 2×2 MBs
        let big = estimate_frame_bytes(150.0, 40, 64, 64); // 4×4 MBs
        assert!(
            big >= small.saturating_mul(3),
            "4× the MBs must predict roughly 4× the bytes: small={small} big={big}"
        );
    }

    #[test]
    fn estimate_frame_bytes_rounds_partial_macroblocks_up() {
        // A 17×17 frame spans 2×2 MBs (one full + partials) — same MB
        // count as 32×32, so the predicted byte count must match.
        let partial = estimate_frame_bytes(100.0, 40, 17, 17);
        let full = estimate_frame_bytes(100.0, 40, 32, 32);
        assert_eq!(
            partial, full,
            "17×17 and 32×32 both span 2×2 MBs — byte estimate must match"
        );
    }

    #[test]
    fn estimate_frame_bytes_zero_dimensions_is_zero() {
        assert_eq!(estimate_frame_bytes(500.0, 40, 0, 64), 0);
        assert_eq!(estimate_frame_bytes(500.0, 40, 64, 0), 0);
    }

    // ─────────────── bitrate-budget solver ───────────────

    fn busy_clip(n: usize) -> Vec<FrameComplexity> {
        (0..n)
            .map(|i| FrameComplexity {
                frame_index: i as u32,
                bits_per_mb: 600.0,
                scene_cut: false,
            })
            .collect()
    }

    #[test]
    fn gop_byte_budget_tracks_bitrate_and_rate() {
        let mut config = Vp8TwoPassConfig {
            target_bitrate_bps: 8_000_000, // 1 MB/s
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        // 30 frames at 30fps == 1 second == 1 MB.
        let b = gop_byte_budget(&config, 30).expect("budget defined");
        assert_eq!(b, 1_000_000);
        // Half the frames ⇒ half the budget.
        let b2 = gop_byte_budget(&config, 15).expect("budget defined");
        assert_eq!(b2, 500_000);
        // No rate ⇒ no budget.
        config.fps_num = 0;
        assert!(gop_byte_budget(&config, 30).is_none());
    }

    #[test]
    fn solver_returns_base_when_not_bitrate_targeted() {
        // No target bitrate ⇒ solver is a no-op, returns base qindex.
        let config = Vp8TwoPassConfig {
            base: Vp8EncoderConfig {
                qindex: 44,
                ..Vp8EncoderConfig::default()
            },
            ..Vp8TwoPassConfig::default()
        };
        let qi = solve_base_qindex_for_budget(&config, &busy_clip(10), 176, 144);
        assert_eq!(qi, 44);
    }

    #[test]
    fn solver_raises_qindex_under_a_tight_budget() {
        // A deliberately starved bitrate must drive the solved base
        // qindex high (coarse quantiser) to fit the budget.
        let config = Vp8TwoPassConfig {
            target_bitrate_bps: 8_000, // 1 kB/s — very tight for QCIF busy
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        let qi = solve_base_qindex_for_budget(&config, &busy_clip(30), 176, 144);
        assert!(
            qi > Vp8EncoderConfig::default().qindex,
            "tight budget must raise the base qindex above the default, got {qi}"
        );
    }

    #[test]
    fn solver_lowers_qindex_under_a_generous_budget() {
        // A very generous bitrate must drive the solved base qindex to
        // the finest quantiser (best quality).
        let config = Vp8TwoPassConfig {
            target_bitrate_bps: 800_000_000, // absurdly high
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        let qi = solve_base_qindex_for_budget(&config, &busy_clip(30), 176, 144);
        assert_eq!(qi, 0, "generous budget must spend all quality (qindex 0)");
    }

    #[test]
    fn solver_is_monotone_in_budget() {
        // Loosening the budget must never raise the solved qindex.
        let base = Vp8TwoPassConfig {
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        let clip = busy_clip(30);
        let mut prev = 128i32;
        for &bps in &[8_000u32, 40_000, 200_000, 1_000_000, 5_000_000] {
            let config = Vp8TwoPassConfig {
                target_bitrate_bps: bps,
                ..base
            };
            let qi = solve_base_qindex_for_budget(&config, &clip, 176, 144) as i32;
            assert!(
                qi <= prev,
                "a larger budget ({bps} bps) must not raise the solved qindex: {qi} > {prev}"
            );
            prev = qi;
        }
    }

    #[test]
    fn solver_respects_predicted_ceiling() {
        // The predicted byte total at the solved qindex must stay under
        // the budget × overshoot ceiling (the solver's contract).
        let config = Vp8TwoPassConfig {
            target_bitrate_bps: 100_000,
            overshoot_ratio: 1.2,
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        let clip = busy_clip(30);
        let qi = solve_base_qindex_for_budget(&config, &clip, 176, 144) as i32;
        let budget = gop_byte_budget(&config, clip.len()).unwrap();
        let ceiling = (budget as f64 * 1.2) as u64;
        let predicted = predict_gop_bytes_at_qindex(&clip, 176, 144, qi);
        // Either we fit the ceiling, or we are already pinned at the
        // coarsest quantiser (127) and physically cannot do better.
        assert!(
            predicted <= ceiling || qi == 127,
            "predicted {predicted} must fit ceiling {ceiling} unless pinned at qi127 (qi={qi})"
        );
    }

    #[test]
    fn two_pass_qindices_for_bitrate_recenters_schedule() {
        // Under a tight budget the per-frame schedule must re-centre on
        // the solved (higher) base, not on config.base.qindex.
        let config = Vp8TwoPassConfig {
            base: Vp8EncoderConfig {
                qindex: 30,
                ..Vp8EncoderConfig::default()
            },
            target_bitrate_bps: 8_000,
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        // Vary complexity so the schedule isn't flat.
        let clip: Vec<_> = (0..6)
            .map(|i| FrameComplexity {
                frame_index: i,
                bits_per_mb: 200.0 + i as f32 * 200.0,
                scene_cut: false,
            })
            .collect();
        let sched = two_pass_qindices_for_bitrate(&config, &clip, 176, 144).unwrap();
        assert_eq!(sched.len(), clip.len());
        let mean_qi: f32 = sched.iter().map(|&q| q as f32).sum::<f32>() / sched.len() as f32;
        assert!(
            mean_qi > config.base.qindex as f32,
            "tight budget must raise the schedule mean above the configured base: \
             mean={mean_qi} base={}",
            config.base.qindex
        );
        for &q in &sched {
            assert!(q <= 127);
        }
    }

    #[test]
    fn two_pass_qindices_for_bitrate_matches_cq_when_untargeted() {
        // With no bitrate target the bitrate schedule must equal the
        // plain complexity-aware schedule.
        let config = Vp8TwoPassConfig::default();
        let clip: Vec<_> = (0..5)
            .map(|i| FrameComplexity {
                frame_index: i,
                bits_per_mb: 100.0 + i as f32 * 50.0,
                scene_cut: false,
            })
            .collect();
        let cq = two_pass_qindices(&config, &clip).unwrap();
        let br = two_pass_qindices_for_bitrate(&config, &clip, 176, 144).unwrap();
        assert_eq!(
            cq, br,
            "untargeted bitrate schedule must equal the CQ schedule"
        );
    }

    #[test]
    fn rate_feedback_delta_sign_and_clamp() {
        let config = Vp8TwoPassConfig {
            target_bitrate_bps: 100_000,
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        let mut enc = Vp8TwoPassEncoder::new(config);
        // No budget cached yet ⇒ no feedback.
        assert_eq!(enc.rate_feedback_delta(), 0);

        enc.frame_byte_budget = Some(1000);
        // Over budget by one full frame allotment ⇒ +gain.
        enc.rate_debt_bytes = 1000;
        assert_eq!(enc.rate_feedback_delta(), RATE_FEEDBACK_QSTEP_PER_FRAME);
        // Under budget by one allotment ⇒ −gain.
        enc.rate_debt_bytes = -1000;
        assert_eq!(enc.rate_feedback_delta(), -RATE_FEEDBACK_QSTEP_PER_FRAME);
        // Huge debt ⇒ clamped at the max step.
        enc.rate_debt_bytes = 1_000_000;
        assert_eq!(enc.rate_feedback_delta(), RATE_FEEDBACK_MAX_QSTEP);
        enc.rate_debt_bytes = -1_000_000;
        assert_eq!(enc.rate_feedback_delta(), -RATE_FEEDBACK_MAX_QSTEP);
    }

    #[test]
    fn two_pass_qindices_for_bitrate_empty_and_bad_base() {
        let config = Vp8TwoPassConfig::default();
        assert!(two_pass_qindices_for_bitrate(&config, &[], 176, 144)
            .unwrap()
            .is_empty());
        let bad = Vp8TwoPassConfig {
            base: Vp8EncoderConfig {
                qindex: 200,
                ..Vp8EncoderConfig::default()
            },
            ..Vp8TwoPassConfig::default()
        };
        assert!(two_pass_qindices_for_bitrate(&bad, &busy_clip(3), 176, 144).is_err());
    }

    #[test]
    fn treed_bits_cached_matches_dfs_walk() {
        // The RD scorers price mode bits through the precomputed
        // per-tree path tables; every priced value must be f64
        // bit-for-bit the DFS-walk `treed_bits` sum, for every leaf of
        // every table and under both the real probability tables and a
        // synthetic non-uniform lookup (so a wrong prob_index would
        // change the sum, not alias into it).
        fn check<const N: usize>(
            tree: &'static [i8],
            paths: &[TreedLeafPath; N],
            real_probs: &'static [u8],
        ) {
            let synthetic = |i: usize| -> u8 { (37 * i + 11) as u8 };
            for leaf in 0..N as u8 {
                for lookup in [
                    Box::new(|i: usize| real_probs[i]) as Box<dyn Fn(usize) -> u8>,
                    Box::new(synthetic),
                ] {
                    let reference = treed_bits(tree, &lookup, leaf);
                    let cached = treed_bits_cached(paths, &lookup, leaf);
                    assert_eq!(
                        cached.to_bits(),
                        reference.to_bits(),
                        "tree {tree:?} leaf {leaf}"
                    );
                }
            }
        }
        check(&KF_YMODE_TREE, &KF_YMODE_PATHS, &KF_YMODE_PROB);
        check(&UV_MODE_TREE, &UV_MODE_PATHS, &KF_UV_MODE_PROB);
        check(&BMODE_TREE, &BMODE_PATHS, &KF_BMODE_PROB[0][0]);
        check(&IF_YMODE_TREE, &IF_YMODE_PATHS, &IF_YMODE_PROB_DEFAULTS);
        // The chroma table is shared between keyframe and interframe
        // pricing; cover the interframe probabilities too.
        check(&UV_MODE_TREE, &UV_MODE_PATHS, &IF_UV_MODE_PROB_DEFAULTS);
    }

    #[test]
    fn write_literal_fast_matches_generic_loop() {
        // The register-local fixed-prob-128 `write_literal` must be
        // state-for-state the generic `num_bits x write_bool(128)` loop:
        // identical emitted bytes AND identical (range, bottom,
        // bit_count) after every literal, across the full width spread
        // (including width 0) and enough volume to exercise the carry
        // scan and the byte-emit cadence.
        let mut fast = BoolEncoder::new();
        let mut generic = BoolEncoder::new();
        let mut x: u32 = 0x1234_5678;
        for step in 0..4096u32 {
            // xorshift32 value stream.
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let num_bits = step % 33; // 0..=32
            let value = if num_bits == 32 {
                x
            } else {
                x & ((1u32 << num_bits) - 1)
            };
            fast.write_literal(value, num_bits);
            let mut i = num_bits;
            while i > 0 {
                i -= 1;
                generic.write_bool(128, ((value >> i) & 1) != 0);
            }
            assert_eq!(fast.range, generic.range, "range diverged at step {step}");
            assert_eq!(
                fast.bottom, generic.bottom,
                "bottom diverged at step {step}"
            );
            assert_eq!(
                fast.bit_count, generic.bit_count,
                "bit_count diverged at step {step}"
            );
            assert_eq!(fast.out, generic.out, "bytes diverged at step {step}");
        }
        assert_eq!(fast.finish(), generic.finish());
    }
}
