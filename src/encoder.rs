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
    /// 128 (i.e. the §7.3 `L(n)` macro). Helper that iterates
    /// [`write_bool`] but keeps the call sites at the layer above more
    /// readable.
    pub fn write_literal(&mut self, value: u32, num_bits: u32) {
        debug_assert!(num_bits <= 32);
        // MSB-first, paired with `BoolDecoder::read_literal`.
        let mut i = num_bits;
        while i > 0 {
            i -= 1;
            self.write_bool(128, ((value >> i) & 1) != 0);
        }
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

/// §9.3 segment-update flags writer.
///
/// Phase 1 only supports the disabled path (`enabled = false`), which
/// emits the single `L(1) = 0` toggle. The `update_mb_segmentation_map`
/// / `update_segment_feature_data` / per-segment quantizer / per-segment
/// loop-filter / `mb_segment_tree_probs` sub-blocks of §9.3 are not
/// written. When a later round needs them, this signature will grow a
/// richer parameter struct.
pub fn write_segment_update_flags(enc: &mut BoolEncoder, enabled: bool) {
    debug_assert!(
        !enabled,
        "Phase 1 encoder only supports segmentation_enabled = false"
    );
    enc.write_bool(128, enabled);
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
pub(crate) const COEFF_UPDATE_PROBS_FLAT: [u8; 1056] = {
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

/// Walk [`ENC_COEFF_TREE`] starting at internal node `start_index` and
/// record the `(prob_index, bit)` sequence that lands at the leaf for
/// `target`. Mirrors the decoder's `treed_read_coef` traversal but
/// runs backwards from a known leaf.
///
/// Returns the list of `(i_half, bit)` pairs in the exact order the
/// decoder will read them. The `start_index` distinguishes the §13.2
/// "may emit EOB" case (`0`) from the "previous coefficient was DCT_0"
/// case (`2`, EOB branch bypassed).
fn token_to_bit_path(token: DctToken, start_index: i8) -> Vec<(usize, bool)> {
    let target = token as i8;

    fn descend(i: i8, target: i8, path: &mut Vec<(usize, bool)>) -> bool {
        for &bit in &[false, true] {
            let child = ENC_COEFF_TREE[i as usize + bit as usize];
            if child <= 0 {
                if -child == target {
                    path.push(((i as usize) >> 1, bit));
                    return true;
                }
            } else {
                path.push(((i as usize) >> 1, bit));
                if descend(child, target, path) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }

    let mut out = Vec::with_capacity(8);
    let ok = descend(start_index, target, &mut out);
    debug_assert!(
        ok,
        "token {token:?} not reachable from coeff_tree index {start_index}"
    );
    out
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

        for (i_half, bit) in token_to_bit_path(token, start) {
            enc.write_bool(probs[i_half], bit);
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
}

/// Encode one macroblock's residual through the §13 / §14 block-set
/// walker — the inverse of the §14.2 reconstruction orchestrator.
///
/// # Behavior (Phase 2 scope)
///
/// * **Prediction**: `DC_PRED` constant 128 (the §12.2 default for a
///   macroblock with no neighbours — i.e. the top-left MB of a frame).
///   The residual is `pixel - 128` per channel.
/// * **Forward transforms**: §14.4 forward DCT on every Y, U, V 4×4
///   sub-block (16 + 4 + 4 = 24 blocks); the 16 Y DCs are collected
///   into a Y2 block and §14.3 forward-WHT'd.
/// * **Quantisation**: §14.1 / §20.4 factors per
///   [`MbDequantFactors::from_quant_indices`], with the `MbCoeffs`
///   plane layout matching what [`decode_and_dequantize_mb`] produces
///   on the decoder side.
/// * **Token coding**: §13.3 walk in residual order Y2 → 16 Y
///   (`YAfterY2`) → 4 U (`UV`) → 4 V (`UV`), threaded through fresh
///   above / left [`MbEntropyCtx`] (i.e. off-frame neighbours, which
///   matches the test fixture's single-MB frame).
///
/// # Roundtrip guarantee
///
/// On a flat-colour Y / U / V macroblock at `yac_qi = 0` the bytes
/// `decode_mb_coeffs` + `MbDequantFactors::dequantize` + the §14.2 /
/// §12.2 reconstruction orchestrator return exactly recover the input
/// pixel value (the FDCT / FWHT chain concentrates a flat block into a
/// single DC, which round-trips losslessly at `q = 0` per the existing
/// `flat_residual_block_roundtrips_losslessly_at_q0` test). The
/// `mb_block_set_roundtrip_flat_color_recovers_within_one_lsb` test in
/// the same module checks this end-to-end.
///
/// # Out of scope (deferred)
///
/// * Non-DC prediction modes (V / H / TM / B_PRED + the 8×8 chroma
///   variants), inter prediction, mode RD search. The encoder will
///   gain those when the §11 mode-selection round lands.
/// * Multi-MB neighbour evolution. This entry encodes one MB against
///   off-frame neighbours; the per-frame raster driver that threads
///   `MbEntropyCtx` columns through a frame is the next layer.
pub fn encode_mb_block_set(
    pixels: &MbPixels,
    yac_qi: u8,
    coeff_probs: &crate::dct_tokens::CoeffProbs,
) -> Result<EncodedMb, TokenEncodeError> {
    // ---- 1. Build the §14.1 dequant factors for this MB's segment.
    // The encoder's quantisation step is the inverse — divide by these.
    let factors =
        crate::dequant::MbDequantFactors::from_base_and_deltas(yac_qi as i32, 0, 0, 0, 0, 0);

    // ---- 2. Forward-transform the residual.
    //
    // §12.2 DC_PRED with no above / left neighbours gives a flat
    // 128-prediction for every plane (§12.2 page 51 "if neither
    // exists, the average is 128"). The residual is `pixel - 128`.
    let dc_pred: i16 = 128;

    let mut raw_coeffs = MbCoeffs::default();

    // 16 Y sub-blocks: extract → FDCT into raster-order arrays.
    // Y sub-block order matches §14.2: index `i*4 + j` for row i,
    // column j of the 4×4 sub-block grid in a 16×16 plane.
    for i in 0..4 {
        for j in 0..4 {
            let mut residual = [0i16; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let py = i * 4 + r;
                    let px = j * 4 + c;
                    residual[r * 4 + c] = pixels.y[py * 16 + px] as i16 - dc_pred;
                }
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&residual, &mut coeffs);
            raw_coeffs.y[i * 4 + j] = coeffs;
        }
    }

    // 4 U sub-blocks.
    for i in 0..2 {
        for j in 0..2 {
            let mut residual = [0i16; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let py = i * 4 + r;
                    let px = j * 4 + c;
                    residual[r * 4 + c] = pixels.u[py * 8 + px] as i16 - dc_pred;
                }
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&residual, &mut coeffs);
            raw_coeffs.u[i * 2 + j] = coeffs;
        }
    }

    // 4 V sub-blocks.
    for i in 0..2 {
        for j in 0..2 {
            let mut residual = [0i16; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let py = i * 4 + r;
                    let px = j * 4 + c;
                    residual[r * 4 + c] = pixels.v[py * 8 + px] as i16 - dc_pred;
                }
            }
            let mut coeffs = [0i16; 16];
            forward_dct_4x4(&residual, &mut coeffs);
            raw_coeffs.v[i * 2 + j] = coeffs;
        }
    }

    // ---- 3. Collect the 16 Y DCs and run §14.3 forward WHT into Y2.
    //
    // §14.2 first paragraph (inverse direction): "the element of the
    // result at row i, column j is used as the 0th coefficient of the
    // Y subblock at position (i, j)" — i.e. Y2[i*4+j] seeds Y[i*4+j]'s
    // DC. The encode-side inverse: take DC[i*4+j] from each Y
    // sub-block, build a 4×4 array, FWHT, then zero out each Y
    // sub-block's DC (since it now lives in Y2).
    let mut y_dc_block = [0i16; 16];
    for (slot, src) in y_dc_block.iter_mut().zip(raw_coeffs.y.iter()) {
        *slot = src[0];
    }
    let mut y2_coeffs = [0i16; 16];
    forward_wht_4x4(&y_dc_block, &mut y2_coeffs);
    raw_coeffs.y2 = y2_coeffs;
    for blk in raw_coeffs.y.iter_mut() {
        blk[0] = 0;
    }

    // ---- 4. Quantise every block against its plane's §14.1 factors.
    //
    // The encoder's quant step is the natural inverse of the
    // decoder's `MbDequantFactors::dequantize`: divide each
    // coefficient by the matching DC / AC factor with round-half-
    // away-from-zero. The Y sub-blocks' DCs are zero (consumed by
    // Y2 above), but we still pass `factors.y1_dc` for the divisor
    // — round-half-away-from-zero of 0 stays 0.
    enc_quantize_block(&mut raw_coeffs.y2, factors.y2_dc, factors.y2_ac);
    for blk in raw_coeffs.y.iter_mut() {
        enc_quantize_block(blk, factors.y1_dc, factors.y1_ac);
    }
    for blk in raw_coeffs.u.iter_mut() {
        enc_quantize_block(blk, factors.uv_dc, factors.uv_ac);
    }
    for blk in raw_coeffs.v.iter_mut() {
        enc_quantize_block(blk, factors.uv_dc, factors.uv_ac);
    }

    // ---- 5. Walk §13.3 residual order, encode each block in scan
    //         order against fresh above / left predictor contexts.
    //
    // We use the same `decode_mb_coeffs` walk order:
    //   1. Y2 (if has_y2)                          — block 24
    //   2. 16 Y sub-blocks (YAfterY2 plane)        — blocks 0..15
    //   3. 4 U sub-blocks (UV plane)               — blocks 16..19
    //   4. 4 V sub-blocks (UV plane)               — blocks 20..23
    //
    // The above / left predictor seeds for the first block are both
    // off-frame ("false") since this entry encodes a single isolated
    // MB. Each block updates its slot in both contexts per §13.3.
    let mut enc = BoolEncoder::new();
    let mut above = MbEntropyCtx::default();
    let mut left = MbEntropyCtx::default();
    let mut nonzero_block_count = 0usize;

    let scan_y2 = raster_to_scan(&raw_coeffs.y2);
    let nz = encode_block_with_ctx(
        &mut enc,
        24,
        BlockType::Y2,
        coeff_probs,
        &scan_y2,
        &mut above,
        &mut left,
    )?;
    if nz != 0 {
        nonzero_block_count += 1;
    }

    for (i, y_block) in raw_coeffs.y.iter().enumerate() {
        let scan = raster_to_scan(y_block);
        let nz = encode_block_with_ctx(
            &mut enc,
            i,
            BlockType::YAfterY2,
            coeff_probs,
            &scan,
            &mut above,
            &mut left,
        )?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    for (i, u_block) in raw_coeffs.u.iter().enumerate() {
        let scan = raster_to_scan(u_block);
        let nz = encode_block_with_ctx(
            &mut enc,
            16 + i,
            BlockType::UV,
            coeff_probs,
            &scan,
            &mut above,
            &mut left,
        )?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    for (i, v_block) in raw_coeffs.v.iter().enumerate() {
        let scan = raster_to_scan(v_block);
        let nz = encode_block_with_ctx(
            &mut enc,
            20 + i,
            BlockType::UV,
            coeff_probs,
            &scan,
            &mut above,
            &mut left,
        )?;
        if nz != 0 {
            nonzero_block_count += 1;
        }
    }

    let bytes = enc.finish();
    Ok(EncodedMb {
        coeffs: raw_coeffs,
        bytes,
        nonzero_block_count,
    })
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

// ───────────────────────── factory + dual-API surface ─────────────────────────

/// Direct factory entry — paired with the workspace's "dual API"
/// convention (`<crate>::encoder::make_encoder` alongside the
/// `oxideav_core::register!` registry path). Phase 1 only emits silent
/// keyframes, so the factory's `encode_keyframe` method is a thin
/// wrapper over [`encode_silent_keyframe`]. The signature keeps the
/// pixel slice in the API for forward compatibility with the §13 /
/// §14 encoder round (which will actually consume it); for now it is
/// ignored.
pub fn make_encoder() -> SilentKeyframeEncoder {
    SilentKeyframeEncoder
}

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

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;
    use crate::coded_header::Vp8CodedHeader;
    use crate::decode_vp8;
    use crate::frame_header::Vp8FrameHeader;

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

    /// `make_encoder()` reaches the same byte sequence as
    /// `encode_silent_keyframe`.
    #[test]
    fn make_encoder_factory_produces_same_bytes_as_top_level_function() {
        let enc = make_encoder();
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
    use crate::frame::{decode_keyframe, MbCoeffs as FrameMbCoeffs};
    use crate::macroblock::{IntraUvMode, IntraYMode, MacroblockModes};

    /// Build a single-macroblock frame with DC_PRED + given
    /// pre-dequantized coefficients and return the reconstructed luma
    /// plane.
    fn decode_single_mb(coeffs: FrameMbCoeffs) -> Vec<u8> {
        let modes = vec![MacroblockModes {
            segment_id: None,
            mb_skip_coeff: false,
            y_mode: IntraYMode::Dc,
            subblock_modes: None,
            uv_mode: IntraUvMode::Dc,
        }];
        let planes = decode_keyframe(1, 1, &modes, &[coeffs]).expect("decode_keyframe");
        planes.y
    }

    /// Encode a flat-color MB through `encode_mb_block_set`, decode the
    /// bytes back into the per-MB raw coefficients, dequantize, and run
    /// the §14.2 reconstruction orchestrator — the recovered luma /
    /// chroma planes must be within ≤ 1 LSB of the input.
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
            // (`DC_QLOOKUP[0] = AC_QLOOKUP[0] = 4`).
            let encoded =
                encode_mb_block_set(&pixels, 0, &DEFAULT_COEFF_PROBS).expect("encode_mb_block_set");

            // Decode the emitted bytes back into raw quantized coeffs
            // through `decode_mb_coeffs` (same predictor seeds: fresh
            // contexts since this is a single isolated MB).
            let mut dec = BoolDecoder::init(&encoded.bytes).expect("encoder emits ≥ 2 bytes");
            let mut above = MbEntropyCtx::default();
            let mut left = MbEntropyCtx::default();
            let mut recovered_raw = decode_mb_coeffs(
                &mut dec,
                true,
                false,
                &DEFAULT_COEFF_PROBS,
                &mut above,
                &mut left,
            )
            .expect("decode_mb_coeffs");

            // The decoder's recovered raw coefficients must match what
            // the encoder produced (the §13.3 byte-stream layer is the
            // tight invariant; this is the same check the standalone
            // `encode_coeff_block` test in this module exercises but
            // composed across all 25 residual blocks).
            assert_eq!(
                recovered_raw, encoded.coeffs,
                "encoded coeffs differ from decoded raw coeffs (pixel = {pixel})"
            );

            // Dequantize and reconstruct.
            let factors = MbDequantFactors::from_base_and_deltas(0, 0, 0, 0, 0, 0);
            factors.dequantize(&mut recovered_raw);
            let y_plane = decode_single_mb(recovered_raw);

            // Verify every reconstructed luma pixel is within ≤ 1 LSB
            // of the input. At yac_qi = 0 the chain is bit-exact for a
            // flat block (per the existing per-block roundtrip test),
            // but the test bounds at ≤ 1 to leave room for the §14.5
            // clamp / rounding behaviour on extreme inputs.
            for (i, &recon) in y_plane.iter().enumerate() {
                assert!(
                    (recon as i32 - pixel as i32).abs() <= 1,
                    "pixel {pixel}: recon[{i}] = {recon} differs by > 1 LSB"
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
        let encoded =
            encode_mb_block_set(&pixels, 16, &DEFAULT_COEFF_PROBS).expect("encode_mb_block_set");

        let mut dec = BoolDecoder::init(&encoded.bytes).expect("encoder emits ≥ 2 bytes");
        let mut above = MbEntropyCtx::default();
        let mut left = MbEntropyCtx::default();
        let mut recovered_raw = decode_mb_coeffs(
            &mut dec,
            true,
            false,
            &DEFAULT_COEFF_PROBS,
            &mut above,
            &mut left,
        )
        .expect("decode_mb_coeffs");
        assert_eq!(recovered_raw, encoded.coeffs);

        let factors = MbDequantFactors::from_base_and_deltas(16, 0, 0, 0, 0, 0);
        factors.dequantize(&mut recovered_raw);
        let y_plane = decode_single_mb(recovered_raw);

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
}
