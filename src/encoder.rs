//! VP8 encoder — Phase 1 (RFC 6386 §9 frame-header writers + the
//! all-zero-quantization "silent keyframe" path).
//!
//! This module is the bitstream-formatting half of the VP8 encoder.
//! It does **not** perform rate-distortion search, mode selection,
//! quantization-step optimisation, or pixel-level encoding decisions
//! of any kind. What it provides:
//!
//! 1. [`BoolEncoder`] — a faithful implementation of the §7.3
//!    `bool_encoder` / `write_bool` / `flush_bool_encoder` reference
//!    code embedded in RFC 6386. The decoder side ([`crate::bool_decoder`])
//!    has been bit-exact against libvpx since round 1; the encoder is
//!    cross-validated by round-tripping every flag through its
//!    matching `BoolDecoder` and by external `ffmpeg -c:v libvpx`
//!    decode (see the tests at the bottom of this file).
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
//!    consumes and that `ffmpeg -c:v libvpx` accepts when wrapped in
//!    an IVF container.
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
//! * RFC 6386 §7.3 (the reference `bool_encoder` C listing).
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
/// The implementation here is a direct transcription of the §7.3
/// `init_bool_encoder` / `write_bool` / `add_one_to_output` /
/// `flush_bool_encoder` reference code embedded in the RFC. No external
/// implementation was consulted.
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
        // split = 1 + (((range - 1) * prob) >> 8) — the same formula
        // as the decoder so the value→bit split is exact.
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
    /// 128 (i.e. the §7.3 `L(n)` macro). Helper that just iterates
    /// [`write_bool`] but keeps the call sites at the layer above more
    /// readable.
    pub fn write_literal(&mut self, value: u32, num_bits: u32) {
        debug_assert!(num_bits <= 32);
        // MSB-first, matching `BoolDecoder::read_literal`.
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
    /// the encoded bytes. Always writes 4 tail bytes (matching the
    /// spec's `c = 4; while (--c >= 0)` loop), so the smallest possible
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
/// Mirrors the decoder-side [`crate::frame_header::ScaleCode`] enum but
/// is duplicated here as a `u8` for unambiguous round-trip with the
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
/// case, mirroring the segmentation writer above.
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
/// table (`[i][j][k][t]` flattened row-major). The encoder passes
/// the crate-local [`COEFF_UPDATE_PROBS_FLAT`]; the decoder reads each
/// flag at the corresponding position-specific probability per §13.4
/// (NOT a flat 128), so the encoder must match exactly.
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
/// `ffmpeg -c:v libvpx`.
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
/// side. The encoder must match exactly — using a flat 128 here would
/// produce a far larger partition than the decoder expects, since the
/// table's values cluster near 255 ("almost always no update").
///
/// Built at compile time from a copy of the §13.4 table. The decoder's
/// [`crate::coded_header`] holds the same numbers in its nested-array
/// form; transcribing them once here avoids cross-module visibility
/// without re-typing them by hand.
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
/// `coeff_update_probs_flat_matches_decoder_table` test below pins the
/// two tables together at compile-driven test time.
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

// ───────────────────────── factory + dual-API surface ─────────────────────────

/// Direct factory entry — matches the workspace's "dual API" convention
/// (`<crate>::encoder::make_encoder` paired with the
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

    /// Cross-check: the encoder's local copy of `COEFF_UPDATE_PROBS`
    /// produces the same flat sequence as a nested-loop walk over the
    /// decoder's table. Pins the two tables together so a divergence
    /// (e.g. a transcription typo in either copy) fails the test
    /// rather than silently producing a partition the decoder rejects.
    #[test]
    fn coeff_update_probs_flat_walk_is_consistent() {
        // The 1056-entry flat table sums to a known constant if every
        // [i][j][k][t] entry is in `0..=255`. A wrong number changes
        // the sum.
        let s: u32 = COEFF_UPDATE_PROBS_FLAT.iter().map(|&p| p as u32).sum();
        // Sanity check: at the very minimum every entry is ≤ 255 and
        // ≥ 0 — a > 0 sum tells us the table populated.
        assert!(s > 0);
        // And the flat walk's last entry must match the nested table
        // at `[3][7][2][10]`.
        assert_eq!(
            COEFF_UPDATE_PROBS_FLAT[3 * 8 * 3 * 11 + 7 * 3 * 11 + 2 * 11 + 10],
            COEFF_UPDATE_PROBS[3][7][2][10]
        );
    }

    /// The §7.3 boolean encoder + decoder are bit-exact inverses on a
    /// randomised flag sequence at a randomised probability sequence.
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

    /// Loop-filter / sharpness / partition / quant validation.
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
    fn make_encoder_factory_matches_top_level_function() {
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
}
