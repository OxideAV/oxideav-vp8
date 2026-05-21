//! VP8 uncompressed frame header (RFC 6386 §9.1 / §19.1).
//!
//! The first thing every VP8 frame ships is a **3-byte frame tag**
//! followed, *only for key frames*, by a 3-byte start code and a pair
//! of 16-bit fields encoding `(scale, dimension)` for the horizontal
//! and vertical axes. The frame tag fields are:
//!
//! * bit 0 — `key_frame` (0 = key frame, 1 = interframe).
//! * bits 1..3 — `version` (3-bit profile index; see
//!   [`ReconstructionFilter`] and [`LoopFilterPolicy`]).
//! * bit 4 — `show_frame` (0 = invisible / alt-ref, 1 = displayed).
//! * bits 5..23 — `first_partition_size` (19-bit byte count of the
//!   "control" partition that immediately follows the uncompressed
//!   header).
//!
//! The 24-bit frame tag is parsed little-endian per RFC 6386 §19.1:
//! `tmp = (c[2] << 16) | (c[1] << 8) | c[0]`. For key frames the
//! three start-code bytes that follow must equal `0x9d 0x01 0x2a`.
//! Each of the two key-frame size words is a little-endian u16 whose
//! low 14 bits are the pixel dimension and whose top 2 bits are the
//! upscale code (0..3; see [`ScaleCode`]).
//!
//! Composition with the rest of the decoder: the uncompressed header
//! describes the byte layout that contains the *boolean-coded*
//! control partition. The `first_partition_size` field is the
//! input length the [`BoolDecoder`](crate::bool_decoder::BoolDecoder)
//! will be initialised over, once round 3 lands the section §19.2
//! parsing.
//!
//! This module is bitstream-structural only: it surfaces the
//! parsed field values; it does not yet drive the bool decoder.

use core::fmt;

/// Errors surfaced by [`Vp8FrameHeader::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameHeaderError {
    /// The input is shorter than 3 bytes (interframe) or 10 bytes
    /// (key frame) and so the uncompressed chunk cannot be read.
    InputTooShort,
    /// The frame's `key_frame` bit is 0 but the three bytes that
    /// should follow the frame tag are not the spec-mandated
    /// `0x9d 0x01 0x2a` start code (RFC 6386 §9.1).
    InvalidStartCode {
        /// The three bytes that were found in the start-code slot.
        got: [u8; 3],
    },
}

impl fmt::Display for FrameHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameHeaderError::InputTooShort => {
                write!(f, "vp8 frame header: input shorter than required")
            }
            FrameHeaderError::InvalidStartCode { got } => {
                write!(
                    f,
                    "vp8 frame header: invalid key-frame start code: got [{:#04x}, {:#04x}, {:#04x}], want [0x9d, 0x01, 0x2a]",
                    got[0], got[1], got[2]
                )
            }
        }
    }
}

impl std::error::Error for FrameHeaderError {}

/// The mandatory three-byte start code that follows the frame tag on
/// key frames, per RFC 6386 §9.1.
pub const KEY_FRAME_START_CODE: [u8; 3] = [0x9d, 0x01, 0x2a];

/// Reconstruction filter implied by the 3-bit `version` field.
///
/// Per RFC 6386 §9.1 Table 1. Sub-pixel reconstruction filters in
/// VP8 are part of the inter-prediction path; versions other than 0
/// reduce sub-pel accuracy in exchange for decode complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionFilter {
    /// Version 0 — bicubic sub-pel filter.
    Bicubic,
    /// Versions 1 and 2 — bilinear sub-pel filter.
    Bilinear,
    /// Version 3 — no sub-pel filtering (full-pixel motion).
    None,
    /// Reserved version > 3.
    Reserved,
}

/// Loop-filter policy implied by the 3-bit `version` field.
///
/// Per RFC 6386 §9.1 Table 1 and surrounding text. Per the RFC the
/// loop-filter policy implied by the version field is only an
/// encoder-side hint — the actual filter parameters that drive the
/// decoder live in the §9.4 frame header. We surface the policy
/// anyway so the parser preserves all four version fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopFilterPolicy {
    /// Version 0 — full normal loop filter.
    Normal,
    /// Version 1 — the encoder hints "simple" loop filter (no
    /// decoder effect on its own).
    Simple,
    /// Versions 2 and 3 — the encoder hints "no loop filter" (no
    /// decoder effect on its own).
    None,
    /// Reserved version > 3.
    Reserved,
}

/// Upscale code for one axis, occupying the top 2 bits of the
/// 16-bit key-frame size word per RFC 6386 §9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleCode {
    /// 0 — no upscaling. The reconstruction buffer is the display.
    None,
    /// 1 — upscale by 5/4.
    FiveFourths,
    /// 2 — upscale by 5/3.
    FiveThirds,
    /// 3 — upscale by 2.
    Double,
}

impl ScaleCode {
    /// Construct from the raw 2-bit code (`0..=3`). Values outside
    /// the range are saturated to [`ScaleCode::None`] under
    /// `debug_assertions`; in release builds the high bits are
    /// masked off and the low 2 bits decoded.
    fn from_code(code: u16) -> Self {
        match code & 0b11 {
            0 => ScaleCode::None,
            1 => ScaleCode::FiveFourths,
            2 => ScaleCode::FiveThirds,
            _ => ScaleCode::Double,
        }
    }
}

/// Parsed VP8 uncompressed frame header per RFC 6386 §9.1.
///
/// On a key frame all fields are populated. On an interframe the
/// six key-frame-only fields (`start_code_ok`, `width`, `height`,
/// `horizontal_scale`, `vertical_scale`, plus the implied
/// `header_size` accounting for the start code + size words) shrink
/// to "frame tag only" — three bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8FrameHeader {
    /// `true` when this frame is a key frame (the §9.1 `frame_type`
    /// bit reads 0). On interframes this is `false`.
    pub key_frame: bool,
    /// Raw `version` field (`0..=7`). The 3-bit value addresses the
    /// `Reconstruction Filter` / `Loop Filter` table in §9.1.
    pub version: u8,
    /// Decoded reconstruction filter implied by `version`.
    pub reconstruction_filter: ReconstructionFilter,
    /// Decoded loop-filter policy implied by `version` (hint only,
    /// see [`LoopFilterPolicy`]).
    pub loop_filter_policy: LoopFilterPolicy,
    /// `show_frame` bit. `false` is used by encoders for invisible
    /// alt-ref frames (RFC 6386 §9.1).
    pub show_frame: bool,
    /// Byte length of the "first partition" — the boolean-coded
    /// control partition that immediately follows the uncompressed
    /// header. 19 bits wide (`0..=0x7_FFFF`).
    pub first_partition_size: u32,
    /// Pixel width (key frame only). 14 bits wide (`0..=0x3FFF`).
    /// `None` on interframes; dimensions are inherited.
    pub width: Option<u16>,
    /// Pixel height (key frame only). 14 bits wide. `None` on
    /// interframes.
    pub height: Option<u16>,
    /// Horizontal upscale (key frame only).
    pub horizontal_scale: Option<ScaleCode>,
    /// Vertical upscale (key frame only).
    pub vertical_scale: Option<ScaleCode>,
    /// Number of bytes consumed from the input: 3 for an interframe,
    /// 10 for a key frame. Callers can advance the cursor by this
    /// amount to land at the start of the first (control)
    /// partition.
    pub header_bytes_consumed: usize,
}

impl Vp8FrameHeader {
    /// Parse the uncompressed VP8 frame header from the start of
    /// `bytes`.
    ///
    /// Required input length: 3 bytes for interframes, 10 bytes for
    /// key frames (3-byte frame tag + 3-byte start code + 4 bytes
    /// of width/height/scale).
    pub fn parse(bytes: &[u8]) -> Result<Self, FrameHeaderError> {
        if bytes.len() < 3 {
            return Err(FrameHeaderError::InputTooShort);
        }

        // RFC 6386 §19.1 frame-tag decomposition. The 24-bit value is
        // assembled little-endian from bytes [0..3] and then split by
        // bit-field shifts as named in the spec.
        let tmp = (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16);

        // bit 0 of the frame tag: `frame_type` (0 = key, 1 = inter).
        let key_frame = (tmp & 0x1) == 0;
        // bits 1..3: 3-bit version.
        let version = ((tmp >> 1) & 0x7) as u8;
        // bit 4: show_frame.
        let show_frame = ((tmp >> 4) & 0x1) != 0;
        // bits 5..23: 19-bit first-partition byte count.
        let first_partition_size = (tmp >> 5) & 0x7_FFFF;

        let (reconstruction_filter, loop_filter_policy) = decode_version(version);

        if !key_frame {
            // Interframe: no start code, no width/height/scale fields.
            return Ok(Vp8FrameHeader {
                key_frame,
                version,
                reconstruction_filter,
                loop_filter_policy,
                show_frame,
                first_partition_size,
                width: None,
                height: None,
                horizontal_scale: None,
                vertical_scale: None,
                header_bytes_consumed: 3,
            });
        }

        // Key frame path: need 7 more bytes (start code + two
        // little-endian 16-bit size words).
        if bytes.len() < 10 {
            return Err(FrameHeaderError::InputTooShort);
        }
        let start_code = [bytes[3], bytes[4], bytes[5]];
        if start_code != KEY_FRAME_START_CODE {
            return Err(FrameHeaderError::InvalidStartCode { got: start_code });
        }

        // The two size words are little-endian per §19.1's
        // `(c[1] << 8) | c[0]` formulation.
        let h_word = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        let v_word = (bytes[8] as u16) | ((bytes[9] as u16) << 8);

        let width = h_word & 0x3FFF;
        let horizontal_scale = ScaleCode::from_code(h_word >> 14);
        let height = v_word & 0x3FFF;
        let vertical_scale = ScaleCode::from_code(v_word >> 14);

        Ok(Vp8FrameHeader {
            key_frame,
            version,
            reconstruction_filter,
            loop_filter_policy,
            show_frame,
            first_partition_size,
            width: Some(width),
            height: Some(height),
            horizontal_scale: Some(horizontal_scale),
            vertical_scale: Some(vertical_scale),
            header_bytes_consumed: 10,
        })
    }
}

/// Map the raw 3-bit `version` field to the reconstruction filter
/// and loop-filter policy described in RFC 6386 §9.1 Table 1.
fn decode_version(version: u8) -> (ReconstructionFilter, LoopFilterPolicy) {
    match version {
        0 => (ReconstructionFilter::Bicubic, LoopFilterPolicy::Normal),
        1 => (ReconstructionFilter::Bilinear, LoopFilterPolicy::Simple),
        2 => (ReconstructionFilter::Bilinear, LoopFilterPolicy::None),
        3 => (ReconstructionFilter::None, LoopFilterPolicy::None),
        _ => (ReconstructionFilter::Reserved, LoopFilterPolicy::Reserved),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_input_below_three_bytes() {
        assert_eq!(
            Vp8FrameHeader::parse(&[]).unwrap_err(),
            FrameHeaderError::InputTooShort
        );
        assert_eq!(
            Vp8FrameHeader::parse(&[0x00, 0x00]).unwrap_err(),
            FrameHeaderError::InputTooShort
        );
    }

    #[test]
    fn rejects_key_frame_with_short_input_below_ten_bytes() {
        // key_frame=0 bit set: byte0 bit0 == 0. Other fields zero.
        // We have a valid 3-byte frame tag but only 5 bytes total —
        // not enough for the start code + size words.
        let buf = [0x00, 0x00, 0x00, 0x9d, 0x01];
        assert_eq!(
            Vp8FrameHeader::parse(&buf).unwrap_err(),
            FrameHeaderError::InputTooShort
        );
    }

    #[test]
    fn rejects_key_frame_with_wrong_start_code() {
        // Frame tag: key_frame=0, version=0, show_frame=0,
        // first_part_size=0. Then a deliberately-wrong start code.
        let buf = [0x00, 0x00, 0x00, 0xde, 0xad, 0xbe, 0x10, 0x00, 0x10, 0x00];
        assert_eq!(
            Vp8FrameHeader::parse(&buf).unwrap_err(),
            FrameHeaderError::InvalidStartCode {
                got: [0xde, 0xad, 0xbe]
            }
        );
    }

    #[test]
    fn interframe_consumes_three_bytes_and_omits_dimensions() {
        // tmp = 0b0...0001 → key_frame_flag bit set → interframe.
        // Pick values that exercise each field independently:
        //   version = 5         → bits 1..3 = 101
        //   show_frame = 1      → bit  4    = 1
        //   first_part_size = 0x1_2345 → bits 5..23 = 0x1_2345
        // Compose: tmp = 0x1 | (0x5 << 1) | (0x1 << 4) | (0x1_2345 << 5)
        //              = 0x1 | 0xA | 0x10 | 0x2_468A_0
        //              = 0x2_468B_B
        let first_part_size: u32 = 0x1_2345;
        let tmp: u32 = 0x1 | (0x5 << 1) | (0x1 << 4) | (first_part_size << 5);
        let bytes = [
            (tmp & 0xff) as u8,
            ((tmp >> 8) & 0xff) as u8,
            ((tmp >> 16) & 0xff) as u8,
            0xFF, // junk that must NOT be consumed
            0xFF,
        ];
        let hdr = Vp8FrameHeader::parse(&bytes).unwrap();
        assert!(!hdr.key_frame);
        assert_eq!(hdr.version, 5);
        assert!(hdr.show_frame);
        assert_eq!(hdr.first_partition_size, first_part_size);
        assert_eq!(hdr.header_bytes_consumed, 3);
        assert_eq!(hdr.width, None);
        assert_eq!(hdr.height, None);
        assert_eq!(hdr.horizontal_scale, None);
        assert_eq!(hdr.vertical_scale, None);
        // version=5 lies in the reserved range of §9.1 Table 1.
        assert_eq!(hdr.reconstruction_filter, ReconstructionFilter::Reserved);
        assert_eq!(hdr.loop_filter_policy, LoopFilterPolicy::Reserved);
    }

    #[test]
    fn version_table_maps_all_four_defined_codes() {
        // (version, expected filter, expected policy) per §9.1 Table 1.
        let cases: &[(u8, ReconstructionFilter, LoopFilterPolicy)] = &[
            (0, ReconstructionFilter::Bicubic, LoopFilterPolicy::Normal),
            (1, ReconstructionFilter::Bilinear, LoopFilterPolicy::Simple),
            (2, ReconstructionFilter::Bilinear, LoopFilterPolicy::None),
            (3, ReconstructionFilter::None, LoopFilterPolicy::None),
        ];
        for &(v, want_filter, want_policy) in cases {
            // Build a minimal interframe tag with the version field
            // set to v and everything else cleared.
            let tmp: u32 = 0x1 | ((v as u32) << 1);
            let bytes = [
                (tmp & 0xff) as u8,
                ((tmp >> 8) & 0xff) as u8,
                ((tmp >> 16) & 0xff) as u8,
            ];
            let hdr = Vp8FrameHeader::parse(&bytes).unwrap();
            assert_eq!(hdr.version, v);
            assert_eq!(hdr.reconstruction_filter, want_filter, "version={v}");
            assert_eq!(hdr.loop_filter_policy, want_policy, "version={v}");
        }
    }

    #[test]
    fn scale_code_decodes_all_four_values() {
        assert_eq!(ScaleCode::from_code(0), ScaleCode::None);
        assert_eq!(ScaleCode::from_code(1), ScaleCode::FiveFourths);
        assert_eq!(ScaleCode::from_code(2), ScaleCode::FiveThirds);
        assert_eq!(ScaleCode::from_code(3), ScaleCode::Double);
    }

    #[test]
    fn key_frame_parses_real_tiny_fixture_bytes() {
        // First 10 bytes of the VP8 frame inside
        // docs/video/vp8/fixtures/tiny-i-only-16x16/input.ivf, lifted
        // by hand from `xxd input.ivf` (offset 0x2c .. 0x36).
        //
        // Frame tag = 0xf0 0x02 0x00. tmp = 0x0002f0.
        //   key_frame_flag = tmp & 1     = 0  (KEY frame)
        //   version        = (tmp>>1)&7  = 0
        //   show_frame     = (tmp>>4)&1  = 1
        //   first_part_size= (tmp>>5)&0x7FFFF = 0x17 = 23
        //
        // Start code = 0x9d 0x01 0x2a (required for key frames).
        // h_word = 0x0010 → width=16, hscale=0.
        // v_word = 0x0010 → height=16, vscale=0.
        let bytes = [0xf0, 0x02, 0x00, 0x9d, 0x01, 0x2a, 0x10, 0x00, 0x10, 0x00];
        let hdr = Vp8FrameHeader::parse(&bytes).unwrap();
        assert!(hdr.key_frame);
        assert_eq!(hdr.version, 0);
        assert!(hdr.show_frame);
        assert_eq!(hdr.first_partition_size, 23);
        assert_eq!(hdr.width, Some(16));
        assert_eq!(hdr.height, Some(16));
        assert_eq!(hdr.horizontal_scale, Some(ScaleCode::None));
        assert_eq!(hdr.vertical_scale, Some(ScaleCode::None));
        assert_eq!(hdr.header_bytes_consumed, 10);
        assert_eq!(hdr.reconstruction_filter, ReconstructionFilter::Bicubic);
        assert_eq!(hdr.loop_filter_policy, LoopFilterPolicy::Normal);
    }

    #[test]
    fn key_frame_max_width_height_round_trip() {
        // Width 0x3FFF (the 14-bit max) and height likewise, with
        // horizontal scale 3 (×2) and vertical scale 2 (×5/3) packed
        // into the top two bits of each size word per §9.1.
        //
        // h_word = (0b11 << 14) | 0x3FFF = 0xFFFF
        // v_word = (0b10 << 14) | 0x3FFF = 0xBFFF
        //
        // For the frame tag use the smallest valid key frame: all
        // zeros. (key_frame_flag=0 → key frame; everything else 0.)
        let bytes = [
            0x00, 0x00, 0x00, // frame tag: all zeros = key, version 0, !show, fps=0
            0x9d, 0x01, 0x2a, // start code
            0xFF, 0xFF, // h_word LE = 0xFFFF
            0xFF, 0xBF, // v_word LE = 0xBFFF
        ];
        let hdr = Vp8FrameHeader::parse(&bytes).unwrap();
        assert!(hdr.key_frame);
        assert_eq!(hdr.width, Some(0x3FFF));
        assert_eq!(hdr.height, Some(0x3FFF));
        assert_eq!(hdr.horizontal_scale, Some(ScaleCode::Double));
        assert_eq!(hdr.vertical_scale, Some(ScaleCode::FiveThirds));
    }

    #[test]
    fn first_partition_size_uses_full_nineteen_bits() {
        // first_part_size = 0x7_FFFF (the 19-bit max). All other
        // bits of the frame tag zero; key_frame_flag=0.
        let fps: u32 = 0x7_FFFF;
        let tmp: u32 = fps << 5;
        let bytes = [
            (tmp & 0xff) as u8,
            ((tmp >> 8) & 0xff) as u8,
            ((tmp >> 16) & 0xff) as u8,
            0x9d,
            0x01,
            0x2a,
            0x00,
            0x00,
            0x00,
            0x00,
        ];
        let hdr = Vp8FrameHeader::parse(&bytes).unwrap();
        assert_eq!(hdr.first_partition_size, 0x7_FFFF);
        // sanity: the bits above bit 23 of tmp must not bleed in.
        assert!(hdr.first_partition_size < (1 << 19));
    }
}
