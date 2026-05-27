//! VP8 frame-tag parsing — the 3-byte tag at the start of every VP8
//! frame, plus the 7-byte keyframe extension (start code + width /
//! height + scale fields).
//!
//! This module restores the 0.1.13 module layout where frame-tag types
//! lived on their own path under `oxideav_vp8::frame_tag::*`. The
//! implementation delegates to the existing
//! [`Vp8FrameHeader::parse`](crate::frame_header::Vp8FrameHeader::parse)
//! so we don't fork the parser — these types are thin wrappers over the
//! already-decoded fields.
//!
//! ## Standalone build
//!
//! This module is reachable with `default-features = false`. It depends
//! only on [`crate::frame_header`] and [`crate::error`], neither of
//! which pulls in `oxideav-core`.
//!
//! ## Surface
//!
//! * [`FrameType`] — `Key` or `Inter`.
//! * [`FrameTag`] — the bit-field shape of the 3-byte frame tag
//!   (`frame_type`, `version`, `show_frame`, `first_partition_size`).
//! * [`KeyframeHeader`] — start-code + width / height + scale codes
//!   (only present on key frames).
//! * [`ParsedHeader`] — `FrameTag` + optional `KeyframeHeader`.
//! * [`parse_header`] — top-level parser; one call returns either form.

use crate::error::{Result, Vp8Error};
use crate::frame_header::{ScaleCode, Vp8FrameHeader, KEY_FRAME_START_CODE};

/// VP8 frame-type discriminator (the §9.1 `frame_type` bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Key frame (the §9.1 `frame_type` bit is 0).
    Key,
    /// Interframe (the §9.1 `frame_type` bit is 1).
    Inter,
}

/// The 3-byte VP8 frame tag (RFC 6386 §9.1, §19.1).
///
/// Decomposes the 24-bit little-endian tag value into its four
/// bit-fields. The companion [`KeyframeHeader`] (when [`FrameType::Key`])
/// covers the 7-byte extension after the tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTag {
    /// `frame_type` bit (bit 0).
    pub frame_type: FrameType,
    /// `version` bits (bits 1..3) — `0..=7`.
    pub version: u8,
    /// `show_frame` bit (bit 4).
    pub show_frame: bool,
    /// `first_partition_size` (bits 5..23) — `0..=0x7_FFFF`.
    pub first_partition_size: u32,
}

/// The 7-byte keyframe extension that follows the frame tag on key
/// frames (RFC 6386 §9.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyframeHeader {
    /// The 3-byte start code that MUST be `[0x9d, 0x01, 0x2a]`.
    pub start_code: [u8; 3],
    /// Pixel width (14-bit field).
    pub width: u16,
    /// Pixel height (14-bit field).
    pub height: u16,
    /// Horizontal upscale code (2-bit field).
    pub horizontal_scale: ScaleCode,
    /// Vertical upscale code (2-bit field).
    pub vertical_scale: ScaleCode,
}

/// Result of [`parse_header`] — the frame tag plus, when the frame is a
/// key frame, the 7-byte extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedHeader {
    /// The decoded 3-byte frame tag.
    pub tag: FrameTag,
    /// The decoded keyframe extension, present iff `tag.frame_type ==
    /// FrameType::Key`.
    pub keyframe: Option<KeyframeHeader>,
    /// Number of bytes consumed from the input — 3 on interframes,
    /// 10 on key frames. The next byte is the first byte of the
    /// boolean-coded control partition.
    pub header_bytes_consumed: usize,
}

/// Parse the VP8 uncompressed frame header from the start of `bytes`.
///
/// Returns a [`ParsedHeader`] containing the [`FrameTag`] and, for key
/// frames, the [`KeyframeHeader`] extension. Required input length is
/// 3 bytes for interframes, 10 bytes for key frames.
///
/// On error this returns [`Vp8Error::InvalidData`] with a message that
/// names the specific failure (too short / wrong start code).
pub fn parse_header(bytes: &[u8]) -> Result<ParsedHeader> {
    let hdr = Vp8FrameHeader::parse(bytes).map_err(|e| Vp8Error::InvalidData(e.to_string()))?;

    let frame_type = if hdr.key_frame {
        FrameType::Key
    } else {
        FrameType::Inter
    };
    let tag = FrameTag {
        frame_type,
        version: hdr.version,
        show_frame: hdr.show_frame,
        first_partition_size: hdr.first_partition_size,
    };

    let keyframe = if hdr.key_frame {
        Some(KeyframeHeader {
            start_code: KEY_FRAME_START_CODE,
            width: hdr.width.unwrap_or(0),
            height: hdr.height.unwrap_or(0),
            horizontal_scale: hdr.horizontal_scale.unwrap_or(ScaleCode::None),
            vertical_scale: hdr.vertical_scale.unwrap_or(ScaleCode::None),
        })
    } else {
        None
    };

    Ok(ParsedHeader {
        tag,
        keyframe,
        header_bytes_consumed: hdr.header_bytes_consumed,
    })
}

/// Adapter over [`parse_header`] that returns only the keyframe-side
/// fields. Returns [`Vp8Error::InvalidData`] if the frame is not a key
/// frame or the header is malformed.
///
/// This is the 0.1.13 `frame_header::parse_keyframe_header` entry point.
pub fn parse_keyframe_header(bytes: &[u8]) -> Result<KeyframeHeader> {
    match parse_header(bytes)? {
        ParsedHeader {
            keyframe: Some(kf), ..
        } => Ok(kf),
        ParsedHeader { keyframe: None, .. } => Err(Vp8Error::InvalidData(
            "vp8 parse_keyframe_header: frame is an interframe, not a key frame".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_frame_bytes() -> [u8; 10] {
        // frame_type=0 (key), version=0, show_frame=1,
        // first_partition_size=0 → byte0 bit0 = 0, byte0 bit4 = 1.
        // tmp = 0x10 (show_frame bit set, everything else zero).
        let mut buf = [0u8; 10];
        buf[0] = 0x10;
        buf[3] = 0x9d;
        buf[4] = 0x01;
        buf[5] = 0x2a;
        // width = 16, height = 16, no scale.
        buf[6] = 0x10;
        buf[7] = 0x00;
        buf[8] = 0x10;
        buf[9] = 0x00;
        buf
    }

    #[test]
    fn parse_header_key_frame() {
        let bytes = key_frame_bytes();
        let parsed = parse_header(&bytes).unwrap();
        assert_eq!(parsed.tag.frame_type, FrameType::Key);
        assert_eq!(parsed.tag.version, 0);
        assert!(parsed.tag.show_frame);
        assert_eq!(parsed.header_bytes_consumed, 10);
        let kf = parsed.keyframe.unwrap();
        assert_eq!(kf.start_code, KEY_FRAME_START_CODE);
        assert_eq!(kf.width, 16);
        assert_eq!(kf.height, 16);
    }

    #[test]
    fn parse_header_interframe() {
        let bytes = [0x11, 0x00, 0x00];
        let parsed = parse_header(&bytes).unwrap();
        assert_eq!(parsed.tag.frame_type, FrameType::Inter);
        assert_eq!(parsed.header_bytes_consumed, 3);
        assert!(parsed.keyframe.is_none());
    }

    #[test]
    fn parse_keyframe_header_rejects_interframe() {
        let bytes = [0x11, 0x00, 0x00];
        assert!(parse_keyframe_header(&bytes).is_err());
    }

    #[test]
    fn parse_keyframe_header_accepts_key_frame() {
        let bytes = key_frame_bytes();
        let kf = parse_keyframe_header(&bytes).unwrap();
        assert_eq!(kf.width, 16);
    }

    #[test]
    fn parse_header_truncated() {
        assert!(parse_header(&[]).is_err());
    }
}
