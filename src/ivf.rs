//! IVF — the de-facto-standard single-codec elementary-stream container
//! commonly used to ship VP8 test fixtures.
//!
//! The 0.1.13 `oxideav-vp8` crate exposed IVF reader / writer helpers
//! alongside the VP8 bitstream layer. This module restores that surface
//! at its historical path (`oxideav_vp8::ivf::*`) so downstream
//! consumers keep building.
//!
//! ## Format overview
//!
//! An IVF file is a 32-byte global header followed by a sequence of
//! per-frame records (`12`-byte frame header + payload). All
//! multi-byte fields are little-endian.
//!
//! Global header layout:
//!
//! | offset | size | field           |
//! |-------:|-----:|-----------------|
//! |      0 |    4 | signature `DKIF`|
//! |      4 |    2 | version         |
//! |      6 |    2 | header length   |
//! |      8 |    4 | fourcc (`VP80`) |
//! |     12 |    2 | width           |
//! |     14 |    2 | height          |
//! |     16 |    4 | framerate num   |
//! |     20 |    4 | framerate den   |
//! |     24 |    4 | frame count     |
//! |     28 |    4 | reserved        |
//!
//! Frame header layout:
//!
//! | offset | size | field         |
//! |-------:|-----:|---------------|
//! |      0 |    4 | payload size  |
//! |      4 |    8 | pts (u64 LE)  |
//!
//! ## Standalone build
//!
//! This module is reachable with `default-features = false`. It uses
//! only `std` primitives and the crate's own [`Vp8Error`] /
//! [`Result`](crate::error::Result) surface.

use crate::error::{Result, Vp8Error};

/// Magic four-byte signature at offset 0 of every IVF file.
pub const IVF_SIGNATURE: [u8; 4] = *b"DKIF";

/// Default fourcc IVF uses for VP8 elementary streams.
pub const IVF_VP8_FOURCC: [u8; 4] = *b"VP80";

/// Length of the IVF global header in bytes.
pub const IVF_HEADER_LEN: usize = 32;

/// Length of one IVF per-frame header in bytes (excluding payload).
pub const IVF_FRAME_HEADER_LEN: usize = 12;

/// Parsed IVF global header — the 32-byte file prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfHeader {
    /// Visible pixel width.
    pub width: u32,
    /// Visible pixel height.
    pub height: u32,
    /// Framerate numerator.
    pub framerate_num: u32,
    /// Framerate denominator.
    pub framerate_den: u32,
    /// Declared frame count (may be 0 if unknown at mux time).
    pub frame_count: u32,
    /// Per-codec fourcc (`*b"VP80"` for VP8).
    pub fourcc: [u8; 4],
}

impl IvfHeader {
    /// Build an [`IvfHeader`] for a VP8 stream with the supplied
    /// dimensions / framerate / frame count. The `fourcc` field is
    /// fixed to [`IVF_VP8_FOURCC`].
    pub fn vp8(width: u32, height: u32, framerate_num: u32, framerate_den: u32) -> Self {
        Self {
            width,
            height,
            framerate_num,
            framerate_den,
            frame_count: 0,
            fourcc: IVF_VP8_FOURCC,
        }
    }
}

/// Parse the IVF global header from the start of `buf`.
///
/// Returns [`Vp8Error::InvalidData`] when `buf` is shorter than
/// [`IVF_HEADER_LEN`] or the `DKIF` magic is missing.
pub fn parse_header(buf: &[u8]) -> Result<IvfHeader> {
    if buf.len() < IVF_HEADER_LEN {
        return Err(Vp8Error::InvalidData(format!(
            "ivf: header truncated ({} < {} bytes)",
            buf.len(),
            IVF_HEADER_LEN
        )));
    }
    let sig = [buf[0], buf[1], buf[2], buf[3]];
    if sig != IVF_SIGNATURE {
        return Err(Vp8Error::InvalidData(format!(
            "ivf: bad signature {:02x?} (expected {:02x?})",
            sig, IVF_SIGNATURE
        )));
    }
    let fourcc = [buf[8], buf[9], buf[10], buf[11]];
    let width = u16::from_le_bytes([buf[12], buf[13]]) as u32;
    let height = u16::from_le_bytes([buf[14], buf[15]]) as u32;
    let framerate_num = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let framerate_den = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let frame_count = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
    Ok(IvfHeader {
        width,
        height,
        framerate_num,
        framerate_den,
        frame_count,
        fourcc,
    })
}

/// Serialise an [`IvfHeader`] into 32 bytes ready to prepend to a
/// stream of [`write_frame`] records.
pub fn write_header(header: &IvfHeader) -> Vec<u8> {
    let mut out = Vec::with_capacity(IVF_HEADER_LEN);
    out.extend_from_slice(&IVF_SIGNATURE); // 0..4
    out.extend_from_slice(&0u16.to_le_bytes()); // 4..6 — version 0
    out.extend_from_slice(&(IVF_HEADER_LEN as u16).to_le_bytes()); // 6..8 — header length
    out.extend_from_slice(&header.fourcc); // 8..12
    out.extend_from_slice(&(header.width as u16).to_le_bytes()); // 12..14
    out.extend_from_slice(&(header.height as u16).to_le_bytes()); // 14..16
    out.extend_from_slice(&header.framerate_num.to_le_bytes()); // 16..20
    out.extend_from_slice(&header.framerate_den.to_le_bytes()); // 20..24
    out.extend_from_slice(&header.frame_count.to_le_bytes()); // 24..28
    out.extend_from_slice(&0u32.to_le_bytes()); // 28..32 — reserved
    debug_assert_eq!(out.len(), IVF_HEADER_LEN);
    out
}

/// Append one IVF frame record (12-byte header + payload) onto `buf`.
pub fn write_frame(buf: &mut Vec<u8>, pts: u64, frame: &[u8]) {
    let size = frame.len() as u32;
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&pts.to_le_bytes());
    buf.extend_from_slice(frame);
}

/// Parsed per-frame IVF record header (the 12 bytes before each
/// frame payload).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfFrameHeader {
    /// Payload byte count.
    pub size: u32,
    /// Presentation timestamp in timebase ticks.
    pub pts: u64,
}

/// Parse the per-frame IVF header from the start of `buf`. Returns
/// [`Vp8Error::InvalidData`] when `buf` is too short to hold the
/// 12-byte header.
pub fn parse_frame_header(buf: &[u8]) -> Result<IvfFrameHeader> {
    if buf.len() < IVF_FRAME_HEADER_LEN {
        return Err(Vp8Error::InvalidData(format!(
            "ivf: frame header truncated ({} < {} bytes)",
            buf.len(),
            IVF_FRAME_HEADER_LEN
        )));
    }
    let size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let pts = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
    ]);
    Ok(IvfFrameHeader { size, pts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_header() {
        let h = IvfHeader::vp8(640, 480, 30, 1);
        let bytes = write_header(&h);
        assert_eq!(bytes.len(), IVF_HEADER_LEN);
        let parsed = parse_header(&bytes).unwrap();
        assert_eq!(parsed.width, 640);
        assert_eq!(parsed.height, 480);
        assert_eq!(parsed.fourcc, IVF_VP8_FOURCC);
    }

    #[test]
    fn rejects_short_header() {
        assert!(parse_header(&[]).is_err());
        assert!(parse_header(&[0; 16]).is_err());
    }

    #[test]
    fn rejects_bad_signature() {
        let mut buf = vec![0u8; IVF_HEADER_LEN];
        buf[0] = b'X';
        assert!(parse_header(&buf).is_err());
    }

    #[test]
    fn write_frame_appends_header_plus_payload() {
        let mut buf = Vec::new();
        let payload = [0xab, 0xcd, 0xef];
        write_frame(&mut buf, 42, &payload);
        assert_eq!(buf.len(), IVF_FRAME_HEADER_LEN + payload.len());
        let fh = parse_frame_header(&buf).unwrap();
        assert_eq!(fh.size, 3);
        assert_eq!(fh.pts, 42);
        assert_eq!(&buf[IVF_FRAME_HEADER_LEN..], &payload);
    }
}
