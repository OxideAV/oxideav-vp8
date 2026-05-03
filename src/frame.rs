//! Standalone frame container returned by `oxideav-vp8`'s
//! framework-free decode API.
//!
//! `Vp8Frame` carries the cropped (no MB padding) YUV 4:2:0 planes a
//! decoded VP8 frame produces. Image-library consumers that just want
//! pixels can use [`crate::decode_vp8`] without dragging the
//! `oxideav-core` dependency tree in.
//!
//! When the `registry` feature is enabled, the gated
//! [`crate::registry`] module provides a conversion to
//! `oxideav_core::VideoFrame` so the trait-side surface (`Decoder`)
//! continues to work unchanged.

/// Decoded VP8 frame in YUV 4:2:0 layout with **tight** strides
/// (no MB padding). Width and height match the keyframe header.
///
/// Plane lengths:
/// * Y is `width * height` bytes, stride `width`.
/// * U and V are `((width + 1) / 2) * ((height + 1) / 2)` bytes each,
///   stride `(width + 1) / 2`.
///
/// `pts` is `None` for the standalone [`crate::decode_vp8`] entry
/// point — that function operates on a single isolated frame buffer
/// without packet timing information. Callers that drive a full
/// stream through the gated `Decoder` trait get PTS values from the
/// containing `Packet`.
#[derive(Clone, Debug)]
pub struct Vp8Frame {
    /// Frame width in pixels (matches the keyframe header).
    pub width: u32,
    /// Frame height in pixels (matches the keyframe header).
    pub height: u32,
    /// Optional presentation timestamp (in the surrounding
    /// container's time base, when known). `None` for the standalone
    /// decode path.
    pub pts: Option<i64>,
    /// Luma plane (`width * height` bytes, stride `width`).
    pub y: Vec<u8>,
    /// Chroma-blue plane (`cw * ch` bytes, stride `cw`).
    pub u: Vec<u8>,
    /// Chroma-red plane (`cw * ch` bytes, stride `cw`).
    pub v: Vec<u8>,
    /// Stride of the luma plane (== `width`).
    pub y_stride: u32,
    /// Stride of the chroma planes (== `(width + 1) / 2`).
    pub uv_stride: u32,
}

impl Vp8Frame {
    /// Width of the U/V planes in samples.
    #[inline]
    pub fn chroma_width(&self) -> u32 {
        (self.width + 1) / 2
    }

    /// Height of the U/V planes in samples.
    #[inline]
    pub fn chroma_height(&self) -> u32 {
        (self.height + 1) / 2
    }
}
