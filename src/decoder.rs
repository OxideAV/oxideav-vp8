//! VP8 per-frame decode driver (`decode_vp8`) and the
//! [`oxideav_core::Decoder`] integration.
//!
//! This module is the top of the keyframe decode chain — the entry point
//! that takes the raw bytes of one VP8 frame ("one packet = one frame")
//! and produces a fully-reconstructed, loop-filtered I420
//! [`Vp8DecodedFrame`] (and, when the `registry` feature is on, an
//! [`oxideav_core::VideoFrame`]).
//!
//! ## Frame layout per RFC 6386 §4 / §9
//!
//! Every VP8 frame ships as:
//!
//! 1. The §9.1 uncompressed header — 3 bytes for an interframe, 10 bytes
//!    for a key frame (3-byte frame tag + 3-byte start code + two
//!    16-bit little-endian width / height words). [`Vp8FrameHeader::parse`]
//!    handles the layout.
//! 2. The "first" (control) partition — `first_partition_size` bytes
//!    immediately following the uncompressed header. The first partition
//!    carries the §19.2 frame header and the §19.3 per-macroblock
//!    *prediction* records (segment id, skip flag, intra Y / sub-block /
//!    UV modes — §11) for every macroblock in raster order.
//! 3. (Only when `log2_nbr_of_dct_partitions > 0`)
//!    `(nbr_of_dct_partitions - 1) * 3` bytes of 3-byte little-endian
//!    DCT-partition sizes (RFC 6386 §9.5). Each entry is the byte length
//!    of the corresponding DCT partition that follows; the last partition's
//!    length is computed by subtraction from the frame total.
//! 4. `nbr_of_dct_partitions` concatenated DCT partitions carrying the
//!    §13 residual-coefficient records (`residual_data()` per §19.3) for
//!    every macroblock. Macroblock rows are striped round-robin across the
//!    partitions: row `r` is read from partition `r % nbr_of_dct_partitions`
//!    (RFC 6386 §20.4 — `for (row, partition = 0; row < mb_rows; row++) { ...;
//!    if (++partition == ctx->token_hdr.partitions) partition = 0; }`).
//!    Each partition uses its own [`BoolDecoder`] instance (RFC 6386
//!    §4 page 9 — "All partitions are decoded using separate instances of
//!    the boolean entropy decoder").
//!
//! ## Scope
//!
//! [`decode_vp8`] is the **stateless** single-frame entry point — it
//! decodes one key frame in isolation and returns
//! [`DecodeError::Unsupported`] on an interframe (since interframes
//! need a reference-frame buffer this function does not own). For
//! multi-frame streams with P-frames, use
//! [`crate::state::Vp8DecoderState::decode_frame`], which owns the
//! RFC 6386 §9 three-slot reference-frame buffer and handles both
//! key frames and interframes end-to-end.

use crate::bool_decoder::BoolDecoder;
use crate::coded_header::{CodedHeaderError, Vp8CodedHeader};
use crate::dct_tokens::{merge_default_token_probs, CoeffProbs, MbCoeffError, MbEntropyCtx};
use crate::dequant::{decode_and_dequantize_mb, MbDequantFactors};
use crate::frame::{decode_keyframe, FrameError, KeyframePlanes, MbCoeffs};
use crate::frame_header::{FrameHeaderError, Vp8FrameHeader};
use crate::loop_filter::{filter_frame, FrameFilterConfig, MAX_MB_SEGMENTS};
use crate::macroblock::{
    parse_key_frame_macroblock_modes, IntraYMode, MacroblockError, MacroblockModes,
};

/// Errors surfaced by [`decode_vp8`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The §9.1 uncompressed header could not be parsed.
    FrameHeader(FrameHeaderError),
    /// The §19.2 boolean-coded frame header could not be parsed.
    CodedHeader(CodedHeaderError),
    /// The §11 / §19.3 macroblock prediction layer could not be parsed.
    Macroblock(MacroblockError),
    /// The §13 residual-coefficient layer could not be parsed.
    MbCoeffs {
        /// `mb_row * mb_cols + mb_col` of the offending macroblock.
        index: usize,
        /// The underlying token-decode error.
        source: MbCoeffError,
    },
    /// The §14.2 per-MB reconstruction step rejected its inputs.
    Frame(FrameError),
    /// The frame is not a key frame — interframes are not yet supported.
    /// VP8 P-frames need §16 motion-vector decoding, §16.2 mv_ref tree
    /// resolution, and the reference-frame compositor (none of which are
    /// landed yet). Callers that receive this on a P-frame should treat
    /// the stream as un-decodable past the last key frame.
    Unsupported(&'static str),
    /// The first partition declared in the frame tag extends past the end
    /// of the supplied input bytes (truncated packet).
    TruncatedFirstPartition {
        /// Bytes available after the uncompressed header.
        available: usize,
        /// Bytes the frame tag's `first_partition_size` claimed.
        declared: usize,
    },
    /// One of the 3-byte DCT-partition size prefixes could not be read
    /// (truncated packet).
    TruncatedPartitionSizes,
    /// A declared DCT partition extends past the end of the supplied
    /// input bytes (truncated packet).
    TruncatedDctPartition {
        /// 0-based partition index (`0..nbr_of_dct_partitions`).
        index: usize,
        /// Bytes available for the partition.
        available: usize,
        /// Bytes the size prefix declared.
        declared: usize,
    },
    /// `width` or `height` from the §9.1 key-frame size words was zero —
    /// a valid VP8 key frame describes at least one pixel.
    ZeroDimension,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::FrameHeader(e) => write!(f, "vp8 decode: {e}"),
            DecodeError::CodedHeader(e) => write!(f, "vp8 decode: {e}"),
            DecodeError::Macroblock(e) => write!(f, "vp8 decode: {e}"),
            DecodeError::MbCoeffs { index, source } => {
                write!(f, "vp8 decode: macroblock {index}: {source}")
            }
            DecodeError::Frame(e) => write!(f, "vp8 decode: {e}"),
            DecodeError::Unsupported(msg) => write!(f, "vp8 decode: {msg}"),
            DecodeError::TruncatedFirstPartition {
                available,
                declared,
            } => write!(
                f,
                "vp8 decode: truncated first partition ({declared} declared, {available} available)"
            ),
            DecodeError::TruncatedPartitionSizes => {
                f.write_str("vp8 decode: truncated DCT-partition size prefixes")
            }
            DecodeError::TruncatedDctPartition {
                index,
                available,
                declared,
            } => write!(
                f,
                "vp8 decode: truncated DCT partition {index} ({declared} declared, {available} available)"
            ),
            DecodeError::ZeroDimension => {
                f.write_str("vp8 decode: zero width or height in key-frame header")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<FrameHeaderError> for DecodeError {
    fn from(value: FrameHeaderError) -> Self {
        DecodeError::FrameHeader(value)
    }
}
impl From<CodedHeaderError> for DecodeError {
    fn from(value: CodedHeaderError) -> Self {
        DecodeError::CodedHeader(value)
    }
}
impl From<MacroblockError> for DecodeError {
    fn from(value: MacroblockError) -> Self {
        DecodeError::Macroblock(value)
    }
}
impl From<FrameError> for DecodeError {
    fn from(value: FrameError) -> Self {
        DecodeError::Frame(value)
    }
}
impl From<crate::bool_decoder::BoolDecoderError> for DecodeError {
    fn from(value: crate::bool_decoder::BoolDecoderError) -> Self {
        // Surface a bool-decoder mishap during interframe macroblock-header
        // reads as a macroblock error (the §19.3 prefix is part of the
        // macroblock layer).
        DecodeError::Macroblock(MacroblockError::from(value))
    }
}
impl From<crate::near_mv::InterMbError> for DecodeError {
    fn from(value: crate::near_mv::InterMbError) -> Self {
        match value {
            crate::near_mv::InterMbError::BoolDecoder(inner) => {
                DecodeError::Macroblock(MacroblockError::from(inner))
            }
            crate::near_mv::InterMbError::SplitNotSupported { .. } => DecodeError::Unsupported(
                "vp8 interframe: SPLITMV reconstruction path tripped an internal mis-dispatch",
            ),
        }
    }
}

/// Fully-decoded, loop-filtered VP8 key-frame in I420 layout.
///
/// `y`/`u`/`v` are the visible-cropped planes — the §9.1 width / height
/// from the key-frame header, rounded for chroma sub-sampling
/// (`uv_width = (width + 1) / 2`, `uv_height = (height + 1) / 2`). Decode
/// pads internally to whole macroblocks; the §15 loop filter runs against
/// that padded buffer, then this output is cropped to the visible area.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp8DecodedFrame {
    /// Visible width in luma pixels (the §9.1 key-frame `width`).
    pub width: u32,
    /// Visible height in luma pixels (the §9.1 key-frame `height`).
    pub height: u32,
    /// Luma plane — `width * height` bytes, row-major (stride == width).
    pub y: Vec<u8>,
    /// U chroma plane — `((width+1)/2) * ((height+1)/2)` bytes.
    pub u: Vec<u8>,
    /// V chroma plane — same dimensions as `u`.
    pub v: Vec<u8>,
}

/// Decode one VP8 frame end-to-end and emit the reconstructed I420 picture.
///
/// `bytes` must hold exactly one VP8 elementary-stream frame (the bytes a
/// container demuxer hands the decoder for one sample / packet). The
/// function is stateless — no reference-frame carry-over, no probability
/// persistence — so it can only handle **key frames**. Interframes return
/// [`DecodeError::Unsupported`].
///
/// See the module docstring for the partition-layout walk this function
/// follows. Production-grade users should prefer [`Vp8Decoder`] (gated on
/// the `registry` feature), which wraps this entry point in the
/// [`oxideav_core::Decoder`] trait.
pub fn decode_vp8(bytes: &[u8]) -> Result<Vp8DecodedFrame, DecodeError> {
    let header = Vp8FrameHeader::parse(bytes)?;
    if !header.key_frame {
        return Err(DecodeError::Unsupported(
            "VP8 inter-frame (§16) decoding is not implemented yet — \
             only key frames are supported in this build",
        ));
    }

    // §9.1: width and height live on the key-frame header (Some) and are
    // both 14 bits; reject zero up-front so the macroblock math below sees
    // a non-degenerate frame.
    let width = header.width.unwrap_or(0) as u32;
    let height = header.height.unwrap_or(0) as u32;
    if width == 0 || height == 0 {
        return Err(DecodeError::ZeroDimension);
    }

    // §4 / §9.1: macroblock dimensions round up to whole 16-pixel blocks.
    let mb_cols = width.div_ceil(16) as usize;
    let mb_rows = height.div_ceil(16) as usize;

    // ---- Carve out the first (control) partition ---------------------
    let first_part_offset = header.header_bytes_consumed;
    let first_part_size = header.first_partition_size as usize;
    let available_after_hdr = bytes.len().saturating_sub(first_part_offset);
    if first_part_size > available_after_hdr {
        return Err(DecodeError::TruncatedFirstPartition {
            available: available_after_hdr,
            declared: first_part_size,
        });
    }
    let first_partition = &bytes[first_part_offset..first_part_offset + first_part_size];

    // ---- §19.2 boolean-coded frame header + §11 macroblock layer -----
    let (coded, mut dec) = Vp8CodedHeader::parse_with_decoder(first_partition, true)?;
    let modes = parse_key_frame_macroblock_modes(&mut dec, &coded, mb_rows, mb_cols)?;
    debug_assert_eq!(modes.len(), mb_rows * mb_cols);

    // ---- §9.5 DCT-partition table ------------------------------------
    // The DCT partitions begin immediately after the first partition,
    // preceded by (nbr_of_dct_partitions - 1) 3-byte LE size words.
    let dct_section_offset = first_part_offset + first_part_size;
    let dct_section = &bytes[dct_section_offset..];
    let num_partitions = coded.nbr_of_dct_partitions as usize;
    let partitions = carve_dct_partitions(dct_section, num_partitions)?;

    // ---- §14.1 per-segment dequant factors ---------------------------
    // Resolve the four segment slots up front. With segmentation off the
    // four entries collapse to the same factor set and the per-MB
    // `segment_id` is `None` (treated as segment 0).
    let dequant_factors = resolve_segment_dequant_factors(&coded);

    // ---- §13 token probability table ---------------------------------
    let coeff_probs: CoeffProbs = merge_default_token_probs(&coded.token_prob_updates);

    // ---- Per-row residual decode -------------------------------------
    let coeffs = decode_residuals(
        &coded,
        &modes,
        &partitions,
        &coeff_probs,
        &dequant_factors,
        mb_rows,
        mb_cols,
    )?;

    // ---- §12 / §14.2 reconstruction ----------------------------------
    let mut planes = decode_keyframe(mb_cols, mb_rows, &modes, &coeffs)?;

    // ---- §15.1 loop-filter post-pass ---------------------------------
    let lf_config = FrameFilterConfig::keyframe(&coded);
    if lf_config.loop_filter_level != 0 {
        filter_frame(&mut planes, &modes, &coeffs, &lf_config);
    }

    // ---- §9.1 crop to visible width / height -------------------------
    Ok(crop_to_visible(&planes, width, height))
}

/// Carve the DCT-partition section into a `Vec` of byte slices.
///
/// Layout per RFC 6386 §9.5: the first `(num_partitions - 1) * 3` bytes
/// are 3-byte little-endian size words, then the partitions follow
/// concatenated. The last partition's length is implied by what is left.
/// `num_partitions == 1` skips the size table entirely.
pub(crate) fn carve_dct_partitions(
    section: &[u8],
    num_partitions: usize,
) -> Result<Vec<&[u8]>, DecodeError> {
    debug_assert!((1..=8).contains(&num_partitions));
    let table_len = (num_partitions - 1) * 3;
    if section.len() < table_len {
        return Err(DecodeError::TruncatedPartitionSizes);
    }
    let (sizes_bytes, body) = section.split_at(table_len);

    let mut sizes: Vec<usize> = Vec::with_capacity(num_partitions);
    for i in 0..(num_partitions - 1) {
        let off = i * 3;
        let sz = (sizes_bytes[off] as usize)
            | ((sizes_bytes[off + 1] as usize) << 8)
            | ((sizes_bytes[off + 2] as usize) << 16);
        sizes.push(sz);
    }
    // Last partition's size: the remaining bytes.
    let consumed: usize = sizes.iter().sum();
    if consumed > body.len() {
        // One of the prefix sizes already overshot the available data.
        // Surface the offending one rather than the last (computed) slot.
        for (i, &sz) in sizes.iter().enumerate() {
            let off: usize = sizes.iter().take(i).sum();
            if off + sz > body.len() {
                return Err(DecodeError::TruncatedDctPartition {
                    index: i,
                    available: body.len().saturating_sub(off),
                    declared: sz,
                });
            }
        }
    }
    sizes.push(body.len() - consumed);

    let mut out: Vec<&[u8]> = Vec::with_capacity(num_partitions);
    let mut cursor = 0usize;
    for (i, &sz) in sizes.iter().enumerate() {
        let end = cursor + sz;
        if end > body.len() {
            return Err(DecodeError::TruncatedDctPartition {
                index: i,
                available: body.len().saturating_sub(cursor),
                declared: sz,
            });
        }
        out.push(&body[cursor..end]);
        cursor = end;
    }
    Ok(out)
}

/// Build the four per-segment [`MbDequantFactors`] for the frame, applying
/// the §10 `quantizer_update` segment override (delta or absolute) on top
/// of the §9.6 frame baseline.
///
/// When segmentation is disabled (the common case for the simplest key
/// frames) every slot resolves to the frame-level
/// [`MbDequantFactors::from_quant_indices`] result, so the per-MB lookup
/// always lands on the correct factors regardless of how the caller's
/// `segment_id` happens to be populated.
fn resolve_segment_dequant_factors(coded: &Vp8CodedHeader) -> [MbDequantFactors; MAX_MB_SEGMENTS] {
    let base = MbDequantFactors::from_quant_indices(&coded.quant_indices);
    let mut out = [base; MAX_MB_SEGMENTS];
    if coded.segmentation_enabled {
        if let Some(seg) = coded.update_segmentation {
            for (i, slot) in out.iter_mut().enumerate() {
                let qval = seg.quantizer_update[i].unwrap_or(0) as i32;
                *slot = MbDequantFactors::for_segment(
                    &coded.quant_indices,
                    qval,
                    seg.segment_feature_mode_absolute,
                );
            }
        }
    }
    out
}

/// Decode the §13 residual coefficients for every macroblock, threading the
/// above / left non-zero predictor contexts per the §13.3 rules and routing
/// each row's reads to the correct DCT partition per the §9.5 / §20.4
/// round-robin row striping.
fn decode_residuals(
    coded: &Vp8CodedHeader,
    modes: &[MacroblockModes],
    partitions: &[&[u8]],
    coeff_probs: &CoeffProbs,
    dequant_factors: &[MbDequantFactors; MAX_MB_SEGMENTS],
    mb_rows: usize,
    mb_cols: usize,
) -> Result<Vec<MbCoeffs>, DecodeError> {
    let num_partitions = partitions.len();
    debug_assert!(num_partitions >= 1);

    // §13: every macroblock is decoded exactly once, in raster order
    // (`mb_row` outer, `mb_col` inner), and its coefficients appended.
    // Reserve the frame's slots and `push` per MB rather than `resize`-ing
    // to `default()` first: each slot is overwritten before it is read, so
    // the bulk default-zeroing (800 bytes per `MbCoeffs`) was dead work.
    let mut coeffs: Vec<MbCoeffs> = Vec::with_capacity(mb_rows * mb_cols);

    // One above-context per macroblock column, lives for the whole frame.
    let mut above: Vec<MbEntropyCtx> = vec![MbEntropyCtx::default(); mb_cols];

    // One bool-decoder per DCT partition that is actually consumed, with
    // the cursor persisting across rows that share a partition (the §20.4
    // round-robin rule). Partitions that no row routes to — e.g. the
    // 1-byte padding partitions of a 16×16 / 4-partition frame, where only
    // partition 0 is ever read — are left `None` and never initialised.
    // Consumed partitions use the tolerant §20.2 `init_bool_decoder` form
    // ([`BoolDecoder::init_partition`]): a DCT partition that rounds down
    // to a 0- or 1-byte slice is spec-legal (`sz < 2` → zero-initialised
    // value register, empty input) and must decode, matching the
    // stateful keyframe path in `state::decode_intra_residuals`. The
    // round-284 `decode_stream_token_descent` fuzz target's
    // cross-entry-point differential caught this path still using the
    // strict control-partition `init` and rejecting such frames with
    // `InputTooShort`.
    let mut partition_decoders: Vec<Option<BoolDecoder<'_>>> =
        (0..num_partitions).map(|_| None).collect();
    for mb_row in 0..mb_rows {
        let part_idx = mb_row % num_partitions;
        if partition_decoders[part_idx].is_none() {
            partition_decoders[part_idx] = Some(BoolDecoder::init_partition(partitions[part_idx]));
        }
    }

    for mb_row in 0..mb_rows {
        // §13.3 page 65: the "left" non-zero predictor context resets at
        // the start of every macroblock row.
        let mut left = MbEntropyCtx::default();
        let part_idx = mb_row % num_partitions;
        let dec = partition_decoders[part_idx]
            .as_mut()
            .expect("partition decoder initialised above for every consumed partition");

        for (mb_col, above_ctx) in above.iter_mut().enumerate() {
            let raster = mb_row * mb_cols + mb_col;
            let mb = &modes[raster];
            let has_y2 = mb.y_mode != IntraYMode::B;
            // §10: a macroblock with no `segment_id` defaults to segment 0
            // (the §10 default before the map is ever updated). When
            // segmentation is off, all four segment factors are identical
            // anyway, so the unwrap_or here cannot mis-route.
            let seg = mb.segment_id.unwrap_or(0) as usize;
            let seg = seg.min(MAX_MB_SEGMENTS - 1);
            let factors = &dequant_factors[seg];

            let mb_coeffs = decode_and_dequantize_mb(
                dec,
                has_y2,
                mb.mb_skip_coeff,
                coeff_probs,
                factors,
                above_ctx,
                &mut left,
            )
            .map_err(|source| DecodeError::MbCoeffs {
                index: raster,
                source,
            })?;
            debug_assert_eq!(coeffs.len(), raster, "coeffs pushed in raster order");
            coeffs.push(mb_coeffs);
        }
    }

    // `coded` is informational at this layer (the per-MB walk above is
    // entirely self-contained), but keep the reference to make the
    // signature unambiguous about what header state drives the loop and to
    // future-proof for the §16 interframe extension (which will need
    // `coded.prob_intra` etc here).
    let _ = coded;

    Ok(coeffs)
}

/// Public-in-crate alias of [`crop_to_visible`] for the multi-frame driver.
pub(crate) fn crop_to_visible_public(
    planes: &KeyframePlanes,
    width: u32,
    height: u32,
) -> Vp8DecodedFrame {
    crop_to_visible(planes, width, height)
}

/// Crop the §14.2 reconstructed plane buffers (sized to whole macroblocks)
/// to the §9.1 visible width / height and return a packed I420 frame.
fn crop_to_visible(planes: &KeyframePlanes, width: u32, height: u32) -> Vp8DecodedFrame {
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);

    // When the visible width equals the macroblock-padded stride there is no
    // per-row gap to skip, so the whole leading `w * h` region of the source
    // plane is already the packed output verbatim — one contiguous copy
    // instead of `h` row-sliced `extend_from_slice` calls. This is the common
    // case for macroblock-aligned dimensions (e.g. 128×128, 16-multiples). The
    // strided path below produces the identical packed bytes when a gap exists.
    let y = if w == planes.y_stride {
        planes.y[..w * h].to_vec()
    } else {
        let mut y = Vec::with_capacity(w * h);
        for r in 0..h {
            let src = r * planes.y_stride;
            y.extend_from_slice(&planes.y[src..src + w]);
        }
        y
    };

    let (u, v) = if uvw == planes.uv_stride {
        (
            planes.u[..uvw * uvh].to_vec(),
            planes.v[..uvw * uvh].to_vec(),
        )
    } else {
        let mut u = Vec::with_capacity(uvw * uvh);
        let mut v = Vec::with_capacity(uvw * uvh);
        for r in 0..uvh {
            let src = r * planes.uv_stride;
            u.extend_from_slice(&planes.u[src..src + uvw]);
            v.extend_from_slice(&planes.v[src..src + uvw]);
        }
        (u, v)
    };

    Vp8DecodedFrame {
        width,
        height,
        y,
        u,
        v,
    }
}

// ─────────────────── oxideav_core::Decoder integration ───────────────────

#[cfg(feature = "registry")]
mod registry {
    use std::collections::VecDeque;

    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{
        CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
        Error, Frame, Packet, PixelFormat, Result, RuntimeContext, VideoFrame,
    };

    use super::{DecodeError, Vp8DecodedFrame};

    /// Canonical codec id under which this decoder registers. Matches the
    /// long-standing `oxideav-vp9` / `oxideav-av1` naming convention.
    pub const VP8_CODEC_ID: &str = "vp8";

    impl From<DecodeError> for Error {
        fn from(value: DecodeError) -> Self {
            match value {
                DecodeError::Unsupported(msg) => Error::unsupported(msg),
                DecodeError::FrameHeader(_)
                | DecodeError::CodedHeader(_)
                | DecodeError::Macroblock(_)
                | DecodeError::MbCoeffs { .. }
                | DecodeError::Frame(_)
                | DecodeError::TruncatedFirstPartition { .. }
                | DecodeError::TruncatedPartitionSizes
                | DecodeError::TruncatedDctPartition { .. }
                | DecodeError::ZeroDimension => Error::invalid(value.to_string()),
            }
        }
    }

    impl From<Vp8DecodedFrame> for VideoFrame {
        fn from(f: Vp8DecodedFrame) -> Self {
            let uv_stride = (f.width as usize).div_ceil(2);
            VideoFrame {
                pts: None,
                planes: vec![
                    VideoPlane {
                        stride: f.width as usize,
                        data: f.y,
                    },
                    VideoPlane {
                        stride: uv_stride,
                        data: f.u,
                    },
                    VideoPlane {
                        stride: uv_stride,
                        data: f.v,
                    },
                ],
            }
        }
    }

    /// `oxideav_core::Decoder` impl driving [`Vp8DecoderState`].
    ///
    /// Each `send_packet` queues one compressed packet; the next
    /// `receive_frame` consumes it and feeds it to
    /// [`Vp8DecoderState::decode_frame`], which threads reference frames
    /// and entropy carry-state across calls. Both key frames and
    /// interframes decode end-to-end; the only `DecodeError::Unsupported`
    /// returned now is "interframe arrived before any key frame" (a
    /// stream-mis-feed error, not a missing feature).
    ///
    /// [`Vp8DecoderState::decode_frame`]: crate::state::Vp8DecoderState::decode_frame
    pub struct Vp8Decoder {
        codec_id: CodecId,
        pending: VecDeque<Packet>,
        eof: bool,
        state: crate::state::Vp8DecoderState,
    }

    impl Vp8Decoder {
        /// Build a fresh decoder bound to the supplied codec id. The id is
        /// reported verbatim by [`Decoder::codec_id`]. The internal
        /// [`Vp8DecoderState`] starts empty — the first packet fed to
        /// `send_packet` / `receive_frame` must be a key frame.
        ///
        /// [`Vp8DecoderState`]: crate::state::Vp8DecoderState
        pub fn new(codec_id: CodecId) -> Self {
            Self {
                codec_id,
                pending: VecDeque::new(),
                eof: false,
                state: crate::state::Vp8DecoderState::new(),
            }
        }
    }

    impl std::fmt::Debug for Vp8Decoder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Vp8Decoder")
                .field("codec_id", &self.codec_id)
                .field("pending", &self.pending.len())
                .field("eof", &self.eof)
                .finish()
        }
    }

    impl Decoder for Vp8Decoder {
        fn codec_id(&self) -> &CodecId {
            &self.codec_id
        }

        fn send_packet(&mut self, packet: &Packet) -> Result<()> {
            self.pending.push_back(packet.clone());
            Ok(())
        }

        fn receive_frame(&mut self) -> Result<Frame> {
            let Some(pkt) = self.pending.pop_front() else {
                return if self.eof {
                    Err(Error::Eof)
                } else {
                    Err(Error::NeedMore)
                };
            };
            let decoded = self.state.decode_frame(&pkt.data)?;
            let mut vf: VideoFrame = decoded.into();
            vf.pts = pkt.pts;
            Ok(Frame::Video(vf))
        }

        fn flush(&mut self) -> Result<()> {
            self.eof = true;
            Ok(())
        }

        fn reset(&mut self) -> Result<()> {
            self.pending.clear();
            self.eof = false;
            self.state = crate::state::Vp8DecoderState::new();
            Ok(())
        }
    }

    /// `make_decoder` factory plugged into the registry. Required by the
    /// `DecoderFactory` function-pointer type.
    pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(Vp8Decoder::new(params.codec_id.clone())))
    }

    /// Register the VP8 codec into the supplied [`CodecRegistry`] under the
    /// canonical [`VP8_CODEC_ID`] codec id, with the container tags VP8
    /// commonly carries (Matroska / WebM `V_VP8`, MP4 / ISOBMFF `vp08`,
    /// IVF / RIFF `VP80` FourCC).
    pub fn register_codecs(reg: &mut CodecRegistry) {
        let caps = CodecCapabilities::video(VP8_CODEC_ID)
            .with_lossy(true)
            .with_max_size(16384, 16384);
        reg.register(
            CodecInfo::new(CodecId::new(VP8_CODEC_ID))
                .capabilities(caps)
                .decoder(make_decoder)
                .tags([
                    CodecTag::fourcc(b"VP80"),
                    CodecTag::fourcc(b"vp08"),
                    CodecTag::matroska("V_VP8"),
                ]),
        );
    }

    /// Unified entry point — installs the VP8 codec into a
    /// [`RuntimeContext`]. Called by `oxideav-meta` via the
    /// [`oxideav_core::register!`] hook in `lib.rs`.
    pub fn register(ctx: &mut RuntimeContext) {
        register_codecs(&mut ctx.codecs);
    }

    /// Helper that the public reflection / fallback path inspects: returns
    /// the pixel format the decoder produces (always I420 in this build).
    pub fn output_pixel_format() -> PixelFormat {
        PixelFormat::Yuv420P
    }
}

#[cfg(feature = "registry")]
pub use registry::{
    make_decoder, output_pixel_format, register, register_codecs, Vp8Decoder, VP8_CODEC_ID,
};

// ───────────────────────────── tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Strip the 32-byte DKIF header and one 12-byte IVF frame header off
    /// the front of an IVF file and return the contained VP8 elementary
    /// frame bytes. This is **not** a container parser — it knows just
    /// enough about the IVF wrapper to lift a single VP8 frame out of a
    /// fixture so the codec test stays self-contained per the workspace
    /// "no cross-crate dev-deps" rule. The IVF wrapper is a 32-byte DKIF
    /// header (`'DKIF'` magic + version + header_len + FourCC + width +
    /// height + fps + scale + frame_count + reserved) followed by one or
    /// more 12-byte frame headers (4-byte LE frame size + 8-byte LE pts)
    /// each preceding the raw frame bytes — read directly off the fixture
    /// `input.ivf` byte layout, no external parser consulted.
    fn strip_single_ivf_frame(ivf: &[u8]) -> &[u8] {
        assert!(ivf.len() >= 32 + 12, "IVF too short for a single frame");
        assert_eq!(&ivf[0..4], b"DKIF", "expected DKIF magic");
        let frame_hdr_off = 32;
        let frame_size = u32::from_le_bytes([
            ivf[frame_hdr_off],
            ivf[frame_hdr_off + 1],
            ivf[frame_hdr_off + 2],
            ivf[frame_hdr_off + 3],
        ]) as usize;
        let payload_off = frame_hdr_off + 12;
        assert!(
            ivf.len() >= payload_off + frame_size,
            "IVF frame body truncated"
        );
        &ivf[payload_off..payload_off + frame_size]
    }

    /// Build a minimal interframe (3-byte frame tag with bit 0 = 1).
    fn interframe_bytes() -> Vec<u8> {
        // tmp = 1: key_frame bit set → interframe; version=0; show=0;
        // first_partition_size=0.
        vec![0x01, 0x00, 0x00]
    }

    #[test]
    fn non_keyframe_rejected_as_unsupported() {
        let res = decode_vp8(&interframe_bytes());
        assert!(matches!(res, Err(DecodeError::Unsupported(_))));
    }

    #[test]
    fn too_short_frame_header_propagates_error() {
        let res = decode_vp8(&[]);
        assert!(matches!(
            res,
            Err(DecodeError::FrameHeader(
                crate::frame_header::FrameHeaderError::InputTooShort
            ))
        ));
    }

    #[test]
    fn truncated_first_partition_surfaces_clean_error() {
        // Build a key-frame header whose first_partition_size is 1000 but
        // append no actual partition bytes. The function should refuse on
        // the slice carve, not panic.
        //
        // first_partition_size = 1000 → tmp = 1000 << 5 = 0x7d00.
        let fps: u32 = 1000;
        let tmp: u32 = fps << 5;
        let mut buf = vec![
            (tmp & 0xff) as u8,
            ((tmp >> 8) & 0xff) as u8,
            ((tmp >> 16) & 0xff) as u8,
        ];
        // Start code + width/height (16x16, scale 0).
        buf.extend_from_slice(&[0x9d, 0x01, 0x2a, 0x10, 0x00, 0x10, 0x00]);
        assert_eq!(buf.len(), 10);
        let res = decode_vp8(&buf);
        assert!(matches!(
            res,
            Err(DecodeError::TruncatedFirstPartition {
                available: 0,
                declared: 1000,
            })
        ));
    }

    /// `crop_to_visible`'s contiguous fast path (taken when the visible
    /// width equals the macroblock-padded stride) must produce byte-for-byte
    /// the same packed I420 output as the strided per-row copy. Sweeps an
    /// aligned case (no per-row gap → contiguous path) and a cropped case
    /// (visible < stride → strided path) against a reference that always
    /// copies row-by-row, on a deterministically textured plane set.
    #[test]
    fn crop_to_visible_contiguous_matches_strided() {
        // Reference: the unconditional strided per-row crop.
        fn crop_strided(planes: &KeyframePlanes, width: u32, height: u32) -> Vp8DecodedFrame {
            let w = width as usize;
            let h = height as usize;
            let uvw = w.div_ceil(2);
            let uvh = h.div_ceil(2);
            let mut y = Vec::with_capacity(w * h);
            for r in 0..h {
                let src = r * planes.y_stride;
                y.extend_from_slice(&planes.y[src..src + w]);
            }
            let mut u = Vec::with_capacity(uvw * uvh);
            let mut v = Vec::with_capacity(uvw * uvh);
            for r in 0..uvh {
                let src = r * planes.uv_stride;
                u.extend_from_slice(&planes.u[src..src + uvw]);
                v.extend_from_slice(&planes.v[src..src + uvw]);
            }
            Vp8DecodedFrame {
                width,
                height,
                y,
                u,
                v,
            }
        }

        // Build a 2×2-macroblock padded plane set (32×32 luma, 16×16 chroma)
        // filled with a position-dependent LCG texture so a stride/offset bug
        // produces visibly wrong bytes.
        let mb_cols = 2usize;
        let mb_rows = 2usize;
        let y_stride = mb_cols * 16; // 32
        let uv_stride = mb_cols * 8; // 16
        let y_rows = mb_rows * 16; // 32
        let uv_rows = mb_rows * 8; // 16
        let mut seed: u32 = 0x1234_5678;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        };
        let y: Vec<u8> = (0..y_stride * y_rows).map(|_| next()).collect();
        let u: Vec<u8> = (0..uv_stride * uv_rows).map(|_| next()).collect();
        let v: Vec<u8> = (0..uv_stride * uv_rows).map(|_| next()).collect();
        let planes = KeyframePlanes {
            y,
            u,
            v,
            y_stride,
            uv_stride,
            mb_cols,
            mb_rows,
        };

        // Aligned case: visible == padded → contiguous fast path.
        let got = crop_to_visible(&planes, 32, 32);
        let want = crop_strided(&planes, 32, 32);
        assert_eq!(got.y, want.y, "aligned luma");
        assert_eq!(got.u, want.u, "aligned U");
        assert_eq!(got.v, want.v, "aligned V");

        // Cropped case: visible < padded on both axes → strided path. Chroma
        // visible width 17.div_ceil(2)=9 < uv_stride 16 keeps the chroma
        // strided path active too.
        let got = crop_to_visible(&planes, 17, 18);
        let want = crop_strided(&planes, 17, 18);
        assert_eq!(got.y, want.y, "cropped luma");
        assert_eq!(got.u, want.u, "cropped U");
        assert_eq!(got.v, want.v, "cropped V");

        // Mixed case: luma aligned (width==stride) but chroma not. width 32 →
        // uvw 16 == uv_stride → both contiguous; use width 32 height 30 so the
        // luma row count differs but width stays aligned (luma contiguous,
        // height truncation handled by the `w*h` prefix length).
        let got = crop_to_visible(&planes, 32, 30);
        let want = crop_strided(&planes, 32, 30);
        assert_eq!(got.y, want.y, "height-truncated luma");
        assert_eq!(got.u, want.u, "height-truncated U");
        assert_eq!(got.v, want.v, "height-truncated V");
    }

    #[test]
    fn zero_dimension_rejected() {
        // Valid key-frame header but with width = height = 0.
        let buf = [0x00, 0x00, 0x00, 0x9d, 0x01, 0x2a, 0x00, 0x00, 0x00, 0x00];
        let res = decode_vp8(&buf);
        assert!(matches!(res, Err(DecodeError::ZeroDimension)));
    }

    /// The tiny-i-only-16x16 fixture is a single-keyframe VP8 elementary
    /// stream packaged in IVF; decoding the contained 57-byte frame should
    /// produce a 16x16 I420 picture.
    ///
    /// `expected.yuv` from the fixture set is the reference decoder's
    /// black-box output for the same input (the `ffmpeg` invocation that
    /// produced it is recorded in the fixture's `notes.md`). The decode is
    /// **bit-exact** against that reference — the whole point of VP8's
    /// "exact pixel values are part of the specification" guarantee (RFC
    /// 6386 §2). We assert the full I420 byte stream (Y then U then V)
    /// equals `expected.yuv` so a regression anywhere in the
    /// bitstream → dequant → reconstruct → loop-filter chain is caught.
    #[test]
    fn tiny_i_only_16x16_fixture_decodes_bit_exact() {
        const IVF: &[u8] = include_bytes!("../tests/fixtures/tiny-i-only-16x16/input.ivf");
        const EXPECTED: &[u8] = include_bytes!("../tests/fixtures/tiny-i-only-16x16/expected.yuv");
        // Sanity: trace.txt lists the contained frame as 57 bytes.
        assert_eq!(
            strip_single_ivf_frame(IVF).len(),
            57,
            "fixture VP8 frame size changed"
        );
        // Concatenate the planes in I420 order and require a byte-exact
        // match against the reference (384 bytes = 256 Y + 64 U + 64 V).
        assert_fixture_bit_exact(IVF, EXPECTED, 16, 16);
    }

    /// A second, multi-macroblock bit-exact fixture: a 64×64 intra-only
    /// key frame (4×4 = 16 macroblocks, one DCT partition). Exercises the
    /// §12 cross-macroblock neighbour propagation (above-row / left-column
    /// strips), the §13.3 above/left non-zero predictor threading across a
    /// full raster of macroblocks, and the §15.1 loop-filter geometry over
    /// internal macroblock edges — none of which the single-MB 16×16
    /// fixture touches. Bit-exact against the reference YUV.
    #[test]
    fn i_only_64x64_fixture_decodes_bit_exact() {
        const IVF: &[u8] = include_bytes!("../tests/fixtures/i-only-64x64/input.ivf");
        const EXPECTED: &[u8] = include_bytes!("../tests/fixtures/i-only-64x64/expected.yuv");
        // 6144 bytes = 64*64 Y + 32*32 U + 32*32 V.
        assert_fixture_bit_exact(IVF, EXPECTED, 64, 64);
    }

    /// Decode a fixture IVF and assert its single key frame is bit-exact
    /// against the reference YUV. `w` / `h` are the visible dimensions.
    fn assert_fixture_bit_exact(ivf: &[u8], expected: &[u8], w: u32, h: u32) {
        let frame_bytes = strip_single_ivf_frame(ivf);
        let decoded = decode_vp8(frame_bytes).expect("fixture should decode");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        let mut got = Vec::with_capacity(expected.len());
        got.extend_from_slice(&decoded.y);
        got.extend_from_slice(&decoded.u);
        got.extend_from_slice(&decoded.v);
        assert_eq!(got.len(), expected.len(), "plane byte count mismatch");
        assert_eq!(got, expected, "decoded I420 must be bit-exact vs reference");
    }

    /// A 16×16 key frame coded with **four DCT partitions** (the
    /// `log2_nbr_of_dct_partitions = 2` case, §9.5). Exercises the 3-byte
    /// little-endian partition-size table parse and the round-robin row
    /// striping (one MB row → partition 0). Bit-exact against the
    /// reference.
    #[test]
    fn partition_padding_16x16_4parts_decodes_bit_exact() {
        const IVF: &[u8] =
            include_bytes!("../tests/fixtures/partition-padding-16x16-4parts/input.ivf");
        const EXPECTED: &[u8] =
            include_bytes!("../tests/fixtures/partition-padding-16x16-4parts/expected.yuv");
        assert_fixture_bit_exact(IVF, EXPECTED, 16, 16);
    }

    /// A 128×128 key frame (8×8 = 64 macroblocks) coded with four DCT
    /// partitions. With eight macroblock rows striped round-robin across
    /// four partitions (rows 0/4 → part 0, 1/5 → part 1, …) this is the
    /// fixture that actually exercises the §20.4 multi-row partition
    /// routing (and the per-partition bool-decoder cursor persisting across
    /// the rows that share a partition). Bit-exact against the reference.
    #[test]
    fn segment_4_partitions_128x128_decodes_bit_exact() {
        const IVF: &[u8] = include_bytes!("../tests/fixtures/segment-4-partitions/input.ivf");
        const EXPECTED: &[u8] =
            include_bytes!("../tests/fixtures/segment-4-partitions/expected.yuv");
        assert_fixture_bit_exact(IVF, EXPECTED, 128, 128);
    }

    /// A batch of single-key-frame fixtures, each decoded bit-exact
    /// against its reference YUV. These cover decode behaviours the
    /// 16×16 / 64×64 / partition fixtures above don't reach:
    ///
    /// * `q-high` — `loop_filter_level = 38` (heavy normal deblocking)
    ///   over a 128×128 frame, the strongest test of the §15.3 `mb_filter`
    ///   / `subblock_filter` parameter ladder.
    /// * `i-only-loopfilter-high` — `loop_filter_level = 33` over 64×64.
    /// * `i-only-loopfilter-off` — `loop_filter_level = 0` (the §15 page-84
    ///   whole-frame filter skip).
    /// * `vp8-with-loopfilter-mode-simple` — the §15.2 simple-filter path
    ///   (`filter_type = 1`, luma-only).
    /// * `q-low` — `loop_filter_level = 0`, 32×32 (a non-multiple-of-16
    ///   path is the next size up; 32 is a clean 2×2 MB grid).
    /// * `gradient-and-noise-128x128` — high-entropy content that exercises
    ///   the full §13 token alphabet (cat ranges, dense blocks).
    #[test]
    fn additional_single_keyframe_fixtures_decode_bit_exact() {
        // (fixture name, input IVF bytes, expected YUV, visible w, visible h)
        type FixtureCase = (&'static str, &'static [u8], &'static [u8], u32, u32);
        let cases: &[FixtureCase] = &[
            (
                "q-high",
                include_bytes!("../tests/fixtures/q-high/input.ivf"),
                include_bytes!("../tests/fixtures/q-high/expected.yuv"),
                128,
                128,
            ),
            (
                "q-low",
                include_bytes!("../tests/fixtures/q-low/input.ivf"),
                include_bytes!("../tests/fixtures/q-low/expected.yuv"),
                32,
                32,
            ),
            (
                "i-only-loopfilter-off",
                include_bytes!("../tests/fixtures/i-only-loopfilter-off/input.ivf"),
                include_bytes!("../tests/fixtures/i-only-loopfilter-off/expected.yuv"),
                64,
                64,
            ),
            (
                "i-only-loopfilter-high",
                include_bytes!("../tests/fixtures/i-only-loopfilter-high/input.ivf"),
                include_bytes!("../tests/fixtures/i-only-loopfilter-high/expected.yuv"),
                64,
                64,
            ),
            (
                "gradient-and-noise-128x128",
                include_bytes!("../tests/fixtures/gradient-and-noise-128x128/input.ivf"),
                include_bytes!("../tests/fixtures/gradient-and-noise-128x128/expected.yuv"),
                128,
                128,
            ),
            (
                "vp8-with-loopfilter-mode-simple",
                include_bytes!("../tests/fixtures/vp8-with-loopfilter-mode-simple/input.ivf"),
                include_bytes!("../tests/fixtures/vp8-with-loopfilter-mode-simple/expected.yuv"),
                64,
                64,
            ),
        ];
        for (name, ivf, expected, w, h) in cases {
            let frame_bytes = strip_single_ivf_frame(ivf);
            let decoded = decode_vp8(frame_bytes)
                .unwrap_or_else(|e| panic!("fixture {name} should decode: {e}"));
            assert_eq!(decoded.width, *w, "{name} width");
            assert_eq!(decoded.height, *h, "{name} height");
            let mut got = Vec::with_capacity(expected.len());
            got.extend_from_slice(&decoded.y);
            got.extend_from_slice(&decoded.u);
            got.extend_from_slice(&decoded.v);
            assert_eq!(
                &got, expected,
                "fixture {name} must be bit-exact against the reference"
            );
        }
    }

    #[test]
    fn carve_dct_partitions_single_partition_no_size_table() {
        // num_partitions = 1: the whole section is the one partition; no
        // 3-byte size table is read.
        let section = [0xAAu8; 7];
        let parts = carve_dct_partitions(&section, 1).expect("single-partition");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], &section[..]);
    }

    #[test]
    fn carve_dct_partitions_two_partitions_uses_three_byte_size_prefix() {
        // num_partitions = 2: 3 bytes of size prefix for partition 0,
        // partition 1 takes the rest.
        //
        // Build: size[0] = 4 (LE 04 00 00) | partition0 (4 bytes) |
        //        partition1 (3 bytes).
        let mut buf = vec![0x04, 0x00, 0x00];
        buf.extend_from_slice(&[0xA1, 0xA2, 0xA3, 0xA4]);
        buf.extend_from_slice(&[0xB1, 0xB2, 0xB3]);
        let parts = carve_dct_partitions(&buf, 2).expect("two-partition");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], &[0xA1, 0xA2, 0xA3, 0xA4][..]);
        assert_eq!(parts[1], &[0xB1, 0xB2, 0xB3][..]);
    }

    #[test]
    fn carve_dct_partitions_truncated_size_table_rejected() {
        // num_partitions = 2 needs 3 size bytes; supply only 2.
        let buf = [0x04u8, 0x00];
        let res = carve_dct_partitions(&buf, 2);
        assert!(matches!(res, Err(DecodeError::TruncatedPartitionSizes)));
    }

    #[test]
    fn carve_dct_partitions_truncated_body_rejected() {
        // size[0] = 10, but only 6 body bytes available.
        let mut buf = vec![0x0Au8, 0x00, 0x00];
        buf.extend_from_slice(&[0xCC; 6]);
        let res = carve_dct_partitions(&buf, 2);
        assert!(matches!(
            res,
            Err(DecodeError::TruncatedDctPartition {
                index: 0,
                available: 6,
                declared: 10,
            })
        ));
    }

    #[cfg(feature = "registry")]
    mod registry_tests {
        use super::*;
        use oxideav_core::{CodecId, Decoder as _, Error, Frame, Packet, TimeBase};

        #[test]
        fn vp8_decoder_returns_unsupported_for_interframe_before_keyframe() {
            // An interframe fed in with no prior key frame can't decode
            // (the §16 inter-prediction layer has no reference to predict
            // from); surfaces `Error::Unsupported` via the
            // `Vp8DecoderState` driver's empty-reference-slot check.
            let mut dec = Vp8Decoder::new(CodecId::new(VP8_CODEC_ID));
            let pkt = Packet::new(0, TimeBase::new(1, 1000), interframe_bytes());
            dec.send_packet(&pkt).expect("queue packet");
            match dec.receive_frame() {
                Err(Error::Unsupported(_)) => {}
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }

        #[test]
        fn vp8_decoder_needs_packet_before_frame() {
            let mut dec = Vp8Decoder::new(CodecId::new(VP8_CODEC_ID));
            match dec.receive_frame() {
                Err(Error::NeedMore) => {}
                other => panic!("expected NeedMore, got {other:?}"),
            }
        }

        #[test]
        fn vp8_decoder_decodes_tiny_keyframe_through_trait_api() {
            const IVF: &[u8] = include_bytes!("../tests/fixtures/tiny-i-only-16x16/input.ivf");
            let frame_bytes = strip_single_ivf_frame(IVF);
            let mut dec = Vp8Decoder::new(CodecId::new(VP8_CODEC_ID));
            let mut pkt = Packet::new(0, TimeBase::new(1, 1000), frame_bytes.to_vec());
            pkt.pts = Some(42);
            pkt.flags.keyframe = true;
            dec.send_packet(&pkt).expect("send packet");
            let frame = dec.receive_frame().expect("decode");
            let Frame::Video(v) = frame else {
                panic!("expected Video frame, got {frame:?}");
            };
            assert_eq!(v.planes.len(), 3, "I420 has three planes");
            assert_eq!(v.planes[0].stride, 16);
            assert_eq!(v.planes[1].stride, 8);
            assert_eq!(v.planes[2].stride, 8);
            assert_eq!(v.planes[0].data.len(), 16 * 16);
            assert_eq!(v.planes[1].data.len(), 8 * 8);
            assert_eq!(v.planes[2].data.len(), 8 * 8);
            assert_eq!(v.pts, Some(42));
            // After draining, the next call needs more input.
            assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
        }

        /// End-to-end multi-frame decode through the `Decoder` trait
        /// surface — feed the two-frame `i-frame-then-p-frame-64x64`
        /// fixture as two consecutive `Packet`s and assert both frames
        /// produce a `Frame::Video` (the bit-exact pixel-match itself is
        /// covered by the state-module tests).
        #[test]
        fn vp8_decoder_decodes_multi_frame_through_trait_api() {
            const IVF: &[u8] =
                include_bytes!("../tests/fixtures/i-frame-then-p-frame-64x64/input.ivf");
            // Walk the IVF (32-byte DKIF + per-frame 12-byte headers).
            let mut frames: Vec<Vec<u8>> = Vec::new();
            let mut cur = 32usize;
            while cur + 12 <= IVF.len() {
                let size = u32::from_le_bytes([IVF[cur], IVF[cur + 1], IVF[cur + 2], IVF[cur + 3]])
                    as usize;
                let body = cur + 12;
                frames.push(IVF[body..body + size].to_vec());
                cur = body + size;
            }
            assert_eq!(frames.len(), 2, "fixture has 2 frames");
            let mut dec = Vp8Decoder::new(CodecId::new(VP8_CODEC_ID));
            for (i, bytes) in frames.iter().enumerate() {
                let pkt = Packet::new(i as u32, TimeBase::new(1, 1000), bytes.clone());
                dec.send_packet(&pkt).expect("send packet");
                let frame = dec.receive_frame().expect("decode through trait API");
                let Frame::Video(v) = frame else {
                    panic!("expected Video frame, got {frame:?}");
                };
                assert_eq!(v.planes.len(), 3, "I420 has 3 planes");
                assert_eq!(v.planes[0].stride, 64);
                assert_eq!(v.planes[1].stride, 32);
                assert_eq!(v.planes[2].stride, 32);
            }
        }

        #[test]
        fn registers_into_codec_registry() {
            use oxideav_core::CodecRegistry;
            let mut reg = CodecRegistry::new();
            register_codecs(&mut reg);
            // The id we registered under must round-trip through
            // `decoder_ids()`.
            let id = CodecId::new(VP8_CODEC_ID);
            let found = reg.decoder_ids().any(|i| i == &id);
            assert!(found, "vp8 codec id should be enumerated");
        }
    }
}
