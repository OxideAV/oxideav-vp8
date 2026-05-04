//! VP8 decoder — key-frame + inter-frame paths.
//!
//! Inter-frame support covers the three reference frames (LAST / GOLDEN
//! / ALTREF), motion-vector decoding, 6-tap luma + bilinear chroma
//! sub-pel reconstruction, and the sign-bias / refresh / copy-buffer
//! flags that manage the reference slots.

#[cfg(feature = "registry")]
use std::collections::VecDeque;

#[cfg(feature = "registry")]
use oxideav_core::Decoder;
#[cfg(feature = "registry")]
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase, VideoFrame, VideoPlane};

use crate::error::{Result, Vp8Error as Error};
use crate::frame::Vp8Frame;

use crate::bool_decoder::BoolDecoder;
use crate::frame_header::{
    parse_inter_header, parse_keyframe_header, FrameHeader, PersistentProbs,
};
use crate::frame_tag::{parse_header, FrameType};
use crate::inter::{sixtap_predict, RefPlane};
use crate::intra::{predict_16x16, predict_4x4, predict_8x8, B4x4Neighbours};
use crate::loopfilter::{
    filter_normal_horizontal, filter_normal_vertical, filter_simple_horizontal,
    filter_simple_vertical, FilterParams,
};
use crate::mv::{clamp_mv_to_border, decode_mv, Mv};
use crate::tables::quant::{
    clamp_qindex, uv_ac_step, uv_dc_step, y2_ac_step, y2_dc_step, y_ac_step, y_dc_step,
};
use crate::tables::trees::{
    decode_tree, BMODE_TREE, B_DC_PRED, B_HE_PRED, B_PRED, B_TM_PRED, B_VE_PRED, DC_PRED, H_PRED,
    KF_BMODE_PROB, KF_UV_MODE_PROBS, KF_UV_MODE_TREE, KF_YMODE_PROBS, KF_YMODE_TREE, MBSPLIT_PROBS,
    MB_SPLITS, MB_SPLIT_COUNT, MB_SPLIT_TREE, MV_COUNTS_TO_PROBS, MV_REF_TREE, NEAREST_MV, NEAR_MV,
    NEW_MV, SPLIT_MV, SUB_MV_REF_PROBS, SUB_MV_REF_TREE, TM_PRED, UV_MODE_TREE, V_PRED, YMODE_TREE,
    ZERO_MV,
};
use crate::tokens::{decode_block, BlockType};
use crate::transform::{idct4x4, iwht4x4};

const REF_INTRA: u8 = 0;
const REF_LAST: u8 = 1;
const REF_GOLDEN: u8 = 2;
const REF_ALT: u8 = 3;

/// Public factory used by the registry.
#[cfg(feature = "registry")]
pub fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp8Decoder::new(params.codec_id.clone())))
}

/// Reference frame storage. Stride is fixed at MB-aligned width.
#[derive(Clone, Default)]
struct RefFrame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    y_h: usize,
    uv_h: usize,
}

impl RefFrame {
    fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    fn y_plane(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.y,
            stride: self.y_stride,
            width: self.y_stride,
            height: self.y_h,
        }
    }

    fn u_plane(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.u,
            stride: self.uv_stride,
            width: self.uv_stride,
            height: self.uv_h,
        }
    }

    fn v_plane(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.v,
            stride: self.uv_stride,
            width: self.uv_stride,
            height: self.uv_h,
        }
    }
}

/// Per-decoder state that persists between frames.
#[derive(Clone)]
struct DecoderState {
    probs: PersistentProbs,
    last: RefFrame,
    golden: RefFrame,
    altref: RefFrame,
    /// Per-frame scratch buffers reused across decode_frame_with_state
    /// calls. Sized lazily at the top of each frame; capacity is
    /// retained between frames so a steady-state stream allocates only
    /// once at the keyframe (when dimensions become known) and never
    /// again. Heap-allocating these per frame is visible in profiles —
    /// an 1080p frame walks ~8160 mb_info entries plus ~120 above-row
    /// scratch entries plus three multi-MB-byte plane buffers.
    scratch: Scratch,
}

/// Scratch buffers reused across `decode_frame_with_state` invocations.
/// All entries are resized to the current frame's MB dimensions on
/// entry. `nz_*_above` / `bmode_above` are explicitly reset at the top
/// of the per-MB walk, so leftover data from the prior frame doesn't
/// leak across. `mb_info` is fully overwritten by the mode-decode loop
/// in raster order before any neighbour-lookup reads it (find_near_mvs,
/// keyframe B_PRED neighbour walk).
///
/// `y_plane` / `u_plane` / `v_plane` are the per-frame reconstructed
/// planes. They are zero-initialised on resize (keyframe) and fully
/// overwritten by the MB reconstruction loop on subsequent frames —
/// every output pixel is produced by intra-prediction or motion
/// compensation, so leftover data never leaks. Hoisting these saves
/// three multi-MB-byte heap allocations per frame; an 1080p stream is
/// 3 × ~3.1 MiB / frame, very visible in profiles. The borrow checker
/// is satisfied by destructuring the scratch sub-struct into disjoint
/// `&mut` field borrows at the top of `decode_frame_with_state` (Rust
/// permits simultaneous mutable borrows of distinct struct fields), so
/// the inter MB loop holds `&mut y_plane` alongside `&state.last`
/// without conflict.
///
/// `padded_parts` is hoisted as `Vec<Vec<u8>>` — token-partition
/// padded copies. Each inner `Vec` is `clear()`-then-
/// `extend_from_slice()`d each frame so the heap allocation persists
/// across frames; the outer `Vec` is resized to `nb_parts` (1, 2, 4
/// or 8 per the 2-bit `log2_nb_partitions` field). The `BoolDecoder`
/// instances built from these slices hold borrows back into the padded
/// data for the duration of the token walk, but the entire walk
/// completes before `decode_frame_with_state` returns, so the borrows
/// never escape.
#[derive(Clone, Default)]
struct Scratch {
    nz_y_above: Vec<[u8; 4]>,
    nz_uv_above: Vec<[u8; 2]>,
    nz_v_above: Vec<[u8; 2]>,
    nz_y2_above: Vec<u8>,
    bmode_above: Vec<[i32; 4]>,
    mb_info: Vec<MbInfo>,
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
    padded_parts: Vec<Vec<u8>>,
}

impl DecoderState {
    fn new() -> Self {
        Self {
            probs: PersistentProbs::defaults(),
            last: RefFrame::default(),
            golden: RefFrame::default(),
            altref: RefFrame::default(),
            scratch: Scratch::default(),
        }
    }
}

#[cfg(feature = "registry")]
pub struct Vp8Decoder {
    codec_id: CodecId,
    queued: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    state: DecoderState,
}

#[cfg(feature = "registry")]
impl Vp8Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            queued: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 1000),
            state: DecoderState::new(),
        }
    }
}

#[cfg(feature = "registry")]
impl Decoder for Vp8Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        // Hidden frames (`show_frame = 0`) are reference-only: they
        // update the LAST/GOLDEN/ALTREF slots but the consumer must
        // never receive them as a video frame. The encoder uses these
        // for look-ahead alt-ref synthesis (the synthesized image is
        // installed in the alt-ref slot via a hidden P-frame). We
        // detect them on the frame tag and suppress the queue push
        // while still letting `decode_frame_with_state` walk through
        // the bitstream and update reference state.
        let parsed = parse_header(&packet.data)?;
        let visible = parsed.tag.show_frame;
        let frame = decode_frame_with_state(&packet.data, &mut self.state)?;
        if visible {
            let mut vf = vp8_frame_to_video_frame(frame);
            vf.pts = self.pending_pts;
            self.queued.push_back(vf);
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        match self.queued.pop_front() {
            Some(v) => Ok(Frame::Video(v)),
            None => Err(oxideav_core::Error::NeedMore),
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> oxideav_core::Result<()> {
        // VP8 carries per-stream entropy probability tables
        // (`PersistentProbs`) plus the three reference frames
        // (LAST / GOLDEN / ALTREF) between packets. All of these must be
        // dropped: probabilities restart at spec defaults on the next
        // keyframe, and a post-seek inter-frame is an error anyway (the
        // reference pictures it names aren't ours). Resetting to a fresh
        // `DecoderState` takes care of both.
        self.state = DecoderState::new();
        self.queued.clear();
        self.pending_pts = None;
        Ok(())
    }
}

/// Standalone decode entry point — accepts a single VP8 frame buffer
/// (typically a keyframe; an inter-frame works only when the caller
/// has already populated the decoder state) and returns a [`Vp8Frame`]
/// with the cropped YUV 4:2:0 planes.
///
/// This API is unconditional: it works whether or not the `registry`
/// feature is enabled, and never references `oxideav-core` types.
/// Image-library consumers that just want pixels should use this entry
/// point.
pub fn decode_vp8(buf: &[u8]) -> Result<Vp8Frame> {
    let mut state = DecoderState::new();
    decode_frame_with_state(buf, &mut state)
}

/// Legacy alias: identical to [`decode_vp8`] but returns the
/// `oxideav-core::VideoFrame` shape instead of a [`Vp8Frame`]. Gated
/// on the `registry` feature; standalone callers should use
/// [`decode_vp8`].
#[cfg(feature = "registry")]
pub fn decode_frame(buf: &[u8]) -> oxideav_core::Result<VideoFrame> {
    let mut state = DecoderState::new();
    decode_frame_with_state(buf, &mut state)
        .map(vp8_frame_to_video_frame)
        .map_err(Into::into)
}

/// Convert a [`Vp8Frame`] (tight-stride YUV planes) to an
/// `oxideav-core::VideoFrame`. The fields map 1:1 — `Vp8Frame` already
/// holds cropped, tight-stride buffers, so no copying or repacking
/// happens beyond constructing the outer `Vec<VideoPlane>`.
#[cfg(feature = "registry")]
pub(crate) fn vp8_frame_to_video_frame(f: Vp8Frame) -> VideoFrame {
    let cw = f.uv_stride as usize;
    VideoFrame {
        pts: f.pts,
        planes: vec![
            VideoPlane {
                stride: f.y_stride as usize,
                data: f.y,
            },
            VideoPlane {
                stride: cw,
                data: f.u,
            },
            VideoPlane {
                stride: cw,
                data: f.v,
            },
        ],
    }
}

/// Decode a single frame using the given (mutable) decoder state.
fn decode_frame_with_state(buf: &[u8], state: &mut DecoderState) -> Result<Vp8Frame> {
    let parsed = parse_header(buf)?;
    let is_keyframe = matches!(parsed.tag.frame_type, FrameType::Key);

    let (width, height) = if is_keyframe {
        let kf = parsed
            .keyframe
            .ok_or_else(|| Error::invalid("VP8: keyframe header missing"))?;
        (kf.width as usize, kf.height as usize)
    } else if !state.last.is_empty() {
        (state.last.width, state.last.height)
    } else {
        return Err(Error::invalid(
            "VP8: inter-frame before any keyframe — no reference available",
        ));
    };

    if is_keyframe {
        // Keyframes reset persistent entropy state to defaults.
        state.probs = PersistentProbs::defaults();
    }

    // --- bool-coded header ---
    let header_buf = &buf[parsed.compressed_offset..];
    let mut hdr_dec = BoolDecoder::new(header_buf)?;
    let header = if is_keyframe {
        parse_keyframe_header(&mut hdr_dec)?
    } else {
        parse_inter_header(&mut hdr_dec, &state.probs)?
    };

    // --- token partitions ---
    let first_part_size = parsed.tag.first_partition_size as usize;
    let after_header_off = parsed.compressed_offset + first_part_size;
    if after_header_off > buf.len() {
        return Err(Error::invalid("VP8: first partition extends past end"));
    }
    let nb_parts = 1usize << header.log2_nb_partitions;
    let mut parts: Vec<&[u8]> = Vec::with_capacity(nb_parts);

    let mut cursor = after_header_off;
    if nb_parts > 1 {
        let sizes_bytes = (nb_parts - 1) * 3;
        if cursor + sizes_bytes > buf.len() {
            return Err(Error::invalid("VP8: partition size table truncated"));
        }
        let mut sizes = Vec::with_capacity(nb_parts - 1);
        for i in 0..nb_parts - 1 {
            let off = cursor + i * 3;
            let sz = (buf[off] as usize)
                | ((buf[off + 1] as usize) << 8)
                | ((buf[off + 2] as usize) << 16);
            sizes.push(sz);
        }
        cursor += sizes_bytes;
        for sz in sizes {
            if cursor + sz > buf.len() {
                return Err(Error::invalid("VP8: partition extends past end"));
            }
            parts.push(&buf[cursor..cursor + sz]);
            cursor += sz;
        }
        parts.push(&buf[cursor..]);
    } else {
        parts.push(&buf[cursor..]);
    }

    let mb_w = (width + 15) / 16;
    let mb_h = (height + 15) / 16;

    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;

    // Hoisted per-frame scratch lives in `state.scratch`. Resizing
    // is a no-op once dimensions stabilise (after the keyframe), so a
    // steady-state stream allocates these buffers exactly once for the
    // lifetime of the decoder. `bmode_above` MUST be zero-initialised
    // on the very first row of the frame (the `row == 0` branch in
    // `decode_mb_mode_info_keyframe` reads `bmode_above[mb_x][col]`),
    // which `resize` accomplishes when growing from 0; for already-
    // sized buffers the explicit reset further down handles it.
    state.scratch.nz_y_above.resize(mb_w, [0u8; 4]);
    state.scratch.nz_uv_above.resize(mb_w, [0u8; 2]);
    state.scratch.nz_v_above.resize(mb_w, [0u8; 2]);
    state.scratch.nz_y2_above.resize(mb_w, 0u8);
    state.scratch.bmode_above.resize(mb_w, [0i32; 4]);
    // Reset the bmode_above row for this frame — neighbour reads from
    // mb_y == 0 must see fresh zeros, and rows further down get
    // overwritten by the per-MB writes anyway.
    for c in &mut state.scratch.bmode_above {
        *c = [0; 4];
    }
    state.scratch.mb_info.resize(mb_w * mb_h, MbInfo::default());

    // Plane buffers: resize-then-zero so the contents start fresh on
    // every frame regardless of prior dimensions. Resize is a no-op
    // once steady-state, and the zero-fill matches the explicit
    // `vec![0u8; ...]` semantics this code replaced. Every output
    // pixel is rewritten by the reconstruction loop below — the zero
    // floor matters only for the cropping copy out of out-of-bounds
    // pad rows on non-multiple-of-16 frame sizes.
    state.scratch.y_plane.resize(y_stride * y_buf_h, 0);
    state.scratch.u_plane.resize(uv_stride * uv_buf_h, 0);
    state.scratch.v_plane.resize(uv_stride * uv_buf_h, 0);
    state.scratch.y_plane.fill(0);
    state.scratch.u_plane.fill(0);
    state.scratch.v_plane.fill(0);

    // Split the scratch borrow so each sub-field is independently
    // mutable. Without this we'd have to choose between holding a
    // single `&mut state.scratch` (blocking simultaneous `&state.last`
    // reads in the inter MB loop) and re-borrowing per call (verbose
    // and easier to get wrong). The plane fields participate in the
    // same destructuring so the inter MB loop can hold `&mut y_plane`
    // alongside `&state.last`.
    let Scratch {
        nz_y_above,
        nz_uv_above,
        nz_v_above,
        nz_y2_above,
        bmode_above,
        mb_info,
        y_plane,
        u_plane,
        v_plane,
        padded_parts,
    } = &mut state.scratch;

    // --- MB mode decode ---
    let mut mb_dec = hdr_dec;

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let info = if is_keyframe {
                decode_mb_mode_info_keyframe(
                    &mut mb_dec,
                    mb_info.as_slice(),
                    mb_x,
                    mb_y,
                    mb_w,
                    &header,
                    bmode_above.as_mut_slice(),
                )?
            } else {
                decode_mb_mode_info_inter(
                    &mut mb_dec,
                    mb_info.as_slice(),
                    mb_x,
                    mb_y,
                    mb_w,
                    mb_h,
                    &header,
                    bmode_above.as_mut_slice(),
                )?
            };
            mb_info[mb_y * mb_w + mb_x] = info;
        }
    }

    // Pad token partitions that are shorter than the BoolDecoder
    // priming size. VP8 allows trailing zeros to be elided; past-EOF
    // reads in the boolean decoder are already defined to return zero
    // so padding is a no-op from a decoding standpoint.
    //
    // The outer `Vec` and inner `Vec<u8>` storage live in
    // `state.scratch.padded_parts` across frames — `clear()` zeros
    // length without freeing capacity, and `extend_from_slice` reuses
    // the prior allocation when the new partition fits (it usually
    // does once the bitstream has settled into a steady state). The
    // resize+take-from-default trick handles the case where the new
    // frame has more partitions than the previous (the new entries
    // start as empty `Vec::new()` and the first frame at that size
    // pays a single allocation).
    padded_parts.resize_with(parts.len(), Vec::new);
    for (dst, src) in padded_parts.iter_mut().zip(parts.iter()) {
        dst.clear();
        dst.extend_from_slice(src);
        while dst.len() < 2 {
            dst.push(0);
        }
    }
    let mut token_decs: Vec<BoolDecoder> = padded_parts
        .iter()
        .map(|p| BoolDecoder::new(p))
        .collect::<Result<_>>()?;

    for c in nz_y_above.iter_mut() {
        *c = [0; 4];
    }
    for c in nz_uv_above.iter_mut() {
        *c = [0; 2];
    }
    for c in nz_v_above.iter_mut() {
        *c = [0; 2];
    }
    for c in nz_y2_above.iter_mut() {
        *c = 0;
    }

    for mb_y in 0..mb_h {
        let mut nz_y_left = [0u8; 4];
        let mut nz_u_left = [0u8; 2];
        let mut nz_v_left = [0u8; 2];
        let mut nz_y2_left = 0u8;
        let part_idx = mb_y % nb_parts;
        let dec = &mut token_decs[part_idx];

        for mb_x in 0..mb_w {
            let mut info = mb_info[mb_y * mb_w + mb_x].clone();
            let skip = info.skip;

            let is_intra = info.ref_frame == REF_INTRA;
            let has_y2 = if is_intra {
                info.y_mode != B_PRED
            } else {
                info.inter_split_mode.is_none() && info.y_mode != B_PRED
            };
            // For inter MBs, Y2 is used when the MB is NOT using SPLITMV.
            let has_y2 = if !is_intra {
                info.inter_split_mode.is_none()
            } else {
                has_y2
            };

            let mut y2_coeffs = [0i16; 16];
            let mut y_coeffs = [[0i16; 16]; 16];
            let mut u_coeffs = [[0i16; 16]; 4];
            let mut v_coeffs = [[0i16; 16]; 4];

            // Track whether the MB has any non-zero coefficient
            // anywhere — needed by the loop-filter sub-block-edge skip
            // rule (RFC 6386 §15.1, libvpx `eob_mask`).
            let mut any_coeffs = false;
            if !skip {
                if has_y2 {
                    let nctx = nz_y2_above[mb_x] + nz_y2_left;
                    let nz = decode_block(
                        dec,
                        &header.coef_probs,
                        BlockType::Y2,
                        nctx,
                        &mut y2_coeffs,
                        0,
                    );
                    let nz_flag = if nz > 0 { 1 } else { 0 };
                    nz_y2_above[mb_x] = nz_flag;
                    nz_y2_left = nz_flag;
                    if nz > 0 {
                        any_coeffs = true;
                    }
                }

                let block_type = if has_y2 {
                    BlockType::YAfterY2
                } else {
                    BlockType::YNoY2
                };
                let start = if has_y2 { 1 } else { 0 };
                for by in 0..4 {
                    for bx in 0..4 {
                        let idx = by * 4 + bx;
                        let above_nz = nz_y_above[mb_x][bx];
                        let left_nz = nz_y_left[by];
                        let nctx = above_nz + left_nz;
                        let nz = decode_block(
                            dec,
                            &header.coef_probs,
                            block_type,
                            nctx,
                            &mut y_coeffs[idx],
                            start,
                        );
                        let nz_flag = if nz > 0 { 1 } else { 0 };
                        nz_y_above[mb_x][bx] = nz_flag;
                        nz_y_left[by] = nz_flag;
                        if nz > 0 {
                            any_coeffs = true;
                        }
                    }
                }

                // Per RFC 6386 §13.1: the residue record has 4 DCTs for
                // all U subblocks (raster order) followed by 4 DCTs for
                // all V subblocks. Do NOT interleave U/V — they are
                // separate contiguous groups in the bitstream, and they
                // also maintain independent above/left nz-flag contexts.
                for by in 0..2 {
                    for bx in 0..2 {
                        let idx = by * 2 + bx;
                        let above_nz = nz_uv_above[mb_x][bx];
                        let left_nz = nz_u_left[by];
                        let nctx = above_nz + left_nz;
                        let nz = decode_block(
                            dec,
                            &header.coef_probs,
                            BlockType::UV,
                            nctx,
                            &mut u_coeffs[idx],
                            0,
                        );
                        let nz_flag = if nz > 0 { 1 } else { 0 };
                        nz_uv_above[mb_x][bx] = nz_flag;
                        nz_u_left[by] = nz_flag;
                        if nz > 0 {
                            any_coeffs = true;
                        }
                    }
                }
                for by in 0..2 {
                    for bx in 0..2 {
                        let idx = by * 2 + bx;
                        let above_nz = nz_v_above[mb_x][bx];
                        let left_nz = nz_v_left[by];
                        let nctx = above_nz + left_nz;
                        let nz = decode_block(
                            dec,
                            &header.coef_probs,
                            BlockType::UV,
                            nctx,
                            &mut v_coeffs[idx],
                            0,
                        );
                        let nz_flag = if nz > 0 { 1 } else { 0 };
                        nz_v_above[mb_x][bx] = nz_flag;
                        nz_v_left[by] = nz_flag;
                        if nz > 0 {
                            any_coeffs = true;
                        }
                    }
                }
            } else {
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
                    nz_v_above[mb_x][bx] = 0;
                    nz_u_left[bx] = 0;
                    nz_v_left[bx] = 0;
                }
            }
            info.has_coeffs = any_coeffs;
            mb_info[mb_y * mb_w + mb_x].has_coeffs = any_coeffs;

            if is_intra {
                reconstruct_intra_mb(
                    &header,
                    &info,
                    has_y2,
                    &y2_coeffs,
                    &y_coeffs,
                    &u_coeffs,
                    &v_coeffs,
                    mb_x,
                    mb_y,
                    mb_w,
                    y_plane.as_mut_slice(),
                    u_plane.as_mut_slice(),
                    v_plane.as_mut_slice(),
                    y_stride,
                    uv_stride,
                );
            } else {
                reconstruct_inter_mb(
                    &header,
                    &state.last,
                    &state.golden,
                    &state.altref,
                    &info,
                    has_y2,
                    &y2_coeffs,
                    &y_coeffs,
                    &u_coeffs,
                    &v_coeffs,
                    mb_x,
                    mb_y,
                    y_plane.as_mut_slice(),
                    u_plane.as_mut_slice(),
                    v_plane.as_mut_slice(),
                    y_stride,
                    uv_stride,
                );
            }
        }
    }

    // Loop filter.
    apply_loop_filter(
        &header,
        mb_info.as_slice(),
        mb_w,
        mb_h,
        y_plane.as_mut_slice(),
        u_plane.as_mut_slice(),
        v_plane.as_mut_slice(),
        y_stride,
        uv_stride,
        y_buf_h,
        uv_buf_h,
        is_keyframe,
    );

    // Update reference frames based on flags. Cloning the destructured
    // `&mut Vec<u8>` field auto-derefs to `Vec<u8>::clone`, producing
    // owned heap copies for the reference snapshots.
    let new_frame = RefFrame {
        y: y_plane.clone(),
        u: u_plane.clone(),
        v: v_plane.clone(),
        width,
        height,
        y_stride,
        uv_stride,
        y_h: y_buf_h,
        uv_h: uv_buf_h,
    };

    if is_keyframe {
        // Keyframes refresh all three references.
        state.last = new_frame.clone();
        state.golden = new_frame.clone();
        state.altref = new_frame;
    } else {
        // Apply copy-to flags first (reference snapshots before updates).
        let prev_last = state.last.clone();
        let prev_golden = state.golden.clone();
        let prev_altref = state.altref.clone();

        match header.copy_buffer_to_golden {
            1 => state.golden = prev_last.clone(),
            2 => state.golden = prev_altref.clone(),
            _ => {}
        }
        match header.copy_buffer_to_alternate {
            1 => state.altref = prev_last.clone(),
            2 => state.altref = prev_golden.clone(),
            _ => {}
        }
        if header.refresh_alternate {
            state.altref = new_frame.clone();
        }
        if header.refresh_golden {
            state.golden = new_frame.clone();
        }
        if header.refresh_last {
            state.last = new_frame;
        }
    }

    // Update persistent probability state if indicated.
    //
    // RFC 6386 §13.5 / libvpx `vp8_decode_frame`:
    // * Keyframes always reset persistent state to defaults at the start
    //   (handled above at `state.probs = PersistentProbs::defaults()`).
    //   They then save the (possibly in-frame-modified) header values
    //   into persistent state ONLY when `refresh_entropy_probs=1`.
    // * P-frames save the (in-frame-modified) header values into
    //   persistent state when `refresh_entropy_probs=1`. When 0, the
    //   persistent state stays at whatever the previous frame saved.
    //
    // The keyframe-with-refresh_entropy=0 case is the small-roi-
    // segmentation fixture: the in-frame coef-probs updates apply to
    // the keyframe ONLY, and the next P-frame must read with the
    // RFC 6386 default coef table — not the keyframe's modified copy.
    if header.refresh_entropy_probs {
        state.probs.coef_probs = header.coef_probs;
        state.probs.ymode_probs = header.ymode_probs;
        state.probs.uv_mode_probs = header.uv_mode_probs;
        state.probs.mv_context = header.mv_context;
        state.probs.mb_skip_prob = header.mb_skip_prob;
        state.probs.mb_skip_enabled = header.mb_skip_enabled;
    }
    // Loop-filter ref/mode deltas persist across frames REGARDLESS of
    // `refresh_entropy_probs` — they're a separate carry-over state per
    // RFC 6386 §9.4 (`mode_ref_lf_delta_update`). Stash whatever the
    // current frame ended up with so the next frame's parse can inherit.
    state.probs.ref_deltas = header.loop_filter.ref_deltas;
    state.probs.mode_deltas = header.loop_filter.mode_deltas;

    // Crop.
    let mut y_out = vec![0u8; width * height];
    for j in 0..height {
        let src = &y_plane[j * y_stride..j * y_stride + width];
        y_out[j * width..j * width + width].copy_from_slice(src);
    }
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let mut u_out = vec![0u8; cw * ch];
    let mut v_out = vec![0u8; cw * ch];
    for j in 0..ch {
        let src_u = &u_plane[j * uv_stride..j * uv_stride + cw];
        u_out[j * cw..j * cw + cw].copy_from_slice(src_u);
        let src_v = &v_plane[j * uv_stride..j * uv_stride + cw];
        v_out[j * cw..j * cw + cw].copy_from_slice(src_v);
    }

    Ok(Vp8Frame {
        width: width as u32,
        height: height as u32,
        pts: None,
        y: y_out,
        u: u_out,
        v: v_out,
        y_stride: width as u32,
        uv_stride: cw as u32,
    })
}

#[derive(Clone, Default)]
struct MbInfo {
    /// Y intra mode, or inter Y-mode code (NEAREST/NEAR/ZERO/NEW/SPLIT) in
    /// the same int namespace (values 10..=14 for inter).
    y_mode: i32,
    bmodes: [i32; 16],
    uv_mode: i32,
    skip: bool,
    #[allow(dead_code)]
    segment_id: u8,
    /// 0 = intra, 1 = LAST, 2 = GOLDEN, 3 = ALT.
    ref_frame: u8,
    /// MV for the MB (used when inter and not SPLITMV).
    mv: Mv,
    /// Per-subblock MVs (inter + SPLITMV).
    sub_mvs: [Mv; 16],
    /// Split mode (when y_mode == SPLIT_MV).
    inter_split_mode: Option<u8>,
    /// True if any non-zero DCT/WHT coefficient was decoded for this
    /// MB (libvpx's "eob_mask"). Loopfilter step 2 + 4 (sub-block edge
    /// filtering) is skipped when this is false AND y_mode is neither
    /// B_PRED nor SPLITMV — see RFC 6386 §15.1.
    has_coeffs: bool,
}

fn decode_mb_mode_info_keyframe(
    dec: &mut BoolDecoder<'_>,
    mb_info: &[MbInfo],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    header: &FrameHeader,
    bmode_above: &mut [[i32; 4]],
) -> Result<MbInfo> {
    let mut info = MbInfo::default();
    info.ref_frame = REF_INTRA;
    if header.segmentation.enabled && header.segmentation.update_map {
        let probs = &header.segmentation.tree_probs;
        let s0 = dec.read_bool(probs[0] as u32) as u8;
        let s = if s0 == 0 {
            dec.read_bool(probs[1] as u32) as u8
        } else {
            2 + dec.read_bool(probs[2] as u32) as u8
        };
        info.segment_id = s;
    }
    info.skip = if header.mb_skip_enabled {
        dec.read_bool(header.mb_skip_prob as u32)
    } else {
        false
    };
    info.y_mode = decode_tree(dec, &KF_YMODE_TREE, &KF_YMODE_PROBS);
    if info.y_mode == B_PRED {
        let mut left_bmodes = if mb_x > 0 {
            let l = &mb_info[mb_y * mb_w + mb_x - 1];
            if l.y_mode == B_PRED {
                [l.bmodes[3], l.bmodes[7], l.bmodes[11], l.bmodes[15]]
            } else {
                [intra_to_b(l.y_mode); 4]
            }
        } else {
            [intra_to_b(0); 4]
        };
        let mut new_above = [0i32; 4];
        for i in 0..16 {
            let row = i / 4;
            let col = i % 4;
            let above_mode = if row == 0 {
                bmode_above[mb_x][col]
            } else {
                info.bmodes[(row - 1) * 4 + col]
            };
            let left_mode = if col == 0 {
                left_bmodes[row]
            } else {
                info.bmodes[row * 4 + col - 1]
            };
            let probs = &KF_BMODE_PROB[above_mode as usize][left_mode as usize];
            let m = decode_tree(dec, &BMODE_TREE, probs);
            info.bmodes[i] = m;
            if row == 3 {
                new_above[col] = m;
            }
            if col == 3 {
                left_bmodes[row] = m;
            }
        }
        bmode_above[mb_x] = new_above;
    } else {
        let bm = intra_to_b(info.y_mode);
        for i in 0..16 {
            info.bmodes[i] = bm;
        }
        bmode_above[mb_x] = [bm; 4];
    }
    info.uv_mode = decode_tree(dec, &KF_UV_MODE_TREE, &KF_UV_MODE_PROBS);
    Ok(info)
}

fn decode_mb_mode_info_inter(
    dec: &mut BoolDecoder<'_>,
    mb_info: &[MbInfo],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    mb_h: usize,
    header: &FrameHeader,
    bmode_above: &mut [[i32; 4]],
) -> Result<MbInfo> {
    let mut info = MbInfo::default();
    if header.segmentation.enabled && header.segmentation.update_map {
        let probs = &header.segmentation.tree_probs;
        let s0 = dec.read_bool(probs[0] as u32) as u8;
        let s = if s0 == 0 {
            dec.read_bool(probs[1] as u32) as u8
        } else {
            2 + dec.read_bool(probs[2] as u32) as u8
        };
        info.segment_id = s;
    }
    info.skip = if header.mb_skip_enabled {
        dec.read_bool(header.mb_skip_prob as u32)
    } else {
        false
    };
    // intra vs inter
    let is_inter = dec.read_bool(header.prob_intra as u32);
    if !is_inter {
        // Intra MB inside inter frame.
        info.ref_frame = REF_INTRA;
        // Use inter Y mode tree + dynamic probs.
        info.y_mode = decode_tree(dec, &YMODE_TREE, &header.ymode_probs);
        if info.y_mode == B_PRED {
            // B_PRED uses its default probs (not context-sensitive) inside
            // inter frames. See RFC 6386 §16.3 — uses `vp8_bmode_prob`.
            let default_bmode_probs: [u8; 9] = [120, 90, 79, 133, 87, 85, 80, 111, 151];
            for i in 0..16 {
                let m = decode_tree(dec, &BMODE_TREE, &default_bmode_probs);
                info.bmodes[i] = m;
            }
            bmode_above[mb_x] = [
                info.bmodes[12],
                info.bmodes[13],
                info.bmodes[14],
                info.bmodes[15],
            ];
        } else {
            let bm = intra_to_b(info.y_mode);
            for i in 0..16 {
                info.bmodes[i] = bm;
            }
            bmode_above[mb_x] = [bm; 4];
        }
        info.uv_mode = decode_tree(dec, &UV_MODE_TREE, &header.uv_mode_probs);
        return Ok(info);
    }
    // Inter MB — pick reference frame.
    info.ref_frame = if dec.read_bool(header.prob_last as u32) {
        if dec.read_bool(header.prob_gf as u32) {
            REF_ALT
        } else {
            REF_GOLDEN
        }
    } else {
        REF_LAST
    };

    // Find nearest / near / best MV context from neighbours.
    // RFC 6386 §16.3 calls `vp8_clamp_mv` on `nearest`, `near`, and
    // `best_mv` at the end of `vp8_find_near_mvs`, before any of them
    // is fed back into the prediction. The clamp restricts each
    // component to within one MB beyond the visible image edge in
    // 1/8-pel units; without it a neighbour-inherited MV at frame-edge
    // MBs (mb_y close to mb_h-1 or mb_x close to mb_w-1) can point
    // arbitrarily far outside the §18.1 1-MB extended border, breaking
    // sub-pel reconstruction at all four ReportOnly P-frame fixtures.
    let (raw_nearest, raw_near, raw_best, cnt) =
        find_near_mvs(mb_info, mb_x, mb_y, mb_w, info.ref_frame, header);
    let nearest = clamp_mv_to_border(raw_nearest, mb_x, mb_y, mb_w, mb_h);
    let near = clamp_mv_to_border(raw_near, mb_x, mb_y, mb_w, mb_h);
    let best_mv = clamp_mv_to_border(raw_best, mb_x, mb_y, mb_w, mb_h);

    let ctx_probs = mv_ref_probs(&cnt);
    // Tree leaves start at 10 in this decoder's int namespace.
    let leaf = decode_tree(dec, &MV_REF_TREE, &ctx_probs);
    info.y_mode = leaf + 10;

    match info.y_mode {
        NEAREST_MV => info.mv = nearest,
        NEAR_MV => info.mv = near,
        ZERO_MV => info.mv = Mv::ZERO,
        NEW_MV => {
            // Decode MV difference, add to best_mv. RFC 6386 §18.1
            // mandates a *secondary* clamp here for NEWMV: "the final
            // motion vector is clamped again after combining the 'best'
            // predictor and the differential vector decoded from the
            // stream." This secondary clamp is NOT applied to SPLITMV
            // sub-MVs (per the same §18.1 paragraph).
            let dmv = decode_mv(dec, &header.mv_context);
            let combined = Mv::new(
                best_mv.row as i32 + dmv.row as i32,
                best_mv.col as i32 + dmv.col as i32,
            );
            info.mv = clamp_mv_to_border(combined, mb_x, mb_y, mb_w, mb_h);
        }
        SPLIT_MV => {
            // Decode split mode then sub-MVs.
            let split = decode_tree(dec, &MB_SPLIT_TREE, &MBSPLIT_PROBS) as u8;
            info.inter_split_mode = Some(split);
            let n = MB_SPLIT_COUNT[split as usize] as usize;
            let partition = &MB_SPLITS[split as usize];
            let mut part_mvs = [Mv::ZERO; 16];
            // For each partition, find its first 4×4 and decode one MV.
            for p in 0..n {
                let first_idx = (0..16).find(|&i| partition[i] as usize == p).unwrap();
                let row = first_idx / 4;
                let col = first_idx % 4;
                // Neighbour sub-MVs.
                let left_mv = if col == 0 {
                    if mb_x > 0 {
                        let l = &mb_info[mb_y * mb_w + mb_x - 1];
                        left_edge_mv(l, row)
                    } else {
                        Mv::ZERO
                    }
                } else {
                    part_mvs[row * 4 + col - 1]
                };
                let above_mv = if row == 0 {
                    if mb_y > 0 {
                        let a = &mb_info[(mb_y - 1) * mb_w + mb_x];
                        top_edge_mv(a, col)
                    } else {
                        Mv::ZERO
                    }
                } else {
                    part_mvs[(row - 1) * 4 + col]
                };
                let sub_prob_row = sub_mv_context(&left_mv, &above_mv);
                let sub_tree_leaf =
                    decode_tree(dec, &SUB_MV_REF_TREE, &SUB_MV_REF_PROBS[sub_prob_row]);
                let chosen = match sub_tree_leaf {
                    0 => left_mv,  // LEFT_4x4
                    1 => above_mv, // ABOVE_4x4
                    2 => Mv::ZERO, // ZERO_4x4
                    _ => {
                        // NEW_4x4 — decode diff from best.
                        let dmv = decode_mv(dec, &header.mv_context);
                        Mv::new(
                            best_mv.row as i32 + dmv.row as i32,
                            best_mv.col as i32 + dmv.col as i32,
                        )
                    }
                };
                for i in 0..16 {
                    if partition[i] as usize == p {
                        part_mvs[i] = chosen;
                    }
                }
            }
            info.sub_mvs = part_mvs;
            // For downstream code, record the MB MV as the bottom-right sub-mv
            // (commonly used for propagation context).
            info.mv = part_mvs[15];
        }
        _ => {
            return Err(Error::invalid("VP8: invalid inter mode"));
        }
    }

    // Populate sub_mvs for non-SPLIT case.
    if info.inter_split_mode.is_none() {
        for s in &mut info.sub_mvs {
            *s = info.mv;
        }
    }

    // UV mode is not coded for inter MBs — it uses the same motion as luma.
    info.uv_mode = DC_PRED;
    // Reset bmode_above propagation for inter MBs (neighbour b-mode becomes
    // "predicted-as-DC").
    bmode_above[mb_x] = [intra_to_b(DC_PRED); 4];
    Ok(info)
}

/// Neighbour 4×4 sub-MV for the *bottom* row of an above-neighbour MB at
/// sub-block column `col`. For inter MBs this is the MB's MV (or, if
/// SPLIT, the sub-block at row 3).
fn top_edge_mv(a: &MbInfo, col: usize) -> Mv {
    if a.ref_frame == REF_INTRA {
        return Mv::ZERO;
    }
    if a.inter_split_mode.is_some() {
        a.sub_mvs[12 + col]
    } else {
        a.mv
    }
}

/// Neighbour 4×4 sub-MV for the *right* column of a left-neighbour MB at
/// sub-block row `row`.
fn left_edge_mv(l: &MbInfo, row: usize) -> Mv {
    if l.ref_frame == REF_INTRA {
        return Mv::ZERO;
    }
    if l.inter_split_mode.is_some() {
        l.sub_mvs[row * 4 + 3]
    } else {
        l.mv
    }
}

/// Pick the SUB_MV_REF_PROBS row based on neighbour MV pair.
///
/// Mirrors RFC 6386 §16.3 `vp8_mvCont` exactly:
///
/// ```c
/// if (left == above && left == 0) return 4; /* LEFT_ABOVE_ZED */
/// if (left == above)              return 3; /* LEFT_ABOVE_SAME (both non-zero) */
/// if (above == 0)                 return 2; /* ABOVE_ZED      (left non-zero) */
/// if (left  == 0)                 return 1; /* LEFT_ZED       (above non-zero) */
/// return 0;                                 /* NORMAL */
/// ```
///
/// The returned value indexes [`SUB_MV_REF_PROBS`] in RFC order. An
/// earlier version of this function used a different (scrambled) row
/// numbering and the wrong probability vector was applied to every
/// SPLITMV sub-block, producing garbage sub-MVs (often
/// 3-digit-magnitude) at frame-edge MBs in P-frames.
fn sub_mv_context(left: &Mv, above: &Mv) -> usize {
    let l_zero = left.row == 0 && left.col == 0;
    let a_zero = above.row == 0 && above.col == 0;
    let same = left == above;
    if same && l_zero {
        4 // LEFT_ABOVE_ZED
    } else if same {
        3 // LEFT_ABOVE_SAME (both non-zero, identical)
    } else if a_zero {
        2 // ABOVE_ZED (left non-zero)
    } else if l_zero {
        1 // LEFT_ZED (above non-zero)
    } else {
        0 // NORMAL (both non-zero, distinct)
    }
}

/// RFC 6386 §16.3 `find_near_mvs`. Returns (nearest, near, best, cnt)
/// where `cnt` is the 4-entry neighbour-count vector used to derive the
/// MV mode reference probabilities.
///
/// This follows the reference C pseudo-code near-verbatim:
///   * `mv` and `cntx` start pointing at slot 0 (ZEROZERO / BEST).
///   * For each of the three neighbours (above, left, aboveleft with
///     weights 2, 2, 1):
///     - If the neighbour is INTRA (`ref_frame == CURRENT_FRAME`), it
///       contributes nothing.
///     - Else if the neighbour's MV (after sign-bias normalisation) is
///       (0,0), its weight is added to `cnt[CNT_ZEROZERO]`.
///     - Else the MV is added as the next distinct candidate (NEAREST
///       → NEAR → a transient slot used as a merge check).
///   * After aboveleft: if a 3rd distinct candidate landed in the
///     SPLITMV temp slot and it equals NEAREST, merge it. Then reset
///     `cnt[CNT_SPLITMV]` to the weighted count of neighbours whose
///     y_mode is SPLITMV.
///   * Swap NEAR/NEAREST if cnt[NEAR] > cnt[NEAREST].
///   * `best` is NEAREST if `cnt[NEAREST] >= cnt[BEST]`, else ZERO.
///
/// Out-of-frame neighbours are treated like INTRA neighbours (they
/// contribute nothing). This is what libvpx does via its mb_info
/// storage layout (the frame border MBs are pre-zeroed as intra).
fn find_near_mvs(
    mb_info: &[MbInfo],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    ref_frame: u8,
    header: &FrameHeader,
) -> (Mv, Mv, Mv, [u8; 4]) {
    // mvs[0]=BEST/ZEROZERO, mvs[1]=NEAREST, mvs[2]=NEAR, mvs[3]=SPLITMV-temp
    let mut mvs: [Mv; 4] = [Mv::ZERO; 4];
    let mut cnt = [0u8; 4];
    // Pointer-like index into mvs/cnt — advances as new distinct MVs are seen.
    let mut mv_idx: usize = 0;

    // Neighbours: above (0,-1,w=2), left (-1,0,w=2), aboveleft (-1,-1,w=1).
    let neighbours: [(isize, isize, u8); 3] = [(0, -1, 2), (-1, 0, 2), (-1, -1, 1)];
    let mut splitmv_count = 0u8; // counts how many neighbours are SPLITMV (weighted)

    for (nb_idx, &(dx, dy, weight)) in neighbours.iter().enumerate() {
        let nx = mb_x as isize + dx;
        let ny = mb_y as isize + dy;
        let out_of_frame = nx < 0 || ny < 0 || nx as usize >= mb_w;
        // Treat out-of-frame as intra (contributes nothing).
        if out_of_frame {
            continue;
        }
        let n = &mb_info[(ny as usize) * mb_w + (nx as usize)];
        // Intra neighbours contribute nothing.
        if n.ref_frame == REF_INTRA {
            continue;
        }

        // Track SPLITMV weighted count for cnt[CNT_SPLITMV] at the end.
        if n.inter_split_mode.is_some() {
            splitmv_count += weight;
        }

        // Apply sign-bias flip if neighbour's ref sign-bias differs from ours.
        let mut nmv = n.mv;
        if ref_frame_sign_bias(ref_frame, header) != ref_frame_sign_bias(n.ref_frame, header) {
            nmv = Mv::new(-(nmv.row as i32), -(nmv.col as i32));
        }

        if nmv.row == 0 && nmv.col == 0 {
            // Zero MV — add weight to current slot. In the RFC, this is
            // `cnt[CNT_ZEROZERO]` for above, but for left/aboveleft, the
            // zero-MV weight still goes to cnt[CNT_ZEROZERO] — see the
            // explicit `cnt[CNT_ZEROZERO] += ...` lines in the left and
            // aboveleft paths.
            cnt[0] += weight;
            continue;
        }

        // Non-zero MV.
        if nb_idx == 0 {
            // First neighbour (above) always becomes NEAREST.
            mv_idx = 1;
            mvs[1] = nmv;
            cnt[1] += weight;
        } else {
            // Merge if matches current slot's MV, else advance slot.
            if mvs[mv_idx] != nmv {
                mv_idx += 1;
                mvs[mv_idx] = nmv;
            }
            cnt[mv_idx] += weight;
        }
    }

    // Post-pass 1: if a 3rd distinct candidate was produced (landed in
    // SPLITMV-temp slot 3) AND it equals NEAREST, merge it.
    if mv_idx == 3 && mvs[3] == mvs[1] {
        cnt[1] += 1;
    }

    // Post-pass 2: reset cnt[CNT_SPLITMV] to the actual weighted count of
    // SPLITMV-mode neighbours (this overwrites any use of cnt[3] as a
    // transient counter above).
    cnt[3] = splitmv_count;

    // Swap NEAR and NEAREST if cnt[NEAR] > cnt[NEAREST].
    if cnt[2] > cnt[1] {
        cnt.swap(1, 2);
        mvs.swap(1, 2);
    }

    // Best is NEAREST if cnt[NEAREST] >= cnt[BEST], else ZERO (slot 0).
    if cnt[1] >= cnt[0] {
        mvs[0] = mvs[1];
    }

    (mvs[1], mvs[2], mvs[0], cnt)
}

/// Compute the MV reference tree probabilities from the neighbour-count
/// vector. RFC 6386 §16.3:
///
/// ```c
/// probs[0] = mv_counts_to_probs[cnt[0]][0];
/// probs[1] = mv_counts_to_probs[cnt[1]][1];
/// probs[2] = mv_counts_to_probs[cnt[2]][2];
/// probs[3] = mv_counts_to_probs[cnt[3]][3];
/// ```
///
/// Each `cnt[i]` indexes a row independently of the others, and
/// selects column `i` from that row. The max count is 2+2+1=5 (the
/// neighbour weights), so counts ≤ 5 always — the `.min(5)` is a
/// safety clamp against malformed input.
fn mv_ref_probs(cnt: &[u8; 4]) -> [u8; 4] {
    [
        MV_COUNTS_TO_PROBS[cnt[0].min(5) as usize][0],
        MV_COUNTS_TO_PROBS[cnt[1].min(5) as usize][1],
        MV_COUNTS_TO_PROBS[cnt[2].min(5) as usize][2],
        MV_COUNTS_TO_PROBS[cnt[3].min(5) as usize][3],
    ]
}

fn ref_frame_sign_bias(rf: u8, header: &FrameHeader) -> bool {
    match rf {
        REF_GOLDEN => header.sign_bias_golden,
        REF_ALT => header.sign_bias_alternate,
        _ => false,
    }
}

fn intra_to_b(intra_mode: i32) -> i32 {
    match intra_mode {
        DC_PRED => B_DC_PRED,
        V_PRED => B_VE_PRED,
        H_PRED => B_HE_PRED,
        TM_PRED => B_TM_PRED,
        _ => B_DC_PRED,
    }
}

/// Effective base luma AC quant index for an MB given the frame header's
/// segmentation data and the MB's segment id. RFC 6386 §10:
///
///   * `enabled = false` → frame-level `quant.y_ac_qi`.
///   * `enabled = true`, `abs_delta = false` → frame qi + per-segment delta.
///   * `enabled = true`, `abs_delta = true`  → per-segment value (absolute).
///
/// The result is clamped to the legal `[0, 127]` qindex range so the
/// quant-step LUTs stay in bounds.
fn mb_base_qi(header: &FrameHeader, segment_id: u8) -> i32 {
    let frame_qi = header.quant.y_ac_qi;
    if !header.segmentation.enabled {
        return clamp_qindex(frame_qi) as i32;
    }
    let s = (segment_id as usize).min(3);
    let raw = if header.segmentation.abs_delta {
        header.segmentation.quant[s]
    } else {
        frame_qi + header.segmentation.quant[s]
    };
    clamp_qindex(raw) as i32
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_intra_mb(
    header: &FrameHeader,
    info: &MbInfo,
    has_y2: bool,
    y2_coeffs: &[i16; 16],
    y_coeffs: &[[i16; 16]; 16],
    u_coeffs: &[[i16; 16]; 4],
    v_coeffs: &[[i16; 16]; 4],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
) {
    let qi = mb_base_qi(header, info.segment_id);
    let y_dc = y_dc_step(qi + header.quant.y_dc_delta);
    let y_ac = y_ac_step(qi);
    let y2_dc = y2_dc_step(qi + header.quant.y2_dc_delta);
    let y2_ac = y2_ac_step(qi + header.quant.y2_ac_delta);
    let uv_dc = uv_dc_step(qi + header.quant.uv_dc_delta);
    let uv_ac = uv_ac_step(qi + header.quant.uv_ac_delta);

    let y2_dc_vals: [i16; 16] = if has_y2 {
        let mut deq = [0i16; 16];
        for i in 0..16 {
            let v = y2_coeffs[i] as i32;
            let q = if i == 0 { y2_dc } else { y2_ac };
            deq[i] = (v * q) as i16;
        }
        iwht4x4(&deq)
    } else {
        [0; 16]
    };

    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;
    if info.y_mode == B_PRED {
        let above_right_extension: [u8; 4] = if mb_y_px > 0 {
            let row = mb_y_px - 1;
            let mut ext = [0u8; 4];
            for k in 0..4 {
                let xx = mb_x_px + 16 + k;
                if xx < mb_w * 16 {
                    ext[k] = y_plane[row * y_stride + xx];
                } else {
                    ext[k] = y_plane[row * y_stride + (mb_x_px + 15)];
                }
            }
            ext
        } else {
            [127; 4]
        };
        for i in 0..16 {
            let by = i / 4;
            let bx = i % 4;
            let dst_x = mb_x_px + bx * 4;
            let dst_y = mb_y_px + by * 4;
            let mut neigh = B4x4Neighbours {
                above: [127; 8],
                left: [129; 4],
                tl: 127,
            };
            if dst_y > 0 {
                for k in 0..4 {
                    neigh.above[k] = y_plane[(dst_y - 1) * y_stride + dst_x + k];
                }
                if bx == 3 && by > 0 {
                    neigh.above[4..8].copy_from_slice(&above_right_extension);
                } else {
                    for k in 4..8 {
                        let xx = dst_x + k;
                        if xx < mb_x_px + 16 {
                            neigh.above[k] = y_plane[(dst_y - 1) * y_stride + xx];
                        } else if by == 0 {
                            if xx < mb_w * 16 {
                                neigh.above[k] = y_plane[(dst_y - 1) * y_stride + xx];
                            } else {
                                neigh.above[k] = y_plane[(dst_y - 1) * y_stride + (mb_x_px + 15)];
                            }
                        } else {
                            neigh.above[k] = above_right_extension[(xx - mb_x_px) - 16];
                        }
                    }
                }
            }
            if dst_x > 0 {
                for k in 0..4 {
                    neigh.left[k] = y_plane[(dst_y + k) * y_stride + dst_x - 1];
                }
            }
            // Top-left "P" pixel. Per RFC 6386 §16.2 and libvpx, the TL
            // defaults depend on availability:
            //   * both available → actual pixel at (dst_y-1, dst_x-1)
            //   * only above available (dst_x == 0, dst_y > 0) → 129
            //     (matches the default for the left column)
            //   * only left available (dst_x > 0, dst_y == 0) → 127
            //     (matches the default for the above row)
            //   * neither (top-left of frame) → 127
            neigh.tl = if dst_x > 0 && dst_y > 0 {
                y_plane[(dst_y - 1) * y_stride + dst_x - 1]
            } else if dst_y > 0 {
                // left column unavailable → default 129
                129
            } else {
                // above unavailable → default 127
                127
            };

            let mut pred = [0u8; 16];
            predict_4x4(info.bmodes[i], &neigh, &mut pred, 4);

            let mut deq = [0i16; 16];
            for k in 0..16 {
                let q = if k == 0 { y_dc } else { y_ac };
                deq[k] = (y_coeffs[i][k] as i32 * q) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[r * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    y_plane[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    } else {
        let mut above = [0u8; 16];
        let mut left = [0u8; 16];
        let above_avail = mb_y_px > 0;
        let left_avail = mb_x_px > 0;
        if above_avail {
            for i in 0..16 {
                above[i] = y_plane[(mb_y_px - 1) * y_stride + mb_x_px + i];
            }
        }
        if left_avail {
            for j in 0..16 {
                left[j] = y_plane[(mb_y_px + j) * y_stride + mb_x_px - 1];
            }
        }
        // TL pixel defaults: when only one neighbour is available, TL
        // takes the *other* neighbour's default (127 for above-default
        // row, 129 for left-default column).
        let tl = if above_avail && left_avail {
            Some(y_plane[(mb_y_px - 1) * y_stride + mb_x_px - 1])
        } else if above_avail {
            // left unavailable → default left column is 129
            Some(129)
        } else if left_avail {
            // above unavailable → default above row is 127
            Some(127)
        } else {
            None
        };
        // Stack-allocated 16×16 luma intra-prediction scratch. Hoisting
        // this out of the heap saves one malloc + one free per intra MB
        // (16 of them on a 64×64 keyframe, scaled per area).
        let mut pred = [0u8; 16 * 16];
        predict_16x16(
            info.y_mode,
            if above_avail { Some(&above) } else { None },
            if left_avail { Some(&left) } else { None },
            tl,
            &mut pred,
            16,
        );
        for i in 0..16 {
            let by = i / 4;
            let bx = i % 4;
            let mut deq = [0i16; 16];
            deq[0] = y2_dc_vals[i];
            for k in 1..16 {
                deq[k] = (y_coeffs[i][k] as i32 * y_ac) as i16;
            }
            let res = idct4x4(&deq);
            let dst_x = mb_x_px + bx * 4;
            let dst_y = mb_y_px + by * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    y_plane[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    // UV — intra.
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for plane_sel in 0..2 {
        let (plane, coeffs) = if plane_sel == 0 {
            (u_plane.as_mut(), u_coeffs)
        } else {
            (v_plane.as_mut(), v_coeffs)
        };
        let mut above = [0u8; 8];
        let mut left = [0u8; 8];
        let above_avail = mb_yc > 0;
        let left_avail = mb_xc > 0;
        if above_avail {
            for i in 0..8 {
                above[i] = plane[(mb_yc - 1) * uv_stride + mb_xc + i];
            }
        }
        if left_avail {
            for j in 0..8 {
                left[j] = plane[(mb_yc + j) * uv_stride + mb_xc - 1];
            }
        }
        // TL pixel defaults: see matching logic in the B_PRED 4×4 path.
        // When only one neighbour is available, TL takes the *other*
        // neighbour's default (127 for above-default, 129 for left-default).
        let tl = if above_avail && left_avail {
            Some(plane[(mb_yc - 1) * uv_stride + mb_xc - 1])
        } else if above_avail {
            // left unavailable → default left column is 129
            Some(129)
        } else if left_avail {
            // above unavailable → default above row is 127
            Some(127)
        } else {
            None
        };
        // Stack-allocated 8×8 chroma intra-prediction scratch — twice
        // per intra MB (U and V planes). Same hoist rationale as the
        // luma block above.
        let mut pred = [0u8; 8 * 8];
        predict_8x8(
            info.uv_mode,
            if above_avail { Some(&above) } else { None },
            if left_avail { Some(&left) } else { None },
            tl,
            &mut pred,
            8,
        );
        for i in 0..4 {
            let by = i / 2;
            let bx = i % 2;
            let mut deq = [0i16; 16];
            deq[0] = (coeffs[i][0] as i32 * uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (coeffs[i][k] as i32 * uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            let dst_x = mb_xc + bx * 4;
            let dst_y = mb_yc + by * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    plane[(dst_y + r) * uv_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_mb(
    header: &FrameHeader,
    last: &RefFrame,
    golden: &RefFrame,
    altref: &RefFrame,
    info: &MbInfo,
    has_y2: bool,
    y2_coeffs: &[i16; 16],
    y_coeffs: &[[i16; 16]; 16],
    u_coeffs: &[[i16; 16]; 4],
    v_coeffs: &[[i16; 16]; 4],
    mb_x: usize,
    mb_y: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
) {
    let qi = mb_base_qi(header, info.segment_id);
    let y_dc = y_dc_step(qi + header.quant.y_dc_delta);
    let y_ac = y_ac_step(qi);
    let y2_dc = y2_dc_step(qi + header.quant.y2_dc_delta);
    let y2_ac = y2_ac_step(qi + header.quant.y2_ac_delta);
    let uv_dc = uv_dc_step(qi + header.quant.uv_dc_delta);
    let uv_ac = uv_ac_step(qi + header.quant.uv_ac_delta);

    let y2_dc_vals: [i16; 16] = if has_y2 {
        let mut deq = [0i16; 16];
        for i in 0..16 {
            let v = y2_coeffs[i] as i32;
            let q = if i == 0 { y2_dc } else { y2_ac };
            deq[i] = (v * q) as i16;
        }
        iwht4x4(&deq)
    } else {
        [0; 16]
    };

    let ref_frame = match info.ref_frame {
        REF_LAST => last,
        REF_GOLDEN => golden,
        REF_ALT => altref,
        _ => last,
    };

    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;

    // --- Luma prediction via sub-pel 6-tap filter ---
    // Each 4×4 luma sub-block has its own MV. We keep MVs throughout in
    // 1/8-luma-pel units (matching the 8-phase sixtap filter), which is
    // what the stream uses after the §18.1 "motion vectors are all
    // doubled" transform. The encoder side stores and emits MVs the
    // same way.
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = mb_x_px + bx * 4;
        let dst_y = mb_y_px + by * 4;
        let mv = info.sub_mvs[i];
        let ref_x_fp = (dst_x as i32) * 8 + mv.col as i32;
        let ref_y_fp = (dst_y as i32) * 8 + mv.row as i32;
        sixtap_predict(
            &ref_frame.y_plane(),
            ref_x_fp,
            ref_y_fp,
            y_plane,
            y_stride,
            dst_x,
            dst_y,
            4,
            4,
        );
    }

    // --- Chroma prediction via 6-tap filter (profile 0) ---
    // Each 4×4 chroma sub-block covers 2×2 luma sub-blocks. The chroma
    // MV is the average of the 4 luma sub-MVs it covers (RFC 6386 §18.1):
    //
    //   chroma_mv = (sum + 4) / 8   for sum >= 0
    //   chroma_mv = -((-sum + 4) / 8)   for sum < 0
    //
    // Result is already in 1/8-chroma-pel units (no further scaling).
    //
    // Profile 0 (the only profile RFC 6386 standardises) uses the SAME
    // 6-tap filter for chroma as for luma — see libvpx's
    // `vp8_setup_version`: `use_bilinear_mc_filter = 0` for `version == 0`.
    // Profiles 1..3 (libvpx speed/quality variants, out of scope per the
    // corpus README) switch to bilinear; we don't decode those.
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for i in 0..4 {
        let by = i / 2;
        let bx = i % 2;
        // 4 luma subs for this chroma 4×4: (2bx..2bx+2, 2by..2by+2).
        let mut sum_r: i32 = 0;
        let mut sum_c: i32 = 0;
        for r in 0..2 {
            for c in 0..2 {
                let li = (2 * by + r) * 4 + (2 * bx + c);
                sum_r += info.sub_mvs[li].row as i32;
                sum_c += info.sub_mvs[li].col as i32;
            }
        }
        let cmv_r = chroma_avg4(sum_r);
        let cmv_c = chroma_avg4(sum_c);
        let dst_x = mb_xc + bx * 4;
        let dst_y = mb_yc + by * 4;
        let ref_x_fp = (dst_x as i32) * 8 + cmv_c;
        let ref_y_fp = (dst_y as i32) * 8 + cmv_r;
        sixtap_predict(
            &ref_frame.u_plane(),
            ref_x_fp,
            ref_y_fp,
            u_plane,
            uv_stride,
            dst_x,
            dst_y,
            4,
            4,
        );
        sixtap_predict(
            &ref_frame.v_plane(),
            ref_x_fp,
            ref_y_fp,
            v_plane,
            uv_stride,
            dst_x,
            dst_y,
            4,
            4,
        );
    }

    // --- Add residuals ---
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let mut deq = [0i16; 16];
        if has_y2 {
            deq[0] = y2_dc_vals[i];
            for k in 1..16 {
                deq[k] = (y_coeffs[i][k] as i32 * y_ac) as i16;
            }
        } else {
            deq[0] = (y_coeffs[i][0] as i32 * y_dc) as i16;
            for k in 1..16 {
                deq[k] = (y_coeffs[i][k] as i32 * y_ac) as i16;
            }
        }
        let res = idct4x4(&deq);
        let dst_x = mb_x_px + bx * 4;
        let dst_y = mb_y_px + by * 4;
        for r in 0..4 {
            for c in 0..4 {
                let p = y_plane[(dst_y + r) * y_stride + dst_x + c] as i32;
                let rr = res[r * 4 + c] as i32;
                y_plane[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }
    for (coeffs, plane) in [(u_coeffs, &mut *u_plane), (v_coeffs, &mut *v_plane)] {
        for i in 0..4 {
            let by = i / 2;
            let bx = i % 2;
            let mut deq = [0i16; 16];
            deq[0] = (coeffs[i][0] as i32 * uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (coeffs[i][k] as i32 * uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            let dst_x = mb_xc + bx * 4;
            let dst_y = mb_yc + by * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let p = plane[(dst_y + r) * uv_stride + dst_x + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    plane[(dst_y + r) * uv_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Average 4 luma MVs (1/4-luma-pel units) into one chroma MV
/// (1/8-chroma-pel units), per RFC 6386 §18.1:
///
/// ```text
/// chroma_mv = (sum + 4) / 8    if sum >= 0
///           = -((-sum + 4) / 8) if sum < 0
/// ```
///
/// The shift divides by 8 rather than 4 because chroma pixels have
/// twice the diameter of luma pixels (so halving is "baked in"). The
/// asymmetric negative-branch handling matches `(s + 4) >> 3` with
/// explicit sign propagation, avoiding the C-undefined-for-negatives
/// right-shift.
#[inline]
fn chroma_avg4(sum: i32) -> i32 {
    if sum >= 0 {
        (sum + 4) / 8
    } else {
        -((-sum + 4) / 8)
    }
}

/// Compute the per-MB loop-filter level after applying segmentation,
/// reference-frame and mode deltas (RFC 6386 §15.2; libvpx
/// `vp8_loop_filter_frame_init`).
///
/// Steps:
/// 1. base = `header.loop_filter.level`, then if segmentation is
///    active: `base = abs ? seg_lf : clamp(base + seg_lf, 0..=63)`.
/// 2. if `mode_ref_delta_enabled`:
///    - (a) `level += ref_deltas[ref_frame]`
///    - (b) `level += mode_deltas[i]` where `i` is selected from the
///      current `(intra/inter, y_mode)` pair as follows:
///      - intra and `y_mode == B_PRED`         → `i = 0`
///      - inter and `y_mode == ZERO_MV`        → `i = 1`
///      - inter and `y_mode in NEAREST/NEAR/NEW_MV` → `i = 2`
///      - inter and `y_mode == SPLIT_MV`       → `i = 3`
/// 3. clamp 0..=63.
fn per_mb_filter_level(header: &FrameHeader, info: &MbInfo) -> u8 {
    let mut lvl = header.loop_filter.level as i32;
    if header.segmentation.enabled {
        let seg = info.segment_id as usize;
        let delta = header.segmentation.lf[seg];
        if header.segmentation.abs_delta {
            lvl = delta;
        } else {
            lvl = (lvl + delta).clamp(0, 63);
        }
    }
    if header.loop_filter.mode_ref_delta_enabled {
        // Ref delta — REF_INTRA = 0, REF_LAST = 1, REF_GOLDEN = 2, REF_ALT = 3.
        lvl += header.loop_filter.ref_deltas[info.ref_frame as usize];
        // Mode delta — mode index map per libvpx:
        //   0 = INTRA + B_PRED
        //   1 = inter + ZERO_MV
        //   2 = inter + NEAREST/NEAR/NEW_MV
        //   3 = inter + SPLIT_MV
        if info.ref_frame == REF_INTRA {
            if info.y_mode == B_PRED {
                lvl += header.loop_filter.mode_deltas[0];
            }
        } else {
            let mode_idx = match info.y_mode {
                ZERO_MV => 1,
                NEAREST_MV | NEAR_MV | NEW_MV => 2,
                SPLIT_MV => 3,
                _ => 1, // shouldn't happen for inter, fall back to ZERO bucket
            };
            lvl += header.loop_filter.mode_deltas[mode_idx];
        }
    }
    lvl.clamp(0, 63) as u8
}

#[allow(clippy::too_many_arguments)]
/// Apply the in-loop filter, iterating macroblocks in raster order and
/// (per RFC 6386 §15.1) doing the four edge passes per-MB in this order:
///
///   1. Left MB-edge vertical filter (when col > 0)
///   2. Inner sub-block vertical filters at x = mb_x*16 + 4/8/12
///      (skipped when MB has no coefficients AND y_mode is neither
///      B_PRED nor SPLITMV)
///   3. Top MB-edge horizontal filter (when row > 0)
///   4. Inner sub-block horizontal filters (same skip rule)
///
/// At each step, luma is filtered first followed by U and V (the simple
/// filter type only filters luma and only the four pixels closest to
/// the edge — RFC §15.2). Chroma has only one inner sub-block edge per
/// MB (the centre at x = mb_x*8 + 4 / y = mb_y*8 + 4).
///
/// The per-MB filter strength derives from `per_mb_filter_level` —
/// segmentation + mode-ref deltas applied per-MB, per RFC §15.2.
fn apply_loop_filter(
    header: &FrameHeader,
    mb_info: &[MbInfo],
    mb_w: usize,
    mb_h: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    y_buf_h: usize,
    uv_buf_h: usize,
    key_frame: bool,
) {
    if header.loop_filter.level == 0 {
        return;
    }
    let lf = &header.loop_filter;
    let simple = lf.filter_type == 1;
    for mb_y in 0..mb_h {
        let y0 = mb_y * 16;
        let y0c = mb_y * 8;
        for mb_x in 0..mb_w {
            let info = &mb_info[mb_y * mb_w + mb_x];
            // Per-MB level: segmentation + ref/mode deltas. When this
            // resolves to 0 the MB's edges are skipped entirely.
            let mb_level = per_mb_filter_level(header, info);
            if mb_level == 0 {
                continue;
            }
            let params_mb = FilterParams::for_mb_typed(mb_level, lf.sharpness, true, key_frame);
            let params_sb = FilterParams::for_mb_typed(mb_level, lf.sharpness, false, key_frame);
            let filter_subblocks =
                info.has_coeffs || info.y_mode == B_PRED || info.inter_split_mode.is_some();
            let x = mb_x * 16;
            let xc = mb_x * 8;

            // 1. Left MB v-edges — luma then U then V (RFC §15.1; the
            //    simple filter type is luma-only per §15.2). Filters
            //    EXACTLY this MB's 16 luma rows (8 chroma rows).
            if mb_x > 0 {
                if simple {
                    filter_simple_vertical(y_plane, y_stride, x, y_stride, y0, 16, params_mb);
                } else {
                    filter_normal_vertical(y_plane, y_stride, x, y_stride, y0, 16, params_mb, true);
                    filter_normal_vertical(
                        u_plane, uv_stride, xc, uv_stride, y0c, 8, params_mb, true,
                    );
                    filter_normal_vertical(
                        v_plane, uv_stride, xc, uv_stride, y0c, 8, params_mb, true,
                    );
                }
            }

            // 2. Inner sub-block v-edges — three for luma, one for U/V.
            if filter_subblocks {
                if simple {
                    for k in 1..4 {
                        filter_simple_vertical(
                            y_plane,
                            y_stride,
                            x + k * 4,
                            y_stride,
                            y0,
                            16,
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
                            y0,
                            16,
                            params_sb,
                            false,
                        );
                    }
                    filter_normal_vertical(
                        u_plane,
                        uv_stride,
                        xc + 4,
                        uv_stride,
                        y0c,
                        8,
                        params_sb,
                        false,
                    );
                    filter_normal_vertical(
                        v_plane,
                        uv_stride,
                        xc + 4,
                        uv_stride,
                        y0c,
                        8,
                        params_sb,
                        false,
                    );
                }
            }

            // 3. Top MB h-edges. Filters EXACTLY this MB's 16 luma cols
            //    (8 chroma cols).
            if mb_y > 0 {
                if simple {
                    filter_simple_horizontal(y_plane, y_stride, y0, y_buf_h, x, 16, params_mb);
                } else {
                    filter_normal_horizontal(
                        y_plane, y_stride, y0, y_buf_h, x, 16, params_mb, true,
                    );
                    filter_normal_horizontal(
                        u_plane, uv_stride, y0c, uv_buf_h, xc, 8, params_mb, true,
                    );
                    filter_normal_horizontal(
                        v_plane, uv_stride, y0c, uv_buf_h, xc, 8, params_mb, true,
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
                            y_buf_h,
                            x,
                            16,
                            params_sb,
                        );
                    }
                } else {
                    for k in 1..4 {
                        filter_normal_horizontal(
                            y_plane,
                            y_stride,
                            y0 + k * 4,
                            y_buf_h,
                            x,
                            16,
                            params_sb,
                            false,
                        );
                    }
                    filter_normal_horizontal(
                        u_plane,
                        uv_stride,
                        y0c + 4,
                        uv_buf_h,
                        xc,
                        8,
                        params_sb,
                        false,
                    );
                    filter_normal_horizontal(
                        v_plane,
                        uv_stride,
                        y0c + 4,
                        uv_buf_h,
                        xc,
                        8,
                        params_sb,
                        false,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `mv_ref_probs` must follow RFC 6386 §16.3 exactly —
    /// each `probs[i]` is taken from `MV_COUNTS_TO_PROBS[cnt[i]][i]`.
    /// A prior version indexed all four columns by `cnt[0]`, which
    /// caused every inter MB after the first to read mode bits with
    /// the wrong probabilities and cascaded coefficient divergence
    /// through the whole P-frame. See commit log.
    #[test]
    fn mv_ref_probs_indexes_each_row_by_matching_cnt() {
        // Spot-check: cnt = [2, 5, 0, 0] should give
        //   [COUNTS[2][0], COUNTS[5][1], COUNTS[0][2], COUNTS[0][3]]
        //   = [135, 188, 1, 143].
        let got = mv_ref_probs(&[2, 5, 0, 0]);
        assert_eq!(
            got,
            [
                MV_COUNTS_TO_PROBS[2][0],
                MV_COUNTS_TO_PROBS[5][1],
                MV_COUNTS_TO_PROBS[0][2],
                MV_COUNTS_TO_PROBS[0][3],
            ]
        );
        assert_eq!(got, [135, 188, 1, 143]);

        // All-zero cnt → row 0 for each column.
        let z = mv_ref_probs(&[0, 0, 0, 0]);
        assert_eq!(z, [7, 1, 1, 143]);

        // Clamping: cnt[i] > 5 should clamp to row 5.
        let clamped = mv_ref_probs(&[6, 7, 10, 100]);
        assert_eq!(
            clamped,
            [
                MV_COUNTS_TO_PROBS[5][0],
                MV_COUNTS_TO_PROBS[5][1],
                MV_COUNTS_TO_PROBS[5][2],
                MV_COUNTS_TO_PROBS[5][3],
            ]
        );
    }

    /// Regression: MV_COUNTS_TO_PROBS row 3 must match RFC 6386
    /// `mv_counts_to_probs[3] = { 60, 56, 128, 65 }`. Previously the
    /// table had [60, 56, 108, 164], causing MV mode bits to decode
    /// at the wrong probability for certain neighbour configurations.
    #[test]
    fn mv_counts_to_probs_row_3_matches_rfc() {
        assert_eq!(MV_COUNTS_TO_PROBS[3], [60, 56, 128, 65]);
    }

    /// Regression: `MBSPLIT_PROBS` must match RFC 6386
    /// `split_mv_probs = { 110, 111, 150 }`. Previously [110, 111, 165].
    #[test]
    fn mbsplit_probs_match_rfc() {
        use crate::tables::trees::MBSPLIT_PROBS;
        assert_eq!(MBSPLIT_PROBS, [110, 111, 150]);
    }
}
