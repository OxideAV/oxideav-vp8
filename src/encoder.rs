//! VP8 encoder — key-frame + P-frame paths (RFC 6386).
//!
//! Scope:
//! * First frame (and on forced refresh) = key-frame (I-frame).
//! * Subsequent frames = P-frames against REF_LAST, picking the best per-MB
//!   mode among SKIP (no residual), ZERO_MV (motion-compensated copy +
//!   residual at `mv=0`) and NEWMV with a small integer-pel motion
//!   search. Sub-pel search and the NEAREST / NEAR / SPLIT modes remain
//!   planned follow-ups.
//! * DC_PRED for every luma 16×16 MB and chroma 8×8 MB in I-frames.
//! * Fixed quantiser (default `qindex = 50`, mid-quality).
//! * Loop filter disabled (`filter_level = 0`).
//! * Single token partition.
//! * Accepted pixel format: `PixelFormat::Yuv420P`.
//!
//! Not yet implemented (candidate for a follow-up):
//! * Sub-pel refinement after the integer search.
//! * NEAREST / NEAR neighbour-predicted MV modes and SPLIT MVs.
//! * Intra-as-fallback-inside-P.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use crate::bool_encoder::BoolEncoder;
use crate::fdct::{fdct4x4, fwht4x4};
use crate::frame_tag::KEYFRAME_SYNC_CODE;
use crate::intra::{predict_16x16, predict_8x8};
use crate::mv::{encode_mv_component, Mv};
use crate::tables::coeff_probs::{CoeffProbs, DEFAULT_COEF_PROBS};
use crate::tables::mv::DEFAULT_MV_CONTEXT;
use crate::tables::quant::{
    clamp_qindex, uv_ac_step, uv_dc_step, y2_ac_step, y2_dc_step, y_ac_step, y_dc_step,
};
use crate::tables::token_tree::{COEF_BANDS, ZIGZAG};
use crate::tables::trees::{DC_PRED, MV_COUNTS_TO_PROBS};
use crate::transform::{idct4x4, iwht4x4};

/// Default qindex. 50 ≈ mid-quality; the codec accepts 0..=127.
pub const DEFAULT_QINDEX: u8 = 50;

/// SAD-per-pixel threshold below which a P-frame MB is emitted as `skip`.
/// Empirically chosen: 3 is low enough that static content always skips
/// while modestly-changed content falls through to coded residual.
const MB_SKIP_SAD_PER_PIXEL: u32 = 3;

/// Half-range (in integer luma pixels) of the NEWMV motion search window.
/// Search spans [-MOTION_SEARCH_RANGE, +MOTION_SEARCH_RANGE] on each axis.
const MOTION_SEARCH_RANGE: i32 = 8;

/// SAD delta, per MB, that NEWMV must beat ZERO_MV by. Tuned as a coarse
/// proxy for the extra bitrate cost of coding the MV itself — NEWMV is
/// only picked when motion search reduces luma SAD by at least this much
/// versus the zero-motion prediction.
const NEWMV_SAD_MARGIN: u32 = 64;

/// Encoder factory used by [`crate::register_codecs`].
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("vp8 encoder: missing height"))?;
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(Error::invalid(format!(
            "vp8 encoder: dimensions {width}x{height} out of range (1..=16383)"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);

    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        qindex: DEFAULT_QINDEX,
        time_base,
        pending: VecDeque::new(),
        eof: false,
        last_frame: None,
    }))
}

/// Build an encoder with an explicit qindex. Useful for tests and for
/// callers that want finer control than the default quality.
pub fn make_encoder_with_qindex(params: &CodecParameters, qindex: u8) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("vp8 encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }
    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);
    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        qindex: qindex.min(127),
        time_base,
        pending: VecDeque::new(),
        eof: false,
        last_frame: None,
    }))
}

/// Reconstructed reference frame (post-quant reconstruction, matching what
/// the decoder will regenerate from the emitted bitstream). Stored in
/// MB-padded stride/height to make per-MB access a straight index calc.
#[derive(Clone)]
struct ReferenceFrame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    y_h: usize,
    uv_h: usize,
}

struct Vp8Encoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    qindex: u8,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
    last_frame: Option<ReferenceFrame>,
}

impl Encoder for Vp8Encoder {
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
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(format!(
                "vp8 encoder: frame dims {}x{} do not match encoder {}x{}",
                v.width, v.height, self.width, self.height
            )));
        }
        if v.format != PixelFormat::Yuv420P {
            return Err(Error::invalid("vp8 encoder: only Yuv420P input frames"));
        }
        if v.planes.len() < 3 {
            return Err(Error::invalid("vp8 encoder: expected 3 planes"));
        }

        let is_keyframe = self.last_frame.is_none();
        let (data, reference) = if is_keyframe {
            let (bitstream, rec) =
                encode_keyframe_and_reconstruct(self.width, self.height, self.qindex, v)?;
            (bitstream, rec)
        } else {
            let reference = self.last_frame.as_ref().unwrap();
            let (bitstream, rec) =
                encode_pframe_and_reconstruct(self.width, self.height, self.qindex, v, reference)?;
            (bitstream, rec)
        };
        self.last_frame = Some(reference);

        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
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

// ---------------------------------------------------------------------------
// Frame assembly — keyframe
// ---------------------------------------------------------------------------

/// Encode one keyframe. Returns the raw VP8 bitstream for the frame.
/// Backwards-compatible with the earlier single-return signature.
pub fn encode_keyframe(width: u32, height: u32, qindex: u8, frame: &VideoFrame) -> Result<Vec<u8>> {
    let (bitstream, _rec) = encode_keyframe_and_reconstruct(width, height, qindex, frame)?;
    Ok(bitstream)
}

fn encode_keyframe_and_reconstruct(
    width: u32,
    height: u32,
    qindex: u8,
    frame: &VideoFrame,
) -> Result<(Vec<u8>, ReferenceFrame)> {
    let mb_w = ((width + 15) / 16) as usize;
    let mb_h = ((height + 15) / 16) as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;

    // Copy (and MB-pad) the source into our own buffers.
    let (src_y, src_u, src_v) = extract_mb_padded(frame, mb_w, mb_h)?;

    // Allocate reconstruction buffers (they track, pixel-for-pixel, what
    // the decoder will produce — needed for intra prediction context in
    // subsequent MBs).
    let mut rec_y = vec![0u8; y_stride * y_buf_h];
    let mut rec_u = vec![0u8; uv_stride * uv_buf_h];
    let mut rec_v = vec![0u8; uv_stride * uv_buf_h];

    // Pre-compute quant steps.
    let qi = clamp_qindex(qindex as i32);
    let q = QuantCtx {
        y_dc: y_dc_step(qi as i32),
        y_ac: y_ac_step(qi as i32),
        y2_dc: y2_dc_step(qi as i32),
        y2_ac: y2_ac_step(qi as i32),
        uv_dc: uv_dc_step(qi as i32),
        uv_ac: uv_ac_step(qi as i32),
    };

    // --- Compressed header ---
    let mut hdr_enc = BoolEncoder::new();
    // color_space + clamping_type (1 bit each)
    hdr_enc.write_literal(1, 0);
    hdr_enc.write_literal(1, 0);
    // segmentation enabled = 0
    hdr_enc.write_bool(128, false);
    // loop filter: filter_type=0, level=0 (disables LF), sharpness=0,
    //              mode_ref_delta_enabled=0.
    hdr_enc.write_literal(1, 0);
    hdr_enc.write_literal(6, 0);
    hdr_enc.write_literal(3, 0);
    hdr_enc.write_bool(128, false);
    // log2_nb_partitions = 0 (1 partition).
    hdr_enc.write_literal(2, 0);
    // Quant: y_ac_qi + zero deltas (each delta is a 1-bit "present" flag = 0).
    hdr_enc.write_literal(7, qi as u32);
    for _ in 0..5 {
        hdr_enc.write_bool(128, false);
    }
    // refresh_entropy_probs = 0 (we keep defaults).
    hdr_enc.write_bool(128, false);
    // Skip per-prob coefficient probability updates — send "no update" for all.
    emit_no_coef_prob_updates(&mut hdr_enc);
    // mb_skip_enabled = 0 (keyframes have no skip mode in this encoder).
    hdr_enc.write_bool(128, false);

    // --- MB mode info (still boolean-coded into the same first partition) ---
    // All MBs: segment id (not written, seg disabled); skip (not written,
    // skip disabled); y_mode = DC_PRED (KF_YMODE_TREE leaf code 1, probs 145);
    // uv_mode = DC_PRED (KF_UV_MODE_TREE leaf code 0, prob 142).
    let mut mb_encoded: Vec<MbEncoded> = Vec::with_capacity(mb_w * mb_h);
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            // Y mode (DC_PRED).
            hdr_enc.write_bool(145, true);
            hdr_enc.write_bool(156, false);
            hdr_enc.write_bool(163, false);
            // UV mode (DC_PRED).
            hdr_enc.write_bool(142, false);

            let mb_rec = encode_intra_mb_dc(
                &src_y, &src_u, &src_v, &mut rec_y, &mut rec_u, &mut rec_v, y_stride, uv_stride,
                y_buf_h, uv_buf_h, mb_x, mb_y, mb_w, mb_h, &q,
            );
            mb_encoded.push(mb_rec);
        }
    }

    let first_partition = hdr_enc.finish();

    // --- Token partition (separate BoolEncoder) ---
    let tok_enc = emit_tokens(mb_w, mb_h, &mb_encoded, &[], &DEFAULT_COEF_PROBS);
    let token_partition = tok_enc.finish();

    let out = assemble_frame_keyframe(width, height, first_partition, token_partition)?;

    let reference = ReferenceFrame {
        y: rec_y,
        u: rec_u,
        v: rec_v,
        y_stride,
        uv_stride,
        y_h: y_buf_h,
        uv_h: uv_buf_h,
    };
    Ok((out, reference))
}

fn assemble_frame_keyframe(
    width: u32,
    height: u32,
    first_partition: Vec<u8>,
    token_partition: Vec<u8>,
) -> Result<Vec<u8>> {
    let part_size = first_partition.len() as u32;
    if part_size >= (1 << 19) {
        return Err(Error::invalid(format!(
            "vp8 encoder: first partition too large ({} bytes)",
            part_size
        )));
    }
    // frame_type=0 (bit 0), version=0 (bits 1..3 all zero), show_frame=1 (bit 4),
    // first_partition_size in bits 5..23.
    let tag_word: u32 = (1u32 << 4) | (part_size << 5);
    let mut out = Vec::with_capacity(10 + first_partition.len() + token_partition.len());
    out.push((tag_word & 0xff) as u8);
    out.push(((tag_word >> 8) & 0xff) as u8);
    out.push(((tag_word >> 16) & 0xff) as u8);

    // --- Keyframe 7-byte header: sync + w/h words ---
    out.extend_from_slice(&KEYFRAME_SYNC_CODE);
    let w = width as u16 & 0x3fff;
    let h = height as u16 & 0x3fff;
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());

    out.extend_from_slice(&first_partition);
    out.extend_from_slice(&token_partition);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Frame assembly — P-frame (inter-frame, ZERO_MV + SKIP only)
// ---------------------------------------------------------------------------

/// Per-MB decision for a P-frame.
#[derive(Clone, Copy, Debug)]
enum PMbDecision {
    /// Copy the reference MB verbatim — no residual coded.
    Skip,
    /// Motion-compensated copy at `mv=0` followed by a coded residual.
    ZeroMv,
    /// Motion-compensated copy at `mv` (integer-pel) followed by a coded
    /// residual. `mv` is in luma 1/8-pel units and row/col are both
    /// multiples of 8 — the decoder's sub-pel filters degenerate to an
    /// integer copy in that case.
    NewMv(Mv),
}

/// Encode one P-frame (inter-frame). All MBs use REF_LAST + ZERO_MV.
/// Returns the raw VP8 bitstream and a reconstructed reference that
/// matches what the decoder will produce.
fn encode_pframe_and_reconstruct(
    width: u32,
    height: u32,
    qindex: u8,
    frame: &VideoFrame,
    reference: &ReferenceFrame,
) -> Result<(Vec<u8>, ReferenceFrame)> {
    let mb_w = ((width + 15) / 16) as usize;
    let mb_h = ((height + 15) / 16) as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;

    // Sanity: reference must have the same stride/geometry.
    if reference.y_stride != y_stride
        || reference.uv_stride != uv_stride
        || reference.y_h != y_buf_h
        || reference.uv_h != uv_buf_h
    {
        return Err(Error::invalid(
            "vp8 encoder: reference frame geometry mismatch (reset required)",
        ));
    }

    let (src_y, src_u, src_v) = extract_mb_padded(frame, mb_w, mb_h)?;

    // Allocate reconstruction buffers.
    let mut rec_y = vec![0u8; y_stride * y_buf_h];
    let mut rec_u = vec![0u8; uv_stride * uv_buf_h];
    let mut rec_v = vec![0u8; uv_stride * uv_buf_h];

    let qi = clamp_qindex(qindex as i32);
    let q = QuantCtx {
        y_dc: y_dc_step(qi as i32),
        y_ac: y_ac_step(qi as i32),
        y2_dc: y2_dc_step(qi as i32),
        y2_ac: y2_ac_step(qi as i32),
        uv_dc: uv_dc_step(qi as i32),
        uv_ac: uv_ac_step(qi as i32),
    };

    // --- First-partition: inter-frame header + MB mode info ---
    let mut hdr_enc = BoolEncoder::new();
    // Inter-header order (matching parse_inter_header exactly):
    //   segmentation enabled=0
    hdr_enc.write_bool(128, false);
    //   loop filter
    hdr_enc.write_literal(1, 0); // filter_type
    hdr_enc.write_literal(6, 0); // level=0 (disables LF)
    hdr_enc.write_literal(3, 0); // sharpness
    hdr_enc.write_bool(128, false); // mode_ref_delta_enabled
                                    //   log2_nb_partitions = 0 (single partition)
    hdr_enc.write_literal(2, 0);
    //   quant: y_ac_qi + 5 "delta present" flags all 0.
    hdr_enc.write_literal(7, qi as u32);
    for _ in 0..5 {
        hdr_enc.write_bool(128, false);
    }
    //   refresh_alt = 1 (refresh all references with the new frame)
    hdr_enc.write_bool(128, true);
    //   refresh_golden = 1
    hdr_enc.write_bool(128, true);
    //   sign_bias_golden, sign_bias_alt
    hdr_enc.write_bool(128, false);
    hdr_enc.write_bool(128, false);
    //   refresh_entropy_probs = 0
    hdr_enc.write_bool(128, false);
    //   refresh_last = 1
    hdr_enc.write_bool(128, true);
    //   coef prob updates — all "no update"
    emit_no_coef_prob_updates(&mut hdr_enc);
    //   mb_skip_enabled = 1, skip prob literal (we use 128 — neutral).
    let mb_skip_prob: u8 = 128;
    hdr_enc.write_bool(128, true);
    hdr_enc.write_literal(8, mb_skip_prob as u32);
    //   prob_intra, prob_last, prob_gf — all literals (8 bits each).
    // We emit inter=true for every MB and REF_LAST for every MB.
    //   read_bool(prob_intra) == true  → inter (what we want). Pick prob_intra=1
    //     so `write_bool(1, true)` is near-free.
    //   read_bool(prob_last)  == false → REF_LAST (what we want). Pick prob_last=1
    //     so `write_bool(1, false)` is near-free.
    //   prob_gf — never read since we never take the "not last" branch.
    let prob_intra: u8 = 1;
    let prob_last: u8 = 1;
    let prob_gf: u8 = 128;
    hdr_enc.write_literal(8, prob_intra as u32);
    hdr_enc.write_literal(8, prob_last as u32);
    hdr_enc.write_literal(8, prob_gf as u32);
    //   y-mode prob update flag = 0 ; uv-mode prob update flag = 0
    hdr_enc.write_bool(128, false);
    hdr_enc.write_bool(128, false);
    //   MV context updates — 19 × 2, all "no update".
    use crate::tables::mv::MV_UPDATE_PROBS;
    for comp in 0..2 {
        for i in 0..19 {
            hdr_enc.write_bool(MV_UPDATE_PROBS[comp][i] as u32, false);
        }
    }

    // --- Per-MB decision + reconstruction ---
    let mut mb_encoded: Vec<MbEncoded> = Vec::with_capacity(mb_w * mb_h);
    let mut mb_decisions: Vec<PMbDecision> = Vec::with_capacity(mb_w * mb_h);
    let mut mb_mvs: Vec<Mv> = vec![Mv::ZERO; mb_w * mb_h];
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            // Zero-MV SAD — the baseline against which NEWMV must improve.
            let zero_sad = mb_luma_sad_at(&src_y, &reference.y, y_stride, mb_x, mb_y, 0, 0);

            // Decision 1: cheap skip test.
            let skip = zero_sad <= MB_SKIP_SAD_PER_PIXEL * (16 * 16);

            // Decision 2: if not skipping, try a small integer-pel search.
            let (decision, chosen_mv) = if skip {
                (PMbDecision::Skip, Mv::ZERO)
            } else {
                let (best_mv_px, best_sad) = integer_motion_search(
                    &src_y,
                    &reference.y,
                    y_stride,
                    mb_x,
                    mb_y,
                    mb_w,
                    mb_h,
                    MOTION_SEARCH_RANGE,
                );
                if best_sad + NEWMV_SAD_MARGIN < zero_sad && best_mv_px != (0, 0) {
                    let mv = Mv::new(best_mv_px.0 * 8, best_mv_px.1 * 8);
                    (PMbDecision::NewMv(mv), mv)
                } else {
                    (PMbDecision::ZeroMv, Mv::ZERO)
                }
            };
            mb_decisions.push(decision);
            mb_mvs[mb_y * mb_w + mb_x] = chosen_mv;

            // Replicate the decoder's find_near_mvs for this MB against
            // already-emitted neighbour MVs so that the probabilities +
            // best-MV delta match.
            let (_nearest, _near, best_for_newmv, cnt) =
                find_near_mvs_enc(&mb_mvs, &mb_decisions, mb_x, mb_y, mb_w);
            let ref_probs = mv_ref_probs_enc(&cnt);

            // Mode-info bits.
            // 1) segment id (skipped — seg disabled).
            // 2) skip flag (mb_skip_enabled=1 so this is coded).
            hdr_enc.write_bool(mb_skip_prob as u32, matches!(decision, PMbDecision::Skip));
            // 3) is_inter = true.
            hdr_enc.write_bool(prob_intra as u32, true);
            // 4) ref_frame bits: prob_last "is not last" = false (→ REF_LAST).
            hdr_enc.write_bool(prob_last as u32, false);

            // 5) MV_REF_TREE leaves (RFC §16.3):
            //      leaf 0 = ZERO_MV  (tree path: 0)
            //      leaf 3 = NEW_MV   (tree path: 1, 1, 1)
            //    Skip & ZeroMv both take the ZERO_MV leaf (no MV coded).
            match decision {
                PMbDecision::Skip | PMbDecision::ZeroMv => {
                    hdr_enc.write_bool(ref_probs[0] as u32, false);
                }
                PMbDecision::NewMv(mv) => {
                    // MV_REF_TREE walk to NEW_MV (leaf 3): probs[0]=true,
                    // probs[1]=true, probs[2]=true, probs[3]=false.
                    hdr_enc.write_bool(ref_probs[0] as u32, true);
                    hdr_enc.write_bool(ref_probs[1] as u32, true);
                    hdr_enc.write_bool(ref_probs[2] as u32, true);
                    hdr_enc.write_bool(ref_probs[3] as u32, false);
                    let dmv = Mv::new(
                        mv.row as i32 - best_for_newmv.row as i32,
                        mv.col as i32 - best_for_newmv.col as i32,
                    );
                    encode_mv_component(&mut hdr_enc, &DEFAULT_MV_CONTEXT[0], dmv.row as i32);
                    encode_mv_component(&mut hdr_enc, &DEFAULT_MV_CONTEXT[1], dmv.col as i32);
                }
            }

            // Per-MB encode: compute residual between src and the
            // motion-compensated prediction.
            let mb_rec = match decision {
                PMbDecision::Skip => {
                    copy_ref_into_rec(
                        &reference.y,
                        &reference.u,
                        &reference.v,
                        &mut rec_y,
                        &mut rec_u,
                        &mut rec_v,
                        y_stride,
                        uv_stride,
                        mb_x,
                        mb_y,
                    );
                    MbEncoded::zero()
                }
                PMbDecision::ZeroMv => encode_inter_mb_at_mv(
                    &src_y,
                    &src_u,
                    &src_v,
                    &reference.y,
                    &reference.u,
                    &reference.v,
                    &mut rec_y,
                    &mut rec_u,
                    &mut rec_v,
                    y_stride,
                    uv_stride,
                    y_buf_h,
                    uv_buf_h,
                    mb_x,
                    mb_y,
                    Mv::ZERO,
                    &q,
                ),
                PMbDecision::NewMv(mv) => encode_inter_mb_at_mv(
                    &src_y,
                    &src_u,
                    &src_v,
                    &reference.y,
                    &reference.u,
                    &reference.v,
                    &mut rec_y,
                    &mut rec_u,
                    &mut rec_v,
                    y_stride,
                    uv_stride,
                    y_buf_h,
                    uv_buf_h,
                    mb_x,
                    mb_y,
                    mv,
                    &q,
                ),
            };
            mb_encoded.push(mb_rec);
        }
    }

    let first_partition = hdr_enc.finish();

    // --- Token partition ---
    let tok_enc = emit_tokens(mb_w, mb_h, &mb_encoded, &mb_decisions, &DEFAULT_COEF_PROBS);
    let token_partition = tok_enc.finish();

    let part_size = first_partition.len() as u32;
    if part_size >= (1 << 19) {
        return Err(Error::invalid(format!(
            "vp8 encoder: first partition too large ({} bytes)",
            part_size
        )));
    }
    // frame_type=1 (P), version=0, show_frame=1.
    let tag_word: u32 = 1u32 | (1u32 << 4) | (part_size << 5);
    let mut out = Vec::with_capacity(3 + first_partition.len() + token_partition.len());
    out.push((tag_word & 0xff) as u8);
    out.push(((tag_word >> 8) & 0xff) as u8);
    out.push(((tag_word >> 16) & 0xff) as u8);
    out.extend_from_slice(&first_partition);
    out.extend_from_slice(&token_partition);

    let reference_out = ReferenceFrame {
        y: rec_y,
        u: rec_u,
        v: rec_v,
        y_stride,
        uv_stride,
        y_h: y_buf_h,
        uv_h: uv_buf_h,
    };
    Ok((out, reference_out))
}

/// SAD of the luma MB at `(mb_x, mb_y)` against the reference plane shifted
/// by `(dy, dx)` integer luma pixels. Out-of-bounds reference samples are
/// clamped (edge replication) to match the decoder's `RefPlane::sample`.
fn mb_luma_sad_at(
    src_y: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    dy: i32,
    dx: i32,
) -> u32 {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    let w = stride as i32;
    let h = (ref_y.len() / stride) as i32;
    let mut sad: u32 = 0;
    for r in 0..16 {
        let ry = (y0 as i32 + r as i32 + dy).clamp(0, h - 1) as usize;
        let row_off_src = (y0 + r) * stride;
        for c in 0..16 {
            let rx = (x0 as i32 + c as i32 + dx).clamp(0, w - 1) as usize;
            let d = src_y[row_off_src + x0 + c] as i32 - ref_y[ry * stride + rx] as i32;
            sad += d.unsigned_abs();
        }
    }
    sad
}

/// Simple full-pel motion search. Scans `[-range, +range]` on each axis
/// around zero, returning `((dy, dx), best_sad)` for the displacement that
/// minimises luma SAD. Ties break toward smaller magnitudes (the search
/// starts at zero and only switches on strict improvement).
#[allow(clippy::too_many_arguments)]
fn integer_motion_search(
    src_y: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    _mb_w: usize,
    _mb_h: usize,
    range: i32,
) -> ((i32, i32), u32) {
    let mut best = (0, 0);
    let mut best_sad = mb_luma_sad_at(src_y, ref_y, stride, mb_x, mb_y, 0, 0);
    for dy in -range..=range {
        for dx in -range..=range {
            if dy == 0 && dx == 0 {
                continue;
            }
            let sad = mb_luma_sad_at(src_y, ref_y, stride, mb_x, mb_y, dy, dx);
            if sad < best_sad {
                best_sad = sad;
                best = (dy, dx);
            }
        }
    }
    (best, best_sad)
}

/// Encode-time replica of `find_near_mvs` in the decoder, restricted to
/// the REF_LAST-only encoder. Returns `(nearest, near, best, cnt)`.
fn find_near_mvs_enc(
    mb_mvs: &[Mv],
    mb_decisions: &[PMbDecision],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
) -> (Mv, Mv, Mv, [u8; 4]) {
    let mut cnt = [0u8; 4];
    let mut mvs: [Mv; 3] = [Mv::ZERO; 3];
    let mut num_mvs = 0usize;
    let neighbours: [(isize, isize, u8); 3] = [(0, -1, 2), (-1, 0, 2), (-1, -1, 1)];
    for &(dx, dy, weight) in &neighbours {
        let nx = mb_x as isize + dx;
        let ny = mb_y as isize + dy;
        if nx < 0 || ny < 0 || (nx as usize) >= mb_w {
            cnt[0] += weight;
            continue;
        }
        let idx = (ny as usize) * mb_w + (nx as usize);
        let nmv = match mb_decisions.get(idx) {
            Some(PMbDecision::Skip) | Some(PMbDecision::ZeroMv) => Mv::ZERO,
            Some(PMbDecision::NewMv(_)) => mb_mvs[idx],
            None => Mv::ZERO,
        };
        if nmv.row == 0 && nmv.col == 0 {
            cnt[0] += weight;
        } else {
            let mut matched = false;
            for i in 0..num_mvs {
                if mvs[i] == nmv {
                    cnt[i + 1] += weight;
                    matched = true;
                    break;
                }
            }
            if !matched && num_mvs < 2 {
                mvs[num_mvs] = nmv;
                cnt[num_mvs + 1] = weight;
                num_mvs += 1;
            }
        }
    }
    let nearest = mvs[0];
    let near = mvs[1];
    let best = if cnt[1] >= cnt[0] { nearest } else { Mv::ZERO };
    (nearest, near, best, cnt)
}

/// Encode-time replica of the decoder's `mv_ref_probs` — selects a row of
/// `MV_COUNTS_TO_PROBS` by `cnt[0]`.
fn mv_ref_probs_enc(cnt: &[u8; 4]) -> [u8; 4] {
    let row = (cnt[0].min(5)) as usize;
    MV_COUNTS_TO_PROBS[row]
}

/// Encode-time replica of the decoder's `chroma_round`. Converts the sum
/// of 4 luma sub-MV components (in 1/8-pel units) into the chroma MV
/// component that the decoder will apply to the chroma plane.
#[inline]
fn chroma_round_enc(sum: i32) -> i32 {
    let sign = if sum < 0 { -1 } else { 1 };
    ((sum + sign * 4) / 8) * 2
}

#[allow(clippy::too_many_arguments)]
fn copy_ref_into_rec(
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    mb_x: usize,
    mb_y: usize,
) {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    for r in 0..16 {
        let off = (y0 + r) * y_stride + x0;
        rec_y[off..off + 16].copy_from_slice(&ref_y[off..off + 16]);
    }
    let xc = mb_x * 8;
    let yc = mb_y * 8;
    for r in 0..8 {
        let off = (yc + r) * uv_stride + xc;
        rec_u[off..off + 8].copy_from_slice(&ref_u[off..off + 8]);
        rec_v[off..off + 8].copy_from_slice(&ref_v[off..off + 8]);
    }
}

// ---------------------------------------------------------------------------
// Inter MB encode (integer-pel MV: integer copy from reference + coded residual)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn encode_inter_mb_at_mv(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    ref_y: &[u8],
    ref_u: &[u8],
    ref_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    y_buf_h: usize,
    uv_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    mv: Mv,
    q: &QuantCtx,
) -> MbEncoded {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    // Integer-pel luma displacement in pixels (MVs come in 1/8-pel units
    // with multiples of 8).
    let dy = (mv.row as i32) >> 3;
    let dx = (mv.col as i32) >> 3;
    let ybh = y_buf_h as i32;
    let ybw = y_stride as i32;

    // Luma prediction = reference shifted by (dy, dx), edge-replicated to
    // match the decoder's `sixtap_predict` when called with integer MVs.
    let mut pred_y = [0u8; 256];
    for r in 0..16 {
        for c in 0..16 {
            let ry = ((mb_yp as i32) + r as i32 + dy).clamp(0, ybh - 1) as usize;
            let rx = ((mb_xp as i32) + c as i32 + dx).clamp(0, ybw - 1) as usize;
            pred_y[r * 16 + c] = ref_y[ry * y_stride + rx];
        }
    }

    // 4×4 DCTs of the luma residual. For inter MBs the Y2 (WHT) path is
    // used exactly as in intra 16×16 DC_PRED.
    let mut raw_dc_y = [0i32; 16];
    let mut raw_ac_y = [[0i32; 16]; 16];
    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut blk = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let src = src_y[(mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c] as i32;
                let p = pred_y[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                blk[r * 4 + c] = src - p;
            }
        }
        let coeffs = fdct4x4(&blk);
        raw_dc_y[bi] = coeffs[0];
        raw_ac_y[bi] = coeffs;
    }

    let y2_raw = fwht4x4(&raw_dc_y);
    let mut y2_q = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_q[i] = quant(y2_raw[i], step);
    }
    let mut y2_deq = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_deq[i] = (y2_q[i] as i32 * step) as i16;
    }
    let rec_dc = iwht4x4(&y2_deq);

    let mut y_q = [[0i16; 16]; 16];
    for bi in 0..16 {
        for k in 1..16 {
            y_q[bi][k] = quant(raw_ac_y[bi][k], q.y_ac);
        }
        y_q[bi][0] = 0;
    }

    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut deq = [0i16; 16];
        deq[0] = rec_dc[bi];
        for k in 1..16 {
            deq[k] = (y_q[bi][k] as i32 * q.y_ac) as i16;
        }
        let res = idct4x4(&deq);
        for r in 0..4 {
            for c in 0..4 {
                let p = pred_y[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                let rr = res[r * 4 + c] as i32;
                let dst_y_idx = (mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c;
                rec_y[dst_y_idx] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }

    // Chroma — same pipeline, integer-copy reference prediction.
    // The decoder derives a per-4×4 chroma MV as the average of the 4
    // covered luma sub-MVs (`chroma_round`). For a non-SPLIT MB every
    // sub-MV equals `mv`, so `sum = 4*mv` and `chroma_round(4*mv)` gives
    // the applied chroma displacement in 1/8-chroma-pel units. Integer
    // luma MVs (multiples of 8) always produce integer chroma MVs
    // (multiples of 8) under this rule, so the bilinear filter degenerates
    // to a straight copy in the reference buffer.
    let cmv_r = chroma_round_enc(4 * mv.row as i32);
    let cmv_c = chroma_round_enc(4 * mv.col as i32);
    debug_assert!(
        (cmv_r & 7) == 0 && (cmv_c & 7) == 0,
        "integer-pel luma MV must yield integer chroma MV"
    );
    let cdy_base = cmv_r >> 3;
    let cdx_base = cmv_c >> 3;
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    let uvbw = uv_stride as i32;
    let uvbh = uv_buf_h as i32;
    let mut u_q = [[0i16; 16]; 4];
    let mut v_q = [[0i16; 16]; 4];
    for plane_sel in 0..2 {
        let (src, refp, rec, q_coeffs) = match plane_sel {
            0 => (src_u, ref_u, &mut *rec_u, &mut u_q),
            _ => (src_v, ref_v, &mut *rec_v, &mut v_q),
        };
        let mut pred_uv = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                let ry = ((mb_yc as i32) + r as i32 + cdy_base).clamp(0, uvbh - 1) as usize;
                let rx = ((mb_xc as i32) + c as i32 + cdx_base).clamp(0, uvbw - 1) as usize;
                pred_uv[r * 8 + c] = refp[ry * uv_stride + rx];
            }
        }
        for bi in 0..4 {
            let by = bi / 2;
            let bx = bi % 2;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let sidx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    let s = src[sidx] as i32;
                    let p = pred_uv[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = s - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.uv_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.uv_ac);
            }
            q_coeffs[bi] = blk_q;
            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let pidx = (by * 4 + r) * 8 + bx * 4 + c;
                    let p = pred_uv[pidx] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let didx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    rec[didx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    MbEncoded {
        y2_coeffs: y2_q,
        y_coeffs: y_q,
        u_coeffs: u_q,
        v_coeffs: v_q,
    }
}

// ---------------------------------------------------------------------------
// Macroblock encode (intra DC_PRED — used by key-frames)
// ---------------------------------------------------------------------------

struct QuantCtx {
    /// Not used by the encoder directly (the decoder ignores Y-block DC
    /// for non-B_PRED MBs and uses the Y2-derived DC instead) — kept for
    /// documentation / future use by a B_PRED path.
    #[allow(dead_code)]
    y_dc: i32,
    y_ac: i32,
    y2_dc: i32,
    y2_ac: i32,
    uv_dc: i32,
    uv_ac: i32,
}

/// Output of per-MB encode: quantised coefficients for each block and the
/// Y2 block (the 16 DC coefficients passed through forward WHT).
struct MbEncoded {
    y2_coeffs: [i16; 16],
    y_coeffs: [[i16; 16]; 16],
    u_coeffs: [[i16; 16]; 4],
    v_coeffs: [[i16; 16]; 4],
}

impl MbEncoded {
    fn zero() -> Self {
        Self {
            y2_coeffs: [0; 16],
            y_coeffs: [[0; 16]; 16],
            u_coeffs: [[0; 16]; 4],
            v_coeffs: [[0; 16]; 4],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_intra_mb_dc(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    _y_buf_h: usize,
    _uv_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    _mb_w: usize,
    _mb_h: usize,
    q: &QuantCtx,
) -> MbEncoded {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    // Gather DC_PRED neighbours for the 16x16 luma prediction.
    let mut above_arr = [0u8; 16];
    let mut left_arr = [0u8; 16];
    let above_avail = mb_yp > 0;
    let left_avail = mb_xp > 0;
    if above_avail {
        for i in 0..16 {
            above_arr[i] = rec_y[(mb_yp - 1) * y_stride + mb_xp + i];
        }
    }
    if left_avail {
        for j in 0..16 {
            left_arr[j] = rec_y[(mb_yp + j) * y_stride + mb_xp - 1];
        }
    }
    let tl = if above_avail && left_avail {
        Some(rec_y[(mb_yp - 1) * y_stride + mb_xp - 1])
    } else if above_avail {
        Some(127)
    } else if left_avail {
        Some(129)
    } else {
        None
    };
    let mut pred = vec![0u8; 16 * 16];
    predict_16x16(
        DC_PRED,
        if above_avail { Some(&above_arr) } else { None },
        if left_avail { Some(&left_arr) } else { None },
        tl,
        &mut pred,
        16,
    );

    let mut raw_dc_y = [0i32; 16];
    let mut raw_ac_y = [[0i32; 16]; 16];
    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut blk = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let src = src_y[(mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c] as i32;
                let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                blk[r * 4 + c] = src - p;
            }
        }
        let coeffs = fdct4x4(&blk);
        raw_dc_y[bi] = coeffs[0];
        raw_ac_y[bi] = coeffs;
    }

    // Forward WHT on the 16 DC values.
    let y2_raw = fwht4x4(&raw_dc_y);
    // Quantise Y2 (DC step = y2_dc, AC step = y2_ac).
    let mut y2_q = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_q[i] = quant(y2_raw[i], step);
    }
    // Dequantise + inverse WHT → reconstructed DCs.
    let mut y2_deq = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_deq[i] = (y2_q[i] as i32 * step) as i16;
    }
    let rec_dc = iwht4x4(&y2_deq);

    let mut y_q = [[0i16; 16]; 16];
    for bi in 0..16 {
        for k in 1..16 {
            y_q[bi][k] = quant(raw_ac_y[bi][k], q.y_ac);
        }
        y_q[bi][0] = 0;
    }

    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut deq = [0i16; 16];
        deq[0] = rec_dc[bi];
        for k in 1..16 {
            deq[k] = (y_q[bi][k] as i32 * q.y_ac) as i16;
        }
        let res = idct4x4(&deq);
        for r in 0..4 {
            for c in 0..4 {
                let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                let rr = res[r * 4 + c] as i32;
                let dst_y_idx = (mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c;
                rec_y[dst_y_idx] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }

    // --- Chroma (8x8 DC_PRED) ---
    let mut u_q = [[0i16; 16]; 4];
    let mut v_q = [[0i16; 16]; 4];
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for plane_sel in 0..2 {
        let (src, rec, q_coeffs) = match plane_sel {
            0 => (src_u, &mut *rec_u, &mut u_q),
            _ => (src_v, &mut *rec_v, &mut v_q),
        };
        let above_avail_c = mb_yc > 0;
        let left_avail_c = mb_xc > 0;
        let mut above = [0u8; 8];
        let mut left = [0u8; 8];
        if above_avail_c {
            for i in 0..8 {
                above[i] = rec[(mb_yc - 1) * uv_stride + mb_xc + i];
            }
        }
        if left_avail_c {
            for j in 0..8 {
                left[j] = rec[(mb_yc + j) * uv_stride + mb_xc - 1];
            }
        }
        let tl = if above_avail_c && left_avail_c {
            Some(rec[(mb_yc - 1) * uv_stride + mb_xc - 1])
        } else if above_avail_c {
            Some(127)
        } else if left_avail_c {
            Some(129)
        } else {
            None
        };
        let mut pred_uv = vec![0u8; 8 * 8];
        predict_8x8(
            DC_PRED,
            if above_avail_c { Some(&above) } else { None },
            if left_avail_c { Some(&left) } else { None },
            tl,
            &mut pred_uv,
            8,
        );
        for bi in 0..4 {
            let by = bi / 2;
            let bx = bi % 2;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let sidx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    let s = src[sidx] as i32;
                    let p = pred_uv[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = s - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.uv_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.uv_ac);
            }
            q_coeffs[bi] = blk_q;
            // Reconstruct.
            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let pidx = (by * 4 + r) * 8 + bx * 4 + c;
                    let p = pred_uv[pidx] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let didx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    rec[didx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    MbEncoded {
        y2_coeffs: y2_q,
        y_coeffs: y_q,
        u_coeffs: u_q,
        v_coeffs: v_q,
    }
}

// ---------------------------------------------------------------------------
// Token partition encoding shared between I and P frames.
// ---------------------------------------------------------------------------

/// Emit the token partition for every MB in raster scan. If `decisions`
/// is empty, every MB contributes its tokens (I-frame path). If
/// `decisions` is populated, SKIP MBs contribute no tokens and still
/// reset their neighbour non-zero contexts to 0.
fn emit_tokens(
    mb_w: usize,
    mb_h: usize,
    mb_encoded: &[MbEncoded],
    decisions: &[PMbDecision],
    coef_probs: &CoeffProbs,
) -> BoolEncoder {
    let mut tok_enc = BoolEncoder::new();
    let mut nz_y_above = vec![[0u8; 4]; mb_w];
    let mut nz_uv_above = vec![[0u8; 2]; mb_w];
    let mut nz_v_above = vec![[0u8; 2]; mb_w];
    let mut nz_y2_above = vec![0u8; mb_w];

    for mb_y in 0..mb_h {
        let mut nz_y_left = [0u8; 4];
        let mut nz_u_left = [0u8; 2];
        let mut nz_v_left = [0u8; 2];
        let mut nz_y2_left = 0u8;
        for mb_x in 0..mb_w {
            let mb_rec = &mb_encoded[mb_y * mb_w + mb_x];
            let is_skip = decisions
                .get(mb_y * mb_w + mb_x)
                .is_some_and(|d| matches!(d, PMbDecision::Skip));
            if is_skip {
                // Decoder: skip clears all neighbour nz for this MB.
                nz_y2_above[mb_x] = 0;
                nz_y2_left = 0;
                for bx in 0..4 {
                    nz_y_above[mb_x][bx] = 0;
                    nz_y_left[bx] = 0;
                }
                for bx in 0..2 {
                    nz_uv_above[mb_x][bx] = 0;
                    nz_u_left[bx] = 0;
                    nz_v_above[mb_x][bx] = 0;
                    nz_v_left[bx] = 0;
                }
                continue;
            }

            // Y2 DC block.
            let nctx = nz_y2_above[mb_x] + nz_y2_left;
            let nz = encode_block(
                &mut tok_enc,
                coef_probs,
                /*plane=*/ 1,
                nctx as usize,
                &mb_rec.y2_coeffs,
                0,
            );
            let nzf = if nz > 0 { 1 } else { 0 };
            nz_y2_above[mb_x] = nzf;
            nz_y2_left = nzf;

            for by in 0..4 {
                for bx in 0..4 {
                    let idx = by * 4 + bx;
                    let nctx = nz_y_above[mb_x][bx] + nz_y_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        0,
                        nctx as usize,
                        &mb_rec.y_coeffs[idx],
                        1,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_y_above[mb_x][bx] = nzf;
                    nz_y_left[by] = nzf;
                }
            }
            for by in 0..2 {
                for bx in 0..2 {
                    let idx = by * 2 + bx;
                    let nctx = nz_uv_above[mb_x][bx] + nz_u_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        2,
                        nctx as usize,
                        &mb_rec.u_coeffs[idx],
                        0,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_uv_above[mb_x][bx] = nzf;
                    nz_u_left[by] = nzf;
                    let nctx = nz_v_above[mb_x][bx] + nz_v_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        2,
                        nctx as usize,
                        &mb_rec.v_coeffs[idx],
                        0,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_v_above[mb_x][bx] = nzf;
                    nz_v_left[by] = nzf;
                }
            }
        }
    }
    tok_enc
}

/// Quantise a single coefficient using `step`. Uses symmetric rounding
/// towards zero with `step/2` bias — close to what libvpx's reference
/// encoder does for intra blocks and adequate for our decoder's
/// multiply-by-step dequantiser.
#[inline]
fn quant(v: i32, step: i32) -> i16 {
    if step <= 0 {
        return 0;
    }
    let half = step / 2;
    let q = if v >= 0 {
        (v + half) / step
    } else {
        -((-v + half) / step)
    };
    q.clamp(-2048, 2047) as i16
}

// ---------------------------------------------------------------------------
// Token (coefficient) entropy encode
// ---------------------------------------------------------------------------

/// Encode one transform block's 16 coefficients into the boolean coder.
/// Returns the number of coefficients encoded (last-non-zero + 1), or 0
/// if the entire block is zero starting from `start`.
///
/// Mirrors [`crate::tokens::decode_block`] bit-for-bit — the decoder's
/// flat `p[0..10]` look-up table is the authoritative tree walk, so the
/// write side uses the exact same branch structure.
fn encode_block(
    enc: &mut BoolEncoder,
    probs: &CoeffProbs,
    plane: usize,
    nctx: usize,
    coeffs: &[i16; 16],
    start: usize,
) -> u8 {
    let plane_probs = &probs[plane];
    let mut last_nz = None::<usize>;
    for n in start..16 {
        let c = coeffs[ZIGZAG[n]];
        if c != 0 {
            last_nz = Some(n);
        }
    }
    let last = match last_nz {
        Some(n) => n,
        None => {
            let p = &plane_probs[COEF_BANDS[start]][nctx];
            enc.write_bool(p[0] as u32, false);
            return 0;
        }
    };

    let mut n = start;
    let mut ctx = nctx;
    let mut p = &plane_probs[COEF_BANDS[n]][ctx];
    enc.write_bool(p[0] as u32, true);

    loop {
        while coeffs[ZIGZAG[n]] == 0 {
            enc.write_bool(p[1] as u32, false);
            n += 1;
            p = &plane_probs[COEF_BANDS[n]][0];
        }
        enc.write_bool(p[1] as u32, true);

        let raw = coeffs[ZIGZAG[n]] as i32;
        let v = raw.unsigned_abs() as i32;
        emit_magnitude(enc, p, v);
        ctx = if v == 1 { 1 } else { 2 };

        enc.write_bool(128, raw < 0);

        n += 1;
        if n == 16 {
            return 16;
        }
        p = &plane_probs[COEF_BANDS[n]][ctx];
        if n > last {
            enc.write_bool(p[0] as u32, false);
            return (last + 1) as u8;
        }
        enc.write_bool(p[0] as u32, true);
    }
}

/// Write the magnitude of a non-zero coefficient following the coef-tree
/// branch structure in [`crate::tokens::decode_block`]. `p` is the
/// 11-element probability array for the current (band, ctx).
fn emit_magnitude(enc: &mut BoolEncoder, p: &[u8; 11], v: i32) {
    // Match the decoder's branch-by-branch ladder.
    if v == 1 {
        enc.write_bool(p[2] as u32, false);
        return;
    }
    enc.write_bool(p[2] as u32, true);
    if v <= 4 {
        enc.write_bool(p[3] as u32, false);
        if v == 2 {
            enc.write_bool(p[4] as u32, false);
        } else {
            enc.write_bool(p[4] as u32, true);
            enc.write_bool(p[5] as u32, v == 4);
        }
        return;
    }
    enc.write_bool(p[3] as u32, true);
    if v <= 10 {
        enc.write_bool(p[6] as u32, false);
        if v <= 6 {
            enc.write_bool(p[7] as u32, false);
            enc.write_bool(159, v == 6);
        } else {
            enc.write_bool(p[7] as u32, true);
            let hi = if v >= 9 { 1 } else { 0 };
            enc.write_bool(165, hi == 1);
            let low = (v - 7 - 2 * hi) as u32;
            enc.write_bool(145, low == 1);
        }
        return;
    }
    enc.write_bool(p[6] as u32, true);
    let (cat, base) = if v < 19 {
        (0, 11)
    } else if v < 35 {
        (1, 19)
    } else if v < 67 {
        (2, 35)
    } else {
        (3, 67)
    };
    let bit1 = (cat >> 1) & 1;
    let bit0 = cat & 1;
    enc.write_bool(p[8] as u32, bit1 == 1);
    enc.write_bool(p[9 + bit1] as u32, bit0 == 1);
    let extra_bits_tab: &[u8] = match cat {
        0 => &[173, 148, 140],
        1 => &[176, 155, 140, 135],
        2 => &[180, 157, 141, 134, 130],
        _ => &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129],
    };
    let extra = (v - base) as u32;
    let nbits = extra_bits_tab.len();
    for i in 0..nbits {
        let bit = ((extra >> (nbits - 1 - i)) & 1) as u8;
        enc.write_bool(extra_bits_tab[i] as u32, bit != 0);
    }
}

/// Emit "no probability updates" for the whole 4×8×3×11 coefficient prob
/// table. Mirrors the decoder's `update_coef_probs` but always sending
/// `read_bool(upd)=false`.
fn emit_no_coef_prob_updates(enc: &mut BoolEncoder) {
    use crate::tables::coeff_probs::COEF_UPDATE_PROBS;
    use crate::tables::token_tree::{NUM_BANDS, NUM_CTX, NUM_PROBS, NUM_TYPES};
    for i in 0..NUM_TYPES {
        for j in 0..NUM_BANDS {
            for k in 0..NUM_CTX {
                for l in 0..NUM_PROBS {
                    let upd = COEF_UPDATE_PROBS[i][j][k][l] as u32;
                    enc.write_bool(upd, false);
                }
            }
        }
    }
}

/// Copy the 3 planes of a video frame into MB-aligned (16/8 pixel) buffers.
/// Edge-replicate when frame dimensions are not multiples of 16.
fn extract_mb_padded(
    v: &VideoFrame,
    mb_w: usize,
    mb_h: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let width = v.width as usize;
    let height = v.height as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_h = mb_h * 16;
    let uv_h = mb_h * 8;

    let y_plane = &v.planes[0];
    let u_plane = &v.planes[1];
    let v_plane = &v.planes[2];

    let mut y_out = vec![0u8; y_stride * y_h];
    for j in 0..y_h {
        let src_row = j.min(height - 1);
        let src_start = src_row * y_plane.stride;
        for i in 0..y_stride {
            let src_col = i.min(width - 1);
            y_out[j * y_stride + i] = y_plane.data[src_start + src_col];
        }
    }
    let uv_w = (width + 1) / 2;
    let uv_src_h = (height + 1) / 2;
    let mut u_out = vec![0u8; uv_stride * uv_h];
    let mut v_out = vec![0u8; uv_stride * uv_h];
    for j in 0..uv_h {
        let src_row = j.min(uv_src_h - 1);
        let u_start = src_row * u_plane.stride;
        let v_start = src_row * v_plane.stride;
        for i in 0..uv_stride {
            let src_col = i.min(uv_w - 1);
            u_out[j * uv_stride + i] = u_plane.data[u_start + src_col];
            v_out[j * uv_stride + i] = v_plane.data[v_start + src_col];
        }
    }
    Ok((y_out, u_out, v_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;
    use crate::tables::coeff_probs::DEFAULT_COEF_PROBS;
    use crate::tokens::{decode_block, BlockType};

    fn roundtrip_one_block(coeffs: &[i16; 16], plane: usize, nctx: u8, start: usize) {
        let mut enc = BoolEncoder::new();
        let _nz_enc = encode_block(
            &mut enc,
            &DEFAULT_COEF_PROBS,
            plane,
            nctx as usize,
            coeffs,
            start,
        );
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let bt = match plane {
            0 => BlockType::YAfterY2,
            1 => BlockType::Y2,
            2 => BlockType::UV,
            _ => BlockType::YNoY2,
        };
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, bt, nctx, &mut out, start);
        for i in start..16 {
            let zz_idx = crate::tables::token_tree::ZIGZAG[i];
            assert_eq!(
                out[zz_idx], coeffs[zz_idx],
                "coeff at zigzag pos {i} (raw {zz_idx}) mismatch: in={} out={}",
                coeffs[zz_idx], out[zz_idx]
            );
        }
    }

    #[test]
    fn block_roundtrip_all_zero() {
        let coeffs = [0i16; 16];
        roundtrip_one_block(&coeffs, 0, 0, 1);
        roundtrip_one_block(&coeffs, 1, 0, 0);
        roundtrip_one_block(&coeffs, 2, 0, 0);
    }

    #[test]
    fn block_roundtrip_dc_only() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5;
        roundtrip_one_block(&coeffs, 1, 0, 0);
        coeffs[0] = -3;
        roundtrip_one_block(&coeffs, 2, 0, 0);
    }

    #[test]
    fn block_roundtrip_small_values() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;
        coeffs[1] = 2;
        coeffs[4] = 3;
        coeffs[8] = -1;
        roundtrip_one_block(&coeffs, 0, 0, 1);
    }

    #[test]
    fn block_roundtrip_y2_negative_dc() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = -19;
        roundtrip_one_block(&coeffs, 1, 0, 0);
    }

    #[test]
    fn block_roundtrip_y2_sparse_ctx1() {
        let coeffs: [i16; 16] = [7, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        roundtrip_one_block(&coeffs, 1, 1, 0);
    }

    #[test]
    fn block_roundtrip_y_with_dc_zeroed() {
        let mut y_blks = [[0i16; 16]; 16];
        for bi in 0..16 {
            y_blks[bi][1] = -3;
            y_blks[bi][4] = -2;
        }
        let mut enc = BoolEncoder::new();
        let mut nz_left = [0u8; 4];
        let mut nz_above = [0u8; 4];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = by * 4 + bx;
                let nctx = nz_above[bx] + nz_left[by];
                let nz = encode_block(
                    &mut enc,
                    &DEFAULT_COEF_PROBS,
                    0,
                    nctx as usize,
                    &y_blks[idx],
                    1,
                );
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_above[bx] = nzf;
                nz_left[by] = nzf;
            }
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let mut nz_left = [0u8; 4];
        let mut nz_above = [0u8; 4];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = by * 4 + bx;
                let nctx = nz_above[bx] + nz_left[by];
                let mut out = [0i16; 16];
                let nz = decode_block(
                    &mut dec,
                    &DEFAULT_COEF_PROBS,
                    BlockType::YAfterY2,
                    nctx,
                    &mut out,
                    1,
                );
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_above[bx] = nzf;
                nz_left[by] = nzf;
                for k in 1..16 {
                    let zz = crate::tables::token_tree::ZIGZAG[k];
                    assert_eq!(out[zz], y_blks[idx][zz], "block {idx} zz pos {k}");
                }
            }
        }
    }

    #[test]
    fn block_roundtrip_two_y2_back_to_back() {
        let a: [i16; 16] = [-79, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        let b: [i16; 16] = [7, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        let mut enc = BoolEncoder::new();
        encode_block(&mut enc, &DEFAULT_COEF_PROBS, 1, 0, &a, 0);
        encode_block(&mut enc, &DEFAULT_COEF_PROBS, 1, 1, &b, 0);
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::Y2, 0, &mut out, 0);
        assert_eq!(out, a);
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::Y2, 1, &mut out, 0);
        assert_eq!(out, b);
    }

    #[test]
    fn block_roundtrip_category_magnitudes() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 6;
        coeffs[1] = 9;
        coeffs[2] = 15;
        coeffs[3] = 25;
        coeffs[4] = -50;
        coeffs[5] = 100;
        roundtrip_one_block(&coeffs, 1, 0, 0);
    }
}
