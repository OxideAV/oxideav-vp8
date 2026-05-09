//! Per-MB segment-map regressions (RFC 6386 §10).
//!
//! These tests pin the new segmentation behaviour added to the encoder:
//!
//! * The encoder's variance-based MB classifier maps low-variance
//!   (smooth) regions into segment 0/1 and high-variance (textured)
//!   regions into segment 2/3.
//! * On a mixed-content fixture (a smooth half spliced with a
//!   high-variance noise half), `enable_segments = true` produces a
//!   strictly smaller bitstream than `enable_segments = false` at the
//!   same nominal qindex (because the high-variance segment quantises
//!   coarser and saves bits where texture masks the loss).
//! * The frame header signals `segmentation_enabled = 1` with the
//!   per-segment quantiser deltas the config supplied, and the in-tree
//!   decoder applies the per-segment dequant correctly (decode round-trip
//!   stays at high PSNR).
//! * `ffmpeg` cross-decodes the segmented bitstream cleanly (skipped when
//!   ffmpeg is not on `PATH`).

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256, DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
    DEFAULT_GOLDEN_INTERVAL, DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
    DEFAULT_KMEANS_CONVERGENCE_THRESHOLD, DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH,
    DEFAULT_SEGMENT_LF_DELTAS, DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL,
};
use oxideav_vp8::frame_header::{parse_inter_header, parse_keyframe_header, PersistentProbs};
use oxideav_vp8::{parse_header, FrameType};

const W: u32 = 96;
const H: u32 = 64;
const QINDEX: u8 = 50;

fn make_frame(y: Vec<u8>, u: Vec<u8>, v: Vec<u8>) -> VideoFrame {
    let cw = (W / 2) as usize;
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: W as usize,
                data: y,
            },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
        ],
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Mixed-content fixture: the left half is a smooth horizontal gradient
/// (low variance per MB), the right half is a high-frequency
/// pseudo-noise pattern (high variance). The two halves split exactly on
/// MB boundaries so each MB lands cleanly in either the smooth or noisy
/// region. With segments enabled the encoder must classify the smooth
/// half into segment 0/1 (low-quant, high quality) and the noisy half
/// into segment 2/3 (coarse-quant, bit savings).
fn make_mixed_clip(n_frames: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let v = if col < (W as usize) / 2 {
                    // Smooth gradient: 64 + col/2 (very low MB variance).
                    (64 + (col as i32) / 2 + (f as i32 % 4)).clamp(0, 255) as u8
                } else {
                    // Pseudo-noise: blockless high-frequency pattern.
                    let mut h: u32 = (row as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add((col as u32).wrapping_mul(40503))
                        .wrapping_add(f as u32 * 17);
                    h ^= h >> 13;
                    (h & 0xff) as u8
                };
                y[row * W as usize + col] = v;
            }
        }
        let u = vec![128u8; cw * ch];
        let v = vec![128u8; cw * ch];
        out.push(make_frame(y, u, v));
    }
    out
}

fn encode_clip(config: Vp8EncoderConfig, clip: &[VideoFrame]) -> Vec<Vec<u8>> {
    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_config(&enc_params, config).expect("encoder");
    let mut packets = Vec::with_capacity(clip.len());
    for f in clip.iter() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("rx");
        packets.push(pkt.data);
    }
    packets
}

fn decode_clip_psnr(packets: &[Vec<u8>], src: &[VideoFrame]) -> Vec<f64> {
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let mut psnrs = Vec::with_capacity(packets.len());
    for (i, p) in packets.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), p.clone()))
            .expect("decode");
        let frame = match dec.receive_frame().expect("rx") {
            Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        psnrs.push(psnr(&frame.planes[0].data, &src[i].planes[0].data));
    }
    psnrs
}

fn cfg_no_segments() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
        segment_lf_deltas: [0; 4],
        // Disable scene-cut detection: this fixture is randomised
        // pseudo-noise that would otherwise trip the cut detector and
        // change the keyframe / size accounting these tests pin.
        enable_scene_cut: false,
        scene_cut_threshold: 0.0,
        scene_cut_quant_boost: 0,
        scene_cut_boost_frames: 0,
        // Disable look-ahead alt-ref synthesis (#209) so packet counts
        // and refresh-flag cadence remain bit-exact with the pre-#209
        // baseline these tests pin.
        enable_lookahead_altref: false,
        lookahead_window: 0,
        loop_filter_mode: LoopFilterMode::Auto,
        simple_lf_max_level: DEFAULT_SIMPLE_LF_MAX_LEVEL,
        y_dc_delta: 0,
        y2_dc_delta: 0,
        y2_ac_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
        adaptive_segment_thresholds: false,
        enable_split_mv_joint_refine: false,
        split_mv_joint_refine_passes: 0,
        lambda_long_ref_scale_x256: 256,
        enable_trellis_quant: false,
        enable_subpel_mv_cost: false,
        enable_psy_rdo: false,
        psy_rd_strength: DEFAULT_PSY_RD_STRENGTH,
        enable_arnr_nlm: false,
        nlm_h2: DEFAULT_NLM_H2,
        enable_trellis_full: false,
        enable_aq: false,
        aq_qindex_range: 8,
        enable_joint_lf_rdo: false,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        enable_mv_cost_aware_snap: false,
        enable_split_mv_rdo_real_context: false,
        enable_split_mv_rdo_real_context_first_pass: false,
        enable_subpel_mv_cost_partition: false,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
    }
}

fn cfg_with_segments() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: true,
        segment_quant_deltas: DEFAULT_SEGMENT_QUANT_DELTAS,
        segment_lf_deltas: DEFAULT_SEGMENT_LF_DELTAS,
        enable_scene_cut: false,
        scene_cut_threshold: 0.0,
        scene_cut_quant_boost: 0,
        scene_cut_boost_frames: 0,
        enable_lookahead_altref: false,
        lookahead_window: 0,
        loop_filter_mode: LoopFilterMode::Auto,
        simple_lf_max_level: DEFAULT_SIMPLE_LF_MAX_LEVEL,
        y_dc_delta: 0,
        y2_dc_delta: 0,
        y2_ac_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
        adaptive_segment_thresholds: false,
        enable_split_mv_joint_refine: false,
        split_mv_joint_refine_passes: 0,
        lambda_long_ref_scale_x256: 256,
        enable_trellis_quant: false,
        enable_subpel_mv_cost: false,
        enable_psy_rdo: false,
        psy_rd_strength: DEFAULT_PSY_RD_STRENGTH,
        enable_arnr_nlm: false,
        nlm_h2: DEFAULT_NLM_H2,
        enable_trellis_full: false,
        enable_aq: false,
        aq_qindex_range: 8,
        enable_joint_lf_rdo: false,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        enable_mv_cost_aware_snap: false,
        enable_split_mv_rdo_real_context: false,
        enable_split_mv_rdo_real_context_first_pass: false,
        enable_subpel_mv_cost_partition: false,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
    }
}

/// Bit-saving-only segment config: every segment quantises at >= the
/// frame qi so segment maps monotonically save bits without ever
/// demanding a finer quant. Used by the size-regression test which
/// proves the encoder's segment classifier picks up the textured MBs
/// and routes them through a coarser quantiser.
fn cfg_with_save_segments() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: true,
        // Smooth segments stay at the frame qi; high-variance segments
        // step up to coarser quants (saves bits where texture masks the
        // loss).
        segment_quant_deltas: [0, 2, 6, 12],
        // Pin LF deltas off so the existing 5%-savings regression
        // pins purely the quant-delta savings path. The DEFAULT
        // (`[-2, -1, 0, +2]`) is exercised by `cfg_with_segments`.
        segment_lf_deltas: [0; 4],
        enable_scene_cut: false,
        scene_cut_threshold: 0.0,
        scene_cut_quant_boost: 0,
        scene_cut_boost_frames: 0,
        enable_lookahead_altref: false,
        lookahead_window: 0,
        loop_filter_mode: LoopFilterMode::Auto,
        simple_lf_max_level: DEFAULT_SIMPLE_LF_MAX_LEVEL,
        y_dc_delta: 0,
        y2_dc_delta: 0,
        y2_ac_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
        adaptive_segment_thresholds: false,
        enable_split_mv_joint_refine: false,
        split_mv_joint_refine_passes: 0,
        lambda_long_ref_scale_x256: 256,
        enable_trellis_quant: false,
        enable_subpel_mv_cost: false,
        enable_psy_rdo: false,
        psy_rd_strength: DEFAULT_PSY_RD_STRENGTH,
        enable_arnr_nlm: false,
        nlm_h2: DEFAULT_NLM_H2,
        enable_trellis_full: false,
        enable_aq: false,
        aq_qindex_range: 8,
        enable_joint_lf_rdo: false,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        enable_mv_cost_aware_snap: false,
        enable_split_mv_rdo_real_context: false,
        enable_split_mv_rdo_real_context_first_pass: false,
        enable_subpel_mv_cost_partition: false,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
    }
}

/// Frame headers must report `segmentation_enabled = 1` and the per-MB
/// `update_map = 1` bit when segments are enabled. Confirms the bool-coded
/// segmentation block round-trips through `parse_keyframe_header` /
/// `parse_inter_header` cleanly.
#[test]
fn frame_headers_signal_segments_on() {
    let clip = make_mixed_clip(4);
    let packets = encode_clip(cfg_with_segments(), &clip);
    let probs = PersistentProbs::defaults();
    use oxideav_vp8::bool_decoder::BoolDecoder;
    for (i, pkt) in packets.iter().enumerate() {
        let parsed = parse_header(pkt).expect("hdr");
        let body = &pkt[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let h = match parsed.tag.frame_type {
            FrameType::Key => parse_keyframe_header(&mut bd).expect("kf"),
            FrameType::Inter => parse_inter_header(&mut bd, &probs).expect("inter"),
        };
        assert!(
            h.segmentation.enabled,
            "frame {i} did not signal segmentation"
        );
        assert!(
            h.segmentation.update_map,
            "frame {i} did not signal update_map"
        );
        assert!(
            h.segmentation.update_data,
            "frame {i} did not signal update_data"
        );
        assert!(
            !h.segmentation.abs_delta,
            "frame {i} should signal delta mode (abs_delta=0)"
        );
        // Per-segment deltas must match what the encoder was configured with.
        assert_eq!(
            h.segmentation.quant, DEFAULT_SEGMENT_QUANT_DELTAS,
            "frame {i} per-segment quant deltas mismatch"
        );
        // Per-segment loop-filter deltas (#337) must match the
        // encoder's configured per-segment LF schedule. With the
        // defaults (`[-2, -1, 0, +2]`) smooth segments take a softer
        // filter and high-variance segments take a stronger one.
        assert_eq!(
            h.segmentation.lf, DEFAULT_SEGMENT_LF_DELTAS,
            "frame {i} per-segment loop-filter deltas mismatch"
        );
    }
}

/// Per-segment loop-filter deltas (#337): an explicit non-zero
/// `segment_lf_deltas` must round-trip through the segmentation block
/// of the frame header into the decoder's `LoopFilterHeader.lf` array.
/// This pins both the encoder bool-coded emit (`emit_segmentation_header`)
/// and the decoder parser (`parse_segmentation`) on the same per-segment
/// LF wire format.
#[test]
fn frame_headers_carry_segment_lf_deltas() {
    let clip = make_mixed_clip(3);
    let custom_lf: [i32; 4] = [-3, 0, 1, 5];
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: true,
        segment_quant_deltas: DEFAULT_SEGMENT_QUANT_DELTAS,
        segment_lf_deltas: custom_lf,
        enable_scene_cut: false,
        scene_cut_threshold: 0.0,
        scene_cut_quant_boost: 0,
        scene_cut_boost_frames: 0,
        enable_lookahead_altref: false,
        lookahead_window: 0,
        loop_filter_mode: LoopFilterMode::Auto,
        simple_lf_max_level: DEFAULT_SIMPLE_LF_MAX_LEVEL,
        y_dc_delta: 0,
        y2_dc_delta: 0,
        y2_ac_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
        adaptive_segment_thresholds: false,
        enable_split_mv_joint_refine: false,
        split_mv_joint_refine_passes: 0,
        lambda_long_ref_scale_x256: 256,
        enable_trellis_quant: false,
        enable_subpel_mv_cost: false,
        enable_psy_rdo: false,
        psy_rd_strength: DEFAULT_PSY_RD_STRENGTH,
        enable_arnr_nlm: false,
        nlm_h2: DEFAULT_NLM_H2,
        enable_trellis_full: false,
        enable_aq: false,
        aq_qindex_range: 8,
        enable_joint_lf_rdo: false,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        enable_mv_cost_aware_snap: false,
        enable_split_mv_rdo_real_context: false,
        enable_split_mv_rdo_real_context_first_pass: false,
        enable_subpel_mv_cost_partition: false,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
    };
    let packets = encode_clip(cfg, &clip);
    let probs = PersistentProbs::defaults();
    use oxideav_vp8::bool_decoder::BoolDecoder;
    for (i, pkt) in packets.iter().enumerate() {
        let parsed = parse_header(pkt).expect("hdr");
        let body = &pkt[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let h = match parsed.tag.frame_type {
            FrameType::Key => parse_keyframe_header(&mut bd).expect("kf"),
            FrameType::Inter => parse_inter_header(&mut bd, &probs).expect("inter"),
        };
        assert!(h.segmentation.enabled, "frame {i} not segmented");
        assert_eq!(
            h.segmentation.lf, custom_lf,
            "frame {i} per-segment LF deltas mismatch (wanted {custom_lf:?}, got {:?})",
            h.segmentation.lf
        );
    }
}

/// Headers must NOT signal segmentation when `enable_segments = false`,
/// preserving the legacy single-segment encoding bit-for-bit.
#[test]
fn frame_headers_keep_legacy_when_segments_off() {
    let clip = make_mixed_clip(2);
    let packets = encode_clip(cfg_no_segments(), &clip);
    use oxideav_vp8::bool_decoder::BoolDecoder;
    let probs = PersistentProbs::defaults();
    for (i, pkt) in packets.iter().enumerate() {
        let parsed = parse_header(pkt).expect("hdr");
        let body = &pkt[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let h = match parsed.tag.frame_type {
            FrameType::Key => parse_keyframe_header(&mut bd).expect("kf"),
            FrameType::Inter => parse_inter_header(&mut bd, &probs).expect("inter"),
        };
        assert!(
            !h.segmentation.enabled,
            "frame {i} unexpectedly signalled segmentation",
        );
    }
}

/// Headline regression: on a high-variance + smooth mixed clip, enabling
/// segments with the bit-saving deltas `[0, +2, +6, +12]` shrinks the
/// bitstream by a measurable margin (>= 5%). Segment 0 stays at the
/// frame qi so the smooth half is unchanged; segments 2/3 step up to
/// coarser quants and save bits where texture masks the loss.
#[test]
fn segments_shrink_mixed_content_clip() {
    let clip = make_mixed_clip(10);

    let off_packets = encode_clip(cfg_no_segments(), &clip);
    let on_packets = encode_clip(cfg_with_save_segments(), &clip);

    let off_bytes: usize = off_packets.iter().map(|p| p.len()).sum();
    let on_bytes: usize = on_packets.iter().map(|p| p.len()).sum();

    let off_psnrs = decode_clip_psnr(&off_packets, &clip);
    let on_psnrs = decode_clip_psnr(&on_packets, &clip);
    let off_avg: f64 = off_psnrs.iter().sum::<f64>() / off_psnrs.len() as f64;
    let on_avg: f64 = on_psnrs.iter().sum::<f64>() / on_psnrs.len() as f64;

    eprintln!(
        "segments OFF: {off_bytes} bytes, avg Y PSNR = {off_avg:.2} dB | \
         segments ON:  {on_bytes} bytes, avg Y PSNR = {on_avg:.2} dB",
    );

    // Hard regression: segments must engage and shrink the bitstream.
    let saved_pct = 100.0 * (off_bytes as f64 - on_bytes as f64) / off_bytes as f64;
    assert!(
        on_bytes < off_bytes,
        "segments did not shrink the bitstream: on={on_bytes} off={off_bytes} ({saved_pct:.2}%)",
    );
    assert!(
        saved_pct >= 5.0,
        "segments saved less than 5%: {saved_pct:.2}% (on={on_bytes} off={off_bytes})",
    );
    // Average PSNR must stay close to the off baseline. Per-frame can
    // dip more aggressively because segment 3 quantises 12 qi steps
    // higher and pseudo-noise dominates the SSE; the ~5% bit saving is
    // the real win we are pinning here.
    assert!(
        on_avg + 4.0 >= off_avg,
        "segments-on avg PSNR collapsed: on={on_avg:.2} off={off_avg:.2}",
    );
}

/// End-to-end: the in-tree decoder must apply per-segment dequant
/// (otherwise a segment-3 MB encoded at a different qi would be
/// reconstructed at the frame's base qi, producing a noticeably worse
/// block instead of a clean reconstruction). Sanity-checks that the
/// decoder honours the per-segment quant deltas signalled in the frame
/// header.
#[test]
fn segments_decode_roundtrip_high_psnr() {
    let clip = make_mixed_clip(6);
    let packets = encode_clip(cfg_with_segments(), &clip);
    let psnrs = decode_clip_psnr(&packets, &clip);
    let avg: f64 = psnrs.iter().sum::<f64>() / psnrs.len() as f64;
    eprintln!("segments-on: avg PSNR = {avg:.2} dB");
    // The pseudo-noise half is near-impossible to encode at any QP, but
    // the smooth half should still reconstruct cleanly. Floor at a low
    // bound to catch a collapsed decoder that ignores segment_id.
    for (i, &p) in psnrs.iter().enumerate() {
        assert!(p >= 10.0, "frame {i} segments PSNR collapsed: {p:.2} dB");
    }
    assert!(avg >= 12.0, "segments-on avg PSNR too low: {avg:.2}");

    // Segments-on PSNR must roughly match segments-off (the segment maps
    // shift bits between regions but should not crater the average). If
    // the decoder were ignoring segment_id, segment-1/2/3 MBs would be
    // dequantised at the wrong qi and the average would collapse by
    // many dB.
    let off_packets = encode_clip(cfg_no_segments(), &clip);
    let off_psnrs = decode_clip_psnr(&off_packets, &clip);
    let off_avg: f64 = off_psnrs.iter().sum::<f64>() / off_psnrs.len() as f64;
    eprintln!("segments-off baseline: avg PSNR = {off_avg:.2} dB");
    assert!(
        avg + 5.0 >= off_avg,
        "segments-on PSNR collapsed vs off baseline: on={avg:.2} off={off_avg:.2}",
    );
}

/// Build a minimal IVF wrapper around the encoded packets so the file
/// can be fed to `ffmpeg` for cross-decode validation. Single video
/// stream, 30 fps, dimensions in `(W, H)`, sequential PTS.
fn ivf_bytes_for_packets(packets: &[Vec<u8>], width: u32, height: u32, fps: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + packets.len() * 32);
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP80");
    out.extend_from_slice(&(width as u16).to_le_bytes());
    out.extend_from_slice(&(height as u16).to_le_bytes());
    out.extend_from_slice(&fps.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (i, p) in packets.iter().enumerate() {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u64).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
}

fn which(prog: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(prog);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Cross-decode validation: write a segmented encode to IVF, ask `ffmpeg`
/// to decode it (raw output to /dev/null), and confirm exit code 0.
/// Skipped silently if `ffmpeg` is not on `PATH` (CI parity).
#[test]
fn ffmpeg_cross_decode_accepts_segmented_stream() {
    let ffmpeg = match which("ffmpeg") {
        Some(p) => p,
        None => {
            eprintln!("ffmpeg not on PATH; skipping segments cross-decode test");
            return;
        }
    };
    let clip = make_mixed_clip(10);
    let packets = encode_clip(cfg_with_segments(), &clip);
    let ivf = ivf_bytes_for_packets(&packets, W, H, 30);
    let tmp = std::env::temp_dir().join("oxideav-vp8-segments-cross-decode.ivf");
    std::fs::write(&tmp, &ivf).expect("write tmp");
    let output = std::process::Command::new(ffmpeg)
        .args(["-loglevel", "error", "-y", "-i"])
        .arg(&tmp)
        .args(["-f", "null", "-"])
        .output()
        .expect("spawn ffmpeg");
    let _ = std::fs::remove_file(&tmp);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "ffmpeg failed to decode segmented bitstream: status={:?}\nstderr:\n{}",
            output.status, stderr
        );
    }
}
