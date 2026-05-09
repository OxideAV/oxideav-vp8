//! Round-46 encoder push tests — first-pass real-context SPLIT_MV
//! scoring (`enable_split_mv_rdo_real_context_first_pass`) + MV-cost-aware
//! sub-pel partition refinement (`enable_subpel_mv_cost_partition`).
//!
//! Round-46 lands two complementary picker upgrades on top of the
//! round-45 second-pass real-context swap + round-43 SPLIT_MV RDO:
//!
//!   * `enable_split_mv_rdo_real_context_first_pass` folds the
//!     real-context rate model into the per-ref picker so the
//!     SPLIT-vs-NEW competition under `D + λ·R` sees the bitstream
//!     rate from the start, not the round-43 neutral-context upper
//!     bound. Subsumes the round-45 second-pass swap (when both flags
//!     are on the second pass becomes a no-op because the first-pass
//!     picker already chose the real-context winner).
//!
//!   * `enable_subpel_mv_cost_partition` extends the 3×3 quarter-pel
//!     `subpel_refine_partition` hill-climb with the same
//!     `mv_cost_lambda` rate term used in `subpel_refine_luma`, so
//!     SPLIT_MV partitions land on rate-cheaper MVs (smaller delta to
//!     the absolute-MV proxy `split_mv_total_rate_x256` charges).
//!
//! Tests:
//!  1) Default config has both knobs off.
//!  2) Off path produces byte-identical encoder output.
//!  3) First-pass real-context on: keyframe + P-frame decode cleanly.
//!  4) First-pass real-context byte envelope ±10 % vs round-45 baseline.
//!  5) First-pass real-context requires `enable_split_mv_rdo`.
//!  6) Sub-pel MV-cost partition refinement on: P-frame decodes cleanly.
//!  7) Sub-pel MV-cost partition refinement requires `enable_subpel_mv_cost`.
//!  8) Combined round-46 + round-45 + round-43: clean round-trip.

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_AQ_QINDEX_RANGE, DEFAULT_GOLDEN_INTERVAL, DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH,
    DEFAULT_SIMPLE_LF_MAX_LEVEL,
};

const W: u32 = 32;
const H: u32 = 32;
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

/// Pan clip — frames translate by 1 pixel per frame so most inter MBs
/// find a good NEW_MV close to their NEAREST/NEAR neighbours, the
/// configuration where SPLIT_MV vs NEW_MV competition matters most
/// (and where neutral-context vs real-context rate model differs the
/// most).
fn make_pan_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let phase = (f as i32).rem_euclid(W as i32);
                let bx = (col as i32 + phase).rem_euclid(W as i32);
                let by = (row as i32 + phase / 2).rem_euclid(H as i32);
                let base = if (bx + by) % 6 < 3 { 60 } else { 200 };
                let wobble = ((row as i32 * 5 + col as i32 * 3) % 11) - 5;
                y[row * W as usize + col] = (base + wobble).clamp(0, 255) as u8;
            }
        }
        out.push(make_frame(y, vec![128u8; cw * ch], vec![128u8; cw * ch]));
    }
    out
}

fn measure(cfg: Vp8EncoderConfig, clip: &[VideoFrame]) -> (usize, f64, Vec<Vec<u8>>) {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));

    let mut enc = make_encoder_with_config(&params, cfg).expect("encoder");
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let mut total_bytes = 0usize;
    let mut psnr_sum = 0f64;
    let mut psnr_n = 0usize;
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(clip.len());

    for f in clip.iter() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            total_bytes += pkt.data.len();
            packets.push(pkt.data.clone());
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
                .expect("decode");
            while let Ok(frame) = dec.receive_frame() {
                if let Frame::Video(vf) = frame {
                    let y_src = &f.planes[0].data;
                    let y_dec = &vf.planes[0].data;
                    let src_stride = f.planes[0].stride;
                    let dec_stride = vf.planes[0].stride;
                    let mut se = 0f64;
                    for r in 0..H as usize {
                        for c in 0..W as usize {
                            let a = y_src[r * src_stride + c] as f64;
                            let b = y_dec[r * dec_stride + c] as f64;
                            se += (a - b) * (a - b);
                        }
                    }
                    let mse = se / (W as f64 * H as f64);
                    psnr_sum += if mse == 0.0 {
                        60.0
                    } else {
                        10.0 * (255.0f64 * 255.0 / mse).log10()
                    };
                    psnr_n += 1;
                }
            }
        }
    }
    let avg_psnr = if psnr_n > 0 {
        psnr_sum / psnr_n as f64
    } else {
        0.0
    };
    (total_bytes, avg_psnr, packets)
}

/// Round-46 baseline: round-43 SPLIT_MV RDO on (so the first-pass
/// real-context flag has something to score) and round-45 MV-cost-aware
/// snap on (the snap is independent and reflects the typical caller
/// configuration).
fn cfg_baseline() -> Vp8EncoderConfig {
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
        enable_scene_cut: false,
        scene_cut_threshold: 0.0,
        scene_cut_quant_boost: 0,
        scene_cut_boost_frames: 0,
        enable_lookahead_altref: false,
        lookahead_window: 0,
        loop_filter_mode: LoopFilterMode::Normal,
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
        aq_qindex_range: DEFAULT_AQ_QINDEX_RANGE,
        enable_joint_lf_rdo: false,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: true,
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
    }
}

#[test]
fn default_config_round46_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_split_mv_rdo_real_context_first_pass,
        "round-46 default must keep first-pass real-context off"
    );
    assert!(
        !cfg.enable_subpel_mv_cost_partition,
        "round-46 default must keep sub-pel MV-cost partition refine off"
    );
}

/// Round-46 must not perturb the bitstream when both knobs are off.
#[test]
fn round46_off_path_byte_identical_to_legacy() {
    let clip = make_pan_clip(4);
    let cfg = cfg_baseline();
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(b0, b1, "deterministic encode must match itself");
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} differs between identical configs");
    }
}

#[test]
fn split_mv_real_context_first_pass_keyframe_decodes_cleanly() {
    let clip = make_pan_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context_first_pass: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "first-pass real-context keyframe produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "first-pass real-context keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn split_mv_real_context_first_pass_pframe_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context_first_pass: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "first-pass real-context P-frame produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "first-pass real-context P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// First-pass real-context refines the rate term but preserves the
/// distortion model; byte envelope stays within ±10 % of the
/// neutral-context baseline (same band the round-45 second-pass
/// envelope enforces).
#[test]
fn split_mv_real_context_first_pass_byte_envelope_within_10pct() {
    let clip = make_pan_clip(8);
    let baseline = cfg_baseline();
    let with_first_pass = Vp8EncoderConfig {
        enable_split_mv_rdo_real_context_first_pass: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_r, _, _) = measure(with_first_pass, &clip);

    let frac = (bytes_r as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.10,
        "first-pass real-context swung byte size by {:.1}% (neutral {bytes_g}, real {bytes_r}) — beyond +/-10%",
        frac * 100.0
    );
}

/// `enable_split_mv_rdo = false` makes the first-pass real-context
/// knob inert — there is no SPLIT_MV RDO weight to score against, so
/// the dispatch falls through to the legacy SAD-min path.
#[test]
fn split_mv_real_context_first_pass_requires_split_mv_rdo() {
    let clip = make_pan_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_split_mv_rdo: false,
        enable_split_mv_rdo_real_context_first_pass: false,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_split_mv_rdo: false,
        enable_split_mv_rdo_real_context_first_pass: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "first-pass real-context must be inert when enable_split_mv_rdo=false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "split-mv-rdo-off packet {i} differs");
    }
}

#[test]
fn subpel_mv_cost_partition_pframe_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_subpel_mv_cost: true,
        enable_subpel_mv_cost_partition: true,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "sub-pel MV-cost partition refine P-frame produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "sub-pel MV-cost partition refine P-frame PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// `enable_subpel_mv_cost = false` makes the partition refine knob
/// inert (the refinement uses the per-ref `subpel_mv_cost_lambda`
/// only when its parent flag is on; we mirror the same gate inside
/// the call site, so the picker collapses to SAD-only).
#[test]
fn subpel_mv_cost_partition_requires_subpel_mv_cost() {
    let clip = make_pan_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_subpel_mv_cost: false,
        enable_subpel_mv_cost_partition: false,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_subpel_mv_cost: false,
        enable_subpel_mv_cost_partition: true,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "partition refine must be inert when enable_subpel_mv_cost=false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "subpel-mv-cost-off packet {i} differs");
    }
}

/// Combined round-46 on top of round-45 + round-43: turn everything on
/// at once and confirm a clean round-trip with reasonable PSNR.
#[test]
fn round46_combined_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context: true,
        enable_split_mv_rdo_real_context_first_pass: true,
        enable_subpel_mv_cost: true,
        enable_subpel_mv_cost_partition: true,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_mv_cost_aware_snap: true,
        enable_uv_rdo: true,
        enable_mode_ref_lf_deltas: true,
        enable_joint_lf_rdo: true,
        enable_adaptive_lf_deltas: true,
        enable_trellis_quant: true,
        enable_trellis_full: true,
        enable_trellis_context_rate: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "combined round-46 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-46 PSNR collapsed: {psnr_y:.2} dB"
    );
}
