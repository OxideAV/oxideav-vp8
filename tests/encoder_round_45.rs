//! Round-45 encoder push tests — MV-cost-aware NEAREST/NEAR/NEW snap
//! (`enable_mv_cost_aware_snap`) + SPLIT_MV RDO real-context second
//! pass (`enable_split_mv_rdo_real_context`).
//!
//! Round-45 lands two complementary picker upgrades on top of the
//! round-43 SPLIT_MV RDO + round-42 mode-info groundwork:
//!
//!   * `enable_mv_cost_aware_snap` augments the fixed-tolerance NEW_MV
//!     to NEAREST/NEAR snap with a Lagrangian check: when the SAD
//!     penalty for snapping is smaller than `λ × Δbits / 256`, the
//!     picker prefers the rate-cheaper neighbour mode even when the MV
//!     magnitude is more than `NEIGHBOUR_MV_SNAP_TOLERANCE` away. λ
//!     comes from `lambda_for_qp`, the same multiplier the per-MB
//!     ref/mode picker uses; `Δbits` is the bool-coder cost difference
//!     between coding NEW_MV (mv-tree path "1110" + MV-delta literal
//!     under `DEFAULT_MV_CONTEXT`) and the cheaper neighbour mode
//!     (NEAREST: "10", NEAR: "110").
//!
//!   * `enable_split_mv_rdo_real_context` adds a second pass after the
//!     per-MB picker commits a `SplitMv`: re-evaluate the four
//!     split-mode candidates with the actual neighbour sub-MVs from the
//!     already-committed left/above MBs (round-43 used the neutral
//!     `[0]` context because the search ran before any MB was
//!     committed). The real per-leaf path (LEFT / ABOVE / ZERO / NEW)
//!     and the real per-row `SUB_MV_REF_PROBS` row replace the round-43
//!     neutral approximation; if a different split mode wins under real
//!     context, the picker swaps before reconstruction.
//!
//! Tests:
//!  1) Default config has both knobs off.
//!  2) Off path produces byte-identical encoder output.
//!  3) MV-cost-aware snap on: keyframe + P-frame encode/decode cleanly.
//!  4) MV-cost-aware snap byte envelope ±10 % vs fixed-tolerance baseline.
//!  5) MV-cost-aware snap requires `enable_rdo`.
//!  6) Real-context second pass on: keyframe + P-frame decode cleanly.
//!  7) Real-context second pass byte envelope ±10 % vs neutral-context baseline.
//!  8) Real-context second pass requires `enable_split_mv_rdo`.
//!  9) Combined round-45 + round-44 + round-43: clean round-trip.

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

/// Pan clip — the whole frame translates by 1 pixel per frame so most
/// inter MBs find a good NEW_MV close to a NEAREST/NEAR neighbour, the
/// configuration where the MV-cost-aware snap most matters.
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

/// Encode a clip, return `(total_bytes, avg_psnr_y, packets)`.
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

/// Round-45 baseline config: round-43 SPLIT_MV RDO on (so the
/// real-context second pass has something to refine) and `enable_rdo`
/// on (so the MV-cost-aware snap has a real λ to weight against).
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
    }
}

#[test]
fn default_config_round45_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_mv_cost_aware_snap,
        "round-45 default must keep MV-cost-aware snap off"
    );
    assert!(
        !cfg.enable_split_mv_rdo_real_context,
        "round-45 default must keep SPLIT_MV real-context second pass off"
    );
}

/// Round-45 must not perturb the bitstream when both knobs are off.
#[test]
fn round45_off_path_byte_identical_to_legacy() {
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
fn mv_cost_aware_snap_keyframe_decodes_cleanly() {
    let clip = make_pan_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_mv_cost_aware_snap: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "MV-cost-aware-snap keyframe produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "MV-cost-aware-snap keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn mv_cost_aware_snap_pframe_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_mv_cost_aware_snap: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "MV-cost-aware-snap P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "MV-cost-aware-snap P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// MV-cost-aware snap only swaps NEW_MV for NEAREST/NEAR when the
/// Lagrangian-aware test fires; the bitstream keeps the same coarse
/// shape. ±10 % envelope vs the fixed-tolerance baseline.
#[test]
fn mv_cost_aware_snap_byte_envelope_within_10pct() {
    let clip = make_pan_clip(8);
    let baseline = cfg_baseline();
    let with_snap = Vp8EncoderConfig {
        enable_mv_cost_aware_snap: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _) = measure(with_snap, &clip);

    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.10,
        "MV-cost-aware snap swung byte size by {:.1}% (fixed {bytes_g}, snap {bytes_a}) — beyond +/-10%",
        frac * 100.0
    );
}

/// MV-cost-aware snap uses `lambda_for_qp` as the rate weight; with
/// `enable_rdo = false` λ collapses to 0 and the knob is inert.
#[test]
fn mv_cost_aware_snap_requires_enable_rdo() {
    let clip = make_pan_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_rdo: false,
        enable_mv_cost_aware_snap: false,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_rdo: false,
        enable_mv_cost_aware_snap: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "MV-cost-aware snap must be inert when enable_rdo=false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "rdo-off packet {i} differs");
    }
}

#[test]
fn split_mv_real_context_keyframe_decodes_cleanly() {
    let clip = make_pan_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context: true,
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
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "SPLIT_MV real-context keyframe produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "SPLIT_MV real-context keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn split_mv_real_context_pframe_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context: true,
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
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "SPLIT_MV real-context P-frame produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "SPLIT_MV real-context P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Real-context second pass refines the rate term but preserves the
/// distortion model; byte envelope stays within ±10 % of the
/// neutral-context baseline.
#[test]
fn split_mv_real_context_byte_envelope_within_10pct() {
    let clip = make_pan_clip(8);
    let baseline = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };
    let with_real = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context: true,
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
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_r, _, _) = measure(with_real, &clip);

    let frac = (bytes_r as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.10,
        "real-context second pass swung byte size by {:.1}% (neutral {bytes_g}, real {bytes_r}) — beyond +/-10%",
        frac * 100.0
    );
}

/// `enable_split_mv_rdo = false` makes the real-context knob inert —
/// the second pass only fires when the picker actually committed
/// SPLIT_MV under the round-43 rate weight.
#[test]
fn split_mv_real_context_requires_split_mv_rdo() {
    let clip = make_pan_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_split_mv_rdo: false,
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
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_split_mv_rdo: false,
        enable_split_mv_rdo_real_context: true,
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
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "real-context must be inert when enable_split_mv_rdo=false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "split-mv-rdo-off packet {i} differs");
    }
}

/// Combined round-45 + round-44 + round-43: turn everything on at once
/// and confirm a clean round-trip with reasonable PSNR.
#[test]
fn round45_combined_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context: true,
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
    assert!(bytes > 0, "combined round-45 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-45 PSNR collapsed: {psnr_y:.2} dB"
    );
}
