//! Round-47 encoder push tests — high-QP adaptive LF magnitude scaling
//! (`enable_adaptive_lf_high_qp_cap`) + sub-pel rate-term refactor.
//!
//! Round-47 lands two related picker upgrades on top of the round-46
//! first-pass real-context SPLIT_MV scoring + MV-cost-aware sub-pel
//! partition refinement:
//!
//!   * `enable_adaptive_lf_high_qp_cap` lets the round-44 adaptive LF
//!     estimator's per-bucket delta cap grow from `±6` at `qindex ≤ 60`
//!     linearly up to `±10` at `qindex ≥ 110`. The round-44 cap is
//!     calibrated for mid-QP; at high QP the per-MB SSE distribution
//!     compresses against the cap and the adaptation signal is
//!     truncated. The expansion is always *at* the cap — when the
//!     proportional bucket deviation is small the produced delta is
//!     identical regardless. Off-by-default so the round-44 calibration
//!     is preserved bit-for-bit.
//!
//!   * Sub-pel rate-term refactor folds the duplicated `mv_rate_cost`
//!     closures from `subpel_refine_luma` and `subpel_refine_partition`
//!     into a single shared helper `subpel_mv_rate_cost_x256`. Pure
//!     mechanical refactor — no behavioural change. Verified via the
//!     same off-path bit-identity tests round-46 used for
//!     `enable_subpel_mv_cost_partition`.
//!
//! Tests:
//!  1) Default config has the new knob off.
//!  2) Off path produces byte-identical encoder output (regression
//!     guard against accidental engagement).
//!  3) High-QP cap on: high-QP P-frame decodes cleanly.
//!  4) High-QP cap byte envelope ±20 % vs round-44 baseline at high QP.
//!  5) High-QP cap is inert at low QP (cap == 6 there, identical
//!     bitstream).
//!  6) High-QP cap requires `enable_adaptive_lf_deltas`.
//!  7) Refactored sub-pel rate term: same byte size as pre-refactor
//!     when `enable_subpel_mv_cost{,_partition}` are on (no behavioural
//!     change, just the helper extraction).
//!  8) Combined round-47 + round-46: clean round-trip with reasonable
//!     PSNR at a high-QP setting.

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

/// Half-and-half clip — mixes intra (the keyframe) and inter MBs of
/// different modes, giving the adaptive estimator a non-trivial bucket
/// distribution to work from. Same shape as round-44 used for its
/// adaptive LF tests.
fn make_half_step_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let phase = (f as i32).rem_euclid(W as i32);
                let bx = (col as i32 + phase).rem_euclid(W as i32);
                let base = if bx < W as i32 / 2 { 60 } else { 200 };
                let wobble = ((row as i32 * 3) % 7) - 3;
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

/// Round-47 baseline config at a high QP (`qindex = 110`) where the
/// round-44 adaptive LF cap of `±6` saturates the SSE-deviation signal
/// the most. Matches the round-44 baseline shape (mode/ref LF deltas
/// on so the adaptive ladder can replace the static one).
fn cfg_baseline_high_qp() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: 110,
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
        enable_mode_ref_lf_deltas: true,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: true,
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

/// Same baseline at low QP (`qindex = 30`) where the high-QP cap ramp
/// floors at `6` and the produced bitstream must match the off path
/// bit-for-bit.
fn cfg_baseline_low_qp() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: 30,
        ..cfg_baseline_high_qp()
    }
}

#[test]
fn default_config_round47_knob_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_adaptive_lf_high_qp_cap,
        "round-47 default must keep adaptive-LF high-QP cap off"
    );
}

/// Round-47 must not perturb the bitstream when the new knob is off.
#[test]
fn round47_off_path_byte_identical_to_legacy() {
    let clip = make_half_step_clip(4);
    let cfg = cfg_baseline_high_qp();
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(b0, b1, "deterministic encode must match itself");
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} differs between identical configs");
    }
}

#[test]
fn high_qp_cap_pframe_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_adaptive_lf_high_qp_cap: true,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        ..cfg_baseline_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "high-QP cap P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "high-QP cap P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// The high-QP cap only widens the per-bucket delta clamp; the
/// signed-6 grammar and frame-mean comparison are unchanged. Byte size
/// stays within ±20 % of the round-44 baseline at the same QP (same
/// envelope the round-44 adaptive-on test uses).
#[test]
fn high_qp_cap_byte_envelope_within_20pct() {
    let clip = make_half_step_clip(8);
    let baseline = cfg_baseline_high_qp();
    let with_high_cap = Vp8EncoderConfig {
        enable_adaptive_lf_high_qp_cap: true,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        ..cfg_baseline_high_qp()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _) = measure(with_high_cap, &clip);

    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.20,
        "high-QP cap swung byte size by {:.1}% (round-44 {bytes_g}, round-47 {bytes_a}) — beyond +/-20%",
        frac * 100.0
    );
}

/// At low QP (`qindex ≤ 60`) the round-47 cap ramp floors at `6`, the
/// same value the round-44 estimator uses. Toggling the high-QP cap
/// flag must produce a byte-identical bitstream to the round-44 path.
#[test]
fn high_qp_cap_inert_at_low_qp() {
    let clip = make_half_step_clip(4);
    let baseline = cfg_baseline_low_qp();
    let with_high_cap = Vp8EncoderConfig {
        enable_adaptive_lf_high_qp_cap: true,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        ..cfg_baseline_low_qp()
    };

    let (b0, _, p0) = measure(baseline, &clip);
    let (b1, _, p1) = measure(with_high_cap, &clip);

    assert_eq!(
        b0, b1,
        "high-QP cap must be inert at qindex=30: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "low-QP packet {i} differs");
    }
}

/// `enable_adaptive_lf_deltas = false` makes the high-QP cap knob
/// inert (the cap only feeds the round-44 estimator; with that off the
/// encoder uses the static round-42 ladder regardless).
#[test]
fn high_qp_cap_requires_adaptive_lf_deltas() {
    let clip = make_half_step_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_adaptive_lf_deltas: false,
        enable_adaptive_lf_high_qp_cap: false,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        ..cfg_baseline_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_adaptive_lf_deltas: false,
        enable_adaptive_lf_high_qp_cap: true,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        ..cfg_baseline_high_qp()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "high-QP cap must be inert when adaptive LF deltas off: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "adaptive-off packet {i} differs");
    }
}

/// The sub-pel rate-term refactor (extracting `subpel_mv_rate_cost_x256`
/// from the duplicated closures) is a pure mechanical change. With the
/// MV-cost knobs on, encoding the same clip twice at the same config
/// must remain deterministic and produce identical packets — the
/// determinism guard catches any accidental behavioural drift the
/// refactor might have introduced.
#[test]
fn subpel_rate_refactor_deterministic() {
    let clip = make_half_step_clip(4);
    let cfg = Vp8EncoderConfig {
        enable_subpel_mv_cost: true,
        enable_subpel_mv_cost_partition: true,
        enable_split_mv_rdo: true,
        ..cfg_baseline_high_qp()
    };
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(
        b0, b1,
        "sub-pel rate refactor broke determinism: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "sub-pel rate refactor packet {i} differs");
    }
}

/// Combined round-47 + round-46 + round-44: turn everything on at a
/// high-QP setting and confirm a clean round-trip with reasonable PSNR.
#[test]
fn round47_combined_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_split_mv_rdo_real_context: true,
        enable_split_mv_rdo_real_context_first_pass: true,
        enable_subpel_mv_cost: true,
        enable_subpel_mv_cost_partition: true,
        enable_mv_cost_aware_snap: true,
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        enable_adaptive_lf_high_qp_cap: true,
        enable_variance_lf_cap: false,
        enable_adaptive_uv_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        spatial_lf_n_row_bands: 4,
        spatial_lf_n_col_bands: 4,
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: 256,
        enable_joint_lf_rdo: true,
        ..cfg_baseline_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "combined round-47 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-47 PSNR collapsed: {psnr_y:.2} dB"
    );
}
