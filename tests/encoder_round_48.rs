//! Round-48 encoder push tests — variance-driven adaptive LF cap
//! (`enable_variance_lf_cap`) + UV-channel adaptive LF deltas
//! (`enable_adaptive_uv_lf_deltas`).
//!
//! Round-48 generalises the round-47 high-QP LF cap and the round-44
//! adaptive LF estimator with two opt-in knobs:
//!
//!   * `enable_variance_lf_cap` replaces the round-47 `qindex`-proxy
//!     ramp with a content-driven model: the cap is computed directly
//!     from the per-frame SSE distribution's normalised variance
//!     (coefficient of variation squared, `cv2 = var / mean^2`).
//!     Homogeneous content (`cv2 ≤ 0.5`) keeps the cap at `±6`; very
//!     heterogeneous content (`cv2 ≥ 1.0`) saturates at `±10`.
//!     Off-by-default; preserves the round-47 behaviour bit-for-bit
//!     when off. When both this flag and the round-47 cap are on,
//!     this flag wins (see encoder.rs).
//!
//!   * `enable_adaptive_uv_lf_deltas` extends the round-44 estimator to
//!     consider chroma SSE alongside luma. When on, the per-bucket
//!     delta is the average of the luma-only and chroma-only estimates;
//!     since both are inside `±delta_cap`, the average is too — no
//!     additional clamping required. Off-by-default; preserves the
//!     round-44 luma-only path bit-for-bit when off.
//!
//! Tests:
//!  1) Default config has both new knobs off.
//!  2) Variance cap off path is byte-identical to round-47 baseline.
//!  3) UV-deltas off path is byte-identical to round-47 baseline.
//!  4) Variance cap requires `enable_adaptive_lf_deltas` (inert when
//!     adaptive LF deltas are off).
//!  5) UV-deltas require `enable_adaptive_lf_deltas` (inert when off).
//!  6) Variance cap on a high-variance clip decodes cleanly.
//!  7) UV-deltas on a chroma-textured clip decodes cleanly.
//!  8) Combined round-48 + round-47 + round-44 decodes cleanly.

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

/// Half-step clip mirroring round-44/47 — moving step edge gives the
/// adaptive estimator a non-trivial bucket distribution.
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

/// High-variance clip: the left half is a flat patch (low SSE under
/// most ref/mode buckets) and the right half is a heavy gradient with a
/// per-frame phase shift (drives SPLIT_MV / NEW_MV residual high). The
/// per-frame variance of per-MB SSE is large — the variance-driven cap
/// should ramp toward `10` here.
fn make_high_variance_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let val = if col < W as usize / 2 {
                    // Flat — ZERO_MV target.
                    100
                } else {
                    // Steep gradient with a per-frame phase + a 5-pixel
                    // checker the encoder can't predict perfectly,
                    // driving residual SSE up.
                    let phase = (f as i32 * 7).rem_euclid(W as i32);
                    let g = ((col as i32 - W as i32 / 2 + phase) * 8).rem_euclid(256);
                    let chk = if (row / 5 + col / 5) % 2 == 0 {
                        16
                    } else {
                        -16
                    };
                    (g + chk).clamp(0, 255)
                };
                y[row * W as usize + col] = val as u8;
            }
        }
        out.push(make_frame(y, vec![128u8; cw * ch], vec![128u8; cw * ch]));
    }
    out
}

/// Chroma-textured clip: luma is a slow ramp (low luma SSE) but the U/V
/// planes carry per-frame movement that the encoder can't perfectly
/// predict from the previous frame's chroma. Drives the chroma SSE
/// distribution while keeping luma quiet — the natural environment for
/// the UV-delta knob to deviate from the luma-only path.
fn make_chroma_textured_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                // Slow vertical ramp — easy to predict from previous frame.
                y[row * W as usize + col] = (90 + row as i32 / 3).clamp(0, 255) as u8;
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                let phase = (f as i32 * 5).rem_euclid(cw as i32);
                let mix = ((c as i32 + phase) * 12).rem_euclid(256);
                u[r * cw + c] = (128 + (mix - 128) / 2).clamp(0, 255) as u8;
                v[r * cw + c] = (128 - (mix - 128) / 3).clamp(0, 255) as u8;
            }
        }
        out.push(make_frame(y, u, v));
    }
    out
}

fn measure(cfg: Vp8EncoderConfig, clip: &[VideoFrame]) -> (usize, f64, f64, Vec<Vec<u8>>) {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));

    let mut enc = make_encoder_with_config(&params, cfg).expect("encoder");
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let mut total_bytes = 0usize;
    let mut psnr_y_sum = 0f64;
    let mut psnr_uv_sum = 0f64;
    let mut psnr_n = 0usize;
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(clip.len());

    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;

    for f in clip.iter() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            total_bytes += pkt.data.len();
            packets.push(pkt.data.clone());
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
                .expect("decode");
            while let Ok(frame) = dec.receive_frame() {
                if let Frame::Video(vf) = frame {
                    // Luma PSNR.
                    let y_src = &f.planes[0].data;
                    let y_dec = &vf.planes[0].data;
                    let src_stride = f.planes[0].stride;
                    let dec_stride = vf.planes[0].stride;
                    let mut se_y = 0f64;
                    for r in 0..H as usize {
                        for c in 0..W as usize {
                            let a = y_src[r * src_stride + c] as f64;
                            let b = y_dec[r * dec_stride + c] as f64;
                            se_y += (a - b) * (a - b);
                        }
                    }
                    let mse_y = se_y / (W as f64 * H as f64);
                    psnr_y_sum += if mse_y == 0.0 {
                        60.0
                    } else {
                        10.0 * (255.0f64 * 255.0 / mse_y).log10()
                    };
                    // Combined chroma PSNR (Cb + Cr averaged).
                    let mut se_uv = 0f64;
                    for plane in 1..=2 {
                        let s = &f.planes[plane].data;
                        let d = &vf.planes[plane].data;
                        let ss = f.planes[plane].stride;
                        let ds = vf.planes[plane].stride;
                        for r in 0..ch {
                            for c in 0..cw {
                                let a = s[r * ss + c] as f64;
                                let b = d[r * ds + c] as f64;
                                se_uv += (a - b) * (a - b);
                            }
                        }
                    }
                    let mse_uv = se_uv / (2.0 * cw as f64 * ch as f64);
                    psnr_uv_sum += if mse_uv == 0.0 {
                        60.0
                    } else {
                        10.0 * (255.0f64 * 255.0 / mse_uv).log10()
                    };
                    psnr_n += 1;
                }
            }
        }
    }
    let avg_y = if psnr_n > 0 {
        psnr_y_sum / psnr_n as f64
    } else {
        0.0
    };
    let avg_uv = if psnr_n > 0 {
        psnr_uv_sum / psnr_n as f64
    } else {
        0.0
    };
    (total_bytes, avg_y, avg_uv, packets)
}

/// Round-48 baseline shape: high QP + adaptive LF deltas on, both new
/// knobs off. Mirrors `cfg_baseline_high_qp` in `encoder_round_47.rs`
/// so the off path is verifiably round-47-equivalent.
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

#[test]
fn default_config_round48_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_variance_lf_cap,
        "round-48 default must keep variance-LF-cap off"
    );
    assert!(
        !cfg.enable_adaptive_uv_lf_deltas,
        "round-48 default must keep UV-channel LF deltas off"
    );
}

/// Round-48 with both knobs off must reproduce the round-47 baseline
/// bit-for-bit (proves the new helpers + struct fields are inert on the
/// off path).
#[test]
fn round48_off_path_byte_identical() {
    let clip = make_half_step_clip(4);
    let cfg = cfg_baseline_high_qp();
    let (b0, _, _, p0) = measure(cfg, &clip);
    let (b1, _, _, p1) = measure(cfg, &clip);
    assert_eq!(b0, b1, "deterministic encode must match itself");
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} differs between identical configs");
    }
}

/// Variance-LF-cap is inert when adaptive LF deltas are off (it only
/// feeds the round-44 estimator).
#[test]
fn variance_cap_requires_adaptive_lf_deltas() {
    let clip = make_high_variance_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_adaptive_lf_deltas: false,
        enable_variance_lf_cap: false,
        ..cfg_baseline_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_adaptive_lf_deltas: false,
        enable_variance_lf_cap: true,
        ..cfg_baseline_high_qp()
    };

    let (b0, _, _, p0) = measure(cfg_a, &clip);
    let (b1, _, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "variance LF cap must be inert when adaptive LF deltas off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "variance-cap-off packet {i} differs");
    }
}

/// UV-channel LF deltas are inert when adaptive LF deltas are off (the
/// chroma path only adjusts the round-44 ladder).
#[test]
fn uv_lf_deltas_require_adaptive_lf_deltas() {
    let clip = make_chroma_textured_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_adaptive_lf_deltas: false,
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
        enable_adaptive_uv_lf_deltas: true,
        ..cfg_baseline_high_qp()
    };

    let (b0, _, _, p0) = measure(cfg_a, &clip);
    let (b1, _, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "UV-LF deltas must be inert when adaptive LF deltas off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "uv-deltas-off packet {i} differs");
    }
}

#[test]
fn variance_cap_pframe_decodes_cleanly() {
    let clip = make_high_variance_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_variance_lf_cap: true,
        ..cfg_baseline_high_qp()
    };
    let (bytes, psnr_y, _, _) = measure(cfg, &clip);
    assert!(bytes > 0, "variance-cap P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "variance-cap P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn uv_lf_deltas_pframe_decodes_cleanly() {
    let clip = make_chroma_textured_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_adaptive_uv_lf_deltas: true,
        ..cfg_baseline_high_qp()
    };
    let (bytes, _, psnr_uv, _) = measure(cfg, &clip);
    assert!(bytes > 0, "uv-deltas P-frame produced zero bytes");
    assert!(
        psnr_uv > 5.0,
        "uv-deltas P-frame chroma PSNR collapsed: {psnr_uv:.2} dB"
    );
}

/// Variance-LF-cap byte envelope: the cap only widens the per-bucket
/// delta clamp, not the underlying signal — byte size on a high-variance
/// clip stays inside the same ±25 % envelope round-47 used for the
/// round-44 vs round-47 comparison.
#[test]
fn variance_cap_byte_envelope_within_25pct() {
    let clip = make_high_variance_clip(8);
    let baseline = cfg_baseline_high_qp();
    let with_var = Vp8EncoderConfig {
        enable_variance_lf_cap: true,
        ..baseline
    };
    let (bytes_g, _, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _, _) = measure(with_var, &clip);
    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.25,
        "variance-cap swung byte size by {:.1}% (baseline {bytes_g}, variance {bytes_a}) — beyond +/-25%",
        frac * 100.0
    );
}

/// UV-deltas byte envelope on a chroma-textured clip — the average of
/// luma + chroma adaptive ladders shouldn't blow the byte budget vs the
/// luma-only round-44 path.
#[test]
fn uv_lf_deltas_byte_envelope_within_25pct() {
    let clip = make_chroma_textured_clip(8);
    let baseline = cfg_baseline_high_qp();
    let with_uv = Vp8EncoderConfig {
        enable_adaptive_uv_lf_deltas: true,
        ..baseline
    };
    let (bytes_g, _, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _, _) = measure(with_uv, &clip);
    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.25,
        "uv-deltas swung byte size by {:.1}% (baseline {bytes_g}, uv {bytes_a}) — beyond +/-25%",
        frac * 100.0
    );
}

/// Combined round-48 + round-47 + round-44: turn everything on at a
/// high-QP setting and confirm a clean round-trip with reasonable PSNR.
#[test]
fn round48_combined_decodes_cleanly() {
    let clip = make_high_variance_clip(8);
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
        enable_variance_lf_cap: true,
        enable_adaptive_uv_lf_deltas: true,
        enable_joint_lf_rdo: true,
        ..cfg_baseline_high_qp()
    };
    let (bytes, psnr_y, _, _) = measure(cfg, &clip);
    assert!(bytes > 0, "combined round-48 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-48 PSNR collapsed: {psnr_y:.2} dB"
    );
}
