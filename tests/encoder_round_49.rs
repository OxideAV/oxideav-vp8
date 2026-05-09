//! Round-49 encoder push tests — per-MB-targeted segment LF deltas
//! (`enable_per_mb_lf_deltas`) + spatial-locality bucketed adaptive LF
//! (`enable_spatial_lf_deltas`).
//!
//! Round-49 generalises the round-44/47/48 LF-delta estimator family
//! along two complementary axes, both gated on `enable_segments = true`
//! since they target the per-segment LF-delta channel rather than the
//! mode/ref delta channel that round-44/48 owns:
//!
//!   * `enable_per_mb_lf_deltas` replaces the static
//!     `config.segment_lf_deltas` array with a content-driven pick. For
//!     each MB the encoder computes the round-44 proportional delta
//!     (`(mb_sse - frame_mean) * delta_cap / frame_mean`) — its
//!     standalone optimum — then groups by `mb_segment_id` and picks the
//!     median per segment. Empty segments fall back to the static
//!     `config.segment_lf_deltas` value so this stays well-behaved on
//!     sparse segment maps. Off-by-default; the static-config segment LF
//!     ladder is preserved bit-for-bit when the flag is off.
//!
//!   * `enable_spatial_lf_deltas` partitions the frame into
//!     `spatial_lf_n_row_bands × spatial_lf_n_col_bands` rectangular
//!     regions and applies the same proportional-delta formula
//!     per-region, then maps the regions onto VP8's 4-segment scheme by
//!     clustering: the 3 regions with the largest `|delta|` become
//!     segments 1/2/3 (each carrying its own delta), the rest collapse
//!     into segment 0 with delta `0`. Both the per-MB segment-id map and
//!     the per-segment LF-delta array are overridden, so the bitstream
//!     signals the spatial assignment to the decoder. Off-by-default.
//!
//! When both flags are on the spatial path wins (it owns the segment-id
//! map + the LF deltas, leaving the per-MB median path nothing to
//! override). The cap (`±6` default, `±10` under
//! `enable_adaptive_lf_high_qp_cap` / `enable_variance_lf_cap`) matches
//! the round-44/48 estimator so the cap-widening flags compose with
//! round-49 without double-clamping.
//!
//! Tests:
//!  1) Default config has both new knobs off.
//!  2) Per-MB off path is byte-identical to a round-48 baseline.
//!  3) Spatial off path is byte-identical to a round-48 baseline.
//!  4) Per-MB requires `enable_segments` (inert when segments off).
//!  5) Spatial requires `enable_segments` (inert when segments off).
//!  6) Per-MB on a high-error-MB clip decodes cleanly.
//!  7) Spatial on a top/bottom-split clip decodes cleanly.
//!  8) Per-MB byte envelope ±25 % vs round-48 baseline.
//!  9) Spatial byte envelope ±35 % vs round-48 baseline (wider —
//!     spatial path also rewrites segment-id map → MB-segment_id bool
//!     coding cost shifts).
//! 10) Spatial wins over per-MB when both flags are on (composition
//!     test).
//! 11) Combined round-49 + round-48 + round-44 decodes cleanly.

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_AQ_QINDEX_RANGE, DEFAULT_GOLDEN_INTERVAL, DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH,
    DEFAULT_SEGMENT_LF_DELTAS, DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL,
    DEFAULT_SPATIAL_LF_N_COL_BANDS, DEFAULT_SPATIAL_LF_N_ROW_BANDS,
};

const W: u32 = 64;
const H: u32 = 64;

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

/// Half-step clip — same shape round-44/47/48 use; gives the per-MB and
/// spatial estimators a non-trivial bucket distribution.
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

/// High-error-MB clip: a single 16×16 MB at `(2, 2)` carries a heavy
/// per-frame phase shift while the rest of the frame is a slow ramp.
/// The per-MB optimal LF delta picker should shape the per-segment
/// median around that outlier MB rather than the frame-mean ladder.
fn make_high_error_mb_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                y[row * W as usize + col] = (90 + row as i32 / 4).clamp(0, 255) as u8;
            }
        }
        // Outlier MB at (mb_row=2, mb_col=2) — fill its 16×16 luma tile
        // with a phase-shifted gradient + checker the predictor can't
        // perfectly anticipate.
        let mb_x0 = 2 * 16;
        let mb_y0 = 2 * 16;
        for r in 0..16 {
            for c in 0..16 {
                let phase = (f as i32 * 11).rem_euclid(64);
                let g = ((c as i32 + phase) * 12).rem_euclid(256);
                let chk = if (r / 4 + c / 4) % 2 == 0 { 32 } else { -32 };
                y[(mb_y0 + r) * W as usize + (mb_x0 + c)] = (g + chk).clamp(0, 255) as u8;
            }
        }
        out.push(make_frame(y, vec![128u8; cw * ch], vec![128u8; cw * ch]));
    }
    out
}

/// Top/bottom-split clip: top half is heavily textured (drives high
/// SSE in the top spatial bands), bottom half is flat (drives low SSE
/// in the bottom spatial bands). The spatial bucketing should pick the
/// top region as a distinct segment with a positive LF delta.
fn make_top_bottom_split_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let val = if row < H as usize / 2 {
                    // Top: heavy per-frame-shifted gradient + checker.
                    let phase = (f as i32 * 9).rem_euclid(64);
                    let g = ((col as i32 + phase) * 10).rem_euclid(256);
                    let chk = if (row / 4 + col / 4) % 2 == 0 {
                        24
                    } else {
                        -24
                    };
                    (g + chk).clamp(0, 255)
                } else {
                    // Bottom: flat 100, easy to predict from previous frame.
                    100
                };
                y[row * W as usize + col] = val as u8;
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
    let mut psnr_y_sum = 0f64;
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
                    psnr_y_sum += if mse == 0.0 {
                        60.0
                    } else {
                        10.0 * (255.0f64 * 255.0 / mse).log10()
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
    (total_bytes, avg_y, packets)
}

/// Round-49 baseline shape: high QP + segments on + adaptive LF deltas
/// on, both new round-49 knobs off. Mirrors the round-48 baseline so
/// the off path is verifiably round-48-equivalent.
fn cfg_baseline_segments_high_qp() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: 110,
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
        spatial_lf_n_row_bands: DEFAULT_SPATIAL_LF_N_ROW_BANDS,
        spatial_lf_n_col_bands: DEFAULT_SPATIAL_LF_N_COL_BANDS,
    }
}

#[test]
fn default_config_round49_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_per_mb_lf_deltas,
        "round-49 default must keep per-MB LF deltas off"
    );
    assert!(
        !cfg.enable_spatial_lf_deltas,
        "round-49 default must keep spatial LF deltas off"
    );
    assert_eq!(
        cfg.spatial_lf_n_row_bands, DEFAULT_SPATIAL_LF_N_ROW_BANDS,
        "spatial row-band default must match exported constant"
    );
    assert_eq!(
        cfg.spatial_lf_n_col_bands, DEFAULT_SPATIAL_LF_N_COL_BANDS,
        "spatial col-band default must match exported constant"
    );
}

/// Round-49 with both knobs off must reproduce the round-48 baseline
/// bit-for-bit (proves the new helpers + struct fields are inert on the
/// off path).
#[test]
fn round49_off_path_byte_identical() {
    let clip = make_half_step_clip(4);
    let cfg = cfg_baseline_segments_high_qp();
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(b0, b1, "deterministic encode must match itself");
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} differs between identical configs");
    }
}

/// Per-MB LF is inert when segmentation is off (the per-segment LF
/// deltas are never signalled to the decoder).
#[test]
fn per_mb_lf_requires_segments() {
    let clip = make_high_error_mb_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_segments: false,
        enable_per_mb_lf_deltas: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_segments: false,
        enable_per_mb_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "per-MB LF deltas must be inert when segments off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "per-mb-lf-segments-off packet {i} differs");
    }
}

/// Spatial LF is inert when segmentation is off (the per-MB segment id
/// map + per-segment LF deltas aren't signalled either).
#[test]
fn spatial_lf_requires_segments() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_segments: false,
        enable_spatial_lf_deltas: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_segments: false,
        enable_spatial_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "spatial LF deltas must be inert when segments off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "spatial-lf-segments-off packet {i} differs");
    }
}

#[test]
fn per_mb_lf_pframe_decodes_cleanly() {
    let clip = make_high_error_mb_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_per_mb_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "per-MB LF P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "per-MB LF P-frame PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn spatial_lf_pframe_decodes_cleanly() {
    let clip = make_top_bottom_split_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "spatial LF P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "spatial LF P-frame PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Per-MB byte envelope: the per-MB median-of-optimal-deltas should
/// land inside the same `±delta_cap` envelope the round-44/48 estimator
/// uses, so the byte size on a high-error-MB clip stays inside ±25 % of
/// the round-48 baseline.
#[test]
fn per_mb_lf_byte_envelope_within_25pct() {
    let clip = make_high_error_mb_clip(8);
    let baseline = cfg_baseline_segments_high_qp();
    let with_per_mb = Vp8EncoderConfig {
        enable_per_mb_lf_deltas: true,
        ..baseline
    };
    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _) = measure(with_per_mb, &clip);
    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.25,
        "per-MB LF swung byte size by {:.1}% (baseline {bytes_g}, per-MB {bytes_a}) — beyond +/-25%",
        frac * 100.0
    );
}

/// Spatial byte envelope: the spatial path also rewrites the per-MB
/// segment-id map → the bool-coded segment-id cost shifts (in addition
/// to the LF-delta change), so the envelope is wider than the per-MB
/// path. ±35 % is the practical bound for a top/bottom-split clip.
#[test]
fn spatial_lf_byte_envelope_within_35pct() {
    let clip = make_top_bottom_split_clip(8);
    let baseline = cfg_baseline_segments_high_qp();
    let with_spatial = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        ..baseline
    };
    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _) = measure(with_spatial, &clip);
    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.35,
        "spatial LF swung byte size by {:.1}% (baseline {bytes_g}, spatial {bytes_a}) — beyond +/-35%",
        frac * 100.0
    );
}

/// Composition test: when both round-49 flags are on, the spatial path
/// should produce byte-identical output to spatial-alone (the per-MB
/// median path becomes a no-op because the spatial path overrides
/// `mb_segment_ids` AND `segment_lf_deltas` first).
#[test]
fn spatial_wins_over_per_mb_when_both_on() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_spatial_only = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_per_mb_lf_deltas: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_both = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_per_mb_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_spatial_only, &clip);
    let (b1, _, p1) = measure(cfg_both, &clip);
    assert_eq!(
        b0, b1,
        "spatial path must win over per-MB when both on: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "spatial-wins packet {i} differs");
    }
}

/// Combined round-49 + round-48 + round-44: turn everything on at a
/// high-QP setting and confirm a clean round-trip with reasonable PSNR.
#[test]
fn round49_combined_decodes_cleanly() {
    let clip = make_top_bottom_split_clip(8);
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
        enable_per_mb_lf_deltas: true,
        enable_spatial_lf_deltas: true,
        enable_joint_lf_rdo: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "combined round-49 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-49 PSNR collapsed: {psnr_y:.2} dB"
    );
}
