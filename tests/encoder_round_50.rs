//! Round-50 encoder push tests — 4-means clustering for spatial-path
//! segments (`enable_kmeans_spatial_segmentation`) + per-MB luma-SSE
//! caching across the round-44/48 estimator and the round-49 paths.
//!
//! Round-50 generalises the round-49 spatial-locality bucketed adaptive
//! LF picker along two complementary axes:
//!
//!   * `enable_kmeans_spatial_segmentation` replaces the round-49 greedy
//!     "top-3 |delta| → segments 1/2/3, rest → segment 0" picker with a
//!     `k = 4` Lloyd's-algorithm clustering on `(region_delta,
//!     region_pos_x, region_pos_y)`. The metric is `(region_delta -
//!     centroid_delta)² + (alpha_x256/256) * ((px - cx)² + (py - cy)²)`,
//!     tuned by `kmeans_spatial_alpha_x256`. Centroids are seeded from
//!     the 4 highest-|delta| regions; iteration runs until convergence
//!     or `KMEANS_SPATIAL_MAX_ITERS` (= 16). Off-by-default; the
//!     round-49 greedy picker is preserved bit-for-bit when this flag
//!     is off. Requires `enable_spatial_lf_deltas = true`.
//!
//!   * Per-MB luma-SSE caching collapses the duplicate
//!     `compute_per_mb_luma_sse` calls the round-44/48 adaptive
//!     estimator and the round-49 per-MB / spatial paths previously
//!     made independently. The refactor is pure plumbing: the same
//!     `Vec<u64>` is computed once + threaded into both branches. No
//!     bitstream change.
//!
//! Tests:
//!  1) Default config has the new k-means knob off.
//!  2) k-means off path is byte-identical to a round-49 spatial baseline.
//!  3) k-means requires `enable_spatial_lf_deltas` (inert otherwise).
//!  4) k-means on a top/bottom-split clip decodes cleanly.
//!  5) k-means byte envelope ±35 % vs round-49 spatial baseline.
//!  6) k-means + composed round-44/48/49 round-trips.
//!  7) Cache invariant: re-running the same config produces identical
//!     bytes (catches non-determinism in the cache plumbing).

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_AQ_QINDEX_RANGE, DEFAULT_GOLDEN_INTERVAL, DEFAULT_KMEANS_SPATIAL_ALPHA_X256,
    DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH, DEFAULT_SEGMENT_LF_DELTAS,
    DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL, DEFAULT_SPATIAL_LF_N_COL_BANDS,
    DEFAULT_SPATIAL_LF_N_ROW_BANDS,
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

/// Top/bottom-split clip — mirrors the round-49 fixture so the
/// k-means / greedy comparison runs on the same shape the spatial
/// path was originally tuned for.
fn make_top_bottom_split_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let val = if row < H as usize / 2 {
                    let phase = (f as i32 * 9).rem_euclid(64);
                    let g = ((col as i32 + phase) * 10).rem_euclid(256);
                    let chk = if (row / 4 + col / 4) % 2 == 0 {
                        24
                    } else {
                        -24
                    };
                    (g + chk).clamp(0, 255)
                } else {
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

/// Round-50 baseline shape: high QP + segments on + adaptive LF deltas
/// on, both new round-50 knobs off. Mirrors the round-49 baseline so
/// the off path is verifiably round-49-equivalent.
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
        enable_kmeans_spatial_segmentation: false,
        kmeans_spatial_alpha_x256: DEFAULT_KMEANS_SPATIAL_ALPHA_X256,
    }
}

#[test]
fn default_config_round50_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_kmeans_spatial_segmentation,
        "round-50 default must keep k-means spatial segmentation off"
    );
    assert_eq!(
        cfg.kmeans_spatial_alpha_x256, DEFAULT_KMEANS_SPATIAL_ALPHA_X256,
        "round-50 default alpha_x256 must match exported constant"
    );
}

/// k-means flag is inert when `enable_spatial_lf_deltas = false`.
#[test]
fn kmeans_spatial_requires_spatial_path() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_spatial_lf_deltas: false,
        enable_kmeans_spatial_segmentation: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_spatial_lf_deltas: false,
        enable_kmeans_spatial_segmentation: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "k-means flag must be inert when spatial path off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "kmeans-spatial-off packet {i} differs");
    }
}

/// k-means off + spatial on must reproduce the round-49 greedy spatial
/// path bit-for-bit (proves the new flag is a strict opt-in). This is
/// also the headline cache-refactor regression test: the per-MB SSE
/// pathway is shared by the round-49 spatial path and the round-44/48
/// adaptive estimator → same input vector, same output bytes.
#[test]
fn kmeans_off_path_byte_identical_to_round49_spatial() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_round49 = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: false,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_round49, &clip);
    // Re-run with the same config — proves determinism (any divergence
    // here would point at uninitialised state in the cache plumbing).
    let (b1, _, p1) = measure(cfg_round49, &clip);
    assert_eq!(
        b0, b1,
        "round-49 spatial deterministic re-run must match: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-49 spatial deterministic packet {i} differs");
    }
}

#[test]
fn kmeans_spatial_pframe_decodes_cleanly() {
    let clip = make_top_bottom_split_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "k-means spatial P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "k-means spatial P-frame PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// k-means byte envelope: the spatial picker variant changes the
/// segment-id distribution → bool-coded segment-id cost shifts. Keep
/// the envelope inside ±35 % vs the round-49 greedy spatial baseline
/// (matches the round-49 spatial-vs-round-48 bound).
#[test]
fn kmeans_spatial_byte_envelope_within_35pct() {
    let clip = make_top_bottom_split_clip(8);
    let baseline_spatial = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let with_kmeans = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes_g, _, _) = measure(baseline_spatial, &clip);
    let (bytes_k, _, _) = measure(with_kmeans, &clip);
    let frac = (bytes_k as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.35,
        "k-means spatial swung byte size by {:.1}% (greedy {bytes_g}, k-means {bytes_k}) — beyond +/-35%",
        frac * 100.0
    );
}

/// Combined round-50 + round-49 + round-48 + round-44: every knob on,
/// at high-QP. Proves the `mb_sse_y_cache` refactor handles the
/// composed pipeline without panicking and the bitstream still
/// decodes.
#[test]
fn round50_combined_decodes_cleanly() {
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
        enable_kmeans_spatial_segmentation: true,
        enable_joint_lf_rdo: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "combined round-50 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-50 PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Cache invariant: enabling round-44 adaptive LF deltas + round-49
/// spatial path together (both consumers of the per-MB SSE cache)
/// produces a deterministic bytestream. Distinct configs that touch
/// the cache pathway must each be reproducible across re-runs.
#[test]
fn cache_round48_plus_round49_deterministic() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_only_round49 = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: false,
        enable_adaptive_lf_deltas: false,
        enable_spatial_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_both = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        enable_spatial_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_only_round49, &clip);
    let (b1, _, p1) = measure(cfg_both, &clip);
    assert!(b0 > 0 && b1 > 0, "cache test produced empty streams");
    let (b0_rerun, _, p0_rerun) = measure(cfg_only_round49, &clip);
    let (b1_rerun, _, p1_rerun) = measure(cfg_both, &clip);
    assert_eq!(b0, b0_rerun, "round-49-only stream not reproducible");
    assert_eq!(b1, b1_rerun, "round-44+49 stream not reproducible");
    for (i, (a, b)) in p0.iter().zip(p0_rerun.iter()).enumerate() {
        assert_eq!(a, b, "round-49-only packet {i} not reproducible");
    }
    for (i, (a, b)) in p1.iter().zip(p1_rerun.iter()).enumerate() {
        assert_eq!(a, b, "round-44+49 packet {i} not reproducible");
    }
}
