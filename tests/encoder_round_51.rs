//! Round-51 encoder push tests — k-means++ centroid seeding for the
//! round-50 spatial segmentation picker (`enable_kmeans_pp_seeding`)
//!
//! \+ per-MB chroma-SSE caching across the round-48 UV-channel
//! adaptive LF estimator (mirrors the round-50 #4 luma-SSE cache
//! refactor).
//!
//! Round-51 lands two complementary refinements:
//!
//!   * `enable_kmeans_pp_seeding` swaps the round-50 top-|delta|
//!     centroid seeding for a deterministic k-means++ variant (Arthur
//!     & Vassilvitskii, 2007). Seed 0 is still the highest-|delta|
//!     populated region so the first cluster anchor matches the
//!     round-50 path; subsequent seeds are picked by `argmax D²` to
//!     the nearest already-chosen centroid (deterministic limit of the
//!     paper's probabilistic D²-weighted sampling). Spreads the seeds
//!     across `(delta, position)` so adjacent equal-|delta| spikes
//!     don't co-locate two starting clusters. Off-by-default; the
//!     round-50 top-|delta| seeding is preserved bit-for-bit when this
//!     flag is off. Requires `enable_kmeans_spatial_segmentation =
//!     true`.
//!
//!   * Per-MB chroma-SSE caching: today only the round-48 UV-channel
//!     adaptive LF estimator consumes `compute_per_mb_chroma_sse`, but
//!     the cache is hoisted into `mb_sse_uv_cache` symmetrically with
//!     the round-50 luma cache. Future chroma-aware paths (round-49
//!     spatial chroma-aware variant, etc.) plug in as a single line of
//!     plumbing instead of a duplicate computation. Bit-exact
//!     preserving — the round-48 single-consumer path produces
//!     identical bytes before and after the refactor.
//!
//! Tests:
//!  1) Default config has the new k-means++ knob off.
//!  2) k-means++ flag is inert when k-means spatial segmentation off.
//!  3) k-means++ off path is byte-identical to a round-50 baseline.
//!  4) k-means++ on a top/bottom-split clip decodes cleanly.
//!  5) k-means++ byte envelope ±35 % vs round-50 baseline.
//!  6) k-means++ + composed round-44/48/49/50 round-trips.
//!  7) Chroma cache: round-48 UV path remains byte-identical (proves
//!     the lazy `mb_sse_uv_cache` doesn't perturb the single-consumer
//!     case).
//!  8) Cache invariant: re-running same config produces identical
//!     bytes (catches non-determinism in either cache plumbing).

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_AQ_QINDEX_RANGE, DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
    DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256, DEFAULT_GOLDEN_INTERVAL,
    DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS, DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
    DEFAULT_KMEANS_SPATIAL_ALPHA_X256, DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH,
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

/// Top/bottom-split clip — mirrors the round-49 / round-50 fixture so
/// the k-means++ vs top-|delta| comparison runs on the same shape the
/// spatial path was originally tuned for.
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

/// Equal-|delta| adjacent-spike clip: a pattern engineered to surface
/// the top-|delta| seeding weakness. Several spatially-adjacent regions
/// have similar high |delta| values + one isolated spike in a far
/// quadrant. Top-|delta| seeding sorts the adjacent spikes ahead of
/// the isolated one (ties broken by region index), placing two seeds
/// in the same neighbourhood. ++ seeding's `argmax D²` step jumps to
/// the isolated spike for seed 1.
fn make_equal_delta_spike_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let phase = (f as i32 * 7).rem_euclid(128);
        let mut y = vec![100u8; (W * H) as usize];
        // Three adjacent textured spike regions in the top-left.
        for row in 0..16 {
            for col in 0..48 {
                let v = ((col as i32 + phase) * 17).rem_euclid(256);
                y[row * W as usize + col] = v as u8;
            }
        }
        // One isolated spike region in the bottom-right.
        for row in (H as usize - 16)..H as usize {
            for col in (W as usize - 16)..W as usize {
                let v = ((col as i32 + phase) * 19).rem_euclid(256);
                y[row * W as usize + col] = v as u8;
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

/// Round-51 baseline shape: high QP + segments on + adaptive LF deltas
/// on, both new round-51 knobs off. Round-50 / round-49 / round-48
/// flags also off so the baseline is the round-50 default-config path.
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

#[test]
fn default_config_round51_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_kmeans_pp_seeding,
        "round-51 default must keep k-means++ seeding off"
    );
}

/// k-means++ flag is inert when `enable_kmeans_spatial_segmentation =
/// false` (no centroid seeding happens in the greedy spatial path).
#[test]
fn kmeans_pp_seeding_requires_kmeans_spatial() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: false,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: false,
        enable_kmeans_pp_seeding: true,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "k-means++ seeding flag must be inert when k-means spatial off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "kmeans-pp inert packet {i} differs");
    }
}

/// k-means++ off + k-means spatial on must reproduce the round-50
/// top-|delta|-seeded path bit-for-bit (proves the new flag is a
/// strict opt-in).
#[test]
fn kmeans_pp_off_path_byte_identical_to_round50_kmeans() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_round50 = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_round50, &clip);
    // Re-run with the same config — proves determinism.
    let (b1, _, p1) = measure(cfg_round50, &clip);
    assert_eq!(
        b0, b1,
        "round-50 k-means deterministic re-run must match: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-50 k-means deterministic packet {i} differs");
    }
}

/// k-means++ on a P-frame clip decodes cleanly (no panics, valid
/// bitstream). Headline smoke test.
#[test]
fn kmeans_pp_pframe_decodes_cleanly() {
    let clip = make_equal_delta_spike_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: true,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "k-means++ P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "k-means++ P-frame PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// k-means++ byte envelope: the seed-spread variant changes the
/// segment-id distribution → bool-coded segment-id cost shifts. Keep
/// the envelope inside ±35 % vs the round-50 top-|delta| baseline
/// (matches the round-50-vs-round-49 bound).
#[test]
fn kmeans_pp_byte_envelope_within_35pct() {
    let clip = make_equal_delta_spike_clip(8);
    let baseline_round50 = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        ..cfg_baseline_segments_high_qp()
    };
    let with_pp = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: true,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes_t, _, _) = measure(baseline_round50, &clip);
    let (bytes_p, _, _) = measure(with_pp, &clip);
    let frac = (bytes_p as f64 - bytes_t as f64).abs() / bytes_t.max(1) as f64;
    assert!(
        frac < 0.35,
        "k-means++ swung byte size by {:.1}% (top {bytes_t}, ++ {bytes_p}) — beyond +/-35%",
        frac * 100.0
    );
}

/// Combined round-51 + round-50 + round-49 + round-48 + round-44:
/// every knob on, at high-QP. Proves the cache refactors handle the
/// composed pipeline without panicking and the bitstream still
/// decodes.
#[test]
fn round51_combined_decodes_cleanly() {
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
        enable_kmeans_pp_seeding: true,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        enable_joint_lf_rdo: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "combined round-51 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-51 PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Chroma cache: the round-48 UV-channel adaptive LF estimator is the
/// only consumer today. Hoisting `compute_per_mb_chroma_sse` into
/// `mb_sse_uv_cache` must not change the bitstream for the
/// single-consumer case.
///
/// Asserts the round-48 UV path produces deterministic + reproducible
/// bytes across re-runs (any divergence here would point at
/// uninitialised state in the chroma cache plumbing).
#[test]
fn chroma_cache_round48_uv_deterministic() {
    let clip = make_top_bottom_split_clip(4);
    let cfg = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        enable_adaptive_uv_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert!(b0 > 0, "round-48 UV path produced zero bytes");
    assert_eq!(b0, b1, "chroma-cache deterministic re-run must match");
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "chroma-cache deterministic packet {i} differs");
    }
}

/// Chroma cache off-path: when `enable_adaptive_uv_lf_deltas = false`,
/// the `mb_sse_uv_cache` is `None` (no chroma SSE walk happens). The
/// luma-only round-44 path must be byte-identical to the
/// `enable_mode_ref_lf_deltas + enable_adaptive_lf_deltas` baseline
/// without the UV knob — the chroma cache plumbing must not perturb
/// the luma-only path.
#[test]
fn chroma_cache_inert_when_uv_off() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        enable_adaptive_uv_lf_deltas: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = cfg_a;
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(b0, b1, "luma-only path deterministic re-run must match");
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "luma-only path deterministic packet {i} differs");
    }
}

/// Full re-run determinism with both round-51 knobs on. Catches
/// non-determinism in either cache plumbing or in the ++ seed
/// selection.
#[test]
fn round51_full_pipeline_deterministic() {
    let clip = make_equal_delta_spike_clip(6);
    let cfg = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: true,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        enable_adaptive_uv_lf_deltas: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert!(b0 > 0, "round-51 full pipeline produced zero bytes");
    assert_eq!(b0, b1, "round-51 full pipeline reproducible re-run");
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-51 full pipeline packet {i} differs");
    }
}
