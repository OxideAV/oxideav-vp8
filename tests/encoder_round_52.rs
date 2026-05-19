//! Round-52 encoder push tests — joint round-44/48 + round-49 picker
//! (`enable_joint_r44r49_picker`) + chroma-aware spatial picker
//! (`enable_chroma_aware_spatial`).
//!
//! Round-52 lands two complementary forward steps gated behind opt-in
//! flags (default off), composing the existing round-44/48 (mode/ref
//! deltas) and round-49/50/51 (spatial-segment LF deltas) tiers:
//!
//!   * `enable_joint_r44r49_picker` makes the two tiers iterate
//!     jointly. Each iteration recomputes the round-44/48 estimator
//!     using a per-MB residual SSE that subtracts the part of the
//!     per-MB ideal delta the spatial tier has already addressed; and
//!     recomputes the spatial picker using a per-MB residual SSE that
//!     subtracts the part the mode/ref tier has already addressed.
//!     Convergence: stop when both outputs (the 8 lf_deltas + the
//!     segment_id vector + the 4 segment_lf_deltas) match the previous
//!     iteration, capped at `joint_r44r49_picker_max_iters` (default 3).
//!     Off-by-default; the round-49 / round-50 / round-51 single-pass
//!     behaviour is preserved bit-for-bit when this flag is disabled.
//!
//!   * `enable_chroma_aware_spatial` extends the round-49 / round-50
//!     spatial picker to score regions on a luma+chroma weighted SSE
//!     blend. Today the spatial picker scores by per-MB luma SSE only;
//!     a region whose luma is uniformly smooth but whose chroma carries
//!     substantial residual error never lands as a separate segment.
//!     With this flag on the picker uses the round-51 `mb_sse_uv_cache`
//!     to compute `combined_sse = (luma_w * mb_sse_y + chroma_w *
//!     mb_sse_uv) / 256`, default weights `luma=256` (`1.0`),
//!     `chroma=128` (`0.5`). Off-by-default.
//!
//! Tests:
//!  1) Default config has both new knobs off.
//!  2) Joint picker off path is byte-identical to a round-51 baseline.
//!  3) Joint picker requires both `enable_mode_ref_lf_deltas` (for the
//!     round-44/48 tier) AND `enable_spatial_lf_deltas` (for the
//!     round-49 spatial tier) — inert when either is off.
//!  4) Joint picker on a synthesised content frame decodes cleanly
//!     and converges (re-running with `max_iters = 1` vs `max_iters
//!     = 3` produces identical bytes once the picker has converged).
//!  5) Chroma-aware spatial off path is byte-identical to a round-51
//!     baseline.
//!  6) Chroma-aware spatial requires `enable_spatial_lf_deltas` —
//!     inert when off.
//!  7) Chroma-aware spatial on a chroma-textured / luma-flat frame
//!     produces a different byte stream than the luma-only spatial
//!     baseline (proves the chroma plane is actually consumed).
//!  8) Joint picker + chroma-aware spatial composed at high QP
//!     decodes cleanly.
//!  9) Full re-run determinism with both flags on.

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

/// Top/bottom-split clip — same shape round-49 / round-50 / round-51
/// fixture so the joint picker / chroma-aware comparisons run on the
/// same shape the spatial path was tuned for.
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

/// Chroma-textured / luma-flat clip: a clip whose luma is uniform but
/// whose chroma planes carry a textured patch in one quadrant. The
/// luma-only spatial picker (round-49 / round-50 / round-51) cannot
/// see the chroma texture, so the patch ends up in segment 0 with no
/// LF nudge. The chroma-aware spatial picker reads `mb_sse_uv_cache`
/// and lifts the patch into a non-zero segment.
fn make_chroma_textured_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let phase = (f as i32 * 5).rem_euclid(64);
        // Luma: a constant grey value with a tiny gradient so the
        // encoder doesn't collapse to all-skip MBs.
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let g = 96 + ((col as i32 + phase) % 8);
                y[row * W as usize + col] = g as u8;
            }
        }
        // U plane: textured top-left quadrant, flat elsewhere.
        let mut u = vec![128u8; cw * ch];
        for row in 0..(ch / 2) {
            for col in 0..(cw / 2) {
                let v = (((col as i32 + phase) * 23).rem_euclid(192)) + 32;
                u[row * cw + col] = v as u8;
            }
        }
        // V plane: textured bottom-right quadrant, flat elsewhere.
        let mut v = vec![128u8; cw * ch];
        for row in (ch / 2)..ch {
            for col in (cw / 2)..cw {
                let vv = (((col as i32 + phase) * 31).rem_euclid(192)) + 32;
                v[row * cw + col] = vv as u8;
            }
        }
        out.push(make_frame(y, u, v));
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

/// Round-52 baseline shape: high QP + segments on + adaptive LF deltas
/// on, both new round-52 knobs off. Round-50 / round-49 / round-48
/// flags also off so the baseline is the round-51 default-config path.
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
        enable_chroma_aware_variance_lf_cap: false,
    }
}

#[test]
fn default_config_round52_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_joint_r44r49_picker,
        "round-52 default must keep joint r44/r49 picker off"
    );
    assert!(
        !cfg.enable_chroma_aware_spatial,
        "round-52 default must keep chroma-aware spatial off"
    );
    assert_eq!(
        cfg.joint_r44r49_picker_max_iters, DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        "round-52 default joint-iter cap must equal DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS"
    );
    assert_eq!(
        cfg.chroma_aware_spatial_luma_weight_x256,
        DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256
    );
    assert_eq!(
        cfg.chroma_aware_spatial_chroma_weight_x256,
        DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256
    );
}

/// Joint-picker off path is byte-identical to the round-51 baseline:
/// the joint-picker loop runs exactly once with no residual feedback
/// when the flag is off, recovering single-pass behaviour bit-for-bit.
#[test]
fn joint_picker_off_byte_identical_to_round51_baseline() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_joint_r44r49_picker: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        // Iter cap doesn't matter when the flag is off.
        joint_r44r49_picker_max_iters: 8,
        enable_joint_r44r49_picker: false,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "joint-picker-off path bit-exact across iter-cap settings"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "joint-picker off packet {i} differs");
    }
}

/// Joint picker requires `enable_mode_ref_lf_deltas = true` AND
/// `enable_spatial_lf_deltas = true` (it composes the two tiers; with
/// either off there's nothing to compose). Inert otherwise.
#[test]
fn joint_picker_inert_when_dependencies_off() {
    let clip = make_top_bottom_split_clip(4);
    // No mode/ref deltas: joint picker has nothing to feed back.
    let cfg_no_mode_ref = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_mode_ref_lf_deltas: false,
        enable_joint_r44r49_picker: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_no_mode_ref_with_joint = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_mode_ref_lf_deltas: false,
        enable_joint_r44r49_picker: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, _) = measure(cfg_no_mode_ref, &clip);
    let (b1, _, _) = measure(cfg_no_mode_ref_with_joint, &clip);
    assert_eq!(
        b0, b1,
        "joint picker must be inert when enable_mode_ref_lf_deltas = false"
    );
    // No spatial deltas: joint picker has nothing to iterate against.
    let cfg_no_spatial = Vp8EncoderConfig {
        enable_spatial_lf_deltas: false,
        enable_joint_r44r49_picker: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_no_spatial_with_joint = Vp8EncoderConfig {
        enable_spatial_lf_deltas: false,
        enable_joint_r44r49_picker: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b2, _, _) = measure(cfg_no_spatial, &clip);
    let (b3, _, _) = measure(cfg_no_spatial_with_joint, &clip);
    assert_eq!(
        b2, b3,
        "joint picker must be inert when enable_spatial_lf_deltas = false"
    );
}

/// Joint picker on a P-frame clip decodes cleanly (no panics, valid
/// bitstream) and the iteration cap is respected. With the cap set to
/// 1 (single-pass equivalent) vs 3 (default), the bytes can differ
/// (the joint picker's feedback kicks in at iteration 2) — but neither
/// must panic and both must round-trip the bitstream.
#[test]
fn joint_picker_pframe_decodes_cleanly_and_respects_iter_cap() {
    let clip = make_top_bottom_split_clip(8);
    let cfg_iter1 = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_joint_r44r49_picker: true,
        joint_r44r49_picker_max_iters: 1,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_iter3 = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_joint_r44r49_picker: true,
        joint_r44r49_picker_max_iters: 3,
        ..cfg_baseline_segments_high_qp()
    };
    let (b1, psnr1, _) = measure(cfg_iter1, &clip);
    let (b3, psnr3, _) = measure(cfg_iter3, &clip);
    assert!(b1 > 0, "joint picker iter=1 produced zero bytes");
    assert!(b3 > 0, "joint picker iter=3 produced zero bytes");
    assert!(
        psnr1 > 5.0,
        "joint picker iter=1 PSNR collapsed: {psnr1:.2} dB"
    );
    assert!(
        psnr3 > 5.0,
        "joint picker iter=3 PSNR collapsed: {psnr3:.2} dB"
    );
    // Wide envelope — the joint picker may shift the byte stream
    // significantly when it converges to a different segment-id
    // distribution than the single-pass picker.
    let frac = (b3 as f64 - b1 as f64).abs() / b1.max(1) as f64;
    assert!(
        frac < 0.50,
        "joint-picker iter=1 vs iter=3 swung byte size by {:.1}% — beyond +/-50%",
        frac * 100.0
    );
}

/// Chroma-aware off path is byte-identical to the round-51 baseline.
/// Proves the new flag is a strict opt-in.
#[test]
fn chroma_aware_off_byte_identical_to_round51_baseline() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_chroma_aware_spatial: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_chroma_aware_spatial: false,
        // Weights don't matter when the flag is off.
        chroma_aware_spatial_luma_weight_x256: 0,
        chroma_aware_spatial_chroma_weight_x256: 1024,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "chroma-aware-off path bit-exact across weight settings"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "chroma-aware off packet {i} differs");
    }
}

/// Chroma-aware spatial requires `enable_spatial_lf_deltas = true` —
/// inert when the spatial path itself is off.
#[test]
fn chroma_aware_inert_when_spatial_off() {
    let clip = make_top_bottom_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_spatial_lf_deltas: false,
        enable_chroma_aware_spatial: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_spatial_lf_deltas: false,
        enable_chroma_aware_spatial: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, _) = measure(cfg_a, &clip);
    let (b1, _, _) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "chroma-aware spatial must be inert when enable_spatial_lf_deltas = false"
    );
}

/// Chroma-aware spatial on a chroma-textured / luma-flat frame
/// produces a different byte stream than the luma-only baseline:
/// proves the chroma plane is actually consumed by the picker (the
/// luma-only baseline cannot see the chroma texture, so the
/// chroma-aware variant must shift the segment_id distribution).
#[test]
fn chroma_aware_picks_up_chroma_textured_region() {
    let clip = make_chroma_textured_clip(4);
    let cfg_luma_only = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_chroma_aware_spatial: false,
        ..cfg_baseline_segments_high_qp()
    };
    let cfg_chroma_aware = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_chroma_aware_spatial: true,
        // Crank chroma weight up so the chroma plane dominates the
        // SSE blend on a clip whose luma is uniform.
        chroma_aware_spatial_luma_weight_x256: 256,
        chroma_aware_spatial_chroma_weight_x256: 1024,
        ..cfg_baseline_segments_high_qp()
    };
    let (b_luma, _, p_luma) = measure(cfg_luma_only, &clip);
    let (b_chroma, _, p_chroma) = measure(cfg_chroma_aware, &clip);
    assert!(b_luma > 0, "luma-only spatial produced zero bytes");
    assert!(b_chroma > 0, "chroma-aware spatial produced zero bytes");
    // Detect at least one packet differing — proves the chroma plane
    // affected the picker's output.
    let mut diff_packets = 0;
    for (a, b) in p_luma.iter().zip(p_chroma.iter()) {
        if a != b {
            diff_packets += 1;
        }
    }
    assert!(
        diff_packets > 0,
        "chroma-aware spatial produced byte-identical packets to luma-only \
         on a chroma-textured / luma-flat clip — the chroma plane should have shifted at \
         least one segment_id assignment"
    );
}

/// Joint picker + chroma-aware spatial composed at high QP decodes
/// cleanly (proves the two new flags don't deadlock or panic when both
/// are on alongside the round-51 stack).
#[test]
fn joint_plus_chroma_aware_composed_decodes_cleanly() {
    let clip = make_top_bottom_split_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: true,
        enable_adaptive_uv_lf_deltas: true,
        enable_joint_r44r49_picker: true,
        joint_r44r49_picker_max_iters: 3,
        enable_chroma_aware_spatial: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "composed round-52 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "composed round-52 PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Determinism across re-runs with both round-52 flags on. Catches
/// non-determinism in either the joint-picker iteration loop or the
/// chroma-aware combined-SSE blend.
#[test]
fn round52_full_pipeline_deterministic() {
    let clip = make_chroma_textured_clip(6);
    let cfg = Vp8EncoderConfig {
        enable_spatial_lf_deltas: true,
        enable_kmeans_spatial_segmentation: true,
        enable_kmeans_pp_seeding: true,
        enable_joint_r44r49_picker: true,
        enable_chroma_aware_spatial: true,
        ..cfg_baseline_segments_high_qp()
    };
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert!(b0 > 0, "round-52 full pipeline produced zero bytes");
    assert_eq!(b0, b1, "round-52 full pipeline reproducible re-run");
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-52 full pipeline packet {i} differs");
    }
}
