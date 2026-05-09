//! Round-40 encoder push tests — joint loop-filter / QP rate-distortion
//! optimisation (`enable_joint_lf_rdo`).
//!
//! When `enable_joint_lf_rdo = true`, the per-frame loop-filter level on
//! P-frames is picked from a ±4-level neighbourhood around the heuristic
//! `15 + qi/8` by scoring each candidate's luma SSE on a centre 32×32
//! patch. Since the LF level is a 6-bit literal in the frame header, the
//! rate term is identical for every candidate — the search reduces to
//! pure distortion minimisation. Picks the level that best preserves
//! source structure under the post-filter reconstruction.
//!
//! Tests:
//!  1) Keyframe-only encode must be bit-identical with/without the flag
//!     (LF-RDO is P-frame only).
//!  2) P-frame encode produces a valid, decodable bitstream that
//!     round-trips with plausible PSNR.
//!  3) PSNR_Y must not regress vs the heuristic-only path on a clip where
//!     the heuristic is already a good pick.
//!  4) Default config has `enable_joint_lf_rdo = false`.

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
    DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS, DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH,
    DEFAULT_SIMPLE_LF_MAX_LEVEL,
};

const W: u32 = 64;
const H: u32 = 64;
const QINDEX: u8 = 50;

// ─── helpers ──────────────────────────────────────────────────────────────────

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

/// Smooth-pan + slight-jitter clip — gives the LF level something to bite
/// on (P-frame reference drift creates blockiness on edges that the LF
/// removes; the RDO pass picks the best level for the given content).
fn make_pan_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        // Diagonal gradient + per-frame phase shift (slow pan).
        for row in 0..H as usize {
            for col in 0..W as usize {
                let v = ((row as i32 + col as i32 + (f as i32) * 2) * 3).rem_euclid(255);
                y[row * W as usize + col] = v as u8;
            }
        }
        // Sprinkle of high-frequency noise on the right edge so the LF has
        // something to deblock at MB boundaries.
        for row in 0..H as usize {
            for col in (W as usize / 2)..(W as usize) {
                let mut h: u32 = (row as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add((col as u32).wrapping_mul(40503))
                    .wrapping_add(f as u32 * 17);
                h ^= h >> 13;
                let n = (h & 0x1f) as i32;
                let cur = y[row * W as usize + col] as i32;
                y[row * W as usize + col] = (cur + n - 16).clamp(0, 255) as u8;
            }
        }
        out.push(make_frame(y, vec![128u8; cw * ch], vec![128u8; cw * ch]));
    }
    out
}

/// Encode a clip and return `(total_bytes, avg_psnr_y, raw_packets)`.
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
        enable_kmeans_pp_seeding: false,
        enable_joint_r44r49_picker: false,
        joint_r44r49_picker_max_iters: DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
        enable_chroma_aware_spatial: false,
        chroma_aware_spatial_luma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
        chroma_aware_spatial_chroma_weight_x256: DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
    }
}

// ─── joint LF-RDO tests ───────────────────────────────────────────────────────

/// LF-RDO is P-frame-only by spec — keyframe encoding must produce a
/// byte-identical bitstream with the flag flipped. This pins that a stray
/// future implementation cannot accidentally engage the RDO search on
/// keyframes (which would disturb every existing keyframe-only consumer
/// that toggles unrelated knobs).
#[test]
fn joint_lf_rdo_keyframe_only_is_byte_identical() {
    let clip = make_pan_clip(1); // single keyframe
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_joint_lf_rdo: true,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(baseline, &clip);
    let (b1, _, p1) = measure(with_rdo, &clip);

    assert_eq!(
        b0, b1,
        "keyframe-only encode byte total must not change with LF-RDO toggle: {b0} vs {b1}"
    );
    assert_eq!(
        p0.len(),
        p1.len(),
        "keyframe-only encode packet count must not change"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(
            a, b,
            "keyframe-only encode packet {i} differs between baseline and LF-RDO toggle"
        );
    }
}

/// LF-RDO produces a valid bitstream that decodes through the in-tree
/// decoder with plausible PSNR. The point is functional — we're not
/// pinning a precise byte/PSNR delta because the LF level swings on
/// content-dependent SSE minima, but the encoder must converge.
#[test]
fn joint_lf_rdo_pframe_bitstream_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_joint_lf_rdo: true,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "LF-RDO produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "LF-RDO encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// The LF-RDO path searches around the heuristic and picks the SSE-minimum
/// level — by construction it cannot produce strictly worse luma fidelity
/// than the heuristic on the test patch. Whole-frame PSNR (which scores
/// the entire frame, not just the centre 32×32 patch) typically tracks
/// the patch metric, so we pin "PSNR_Y not worse than heuristic by more
/// than a small slack" rather than "strictly better".
#[test]
fn joint_lf_rdo_psnr_does_not_regress_significantly() {
    let clip = make_pan_clip(8);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_joint_lf_rdo: true,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        ..cfg_baseline()
    };

    let (bytes_h, psnr_h, _) = measure(baseline, &clip);
    let (bytes_r, psnr_r, _) = measure(with_rdo, &clip);

    eprintln!(
        "heuristic: {bytes_h} bytes, PSNR_Y {psnr_h:.2} dB\n\
         lf-rdo:    {bytes_r} bytes, PSNR_Y {psnr_r:.2} dB"
    );

    // Slack for the difference between centre-patch SSE (the RDO target)
    // and whole-frame PSNR (the test metric). LF level is shared across
    // the frame, so picking the centre-patch optimum can be slightly off
    // for the periphery; budget 0.5 dB.
    assert!(
        psnr_r >= psnr_h - 0.5,
        "LF-RDO regressed PSNR_Y by {:.3} dB beyond 0.5 dB slack (heuristic {psnr_h:.2}, rdo {psnr_r:.2})",
        psnr_h - psnr_r
    );
}

/// The RDO search must never explode the bitstream — picking a wider LF
/// level can change the reconstruction's reference for subsequent
/// P-frames, and a misbehaving search could in principle produce a
/// catastrophically over- or under-filtered loop that drifts further
/// each P-frame. Pin the byte total within ±15% of the heuristic
/// baseline (the RDO is a small per-frame perturbation — large drifts
/// indicate a bug).
#[test]
fn joint_lf_rdo_byte_budget_within_envelope() {
    let clip = make_pan_clip(10);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_joint_lf_rdo: true,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        ..cfg_baseline()
    };

    let (bytes_h, _, _) = measure(baseline, &clip);
    let (bytes_r, _, _) = measure(with_rdo, &clip);

    let frac = (bytes_r as f64 - bytes_h as f64).abs() / bytes_h.max(1) as f64;
    assert!(
        frac < 0.15,
        "LF-RDO swung byte size by {:.1}% (heuristic {bytes_h}, rdo {bytes_r}) — beyond +/-15% envelope",
        frac * 100.0
    );
}

/// The RDO behaviour combines cleanly with segmentation (where per-MB LF
/// deltas alter the effective per-MB filter level on top of the
/// frame-wide level). Pin that the combined path produces a valid,
/// decodable stream.
#[test]
fn joint_lf_rdo_with_segmentation_decodes_cleanly() {
    let clip = make_pan_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_segments: true,
        segment_quant_deltas: [-8, -4, 0, 4],
        segment_lf_deltas: [-2, -1, 0, 2],
        enable_joint_lf_rdo: true,
        enable_bpred_rdo: false,
        enable_uv_rdo: false,
        enable_mode_ref_lf_deltas: false,
        enable_split_mv_rdo: false,
        enable_adaptive_lf_deltas: false,
        enable_trellis_context_rate: false,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "LF-RDO + segments produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "LF-RDO + segments PSNR collapsed: {psnr_y:.2} dB"
    );
}

// ─── default config regression ────────────────────────────────────────────────

#[test]
fn default_config_lf_rdo_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_joint_lf_rdo,
        "round-40 default must be off (preserves bit-exact heuristic path)"
    );
}
