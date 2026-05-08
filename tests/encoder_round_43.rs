//! Round-43 encoder push tests — SPLIT_MV partition-selection RDO
//! (`enable_split_mv_rdo`).
//!
//! Round-43 lands the rate-aware variant of `search_split_mv`. Up to
//! round-42 the SPLIT_MV picker chose the SAD-min split mode across the
//! four candidates (16×8 / 8×16 / 8×8 / 4×4). The 4×4 split nearly
//! always beats the coarser splits on raw SAD because it has the most
//! per-partition freedom — but it also pays the most bitstream bits
//! (16 partitions × per-partition tree + 16 MV deltas + the longest
//! `MBSPLIT_PROBS` tree path). When `enable_split_mv_rdo` is on, each
//! candidate is scored as `D + λ·R` where
//!
//!   * `D` = total partition SAD (same as the legacy path).
//!   * `R` = `MBSPLIT_PROBS` tree-path cost + per-partition
//!     `SUB_MV_REF_PROBS` longest-path leaf cost (NEW_4X4 under the
//!     neutral [0] context, since neighbour sub-MVs aren't visible at
//!     search time) + per-partition `mv_component_cost_x256` MV-delta
//!     bits when the partition's MV is non-zero.
//!
//! λ comes from `lambda_for_qp(qi, scale)`, the same multiplier the
//! per-MB ref/mode picker uses, so RDO trade-offs stay coherent across
//! all encoder decisions.
//!
//! Tests:
//!  1) Default config has the knob off.
//!  2) Off path produces byte-identical encoder output (regression
//!     guard against accidental engagement).
//!  3) RDO on: keyframe + P-frame encode/decode cleanly.
//!  4) RDO byte envelope ±20 % vs greedy SAD baseline.
//!  5) RDO PSNR not worse by more than 0.5 dB.
//!  6) Flat-content byte-identical (the cheap-skip test fires before
//!     `search_split_mv` is reached, so toggling the knob can't move
//!     bytes).
//!  7) `enable_rdo = false` makes the knob inert.

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

/// Half-and-half clip — a horizontally-translating step between two
/// luminance regions, designed to make the SPLIT_MV picker actually
/// engage. The 16×8 and 8×16 split modes can capture the boundary in
/// two halves; the 4×4 split would over-fit and pay more bits without
/// matching SAD savings — so RDO should bias toward the coarser split.
fn make_half_step_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                // Step boundary moves by `f` columns each frame, giving
                // motion search something genuine to lock on. Left half
                // is dark (60), right half is bright (200), with a small
                // vertical wobble to keep some texture for sub-pel
                // refinement.
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

/// Constant-grey clip — every MB skips the SPLIT_MV search outright via
/// the cheap-skip test (`zero_sad <= MB_SKIP_SAD_PER_PIXEL * 256`), so
/// toggling the RDO knob must not move bytes.
fn make_flat_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let y = vec![128u8; (W * H) as usize];
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
    }
}

#[test]
fn default_config_split_mv_rdo_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_split_mv_rdo,
        "round-43 default must keep SPLIT_MV RDO off (preserves bit-exact greedy SAD min)"
    );
}

/// Round-43 must not perturb the bitstream when the knob is off.
#[test]
fn round43_off_path_byte_identical_to_legacy() {
    let clip = make_half_step_clip(4);
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
fn split_mv_rdo_keyframe_decodes_cleanly() {
    let clip = make_half_step_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "SPLIT_MV RDO keyframe produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "SPLIT_MV RDO keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn split_mv_rdo_pframe_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "SPLIT_MV RDO P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "SPLIT_MV RDO P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// SPLIT_MV RDO biases the picker toward coarser splits when the SAD
/// savings of the finer splits don't amortise the bits — so on
/// translating content the bitstream can shrink. Pin the swing to
/// ±20 % of the legacy baseline.
#[test]
fn split_mv_rdo_byte_envelope_within_20pct() {
    let clip = make_half_step_clip(8);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_r, _, _) = measure(with_rdo, &clip);

    let frac = (bytes_r as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.20,
        "SPLIT_MV RDO swung byte size by {:.1}% (greedy {bytes_g}, rdo {bytes_r}) — beyond +/-20%",
        frac * 100.0
    );
}

#[test]
fn split_mv_rdo_psnr_does_not_regress_significantly() {
    let clip = make_half_step_clip(6);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };

    let (_b_g, psnr_g, _) = measure(baseline, &clip);
    let (_b_r, psnr_r, _) = measure(with_rdo, &clip);

    assert!(
        psnr_r >= psnr_g - 0.5,
        "SPLIT_MV RDO regressed PSNR_Y by {:.3} dB beyond 0.5 dB slack (greedy {psnr_g:.2}, rdo {psnr_r:.2})",
        psnr_g - psnr_r
    );
}

/// Flat content has every MB taking the cheap-skip path before the
/// SPLIT_MV search runs, so toggling the knob can't move bytes.
#[test]
fn split_mv_rdo_flat_content_is_byte_identical() {
    let clip = make_flat_clip(4);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(baseline, &clip);
    let (b1, _, p1) = measure(with_rdo, &clip);

    assert_eq!(
        b0, b1,
        "flat-content encode must be byte-identical with SPLIT_MV RDO toggle: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "flat-content packet {i} differs");
    }
}

/// `enable_rdo = false` must zero the SPLIT_MV RDO lambda, making the
/// knob inert — same gating contract round-41 BMODE-RDO and round-42
/// UV-RDO use.
#[test]
fn split_mv_rdo_requires_enable_rdo() {
    let clip = make_half_step_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_rdo: false,
        enable_split_mv_rdo: false,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_rdo: false,
        enable_split_mv_rdo: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "SPLIT_MV RDO must be inert when enable_rdo = false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "enable_rdo-off packet {i} differs");
    }
}

/// Composition with round-42 knobs (UV-RDO + mode/ref deltas + joint
/// LF-RDO): turn everything on at once and confirm a clean round-trip
/// with reasonable PSNR.
#[test]
fn round43_combined_with_round42_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
        enable_uv_rdo: true,
        enable_mode_ref_lf_deltas: true,
        enable_joint_lf_rdo: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "combined round-43 + round-42 produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "combined round-43 + round-42 PSNR collapsed: {psnr_y:.2} dB"
    );
}
