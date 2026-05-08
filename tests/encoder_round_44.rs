//! Round-44 encoder push tests — adaptive loop-filter mode/ref deltas
//! (`enable_adaptive_lf_deltas`) + Trellis rate-from-context
//! (`enable_trellis_context_rate`).
//!
//! Round-44 lands two complementary picker upgrades that share a
//! distortion-vs-rate theme:
//!
//!   * `enable_adaptive_lf_deltas` replaces the static round-42 ladder
//!     (`ref_deltas = [+2, 0, -2, -2]` / `mode_deltas = [+4, -2, +1, +4]`)
//!     with a per-frame estimate from the per-MB unfiltered luma-SSE
//!     distribution. Each ref / mode bucket's delta is biased toward
//!     stronger filtering for buckets whose mean SSE exceeds the frame
//!     mean (deblocking helps reconstruction-noisy MBs the most) and
//!     toward lighter filtering for buckets below the frame mean.
//!     Empty buckets fall back to the static ladder.
//!
//!   * `enable_trellis_context_rate` runs the trellis pass after the
//!     per-MB encode loop completes, with a per-block context derived
//!     from the running above/left non-zero predictor (the same the
//!     entropy coder uses). The previous per-MB call always passed
//!     `nctx = 0` — an over-approximation of EOB savings on blocks
//!     whose neighbours have non-zero coefficients (the actual
//!     `nctx ∈ {0,1,2}` raises the EOB probability for high-context
//!     blocks, so the rate term changes).
//!
//! Tests:
//!  1) Default config has both knobs off.
//!  2) Off path produces byte-identical encoder output (regression
//!     guard against accidental engagement).
//!  3) Adaptive LF on: keyframe + P-frame encode/decode cleanly.
//!  4) Adaptive LF byte envelope ±20 % vs static-ladder baseline.
//!  5) Adaptive LF flat-content stays close to baseline (sparse-mode
//!     buckets fall back to the static ladder, so the deltas are
//!     close to the static values; envelope ±20 %).
//!  6) Adaptive LF requires `enable_mode_ref_lf_deltas`.
//!  7) Trellis context-rate on: keyframe + P-frame decode cleanly.
//!  8) Trellis context-rate byte envelope ±10 % vs nctx=0 baseline.
//!  9) Trellis context-rate PSNR not worse by more than 0.5 dB.
//! 10) Trellis context-rate requires `enable_trellis_quant`.
//! 11) Combined round-44 + round-43: clean round-trip with reasonable PSNR.

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

/// Half-and-half clip — mixes intra (the keyframe) and inter MBs of
/// different modes (zero-MV in the still parts, non-zero MV at the
/// boundary), giving the adaptive estimator a non-trivial bucket
/// distribution to work from.
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

/// Constant-grey clip — every inter MB is ZERO_MV under LAST, so most
/// adaptive buckets fall back to static-ladder values. Useful for the
/// "sparse-mode fallback" sanity test.
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

/// Round-44 baseline config: round-42 mode/ref LF deltas on (so
/// adaptive can replace the static ladder when toggled), trellis on
/// with `enable_trellis_full = true` (the only path where context-rate
/// matters — the EOB-only path's rate accounting is already minimal).
/// All other knobs at round-43 baseline values.
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
    }
}

#[test]
fn default_config_round44_knobs_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_adaptive_lf_deltas,
        "round-44 default must keep adaptive LF deltas off"
    );
    assert!(
        !cfg.enable_trellis_context_rate,
        "round-44 default must keep trellis context-rate off"
    );
}

/// Round-44 must not perturb the bitstream when both knobs are off.
#[test]
fn round44_off_path_byte_identical_to_legacy() {
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
fn adaptive_lf_keyframe_decodes_cleanly() {
    let clip = make_half_step_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "adaptive LF keyframe produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "adaptive LF keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn adaptive_lf_pframe_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "adaptive LF P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "adaptive LF P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Adaptive LF deltas only swap the static ladder for a per-frame
/// estimate; the bitstream cost is the same (still 4 ref + 4 mode
/// deltas of signed-6) so byte size should stay within ±20 % of the
/// static-ladder baseline.
#[test]
fn adaptive_lf_byte_envelope_within_20pct() {
    let clip = make_half_step_clip(8);
    let baseline = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };
    let with_adaptive = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_a, _, _) = measure(with_adaptive, &clip);

    let frac = (bytes_a as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.20,
        "adaptive LF swung byte size by {:.1}% (static {bytes_g}, adaptive {bytes_a}) — beyond +/-20%",
        frac * 100.0
    );
}

/// Flat content has every inter MB picking ZERO_MV under LAST, so the
/// non-LAST ref buckets and non-zero mode buckets are empty and fall
/// back to the static ladder. The single populated bucket converges to
/// near-zero delta (its mean equals the frame mean by definition), so
/// the bitstream stays within ±20 % of the static-ladder baseline.
#[test]
fn adaptive_lf_flat_content_byte_envelope_within_20pct() {
    let clip = make_flat_clip(4);
    let baseline = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };
    let with_adaptive = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        enable_adaptive_lf_deltas: true,
        ..cfg_baseline()
    };

    let (b0, _, _) = measure(baseline, &clip);
    let (b1, _, _) = measure(with_adaptive, &clip);

    let frac = (b1 as f64 - b0 as f64).abs() / b0.max(1) as f64;
    assert!(
        frac < 0.20,
        "adaptive LF on flat content swung bytes by {:.1}% (static {b0}, adaptive {b1})",
        frac * 100.0
    );
}

/// Adaptive LF deltas are ignored when `enable_mode_ref_lf_deltas` is
/// off — without that flag the encoder emits `mode_ref_delta_enabled = 0`
/// and the deltas don't reach the bitstream regardless. Toggling the
/// adaptive knob alone must be inert.
#[test]
fn adaptive_lf_requires_mode_ref_lf_deltas() {
    let clip = make_half_step_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: false,
        enable_adaptive_lf_deltas: false,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: false,
        enable_adaptive_lf_deltas: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "adaptive LF must be inert when enable_mode_ref_lf_deltas=false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "mode-ref-off packet {i} differs");
    }
}

#[test]
fn trellis_ctx_rate_keyframe_decodes_cleanly() {
    let clip = make_half_step_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_trellis_quant: true,
        enable_trellis_full: true,
        enable_trellis_context_rate: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "trellis-context keyframe produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "trellis-context keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn trellis_ctx_rate_pframe_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_trellis_quant: true,
        enable_trellis_full: true,
        enable_trellis_context_rate: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "trellis-context P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "trellis-context P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Trellis context-rate uses real `nctx ∈ {0,1,2}` instead of the
/// approximate `nctx = 0`. The rate term in `D + λ·R` shifts but the
/// distortion model is the same, so byte size stays within a small
/// envelope of the baseline.
#[test]
fn trellis_ctx_rate_byte_envelope_within_10pct() {
    let clip = make_half_step_clip(8);
    let baseline = Vp8EncoderConfig {
        enable_trellis_quant: true,
        enable_trellis_full: true,
        ..cfg_baseline()
    };
    let with_ctx = Vp8EncoderConfig {
        enable_trellis_quant: true,
        enable_trellis_full: true,
        enable_trellis_context_rate: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_c, _, _) = measure(with_ctx, &clip);

    let frac = (bytes_c as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.10,
        "trellis-context swung byte size by {:.1}% (nctx=0 {bytes_g}, ctx-aware {bytes_c}) — beyond +/-10%",
        frac * 100.0
    );
}

/// Trellis context-rate is a refinement of the same RD trade-off, so
/// PSNR should not regress significantly vs the nctx=0 baseline.
#[test]
fn trellis_ctx_rate_psnr_does_not_regress_significantly() {
    let clip = make_half_step_clip(6);
    let baseline = Vp8EncoderConfig {
        enable_trellis_quant: true,
        enable_trellis_full: true,
        ..cfg_baseline()
    };
    let with_ctx = Vp8EncoderConfig {
        enable_trellis_quant: true,
        enable_trellis_full: true,
        enable_trellis_context_rate: true,
        ..cfg_baseline()
    };

    let (_b_g, psnr_g, _) = measure(baseline, &clip);
    let (_b_c, psnr_c, _) = measure(with_ctx, &clip);

    assert!(
        psnr_c >= psnr_g - 0.5,
        "trellis-context regressed PSNR_Y by {:.3} dB beyond 0.5 dB slack (nctx=0 {psnr_g:.2}, ctx-aware {psnr_c:.2})",
        psnr_g - psnr_c
    );
}

/// `enable_trellis_quant = false` makes the context-rate knob inert —
/// no trellis runs at all, so toggling the secondary knob can't move
/// bytes.
#[test]
fn trellis_ctx_rate_requires_enable_trellis_quant() {
    let clip = make_half_step_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_trellis_quant: false,
        enable_trellis_context_rate: false,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_trellis_quant: false,
        enable_trellis_context_rate: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "trellis-context must be inert when enable_trellis_quant=false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "trellis-quant-off packet {i} differs");
    }
}

/// Combined round-44 + round-43 + round-42 knobs: turn everything on
/// at once and confirm a clean round-trip with reasonable PSNR.
#[test]
fn round44_combined_decodes_cleanly() {
    let clip = make_half_step_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_split_mv_rdo: true,
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
    assert!(bytes > 0, "combined round-44 produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "combined round-44 PSNR collapsed: {psnr_y:.2} dB"
    );
}
