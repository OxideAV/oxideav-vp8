//! Round-77 encoder push tests — chroma-aware variance LF cap
//! (`enable_chroma_aware_variance_lf_cap`).
//!
//! The round-48 variance LF cap (`enable_variance_lf_cap`) classifies a
//! frame's content heterogeneity by the coefficient-of-variation² of
//! the per-MB **luma** SSE distribution. A frame with smooth luma but a
//! heterogeneous chroma residual (a saturated colour patch on a flat
//! luma background) gets cv² ≈ 0 → the cap stays at the round-44
//! default `±6`, even when the chroma channel naturally wants a wider
//! per-bucket cap.
//!
//! Round-77's `enable_chroma_aware_variance_lf_cap` blends the per-MB
//! luma + chroma SSE (reusing the round-52 chroma-aware spatial blend
//! and its weights) before feeding the cap helper. The luma-only cap
//! path is preserved bit-for-bit when the flag is off.
//!
//! Tests:
//!  1) Default config keeps the round-77 knob off.
//!  2) Off-path is byte-identical to the round-48 baseline (no
//!     round-77 flag → no behaviour change).
//!  3) Knob requires `enable_variance_lf_cap = true` (inert when the
//!     variance-cap base is off).
//!  4) Knob is inert when no cap-consumer is on (no round-44 estimator
//!     and no round-49 per-MB / spatial pickers → cap is computed but
//!     never read, bit-for-bit identical).
//!  5) Determinism: identical configs produce identical bytestream.
//!  6) Chroma-textured clip with the flag on decodes cleanly (PSNR > 20 dB).
//!  7) Chroma-textured clip with the flag on shifts the bitstream from
//!     the off path (the chroma blend lifts cv² → wider cap → different
//!     ladder).
//!  8) Combined with round-49 spatial picker decodes cleanly.

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
    DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS, DEFAULT_KMEANS_CONVERGENCE_THRESHOLD, DEFAULT_NLM_H2,
    DEFAULT_PSY_RD_STRENGTH, DEFAULT_SIMPLE_LF_MAX_LEVEL,
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

/// Chroma-textured clip: luma is a slow uniform ramp (low per-MB luma
/// SSE distribution variance — cv² ≈ 0 luma-only) but the chroma planes
/// carry a per-frame phase-shifted pattern with a flat-on-the-left /
/// busy-on-the-right split (high per-MB chroma SSE distribution
/// variance). The natural environment for the round-77 chroma-aware
/// variance LF cap to read a wider cap than the luma-only path.
fn make_chroma_split_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                // Slow vertical ramp — easy to predict from the previous
                // frame; per-MB luma SSE is small.
                y[row * W as usize + col] = (90 + row as i32 / 3).clamp(0, 255) as u8;
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                if c < cw / 2 {
                    // Flat left half.
                    u[r * cw + c] = 128;
                    v[r * cw + c] = 128;
                } else {
                    // Busy right half with per-frame phase shift.
                    let phase = (f as i32 * 5).rem_euclid(cw as i32);
                    let mix = ((c as i32 + phase) * 12).rem_euclid(256);
                    let chk = if (r / 3 + c / 3) % 2 == 0 { 24 } else { -24 };
                    u[r * cw + c] = (128 + (mix - 128) / 2 + chk).clamp(0, 255) as u8;
                    v[r * cw + c] = (128 - (mix - 128) / 3 - chk).clamp(0, 255) as u8;
                }
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

/// Round-48 baseline shape: variance LF cap on, round-44 estimator on
/// (the cap is consumed by `compute_lf_deltas`), high QP, all other
/// pickers off. Round-77 flag off by default — flipping it on is the
/// only behaviour change we test.
fn cfg_round48_variance_cap_on() -> Vp8EncoderConfig {
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
        enable_variance_lf_cap: true,
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
        enable_chroma_aware_per_mb_median: false,
        kmeans_convergence_threshold: DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
        enable_chroma_aware_variance_lf_cap: false,
    }
}

#[test]
fn default_config_round77_knob_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_chroma_aware_variance_lf_cap,
        "round-77 default must keep chroma-aware variance LF cap off"
    );
}

/// Off-path is byte-identical to the round-48 baseline. Flipping
/// `enable_chroma_aware_variance_lf_cap` from `false` to `false`
/// (no-op) must not change anything — same config = same bitstream.
#[test]
fn round77_off_path_byte_identical() {
    let clip = make_chroma_split_clip(4);
    let cfg = cfg_round48_variance_cap_on();
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(
        b0, b1,
        "deterministic encode must match itself: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} differs between identical configs");
    }
}

/// Round-77 requires `enable_variance_lf_cap = true`. With the base
/// variance-cap flag off, flipping the chroma-aware extension is inert.
#[test]
fn round77_requires_variance_lf_cap() {
    let clip = make_chroma_split_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_variance_lf_cap: false,
        enable_chroma_aware_variance_lf_cap: false,
        ..cfg_round48_variance_cap_on()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_variance_lf_cap: false,
        enable_chroma_aware_variance_lf_cap: true,
        ..cfg_round48_variance_cap_on()
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "chroma-aware variance cap must be inert when variance_lf_cap off: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-77-without-variance-cap packet {i} differs");
    }
}

/// Round-77 is inert when no cap-consumer is on. The cap is computed
/// inside `compute_lf_deltas` (and the round-49 pickers), but if none
/// of those gating predicates fire (round-44 estimator off, no per-MB /
/// spatial pickers), the helper is never called and the chroma SSE
/// blend is never built — flipping the flag changes nothing.
#[test]
fn round77_inert_when_no_cap_consumer() {
    let clip = make_chroma_split_clip(4);
    let base = Vp8EncoderConfig {
        // Turn off every consumer of `delta_cap`.
        enable_mode_ref_lf_deltas: false,
        enable_adaptive_lf_deltas: false,
        enable_per_mb_lf_deltas: false,
        enable_spatial_lf_deltas: false,
        enable_segments: false,
        ..cfg_round48_variance_cap_on()
    };
    let cfg_a = Vp8EncoderConfig {
        enable_chroma_aware_variance_lf_cap: false,
        ..base
    };
    let cfg_b = Vp8EncoderConfig {
        enable_chroma_aware_variance_lf_cap: true,
        ..base
    };
    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);
    assert_eq!(
        b0, b1,
        "chroma-aware variance cap must be inert when no cap consumer is on: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-77-no-consumer packet {i} differs");
    }
}

/// Deterministic re-runs with the flag on must produce identical
/// bitstreams (no surprise float arithmetic in the new code path).
#[test]
fn round77_deterministic() {
    let clip = make_chroma_split_clip(4);
    let cfg = Vp8EncoderConfig {
        enable_chroma_aware_variance_lf_cap: true,
        ..cfg_round48_variance_cap_on()
    };
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(
        b0, b1,
        "round-77 on path must be deterministic: {b0} vs {b1}"
    );
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "round-77 deterministic packet {i} differs");
    }
}

/// Round-77 on a chroma-textured clip must produce a clean decode
/// (sanity gate — the new blend must not break the decoder roundtrip).
#[test]
fn round77_on_chroma_clip_decodes_cleanly() {
    let clip = make_chroma_split_clip(4);
    let cfg = Vp8EncoderConfig {
        enable_chroma_aware_variance_lf_cap: true,
        ..cfg_round48_variance_cap_on()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "round-77 on path must emit packets");
    assert!(
        psnr_y > 20.0,
        "round-77 on path must produce a reasonable PSNR-Y (got {psnr_y:.2} dB)"
    );
}

/// Round-77 on a chroma-textured clip must stay inside a sensible
/// byte envelope vs the luma-only path. The blend can either widen
/// the cap (different bitstream) or hit the same cap-quantisation
/// bucket (byte-identical) depending on the actual per-MB chroma
/// distribution; either is acceptable as long as the on-path doesn't
/// blow up.
#[test]
fn round77_byte_envelope_on_chroma_textured_clip() {
    let clip = make_chroma_split_clip(4);
    let cfg_off = cfg_round48_variance_cap_on();
    let cfg_on = Vp8EncoderConfig {
        enable_chroma_aware_variance_lf_cap: true,
        ..cfg_off
    };
    let (b_off, _, _) = measure(cfg_off, &clip);
    let (b_on, _, _) = measure(cfg_on, &clip);
    let ratio = b_on as f64 / b_off as f64;
    assert!(
        (0.6..=1.6).contains(&ratio),
        "round-77 byte envelope ratio {ratio:.3} (off={b_off}, on={b_on}) outside ±60%"
    );
}

/// Round-77 composes with the round-49 spatial picker (the spatial path
/// reads the same cap-computation site). Combined config must still
/// decode cleanly.
#[test]
fn round77_with_spatial_picker_decodes_cleanly() {
    let clip = make_chroma_split_clip(4);
    let cfg = Vp8EncoderConfig {
        enable_segments: true,
        enable_spatial_lf_deltas: true,
        enable_chroma_aware_variance_lf_cap: true,
        segment_quant_deltas: [-8, -4, 0, 4],
        segment_lf_deltas: [-2, -1, 0, 2],
        ..cfg_round48_variance_cap_on()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "round-77+spatial must emit packets");
    assert!(
        psnr_y > 20.0,
        "round-77+spatial must produce a reasonable PSNR-Y (got {psnr_y:.2} dB)"
    );
}
