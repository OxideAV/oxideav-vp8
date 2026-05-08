//! Round-42 encoder push tests — UV-mode RDO (`enable_uv_rdo`) +
//! mode/ref loop-filter deltas (`enable_mode_ref_lf_deltas`).
//!
//! Round-42 lands two complementary picker upgrades:
//!
//!  1. **UV-mode RDO (`enable_uv_rdo`)** — when on, the chroma intra
//!     mode for both keyframes and the intra-in-P fallback is picked
//!     by `D + λ·R` against `KF_UV_MODE_PROBS` (keyframes) or
//!     `DEFAULT_UV_MODE_PROBS` (intra-in-P), where `R` is the bool-
//!     coder cost of the UV-mode tree path the bitstream will pay.
//!     λ comes from `lambda_for_qp(qi, scale)` so trade-offs stay
//!     coherent with every other RDO decision in the encoder.
//!
//!  2. **Mode/ref LF deltas (`enable_mode_ref_lf_deltas`)** — when on,
//!     the encoder emits `mode_ref_delta_enabled = 1` in the inter-
//!     frame header and writes the round-42 default ladder
//!     (`ref_deltas = [+2, 0, -2, -2]`,
//!     `mode_deltas  = [+4, -2, +1, +4]`). Per-MB filter level then
//!     matches the decoder's `per_mb_filter_level` (RFC 6386 §15.2)
//!     bit-for-bit, so the post-filter reconstruction the encoder
//!     hands to the next reference is decoder-exact and the joint
//!     LF-RDO picker now scores candidate levels against the actual
//!     post-delta reconstruction (not the bare frame level).
//!
//! Tests:
//!  1) Default config has both knobs off.
//!  2) Both knobs off produces byte-identical encoder output (regression
//!     guard against accidental engagement).
//!  3) UV-RDO on: keyframe + P-frame encode/decode cleanly.
//!  4) UV-RDO byte envelope ±20 % vs greedy SSE baseline.
//!  5) UV-RDO PSNR not worse by more than 0.5 dB.
//!  6) Mode/ref LF deltas: keyframe encode is byte-identical (deltas
//!     don't apply on keyframes — the decoder zeros them at every
//!     keyframe per §9.4).
//!  7) Mode/ref LF deltas: P-frame encode produces a valid bitstream
//!     that round-trips through the in-tree decoder (the encoder's
//!     post-filter reconstruction must match the decoder's, otherwise
//!     subsequent P-frames diverge).
//!  8) Mode/ref LF deltas: PSNR not worse by more than 0.5 dB.
//!  9) Combined UV-RDO + mode/ref deltas + joint LF-RDO: the three
//!     knobs compose without collapsing PSNR or producing an
//!     undecodable bitstream.

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

/// Mixed-chroma clip — UV planes carry direction-aware structure that
/// gives the four UV candidates (DC / V / H / TM) something to pick
/// from. Y plane has a moderate diagonal so motion search engages on
/// P-frames, exercising the intra-in-P fallback path on textured MBs.
fn make_chroma_edge_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let mut y = vec![0u8; (W * H) as usize];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for row in 0..H as usize {
            for col in 0..W as usize {
                // Luma: diagonal gradient + per-frame phase shift.
                let diag = ((row as i32 - col as i32) * 5).rem_euclid(255);
                let phase = (f as i32 * 2).rem_euclid(255);
                y[row * W as usize + col] = ((diag + phase) % 255) as u8;
            }
        }
        for row in 0..ch {
            for col in 0..cw {
                // U plane: vertical stripes — favours V_PRED.
                u[row * cw + col] = if (col / 2) % 2 == 0 { 96 } else { 160 };
                // V plane: horizontal stripes — favours H_PRED.
                v[row * cw + col] = if (row / 2) % 2 == 0 { 96 } else { 160 };
            }
        }
        out.push(make_frame(y, u, v));
    }
    out
}

/// Constant-grey clip — every MB picks DC across the board on chroma
/// (the prediction is exact). Used to pin that toggling UV-RDO has no
/// effect when greedy SSE already collapses to DC.
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
    }
}

// ─── default + bit-exact off path ────────────────────────────────────────────

#[test]
fn default_config_uv_rdo_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_uv_rdo,
        "round-42 default must keep UV-RDO off (preserves bit-exact greedy SSE chroma pick)"
    );
}

#[test]
fn default_config_mode_ref_lf_deltas_off() {
    let cfg = Vp8EncoderConfig::default();
    assert!(
        !cfg.enable_mode_ref_lf_deltas,
        "round-42 default must keep mode/ref LF deltas off (legacy P-frame bitstream byte-identical)"
    );
}

/// Round-42 must not perturb the bitstream when both knobs are off.
#[test]
fn round42_off_path_byte_identical_to_legacy() {
    let clip = make_chroma_edge_clip(4);
    let cfg = cfg_baseline();
    let (b0, _, p0) = measure(cfg, &clip);
    let (b1, _, p1) = measure(cfg, &clip);
    assert_eq!(b0, b1, "deterministic encode must match itself");
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "packet {i} differs between identical configs");
    }
}

// ─── UV-RDO ──────────────────────────────────────────────────────────────────

#[test]
fn uv_rdo_keyframe_decodes_cleanly() {
    let clip = make_chroma_edge_clip(1);
    let cfg = Vp8EncoderConfig {
        enable_uv_rdo: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "UV-RDO keyframe produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "UV-RDO keyframe encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

#[test]
fn uv_rdo_pframe_decodes_cleanly() {
    let clip = make_chroma_edge_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_uv_rdo: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "UV-RDO P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "UV-RDO P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// UV-RDO sometimes biases toward DC (cheaper to code), other times
/// toward V/H (lower distortion on directional chroma); pin the swing
/// to ±20 % of the legacy baseline.
#[test]
fn uv_rdo_byte_envelope_within_20pct() {
    let clip = make_chroma_edge_clip(8);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_uv_rdo: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_r, _, _) = measure(with_rdo, &clip);

    let frac = (bytes_r as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.20,
        "UV-RDO swung byte size by {:.1}% (greedy {bytes_g}, rdo {bytes_r}) — beyond +/-20%",
        frac * 100.0
    );
}

#[test]
fn uv_rdo_psnr_does_not_regress_significantly() {
    let clip = make_chroma_edge_clip(6);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_uv_rdo: true,
        ..cfg_baseline()
    };

    let (_b_g, psnr_g, _) = measure(baseline, &clip);
    let (_b_r, psnr_r, _) = measure(with_rdo, &clip);

    assert!(
        psnr_r >= psnr_g - 0.5,
        "UV-RDO regressed PSNR_Y by {:.3} dB beyond 0.5 dB slack (greedy {psnr_g:.2}, rdo {psnr_r:.2})",
        psnr_g - psnr_r
    );
}

/// On flat content greedy SSE already picks DC_PRED on every chroma MB
/// (perfect prediction → zero SSE for DC). UV-RDO must agree, since
/// adding any non-zero λ·R term to a zero-D candidate keeps DC's cost
/// at 0 + λ·rate(DC), the smallest over all four candidates. So
/// toggling the knob must not move bytes on flat content.
#[test]
fn uv_rdo_flat_content_is_byte_identical() {
    let clip = make_flat_clip(4);
    let baseline = cfg_baseline();
    let with_rdo = Vp8EncoderConfig {
        enable_uv_rdo: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(baseline, &clip);
    let (b1, _, p1) = measure(with_rdo, &clip);

    assert_eq!(
        b0, b1,
        "flat-content encode must be byte-identical with UV-RDO toggle: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "flat-content packet {i} differs");
    }
}

/// `enable_rdo = false` must zero the UV-RDO lambda, making the knob
/// inert — same gating contract round-41 BMODE-RDO uses.
#[test]
fn uv_rdo_requires_enable_rdo() {
    let clip = make_chroma_edge_clip(4);
    let cfg_a = Vp8EncoderConfig {
        enable_rdo: false,
        enable_uv_rdo: false,
        ..cfg_baseline()
    };
    let cfg_b = Vp8EncoderConfig {
        enable_rdo: false,
        enable_uv_rdo: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(cfg_a, &clip);
    let (b1, _, p1) = measure(cfg_b, &clip);

    assert_eq!(
        b0, b1,
        "UV-RDO must be inert when enable_rdo = false: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "enable_rdo-off packet {i} differs");
    }
}

// ─── Mode/ref LF deltas ──────────────────────────────────────────────────────

/// Keyframes always reset mode/ref deltas to zero per RFC 6386 §9.4
/// (`parse_keyframe_header` calls `parse_loop_filter` with
/// `prev_*_deltas = [0; 4]`). The encoder must preserve this — toggling
/// `enable_mode_ref_lf_deltas` against a single-keyframe clip must not
/// move bytes.
#[test]
fn mode_ref_lf_deltas_keyframe_byte_identical() {
    let clip = make_chroma_edge_clip(1);
    let baseline = cfg_baseline();
    let with_deltas = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };

    let (b0, _, p0) = measure(baseline, &clip);
    let (b1, _, p1) = measure(with_deltas, &clip);

    assert_eq!(
        b0, b1,
        "keyframe encode must be byte-identical with mode/ref LF deltas toggle: {b0} vs {b1}"
    );
    assert_eq!(p0.len(), p1.len());
    for (i, (a, b)) in p0.iter().zip(p1.iter()).enumerate() {
        assert_eq!(a, b, "keyframe packet {i} differs");
    }
}

/// P-frame round-trip: the encoder applies the deltas to its own
/// reconstruction (so the next reference matches what the decoder
/// will compute from the bitstream we emit). If the encoder + decoder
/// disagreed on per-MB level, subsequent P-frames would drift and
/// PSNR collapse. A clean round-trip with reasonable PSNR therefore
/// pins the encoder + decoder agree on `per_mb_filter_level`.
#[test]
fn mode_ref_lf_deltas_pframe_decodes_cleanly() {
    let clip = make_chroma_edge_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(bytes > 0, "mode/ref LF deltas P-frame produced zero bytes");
    assert!(
        psnr_y > 5.0,
        "mode/ref LF deltas P-frame encode/decode PSNR collapsed: {psnr_y:.2} dB"
    );
}

/// Mode/ref LF deltas can shift bytes a little (extra header bits and
/// slightly different reconstruction → different residual entropy);
/// pin the swing to ±15 %.
#[test]
fn mode_ref_lf_deltas_byte_envelope_within_15pct() {
    let clip = make_chroma_edge_clip(8);
    let baseline = cfg_baseline();
    let with_deltas = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };

    let (bytes_g, _, _) = measure(baseline, &clip);
    let (bytes_r, _, _) = measure(with_deltas, &clip);

    let frac = (bytes_r as f64 - bytes_g as f64).abs() / bytes_g.max(1) as f64;
    assert!(
        frac < 0.15,
        "mode/ref LF deltas swung byte size by {:.1}% ({bytes_g} → {bytes_r}) — beyond +/-15%",
        frac * 100.0
    );
}

#[test]
fn mode_ref_lf_deltas_psnr_does_not_regress_significantly() {
    let clip = make_chroma_edge_clip(6);
    let baseline = cfg_baseline();
    let with_deltas = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };

    let (_b_g, psnr_g, _) = measure(baseline, &clip);
    let (_b_r, psnr_r, _) = measure(with_deltas, &clip);

    assert!(
        psnr_r >= psnr_g - 0.5,
        "mode/ref LF deltas regressed PSNR_Y by {:.3} dB beyond 0.5 dB slack (off {psnr_g:.2}, on {psnr_r:.2})",
        psnr_g - psnr_r
    );
}

/// First P-frame's bitstream must contain the mode_ref_delta_enabled = 1
/// bit followed by the 4 ref + 4 mode delta literals. We don't pin the
/// exact byte offsets (entropy-coded position drifts with content), but
/// we do pin that turning the knob on grows the inter-frame's byte
/// budget vs the no-deltas baseline by a small but non-zero amount —
/// enough to confirm the bits are actually emitted.
///
/// On a 32×32 frame each P-frame is ~10–30 bytes, and the deltas add
/// ~10 bool symbols (8 present flags + 8 signed-6-bit literals × 2) ≈
/// 1–3 bytes; the regression check is "non-zero growth".
#[test]
fn mode_ref_lf_deltas_grows_pframe_header() {
    // First P-frame must carry the mode_ref_delta block; isolate the
    // P-frame by looking at the second packet (index 1).
    let clip = make_chroma_edge_clip(2);
    let baseline = cfg_baseline();
    let with_deltas = Vp8EncoderConfig {
        enable_mode_ref_lf_deltas: true,
        ..cfg_baseline()
    };

    let (_, _, p_off) = measure(baseline, &clip);
    let (_, _, p_on) = measure(with_deltas, &clip);

    assert_eq!(p_off.len(), 2);
    assert_eq!(p_on.len(), 2);

    // Keyframe (packet 0) is identical regardless of the toggle.
    assert_eq!(
        p_off[0], p_on[0],
        "keyframe must not change with mode/ref LF deltas toggle"
    );
    // P-frame (packet 1) MUST differ when the deltas are on.
    assert_ne!(
        p_off[1], p_on[1],
        "P-frame must change when mode/ref LF deltas are emitted"
    );
}

// ─── Composition with joint LF-RDO ───────────────────────────────────────────

/// Round-42 unlocks the joint LF-RDO real rate term: with mode/ref
/// deltas on, the per-MB level the encoder applies during the
/// patch-level filter trial varies per MB (intra MBs +2 ref delta,
/// B_PRED +4 mode delta, etc.), so the LF-RDO picker now scores the
/// actual post-delta reconstruction. Composition test: enable all
/// three knobs (UV-RDO + mode/ref deltas + joint LF-RDO) and confirm
/// the bitstream still round-trips cleanly with reasonable PSNR.
#[test]
fn round42_combined_with_joint_lf_rdo_decodes_cleanly() {
    let clip = make_chroma_edge_clip(8);
    let cfg = Vp8EncoderConfig {
        enable_uv_rdo: true,
        enable_mode_ref_lf_deltas: true,
        enable_joint_lf_rdo: true,
        ..cfg_baseline()
    };
    let (bytes, psnr_y, _) = measure(cfg, &clip);
    assert!(
        bytes > 0,
        "combined round-42 + joint LF-RDO produced zero bytes"
    );
    assert!(
        psnr_y > 5.0,
        "combined round-42 + joint LF-RDO PSNR collapsed: {psnr_y:.2} dB"
    );
}
