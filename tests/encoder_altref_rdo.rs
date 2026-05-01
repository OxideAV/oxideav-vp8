//! Alt-ref / golden-ref planning + Lagrangian RDO regressions.
//!
//! These tests pin the new behaviour added in this round:
//!
//! * The encoder periodically refreshes GOLDEN and ALTREF on the cadence
//!   from `Vp8EncoderConfig`; the bitstream's `refresh_golden_frame` and
//!   `refresh_alternate_frame` flags follow that schedule and are
//!   correctly parsed back by the decoder.
//! * With multi-ref enabled, motion that recurs between frame N-2 and
//!   frame N is reproducible by referencing the older anchor (GOLDEN /
//!   ALTREF) at high PSNR.
//! * The Lagrangian RDO path (`enable_rdo = true`) does not regress
//!   PSNR vs the SAD-only path on a 30-frame synthetic clip; in
//!   practice it produces a smaller bitstream because the rate term
//!   penalises modes whose entropy cost outweighs the SSE gain.
//! * Toggling `enable_multi_ref` off recovers the previous single-ref
//!   behaviour (no GOLDEN / ALTREF refresh in the bitstream), proving
//!   the new path is opt-out cleanly.

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL, DEFAULT_GOLDEN_INTERVAL,
};
use oxideav_vp8::{parse_header, FrameType};

const W: u32 = 64;
const H: u32 = 64;
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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Synthetic 30-frame clip with a 16-px vertical-stripe pattern that
/// pans diagonally by (+1, +1) pixel per frame, with a hard cut every
/// 16 frames back to the original phase. The cut cadence specifically
/// exercises the alt-ref / golden anchor: long-pinned references see the
/// pre-cut content again on cycle restart, so picking GOLDEN or ALTREF
/// over the (post-cut) LAST reference can radically reduce the residual.
fn make_clip(n_frames: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        // Pan within a 16-frame cycle.
        let p = (f % 16) as i32;
        let mut y = vec![0u8; (W * H) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let cc = (col as i32 + p).rem_euclid(W as i32) as usize;
                let stripe = (cc / 16) & 1;
                y[row * W as usize + col] = if stripe == 0 { 40 } else { 200 };
            }
        }
        let u = vec![128u8; cw * ch];
        let v = vec![128u8; cw * ch];
        out.push(make_frame(y, u, v));
    }
    out
}

fn encode_clip(config: Vp8EncoderConfig, clip: &[VideoFrame]) -> (Vec<Vec<u8>>, Vec<bool>) {
    let mut enc_params = CodecParameters::video(CodecId::new("vp8"));
    enc_params.width = Some(W);
    enc_params.height = Some(H);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_config(&enc_params, config).expect("encoder");
    let mut packets = Vec::with_capacity(clip.len());
    let mut keyflags = Vec::with_capacity(clip.len());
    for f in clip.iter() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("rx");
        keyflags.push(pkt.flags.keyframe);
        packets.push(pkt.data);
    }
    (packets, keyflags)
}

fn decode_clip_psnr(packets: &[Vec<u8>], src: &[VideoFrame]) -> Vec<f64> {
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let mut psnrs = Vec::with_capacity(packets.len());
    for (i, p) in packets.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), p.clone()))
            .expect("decode");
        let frame = match dec.receive_frame().expect("rx") {
            Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        psnrs.push(psnr(&frame.planes[0].data, &src[i].planes[0].data));
    }
    psnrs
}

/// The encoder's per-frame plan refreshes GOLDEN every
/// `golden_interval` P-frames and ALTREF every `alt_ref_interval`
/// P-frames. The decoder must agree.
#[test]
fn refresh_flags_follow_plan() {
    let clip = make_clip(20); // 1 keyframe + 19 P-frames
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: 4,
        alt_ref_interval: 7,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (packets, keyflags) = encode_clip(cfg, &clip);
    assert!(keyflags[0], "first frame must be a keyframe");

    // Walk the per-packet header tags + parse just enough of the inter
    // header to confirm refresh_golden / refresh_alt match the plan.
    use oxideav_vp8::bool_decoder::BoolDecoder;
    use oxideav_vp8::frame_header::{parse_inter_header, PersistentProbs};
    let probs = PersistentProbs::defaults();
    let mut p_idx = 0u32; // counts P-frames specifically
    for (i, pkt) in packets.iter().enumerate() {
        let parsed = parse_header(pkt).expect("hdr");
        if matches!(parsed.tag.frame_type, FrameType::Key) {
            assert_eq!(i, 0, "only frame 0 should be a keyframe");
            continue;
        }
        p_idx += 1;
        let body = &pkt[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let h = parse_inter_header(&mut bd, &probs).expect("hdr");
        let want_golden = p_idx % 4 == 0;
        let want_alt = p_idx % 7 == 0;
        assert_eq!(
            h.refresh_golden, want_golden,
            "P-frame {p_idx} refresh_golden mismatch"
        );
        assert_eq!(
            h.refresh_alternate, want_alt,
            "P-frame {p_idx} refresh_alt mismatch"
        );
    }
}

/// Disabling multi-ref must keep refresh_golden / refresh_alt = false on
/// all P-frames (legacy single-ref behaviour). Confirms the new path is
/// opt-out cleanly.
#[test]
fn multi_ref_off_keeps_legacy_refresh_flags() {
    let clip = make_clip(10);
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: 4,
        alt_ref_interval: 7,
        enable_rdo: false,
        lambda_scale: 0,
        enable_multi_ref: false,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (packets, keyflags) = encode_clip(cfg, &clip);
    assert!(keyflags[0]);
    use oxideav_vp8::bool_decoder::BoolDecoder;
    use oxideav_vp8::frame_header::{parse_inter_header, PersistentProbs};
    let probs = PersistentProbs::defaults();
    for (i, pkt) in packets.iter().enumerate().skip(1) {
        let parsed = parse_header(pkt).expect("hdr");
        let body = &pkt[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let h = parse_inter_header(&mut bd, &probs).expect("hdr");
        assert!(
            !h.refresh_golden,
            "P-frame {i} unexpectedly refreshed GOLDEN"
        );
        assert!(
            !h.refresh_alternate,
            "P-frame {i} unexpectedly refreshed ALT"
        );
    }
}

/// End-to-end: encode 30 frames with multi-ref + RDO enabled, decode
/// the bitstream back, confirm every frame stays at high PSNR.
#[test]
fn multi_ref_rdo_pipeline_roundtrips_high_psnr() {
    let clip = make_clip(30);
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (packets, _kf) = encode_clip(cfg, &clip);
    let psnrs = decode_clip_psnr(&packets, &clip);
    let total_bytes: usize = packets.iter().map(|p| p.len()).sum();
    let avg_psnr: f64 = psnrs.iter().sum::<f64>() / psnrs.len() as f64;
    eprintln!(
        "multi-ref+RDO: {} bytes total, avg PSNR Y = {avg_psnr:.2} dB",
        total_bytes,
    );
    for (i, &p) in psnrs.iter().enumerate() {
        assert!(
            p >= 30.0,
            "frame {i} multi-ref+RDO Y PSNR too low: {p:.2} dB",
        );
    }
    assert!(
        avg_psnr >= 35.0,
        "multi-ref+RDO avg Y PSNR too low: {avg_psnr:.2}"
    );
}

/// BD-rate-style measurement: encode the same clip once with multi-ref
/// + alt-ref planning enabled, once without.
///
/// Prints both `(bytes, avg-PSNR)` pairs to the test log and pins a
/// regression: the multi-ref + RDO path's PSNR must not collapse vs
/// the SAD-only single-ref baseline. On this stripe-pan clip multi-ref
/// is expected to win clearly (the cyclic content recurs every 16
/// frames, GOLDEN catches it), but the assertion is structured so
/// either path can dominate as long as quality holds.
#[test]
fn alt_ref_vs_off_bd_rate_comparison() {
    let clip = make_clip(30);

    // OFF: single reference, SAD-only mode decision.
    let off = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: 0,
        alt_ref_interval: 0,
        enable_rdo: false,
        lambda_scale: 0,
        enable_multi_ref: false,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (off_packets, _) = encode_clip(off, &clip);
    let off_psnrs = decode_clip_psnr(&off_packets, &clip);

    // ON: GOLDEN every 8 P-frames, ALTREF every 13, RDO on.
    let on = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (on_packets, _) = encode_clip(on, &clip);
    let on_psnrs = decode_clip_psnr(&on_packets, &clip);

    let off_bytes: usize = off_packets.iter().map(|p| p.len()).sum();
    let on_bytes: usize = on_packets.iter().map(|p| p.len()).sum();
    let off_avg: f64 = off_psnrs.iter().sum::<f64>() / off_psnrs.len() as f64;
    let on_avg: f64 = on_psnrs.iter().sum::<f64>() / on_psnrs.len() as f64;
    eprintln!(
        "alt-ref OFF: {off_bytes} bytes, avg Y PSNR = {off_avg:.2} dB | \
         alt-ref+RDO ON: {on_bytes} bytes, avg Y PSNR = {on_avg:.2} dB",
    );

    // Both encoders should produce decodable streams at high quality;
    // the multi-ref encoder MUST not collapse PSNR. Allow up to 2 dB
    // worse than OFF as a safety margin (RDO can pick smaller-residual
    // modes whose reconstruction differs slightly).
    assert!(
        on_avg >= off_avg - 2.0,
        "multi-ref+RDO avg PSNR collapsed: on={on_avg:.2} off={off_avg:.2}"
    );
    // Both must be reasonable absolute numbers. The SAD-only single-ref
    // path lands in the ~28-30 dB range on this stripe-pan clip — its
    // motion search misses some shifts that the multi-ref+RDO path
    // catches via GOLDEN, so pin a permissive floor here.
    assert!(off_avg >= 25.0, "off avg PSNR too low: {off_avg:.2}");
    assert!(on_avg >= 30.0, "on avg PSNR too low: {on_avg:.2}");
    // The win on this clip is large (multi-ref catches the cyclic pan
    // via GOLDEN); pin that the multi-ref path is the better encoder.
    assert!(
        on_avg > off_avg,
        "multi-ref+RDO did not beat SAD-only baseline: on={on_avg:.2} off={off_avg:.2}"
    );
    assert!(
        on_bytes <= off_bytes,
        "multi-ref+RDO produced a larger bitstream than SAD-only: on={on_bytes} off={off_bytes}"
    );
}

/// Build a minimal IVF wrapper around the encoded packets so the file
/// can be fed to `ffmpeg` for cross-decode validation. Single video
/// stream, 30 fps, the dimensions in `(W, H)`, sequential PTS = 0..n.
fn ivf_bytes_for_packets(packets: &[Vec<u8>], width: u32, height: u32, fps: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + packets.len() * 32);
    // 32-byte IVF file header
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes()); // version
    out.extend_from_slice(&32u16.to_le_bytes()); // header length
    out.extend_from_slice(b"VP80");
    out.extend_from_slice(&(width as u16).to_le_bytes());
    out.extend_from_slice(&(height as u16).to_le_bytes());
    out.extend_from_slice(&fps.to_le_bytes()); // timebase den
    out.extend_from_slice(&1u32.to_le_bytes()); // timebase num
    out.extend_from_slice(&(packets.len() as u32).to_le_bytes()); // num frames
    out.extend_from_slice(&0u32.to_le_bytes()); // unused
                                                // Per-frame: 12-byte header (4-byte size + 8-byte pts) + payload.
    for (i, p) in packets.iter().enumerate() {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u64).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
}

/// Cross-decode validation: write the alt-ref-on encode to a temporary
/// IVF, ask `ffmpeg` to decode it (raw output to /dev/null), and assert
/// the exit code is 0 (= ffmpeg accepted every frame). Skipped silently
/// if `ffmpeg` is not on PATH so the test still passes on CI.
#[test]
fn ffmpeg_cross_decode_accepts_alt_ref_stream() {
    let ffmpeg = match which("ffmpeg") {
        Some(p) => p,
        None => {
            eprintln!("ffmpeg not on PATH; skipping cross-decode test");
            return;
        }
    };
    let clip = make_clip(30);
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (packets, _) = encode_clip(cfg, &clip);
    let ivf = ivf_bytes_for_packets(&packets, W, H, 30);
    let tmp = std::env::temp_dir().join("oxideav-vp8-altref-cross-decode.ivf");
    std::fs::write(&tmp, &ivf).expect("write tmp");
    let output = std::process::Command::new(ffmpeg)
        .args(["-loglevel", "error", "-y", "-i"])
        .arg(&tmp)
        .args(["-f", "null", "-"])
        .output()
        .expect("spawn ffmpeg");
    let _ = std::fs::remove_file(&tmp);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "ffmpeg failed to decode alt-ref bitstream: status={:?}\nstderr:\n{}",
            output.status, stderr
        );
    }
}

fn which(prog: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(prog);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Smoke test: the legacy single-ref path (multi-ref off) still decodes
/// every frame at the previously-shipped PSNR bar (>= 30 dB on this
/// clip).
#[test]
fn legacy_single_ref_path_still_works() {
    let clip = make_clip(10);
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: 0,
        alt_ref_interval: 0,
        enable_rdo: false,
        lambda_scale: 0,
        enable_multi_ref: false,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
    };
    let (packets, _) = encode_clip(cfg, &clip);
    let psnrs = decode_clip_psnr(&packets, &clip);
    let avg: f64 = psnrs.iter().sum::<f64>() / psnrs.len() as f64;
    eprintln!("legacy single-ref avg Y PSNR = {avg:.2} dB");
    for (i, &p) in psnrs.iter().enumerate() {
        assert!(p >= 25.0, "frame {i} legacy Y PSNR too low: {p:.2}");
    }
    assert!(avg >= 30.0, "legacy avg Y PSNR too low: {avg:.2}");
}
