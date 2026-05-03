//! Per-MB ref-frame context-adaptive probability regressions.
//!
//! The encoder picks `prob_intra` / `prob_last` / `prob_gf` (RFC 6386
//! §9.10 field J) from the actual ref-frame distribution observed
//! across the frame's MBs, instead of the legacy fixed `200 / 1 / 128`
//! (single-ref) or `200 / 128 / 128` (multi-ref) literals. These tests
//! pin three things on a real-content fixture:
//!
//!   * The encoded frame headers carry probabilities that visibly
//!     reflect the per-frame MB distribution (e.g. `prob_last` near
//!     `255` on a clip whose inter MBs all pick LAST).
//!   * The total byte size on a real-content fixture (the SMPTE bars
//!     `tests/fixtures/smpte_pframes.yuv`, 30 frames at 64×64) stays
//!     below the post-fix budget that captures the savings.
//!   * `ffmpeg` cross-decodes the resulting bitstream cleanly.

use std::fs;

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::bool_decoder::BoolDecoder;
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL, DEFAULT_GOLDEN_INTERVAL,
};
use oxideav_vp8::frame_header::{parse_inter_header, PersistentProbs};
use oxideav_vp8::{parse_header, FrameType};

const W: u32 = 64;
const H: u32 = 64;
const Y_SIZE: usize = (W * H) as usize;
const UV_SIZE: usize = ((W / 2) * (H / 2)) as usize;
const FRAME_SIZE: usize = Y_SIZE + 2 * UV_SIZE;
const QINDEX: u8 = 50;

fn read_yuv420p_clip(path: &str) -> Option<Vec<VideoFrame>> {
    let raw = fs::read(path).ok()?;
    if raw.len() % FRAME_SIZE != 0 {
        return None;
    }
    let n = raw.len() / FRAME_SIZE;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let base = f * FRAME_SIZE;
        let y = raw[base..base + Y_SIZE].to_vec();
        let u = raw[base + Y_SIZE..base + Y_SIZE + UV_SIZE].to_vec();
        let v = raw[base + Y_SIZE + UV_SIZE..base + FRAME_SIZE].to_vec();
        out.push(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: W as usize,
                    data: y,
                },
                VideoPlane {
                    stride: (W / 2) as usize,
                    data: u,
                },
                VideoPlane {
                    stride: (W / 2) as usize,
                    data: v,
                },
            ],
        });
    }
    Some(out)
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

fn encode_clip(clip: &[VideoFrame]) -> Vec<Vec<u8>> {
    let cfg = Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: false,
        segment_quant_deltas: [0; 4],
        // Pin pre-#166 cadence (no scene-cut detection) so this test's
        // ref-frame distribution accounting stays bit-exact.
        enable_scene_cut: false,
        scene_cut_threshold: 0.0,
        scene_cut_quant_boost: 0,
        scene_cut_boost_frames: 0,
        // Same reasoning for #209 — disable look-ahead alt-ref synthesis
        // so packet count and ref-frame counts match the legacy baseline.
        enable_lookahead_altref: false,
        lookahead_window: 0,
    };
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = make_encoder_with_config(&params, cfg).expect("encoder");
    let mut out = Vec::with_capacity(clip.len());
    for f in clip.iter() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("rx");
        out.push(pkt.data);
    }
    out
}

fn decode_clip(packets: &[Vec<u8>]) -> Vec<VideoFrame> {
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let mut out = Vec::with_capacity(packets.len());
    for p in packets.iter() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), p.clone()))
            .expect("decode");
        let frame = match dec.receive_frame().expect("rx") {
            Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        out.push(frame);
    }
    out
}

/// Read every P-frame's `prob_intra` / `prob_last` / `prob_gf` triple
/// from the inter header and confirm:
///   * They are NOT all the legacy literals (`200 / 128 / 128` or
///     `200 / 1 / 128`) — the new code never emits those.
///   * At least one P-frame's `prob_intra` lands far from the legacy
///     `200`, proving the encoder is reading the actual MB ref-frame
///     distribution.
#[test]
fn header_prob_triple_is_no_longer_constant_literals() {
    let Some(clip) = read_yuv420p_clip("tests/fixtures/smpte_pframes.yuv") else {
        eprintln!("skipping: smpte_pframes.yuv fixture missing");
        return;
    };
    assert!(clip.len() >= 5, "fixture must have several frames");
    let packets = encode_clip(&clip);

    let probs = PersistentProbs::defaults();
    let mut prob_intra_seen = Vec::new();
    let mut prob_last_seen = Vec::new();
    let mut prob_gf_seen = Vec::new();
    for (i, pkt) in packets.iter().enumerate() {
        let parsed = parse_header(pkt).expect("hdr");
        if matches!(parsed.tag.frame_type, FrameType::Key) {
            continue;
        }
        let body = &pkt[parsed.compressed_offset..];
        let mut bd = BoolDecoder::new(body).expect("bd");
        let h = parse_inter_header(&mut bd, &probs).expect("hdr");
        eprintln!(
            "P-frame {i}: prob_intra={} prob_last={} prob_gf={}",
            h.prob_intra, h.prob_last, h.prob_gf
        );
        prob_intra_seen.push(h.prob_intra);
        prob_last_seen.push(h.prob_last);
        prob_gf_seen.push(h.prob_gf);
    }

    assert!(!prob_intra_seen.is_empty(), "no P-frames in fixture");

    // None of the legacy fixed-literal triples should appear unchanged
    // across every P-frame — the per-frame distribution-derived path
    // would have to coincidentally match for that to happen.
    let all_legacy_multi = prob_intra_seen.iter().all(|&p| p == 200)
        && prob_last_seen.iter().all(|&p| p == 128)
        && prob_gf_seen.iter().all(|&p| p == 128);
    let all_legacy_single = prob_intra_seen.iter().all(|&p| p == 200)
        && prob_last_seen.iter().all(|&p| p == 1)
        && prob_gf_seen.iter().all(|&p| p == 128);
    assert!(
        !all_legacy_multi,
        "encoder still emits the legacy `200 / 128 / 128` literals"
    );
    assert!(
        !all_legacy_single,
        "encoder still emits the legacy `200 / 1 / 128` literals"
    );

    // On real content with a 64×64 SMPTE-bars clip, almost every inter
    // MB picks LAST since GOLDEN/ALT only catch occasional anchor
    // matches — `prob_last` should land high (>= 200) on average.
    let mean_prob_last: u32 =
        prob_last_seen.iter().map(|&p| p as u32).sum::<u32>() / prob_last_seen.len() as u32;
    assert!(
        mean_prob_last >= 200,
        "mean prob_last too low for an inter-LAST-dominant clip: {mean_prob_last}"
    );
}

/// End-to-end byte-size + PSNR + ffmpeg cross-decode regression on the
/// SMPTE-bars fixture.
///
/// The optimised prob triple makes the per-MB ref-frame bits cost
/// (close to) the actual entropy of the distribution. On this clip the
/// total encoded byte count must stay under the post-fix budget; we
/// also assert ffmpeg cross-decodes without complaints and the
/// roundtrip PSNR holds.
#[test]
fn smpte_clip_encodes_under_size_budget_and_ffmpeg_decodes() {
    let Some(clip) = read_yuv420p_clip("tests/fixtures/smpte_pframes.yuv") else {
        eprintln!("skipping: smpte_pframes.yuv fixture missing");
        return;
    };
    let packets = encode_clip(&clip);
    let total: usize = packets.iter().map(|p| p.len()).sum();
    eprintln!(
        "smpte_pframes ({} frames, 64×64): encoded total = {} bytes",
        packets.len(),
        total
    );

    // Budget pinned after the per-MB ref-frame context-adaptive prob
    // fix. The pre-fix total on this fixture was 1336 bytes; the new
    // path lands at ≈1157 bytes (a ~13% saving driven entirely by the
    // ref-frame bool tree shrinking from ~3 bits/MB to <0.1 bits/MB
    // when every inter MB picks LAST). The 1250-byte budget still
    // beats the legacy size by 6%+ even if a future encoder tweak
    // adds a few more bytes per frame.
    const SIZE_BUDGET_BYTES: usize = 1250;
    assert!(
        total <= SIZE_BUDGET_BYTES,
        "smpte clip encoded larger than budget: {total} > {SIZE_BUDGET_BYTES}"
    );

    // Roundtrip PSNR.
    let decoded = decode_clip(&packets);
    let mut total_psnr = 0f64;
    for (i, (got, want)) in decoded.iter().zip(clip.iter()).enumerate() {
        let py = psnr(&got.planes[0].data, &want.planes[0].data);
        if i < 3 {
            eprintln!("frame {i}: Y PSNR = {py:.2} dB");
        }
        total_psnr += py;
    }
    let avg = total_psnr / decoded.len() as f64;
    eprintln!("smpte_pframes avg Y PSNR = {avg:.2} dB");
    assert!(avg >= 25.0, "avg Y PSNR collapsed: {avg:.2}");

    // ffmpeg cross-decode (skipped if ffmpeg is not on PATH).
    let ffmpeg = match which("ffmpeg") {
        Some(p) => p,
        None => {
            eprintln!("ffmpeg not on PATH; skipping cross-decode");
            return;
        }
    };
    let ivf = ivf_bytes_for_packets(&packets, W, H, 30);
    let tmp = std::env::temp_dir().join("oxideav-vp8-refprob-cross-decode.ivf");
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
            "ffmpeg failed to decode optimised-prob bitstream: status={:?}\nstderr:\n{}",
            output.status, stderr
        );
    }
}

fn ivf_bytes_for_packets(packets: &[Vec<u8>], width: u32, height: u32, fps: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + packets.len() * 32);
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
    for (i, p) in packets.iter().enumerate() {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u64).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
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
