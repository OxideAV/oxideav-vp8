//! End-to-end exercise of the `oxideav-vp8` standalone API surface
//! (the symbols reachable under `cargo test -p oxideav-vp8
//! --no-default-features`).
//!
//! The crate's README documents a no-`oxideav-core` build that exposes
//! the keyframe encoder, the stateful multi-frame decoder, the two-pass
//! encoder family, the IVF container helpers and the WebP-canonical
//! `0..=100` → qindex `quality_to_qindex` knob. This file pins that the standalone-build
//! contract is **runtime**-correct, not just compile-time: each of those
//! entry points is driven against a synthetic I420 source and the
//! result is checked numerically.
//!
//! Every symbol imported here MUST resolve without the `registry`
//! feature; the `tests/api_compat_0_1_13.rs` suite already pins the
//! compile-time surface, so this file specifically exercises the
//! *behaviour* — encode succeeds, decode succeeds, PSNR clears a
//! sensible floor, qindex schedule varies across complexity, IVF
//! round-trips through the parse / write helpers.
//!
//! The test is self-contained — no external codec is consulted, only
//! the crate's own decode + encode entry points.

use oxideav_vp8::ivf::{
    parse_frame_header, parse_header, write_frame, write_header, IvfHeader, IVF_FRAME_HEADER_LEN,
    IVF_HEADER_LEN, IVF_VP8_FOURCC,
};
use oxideav_vp8::{
    decode_vp8, encode_keyframe, first_pass_analyze, quality_to_qindex, two_pass_qindices,
    FrameComplexity, I420Frame, KeyframeParams, Vp8DecodedFrame, Vp8DecoderState, Vp8TwoPassConfig,
    Vp8TwoPassEncoder,
};

// ─────────────────────────── synthetic sources ───────────────────────────

/// Packed I420 source picture.
struct Source {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Source {
    fn frame(&self) -> I420Frame<'_> {
        I420Frame::packed(self.width, self.height, &self.y, &self.u, &self.v)
    }
}

/// Build a synthetic `width × height` I420 picture combining a smooth
/// luma gradient with a flat chroma background and a flat luma square —
/// a mix of structured and uniform regions that exercises both the
/// whole-block and `B_PRED` intra paths.
fn synthetic_source(width: u32, height: u32) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;

    let mut y = vec![0u8; w * h];
    for (row, chunk) in y.chunks_mut(w).enumerate() {
        for (col, px) in chunk.iter_mut().enumerate() {
            let mut v = ((col * 256 / w + row * 256 / h) / 2) as u8;
            if col >= w / 4 && col < w * 3 / 4 && row >= h / 4 && row < h * 3 / 4 {
                v = 128;
            }
            *px = v;
        }
    }

    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = (120 + (col * 16 / cw)) as u8;
            v[row * cw + col] = (130 + (row * 16 / ch)) as u8;
        }
    }

    Source {
        width,
        height,
        y,
        u,
        v,
    }
}

/// Per-frame source for the multi-frame sequence: small per-frame drift
/// so the ZERO_MV inter encoder can absorb the residual at a mid
/// quantiser.
fn slow_drift_source(width: u32, height: u32, frame_idx: usize) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;

    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let v = 40 + (r as i16) + (c as i16) + (frame_idx as i16);
            y[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
    let u_val = (130 + frame_idx as i32).clamp(0, 255) as u8;
    let v_val = (120 - frame_idx as i32).clamp(0, 255) as u8;
    Source {
        width,
        height,
        y,
        u: vec![u_val; cw * ch],
        v: vec![v_val; cw * ch],
    }
}

// ─────────────────────────── PSNR helpers ───────────────────────────────

fn plane_mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "plane length mismatch");
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    sum / a.len() as f64
}

/// Luma PSNR (8-bit peak = 255). Returns `f64::INFINITY` on identical
/// inputs.
fn luma_psnr(src: &[u8], dec: &[u8]) -> f64 {
    let mse = plane_mse(src, dec);
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

// ────────────────────── 1. Keyframe roundtrip standalone ──────────────────

/// `I420Frame::packed` + `KeyframeParams::new` + `encode_keyframe` +
/// `decode_vp8` must round-trip a synthetic gradient frame at PSNR-Y
/// ≥ 30 dB without touching the `registry` feature.
#[test]
fn standalone_keyframe_roundtrip_psnr_meets_30db() {
    let src = synthetic_source(64, 64);
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let bytes = encode_keyframe(&src.frame(), &params).expect("encode_keyframe");
    let dec: Vp8DecodedFrame = decode_vp8(&bytes).expect("decode_vp8");

    assert_eq!(dec.width, 64);
    assert_eq!(dec.height, 64);
    assert_eq!(dec.y.len(), 64 * 64);
    assert_eq!(dec.u.len(), 32 * 32);
    assert_eq!(dec.v.len(), 32 * 32);

    let psnr_y = luma_psnr(&src.y, &dec.y);
    eprintln!("standalone keyframe 64x64 qi=32: PSNR-Y = {psnr_y:.2} dB");
    assert!(
        psnr_y >= 30.0,
        "standalone keyframe PSNR-Y {psnr_y:.2} dB below 30 dB floor"
    );
}

// ─────────────────── 2. Multi-frame inter roundtrip standalone ──────────────

/// A 4-frame synthetic I420 sequence encoded one frame at a time through
/// the two-pass encoder (which is the standalone-reachable inter
/// encoder driver per the README) and decoded back through
/// `Vp8DecoderState`. Every frame must decode and clear PSNR-Y ≥ 28 dB.
///
/// The two-pass encoder is used because `Vp8InterStreamEncoder` is also
/// reachable standalone, but the two-pass driver is the README-documented
/// path for multi-frame standalone encoding (its `encode_frame` builds
/// the K + P stream end-to-end without requiring the caller to
/// hand-thread the reference slots).
#[test]
fn standalone_multi_frame_inter_decode_via_decoder_state() {
    let (w, h) = (64u32, 64u32);
    let frames: Vec<Source> = (0..4).map(|i| slow_drift_source(w, h, i)).collect();
    let i420: Vec<I420Frame<'_>> = frames.iter().map(|s| s.frame()).collect();

    let config = Vp8TwoPassConfig::default();
    let mut encoder = Vp8TwoPassEncoder::new(config);
    let stats = encoder
        .first_pass_analyze(&i420)
        .expect("first_pass_analyze must succeed");
    assert_eq!(stats.len(), 4, "one stat per frame");

    let mut decoder = Vp8DecoderState::new();
    for (i, (frame, stat)) in i420.iter().zip(stats.iter()).enumerate() {
        let bytes = encoder
            .encode_frame(frame, *stat)
            .expect("encode_frame must succeed");
        assert!(!bytes.is_empty(), "frame {i} must produce bytes");
        let dec = decoder
            .decode_frame(&bytes)
            .unwrap_or_else(|e| panic!("frame {i} must decode via Vp8DecoderState: {e:?}"));
        assert_eq!(dec.width, w, "frame {i} width");
        assert_eq!(dec.height, h, "frame {i} height");

        let psnr_y = luma_psnr(&frames[i].y, &dec.y);
        eprintln!(
            "standalone inter frame {i}: {} B, PSNR-Y = {psnr_y:.2} dB",
            bytes.len()
        );
        assert!(
            psnr_y >= 28.0,
            "frame {i} PSNR-Y {psnr_y:.2} dB below 28 dB floor"
        );
    }
}

// ───────────────────── 3. Two-pass standalone ──────────────────────────

/// `first_pass_analyze` + `two_pass_qindices` + `Vp8TwoPassEncoder` all
/// drive a varied-complexity GOP end-to-end through the standalone API.
/// The qindex schedule must vary across frames (the whole point of
/// two-pass), and each emitted frame must decode through
/// `Vp8DecoderState`.
#[test]
fn standalone_two_pass_qindex_schedule_and_decode() {
    let (w, h) = (32u32, 32u32);

    // Solid → gradient → checker → noise: deliberately varied complexity
    // so the qindex schedule must respond.
    let mut frames: Vec<Source> = Vec::new();
    // Solid grey.
    frames.push(Source {
        width: w,
        height: h,
        y: vec![128u8; (w * h) as usize],
        u: vec![128u8; ((w * h) / 4) as usize],
        v: vec![128u8; ((w * h) / 4) as usize],
    });
    // Gradient.
    {
        let mut y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                y[r * w as usize + c] = (c * 8).min(255) as u8;
            }
        }
        frames.push(Source {
            width: w,
            height: h,
            y,
            u: vec![120u8; ((w * h) / 4) as usize],
            v: vec![136u8; ((w * h) / 4) as usize],
        });
    }
    // Checkerboard.
    {
        let mut y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                let block = ((r / 4) + (c / 4)) & 1;
                y[r * w as usize + c] = if block == 0 { 32 } else { 224 };
            }
        }
        frames.push(Source {
            width: w,
            height: h,
            y,
            u: vec![128u8; ((w * h) / 4) as usize],
            v: vec![128u8; ((w * h) / 4) as usize],
        });
    }
    // Pseudo-random noise.
    {
        let mut y = vec![0u8; (w * h) as usize];
        let mut state: u32 = 0xc0ffee01;
        for px in y.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *px = (state >> 16) as u8;
        }
        frames.push(Source {
            width: w,
            height: h,
            y,
            u: vec![128u8; ((w * h) / 4) as usize],
            v: vec![128u8; ((w * h) / 4) as usize],
        });
    }

    let i420: Vec<I420Frame<'_>> = frames.iter().map(|s| s.frame()).collect();
    let config = Vp8TwoPassConfig::default();

    let stats = first_pass_analyze(&i420, &config).expect("first_pass_analyze");
    assert_eq!(stats.len(), 4);

    let schedule = two_pass_qindices(&config, &stats).expect("two_pass_qindices");
    assert_eq!(schedule.len(), 4);
    for &qi in &schedule {
        assert!(qi <= 127, "schedule respects 0..=127 (got {qi})");
    }
    let min = *schedule.iter().min().unwrap();
    let max = *schedule.iter().max().unwrap();
    eprintln!("standalone two-pass schedule: {schedule:?}");
    assert!(
        max > min,
        "schedule must vary across complexity-varied GOP: {schedule:?}"
    );

    // Drive the encoder + verify each emitted frame decodes.
    let mut encoder = Vp8TwoPassEncoder::new(config);
    let stats2 = encoder
        .first_pass_analyze(&i420)
        .expect("encoder first_pass_analyze");
    assert_eq!(stats2, stats, "encoder stats must match free-fn stats");

    let mut decoder = Vp8DecoderState::new();
    for (i, (frame, stat)) in i420.iter().zip(stats.iter()).enumerate() {
        let bytes = encoder
            .encode_frame(frame, *stat)
            .expect("encode_frame must succeed");
        let dec = decoder
            .decode_frame(&bytes)
            .unwrap_or_else(|e| panic!("frame {i} must decode: {e:?}"));
        assert_eq!(dec.width, w);
        assert_eq!(dec.height, h);
    }
}

// ───────────────── 4. IVF container roundtrip standalone ────────────────

/// `ivf::write_header` + `ivf::write_frame` produce bytes that
/// `ivf::parse_header` + `ivf::parse_frame_header` re-parse into the
/// same `IvfHeader` + frame payload — exercises the standalone IVF
/// reader / writer the README documents.
#[test]
fn standalone_ivf_container_roundtrip() {
    let hdr = IvfHeader::vp8(640, 480, 30, 1);
    let mut buf = write_header(&hdr);
    assert_eq!(buf.len(), IVF_HEADER_LEN);

    // Several frames with distinct PTSes and payloads.
    let payloads: [&[u8]; 3] = [&[0xde, 0xad, 0xbe, 0xef], &[0x11; 16], &[0xab; 1]];
    let ptses: [u64; 3] = [0, 1, 1_000_000];
    for (pts, payload) in ptses.iter().zip(payloads.iter()) {
        write_frame(&mut buf, *pts, payload);
    }
    let expected_extra: usize = payloads
        .iter()
        .map(|p| IVF_FRAME_HEADER_LEN + p.len())
        .sum();
    assert_eq!(buf.len(), IVF_HEADER_LEN + expected_extra);

    let parsed_hdr = parse_header(&buf).expect("parse_header");
    assert_eq!(parsed_hdr.width, 640);
    assert_eq!(parsed_hdr.height, 480);
    assert_eq!(parsed_hdr.framerate_num, 30);
    assert_eq!(parsed_hdr.framerate_den, 1);
    assert_eq!(parsed_hdr.fourcc, IVF_VP8_FOURCC);
    assert_eq!(parsed_hdr, hdr);

    // Walk each frame record. Each record is 12 bytes of header + N
    // bytes of payload; advance the cursor by both.
    let mut cursor = IVF_HEADER_LEN;
    for (pts, payload) in ptses.iter().zip(payloads.iter()) {
        let fh = parse_frame_header(&buf[cursor..]).expect("parse_frame_header");
        assert_eq!(fh.pts, *pts);
        assert_eq!(fh.size as usize, payload.len());
        let start = cursor + IVF_FRAME_HEADER_LEN;
        let end = start + payload.len();
        assert_eq!(&buf[start..end], *payload, "payload bytes round-trip");
        cursor = end;
    }
    assert_eq!(cursor, buf.len(), "consumed every byte");
}

/// Short-buffer rejection is part of the standalone contract: a
/// truncated header must not panic, it must return
/// [`oxideav_vp8::Vp8Error::InvalidData`].
#[test]
fn standalone_ivf_short_buffers_reject_cleanly() {
    assert!(parse_header(&[]).is_err());
    assert!(parse_header(&[0u8; 16]).is_err());
    assert!(parse_frame_header(&[]).is_err());
    assert!(parse_frame_header(&[0u8; 4]).is_err());
}

// ─────────────────────── 5. Quality knob standalone ─────────────────────

/// The `quality_to_qindex` table the README documents (0 → 127, 75 → 32,
/// 100 → 0, NaN → 127, out-of-range clamps) must hold under the
/// standalone build.
#[test]
fn standalone_quality_to_qindex_table() {
    assert_eq!(quality_to_qindex(0.0), 127, "quality 0 → worst qindex 127");
    assert_eq!(
        quality_to_qindex(75.0),
        32,
        "quality 75 → WebP-canonical default qindex 32"
    );
    assert_eq!(quality_to_qindex(100.0), 0, "quality 100 → best qindex 0");
    assert_eq!(quality_to_qindex(50.0), 64, "quality 50 → mid qindex 64");

    // NaN is the documented fallback to the smallest-file choice.
    assert_eq!(quality_to_qindex(f32::NAN), 127);

    // Out-of-range clamps to the corresponding extreme.
    assert_eq!(quality_to_qindex(-1.0), 127);
    assert_eq!(quality_to_qindex(-100.0), 127);
    assert_eq!(quality_to_qindex(101.0), 0);
    assert_eq!(quality_to_qindex(1_000_000.0), 0);

    // Monotonic non-increasing across the documented range.
    let mut prev = quality_to_qindex(0.0);
    for q in 1..=100 {
        let cur = quality_to_qindex(q as f32);
        assert!(
            cur <= prev,
            "monotonicity broken at quality {q}: {cur} > {prev}"
        );
        prev = cur;
    }
}

// ─────────────────── 6. Empty / degenerate two-pass input ───────────────

/// The two-pass entry points must accept an empty frame slice without
/// panicking — documented in the README + the existing in-tree tests.
#[test]
fn standalone_two_pass_empty_input_is_quiet() {
    let config = Vp8TwoPassConfig::default();
    let stats = first_pass_analyze(&[], &config).expect("empty input must succeed");
    assert!(stats.is_empty());
    let schedule = two_pass_qindices(&config, &[]).expect("empty schedule must succeed");
    assert!(schedule.is_empty());
}

// ──────────────── 7. Single-FrameComplexity record + scene-cut ──────────

/// A single-frame `FrameComplexity` schedule must produce a single
/// `0..=127` qindex via `two_pass_qindices`.
#[test]
fn standalone_two_pass_single_frame_schedule() {
    let config = Vp8TwoPassConfig::default();
    let fc = FrameComplexity {
        frame_index: 0,
        bits_per_mb: 100.0,
        scene_cut: false,
    };
    let schedule = two_pass_qindices(&config, &[fc]).expect("schedule");
    assert_eq!(schedule.len(), 1);
    assert!(schedule[0] <= 127);
}
