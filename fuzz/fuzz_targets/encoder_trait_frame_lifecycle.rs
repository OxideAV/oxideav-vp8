#![no_main]

//! Fuzz: the `oxideav_core::Encoder` trait drivers behind the public
//! factory surface (`make_encoder` / `make_encoder_with_qindex` /
//! `make_encoder_with_quality` / `encoder::make_encoder_with_config`).
//!
//! `decoder_trait_packet_lifecycle` covers the framework *decode*
//! adapter; the encode adapters had no fuzz coverage at all. Two
//! distinct trait impls sit behind the factories — the per-frame
//! keyframe encoder (the qindex / quality factories) and the lagged
//! lookahead encoder wrapping the altref stream driver (the config
//! factory) — and both own plumbing no direct-API target reaches:
//!
//! * The `CodecParameters` validation gate (missing / zero / oversize
//!   dimensions, wrong pixel format, wrong media type) and the raw
//!   `qindex` (0..=255) / `quality` (arbitrary f32 bit patterns, NaN
//!   and infinities included) / `Vp8EncoderConfig` parameter envelope.
//! * The `VideoFrame → I420Frame` plane extraction: attacker-shaped
//!   plane counts, strides, and buffer lengths. A mis-shaped frame must
//!   surface `Err`, never a panic or an out-of-bounds slice — this is
//!   the exact seam where a `stride < width` row repack or a
//!   `stride * height` overflow would detonate.
//! * The `send_frame` / `receive_packet` / `flush` lifecycle contract:
//!   `NeedMore` before EOF, `Eof` after the flush drain, and the
//!   lagged impl's lookahead buffering across sends.
//!
//! Oracles beyond panic-freedom:
//!
//! 1. Factory polarity — in-range parameters (Yuv420P, 1..=16383 dims,
//!    qindex ≤ 127) must produce an encoder; the hostile-parameter leg
//!    must produce `Err`.
//! 2. `output_params()` carries the negotiated width / height / codec.
//! 3. Every packet emitted from well-formed sends decodes cleanly in
//!    one long-lived `Vp8DecoderState` with the negotiated visible
//!    dimensions (the two encode impls only ever emit self-consistent
//!    streams; a decoder rejection is a real cross-side finding).
//! 4. If at least one well-formed frame was sent, flush + drain must
//!    yield at least one packet (an encoder that swallows its input is
//!    a finding), and the drain must terminate with `Eof`.
//!
//! Caps: encode dimensions ≤ 64×64 luma, ≤ 6 sends per iteration,
//! input ≤ 4 KiB. The hostile-parameter factory leg never encodes, so
//! wire-legal 16383×16383 declarations stay allocation-free.

use libfuzzer_sys::fuzz_target;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_vp8::encoder::{make_encoder_with_config, Vp8EncoderConfig};
use oxideav_vp8::{
    make_encoder, make_encoder_with_qindex, make_encoder_with_quality, LoopFilterMode,
    TrellisStrength, Vp8DecoderState,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 16;
const MAX_DIM: u32 = 64;
const MAX_SENDS: usize = 6;

/// Build the canonical VP8 `CodecParameters`.
fn vp8_params(width: u32, height: u32) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params
}

/// Build a well-formed `width × height` I420 `VideoFrame` with an
/// attacker-chosen fill and optional benign stride padding.
fn well_formed_frame(width: u32, height: u32, fill: u8, pad: usize, pts: i64) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    VideoFrame {
        pts: Some(pts),
        planes: vec![
            VideoPlane {
                stride: w + pad,
                data: vec![fill; (w + pad) * h],
            },
            VideoPlane {
                stride: uvw + pad,
                data: vec![fill ^ 0x40; (uvw + pad) * uvh],
            },
            VideoPlane {
                stride: uvw + pad,
                data: vec![fill ^ 0x80; (uvw + pad) * uvh],
            },
        ],
    }
}

/// Build a deliberately mis-shaped `VideoFrame`. Every shape must be
/// answered with `Err`, never a panic.
fn hostile_frame(width: u32, height: u32, shape: u8, fill: u8) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let plane = |stride: usize, len: usize| VideoPlane {
        stride,
        data: vec![fill; len],
    };
    match shape % 5 {
        // Too few planes.
        0 => VideoFrame {
            pts: Some(0),
            planes: vec![plane(w, w * h), plane(uvw, uvw * uvh)],
        },
        // Luma stride narrower than the row width (the row-repack
        // overread shape: len == stride * h passes a naive length
        // check, but the last row needs stride*(h-1)+w bytes).
        1 if w > 1 => VideoFrame {
            pts: Some(0),
            planes: vec![
                plane(w - 1, (w - 1) * h),
                plane(uvw, uvw * uvh),
                plane(uvw, uvw * uvh),
            ],
        },
        // Short luma buffer under an honest stride.
        2 => VideoFrame {
            pts: Some(0),
            planes: vec![
                plane(w, (w * h).saturating_sub(1)),
                plane(uvw, uvw * uvh),
                plane(uvw, uvw * uvh),
            ],
        },
        // Overflow probe: an absurd stride whose `stride * (height-1)`
        // wraps usize. The validation must use checked arithmetic.
        // (With a single luma row the stride never multiplies, so the
        // frame is legal — the guard keeps this arm honestly hostile.)
        3 if h > 1 => VideoFrame {
            pts: Some(0),
            planes: vec![
                plane(usize::MAX / 2 + 1, w * h),
                plane(uvw, uvw * uvh),
                plane(uvw, uvw * uvh),
            ],
        },
        // Zero-stride chroma with a plausible buffer.
        _ => VideoFrame {
            pts: Some(0),
            planes: vec![plane(w, w * h), plane(0, uvw * uvh), plane(0, uvw * uvh)],
        },
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = 1 + u32::from(data[0]) % MAX_DIM;
    let height = 1 + u32::from(data[1]) % MAX_DIM;
    let factory_pick = data[2] % 4;
    let qindex_raw = data[3];
    let quality_raw = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let config = Vp8EncoderConfig {
        qindex: data[8] % 128,
        lf_level: data[9] % 64,
        lf_mode: match data[10] % 3 {
            0 => LoopFilterMode::Auto,
            1 => LoopFilterMode::Normal,
            _ => LoopFilterMode::Simple,
        },
        golden_interval: u32::from(data[11] % 8),
        alt_ref_interval: u32::from(data[12] % 5),
        lookahead_window: u32::from(data[13] % 5),
        target_bitrate_bps: u32::from(data[14]) * 1024,
        auto_loop_filter: (data[15] & 0x01) != 0,
        trellis_strength: if (data[15] & 0x02) != 0 {
            TrellisStrength::OFF
        } else {
            TrellisStrength::DEFAULT
        },
    };
    let payload = &data[HEADER_BYTES..];

    // ---- Hostile-parameter factory leg (no allocation, no encode) ----
    // Each shape must be rejected with Err.
    {
        let mut missing = CodecParameters::video(CodecId::new("vp8"));
        missing.media_type = MediaType::Video;
        assert!(
            make_encoder(&missing).is_err(),
            "missing dimensions must be rejected"
        );
        assert!(
            make_encoder(&vp8_params(0, height)).is_err(),
            "zero width must be rejected"
        );
        assert!(
            make_encoder(&vp8_params(16384, height)).is_err(),
            "15-bit width must be rejected"
        );
        let mut bad_fmt = vp8_params(width, height);
        bad_fmt.pixel_format = Some(PixelFormat::Yuv444P);
        assert!(
            make_encoder(&bad_fmt).is_err(),
            "non-4:2:0 pixel format must be rejected"
        );
        if qindex_raw > 127 {
            assert!(
                make_encoder_with_qindex(&vp8_params(width, height), qindex_raw).is_err(),
                "qindex {qindex_raw} must be rejected"
            );
        }
    }

    // ---- Constructing factory leg -----------------------------------
    let params = vp8_params(width, height);
    let mut enc: Box<dyn Encoder> = match factory_pick {
        0 => make_encoder(&params).expect("make_encoder must accept in-range params"),
        1 => make_encoder_with_qindex(&params, qindex_raw % 128)
            .expect("make_encoder_with_qindex must accept qindex <= 127"),
        // Arbitrary f32 bit patterns (NaN / ±inf included) are
        // documented as clamped, never rejected.
        2 => make_encoder_with_quality(&params, quality_raw)
            .expect("make_encoder_with_quality must accept any quality scalar"),
        _ => make_encoder_with_config(&params, config)
            .expect("make_encoder_with_config must accept an in-range config"),
    };

    let out = enc.output_params();
    assert_eq!(out.width, Some(width), "output_params width drift");
    assert_eq!(out.height, Some(height), "output_params height drift");

    // ---- send / receive / flush lifecycle ----------------------------
    let sends = 1 + payload.first().map_or(0, |b| usize::from(*b)) % MAX_SENDS;
    let mut dec = Vp8DecoderState::new();
    let mut well_formed_sent = 0usize;
    let mut packets = Vec::new();

    let drain = |enc: &mut Box<dyn Encoder>, packets: &mut Vec<Vec<u8>>| loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p.data),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet surfaced an unexpected error: {e:?}"),
        }
    };

    for i in 0..sends {
        let ctl = payload.get(1 + i).copied().unwrap_or(0);
        let fill = payload
            .get(1 + MAX_SENDS + i)
            .copied()
            .unwrap_or(0x80)
            .wrapping_add(i as u8);
        if ctl & 0x03 == 0x03 {
            // Hostile-frame leg: must Err, must not perturb the stream.
            let bad = hostile_frame(width, height, ctl >> 2, fill);
            assert!(
                enc.send_frame(&Frame::Video(bad)).is_err(),
                "mis-shaped VideoFrame (shape {}) must be rejected",
                (ctl >> 2) % 5
            );
        } else {
            let pad = usize::from(ctl >> 4) % 5;
            let frame = well_formed_frame(width, height, fill, pad, i as i64);
            enc.send_frame(&Frame::Video(frame))
                .expect("well-formed frame must be accepted");
            well_formed_sent += 1;
        }
        drain(&mut enc, &mut packets);
    }

    enc.flush().expect("flush must succeed");
    // Post-flush drain must terminate with Eof, never hang on NeedMore.
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p.data),
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => panic!("receive_packet returned NeedMore after flush"),
            Err(e) => panic!("post-flush receive_packet surfaced {e:?}"),
        }
    }

    if well_formed_sent > 0 {
        assert!(
            !packets.is_empty(),
            "{well_formed_sent} well-formed frames in, zero packets out"
        );
    }

    // ---- decode lockstep ---------------------------------------------
    for (i, bytes) in packets.iter().enumerate() {
        let decoded = dec
            .decode_frame(bytes)
            .unwrap_or_else(|e| panic!("decoder rejected encoder packet {i}: {e:?}"));
        assert_eq!(decoded.width, width, "decoded width drift on packet {i}");
        assert_eq!(decoded.height, height, "decoded height drift on packet {i}");
    }
});
