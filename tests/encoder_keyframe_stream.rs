//! Multi-frame VP8 keyframe stream encoder driver round-trip
//! (RFC 6386 §4 / §9.1 / §9.7 / §9.8 — "one packet = one frame",
//! key frames refresh all three reference slots).
//!
//! Builds a synthetic 5-frame I420 sequence whose per-frame pixel
//! pattern changes from frame to frame (so each round-trip really has
//! to follow the per-frame input, not just regurgitate the first
//! frame), drives the sequence through
//! [`oxideav_vp8::Vp8KeyframeStreamEncoder::encode_frame`], replays the
//! emitted bytes through [`oxideav_vp8::Vp8DecoderState::decode_frame`]
//! (the same multi-frame decoder a real consumer would use), and
//! asserts every recovered frame reaches the round target of PSNR ≥ 30
//! dB against its corresponding input.
//!
//! This is a fully self-contained black-box check: no external codec
//! is consulted, only the crate's own encoder + decoder.

use oxideav_vp8::{
    I420Frame, KeyframeParams, StreamEncodeError, Vp8DecodedFrame, Vp8DecoderState,
    Vp8KeyframeStreamEncoder,
};

/// A source I420 picture with tightly-packed planes.
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

/// Build a `frame_idx`-parameterised synthetic I420 picture so each
/// frame in the sequence is meaningfully different.
///
/// * Luma is a shifted diagonal gradient (`(col + row + 32 * frame_idx) % 256`)
///   plus a moving 128-flat square whose top-left corner walks by 4
///   pixels per frame on both axes — exercises both whole-block and
///   `B_PRED` intra paths and forces the per-frame reconstruction to
///   genuinely follow the per-frame input.
/// * Chroma is a flat per-frame DC value (`120 + 5 * frame_idx`,
///   `130 - 5 * frame_idx`) so a frame-to-frame leak in either chroma
///   plane would also show up.
fn synthetic_frame(width: u32, height: u32, frame_idx: usize) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;

    let phase = (frame_idx * 32) as i32;
    let mut y = vec![0u8; w * h];
    for (row, chunk) in y.chunks_mut(w).enumerate() {
        for (col, px) in chunk.iter_mut().enumerate() {
            let mut v = (((col as i32 + row as i32 + phase) & 0xff) as u8) / 2 + 32;

            // Moving flat-128 square.
            let sq_x0 = (4 * frame_idx).min(w.saturating_sub(8));
            let sq_y0 = (4 * frame_idx).min(h.saturating_sub(8));
            let sq_w = (w / 3).max(4);
            let sq_h = (h / 3).max(4);
            if col >= sq_x0 && col < sq_x0 + sq_w && row >= sq_y0 && row < sq_y0 + sq_h {
                v = 128;
            }
            *px = v;
        }
    }

    let u_val = (120 + 5 * frame_idx as i32).clamp(0, 255) as u8;
    let v_val = (130 - 5 * frame_idx as i32).clamp(0, 255) as u8;
    let u = vec![u_val; cw * ch];
    let v = vec![v_val; cw * ch];

    Source {
        width,
        height,
        y,
        u,
        v,
    }
}

/// Mean-squared error between two equal-length byte planes.
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

/// Whole-frame PSNR across the three planes (combined MSE over all
/// luma + chroma samples). 8-bit peak = 255.
fn frame_psnr(src: &Source, dec: &Vp8DecodedFrame) -> f64 {
    let total = src.y.len() + src.u.len() + src.v.len();
    let combined_se = plane_mse(&src.y, &dec.y) * src.y.len() as f64
        + plane_mse(&src.u, &dec.u) * src.u.len() as f64
        + plane_mse(&src.v, &dec.v) * src.v.len() as f64;
    let mse = combined_se / total as f64;
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// The round target: a 5-frame synthetic sequence at mid quantiser.
/// Every per-frame self-decode through `Vp8DecoderState` must clear
/// 30 dB against the *corresponding* input frame.
#[test]
fn five_frame_keyframe_stream_mid_quant_meets_30db_per_frame() {
    let width = 48u32;
    let height = 32u32;
    let qi = 32u8;
    let n_frames = 5usize;

    // Build the per-frame source pictures and the encoded byte buffers.
    let sources: Vec<Source> = (0..n_frames)
        .map(|i| synthetic_frame(width, height, i))
        .collect();

    let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    });
    let mut byte_streams: Vec<Vec<u8>> = Vec::with_capacity(n_frames);
    for (i, src) in sources.iter().enumerate() {
        let frame = src.frame();
        let bytes = enc
            .encode_frame(&frame)
            .unwrap_or_else(|e| panic!("encode_frame[{i}]: {e}"));
        assert!(!bytes.is_empty(), "frame {i} bytes non-empty");
        byte_streams.push(bytes);
    }
    assert_eq!(enc.frame_count(), n_frames as u64);
    assert_eq!(enc.dimensions(), Some((width, height)));

    // Sanity: all three reference slots are populated by the keyframe
    // refresh.
    assert!(enc.last().is_some(), "LAST slot populated");
    assert!(enc.golden().is_some(), "GOLDEN slot populated");
    assert!(enc.altref().is_some(), "ALTREF slot populated");

    // Now replay through the multi-frame decoder driver.
    let mut dec = Vp8DecoderState::new();
    let mut psnrs = Vec::with_capacity(n_frames);
    for (i, bytes) in byte_streams.iter().enumerate() {
        let out = dec
            .decode_frame(bytes)
            .unwrap_or_else(|e| panic!("decode_frame[{i}]: {e:?}"));
        assert_eq!(out.width, width, "frame {i} decoded width");
        assert_eq!(out.height, height, "frame {i} decoded height");

        let psnr = frame_psnr(&sources[i], &out);
        eprintln!("frame {i} self-decode PSNR = {psnr:.2} dB");
        assert!(
            psnr >= 30.0,
            "frame {i} PSNR {psnr:.2} dB below the 30.0 dB target"
        );
        psnrs.push(psnr);
    }

    // Whole-stream summary line for the CI log.
    let mean: f64 = psnrs.iter().sum::<f64>() / psnrs.len() as f64;
    eprintln!("5-frame mean self-decode PSNR = {mean:.2} dB");
}

/// Confirms each frame of the sequence is structurally a key frame
/// (RFC 6386 §9.1 — the `key_frame` bit lives in the bottom bit of the
/// frame tag's first byte, *cleared* on a key frame).
#[test]
fn every_emitted_frame_is_a_keyframe() {
    let width = 32u32;
    let height = 32u32;
    let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
    let mut byte_streams = Vec::new();
    for i in 0..4 {
        let src = synthetic_frame(width, height, i);
        let bytes = enc.encode_frame(&src.frame()).expect("encode");
        // RFC 6386 §9.1: bit 0 of the first uncompressed-header byte =
        // `key_frame == 0` flags a key frame.
        let key_bit = bytes[0] & 0x01;
        assert_eq!(key_bit, 0, "frame {i} should be a key frame");
        // RFC 6386 §9.1: a key frame is followed by the 3-byte start
        // code `0x9d 0x01 0x2a` at offset 3.
        assert_eq!(&bytes[3..6], &[0x9d, 0x01, 0x2a], "frame {i} start code");
        byte_streams.push(bytes);
    }

    // The §9.1 / §16.1 invariant: every frame independently decodable
    // from a fresh decoder state (no carry from earlier frames is
    // required), the defining property of "all keyframes".
    for (i, bytes) in byte_streams.iter().enumerate() {
        let mut fresh = Vp8DecoderState::new();
        let out = fresh
            .decode_frame(bytes)
            .unwrap_or_else(|e| panic!("fresh decode of frame {i}: {e:?}"));
        assert_eq!(out.width, width);
        assert_eq!(out.height, height);
    }
}

/// Dimensions are locked at the first `encode_frame` call; a
/// subsequent frame with different dimensions surfaces
/// `StreamEncodeError::DimensionsChanged`.
#[test]
fn stream_rejects_mid_stream_resize() {
    let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
    let first = synthetic_frame(32, 32, 0);
    enc.encode_frame(&first.frame()).expect("first frame");

    let second = synthetic_frame(48, 32, 1);
    let err = enc
        .encode_frame(&second.frame())
        .expect_err("resize should be rejected");
    assert!(
        matches!(
            err,
            StreamEncodeError::DimensionsChanged {
                first: (32, 32),
                got: (48, 32)
            }
        ),
        "unexpected error: {err:?}"
    );
    // Failed call must not advance the counter.
    assert_eq!(enc.frame_count(), 1);
    assert_eq!(enc.dimensions(), Some((32, 32)));
}

/// Stream encoder behaviour matches the standalone `encode_keyframe`
/// for the first frame (PSNR within 1e-9 — same code path, same
/// reconstruction).
#[test]
fn first_frame_matches_standalone_encode_keyframe_psnr() {
    use oxideav_vp8::{decode_vp8, encode_keyframe};

    let width = 32u32;
    let height = 32u32;
    let src = synthetic_frame(width, height, 0);

    let params = KeyframeParams::default();
    let direct_bytes = encode_keyframe(&src.frame(), &params).expect("direct");

    let mut enc = Vp8KeyframeStreamEncoder::new(params);
    let stream_bytes = enc.encode_frame(&src.frame()).expect("stream");

    // The stream encoder builds the same frame as the standalone API
    // for an identical first frame — byte-equal.
    assert_eq!(
        direct_bytes, stream_bytes,
        "stream encoder must agree with encode_keyframe on the first frame"
    );

    let dec_direct = decode_vp8(&direct_bytes).expect("decode direct");
    let dec_stream = decode_vp8(&stream_bytes).expect("decode stream");
    let psnr_direct = frame_psnr(&src, &dec_direct);
    let psnr_stream = frame_psnr(&src, &dec_stream);
    assert!(
        (psnr_direct - psnr_stream).abs() < 1e-9,
        "psnr mismatch direct={psnr_direct} stream={psnr_stream}"
    );
}
