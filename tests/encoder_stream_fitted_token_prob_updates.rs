//! Round-159 follow-up to the round-157 (keyframe) / round-158 (inter)
//! §13.4 observed-counts fitter: thread the fitter through the multi-
//! frame stream drivers.
//!
//! Pins:
//!
//! 1. [`Vp8KeyframeStreamEncoder::encode_frame_with_fitted_token_prob_updates`]
//!    never grows the wire relative to
//!    [`Vp8KeyframeStreamEncoder::encode_frame`] on any frame of a
//!    multi-frame sequence (the round-157 safety guard, lifted into the
//!    stream driver).
//! 2. The fitted-stream bytes replay through
//!    [`Vp8DecoderState::decode_frame`] and every recovered frame clears
//!    the round target of PSNR ≥ 30 dB against its corresponding source.
//! 3. The §9.7 / §9.8 three-slot reference-frame ladder after a fitted
//!    K-frame is byte-equal across all three slots — the fitter's
//!    matching-planes guarantee carries through the stream driver so
//!    `LAST`, `GOLDEN`, and `ALTREF` are populated with the same
//!    reconstruction (mirrors the keyframe driver's existing §9.7 / §9.8
//!    "all three slots take the post-§15 reconstruction" invariant).
//! 4. [`Vp8InterStreamEncoder::encode_frame_with_fitted_token_prob_updates`]
//!    never grows the wire relative to
//!    [`Vp8InterStreamEncoder::encode_frame`] across a multi-frame I + P
//!    sequence; the recovered frames clear the same 30 dB target.
//! 5. The K/P interleave the fitted inter stream produces matches the
//!    non-fitted inter stream's schedule frame-by-frame (the fitter
//!    has zero effect on the scheduling layer).
//! 6. `force_keyframe = true` on the fitted-inter entry-point re-anchors
//!    the keyframe interval just like the non-fitted entry-point.
//!
//! No external decoder is invoked: every check is the crate's own
//! encoder fed straight into the crate's own
//! [`Vp8DecoderState::decode_frame`].

use oxideav_vp8::{
    FrameKind, I420Frame, KeyframeParams, Vp8DecodedFrame, Vp8DecoderState, Vp8InterStreamEncoder,
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

/// Build a `frame_idx`-parameterised synthetic I420 picture with enough
/// per-frame variation that fitter benefit isn't trivially zero.
///
/// * Luma: diagonal gradient with a per-frame phase shift; a moving
///   flat-128 inset square (forces both whole-block and B_PRED intra
///   activity for the keyframe path, and a translating residual block
///   for the inter path).
/// * Chroma: per-frame DC drift so a leak between frames in either
///   chroma plane is obvious.
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

fn keyframe_params(qi: u8) -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    }
}

// ---- Vp8KeyframeStreamEncoder ---------------------------------------------

/// Fitted keyframe-stream encode never grows the wire relative to the
/// caller-driven (default) keyframe-stream encode, frame-by-frame.
#[test]
fn fitted_keyframe_stream_never_grows_the_wire() {
    let width = 64u32;
    let height = 64u32;
    let qi = 32u8;
    let n_frames = 4usize;

    let sources: Vec<Source> = (0..n_frames)
        .map(|i| synthetic_frame(width, height, i))
        .collect();

    let mut enc_def = Vp8KeyframeStreamEncoder::new(keyframe_params(qi));
    let mut enc_fit = Vp8KeyframeStreamEncoder::new(keyframe_params(qi));
    for (i, src) in sources.iter().enumerate() {
        let bytes_def = enc_def
            .encode_frame(&src.frame())
            .unwrap_or_else(|e| panic!("default[{i}]: {e}"));
        let bytes_fit = enc_fit
            .encode_frame_with_fitted_token_prob_updates(&src.frame())
            .unwrap_or_else(|e| panic!("fitted[{i}]: {e}"));
        assert!(
            bytes_fit.len() <= bytes_def.len(),
            "fitted-keyframe-stream wire grew on frame {i}: {} > {}",
            bytes_fit.len(),
            bytes_def.len()
        );
        eprintln!(
            "frame {i}: default={} fitted={} delta={:+}",
            bytes_def.len(),
            bytes_fit.len(),
            bytes_fit.len() as isize - bytes_def.len() as isize
        );
    }
    assert_eq!(enc_fit.frame_count(), n_frames as u64);
    assert_eq!(enc_fit.dimensions(), Some((width, height)));
}

/// Fitted keyframe-stream bytes replay through the multi-frame
/// `Vp8DecoderState` and every frame clears the 30 dB target.
#[test]
fn fitted_keyframe_stream_round_trips_at_mid_quant() {
    let width = 48u32;
    let height = 32u32;
    let qi = 32u8;
    let n_frames = 5usize;

    let sources: Vec<Source> = (0..n_frames)
        .map(|i| synthetic_frame(width, height, i))
        .collect();

    let mut enc = Vp8KeyframeStreamEncoder::new(keyframe_params(qi));
    let mut byte_streams: Vec<Vec<u8>> = Vec::with_capacity(n_frames);
    for (i, src) in sources.iter().enumerate() {
        let bytes = enc
            .encode_frame_with_fitted_token_prob_updates(&src.frame())
            .unwrap_or_else(|e| panic!("encode[{i}]: {e}"));
        assert!(!bytes.is_empty(), "frame {i} bytes non-empty");
        // §9.1 bit 0 of byte 0 = 0 ⇒ key frame.
        assert_eq!(bytes[0] & 0x01, 0, "frame {i} should be a key frame");
        // §9.1 start code at offset 3.
        assert_eq!(&bytes[3..6], &[0x9d, 0x01, 0x2a], "frame {i} start code");
        byte_streams.push(bytes);
    }

    let mut dec = Vp8DecoderState::new();
    for (i, bytes) in byte_streams.iter().enumerate() {
        let out = dec
            .decode_frame(bytes)
            .unwrap_or_else(|e| panic!("decode_frame[{i}]: {e:?}"));
        assert_eq!(out.width, width);
        assert_eq!(out.height, height);
        let psnr = frame_psnr(&sources[i], &out);
        eprintln!("frame {i} fitted-stream self-decode PSNR = {psnr:.2} dB");
        assert!(
            psnr >= 30.0,
            "frame {i} PSNR {psnr:.2} dB below the 30 dB target"
        );
    }
}

/// The §9.7 / §9.8 keyframe slot-refresh invariant — all three slots
/// hold the same reconstruction after a fitted K-frame.
#[test]
fn fitted_keyframe_stream_refreshes_all_three_slots() {
    let src = synthetic_frame(32, 32, 0);
    let mut enc = Vp8KeyframeStreamEncoder::new(keyframe_params(32));
    enc.encode_frame_with_fitted_token_prob_updates(&src.frame())
        .expect("encode");
    let last = enc.last().expect("LAST");
    let golden = enc.golden().expect("GOLDEN");
    let altref = enc.altref().expect("ALTREF");
    assert_eq!(last.y, golden.y, "LAST.y == GOLDEN.y");
    assert_eq!(last.u, altref.u, "LAST.u == ALTREF.u");
    assert_eq!(golden.v, altref.v, "GOLDEN.v == ALTREF.v");
}

// ---- Vp8InterStreamEncoder ------------------------------------------------

/// Fitted inter-stream encode never grows the wire relative to the
/// caller-driven inter-stream encode, frame-by-frame, across an I + P
/// interleave at keyframe interval 3.
#[test]
fn fitted_inter_stream_never_grows_the_wire() {
    let width = 64u32;
    let height = 64u32;
    let qi = 32u8;
    let n_frames = 6usize;
    let interval = 3u64;

    let sources: Vec<Source> = (0..n_frames)
        .map(|i| synthetic_frame(width, height, i))
        .collect();

    let mut enc_def =
        Vp8InterStreamEncoder::new(keyframe_params(qi), interval).expect("non-zero interval");
    let mut enc_fit =
        Vp8InterStreamEncoder::new(keyframe_params(qi), interval).expect("non-zero interval");
    for (i, src) in sources.iter().enumerate() {
        let out_def = enc_def
            .encode_frame(&src.frame())
            .unwrap_or_else(|e| panic!("default[{i}]: {e}"));
        let out_fit = enc_fit
            .encode_frame_with_fitted_token_prob_updates(&src.frame())
            .unwrap_or_else(|e| panic!("fitted[{i}]: {e}"));
        // The fitter doesn't touch scheduling — kind must match.
        assert_eq!(
            out_def.kind, out_fit.kind,
            "frame {i} kind mismatch: default={:?} fitted={:?}",
            out_def.kind, out_fit.kind
        );
        assert_eq!(out_def.frame_index, out_fit.frame_index);
        assert!(
            out_fit.bytes.len() <= out_def.bytes.len(),
            "fitted-inter-stream wire grew on frame {i} ({:?}): {} > {}",
            out_fit.kind,
            out_fit.bytes.len(),
            out_def.bytes.len()
        );
        eprintln!(
            "frame {i} ({:?}): default={} fitted={} delta={:+}",
            out_fit.kind,
            out_def.bytes.len(),
            out_fit.bytes.len(),
            out_fit.bytes.len() as isize - out_def.bytes.len() as isize
        );
    }
}

/// Fitted inter-stream bytes replay through `Vp8DecoderState` and every
/// frame clears the 30 dB target.
#[test]
fn fitted_inter_stream_round_trips_at_mid_quant() {
    let width = 48u32;
    let height = 32u32;
    let qi = 32u8;
    let n_frames = 5usize;
    let interval = 3u64;

    let sources: Vec<Source> = (0..n_frames)
        .map(|i| synthetic_frame(width, height, i))
        .collect();

    let mut enc =
        Vp8InterStreamEncoder::new(keyframe_params(qi), interval).expect("non-zero interval");
    let mut emitted: Vec<(FrameKind, Vec<u8>)> = Vec::with_capacity(n_frames);
    for (i, src) in sources.iter().enumerate() {
        let out = enc
            .encode_frame_with_fitted_token_prob_updates(&src.frame())
            .unwrap_or_else(|e| panic!("encode[{i}]: {e}"));
        // §9.1 bit 0 of byte 0: 0 ⇒ K, 1 ⇒ P.
        match out.kind {
            FrameKind::Key => assert_eq!(out.bytes[0] & 0x01, 0, "frame {i} K-tag bit"),
            FrameKind::InterZeroMv => {
                assert_eq!(out.bytes[0] & 0x01, 1, "frame {i} P-tag bit")
            }
        }
        emitted.push((out.kind, out.bytes));
    }
    // First frame is always K.
    assert!(matches!(emitted[0].0, FrameKind::Key));

    let mut dec = Vp8DecoderState::new();
    for (i, (_, bytes)) in emitted.iter().enumerate() {
        let out = dec
            .decode_frame(bytes)
            .unwrap_or_else(|e| panic!("decode[{i}]: {e:?}"));
        assert_eq!(out.width, width);
        assert_eq!(out.height, height);
        let psnr = frame_psnr(&sources[i], &out);
        eprintln!("frame {i} fitted-inter-stream PSNR = {psnr:.2} dB");
        assert!(
            psnr >= 30.0,
            "frame {i} PSNR {psnr:.2} dB below the 30 dB target"
        );
    }
}

/// `force_keyframe = true` re-anchors the interval the same way it does
/// on the non-fitted entry-point — the fitter has zero effect on
/// scheduling.
#[test]
fn fitted_inter_stream_force_keyframe_reanchors_interval() {
    let width = 32u32;
    let height = 32u32;
    let qi = 32u8;
    let mut enc = Vp8InterStreamEncoder::new(keyframe_params(qi), 4).expect("non-zero interval");

    // K, P, P-forced-K, P, P, P, K (re-anchored at frame 2) — same
    // expected schedule the non-fitted equivalent test pins on the
    // non-fitted entry-point in `crate::stream::tests`.
    let schedule = [
        (false, FrameKind::Key),
        (false, FrameKind::InterZeroMv),
        (true, FrameKind::Key),
        (false, FrameKind::InterZeroMv),
        (false, FrameKind::InterZeroMv),
        (false, FrameKind::InterZeroMv),
        (false, FrameKind::Key),
    ];
    for (i, (force, expected)) in schedule.iter().enumerate() {
        let src = synthetic_frame(width, height, i);
        let out = enc
            .encode_frame_with_force_and_fitted_token_prob_updates(&src.frame(), *force)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));
        assert_eq!(out.kind, *expected, "frame {i} kind mismatch");
    }
    assert_eq!(enc.last_keyframe_index(), Some(6));
}
