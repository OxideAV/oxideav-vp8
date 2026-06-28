//! Encoder side of the §9.4 `filter_type` knob: the encoder must
//! offer the §15.2 *simple* loop filter alongside the existing §15.3
//! *normal* path, and on both branches the encoder's stored
//! reconstruction must match what the decoder rebuilds from the same
//! wire byte-for-byte (lockstep).
//!
//! RFC 6386 §9.4 page 31: `filter_type` is a 1-bit field — `false`
//! selects the §15.3 normal loop filter, `true` the §15.2 simple
//! filter. Both ends consume the same bit; the §15 implementation in
//! `crate::loop_filter::filter_inter_frame` already branches on
//! `FrameFilterConfig::simple`, and the §19.2 frame header parse path
//! sets `header.filter_type` from the same wire bit.
//!
//! Before this round the encoder hardwired `filter_type = false` on
//! every entry point. `KeyframeParams::filter_type` now exposes the
//! choice and threads it into BOTH:
//!
//!   * `write_loop_filter` / `write_loop_filter_with_deltas` — the
//!     §19.2 bit the decoder reads.
//!   * `FrameFilterConfig::simple` — the encoder's own §15 post-walk
//!     pass that mutates its reconstruction buffer.
//!
//! Each test below encodes a 32×32 I + P pair at `loop_filter_level
//! = 32` (§15 active on both sides), once at `filter_type = false`
//! (the historical wire) and once at `filter_type = true`, and pins:
//!
//!   1. The encoder's stored P-frame reconstruction equals the
//!      decoder's self-decoded P-frame byte-for-byte on every plane.
//!   2. Flipping `filter_type` produces an *observably different*
//!      decoded picture (the §15.2 simple filter touches different
//!      pixel sets than §15.3 normal — chroma planes are
//!      *un*touched, and only edge pixels at MB / sub-block
//!      boundaries are filtered without the inner-window high-edge-
//!      variance branch — so on any frame with content at MB seams
//!      the two outputs cannot be identical).
//!   3. `KeyframeParams::default()` keeps `filter_type = false`
//!      (round-153 wire compatibility).

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_zero_mv, I420Frame, KeyframeParams,
    Vp8DecoderState,
};

/// Build a 32×32 I420 frame with a deliberately MB-seam-crossing Y
/// gradient (different ramp on each half) and gently varying chroma —
/// content the §15 filter actually does work on. Deterministic, no
/// PRNG.
fn seam_frame_32x32(luma_offset: i16, chroma_offset: i16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // Two ramps, one per MB column, so the vertical MB seam at
            // x = 16 carries an actual luminance step — §15 has
            // something to filter there.
            let base = if c < 16 {
                40 + (c as i16)
            } else {
                100 + (c as i16)
            };
            let v = base + (r as i16) / 2 + luma_offset;
            y[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for cc in 0..cw {
            let uv = 100 + (r as i16) + chroma_offset;
            u[r * cw + cc] = uv.clamp(0, 255) as u8;
            let vv = 150 - (cc as i16) + chroma_offset;
            v[r * cw + cc] = vv.clamp(0, 255) as u8;
        }
    }
    (y, u, v)
}

fn encode_pair_and_decode(
    filter_type: bool,
) -> (oxideav_vp8::Vp8DecodedFrame, oxideav_vp8::KeyframePlanes) {
    let (w, h) = (32u32, 32u32);
    let (y0, u0, v0) = seam_frame_32x32(0, 0);
    let (y1, u1, v1) = seam_frame_32x32(3, 2);
    let frame_i = I420Frame::packed(w, h, &y0, &u0, &v0);
    let frame_p = I420Frame::packed(w, h, &y1, &u1, &v1);

    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 32,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I frame");
    let (p_bytes, p_recon) =
        encode_p_frame_zero_mv(&frame_p, &i_recon, &params).expect("encode P frame");

    let mut state = Vp8DecoderState::new();
    let _ = state.decode_frame(&i_bytes).expect("decode I frame");
    let p_decoded = state.decode_frame(&p_bytes).expect("decode P frame");

    (p_decoded, p_recon)
}

fn assert_lockstep(
    decoded: &oxideav_vp8::Vp8DecodedFrame,
    recon: &oxideav_vp8::KeyframePlanes,
    label: &str,
) {
    assert_eq!(
        decoded.y.as_slice(),
        recon.y.as_slice(),
        "{label}: Y plane encoder-recon vs decoder-recon must agree byte-for-byte"
    );
    assert_eq!(
        decoded.u.as_slice(),
        recon.u.as_slice(),
        "{label}: U plane encoder-recon vs decoder-recon must agree byte-for-byte"
    );
    assert_eq!(
        decoded.v.as_slice(),
        recon.v.as_slice(),
        "{label}: V plane encoder-recon vs decoder-recon must agree byte-for-byte"
    );
}

#[test]
fn p_frame_encoder_decoder_lockstep_normal_filter() {
    // Sanity baseline: with `filter_type = false` the round-153 wire
    // and post-walk filter selection still produce a lockstep
    // reconstruction. This is the same coverage the round-152b
    // `encoder_pframe_loop_filter_recon` test pins, repeated here on
    // a deliberately seam-crossing fixture so the §15.3 filter has
    // something to filter (and so the contrast with the simple-filter
    // case below is a real not-trivial difference).
    let (decoded, recon) = encode_pair_and_decode(false);
    assert_lockstep(&decoded, &recon, "filter_type = false (normal)");
}

#[test]
fn p_frame_encoder_decoder_lockstep_simple_filter() {
    // With `filter_type = true` the encoder writes the §15.2 bit on
    // the wire AND switches its own post-walk filter into the §15.2
    // simple branch. Both ends consume the same wire and run the
    // same filter, so the encoder's stored reconstruction must match
    // the decoder's self-decode byte-for-byte just like the normal-
    // filter case above.
    let (decoded, recon) = encode_pair_and_decode(true);
    assert_lockstep(&decoded, &recon, "filter_type = true (simple)");
}

#[test]
fn simple_filter_output_differs_from_normal_filter_output() {
    // The §15.2 simple filter is a different kernel from §15.3 normal
    // (no chroma plane and only a 4-pixel kernel on luma without the
    // inner-window high-edge-variance branch). On any seam-crossing
    // content the two cannot produce the same decoded pixels —
    // otherwise our `filter_type` knob would be load-bearing on the
    // header but a no-op on the decoded picture, which would be a
    // bug.
    let (decoded_normal, _) = encode_pair_and_decode(false);
    let (decoded_simple, _) = encode_pair_and_decode(true);
    assert_ne!(
        decoded_normal.y, decoded_simple.y,
        "normal vs simple §15 filter must produce different Y planes on seam-crossing content"
    );
}

#[test]
fn keyframe_params_default_keeps_filter_type_false() {
    // `KeyframeParams::default()` must continue to be the round-153
    // wire — adding `filter_type` cannot silently rewrite the bytes
    // every existing caller has been emitting. Defaulting to
    // `filter_type = false` keeps the §15.3 normal filter selected
    // for every legacy call site.
    let params = KeyframeParams::default();
    assert!(
        !params.filter_type,
        "KeyframeParams::default() must keep filter_type = false for round-153 wire compatibility"
    );
}
