//! Encoder-side §14.1 dequant correctness regression: the inter MB
//! picker must dequantise its forward-transform output before handing
//! it to the reconstruction orchestrators.
//!
//! RFC 6386 §14.1 (pages 78-80): "the coefficients output by the
//! forward DCT/WHT are quantised by integer division against the
//! per-block §14.1 step tables; the reverse path (the path the decoder
//! walks, and the path the encoder's own reconstruction must walk to
//! stay in lock-step) multiplies by the same step before the inverse
//! transform." The `motion_comp::reconstruct_inter_mb` and
//! `reconstruct_split_mv_mb` orchestrators document this on their
//! parameter names (`*_coeffs_dequant`) and the keyframe encoder path
//! already performs the dequant step on a copy of `MbCoeffs` before
//! calling its keyframe-mode reconstructor.
//!
//! Prior to the fix exercised here, the inter MB picker handed the
//! still-quantised forward-transform output straight into the inter
//! reconstructor, so the encoder's stored reconstruction was off by
//! the §14.1 dequant factor on every coded sub-block. The decoder
//! self-decode walked the spec-correct path and recovered the
//! quantised-then-dequantised pixels, leaving a mismatch between the
//! encoder's `_p_reconstruction` and the decoder's `Vp8DecodedFrame`
//! on the same wire.
//!
//! The two tests below pin that the encoder reconstruction matches
//! the decoder self-decode at two §15 loop-filter operating points:
//!
//!   * `loop_filter_level = 0` — §15 is gated off in both directions,
//!     so any residual mismatch is purely the §14.2 / §14.5 inverse
//!     transform + clamp path and isolates the §14.1 dequant defect
//!     this fix targets.
//!   * `loop_filter_level = 32` — §15 runs on both sides; the encoder
//!     applies its own post-walk filter pass and the decoder's §15
//!     pass runs over the same wire. With the dequant fix in place
//!     both ends feed §15 the same pre-filter pixels, so the post-§15
//!     reconstructions agree as well.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_zero_mv, I420Frame, KeyframeParams,
    Vp8DecoderState,
};

/// Build a 32×32 I420 frame with a smooth Y gradient and gently
/// varying chroma — deterministic, no PRNG. Small enough that the
/// raster walks fast under cargo test, large enough to exercise the
/// inter MB picker's §16.2/§16.4 branches.
fn structured_frame_32x32(luma_offset: i16, chroma_offset: i16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let v = 40 + (r as i16) + (c as i16) + luma_offset;
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

fn encode_and_self_decode_p_frame(
    loop_filter_level: u8,
) -> (oxideav_vp8::Vp8DecodedFrame, oxideav_vp8::KeyframePlanes) {
    let (w, h) = (32u32, 32u32);
    let (y0, u0, v0) = structured_frame_32x32(0, 0);
    let (y1, u1, v1) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(w, h, &y0, &u0, &v0);
    let frame_p = I420Frame::packed(w, h, &y1, &u1, &v1);

    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
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

fn assert_planes_match(
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
fn p_frame_encoder_recon_matches_decoder_recon_lf_disabled() {
    // §15 gated off on both sides: the only thing standing between
    // the encoder's stored reconstruction and the decoder's
    // self-decoded output is the §14.1 dequant / §14.2 inverse
    // transform / §14.5 clamp path. A mismatch here is the dequant
    // defect this fix targets.
    let (decoded, recon) = encode_and_self_decode_p_frame(0);
    assert_planes_match(&decoded, &recon, "loop_filter_level = 0");
}

#[test]
fn p_frame_encoder_recon_matches_decoder_recon_lf_enabled() {
    // §15 runs on both sides. For the post-§15 reconstructions to
    // agree, the pre-§15 pixels must agree, which requires the §14.1
    // dequant step on the encoder's reconstruction path.
    let (decoded, recon) = encode_and_self_decode_p_frame(32);
    assert_planes_match(&decoded, &recon, "loop_filter_level = 32");
}
