//! End-to-end I + P self-decode roundtrip test for the minimum-viable
//! VP8 inter encoder ([`oxideav_vp8::encode_p_frame_zero_mv`]).
//!
//! Encodes a 2-frame sequence:
//!
//!   * Frame 0: a structured I420 picture, emitted as a key frame via
//!     [`oxideav_vp8::encode_keyframe_with_reconstruction`]. The post-§15
//!     reconstruction is captured and threaded into the P-frame encoder
//!     as the LAST reference.
//!   * Frame 1: the same picture with a small constant offset applied
//!     to every plane (simulating a slow brightness drift / pan).
//!     Emitted as a P-frame via [`oxideav_vp8::encode_p_frame_zero_mv`]
//!     with every macroblock as ZEROMV from LAST.
//!
//! Both frames are then fed sequentially into [`oxideav_vp8::Vp8DecoderState`],
//! and the P-frame's self-decode whole-frame PSNR must clear the round's
//! 30 dB bar at a mid quantiser (`yac_qi = 32`). This pins:
//!
//!   * the §9.1 inter frame-tag layout (`frame_type = 1`, no resize, no
//!     start code) the encoder emits;
//!   * the §19.2 inter-frame coded-header bit sequence (refresh ladder,
//!     sign biases, refresh_entropy_probs, prob_intra / prob_last /
//!     prob_gf, the no-update tails on the §13.4 / §17.2 update blocks);
//!   * the §19.3 per-MB header layout (segment_id absent →
//!     mb_skip_coeff → is_inter_mb → ref_frame selector → inter-mode
//!     tree), with the §16.3 census-driven probability evolution
//!     matching what the decoder reconstructs as it walks the same MBs;
//!   * the §18 motion-comp identity copy at MV (0,0) on the encoder
//!     side matching the decoder's interpretation of ZEROMV/LAST.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_zero_mv, I420Frame, KeyframeParams,
    Vp8DecoderState,
};

/// Build a 64×64 I420 frame with a smooth Y gradient and gently
/// varying chroma — repeatable, deterministic, no PRNG.
fn structured_frame_64x64(luma_offset: i16, chroma_offset: i16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (32usize, 32usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // Diagonal gradient in 40..200, clamped, with a per-frame
            // brightness offset to simulate slow scene drift.
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

fn plane_psnr(src: &[u8], rec: &[u8]) -> f64 {
    assert_eq!(src.len(), rec.len());
    let mut sse: u64 = 0;
    for (a, b) in src.iter().zip(rec.iter()) {
        let d = *a as i32 - *b as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / src.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn i_plus_p_zero_mv_self_decode_psnr_clears_30db_at_mid_qi() {
    // Two-frame synthetic scene: slow constant brightness drift between
    // the I frame and the P frame. The P-frame's ZEROMV identity copy
    // therefore predicts the I-frame's pixels at the same position,
    // with the small residual (the drift) absorbed by the §14 quant.
    let (w, h) = (64u32, 64u32);
    let (y0, u0, v0) = structured_frame_64x64(0, 0);
    let (y1, u1, v1) = structured_frame_64x64(4, 2);
    let frame_i = I420Frame::packed(w, h, &y0, &u0, &v0);
    let frame_p = I420Frame::packed(w, h, &y1, &u1, &v1);

    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    // ---- Encode the I frame and capture its post-§15 reconstruction
    let (i_bytes, i_reconstruction) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I frame");
    assert!(!i_bytes.is_empty(), "I-frame bytes non-empty");

    // ---- Encode the P frame ZEROMV/LAST against the I-frame reference
    let (p_bytes, _p_reconstruction) =
        encode_p_frame_zero_mv(&frame_p, &i_reconstruction, &params).expect("encode P frame");
    assert!(!p_bytes.is_empty(), "P-frame bytes non-empty");

    // ---- Self-decode through the stateful driver.
    let mut state = Vp8DecoderState::new();
    let i_decoded = state.decode_frame(&i_bytes).expect("decode I frame");
    assert_eq!(i_decoded.width, w);
    assert_eq!(i_decoded.height, h);

    let p_decoded = state.decode_frame(&p_bytes).expect("decode P frame");
    assert_eq!(p_decoded.width, w);
    assert_eq!(p_decoded.height, h);

    let y_psnr = plane_psnr(&y1, &p_decoded.y);
    let u_psnr = plane_psnr(&u1, &p_decoded.u);
    let v_psnr = plane_psnr(&v1, &p_decoded.v);
    eprintln!("P-frame self-decode PSNR: Y={y_psnr:.2} dB, U={u_psnr:.2} dB, V={v_psnr:.2} dB");

    // The whole-frame combined PSNR (weighted by plane sample count).
    let total_sse: f64 = {
        let mut s = 0.0f64;
        for (a, b) in y1.iter().zip(p_decoded.y.iter()) {
            let d = *a as f64 - *b as f64;
            s += d * d;
        }
        for (a, b) in u1.iter().zip(p_decoded.u.iter()) {
            let d = *a as f64 - *b as f64;
            s += d * d;
        }
        for (a, b) in v1.iter().zip(p_decoded.v.iter()) {
            let d = *a as f64 - *b as f64;
            s += d * d;
        }
        s
    };
    let total_samples = (y1.len() + u1.len() + v1.len()) as f64;
    let whole_psnr = 10.0 * (255.0f64 * 255.0 / (total_sse / total_samples)).log10();
    eprintln!("P-frame whole-frame PSNR: {whole_psnr:.2} dB");

    assert!(
        whole_psnr >= 30.0,
        "P-frame self-decode whole-frame PSNR {whole_psnr:.2} dB < 30 dB at qi=32"
    );
}

/// Build a packed flat I420 frame of a single (Y, U, V) constant.
fn flat_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    (vec![y; w * h], vec![u; cw * ch], vec![v; cw * ch])
}

#[test]
fn p_frame_emits_inter_frame_tag() {
    // The §9.1 frame tag's bit 0 is `frame_type`. A key frame is 0; a
    // P frame is 1. This pins that bit and the absence of the
    // §9.1 start code on the encoder's output.
    let (w, h) = (32u32, 32u32);
    let (y, u, v) = flat_frame(w, h, 128, 128, 128);
    let frame = I420Frame::packed(w, h, &y, &u, &v);
    let params = KeyframeParams::default();

    let (_i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame, &params).expect("encode I");
    let (p_bytes, _) = encode_p_frame_zero_mv(&frame, &i_recon, &params).expect("encode P");

    // §9.1: bit 0 of byte 0 is frame_type. For an inter frame it must
    // be 1 (set).
    assert_eq!(
        p_bytes[0] & 0x01,
        0x01,
        "P-frame must carry frame_type = 1 in §9.1 tag"
    );
    // §9.1: the 3-byte start code 0x9d 0x01 0x2a appears at offset 3..6
    // for key frames only.
    if p_bytes.len() >= 6 {
        assert_ne!(
            &p_bytes[3..6],
            &[0x9d, 0x01, 0x2a],
            "P-frame must not carry the keyframe start code"
        );
    }
}

#[test]
fn p_frame_reference_dimensions_mismatch_rejected() {
    use oxideav_vp8::EncodeError;
    // Encode a small I frame and pass a differently-sized reference to
    // the P encoder.
    let (y_small, u_small, v_small) = flat_frame(16, 16, 128, 128, 128);
    let (y_big, u_big, v_big) = flat_frame(64, 64, 128, 128, 128);
    let frame_small = I420Frame::packed(16, 16, &y_small, &u_small, &v_small);
    let frame_big = I420Frame::packed(64, 64, &y_big, &u_big, &v_big);
    let params = KeyframeParams::default();
    let (_i_bytes, i_recon_big) =
        encode_keyframe_with_reconstruction(&frame_big, &params).expect("encode big I");
    let err = encode_p_frame_zero_mv(&frame_small, &i_recon_big, &params).unwrap_err();
    assert!(matches!(
        err,
        EncodeError::ReferenceDimensionsMismatch { .. }
    ));
}
