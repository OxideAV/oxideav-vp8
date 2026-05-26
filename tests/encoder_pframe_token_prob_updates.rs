//! Encoder-side §13.4 `token_prob_update()` payload for the inter
//! (P-frame) path — exposes caller-driven token-prob updates through the
//! new
//! [`oxideav_vp8::encode_p_frame_multi_ref_with_token_updates`] /
//! [`oxideav_vp8::encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates`]
//! entry-points (and the matching
//! [`oxideav_vp8::Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_token_updates`]
//! stream method).
//!
//! RFC 6386 §13.4 page 68 specifies the 4×8×3×11 = 1056-position
//! `coeff_prob_update_flag` / `coeff_prob` sub-block read against the
//! per-position `coeff_update_probs` table — on EVERY frame (key or
//! inter). Through round 155 the inter encoder hardwired the sub-block
//! to 1056 zero flags; this round mirrors the r155 keyframe pattern on
//! the inter path.
//!
//! Each test below pins one corner of the round-trip:
//!
//!   1. All-`None` updates ⇒ the new entry-point emits the same bytes
//!      [`oxideav_vp8::encode_p_frame_multi_ref`] does (no-op
//!      equivalence; the round-155 inter wire is preserved).
//!   2. A non-trivial `Some(prob)` payload produces an observably
//!      different wire AND still self-decodes through
//!      [`oxideav_vp8::Vp8DecoderState`] to a sound picture.
//!   3. Round-tripping the §19.2 header recovers the exact
//!      `TokenProbUpdates` array the encoder transmitted.
//!   4. The [`Vp8InterStreamEncoder`] stream method preserves the
//!      no-op equivalence when handed `token_updates = None`, and an
//!      end-to-end I + P stream with a non-trivial inter-frame update
//!      decodes through the in-tree decoder state machine.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates,
    encode_p_frame_multi_ref_with_token_updates, I420Frame, KeyframeParams, LoopFilterDeltas,
    RefreshControls, TokenProbUpdates, Vp8CodedHeader, Vp8DecoderState, Vp8FrameHeader,
    Vp8InterStreamEncoder,
};

/// Build a 32×32 I420 picture with a smooth gradient — repeatable,
/// deterministic, no PRNG. The luma / chroma offsets exercise the
/// inter-frame residual path.
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

fn params() -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    }
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

fn frame_psnr(src_y: &[u8], src_u: &[u8], src_v: &[u8], dec: &oxideav_vp8::Vp8DecodedFrame) -> f64 {
    let py = plane_psnr(src_y, &dec.y);
    let pu = plane_psnr(src_u, &dec.u);
    let pv = plane_psnr(src_v, &dec.v);
    // Geometric mean of finite PSNRs (saturating ∞ to 99 dB for the
    // floor check). This is enough for the sound-picture assertion;
    // exact PSNR vs. baseline is NOT asserted because the merged
    // entropy table doesn't change reconstruction.
    let clamp = |p: f64| if p.is_finite() { p } else { 99.0 };
    (clamp(py) + clamp(pu) + clamp(pv)) / 3.0
}

/// All-`None` updates ⇒ the new inter entry-point emits the same bytes
/// [`encode_p_frame_multi_ref`] does (the round-155 inter wire).
#[test]
fn empty_token_prob_updates_matches_round155_inter_wire() {
    let (yi, ui, vi) = structured_frame_32x32(0, 0);
    let (yp, up, vp) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(32, 32, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(32, 32, &yp, &up, &vp);
    let kp = params();

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");

    let no_updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    let (bytes_default, _) =
        encode_p_frame_multi_ref(&frame_p, &i_recon, None, None, &kp).expect("encode P (default)");
    let (bytes_with_empty, _) =
        encode_p_frame_multi_ref_with_token_updates(&frame_p, &i_recon, None, None, &kp, None)
            .expect("encode P (token_updates = None)");
    let (bytes_with_empty_arr, _) = encode_p_frame_multi_ref_with_token_updates(
        &frame_p,
        &i_recon,
        None,
        None,
        &kp,
        Some(&no_updates),
    )
    .expect("encode P (token_updates = all-None)");

    assert_eq!(
        bytes_default, bytes_with_empty,
        "token_updates = None must reproduce the round-155 inter wire byte-for-byte"
    );
    assert_eq!(
        bytes_default, bytes_with_empty_arr,
        "token_updates = Some(all-None) must reproduce the round-155 inter wire byte-for-byte"
    );
}

/// A non-trivial `Some(p)` payload changes the inter wire but still
/// self-decodes through `Vp8DecoderState` (the decoder overlays the
/// same updates on its carried base, which came from the key frame
/// with `refresh_entropy_probs = 1` — i.e. the §13.5 defaults — so
/// the merged table the encoder used is exactly what the decoder
/// rebuilds for this frame).
#[test]
fn nontrivial_token_prob_updates_changes_wire_and_self_decodes() {
    let (yi, ui, vi) = structured_frame_32x32(0, 0);
    let (yp, up, vp) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(32, 32, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(32, 32, &yp, &up, &vp);
    let kp = params();

    // A scattered set of replacement probabilities at "safe" positions
    // (away from §13.5 entries set to 0 or 255 where an extreme override
    // would mis-price the rare branch). Arbitrary values; the round-trip
    // and wire-difference are the properties under test.
    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    updates[0][0][0][0] = Some(200);
    updates[1][2][1][3] = Some(100);
    updates[2][5][2][0] = Some(180);
    updates[3][7][0][2] = Some(64);

    let (i_bytes, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");
    let (p_bytes_default, _) =
        encode_p_frame_multi_ref(&frame_p, &i_recon, None, None, &kp).expect("encode P (default)");
    let (p_bytes_updated, _) = encode_p_frame_multi_ref_with_token_updates(
        &frame_p,
        &i_recon,
        None,
        None,
        &kp,
        Some(&updates),
    )
    .expect("encode P (token_updates)");

    // Wires MUST differ — the §13.4 sub-block carries different bits
    // and the token-encode pass codes against a different merged
    // coeff_probs.
    assert_ne!(
        p_bytes_default, p_bytes_updated,
        "non-trivial inter token updates must change the emitted wire"
    );

    // Both inter wires decode through the in-tree decoder when fed in
    // after the same I-frame.
    let mut dec_default = Vp8DecoderState::new();
    let _ = dec_default.decode_frame(&i_bytes).expect("I-frame decode");
    let pframe_default = dec_default
        .decode_frame(&p_bytes_default)
        .expect("default P-frame decode");

    let mut dec_updated = Vp8DecoderState::new();
    let _ = dec_updated.decode_frame(&i_bytes).expect("I-frame decode");
    let pframe_updated = dec_updated
        .decode_frame(&p_bytes_updated)
        .expect("updated P-frame decode");

    assert_eq!(pframe_default.width, 32);
    assert_eq!(pframe_default.height, 32);
    assert_eq!(pframe_updated.width, 32);
    assert_eq!(pframe_updated.height, 32);

    // The merged coeff_probs governs entropy coding, not reconstruction —
    // the decoded picture must clear the round's 25 dB sound-picture bar
    // on either path. (Exact byte-equality is NOT asserted: the §16.2
    // ref-frame-tree bits use a Laplace-of-counts fit to the picker's
    // distribution and the picker is deterministic in its SAD score, so
    // the decoded planes are byte-identical only by accident.)
    let psnr_default = frame_psnr(&yp, &up, &vp, &pframe_default);
    let psnr_updated = frame_psnr(&yp, &up, &vp, &pframe_updated);
    assert!(
        psnr_default >= 25.0,
        "default inter PSNR < 25 dB: {psnr_default}"
    );
    assert!(
        psnr_updated >= 25.0,
        "updated inter PSNR < 25 dB: {psnr_updated}"
    );
}

/// The §19.2 inter-frame parser recovers the exact `TokenProbUpdates`
/// array the encoder transmitted: per-position `Some(prob)` survives
/// and `None` stays `None`.
#[test]
fn inter_token_prob_updates_round_trip_through_header_parser() {
    let (yi, ui, vi) = structured_frame_32x32(0, 0);
    let (yp, up, vp) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(32, 32, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(32, 32, &yp, &up, &vp);
    let kp = params();

    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    // Pick positions across each plane so any indexing bug stands out.
    updates[0][1][0][0] = Some(176);
    updates[0][1][0][1] = Some(246);
    updates[1][0][0][0] = Some(217);
    updates[2][3][2][5] = Some(50);
    updates[3][6][1][9] = Some(220);

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");
    let (bytes, _) = encode_p_frame_multi_ref_with_token_updates(
        &frame_p,
        &i_recon,
        None,
        None,
        &kp,
        Some(&updates),
    )
    .expect("encode P");

    // Decode just the uncompressed + first-partition header.
    let raw_hdr = Vp8FrameHeader::parse(&bytes).expect("uncompressed header parse");
    assert!(
        !raw_hdr.key_frame,
        "this fixture must be an inter frame for the test to be meaningful"
    );
    let first_partition_size = raw_hdr.first_partition_size as usize;
    let start = raw_hdr.header_bytes_consumed;
    let partition = &bytes[start..start + first_partition_size];
    let coded = Vp8CodedHeader::parse(partition, raw_hdr.key_frame).expect("coded header parse");

    for (i, plane) in updates.iter().enumerate() {
        for (j, band) in plane.iter().enumerate() {
            for (k, ctx) in band.iter().enumerate() {
                for (t, slot) in ctx.iter().enumerate() {
                    assert_eq!(
                        coded.token_prob_updates[i][j][k][t], *slot,
                        "round-trip mismatch at [{i}][{j}][{k}][{t}]"
                    );
                }
            }
        }
    }
}

/// `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates`
/// with `token_updates = None` reproduces the round-151 wire byte-for-
/// byte when the refresh + lf-delta config also matches their round-151
/// defaults — i.e. the new entry-point is a strict superset of the
/// round-151 contract.
#[test]
fn full_inter_entry_point_token_none_matches_round151_wire() {
    let (yi, ui, vi) = structured_frame_32x32(0, 0);
    let (yp, up, vp) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(32, 32, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(32, 32, &yp, &up, &vp);
    let kp = params();

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");

    let (bytes_round151, _) =
        encode_p_frame_multi_ref(&frame_p, &i_recon, None, None, &kp).expect("round-151 P");
    let (bytes_full_none, _) =
        encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates(
            &frame_p,
            &i_recon,
            None,
            None,
            &kp,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
            [0; 4],
            [0; 4],
            None,
        )
        .expect("full entry-point, token_updates = None");

    assert_eq!(
        bytes_round151, bytes_full_none,
        "full inter entry-point with `token_updates = None` must reproduce the round-151 wire byte-for-byte"
    );
}

/// `Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_token_updates`
/// drives the stream encoder through the new payload end-to-end:
///
///   * `token_updates = None` on the stream method must match
///     `encode_p_frame_with_refresh_and_lf_deltas` byte-for-byte.
///   * A non-trivial `Some(u)` payload changes the wire AND still
///     self-decodes through `Vp8DecoderState`.
#[test]
fn inter_stream_token_updates_no_op_equivalence_and_round_trip() {
    let (yi, ui, vi) = structured_frame_32x32(0, 0);
    let (yp, up, vp) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(32, 32, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(32, 32, &yp, &up, &vp);
    let kp = params();

    // ---- No-op equivalence on the stream method ----------------------
    let mut enc_a = Vp8InterStreamEncoder::new(kp, 100).expect("non-zero interval");
    let _ki_a = enc_a.encode_frame(&frame_i).expect("K-frame");
    let p_a = enc_a
        .encode_p_frame_with_refresh_and_lf_deltas(
            &frame_p,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
        )
        .expect("P-frame (round-151 stream method)");

    let mut enc_b = Vp8InterStreamEncoder::new(kp, 100).expect("non-zero interval");
    let _ki_b = enc_b.encode_frame(&frame_i).expect("K-frame");
    let p_b = enc_b
        .encode_p_frame_with_refresh_and_lf_deltas_and_token_updates(
            &frame_p,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
            None,
        )
        .expect("P-frame (new stream method, token_updates = None)");

    assert_eq!(
        p_a.bytes, p_b.bytes,
        "stream method with `token_updates = None` must match the round-151 stream wire byte-for-byte"
    );

    // ---- End-to-end stream with a non-trivial inter update -----------
    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    updates[0][0][0][0] = Some(200);
    updates[1][2][1][3] = Some(100);
    updates[2][5][2][0] = Some(180);
    updates[3][7][0][2] = Some(64);

    let mut enc_c = Vp8InterStreamEncoder::new(kp, 100).expect("non-zero interval");
    let ki_c = enc_c.encode_frame(&frame_i).expect("K-frame");
    let p_c = enc_c
        .encode_p_frame_with_refresh_and_lf_deltas_and_token_updates(
            &frame_p,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
            Some(&updates),
        )
        .expect("P-frame (new stream method, token_updates = Some)");

    assert_ne!(
        p_b.bytes, p_c.bytes,
        "non-trivial inter token updates must change the stream wire"
    );

    let mut dec = Vp8DecoderState::new();
    let dec_i = dec.decode_frame(&ki_c.bytes).expect("decode K-frame");
    let dec_p = dec.decode_frame(&p_c.bytes).expect("decode P-frame");
    assert_eq!(dec_i.width, 32);
    assert_eq!(dec_i.height, 32);
    assert_eq!(dec_p.width, 32);
    assert_eq!(dec_p.height, 32);

    let psnr = frame_psnr(&yp, &up, &vp, &dec_p);
    assert!(
        psnr >= 25.0,
        "inter token-update stream PSNR < 25 dB: {psnr}"
    );

    // The §13.4 layer is THIS-frame-only (we emit
    // `refresh_entropy_probs = 0` on inter), so a follow-up P-frame
    // with `token_updates = None` must self-decode against the
    // SAME carried base the round-151 stream method uses — i.e. the
    // §13.5 defaults restored after the previous P-frame. Pin that
    // here by encoding one more P-frame and checking the decoder still
    // accepts it.
    let p_d = enc_c
        .encode_p_frame_with_refresh_and_lf_deltas(
            &frame_p,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
        )
        .expect("follow-up P-frame (round-151 method)");
    let dec_d = dec.decode_frame(&p_d.bytes).expect("decode follow-up P");
    assert_eq!(dec_d.width, 32);
    let psnr_d = frame_psnr(&yp, &up, &vp, &dec_d);
    assert!(psnr_d >= 25.0, "follow-up P PSNR < 25 dB: {psnr_d}");
}
