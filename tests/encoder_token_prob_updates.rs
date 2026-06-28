//! Encoder-side §13.4 `token_prob_update()` payload — exposes
//! caller-driven token-prob updates through the new
//! [`encode_keyframe_with_token_prob_updates`] entry-point.
//!
//! RFC 6386 §13.4 page 68 specifies the 4×8×3×11 = 1056-position
//! `coeff_prob_update_flag` / `coeff_prob` sub-block read against the
//! per-position `coeff_update_probs` table. The encoder writes this
//! sub-block on every frame (key or inter); through round 154 the wire
//! payload was always "every flag = 0" (defaults retained). This round
//! adds:
//!
//!   * [`oxideav_vp8::write_token_prob_updates`] — the per-position
//!     writer parallel to the parser in [`Vp8CodedHeader::parse`].
//!   * [`oxideav_vp8::encode_keyframe_with_token_prob_updates`] — a
//!     keyframe entry-point that threads a caller-supplied
//!     `TokenProbUpdates` array through both the picker's RD estimate
//!     and the §13.3 token-encode pass, then writes the §13.4 layer so
//!     the decoder rebuilds the same merged `coeff_probs` table.
//!
//! Each test below pins one corner of the round-trip:
//!
//!   1. An all-`None` updates array reproduces the round-154 wire
//!      byte-for-byte (the no-op path).
//!   2. A non-trivial updates array (a handful of `Some(prob)`
//!      positions) produces an *observably different* wire than the
//!      no-op path AND still round-trips through `decode_vp8` to a
//!      sound picture at the same PSNR floor as the no-op path.
//!   3. Round-tripping the §19.2 header on the new wire recovers the
//!      exact `TokenProbUpdates` array the encoder transmitted (per-
//!      position `Some(p)` / `None` shape preserved).
//!   4. The `write_token_prob_updates` free function emits the same
//!      bytes `write_no_token_prob_updates` does when handed an
//!      all-`None` array — proving the new writer extends the old
//!      writer's contract without breaking the trivial path.

use oxideav_vp8::{
    decode_vp8, encode_keyframe, encode_keyframe_with_token_prob_updates,
    write_no_token_prob_updates, write_token_prob_updates, BoolEncoder, I420Frame, KeyframeParams,
    TokenProbUpdates, Vp8CodedHeader, Vp8FrameHeader,
};

/// Build a small synthetic I420 picture exercising both the whole-block
/// and `B_PRED` intra paths.
fn synthetic_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..h {
        for c in 0..w {
            let g = ((r * 8 + c * 4) % 256) as u8;
            y[r * w + c] = g;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (120 + (c * 16 / cw)) as u8;
            v[r * cw + c] = (130 + (r * 16 / ch)) as u8;
        }
    }
    (y, u, v)
}

fn synthetic_params() -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    }
}

/// Whole-frame PSNR over a synthetic 32×32 source and a decoded frame.
fn frame_psnr(src_y: &[u8], src_u: &[u8], src_v: &[u8], dec: &oxideav_vp8::Vp8DecodedFrame) -> f64 {
    let total = src_y.len() + src_u.len() + src_v.len();
    let plane_se = |a: &[u8], b: &[u8]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum::<f64>()
    };
    let total_se = plane_se(src_y, &dec.y) + plane_se(src_u, &dec.u) + plane_se(src_v, &dec.v);
    let mse = total_se / total as f64;
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// All-`None` updates ⇒ the new entry-point emits the same bytes the
/// round-154 [`encode_keyframe`] does.
#[test]
fn empty_token_prob_updates_matches_round154_wire() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    let no_updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    let bytes_default = encode_keyframe(&frame, &params).expect("encode_keyframe (default)");
    let bytes_with_empty =
        encode_keyframe_with_token_prob_updates(&frame, &params, &no_updates).expect("encode");

    assert_eq!(
        bytes_default, bytes_with_empty,
        "all-None token_updates must reproduce the round-154 wire byte-for-byte"
    );
}

/// A non-trivial `Some(p)` payload changes the wire but still
/// round-trips through `decode_vp8` (the merged `coeff_probs` is what
/// the decoder rebuilds).
#[test]
fn nontrivial_token_prob_updates_changes_wire_and_round_trips() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    // A scattered set of replacement probabilities at "safe" positions
    // (away from the §13.5 entries set to 0 or 255 where a near-extreme
    // override would mis-price the rare branch). The values are
    // arbitrary; the round-trip is the property that matters.
    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    updates[0][0][0][0] = Some(200);
    updates[1][2][1][3] = Some(100);
    updates[2][5][2][0] = Some(180);
    updates[3][7][0][2] = Some(64);

    let bytes_default = encode_keyframe(&frame, &params).expect("encode_keyframe (default)");
    let bytes_updated = encode_keyframe_with_token_prob_updates(&frame, &params, &updates)
        .expect("encode_keyframe_with_token_prob_updates");

    // The two wires MUST differ — the §13.4 sub-block carries different
    // bits and the token-encode pass codes against a different merged
    // `coeff_probs`.
    assert_ne!(
        bytes_default, bytes_updated,
        "non-trivial updates must change the emitted wire"
    );

    // Both wires decode through the crate's own decoder.
    let dec_default = decode_vp8(&bytes_default).expect("decode_vp8 (default)");
    let dec_updated = decode_vp8(&bytes_updated).expect("decode_vp8 (updated)");

    assert_eq!(dec_default.width, 32);
    assert_eq!(dec_default.height, 32);
    assert_eq!(dec_updated.width, 32);
    assert_eq!(dec_updated.height, 32);

    // Decoded pictures must reach the same PSNR floor — the §13.4
    // updates affect only the entropy layer, not the quantised
    // coefficients or the §11 / §14 reconstruction.
    let psnr_default = frame_psnr(&y, &u, &v, &dec_default);
    let psnr_updated = frame_psnr(&y, &u, &v, &dec_updated);
    assert!(
        psnr_default >= 25.0,
        "default wire decoded PSNR < 25 dB: {psnr_default}"
    );
    assert!(
        psnr_updated >= 25.0,
        "updated wire decoded PSNR < 25 dB: {psnr_updated}"
    );

    // The picker is deterministic in the SSD it scores (the merged
    // coeff_probs only re-prices the token bits in the J = SSD + lambda
    // * R RD pick, so the chosen modes / coefficients can shift). The
    // decoded Y planes must therefore be *valid* but are not required
    // to be byte-identical.
    assert_eq!(dec_default.y.len(), dec_updated.y.len());
}

/// The §19.2 parser recovers the exact `TokenProbUpdates` array the
/// encoder transmitted: per-position `Some(prob)` survives and `None`
/// stays `None`.
#[test]
fn token_prob_updates_round_trip_through_header_parser() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    // Pick positions across each plane so any indexing bug stands out.
    updates[0][1][0][0] = Some(176);
    updates[0][1][0][1] = Some(246);
    updates[1][0][0][0] = Some(217);
    updates[2][3][2][5] = Some(50);
    updates[3][6][1][9] = Some(220);

    let bytes = encode_keyframe_with_token_prob_updates(&frame, &params, &updates)
        .expect("encode_keyframe_with_token_prob_updates");

    // Decode just the uncompressed + first-partition header.
    let raw_hdr = Vp8FrameHeader::parse(&bytes).expect("uncompressed header parse");
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

/// The free `write_token_prob_updates` writer emits the same bytes
/// `write_no_token_prob_updates` does when handed an all-`None`
/// payload — the new writer is a strict superset of the old contract.
#[test]
fn write_token_prob_updates_all_none_matches_no_update_writer() {
    // The §13.4 `coeff_update_probs[4][8][3][11]` flag-probability
    // table (transcribed from RFC 6386 §13.4 page 69). The crate-local
    // copy used by both writers is `pub(crate)`, so we transcribe the
    // same 1056-entry flat view here for the call. Any divergence
    // between the two writers on the all-None path would mean the new
    // writer doesn't preserve the round-154 wire byte-for-byte.
    //
    // Rather than re-typing the entire table, we exercise the
    // round-154 wire equivalence through the encoder entry-point
    // above; here we just verify the two writers walk the same number
    // of bits when handed the same flag table and an all-None array.
    let flag_probs = [128u8; 1056]; // arbitrary placeholder

    let no_updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];

    let mut a = BoolEncoder::new();
    write_no_token_prob_updates(&mut a, &flag_probs);
    let bytes_a = a.finish();

    let mut b = BoolEncoder::new();
    write_token_prob_updates(&mut b, &no_updates, &flag_probs);
    let bytes_b = b.finish();

    assert_eq!(
        bytes_a, bytes_b,
        "writer outputs must agree on the all-None payload"
    );
}
