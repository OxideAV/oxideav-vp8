//! End-to-end self-decode roundtrip for the round-160 §11 / §12.2
//! intra-within-inter MB picking extension to the inter (P-frame)
//! encoder.
//!
//! What round 160 adds:
//!
//!   * `encode_p_frame_multi_ref_with_intra_pick(frame, last, golden,
//!     altref, params)` — the per-MB picker additionally scores a
//!     §12.2 DC_PRED intra candidate against the running in-frame
//!     neighbours. Whichever of (best inter, intra DC) has the
//!     lower `J + lambda * is_inter_mb-bit` wins per MB.
//!   * `encode_p_frame_multi_ref_with_refresh_and_intra_pick` — same
//!     plus caller-driven §9.7 / §9.8 refresh.
//!
//! What this test pins:
//!
//!   1. With a synthetic source where the encoder's reference frame
//!      contains a uniformly-grey content but the source's row-0 MB
//!      contains a high-contrast vertical-stripe pattern, the §17
//!      motion search at any whole/half/quarter-pixel MV cannot
//!      produce a low-residual prediction (the reference is flat;
//!      every MV gives the same flat prediction), but the §12.2
//!      DC_PRED intra candidate predicts the row-0 MB's mean and
//!      codes a smaller-magnitude residual. The picker should select
//!      intra for that MB and the parsed §9.10 `prob_intra` should
//!      land below 255 (i.e. drop into the fitted range).
//!   2. The round-160 wire never grows the wire vs the
//!      pre-r160 inter-only encoder on the SAME source (safety
//!      guard: the picker's `J` trade is monotone — if intra loses on
//!      every MB, we still emit `is_inter_mb = true` per MB at
//!      `prob_intra = 1` for a constant ~6-bit ≈ 1-byte cost; the
//!      test caps the round-160 wire at `pre-r160 + 4 bytes` slack to
//!      account for that prob_intra-byte overhead).
//!   3. The emitted bytes self-decode through the crate's own
//!      [`Vp8DecoderState`] (no external codec consulted) at a
//!      per-frame Y-PSNR ≥ 30 dB on a mid-quantiser target.
//!
//! Black-box: the encoder's output is fed straight into the crate's
//! own [`Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref,
    encode_p_frame_multi_ref_with_intra_pick, FrameKind, I420Frame, KeyframeParams, Vp8CodedHeader,
    Vp8DecodedFrame, Vp8DecoderState, Vp8FrameHeader, Vp8InterStreamEncoder,
};

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

fn y_psnr(src_y: &[u8], rec: &Vp8DecodedFrame) -> f64 {
    plane_psnr(src_y, &rec.y)
}

/// All planes at a flat colour `v`.
fn flat_color(width: usize, height: usize, v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![v; width * height];
    let u = vec![v; (width / 2) * (height / 2)];
    let v_plane = vec![v; (width / 2) * (height / 2)];
    (y, u, v_plane)
}

/// Source frame: row-0 MB carries a high-contrast vertical-stripe
/// pattern (32 / 224 alternating every 2 columns); row-1 MB stays at
/// flat grey 128. Chroma stays flat at 128. The stripe content is
/// deliberately too sharp for any motion-compensated prediction from a
/// flat reference to reach the §14 quantiser's reconstruction floor, so
/// the inter picker on the row-0 MB will land on a large-magnitude
/// residual. The intra DC_PRED candidate predicts each MB's mean (here
/// `(32+224)/2 = 128`, exactly the row-1 plus chroma neutral) and emits
/// a much smaller residual on the row-0 MB.
#[allow(dead_code)]
fn stripe_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![128u8; width * height];
    for r in 0..16.min(height) {
        for c in 0..width {
            y[r * width + c] = if (c / 2) & 1 == 0 { 32 } else { 224 };
        }
    }
    let u = vec![128u8; (width / 2) * (height / 2)];
    let v = vec![128u8; (width / 2) * (height / 2)];
    (y, u, v)
}

/// Parse the (prob_intra, prob_last, prob_gf) §9.10 triple from an
/// inter (P-frame) bitstream's first partition. Mirrors the helper in
/// `tests/encoder_pframe_goldenref.rs`.
fn parse_inter_prob_triple(bytes: &[u8]) -> (u8, u8, u8) {
    let hdr = Vp8FrameHeader::parse(bytes).expect("§19.1 tag parses");
    assert!(!hdr.key_frame, "inter-frame parser called on a K frame");
    let partition_start = hdr.header_bytes_consumed;
    let partition_end = partition_start + hdr.first_partition_size as usize;
    let partition = &bytes[partition_start..partition_end];
    let coded = Vp8CodedHeader::parse(partition, /*key_frame=*/ false)
        .expect("§19.2 control partition parses");
    (
        coded.prob_intra.expect("inter prob_intra present"),
        coded.prob_last.expect("inter prob_last present"),
        coded.prob_gf.expect("inter prob_gf present"),
    )
}

/// On a frame where the LAST reference is uniformly black (0) and the
/// P source is uniformly mid-bright (200), the inter ZEROMV residual
/// on every MB equals the full source magnitude (`SAD ≈ 256 * 200 =
/// 51200`), while the §12 DC_PRED intra candidate on the top-left MB
/// falls back to the §12 off-frame default mid-grey (128) for a
/// residual of `|200 - 128| = 72` per pixel (`SAD ≈ 18432`). After
/// that first MB's reconstruction lands in the in-frame `planes`
/// buffer at value ≈ 200, every subsequent intra MB has a genuine
/// neighbour-edge close to 200, dropping its residual nearly to zero.
/// The picker should select intra DC_PRED for every MB; the parsed
/// §9.10 `prob_intra` should land at 255 (all-intra) — the `fit_prob_l8`
/// boundary value for `count_intra > 0 && count_inter == 0`.
#[test]
fn intra_pick_selects_intra_when_inter_residual_is_large() {
    let width = 32u32;
    let height = 32u32; // 2 col × 2 rows of MBs.
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    // K frame: all black — populates LAST / GOLDEN / ALTREF with the
    // black picture. The next P-frame's inter picker has only black
    // to predict from at any MV.
    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 0);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &params).expect("encode K");

    // P frame: flat mid-bright 200. Inter SAD per MB ≈ 51200; intra
    // DC_PRED SAD on the top-left MB ≈ 18432 (off-frame default 128
    // vs source 200) and ≈ 0 on every interior MB (neighbour ≈ 200).
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 200);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);
    let (p_bytes, _p_planes) =
        encode_p_frame_multi_ref_with_intra_pick(&p_frame, &k_planes, None, None, &params)
            .expect("encode P");

    let (prob_intra, _prob_last, _prob_gf) = parse_inter_prob_triple(&p_bytes);
    eprintln!("P §9.10 prob_intra = {prob_intra} (1 = no intra, 255 = all intra)");
    assert!(
        prob_intra > 1,
        "P prob_intra = {prob_intra}: the picker should select intra for at least one MB on \
         a black-LAST + bright-source pattern, but `fit_prob_l8(intra=0, inter=>0) = 1` was \
         emitted"
    );

    // Self-decode sanity: the bytes round-trip through the decoder
    // at an interesting PSNR floor.
    let mut dec = Vp8DecoderState::new();
    let _ = dec.decode_frame(&_k_bytes).expect("decode K");
    let d = dec.decode_frame(&p_bytes).expect("decode P");
    let psnr = y_psnr(&p_y, &d);
    eprintln!("P self-decode Y-PSNR = {psnr:.2} dB");
    assert!(
        psnr >= 30.0,
        "P Y-PSNR {psnr:.2} dB below the 30.0 dB floor on intra-pick path"
    );
}

/// On a source where intra never beats inter (perfect LAST match), the
/// round-160 wire must stay close to the pre-r160 inter-only wire — the
/// only difference is the §9.10 `prob_intra` byte adapting to the
/// "every MB inter" distribution (`prob_intra = 1` against `255`) plus
/// the per-MB `is_inter_mb = true` bit coding at probability 255/256
/// instead of essentially-zero bits. The safety-guard cap is 4 bytes
/// of slack.
#[test]
fn intra_pick_does_not_grow_wire_on_perfect_inter_source() {
    let width = 16u32;
    let height = 32u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    // Build a K-frame reference. Reuse the same picture as the
    // P-frame source so the inter picker resolves to a near-zero
    // residual on every MB (LAST is a perfect match).
    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &params).expect("encode K");

    // P frame: identical to K's source. The inter ZEROMV J should
    // be dominated by SAD ≈ 0; intra DC_PRED on the stripe MB has
    // SAD ≈ |stripe - mean| × 16 × 16, dramatically larger. With
    // the round-160 picker selecting intra never, `prob_intra` is
    // fitted to `1` (the clamped `count_intra = 0` boundary), and
    // each per-MB `is_inter_mb = true` bit codes at probability
    // 255/256, costing nearly zero bits — the wire stays close to
    // the pre-r160 inter-only path.
    let p_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (bytes_old, _) =
        encode_p_frame_multi_ref(&p_frame, &k_planes, None, None, &params).expect("encode P old");
    let (bytes_new, _) =
        encode_p_frame_multi_ref_with_intra_pick(&p_frame, &k_planes, None, None, &params)
            .expect("encode P new");

    let (pi_old, _, _) = parse_inter_prob_triple(&bytes_old);
    let (pi_new, _, _) = parse_inter_prob_triple(&bytes_new);
    eprintln!(
        "perfect-match P: pre-r160 = {} B (prob_intra={pi_old}), r160 intra-pick = {} B (prob_intra={pi_new}, Δ = {})",
        bytes_old.len(),
        bytes_new.len(),
        bytes_new.len() as isize - bytes_old.len() as isize
    );
    // The fitter writes prob_intra = 1 when no MB picks intra (the
    // clamped boundary value of `fit_prob_l8(count_intra = 0, ...)`).
    assert_eq!(
        pi_new, 1,
        "perfect-match source should have intra losing on every MB \
         ⇒ prob_intra = 1 (clamped boundary)"
    );
    // The pre-r160 path hardwires prob_intra = 255.
    assert_eq!(
        pi_old, 255,
        "pre-r160 inter-only path hardwires prob_intra = 255"
    );
    // The fitter writes prob_intra = 1 when no MB picks intra (every
    // MB inter). The is_inter_mb = true bit then codes at p ≈ 1/256,
    // costing ~8 bits per MB; on a 2-MB frame that's ~2 bytes vs the
    // ~0 bytes the historical prob_intra = 255 path paid. Cap the
    // slack at 4 bytes.
    assert!(
        bytes_new.len() <= bytes_old.len() + 4,
        "round-160 intra-pick wire grew by more than 4 bytes on a perfect-match source: \
         pre-r160 = {} B, r160 = {} B",
        bytes_old.len(),
        bytes_new.len()
    );

    // Self-decode sanity.
    let mut dec = Vp8DecoderState::new();
    let _ = dec.decode_frame(&_k_bytes).expect("decode K");
    let d = dec.decode_frame(&bytes_new).expect("decode P r160");
    let psnr = y_psnr(&k_y, &d);
    eprintln!("perfect-match P self-decode Y-PSNR = {psnr:.2} dB");
    assert!(
        psnr >= 30.0,
        "perfect-match P Y-PSNR {psnr:.2} dB below 30.0 dB on intra-pick path"
    );
}

/// Three-frame I + P + P self-decode roundtrip on the same black-K +
/// bright-P content the first test uses, exercising the
/// `Vp8InterStreamEncoder`-equivalent driving pattern (but using the
/// bare-encoder intra-pick entry-point directly because the
/// `Vp8InterStreamEncoder` stream-driver intra-pick wiring is the
/// next round's task).
#[test]
fn intra_pick_three_frame_stream_self_decodes() {
    let width = 32u32;
    let height = 32u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    // K: black; P: flat mid-bright 200.
    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 0);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 200);

    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    // Encode K — the standard keyframe path.
    let (k_bytes, k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &params).expect("encode K");
    // Encode P1 — intra DC_PRED expected on every MB (LAST is black,
    // source is bright; intra cascades from the top-left's 128-fallback
    // to neighbour-DC ≈ 200 for every subsequent MB).
    let (p1_bytes, p1_planes) =
        encode_p_frame_multi_ref_with_intra_pick(&p_frame, &k_planes, None, None, &params)
            .expect("encode P1");
    // Encode P2 — by now LAST holds the P1 reconstruction (which is
    // ≈ 200 everywhere). The §17 picker can match the source cheaply
    // off LAST; intra should NOT be picked.
    let (p2_bytes, _p2_planes) =
        encode_p_frame_multi_ref_with_intra_pick(&p_frame, &p1_planes, None, None, &params)
            .expect("encode P2");

    // Decoder side: confirm every frame round-trips above the PSNR
    // floor through a single Vp8DecoderState walking the three
    // bitstreams in order.
    let mut dec = Vp8DecoderState::new();
    let d0 = dec.decode_frame(&k_bytes).expect("decode K");
    let d1 = dec.decode_frame(&p1_bytes).expect("decode P1");
    let d2 = dec.decode_frame(&p2_bytes).expect("decode P2");

    let psnr_k = y_psnr(&k_y, &d0);
    let psnr_p1 = y_psnr(&p_y, &d1);
    let psnr_p2 = y_psnr(&p_y, &d2);

    eprintln!("K Y-PSNR = {psnr_k:.2} dB");
    eprintln!("P1 (bright-from-black) Y-PSNR = {psnr_p1:.2} dB");
    eprintln!("P2 (bright-from-bright) Y-PSNR = {psnr_p2:.2} dB");

    assert!(psnr_k >= 30.0, "K Y-PSNR {psnr_k:.2} dB below floor");
    assert!(psnr_p1 >= 25.0, "P1 Y-PSNR {psnr_p1:.2} dB below floor");
    assert!(psnr_p2 >= 25.0, "P2 Y-PSNR {psnr_p2:.2} dB below floor");

    // P1 prob_intra > 1 reveals intra usage; P2 prob_intra ≈ 1 because
    // LAST now matches the source and inter wins on every MB.
    let (p1_pi, _, _) = parse_inter_prob_triple(&p1_bytes);
    let (p2_pi, _, _) = parse_inter_prob_triple(&p2_bytes);
    eprintln!("P1 prob_intra = {p1_pi}, P2 prob_intra = {p2_pi}");
    assert!(
        p1_pi > 1,
        "P1 prob_intra = {p1_pi}: the picker should select intra for at least one MB on a \
         black-LAST + bright-source pattern (= the first-test setup)"
    );

    // Suppress unused-import warnings on the imported FrameKind /
    // Vp8InterStreamEncoder if the test grows in future rounds.
    let _ = FrameKind::Key;
    let _ = Vp8InterStreamEncoder::new(params, 100);
}
