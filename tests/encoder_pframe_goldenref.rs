//! End-to-end I + P + P self-decode roundtrip for the round-148
//! §16.2 `ref_frame_tree` GOLDEN / ALTREF reference-frame selector
//! extension ([`oxideav_vp8::encode_p_frame_multi_ref`]'s J = SAD +
//! λ·(mv_ref_tree_bits + ref_frame_tree_bits) picker now scores
//! LAST / GOLDEN / ALTREF per MB and emits whichever wins).
//!
//! Encodes a 3-frame I+P+P sequence where the picture content
//! deliberately whipsaws so that the LAST reference for the 3rd frame
//! is a poor predictor for some MBs but the GOLDEN reference (frozen
//! at the §9.7 ladder = the most-recent keyframe's reconstruction)
//! is a good one:
//!
//!   * Frame 0 (K): a high-contrast vertical-stripe pattern in the
//!     row-0 MB; the row-1 MB is flat gray. The keyframe
//!     reconstruction populates LAST = GOLDEN = ALTREF with that
//!     picture.
//!   * Frame 1 (P): the row-0 MB flips to flat gray (a low-detail
//!     content). The P-frame's reconstruction drifts row-0's LAST
//!     slot to flat gray; GOLDEN / ALTREF stay at the original
//!     stripe pattern.
//!   * Frame 2 (P): the row-0 MB returns to the original stripe
//!     pattern. For that MB, GOLDEN (or ALTREF, the same content)
//!     is the closer match than LAST (which holds the flat gray
//!     from P1).
//!
//! What this pins:
//!
//!   * the §16.2 `ref_frame_tree` non-LAST path emission on a real
//!     encoder bitstream (`B(prob_last)` reads `true` for at least
//!     one MB);
//!   * the §9.10 `prob_last` / `prob_gf` distribution fit (the wire
//!     probabilities adapt to the picker's observed per-MB
//!     reference counts so the §16.2 selector bits don't blow up on
//!     a high non-LAST ratio);
//!   * the `Vp8InterStreamEncoder` reference-slot threading: GOLDEN
//!     / ALTREF slots are passed to `encode_p_frame_multi_ref`
//!     unmodified across the §9.7 inter refresh ladder.
//!   * the §18 prediction reading from the chosen ref's planes: the
//!     P2 self-decode Y-PSNR has to clear a non-trivial floor (well
//!     above what a LAST-only encoder would manage on this content
//!     at this quantiser).
//!
//! Black-box: the encoder's output is fed straight into the crate's
//! own [`oxideav_vp8::Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref, FrameKind, I420Frame,
    KeyframeParams, Vp8CodedHeader, Vp8DecodedFrame, Vp8DecoderState, Vp8FrameHeader,
    Vp8InterStreamEncoder,
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

/// Render a "stripe" pattern: row-0 MB carries a high-contrast 2-pixel
/// vertical-stripe luma signal that the §14 quantiser would crater at
/// low bitrate; row-1 MB stays at flat gray 128. Chroma stays flat at
/// 128.
fn stripe_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![128u8; width * height];
    for r in 0..16.min(height) {
        for c in 0..width {
            // Two-pixel period vertical stripes: 32 / 224 alternation.
            y[r * width + c] = if (c / 2) & 1 == 0 { 32 } else { 224 };
        }
    }
    let cw = width / 2;
    let ch = height / 2;
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    (y, u, v)
}

/// Render a flat-gray pattern: every pixel = 128. Used as the P1
/// source to drift the LAST reference away from the K reconstruction.
fn flat_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![128u8; width * height];
    let cw = width / 2;
    let ch = height / 2;
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    (y, u, v)
}

/// Extract the §9.10 `prob_intra` / `prob_last` / `prob_gf` triple
/// from a P-frame's emitted bytes by re-parsing the §19.1 uncompressed
/// frame tag + §19.2 control partition.
///
/// The uncompressed tag is the first 3 bytes; the control partition
/// is the next `first_partition_size` bytes, parsed via
/// [`Vp8CodedHeader::parse(_, key_frame = false)`].
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

#[test]
fn p_frame_picker_selects_golden_when_last_has_drifted() {
    let width = 16u32;
    let height = 32u32; // 1 col, 2 rows of MBs.
    let qi = 16u8;

    let mut enc = Vp8InterStreamEncoder::new(
        KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            sharpness_level: 0,
            nbr_of_dct_partitions: 1,
            filter_type: false,
            trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
        },
        100, // big interval — K is only frame 0.
    )
    .expect("non-zero interval");

    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let (p1_y, p1_u, p1_v) = flat_frame(width as usize, height as usize);
    let (p2_y, p2_u, p2_v) = stripe_frame(width as usize, height as usize);

    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p1_frame = I420Frame::packed(width, height, &p1_y, &p1_u, &p1_v);
    let p2_frame = I420Frame::packed(width, height, &p2_y, &p2_u, &p2_v);

    let e0 = enc.encode_frame(&k_frame).expect("encode K");
    let e1 = enc.encode_frame(&p1_frame).expect("encode P1");
    let e2 = enc.encode_frame(&p2_frame).expect("encode P2");

    assert_eq!(e0.kind, FrameKind::Key, "frame 0 must be K");
    assert_eq!(e1.kind, FrameKind::InterZeroMv, "frame 1 must be P");
    assert_eq!(e2.kind, FrameKind::InterZeroMv, "frame 2 must be P");

    // Decode through the production decoder driver and verify the P2
    // frame's Y-plane PSNR clears 30 dB. The stripe content on row 0
    // is sharp enough that quantiser-only recovery from a flat-gray
    // LAST (P1) would crater PSNR; selecting GOLDEN (which holds the
    // original stripe pattern) recovers it.
    let mut dec = Vp8DecoderState::new();
    let _d0 = dec.decode_frame(&e0.bytes).expect("decode K");
    let _d1 = dec.decode_frame(&e1.bytes).expect("decode P1");
    let d2 = dec.decode_frame(&e2.bytes).expect("decode P2");

    let p2_psnr_y = y_psnr(&p2_y, &d2);
    eprintln!("P2 self-decode Y-PSNR = {p2_psnr_y:.2} dB (stripe-back-from-flat)");
    assert!(
        p2_psnr_y >= 30.0,
        "P2 Y-PSNR {p2_psnr_y:.2} dB below the 30.0 dB target",
    );

    // The §9.10 prob_last byte the encoder emitted is the strongest
    // proxy for "did the picker select a non-LAST ref for any MB?".
    // `fit_prob_l8` sets prob_last = floor(256 * count_last / total)
    // clamped to 1..=255, so prob_last < 255 iff at least one MB
    // selected GOLDEN or ALTREF over LAST.
    let (prob_intra, prob_last, prob_gf) = parse_inter_prob_triple(&e2.bytes);
    eprintln!("P2 §9.10 prob_intra={prob_intra} prob_last={prob_last} prob_gf={prob_gf}");
    assert_eq!(
        prob_intra, 255,
        "every MB this round is still coded as inter (prob_intra = 255)"
    );
    assert!(
        prob_last < 255,
        "P2's prob_last = {prob_last} (== 255 means the picker selected \
         LAST for every MB); round-148 should have selected GOLDEN / \
         ALTREF for at least one MB"
    );
}

/// Regression-free guard: when no GOLDEN / ALTREF is provided to
/// `encode_p_frame_multi_ref`, the wire `prob_last` must land at 255
/// (every MB picks LAST) and the encoder's behaviour collapses to the
/// single-ref `encode_p_frame_zero_mv` path. Protects against an
/// accidental change in the distribution-fit logic that would over-
/// spend bits when no non-LAST refs are available.
#[test]
fn multi_ref_with_no_golden_altref_collapses_to_last_only() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) = encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let (p_bytes, _p_planes) =
        encode_p_frame_multi_ref(&p_frame, &k_planes, None, None, &params).expect("P");

    let (prob_intra, prob_last, _prob_gf) = parse_inter_prob_triple(&p_bytes);
    assert_eq!(prob_intra, 255);
    assert_eq!(
        prob_last, 255,
        "no GOLDEN / ALTREF passed ⇒ every MB picks LAST ⇒ prob_last = 255"
    );

    // Self-decode sanity.
    let mut dec = Vp8DecoderState::new();
    let _ = dec.decode_frame(&_k_bytes).expect("decode K");
    let _ = dec.decode_frame(&p_bytes).expect("decode P");
}

/// When GOLDEN / ALTREF are passed but the picker still prefers LAST
/// for every MB (because LAST is a perfect match), the wire
/// `prob_last` must still land at 255. This pins that the picker's
/// `J + lambda * ref_frame_bits` trade prefers the lower-bit-cost
/// LAST path on a tie.
#[test]
fn multi_ref_with_identical_refs_still_picks_last_only() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) = encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    // Re-use the same K-frame reconstruction as LAST, GOLDEN, and
    // ALTREF. Every ref is bit-identical; the picker should bias to
    // LAST because LAST costs 1 selector bit vs. 2 for GOLDEN / ALTREF.
    let p_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (p_bytes, _p_planes) = encode_p_frame_multi_ref(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
    )
    .expect("P");

    let (prob_intra, prob_last, _prob_gf) = parse_inter_prob_triple(&p_bytes);
    assert_eq!(prob_intra, 255);
    assert_eq!(
        prob_last, 255,
        "all refs equal ⇒ picker prefers LAST (lower selector-bit \
         cost); prob_last = {prob_last}"
    );
}
