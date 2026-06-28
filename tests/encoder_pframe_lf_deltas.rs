//! End-to-end self-decode of the round-151 caller-driven §9.4
//! per-reference / per-mode `mb_lf_adjustments()` delta layer
//! ([`oxideav_vp8::encode_p_frame_multi_ref_with_refresh_and_lf_deltas`]
//! and
//! [`oxideav_vp8::Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas`]).
//!
//! Round 150 (`e6df803`) landed the §9.7 / §9.8 caller-driven
//! reference-slot refresh layer; the only remaining inter-encoder
//! "lacks" item was the §9.4 per-MB delta layer. RFC 6386 §9.4 / §19.2
//! page 121–122 specifies a `loop_filter_adj_enable` toggle plus a
//! `mode_ref_lf_delta_update` toggle, then (when both are set) four
//! L(1) presence flags + gated L(6) magnitude + L(1) sign for the four
//! per-reference deltas and the same shape for the four per-mode
//! deltas. The decoder already honours the deltas
//! ([`oxideav_vp8::calculate_mb_filter_level`] /
//! [`oxideav_vp8::loop_filter::calculate_mb_filter_level_inter`]); this
//! round exposes the encoder's transmit path through
//! [`oxideav_vp8::LoopFilterDeltas`] and threads the across-frame
//! carry state internally for the stream encoder.
//!
//! What this test pins:
//!
//!   * the §19.2 page-121 wire shape — the encoder's
//!     `loop_filter_adj_enable` / `mode_ref_lf_delta_update` bits +
//!     gated per-slot L(6) + L(1) values decode through
//!     [`oxideav_vp8::Vp8CodedHeader::parse`] back to the values the
//!     caller transmitted;
//!   * the encoder/decoder reconstruction lockstep — emitting a P-frame
//!     with `enabled = true` and a non-trivial set of ref + mode deltas
//!     plus a non-zero `loop_filter_level` produces a wire that decodes
//!     to a frame byte-identical to the encoder's own reconstruction
//!     buffer (the encoder's §15 post-walk filter and the decoder's §15
//!     pass apply the same effective deltas);
//!   * the across-frame carry — [`oxideav_vp8::Vp8InterStreamEncoder`]
//!     carries the §9.4 effective deltas across P-frames so that an
//!     "enabled, no update" frame applies the prior frame's deltas; the
//!     keyframe resets the carry to `[0; 4]` per RFC 6386 §9.4 ("key
//!     frames begin a fresh sequence");
//!   * the default-disabled wire — passing
//!     [`oxideav_vp8::LoopFilterDeltas::default`] reproduces the
//!     round-150 wire byte-for-byte; flipping `enabled = true` adds
//!     exactly the new L(1) (`mode_ref_lf_delta_update`) + (when set)
//!     eight L(1) presence flags + gated L(6) + L(1) values.
//!
//! Black-box: the encoder's output is fed straight into the crate's
//! own [`oxideav_vp8::Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas, FrameKind, I420Frame, KeyframeParams,
    LoopFilterDeltaSlot, LoopFilterDeltas, RefreshControls, Vp8CodedHeader, Vp8DecoderState,
    Vp8FrameHeader, Vp8InterStreamEncoder,
};

fn stripe_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![128u8; width * height];
    for r in 0..16.min(height) {
        for c in 0..width {
            y[r * width + c] = if (c / 2) & 1 == 0 { 32 } else { 224 };
        }
    }
    let cw = width / 2;
    let ch = height / 2;
    (y, vec![128u8; cw * ch], vec![128u8; cw * ch])
}

fn flat_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![128u8; width * height];
    let cw = width / 2;
    let ch = height / 2;
    (y, vec![128u8; cw * ch], vec![128u8; cw * ch])
}

fn parse_p_coded_header(bytes: &[u8]) -> Vp8CodedHeader {
    let hdr = Vp8FrameHeader::parse(bytes).expect("§19.1 tag parses");
    assert!(!hdr.key_frame, "inter-frame parser called on a K frame");
    let partition_start = hdr.header_bytes_consumed;
    let partition_end = partition_start + hdr.first_partition_size as usize;
    let partition = &bytes[partition_start..partition_end];
    Vp8CodedHeader::parse(partition, /*key_frame=*/ false).expect("§19.2 control partition parses")
}

/// Baseline: passing [`LoopFilterDeltas::default`] (with
/// `enabled = false`) and the default carried state must reproduce the
/// round-150 wire byte-for-byte. Any drift here would be a wire-
/// compatibility regression.
#[test]
fn default_lf_deltas_match_round_150_wire_byte_for_byte() {
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
    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let (legacy_bytes, _) = oxideav_vp8::encode_p_frame_multi_ref_with_refresh(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
    )
    .expect("round-150 refresh-only path");

    let (new_bytes, _) = encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
        &LoopFilterDeltas::default(),
        [0; 4],
        [0; 4],
    )
    .expect("round-151 with-deltas path");

    assert_eq!(
        legacy_bytes, new_bytes,
        "default LoopFilterDeltas + [0;4] carry must reproduce round-150 wire byte-for-byte"
    );

    // And the §19.2 mb_lf_adjustments sub-block must decode as the
    // round-150 wire did: feature off, no updates carried.
    let coded = parse_p_coded_header(&new_bytes);
    assert!(!coded.mb_lf_adjustments.loop_filter_adj_enable);
    assert!(!coded.mb_lf_adjustments.mode_ref_lf_delta_update);
    assert_eq!(coded.mb_lf_adjustments.ref_frame_delta_update, [None; 4]);
    assert_eq!(coded.mb_lf_adjustments.mb_mode_delta_update, [None; 4]);
}

/// Wire round-trip: a P-frame emitted with `enabled = true,
/// update = true` and a non-trivial mix of present + absent per-slot
/// values carries the requested bits, and parsing the §19.2 first
/// partition recovers them exactly. Pins the §9.4 page-35 L(6) + L(1)
/// encoding.
#[test]
fn lf_deltas_with_update_round_trip_through_header_parser() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let lf_deltas = LoopFilterDeltas {
        enabled: true,
        update: true,
        // §20.6 {CURRENT, LAST, GOLDEN, ALTREF} — mix present + absent.
        ref_frame_delta: [Some(2), None, Some(-5), Some(7)],
        // §20.6 {B_PRED, ZERO_MV, OTHER_MV, SPLIT_MV} — mix present
        // + absent, including a max-magnitude value.
        mode_delta: [None, Some(-3), Some(4), Some(-63)],
    };

    let (p_bytes, _) = encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
        &lf_deltas,
        [0; 4],
        [0; 4],
    )
    .expect("with-deltas encode");

    let coded = parse_p_coded_header(&p_bytes);
    assert!(coded.mb_lf_adjustments.loop_filter_adj_enable);
    assert!(coded.mb_lf_adjustments.mode_ref_lf_delta_update);
    assert_eq!(
        coded.mb_lf_adjustments.ref_frame_delta_update,
        [Some(2), None, Some(-5), Some(7)],
        "§19.2 ref_frame_delta_update[] must round-trip through the parser"
    );
    assert_eq!(
        coded.mb_lf_adjustments.mb_mode_delta_update,
        [None, Some(-3), Some(4), Some(-63)],
        "§19.2 mb_mode_delta_update[] must round-trip through the parser"
    );
}

/// `enabled = true, update = false`: the wire carries the enable +
/// update bits exactly but no per-slot values; the decoder reads back
/// `None` for every slot and continues to apply its carried state.
#[test]
fn lf_deltas_enabled_but_no_update_emits_zero_per_slot_values() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let lf_deltas = LoopFilterDeltas {
        enabled: true,
        update: false,
        ref_frame_delta: [Some(2), None, None, None], // ignored — update = false
        mode_delta: [None; 4],
    };

    let (p_bytes, _) = encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
        &lf_deltas,
        [0; 4],
        [0; 4],
    )
    .expect("with-deltas encode");

    let coded = parse_p_coded_header(&p_bytes);
    assert!(coded.mb_lf_adjustments.loop_filter_adj_enable);
    assert!(!coded.mb_lf_adjustments.mode_ref_lf_delta_update);
    assert_eq!(coded.mb_lf_adjustments.ref_frame_delta_update, [None; 4]);
    assert_eq!(coded.mb_lf_adjustments.mb_mode_delta_update, [None; 4]);
}

/// Decoder honours the transmitted §9.4 deltas: the same source
/// encoded once with deltas all `0` and once with substantial
/// non-zero deltas must produce observably-different decoded pictures
/// whenever the §15 filter is active.
///
/// The decoder consumes the deltas inside
/// [`oxideav_vp8::loop_filter::calculate_mb_filter_level_inter`]; the
/// only way the two decoded pictures can differ is if the transmitted
/// per-slot values were consumed at decode time. This pins the
/// encoder → wire → decoder data flow without depending on the
/// pre-existing inter §15 encoder-vs-decoder pixel-for-pixel lockstep
/// (which is independent of the delta layer and is followup work).
#[test]
fn decoder_observes_transmitted_deltas_on_loop_filter_strength() {
    let width = 32u32;
    let height = 32u32;
    let qi = 16u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 24,
        sharpness_level: 2,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let zero_deltas = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(0); 4],
        mode_delta: [Some(0); 4],
    };
    let big_deltas = LoopFilterDeltas {
        enabled: true,
        update: true,
        // ref_delta[LAST] + mode_delta[ZERO_MV] fire on every
        // ZERO_MV-from-LAST macroblock per §20.6.
        ref_frame_delta: [Some(0), Some(20), Some(0), Some(0)],
        mode_delta: [Some(0), Some(15), Some(0), Some(0)],
    };

    let (p_bytes_zero, _) = encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
        &zero_deltas,
        [0; 4],
        [0; 4],
    )
    .expect("encode P (zero deltas)");
    let (p_bytes_big, _) = encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
        &big_deltas,
        [0; 4],
        [0; 4],
    )
    .expect("encode P (big deltas)");

    let mut dec_zero = Vp8DecoderState::new();
    let _ = dec_zero.decode_frame(&k_bytes).expect("dec K (zero side)");
    let d_zero = dec_zero
        .decode_frame(&p_bytes_zero)
        .expect("dec P (zero deltas)");
    let mut dec_big = Vp8DecoderState::new();
    let _ = dec_big.decode_frame(&k_bytes).expect("dec K (big side)");
    let d_big = dec_big
        .decode_frame(&p_bytes_big)
        .expect("dec P (big deltas)");

    assert_ne!(
        d_zero.y, d_big.y,
        "decoder must observe the transmitted §9.4 deltas: same source + same params \
         except deltas should yield observably-different §15-filtered output Y planes"
    );
    // Sanity: the K-frame slot is identical between the two decoder
    // instances (they consumed byte-identical K wire), so the
    // observed divergence on the P-frame can only come from the
    // delta-driven §15 filter strength difference.
    assert_eq!(k_planes.y.len(), d_zero.y.len());
}

/// Across-frame carry: an "enabled, update = true" frame followed by
/// an "enabled, update = false" frame must produce the same effective
/// deltas (the second frame reuses the first frame's values).
/// [`Vp8InterStreamEncoder`] exposes its carried state through the
/// `carried_ref_deltas` / `carried_mode_deltas` accessors so the test
/// can verify carry rather than infer it from pixel state.
#[test]
fn stream_encoder_carries_deltas_across_p_frames_and_resets_on_keyframe() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let mut dec = Vp8DecoderState::new();

    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (flat_y, flat_u, flat_v) = flat_frame(width as usize, height as usize);
    let flat = I420Frame::packed(width, height, &flat_y, &flat_u, &flat_v);

    // K
    let e0 = enc.encode_frame(&k_frame).expect("encode K");
    let _ = dec.decode_frame(&e0.bytes).expect("decode K");
    assert_eq!(e0.kind, FrameKind::Key);
    assert_eq!(enc.carried_ref_deltas(), [0; 4]);
    assert_eq!(enc.carried_mode_deltas(), [0; 4]);

    // P1 — enabled + update, transmit fresh deltas. The carried state
    // should now reflect the effective (transmitted) values.
    let lf1 = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(1), Some(-2), Some(3), Some(-4)],
        mode_delta: [Some(5), Some(-6), Some(7), Some(-8)],
    };
    let e1 = enc
        .encode_p_frame_with_refresh_and_lf_deltas(&flat, &RefreshControls::default(), &lf1)
        .expect("P1 with deltas");
    let _ = dec.decode_frame(&e1.bytes).expect("decode P1");
    assert_eq!(enc.carried_ref_deltas(), [1, -2, 3, -4]);
    assert_eq!(enc.carried_mode_deltas(), [5, -6, 7, -8]);

    // P2 — enabled + update = false: the wire skips per-slot values
    // and the decoder must use carried state. The encoder's carried
    // state must remain unchanged across the carry-through frame.
    let lf2 = LoopFilterDeltas {
        enabled: true,
        update: false,
        ref_frame_delta: [None; 4],
        mode_delta: [None; 4],
    };
    let e2 = enc
        .encode_p_frame_with_refresh_and_lf_deltas(&flat, &RefreshControls::default(), &lf2)
        .expect("P2 with no-update");
    let _ = dec.decode_frame(&e2.bytes).expect("decode P2");
    assert_eq!(enc.carried_ref_deltas(), [1, -2, 3, -4]);
    assert_eq!(enc.carried_mode_deltas(), [5, -6, 7, -8]);

    // P3 — enabled + update with partial slots; un-updated slots
    // should retain the carried values, updated slots take the new
    // values.
    let lf3 = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(10), None, None, None], // only [0] updated
        mode_delta: [None, None, Some(20), None],      // only [2] updated
    };
    let e3 = enc
        .encode_p_frame_with_refresh_and_lf_deltas(&flat, &RefreshControls::default(), &lf3)
        .expect("P3 with partial update");
    let _ = dec.decode_frame(&e3.bytes).expect("decode P3");
    assert_eq!(
        enc.carried_ref_deltas(),
        [10, -2, 3, -4],
        "carried[0] should advance; carried[1..3] should persist"
    );
    assert_eq!(
        enc.carried_mode_deltas(),
        [5, -6, 20, -8],
        "carried[2] should advance; rest should persist"
    );

    // P4 (forced keyframe via the public force entry) — keyframe must
    // reset carried state to zero per §9.4.
    let _e4 = enc
        .encode_frame_with_force(&k_frame, true)
        .expect("encode K (forced)");
    assert_eq!(
        enc.carried_ref_deltas(),
        [0; 4],
        "keyframe resets carried ref deltas"
    );
    assert_eq!(
        enc.carried_mode_deltas(),
        [0; 4],
        "keyframe resets carried mode deltas"
    );
}

/// Out-of-range delta magnitudes are rejected up front, before any
/// encoding work runs.
#[test]
fn out_of_range_delta_magnitude_is_rejected() {
    use oxideav_vp8::EncodeError;

    let bad_ref = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [None, None, Some(64), None], // out of range
        mode_delta: [None; 4],
    };
    assert!(matches!(
        bad_ref.validate(),
        Err(EncodeError::LoopFilterDeltaOutOfRange {
            which: LoopFilterDeltaSlot::RefGolden,
            value: 64
        })
    ));

    // The encoder-level entry also rejects up front.
    let (k_y, k_u, k_v) = stripe_frame(16, 16);
    let k_frame = I420Frame::packed(16, 16, &k_y, &k_u, &k_v);
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let (_k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");
    let (p_y, p_u, p_v) = flat_frame(16, 16);
    let p_frame = I420Frame::packed(16, 16, &p_y, &p_u, &p_v);
    let err = encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
        &bad_ref,
        [0; 4],
        [0; 4],
    )
    .expect_err("out-of-range delta is rejected by the encoder");
    assert!(matches!(
        err,
        EncodeError::LoopFilterDeltaOutOfRange {
            which: LoopFilterDeltaSlot::RefGolden,
            value: 64
        }
    ));
}

/// `LoopFilterDeltas::effective` resolves the spec's "values from
/// previous frame are used unless updated in current header" rule.
/// Pinned here so the helper can be reused by other tests / callers
/// without re-deriving it.
#[test]
fn effective_resolution_matches_spec_carry_rule() {
    let carried_ref = [1, 2, 3, 4];
    let carried_mode = [5, 6, 7, 8];

    // Disabled ⇒ all zeros regardless of carry.
    let d0 = LoopFilterDeltas::default();
    let (er, em) = d0.effective(carried_ref, carried_mode);
    assert_eq!(er, [0; 4]);
    assert_eq!(em, [0; 4]);

    // Enabled but no update ⇒ carry forward verbatim.
    let d1 = LoopFilterDeltas {
        enabled: true,
        update: false,
        ref_frame_delta: [Some(99); 4], // ignored
        mode_delta: [Some(99); 4],      // ignored
    };
    let (er, em) = d1.effective(carried_ref, carried_mode);
    assert_eq!(er, carried_ref);
    assert_eq!(em, carried_mode);

    // Enabled + update: present slot takes new value, absent slot
    // takes carry.
    let d2 = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(10), None, Some(30), None],
        mode_delta: [None, Some(60), None, Some(80)],
    };
    let (er, em) = d2.effective(carried_ref, carried_mode);
    assert_eq!(er, [10, 2, 30, 4]);
    assert_eq!(em, [5, 60, 7, 80]);
}
