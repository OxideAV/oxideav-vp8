//! Round-163 stream-driver composition of the §9.4 `mb_lf_adjustments()`
//! delta layer with the round-160 / round-161 §11 intra-within-inter
//! MB picker on the refresh path.
//!
//! Round 162 threaded the picker into [`Vp8InterStreamEncoder`] as a
//! family of `_with_intra_pick` entry-points that mirror the existing
//! `encode_frame*` family. Its own next-step ladder named the
//! follow-up that this test pins:
//!
//!   > (3) compose the §9.4 `mb_lf_adjustments()` deltas with the
//!   > intra-pick on the refresh path
//!   > (`encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick`).
//!
//! Round 163 lands that composition. The new entry-point sits at the
//! intersection of two existing axes — caller-driven §9.4 deltas
//! (round 151) and the round-162 stream intra-pick path — and obeys
//! the carry rules of the first while activating the picker of the
//! second.
//!
//! Pins:
//!
//!   1. `disabled_deltas_byte_match_intra_pick_only_path` — passing
//!      [`LoopFilterDeltas::default`] (`enabled = false`) reproduces
//!      [`Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_intra_pick`]
//!      byte-for-byte. This is the round-162 wire — the round-163
//!      composition does not perturb it when the §9.4 layer is off.
//!   2. `bare_encoder_byte_match_on_composition` — bytes the stream
//!      driver emits on a K + P sequence with both deltas + intra-pick
//!      engaged are byte-identical to the equivalent
//!      [`encode_keyframe_with_reconstruction`] +
//!      [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick`]
//!      bare-encoder composition (the same byte-equality the round-162
//!      `_with_intra_pick` test pinned for the no-deltas variant).
//!   3. `carries_deltas_and_resets_on_keyframe` — the across-frame
//!      §9.4 carry rule applies on the composed path exactly as on the
//!      non-intra-pick [`Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas`]
//!      path: adj-enabled frames write back the effective deltas,
//!      keyframes reset to zero. The intra-pick toggle does not
//!      perturb the carry.
//!   4. `refresh_errors_when_no_last` — refresh-driven call before any
//!      prior frame populates `LAST` surfaces
//!      [`StreamEncodeError::NoLastReference`]; the frame counter is
//!      not advanced. Matches the round-162 sibling
//!      `stream_intra_pick_refresh_errors_when_no_last`.
//!   5. `dimensions_change_rejected` — dimensions-lock is preserved.
//!
//! Black-box self-decode — no external codec consulted.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick, FrameKind, I420Frame,
    KeyframeParams, LoopFilterDeltas, RefreshControls, StreamEncodeError, Vp8CodedHeader,
    Vp8DecodedFrame, Vp8DecoderState, Vp8FrameHeader, Vp8InterStreamEncoder,
};

fn flat_color(width: usize, height: usize, v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![v; width * height];
    let u = vec![v; (width / 2) * (height / 2)];
    let v_plane = vec![v; (width / 2) * (height / 2)];
    (y, u, v_plane)
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

fn y_psnr(src_y: &[u8], rec: &Vp8DecodedFrame) -> f64 {
    plane_psnr(src_y, &rec.y)
}

/// Pull the §9.10 `prob_intra` byte out of an inter-frame bitstream.
fn prob_intra_of(bytes: &[u8]) -> u8 {
    let hdr = Vp8FrameHeader::parse(bytes).expect("§19.1 tag parses");
    assert!(!hdr.key_frame, "prob_intra_of called on a K frame");
    let partition_start = hdr.header_bytes_consumed;
    let partition_end = partition_start + hdr.first_partition_size as usize;
    let partition = &bytes[partition_start..partition_end];
    let coded = Vp8CodedHeader::parse(partition, /*key_frame=*/ false)
        .expect("§19.2 control partition parses");
    coded.prob_intra.expect("inter prob_intra present")
}

/// Wire compatibility: `LoopFilterDeltas::default()` makes the
/// composed path byte-equal to the round-162
/// `encode_p_frame_with_refresh_and_intra_pick` wire.
#[test]
fn disabled_deltas_byte_match_intra_pick_only_path() {
    let width = 32u32;
    let height = 32u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 0);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 200);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let refresh = RefreshControls::default();

    // Path A — round-162 intra-pick-only stream entry-point.
    let mut enc_a = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let _ka = enc_a.encode_frame_with_intra_pick(&k_frame).expect("Ka");
    let pa = enc_a
        .encode_p_frame_with_refresh_and_intra_pick(&p_frame, &refresh)
        .expect("Pa");

    // Path B — round-163 composed entry-point with default (disabled)
    // §9.4 deltas. Both paths run on a fresh encoder state so the §9.4
    // carry across frames does not enter the comparison.
    let mut enc_b = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let _kb = enc_b.encode_frame_with_intra_pick(&k_frame).expect("Kb");
    let pb = enc_b
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(
            &p_frame,
            &refresh,
            &LoopFilterDeltas::default(),
        )
        .expect("Pb");

    assert_eq!(pa.kind, FrameKind::InterZeroMv);
    assert_eq!(pb.kind, FrameKind::InterZeroMv);
    assert_eq!(
        pa.bytes, pb.bytes,
        "composed path with default (disabled) deltas must reproduce the round-162 \
         encode_p_frame_with_refresh_and_intra_pick wire byte-for-byte"
    );

    // Carry must remain at zero: default deltas have enabled = false,
    // which the round-151 carry rule documents as "leave carry
    // unchanged" — and the prior keyframe just reset it to zero.
    assert_eq!(enc_b.carried_ref_deltas(), [0; 4]);
    assert_eq!(enc_b.carried_mode_deltas(), [0; 4]);
}

/// Composition byte-equality vs. the bare-encoder
/// `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick`
/// when both knobs are engaged simultaneously.
#[test]
fn bare_encoder_byte_match_on_composition() {
    let width = 32u32;
    let height = 32u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 0);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 200);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let refresh = RefreshControls {
        refresh_last: true,
        refresh_golden_frame: false,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 0,
        copy_buffer_to_alternate: 0,
    };
    let lf = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(2), Some(-3), Some(4), Some(-5)],
        mode_delta: [Some(1), Some(-2), Some(3), Some(-4)],
    };

    // ── Bare-encoder reference: encode_keyframe_with_reconstruction
    // populates LAST; bare wrapper consumes the carried-zero state
    // a fresh stream would hold immediately after the K.
    let (bare_k_bytes, bare_k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &params).expect("bare K");
    let (bare_p_bytes, _bare_p_planes) =
        encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick(
            &p_frame,
            &bare_k_planes,
            None,
            None,
            &params,
            &refresh,
            &lf,
            /* carried_ref_deltas  = */ [0; 4],
            /* carried_mode_deltas = */ [0; 4],
        )
        .expect("bare P composed");

    // ── Stream-driver composed path.
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let stream_k = enc
        .encode_frame_with_intra_pick(&k_frame)
        .expect("stream K");
    let stream_p = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&p_frame, &refresh, &lf)
        .expect("stream P composed");

    assert_eq!(stream_k.kind, FrameKind::Key);
    assert_eq!(stream_p.kind, FrameKind::InterZeroMv);
    assert_eq!(
        stream_k.bytes, bare_k_bytes,
        "stream K must match bare K bytes"
    );
    assert_eq!(
        stream_p.bytes, bare_p_bytes,
        "stream composed P bytes must match the bare encoder's \
         encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick output"
    );

    // Picker activated. On a black-LAST + bright-source pattern the
    // intra mode dominates every MB ⇒ fitter clamps `prob_intra` to
    // `255` (the all-intra boundary, NOT the historical sentinel —
    // only the fitter ever writes that exact value). On any frame
    // where the picker engaged at all, `prob_intra > 1`.
    let pi = prob_intra_of(&stream_p.bytes);
    eprintln!("composed P §9.10 prob_intra = {pi}");
    assert!(
        pi > 1,
        "composed P prob_intra = {pi}: picker should select intra for at least one MB on a \
         black-LAST + bright-source pattern (matches the round-162 intra-pick sibling test)"
    );

    // Self-decode round-trip.
    let mut dec = Vp8DecoderState::new();
    let dk = dec.decode_frame(&stream_k.bytes).expect("decode K");
    let dp = dec.decode_frame(&stream_p.bytes).expect("decode P");
    let psnr_k = y_psnr(&k_y, &dk);
    let psnr_p = y_psnr(&p_y, &dp);
    eprintln!("composed K Y-PSNR = {psnr_k:.2} dB, composed P Y-PSNR = {psnr_p:.2} dB");
    assert!(psnr_k >= 30.0, "K PSNR {psnr_k:.2} dB below floor");
    assert!(psnr_p >= 25.0, "P PSNR {psnr_p:.2} dB below floor");

    // Stream-side carry: §9.4 enabled + update with fresh deltas ⇒
    // the carried state now holds the effective (transmitted) values.
    assert_eq!(enc.carried_ref_deltas(), [2, -3, 4, -5]);
    assert_eq!(enc.carried_mode_deltas(), [1, -2, 3, -4]);
}

/// §9.4 across-frame delta carry rule applies on the composed path
/// identically to the non-intra-pick
/// [`Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas`].
/// Keyframes reset the carry to zero per §9.4.
#[test]
fn carries_deltas_and_resets_on_keyframe() {
    let width = 16u32;
    let height = 16u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let mut dec = Vp8DecoderState::new();

    // Use a P-frame source where intra never beats inter (flat 128).
    // The carry behavior under test is unrelated to which candidate
    // the picker chose; this just keeps the picker happy.
    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 128);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 128);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);
    let refresh = RefreshControls::default();

    // K — populates LAST, clears carry.
    let e0 = enc.encode_frame_with_intra_pick(&k_frame).expect("K");
    let _ = dec.decode_frame(&e0.bytes).expect("decode K");
    assert_eq!(e0.kind, FrameKind::Key);
    assert_eq!(enc.carried_ref_deltas(), [0; 4]);
    assert_eq!(enc.carried_mode_deltas(), [0; 4]);

    // P1 — enabled + update with fresh deltas. Carry advances to the
    // effective values.
    let lf1 = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(1), Some(-2), Some(3), Some(-4)],
        mode_delta: [Some(5), Some(-6), Some(7), Some(-8)],
    };
    let e1 = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&p_frame, &refresh, &lf1)
        .expect("P1");
    let _ = dec.decode_frame(&e1.bytes).expect("decode P1");
    assert_eq!(enc.carried_ref_deltas(), [1, -2, 3, -4]);
    assert_eq!(enc.carried_mode_deltas(), [5, -6, 7, -8]);

    // P2 — enabled + update = false. The wire skips per-slot values
    // and the decoder uses carried state; the encoder's carried state
    // must remain unchanged across the carry-through frame.
    let lf2 = LoopFilterDeltas {
        enabled: true,
        update: false,
        ref_frame_delta: [None; 4],
        mode_delta: [None; 4],
    };
    let e2 = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&p_frame, &refresh, &lf2)
        .expect("P2");
    let _ = dec.decode_frame(&e2.bytes).expect("decode P2");
    assert_eq!(enc.carried_ref_deltas(), [1, -2, 3, -4]);
    assert_eq!(enc.carried_mode_deltas(), [5, -6, 7, -8]);

    // P3 — enabled + update with partial slots; un-updated slots
    // retain the carried values, updated slots take the new values.
    let lf3 = LoopFilterDeltas {
        enabled: true,
        update: true,
        ref_frame_delta: [Some(10), None, None, None],
        mode_delta: [None, None, Some(20), None],
    };
    let e3 = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&p_frame, &refresh, &lf3)
        .expect("P3");
    let _ = dec.decode_frame(&e3.bytes).expect("decode P3");
    assert_eq!(
        enc.carried_ref_deltas(),
        [10, -2, 3, -4],
        "carried[0] advances; carried[1..=3] persist"
    );
    assert_eq!(
        enc.carried_mode_deltas(),
        [5, -6, 20, -8],
        "carried[2] advances; rest persist"
    );

    // P4 — adj-disabled. Per §9.4 the spec keeps the carried state
    // untouched on adj-disabled frames; mirror the non-intra-pick
    // sibling's expectation.
    let lf4 = LoopFilterDeltas {
        enabled: false,
        update: false,
        ref_frame_delta: [None; 4],
        mode_delta: [None; 4],
    };
    let e4 = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&p_frame, &refresh, &lf4)
        .expect("P4");
    let _ = dec.decode_frame(&e4.bytes).expect("decode P4");
    assert_eq!(
        enc.carried_ref_deltas(),
        [10, -2, 3, -4],
        "adj-disabled frame must not touch carried state"
    );
    assert_eq!(enc.carried_mode_deltas(), [5, -6, 20, -8]);

    // K (forced) — §9.4 reset.
    let _e5 = enc
        .encode_frame_with_force_and_intra_pick(&k_frame, true)
        .expect("K (forced)");
    assert_eq!(
        enc.carried_ref_deltas(),
        [0; 4],
        "keyframe resets ref-delta carry per §9.4"
    );
    assert_eq!(
        enc.carried_mode_deltas(),
        [0; 4],
        "keyframe resets mode-delta carry per §9.4"
    );
}

/// Without any prior frame populating `LAST` the refresh-driven entry
/// errors with `NoLastReference` and does not advance the counter.
/// Matches the round-162 sibling
/// `stream_intra_pick_refresh_errors_when_no_last`.
#[test]
fn refresh_errors_when_no_last() {
    let width = 16u32;
    let height = 16u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let (y, u, v) = flat_color(width as usize, height as usize, 128);
    let frame = I420Frame::packed(width, height, &y, &u, &v);
    let refresh = RefreshControls::default();
    let lf = LoopFilterDeltas::default();
    let mut enc = Vp8InterStreamEncoder::new(params, 4).expect("non-zero interval");
    let err = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&frame, &refresh, &lf)
        .expect_err("should refuse without a LAST slot");
    assert!(
        matches!(err, StreamEncodeError::NoLastReference),
        "expected NoLastReference, got {err:?}"
    );
    assert_eq!(enc.frame_count(), 0, "failure must not advance the counter");
}

/// Dimensions-lock semantics are preserved on the composed path.
#[test]
fn dimensions_change_rejected() {
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let (y1, u1, v1) = flat_color(32, 32, 128);
    let f1 = I420Frame::packed(32, 32, &y1, &u1, &v1);
    let (y2, u2, v2) = flat_color(48, 48, 64);
    let f2 = I420Frame::packed(48, 48, &y2, &u2, &v2);

    let refresh = RefreshControls::default();
    let lf = LoopFilterDeltas::default();

    let mut enc = Vp8InterStreamEncoder::new(params, 4).expect("non-zero interval");
    // Need a populated LAST so we get past the NoLastReference check
    // and into the dimensions branch on f2.
    enc.encode_frame_with_intra_pick(&f1)
        .expect("first frame locks dims");
    let err = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&f2, &refresh, &lf)
        .expect_err("differently-sized second frame");
    assert!(
        matches!(
            err,
            StreamEncodeError::DimensionsChanged {
                first: (32, 32),
                got: (48, 48),
            }
        ),
        "expected DimensionsChanged((32,32) → (48,48)), got {err:?}"
    );
    assert_eq!(enc.frame_count(), 1, "failure must not advance the counter");
}
