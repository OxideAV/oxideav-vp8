//! Round-165 stream-driver composition of the round-160 / 161 §11
//! intra-within-inter MB picker **with** the round-157 / 158 §13.4
//! token-prob observed-counts fitter, on top of the §9.7 / §9.8
//! reference-slot refresh control and the §9.4 `mb_lf_adjustments()`
//! delta layer.
//!
//! Round 163 wired the picker through the refresh + §9.4 deltas axis
//! ([`Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick`]).
//! Round 164 wired the fitter through the same axis
//! ([`Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`]).
//! Each composed individually with the lf-deltas axis but neither
//! composed with the other.
//!
//! Round 164's next-step ladder named exactly this composition as
//! item (5):
//!
//! > *(5) parallel fitter composition on the intra-pick + refresh +
//! > lf-deltas axis (combining r163 + r164 — the picker on the
//! > fitted refresh path).*
//!
//! Pins:
//!
//!   1. `bare_encoder_byte_match_on_composition` — bytes the new stream
//!      entry-point emits on a K + P sequence are byte-identical to the
//!      equivalent
//!      [`encode_keyframe_with_reconstruction`] +
//!      [`encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates`]
//!      bare-encoder composition. The cross-frame §9.4 carry feeds the
//!      bare-encoder call exactly as the stream method would.
//!   2. `never_grows_wire_vs_caller_driven_intra_pick_default` — on the
//!      same inputs, the fitted-composed stream bytes are never larger
//!      than the round-163 caller-driven
//!      [`Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick`]
//!      default. This is the round-158 bare-encoder safety guard
//!      lifted into the composed stream driver.
//!   3. `carries_deltas_and_resets_on_keyframe` — the across-frame §9.4
//!      carry rule applies on the composed path exactly as on every
//!      other refresh + lf-deltas sibling: adj-enabled frames write
//!      back the effective deltas; adj-disabled frames leave the carry
//!      untouched; keyframes reset to zero. Neither the §11 picker nor
//!      the §13.4 fitter perturbs the carry.
//!   4. `refresh_errors_when_no_last` — refresh-driven call before any
//!      prior frame populates `LAST` surfaces
//!      [`StreamEncodeError::NoLastReference`]; the frame counter is
//!      not advanced. Matches every other refresh-aware sibling.
//!   5. `dimensions_change_rejected` — dimensions-lock is preserved on
//!      the composed path; the counter is not advanced on rejection.
//!
//! Black-box self-decode — no external codec consulted.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates,
    FrameKind, I420Frame, KeyframeParams, LoopFilterDeltas, RefreshControls, StreamEncodeError,
    Vp8DecodedFrame, Vp8DecoderState, Vp8InterStreamEncoder,
};

/// Build a `frame_idx`-parameterised synthetic I420 picture with enough
/// per-frame variation that the fitter's observed-counts saving model
/// can produce a positive payload (mirrors the round-164 sibling's
/// helper, kept local to keep test files independent).
fn synthetic_frame(width: u32, height: u32, frame_idx: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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
    (y, u, v)
}

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

/// Composition byte-equality vs. the bare-encoder
/// `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates`
/// when all three knobs (refresh + §9.4 deltas + §11 picker + §13.4
/// fitter) are engaged simultaneously.
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

    let (k_y, k_u, k_v) = synthetic_frame(width, height, 0);
    let (p_y, p_u, p_v) = synthetic_frame(width, height, 1);
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
    // populates LAST; bare wrapper consumes the carried-zero state a
    // fresh stream would hold immediately after the K.
    let (bare_k_bytes, bare_k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &params).expect("bare K");
    let (bare_p_bytes, _bare_p_planes) =
        encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
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
    let stream_k = enc.encode_frame(&k_frame).expect("stream K");
    let stream_p = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame, &refresh, &lf,
        )
        .expect("stream P composed");

    assert_eq!(stream_k.kind, FrameKind::Key);
    assert_eq!(stream_p.kind, FrameKind::InterZeroMv);
    assert_eq!(
        stream_k.bytes, bare_k_bytes,
        "stream K must match bare K bytes (§13.5-default keyframe)"
    );
    assert_eq!(
        stream_p.bytes, bare_p_bytes,
        "stream composed P bytes must match the bare-encoder \
         encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates output"
    );

    // Self-decode round-trip — the fitter's bytes-vs-default safety
    // guard and the bare encoder's matched-planes invariant together
    // guarantee both arms of the fitter (win vs. fallback) decode
    // cleanly through `Vp8DecoderState`.
    let mut dec = Vp8DecoderState::new();
    let dk = dec.decode_frame(&stream_k.bytes).expect("decode K");
    let dp = dec.decode_frame(&stream_p.bytes).expect("decode P");
    let psnr_k = y_psnr(&k_y, &dk);
    let psnr_p = y_psnr(&p_y, &dp);
    eprintln!("composed K Y-PSNR = {psnr_k:.2} dB, composed P Y-PSNR = {psnr_p:.2} dB");
    assert!(psnr_k >= 25.0, "K PSNR {psnr_k:.2} dB below floor");
    assert!(psnr_p >= 20.0, "P PSNR {psnr_p:.2} dB below floor");

    // Stream-side carry: §9.4 enabled + update with fresh deltas ⇒
    // the carried state now holds the effective (transmitted) values.
    // Neither the §11 picker nor the §13.4 fitter perturbs the §9.4
    // carry.
    assert_eq!(enc.carried_ref_deltas(), [2, -3, 4, -5]);
    assert_eq!(enc.carried_mode_deltas(), [1, -2, 3, -4]);
}

/// Round-158 safety-guard lifted into the composed stream driver: the
/// composed-fitter wire is never larger than the round-163 caller-driven
/// `encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick` default on
/// the same inputs.
#[test]
fn never_grows_wire_vs_caller_driven_intra_pick_default() {
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

    // Two parallel encoder instances seeded with the same K. The
    // composed safety-guard property is per-frame: P[i] fitted-composed
    // bytes <= P[i] intra-pick-only default bytes across the sequence.
    let mut enc_default = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let mut enc_fitted = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");

    let (k_y, k_u, k_v) = synthetic_frame(width, height, 0);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let _kd = enc_default.encode_frame(&k_frame).expect("Kd");
    let _kf = enc_fitted.encode_frame(&k_frame).expect("Kf");

    for i in 1..5 {
        let (py, pu, pv) = synthetic_frame(width, height, i);
        let p_frame = I420Frame::packed(width, height, &py, &pu, &pv);
        let pd = enc_default
            .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&p_frame, &refresh, &lf)
            .expect("Pd");
        let pf = enc_fitted
            .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
                &p_frame, &refresh, &lf,
            )
            .expect("Pf");
        assert!(
            pf.bytes.len() <= pd.bytes.len(),
            "frame {i}: fitted-composed wire ({fl} B) must NOT grow vs. intra-pick \
             default caller-driven wire ({dl} B) — round-158 safety guard",
            fl = pf.bytes.len(),
            dl = pd.bytes.len(),
        );
    }
}

/// §9.4 across-frame delta carry rule applies on the composed path
/// identically to every other refresh + lf-deltas sibling. Keyframes
/// reset the carry to zero per §9.4. Neither the §11 picker nor the
/// §13.4 fitter perturbs the carry.
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

    // Use a P-frame source where the fitter is well-behaved (flat 128).
    // The carry behavior under test is independent of the fitter's
    // win/fall-back decision; this just keeps the test deterministic.
    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 128);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 128);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);
    let refresh = RefreshControls::default();

    // K — populates LAST, clears carry.
    let e0 = enc.encode_frame(&k_frame).expect("K");
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
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame, &refresh, &lf1,
        )
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
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame, &refresh, &lf2,
        )
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
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame, &refresh, &lf3,
        )
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
    // untouched on adj-disabled frames; mirror every other sibling.
    let lf4 = LoopFilterDeltas {
        enabled: false,
        update: false,
        ref_frame_delta: [None; 4],
        mode_delta: [None; 4],
    };
    let e4 = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame, &refresh, &lf4,
        )
        .expect("P4");
    let _ = dec.decode_frame(&e4.bytes).expect("decode P4");
    assert_eq!(enc.carried_ref_deltas(), [10, -2, 3, -4]);
    assert_eq!(enc.carried_mode_deltas(), [5, -6, 20, -8]);

    // K (forced) — §9.4 carry resets to zero per spec. Matches every
    // other refresh-aware sibling.
    let e5 = enc
        .encode_frame_with_force(&k_frame, true)
        .expect("K (forced)");
    let _ = dec.decode_frame(&e5.bytes).expect("decode K2");
    assert_eq!(e5.kind, FrameKind::Key);
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

/// Refusing a refresh-driven P-frame call before any LAST exists must
/// surface `NoLastReference` and leave the frame counter unchanged.
#[test]
fn refresh_errors_when_no_last() {
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
    let (py, pu, pv) = flat_color(width as usize, height as usize, 128);
    let p_frame = I420Frame::packed(width, height, &py, &pu, &pv);

    let err = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
        )
        .expect_err("no LAST yet");
    assert!(matches!(err, StreamEncodeError::NoLastReference));
    assert_eq!(enc.frame_count(), 0);
}

/// Dimensions-lock is preserved on the composed path.
#[test]
fn dimensions_change_rejected() {
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 16,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let (k_y, k_u, k_v) = flat_color(16, 16, 128);
    let k_frame = I420Frame::packed(16, 16, &k_y, &k_u, &k_v);
    let _k = enc.encode_frame(&k_frame).expect("K");
    let count_before = enc.frame_count();

    let (py, pu, pv) = flat_color(32, 32, 128);
    let p_frame = I420Frame::packed(32, 32, &py, &pu, &pv);
    let err = enc
        .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &p_frame,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
        )
        .expect_err("dimensions differ");
    assert!(matches!(
        err,
        StreamEncodeError::DimensionsChanged {
            first: (16, 16),
            got: (32, 32),
        }
    ));
    assert_eq!(enc.frame_count(), count_before, "counter not advanced");
}
