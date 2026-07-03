//! Equivalence + lockstep coverage for the round-387 generic coding-
//! options front doors:
//!
//!   * [`encode_keyframe_with_reconstruction_and_coding_options`] /
//!     [`KeyframeCodingOptions`],
//!   * [`encode_p_frame_multi_ref_with_refresh_and_coding_options`] /
//!     [`InterCodingOptions`],
//!   * [`encode_invisible_altref_update_with_coding_options`].
//!
//! What this pins:
//!
//!   1. every single-toggle configuration is **byte-identical** to the
//!      historical named entry point it subsumes (the options struct is
//!      a re-plumbing, not a re-encode);
//!   2. the previously unreachable combinations (auto-LF + fitted
//!      updates, on both frame kinds) self-decode in pixel lockstep
//!      through [`Vp8DecoderState`] and never exceed the non-fitted
//!      wire size;
//!   3. the invisible-anchor variant keeps the §9.1 `show_frame = 0`
//!      bit under every toggle set.
//!
//! Black-box: encoder output feeds the crate's own decoder only.

use oxideav_vp8::{
    encode_invisible_altref_update, encode_invisible_altref_update_with_coding_options,
    encode_keyframe_auto_loop_filter_with_reconstruction, encode_keyframe_with_reconstruction,
    encode_keyframe_with_reconstruction_and_coding_options,
    encode_keyframe_with_reconstruction_and_fitted_token_prob_updates,
    encode_p_frame_multi_ref_auto_loop_filter, encode_p_frame_multi_ref_with_refresh,
    encode_p_frame_multi_ref_with_refresh_and_coding_options,
    encode_p_frame_multi_ref_with_refresh_and_intra_pick,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates,
    I420Frame, InterCodingOptions, KeyframeCodingOptions, KeyframeParams, LoopFilterDeltas,
    RefreshControls, Vp8DecoderState,
};

const W: usize = 64;
const H: usize = 64;

/// Deterministic textured source: smooth gradients + a diagonal edge +
/// mid-frequency ripple, shifted by `dx` pixels so consecutive frames
/// carry trackable motion. Busy enough that the §13.4 fitter finds
/// profitable slots and the §9.4 selector picks a non-trivial level.
fn source(dx: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    let mut u = vec![0u8; (W / 2) * (H / 2)];
    let mut v = vec![0u8; (W / 2) * (H / 2)];
    for r in 0..H {
        for c in 0..W {
            let cc = c + dx;
            let ripple = (((cc * 13) % 31) as i32 - 15) * 2;
            let edge = if (r + cc) % 17 < 3 { 40 } else { 0 };
            let base = 60 + ((r * 2 + cc) % 120) as i32;
            y[r * W + c] = (base + ripple + edge).clamp(0, 255) as u8;
        }
    }
    for r in 0..H / 2 {
        for c in 0..W / 2 {
            u[r * (W / 2) + c] = (100 + ((r + c + dx) % 40)) as u8;
            v[r * (W / 2) + c] = (140 + ((r * 2 + c + dx) % 30)) as u8;
        }
    }
    (y, u, v)
}

fn params() -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: 40,
        loop_filter_level: 12,
        ..KeyframeParams::default()
    }
}

// ───────────────────────── keyframe equivalences ─────────────────────────

#[test]
fn keyframe_options_default_matches_plain_reconstruction_entry() {
    let (y, u, v) = source(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (a, pa) = encode_keyframe_with_reconstruction(&frame, &p).expect("plain");
    let (b, pb) = encode_keyframe_with_reconstruction_and_coding_options(
        &frame,
        &p,
        &KeyframeCodingOptions::default(),
    )
    .expect("options default");
    assert_eq!(a, b, "default options must reproduce the plain wire");
    assert_eq!(pa.y, pb.y, "reconstructions must match too");
}

#[test]
fn keyframe_options_auto_lf_matches_dedicated_entry() {
    let (y, u, v) = source(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (a, _) = encode_keyframe_auto_loop_filter_with_reconstruction(&frame, &p).expect("auto");
    let (b, _) = encode_keyframe_with_reconstruction_and_coding_options(
        &frame,
        &p,
        &KeyframeCodingOptions {
            auto_loop_filter: true,
            ..Default::default()
        },
    )
    .expect("options auto");
    assert_eq!(
        a, b,
        "auto_loop_filter toggle must reproduce the dedicated wire"
    );
}

#[test]
fn keyframe_options_fitted_matches_dedicated_entry() {
    let (y, u, v) = source(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (a, _) =
        encode_keyframe_with_reconstruction_and_fitted_token_prob_updates(&frame, &p).expect("fit");
    let (b, _) = encode_keyframe_with_reconstruction_and_coding_options(
        &frame,
        &p,
        &KeyframeCodingOptions {
            fitted_token_prob_updates: true,
            ..Default::default()
        },
    )
    .expect("options fit");
    assert_eq!(a, b, "fitted toggle must reproduce the dedicated wire");
}

#[test]
fn keyframe_auto_lf_plus_fitted_decodes_lockstep_and_never_grows() {
    let (y, u, v) = source(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (baseline, _) =
        encode_keyframe_auto_loop_filter_with_reconstruction(&frame, &p).expect("auto only");
    let (bytes, planes) = encode_keyframe_with_reconstruction_and_coding_options(
        &frame,
        &p,
        &KeyframeCodingOptions {
            auto_loop_filter: true,
            fitted_token_prob_updates: true,
        },
    )
    .expect("combined");
    assert!(
        bytes.len() <= baseline.len(),
        "fitted pass must never grow the wire ({} > {})",
        bytes.len(),
        baseline.len()
    );

    // 64×64 is MB-aligned, so the decoded visible picture equals the
    // encoder's reconstruction planes byte-for-byte.
    let mut state = Vp8DecoderState::new();
    let dec = state.decode_frame(&bytes).expect("keyframe must decode");
    assert_eq!(dec.y, planes.y, "encoder/decoder luma lockstep");
    assert_eq!(dec.u, planes.u, "encoder/decoder U lockstep");
    assert_eq!(dec.v, planes.v, "encoder/decoder V lockstep");
}

// ───────────────────────── inter equivalences ─────────────────────────

/// Encode the shared keyframe and return (bytes, reconstruction).
fn keyed_reference() -> (Vec<u8>, oxideav_vp8::KeyframePlanes) {
    let (y, u, v) = source(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    encode_keyframe_with_reconstruction(&frame, &params()).expect("keyframe")
}

#[test]
fn inter_options_default_matches_with_refresh_entry() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(2);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let refresh = RefreshControls::default();
    let (a, pa) = encode_p_frame_multi_ref_with_refresh(&frame, &kf, None, None, &p, &refresh)
        .expect("plain");
    let (b, pb) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &frame,
        &kf,
        None,
        None,
        &p,
        &refresh,
        &InterCodingOptions::default(),
    )
    .expect("options default");
    assert_eq!(a, b, "default options must reproduce the with_refresh wire");
    assert_eq!(pa.y, pb.y);
}

#[test]
fn inter_options_intra_pick_matches_dedicated_entry() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(2);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let refresh = RefreshControls::default();
    let (a, _) =
        encode_p_frame_multi_ref_with_refresh_and_intra_pick(&frame, &kf, None, None, &p, &refresh)
            .expect("intra pick");
    let (b, _) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &frame,
        &kf,
        None,
        None,
        &p,
        &refresh,
        &InterCodingOptions {
            intra_pick: true,
            ..Default::default()
        },
    )
    .expect("options intra pick");
    assert_eq!(a, b, "intra_pick toggle must reproduce the dedicated wire");
}

#[test]
fn inter_options_auto_lf_plus_intra_pick_matches_dedicated_entry() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(2);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (a, _) = encode_p_frame_multi_ref_auto_loop_filter(&frame, &kf, None, None, &p)
        .expect("auto lf entry");
    let (b, _) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &frame,
        &kf,
        None,
        None,
        &p,
        &RefreshControls::default(),
        &InterCodingOptions {
            intra_pick: true,
            auto_loop_filter: true,
            ..Default::default()
        },
    )
    .expect("options auto lf");
    assert_eq!(
        a, b,
        "intra_pick + auto_loop_filter must reproduce encode_p_frame_multi_ref_auto_loop_filter"
    );
}

#[test]
fn inter_options_intra_pick_fitted_matches_dedicated_entry() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(2);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let refresh = RefreshControls::default();
    let (a, _) =
        encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
            &frame,
            &kf,
            None,
            None,
            &p,
            &refresh,
            &LoopFilterDeltas::default(),
            [0; 4],
            [0; 4],
        )
        .expect("fitted intra pick");
    let (b, _) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &frame,
        &kf,
        None,
        None,
        &p,
        &refresh,
        &InterCodingOptions {
            intra_pick: true,
            fitted_token_prob_updates: true,
            ..Default::default()
        },
    )
    .expect("options fitted intra pick");
    assert_eq!(
        a, b,
        "intra_pick + fitted toggles must reproduce the dedicated fitter wire"
    );
}

#[test]
fn inter_all_toggles_decodes_lockstep_and_never_grows() {
    let (kf_bytes, kf) = keyed_reference();
    let (y, u, v) = source(2);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let refresh = RefreshControls::default();

    let (baseline, _) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &frame,
        &kf,
        None,
        None,
        &p,
        &refresh,
        &InterCodingOptions {
            intra_pick: true,
            auto_loop_filter: true,
            fitted_token_prob_updates: false,
        },
    )
    .expect("no-fit baseline");
    let (bytes, planes) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &frame,
        &kf,
        None,
        None,
        &p,
        &refresh,
        &InterCodingOptions {
            intra_pick: true,
            auto_loop_filter: true,
            fitted_token_prob_updates: true,
        },
    )
    .expect("all toggles");
    assert!(
        bytes.len() <= baseline.len(),
        "fitted pass must never grow the wire ({} > {})",
        bytes.len(),
        baseline.len()
    );

    let mut state = Vp8DecoderState::new();
    state.decode_frame(&kf_bytes).expect("keyframe");
    let dec = state.decode_frame(&bytes).expect("P-frame must decode");
    assert_eq!(dec.y, planes.y, "encoder/decoder luma lockstep");
    assert_eq!(dec.u, planes.u, "encoder/decoder U lockstep");
    assert_eq!(dec.v, planes.v, "encoder/decoder V lockstep");
}

// ───────────────────────── invisible anchor ─────────────────────────

#[test]
fn invisible_update_options_intra_pick_matches_dedicated_entry() {
    let (_, kf) = keyed_reference();
    let (y, u, v) = source(4);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (a, _) = encode_invisible_altref_update(&frame, &kf, Some(&kf), Some(&kf), &p)
        .expect("dedicated invisible");
    let (b, _) = encode_invisible_altref_update_with_coding_options(
        &frame,
        &kf,
        Some(&kf),
        Some(&kf),
        &p,
        &InterCodingOptions {
            intra_pick: true,
            ..Default::default()
        },
    )
    .expect("options invisible");
    assert_eq!(
        a, b,
        "intra_pick invisible update must reproduce encode_invisible_altref_update"
    );
}

#[test]
fn invisible_update_all_toggles_stays_invisible_and_updates_altref_only() {
    let (kf_bytes, kf) = keyed_reference();
    let (y, u, v) = source(4);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    let p = params();
    let (bytes, planes) = encode_invisible_altref_update_with_coding_options(
        &frame,
        &kf,
        Some(&kf),
        Some(&kf),
        &p,
        &InterCodingOptions {
            intra_pick: true,
            auto_loop_filter: true,
            fitted_token_prob_updates: true,
        },
    )
    .expect("all-toggle invisible");
    // §9.1 tag byte 0 bit 4 must be clear (show_frame = 0).
    assert_eq!(bytes[0] & 0x10, 0, "show_frame bit must be 0");

    let mut state = Vp8DecoderState::new();
    state.decode_frame(&kf_bytes).expect("keyframe");
    let dec = state.decode_frame(&bytes).expect("invisible must decode");
    assert_eq!(
        state.last_frame_shown(),
        Some(false),
        "decoder must report not-shown"
    );
    // The invisible frame's reconstruction (what landed in ALTREF) is
    // in decoder lockstep with the encoder's planes (64×64 aligned).
    assert_eq!(dec.y, planes.y, "invisible-frame reconstruction lockstep");

    // LAST was not perturbed (refresh_last = 0): a follow-up P-frame
    // encoded against the keyframe's LAST still decodes in lockstep.
    let (py, pu, pv) = source(4);
    let pframe = I420Frame::packed(W as u32, H as u32, &py, &pu, &pv);
    let (pbytes, pplanes) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &pframe,
        &kf,
        Some(&kf),
        Some(&planes),
        &p,
        &RefreshControls::default(),
        &InterCodingOptions::default(),
    )
    .expect("follow-up P-frame");
    let pdec = state.decode_frame(&pbytes).expect("P-frame must decode");
    assert_eq!(pdec.y, pplanes.y, "follow-up P-frame lockstep");
}
