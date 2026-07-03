//! End-to-end coverage of the §9.1 / §9.7 **invisible altref-update
//! frame** ([`oxideav_vp8::encode_invisible_altref_update`]) — the
//! building block of B-frame-less GOLDEN / ALTREF management.
//!
//! Scenario (64×64, three encoded frames):
//!
//!   * Frame 0 (K, visible): a textured base scene. The keyframe
//!     reconstruction seeds LAST = GOLDEN = ALTREF.
//!   * Frame 1 (invisible altref update): a *future* picture — the
//!     scene translated by a few pixels plus a new detail block. The
//!     wire carries `show_frame = 0` and the §9.7 ladder
//!     `refresh_alternate_frame = 1, refresh_last = 0`, so a decoder
//!     installs the reconstruction **only** into ALTREF and drops the
//!     picture from presentation.
//!   * Frame 2 (P, visible): source equal to the future picture. With
//!     the altref anchor in place the per-MB §16.2 `ref_frame`
//!     selector can predict every MB from ALTREF at near-zero
//!     residual; without the update the same frame must be coded off
//!     the keyframe.
//!
//! What this pins:
//!
//!   * the emitted invisible frame differs from its visible twin in
//!     exactly the §9.1 bit-4 of byte 0 (a single-bit rewrite — every
//!     partition byte is identical);
//!   * [`Vp8DecoderState::last_frame_shown`] reads `Some(false)` after
//!     the invisible frame and `Some(true)` after both visible ones;
//!   * the decoder's LAST slot is **not** perturbed by the invisible
//!     frame (frame 2 was encoded against the keyframe's LAST, and its
//!     self-decode is pixel-exact through the stateful decoder);
//!   * the ALTREF anchor pays for itself: frame 2's bytes with the
//!     altref update present are smaller than the same source encoded
//!     without it;
//!   * full encoder↔decoder pixel lockstep on all three frames.
//!
//! Black-box: the encoder's output feeds the crate's own
//! [`Vp8DecoderState`]; no external codec is consulted.

use oxideav_vp8::{
    encode_invisible_altref_update, encode_keyframe_with_reconstruction, encode_p_frame_multi_ref,
    I420Frame, KeyframeParams, KeyframePlanes, Vp8DecoderState, Vp8FrameHeader,
};

const W: usize = 64;
const H: usize = 64;

/// Textured base scene: a diagonal luma gradient with an 8-pixel
/// checker overlay, chroma at mid-range with a slow ramp.
fn base_scene() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let g = (r * 2 + c) as i32 & 0xff;
            let checker = if ((r / 8) + (c / 8)) & 1 == 0 { 24 } else { 0 };
            y[r * W + c] = (g / 2 + 64 + checker) as u8;
        }
    }
    let (cw, ch) = (W / 2, H / 2);
    let mut u = vec![128u8; cw * ch];
    let mut v = vec![128u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (110 + r) as u8;
            v[r * cw + c] = (140 - c as i32).max(0) as u8;
        }
    }
    (y, u, v)
}

/// The "future" picture: the base scene translated by (+4, +2) pixels
/// (edge-clamped) with a bright 16×16 detail block dropped in at MB
/// (2, 2) — content a LAST-only encoder must spend real bits on.
fn future_scene() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (by, bu, bv) = base_scene();
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let sr = r.saturating_sub(2).min(H - 1);
            let sc = c.saturating_sub(4).min(W - 1);
            y[r * W + c] = by[sr * W + sc];
        }
    }
    // New detail: high-contrast horizontal stripes in one MB.
    for r in 32..48 {
        for c in 32..48 {
            y[r * W + c] = if r & 2 == 0 { 40 } else { 216 };
        }
    }
    let (cw, ch) = (W / 2, H / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            let sr = r.saturating_sub(1).min(ch - 1);
            let sc = c.saturating_sub(2).min(cw - 1);
            u[r * cw + c] = bu[sr * cw + sc];
            v[r * cw + c] = bv[sr * cw + sc];
        }
    }
    (y, u, v)
}

/// Compare a decoded visible frame against the encoder's own
/// reconstruction planes (64×64 is macroblock-aligned, so the visible
/// crop is the whole plane).
fn assert_lockstep(decoded: &oxideav_vp8::Vp8Frame, recon: &KeyframePlanes, what: &str) {
    assert_eq!(decoded.width as usize, W);
    assert_eq!(decoded.height as usize, H);
    assert_eq!(
        &decoded.y[..],
        &recon.y[..W * H],
        "{what}: Y plane mismatch"
    );
    let (cw, ch) = (W / 2, H / 2);
    assert_eq!(
        &decoded.u[..],
        &recon.u[..cw * ch],
        "{what}: U plane mismatch"
    );
    assert_eq!(
        &decoded.v[..],
        &recon.v[..cw * ch],
        "{what}: V plane mismatch"
    );
}

#[test]
fn invisible_altref_update_round_trips_and_pays_off() {
    let (ky, ku, kv) = base_scene();
    let (fy, fu, fv) = future_scene();
    let kf_src = I420Frame::packed(W as u32, H as u32, &ky, &ku, &kv);
    let fut_src = I420Frame::packed(W as u32, H as u32, &fy, &fu, &fv);

    let params = KeyframeParams {
        y_ac_qi: 40,
        ..KeyframeParams::default()
    };

    // ---- Frame 0: visible keyframe ---------------------------------
    let (kf_bytes, kf_recon) =
        encode_keyframe_with_reconstruction(&kf_src, &params).expect("keyframe encodes");

    // ---- Frame 1: invisible altref update (the future picture) -----
    let (alt_bytes, alt_recon) = encode_invisible_altref_update(
        &fut_src,
        &kf_recon,
        Some(&kf_recon),
        Some(&kf_recon),
        &params,
    )
    .expect("invisible altref update encodes");

    // The wire says: interframe, not for display.
    let hdr = Vp8FrameHeader::parse(&alt_bytes).expect("altref update header parses");
    assert!(!hdr.key_frame, "altref update must be an interframe");
    assert!(!hdr.show_frame, "altref update must carry show_frame = 0");

    // Single-bit property: the invisible wire differs from its visible
    // twin in exactly §9.1 bit-4 of byte 0.
    {
        let refresh = oxideav_vp8::RefreshControls {
            refresh_alternate_frame: true,
            refresh_last: false,
            ..oxideav_vp8::RefreshControls::default()
        };
        let (visible_twin, _) = oxideav_vp8::encode_p_frame_multi_ref_with_refresh_and_intra_pick(
            &fut_src,
            &kf_recon,
            Some(&kf_recon),
            Some(&kf_recon),
            &params,
            &refresh,
        )
        .expect("visible twin encodes");
        assert_eq!(alt_bytes.len(), visible_twin.len());
        assert_eq!(alt_bytes[0] ^ visible_twin[0], 0x10, "only bit 4 differs");
        assert_eq!(&alt_bytes[1..], &visible_twin[1..], "payload identical");
    }

    // ---- Frame 2: visible P-frame, source == the future picture ----
    // With the anchor: ALTREF holds the future picture's reconstruction.
    let (p_with, p_with_recon) = encode_p_frame_multi_ref(
        &fut_src,
        &kf_recon, // LAST is still the keyframe (refresh_last was 0)
        Some(&kf_recon),
        Some(&alt_recon),
        &params,
    )
    .expect("P-frame with altref anchor encodes");

    // Without the anchor: all three slots still hold the keyframe.
    let (p_without, _) = encode_p_frame_multi_ref(
        &fut_src,
        &kf_recon,
        Some(&kf_recon),
        Some(&kf_recon),
        &params,
    )
    .expect("P-frame without altref anchor encodes");

    assert!(
        p_with.len() < p_without.len(),
        "the altref anchor must pay for itself: {} (with) vs {} (without) bytes",
        p_with.len(),
        p_without.len()
    );

    // ---- Stateful decode: K → invisible altref → P ------------------
    let mut dec = Vp8DecoderState::new();
    assert_eq!(dec.last_frame_shown(), None, "no frame decoded yet");

    let d0 = dec.decode_frame(&kf_bytes).expect("keyframe decodes");
    assert_eq!(dec.last_frame_shown(), Some(true));
    assert_lockstep(&d0, &kf_recon, "keyframe");

    let d1 = dec.decode_frame(&alt_bytes).expect("altref update decodes");
    assert_eq!(
        dec.last_frame_shown(),
        Some(false),
        "invisible frame must decode as not-for-display"
    );
    // The dropped picture is still the altref reconstruction (a player
    // discards it; the reference side effect is what matters).
    assert_lockstep(&d1, &alt_recon, "altref update");

    // Frame 2 was encoded against LAST = keyframe reconstruction. Its
    // pixel-exact decode proves the invisible frame left the decoder's
    // LAST slot untouched (refresh_last = 0 honoured) and installed the
    // anchor into ALTREF (the frame's altref-predicted MBs reconstruct
    // exactly).
    let d2 = dec.decode_frame(&p_with).expect("anchored P-frame decodes");
    assert_eq!(dec.last_frame_shown(), Some(true));
    assert_lockstep(&d2, &p_with_recon, "anchored P-frame");
}
