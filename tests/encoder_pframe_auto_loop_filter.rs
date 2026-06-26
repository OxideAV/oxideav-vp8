//! RD-driven §9.4 `loop_filter_level` auto-selection on the inter
//! (P-frame) path (round 373).
//!
//! `encode_p_frame_multi_ref_auto_loop_filter` ignores
//! `params.loop_filter_level` and lets
//! [`oxideav_vp8::loop_filter::select_filter_level`] choose the level that
//! minimises the post-§15 reconstruction SSD of the encoder's own picture
//! against the source, writing the winner to the §9.4 header. The inter
//! filter consults the per-MB §9.4 reference / mode deltas, so the
//! selector runs the identical inter §15 pass for each candidate — these
//! tests pin that the chosen level reaches the wire (byte-exact
//! encoder↔decoder lockstep through a stateful decode of I then P) and
//! that it never increases distortion versus an unfiltered P-frame.

use oxideav_vp8::loop_filter::{reconstruction_ssd, SourcePlanes};
use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_auto_loop_filter,
    encode_p_frame_multi_ref_with_refresh_and_intra_pick, I420Frame, KeyframeParams,
    RefreshControls, Vp8DecoderState,
};

/// The two I420 source frames of a deterministic 48×48 pair.
struct FramePair {
    y0: Vec<u8>,
    u0: Vec<u8>,
    v0: Vec<u8>,
    y1: Vec<u8>,
    u1: Vec<u8>,
    v1: Vec<u8>,
}

/// Two deterministic 48×48 I420 frames whose per-MB plateaus are
/// re-shuffled between them, so the P-frame carries a large blocky inter
/// residual whose coded coefficients give the §15 filter genuine MB /
/// sub-block edge error to reduce.
fn frame_pair() -> FramePair {
    let (w, h) = (48usize, 48usize);
    let (cw, ch) = (24usize, 24usize);
    let mut y0 = vec![0u8; w * h];
    let mut y1 = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let mb = ((r / 16) * 9 + (c / 16) * 17) % 200;
            let ramp = ((r % 16) + (c % 16)) / 5;
            y0[r * w + c] = (20 + mb + ramp).min(255) as u8;
            // Frame 1: re-shuffle the per-MB plateaus so each MB carries a
            // large, blocky inter residual — the coded coefficients give
            // the §15 inter pass real MB / sub-block edge error to reduce.
            let mb1 = ((r / 16) * 31 + (c / 16) * 53) % 220;
            y1[r * w + c] = (18 + mb1 + ramp).min(255) as u8;
        }
    }
    let mut u0 = vec![0u8; cw * ch];
    let mut v0 = vec![0u8; cw * ch];
    let mut u1 = vec![0u8; cw * ch];
    let mut v1 = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            let mb = ((r / 8) * 7 + (c / 8) * 3) % 80;
            let mb1 = ((r / 8) * 23 + (c / 8) * 13) % 90;
            u0[r * cw + c] = (110 + mb) as u8;
            v0[r * cw + c] = (140 + mb) as u8;
            u1[r * cw + c] = (100 + mb1) as u8;
            v1[r * cw + c] = (130 + mb1) as u8;
        }
    }
    FramePair {
        y0,
        u0,
        v0,
        y1,
        u1,
        v1,
    }
}

#[test]
fn p_frame_auto_lf_pixel_lockstep() {
    let (w, h) = (48u32, 48u32);
    let fp = frame_pair();
    let frame_i = I420Frame::packed(w, h, &fp.y0, &fp.u0, &fp.v0);
    let frame_p = I420Frame::packed(w, h, &fp.y1, &fp.u1, &fp.v1);

    // Coarse quantiser → blocky → the §15 filter genuinely helps.
    let params = KeyframeParams {
        y_ac_qi: 96,
        ..KeyframeParams::default()
    };

    let (i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I");
    let (p_bytes, p_recon) =
        encode_p_frame_multi_ref_auto_loop_filter(&frame_p, &i_recon, None, None, &params)
            .expect("encode P auto-LF");

    let mut state = Vp8DecoderState::new();
    let _ = state.decode_frame(&i_bytes).expect("decode I");
    let p_dec = state.decode_frame(&p_bytes).expect("decode P");

    // The decoder's self-decoded P picture must byte-equal the encoder's
    // own post-§15 reconstruction — the auto-selected level reached the
    // wire and both ends ran the identical §15 inter pass.
    assert_eq!(
        p_dec.y.as_slice(),
        p_recon.y.as_slice(),
        "P-frame Y encoder-recon vs decoder-recon"
    );
    assert_eq!(
        p_dec.u.as_slice(),
        p_recon.u.as_slice(),
        "P-frame U encoder-recon vs decoder-recon"
    );
    assert_eq!(
        p_dec.v.as_slice(),
        p_recon.v.as_slice(),
        "P-frame V encoder-recon vs decoder-recon"
    );
}

#[test]
fn p_frame_auto_lf_never_worse_than_unfiltered() {
    let (w, h) = (48u32, 48u32);
    let fp = frame_pair();
    let frame_i = I420Frame::packed(w, h, &fp.y0, &fp.u0, &fp.v0);
    let frame_p = I420Frame::packed(w, h, &fp.y1, &fp.u1, &fp.v1);

    let qi = 100;
    let unfiltered_params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        ..KeyframeParams::default()
    };
    let auto_params = KeyframeParams {
        y_ac_qi: qi,
        ..KeyframeParams::default()
    };

    // Shared I-frame reference (unfiltered, lf=0) so both P encodes
    // predict from the identical LAST picture and only the P-frame's own
    // §15 choice differs.
    let (_i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &unfiltered_params).expect("encode I");

    // Baseline must take the IDENTICAL mode-decision path as the auto
    // entry (intra-pick on, default refresh) so the only difference is the
    // §15 level: lf=0 here vs the RD-selected level in the auto encode.
    let (_p0, p_recon0) = encode_p_frame_multi_ref_with_refresh_and_intra_pick(
        &frame_p,
        &i_recon,
        None,
        None,
        &unfiltered_params,
        &RefreshControls::default(),
    )
    .expect("encode P lf=0");
    let (_pa, p_recon_auto) =
        encode_p_frame_multi_ref_auto_loop_filter(&frame_p, &i_recon, None, None, &auto_params)
            .expect("encode P auto-LF");

    let sp = SourcePlanes {
        width: w as usize,
        height: h as usize,
        y: &fp.y1,
        u: &fp.u1,
        v: &fp.v1,
        y_stride: w as usize,
        uv_stride: w.div_ceil(2) as usize,
    };
    let ssd0 = reconstruction_ssd(&p_recon0, &sp);
    let ssd_auto = reconstruction_ssd(&p_recon_auto, &sp);
    eprintln!("P-frame unfiltered SSD {ssd0} / auto-LF SSD {ssd_auto}");
    assert!(
        ssd_auto <= ssd0,
        "P-frame auto-LF distortion {ssd_auto} must not exceed unfiltered {ssd0}"
    );
}
