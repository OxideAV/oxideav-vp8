//! Round-387 regression: `pick_intra = true` with an intra MB
//! preceding a SPLITMV MB in raster order.
//!
//! The inter walk kept a per-MB `split_candidates` table for the §16.4
//! wire emit. The intra branch pushed a `None` slot AND fell through to
//! the shared push site, double-pushing for every intra MB — from the
//! frame's first intra pick onward every MB read its left neighbour's
//! slot. A later SPLITMV MB then found `None` (panic) or, worse, a
//! *different MB's* split candidate (silent §16.4 wire corruption:
//! encoder reconstruction and emitted bytes disagree).
//!
//! This source (flat band + gradient + busy texture, drifted 3 px)
//! reliably makes the picker choose both intra MBs and SPLITMV MBs in
//! the same frame; before the fix the encode below panicked.
//!
//! Black-box: encoder output feeds the crate's own decoder only.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_with_refresh_and_intra_pick,
    I420Frame, KeyframeParams, RefreshControls, Vp8DecoderState,
};

const W: usize = 96;
const H: usize = 96;

fn source(dx: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let cc = c + dx;
            let v = if r < H / 4 {
                200
            } else if r < H / 2 {
                (80 + (cc / 2) % 100) as u8
            } else {
                let t = ((r * 13 + cc * 7) % 61) as i32 * 3 - 90;
                let e = if (cc / 4) % 2 == 0 { 50 } else { -50 };
                (128 + t / 2 + e).clamp(0, 255) as u8
            };
            y[r * W + c] = v;
        }
    }
    let (cw, ch) = (W / 2, H / 2);
    (y, vec![118u8; cw * ch], vec![132u8; cw * ch])
}

#[test]
fn intra_mb_before_splitmv_mb_encodes_and_decodes_lockstep() {
    let params = KeyframeParams {
        y_ac_qi: 44,
        loop_filter_level: 16,
        ..KeyframeParams::default()
    };
    let (ky, ku, kv) = source(0);
    let kf_src = I420Frame::packed(W as u32, H as u32, &ky, &ku, &kv);
    let (kf_bytes, kf) = encode_keyframe_with_reconstruction(&kf_src, &params).expect("keyframe");

    let (py, pu, pv) = source(3);
    let p_src = I420Frame::packed(W as u32, H as u32, &py, &pu, &pv);
    let (p_bytes, p_planes) = encode_p_frame_multi_ref_with_refresh_and_intra_pick(
        &p_src,
        &kf,
        None,
        None,
        &params,
        &RefreshControls::default(),
    )
    .expect("intra-pick P-frame with SPLITMV content must encode");

    // Full stateful self-decode in pixel lockstep — the emitted §16.4
    // split data must be the candidates the reconstruction used.
    let mut dec = Vp8DecoderState::new();
    dec.decode_frame(&kf_bytes).expect("keyframe decodes");
    let dp = dec.decode_frame(&p_bytes).expect("P-frame decodes");
    assert_eq!(dp.y, p_planes.y, "luma lockstep");
    assert_eq!(dp.u, p_planes.u, "U lockstep");
    assert_eq!(dp.v, p_planes.v, "V lockstep");
}
