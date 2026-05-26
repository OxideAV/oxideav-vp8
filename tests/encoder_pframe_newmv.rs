//! End-to-end I + P self-decode roundtrip for the Phase-11 NEWMV path
//! ([`oxideav_vp8::encode_p_frame_zero_mv`]'s per-MB ZEROMV-vs-NEWMV
//! rate-distortion picker).
//!
//! Encodes a 2-frame sequence with a **deliberate whole-pixel
//! translation** between the I-frame source and the P-frame source: a
//! distinctive feature block is offset on the P-frame by exactly
//! `(dy_pixels, dx_pixels)` luma pixels (multiples of the §17.1
//! whole-pixel grid, well inside the §16.3 / §18.1 per-MB clamp
//! rectangle for every macroblock). The encoder's
//! [`oxideav_vp8::small_diamond_search_luma`] descent should converge
//! to a non-zero MV per affected MB and the §17 / §16.2 RD trade should
//! pick NEWMV over ZEROMV for at least one MB.
//!
//! What this pins:
//!
//!   * the §17.2 `read_mv` / `write_mv` round-trip on a real encoder
//!     bitstream (the decoder must re-read the same MV the encoder
//!     wrote and reproduce the same prediction);
//!   * the §16.2 `mv_ref_tree` NEWMV path "1110" emission and the
//!     §16.3 census-driven probability evolution at non-trivial MVs;
//!   * the §18.1 / §20.11 `clamp_mv` of the best predictor that NEWMV
//!     adds its differential to;
//!   * the §18.2 / §18.3 whole-pixel copy path the (whole-pixel) MV
//!     selects (no sub-pixel filter pass runs);
//!   * the self-decode PSNR target on a translated scene the §14
//!     quantiser cannot easily absorb at MV (0, 0) — i.e. a scene
//!     where the picker MUST find non-zero MVs to clear the PSNR floor.
//!
//! Black-box: the encoder's output is fed straight into the crate's own
//! [`oxideav_vp8::Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    decode_split_mv, default_mv_contexts, encode_keyframe_with_reconstruction,
    encode_p_frame_zero_mv, find_near_mvs, mv_ref_probs, read_inter_mode, read_mv, BoolDecoder,
    I420Frame, InterMode, KeyframeParams, MbInfo, Mv, RefFrame, SignBias, Vp8CodedHeader,
    Vp8DecoderState, Vp8FrameHeader,
};

/// Build a 64×64 I420 frame: smooth diagonal background + one distinct
/// 16×16 high-contrast feature square placed at `(fx, fy)` luma pixels.
/// The feature dominates the per-MB SAD inside its enclosing macroblock
/// so the motion search has a clean global minimum at the MV that
/// aligns the candidate patch's feature with the source's feature.
fn frame_with_feature(fx: usize, fy: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (32usize, 32usize);

    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // Slow diagonal background; the feature dominates the per-MB
            // SAD anyway.
            let bg = 70 + (r as i16) / 8 + (c as i16) / 8;
            y[r * w + c] = bg.clamp(0, 200) as u8;
        }
    }
    // 16×16 brighter square at (fx, fy).
    for r in 0..16 {
        for c in 0..16 {
            let py = fy + r;
            let px = fx + c;
            if px < w && py < h {
                y[py * w + px] = 240;
            }
        }
    }
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    (y, u, v)
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

/// Re-walk the §11/§16 macroblock mode layer of a P-frame the encoder
/// emitted with `prob_intra = 255`, `prob_last = 255`, and the §17.2
/// MV_UPDATE block's F-gates all zero (the
/// [`oxideav_vp8::encode_p_frame_zero_mv`] convention). Returns the
/// per-MB `(InterMode, Mv)` the encoder wrote, in raster order.
///
/// This is a stripped-down mirror of the decoder's §11/§16/§17 walk
/// that doesn't require the full state-machine: every MB is inter
/// against LAST, no segmentation, no segment-id, no skip-coeff context
/// (we read the bit but ignore its value), no intra-mode probabilities.
fn parse_p_frame_inter_modes(bytes: &[u8], mb_cols: usize, mb_rows: usize) -> Vec<(InterMode, Mv)> {
    let header = Vp8FrameHeader::parse(bytes).expect("parse P-frame header");
    let off = header.header_bytes_consumed;
    let size = header.first_partition_size as usize;
    let first = &bytes[off..off + size];
    let (coded, mut dec) =
        Vp8CodedHeader::parse_with_decoder(first, false).expect("parse coded header");

    let prob_skip_false = coded.prob_skip_false.expect("prob_skip_false present");
    let prob_intra = coded.prob_intra.expect("prob_intra present");
    let prob_last = coded.prob_last.expect("prob_last present");
    let mv_contexts = default_mv_contexts();
    // §17.2 mv_prob_update was emitted with every F-gate zero, so the
    // decoder reads against the defaults. The header parser may also
    // surface a `mv_contexts` field; in the encoder's hand-crafted
    // emission we know it's the default.

    let mut above: Vec<MbInfo> = vec![MbInfo::border(); mb_cols];
    let mut out: Vec<(InterMode, Mv)> = Vec::with_capacity(mb_rows * mb_cols);

    for mb_row in 0..mb_rows {
        let mut left = MbInfo::border();
        let mut aboveleft = MbInfo::border();
        for (mb_col, above_slot) in above.iter_mut().enumerate() {
            // 1. mb_skip_coeff (we don't care about the value here).
            let _ = dec.read_bool(prob_skip_false).expect("skip bit");
            // 2. is_inter_mb — every MB is inter (prob_intra = 255).
            let _ = dec.read_bool(prob_intra).expect("inter bit");
            // 3. ref_frame selector — LAST is the "false" branch.
            let _ = dec.read_bool(prob_last).expect("ref-frame bit");
            // 4. inter-mode tree walk against the §16.3 census.
            let near = find_near_mvs(
                above_slot,
                &left,
                &aboveleft,
                RefFrame::Last,
                SignBias::default(),
            );
            let probs = mv_ref_probs(&near.cnt);
            let mode = read_inter_mode(&mut dec, &probs).expect("inter mode");
            let bounds = oxideav_vp8::MvClampRect::for_mb(mb_col, mb_row, mb_cols, mb_rows);
            let (mv, cur) = match mode {
                InterMode::Zero => (
                    Mv::default(),
                    MbInfo {
                        ref_frame: Some(RefFrame::Last),
                        mv: Mv::default(),
                        is_split: false,
                        split_mvs: None,
                    },
                ),
                InterMode::Nearest => {
                    let mv = oxideav_vp8::clamp_mv(near.mvs[1], &bounds);
                    (
                        mv,
                        MbInfo {
                            ref_frame: Some(RefFrame::Last),
                            mv,
                            is_split: false,
                            split_mvs: None,
                        },
                    )
                }
                InterMode::Near => {
                    let mv = oxideav_vp8::clamp_mv(near.mvs[2], &bounds);
                    (
                        mv,
                        MbInfo {
                            ref_frame: Some(RefFrame::Last),
                            mv,
                            is_split: false,
                            split_mvs: None,
                        },
                    )
                }
                InterMode::New => {
                    let best = oxideav_vp8::clamp_mv(near.mvs[0], &bounds);
                    let diff = read_mv(&mut dec, &mv_contexts).expect("NEWMV diff");
                    let mv = Mv {
                        row: best.row.wrapping_add(diff.row),
                        col: best.col.wrapping_add(diff.col),
                    };
                    (
                        mv,
                        MbInfo {
                            ref_frame: Some(RefFrame::Last),
                            mv,
                            is_split: false,
                            split_mvs: None,
                        },
                    )
                }
                InterMode::Split => {
                    let best = oxideav_vp8::clamp_mv(near.mvs[0], &bounds);
                    let split = decode_split_mv(&mut dec, above_slot, &left, best, &mv_contexts)
                        .expect("SPLITMV decode");
                    (
                        split.split_mvs[15],
                        MbInfo {
                            ref_frame: Some(RefFrame::Last),
                            mv: split.split_mvs[15],
                            is_split: true,
                            split_mvs: Some(split.split_mvs),
                        },
                    )
                }
            };
            out.push((mode, mv));
            aboveleft = *above_slot;
            left = cur;
            *above_slot = cur;
        }
    }
    out
}

/// Two-frame I+P with a clean +4 / +4 whole-luma-pixel translation of a
/// 16×16 feature square. The encoder MUST pick at least one NEWMV MB
/// (the feature alignment is impossible with MV (0, 0)) and the
/// self-decode whole-frame PSNR MUST clear the round's 30 dB bar.
#[test]
fn i_plus_p_translated_feature_emits_newmv_and_clears_30db() {
    let (w, h) = (64u32, 64u32);
    // I-frame: feature at MB(1, 1)'s top-left (16, 16).
    // P-frame: feature translated by +4 luma pixels both axes (the MV
    // that aligns the P-frame's MB(1, 1) source patch with the I-frame
    // is +4 row, +4 col WHOLE pixels = +16 row, +16 col §17 quarter-
    // pixel units).
    let (yi, ui, vi) = frame_with_feature(16, 16);
    let (yp, up, vp) = frame_with_feature(20, 20);
    let frame_i = I420Frame::packed(w, h, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(w, h, &yp, &up, &vp);

    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };

    let (i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I");
    let (p_bytes, _) = encode_p_frame_zero_mv(&frame_p, &i_recon, &params).expect("encode P");

    // ---- Self-decode through the stateful driver ------------------------
    let mut state = Vp8DecoderState::new();
    let _i_decoded = state.decode_frame(&i_bytes).expect("decode I");
    let p_decoded = state.decode_frame(&p_bytes).expect("decode P");
    assert_eq!(p_decoded.width, w);
    assert_eq!(p_decoded.height, h);

    let y_psnr = plane_psnr(&yp, &p_decoded.y);
    eprintln!(
        "translated-feature P-frame self-decode Y-PSNR: {y_psnr:.2} dB \
         (chroma plane is uniform — skip)"
    );
    assert!(
        y_psnr >= 30.0,
        "translated-feature P-frame Y-plane self-decode PSNR {y_psnr:.2} dB < 30 dB floor"
    );

    // ---- Verify NEWMV emission by re-parsing the macroblock modes ------
    let mb_cols = (w.div_ceil(16)) as usize;
    let mb_rows = (h.div_ceil(16)) as usize;
    let modes = parse_p_frame_inter_modes(&p_bytes, mb_cols, mb_rows);
    assert_eq!(modes.len(), mb_cols * mb_rows);

    let newmv_count = modes.iter().filter(|(m, _)| *m == InterMode::New).count();
    let split_count = modes.iter().filter(|(m, _)| *m == InterMode::Split).count();
    eprintln!(
        "P-frame inter-mode pick: {newmv_count}/{} NEWMV MBs, \
         {split_count}/{} SPLITMV MBs",
        modes.len(),
        modes.len()
    );
    // Either NEWMV (the whole-MB §17 motion-search path) or SPLITMV
    // (the round-147 §16.4 per-sub-block path, which can also recover
    // a translated feature via per-group whole-pixel searches) must
    // fire. A picker stuck on ZEROMV / NEARESTMV / NEARMV alone would
    // mean both the §17 search and the §16.4 walk are inactive.
    assert!(
        newmv_count + split_count >= 1,
        "expected ≥ 1 NEWMV-or-SPLITMV MB on a translated-feature scene, \
         got {newmv_count} NEWMV / {split_count} SPLITMV / {} MBs total",
        modes.len()
    );

    // Every NEWMV MB must carry a non-zero MV (a NEWMV emission with MV
    // (0, 0) would be a wasted-bits picker bug — the ZEROMV path costs
    // strictly fewer bits at identical SAD).
    for (i, (mode, mv)) in modes.iter().enumerate() {
        if *mode == InterMode::New {
            assert_ne!(
                *mv,
                Mv::default(),
                "MB {i}: NEWMV emitted with MV (0, 0) — picker should have chosen ZEROMV"
            );
            // §17.1: each component in [-1023, +1023].
            assert!(
                (-1023..=1023).contains(&mv.row) && (-1023..=1023).contains(&mv.col),
                "MB {i}: §17.1 range violation, mv={mv:?}"
            );
        }
    }
}

/// A flat scene (no motion possible) must still emit a decodable P-frame
/// whose per-MB resolved MV is always `(0, 0)` — the picker may legitimately
/// pick ZEROMV or SPLITMV-with-all-zero-sub-blocks (which can underrun
/// ZEROMV's bit cost when `cnt = [0,0,0,0]` makes `mv_ref_probs[0] = 7`
/// turn the "0" path into a ~5-bit emission, see
/// `flat_scene_picker_resolves_to_zero_mv` in
/// `tests/encoder_pframe_nearestmv.rs`). What the picker must NEVER do is
/// emit a non-zero resolved MV on a flat scene — that would inflate the
/// §14 residue.
#[test]
fn flat_scene_picker_stays_on_zeromv_path() {
    let (w, h) = (64u32, 64u32);
    let (cw, ch) = ((w / 2) as usize, (h / 2) as usize);
    let y_flat = vec![128u8; (w * h) as usize];
    let u_flat = vec![128u8; cw * ch];
    let v_flat = vec![128u8; cw * ch];

    let frame_i = I420Frame::packed(w, h, &y_flat, &u_flat, &v_flat);
    let frame_p = I420Frame::packed(w, h, &y_flat, &u_flat, &v_flat);

    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };

    let (i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I");
    let (p_bytes, _) = encode_p_frame_zero_mv(&frame_p, &i_recon, &params).expect("encode P");

    let mut state = Vp8DecoderState::new();
    let _ = state.decode_frame(&i_bytes).expect("decode I");
    let p_decoded = state.decode_frame(&p_bytes).expect("decode P");
    let y_psnr = plane_psnr(&y_flat, &p_decoded.y);
    assert!(
        y_psnr.is_infinite() || y_psnr >= 40.0,
        "flat-scene P-frame Y PSNR {y_psnr:.2} dB < 40 dB"
    );

    // The resolved per-MB MV must always be `(0, 0)` on a flat scene
    // (the §14 residue would inflate under any non-zero MV); the mode
    // may be ZEROMV or SPLITMV-with-zero-sub-blocks depending on the
    // §16.2 / §16.4 bit-cost trade.
    let mb_cols = (w.div_ceil(16)) as usize;
    let mb_rows = (h.div_ceil(16)) as usize;
    let modes = parse_p_frame_inter_modes(&p_bytes, mb_cols, mb_rows);
    for (i, (mode, mv)) in modes.iter().enumerate() {
        assert_eq!(
            *mv,
            Mv::default(),
            "MB {i}: flat-scene picker emitted {mode:?} with non-zero \
             resolved MV {mv:?} — the §14 residue would inflate"
        );
    }
}

// Silence the unused-import lint when the test binary's import block is
// pruned by future maintenance — `BoolDecoder` is used inside
// `parse_p_frame_inter_modes` via the `bool_decoder::read_bool` method,
// but rustc's unused-import check counts only direct uses.
#[allow(dead_code)]
fn _unused_imports_kept_alive(_d: BoolDecoder<'_>) {}
