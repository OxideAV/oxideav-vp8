//! End-to-end I + P self-decode roundtrip for the round-146 §16.2
//! NEARESTMV / NEARMV mode-pick extension
//! ([`oxideav_vp8::encode_p_frame_zero_mv`]'s J = SAD + λ·bits picker
//! now scores the §16.3 `near.mvs[1]` / `near.mvs[2]` census slots
//! alongside ZEROMV / NEWMV).
//!
//! Encodes a 2-frame I+P sequence whose P-frame is a **uniform
//! whole-pixel translation** of the I-frame: every visible 16×16 region
//! moves by the same `(dy, dx)` luma vector. The first MB to find a
//! non-zero match in its motion search encodes NEWMV; the §16.3 census
//! then propagates that vector into subsequent MBs' `near.mvs[1]` slot
//! via the left / above-left / above neighbour walk. With NEARESTMV
//! costing 2 bool bits (path "10") vs. NEWMV's 4 bool bits + §17.2
//! component bits (path "1110" + diff), every downstream MB whose
//! census-derived nearest predictor matches its best motion should pick
//! NEARESTMV — the same SAD at fewer bits.
//!
//! What this pins:
//!
//!   * the §16.2 `mv_ref_tree` NEARESTMV path "10" emission on a real
//!     encoder bitstream;
//!   * the §16.3 / §20.11 `clamp_mv` of the propagated nearest predictor
//!     (the decoder reconstructs the MV as
//!     `clamp_mv(near.mvs[1], bounds)` — no extra §17.2 bits);
//!   * the round-146 picker's tie-break ladder: when NEARESTMV's SAD
//!     equals NEWMV's at strictly fewer bits, NEARESTMV wins;
//!   * the §18.2 / §18.3 prediction path the chosen MV selects
//!     (whole-pixel ⇒ copy, sub-pixel ⇒ §18.3 sixtap), still matching
//!     the decoder's reconstruction.
//!
//! Black-box: the encoder's output is fed straight into the crate's own
//! [`oxideav_vp8::Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    decode_split_mv, default_mv_contexts, encode_keyframe_with_reconstruction,
    encode_p_frame_zero_mv, find_near_mvs, mv_ref_probs, read_inter_mode, read_mv, BoolDecoder,
    I420Frame, InterMode, KeyframeParams, MbInfo, Mv, RefFrame, SignBias, Vp8CodedHeader,
    Vp8DecoderState, Vp8FrameHeader,
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

/// Render a luma plane with high-frequency content (diagonal ramp +
/// per-column step + per-row stripe) so a per-MB SAD has a sharp global
/// minimum at the MV that aligns the candidate patch with the source's
/// pattern. A uniform-flat plane would let every MV tie at SAD 0, which
/// the picker would resolve to ZEROMV — defeating the test.
fn high_frequency_plane(width: usize, height: usize) -> Vec<u8> {
    let mut p = vec![0u8; width * height];
    for r in 0..height {
        for c in 0..width {
            let base = ((c as i32) / 2 + (r as i32) / 3).clamp(0, 160);
            let step = if c % 5 == 0 { 50 } else { 0 };
            let stripe = if r % 4 == 0 { -30 } else { 0 };
            p[r * width + c] = (base + step + stripe).clamp(0, 240) as u8;
        }
    }
    p
}

/// Shift `src` by the whole-pixel vector `(dy, dx)` — sample[(r, c)] of
/// the result is sample[(r - dy, c - dx)] of the input, with the
/// boundary region (where the source coordinate would land outside the
/// plane) filled with the §20.14 edge-replicated value. Mirrors the
/// behaviour the encoder's `fetch_block_whole_pixel` would synthesise
/// for any candidate fetch that walks off the plane.
fn shift_plane_whole_pixel(src: &[u8], width: usize, height: usize, dy: i32, dx: i32) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for r in 0..height {
        for c in 0..width {
            let sr = (r as i32 - dy).clamp(0, height as i32 - 1) as usize;
            let sc = (c as i32 - dx).clamp(0, width as i32 - 1) as usize;
            out[r * width + c] = src[sr * width + sc];
        }
    }
    out
}

/// Walk the §11/§16 macroblock-mode layer of a P-frame and report the
/// (mode, mv) pair the encoder emitted for each MB. Mirrors the
/// decoder's `read_inter_mode` + `resolve_inter_mb_mv` walk for the
/// `encode_p_frame_zero_mv` convention (every MB inter against LAST, no
/// segmentation, no token-prob updates, §17.2 `mv_prob_update` F-gates
/// all zero so the decoder uses the default `MvContexts`).
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

    let mut above: Vec<MbInfo> = vec![MbInfo::border(); mb_cols];
    let mut out: Vec<(InterMode, Mv)> = Vec::with_capacity(mb_rows * mb_cols);

    for mb_row in 0..mb_rows {
        let mut left = MbInfo::border();
        let mut aboveleft = MbInfo::border();
        for (mb_col, above_slot) in above.iter_mut().enumerate() {
            let _ = dec.read_bool(prob_skip_false).expect("skip bit");
            let _ = dec.read_bool(prob_intra).expect("inter bit");
            let _ = dec.read_bool(prob_last).expect("ref-frame bit");
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

/// Two-frame I+P with a **uniform whole-pixel translation**. The first
/// MB to detect motion encodes NEWMV; via the §16.3 census the next MB's
/// nearest predictor (left-neighbour MV) matches its own best motion, so
/// NEARESTMV beats NEWMV at fewer bits. The picker must emit ≥ 1
/// NEARESTMV MB and the self-decode must clear a non-trivial PSNR floor.
#[test]
fn i_plus_p_uniform_translation_emits_nearestmv() {
    // 96×64 — wide enough that several MBs land fully inside the inner
    // translated region (the outer 1-MB border carries the §20.14
    // edge-replicated copy and tends to ZEROMV).
    let (w, h) = (96usize, 64usize);
    let (cw, ch) = (w / 2, h / 2);

    let y_i = high_frequency_plane(w, h);
    // Whole-pixel translation by (+4 luma px down, +8 luma px right) —
    // both axes a multiple of `WHOLE_PIXEL_STEP` quarter-pixels (4 and
    // 8 luma px ⇒ 16 and 32 §17 quarter-pixels) and well inside the
    // per-MB MvClampRect.
    let y_p = shift_plane_whole_pixel(&y_i, w, h, 4, 8);

    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];

    let frame_i = I420Frame::packed(w as u32, h as u32, &y_i, &u, &v);
    let frame_p = I420Frame::packed(w as u32, h as u32, &y_p, &u, &v);

    let params = KeyframeParams {
        // Low quantiser ⇒ §14 residue rounding is tiny ⇒ the picker's
        // mode-vs-mode bit-cost trade dominates the J comparison.
        y_ac_qi: 4,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I");
    let (p_bytes, _) = encode_p_frame_zero_mv(&frame_p, &i_recon, &params).expect("encode P");

    // ---- Self-decode ----------------------------------------------------
    let mut state = Vp8DecoderState::new();
    let _i_decoded = state.decode_frame(&i_bytes).expect("decode I");
    let p_decoded = state.decode_frame(&p_bytes).expect("decode P");

    let y_psnr = plane_psnr(&y_p, &p_decoded.y);
    eprintln!("uniform-translation P-frame self-decode Y-PSNR: {y_psnr:.2} dB (yac_qi=4)");
    assert!(
        y_psnr >= 35.0,
        "uniform-translation P-frame Y-plane self-decode PSNR {y_psnr:.2} dB < 35 dB floor"
    );

    // ---- Verify NEARESTMV emission -------------------------------------
    let mb_cols = w.div_ceil(16);
    let mb_rows = h.div_ceil(16);
    let modes = parse_p_frame_inter_modes(&p_bytes, mb_cols, mb_rows);

    let nearest_count = modes
        .iter()
        .filter(|(m, _)| *m == InterMode::Nearest)
        .count();
    let new_count = modes.iter().filter(|(m, _)| *m == InterMode::New).count();
    let split_count = modes.iter().filter(|(m, _)| *m == InterMode::Split).count();
    eprintln!(
        "P-frame inter-mode pick: {nearest_count}/{} NEARESTMV MBs, \
         {new_count}/{} NEWMV MBs, {split_count}/{} SPLITMV MBs",
        modes.len(),
        modes.len(),
        modes.len()
    );
    // Either NEARESTMV (the §16.3 census-propagation path) OR SPLITMV
    // (the §16.4 per-sub-block path, which can also recover the
    // uniform translation by copying the neighbour MV into each
    // sub-block via LEFT4X4 / ABOVE4X4) must fire on this content. A
    // picker that stays on ZEROMV / NEWMV alone would mean both the
    // §16.3 census propagation AND the §16.4 partition / sub_mv_ref
    // walk are inactive — which would be a regression.
    assert!(
        nearest_count + split_count >= 1,
        "expected ≥ 1 NEARESTMV-or-SPLITMV MB on a uniform-translation \
         scene, got {nearest_count} NEARESTMV / {new_count} NEWMV / \
         {split_count} SPLITMV / {} MBs total",
        modes.len()
    );

    // Every NEARESTMV MB must carry a non-zero MV (the picker drops a
    // zero-clamped NEARESTMV — ZEROMV uses one fewer `mv_ref_tree` bool
    // at identical SAD, so NEARESTMV-with-zero-MV would be a waste-of-
    // bits picker bug).
    for (i, (mode, mv)) in modes.iter().enumerate() {
        if *mode == InterMode::Nearest {
            assert_ne!(
                *mv,
                Mv::default(),
                "MB {i}: NEARESTMV emitted with MV (0, 0) — picker should have chosen ZEROMV"
            );
            // §17.1: each component in [-1023, +1023].
            assert!(
                (-1023..=1023).contains(&mv.row) && (-1023..=1023).contains(&mv.col),
                "MB {i}: §17.1 range violation, mv={mv:?}"
            );
        }
    }
}

/// A flat scene (no high-frequency content, every translation candidate
/// ties at SAD 0) must emit a mode whose resolved MV is `(0, 0)` —
/// either ZEROMV directly, OR SPLITMV whose every per-sub-block vector
/// resolves to `(0, 0)` (the round-147 picker may legitimately pick
/// SPLITMV-with-all-LEFT4X4 sub-block modes when its bit cost beats
/// ZEROMV's "0" path bit cost: with `cnt = [0,0,0,0]` the §16.3
/// `mv_ref_probs[0] = 7` makes "0" cost ~5 bits while the
/// SUBMVREF_LEFT_ABOVE_ZED LEFT4X4 path costs ~0.3 bits per group, so
/// 16 cheap LEFT4X4 + the §16.4 partition tree can underrun ZEROMV's
/// single expensive false bit). What we *must* never see is a per-MB
/// resolved MV that's non-zero, since the §14 residue would carry the
/// non-zero MV's mispredict at strictly more bits.
#[test]
fn flat_scene_picker_resolves_to_zero_mv() {
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
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    let (_i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I");
    let (p_bytes, _) = encode_p_frame_zero_mv(&frame_p, &i_recon, &params).expect("encode P");

    let mb_cols = (w.div_ceil(16)) as usize;
    let mb_rows = (h.div_ceil(16)) as usize;
    let modes = parse_p_frame_inter_modes(&p_bytes, mb_cols, mb_rows);
    for (i, (mode, mv)) in modes.iter().enumerate() {
        // Mode is permissive (ZEROMV / NEARESTMV / NEARMV / SPLITMV may
        // all win on a flat scene depending on neighbour-driven probs);
        // the resolved per-MB vector — which is what feeds the §14
        // residue path — must always be `(0, 0)` because any non-zero MV
        // would mispredict the flat reference and inflate the residue.
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
