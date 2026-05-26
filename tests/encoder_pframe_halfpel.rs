//! End-to-end I + P self-decode roundtrip for the round-144 §18.3
//! half-pixel refinement path
//! ([`oxideav_vp8::encode_p_frame_zero_mv`]'s
//! [`oxideav_vp8::half_pixel_refine_luma`] post-pass on the whole-pixel
//! search result).
//!
//! Encodes a 2-frame sequence where the P-frame source is the §18.3
//! six-tap synthesis of the I-frame source at a deliberate **half-pixel**
//! MV — a translation that is fundamentally unreachable from a whole-
//! pixel-only descent (any whole-pixel MV leaves a residue the §14
//! quantiser must absorb, capping PSNR). With the half-pixel refinement
//! the encoder should pick at least one half-pixel MV per affected MB
//! and produce a self-decode that bit-exactly recovers the source modulo
//! the §14 quantiser's residual rounding.
//!
//! What this pins:
//!
//!   * the §18.3 `filter_block_4x4` six-tap synthesis on the encoder's
//!     prediction path (now also exercised at non-zero fractional MV
//!     components, not just whole-pixel);
//!   * the §17.2 `write_mv` round-trip carrying a half-pixel
//!     differential (the `(mv & 7)` fractional bits propagate);
//!   * the §-non-normative RD trade picking a half-pixel NEWMV over
//!     ZEROMV when the residue savings cover the §17.2 bit cost.
//!
//! Black-box: the encoder's output is fed straight into the crate's own
//! [`oxideav_vp8::Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    decode_split_mv, default_mv_contexts, encode_keyframe_with_reconstruction,
    encode_p_frame_zero_mv, filter_block_4x4, filter_set_for_version, find_near_mvs, mv_ref_probs,
    read_inter_mode, read_mv, stored_luma_mv, BoolDecoder, I420Frame, InterMode, KeyframeParams,
    MbInfo, Mv, RefFrame, SignBias, Vp8CodedHeader, Vp8DecoderState, Vp8FrameHeader,
    HALF_PIXEL_STEP,
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

/// Build an I-frame luma plane with a non-degenerate 2D ramp + a
/// distinct high-contrast 16×16 feature square at `(fx, fy)`. The ramp
/// gives the §18.3 sixtap a non-zero gradient response at every half-
/// pixel position (so a horizontal-only half-pixel MV is distinguishable
/// from a diagonal half-pixel MV).
fn frame_with_feature(width: usize, height: usize, fx: usize, fy: usize) -> Vec<u8> {
    let mut y = vec![0u8; width * height];
    for r in 0..height {
        for c in 0..width {
            // 2D ramp: distinct slopes per axis.
            let bg = (40 + (r as i32) / 4 + (c as i32) / 6).clamp(0, 200);
            y[r * width + c] = bg as u8;
        }
    }
    for r in 0..16 {
        for c in 0..16 {
            let py = fy + r;
            let px = fx + c;
            if px < width && py < height {
                y[py * width + px] = 240;
            }
        }
    }
    y
}

/// Synthesise the P-frame source as the §18.3 six-tap prediction of the
/// I-frame plane at a global half-pixel MV. The result is the "exact"
/// pixel data a half-pixel-aware encoder should be able to reconstruct
/// with a zero residue (modulo §14 quantiser rounding).
fn shift_plane_by_half_pixel(src: &[u8], width: usize, height: usize, mv_quarter: Mv) -> Vec<u8> {
    let filters = filter_set_for_version(0).taps();
    let mv_eighth = stored_luma_mv(mv_quarter);
    let mut out = vec![0u8; width * height];
    for blk_y in (0..height).step_by(4) {
        for blk_x in (0..width).step_by(4) {
            let patch =
                filter_block_4x4(src, width, width, height, blk_x, blk_y, mv_eighth, filters);
            for r in 0..4 {
                for c in 0..4 {
                    let dy = blk_y + r;
                    let dx = blk_x + c;
                    if dy < height && dx < width {
                        out[dy * width + dx] = patch[r * 4 + c];
                    }
                }
            }
        }
    }
    out
}

/// Strip down the §11/§16 macroblock-mode walk to extract per-MB
/// `(InterMode, Mv)` from an encoded P-frame (every MB inter against
/// LAST, no segmentation, no token-prob updates, §17.2 mv_prob_update
/// F-gates all zero — the [`oxideav_vp8::encode_p_frame_zero_mv`]
/// convention).
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

/// A 2-frame I+P sequence whose P-frame source is the §18.3 sixtap
/// synthesis of the I-frame source at MV (row=0, col=+HALF_PIXEL_STEP).
/// The half-pixel-aware picker MUST emit at least one half-pixel NEWMV
/// (the only MV that recovers the +0.5 px shift with a zero residue),
/// and the self-decode Y-PSNR must clear a tighter threshold than the
/// whole-pixel-only round-143 floor (which had to absorb the half-pixel
/// shift through the §14 quantiser).
#[test]
fn i_plus_p_half_pixel_shift_emits_half_pixel_newmv_and_clears_44db() {
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (w / 2, h / 2);

    let y_i = frame_with_feature(w, h, 16, 16);
    // P-frame source = §18.3 sixtap of I at MV (0, +1/2 px). Apply only
    // inside the inner region — the outer 16-pixel ring uses the I-frame
    // values verbatim so the encoder's edge-replicated half-pixel halo
    // does not introduce uncontrolled SAD.
    let mut y_p = shift_plane_by_half_pixel(
        &y_i,
        w,
        h,
        Mv {
            row: 0,
            col: HALF_PIXEL_STEP,
        },
    );
    // Outer 8 px border = I-frame values verbatim so MB(0,*) / MB(*,0) /
    // MB(rows-1, *) / MB(*, cols-1) have ZEROMV as their winning MV
    // (their content didn't move). We only care that the INNER MBs (the
    // ones that fully landed inside the half-pixel-shifted region) pick
    // a half-pixel MV.
    for r in 0..h {
        for c in 0..w {
            let in_border = r < 8 || r >= h - 8 || c < 8 || c >= w - 8;
            if in_border {
                y_p[r * w + c] = y_i[r * w + c];
            }
        }
    }

    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];

    let frame_i = I420Frame::packed(w as u32, h as u32, &y_i, &u, &v);
    let frame_p = I420Frame::packed(w as u32, h as u32, &y_p, &u, &v);

    let params = KeyframeParams {
        y_ac_qi: 4, // low quantiser ⇒ §14 residue rounding is tiny ⇒ the
        // half-pixel-MV PSNR gap vs. whole-pixel-only is what dominates.
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };

    let (i_bytes, i_recon) =
        encode_keyframe_with_reconstruction(&frame_i, &params).expect("encode I");
    let (p_bytes, _) = encode_p_frame_zero_mv(&frame_p, &i_recon, &params).expect("encode P");

    let mut state = Vp8DecoderState::new();
    let _i_decoded = state.decode_frame(&i_bytes).expect("decode I");
    let p_decoded = state.decode_frame(&p_bytes).expect("decode P");

    let y_psnr = plane_psnr(&y_p, &p_decoded.y);
    eprintln!("half-pixel-shifted P-frame self-decode Y-PSNR: {y_psnr:.2} dB (yac_qi=4)");
    // A whole-pixel-only encoder absorbs the +0.5 px shift through the
    // §14 residue, capping Y-PSNR around the mid 30s on this content;
    // the half-pixel-aware encoder should clear 44 dB once the picker
    // is choosing half-pixel MVs.
    assert!(
        y_psnr >= 44.0,
        "half-pixel P-frame Y-plane self-decode PSNR {y_psnr:.2} dB < 44 dB"
    );

    // ---- Verify at least one half-pixel MV was emitted -----------
    //
    // The round-147 SPLITMV picker can match a half-pixel-shifted scene
    // by copying a NEWMV neighbour's half-pixel MV through LEFT4X4 /
    // ABOVE4X4 sub-block modes (the SPLITMV per-group search itself is
    // whole-pixel-only this round). Either path — a whole-MB half-pixel
    // NEWMV emission, OR a SPLITMV MB whose `split_mvs[15]` lands at
    // half-pixel resolution — proves the encoder is exercising the
    // §18.3 sixtap synthesis. A picker stuck on whole-pixel-only MVs
    // would NOT clear the 44 dB bar this test enforces.
    let mb_cols = w.div_ceil(16);
    let mb_rows = h.div_ceil(16);
    let modes = parse_p_frame_inter_modes(&p_bytes, mb_cols, mb_rows);
    let half_pixel_newmv_count = modes
        .iter()
        .filter(|(mode, mv)| *mode == InterMode::New && ((mv.row & 3) != 0 || (mv.col & 3) != 0))
        .count();
    let half_pixel_split_count = modes
        .iter()
        .filter(|(mode, mv)| *mode == InterMode::Split && ((mv.row & 3) != 0 || (mv.col & 3) != 0))
        .count();
    eprintln!(
        "P-frame inter-mode pick: {half_pixel_newmv_count}/{} half-pixel-grid NEWMV MBs, \
         {half_pixel_split_count}/{} half-pixel-grid SPLITMV MBs",
        modes.len(),
        modes.len()
    );
    assert!(
        half_pixel_newmv_count + half_pixel_split_count >= 1,
        "expected at least 1 half-pixel NEWMV-or-SPLITMV MB on a +1/2 px \
         shifted scene, got {} NEWMV / {} SPLITMV / {} MBs total",
        half_pixel_newmv_count,
        half_pixel_split_count,
        modes.len()
    );

    // Every NEWMV MB must still satisfy §17.1 [-1023, +1023].
    for (i, (mode, mv)) in modes.iter().enumerate() {
        if *mode == InterMode::New {
            assert!(
                (-1023..=1023).contains(&mv.row) && (-1023..=1023).contains(&mv.col),
                "MB {i}: §17.1 range violation, mv={mv:?}"
            );
        }
    }
}

// Silence the unused-import lint when the test binary's import block is
// pruned by future maintenance — `BoolDecoder` is used inside
// `parse_p_frame_inter_modes` via the `bool_decoder::read_bool` method,
// but rustc's unused-import check counts only direct uses.
#[allow(dead_code)]
fn _unused_imports_kept_alive(_d: BoolDecoder<'_>) {}
