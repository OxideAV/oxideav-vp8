//! End-to-end I + P self-decode roundtrip for the round-147 §16.4
//! SPLITMV picker extension
//! ([`oxideav_vp8::encode_p_frame_zero_mv`]'s J = SAD + λ·bits picker
//! now scores the four §16.4 partition shapes (TopBottom / LeftRight /
//! Quarters / Mv16) with a per-group whole-pixel sub-block diamond
//! search and a §16.4 `sub_mv_ref_tree` mode pick).
//!
//! Encodes a 2-frame I+P sequence whose P-frame is a **divergent
//! per-sub-MB translation**: the source content moves differently in
//! the four 2×2 sub-MB quadrants of each affected macroblock. A
//! whole-MB MV cannot align all four quadrants simultaneously, so the
//! per-quadrant SAD floor is non-trivial under any single whole-MB
//! ZEROMV / NEAREST / NEAR / NEWMV pick. The §16.4 `Quarters`
//! partition — four independent 2×2 sub-block groups — CAN align each
//! quadrant separately, producing a much lower per-MB SAD at the cost
//! of the partition's §16.4 / §17.2 NEW4X4 component bits. The
//! round-147 picker must therefore emit at least one SPLITMV MB on
//! this content, and the self-decode Y-PSNR must clear a non-trivial
//! floor that a whole-MB-only encoder could not.
//!
//! What this pins:
//!
//!   * the §16.2 `mv_ref_tree` SPLITMV path "1111" emission on a real
//!     encoder bitstream;
//!   * the §16.4 `mvpartition_tree` partition-id round-trip
//!     (encoder writes a partition id, decoder reads it back via
//!     [`oxideav_vp8::read_mv_partition`]);
//!   * the §16.4 `sub_mv_ref_tree` per-group mode round-trip
//!     (NEW4X4 followed by its §17.2 component differential);
//!   * the §18 / §14 SPLITMV reconstruction
//!     ([`oxideav_vp8::reconstruct_split_mv_mb`]) — the decoder must
//!     reproduce the per-sub-block prediction the encoder used;
//!   * the §15 loop-filter "filter internal edges" rule for SPLITMV
//!     (the encoder records `y_mode = B_PRED` so the filter geometry
//!     follows the spec's "B_PRED or SPLITMV" branch).
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

/// Render a high-frequency luma plane with distinct per-quadrant
/// patterns inside each 16×16 macroblock. Every 16×16 MB is split into
/// four 8×8 quadrants — TL / TR / BL / BR — each carrying a distinct
/// diagonal-ramp signature so the §16.4 `Quarters` partition has a
/// clean per-quadrant SAD minimum. A whole-MB MV cannot simultaneously
/// align all four quadrants when they each move differently.
fn quadranted_plane(width: usize, height: usize) -> Vec<u8> {
    let mut p = vec![0u8; width * height];
    for r in 0..height {
        for c in 0..width {
            let mb_r = r % 16;
            let mb_c = c % 16;
            // Per-quadrant signature: a unique diagonal ramp + offset
            // for each of the four 8×8 quadrants of the MB so a
            // per-quadrant MV has a clean SAD-zero target.
            let (offset, ramp_sign): (i32, i32) = match (mb_r < 8, mb_c < 8) {
                (true, true) => (60, 3),    // top-left
                (true, false) => (110, -2), // top-right
                (false, true) => (150, 2),  // bottom-left
                (false, false) => (90, -3), // bottom-right
            };
            let local_r = (mb_r as i32) % 8;
            let local_c = (mb_c as i32) % 8;
            let val = offset + ramp_sign * (local_r + local_c);
            p[r * width + c] = val.clamp(20, 235) as u8;
        }
    }
    p
}

/// Apply per-quadrant translation: every 8×8 quadrant of every 16×16
/// macroblock moves by its own `(dy, dx)` whole-luma-pixel vector.
/// Used to force the §16.4 `Quarters` partition to win — a single
/// whole-MB MV cannot simultaneously align the four quadrants.
fn shift_per_quadrant(
    src: &[u8],
    width: usize,
    height: usize,
    tl: (i32, i32),
    tr: (i32, i32),
    bl: (i32, i32),
    br: (i32, i32),
) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for r in 0..height {
        for c in 0..width {
            let mb_r = r % 16;
            let mb_c = c % 16;
            let (dy, dx) = match (mb_r < 8, mb_c < 8) {
                (true, true) => tl,
                (true, false) => tr,
                (false, true) => bl,
                (false, false) => br,
            };
            // Source pixel (clamped to plane bounds via §20.14
            // edge-replication, same convention as the encoder's
            // `fetch_block_whole_pixel`).
            let sr = (r as i32 - dy).clamp(0, height as i32 - 1) as usize;
            let sc = (c as i32 - dx).clamp(0, width as i32 - 1) as usize;
            out[r * width + c] = src[sr * width + sc];
        }
    }
    out
}

/// Walk the §11/§16 macroblock-mode layer of a P-frame and report the
/// (mode, mv) pair the encoder emitted for each MB. The SPLITMV branch
/// drives through [`oxideav_vp8::decode_split_mv`] so the decoder's
/// per-sub-block walk runs end-to-end against the encoder's emission.
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

/// Two-frame I+P with **divergent per-quadrant translation**. Each
/// 16×16 macroblock's four 8×8 quadrants each move by their own
/// whole-pixel vector; no single whole-MB MV can simultaneously align
/// all four. The §16.4 `Quarters` partition (four 2×2-subblock groups)
/// is the only shape whose four per-group MVs can cleanly recover the
/// pattern, so the round-147 picker MUST emit at least one SPLITMV MB
/// and the self-decode PSNR MUST clear the round's 30 dB bar (a
/// whole-MB-only encoder would absorb the per-quadrant SAD floor as
/// §14 residue, capping the PSNR around the high 20s).
#[test]
fn i_plus_p_per_quadrant_translation_emits_splitmv_and_clears_30db() {
    // 64×64 — 4×4 = 16 MBs. The outer ring carries the edge-replicated
    // §20.14 halo and tends to ZEROMV / NEARESTMV; the inner 2×2 MBs
    // are the ones whose quadrant pattern hits the §16.4 `Quarters`
    // partition sweet spot.
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (w / 2, h / 2);

    let y_i = quadranted_plane(w, h);
    // Per-quadrant translation: top-left moves +2 luma px down + right,
    // top-right moves -2 luma px down + +2 right, bottom-left moves
    // -2/+2, bottom-right +2/-2. Each MV is a multiple of
    // `WHOLE_PIXEL_STEP = 4` quarter-pixels (here whole-pixel) and well
    // inside §17.1's `[MV_MIN, MV_MAX]`.
    let y_p = shift_per_quadrant(
        &y_i,
        w,
        h,
        (2, 2),   // TL
        (-2, 2),  // TR
        (-2, -2), // BL
        (2, -2),  // BR
    );

    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];

    let frame_i = I420Frame::packed(w as u32, h as u32, &y_i, &u, &v);
    let frame_p = I420Frame::packed(w as u32, h as u32, &y_p, &u, &v);

    let params = KeyframeParams {
        // Mid quantiser — the §14 residue absorbs some of the per-MB
        // SAD that whole-MB candidates leave on this content; SPLITMV
        // wins by spending bits on per-group MVs that drive the
        // post-residue distortion materially down.
        y_ac_qi: 32,
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
    eprintln!("per-quadrant-translation P-frame self-decode Y-PSNR: {y_psnr:.2} dB (yac_qi=32)");
    assert!(
        y_psnr >= 30.0,
        "per-quadrant-translation P-frame Y-plane self-decode PSNR \
         {y_psnr:.2} dB < 30 dB floor"
    );

    // ---- Verify ≥ 1 SPLITMV MB was emitted ------------------------------
    let mb_cols = w.div_ceil(16);
    let mb_rows = h.div_ceil(16);
    let modes = parse_p_frame_inter_modes(&p_bytes, mb_cols, mb_rows);
    let split_count = modes.iter().filter(|(m, _)| *m == InterMode::Split).count();
    let zero_count = modes.iter().filter(|(m, _)| *m == InterMode::Zero).count();
    let nearest_count = modes
        .iter()
        .filter(|(m, _)| *m == InterMode::Nearest)
        .count();
    let near_count = modes.iter().filter(|(m, _)| *m == InterMode::Near).count();
    let new_count = modes.iter().filter(|(m, _)| *m == InterMode::New).count();
    eprintln!(
        "P-frame inter-mode pick: {zero_count}/{} ZEROMV, {nearest_count}/{} NEARESTMV, \
         {near_count}/{} NEARMV, {new_count}/{} NEWMV, {split_count}/{} SPLITMV",
        modes.len(),
        modes.len(),
        modes.len(),
        modes.len(),
        modes.len()
    );
    assert!(
        split_count >= 1,
        "expected ≥ 1 SPLITMV MB on a per-quadrant-translation scene \
         (where a single whole-MB MV cannot align all four 8×8 quadrants); \
         got {split_count} SPLITMV / {} MBs total",
        modes.len()
    );
}

// Silence the unused-import lint when the test binary's import block is
// pruned by future maintenance — `BoolDecoder` is used inside
// `parse_p_frame_inter_modes` via the `bool_decoder::read_bool` method,
// but rustc's unused-import check counts only direct uses.
#[allow(dead_code)]
fn _unused_imports_kept_alive(_d: BoolDecoder<'_>) {}
