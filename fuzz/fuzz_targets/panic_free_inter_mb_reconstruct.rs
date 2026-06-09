#![no_main]

//! Fuzz: panic-freedom of the public §16 inter-MB reconstruction
//! surface (RFC 6386 §16 / §16.2 / §16.4 / §18 / §18.1 / §18.3).
//!
//! Surface coverage (all entry points reached directly from attacker
//! bytes, plus the §18.1 vector-adjustment primitives that gate every
//! call):
//!
//! * `reconstruct_inter_mb_whole_pixel(reference, mb_col, mb_row,
//!   luma_mv, full_pixel, mb_skip_coeff, y2_coeffs_dequant,
//!   y_coeffs_dequant, u_coeffs_dequant, v_coeffs_dequant)` — §16.2
//!   non-SPLITMV whole-pixel reconstruction (`§18.2` prediction +
//!   `§14.2` inverse-WHT + `§14.4` inverse-DCT + `§14.5` residue add).
//!   Returns `MotionCompError::SubPixelNotSupported` on every non-
//!   whole-pixel vector; the harness asserts the discriminator agrees
//!   with the §18.1 fractional gate it computes itself.
//! * `reconstruct_inter_mb(reference, …, filters, …)` — §16.2 / §18.3
//!   full sub-pixel path. Reachable on every (mx, my) ∈ {0..7}²
//!   fractional combination of the post-§18.1 luma + chroma vectors.
//! * `reconstruct_split_mv_mb(reference, …, split_luma_mvs, …)` —
//!   §16.4 SPLITMV: sixteen per-luma-sub-block vectors + four §18.1-
//!   averaged chroma vectors, no Y2 block (the §14.2 SPLITMV
//!   short-circuit).
//! * `predict_inter_mb_whole_pixel` / `predict_inter_mb` /
//!   `predict_split_mv` — the residue-free counterparts of each
//!   `reconstruct_*` entry point. The harness drives them in addition
//!   to the reconstruct legs because the §11.1 skip short-circuit
//!   inside `reconstruct_*` means `mb_skip_coeff == true` collapses
//!   into the predict-only path; we want both polarities exercised
//!   from the same input bytes.
//! * `select_ref_frame(dec, prob_last, prob_gf)` — §16.2 reference-
//!   frame discriminator. A short attacker-shaped partition feeds the
//!   bool coder; every variant of `RefFrame` (`Last` / `Golden` /
//!   `AltRef`) is reachable.
//! * §18.1 vector-adjustment primitives `stored_luma_mv`, `chroma_mv`,
//!   `apply_full_pixel`, `whole_pixel_fraction_is_zero` — each is
//!   pure and panic-free by construction; they are driven on every
//!   iteration so any future refactor that introduces a panic gate
//!   (e.g. an overflow check) surfaces here.
//! * §18.1 chroma index mapping `chroma_idx_for_luma_subblock` and
//!   the four-vector chroma average `split_chroma_mvs` — the §16.4
//!   primitives the SPLITMV path consumes; driven independently from
//!   the SPLITMV reconstruct call so the dispatch table is exercised
//!   on every input.
//! * `filter_set_for_version(version)` — both arms (`Sixtap` for
//!   `version == 0`, `Bilinear` for any other byte). The selected
//!   tap set feeds the full-sub-pixel `reconstruct_inter_mb` and
//!   `reconstruct_split_mv_mb` calls.
//!
//! Surface gap the existing fuzz targets leave cold:
//!
//! The thirteen fuzz targets above reach §16 only indirectly: the
//! decode-side targets feed bytes through `decode_vp8` /
//! `Vp8DecoderState::decode_frame`, which gate the inter-MB path
//! behind a fully-formed previous keyframe + §9.7 reference refresh
//! state machine — every (`prev`, `golden`, `altref`) reference plane
//! is well-formed and every motion vector was already snap-clamped
//! by §17.1. The encode-side targets reach `reconstruct_inter_mb`
//! through `encode_p_frame_multi_ref` / `Vp8TwoPassEncoder::encode_frame`
//! and feed the §14 residue blocks with §9.6-clamped quantiser
//! tables. None of the thirteen drive `reconstruct_inter_mb` /
//! `reconstruct_inter_mb_whole_pixel` / `reconstruct_split_mv_mb`
//! with attacker-shaped `(mb_col, mb_row, luma_mv, full_pixel,
//! mb_skip_coeff, y2_coeffs_dequant, y_coeffs_dequant,
//! u_coeffs_dequant, v_coeffs_dequant)` tuples directly. The
//! round-256 `panic_free_motion_search_descent` and round-257
//! `panic_free_sixtap_subpel` targets reach the §18.3 sub-pixel
//! synthesis primitive layer, but never the §16 macroblock-level
//! reconstruction orchestrator. The round-262
//! `panic_free_transform_4x4_roundtrip` target reaches §14, but
//! never feeds the residue into a §16 reconstruct call.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | flags byte: bit0 `full_pixel`; bit1 `mb_skip_coeff`; bits 2..=3 path discriminator (0 = whole-pixel, 1 = sub-pixel, 2 = SPLITMV, 3 = SPLITMV); bit4 `version` selector for `filter_set_for_version`; bit5 toggles all-zero residue vs payload-tiled residue |
//! | `[1]`     | `mb_cols` selector — `(data[1] % 3) + 1` ∈ {1, 2, 3} |
//! | `[2]`     | `mb_rows` selector — `(data[2] % 3) + 1` ∈ {1, 2, 3} |
//! | `[3]`     | `mb_col` selector — saturated against `mb_cols` |
//! | `[4]`     | `mb_row` selector — saturated against `mb_rows` |
//! | `[5..7]`  | `luma_mv.row` — two raw bytes interpreted as `i16` and pre-clamped to the §17.1 envelope `-1023..=1023` |
//! | `[7..9]`  | `luma_mv.col` — same shape as row |
//! | `[9]`     | `prob_last` for `select_ref_frame` |
//! | `[10]`    | `prob_gf` for `select_ref_frame` |
//! | `[11..]`  | reference-plane payload — tiled into the luma + U + V planes; also tiled into the per-sub-block §14 residue coefficient arrays so the §14.2 / §14.4 inverse path consumes attacker-shaped quantised residuals |
//!
//! The harness caps input ≤ 4 KiB (libFuzzer default; re-checked at
//! entry as defence-in-depth) and requires ≥ 11 bytes for the fixed
//! header. Per-iteration allocations: three `Vec<u8>` for the I420
//! reference planes (`mb_cols * mb_rows` ≤ 9 MBs ⇒ ≤ 9 × 384 = 3456 B
//! per MB array) and no other heap traffic — every coefficient array
//! is stack-allocated.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    bool_decoder::BoolDecoder,
    motion_comp::{
        apply_full_pixel, chroma_idx_for_luma_subblock, chroma_mv, filter_set_for_version,
        predict_inter_mb, predict_inter_mb_whole_pixel, predict_split_mv, reconstruct_inter_mb,
        reconstruct_inter_mb_whole_pixel, reconstruct_split_mv_mb, select_ref_frame,
        split_chroma_mvs, stored_luma_mv, whole_pixel_fraction_is_zero, FilterSet, MotionCompError,
        ReferencePlanes,
    },
    motion_vector::Mv,
};

const MIN_INPUT_BYTES: usize = 11;
const MAX_INPUT_BYTES: usize = 4 * 1024;

/// §17.1 motion-vector component clamp envelope.
const MV_CLAMP: i16 = 1023;

/// Read two raw bytes as an `i16`, then clamp into the §17.1 envelope.
fn read_clamped_mv_component(lo: u8, hi: u8) -> i16 {
    let v = i16::from_le_bytes([lo, hi]);
    v.clamp(-MV_CLAMP, MV_CLAMP)
}

/// Tile `payload` into a destination slice of `len` bytes (repeating
/// the payload until full). An empty payload writes all zeros.
fn tile_into(payload: &[u8], len: usize) -> Vec<u8> {
    if payload.is_empty() {
        return vec![0u8; len];
    }
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let take = (len - out.len()).min(payload.len());
        out.extend_from_slice(&payload[..take]);
    }
    out
}

/// Fill a 16-coefficient block from a tiled payload, interpreting each
/// pair of bytes as a signed `i16` and clamping into the §14.2 envelope.
fn fill_coeff_block(payload: &[u8], start: usize) -> [i16; 16] {
    let mut out = [0i16; 16];
    if payload.is_empty() {
        return out;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let p = (start + i * 2) % payload.len();
        let q = (start + i * 2 + 1) % payload.len();
        let raw = i16::from_le_bytes([payload[p], payload[q]]);
        // §14.2 dequantised-residue envelope is ±2047 (the inverse-DCT
        // intermediate-product bound documented in §14.4). Clamp the
        // attacker bytes there so the §14.4 i32 butterfly multiplies
        // stay inside `i32`; panic-freedom is still asserted across
        // every legal magnitude.
        *slot = raw.clamp(-2047, 2047);
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < MIN_INPUT_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let full_pixel = (flags & 0b0000_0001) != 0;
    let mb_skip_coeff = (flags & 0b0000_0010) != 0;
    let path = (flags >> 2) & 0b0000_0011;
    let version = (flags >> 4) & 1;
    let zero_residue = (flags & 0b0010_0000) != 0;

    // §18.2 reference-plane dimensions — always whole-MB and ≥ one MB.
    let mb_cols = ((data[1] as usize) % 3) + 1;
    let mb_rows = ((data[2] as usize) % 3) + 1;
    let lw = mb_cols * 16;
    let lh = mb_rows * 16;
    let cw = mb_cols * 8;
    let ch = mb_rows * 8;

    // §16 macroblock position — saturated inside the plane.
    let mb_col = (data[3] as usize) % mb_cols;
    let mb_row = (data[4] as usize) % mb_rows;

    // §17.1 motion-vector envelope.
    let luma_mv = Mv {
        row: read_clamped_mv_component(data[5], data[6]),
        col: read_clamped_mv_component(data[7], data[8]),
    };

    // §16.2 reference-frame selector — a short attacker partition
    // drives `select_ref_frame`. Two-byte minimum for the §7.3 init.
    let part_start = 11.min(data.len());
    let part_end = (part_start + 16).min(data.len());
    if part_end - part_start >= 2 {
        if let Ok(mut dec) = BoolDecoder::init(&data[part_start..part_end]) {
            let _ = select_ref_frame(&mut dec, data[9], data[10]);
        }
    }

    // §18.1 vector-adjustment primitives — pure, panic-free on every
    // input. Driven on every iteration so any future refactor that
    // introduces a panic gate surfaces here. The §18.1 doubling is
    // bounded by `wrapping_mul(2)` per the spec text — i16 wrap is
    // by construction.
    let ymv = stored_luma_mv(luma_mv);
    let uvmv = chroma_mv(ymv);
    let _ymv_fp = apply_full_pixel(ymv);
    let _uvmv_fp = apply_full_pixel(uvmv);
    let _ymv_frac0 = whole_pixel_fraction_is_zero(ymv);
    let _uvmv_frac0 = whole_pixel_fraction_is_zero(uvmv);
    for b in 0..16 {
        let _ = chroma_idx_for_luma_subblock(b);
    }

    // §16.4 SPLITMV: sixteen per-luma-sub-block vectors derived from
    // the attacker payload (the §17.1 envelope still applies). All
    // sixteen feed `split_chroma_mvs` for the §18.1 averaging cross-
    // check on every iteration.
    let payload = &data[11..];
    let mut split_luma_mvs = [Mv::default(); 16];
    for (b, slot) in split_luma_mvs.iter_mut().enumerate() {
        let off = (b * 4) % payload.len().max(1);
        if payload.len() >= off + 4 {
            *slot = Mv {
                row: read_clamped_mv_component(payload[off], payload[off + 1]),
                col: read_clamped_mv_component(payload[off + 2], payload[off + 3]),
            };
        }
    }
    let _chroma_split = split_chroma_mvs(&split_luma_mvs);

    // §18.3 filter set: both arms reachable from one bit of the flags
    // byte.
    let filter_set = filter_set_for_version(version);
    let filters = match filter_set {
        FilterSet::Sixtap => filter_set.taps(),
        FilterSet::Bilinear => filter_set.taps(),
    };

    // §18.2 reference planes — payload-tiled luma + chroma.
    let y_plane = tile_into(payload, lw * lh);
    let u_plane = tile_into(payload, cw * ch);
    let v_plane = tile_into(payload, cw * ch);
    let reference = ReferencePlanes {
        y: &y_plane,
        u: &u_plane,
        v: &v_plane,
        y_stride: lw,
        uv_stride: cw,
        mb_cols,
        mb_rows,
    };

    // §14.2 / §14.4 dequantised residue blocks. The harness either
    // zero-fills (the `mb_skip_coeff == true` short-circuit path
    // produces the same output regardless, but a zero residue exercises
    // the §14.5 `add_residue_4x4` identity) or payload-tiles per
    // §14.2 envelope.
    let mut y2 = [0i16; 16];
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut u_coeffs = [[0i16; 16]; 4];
    let mut v_coeffs = [[0i16; 16]; 4];
    if !zero_residue {
        y2 = fill_coeff_block(payload, 0);
        for (i, blk) in y_coeffs.iter_mut().enumerate() {
            *blk = fill_coeff_block(payload, 32 + i * 32);
        }
        for (i, blk) in u_coeffs.iter_mut().enumerate() {
            *blk = fill_coeff_block(payload, 32 * 17 + i * 32);
        }
        for (i, blk) in v_coeffs.iter_mut().enumerate() {
            *blk = fill_coeff_block(payload, 32 * 21 + i * 32);
        }
    }

    // §16.2 / §16.4 reconstruct dispatch.
    match path {
        0 => {
            // §16.2 whole-pixel path. The §18.1 fractional gate is the
            // documented `SubPixelNotSupported` predicate; the harness
            // cross-checks the dispatcher's discriminator against the
            // gate it computes itself. We must mirror the public
            // `predict_inter_mb_whole_pixel` exactly: §18.1 stored-luma
            // doubling + chroma averaging FIRST, then the
            // `full_pixel`-gated truncation on BOTH planes, only THEN
            // the fraction test.
            let (gate_ymv, gate_uvmv) = if full_pixel {
                (apply_full_pixel(ymv), apply_full_pixel(uvmv))
            } else {
                (ymv, uvmv)
            };
            let want_subpel_err =
                !whole_pixel_fraction_is_zero(gate_ymv) || !whole_pixel_fraction_is_zero(gate_uvmv);
            let pred_res =
                predict_inter_mb_whole_pixel(&reference, mb_col, mb_row, luma_mv, full_pixel);
            let recon_res = reconstruct_inter_mb_whole_pixel(
                &reference,
                mb_col,
                mb_row,
                luma_mv,
                full_pixel,
                mb_skip_coeff,
                &y2,
                &y_coeffs,
                &u_coeffs,
                &v_coeffs,
            );
            // Predict and reconstruct must agree on the discriminator.
            assert_eq!(
                pred_res.is_err(),
                recon_res.is_err(),
                "whole-pixel predict/reconstruct discriminator mismatch"
            );
            // The §16.2 path returns SubPixelNotSupported on every
            // non-zero fractional component after §18.1 doubling +
            // chroma averaging + (optional) full-pel truncation.
            if want_subpel_err {
                assert!(
                    matches!(pred_res, Err(MotionCompError::SubPixelNotSupported)),
                    "non-whole-pixel vector did not surface §18.1 SubPixelNotSupported"
                );
            }
            // On the in-range leg with `mb_skip_coeff == true`, the
            // reconstruct call collapses into the prediction — the
            // §11.1 skip short-circuit guarantees byte equality.
            if let (Ok(pred), Ok(recon)) = (pred_res, recon_res) {
                if mb_skip_coeff {
                    assert_eq!(
                        pred, recon,
                        "mb_skip_coeff path: reconstruct != predict (§11.1 skip)"
                    );
                }
            }
        }
        1 => {
            // §16.2 / §18.3 full sub-pixel path.
            let pred = predict_inter_mb(&reference, mb_col, mb_row, luma_mv, full_pixel, filters);
            let recon = reconstruct_inter_mb(
                &reference,
                mb_col,
                mb_row,
                luma_mv,
                full_pixel,
                filters,
                mb_skip_coeff,
                &y2,
                &y_coeffs,
                &u_coeffs,
                &v_coeffs,
            );
            if mb_skip_coeff {
                assert_eq!(
                    pred, recon,
                    "mb_skip_coeff path (sub-pixel): reconstruct != predict (§11.1 skip)"
                );
            }
        }
        _ => {
            // §16.4 SPLITMV path. Two flag-bit values 2 / 3 both map
            // here so the path discriminator's full envelope is
            // exercised.
            let pred = predict_split_mv(
                &reference,
                mb_col,
                mb_row,
                &split_luma_mvs,
                full_pixel,
                filters,
            );
            let recon = reconstruct_split_mv_mb(
                &reference,
                mb_col,
                mb_row,
                &split_luma_mvs,
                full_pixel,
                filters,
                mb_skip_coeff,
                &y_coeffs,
                &u_coeffs,
                &v_coeffs,
            );
            if mb_skip_coeff {
                assert_eq!(
                    pred, recon,
                    "mb_skip_coeff path (SPLITMV): reconstruct != predict (§11.1 skip)"
                );
            }
        }
    }
});
