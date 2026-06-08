#![no_main]

//! Fuzz: panic-freedom of the public §12 intra-prediction pixel kernels.
//!
//! The seven existing fuzz targets reach §12 only indirectly through
//! `decode_vp8` / `Vp8DecoderState::decode_frame` /
//! `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which gate
//! the per-block prediction kernels behind a fully-formed
//! reconstruction raster: every neighbour pixel arrives already
//! reconstructed, every (above, left, top-left) triple has been
//! sourced from the same valid frame, and `above`-extension pixels
//! (positions `(-1, 16) .. (-1, 19)` for right-edge sub-blocks) have
//! been pre-clamped per the §12.3 right-edge-fixup rules.
//!
//! That leaves the §12 primitive surface — the seven public 16×16 /
//! 8×8 mode kernels and the ten-arm `predict_b4x4` sub-block
//! dispatcher — directly under-fuzzed. The round-258
//! `intra_predict_dc16` criterion bench likewise only exercises
//! `predict_y16x16_dc` / `_v` / `_h` against a fixed `[128u8; 16] /
//! [129u8; 16]` neighbour pair, never the 4×4 sub-block kernel and
//! never the corner / edge / wrap-around envelope of the §12.3 `Vr`,
//! `Vl`, `Hd`, `Hu`, `Ld`, `Rd` diagonal arms.
//!
//! Surface coverage (every symbol below is exported from `oxideav_vp8`):
//!
//! * `predict_y16x16_dc(out, above: Option, left: Option)` — both
//!   edges present (canonical mid-frame), both edges absent (top-left
//!   macroblock → `DEFAULT_TOPLEFT_DC` fallback), and the two
//!   single-edge fallbacks (top row / left column).
//! * `predict_y16x16_v(out, above: &[u8; 16])` — full vertical fill.
//! * `predict_y16x16_h(out, left: &[u8; 16])` — full horizontal fill.
//! * `predict_y16x16_tm(out, above, left, p)` — every byte position
//!   of the 256-pixel output is `clamp255(left[r] + above[c] - p)` so
//!   adversarial `(above, left, p)` triples (e.g. all-zero left +
//!   all-saturated above + `p == 255`) exercise both the lower and
//!   upper saturating clamps.
//! * `predict_y16x16(out, mode, above, left, p)` dispatcher — driven
//!   across every variant of `IntraYMode` (`Dc`, `V`, `H`, `Tm`, `B`)
//!   with both `Option` polarities so the §12 `B_PRED` `None`-return
//!   path is exercised and the `unwrap_or([DEFAULT_*; 16])` fallbacks
//!   on `V` / `H` / `Tm` are exercised.
//! * `predict_uv8x8_dc` / `_v` / `_h` / `_tm` — the 8×8 chroma
//!   partners, same fallback envelope as the 16×16 luma family.
//! * `predict_uv8x8(out, mode, above, left, p)` dispatcher — every
//!   `IntraUvMode` variant, both `Option` polarities.
//! * `predict_b4x4(out, mode, above: &[u8; 8], left: &[u8; 4], p)` —
//!   driven across every variant of `IntraBmode` (`Dc`, `Tm`, `Ve`,
//!   `He`, `Ld`, `Rd`, `Vr`, `Vl`, `Hd`, `Hu`). The 8-byte `above`
//!   slot covers both the directly-above 4 pixels (`above[0..4]`)
//!   AND the right-extension 4 pixels (`above[4..8]`) the §12.3
//!   right-edge sub-block fixup feeds in for sub-blocks 3 / 7 / 11 /
//!   15. Every diagonal arm references different positions of `E`
//!   (`Rd`, `Vr`, `Hd`) or `above[0..=7]` (`Vl`, `Ld`); the harness
//!   walks them all so an off-by-one slip in any of the 16-pixel
//!   assignment lists would surface as a buffer write outside the
//!   fixed-size `[u8; 16]` destination — which the type system
//!   already rejects, but the harness's `mode`-sweep also stresses
//!   the helper functions `avg2p`, `avg3p`, `avg3` and the §12.3
//!   `E[0..=8]` array assembly that every diagonal arm reads.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes        | Meaning |
//! |-------------:|---------|
//! | `[0]`        | flags byte: bit0 `above_present`, bit1 `left_present`, bits 2-4 `IntraYMode` selector (0..=4), bits 5-6 `IntraUvMode` selector (0..=3), bit7 chained-pass toggle |
//! | `[1]`        | `IntraBmode` selector (0..=9 mod 10) |
//! | `[2]`        | `p` — the §12.3 top-left pixel (B-mode + TM-mode `p`) |
//! | `[3..=18]`   | 16-byte `above` row (positions `(-1, 0)..=(-1, 15)`) for the 16×16 luma kernels |
//! | `[19..=34]`  | 16-byte `left` column (positions `(0, -1)..=(15, -1)`) for the 16×16 luma kernels |
//! | `[35..=42]`  | 8-byte `above` row for the 8×8 chroma kernels |
//! | `[43..=50]`  | 8-byte `left` column for the 8×8 chroma kernels |
//! | `[51..=58]`  | 8-byte `above` row for `predict_b4x4` (covers the right-extension 4 pixels too) |
//! | `[59..=62]`  | 4-byte `left` column for `predict_b4x4` |
//!
//! Each kernel writes into a fresh stack-allocated output array of
//! the correct fixed size (`[u8; 256]` / `[u8; 64]` / `[u8; 16]`);
//! the type system already enforces the bounds on the destination,
//! and the harness asserts that every call returns without panic.
//!
//! Sweeps the §12 primitive layer for panic-freedom. All allocations
//! live on the stack; no heap touches.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    predict_b4x4, predict_uv8x8, predict_uv8x8_dc, predict_uv8x8_h, predict_uv8x8_tm,
    predict_uv8x8_v, predict_y16x16, predict_y16x16_dc, predict_y16x16_h, predict_y16x16_tm,
    predict_y16x16_v, IntraBmode, IntraUvMode, IntraYMode,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 63;

fn ymode_from(sel: u8) -> IntraYMode {
    match sel % 5 {
        0 => IntraYMode::Dc,
        1 => IntraYMode::V,
        2 => IntraYMode::H,
        3 => IntraYMode::Tm,
        _ => IntraYMode::B,
    }
}

fn uvmode_from(sel: u8) -> IntraUvMode {
    match sel % 4 {
        0 => IntraUvMode::Dc,
        1 => IntraUvMode::V,
        2 => IntraUvMode::H,
        _ => IntraUvMode::Tm,
    }
}

fn bmode_from(sel: u8) -> IntraBmode {
    match sel % 10 {
        0 => IntraBmode::Dc,
        1 => IntraBmode::Tm,
        2 => IntraBmode::Ve,
        3 => IntraBmode::He,
        4 => IntraBmode::Ld,
        5 => IntraBmode::Rd,
        6 => IntraBmode::Vr,
        7 => IntraBmode::Vl,
        8 => IntraBmode::Hd,
        _ => IntraBmode::Hu,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let above_present = (flags & 0b0000_0001) != 0;
    let left_present = (flags & 0b0000_0010) != 0;
    let ymode_sel = (flags >> 2) & 0b0000_0111;
    let uvmode_sel = (flags >> 5) & 0b0000_0011;
    let chained = (flags & 0b1000_0000) != 0;

    let bmode_sel = data[1];
    let p = data[2];

    // 16-byte `above` / `left` for the luma 16×16 kernels.
    let mut above_y = [0u8; 16];
    let mut left_y = [0u8; 16];
    above_y.copy_from_slice(&data[3..19]);
    left_y.copy_from_slice(&data[19..35]);

    // 8-byte `above` / `left` for the chroma 8×8 kernels.
    let mut above_uv = [0u8; 8];
    let mut left_uv = [0u8; 8];
    above_uv.copy_from_slice(&data[35..43]);
    left_uv.copy_from_slice(&data[43..51]);

    // 8-byte `above` (includes right-extension) + 4-byte `left` for
    // the §12.3 `predict_b4x4` sub-block dispatcher.
    let mut above_b = [0u8; 8];
    let mut left_b = [0u8; 4];
    above_b.copy_from_slice(&data[51..59]);
    left_b.copy_from_slice(&data[59..63]);

    let ymode = ymode_from(ymode_sel);
    let uvmode = uvmode_from(uvmode_sel);
    let bmode = bmode_from(bmode_sel);

    // ---- (1) 16×16 luma kernels — direct calls. ----
    let mut out16 = [0u8; 256];

    // DC: every Option polarity.
    predict_y16x16_dc(&mut out16, Some(&above_y), Some(&left_y));
    predict_y16x16_dc(&mut out16, Some(&above_y), None);
    predict_y16x16_dc(&mut out16, None, Some(&left_y));
    predict_y16x16_dc(&mut out16, None, None);

    // V / H — single-edge kernels.
    predict_y16x16_v(&mut out16, &above_y);
    predict_y16x16_h(&mut out16, &left_y);

    // TM — both edges + corner `p`. Drive raw `p` straight from the
    // input so the §12.2 `clamp255` floor and ceiling are exercised
    // against the adversarial `above` / `left` byte patterns above.
    predict_y16x16_tm(&mut out16, &above_y, &left_y, p);

    // ---- (2) 16×16 luma dispatcher — every mode × every Option
    // polarity decided by the `above_present` / `left_present` flag
    // bits. Drives the `IntraYMode::B → None` short-circuit too. ----
    let above_y_opt = if above_present { Some(&above_y) } else { None };
    let left_y_opt = if left_present { Some(&left_y) } else { None };
    let _ = predict_y16x16(&mut out16, ymode, above_y_opt, left_y_opt, p);

    // Also sweep every `IntraYMode` so the dispatcher's `B`-arm
    // `None` return is reachable from every input.
    for m in [
        IntraYMode::Dc,
        IntraYMode::V,
        IntraYMode::H,
        IntraYMode::Tm,
        IntraYMode::B,
    ] {
        let _ = predict_y16x16(&mut out16, m, above_y_opt, left_y_opt, p);
    }

    // ---- (3) 8×8 chroma kernels — direct calls + dispatcher. ----
    let mut out8 = [0u8; 64];

    predict_uv8x8_dc(&mut out8, Some(&above_uv), Some(&left_uv));
    predict_uv8x8_dc(&mut out8, Some(&above_uv), None);
    predict_uv8x8_dc(&mut out8, None, Some(&left_uv));
    predict_uv8x8_dc(&mut out8, None, None);

    predict_uv8x8_v(&mut out8, &above_uv);
    predict_uv8x8_h(&mut out8, &left_uv);
    predict_uv8x8_tm(&mut out8, &above_uv, &left_uv, p);

    let above_uv_opt = if above_present { Some(&above_uv) } else { None };
    let left_uv_opt = if left_present { Some(&left_uv) } else { None };
    predict_uv8x8(&mut out8, uvmode, above_uv_opt, left_uv_opt, p);

    // Sweep every `IntraUvMode` so the four arms are each reachable
    // from every input.
    for m in [
        IntraUvMode::Dc,
        IntraUvMode::V,
        IntraUvMode::H,
        IntraUvMode::Tm,
    ] {
        predict_uv8x8(&mut out8, m, above_uv_opt, left_uv_opt, p);
    }

    // ---- (4) 4×4 sub-block dispatcher — sweep every `IntraBmode`
    // arm. Each arm references a different slice of `above[0..=7]` /
    // `left[0..=3]` and the synthetic §12.3 `E[0..=8]` array; sweeping
    // every arm against the same `(above_b, left_b, p)` triple
    // exercises the assignment-list arithmetic of every diagonal mode
    // (`Ld`, `Rd`, `Vr`, `Vl`, `Hd`, `Hu`) plus the simpler `Dc` /
    // `Tm` / `Ve` / `He` arms. ----
    let mut out4 = [0u8; 16];
    for m in [
        IntraBmode::Dc,
        IntraBmode::Tm,
        IntraBmode::Ve,
        IntraBmode::He,
        IntraBmode::Ld,
        IntraBmode::Rd,
        IntraBmode::Vr,
        IntraBmode::Vl,
        IntraBmode::Hd,
        IntraBmode::Hu,
    ] {
        predict_b4x4(&mut out4, m, &above_b, &left_b, p);
    }

    // Per-iteration variant picked by the input.
    predict_b4x4(&mut out4, bmode, &above_b, &left_b, p);

    // ---- (5) Chained leg — when the `chained` flag bit is set,
    // re-feed the 16×16 luma TM output's first row into the chroma
    // `above` slot and the first column into the chroma `left` slot,
    // simulating the cross-plane neighbour reuse the §11 / §12
    // macroblock walker performs when both luma and chroma planes
    // come from the same reconstructed neighbour MB. The chroma
    // dispatcher is then re-driven with the synthesised neighbour
    // pair to exercise a kernel-output-as-kernel-input data-flow path
    // the per-call legs above do not visit. ----
    if chained {
        predict_y16x16_tm(&mut out16, &above_y, &left_y, p);

        let mut chained_above_uv = [0u8; 8];
        let mut chained_left_uv = [0u8; 8];
        // First row of `out16` (positions `(0, 0)..=(0, 7)`).
        chained_above_uv.copy_from_slice(&out16[0..8]);
        // First column of `out16` (positions `(0, 0)..=(7, 0)`).
        for (r, slot) in chained_left_uv.iter_mut().enumerate() {
            *slot = out16[r * 16];
        }

        predict_uv8x8(
            &mut out8,
            uvmode,
            Some(&chained_above_uv),
            Some(&chained_left_uv),
            p,
        );
    }
});
