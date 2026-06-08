#![no_main]

//! Fuzz: panic-freedom of the public §17.1 / §18.3 luma motion-search
//! descent ladder.
//!
//! The pre-existing fuzz targets reach the §17 search shape only
//! indirectly through `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`,
//! which gate the per-MB motion-vector picker behind a fully-formed
//! I420 frame + §11 mode picker + §13 token emitter cascade. That
//! leaves the §17 / §18.3 primitive surface itself —
//! `small_diamond_search_luma`, `half_pixel_refine_luma`,
//! `quarter_pixel_refine_luma`, `mb_luma_sad_at_whole_mv` and
//! `mb_luma_sad_at_mv` — directly under-fuzzed.
//!
//! In particular, every per-candidate fetch inside the half- and
//! quarter-pixel refines walks the §18.3 sixtap synthesis through
//! `filter_block_4x4`, which fans out to either the whole-pixel
//! `fetch_block_whole_pixel` copy path or the `fetch_block_halo`
//! 9×9 edge-replicated-halo + `sixtap_2d` convolution. The §20.14
//! edge-replication clamp inside `fetch_block_halo` is exercised
//! only when the (mb_col, mb_row) + (mv >> 3) origin lands within
//! two pixels of a picture boundary — a slice of the configuration
//! space the round-255 `motion_search_descent` criterion bench
//! never visits (it pins the MB at (1, 1) inside a 64×64 plane so
//! the SAD landscape stays clear of the clamp). A caller that
//! parks the MB at (0, 0) and starts the descent with a large
//! negative center MV would land every per-candidate fetch in the
//! clamp; a malformed plane dimension (width or height = 0, or one
//! pixel smaller than the MB span) would land it off the back of
//! the plane entirely.
//!
//! This target drives that surface with a deliberately wide
//! (mb-position, mv-center, plane-dimension, source-block) envelope
//! so libFuzzer can hit every §17.1 component clamp at the
//! ±`MV_MIN`/`MV_MAX` boundary, every §20.14 edge-replication
//! branch inside the halo fetch, and every fan-out of the 9-row
//! sixtap intermediate the §18.3 convolution synthesises.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | flags byte: bit0 selects `mb_luma_sad_at_whole_mv` (else `mb_luma_sad_at_mv`); bits 1..=2 pick which descent stage to run (0 = whole-pixel only, 1 = + half-pixel, 2 = + quarter-pixel, 3 = full ladder); bit3 toggles the seed pattern between gradient and constant |
//! | `[1]`     | plane width selector — `width = (data[1] as usize % 4) * 8 + 16` (∈ {16, 24, 32, 40}, all ≥ one 16×16 MB) |
//! | `[2]`     | plane height selector — `height = (data[2] as usize % 4) * 8 + 16` (∈ {16, 24, 32, 40}) |
//! | `[3]`     | `mb_col` selector — saturated against `width / 16` so the macroblock origin stays inside the plane |
//! | `[4]`     | `mb_row` selector — saturated against `height / 16` |
//! | `[5..7]`  | center MV row — two raw bytes interpreted as `i16`, then clamped into `[MV_MIN, MV_MAX]` before the search begins (the `small_diamond_search_luma` API does its own snap-and-clamp; we pre-clamp for the half/quarter refine variants which the orchestrator drives with the post-search whole-pixel result) |
//! | `[7..9]`  | center MV col — same layout as row |
//! | `[9]`     | `max_iters` for the whole-pixel diamond descent (saturated to `≤ MAX_DIAMOND_ITERS`) |
//! | `[10..]`  | reference-plane payload — tiled into a `Vec<u8>` of length `width * height` |
//!
//! The source 16×16 block is synthesised independently from the flags
//! byte (gradient or constant) so the SAD landscape against the
//! payload-tiled reference plane is non-degenerate and the descent
//! makes a non-zero number of probes per stage. The harness:
//!
//! 1. Synthesises a `width × height` reference plane from the
//!    payload (tile-extended if shorter), then a 16×16 source block
//!    seeded from the flags byte.
//! 2. Calls `small_diamond_search_luma` with the clamped center MV
//!    and saturated `max_iters` — its return value snaps onto the
//!    whole-pixel grid by construction.
//! 3. Stages 1..=3 chain `half_pixel_refine_luma` and
//!    `quarter_pixel_refine_luma` against the snapped intermediate.
//!    Each refine asserts its input grid in debug builds; the
//!    chain guarantees the asserts always hold by construction.
//! 4. On the bit0 leg, additionally invokes the per-candidate
//!    evaluators `mb_luma_sad_at_whole_mv` and `mb_luma_sad_at_mv`
//!    against a small fixed sweep of MVs from the center (whole-,
//!    half-, quarter-pixel offsets) so the §18.3 sixtap synthesis
//!    inside `mb_luma_sad_at_mv` is exercised at every fractional
//!    offset, not just the ones the descent happened to land on.
//!
//! Sweeps the §17 / §18.3 primitive layer for panic-freedom, with
//! the reference plane and 16×16 source block as the only
//! allocations.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at
//! harness entry as defence-in-depth).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::motion_search::{
    half_pixel_refine_luma, mb_luma_sad_at_mv, mb_luma_sad_at_whole_mv, quarter_pixel_refine_luma,
    small_diamond_search_luma, LumaRef, HALF_PIXEL_STEP, MV_MAX, MV_MIN, QUARTER_PIXEL_STEP,
    WHOLE_PIXEL_STEP,
};
use oxideav_vp8::motion_vector::Mv;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 10;
/// Hard cap on the diamond descent so a maximally adversarial
/// `max_iters` byte cannot make the harness time out. Each iteration
/// probes the 4-neighbour ring, so 8 iterations are enough to walk a
/// dimensional cross-section of the search-plane / 4 ≈ 8 pixels —
/// well past the bench's representative value.
const MAX_DIAMOND_ITERS: u32 = 8;

fn clamp_to_range(v: i32) -> i16 {
    v.clamp(MV_MIN as i32, MV_MAX as i32) as i16
}

fn snap_whole(v: i16) -> i16 {
    (v / WHOLE_PIXEL_STEP) * WHOLE_PIXEL_STEP
}

fn snap_half(v: i16) -> i16 {
    (v / HALF_PIXEL_STEP) * HALF_PIXEL_STEP
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let use_whole_pixel_evaluator = (flags & 0b0000_0001) != 0;
    let descent_stage = (flags >> 1) & 0b11;
    let constant_seed = (flags & 0b0000_1000) != 0;

    // Plane dimensions: ∈ {16, 24, 32, 40} per axis. Each is a
    // multiple of 4 (the sixtap halo support is 9 cols/rows, so the
    // smallest legal plane is 16 = one MB; larger planes give the
    // descent room to wander before the §17.1 boundary clamp kicks
    // in).
    let width = (data[1] as usize % 4) * 8 + 16;
    let height = (data[2] as usize % 4) * 8 + 16;

    // MB origin must stay inside the plane. width / 16 is the number
    // of whole MB columns; saturate the selector against
    // `mb_cols - 1`.
    let mb_cols = width / 16;
    let mb_rows = height / 16;
    let mb_col = (data[3] as usize) % mb_cols;
    let mb_row = (data[4] as usize) % mb_rows;

    // Center MV — two-byte signed components, pre-clamped into the
    // §17.1 envelope. `small_diamond_search_luma` does its own
    // snap-and-clamp; the pre-clamp keeps the value in range so the
    // post-search whole-pixel center we feed into the half/quarter
    // refines (which debug-assert their inputs) never falls outside.
    let center_row_raw = i16::from_le_bytes([data[5], data[6]]);
    let center_col_raw = i16::from_le_bytes([data[7], data[8]]);
    let center = Mv {
        row: clamp_to_range(center_row_raw as i32),
        col: clamp_to_range(center_col_raw as i32),
    };

    let max_iters = (data[9] as u32) % (MAX_DIAMOND_ITERS + 1);

    // Build the reference plane: tile the payload (or seed from the
    // flags byte if the payload is empty) so the SAD landscape is
    // non-degenerate. A wholly-zero plane would collapse the descent
    // to a single trivial probe and skip the boundary clamps.
    let payload = &data[HEADER_BYTES..];
    let mut ref_plane = vec![0u8; width * height];
    if payload.is_empty() {
        for (i, slot) in ref_plane.iter_mut().enumerate() {
            *slot = flags.wrapping_add(i as u8);
        }
    } else {
        for (i, slot) in ref_plane.iter_mut().enumerate() {
            *slot = payload[i % payload.len()];
        }
    }

    let reference = LumaRef {
        plane: &ref_plane,
        stride: width,
        width,
        height,
    };

    // Build the 16×16 source block. Distinct from the reference
    // pattern so the SAD is non-zero at the center and the descent
    // has a non-trivial landscape to walk.
    let mut src_y = [0u8; 256];
    if constant_seed {
        // Constant fill — only the boundary-clamp / sixtap-halo
        // paths see any variation against the reference plane.
        let v = flags.wrapping_add(0x33);
        src_y.iter_mut().for_each(|s| *s = v);
    } else {
        for (i, s) in src_y.iter_mut().enumerate() {
            let r = i / 16;
            let c = i % 16;
            *s = ((r as u32 * 7)
                .wrapping_add(c as u32 * 11)
                .wrapping_add(flags as u32 * 13)
                ^ (r as u32 * c as u32)) as u8;
        }
    }

    // ---- (1) Whole-pixel diamond descent. ----
    // `small_diamond_search_luma` snaps + clamps the center
    // internally; its result is on the whole-pixel grid by
    // construction.
    let whole = small_diamond_search_luma(reference, mb_col, mb_row, &src_y, center, max_iters);

    // ---- (2) Half-pixel refine (debug-asserts the input is on the
    // whole-pixel grid). The diamond result already satisfies it. ----
    let half = if descent_stage >= 1 {
        let r = half_pixel_refine_luma(reference, mb_col, mb_row, &src_y, whole.mv);
        // Defence-in-depth: r.mv must be on the half-pixel grid by
        // construction. Skip the assert under fuzzing (it would
        // count as a panic) — instead, snap before feeding into
        // quarter to keep that stage's debug-assert satisfied even
        // if a future regression broke half_pixel_refine_luma's
        // grid invariant.
        let snapped = Mv {
            row: snap_half(r.mv.row),
            col: snap_half(r.mv.col),
        };
        Some(snapped)
    } else {
        None
    };

    // ---- (3) Quarter-pixel refine (debug-asserts the input is on
    // the half-pixel grid). The snapped half result satisfies it. ----
    if let Some(half_mv) = half {
        if descent_stage >= 2 {
            let _ = quarter_pixel_refine_luma(reference, mb_col, mb_row, &src_y, half_mv);
        }
    }

    // ---- (4) Full ladder rerun on the unsnapped post-half result:
    // exercises the descent-stage = 3 leg the bench's
    // `full_descent_whole_half_quarter` group uses, where the
    // intermediate result is NOT pre-snapped between stages. ----
    if descent_stage == 3 {
        let full_half = half_pixel_refine_luma(reference, mb_col, mb_row, &src_y, whole.mv);
        // full_half.mv is on the half-pixel grid by construction;
        // pass directly without a defensive snap so the chain
        // matches the encoder's actual call shape.
        let _ = quarter_pixel_refine_luma(reference, mb_col, mb_row, &src_y, full_half.mv);
    }

    // ---- (5) Per-candidate evaluator sweep — exercises the §18.3
    // sixtap synthesis at every fractional offset (mx, my) ∈ {0, 2,
    // 4, 6}², not just the ones the descent landed on. The
    // whole-pixel offsets walk the `fetch_block_whole_pixel` copy
    // path; the non-zero fractional offsets walk `sixtap_2d` against
    // the `fetch_block_halo` 9×9 edge-replicated halo. ----
    if use_whole_pixel_evaluator {
        // Sweep the 5 whole-pixel offsets around the snapped center
        // (NSWE + center) — `mb_luma_sad_at_whole_mv` debug-asserts
        // its input is on the whole-pixel grid.
        let whole_center = Mv {
            row: snap_whole(clamp_to_range(center.row as i32)),
            col: snap_whole(clamp_to_range(center.col as i32)),
        };
        let step = WHOLE_PIXEL_STEP as i32;
        let probes: [(i32, i32); 5] = [(0, 0), (-step, 0), (step, 0), (0, -step), (0, step)];
        for (drow, dcol) in probes {
            let cand = Mv {
                row: clamp_to_range(whole_center.row as i32 + drow),
                col: clamp_to_range(whole_center.col as i32 + dcol),
            };
            // Snap the clamped result back onto the whole-pixel grid
            // — §17.1's ±1023 boundary is not a multiple of
            // WHOLE_PIXEL_STEP.
            let cand = Mv {
                row: snap_whole(cand.row),
                col: snap_whole(cand.col),
            };
            let _ = mb_luma_sad_at_whole_mv(reference, mb_col, mb_row, &src_y, cand);
        }
    } else {
        // Sweep the 9 quarter-pixel offsets around the snapped
        // center (3×3 ring at QUARTER_PIXEL_STEP). The non-zero
        // fractional candidates walk the §18.3 sixtap synthesis.
        let qp_center = Mv {
            row: clamp_to_range(center.row as i32),
            col: clamp_to_range(center.col as i32),
        };
        let qstep = QUARTER_PIXEL_STEP as i32;
        for drow in [-qstep, 0, qstep] {
            for dcol in [-qstep, 0, qstep] {
                let cand = Mv {
                    row: clamp_to_range(qp_center.row as i32 + drow),
                    col: clamp_to_range(qp_center.col as i32 + dcol),
                };
                let _ = mb_luma_sad_at_mv(reference, mb_col, mb_row, &src_y, cand);
            }
        }
    }
});
