#![no_main]

//! Fuzz: panic-freedom of the MB-scale §18.3 / §20.14 batched
//! motion-compensation primitives landed in rounds 270–272 —
//! `fetch_luma_mb_halo`, `sixtap_mb_luma`, `fetch_chroma_mb_halo`,
//! `sixtap_mb_chroma`, `fetch_luma_mb_whole_pixel`, and
//! `fetch_chroma_mb_whole_pixel`.
//!
//! These six public functions synthesise (or copy) a whole macroblock's
//! luma / chroma prediction in one pass instead of the per-4×4-sub-block
//! assembly the round-257 `panic_free_sixtap_subpel` target drives
//! (`fetch_block_halo` / `sixtap_2d` / `fetch_block_whole_pixel`). They
//! landed AFTER that target was written, so no existing fuzz harness
//! reaches them: the fourteen targets above hit §18 only through either
//! the §17 motion-search descent ladder (which snaps every per-candidate
//! MV to a sub-block grid and never reaches the MB-scale orchestrator) or
//! through `decode_vp8` / `Vp8DecoderState::decode_frame` /
//! `encode_p_frame_multi_ref` (which gate the MB-scale fetch behind a
//! fully-formed reference picture + §9.7 refresh state machine, so the
//! §20.14 `build_mc_border` clamp inside the MB-halo fetch never sees an
//! origin parked across a picture boundary by an arbitrary `i16` vector).
//!
//! This target drives the MB-scale surface directly with an
//! attacker-shaped `(plane dimension, mb origin, mv, fractional offset,
//! filter set, border-position class)` envelope:
//!
//! * **`fetch_luma_mb_halo`** — the 21×21 edge-replicated luma halo, then
//!   **`sixtap_mb_luma`** convolving it into a 16×16 block. Every border
//!   class (mid-plane fast path, top-left corner, bottom-right corner,
//!   adversarial full-`i16` MV) and every `(mx, my) ∈ {0..7}²` fraction.
//! * **`fetch_chroma_mb_halo`** — the 13×13 chroma halo, then
//!   **`sixtap_mb_chroma`** convolving it into an 8×8 block. Same envelope.
//! * **`fetch_luma_mb_whole_pixel`** / **`fetch_chroma_mb_whole_pixel`**
//!   — the whole-pixel non-SPLITMV copy paths (round 272), driven with a
//!   whole-pixel MV (`mv & 7 == 0`) so the contiguous-copy branch fires.
//!
//! Equivalence cross-checks (panic on mismatch — any drift between the
//! batched and per-sub-block paths surfaces as a harness `assert_eq!`):
//!
//! * The 21×21 MB luma halo must contain each per-sub-block 9×9
//!   `fetch_block_halo` window at offset `(sb*4, sc*4)` — the round-270
//!   `fetch_luma_mb_halo_matches_subblock_halos_in_bounds` invariant,
//!   re-asserted here on attacker-shaped (plane, mv) inputs INCLUDING the
//!   border-clamp envelope the in-tree test (mid-plane only) never reaches.
//! * The 13×13 MB chroma halo must contain each per-sub-block 9×9 window
//!   at offset `(sb*4, sc*4)` — the chroma analogue.
//! * The whole-pixel MB luma/chroma copy must equal the per-sub-block
//!   `fetch_block_whole_pixel` assembly tiled into the MB raster.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | flags: bits 0..=1 border-position class (0 mid-plane, 1 top-left, 2 bottom-right, 3 adversarial); bit2 selects luma-vs-chroma emphasis for the whole-pixel leg's MV magnitude (cosmetic) |
//! | `[1]`     | luma plane width selector — `w = (data[1] % 4) * 16 + 32` (∈ {32, 48, 64, 80}) |
//! | `[2]`     | luma plane height selector — same shape |
//! | `[3]`     | MB column index (in 16-px luma MB units), saturated into the plane |
//! | `[4]`     | MB row index (in 16-px luma MB units), saturated into the plane |
//! | `[5]`     | `version` byte → `filter_set_for_version` (0 = six-tap, else bilinear) |
//! | `[6]`     | `mx` fractional — `data[6] & 7` |
//! | `[7]`     | `my` fractional — `data[7] & 7` |
//! | `[8..10]` | MV column integer-pixel offset (signed `i16`) |
//! | `[10..12]`| MV row integer-pixel offset (signed `i16`) |
//! | `[12..]`  | reference-plane payload tiled into the luma + chroma planes |
//!
//! The chroma plane is the standard 4:2:0 half-resolution
//! (`w.div_ceil(2) × h.div_ceil(2)`), and the chroma MB origin /
//! dimensions are derived from the luma ones so both fetches see a
//! consistent geometry.
//!
//! Hard caps: input ≤ 4 KiB; luma plane ≤ 80 × 80 (≤ 6 400 bytes), chroma
//! ≤ 40 × 40; all batched buffers (441-byte luma halo, 169-byte chroma
//! halo, 256/64-byte outputs) are stack-resident; no internal iteration
//! beyond the fixed 4×4 / 2×2 sub-block cross-check loops.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::motion_comp::{
    fetch_block_halo, fetch_block_whole_pixel, fetch_chroma_mb_halo, fetch_chroma_mb_whole_pixel,
    fetch_luma_mb_halo, fetch_luma_mb_whole_pixel, filter_set_for_version, sixtap_mb_chroma,
    sixtap_mb_luma,
};
use oxideav_vp8::motion_vector::Mv;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 12;

/// Build an eighth-pixel MV whose low three bits are exactly `(my, mx)`
/// and whose integer-pixel offset is `(int_row, int_col)`. Wrapping
/// shifts keep the `i16` truncation well-defined under the fuzzer's full
/// envelope; the §20.14 clamp inside the fetch absorbs whatever origin
/// the truncated offset produces.
fn make_mv_eighth(int_row: i32, int_col: i32, my: u8, mx: u8) -> Mv {
    let row_i32 = int_row.wrapping_shl(3).wrapping_add(my as i32);
    let col_i32 = int_col.wrapping_shl(3).wrapping_add(mx as i32);
    Mv {
        row: row_i32 as i16,
        col: col_i32 as i16,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let border_class = flags & 0b11;

    // Luma plane axes ∈ {32, 48, 64, 80}: at least two MB spans so an
    // interior MB origin exists. Chroma is the 4:2:0 half-plane.
    let w = (data[1] as usize % 4) * 16 + 32;
    let h = (data[2] as usize % 4) * 16 + 32;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    // MB origin in pixels. There are `w / 16` MB columns; saturate the
    // selector into `[0, mb_cols)` then convert to a 16-px luma origin /
    // 8-px chroma origin.
    let mb_cols = w / 16;
    let mb_rows = h / 16;
    let mb_col = (data[3] as usize) % mb_cols;
    let mb_row = (data[4] as usize) % mb_rows;
    let luma_mb_x = mb_col * 16;
    let luma_mb_y = mb_row * 16;
    let chroma_mb_x = mb_col * 8;
    let chroma_mb_y = mb_row * 8;

    let filter_set = filter_set_for_version(data[5]);
    let filters = filter_set.taps();

    let mx = (data[6] & 7) as usize;
    let my = (data[7] & 7) as usize;

    let raw_col_int = i16::from_le_bytes([data[8], data[9]]) as i32;
    let raw_row_int = i16::from_le_bytes([data[10], data[11]]) as i32;

    // Border-position class — choose an integer offset that parks the
    // MB halo mid-plane (round-170/270 fast path), across the top-left or
    // bottom-right edge (every §20.14 clamp branch), or anywhere in the
    // full `i16` envelope (adversarial).
    let (int_row, int_col) = match border_class {
        0 => (0, 0),
        1 => (-((luma_mb_y + 8) as i32), -((luma_mb_x + 8) as i32)),
        2 => (
            (h as i32 - luma_mb_y as i32 + 8),
            (w as i32 - luma_mb_x as i32 + 8),
        ),
        _ => (raw_row_int, raw_col_int),
    };

    // Reference planes: tile the payload (or a flags-seeded ramp when the
    // payload is empty) so the convolution landscape is non-degenerate.
    let payload = &data[HEADER_BYTES..];
    let seed_byte = flags;
    let fill = |len: usize| -> Vec<u8> {
        if payload.is_empty() {
            (0..len)
                .map(|i| (i as u8).wrapping_add(seed_byte))
                .collect()
        } else {
            (0..len).map(|i| payload[i % payload.len()]).collect()
        }
    };
    let luma = fill(w * h);
    let chroma = fill(cw * ch);

    // The chroma MV is the §18.1 averaged vector; for fuzzing we drive
    // the chroma fetch with the same eighth-pixel MV so both planes see a
    // consistent integer + fractional offset (the cross-checks below only
    // require the same MV reaches the batched and per-sub-block paths).
    let mv = make_mv_eighth(int_row, int_col, my as u8, mx as u8);

    // ---- luma: 21×21 halo + 16×16 convolution -----------------------
    let luma_halo = fetch_luma_mb_halo(&luma, w, w, h, luma_mb_x, luma_mb_y, mv);
    let _luma_block = sixtap_mb_luma(&luma_halo, mx, my, filters);

    // Cross-check: the 21×21 MB halo contains every per-sub-block 9×9
    // `fetch_block_halo` window at offset (sb*4, sc*4) — the round-270
    // containment invariant, now under the attacker-shaped clamp envelope.
    for sb in 0..4 {
        for sc in 0..4 {
            let sub = fetch_block_halo(&luma, w, w, h, luma_mb_x + sc * 4, luma_mb_y + sb * 4, mv);
            for r in 0..9 {
                for c in 0..9 {
                    assert_eq!(
                        luma_halo[(sb * 4 + r) * 21 + (sc * 4 + c)],
                        sub[r * 9 + c],
                        "luma halo mismatch sb={sb} sc={sc} ({r},{c})"
                    );
                }
            }
        }
    }

    // ---- chroma: 13×13 halo + 8×8 convolution -----------------------
    let chroma_halo = fetch_chroma_mb_halo(&chroma, cw, cw, ch, chroma_mb_x, chroma_mb_y, mv);
    let _chroma_block = sixtap_mb_chroma(&chroma_halo, mx, my, filters);

    // Cross-check: the 13×13 MB halo contains every per-sub-block 9×9
    // window at offset (sb*4, sc*4) for the four chroma sub-blocks.
    for sb in 0..2 {
        for sc in 0..2 {
            let sub = fetch_block_halo(
                &chroma,
                cw,
                cw,
                ch,
                chroma_mb_x + sc * 4,
                chroma_mb_y + sb * 4,
                mv,
            );
            for r in 0..9 {
                for c in 0..9 {
                    assert_eq!(
                        chroma_halo[(sb * 4 + r) * 13 + (sc * 4 + c)],
                        sub[r * 9 + c],
                        "chroma halo mismatch sb={sb} sc={sc} ({r},{c})"
                    );
                }
            }
        }
    }

    // ---- whole-pixel copy paths (round 272) -------------------------
    // Strip the fractional bits so the contiguous-copy branch fires.
    let wp_mv = make_mv_eighth(int_row, int_col, 0, 0);

    let luma_wp = fetch_luma_mb_whole_pixel(&luma, w, w, h, luma_mb_x, luma_mb_y, wp_mv);
    // Equals the per-sub-block `fetch_block_whole_pixel` assembly tiled
    // into the 16×16 MB raster.
    for sb in 0..4 {
        for sc in 0..4 {
            let sub = fetch_block_whole_pixel(
                &luma,
                w,
                w,
                h,
                luma_mb_x + sc * 4,
                luma_mb_y + sb * 4,
                wp_mv,
            );
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(
                        luma_wp[(sb * 4 + r) * 16 + (sc * 4 + c)],
                        sub[r * 4 + c],
                        "luma whole-pixel mismatch sb={sb} sc={sc} ({r},{c})"
                    );
                }
            }
        }
    }

    let chroma_wp =
        fetch_chroma_mb_whole_pixel(&chroma, cw, cw, ch, chroma_mb_x, chroma_mb_y, wp_mv);
    for sb in 0..2 {
        for sc in 0..2 {
            let sub = fetch_block_whole_pixel(
                &chroma,
                cw,
                cw,
                ch,
                chroma_mb_x + sc * 4,
                chroma_mb_y + sb * 4,
                wp_mv,
            );
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(
                        chroma_wp[(sb * 4 + r) * 8 + (sc * 4 + c)],
                        sub[r * 4 + c],
                        "chroma whole-pixel mismatch sb={sb} sc={sc} ({r},{c})"
                    );
                }
            }
        }
    }
});
