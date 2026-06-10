#![no_main]

//! Fuzz: panic-freedom + write-equivalence of the §16.4 SPLITMV
//! strided-write motion-comp primitive `filter_block_4x4_into` landed in
//! round 274.
//!
//! `filter_block_4x4_into` is the strided-write companion of
//! `filter_block_4x4`: instead of returning a fixed `[u8; 16]` block the
//! caller then re-copies row-by-row into a strided macroblock buffer, it
//! synthesises one 4×4 sub-block (RFC 6386 §20.14 `filter_block`) and
//! writes it directly into a destination raster at `(dst_x, dst_y)` with an
//! arbitrary `dst_stride` — the whole-pixel branch copies source rows
//! straight in (§18.3 "simply copied" / §20.14 `build_mc_border` edge
//! replication on the border-straddle path), the sub-pixel branch
//! delegates to `filter_block_4x4` verbatim and writes strided.
//!
//! It landed AFTER the round-257 `panic_free_sixtap_subpel` target (which
//! drives `filter_block_4x4` itself) and the round-273
//! `panic_free_mb_batch_motion_comp` target (which drives the MB-scale
//! batched fetchers), so no existing harness reaches it: the in-tree
//! round-274 equivalence tests
//! (`filter_block_4x4_into_matches_filter_block_4x4`,
//! `strided_into_assembly_matches_predict_split_mv`) drive only fixed
//! inputs on a single mid-plane plane geometry, and the decode / encode
//! stack never exposes the destination-raster triple (`dst_x`, `dst_y`,
//! `dst_stride`) to an attacker — `predict_split_mv` keeps the shipped
//! scratch-copy form, so `filter_block_4x4_into` is reached only as a
//! retained public primitive.
//!
//! This target drives that surface directly with an attacker-shaped
//! `(plane dimension, block origin, mv, fraction, filter set,
//! border-position class, destination geometry)` envelope and asserts the
//! write-equivalence contract on every iteration (panic on mismatch — any
//! drift between the strided write and the `filter_block_4x4`-then-copy
//! reference surfaces as a harness `assert_eq!`):
//!
//! * The 4×4 footprint written at `(dst_x, dst_y)` must equal, byte for
//!   byte, the `[u8; 16]` block `filter_block_4x4` returns for the same
//!   `(plane, blk_x, blk_y, mv, filters)` — the round-274
//!   `filter_block_4x4_into_matches_filter_block_4x4` invariant, here under
//!   the attacker-shaped clamp + destination-geometry envelope the
//!   mid-plane-only in-tree test never reaches.
//! * Every destination byte OUTSIDE the 4×4 footprint must be untouched
//!   (the pre-fill sentinel survives) — a regression that strided past the
//!   four-row / four-column window (a wrong `dst_stride` multiply, or a
//!   copy length other than 4) corrupts a neighbour and is caught here.
//!
//! Both whole-pixel (`mv & 7 == 0`, the copy branch) and sub-pixel
//! (`sixtap_2d`) vectors are driven, across every border class:
//! mid-plane fast path, top-left corner, bottom-right corner, and an
//! adversarial full-`i16` MV that parks the source origin anywhere in the
//! `i16` envelope so the §20.14 clamp absorbs it.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | flags: bits 0..=1 border-position class (0 mid-plane, 1 top-left, 2 bottom-right, 3 adversarial); bit2 forces a whole-pixel MV (copy branch) when set, else keeps the `[6]`/`[7]` fractions |
//! | `[1]`     | plane width selector — `w = (data[1] % 6) * 8 + 16` (∈ {16, 24, 32, 40, 48, 56}) |
//! | `[2]`     | plane height selector — same shape |
//! | `[3]`     | block column origin (in 4-px sub-block units), saturated into the plane |
//! | `[4]`     | block row origin (in 4-px sub-block units), saturated into the plane |
//! | `[5]`     | `version` byte → `filter_set_for_version` (0 = six-tap, else bilinear) |
//! | `[6]`     | `mx` fractional — `data[6] & 7` |
//! | `[7]`     | `my` fractional — `data[7] & 7` |
//! | `[8..10]` | MV column integer-pixel offset (signed `i16`) |
//! | `[10..12]`| MV row integer-pixel offset (signed `i16`) |
//! | `[12]`    | destination stride selector — `dst_stride = 16 + (data[12] % 17)` (∈ 16..=32, always ≥ 4 + dst_x cushion) |
//! | `[13]`    | destination x-origin — `dst_x = data[13] % (dst_stride - 4 + 1)` (keeps the 4-col footprint in-row) |
//! | `[14]`    | destination y-origin — `dst_y = data[14] % 9` (≤ 8 spare rows below) |
//! | `[15..]`  | reference-plane payload tiled into the plane |
//!
//! Hard caps: input ≤ 4 KiB; plane ≤ 56 × 56 (≤ 3 136 bytes); the
//! destination raster is `dst_stride × (dst_y + 4 + 4)` ≤ 32 × 17 ≤ 544
//! bytes; all heap buffers bounded; no internal iteration beyond the fixed
//! 4×4 footprint compare.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::motion_comp::{filter_block_4x4, filter_block_4x4_into, filter_set_for_version};
use oxideav_vp8::motion_vector::Mv;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 15;

/// Build an eighth-pixel MV whose low three bits are exactly `(my, mx)`
/// and whose integer-pixel offset is `(int_row, int_col)`. Wrapping
/// shifts keep the `i16` truncation well-defined under the fuzzer's full
/// envelope; the §20.14 clamp inside the fetch absorbs whatever origin the
/// truncated offset produces.
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
    let force_whole_pixel = flags & 0b100 != 0;

    // Plane axes ∈ {16, 24, 32, 40, 48, 56}: at least one full 4×4 grid.
    let w = (data[1] as usize % 6) * 8 + 16;
    let h = (data[2] as usize % 6) * 8 + 16;

    // Block origin on the 4-px sub-block grid, saturated into the plane.
    let blk_cols = w / 4;
    let blk_rows = h / 4;
    let blk_x = ((data[3] as usize) % blk_cols) * 4;
    let blk_y = ((data[4] as usize) % blk_rows) * 4;

    let filter_set = filter_set_for_version(data[5]);
    let filters = filter_set.taps();

    // Fractions: zero them when the whole-pixel (copy) branch is forced.
    let (mx, my) = if force_whole_pixel {
        (0u8, 0u8)
    } else {
        (data[6] & 7, data[7] & 7)
    };

    let raw_col_int = i16::from_le_bytes([data[8], data[9]]) as i32;
    let raw_row_int = i16::from_le_bytes([data[10], data[11]]) as i32;

    // Border-position class — choose an integer offset that parks the
    // sub-block support mid-plane (fast path), across the top-left or
    // bottom-right edge (every §20.14 clamp branch), or anywhere in the
    // full `i16` envelope (adversarial).
    let (int_row, int_col) = match border_class {
        0 => (0, 0),
        1 => (-((blk_y + 4) as i32), -((blk_x + 4) as i32)),
        2 => ((h as i32 - blk_y as i32 + 4), (w as i32 - blk_x as i32 + 4)),
        _ => (raw_row_int, raw_col_int),
    };

    let mv = make_mv_eighth(int_row, int_col, my, mx);

    // Reference plane: tile the payload (or a flags-seeded ramp when the
    // payload is empty) so the convolution landscape is non-degenerate.
    let payload = &data[HEADER_BYTES..];
    let seed_byte = flags;
    let plane: Vec<u8> = if payload.is_empty() {
        (0..w * h)
            .map(|i| (i as u8).wrapping_add(seed_byte))
            .collect()
    } else {
        (0..w * h).map(|i| payload[i % payload.len()]).collect()
    };

    // Destination raster geometry — attacker-shaped but kept large enough
    // that the 4×4 footprint at (dst_x, dst_y) is fully in-bounds.
    let dst_stride = 16 + (data[12] as usize % 17); // ∈ 16..=32
    let dst_x = (data[13] as usize) % (dst_stride - 4 + 1); // footprint stays in-row
    let dst_y = (data[14] as usize) % 9; // ≤ 8 spare rows below
    let dst_rows = dst_y + 4 + 4; // four written rows + cushion
    const SENTINEL: u8 = 0xA5;
    let mut dst = vec![SENTINEL; dst_stride * dst_rows];

    // ---- drive the strided-write primitive --------------------------
    filter_block_4x4_into(
        &mut dst, dst_stride, dst_x, dst_y, &plane, w, w, h, blk_x, blk_y, mv, filters,
    );

    // Reference: the round-274 contract is byte-identity with
    // `filter_block_4x4` then a four-row strided copy.
    let expected = filter_block_4x4(&plane, w, w, h, blk_x, blk_y, mv, filters);

    for r in 0..4 {
        for c in 0..4 {
            let d = (dst_y + r) * dst_stride + dst_x + c;
            assert_eq!(
                dst[d],
                expected[r * 4 + c],
                "strided-write mismatch mv={mv:?} blk=({blk_x},{blk_y}) \
                 dst=({dst_x},{dst_y})/{dst_stride} ({r},{c})"
            );
        }
    }

    // Every byte outside the 4×4 footprint must retain the sentinel — a
    // stride / length regression that wrote past the window is caught here.
    for (idx, &b) in dst.iter().enumerate() {
        let row = idx / dst_stride;
        let col = idx % dst_stride;
        let in_footprint = row >= dst_y && row < dst_y + 4 && col >= dst_x && col < dst_x + 4;
        if !in_footprint {
            assert_eq!(
                b, SENTINEL,
                "out-of-footprint corruption at ({row},{col}) mv={mv:?} \
                 dst=({dst_x},{dst_y})/{dst_stride}"
            );
        }
    }
});
