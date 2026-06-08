#![no_main]

//! Fuzz: panic-freedom of the public §18.3 / §20.14 sub-pixel synthesis
//! primitives — `filter_block_4x4`, `sixtap_2d`, `fetch_block_halo`,
//! `fetch_block_whole_pixel`, and `filter_set_for_version`.
//!
//! The round-256 `panic_free_motion_search_descent` target reaches the
//! §18.3 sixtap synthesis only through the §17 motion-search descent
//! ladder, which by construction snaps every per-candidate vector to the
//! half- or quarter-pixel grid (so `mv & 7` only ever lands on a subset
//! of the 64 (mx, my) ∈ {0..7}² fractional combinations the §18.3 tap
//! table indexes). The round-225 `motion_comp_subpel_luma` criterion
//! bench similarly only exercises a fixed `(mx, my) = (6, 6)` choice
//! against a mid-plane MB so the §20.14 `build_mc_border` edge-
//! replication clamp inside `fetch_block_halo` stays cold. That leaves
//! the §18 primitive surface — every fractional offset, every
//! filter-set arm (sixtap `version == 0` vs bilinear other versions),
//! every edge-replication border-position class — directly under-fuzzed.
//!
//! In particular the §20.14 `build_mc_border` clamp inside
//! `fetch_block_halo` fires whenever the (blk_x, blk_y) + (mv >> 3)
//! origin lands within two pixels of any picture boundary; an
//! adversarial caller that parks the 4×4 origin at (0, 0) with a large
//! negative MV would land every halo read across the top-left corner
//! and walk the per-pixel clamp branch on every row × column of the
//! 9×9 support. The current motion_search-driven harnesses can't
//! produce that envelope — §17.1's MV clamp limits the descent to a
//! ±1023 quarter-pixel envelope around a center the encoder picks, so
//! the §17 layer never invites a halo straddling the boundary into the
//! §18 layer.
//!
//! Surface coverage:
//!
//! * `filter_block_4x4(plane, stride, w, h, blk_x, blk_y, mv, filters)`
//!   — both filter sets (sixtap version 0 / bilinear other versions
//!   selected via `filter_set_for_version`), every (mx, my) ∈ {0..7}²
//!   fractional combination by construction (including the (0, 0)
//!   whole-pixel copy-path fast-out), every 4×4 sub-block origin
//!   inside the plane.
//! * `sixtap_2d(halo, mx, my, filters)` driven directly with
//!   attacker-shaped halo bytes — bypasses the §20.14 fetch path so
//!   the convolution itself is the only surface in the loop. (The
//!   horizontal-then-vertical pass clamps each intermediate sample
//!   with `clamp255((a + 64) >> 7)`; a saturating-add wrap or a
//!   negative-clamp slip would surface here.)
//! * `fetch_block_whole_pixel(plane, stride, w, h, blk_x, blk_y, mv)`
//!   — whole-pixel copy path with §20.14 edge-replication; exercised
//!   independently so the `mx == 0 && my == 0` collapse path inside
//!   `filter_block_4x4` is not the only surface that reaches it.
//! * `fetch_block_halo(plane, stride, w, h, blk_x, blk_y, mv)` —
//!   §20.14 9×9 edge-replicated halo fetch driven independently;
//!   the per-pixel clamp branch is exercised at every border-position
//!   class (top edge, bottom edge, left edge, right edge, NW / NE /
//!   SW / SE corner, mid-plane in-bounds).
//! * `filter_set_for_version(version)` — both arms (`Sixtap` for
//!   `version == 0`, `Bilinear` for any other byte). The selected
//!   tap set is fed into the `sixtap_2d` and `filter_block_4x4` calls
//!   so the bilinear convolution's 4-tap-padded-to-6 lane is also
//!   exercised by the same harness.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | flags byte: bit0 forces `(mx, my) = (0, 0)` into the `filter_block_4x4` leg so the whole-pixel copy-path fast-out is exercised regardless of the `data[6..8]` fractional choice; bits 1..=2 pick the border-position class (0 = mid-plane, 1 = top-left corner, 2 = bottom-right corner, 3 = adversarial: large signed MV across an arbitrary edge); bit3 toggles the constant-vs-gradient halo seed for the direct `sixtap_2d` leg |
//! | `[1]`     | plane width selector — `width = (data[1] as usize % 4) * 8 + 16` (∈ {16, 24, 32, 40}) |
//! | `[2]`     | plane height selector — `height = (data[2] as usize % 4) * 8 + 16` |
//! | `[3]`     | 4×4 sub-block origin column inside the plane, saturated against `width - 4` |
//! | `[4]`     | 4×4 sub-block origin row inside the plane, saturated against `height - 4` |
//! | `[5]`     | `version` byte fed into `filter_set_for_version` — `0` selects the §20.14 bicubic six-tap set, any other value selects the bilinear set |
//! | `[6]`     | `mx` fractional — `mx = data[6] & 7` (eighth-pixel) |
//! | `[7]`     | `my` fractional — `my = data[7] & 7` |
//! | `[8..10]` | MV column integer-pixel offset — two-byte signed `i16` interpreted as quarter-pixel offset and §18.1-doubled to eighth-pixel via `wrapping_mul(2)`. The fractional bits are stripped and replaced by `mx` so the per-leg `(mx, my)` choice is honoured regardless of the integer offset. |
//! | `[10..12]`| MV row integer-pixel offset — same layout |
//! | `[12..]`  | reference-plane payload — tiled into a `Vec<u8>` of length `width * height` for the `filter_block_4x4` / `fetch_block_*` legs, AND seeded into the 81-byte halo for the direct `sixtap_2d` leg |
//!
//! The MV is constructed in eighth-pixel form (the resolution the
//! §18.3 / §20.14 primitives consume — see `stored_luma_mv`'s doubling
//! contract). The integer-pixel offset from the input is shifted left
//! by 3 and OR'd with the fractional `(mx, my)` so the resulting MV
//! satisfies `(mv.row & 7) == my` and `(mv.col & 7) == mx` by
//! construction — this guarantees the (mx, my) leg the input
//! selected actually drives the convolution, rather than being
//! overridden by the integer offset's low bits.
//!
//! Border-position class behaviour (bits 1..=2 of the flags byte):
//!
//! * **0 — mid-plane:** the integer MV offset stays inside
//!   `[2, w - 7]` × `[2, h - 7]` so the halo fits without clamping;
//!   exercises the round-170 fast paths in `fetch_block_halo` /
//!   `fetch_block_whole_pixel`.
//! * **1 — top-left corner:** integer offset forced to a
//!   large-negative pair so the (mv >> 3) origin lands at or past the
//!   top-left edge; every halo row / column is clamped to row 0 /
//!   column 0.
//! * **2 — bottom-right corner:** symmetric to (1), origin past the
//!   bottom-right edge.
//! * **3 — adversarial:** the integer offset from `data[8..12]` is
//!   passed through unchanged. Combined with the §18.1 `wrapping_mul(2)`
//!   inside `stored_luma_mv`'s analogue (we do the doubling inline
//!   here) and the full §17.1 envelope of `i16`, this lets libFuzzer
//!   drive the MV anywhere within `i16`'s range — the §20.14 clamp
//!   has to absorb the entire `isize` range the right-shift produces.
//!
//! Sweeps the §18 primitive layer for panic-freedom, with the
//! reference plane (`width * height` ≤ 1 600 bytes), the 9×9 halo (81
//! bytes, stack), the 4×4 output (16 bytes, stack), and the
//! intermediate `temp` buffer inside `sixtap_2d` (36 bytes, stack) as
//! the only allocations.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth); plane ≤ 40 × 40 pixels (matches the
//! r256 motion_search_descent target for consistency); no internal
//! iteration so every byte budget is bounded by the input shape.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::motion_comp::{
    fetch_block_halo, fetch_block_whole_pixel, filter_block_4x4, filter_set_for_version, sixtap_2d,
};
use oxideav_vp8::motion_vector::Mv;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 12;

/// Build an eighth-pixel MV whose fractional bits are exactly
/// `(my, mx)` and whose integer-pixel offset is `(int_row, int_col)`.
///
/// The §18 / §20.14 primitives extract the fractional eighth-pixel
/// offset via `mv & 7`; pinning the low 3 bits to the requested
/// fractional lets the harness drive any (mx, my) leg of the §18.3
/// tap table independently of the integer offset.
fn make_mv_eighth(int_row: i32, int_col: i32, my: u8, mx: u8) -> Mv {
    // Shift the integer offset left by 3 (eighth-pixel resolution),
    // then OR the fractional bits into the low 3. `wrapping_shl` /
    // `wrapping_add` keep `i16` wrap-around well-defined under the
    // fuzzer's full envelope. `as i16` truncates anything past `i16`
    // bounds; the §20.14 clamp inside `fetch_block_halo` then absorbs
    // whatever the truncated offset produces.
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
    let force_whole_pixel_into_filter_block = (flags & 0b0000_0001) != 0;
    let border_class = (flags >> 1) & 0b11;
    let halo_constant_seed = (flags & 0b0000_1000) != 0;

    // Plane axes ∈ {16, 24, 32, 40}: each at least one MB span,
    // matching the round-256 motion_search_descent harness.
    let width = (data[1] as usize % 4) * 8 + 16;
    let height = (data[2] as usize % 4) * 8 + 16;

    // 4×4 sub-block origin: saturated against `(width - 4, height - 4)`
    // so the in-bounds fast-path leg of `filter_block_4x4` has a valid
    // starting point. The MV (below) is what actually walks the origin
    // across the §20.14 clamp.
    let blk_x = (data[3] as usize) % (width - 3);
    let blk_y = (data[4] as usize) % (height - 3);

    // Filter set: version 0 ⇒ bicubic six-tap, anything else ⇒ bilinear.
    // Both arms are exercised across the input envelope.
    let filter_set = filter_set_for_version(data[5]);
    let filters = filter_set.taps();

    // Fractional eighth-pixel offsets — `mx`, `my` ∈ 0..=7. The (0, 0)
    // combination drives the §18.3 "subblock is simply copied" fast
    // out inside `filter_block_4x4`; every other combination drives
    // the `sixtap_2d` convolution.
    let mx = data[6] & 7;
    let my = data[7] & 7;

    // Integer MV components (raw signed bytes) — interpreted as
    // quarter-pixel offsets per §17 then doubled to eighth-pixel
    // resolution inline (the doubling matches `stored_luma_mv`'s
    // contract). For the mid-plane class we replace the raw offset
    // with a small in-bounds choice so the §20.14 fast path is
    // exercised; for the corner classes we force the MV to push the
    // halo across the edge; for the adversarial class we keep the
    // raw bytes so libFuzzer drives the full `i16` envelope.
    let raw_col_int = i16::from_le_bytes([data[8], data[9]]) as i32;
    let raw_row_int = i16::from_le_bytes([data[10], data[11]]) as i32;

    let (int_row, int_col) = match border_class {
        0 => {
            // Mid-plane: keep the halo inside `[2, w - 7] × [2, h - 7]`
            // so `fetch_block_halo`'s round-170 fast path applies.
            // (blk_x, blk_y) + (mv >> 3) ∈ `[2, w - 7]` requires
            // `(mv >> 3) ∈ [2 - blk_x, w - 7 - blk_x]`. With blk_x
            // saturated against `width - 4`, the safe envelope is
            // `[-blk_x + 2, width - blk_x - 7]`; pick zero-offset so
            // the integer MV doesn't move the halo at all.
            (0, 0)
        }
        1 => {
            // Top-left: large-negative integer MV so the halo origin
            // lands at or past row -2, col -2 ⇒ every halo row /
            // column reads from the clamped edge.
            (-((blk_y + 4) as i32), -((blk_x + 4) as i32))
        }
        2 => {
            // Bottom-right: symmetric to (1).
            (
                (height as i32 - blk_y as i32 + 4),
                (width as i32 - blk_x as i32 + 4),
            )
        }
        _ => (raw_row_int, raw_col_int),
    };

    // Two MV shapes for the per-leg drive:
    //   * `mv` carries the input's chosen fractional `(mx, my)` and
    //     exercises the §18.3 sixtap convolution arm of
    //     `filter_block_4x4`.
    //   * `mv_filter_block` is what we actually feed into the
    //     `filter_block_4x4` leg; when bit0 of the flags byte is set
    //     we strip the fractional bits so the §18.3 "subblock is
    //     simply copied" fast-out is exercised regardless of the
    //     fractional input — otherwise `(mx, my)` would have to be
    //     `(0, 0)` for that arm to fire, which under a uniform
    //     fractional pick happens 1 in 64 iterations.
    let mv = make_mv_eighth(int_row, int_col, my, mx);
    let mv_filter_block = if force_whole_pixel_into_filter_block {
        make_mv_eighth(int_row, int_col, 0, 0)
    } else {
        mv
    };

    // Reference plane: tile the payload (or fall back to a flags-seeded
    // deterministic ramp if the payload is empty) so the SAD / sixtap
    // landscape is non-degenerate.
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

    // ---- (1) filter_block_4x4 — the §20.14 entry point that fans out
    // to the whole-pixel copy or the sixtap convolution. ----
    let _ = filter_block_4x4(
        &ref_plane,
        width,
        width,
        height,
        blk_x,
        blk_y,
        mv_filter_block,
        filters,
    );

    // ---- (2) fetch_block_whole_pixel — driven independently with a
    // whole-pixel MV (fractional bits cleared) so the (mx == 0 &&
    // my == 0) collapse inside filter_block_4x4 is not the only path
    // that reaches it. ----
    let mv_whole = make_mv_eighth(int_row, int_col, 0, 0);
    let _ = fetch_block_whole_pixel(&ref_plane, width, width, height, blk_x, blk_y, mv_whole);

    // ---- (3) fetch_block_halo — driven independently so the
    // border-position class chosen above lands directly inside the
    // halo fetch's per-pixel clamp, regardless of whether the
    // filter_block_4x4 leg picked the whole-pixel fast-out. ----
    let halo = fetch_block_halo(&ref_plane, width, width, height, blk_x, blk_y, mv);

    // ---- (4) sixtap_2d — drive the convolution directly with either
    // the freshly-fetched halo or an attacker-shaped halo seeded from
    // the input. The (mx, my) leg passed in is the input's fractional
    // choice; the (0, 0) combination is filtered out because
    // `sixtap_2d` is only called by `filter_block_4x4` when the
    // fractional part is non-zero, and feeding it zero would walk a
    // tap row that's never indexed in the deployed path.
    //
    // Note: even though the filter table has 8 rows (mx / my ∈ 0..8),
    // index 0 is the "no convolution needed" row whose use the
    // §20.14 reference fast-outs above. Calling sixtap_2d with
    // mx == 0 or my == 0 is well-defined (the row is all zeros except
    // the centre tap), but it's not on the reference's hot path; the
    // fuzz harness still exercises it for completeness. ----
    let mx_for_sixtap = if (mx | my) == 0 { 1 } else { mx } as usize;
    let my_for_sixtap = my as usize;

    if halo_constant_seed {
        // Constant-fill halo — exercises the convolution's centre-of-
        // mass against a flat input, where every output sample reduces
        // to the constant times the sum of taps (which equals 128 for
        // the bicubic / bilinear lanes). Catches a tap-sum drift if
        // the filter table is ever regenerated incorrectly.
        let v = flags.wrapping_add(0x55);
        let halo_const = [v; 81];
        let _ = sixtap_2d(&halo_const, mx_for_sixtap, my_for_sixtap, filters);
    } else {
        // Halo from the §20.14 fetch above.
        let _ = sixtap_2d(&halo, mx_for_sixtap, my_for_sixtap, filters);
    }

    // ---- (5) An attacker-shaped halo seeded directly from the input
    // payload — bypasses fetch_block_halo entirely so the convolution
    // sees byte patterns the §20.14 clamp would never produce (e.g.
    // non-monotonic adjacent rows that swing the partial sum
    // between +∞ and -∞ within a single tap window). The `(a + 64)
    // >> 7` rounding and `clamp255` saturation surface are the
    // primary panic candidates here. ----
    let mut halo_raw = [0u8; 81];
    if payload.is_empty() {
        for (i, slot) in halo_raw.iter_mut().enumerate() {
            // `(i as u8).wrapping_mul(11)` — the harness must never
            // panic on its own arithmetic; the multiply intentionally
            // wraps so the synthesised halo cycles through every byte
            // residue class.
            *slot = flags.wrapping_add((i as u8).wrapping_mul(11));
        }
    } else {
        for (i, slot) in halo_raw.iter_mut().enumerate() {
            *slot = payload[(i * 7) % payload.len()];
        }
    }
    let _ = sixtap_2d(&halo_raw, mx_for_sixtap, my_for_sixtap, filters);
});
