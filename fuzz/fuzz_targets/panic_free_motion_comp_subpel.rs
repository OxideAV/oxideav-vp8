#![no_main]

//! Fuzz: panic-freedom of the public §18.3 / §20.14 sub-pixel
//! motion-compensation primitives.
//!
//! The five existing fuzz targets reach §18 only indirectly through
//! `decode_vp8` / `Vp8DecoderState::decode_frame` / `encode_keyframe` /
//! `Vp8TwoPassEncoder`, all of which gate the sub-pel kernels behind a
//! fully-formed inter-frame state machine (reference plane plus a
//! decoded motion vector tree). That leaves the primitive surface
//! itself — `fetch_block_whole_pixel`, `fetch_block_halo`, `sixtap_2d`,
//! `filter_block_4x4`, plus the §18.1 helpers `stored_luma_mv` /
//! `chroma_mv` / `apply_full_pixel` / `whole_pixel_fraction_is_zero` and
//! the §18.3 `filter_set_for_version` / `interp` — directly
//! under-fuzzed: a caller that picks an unfortunate `(plane.len(),
//! stride, w, h, blk_x, blk_y, mv)` tuple could land an out-of-bounds
//! index that the higher-level harnesses never reach.
//!
//! This target exercises that surface with a wide-envelope reference
//! plane and an arbitrary motion vector so libFuzzer can drive every
//! edge-replication clamp in `build_mc_border` (the §20.14 halo /
//! whole-pixel fetch), every saturating arithmetic in `interp` (the
//! `(a + 64) >> 7` rounding plus `clamp255`), and every phase of the
//! 8-position eighth-pel filter tables for both the sixtap (`version
//! == 0`) and bilinear (`version != 0`) sets.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`   | `version_byte` — drives [`filter_set_for_version`] (bit0 picks sixtap vs bilinear; remaining bits are passed through verbatim so the bilinear branch sees the full §9.1 envelope) |
//! | `[1]`   | `w_sel` — plane width selector in 1..=64 px (`1 + (b & 0x3f)`) |
//! | `[2]`   | `h_sel` — plane height selector in 1..=64 px (`1 + (b & 0x3f)`) |
//! | `[3]`   | `stride_pad` — extra padding bytes appended to each row so `stride = w + (b & 0x0f)` (exercises the stride-vs-width split that production decoders use to pad each row to a SIMD boundary) |
//! | `[4]`   | `blk_x` — 4×4 sub-block top-left column in pixels (`b % w`) |
//! | `[5]`   | `blk_y` — 4×4 sub-block top-left row in pixels (`b % h`) |
//! | `[6..8]` | `mv_row` raw little-endian `i16` (full -32768..=32767; clamped by §18.1 callers but the primitives must remain panic-free outside that range) |
//! | `[8..10]` | `mv_col` raw little-endian `i16` (full -32768..=32767) |
//! | `[10]`  | flag byte: bit0 picks the chained `predict_inter_mb_whole_pixel` leg (whole-pixel only); bit1 picks the §18.1 `apply_full_pixel` round-trip on the MV; bit2 picks `chroma_mv` derivation; bit3 toggles whether `stored_luma_mv` is composed in too |
//! | `[11..]` | plane payload — tiled into a `(stride * h)`-byte plane so even short fuzz inputs synthesise a fully-populated reference |
//!
//! The harness:
//!
//! 1. Builds the §18.1 `Mv` from `data[6..10]`, then derives the four
//!    related MVs (`stored_luma_mv`, `apply_full_pixel`, `chroma_mv`)
//!    so every §18.1 helper sees the raw envelope (NOT just §9.1-clamped
//!    values).
//! 2. Tests [`whole_pixel_fraction_is_zero`] against the raw MV — purely
//!    a panic-freedom assertion.
//! 3. Drives [`filter_set_for_version`] with `data[0]` and pulls the
//!    `&[[i32; 6]; 8]` tap table out via [`FilterSet::taps`] so the
//!    sixtap-vs-bilinear branch is sampled per iteration.
//! 4. Calls [`fetch_block_whole_pixel`] with the raw MV (forces the
//!    edge-replication clamp branch when the MV pushes the sub-block
//!    off-plane).
//! 5. Calls [`fetch_block_halo`] with the same MV (forces the 9×9
//!    `build_mc_border` clamp branch).
//! 6. Calls [`sixtap_2d`] on the halo with every `(mx, my)` phase pair
//!    from `mv & 7` and `(mv ^ flags) & 7` — covers a deterministic
//!    pair so the harness exercises BOTH the in-bounds (mid-frame) and
//!    out-of-bounds (border-straddling) halo cases at every phase.
//! 7. Calls [`filter_block_4x4`] — the dispatch wrapper that picks the
//!    whole-pixel fast path on a zero fractional or the `sixtap_2d` slow
//!    path otherwise. Drives BOTH the picked filter set and the *other*
//!    set so the dispatcher's parameter routing is independently
//!    fuzzed.
//! 8. When `data[10] & 1 != 0`, drives [`predict_inter_mb_whole_pixel`]
//!    on the constructed plane: the 16×16 luma / 8×8 chroma full-pel
//!    aggregate of the per-sub-block fetches, including the §18.4
//!    `MotionCompError::SubPixelNotSupported` rejection on a fractional
//!    MV. The §18.4 happy path is also reached via [`predict_inter_mb`]
//!    on the sub-pel leg.
//! 9. Drives [`interp`] in isolation on a synthetic 6-tap support
//!    (`data[11..17]` tiled to 6 bytes) under every row of the chosen
//!    tap table so the `(a + 64) >> 7 -> clamp255` rounding ladder is
//!    sampled across all 8 phases.
//!
//! No oracle, no decoder, no allocations beyond the working plane and
//! the per-phase scratch buffers: pure differential-against-spec sanity
//! at the §18 primitive layer.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth); plane ≤ 80 × 64 ≈ 5 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::motion_comp::{
    apply_full_pixel, chroma_mv, fetch_block_halo, fetch_block_whole_pixel, filter_block_4x4,
    filter_set_for_version, interp, predict_inter_mb, predict_inter_mb_whole_pixel, stored_luma_mv,
    whole_pixel_fraction_is_zero, ReferencePlanes,
};
use oxideav_vp8::motion_vector::Mv;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 11;
const MAX_WIDTH: usize = 64;
const MAX_HEIGHT: usize = 64;
const MAX_STRIDE_PAD: usize = 16;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let version_byte = data[0];
    let w_sel = data[1];
    let h_sel = data[2];
    let stride_pad_sel = data[3];
    let blk_x_sel = data[4];
    let blk_y_sel = data[5];
    let mv_row = i16::from_le_bytes([data[6], data[7]]);
    let mv_col = i16::from_le_bytes([data[8], data[9]]);
    let flags = data[10];

    // Plane geometry: 1..=MAX_WIDTH px wide, 1..=MAX_HEIGHT px tall,
    // stride = width + 0..=MAX_STRIDE_PAD.
    let w = 1 + (w_sel as usize & (MAX_WIDTH - 1));
    let h = 1 + (h_sel as usize & (MAX_HEIGHT - 1));
    let stride = w + (stride_pad_sel as usize & MAX_STRIDE_PAD);
    let plane_len = stride.checked_mul(h).unwrap_or(0);
    if plane_len == 0 || plane_len > MAX_INPUT_BYTES {
        return;
    }

    // Sub-block top-left within the plane. We intentionally allow
    // (blk_x, blk_y) right up to the plane bounds so an out-of-range MV
    // pushes the source position past the picture edge — the
    // `build_mc_border` clamp is what we're fuzzing.
    let blk_x = (blk_x_sel as usize) % w;
    let blk_y = (blk_y_sel as usize) % h;

    // Tile the input payload into a fully-populated plane. Even a short
    // input synthesises a non-zero plane so the edge clamp produces a
    // meaningful filtered result.
    let payload = &data[HEADER_BYTES..];
    let mut plane = vec![0u8; plane_len];
    if payload.is_empty() {
        for (i, slot) in plane.iter_mut().enumerate() {
            *slot = (i as u8) ^ flags;
        }
    } else {
        for (i, slot) in plane.iter_mut().enumerate() {
            *slot = payload[i % payload.len()];
        }
    }

    // §18.1 helpers — exercise the raw envelope (NOT clamped to
    // §17 -1023..=1023). Each function is total over `i16` and may
    // never panic.
    let mv = Mv {
        row: mv_row,
        col: mv_col,
    };
    let _ = whole_pixel_fraction_is_zero(mv);
    let stored = stored_luma_mv(mv);
    let chroma = chroma_mv(mv);
    let full_pel = apply_full_pixel(mv);
    let _ = whole_pixel_fraction_is_zero(stored);
    let _ = whole_pixel_fraction_is_zero(chroma);
    let _ = whole_pixel_fraction_is_zero(full_pel);

    // Pick the MV the rest of the harness drives.
    let driving_mv = match flags & 0b1110 {
        0b0010 => full_pel,
        0b0100 => chroma,
        0b1000 => stored,
        _ => mv,
    };

    // §18.3 filter-set selection. Drive the boolean dispatch with the
    // full version byte (not just bit0) so the bilinear / sixtap
    // boundary is independent of any header normalisation.
    let primary_set = filter_set_for_version(version_byte);
    let secondary_set = filter_set_for_version(version_byte.wrapping_add(1));
    let primary_taps = primary_set.taps();
    let secondary_taps = secondary_set.taps();

    // (1) Whole-pixel fetch — drives the §20.14 `build_mc_border`
    // whole-pixel branch.
    let _ = fetch_block_whole_pixel(&plane, stride, w, h, blk_x, blk_y, driving_mv);

    // (2) Halo fetch — drives the §20.14 9×9 `build_mc_border` halo
    // branch.
    let halo = fetch_block_halo(&plane, stride, w, h, blk_x, blk_y, driving_mv);

    // (3) sixtap_2d on the halo with the deterministic (mx, my) pair
    // derived from the driving MV — both primary and secondary tap
    // sets so the dispatcher's parameter routing is independently
    // sampled per iteration.
    let mx = (driving_mv.col & 7) as usize;
    let my = (driving_mv.row & 7) as usize;
    let _ = oxideav_vp8::motion_comp::sixtap_2d(&halo, mx, my, primary_taps);
    let _ = oxideav_vp8::motion_comp::sixtap_2d(&halo, mx, my, secondary_taps);

    // (4) sixtap_2d cross-phase: drive an alternate `(mx', my')` pair
    // from `(mv ^ flags) & 7` so a single iteration exercises two
    // distinct phases — phase 0 (whole-pel) often dominates input
    // when fuzzer bytes are zero-biased, this guarantees the
    // fractional-phase branch is reached.
    let mx_alt = ((driving_mv.col ^ flags as i16) & 7) as usize;
    let my_alt = ((driving_mv.row ^ flags as i16) & 7) as usize;
    let _ = oxideav_vp8::motion_comp::sixtap_2d(&halo, mx_alt, my_alt, primary_taps);
    let _ = oxideav_vp8::motion_comp::sixtap_2d(&halo, mx_alt, my_alt, secondary_taps);

    // (5) The dispatch wrapper — exercises both the whole-pixel fast
    // path (mx == my == 0) and the sixtap_2d slow path. Both filter
    // sets per call.
    let _ = filter_block_4x4(&plane, stride, w, h, blk_x, blk_y, driving_mv, primary_taps);
    let _ = filter_block_4x4(
        &plane,
        stride,
        w,
        h,
        blk_x,
        blk_y,
        driving_mv,
        secondary_taps,
    );

    // (6) interp on a synthetic 6-tap support across all 8 phases of
    // both tap tables. The support pixels are drawn from the tail of
    // the payload (or a flag-tiled fallback when empty) so the
    // `(a + 64) >> 7` ladder is sampled across the full 0..=255 range,
    // not just the bytes the halo happened to materialise.
    let mut support = [0u8; 6];
    if payload.len() >= 6 {
        support.copy_from_slice(&payload[..6]);
    } else if !payload.is_empty() {
        for (i, b) in support.iter_mut().enumerate() {
            *b = payload[i % payload.len()];
        }
    } else {
        for (i, b) in support.iter_mut().enumerate() {
            *b = flags.wrapping_add(i as u8);
        }
    }
    for phase in 0..8 {
        let _ = interp(&primary_taps[phase], &support);
        let _ = interp(&secondary_taps[phase], &support);
    }

    // (7) Optional 16×16 / 8×8 aggregate via the public MB-level
    // wrappers. Only drive this when the flag bit asks for it (the
    // aggregate allocates a `[u8; 256]` luma + two `[u8; 64]` chroma
    // buffers, so we skip it on most iterations to keep exec/s high).
    //
    // The §18.2 macroblock layer demands a strict MB-grid plane
    // (`y_stride == mb_cols * 16`, `uv_stride == mb_cols * 8`, plane
    // length == `mb_cols * 16 * mb_rows * 16` for luma and the
    // half-resolution analogue for chroma). We build a fresh, tightly
    // sized luma + chroma plane out of the same tiled byte source so
    // the kernel sees a fully-populated MB-grid.
    if flags & 0b0000_0001 != 0 {
        // Aim for a 1..=2 MB-wide × 1..=2 MB-tall plane (16..=32 px
        // per axis) so the MB grid covers a non-trivial number of
        // sub-blocks while keeping the per-iteration allocation small.
        let mb_cols = 1 + ((flags >> 4) as usize & 1);
        let mb_rows = 1 + ((flags >> 5) as usize & 1);
        let y_stride = mb_cols * 16;
        let uv_stride = mb_cols * 8;
        let y_plane_len = y_stride * mb_rows * 16;
        let uv_plane_len = uv_stride * mb_rows * 8;

        let mut y_plane = vec![0u8; y_plane_len];
        let mut uv_plane = vec![0u8; uv_plane_len];
        if payload.is_empty() {
            for (i, slot) in y_plane.iter_mut().enumerate() {
                *slot = (i as u8) ^ flags;
            }
            for (i, slot) in uv_plane.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_add(flags);
            }
        } else {
            for (i, slot) in y_plane.iter_mut().enumerate() {
                *slot = payload[i % payload.len()];
            }
            for (i, slot) in uv_plane.iter_mut().enumerate() {
                *slot = payload[(i.wrapping_add(7)) % payload.len()];
            }
        }

        let mb_col = (blk_x_sel as usize) % mb_cols;
        let mb_row = (blk_y_sel as usize) % mb_rows;
        let refs = ReferencePlanes {
            y: &y_plane,
            u: &uv_plane,
            v: &uv_plane,
            y_stride,
            uv_stride,
            mb_cols,
            mb_rows,
        };

        // Whole-pixel MB predict: returns `SubPixelNotSupported` if
        // `driving_mv` has a non-zero fractional, which is expected —
        // the call must still be panic-free either way.
        let _ = predict_inter_mb_whole_pixel(
            &refs,
            mb_col,
            mb_row,
            driving_mv,
            /* full_pixel_chroma = */ (version_byte & 1) != 0,
        );

        // Sub-pel MB predict: panic-free for any `driving_mv`. Both tap
        // tables routed through so the dispatcher's filter parameter is
        // exercised at the §18.4 layer.
        let _ = predict_inter_mb(
            &refs,
            mb_col,
            mb_row,
            driving_mv,
            /* full_pixel_chroma = */ (version_byte & 1) != 0,
            primary_taps,
        );
        let _ = predict_inter_mb(
            &refs,
            mb_col,
            mb_row,
            driving_mv,
            /* full_pixel_chroma = */ (version_byte & 1) != 0,
            secondary_taps,
        );
    }
});
