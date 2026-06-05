#![no_main]

//! Fuzz: panic-freedom of the public §15 per-segment loop-filter
//! primitives.
//!
//! The four existing fuzz targets reach §15 only indirectly through
//! `decode_vp8` / `Vp8DecoderState::decode_frame` / `encode_keyframe`,
//! which gate the per-segment kernels behind a fully-formed
//! reconstruction raster. That leaves the primitive surface itself —
//! `common_adjust`, `simple_segment`, `subblock_filter`, `mb_filter`,
//! and `LoopFilterParams::derive` — directly under-fuzzed: a caller
//! that picks an unfortunate `(seg.len(), base)` pair would land an
//! out-of-bounds index that the higher-level harnesses never reach.
//!
//! This target exercises that surface with a deliberately
//! wide-envelope `(base, segment-bytes)` shape so libFuzzer can drive
//! every clamp / saturating-add boundary in §15.2 / §15.3 plus the
//! `(loop_filter_level, sharpness_level, key_frame)` lattice of
//! §15.4.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`  | `loop_filter_level` raw byte (full 0..=255 — §15.4 caller normalises) |
//! | `[1]`  | `sharpness_level` raw byte (full 0..=255) |
//! | `[2]`  | `hev_threshold` raw byte (full 0..=255) |
//! | `[3]`  | `interior_limit` raw byte (full 0..=255) |
//! | `[4]`  | `edge_limit` raw byte (full 0..=255) |
//! | `[5]`  | flag byte: bit0 `key_frame`, bit1 `use_outer_taps`, bit2 picks `simple` vs `normal` filter on the §15.2 leg |
//! | `[6]`  | `base` offset selector (saturated against the buffer length) |
//! | `[7..]`| segment payload — copied into a `Vec<u8>` of length ≥ 8 |
//!
//! The harness:
//!
//! 1. Calls [`LoopFilterParams::derive`] with the raw §15.4 byte triple
//!    and asserts the four returned limits are reachable (no panic).
//! 2. Builds a working buffer of length `max(8, payload.len())`,
//!    populates it from `data[7..]` (tile-extended if shorter than
//!    8 bytes), and picks `base` from `data[6]` saturated so
//!    `base + 8 <= buf.len()` — both kernel families read up to 8
//!    bytes ahead of `base`.
//! 3. Drives all four public primitives in turn with the derived
//!    parameters AND with an independent raw-bytes parameter set from
//!    `data[2..5]`, snapshotting + restoring the buffer between calls
//!    so each primitive sees a fresh segment.
//!
//! Sweeps the §15 primitive layer for panic-freedom, with the working
//! buffer as the only allocation.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::loop_filter::{
    common_adjust, mb_filter, simple_segment, subblock_filter, LoopFilterParams,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;
const SEGMENT_WINDOW: usize = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let loop_filter_level = data[0];
    let sharpness_level = data[1];
    let hev_threshold_raw = data[2];
    let interior_limit_raw = data[3];
    let edge_limit_raw = data[4];
    let flags = data[5];
    let key_frame = (flags & 0b001) != 0;
    let use_outer_taps = (flags & 0b010) != 0;
    let prefer_simple = (flags & 0b100) != 0;
    let base_selector = data[6];

    // §15.4: derive limits from the raw byte triple. The body of
    // `derive` is total over `u8` so no panic should ever surface; the
    // call still exercises the saturating-sub / min-cap clamps for the
    // sharpness ladder and the `interior_limit == 0 -> 1` floor.
    let derived = LoopFilterParams::derive(loop_filter_level, sharpness_level, key_frame);

    // Build the working buffer. The primitives all read at least
    // `base + 8` bytes (subblock_filter / mb_filter) or `base + 4`
    // bytes (simple_segment / common_adjust), so guarantee a SEGMENT
    // window at the chosen base.
    let payload = &data[HEADER_BYTES..];
    let buf_len = SEGMENT_WINDOW.max(payload.len());
    let mut buf = vec![0u8; buf_len];
    if payload.is_empty() {
        // Seed with the flags byte so the buffer isn't all-zero (the
        // edge metric trips trivially on a zero segment, skipping the
        // adjustment paths).
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = flags.wrapping_add(i as u8);
        }
    } else {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = payload[i % payload.len()];
        }
    }

    // Pick `base` so `base + SEGMENT_WINDOW <= buf_len`. With
    // `buf_len >= SEGMENT_WINDOW` the modulo is well-defined; the
    // saturating-add pattern keeps the bound true even when `buf_len
    // == SEGMENT_WINDOW` (base == 0).
    let max_base = buf_len - SEGMENT_WINDOW;
    let base = if max_base == 0 {
        0
    } else {
        (base_selector as usize) % (max_base + 1)
    };

    // Snapshot so each primitive sees an identical starting segment;
    // the §15.2 / §15.3 kernels mutate the segment in place.
    let snapshot = buf.clone();

    // ---- (1) common_adjust on the 4-pixel inner window. ----
    let _ = common_adjust(use_outer_taps, &mut buf, base);
    buf.copy_from_slice(&snapshot);

    // ---- (2) simple_segment on the 4-pixel window. ----
    // §15.2: 4-byte window starting at `base`.
    simple_segment(edge_limit_raw, &mut buf, base);
    buf.copy_from_slice(&snapshot);

    // ---- (3) subblock_filter on the 8-pixel window with derived
    // limits, then with the raw triple. ----
    subblock_filter(
        derived.hev_threshold,
        derived.interior_limit,
        derived.sub_bedge_limit,
        &mut buf,
        base,
    );
    buf.copy_from_slice(&snapshot);

    subblock_filter(
        hev_threshold_raw,
        interior_limit_raw,
        edge_limit_raw,
        &mut buf,
        base,
    );
    buf.copy_from_slice(&snapshot);

    // ---- (4) mb_filter on the 8-pixel window, both parameter sets.
    mb_filter(
        derived.hev_threshold,
        derived.interior_limit,
        derived.mbedge_limit,
        &mut buf,
        base,
    );
    buf.copy_from_slice(&snapshot);

    mb_filter(
        hev_threshold_raw,
        interior_limit_raw,
        edge_limit_raw,
        &mut buf,
        base,
    );

    // ---- (5) chained pass: when the flag's `prefer_simple` bit is
    // set, run simple_segment then mb_filter back-to-back on the same
    // mutated buffer so any state hand-off across primitives gets
    // exercised; otherwise run subblock_filter then mb_filter. ----
    buf.copy_from_slice(&snapshot);
    if prefer_simple {
        simple_segment(edge_limit_raw, &mut buf, base);
        mb_filter(
            derived.hev_threshold,
            derived.interior_limit,
            derived.mbedge_limit,
            &mut buf,
            base,
        );
    } else {
        subblock_filter(
            derived.hev_threshold,
            derived.interior_limit,
            derived.sub_bedge_limit,
            &mut buf,
            base,
        );
        mb_filter(
            derived.hev_threshold,
            derived.interior_limit,
            derived.mbedge_limit,
            &mut buf,
            base,
        );
    }
});
