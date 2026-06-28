#![no_main]

//! Fuzz: panic-freedom of the public `encode_keyframe` driver.
//!
//! Mirror of `panic_free_decode_keyframe` on the encode side. The
//! contract: every `(I420Frame, KeyframeParams)` pair the caller can
//! plausibly construct from arbitrary bytes MUST produce either an
//! emitted byte stream or a structured `Err` — never a panic, abort,
//! debug-arithmetic overflow, or out-of-bounds index.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`             | Visible width MB-units (1..=16, so 16..=256 px) |
//! | `[1]`             | Visible height MB-units (1..=16, so 16..=256 px) |
//! | `[2]`             | `y_ac_qi` raw byte (full 0..=255 range — encoder MUST surface `QuantIndexOutOfRange` for > 127) |
//! | `[3]`             | `loop_filter_level` raw byte (encoder MUST surface `LoopFilterLevelOutOfRange` for > 63) |
//! | `[4]`             | `sharpness_level` raw byte (encoder MUST surface `SharpnessLevelOutOfRange` for > 7) |
//! | `[5]`             | `nbr_of_dct_partitions` raw byte (encoder MUST surface `InvalidDctPartitionCount` for non-{1,2,4,8}) |
//! | `[6]`             | `filter_type` flag (low bit) |
//! | `[7..]`           | I420 pixel payload — tiled / truncated to fit the three plane lengths |
//!
//! The width / height bytes are normalised into the §9.1 wire range by
//! `1 + (b % 16)` MB-units so the dimensions land in the legal
//! `1..=256` luma-pixel range a single fuzz iteration can afford to
//! encode. (The encoder accepts MB-misaligned dimensions and
//! edge-replicates a partial right / bottom MB, but the OOM guard
//! tracks total pixels, not the alignment.) Every other knob is fed
//! raw so the encoder's own range-validation surfaces the
//! `EncodeError` variants the public surface promises — exercising
//! both the happy path AND the parameter-rejection paths in one
//! target.
//!
//! Pixel payload is tiled across the three planes from the tail of
//! `data` (modular indexing so short inputs still produce a populated
//! plane). No dimension preflight beyond the 256 px-per-axis cap
//! (matching `panic_free_decode_keyframe`'s 256×256 luma-pixel
//! budget); the bool-coder partition emission is allocation-free per
//! token so even the maximum frame fits comfortably inside the fuzz
//! runner's per-iteration memory cap.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{encode_keyframe, I420Frame, KeyframeParams};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions in MB units (16 px each). `1 + (b % 16)` lands
    // in 1..=16 MBs i.e. 16..=256 luma pixels per axis — the same
    // OOM envelope `panic_free_decode_keyframe` uses.
    let mb_w = 1u32 + u32::from(data[0] % 16);
    let mb_h = 1u32 + u32::from(data[1] % 16);
    let width = mb_w * 16;
    let height = mb_h * 16;

    let params = KeyframeParams {
        y_ac_qi: data[2],
        loop_filter_level: data[3],
        sharpness_level: data[4],
        nbr_of_dct_partitions: data[5],
        filter_type: (data[6] & 1) != 0,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    // I420 plane lengths for tightly-packed strides at the chosen
    // dimensions.
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;

    // Tile the payload tail across the three planes so a short input
    // still produces a populated buffer. The modular indexing is
    // panic-safe even when `payload` is empty (we fall back to a
    // single-byte buffer of zeros so the modulo arithmetic stays in
    // bounds).
    let payload = &data[HEADER_BYTES..];
    let mut y_plane = vec![0u8; y_len];
    let mut u_plane = vec![0u8; uv_len];
    let mut v_plane = vec![0u8; uv_len];
    if !payload.is_empty() {
        for (i, slot) in y_plane.iter_mut().enumerate() {
            *slot = payload[i % payload.len()];
        }
        for (i, slot) in u_plane.iter_mut().enumerate() {
            *slot = payload[(i + y_len) % payload.len()];
        }
        for (i, slot) in v_plane.iter_mut().enumerate() {
            *slot = payload[(i + y_len + uv_len) % payload.len()];
        }
    }

    let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
    let _ = encode_keyframe(&frame, &params);
});
