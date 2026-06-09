#![no_main]

//! Fuzz: per-frame symmetric encode→decode round-trip
//! (RFC 6386 §7 / §9 / §11 / §13 / §14 / §15).
//!
//! The thirteen existing targets cover each leg of the pipeline in
//! isolation: `panic_free_encode_keyframe` exercises `encode_keyframe`
//! with attacker-shaped `(I420Frame, KeyframeParams)` but stops at the
//! emitted byte buffer; `panic_free_decode_keyframe` drives `decode_vp8`
//! with arbitrary attacker bytes but the §9.1 frame-tag pre-screen
//! rejects anything that isn't already a structurally-valid keyframe
//! start. Neither target locks the two halves in lockstep — i.e. it
//! has never been the case that an `encode_keyframe` output is fed
//! back into `decode_vp8` against fuzz-derived parameters in the same
//! libFuzzer iteration. That seam is exactly where a §7.3 bool-coder
//! asymmetry, a §13 token miscount, a §9.6 quant-index sign confusion,
//! or a §15.4 loop-filter derivation skew would hide: the encoder's
//! own structural invariants suppress any panic during emission, and
//! the decoder's strict early validation rejects everything except
//! self-encoded bitstreams. This target stitches them together.
//!
//! Surface coverage (entry points exercised back-to-back per
//! iteration):
//!
//! * `encode_keyframe(&I420Frame, &KeyframeParams)` — drives both the
//!   happy-path §11 mode pick → §14 forward DCT/WHT → §13 token
//!   emission → §15 loop-filter writeback chain AND the parameter-
//!   rejection envelope (`QuantIndexOutOfRange` /
//!   `LoopFilterLevelOutOfRange` / `SharpnessLevelOutOfRange` /
//!   `InvalidDctPartitionCount`).
//! * `decode_vp8(&[u8])` — on every `Ok(bytes)` from the encoder, the
//!   target feeds the emitted bitstream straight into the public §9.1
//!   keyframe decoder, walking the §7 boolean range coder, the §9
//!   uncompressed + §19.2 coded headers, the §11 macroblock-mode
//!   layer, the §13 token decode, the §14 dequant + inverse transform,
//!   the §15 loop-filter post-pass and the visible-cropped I420
//!   emission. Any panic, abort, or debug-arithmetic overflow at any
//!   of those stages — on bitstreams the encoder itself just produced
//!   — surfaces here.
//! * Symmetric correctness sanity-check (light): when `decode_vp8`
//!   succeeds, the harness asserts the decoded dimensions match the
//!   visible width / height the encoder was driven with. The encoder
//!   pads up to the MB-aligned reconstruction grid and the decoder
//!   crops back; any cross-side drift in the §9.1 width / height
//!   wire-form would surface as a mismatch.
//!
//! Cross-target overlap vs the round-261 `panic_free_bool_codec`
//! target is intentional but distinct: r261 drives the §7.3 bool
//! coder primitives in isolation against an *attacker-shaped*
//! probability schedule; this target drives them inside the full
//! §11 / §13 / §14 production chain on encoder-shaped probabilities,
//! exposing any path where the encoder's range-tracking diverges
//! from the decoder's even though both halves pass their isolated
//! sweeps.
//!
//! Input layout (consumed from the front of the libFuzzer `data`,
//! mirroring `panic_free_encode_keyframe` so cross-corpus reuse is
//! trivial):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`             | Visible width MB-units (1..=16, so 16..=256 px) |
//! | `[1]`             | Visible height MB-units (1..=16, so 16..=256 px) |
//! | `[2]`             | `y_ac_qi` raw byte (encoder MUST surface `QuantIndexOutOfRange` for > 127) |
//! | `[3]`             | `loop_filter_level` raw byte (encoder MUST surface `LoopFilterLevelOutOfRange` for > 63) |
//! | `[4]`             | `sharpness_level` raw byte (encoder MUST surface `SharpnessLevelOutOfRange` for > 7) |
//! | `[5]`             | `nbr_of_dct_partitions` raw byte (encoder MUST surface `InvalidDctPartitionCount` for non-{1,2,4,8}) |
//! | `[6]`             | `filter_type` flag (low bit) |
//! | `[7..]`           | I420 pixel payload — tiled across the three planes |
//!
//! Dimension cap matches the rest of the harness via
//! [`oxideav_vp8_fuzz::accept_dimensions`] (≤ 256 × 256 luma pixels
//! per frame). Hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{decode_vp8, encode_keyframe, I420Frame, KeyframeParams};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions in MB units (16 px each). `1 + (b % 16)` lands
    // in 1..=16 MBs i.e. 16..=256 luma pixels per axis, matching the
    // budget the rest of the fuzz harness operates under.
    let mb_w = 1u32 + u32::from(data[0] % 16);
    let mb_h = 1u32 + u32::from(data[1] % 16);
    let width = mb_w * 16;
    let height = mb_h * 16;

    if !accept_dimensions(width, height) {
        return;
    }

    let params = KeyframeParams {
        y_ac_qi: data[2],
        loop_filter_level: data[3],
        sharpness_level: data[4],
        nbr_of_dct_partitions: data[5],
        filter_type: (data[6] & 1) != 0,
    };

    // I420 plane lengths for tightly-packed strides at the chosen
    // dimensions.
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;

    // Tile the payload tail across the three planes. The modular
    // indexing stays panic-safe even on an empty payload (we fall
    // back to all-zero planes).
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

    // Encode leg: any panic here is the same defect
    // `panic_free_encode_keyframe` would catch, but we still drive it
    // because the encode→decode symmetry is the contribution of this
    // target. A structured `Err` is a legitimate outcome (parameter-
    // rejection envelope); only a panic fails.
    let bytes = match encode_keyframe(&frame, &params) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Decode leg: feed the encoder's own output straight back into
    // the public single-frame decoder. The encoder guarantees a
    // structurally-valid §9.1 keyframe start, so the decoder's early
    // pre-screen will not short-circuit — every stage from the §7
    // bool coder through the §15 loop filter is reached on real
    // bitstream bytes for every accepted (`I420Frame`,
    // `KeyframeParams`) pair. Any panic here is a cross-side seam
    // defect.
    let decoded = match decode_vp8(&bytes) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Symmetric sanity-check: the §9.1 visible dimensions must round-
    // trip. An encoder/decoder split on the width / height wire form
    // (scale-code bits, MB-aligned vs visible-cropped semantics) would
    // surface as a mismatch here. We only assert when both halves
    // produce data; the encoder's own out-of-range fail paths and
    // the decoder's `DecodeError` paths are panic-free returns and do
    // not reach this point.
    debug_assert_eq!(decoded.width, width);
    debug_assert_eq!(decoded.height, height);
});
