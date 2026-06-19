#![no_main]

//! Fuzz: §9.3 / §10 segment-based adaptive-quant encode→decode
//! pixel-exact lockstep (RFC 6386 §9 / §10 / §12 / §14 / §15 / §19.2).
//!
//! The round-346 `encode_keyframe_adaptive_quant` path is the first
//! encoder that emits `segmentation_enabled = true`: it classifies each
//! macroblock into one of four §10 segments by luma variance and
//! quantises it at that segment's effective qindex, emitting the §9.3
//! `update_segmentation()` block (delta-mode per-segment quantizer
//! values + `mb_segment_tree_probs`) and the per-MB §10 `segment_id`.
//! The existing `encode_decode_pixel_lockstep` target only drives the
//! `segmentation_enabled = false` keyframe writer, so none of the new
//! segmentation emission — the variance classifier, the four per-segment
//! dequant factors, the per-MB segment-id tree codes, the segmentation
//! header sub-block — has been fuzzed in lockstep.
//!
//! Oracle (same contract as `encode_decode_pixel_lockstep`): the encoder
//! quantises and dequantises each macroblock with the *same* per-segment
//! factor and predicts from its own reconstruction, so a compliant
//! `decode_vp8` — which rebuilds the identical four
//! `MbDequantFactors` from the on-wire quantizer deltas and routes each
//! MB by its decoded `segment_id` — must reproduce the encoder's post-§15
//! reconstruction planes byte-for-byte (cropped to the §9.1 visible
//! window). The harness panics on encode failure, decode failure,
//! dimension drift, MB-grid drift, or any single-pixel mismatch — each is
//! a finding, because every parameter here is normalised into its legal
//! range (the rejection envelope is `panic_free_encode_keyframe`'s job).
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | Visible width: `1 + (b % 64)` luma px |
//! | `[1]`     | Visible height: `1 + (b % 144)` luma px |
//! | `[2]`     | `base_y_ac_qi = b % 128` |
//! | `[3]`     | `loop_filter_level = b % 64` (0 ⇒ §15 skip path) |
//! | `[4]`     | `sharpness_level = b % 8` |
//! | `[5]`     | flags: bit 0 = `filter_type` (§15.2 simple vs §15.3 normal) |
//! | `[6..10]` | four signed per-segment quant deltas (`b as i8` clamped to ±120) |
//! | `[10..13]`| three variance boundaries (each `b * 32`, ascending-ish) |
//! | `[13..]`  | I420 pixel payload — tiled across the three planes |
//!
//! Max area 64 × 144 = 9 216 luma px (≤ 36 MBs), inside the shared
//! [`oxideav_vp8_fuzz::accept_dimensions`] budget; hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    decode_vp8, encode_keyframe_adaptive_quant_with_reconstruction, AdaptiveQuantConfig, I420Frame,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
/// 6 scalar bytes + 4 quant deltas + 3 variance boundaries.
const HEADER_BYTES: usize = 6 + 4 + 3;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = 1u32 + u32::from(data[0] % 64);
    let height = 1u32 + u32::from(data[1] % 144);
    if !accept_dimensions(width, height) {
        return;
    }

    // Per-segment quant deltas: any signed value the §9.3 7-bit magnitude
    // can carry. Clamp to ±120 so `base + delta` stays representable; the
    // encoder itself re-clamps the effective qindex to 0..=127.
    let clamp_delta = |b: u8| -> i8 {
        let v = b as i8;
        v.clamp(-120, 120)
    };

    let config = AdaptiveQuantConfig {
        base_y_ac_qi: data[2] % 128,
        loop_filter_level: data[3] % 64,
        sharpness_level: data[4] % 8,
        filter_type: (data[5] & 0x01) != 0,
        quant_delta: [
            clamp_delta(data[6]),
            clamp_delta(data[7]),
            clamp_delta(data[8]),
            clamp_delta(data[9]),
        ],
        variance_boundaries: [
            u32::from(data[10]) * 32,
            u32::from(data[11]) * 32,
            u32::from(data[12]) * 32,
        ],
    };

    // I420 plane payload, tiled from the input tail (tight strides).
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;
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

    // Encode leg — every parameter is in-range, so failure is a defect.
    let (bytes, recon) = match encode_keyframe_adaptive_quant_with_reconstruction(&frame, &config) {
        Ok(ok) => ok,
        Err(e) => panic!(
            "adaptive-quant encode rejected an in-range input: {e:?} \
                 ({width}x{height}, config {config:?})"
        ),
    };

    // Decode leg.
    let decoded = match decode_vp8(&bytes) {
        Ok(d) => d,
        Err(e) => panic!(
            "decode_vp8 rejected an adaptive-quant bitstream: {e:?} \
             ({width}x{height}, config {config:?}, {} bytes)",
            bytes.len()
        ),
    };

    assert_eq!(decoded.width, width, "visible width drift");
    assert_eq!(decoded.height, height, "visible height drift");
    assert_eq!(recon.mb_cols, w.div_ceil(16), "encoder mb_cols drift");
    assert_eq!(recon.mb_rows, h.div_ceil(16), "encoder mb_rows drift");

    // Pixel-exact differential against the encoder's own reconstruction.
    for row in 0..h {
        let dec = &decoded.y[row * w..row * w + w];
        let enc = &recon.y[row * recon.y_stride..row * recon.y_stride + w];
        assert_eq!(
            dec, enc,
            "luma drift at row {row} ({width}x{height}, {config:?})"
        );
    }
    for row in 0..uvh {
        let dec_u = &decoded.u[row * uvw..row * uvw + uvw];
        let enc_u = &recon.u[row * recon.uv_stride..row * recon.uv_stride + uvw];
        assert_eq!(
            dec_u, enc_u,
            "U drift at row {row} ({width}x{height}, {config:?})"
        );
        let dec_v = &decoded.v[row * uvw..row * uvw + uvw];
        let enc_v = &recon.v[row * recon.uv_stride..row * recon.uv_stride + uvw];
        assert_eq!(
            dec_v, enc_v,
            "V drift at row {row} ({width}x{height}, {config:?})"
        );
    }
});
