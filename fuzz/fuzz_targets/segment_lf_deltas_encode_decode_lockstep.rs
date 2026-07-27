#![no_main]

//! Fuzz: the §10 per-segment **loop-filter feature** encode→decode
//! pixel-exact lockstep —
//! [`encode_keyframe_adaptive_quant_with_segment_lf_deltas`] and its
//! trellis companion
//! [`encode_keyframe_adaptive_quant_with_segment_lf_deltas_and_trellis`].
//!
//! The existing `adaptive_quant_encode_decode_lockstep` target drives
//! the plain adaptive-quant writer, whose §9.3 `update_segmentation()`
//! block carries only the four quantizer deltas. The `_with_segment_lf_deltas`
//! variants additionally emit the four `loop_filter_update` values
//! (delta mode), and the encoder's own §15 post-walk filter resolves
//! each MB's level through the same segment override the decoder
//! applies (`calculate_mb_filter_level`, §20.6 clamp to `0..=63`).
//! None of that wire — the segmentation sub-block with *both* feature
//! arrays populated, nor the per-segment filter-level resolution on
//! the encode side — was fuzzed before this target.
//!
//! Also first-fuzzed here: the WebP-canonical quality mappings
//! [`quality_to_qindex`] (drives `base_y_ac_qi`; must land in
//! `0..=127`) and [`quality_to_trellis_strength`] (drives the trellis
//! knob on the `_and_trellis` leg).
//!
//! Oracle (same contract as the sibling lockstep targets): every
//! parameter fed to the encode leg is normalised into its legal range,
//! so an encode error is a finding; the emitted bytes must decode
//! through [`decode_vp8`]; and the decoded planes must equal the
//! encoder's own post-§15 reconstruction byte-for-byte (cropped to the
//! §9.1 visible window) — a divergence in the per-segment filter-level
//! resolution between the two sides surfaces as a pixel mismatch.
//! Deltas outside the §9.3 7-bit ±63 envelope must instead be
//! rejected with a structured error (asserted on a dedicated probe).
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`      | Visible width: `1 + (b % 64)` luma px |
//! | `[1]`      | Visible height: `1 + (b % 144)` luma px |
//! | `[2]`      | Quality dial `b % 101` → `quality_to_qindex` / `quality_to_trellis_strength` |
//! | `[3]`      | `loop_filter_level = b % 64` (0 ⇒ §15 skip regardless of deltas) |
//! | `[4]`      | `sharpness_level = b % 8` |
//! | `[5]`      | flags: bit 0 = `filter_type`; bit 1 = trellis leg; bit 2 = out-of-range-delta probe |
//! | `[6..10]`  | four per-segment quant deltas (`b as i8` clamped to ±120) |
//! | `[10..14]` | four per-segment LF deltas (`b as i8` clamped to ±63; probe un-clamps one) |
//! | `[14..17]` | three variance boundaries (each `b * 32`) |
//! | `[17..]`   | I420 pixel payload — tiled across the three planes |
//!
//! Max area 64 × 144 = 9 216 luma px (≤ 36 MBs), inside the shared
//! [`oxideav_vp8_fuzz::accept_dimensions`] budget; hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    decode_vp8, encode_keyframe_adaptive_quant_with_segment_lf_deltas,
    encode_keyframe_adaptive_quant_with_segment_lf_deltas_and_trellis, quality_to_qindex,
    quality_to_trellis_strength, AdaptiveQuantConfig, I420Frame,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
/// 6 scalar bytes + 4 quant deltas + 4 LF deltas + 3 variance boundaries.
const HEADER_BYTES: usize = 6 + 4 + 4 + 3;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = 1u32 + u32::from(data[0] % 64);
    let height = 1u32 + u32::from(data[1] % 144);
    if !accept_dimensions(width, height) {
        return;
    }

    // The quality dial covers both pure mapping helpers; the qindex
    // contract (0..=127) is asserted here since the config consumes it
    // as `base_y_ac_qi` directly.
    let quality = f32::from(data[2] % 101);
    let base_y_ac_qi = quality_to_qindex(quality);
    assert!(
        base_y_ac_qi <= 127,
        "quality_to_qindex({quality}) escaped 0..=127: {base_y_ac_qi}"
    );

    let clamp_quant_delta = |b: u8| -> i8 { (b as i8).clamp(-120, 120) };
    let clamp_lf_delta = |b: u8| -> i8 { (b as i8).clamp(-63, 63) };

    let config = AdaptiveQuantConfig {
        base_y_ac_qi,
        loop_filter_level: data[3] % 64,
        sharpness_level: data[4] % 8,
        filter_type: (data[5] & 0x01) != 0,
        quant_delta: [
            clamp_quant_delta(data[6]),
            clamp_quant_delta(data[7]),
            clamp_quant_delta(data[8]),
            clamp_quant_delta(data[9]),
        ],
        variance_boundaries: [
            u32::from(data[14]) * 32,
            u32::from(data[15]) * 32,
            u32::from(data[16]) * 32,
        ],
    };

    let lf_deltas = [
        clamp_lf_delta(data[10]),
        clamp_lf_delta(data[11]),
        clamp_lf_delta(data[12]),
        clamp_lf_delta(data[13]),
    ];

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

    // Out-of-range-delta probe: a delta beyond ±63 must be rejected
    // with a structured error, never silently clamped or emitted.
    if data[5] & 0x04 != 0 {
        let mut bad = lf_deltas;
        bad[usize::from(data[10]) % 4] = if data[11] & 1 != 0 { 64 } else { -64 };
        if encode_keyframe_adaptive_quant_with_segment_lf_deltas(&frame, &config, &bad).is_ok() {
            panic!("segment LF delta beyond +/-63 was accepted: {bad:?}");
        }
        return;
    }

    // Encode leg — every parameter is in-range, so failure is a defect.
    let use_trellis = (data[5] & 0x02) != 0;
    let result = if use_trellis {
        encode_keyframe_adaptive_quant_with_segment_lf_deltas_and_trellis(
            &frame,
            &config,
            &lf_deltas,
            quality_to_trellis_strength(quality),
        )
    } else {
        encode_keyframe_adaptive_quant_with_segment_lf_deltas(&frame, &config, &lf_deltas)
    };
    let (bytes, recon) = match result {
        Ok(ok) => ok,
        Err(e) => panic!(
            "segment-LF-deltas encode rejected an in-range input: {e:?} \
             ({width}x{height}, config {config:?}, lf_deltas {lf_deltas:?})"
        ),
    };

    // Decode leg.
    let decoded = match decode_vp8(&bytes) {
        Ok(d) => d,
        Err(e) => panic!(
            "decode_vp8 rejected a segment-LF-deltas bitstream: {e:?} \
             ({width}x{height}, config {config:?}, lf_deltas {lf_deltas:?}, {} bytes)",
            bytes.len()
        ),
    };

    assert_eq!(decoded.width, width, "visible width drift");
    assert_eq!(decoded.height, height, "visible height drift");
    assert_eq!(recon.mb_cols, w.div_ceil(16), "encoder mb_cols drift");
    assert_eq!(recon.mb_rows, h.div_ceil(16), "encoder mb_rows drift");

    // Pixel-exact differential against the encoder's own reconstruction
    // — the per-segment filter-level resolution must agree on both sides.
    for row in 0..h {
        let dec = &decoded.y[row * w..row * w + w];
        let enc = &recon.y[row * recon.y_stride..row * recon.y_stride + w];
        assert_eq!(
            dec, enc,
            "luma drift at row {row} ({width}x{height}, {config:?}, lf_deltas {lf_deltas:?})"
        );
    }
    for row in 0..uvh {
        let dec_u = &decoded.u[row * uvw..row * uvw + uvw];
        let enc_u = &recon.u[row * recon.uv_stride..row * recon.uv_stride + uvw];
        assert_eq!(
            dec_u, enc_u,
            "U drift at row {row} ({width}x{height}, {config:?}, lf_deltas {lf_deltas:?})"
        );
        let dec_v = &decoded.v[row * uvw..row * uvw + uvw];
        let enc_v = &recon.v[row * recon.uv_stride..row * recon.uv_stride + uvw];
        assert_eq!(
            dec_v, enc_v,
            "V drift at row {row} ({width}x{height}, {config:?}, lf_deltas {lf_deltas:?})"
        );
    }
});
