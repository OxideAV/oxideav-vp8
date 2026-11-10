#![no_main]

//! Fuzz: key-frame encode options — effort level × thread budget ×
//! loop-filter / token-probability switches — in pixel lockstep
//! (RFC 6386 §9 / §11 / §12 / §13 / §14 / §15).
//!
//! `encode_keyframe_with_reconstruction_and_options` adds two axes to the
//! key-frame encoder that no other target drives: the mode-search
//! [`EncodeEffort`] (three of the four levels run a different
//! per-macroblock search with its own mode shortlists, cost model and
//! winner quantisation) and the macroblock-row wavefront that spreads the
//! analysis over a thread budget. Every iteration checks four things:
//!
//! 1. **Thread-budget invariance** — the bytes and all three
//!    reconstruction planes of a threaded encode equal the serial
//!    encode's, for the fuzz-chosen budget (2..=8 threads; pictures are
//!    1..=6 macroblock rows tall, so budgets above, at and below the row
//!    count all occur).
//! 2. **Decoder lockstep** — `decode_vp8` of the bytes reproduces the
//!    encoder's post-§15 reconstruction over the visible window, at every
//!    effort level (the reduced levels predict from their own
//!    reconstruction exactly as the exhaustive search does).
//! 3. **Default identity** — at `Thorough` with no switches the bytes
//!    equal the option-less `encode_keyframe`.
//! 4. No encode or decode error for in-range parameters.
//!
//! Input layout:
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]` | visible width `1 + (b % 96)` luma px |
//! | `[1]` | visible height `1 + (b % 96)` luma px |
//! | `[2]` | `y_ac_qi = b % 128` |
//! | `[3]` | `loop_filter_level = b % 64` |
//! | `[4]` | `sharpness_level = b % 8` |
//! | `[5]` | bits 0..1: `nbr_of_dct_partitions = 1 << n`; bits 2..4: thread budget `2 + n`; bits 5..7: `TrellisStrength` (`0` = OFF) |
//! | `[6]` | bits 0..1: effort level; bit 2: `filter_type`; bit 3: `auto_loop_filter`; bit 4: `fitted_token_prob_updates` |
//! | `[7..]` | I420 pixel payload, tiled across the three planes |
//!
//! Max area 96 × 96 luma px (36 macroblocks); hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    decode_vp8, encode_keyframe, encode_keyframe_with_reconstruction_and_options, EncodeEffort,
    I420Frame, KeyframeEncodeOptions, KeyframeParams, TrellisStrength,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }
    let width = 1u32 + u32::from(data[0] % 96);
    let height = 1u32 + u32::from(data[1] % 96);
    if !accept_dimensions(width, height) {
        return;
    }
    let params = KeyframeParams::default()
        .with_y_ac_qi(data[2] % 128)
        .with_loop_filter_level(data[3] % 64)
        .with_sharpness_level(data[4] % 8)
        .with_nbr_of_dct_partitions(1u8 << (data[5] & 0x03))
        .with_filter_type((data[6] & 0x04) != 0)
        .with_trellis_strength(TrellisStrength::new(f64::from(data[5] >> 5)));
    let threads = 2 + usize::from((data[5] >> 2) & 0x07);
    let effort = EncodeEffort::from_level(data[6] & 0x03);
    let auto_loop_filter = (data[6] & 0x08) != 0;
    let fitted = (data[6] & 0x10) != 0;
    let options = KeyframeEncodeOptions::new()
        .with_effort(effort)
        .with_auto_loop_filter(auto_loop_filter)
        .with_fitted_token_prob_updates(fitted);

    let w = width as usize;
    let h = height as usize;
    let (uvw, uvh) = (w.div_ceil(2), h.div_ceil(2));
    let (y_len, uv_len) = (w * h, uvw * uvh);
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

    let (bytes, recon) =
        match encode_keyframe_with_reconstruction_and_options(&frame, &params, &options) {
            Ok(ok) => ok,
            Err(e) => panic!(
                "serial encode rejected an in-range input: {e:?} \
                 ({width}x{height}, {params:?}, {options:?})"
            ),
        };

    // 1. Thread-budget invariance.
    let (threaded_bytes, threaded_recon) = encode_keyframe_with_reconstruction_and_options(
        &frame,
        &params,
        &options.with_threads(threads),
    )
    .unwrap_or_else(|e| panic!("threaded encode failed: {e:?} ({threads} threads)"));
    assert_eq!(
        threaded_bytes, bytes,
        "bytes changed with the thread budget ({threads} threads, {width}x{height}, \
         {params:?}, {options:?})"
    );
    assert!(
        threaded_recon.y == recon.y && threaded_recon.u == recon.u && threaded_recon.v == recon.v,
        "reconstruction changed with the thread budget ({threads} threads)"
    );

    // 2. Decoder lockstep over the visible window.
    let decoded = decode_vp8(&bytes).unwrap_or_else(|e| {
        panic!("own decoder rejected the frame: {e:?} ({width}x{height}, {params:?}, {options:?})")
    });
    assert_eq!((decoded.width, decoded.height), (width, height));
    for row in 0..h {
        assert_eq!(
            decoded.y[row * w..(row + 1) * w],
            recon.y[row * recon.y_stride..row * recon.y_stride + w],
            "luma row {row} differs ({width}x{height}, {params:?}, {options:?})"
        );
    }
    for row in 0..uvh {
        let at = row * recon.uv_stride;
        assert_eq!(
            decoded.u[row * uvw..(row + 1) * uvw],
            recon.u[at..at + uvw],
            "U row {row} differs ({width}x{height}, {params:?}, {options:?})"
        );
        assert_eq!(
            decoded.v[row * uvw..(row + 1) * uvw],
            recon.v[at..at + uvw],
            "V row {row} differs ({width}x{height}, {params:?}, {options:?})"
        );
    }

    // 3. The default options are the option-less encoder.
    if effort == EncodeEffort::Thorough && !auto_loop_filter && !fitted {
        let plain = encode_keyframe(&frame, &params).expect("plain encode");
        assert_eq!(plain, bytes, "default options drifted from encode_keyframe");
    }
});
