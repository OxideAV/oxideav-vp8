#![no_main]

//! Fuzz: the round-384 lagged auto-altref stream driver
//! ([`Vp8AltrefStreamEncoder`]) end-to-end — panic-freedom of the
//! encoder **plus** a full stateful-decode oracle over every emitted
//! packet, including the invisible ARNR anchors and the §9.7
//! ALTREF → GOLDEN copy-ladder promotion frames that no other target
//! reaches.
//!
//! What random inputs shake out here that the unit / integration tests
//! can't enumerate:
//!
//! * arbitrary (window, keyframe-interval, frame-count) interleavings —
//!   groups closed early by scheduled keys, single-frame tail groups,
//!   `finish()` on every lag state;
//! * the ARNR filter across degenerate content (flat planes, tiled
//!   noise, sub-16px partial-MB frames at any strength, including the
//!   clamped-out-of-range ones);
//! * the invisible-frame §9.1 bit + §9.7 refresh ladder through the
//!   *decoder's* slot walk at every emission position.
//!
//! Contract per packet: it decodes, its wire visibility
//! (`Vp8DecoderState::last_frame_shown`) matches the packet
//! classification, and the visible packets map 1:1 in order onto the
//! source frames. Any structured encoder error aborts the iteration
//! quietly **only** for parameter-envelope rejections that the harness
//! deliberately leaves reachable (none today — the envelope below is
//! always-valid, so every accepted input must encode).
//!
//! Input layout (single-byte fields, consumed front-to-back):
//!
//! | Byte | Meaning |
//! |-----:|---------|
//! | `[0]` | visible width  (`1 + b % 48` px) |
//! | `[1]` | visible height (`1 + b % 48` px) |
//! | `[2]` | `y_ac_qi` (`% 128`) |
//! | `[3]` | `loop_filter_level` (`% 64`) |
//! | `[4]` | `altref_window` (`b % 6` — `0` pins the constructor rejection) |
//! | `[5]` | `keyframe_interval` (`b % 8`) |
//! | `[6]` | ARNR strength raw byte (full range — `ArnrConfig::new` clamps) |
//! | `[7]` | bit 0: `golden_promotion`; bits 1..: frame count seed |
//! | `[8..]` | per-frame fill seeds, then tiled pixel payload |
//!
//! Caps: ≤ 48 × 48 px, ≤ 10 source frames, input ≤ 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    AltrefStreamConfig, ArnrConfig, I420Frame, KeyframeParams, Vp8AltrefStreamEncoder,
    Vp8DecoderState,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 8;
const MAX_FRAMES: usize = 10;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = 1u32 + u32::from(data[0] % 48);
    let height = 1u32 + u32::from(data[1] % 48);
    if !accept_dimensions(width, height) {
        return;
    }

    // Always-valid parameter envelope: every accepted input must encode.
    let params = KeyframeParams {
        y_ac_qi: data[2] % 128,
        loop_filter_level: data[3] % 64,
        ..KeyframeParams::default()
    };
    let altref_window = usize::from(data[4] % 6);
    let config = AltrefStreamConfig {
        params,
        keyframe_interval: u64::from(data[5] % 8),
        altref_window,
        arnr: ArnrConfig::new(data[6]),
        golden_promotion: (data[7] & 0x01) != 0,
    };

    let mut enc = match Vp8AltrefStreamEncoder::new(config) {
        Some(e) => {
            assert_ne!(altref_window, 0, "window 0 must be rejected");
            e
        }
        None => {
            assert_eq!(altref_window, 0, "only window 0 may be rejected");
            return;
        }
    };

    let frame_count = 1usize + usize::from((data[7] >> 1) % (MAX_FRAMES as u8));
    let ctrl = &data[HEADER_BYTES..];
    let payload_start = (HEADER_BYTES + frame_count).min(data.len());
    let payload = &data[payload_start..];

    let w = width as usize;
    let h = height as usize;
    let uv_len = w.div_ceil(2) * h.div_ceil(2);
    let y_len = w * h;

    let mut packets = Vec::new();
    for f in 0..frame_count {
        let seed = ctrl.get(f).copied().unwrap_or(0);
        let base = if payload.is_empty() {
            seed
        } else {
            payload[usize::from(seed) % payload.len()]
        };
        // Distinct fill per frame plus a payload-driven stripe so the
        // ARNR aligner sees non-trivial (but bounded) structure.
        let mut y_plane = vec![base.wrapping_add(f as u8); y_len];
        for (i, px) in y_plane.iter_mut().enumerate() {
            if !payload.is_empty() {
                *px = px.wrapping_add(payload[i % payload.len()] & 0x1f);
            }
        }
        let u_plane = vec![base ^ 0x40; uv_len];
        let v_plane = vec![base ^ 0x80; uv_len];
        let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
        match enc.push_frame(&frame) {
            Ok(emitted) => packets.extend(emitted),
            Err(e) => panic!(
                "altref stream encoder rejected an in-range frame {f}: {e:?} \
                 ({width}x{height}, config {config:?})"
            ),
        }
    }
    match enc.finish() {
        Ok(emitted) => packets.extend(emitted),
        Err(e) => panic!("finish() failed after {frame_count} frames: {e:?}"),
    }
    assert!(
        enc.finish().expect("second finish is Ok").is_empty(),
        "finish() must be idempotent"
    );

    // Decode oracle: every packet decodes in emission order; wire
    // visibility matches the classification; visible packets cover the
    // source indices 0..frame_count in order.
    let mut dec = Vp8DecoderState::new();
    let mut next_visible = 0u64;
    for (i, p) in packets.iter().enumerate() {
        let picture = match dec.decode_frame(&p.bytes) {
            Ok(d) => d,
            Err(e) => panic!(
                "stateful decoder rejected packet {i} ({:?}): {e:?} \
                 ({width}x{height}, {} bytes)",
                p.kind,
                p.bytes.len()
            ),
        };
        assert_eq!(picture.width, width, "visible width drift at packet {i}");
        assert_eq!(picture.height, height, "visible height drift at packet {i}");
        assert_eq!(
            dec.last_frame_shown(),
            Some(p.is_visible()),
            "wire visibility drift at packet {i} ({:?})",
            p.kind
        );
        if let Some(src) = p.source_index {
            assert_eq!(src, next_visible, "visible packet order drift at {i}");
            next_visible += 1;
        }
    }
    assert_eq!(
        next_visible, frame_count as u64,
        "every source frame must come out visible exactly once"
    );
});
