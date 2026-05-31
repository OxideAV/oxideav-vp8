#![no_main]

//! Fuzz: arbitrary bytes → `decode_vp8` (one-shot keyframe decode).
//!
//! Contract: `decode_vp8` MUST return a `Result` for every input and
//! never panic. The §9.1 frame tag pre-screen inside `decode_vp8`
//! rejects most random byte sequences early (key-frame bit, 3-byte
//! `0x9d 0x01 0x2a` start-code, version field, …); we add an
//! additional pre-flight on the visible width/height to guard
//! against decode-pipeline OOM at the wire-legal extremes — see
//! [`oxideav_vp8_fuzz::accept_dimensions`].
//!
//! Errors from `decode_vp8` are fine — the trait is defined to surface
//! them. We only fail on a Rust-level panic.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{decode_vp8, frame_header::Vp8FrameHeader};
use oxideav_vp8_fuzz::accept_dimensions;

fuzz_target!(|data: &[u8]| {
    // Pre-flight: parse just the §9.1 uncompressed header (10 bytes
    // for keyframes) so we can apply the dimension cap before
    // `decode_vp8` allocates any reconstruction planes. If the header
    // parse itself fails — or this is an inter-frame (no width /
    // height; `decode_vp8` would reject it anyway) — that's a
    // Result-return, not a panic, and we simply stop.
    let Ok(header) = Vp8FrameHeader::parse(data) else {
        return;
    };
    let (Some(w), Some(h)) = (header.width, header.height) else {
        return;
    };
    if !accept_dimensions(u32::from(w), u32::from(h)) {
        return;
    }
    let _ = decode_vp8(data);
});
