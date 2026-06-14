#![no_main]

//! Fuzz: the public [`Vp8KeyframeStreamEncoder`] multi-frame driver —
//! both the plain [`Vp8KeyframeStreamEncoder::encode_frame`] path and
//! the §13.4 fitted companion
//! [`Vp8KeyframeStreamEncoder::encode_frame_with_fitted_token_prob_updates`].
//!
//! Every existing encode target stops short of this driver:
//!
//! * `panic_free_encode_keyframe` calls the one-shot
//!   `encode_keyframe(&I420Frame, &KeyframeParams)` — it never builds a
//!   cross-frame encoder, so the dimension-lock guard, the running
//!   frame counter, and the §9.7 / §9.8 three-slot reference refresh
//!   are out of its reach.
//! * `panic_free_two_pass_stream` drives `Vp8TwoPassEncoder` (the
//!   complexity-aware rate-controlled I + P state machine) and
//!   `inter_stream_encode_decode_sequence` drives
//!   `Vp8InterStreamEncoder` (the ZERO_MV-P interleave) — both are the
//!   *inter* stream drivers. Neither touches
//!   `Vp8KeyframeStreamEncoder`, the all-key-frame stream driver, nor
//!   its §13.4 fitted-token-prob-update entry point.
//!
//! So three public surfaces reach the fuzzer here for the first time:
//!
//! 1. **The dimension-lock state machine.** The first successful
//!    `encode_frame` locks `(width, height)`; a later frame with
//!    different dimensions must surface
//!    `StreamEncodeError::DimensionsChanged` and leave the encoder
//!    state (frame count + reference slots) unchanged. This target
//!    deliberately changes dimensions mid-stream on a fuzz-chosen frame
//!    index and asserts the guard fired without advancing the counter.
//!
//! 2. **The §9.7 / §9.8 three-slot keyframe refresh.** After every
//!    successful encode, all three reference slots
//!    (`last` / `golden` / `altref`) must be populated with a clone of
//!    the *same* post-§15 reconstruction. `RefFrameSlot: PartialEq`, so
//!    the harness asserts `last == golden == altref` after each
//!    success — a stale-slot or partial-refresh bug surfaces directly.
//!
//! 3. **The fitted-token-prob-update fitter through a stream driver.**
//!    `encode_frame_with_fitted_token_prob_updates` routes through
//!    `encode_keyframe_with_reconstruction_and_fitted_token_prob_updates`,
//!    whose round-157 safety guard may fall back to the default pass;
//!    either outcome must keep the slot reconstruction consistent with
//!    the emitted bytes. A fuzz-chosen per-frame bit selects the fitted
//!    vs plain path so both run inside one encoder lifetime.
//!
//! Beyond panic-freedom, two byte-level oracles run on every emitted
//! frame: it must self-decode through a fresh `Vp8DecoderState` (a key
//! frame is independently decodable, so a stateless decoder suffices),
//! and the decoded visible dimensions must equal the locked stream
//! dimensions. Every decoded plane byte is folded into an FNV-1a
//! accumulator so a short-write / stale-stride bug surfaces under ASan.
//!
//! Input layout (consumed front-to-back; all single-byte fields):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`        | Width  MB-units (`1 + b%MAX_MB` → 16..=64 px) |
//! | `[1]`        | Height MB-units (`1 + b%MAX_MB` → 16..=64 px) |
//! | `[2]`        | `y_ac_qi` raw byte (full 0..=255 — > 127 surfaces `QuantIndexOutOfRange`) |
//! | `[3]`        | `loop_filter_level` raw byte (> 63 surfaces `LoopFilterLevelOutOfRange`) |
//! | `[4]`        | `sharpness_level` raw byte (> 7 surfaces `SharpnessLevelOutOfRange`) |
//! | `[5]`        | `nbr_of_dct_partitions` raw byte (non-{1,2,4,8} surfaces `InvalidDctPartitionCount`) |
//! | `[6]`        | bit0 `filter_type`; bit1.. fitted-path bitmap base |
//! | `[7]`        | Frame count (`1 + b%MAX_FRAMES`) |
//! | `[8]`        | Fitted-path bitmap (bit `i` → use the fitted entry point on frame `i`) |
//! | `[9]`        | Resize-trigger frame index (`b%MAX_FRAMES`; that frame is fed altered dimensions) |
//! | `[10..]`     | Pseudo-random I420 pixel payload, tiled across the frame planes |
//!
//! Hard caps: input ≤ 4 KiB; ≤ 4 frames per iteration; ≤ 64 × 64 px per
//! frame (well inside the shared `accept_dimensions` budget), so even
//! four frames plus self-decode stay within the per-iteration wall-time
//! and memory budget.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    I420Frame, KeyframeParams, StreamEncodeError, Vp8DecoderState, Vp8KeyframeStreamEncoder,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 10;
const MAX_FRAMES: usize = 4;
const MAX_MB_PER_AXIS: u32 = 4; // 4 × 16 = 64 px per axis

/// Fold a byte slice into a running FNV-1a hash so ASan reads every
/// decoded plane byte the self-decode claims to have produced.
fn fnv1a(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= u64::from(b);
        *acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Build one tiled I420 backing buffer for frame `i` from the payload
/// tail. Distinct per-frame offset keeps content fresh frame-to-frame.
fn fill_planes(
    payload: &[u8],
    i: usize,
    y_len: usize,
    uv_len: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; y_len];
    let mut u = vec![0u8; uv_len];
    let mut v = vec![0u8; uv_len];
    if !payload.is_empty() {
        let base = i.wrapping_mul(31);
        for (k, slot) in y.iter_mut().enumerate() {
            *slot = payload[(k + base) % payload.len()];
        }
        for (k, slot) in u.iter_mut().enumerate() {
            *slot = payload[(k + base + y_len) % payload.len()];
        }
        for (k, slot) in v.iter_mut().enumerate() {
            *slot = payload[(k + base + y_len + uv_len) % payload.len()];
        }
    }
    (y, u, v)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions in MB units (16 px each); cap at 64 px so four
    // frames plus a self-decode each stay inside the per-iteration
    // budget.
    let mb_w = 1u32 + u32::from(data[0]) % MAX_MB_PER_AXIS;
    let mb_h = 1u32 + u32::from(data[1]) % MAX_MB_PER_AXIS;
    let width = mb_w * 16;
    let height = mb_h * 16;

    // §9.6 quant + §9.4 loop-filter + §9.5 partition knobs, fed raw so
    // the encoder's own range validation surfaces a structured
    // `EncodeError` for the out-of-range tail of each byte's domain.
    let params = KeyframeParams {
        y_ac_qi: data[2],
        loop_filter_level: data[3],
        sharpness_level: data[4],
        nbr_of_dct_partitions: data[5],
        filter_type: (data[6] & 1) != 0,
    };

    let frame_count = 1 + (usize::from(data[7]) % MAX_FRAMES);
    let fitted_mask = data[8];
    let resize_at = usize::from(data[9]) % MAX_FRAMES;

    let payload = &data[HEADER_BYTES..];

    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;

    // Alternate dimensions for the mid-stream resize probe: bump each
    // axis by one MB so the geometry genuinely differs from the locked
    // one (and stays inside the budget).
    let alt_width = (mb_w + 1) * 16;
    let alt_height = (mb_h + 1) * 16;
    let aw = alt_width as usize;
    let ah = alt_height as usize;
    let alt_uvw = aw.div_ceil(2);
    let alt_uvh = ah.div_ceil(2);
    let alt_y_len = aw * ah;
    let alt_uv_len = alt_uvw * alt_uvh;

    let mut enc = Vp8KeyframeStreamEncoder::new(params);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;

    for i in 0..frame_count {
        // On the resize-trigger frame (but never the first, which would
        // simply lock the alternate dimensions instead of testing the
        // guard) feed altered dimensions and assert the lock fired.
        let use_alt = i == resize_at && i != 0;
        let (fw, fh, fy, fu, fv) = if use_alt {
            let (y, u, v) = fill_planes(payload, i, alt_y_len, alt_uv_len);
            (alt_width, alt_height, y, u, v)
        } else {
            let (y, u, v) = fill_planes(payload, i, y_len, uv_len);
            (width, height, y, u, v)
        };
        let frame = I420Frame::packed(fw, fh, &fy, &fu, &fv);

        let count_before = enc.frame_count();
        let use_fitted = (fitted_mask >> (i % 8)) & 1 != 0;
        let result = if use_fitted {
            enc.encode_frame_with_fitted_token_prob_updates(&frame)
        } else {
            enc.encode_frame(&frame)
        };

        match result {
            Ok(bytes) => {
                // A non-first frame fed alternate dimensions must NOT
                // succeed — the lock guard should have rejected it.
                if use_alt {
                    panic!("encode_frame accepted a mid-stream dimension change");
                }

                // §9.7 / §9.8 three-slot keyframe refresh: every key
                // frame replaces all three slots with a clone of the
                // same reconstruction.
                let last = enc.last().expect("LAST slot populated after success");
                let golden = enc.golden().expect("GOLDEN slot populated after success");
                let altref = enc.altref().expect("ALTREF slot populated after success");
                if last != golden || golden != altref {
                    panic!("three-slot keyframe refresh left slots divergent");
                }

                // Frame counter advances by exactly one on success.
                if enc.frame_count() != count_before + 1 {
                    panic!("frame_count did not advance by one on success");
                }

                // Dimensions lock to the first accepted frame's size.
                if enc.dimensions() != Some((width, height)) {
                    panic!("dimensions not locked to first accepted frame");
                }

                // Self-decode: a key frame is independently decodable.
                let mut dec = Vp8DecoderState::new();
                match dec.decode_frame(&bytes) {
                    Ok(decoded) => {
                        if decoded.width != width || decoded.height != height {
                            panic!("decoded dimensions drifted from locked stream geometry");
                        }
                        fnv1a(&mut hash, &decoded.y);
                        fnv1a(&mut hash, &decoded.u);
                        fnv1a(&mut hash, &decoded.v);
                    }
                    Err(_) => {
                        // The stream encoder promises its emitted key
                        // frames decode; a decode error on an accepted
                        // frame is a real finding.
                        panic!("emitted key frame failed to self-decode");
                    }
                }
            }
            Err(StreamEncodeError::DimensionsChanged { .. }) => {
                // Expected only on the alternate-dimension probe; the
                // guard must leave the counter unchanged.
                if enc.frame_count() != count_before {
                    panic!("DimensionsChanged advanced the frame counter");
                }
            }
            Err(_) => {
                // Any other structured error (e.g. an out-of-range
                // quant / loop-filter / partition knob) is acceptable —
                // those reject the whole stream uniformly, so further
                // frames would reject identically. Stop the loop.
                break;
            }
        }
    }

    std::hint::black_box(hash);
});
