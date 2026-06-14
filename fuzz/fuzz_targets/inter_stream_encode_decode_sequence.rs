#![no_main]

//! Fuzz: multi-frame I + P stream encode→stateful-decode sequence
//! (RFC 6386 §9 / §14 / §15 / §16.2 / §18 / §19.2).
//!
//! Every existing encode-side target stops at a *single* key frame
//! (`encode_keyframe`, `encode_keyframe_with_reconstruction*`): the
//! cross-frame layer — `Vp8InterStreamEncoder` and the §9 three-slot
//! reference buffer it owns — has never been driven by a fuzzer. That
//! is exactly the state-heavy surface where a stream encoder can drift:
//! the §9.7 reference-refresh ladder (a key frame replaces all three
//! slots, a ZERO_MV P-frame refreshes LAST only), the keyframe-interval
//! scheduler with its `force_keyframe` re-anchoring, the across-frame
//! §9.4 `ref_frame_delta[]` / `mode_delta[]` carry, and the
//! locked-after-first-frame dimension guard.
//!
//! This target stitches the producer against the *stateful* decoder
//! (`Vp8DecoderState::decode_frame`) so the whole loop runs end-to-end:
//! a fuzz-shaped sequence of I420 frames is pushed through one
//! `Vp8InterStreamEncoder`, and each emitted frame's bytes are fed into
//! one long-lived `Vp8DecoderState`. The decoder therefore carries its
//! own §9 reference slots forward across the stream and reconstructs
//! the ZERO_MV P-frames from the LAST it kept — the §16.2 / §18 inter
//! prediction path the single-frame targets cannot reach.
//!
//! All encoder parameters are normalised into their §9.4 / §9.5 / §9.6
//! legal range, so the encoder must never reject an accepted input and
//! the decoder must always accept the encoder's own output. An `Err`
//! from either side, a dimension drift, or a K/P-classification drift
//! relative to the encoder's own scheduler is a finding; the harness
//! panics on each.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`     | Visible width: `1 + (b % 48)` luma px |
//! | `[1]`     | Visible height: `1 + (b % 48)` luma px |
//! | `[2]`     | `y_ac_qi = b % 128` |
//! | `[3]`     | `loop_filter_level = b % 64` (0 ⇒ §15 skip path) |
//! | `[4]`     | `sharpness_level = b % 8` |
//! | `[5]`     | `nbr_of_dct_partitions ∈ {1, 2, 4, 8}` via `b % 4` |
//! | `[6]`     | flags: bit 0 = `filter_type` (§15.2 simple vs §15.3 normal) |
//! | `[7]`     | `keyframe_interval = 1 + (b % 8)` |
//! | `[8]`     | frame count: `1 + (b % 12)` |
//! | `[9..N]`  | one control byte per frame: bit 0 = `force_keyframe`; the rest seeds that frame's plane fill |
//! | `[N..]`   | I420 pixel payload tiled across the three planes |
//!
//! Visible area is capped at 48 × 48 = 2 304 luma px (≤ 9 MBs) and the
//! frame count at 12, so the worst case is a dozen sub-9-MB
//! encode+decode round-trips — well inside the per-iteration wall-time
//! and the shared [`oxideav_vp8_fuzz::accept_dimensions`] memory
//! budget. Hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{FrameKind, I420Frame, KeyframeParams, Vp8DecoderState, Vp8InterStreamEncoder};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 9;
const MAX_FRAMES: usize = 12;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions, kept small so a dozen frames stay cheap.
    let width = 1u32 + u32::from(data[0] % 48);
    let height = 1u32 + u32::from(data[1] % 48);
    if !accept_dimensions(width, height) {
        return;
    }

    // Always-valid §9.4 / §9.5 / §9.6 parameter envelope: the rejection
    // surface belongs to `panic_free_encode_keyframe`; here every
    // accepted input must encode and decode cleanly.
    let params = KeyframeParams {
        y_ac_qi: data[2] % 128,
        loop_filter_level: data[3] % 64,
        sharpness_level: data[4] % 8,
        nbr_of_dct_partitions: 1u8 << (data[5] % 4),
        filter_type: (data[6] & 0x01) != 0,
    };

    let keyframe_interval = 1u64 + u64::from(data[7] % 8);
    let frame_count = 1usize + usize::from(data[8] % (MAX_FRAMES as u8));

    // Per-frame control bytes follow the fixed header; the remainder is
    // the pixel payload tiled into each frame's planes.
    let ctrl = &data[HEADER_BYTES..];
    let payload_start = (HEADER_BYTES + frame_count).min(data.len());
    let payload = &data[payload_start..];

    let mut enc = match Vp8InterStreamEncoder::new(params, keyframe_interval) {
        Some(e) => e,
        // keyframe_interval is built as >= 1, so this is unreachable;
        // bail rather than unwrap to keep the harness panic-free on its
        // own account.
        None => return,
    };
    let mut dec = Vp8DecoderState::new();

    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;

    for f in 0..frame_count {
        // Each frame gets a distinct flat fill so successive P-frames
        // carry a real (if synthetic) residual against the prior LAST.
        let seed = ctrl.get(f).copied().unwrap_or(0);
        let force_keyframe = (seed & 0x01) != 0;
        let base = if payload.is_empty() {
            seed
        } else {
            payload[usize::from(seed) % payload.len()]
        };
        let fill = base.wrapping_add(f as u8);

        let y_plane = vec![fill; y_len];
        let u_plane = vec![fill ^ 0x40; uv_len];
        let v_plane = vec![fill ^ 0x80; uv_len];
        let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);

        // The encoder's own scheduler decides K vs P; capture its
        // pre-encode verdict so we can assert the emitted classification
        // matches (force overrides it to Key).
        let predicted = enc.next_frame_is_keyframe();

        let emitted = match enc.encode_frame_with_force(&frame, force_keyframe) {
            Ok(e) => e,
            Err(e) => panic!(
                "inter stream encoder rejected an in-range frame {f}: {e:?} \
                 ({width}x{height}, interval {keyframe_interval}, params {params:?})"
            ),
        };

        // K/P classification lockstep: a forced frame must be Key;
        // otherwise it must match the scheduler's own pre-encode verdict.
        let expected_kind = if force_keyframe {
            FrameKind::Key
        } else {
            predicted
        };
        assert_eq!(
            emitted.kind, expected_kind,
            "K/P classification drift at frame {f} \
             (force {force_keyframe}, interval {keyframe_interval})"
        );
        assert_eq!(
            emitted.frame_index, f as u64,
            "stream frame_index drift at frame {f}"
        );

        // Decode leg: one long-lived stateful decoder carries the §9
        // reference slots across the stream, so P-frames reconstruct
        // from the LAST it kept.
        let decoded = match dec.decode_frame(&emitted.bytes) {
            Ok(d) => d,
            Err(e) => panic!(
                "stateful decoder rejected encoder frame {f} ({:?}): {e:?} \
                 ({width}x{height}, {} bytes)",
                emitted.kind,
                emitted.bytes.len()
            ),
        };

        assert_eq!(decoded.width, width, "visible width drift at frame {f}");
        assert_eq!(decoded.height, height, "visible height drift at frame {f}");
        assert_eq!(
            decoded.y.len(),
            y_len,
            "decoded luma length drift at frame {f}"
        );
        assert_eq!(
            decoded.u.len(),
            uv_len,
            "decoded U length drift at frame {f}"
        );
        assert_eq!(
            decoded.v.len(),
            uv_len,
            "decoded V length drift at frame {f}"
        );
    }
});
