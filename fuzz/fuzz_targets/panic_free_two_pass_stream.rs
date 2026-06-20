#![no_main]

//! Fuzz: panic-freedom of the public [`Vp8TwoPassEncoder`] multi-frame
//! driver.
//!
//! Sibling to `panic_free_encode_keyframe` but on the **inter** /
//! rate-controlled side: this target exercises the public two-pass
//! state machine (the `Vp8TwoPassEncoder::first_pass_analyze` →
//! `encode_frame` loop) which is the encoder's main long-running
//! stream entry point and which `panic_free_encode_keyframe` cannot
//! reach (a one-shot `encode_keyframe` never crosses into
//! `encode_p_frame_multi_ref`, the §9.7 reference-frame refresh
//! ladder, the scene-cut keyframe-force, or the
//! `golden_interval`-elapsed keyframe-force). Random-byte inputs
//! routinely cook up degenerate configs (`golden_interval = 0`,
//! out-of-range `qindex`, overshoot ratio `NaN`, scene-cut on every
//! frame, …) and the contract is that the encoder either emits the
//! bytes or surfaces a structured `Vp8Error` — never a panic, abort,
//! debug-arithmetic overflow, or OOB index.
//!
//! Input layout (consumed front-to-back; all single-byte fields):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`             | Visible width  MB-units (`1 + b%8` → 16..=128 px) |
//! | `[1]`             | Visible height MB-units (`1 + b%8` → 16..=128 px) |
//! | `[2]`             | Base `qindex` raw byte (full 0..=255 range — out-of-range surfaces as `Vp8Error`) |
//! | `[3]`             | `lf_level` raw byte (full 0..=255 range) |
//! | `[4]`             | `golden_interval` raw byte (full 0..=255 — the harness clamps the frame loop to `MAX_FRAMES` so even `golden_interval=0` is bounded) |
//! | `[5]`             | `alt_ref_interval` raw byte (full 0..=255 range) |
//! | `[6]`             | Frame count (`1 + b%MAX_FRAMES`) |
//! | `[7]`             | Scene-cut bitmap (low `frame_count` bits — bit `i` forces a scene-cut on frame `i`) |
//! | `[8..8+N]`        | `bits_per_mb` raw bytes (one per frame; rescaled into `0.0..=1024.0` so the qindex picker exercises the full delta envelope) |
//! | `[remainder]`     | Pseudo-random I420 pixel payload, tiled across the frame planes |
//!
//! The frame count is hard-capped at [`MAX_FRAMES`] (= 4) so a single
//! libFuzzer iteration stays within the per-iteration wall-time and
//! memory budget even at the maximum 128×128 frame size (4 × 128×128 ≈
//! 96 KiB of luma). The pixel tail is tiled across all frames so a
//! short input still produces populated planes.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth); ≤ 4 frames per iteration; ≤ 128×128 px
//! per frame.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::encoder::{
    FrameComplexity, LoopFilterMode, Vp8EncoderConfig, Vp8TwoPassConfig, Vp8TwoPassEncoder,
};
use oxideav_vp8::I420Frame;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 8;
const MAX_FRAMES: usize = 4;
const MAX_MB_PER_AXIS: u32 = 8; // 8 × 16 = 128 px per axis

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions in MB units (16 px each); cap at 128 px to bound
    // the per-iteration memory footprint even with `MAX_FRAMES` frames.
    let mb_w = 1u32 + u32::from(data[0]) % MAX_MB_PER_AXIS;
    let mb_h = 1u32 + u32::from(data[1]) % MAX_MB_PER_AXIS;
    let width = mb_w * 16;
    let height = mb_h * 16;

    // §9.6 quant + §15 loop-filter knobs, fed raw so the encoder's own
    // range-validation surfaces a structured `Vp8Error` for the
    // out-of-range tail of each byte's domain (rather than the
    // happy-path encode for the in-range head).
    let qindex = data[2];
    let lf_level = data[3];
    let golden_interval = u32::from(data[4]);
    let alt_ref_interval = u32::from(data[5]);
    let frame_count = 1 + (usize::from(data[6]) % MAX_FRAMES);
    let scene_cut_mask = data[7];

    let cfg = Vp8TwoPassConfig {
        base: Vp8EncoderConfig {
            qindex,
            lf_level,
            lf_mode: LoopFilterMode::Auto,
            golden_interval,
            alt_ref_interval,
            lookahead_window: 0,
            target_bitrate_bps: 0,
        },
        target_bitrate_bps: 0,
        overshoot_ratio: 1.2,
        fps_num: 0,
        fps_den: 1,
    };

    // I420 plane sizes for tightly-packed strides at the chosen
    // dimensions.
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;

    // Decompose the input tail into per-frame `bits_per_mb` bytes plus
    // a pixel payload. The `bits_per_mb` bytes start immediately after
    // the eight-byte header; the pixel payload starts immediately
    // after the per-frame bytes.
    let bpm_start = HEADER_BYTES;
    let bpm_end = bpm_start.saturating_add(frame_count).min(data.len());
    let bpm_bytes = &data[bpm_start..bpm_end];
    let payload = if bpm_end < data.len() {
        &data[bpm_end..]
    } else {
        &[][..]
    };

    // Synthesise one tiled I420 backing buffer per frame so the
    // encoder loop sees fresh content frame-to-frame (a single shared
    // buffer would short-circuit the inter prediction path to "all
    // zero residue"). Buffers are owned outside the loop so the
    // I420Frame borrows live long enough.
    let mut planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
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
        planes.push((y, u, v));
    }

    let frames: Vec<I420Frame<'_>> = planes
        .iter()
        .map(|(y, u, v)| I420Frame::packed(width, height, y, u, v))
        .collect();

    let mut encoder = Vp8TwoPassEncoder::new(cfg);

    // First-pass analyze — surfaces `Vp8Error` for unsupported configs
    // (e.g. zero-sized frames) without panicking. We continue to the
    // second pass either way so the harness exercises the
    // first-pass-skipped fallback too: `Vp8TwoPassEncoder::encode_frame`
    // is contracted to fall back to `config.base.qindex` when
    // `first_pass_analyze` hasn't populated the mean.
    let analysed = encoder.first_pass_analyze(&frames).ok();

    for (i, frame) in frames.iter().enumerate() {
        let complexity = match analysed.as_ref() {
            Some(stats) if i < stats.len() => {
                // Use the analysed stats but optionally override the
                // scene-cut flag from the fuzzer-provided bitmap so we
                // exercise the §9.7 force-keyframe-on-scene-cut path
                // even mid-stream.
                let mut c = stats[i];
                if scene_cut_mask & (1 << i) != 0 {
                    c.scene_cut = true;
                }
                c
            }
            _ => {
                // Synthesise a complexity hint from the fuzzer's
                // per-frame `bits_per_mb` byte so the qindex picker
                // exercises the full delta envelope even on the
                // first-pass-skipped fallback path.
                let bpm = bpm_bytes
                    .get(i)
                    .copied()
                    .map(|b| f32::from(b) * 4.0) // 0..=1020.0
                    .unwrap_or(0.0);
                FrameComplexity {
                    frame_index: i as u32,
                    bits_per_mb: bpm,
                    scene_cut: scene_cut_mask & (1 << i) != 0,
                }
            }
        };

        // The contract: a structured `Err` is fine, a panic is not.
        let _ = encoder.encode_frame(frame, complexity);
    }
});
