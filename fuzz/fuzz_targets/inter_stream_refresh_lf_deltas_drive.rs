#![no_main]

//! Fuzz: the caller-driven §9.7/§9.8 refresh + §9.4 lf-deltas family of
//! `Vp8InterStreamEncoder` P-frame entry points, in decode lockstep.
//!
//! `inter_stream_encode_decode_sequence` drives the scheduler front
//! door (`encode_frame_with_force`) only: the seven explicit-refresh
//! siblings — `encode_p_frame_with_refresh`, `…_and_intra_pick`,
//! `…_and_lf_deltas`, `…_and_lf_deltas_and_token_updates`,
//! `…_and_lf_deltas_and_intra_pick`,
//! `…_and_lf_deltas_and_fitted_token_prob_updates`, and
//! `…_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates` —
//! were reachable by no fuzz target. They own three seams the
//! scheduler path never exercises:
//!
//! * The **caller-supplied refresh ladder**: every
//!   `(refresh_golden, refresh_alternate, copy_to_golden,
//!   copy_to_alternate, refresh_last)` combination, including
//!   `refresh_last = false` (predict the next frame off a stale LAST)
//!   and the out-of-range copy selectors that must be rejected via
//!   `RefreshControls::validate`.
//! * The **§9.4 carried-delta state machine**: `LoopFilterDeltas`
//!   with per-slot `Some`/`None` mixes threads the across-frame
//!   `carried_ref_deltas` / `carried_mode_deltas` carry, which the
//!   harness locks against the public `LoopFilterDeltas::effective`
//!   resolution on every frame. Out-of-range magnitudes (|Δ| > 63)
//!   must be rejected by `validate` without perturbing the carry.
//! * The **§13.4 token-updates plumbing** on the `…_token_updates`
//!   sibling (attacker-shaped sparse `TokenProbUpdates` grids with raw
//!   0..=255 probability bytes) and the two-pass fitter siblings.
//!
//! Oracles beyond panic-freedom: every accepted P-frame decodes in a
//! long-lived `Vp8DecoderState` with locked dimensions; a P-frame
//! call before the first keyframe must surface `NoLastReference`; a
//! hostile copy selector must surface `Frame(_)`; the carried-delta
//! accessors must track `effective()` exactly; `frame_count` must
//! count every accepted frame.
//!
//! Caps: ≤ 40×40 luma, ≤ 8 P-frames per iteration, input ≤ 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    FrameKind, I420Frame, KeyframeParams, LoopFilterDeltas, RefreshControls, StreamEncodeError,
    TokenProbUpdates, TrellisStrength, Vp8DecoderState, Vp8InterStreamEncoder,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 8;
const MAX_P_FRAMES: usize = 8;
const MAX_DIM: u32 = 40;
const PER_FRAME_CTRL: usize = 12;

/// Decode `bytes` in the shared stateful decoder and assert geometry.
fn decode_and_check(dec: &mut Vp8DecoderState, bytes: &[u8], width: u32, height: u32, tag: &str) {
    let d = dec
        .decode_frame(bytes)
        .unwrap_or_else(|e| panic!("decoder rejected {tag}: {e:?} ({} bytes)", bytes.len()));
    assert_eq!(d.width, width, "{tag}: width drift");
    assert_eq!(d.height, height, "{tag}: height drift");
}

/// Build a `LoopFilterDeltas` from six control bytes. Magnitudes are
/// pre-masked into the legal −63..=63 band; the out-of-range rejection
/// leg is driven separately.
fn lf_deltas_from(ctl: &[u8]) -> LoopFilterDeltas {
    let delta = |b: u8| -> Option<i8> {
        if b & 0x01 == 0 {
            None
        } else {
            let mag = i16::from((b >> 2) & 0x3f);
            Some(if b & 0x02 != 0 { -mag } else { mag } as i8)
        }
    };
    LoopFilterDeltas {
        enabled: ctl[0] & 0x01 != 0,
        update: ctl[0] & 0x02 != 0,
        ref_frame_delta: [delta(ctl[1]), delta(ctl[2]), delta(ctl[3]), delta(ctl[4])],
        mode_delta: [
            delta(ctl[5]),
            delta(ctl[1].rotate_left(3)),
            delta(ctl[2].rotate_left(5)),
            delta(ctl[3].rotate_left(7)),
        ],
    }
}

/// Sparse attacker-shaped `TokenProbUpdates` grid: `n` cells set to raw
/// probability bytes at positions walked from `seed`.
fn token_updates_from(seed: u8, prob: u8, n: usize) -> TokenProbUpdates {
    let mut out: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    let mut s = usize::from(seed);
    for k in 0..n {
        let i = s % 4;
        let j = (s / 4) % 8;
        let c = (s / 32) % 3;
        let t = (s / 96) % 11;
        out[i][j][c][t] = Some(prob.wrapping_add(k as u8));
        s = s.wrapping_mul(31).wrapping_add(17);
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = 1 + u32::from(data[0]) % MAX_DIM;
    let height = 1 + u32::from(data[1]) % MAX_DIM;
    let params = KeyframeParams {
        y_ac_qi: data[2] % 128,
        loop_filter_level: data[3] % 64,
        sharpness_level: data[4] % 8,
        nbr_of_dct_partitions: 1u8 << (data[5] % 4),
        filter_type: (data[6] & 0x01) != 0,
        trellis_strength: if data[6] & 0x02 != 0 {
            TrellisStrength::OFF
        } else {
            TrellisStrength::DEFAULT
        },
    };
    let p_frames = 1 + usize::from(data[7]) % MAX_P_FRAMES;
    let ctrl = &data[HEADER_BYTES..];

    let mut enc = match Vp8InterStreamEncoder::new(params, 1_000_000) {
        Some(e) => e,
        None => return,
    };
    let mut dec = Vp8DecoderState::new();

    let w = width as usize;
    let h = height as usize;
    let uv_len = w.div_ceil(2) * h.div_ceil(2);
    let make_frame = |fill: u8| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| fill.wrapping_add((i % 251) as u8 & 0x1f))
            .collect();
        (y, vec![fill ^ 0x40; uv_len], vec![fill ^ 0x80; uv_len])
    };

    // ---- NoLastReference leg: any P entry point before frame 0 -------
    {
        let (y, u, v) = make_frame(0x60);
        let frame = I420Frame::packed(width, height, &y, &u, &v);
        match enc.encode_p_frame_with_refresh(&frame, &RefreshControls::default()) {
            Err(StreamEncodeError::NoLastReference) => {}
            other => panic!("P-frame before first keyframe must be NoLastReference, got {other:?}"),
        }
    }

    // ---- Seed keyframe -------------------------------------------------
    {
        let (y, u, v) = make_frame(0x80);
        let frame = I420Frame::packed(width, height, &y, &u, &v);
        let emitted = enc.encode_frame(&frame).expect("seed keyframe must encode");
        assert_eq!(emitted.kind, FrameKind::Key, "frame 0 must be Key");
        decode_and_check(&mut dec, &emitted.bytes, width, height, "seed keyframe");
    }

    // Across-frame §9.4 carry model, locked against the accessors.
    let mut model_ref = [0i16; 4];
    let mut model_mode = [0i16; 4];
    let mut expected_count = 1u64;

    for f in 0..p_frames {
        let base = f * PER_FRAME_CTRL;
        let get = |k: usize| ctrl.get(base + k).copied().unwrap_or(0x55);
        let sel = get(0) % 7;
        // §9.7 wire rule: the copy selectors are only coded when the
        // matching refresh flag is 0 — `validate` rejects a nonzero
        // selector alongside a set refresh flag, so the in-range leg
        // zeroes the selector in that case.
        let refresh_golden = get(1) & 0x01 != 0;
        let refresh_alt = get(1) & 0x02 != 0;
        let rc = RefreshControls {
            refresh_golden_frame: refresh_golden,
            refresh_alternate_frame: refresh_alt,
            copy_buffer_to_golden: if refresh_golden { 0 } else { (get(1) >> 2) % 3 },
            copy_buffer_to_alternate: if refresh_alt { 0 } else { (get(1) >> 4) % 3 },
            refresh_last: get(1) & 0x40 != 0,
        };
        let lf_ctl = [get(2), get(3), get(4), get(5), get(6), get(7)];
        let lf = lf_deltas_from(&lf_ctl);
        let (y, u, v) = make_frame(get(8).wrapping_add(f as u8));
        let frame = I420Frame::packed(width, height, &y, &u, &v);

        // Hostile-selector rejection leg (state must be untouched).
        if get(9) & 0x01 != 0 {
            let bad = RefreshControls {
                copy_buffer_to_golden: 3 + get(9) % 8,
                ..rc
            };
            match enc.encode_p_frame_with_refresh_and_lf_deltas(&frame, &bad, &lf) {
                Err(StreamEncodeError::Frame(_)) => {}
                other => panic!("copy selector > 2 must be rejected, got {other:?}"),
            }
            assert_eq!(
                enc.frame_count(),
                expected_count,
                "rejected frame must not advance frame_count"
            );
        }

        // Out-of-range delta magnitude rejection leg.
        if get(9) & 0x02 != 0 && !lf.enabled {
            let mut bad_lf = lf;
            bad_lf.enabled = true;
            bad_lf.update = true;
            bad_lf.ref_frame_delta[0] = Some(64 + (get(9) % 60) as i8);
            match enc.encode_p_frame_with_refresh_and_lf_deltas(&frame, &rc, &bad_lf) {
                Err(StreamEncodeError::Frame(_)) => {}
                other => panic!("|delta| > 63 must be rejected, got {other:?}"),
            }
        }

        let lf_active = sel >= 2;
        let result = match sel {
            0 => enc.encode_p_frame_with_refresh(&frame, &rc),
            1 => enc.encode_p_frame_with_refresh_and_intra_pick(&frame, &rc),
            2 => enc.encode_p_frame_with_refresh_and_lf_deltas(&frame, &rc, &lf),
            3 => {
                let tu = if get(10) & 0x01 != 0 {
                    Some(token_updates_from(get(10), get(11), 1 + usize::from(get(11)) % 4))
                } else {
                    None
                };
                enc.encode_p_frame_with_refresh_and_lf_deltas_and_token_updates(
                    &frame,
                    &rc,
                    &lf,
                    tu.as_ref(),
                )
            }
            4 => enc.encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(&frame, &rc, &lf),
            5 => enc.encode_p_frame_with_refresh_and_lf_deltas_and_fitted_token_prob_updates(
                &frame, &rc, &lf,
            ),
            _ => enc
                .encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates(
                    &frame, &rc, &lf,
                ),
        };
        let emitted = result.unwrap_or_else(|e| {
            panic!(
                "sibling {sel} rejected an in-range P-frame {f}: {e:?} \
                 ({width}x{height}, rc {rc:?}, lf {lf:?})"
            )
        });
        assert_eq!(emitted.kind, FrameKind::InterZeroMv, "P-frame kind drift");
        expected_count += 1;
        assert_eq!(enc.frame_count(), expected_count, "frame_count drift");
        decode_and_check(
            &mut dec,
            &emitted.bytes,
            width,
            height,
            &format!("P-frame {f} (sibling {sel})"),
        );

        // §9.4 carry lockstep. The intra-pick-only sibling (1) and the
        // bare-refresh sibling (0) run with default (disabled) deltas
        // and must leave the carry untouched.
        if lf_active && lf.enabled {
            let (er, em) = lf.effective(model_ref, model_mode);
            model_ref = er;
            model_mode = em;
        }
        assert_eq!(
            enc.carried_ref_deltas(),
            model_ref,
            "carried ref-delta drift at P-frame {f} (sibling {sel})"
        );
        assert_eq!(
            enc.carried_mode_deltas(),
            model_mode,
            "carried mode-delta drift at P-frame {f} (sibling {sel})"
        );
    }
});
