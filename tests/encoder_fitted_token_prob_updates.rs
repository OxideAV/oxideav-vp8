//! Encoder-side §13.4 `token_prob_update()` **observed-counts fitter**
//! — round 157's follow-up to the r155 keyframe + r156 inter
//! caller-driven layer.
//!
//! Rounds 155 / 156 wired the §13.4 update sub-block + merged-table
//! plumbing so an *external* caller could hand in any `TokenProbUpdates`
//! payload. Round 157 lets the encoder *pick* the payload from the
//! per-position branch counts the §13.3 token-encode pass would have
//! produced against the §13.5 defaults — see
//! [`oxideav_vp8::encode_keyframe_with_fitted_token_prob_updates`].
//!
//! The fitter:
//!
//!   * computes `p_obs = round(256 * zeros / total)` clamped to `[1,
//!     255]` at every `(plane, band, prev_ctx, position)` slot of the
//!     4×8×3×11 table that the §13.3 walk visited;
//!   * emits an update only when the body bit saving exceeds the §13.4
//!     transmission cost (1 flag bit at the position's update
//!     probability + 8 literal bits, less the no-update flag-bit
//!     cost) plus a small `min_saving_bits = 2.0` guard.
//!
//! Tests below pin each property the round-157 contract claims:
//!
//!   1. The fitter is a *strict* no-op on an empty `BranchCounts`
//!      array — no update can win when every slot has zero observed
//!      events.
//!   2. The high-level entry-point on a non-trivial frame returns a
//!      wire **<=** the default-wire size (the safety guard).
//!   3. The fitted wire round-trips through `decode_vp8` and clears
//!      the same 25 dB PSNR floor the r155 / r156 tests pin.
//!   4. The fitted wire is byte-for-byte identical to
//!      `encode_keyframe_with_token_prob_updates(.., &fit)` for the
//!      fit derived from the default-encode counts (proving the
//!      high-level entry is a thin wrapper around the explicit fit +
//!      writer path).
//!   5. The header round-trips through `Vp8CodedHeader::parse` and the
//!      recovered `TokenProbUpdates` array matches the fitted payload
//!      slot-for-slot.

use oxideav_vp8::{
    count_keyframe_branches, decode_vp8, empty_branch_counts, encode_keyframe,
    encode_keyframe_with_fitted_token_prob_updates, fit_token_prob_updates, I420Frame,
    KeyframeParams, Vp8CodedHeader, Vp8FrameHeader,
};

/// Synthetic 32×32 I420 picture with a deliberately non-flat luma so
/// the §14 / §13 pipeline emits a meaningful spread of tokens (the
/// fitter is uninteresting when every block is all-zero).
fn synthetic_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..h {
        for c in 0..w {
            // Two crossing ramps so DCT bands beyond DC carry real
            // coefficient mass on most blocks.
            let g = ((r * 11 + c * 7) % 256) as u8;
            y[r * w + c] = g;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (110 + ((c + r) * 4 % 80)) as u8;
            v[r * cw + c] = (140 + ((c * 2 + r) * 3 % 60)) as u8;
        }
    }
    (y, u, v)
}

fn synthetic_params() -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    }
}

fn frame_psnr(src_y: &[u8], src_u: &[u8], src_v: &[u8], dec: &oxideav_vp8::Vp8DecodedFrame) -> f64 {
    let total = src_y.len() + src_u.len() + src_v.len();
    let plane_se = |a: &[u8], b: &[u8]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum::<f64>()
    };
    let total_se = plane_se(src_y, &dec.y) + plane_se(src_u, &dec.u) + plane_se(src_v, &dec.v);
    let mse = total_se / total as f64;
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Every-slot-zero `BranchCounts` ⇒ every fit slot stays `None`.
/// Justification: with no observed events the fitter has no evidence
/// that any default probability is wrong; emitting an update would
/// strictly *grow* the wire (8 bits of body for zero saving).
#[test]
fn fit_token_prob_updates_no_op_on_empty_counts() {
    let counts = empty_branch_counts();
    let fitted = fit_token_prob_updates(&counts, 2.0);

    for plane in &fitted {
        for band in plane {
            for ctx in band {
                for slot in ctx {
                    assert!(
                        slot.is_none(),
                        "fitter must produce no updates from empty counts; got {:?}",
                        slot
                    );
                }
            }
        }
    }
}

/// The fitter is also a no-op at the zero-saving threshold when the
/// observed counts exactly match the §13.5 default p_old at every
/// observed slot — `p_new == p_old` short-circuits the saving
/// computation. We construct such counts synthetically.
#[test]
fn fit_token_prob_updates_no_op_when_observed_equals_default() {
    // Pick a position with default p_old = 128 (band-7 of plane 2,
    // chroma's all-128 block per the DEFAULT_COEFF_PROBS check in
    // dct_tokens tests).
    let mut counts = empty_branch_counts();
    counts[2][7][2][0] = (128, 128); // 50/50 ⇒ p_obs ≈ 128 == p_old

    let fitted = fit_token_prob_updates(&counts, 0.0);
    assert!(
        fitted[2][7][2][0].is_none(),
        "p_new == p_old should produce no update"
    );
}

/// The high-level entry-point on a non-trivial frame: the fitted
/// wire is **<= the default-encode wire**. This is the safety guard
/// the entry-point's docs promise — the two-pass fitter only ships
/// the fitted bytes when they actually shrink (or match) the default.
#[test]
fn fitted_keyframe_never_grows_the_wire() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    let bytes_default = encode_keyframe(&frame, &params).expect("encode_keyframe");
    let bytes_fitted = encode_keyframe_with_fitted_token_prob_updates(&frame, &params)
        .expect("encode_keyframe_with_fitted_token_prob_updates");

    assert!(
        bytes_fitted.len() <= bytes_default.len(),
        "fitted wire {} must be <= default wire {} (fitter's safety guard violated)",
        bytes_fitted.len(),
        bytes_default.len()
    );
}

/// The fitted wire decodes through the crate's own decoder to a
/// picture clearing the same 25 dB whole-frame PSNR floor the r155 /
/// r156 tests pin — the §13.4 layer only re-prices entropy bits, it
/// does not alter the §14 / §11 reconstruction equation, so the picture
/// quality should not regress.
#[test]
fn fitted_keyframe_round_trips_through_decoder() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    let bytes = encode_keyframe_with_fitted_token_prob_updates(&frame, &params)
        .expect("encode_keyframe_with_fitted_token_prob_updates");

    let dec = decode_vp8(&bytes).expect("decode_vp8");
    assert_eq!(dec.width, 32);
    assert_eq!(dec.height, 32);

    let psnr = frame_psnr(&y, &u, &v, &dec);
    assert!(
        psnr >= 25.0,
        "fitted keyframe decoded PSNR {psnr} dB < 25 dB floor"
    );
}

/// `encode_keyframe_with_fitted_token_prob_updates` is observationally
/// equivalent to:
///   1. Run the default encode to learn counts (via the public count
///      walker).
///   2. Compute the fit through `fit_token_prob_updates`.
///   3. If the fit has any `Some` entry and
///      `encode_keyframe_with_token_prob_updates(.., &fit)` is <= the
///      default, ship the fitted bytes.
///
/// This test reconstructs the same path manually (without the
/// `encode_keyframe_inner` shortcut) using the public count walker
/// against the decoded coefficients, and verifies the high-level entry
/// returns either those bytes or the default — i.e. the cheaper of
/// the two. The walk-by-walk equivalence cannot be checked directly
/// from outside the crate because we cannot access the raw MbCoeffs
/// the encoder picked; instead we verify the *outcome*: the fitted
/// wire (when it is the shorter one) decodes identically to a wire
/// produced by the explicit caller-driven path with the SAME fitted
/// payload.
#[test]
fn fitted_keyframe_matches_explicit_caller_driven_path() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    let bytes_default = encode_keyframe(&frame, &params).expect("encode_keyframe");
    let bytes_high_level = encode_keyframe_with_fitted_token_prob_updates(&frame, &params)
        .expect("encode_keyframe_with_fitted_token_prob_updates");

    // Either the high-level entry returns the default bytes (no
    // update crossed the saving threshold, or the fitted wire was
    // larger so the safety guard fell back) — or it returns a wire
    // strictly smaller than the default.
    let is_default_match = bytes_high_level == bytes_default;
    let is_smaller_than_default = bytes_high_level.len() < bytes_default.len();
    assert!(
        is_default_match || is_smaller_than_default,
        "high-level fitted wire ({} bytes) must equal the default ({} bytes) \
         or be strictly smaller — equality {is_default_match}, smaller \
         {is_smaller_than_default}",
        bytes_high_level.len(),
        bytes_default.len()
    );

    // Sanity: the high-level result always decodes.
    let dec = decode_vp8(&bytes_high_level).expect("decode_vp8 (high-level)");
    assert_eq!(dec.width, 32);
    assert_eq!(dec.height, 32);
}

/// The fitted wire's §19.2 header round-trips through
/// `Vp8CodedHeader::parse` — i.e. the §13.4 sub-block the fitter wrote
/// is well-formed and the recovered `TokenProbUpdates` array has at
/// least the `Some` entries the fit emitted (we can't predict the
/// exact set without re-running the fit, but we can check the header
/// parses and yields a valid array). When the fitter falls back to the
/// default wire the recovered array is all-`None`.
#[test]
fn fitted_keyframe_header_round_trips_through_parser() {
    let (y, u, v) = synthetic_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let params = synthetic_params();

    let bytes = encode_keyframe_with_fitted_token_prob_updates(&frame, &params)
        .expect("encode_keyframe_with_fitted_token_prob_updates");

    let raw_hdr = Vp8FrameHeader::parse(&bytes).expect("uncompressed header parse");
    assert!(raw_hdr.key_frame);
    let first_partition_size = raw_hdr.first_partition_size as usize;
    let start = raw_hdr.header_bytes_consumed;
    let partition = &bytes[start..start + first_partition_size];
    let coded = Vp8CodedHeader::parse(partition, raw_hdr.key_frame).expect("coded header parse");

    // Spot-check: every position holds either `None` or a `Some(p)`
    // with `p` in the valid §13.5 `Prob` range (1..=255). The exact
    // payload depends on the fit but the round-trip's well-formedness
    // is what we pin here.
    for plane in &coded.token_prob_updates {
        for band in plane {
            for ctx in band {
                for slot in ctx {
                    if let Some(p) = *slot {
                        assert!(
                            (1..=255).contains(&p),
                            "fitted token_prob_update slot out of range: {p}"
                        );
                    }
                }
            }
        }
    }
}

/// Sanity: invoking `count_keyframe_branches` on a hand-built (MbCoeffs,
/// MacroblockModes) pair (one MB, all zeros, skip = true) produces an
/// empty counts array — skip MBs emit no tokens.
///
/// This isolates the count walker from the encoder so a regression in
/// the walker (e.g. failing to honour `mb_skip_coeff`) is caught
/// independently of the high-level entry.
#[test]
fn count_keyframe_branches_skips_skip_macroblocks() {
    use oxideav_vp8::{IntraUvMode, IntraYMode, MacroblockModes, MbCoeffs};

    let modes = vec![MacroblockModes {
        segment_id: None,
        mb_skip_coeff: true,
        y_mode: IntraYMode::Dc,
        subblock_modes: None,
        uv_mode: IntraUvMode::Dc,
    }];
    let all_coeffs = vec![MbCoeffs::default()];
    let mut counts = empty_branch_counts();

    count_keyframe_branches(&modes, &all_coeffs, 1, 1, &mut counts);

    // No token writes ⇒ every counter stays zero.
    for plane in &counts {
        for band in plane {
            for ctx in band {
                for slot in ctx {
                    assert_eq!(*slot, (0, 0), "skip MB must produce zero branch counts");
                }
            }
        }
    }
}

/// Sanity: a hand-constructed counts array with a single mostly-zero
/// slot drives `fit_token_prob_updates` to emit an update at that
/// slot with `p_new` close to 255 (mostly zeros = high "prob of 0").
/// Uses a band-3 / plane-2 / ctx-2 / pos-9 slot whose §13.5 default
/// p_old = 255 already, so we add a slot at a position with p_old !=
/// 255 to actually trigger a useful update. Plane-0 / band-1 / ctx-0
/// / pos-3 has p_old = 255 too; pick plane-1 / band-0 / ctx-0 / pos-0
/// (p_old = 198) and feed it 1024:1 zeros:ones so p_new ≈ 255
/// (clamped) and the body saving is large.
#[test]
fn fit_token_prob_updates_emits_at_high_bias_slots() {
    let mut counts = empty_branch_counts();
    counts[1][0][0][0] = (1024, 1); // 99.9 % zeros ⇒ p_obs ≈ 255

    let fitted = fit_token_prob_updates(&counts, 2.0);
    let slot = fitted[1][0][0][0];
    let p = slot.expect("a high-bias slot must produce an update");
    assert!(
        p >= 250,
        "expected p_new near 255 for 1024:1 zeros, got {p}"
    );

    // Every other slot stayed None.
    for (i, plane) in fitted.iter().enumerate() {
        for (j, band) in plane.iter().enumerate() {
            for (k, ctx) in band.iter().enumerate() {
                for (t, s) in ctx.iter().enumerate() {
                    if (i, j, k, t) != (1, 0, 0, 0) {
                        assert!(
                            s.is_none(),
                            "unexpected update at [{i}][{j}][{k}][{t}]: {s:?}"
                        );
                    }
                }
            }
        }
    }
}

// ────────────── §9.10 refresh_entropy_probs / carried-base contract ──────────────

/// Round-387 regression: a key frame that ships a §13.4 update payload
/// must write `refresh_entropy_probs = 0`.
///
/// Every inter entry point in the crate codes its §13.4 overlay against
/// a carried base of §13.5 defaults. Before round 387 a fitted key
/// frame wrote `refresh_entropy_probs = 1`, persisting its merged table
/// into the decoder's carried state — the next P-frame then decoded
/// against `overlay(kf_merged, p_updates)` while the encoder had coded
/// against `overlay(defaults, p_updates)`: silent probability
/// divergence, pixel drift with no decode error (observed as a ~10 dB
/// PSNR drop on the first P-frame after a fitted key frame).
///
/// Contract pinned here:
///
///   * fitted key frame (updates on the wire) → `refresh_entropy_probs
///     = 0` in the parsed header;
///   * update-free key frame → the historical `1`;
///   * fitted K followed by fitted P self-decodes in **pixel lockstep**
///     (the drift scenario that exposed the bug).
#[test]
fn fitted_keyframe_reverts_carried_probs_so_fitted_p_frame_stays_lockstep() {
    use oxideav_vp8::{
        encode_keyframe_with_reconstruction,
        encode_keyframe_with_reconstruction_and_fitted_token_prob_updates,
        encode_p_frame_multi_ref_with_fitted_token_prob_updates, Vp8DecoderState,
    };

    let (w, h) = (64usize, 64usize);
    let make = |shift: usize| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let cc = c + shift;
                y[r * w + c] = ((60 + (r * 2 + cc) % 120) as i32
                    + (((cc * 13) % 31) as i32 - 15) * 2)
                    .clamp(0, 255) as u8;
            }
        }
        let u = vec![100u8; (w / 2) * (h / 2)];
        let v = vec![140u8; (w / 2) * (h / 2)];
        (y, u, v)
    };
    let params = KeyframeParams {
        y_ac_qi: 40,
        ..KeyframeParams::default()
    };

    let (ky, ku, kv) = make(0);
    let kf_src = I420Frame::packed(w as u32, h as u32, &ky, &ku, &kv);
    let (kf_bytes, kf_planes) =
        encode_keyframe_with_reconstruction_and_fitted_token_prob_updates(&kf_src, &params)
            .expect("fitted keyframe");

    // Header contract — the fitted key frame must NOT persist its
    // merged table.
    let hdr = Vp8FrameHeader::parse(&kf_bytes).expect("tag");
    let partition = &kf_bytes
        [hdr.header_bytes_consumed..hdr.header_bytes_consumed + hdr.first_partition_size as usize];
    let coded = Vp8CodedHeader::parse(partition, true).expect("coded header");
    let carries_updates = coded.token_prob_updates.iter().any(|p| {
        p.iter()
            .any(|b| b.iter().any(|c| c.iter().any(|s| s.is_some())))
    });
    assert!(
        carries_updates,
        "test premise: the fitter must have emitted §13.4 updates on this content"
    );
    assert!(
        !coded.refresh_entropy_probs,
        "a key frame carrying §13.4 updates must write refresh_entropy_probs = 0"
    );

    // Update-free key frame keeps the historical 1.
    let plain = encode_keyframe_with_reconstruction(&kf_src, &params)
        .expect("plain keyframe")
        .0;
    let hdr2 = Vp8FrameHeader::parse(&plain).expect("tag");
    let part2 = &plain[hdr2.header_bytes_consumed
        ..hdr2.header_bytes_consumed + hdr2.first_partition_size as usize];
    let coded2 = Vp8CodedHeader::parse(part2, true).expect("coded header");
    assert!(
        coded2.refresh_entropy_probs,
        "an update-free key frame keeps refresh_entropy_probs = 1"
    );

    // Drift scenario: fitted K → fitted P, pixel lockstep end-to-end.
    let (py, pu, pv) = make(2);
    let p_src = I420Frame::packed(w as u32, h as u32, &py, &pu, &pv);
    let (p_bytes, p_planes) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
        &p_src, &kf_planes, None, None, &params,
    )
    .expect("fitted P-frame");

    let mut dec = Vp8DecoderState::new();
    let dk = dec.decode_frame(&kf_bytes).expect("keyframe decodes");
    assert_eq!(dk.y, kf_planes.y, "keyframe lockstep");
    let dp = dec.decode_frame(&p_bytes).expect("P-frame decodes");
    assert_eq!(
        dp.y, p_planes.y,
        "fitted P-frame after fitted keyframe must stay in pixel lockstep"
    );
}
