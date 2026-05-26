//! Encoder-side §13.4 `token_prob_update()` **observed-counts fitter**
//! for the **inter (P-frame) path** — round 158's mirror of the round-
//! 157 keyframe fitter.
//!
//! Round 155 wired the keyframe caller-driven layer; round 156 mirrored
//! it on the inter path; round 157 added the keyframe fitter
//! (`encode_keyframe_with_fitted_token_prob_updates`). Round 158 closes
//! the symmetry by adding the analogous inter entry-point
//! (`encode_p_frame_multi_ref_with_fitted_token_prob_updates` and the
//! full-surface
//! `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`).
//!
//! The fitter shares its cost-model and counter type with the keyframe
//! path; only the per-frame collection walker
//! ([`oxideav_vp8::count_inter_frame_branches`]) is inter-specific —
//! the inter picker stamps `IntraYMode::Dc` onto every MB so the "no
//! Y2" decision cannot be recovered from `y_mode`; the walker consumes
//! an explicit `use_bpred_per_mb` slice instead.
//!
//! Tests below pin each property the round-158 contract claims:
//!
//!   1. The high-level inter entry on a non-trivial frame returns a
//!      wire **<=** the default-encode inter wire (the safety guard).
//!   2. The fitted inter wire decodes through `Vp8DecoderState` after
//!      its I-frame and clears the same 25 dB PSNR floor the r155 /
//!      r156 / r157 tests pin.
//!   3. The fitted inter §19.2 header round-trips through
//!      `Vp8CodedHeader::parse` and every recovered `Some(p)` lies in
//!      the valid §13.5 `Prob` range `[1, 255]`.
//!   4. `count_inter_frame_branches` honours `mb_skip_coeff` — a single
//!      skip-MB frame produces zero counts.
//!   5. The full-surface entry-point with `RefreshControls::default` /
//!      `LoopFilterDeltas::default` / `[0; 4]` carried state matches
//!      the thin-wrapper entry-point byte-for-byte (the wrapper proof).

use oxideav_vp8::{
    count_inter_frame_branches, empty_branch_counts, encode_keyframe_with_reconstruction,
    encode_p_frame_multi_ref, encode_p_frame_multi_ref_with_fitted_token_prob_updates,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates, I420Frame,
    IntraUvMode, IntraYMode, KeyframeParams, LoopFilterDeltas, MacroblockModes, MbCoeffs,
    RefreshControls, Vp8CodedHeader, Vp8DecoderState, Vp8FrameHeader,
};

/// Build a 32×32 I420 picture with a smooth gradient — repeatable,
/// deterministic, no PRNG. The luma / chroma offsets exercise the
/// inter-frame residual path.
fn structured_frame_32x32(luma_offset: i16, chroma_offset: i16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let v = 40 + (r as i16) + (c as i16) + luma_offset;
            y[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for cc in 0..cw {
            let uv = 100 + (r as i16) + chroma_offset;
            u[r * cw + cc] = uv.clamp(0, 255) as u8;
            let vv = 150 - (cc as i16) + chroma_offset;
            v[r * cw + cc] = vv.clamp(0, 255) as u8;
        }
    }
    (y, u, v)
}

/// A larger 64×64 picture with crossing ramps + a quadratic radial — the
/// inter residual is significantly noisier than the smooth 32×32
/// gradient, giving the fitter more material to exploit and surfacing
/// a measurable wire shrinkage above the round-156 inter baseline.
fn synthetic_frame_64x64(luma_offset: i16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (64usize, 64usize);
    let (cw, ch) = (32usize, 32usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..h {
        for c in 0..w {
            let g = ((r * 11 + c * 7) % 256) as u8;
            let g = (g as i16 + luma_offset).clamp(0, 255) as u8;
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

fn params() -> KeyframeParams {
    KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    }
}

fn plane_psnr(src: &[u8], rec: &[u8]) -> f64 {
    assert_eq!(src.len(), rec.len());
    let mut sse: u64 = 0;
    for (a, b) in src.iter().zip(rec.iter()) {
        let d = *a as i32 - *b as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / src.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn frame_psnr(src_y: &[u8], src_u: &[u8], src_v: &[u8], dec: &oxideav_vp8::Vp8DecodedFrame) -> f64 {
    let py = plane_psnr(src_y, &dec.y);
    let pu = plane_psnr(src_u, &dec.u);
    let pv = plane_psnr(src_v, &dec.v);
    let clamp = |p: f64| if p.is_finite() { p } else { 99.0 };
    (clamp(py) + clamp(pu) + clamp(pv)) / 3.0
}

/// The high-level inter entry-point on a non-trivial frame: the fitted
/// wire is **<= the default-encode inter wire**. This is the safety
/// guard the entry-point's docs promise — the two-pass fitter only
/// ships the fitted bytes when they actually shrink (or match) the
/// default.
#[test]
fn fitted_pframe_never_grows_the_wire() {
    let (yi, ui, vi) = synthetic_frame_64x64(0);
    let (yp, up, vp) = synthetic_frame_64x64(4);
    let frame_i = I420Frame::packed(64, 64, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(64, 64, &yp, &up, &vp);
    let kp = params();

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");

    let (bytes_default, _) =
        encode_p_frame_multi_ref(&frame_p, &i_recon, None, None, &kp).expect("encode P (default)");
    let (bytes_fitted, _) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
        &frame_p, &i_recon, None, None, &kp,
    )
    .expect("encode P (fitted)");

    assert!(
        bytes_fitted.len() <= bytes_default.len(),
        "fitted inter wire {} must be <= default inter wire {} (fitter's safety guard violated)",
        bytes_fitted.len(),
        bytes_default.len()
    );
}

/// The fitted inter wire decodes through the in-tree
/// `Vp8DecoderState` after its I-frame predecessor, reproducing the
/// source within the §14 quantiser's distortion (≥ 25 dB whole-frame
/// PSNR at mid quantiser) — the §13.4 layer only re-prices entropy
/// bits, it does not alter the §14 / §17 reconstruction equation, so
/// the picture quality should not regress.
#[test]
fn fitted_pframe_round_trips_through_decoder() {
    let (yi, ui, vi) = synthetic_frame_64x64(0);
    let (yp, up, vp) = synthetic_frame_64x64(4);
    let frame_i = I420Frame::packed(64, 64, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(64, 64, &yp, &up, &vp);
    let kp = params();

    let (i_bytes, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");
    let (p_bytes_fitted, _) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
        &frame_p, &i_recon, None, None, &kp,
    )
    .expect("encode P (fitted)");

    let mut dec = Vp8DecoderState::new();
    let _ = dec.decode_frame(&i_bytes).expect("I-frame decode");
    let pframe = dec.decode_frame(&p_bytes_fitted).expect("P-frame decode");
    assert_eq!(pframe.width, 64);
    assert_eq!(pframe.height, 64);

    let psnr = frame_psnr(&yp, &up, &vp, &pframe);
    assert!(
        psnr >= 25.0,
        "fitted P-frame decoded PSNR {psnr} dB < 25 dB floor"
    );
}

/// The fitted inter §19.2 header round-trips through
/// `Vp8CodedHeader::parse` — i.e. the §13.4 sub-block the fitter wrote
/// is well-formed and the recovered `TokenProbUpdates` array yields
/// valid `Prob` values. When the fitter falls back to the default-
/// wire the recovered array is all-`None`.
#[test]
fn fitted_pframe_header_round_trips_through_parser() {
    let (yi, ui, vi) = synthetic_frame_64x64(0);
    let (yp, up, vp) = synthetic_frame_64x64(4);
    let frame_i = I420Frame::packed(64, 64, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(64, 64, &yp, &up, &vp);
    let kp = params();

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");
    let (bytes, _) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
        &frame_p, &i_recon, None, None, &kp,
    )
    .expect("encode P (fitted)");

    let raw_hdr = Vp8FrameHeader::parse(&bytes).expect("uncompressed header parse");
    assert!(
        !raw_hdr.key_frame,
        "this fixture must be an inter frame for the test to be meaningful"
    );
    let first_partition_size = raw_hdr.first_partition_size as usize;
    let start = raw_hdr.header_bytes_consumed;
    let partition = &bytes[start..start + first_partition_size];
    let coded = Vp8CodedHeader::parse(partition, raw_hdr.key_frame).expect("coded header parse");

    for plane in &coded.token_prob_updates {
        for band in plane {
            for ctx in band {
                for slot in ctx {
                    if let Some(p) = *slot {
                        assert!(
                            (1..=255).contains(&p),
                            "fitted inter token_prob_update slot out of range: {p}"
                        );
                    }
                }
            }
        }
    }
}

/// Sanity: invoking `count_inter_frame_branches` on a hand-built
/// `(MacroblockModes, MbCoeffs, use_bpred_per_mb)` triple (one MB, all
/// zeros, skip = true) produces an empty counts array — skip MBs emit
/// no tokens.
///
/// This isolates the inter count walker from the encoder so a
/// regression in the walker (e.g. failing to honour `mb_skip_coeff`)
/// is caught independently of the high-level entry.
#[test]
fn count_inter_frame_branches_skips_skip_macroblocks() {
    let modes = vec![MacroblockModes {
        segment_id: None,
        mb_skip_coeff: true,
        // Inter picker stamps Dc onto every MB regardless of whether
        // the MB has Y2 on the wire — that's why the walker takes an
        // explicit `use_bpred_per_mb` slice rather than reading
        // `y_mode`.
        y_mode: IntraYMode::Dc,
        subblock_modes: None,
        uv_mode: IntraUvMode::Dc,
    }];
    let use_bpred_per_mb = vec![false];
    let all_coeffs = vec![MbCoeffs::default()];
    let mut counts = empty_branch_counts();

    count_inter_frame_branches(&modes, &use_bpred_per_mb, &all_coeffs, 1, 1, &mut counts);

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

/// The thin-wrapper entry-point
/// (`encode_p_frame_multi_ref_with_fitted_token_prob_updates`) and the
/// full-surface entry-point
/// (`..._with_refresh_and_lf_deltas_and_fitted_token_prob_updates`)
/// produce byte-for-byte the same wire when the full-surface caller
/// passes `RefreshControls::default` / `LoopFilterDeltas::default` /
/// `[0; 4]` carried state. The wrapper proof.
#[test]
fn fitted_inter_thin_wrapper_matches_full_surface_with_defaults() {
    let (yi, ui, vi) = structured_frame_32x32(0, 0);
    let (yp, up, vp) = structured_frame_32x32(4, 2);
    let frame_i = I420Frame::packed(32, 32, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(32, 32, &yp, &up, &vp);
    let kp = params();

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");

    let (bytes_thin, planes_thin) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
        &frame_p, &i_recon, None, None, &kp,
    )
    .expect("thin-wrapper encode");

    let (bytes_full, planes_full) =
        encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates(
            &frame_p,
            &i_recon,
            None,
            None,
            &kp,
            &RefreshControls::default(),
            &LoopFilterDeltas::default(),
            [0; 4],
            [0; 4],
        )
        .expect("full-surface encode");

    assert_eq!(
        bytes_thin, bytes_full,
        "thin wrapper must match full-surface defaults byte-for-byte"
    );
    assert_eq!(planes_thin.y.len(), planes_full.y.len());
    assert_eq!(planes_thin.u.len(), planes_full.u.len());
    assert_eq!(planes_thin.v.len(), planes_full.v.len());
    assert_eq!(planes_thin.y, planes_full.y);
    assert_eq!(planes_thin.u, planes_full.u);
    assert_eq!(planes_thin.v, planes_full.v);
}

/// The fitted inter wire is **strictly** smaller than the default-
/// encode inter wire on a noisy 64×64 frame at `y_ac_qi = 32` — the
/// observed-counts fitter must find at least one slot whose update
/// nets a positive bit saving. (Asserting strict shrinkage on a frame
/// with enough coefficient mass to amortise the §13.4 transmission
/// cost; the safety-guard test above already covers the degenerate
/// "no slot crosses threshold" case.)
#[test]
fn fitted_pframe_shrinks_on_noisy_residual() {
    let (yi, ui, vi) = synthetic_frame_64x64(0);
    let (yp, up, vp) = synthetic_frame_64x64(4);
    let frame_i = I420Frame::packed(64, 64, &yi, &ui, &vi);
    let frame_p = I420Frame::packed(64, 64, &yp, &up, &vp);
    let kp = params();

    let (_, i_recon) = encode_keyframe_with_reconstruction(&frame_i, &kp).expect("encode I");

    let (bytes_default, _) =
        encode_p_frame_multi_ref(&frame_p, &i_recon, None, None, &kp).expect("encode P (default)");
    let (bytes_fitted, _) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
        &frame_p, &i_recon, None, None, &kp,
    )
    .expect("encode P (fitted)");

    assert!(
        bytes_fitted.len() < bytes_default.len(),
        "expected strict shrinkage on noisy 64×64 residual: fitted {} >= default {}",
        bytes_fitted.len(),
        bytes_default.len()
    );
}
