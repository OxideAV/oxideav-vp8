//! Measured PSNR delta of §9.4 RD loop-filter auto-selection (round 373).
//!
//! The auto selector minimises *per-frame* reconstruction SSD against the
//! source. Two regimes:
//!
//!  * **Keyframe / intra (no forward reference dependence)** — the choice
//!    is unambiguously good: a non-zero level only wins when it lowers the
//!    decoded-vs-source error, and level 0 is always a candidate. This
//!    test pins a strict PSNR gain on a blocky coarse-quantised keyframe.
//!
//!  * **Inter stream (filtered recon feeds the next prediction)** —
//!    greedy per-frame selection is *not* globally optimal: each filtered
//!    reconstruction becomes the prediction reference for the following
//!    P-frame, so optimising frame `i` in isolation can slightly degrade
//!    the reference quality for frame `i+1`. The second test **measures**
//!    this stream-level delta (it is documented as informational, not a
//!    strict inequality) so the trade-off is visible and tracked. A future
//!    round can make the inter selector reference-aware (e.g. bias toward
//!    lighter filtering on frames that refresh a long-lived reference).

use oxideav_vp8::{
    encode_keyframe_auto_loop_filter_with_reconstruction, encode_keyframe_with_reconstruction,
    encode_p_frame_multi_ref_auto_loop_filter,
    encode_p_frame_multi_ref_with_refresh_and_intra_pick, I420Frame, KeyframeParams,
    RefreshControls, Vp8DecodedFrame, Vp8DecoderState,
};

struct Src {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Src {
    fn frame(&self) -> I420Frame<'_> {
        I420Frame::packed(self.width, self.height, &self.y, &self.u, &self.v)
    }
}

/// Blocky 64×48 I420 source drifting per frame: per-MB plateaus give the
/// §15 filter genuine MB-edge error, and the per-frame drift gives the
/// P-frames real (small-MV) inter residual.
fn blocky_frame(frame_idx: usize) -> Src {
    let (w, h) = (64usize, 48usize);
    let (cw, ch) = (32usize, 24usize);
    let f = frame_idx;
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let mb = ((r / 16) * 7 + (c / 16) * 13 + f * 5) % 200;
            let ramp = ((r % 16) + (c % 16)) / 4;
            y[r * w + c] = (20 + mb + ramp).min(255) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            let mb = ((r / 8) * 11 + (c / 8) * 5 + f * 3) % 80;
            u[r * cw + c] = (110 + mb) as u8;
            v[r * cw + c] = (140 + mb) as u8;
        }
    }
    Src {
        width: w as u32,
        height: h as u32,
        y,
        u,
        v,
    }
}

fn plane_se(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum()
}

fn frame_psnr(src: &Src, dec: &Vp8DecodedFrame) -> f64 {
    let total = src.y.len() + src.u.len() + src.v.len();
    let se = plane_se(&src.y, &dec.y) + plane_se(&src.u, &dec.u) + plane_se(&src.v, &dec.v);
    let mse = se / total as f64;
    if mse <= f64::EPSILON {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Encode an `n`-frame I+P… stream, decode statefully, return the mean
/// decoded PSNR. `auto_lf` toggles RD level selection; otherwise
/// `loop_filter_level = 0` (unfiltered) is used. Both paths take the same
/// mode-decision route (intra-pick, default refresh) so the only
/// difference is the §15 level.
fn mean_stream_psnr(n: usize, qi: u8, auto_lf: bool) -> f64 {
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        ..KeyframeParams::default()
    };

    let mut last_recon = None;
    let mut dec = Vp8DecoderState::new();
    let mut psnr_sum = 0.0;

    for i in 0..n {
        let src = blocky_frame(i);
        let (bytes, recon) = if i == 0 {
            if auto_lf {
                encode_keyframe_auto_loop_filter_with_reconstruction(&src.frame(), &params)
            } else {
                encode_keyframe_with_reconstruction(&src.frame(), &params)
            }
            .expect("keyframe encode")
        } else {
            let reference = last_recon
                .as_ref()
                .expect("reference exists after keyframe");
            if auto_lf {
                encode_p_frame_multi_ref_auto_loop_filter(
                    &src.frame(),
                    reference,
                    None,
                    None,
                    &params,
                )
                .expect("auto P encode")
            } else {
                encode_p_frame_multi_ref_with_refresh_and_intra_pick(
                    &src.frame(),
                    reference,
                    None,
                    None,
                    &params,
                    &RefreshControls::default(),
                )
                .expect("fixed P encode")
            }
        };
        last_recon = Some(recon);

        let decoded = dec.decode_frame(&bytes).expect("stream frame decodes");
        psnr_sum += frame_psnr(&src, &decoded);
    }
    psnr_sum / n as f64
}

#[test]
fn auto_loop_filter_keyframe_psnr_strictly_better() {
    // A single keyframe has no forward reference dependence, so the RD
    // selector's per-frame choice is unambiguously good: it must strictly
    // beat the unfiltered baseline on this blocky coarse-quantised source.
    let qi = 100;
    let unfiltered = mean_stream_psnr(1, qi, false);
    let auto = mean_stream_psnr(1, qi, true);
    eprintln!(
        "keyframe PSNR (qi {qi}): unfiltered {unfiltered:.3} dB / auto-LF {auto:.3} dB / delta {:+.3} dB",
        auto - unfiltered
    );
    assert!(
        auto > unfiltered,
        "auto-LF keyframe PSNR {auto:.4} must strictly beat unfiltered {unfiltered:.4}"
    );
}

#[test]
fn auto_loop_filter_inter_stream_psnr_is_measured() {
    // Informational: greedy per-frame loop-filter selection on an inter
    // stream is not globally optimal because the filtered reconstruction
    // feeds the next prediction. We measure the stream-level delta rather
    // than assert a (false) strict inequality, and only require that the
    // auto path stays within a small tolerance of the unfiltered baseline
    // (it must not collapse the chain).
    let n = 6;
    let qi = 100;
    let unfiltered = mean_stream_psnr(n, qi, false);
    let auto = mean_stream_psnr(n, qi, true);
    let delta = auto - unfiltered;
    eprintln!(
        "mean inter-stream PSNR ({n} frames, qi {qi}): unfiltered {unfiltered:.3} dB / auto-LF {auto:.3} dB / delta {delta:+.3} dB"
    );
    // The greedy choice may cost a little at the stream level; bound it so
    // a regression that badly destabilises the reference chain still fails.
    assert!(
        delta > -1.0,
        "auto-LF inter-stream PSNR delta {delta:.4} dB must stay within 1 dB of baseline"
    );
    // Both streams must remain plausible pictures.
    assert!(unfiltered > 30.0 && auto > 30.0, "both streams clear 30 dB");
}
