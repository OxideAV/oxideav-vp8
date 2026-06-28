//! §9.3 / §10 segment-based adaptive-quant key-frame encoder
//! (`encode_keyframe_adaptive_quant`) end-to-end tests.
//!
//! The encoder sorts each macroblock into one of four §10 segments by
//! luma variance and quantises it at that segment's effective qindex
//! (`clamp(base_y_ac_qi + quant_delta[seg], 0, 127)`). It emits the §9.3
//! `update_segmentation()` block (delta-mode per-segment `quantizer_update`
//! values + the `mb_segment_tree_probs`) and the per-MB §10 `segment_id`
//! in the §11 mode layer.
//!
//! The contract these tests pin:
//!
//! 1. **Pixel-exact lockstep.** The encoder predicts each MB from its own
//!    per-segment reconstruction, so a compliant `decode_vp8` reproduces
//!    the encoder's post-§15 reconstruction planes byte-for-byte (cropped
//!    from the MB-padded raster to the §9.1 visible window). This proves
//!    the decoder rebuilds the *same four* per-segment dequant factors and
//!    routes each MB to the right one via its decoded `segment_id`.
//! 2. **Self-decode validity** at non-MB-aligned dimensions and with the
//!    §15 loop filter on.
//! 3. **Quality** clears a mid-quantiser PSNR bar against the source.
//!
//! Fully self-contained: only the crate's own encode + decode entry
//! points are consulted; no external codec.

use oxideav_vp8::{
    decode_vp8, encode_keyframe_adaptive_quant, encode_keyframe_adaptive_quant_with_reconstruction,
    encode_keyframe_adaptive_quant_with_trellis, AdaptiveQuantConfig, I420Frame, TrellisStrength,
};

struct Source {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Source {
    /// A picture with a clear activity split: a flat-128 quadrant (→ low
    /// variance, segment 0), a smooth gradient (mid), and a high-contrast
    /// checkerboard quadrant (→ high variance, top segment). This makes
    /// the §10 classifier actually exercise multiple segments.
    fn new(width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut y = vec![0u8; w * h];
        for (r, row) in y.chunks_exact_mut(w).enumerate() {
            for (c, px) in row.iter_mut().enumerate() {
                let left = c < w / 2;
                let top = r < h / 2;
                *px = match (top, left) {
                    // Flat region.
                    (true, true) => 128,
                    // Smooth diagonal gradient.
                    (true, false) => ((c * 256 / w + r * 256 / h) / 2) as u8,
                    // Vertical ramp.
                    (false, true) => ((r * 255 / h) % 256) as u8,
                    // High-contrast checkerboard (high local variance).
                    (false, false) => {
                        if (c / 2 + r / 2) % 2 == 0 {
                            16
                        } else {
                            240
                        }
                    }
                };
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for (r, (urow, vrow)) in u
            .chunks_exact_mut(cw)
            .zip(v.chunks_exact_mut(cw))
            .enumerate()
        {
            for c in 0..cw {
                urow[c] = (120 + (c * 16 / cw)) as u8;
                vrow[c] = (136 - (r * 12 / ch)) as u8;
            }
        }
        Source {
            width,
            height,
            y,
            u,
            v,
        }
    }

    fn frame(&self) -> I420Frame<'_> {
        I420Frame::packed(self.width, self.height, &self.y, &self.u, &self.v)
    }
}

fn plane_mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    sum / a.len() as f64
}

fn frame_psnr(src: &Source, dec: &oxideav_vp8::Vp8DecodedFrame) -> f64 {
    let total = src.y.len() + src.u.len() + src.v.len();
    let se = plane_mse(&src.y, &dec.y) * src.y.len() as f64
        + plane_mse(&src.u, &dec.u) * src.u.len() as f64
        + plane_mse(&src.v, &dec.v) * src.v.len() as f64;
    let mse = se / total as f64;
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Encode with the adaptive-quant path, decode the emitted bytes, and
/// assert the decoder's visible planes byte-equal the encoder's own
/// post-§15 reconstruction — the differential oracle that proves the
/// per-segment dequant factors round-trip.
fn assert_pixel_lockstep(width: u32, height: u32, config: &AdaptiveQuantConfig) {
    let src = Source::new(width, height);
    let (bytes, recon) = encode_keyframe_adaptive_quant_with_reconstruction(&src.frame(), config)
        .expect("encode must accept in-range config");
    let dec = decode_vp8(&bytes).expect("decoder must accept adaptive-quant output");

    assert_eq!(dec.width, width, "visible width");
    assert_eq!(dec.height, height, "visible height");
    let w = width as usize;
    let h = height as usize;
    assert_eq!(recon.mb_cols, w.div_ceil(16), "encoder mb_cols");
    assert_eq!(recon.mb_rows, h.div_ceil(16), "encoder mb_rows");

    for row in 0..h {
        assert_eq!(
            &dec.y[row * w..row * w + w],
            &recon.y[row * recon.y_stride..row * recon.y_stride + w],
            "luma drift at row {row} ({width}x{height}, {config:?})"
        );
    }
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    for row in 0..ch {
        assert_eq!(
            &dec.u[row * cw..row * cw + cw],
            &recon.u[row * recon.uv_stride..row * recon.uv_stride + cw],
            "U drift at row {row} ({width}x{height}, {config:?})"
        );
        assert_eq!(
            &dec.v[row * cw..row * cw + cw],
            &recon.v[row * recon.uv_stride..row * recon.uv_stride + cw],
            "V drift at row {row} ({width}x{height}, {config:?})"
        );
    }
}

#[test]
fn adaptive_quant_pixel_lockstep_default_config() {
    // 64×64 (4×4 MBs) with the default AQ gradient and no loop filter —
    // the cleanest lockstep: every MB's per-segment quantiser must
    // round-trip through the decoded segment_id + §9.3 quantizer deltas.
    assert_pixel_lockstep(64, 64, &AdaptiveQuantConfig::default());
}

#[test]
fn adaptive_quant_pixel_lockstep_partial_mbs() {
    // 47×33: partial right column + partial bottom row of macroblocks,
    // exercising the encoder edge-replication padding alongside the
    // per-MB segment classification.
    assert_pixel_lockstep(47, 33, &AdaptiveQuantConfig::default());
}

#[test]
fn adaptive_quant_pixel_lockstep_with_loop_filter() {
    // §15.3 normal filter over the per-segment reconstruction.
    let config = AdaptiveQuantConfig {
        loop_filter_level: 24,
        sharpness_level: 1,
        ..AdaptiveQuantConfig::default()
    };
    assert_pixel_lockstep(64, 48, &config);
}

#[test]
fn adaptive_quant_pixel_lockstep_simple_filter() {
    // §15.2 simple filter arm.
    let config = AdaptiveQuantConfig {
        loop_filter_level: 40,
        sharpness_level: 3,
        filter_type: true,
        ..AdaptiveQuantConfig::default()
    };
    assert_pixel_lockstep(48, 48, &config);
}

#[test]
fn adaptive_quant_pixel_lockstep_absolute_style_deltas() {
    // A steeper quant gradient and a different base, still in delta mode.
    let config = AdaptiveQuantConfig {
        base_y_ac_qi: 48,
        quant_delta: [20, 4, -10, -24],
        ..AdaptiveQuantConfig::default()
    };
    assert_pixel_lockstep(80, 64, &config);
}

#[test]
fn adaptive_quant_pixel_lockstep_uniform_deltas_equivalent_to_flat() {
    // All-zero deltas: every segment collapses to the base qindex. The
    // segmentation block + per-MB ids are still emitted (and decoded), so
    // this proves the segment plumbing is a no-op on the pixels when the
    // features are uniform.
    let config = AdaptiveQuantConfig {
        base_y_ac_qi: 32,
        quant_delta: [0, 0, 0, 0],
        ..AdaptiveQuantConfig::default()
    };
    assert_pixel_lockstep(64, 64, &config);
}

#[test]
fn adaptive_quant_meets_quality_bar() {
    let src = Source::new(64, 64);
    let bytes = encode_keyframe_adaptive_quant(&src.frame(), &AdaptiveQuantConfig::default())
        .expect("encode");
    let dec = decode_vp8(&bytes).expect("decode");
    let psnr = frame_psnr(&src, &dec);
    assert!(
        psnr >= 28.0,
        "adaptive-quant PSNR {psnr:.2} dB below 28 dB target"
    );
    eprintln!("adaptive-quant 64x64 default PSNR = {psnr:.2} dB");
}

#[test]
fn adaptive_quant_gradient_differs_from_uniform() {
    // On a frame with a real activity split, a non-trivial per-segment
    // quant gradient must change the emitted bytes vs the all-zero-delta
    // (uniform) encode — proving the §10 classifier routes MBs to
    // different segments and those segments quantise differently. Both
    // still self-decode (the lockstep tests above prove correctness); here
    // we only assert the segmentation is *doing something*.
    let src = Source::new(64, 64);
    let uniform = AdaptiveQuantConfig {
        quant_delta: [0, 0, 0, 0],
        ..AdaptiveQuantConfig::default()
    };
    let gradient = AdaptiveQuantConfig::default(); // deltas [8, 0, -6, -12]
    let a = encode_keyframe_adaptive_quant(&src.frame(), &uniform).expect("uniform encode");
    let b = encode_keyframe_adaptive_quant(&src.frame(), &gradient).expect("gradient encode");
    assert_ne!(
        a, b,
        "adaptive-quant gradient produced identical bytes to the uniform encode"
    );
}

#[test]
fn adaptive_quant_trellis_strength_trims_bytes_and_round_trips() {
    // The §13 trellis-strength knob reaches the segment-based adaptive-quant
    // path too: at a coarser base quantiser a strong setting shrinks the
    // frame versus OFF / DEFAULT, the DEFAULT entry matches the historical
    // bytes, and every strength still decodes.
    let src = Source::new(64, 64);
    let config = AdaptiveQuantConfig {
        base_y_ac_qi: 56,
        ..AdaptiveQuantConfig::default()
    };

    let off =
        encode_keyframe_adaptive_quant_with_trellis(&src.frame(), &config, TrellisStrength::OFF)
            .expect("off");
    let def_via_knob = encode_keyframe_adaptive_quant_with_trellis(
        &src.frame(),
        &config,
        TrellisStrength::DEFAULT,
    )
    .expect("default via knob");
    let def_legacy = encode_keyframe_adaptive_quant(&src.frame(), &config).expect("legacy default");
    let hot = encode_keyframe_adaptive_quant_with_trellis(
        &src.frame(),
        &config,
        TrellisStrength::new(4.0),
    )
    .expect("hot");

    // The DEFAULT-strength entry is byte-identical to the historical path.
    assert_eq!(
        def_via_knob, def_legacy,
        "DEFAULT trellis strength must reproduce the historical adaptive-quant bytes"
    );
    // Monotone: OFF (plain rounding) >= DEFAULT >= hot, and the knob moves
    // the wire at this coarse quantiser.
    assert!(
        def_legacy.len() <= off.len() && hot.len() <= def_legacy.len(),
        "monotone OFF {} >= DEFAULT {} >= hot {} expected",
        off.len(),
        def_legacy.len(),
        hot.len()
    );
    assert!(
        hot.len() < off.len(),
        "strength 4.0 ({} B) should beat OFF ({} B)",
        hot.len(),
        off.len()
    );

    // Every strength decodes cleanly.
    for bytes in [&off, &def_via_knob, &hot] {
        let dec = decode_vp8(bytes).expect("decode at this strength");
        assert_eq!(dec.width, 64);
        assert_eq!(dec.height, 64);
    }
}

#[test]
fn adaptive_quant_rejects_out_of_range_base_qindex() {
    let src = Source::new(32, 32);
    let config = AdaptiveQuantConfig {
        base_y_ac_qi: 200,
        ..AdaptiveQuantConfig::default()
    };
    assert!(encode_keyframe_adaptive_quant(&src.frame(), &config).is_err());
}

#[test]
fn adaptive_quant_rejects_out_of_range_loop_filter_level() {
    let src = Source::new(32, 32);
    let config = AdaptiveQuantConfig {
        loop_filter_level: 64,
        ..AdaptiveQuantConfig::default()
    };
    assert!(encode_keyframe_adaptive_quant(&src.frame(), &config).is_err());
}
