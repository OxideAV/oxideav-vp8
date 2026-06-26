//! RD-driven §9.4 `loop_filter_level` auto-selection (round 373).
//!
//! RFC 6386 §15 defines the loop filter but is deliberately silent on the
//! encoder's choice of `loop_filter_level` — that is an encoder-side
//! rate-distortion decision. `encode_keyframe_auto_loop_filter` searches
//! the §9.4 level range `0..=63` for the level that minimises the
//! post-§15 visible-window SSD of the encoder's own reconstruction
//! against the source picture, writes that level into the §9.4 header, and
//! applies it to the reconstruction. These tests pin three properties:
//!
//!   * **Lockstep** — the encoder's post-§15 reconstruction byte-equals a
//!     compliant decoder's output of the same bytes (the auto-selected
//!     level reaches the wire, so both ends run the identical §15 pass).
//!   * **Never-harmful** — the auto path's distortion against the source
//!     is no worse than the `loop_filter_level = 0` (unfiltered) path on
//!     the same source / quantiser. Level 0 is always a search candidate,
//!     so the selector cannot *increase* distortion.
//!   * **Engages** — on a blocky coarse-quantised source the selector
//!     picks a non-zero level (filtering genuinely reduces block-edge
//!     error), demonstrating the search is live and not a no-op.

use oxideav_vp8::loop_filter::{reconstruction_ssd, SourcePlanes};
use oxideav_vp8::{
    decode_vp8, encode_keyframe_auto_loop_filter,
    encode_keyframe_auto_loop_filter_with_reconstruction, encode_keyframe_with_reconstruction,
    I420Frame, KeyframeParams,
};

/// Deterministic blocky I420 source: each 16×16 macroblock is a near-flat
/// patch at a distinct brightness, so the coarse-quantiser reconstruction
/// has pronounced MB-edge discontinuities for the §15 filter to smooth.
struct Blocky {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Blocky {
    fn new(width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                // Per-MB plateau (16×16) plus a faint intra-MB ramp so the
                // intra picker has a little structure to fit.
                let mb = ((r / 16) * 7 + (c / 16) * 13) % 200;
                let ramp = ((r % 16) + (c % 16)) / 4;
                y[r * w + c] = (20 + mb + ramp).min(255) as u8;
            }
        }
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                let mb = ((r / 8) * 11 + (c / 8) * 5) % 80;
                u[r * cw + c] = (110 + mb) as u8;
                v[r * cw + c] = (140 + mb) as u8;
            }
        }
        Blocky {
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

    fn source_planes(&self) -> SourcePlanes<'_> {
        SourcePlanes {
            width: self.width as usize,
            height: self.height as usize,
            y: &self.y,
            u: &self.u,
            v: &self.v,
            y_stride: self.width as usize,
            uv_stride: self.width.div_ceil(2) as usize,
        }
    }
}

/// The §9.4 `loop_filter_level` field a key-frame bitstream actually
/// carries (6 bits, immediately after `filter_type`).
///
/// The first (control) partition of a key frame begins at byte offset 10:
/// the 3-byte frame tag (§9.1) + the 7-byte key-frame extension (start
/// code + dimensions). The §19.2 leading bool-coded fields up to the
/// loop-filter level are: §9.2 `color_space` + `clamping_type`, §9.3
/// `update_mb_segmentation` (off here), §9.4 `filter_type`, then the
/// 6-bit `loop_filter_level`.
fn header_filter_level(bytes: &[u8]) -> u8 {
    use oxideav_vp8::bool_decoder::BoolDecoder;
    let part = &bytes[10..];
    let mut bd = BoolDecoder::init(part).expect("init first partition");
    let _ = bd.read_bool(128).unwrap(); // color_space
    let _ = bd.read_bool(128).unwrap(); // clamping_type
    let _ = bd.read_bool(128).unwrap(); // update_mb_segmentation
    let _filter_type = bd.read_bool(128).unwrap();
    bd.read_literal(6).unwrap() as u8
}

fn psnr(ssd: u64, pixels: u64) -> f64 {
    if ssd == 0 {
        return 99.0;
    }
    let mse = ssd as f64 / pixels as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn auto_lf_pixel_lockstep() {
    // The auto-selected level must reach the wire so the decoder
    // reproduces the encoder's post-§15 reconstruction byte-for-byte.
    let src = Blocky::new(48, 32);
    let params = KeyframeParams {
        y_ac_qi: 96, // coarse → blocky → filtering helps
        ..KeyframeParams::default()
    };
    let (bytes, recon) =
        encode_keyframe_auto_loop_filter_with_reconstruction(&src.frame(), &params)
            .expect("auto-LF encode");
    let dec = decode_vp8(&bytes).expect("decode auto-LF output");

    let w = src.width as usize;
    let h = src.height as usize;
    for row in 0..h {
        assert_eq!(
            &dec.y[row * w..row * w + w],
            &recon.y[row * recon.y_stride..row * recon.y_stride + w],
            "luma drift at row {row}"
        );
    }
    let cw = src.width.div_ceil(2) as usize;
    let ch = src.height.div_ceil(2) as usize;
    for row in 0..ch {
        assert_eq!(
            &dec.u[row * cw..row * cw + cw],
            &recon.u[row * recon.uv_stride..row * recon.uv_stride + cw],
            "U drift at row {row}"
        );
        assert_eq!(
            &dec.v[row * cw..row * cw + cw],
            &recon.v[row * recon.uv_stride..row * recon.uv_stride + cw],
            "V drift at row {row}"
        );
    }
}

#[test]
fn auto_lf_never_worse_than_unfiltered() {
    let src = Blocky::new(64, 48);
    let qi = 100;

    // Baseline: explicit loop_filter_level = 0 (no §15 filtering).
    let unfiltered_params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        ..KeyframeParams::default()
    };
    let (_b0, recon0) =
        encode_keyframe_with_reconstruction(&src.frame(), &unfiltered_params).expect("encode lf=0");

    // Auto path: the selector chooses the distortion-minimising level.
    let auto_params = KeyframeParams {
        y_ac_qi: qi,
        ..KeyframeParams::default()
    };
    let (_b1, recon_auto) =
        encode_keyframe_auto_loop_filter_with_reconstruction(&src.frame(), &auto_params)
            .expect("auto encode");

    let sp = src.source_planes();
    let ssd0 = reconstruction_ssd(&recon0, &sp);
    let ssd_auto = reconstruction_ssd(&recon_auto, &sp);

    assert!(
        ssd_auto <= ssd0,
        "auto-LF distortion {ssd_auto} must not exceed unfiltered {ssd0}"
    );

    let pixels = (src.width as u64 * src.height as u64) * 3 / 2;
    let p0 = psnr(ssd0, pixels);
    let pa = psnr(ssd_auto, pixels);
    eprintln!(
        "unfiltered PSNR {p0:.3} dB / auto-LF PSNR {pa:.3} dB / ssd0={ssd0} ssd_auto={ssd_auto}"
    );
    assert!(pa >= p0 - 1e-9, "auto-LF PSNR must be >= unfiltered PSNR");
}

#[test]
fn auto_lf_engages_on_blocky_source() {
    // A coarse-quantised blocky source has real MB-edge error the §15
    // filter reduces, so the selector should land on a non-zero level and
    // write it to the header.
    let src = Blocky::new(80, 64);
    let params = KeyframeParams {
        y_ac_qi: 110,
        ..KeyframeParams::default()
    };
    let bytes = encode_keyframe_auto_loop_filter(&src.frame(), &params).expect("auto encode");
    let level = header_filter_level(&bytes);
    assert!(
        level > 0,
        "expected a non-zero auto-selected loop_filter_level, got {level}"
    );
    assert!(level <= 63, "level {level} out of §9.4 range");
    // The chosen level must decode cleanly.
    let dec = decode_vp8(&bytes).expect("decode");
    assert_eq!(dec.width, src.width);
    assert_eq!(dec.height, src.height);
}
