//! Pixel-exact encode→decode lockstep anchors (RFC 6386 §9 / §12 /
//! §13.4 / §14 / §15 / §19.2) — the deterministic in-CI companions of
//! the round-280 `encode_decode_pixel_lockstep` fuzz target.
//!
//! The §15.1 design contract is that the encoder predicts each
//! macroblock from its own reconstruction so a compliant decoder
//! reproduces the encoder's post-§15 reconstruction planes *exactly*.
//! `encode_keyframe_with_reconstruction_and_token_updates` returns
//! those planes, which makes them a bit-exact differential oracle for
//! `decode_vp8`: every visible Y / U / V byte of the decoded output
//! must equal the encoder's reconstruction (cropped from the
//! MB-padded raster to the §9.1 visible window). The older roundtrip
//! suite (`tests/encoder_keyframe_roundtrip.rs`) gates on PSNR
//! against the *source* — a much looser oracle that cannot see a
//! drift where both halves remain plausible pictures.
//!
//! Each test here locks one corner the fuzz target sweeps:
//!
//! * non-MB-aligned dimensions (partial right / bottom macroblocks —
//!   encoder edge-replication padding, §15 filtering over the padded
//!   raster, decoder visible-crop);
//! * the §9.5 multi-partition layouts, including a partition count
//!   exceeding the MB-row count (empty trailing partitions) and the
//!   all-8-partitions-populated tall frame;
//! * both §9.4 `filter_type` arms at the loop-filter level / sharpness
//!   extremes, plus the `loop_filter_level = 0` §15 skip path;
//! * the §13.4 `token_prob_update()` write path at the L(8)
//!   probability extremes (`0` and `255`) — outside the `[1, 255]`
//!   band the in-tree fitter emits.

use oxideav_vp8::{
    decode_vp8, encode_keyframe_with_reconstruction_and_token_updates, I420Frame, KeyframeParams,
    TokenProbUpdates,
};

/// Deterministic synthetic I420 source (mixed gradients + a diagonal
/// edge so the §11 mode pick exercises several intra modes and the
/// §15 loop filter has real block-boundary activity to smooth).
struct Source {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Source {
    fn new(width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut y = vec![0u8; w * h];
        for (r, row) in y.chunks_exact_mut(w).enumerate() {
            for (c, px) in row.iter_mut().enumerate() {
                let diag = if c + r * 2 > w { 90 } else { 0 };
                *px = ((c * 5 + r * 3 + diag) % 256) as u8;
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
                urow[c] = ((128 + c * 2 + r) % 256) as u8;
                vrow[c] = ((128usize.wrapping_sub(c) + r * 3) % 256) as u8;
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

/// Encode with `params` (+ optional §13.4 updates), decode the emitted
/// bytes, and assert the decoder's visible planes byte-equal the
/// encoder's own post-§15 reconstruction.
fn assert_pixel_lockstep(
    width: u32,
    height: u32,
    params: &KeyframeParams,
    updates: Option<&TokenProbUpdates>,
) {
    let src = Source::new(width, height);
    let (bytes, recon) =
        encode_keyframe_with_reconstruction_and_token_updates(&src.frame(), params, updates)
            .expect("encode must accept in-range parameters");
    let dec = decode_vp8(&bytes).expect("decoder must accept encoder output");

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
            "luma drift at row {row} ({width}x{height}, {params:?})"
        );
    }
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    for row in 0..ch {
        assert_eq!(
            &dec.u[row * cw..row * cw + cw],
            &recon.u[row * recon.uv_stride..row * recon.uv_stride + cw],
            "U drift at row {row} ({width}x{height}, {params:?})"
        );
        assert_eq!(
            &dec.v[row * cw..row * cw + cw],
            &recon.v[row * recon.uv_stride..row * recon.uv_stride + cw],
            "V drift at row {row} ({width}x{height}, {params:?})"
        );
    }
}

#[test]
fn lockstep_partial_mb_normal_filter() {
    // 33×17: one partial column + one partial row of macroblocks; the
    // §15.3 normal filter runs over the padded raster before the crop.
    let params = KeyframeParams {
        y_ac_qi: 24,
        loop_filter_level: 20,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    assert_pixel_lockstep(33, 17, &params, None);
}

#[test]
fn lockstep_simple_filter_extremes() {
    // §15.2 simple filter at the §9.4 level / sharpness ceiling.
    let params = KeyframeParams {
        y_ac_qi: 12,
        loop_filter_level: 63,
        sharpness_level: 7,
        nbr_of_dct_partitions: 2,
        filter_type: true,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    assert_pixel_lockstep(48, 48, &params, None);
}

#[test]
fn lockstep_one_pixel_strips() {
    // Degenerate 1-px-wide / 1-px-tall visible windows: 15 of every
    // 16 reconstruction columns / rows are padding.
    let params = KeyframeParams {
        y_ac_qi: 64,
        loop_filter_level: 35,
        sharpness_level: 2,
        nbr_of_dct_partitions: 1,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    assert_pixel_lockstep(1, 1, &params, None);
    assert_pixel_lockstep(1, 33, &params, None);
    assert_pixel_lockstep(33, 1, &params, None);
}

#[test]
fn lockstep_eight_partitions_all_populated_and_empty() {
    // 64×130 → 9 MB rows: all 8 §9.5 partitions populated via the
    // §20.4 `row % 8` round-robin. 24×24 → 2 MB rows against 8
    // partitions: six trailing partitions stay empty.
    let params = KeyframeParams {
        y_ac_qi: 40,
        loop_filter_level: 10,
        sharpness_level: 1,
        nbr_of_dct_partitions: 8,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    assert_pixel_lockstep(64, 130, &params, None);
    assert_pixel_lockstep(24, 24, &params, None);
}

#[test]
fn lockstep_token_prob_updates_at_l8_extremes() {
    // §13.4 token_prob_update() write path at the full L(8) wire
    // extremes (0 and 255) — outside the [1, 255] band the in-tree
    // fitter produces — plus a mid value, across both the §15 skip
    // path (level 0) and a filtered configuration.
    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    updates[0][0][0][0] = Some(0);
    updates[1][3][1][4] = Some(255);
    updates[2][7][2][10] = Some(1);
    updates[3][1][0][6] = Some(128);

    let unfiltered = KeyframeParams {
        y_ac_qi: 8,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 4,
        filter_type: false,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    assert_pixel_lockstep(47, 32, &unfiltered, Some(&updates));

    let filtered = KeyframeParams {
        y_ac_qi: 100,
        loop_filter_level: 28,
        sharpness_level: 4,
        nbr_of_dct_partitions: 1,
        filter_type: true,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };
    assert_pixel_lockstep(40, 24, &filtered, Some(&updates));
}
