//! End-to-end §9.5 multi-partition inter-frame round-trip for the
//! round-149 extension that splits the P-frame DCT-token output across
//! `params.nbr_of_dct_partitions` partitions.
//!
//! The §9.5 DCT-coefficient partition count is a layout-only choice
//! that applies symmetrically to key and inter frames per RFC 6386
//! §9.5 / §20.4: macroblock rows are distributed round-robin (row `r`
//! → partition `r % N`) and each partition is a separate `BoolEncoder`
//! instance finalised with its own §7.3 4-byte flush trailer. The
//! residual coding inside each partition is bit-identical to the
//! 1-partition case (the §13.3 above-context is frame-lived and
//! column-wise — shared across partitions; the "left" context resets
//! at every macroblock-row boundary so it never has to cross a
//! partition seam), so the self-decoded picture is unchanged across
//! all four legal partition counts.
//!
//! Before round 149 the P-frame encoder ([`encode_p_frame_multi_ref`])
//! always emitted exactly one DCT partition. The keyframe encoder
//! ([`encode_keyframe`]) gained the 1 / 2 / 4 / 8 split in round 137;
//! round 149 mirrors the keyframe pattern on the inter path.
//!
//! This test pins three independent invariants:
//!
//! 1. **Self-decoded picture is partition-count invariant.** The
//!    same source frame encoded into 1, 2, 4, and 8 partitions
//!    self-decodes to bit-identical samples (PSNR delta < `1e-9` dB).
//!    Any drift between counts would mean the §13.3 above / left
//!    predictor contexts had been wired incorrectly across partition
//!    boundaries.
//! 2. **Encoded-byte length grows monotonically with partition
//!    count.** Each extra partition pays a §7.3 4-byte flush trailer
//!    plus a 3-byte §9.5 size-table entry, so the wire length cannot
//!    shrink as `N` rises (the residual data is identical).
//! 3. **Out-of-range partition counts are rejected** before the
//!    long per-MB pick walk runs, mirroring the keyframe path's
//!    `EncodeError::InvalidDctPartitionCount` guard.
//!
//! Black-box: encoder output is fed straight into the crate's own
//! [`Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref, EncodeError, I420Frame,
    KeyframeParams, Vp8DecodedFrame, Vp8DecoderState,
};

/// Whole-Y-plane PSNR (8-bit peak = 255). Both planes must have the
/// same length.
fn y_psnr(src_y: &[u8], dec: &Vp8DecodedFrame) -> f64 {
    assert_eq!(src_y.len(), dec.y.len());
    let mut sse: u64 = 0;
    for (a, b) in src_y.iter().zip(dec.y.iter()) {
        let d = *a as i32 - *b as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / src_y.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Build a synthetic I420 source whose row-`r`, column-`c` luma
/// samples vary by `(r + c) & 0xff` and whose chroma planes hold a
/// flat 128. The pattern has non-trivial vertical structure so the
/// §20.4 round-robin actually exercises every partition (every MB
/// row carries non-skip tokens at the chosen quantiser).
fn gradient_source(width: u32, height: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = ((r + c) & 0xff) as u8;
        }
    }
    let cw = w / 2;
    let ch = h / 2;
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    (y, u, v)
}

/// Encode an I + P pair at the given partition count and return
/// (P-frame encoded bytes, P-frame self-decode Y-PSNR vs. the P
/// source).
///
/// Uses a frame size whose MB row count is `≥ 8` so the §20.4
/// round-robin routes work into every partition at `N = 8`.
fn encode_ip_at(partitions: u8) -> (Vec<u8>, f64) {
    let width = 32u32;
    let height = 128u32; // 8 MB rows.
    let qi = 16u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: partitions,
    };

    // The K-frame stays at 1 partition for every iteration — only the
    // P-frame partition count varies in this test (so the K-frame's
    // reconstruction is the same across the sweep and any picture
    // delta is attributable to the P-frame's layout choice).
    let k_params = KeyframeParams {
        nbr_of_dct_partitions: 1,
        ..params
    };

    let (k_y, k_u, k_v) = gradient_source(width, height);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &k_params).expect("encode K");

    // The P-frame source is the same gradient shifted vertically by
    // 8 rows — every MB row produces a non-trivial residual that
    // routes a payload into its assigned §9.5 partition.
    let mut p_y = vec![0u8; (width * height) as usize];
    let w = width as usize;
    let h = height as usize;
    for r in 0..h {
        for c in 0..w {
            p_y[r * w + c] = ((r + c + 8) & 0xff) as u8;
        }
    }
    let cw = (width / 2) as usize;
    let ch = (height / 2) as usize;
    let p_u = vec![128u8; cw * ch];
    let p_v = vec![128u8; cw * ch];
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let (p_bytes, _p_planes) =
        encode_p_frame_multi_ref(&p_frame, &k_planes, None, None, &params).expect("encode P");

    let mut dec = Vp8DecoderState::new();
    let _d_k = dec.decode_frame(&_k_bytes).expect("decode K");
    let d_p = dec.decode_frame(&p_bytes).expect("decode P");

    let psnr = y_psnr(&p_y, &d_p);
    (p_bytes, psnr)
}

/// §9.5 multi-partition self-decode round-trip on a P-frame.
///
/// Encodes the same I + P pair at `N ∈ {1, 2, 4, 8}` and asserts the
/// P-frame self-decode Y-PSNR is bit-identical to the 1-partition
/// baseline (the layout choice cannot alter sample values) and that
/// the encoded byte length grows monotonically with `N` (each extra
/// partition pays a §7.3 flush trailer + a §9.5 size-table entry).
#[test]
fn p_frame_multi_partition_psnr_matches_single_partition_baseline() {
    let mut psnrs = [0f64; 4];
    let mut byte_lens = [0usize; 4];
    for (slot, &count) in [1u8, 2, 4, 8].iter().enumerate() {
        let (bytes, psnr) = encode_ip_at(count);
        eprintln!(
            "P-frame partitions={count}: {} B, Y-PSNR = {psnr:.4} dB",
            bytes.len()
        );
        psnrs[slot] = psnr;
        byte_lens[slot] = bytes.len();
    }

    // PSNR invariance — multi-partition is a byte-layout reorganisation,
    // not a residual-coding change, so the reconstructed Y plane must
    // be bit-identical across all four counts.
    let baseline = psnrs[0];
    for (slot, &count) in [1u8, 2, 4, 8].iter().enumerate() {
        let psnr = psnrs[slot];
        assert!(
            (psnr - baseline).abs() < 1e-9,
            "P-frame partitions={count} PSNR {psnr} differs from \
             1-partition baseline {baseline} — the §13.3 predictor \
             contexts must be wired identically across partition seams",
        );
    }

    // Length monotonicity — each extra partition pays a §7.3 flush
    // trailer + a §9.5 size-table entry. The residual data is the
    // same across counts; only the layout grows.
    for w in byte_lens.windows(2) {
        assert!(
            w[1] >= w[0],
            "P-frame byte length must not shrink as partition count \
             grows: {byte_lens:?}",
        );
    }

    // And the headline floor still clears the project-wide 30 dB bar
    // (it does; the I+P pair encodes near-losslessly at qi=16).
    assert!(
        baseline >= 30.0,
        "P-frame Y-PSNR baseline {baseline:.2} dB below 30.0 dB target",
    );
}

/// §9.5 short-frame coverage on a P-frame: when the source has fewer
/// macroblock rows than partitions, some partitions are never written
/// to (the §20.4 round-robin only reaches the first `mb_rows`
/// partitions). The encoder must still emit a valid frame whose
/// unused partitions are minimal §7.3 flush trailers, and the decoder
/// must consume them without complaint.
#[test]
fn p_frame_multi_partition_short_frame_roundtrip() {
    let width = 32u32;
    let height = 32u32; // 2 MB rows.
    let qi = 16u8;

    let mut prev_psnr: Option<f64> = None;
    for count in [1u8, 2, 4, 8] {
        let params = KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            sharpness_level: 0,
            nbr_of_dct_partitions: count,
        };
        let k_params = KeyframeParams {
            nbr_of_dct_partitions: 1,
            ..params
        };

        let (k_y, k_u, k_v) = gradient_source(width, height);
        let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
        let (k_bytes, k_planes) =
            encode_keyframe_with_reconstruction(&k_frame, &k_params).expect("encode K");

        // Drive the P-frame with a slight horizontal shift so the
        // residual is non-trivial in both MB rows.
        let mut p_y = vec![0u8; (width * height) as usize];
        let w = width as usize;
        let h = height as usize;
        for r in 0..h {
            for c in 0..w {
                p_y[r * w + c] = ((r + c + 4) & 0xff) as u8;
            }
        }
        let cw = (width / 2) as usize;
        let ch = (height / 2) as usize;
        let p_u = vec![128u8; cw * ch];
        let p_v = vec![128u8; cw * ch];
        let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

        let (p_bytes, _) =
            encode_p_frame_multi_ref(&p_frame, &k_planes, None, None, &params).expect("encode P");

        let mut dec = Vp8DecoderState::new();
        let _d_k = dec.decode_frame(&k_bytes).expect("decode K");
        let d_p = dec.decode_frame(&p_bytes).expect("decode P");

        let psnr = y_psnr(&p_y, &d_p);
        eprintln!(
            "short-frame P partitions={count} ({h} px tall): {} B, Y-PSNR = {psnr:.4} dB",
            p_bytes.len()
        );

        match prev_psnr {
            None => prev_psnr = Some(psnr),
            Some(prev) => assert!(
                (psnr - prev).abs() < 1e-9,
                "short-frame partitions={count} PSNR {psnr} differs from \
                 prior count's PSNR {prev}",
            ),
        }
    }
}

/// The encoder rejects partition counts outside the §9.5 four-value
/// table (1 / 2 / 4 / 8) before running the long per-MB pick walk.
#[test]
fn p_frame_invalid_partition_count_rejected() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;

    // Build a valid K reference once — we only care about whether the
    // P-frame call rejects the bad count before it reaches the pick
    // walk.
    let k_params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
    };
    let (k_y, k_u, k_v) = gradient_source(width, height);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &k_params).expect("encode K");

    let p_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);

    for bad in [0u8, 3, 5, 6, 7, 9, 16, 255] {
        let params = KeyframeParams {
            y_ac_qi: qi,
            loop_filter_level: 0,
            sharpness_level: 0,
            nbr_of_dct_partitions: bad,
        };
        match encode_p_frame_multi_ref(&p_frame, &k_planes, None, None, &params) {
            Err(EncodeError::InvalidDctPartitionCount { value }) => assert_eq!(
                value, bad,
                "InvalidDctPartitionCount surfaced wrong value for {bad}",
            ),
            other => panic!("expected InvalidDctPartitionCount for {bad}, got {other:?}"),
        }
    }
}
