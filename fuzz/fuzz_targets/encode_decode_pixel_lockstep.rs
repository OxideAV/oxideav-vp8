#![no_main]

//! Fuzz: pixel-exact encode→decode lockstep differential
//! (RFC 6386 §9 / §12 / §13.4 / §14 / §15 / §19.2).
//!
//! The round-264 `panic_free_encode_decode_e2e` target stitched
//! `encode_keyframe` → `decode_vp8` into one iteration, but its oracle
//! stops at the §9.1 visible width / height round-trip — the decoded
//! *pixels* are never compared against anything. The encoder, however,
//! makes a much stronger promise: `encode_keyframe_with_reconstruction`
//! returns the post-§15 reconstruction planes the encoder itself
//! produced while walking the frame, and the §15.1 design contract is
//! that a compliant decoder reproduces those exact pixels (the encoder
//! predicts each macroblock from its own reconstruction precisely so
//! the two sides stay in lockstep). That makes the encoder's own
//! reconstruction a *bit-exact differential oracle* for the decoder —
//! any single-pixel drift anywhere in the §12 intra prediction, §14
//! dequant + inverse transform, §15 loop-filter, or §9.1
//! visible-cropping chain surfaces as a byte mismatch instead of
//! hiding inside "both halves were wrong the same way" (which a
//! dimensions-only or self-roundtrip oracle cannot see... unless the
//! two wrongs differ).
//!
//! Three coverage gaps closed relative to the seventeen existing
//! targets:
//!
//! 1. **Pixel-differential oracle** — first target to assert decoded
//!    pixel content (full visible Y + U + V planes) against an
//!    independent in-process reference on every iteration.
//! 2. **Non-MB-aligned dimensions** — the e2e target derives its
//!    dimensions in whole MB units (`1..=16` MBs per axis), so the §9.1
//!    partial-macroblock path (encoder edge-replication padding on the
//!    right / bottom partial MB + decoder visible-crop of the padded
//!    reconstruction, §15 loop filter running over the *padded* raster
//!    before the crop) has never been fuzzed in lockstep. This target
//!    draws each axis in raw luma pixels (width `1..=64`, height
//!    `1..=144`), so almost every iteration carries partial MBs —
//!    including the 1-pixel-wide / 1-pixel-tall degenerate strips.
//! 3. **§13.4 token-prob-update write path** — no existing fuzz target
//!    drives the encoder-side `token_prob_update()` emitter
//!    (`encode_keyframe_with_reconstruction_and_token_updates`);
//!    `parse_headers` / `panic_free_token_block` fuzz only the *read*
//!    side. Half of all iterations here thread a fuzz-shaped sparse
//!    `TokenProbUpdates` payload (up to 8 positions, raw `0..=255`
//!    probability bytes — the full §13.4 L(8) wire range, wider than
//!    the `[1, 255]` band the in-tree fitter emits) through the
//!    encode, so the merged `coeff_probs` table used by the §13.3
//!    token coder and re-derived by the decoder from the on-wire
//!    update flags is locked pixel-exact end-to-end, at probability
//!    extremes the production fitter never produces.
//!
//! Unlike the `panic_free_*` family this target feeds the encoder only
//! *valid* parameters (each raw byte is normalised into its §9.4 /
//! §9.5 / §9.6 legal range — the rejection envelope is
//! `panic_free_encode_keyframe`'s job), so an `Err` from either side
//! is itself a finding: the harness panics on encode failure, decode
//! failure, dimension drift, MB-grid drift, or any pixel mismatch.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`      | Visible width: `1 + (b % 64)` luma px |
//! | `[1]`      | Visible height: `1 + (b % 144)` luma px (≥ 113 px populates all 8 §9.5 DCT partitions via the §20.4 `row % N` round-robin) |
//! | `[2]`      | `y_ac_qi = b % 128` |
//! | `[3]`      | `loop_filter_level = b % 64` (0 ⇒ §15 skip path) |
//! | `[4]`      | `sharpness_level = b % 8` |
//! | `[5]`      | `nbr_of_dct_partitions ∈ {1, 2, 4, 8}` via `b % 4` |
//! | `[6]`      | flags: bit 0 = `filter_type` (§15.2 simple vs §15.3 normal), bit 1 = thread a §13.4 update payload, bits 4..7 = §13 `TrellisStrength` (`b >> 4`, clamped `0..=8`; `0` = OFF / plain rounding) |
//! | `[7..31]`  | eight §13.4 update triples: 2-byte LE slot selector (`% 1056` → `(plane, band, ctx, position)`) + 1 raw probability byte |
//! | `[31..]`   | I420 pixel payload — tiled across the three planes |
//!
//! Max area 64 × 144 = 9 216 luma px (≤ 36 MBs), well inside the
//! shared [`oxideav_vp8_fuzz::accept_dimensions`] budget; hard input
//! cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    decode_vp8, encode_keyframe_with_reconstruction_and_token_updates, I420Frame, KeyframeParams,
    TokenProbUpdates, TrellisStrength,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
/// 7 scalar bytes + 8 × 3-byte §13.4 update triples.
const HEADER_BYTES: usize = 7 + 8 * 3;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions in raw luma pixels — deliberately NOT MB-aligned
    // so the partial-macroblock padding / cropping seam is on the hot
    // path of nearly every iteration.
    let width = 1u32 + u32::from(data[0] % 64);
    let height = 1u32 + u32::from(data[1] % 144);
    if !accept_dimensions(width, height) {
        return;
    }

    // Normalised (always-valid) parameter envelope. The rejection
    // surface is covered by `panic_free_encode_keyframe`; here every
    // accepted input must produce a decodable, pixel-exact bitstream.
    // §13 trellis aggressiveness, fuzz-derived from the high nibble of the
    // flags byte so the pixel-lockstep differential covers the whole knob
    // range — including OFF (plain rounding) and the strong settings — at
    // every quantiser / dimension. The encoder must stay bit-exact with the
    // decoder at *every* strength (the reconstruction always reflects the
    // trellis-chosen levels), so a mismatch here is a real lockstep defect.
    let trellis_strength = TrellisStrength::new(f64::from(data[6] >> 4));

    let params = KeyframeParams {
        y_ac_qi: data[2] % 128,
        loop_filter_level: data[3] % 64,
        sharpness_level: data[4] % 8,
        nbr_of_dct_partitions: 1u8 << (data[5] % 4),
        filter_type: (data[6] & 0x01) != 0,
        trellis_strength,
    };

    // Optional sparse §13.4 token_prob_update() payload: up to eight
    // (plane, band, ctx, position) slots replaced with raw fuzz-chosen
    // probability bytes. The full 0..=255 L(8) wire range is allowed —
    // wider than the [1, 255] band the in-tree fitter clamps to — so
    // the §7.3 bool coder is locked at both probability extremes.
    let updates: Option<TokenProbUpdates> = if (data[6] & 0x02) != 0 {
        let mut u: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
        for triple in data[7..HEADER_BYTES].chunks_exact(3) {
            let slot = usize::from(u16::from_le_bytes([triple[0], triple[1]])) % (4 * 8 * 3 * 11);
            let (i, rest) = (slot / (8 * 3 * 11), slot % (8 * 3 * 11));
            let (j, rest) = (rest / (3 * 11), rest % (3 * 11));
            let (k, t) = (rest / 11, rest % 11);
            u[i][j][k][t] = Some(triple[2]);
        }
        Some(u)
    } else {
        None
    };

    // I420 plane payload, tiled from the input tail (tight strides).
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;
    let payload = &data[HEADER_BYTES..];
    let mut y_plane = vec![0u8; y_len];
    let mut u_plane = vec![0u8; uv_len];
    let mut v_plane = vec![0u8; uv_len];
    if !payload.is_empty() {
        for (i, slot) in y_plane.iter_mut().enumerate() {
            *slot = payload[i % payload.len()];
        }
        for (i, slot) in u_plane.iter_mut().enumerate() {
            *slot = payload[(i + y_len) % payload.len()];
        }
        for (i, slot) in v_plane.iter_mut().enumerate() {
            *slot = payload[(i + y_len + uv_len) % payload.len()];
        }
    }
    let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);

    // Encode leg — every parameter is in-range, so failure is a defect.
    let (bytes, recon) = match encode_keyframe_with_reconstruction_and_token_updates(
        &frame,
        &params,
        updates.as_ref(),
    ) {
        Ok(ok) => ok,
        Err(e) => panic!(
            "encode_keyframe rejected an in-range input: {e:?} \
                 ({width}x{height}, params {params:?})"
        ),
    };

    // Decode leg — the encoder's §19.2 output must decode through the
    // crate's own §9.1 keyframe decoder.
    let decoded = match decode_vp8(&bytes) {
        Ok(d) => d,
        Err(e) => panic!(
            "decode_vp8 rejected an encoder-produced bitstream: {e:?} \
             ({width}x{height}, params {params:?}, {} bytes)",
            bytes.len()
        ),
    };

    // §9.1 dimension + MB-grid lockstep.
    assert_eq!(decoded.width, width, "visible width drift");
    assert_eq!(decoded.height, height, "visible height drift");
    assert_eq!(recon.mb_cols, w.div_ceil(16), "encoder mb_cols drift");
    assert_eq!(recon.mb_rows, h.div_ceil(16), "encoder mb_rows drift");

    // Pixel-exact differential: the decoder's visible-cropped planes
    // must byte-equal the encoder's own post-§15 reconstruction
    // (cropped from the MB-padded raster to the §9.1 visible window).
    for row in 0..h {
        let dec = &decoded.y[row * w..row * w + w];
        let enc = &recon.y[row * recon.y_stride..row * recon.y_stride + w];
        assert_eq!(
            dec, enc,
            "luma drift at row {row} ({width}x{height}, {params:?})"
        );
    }
    for row in 0..uvh {
        let dec_u = &decoded.u[row * uvw..row * uvw + uvw];
        let enc_u = &recon.u[row * recon.uv_stride..row * recon.uv_stride + uvw];
        assert_eq!(
            dec_u, enc_u,
            "U drift at row {row} ({width}x{height}, {params:?})"
        );
        let dec_v = &decoded.v[row * uvw..row * uvw + uvw];
        let enc_v = &recon.v[row * recon.uv_stride..row * recon.uv_stride + uvw];
        assert_eq!(
            dec_v, enc_v,
            "V drift at row {row} ({width}x{height}, {params:?})"
        );
    }
});
