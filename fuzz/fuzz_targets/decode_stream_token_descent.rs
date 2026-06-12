#![no_main]

//! Fuzz: full-frame decode driver aimed at the round-283 hot-path
//! rewrite — the fused §13.2 coefficient-tree descent
//! (`dct_tokens::decode_block_core`, the branch-by-branch walk with the
//! DCT_0 zero-run re-entry loop), the §20.16 zigzag-direct coefficient
//! writes in `decode_mb_coeffs`, and the §7.3 batched bool-decoder
//! renormalisation (`bool_decoder::renormalize`, the
//! `leading_zeros`-derived shift + single byte splice).
//!
//! Random bytes almost never survive the §9.1 / §19.2 header validation
//! long enough to reach a deep token descent, so this target is seeded
//! with packet streams derived from the crate's own `tests/fixtures/`
//! IVF corpus — keyframes AND inter frames (`i-frame-then-p-frame-64x64`,
//! `golden-update-cycle`, `altref-arnr-on`, …) re-framed into the
//! length-prefixed packet format below. libFuzzer mutations of those
//! seeds keep the leading keyframe structurally valid while corrupting
//! the §13 token partitions and the inter-frame §16 mode/MV data — the
//! exact adversarial envelope the fused descent and the batched
//! renormaliser must survive (truncated partitions, mid-descent EOF,
//! zero-run loops crossing block boundaries, hostile probability
//! tables via §13.4 `token_prob_update`).
//!
//! Input format: repeated `[u16-LE len][len payload bytes]` packets
//! (see `oxideav_vp8_fuzz::split_packets`), each payload one VP8
//! elementary-stream frame fed to `Vp8DecoderState::decode_frame`.
//!
//! Three oracles beyond panic-freedom:
//!
//! 1. **Cross-entry-point differential** — the first packet is also fed
//!    to the stateless `decode_vp8`. A fresh `Vp8DecoderState` and the
//!    one-shot entry point both start from the §13.5 / §16.1 / §17.2
//!    defaults, so on a key frame the two paths MUST agree: same
//!    Ok/Err-ness, and byte-identical visible Y / U / V planes on
//!    success. Any drift between the two keyframe decode paths is a
//!    finding (panic).
//! 2. **Plane materialisation** — every decoded plane byte is folded
//!    into an FNV-1a accumulator, so the harness reads every byte the
//!    decoder claims to have produced (a stale-length / short-write bug
//!    surfaces under ASan here).
//! 3. **Scalar-vs-SIMD kernel differential** (`simd` feature, the
//!    default for this fuzz crate — cargo-fuzz always builds on
//!    nightly): the §14.1 dequant, §14.3 inverse WHT, §14.4 inverse
//!    DCT, §12.2 TM_PRED, and §18.3 / §20.14 six-tap kernels are each
//!    run through their scalar AND SIMD implementations on
//!    attacker-shaped inputs drawn from the same fuzz bytes, and any
//!    divergence panics. The full-frame decode above runs on the SIMD
//!    dispatch path, so between the two legs both halves of every
//!    kernel pair stay under fuzz.
//!
//! OOM / wall-time caps: input ≤ 16 KiB (the fixture-derived seeds top
//! out under 3 KiB, so this leaves mutation headroom), ≤ 12 packets per
//! iteration, and the §9.1 dimension cap from
//! [`oxideav_vp8_fuzz::accept_dimensions`] applied to every packet that
//! parses as a key frame (an inter frame inherits the locked geometry,
//! which was itself capped when the key frame passed).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{decode_vp8, frame_header::Vp8FrameHeader, Vp8DecodedFrame, Vp8DecoderState};
use oxideav_vp8_fuzz::{accept_dimensions, split_packets};

const MAX_INPUT_BYTES: usize = 16 * 1024;
const MAX_PACKETS_PER_ITER: usize = 12;

/// FNV-1a 64-bit fold — forces every decoded plane byte to be read.
fn fnv1a(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= u64::from(b);
        *acc = acc.wrapping_mul(0x100_0000_01b3);
    }
}

fn hash_frame(acc: &mut u64, frame: &Vp8DecodedFrame) {
    fnv1a(acc, &frame.y);
    fnv1a(acc, &frame.u);
    fnv1a(acc, &frame.v);
}

/// Oracle 1: a fresh stateful decoder and the stateless one-shot entry
/// point must agree on the first packet (both start from spec defaults).
/// Only accept/reject polarity plus pixels are compared — the two entry
/// points legitimately phrase some reject reasons differently.
fn assert_first_packet_parity(pkt: &[u8], stateful: Option<&Vp8DecodedFrame>) {
    let oneshot = decode_vp8(pkt);
    match (stateful, &oneshot) {
        (Some(a), Ok(b)) => {
            assert_eq!(
                (a.width, a.height),
                (b.width, b.height),
                "§9.1 dimension drift between Vp8DecoderState and decode_vp8"
            );
            assert_eq!(
                a.y, b.y,
                "Y-plane drift between Vp8DecoderState and decode_vp8"
            );
            assert_eq!(
                a.u, b.u,
                "U-plane drift between Vp8DecoderState and decode_vp8"
            );
            assert_eq!(
                a.v, b.v,
                "V-plane drift between Vp8DecoderState and decode_vp8"
            );
        }
        (Some(_), Err(e)) => {
            panic!("fresh Vp8DecoderState accepted a first packet decode_vp8 rejects: {e:?}")
        }
        (None, Ok(_)) => {
            panic!("decode_vp8 accepted a first packet a fresh Vp8DecoderState rejects")
        }
        (None, Err(_)) => {}
    }
}

/// Oracle 3: scalar-vs-SIMD kernel differential on attacker-shaped
/// inputs cycled out of the raw fuzz bytes. Divergence panics.
#[cfg(feature = "simd")]
fn simd_parity_leg(data: &[u8]) {
    use oxideav_vp8::motion_comp::{BILINEAR_FILTERS, SIXTAP_FILTERS};

    if data.is_empty() {
        return;
    }
    let at = |i: usize| data[i % data.len()];
    let i16_at = |i: usize| i16::from_le_bytes([at(2 * i), at(2 * i + 1)]);

    // §14.1 dequant: full-i16 coefficients and factors — the scalar and
    // SIMD paths both wrap identically in i32 → i16 truncation, so the
    // whole input space is legal for the pair.
    let mut blk = [0i16; 16];
    for (k, c) in blk.iter_mut().enumerate() {
        *c = i16_at(k);
    }
    let (s, v) = oxideav_vp8::dequant::dequant_block_parity_pair(&blk, i16_at(16), i16_at(17));
    assert_eq!(s, v, "§14.1 dequant scalar/SIMD divergence on {blk:?}");

    // §14.3 / §14.4 inverse transforms: inputs clamped to the ±1023
    // §14.2 coefficient envelope (the documented contract of the public
    // entry points — outside it the intermediate i32 arithmetic is
    // undefined by the spec, so a wrap difference there is not a bug).
    let mut coeffs = [0i16; 16];
    for (k, c) in coeffs.iter_mut().enumerate() {
        *c = i16_at(18 + k).clamp(-1023, 1023);
    }
    let (s, v) = oxideav_vp8::inverse_transform::inverse_wht_4x4_parity_pair(&coeffs);
    assert_eq!(
        s, v,
        "§14.3 inverse WHT scalar/SIMD divergence on {coeffs:?}"
    );
    let (s, v) = oxideav_vp8::inverse_transform::inverse_dct_4x4_parity_pair(&coeffs);
    assert_eq!(
        s, v,
        "§14.4 inverse DCT scalar/SIMD divergence on {coeffs:?}"
    );

    // §12.2 TM_PRED: both block widths, raw neighbour bytes.
    let mut above16 = [0u8; 16];
    let mut left16 = [0u8; 16];
    for k in 0..16 {
        above16[k] = at(50 + k);
        left16[k] = at(66 + k);
    }
    let p = at(82);
    let (s, v) = oxideav_vp8::intra_predict::predict_tm_parity_pair(16, &above16, &left16, p);
    assert_eq!(s, v, "§12.2 TM_PRED 16×16 scalar/SIMD divergence");
    let (s, v) =
        oxideav_vp8::intra_predict::predict_tm_parity_pair(8, &above16[..8], &left16[..8], p);
    assert_eq!(s, v, "§12.2 TM_PRED 8×8 scalar/SIMD divergence");

    // §18.3 / §20.14 six-tap synthesis: every kernel scale, both filter
    // sets, fractional offsets masked into the {0..7}² tap-table range.
    let mx = usize::from(at(83)) & 7;
    let my = usize::from(at(84)) & 7;
    let filters = if at(85) & 1 == 0 {
        &SIXTAP_FILTERS
    } else {
        &BILINEAR_FILTERS
    };
    let mut halo_4x4 = [0u8; 81];
    for (k, h) in halo_4x4.iter_mut().enumerate() {
        *h = at(86 + k);
    }
    let (s, v) = oxideav_vp8::motion_comp::sixtap_2d_parity_pair(&halo_4x4, mx, my, filters);
    assert_eq!(
        s, v,
        "§18.3 sixtap_2d scalar/SIMD divergence at ({mx},{my})"
    );
    let mut halo_luma = [0u8; 21 * 21];
    for (k, h) in halo_luma.iter_mut().enumerate() {
        *h = at(167 + k);
    }
    let (s, v) = oxideav_vp8::motion_comp::sixtap_mb_luma_parity_pair(&halo_luma, mx, my, filters);
    assert_eq!(
        s, v,
        "§18.3 sixtap_mb_luma scalar/SIMD divergence at ({mx},{my})"
    );
    let mut halo_chroma = [0u8; 13 * 13];
    for (k, h) in halo_chroma.iter_mut().enumerate() {
        *h = at(608 + k);
    }
    let (s, v) =
        oxideav_vp8::motion_comp::sixtap_mb_chroma_parity_pair(&halo_chroma, mx, my, filters);
    assert_eq!(
        s, v,
        "§18.3 sixtap_mb_chroma scalar/SIMD divergence at ({mx},{my})"
    );
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let packets = split_packets(data, MAX_PACKETS_PER_ITER);

    let mut state = Vp8DecoderState::new();
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for (idx, pkt) in packets.iter().enumerate() {
        // §9.1 dimension cap: only key frames carry dimensions; an
        // inter frame inherits the already-capped locked geometry.
        if let Ok(header) = Vp8FrameHeader::parse(pkt) {
            if let (Some(w), Some(h)) = (header.width, header.height) {
                if !accept_dimensions(u32::from(w), u32::from(h)) {
                    return;
                }
            }
        }
        let decoded = state.decode_frame(pkt);
        if idx == 0 {
            assert_first_packet_parity(pkt, decoded.as_ref().ok());
        }
        if let Ok(frame) = &decoded {
            hash_frame(&mut acc, frame);
        }
    }
    std::hint::black_box(acc);

    #[cfg(feature = "simd")]
    simd_parity_leg(data);
});
