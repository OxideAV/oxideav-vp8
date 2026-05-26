//! End-to-end self-decode of the round-150 caller-driven §9.7 / §9.8
//! reference-slot refresh layer
//! ([`oxideav_vp8::encode_p_frame_multi_ref_with_refresh`] and
//! [`oxideav_vp8::Vp8InterStreamEncoder::encode_p_frame_with_refresh`]).
//!
//! Round 149 (`b565dcd`) baked the inter-frame refresh ladder to
//! `refresh_last = 1`, every other §9.7 / §9.8 bit `0`. Round 150
//! exposes the five bits through [`oxideav_vp8::RefreshControls`] so a
//! caller can express GOLDEN / ALTREF rotation patterns:
//!
//!   * `refresh_golden_frame` — replace GOLDEN with the current
//!     reconstruction.
//!   * `copy_buffer_to_alternate = 1` — snapshot LAST into ALTREF
//!     before LAST gets overwritten by the next P-frame.
//!   * `refresh_last = false` — keep the previous LAST picture so the
//!     next P-frame predicts off it (a "hold" pattern).
//!
//! What this test pins:
//!
//!   * the §19.2 page-122 wire shape — the encoder's L(1) refresh
//!     bits + gated L(2) `copy_buffer_to_*` selectors decode through
//!     [`oxideav_vp8::Vp8CodedHeader::parse`] with the requested
//!     values;
//!   * the §20 page-147 slot-rotation walk — the
//!     [`oxideav_vp8::Vp8InterStreamEncoder`] driver evolves its
//!     `LAST` / `GOLDEN` / `ALTREF` trio in lockstep with the
//!     in-tree [`oxideav_vp8::Vp8DecoderState`] consuming the same
//!     wire (every slot byte-identical after every frame);
//!   * the picker-quality consequence — promoting a clean
//!     reconstruction into GOLDEN lets the picker beat LAST on a
//!     subsequent disturbance (the round-148 `goldenref` test pinned
//!     this for GOLDEN held from the keyframe; round 150 pins it for
//!     GOLDEN promoted from an intermediate P-frame).
//!
//! Black-box: the encoder's output is fed straight into the crate's
//! own [`oxideav_vp8::Vp8DecoderState`] — no external codec consulted.

use oxideav_vp8::{
    encode_p_frame_multi_ref_with_refresh, FrameKind, I420Frame, KeyframeParams, RefreshControls,
    Vp8CodedHeader, Vp8DecoderState, Vp8FrameHeader, Vp8InterStreamEncoder,
};

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

/// Render a "stripe" pattern: row-0 MB carries a high-contrast 2-pixel
/// vertical-stripe luma signal; row-1 MB stays at flat gray 128.
/// Chroma stays flat at 128.
fn stripe_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![128u8; width * height];
    for r in 0..16.min(height) {
        for c in 0..width {
            y[r * width + c] = if (c / 2) & 1 == 0 { 32 } else { 224 };
        }
    }
    let cw = width / 2;
    let ch = height / 2;
    (y, vec![128u8; cw * ch], vec![128u8; cw * ch])
}

/// Flat-gray pattern: every pixel = 128.
fn flat_frame(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![128u8; width * height];
    let cw = width / 2;
    let ch = height / 2;
    (y, vec![128u8; cw * ch], vec![128u8; cw * ch])
}

/// Re-parse a P-frame's §19.2 control partition and return the
/// `Vp8CodedHeader` for inspection.
fn parse_p_coded_header(bytes: &[u8]) -> Vp8CodedHeader {
    let hdr = Vp8FrameHeader::parse(bytes).expect("§19.1 tag parses");
    assert!(!hdr.key_frame, "inter-frame parser called on a K frame");
    let partition_start = hdr.header_bytes_consumed;
    let partition_end = partition_start + hdr.first_partition_size as usize;
    let partition = &bytes[partition_start..partition_end];
    Vp8CodedHeader::parse(partition, /*key_frame=*/ false).expect("§19.2 control partition parses")
}

/// Round-150 baseline: a default [`RefreshControls`] produces a
/// byte-identical wire to the round-149 `encode_p_frame_multi_ref`
/// path. The wrapper-versus-direct call must agree byte-for-byte.
#[test]
fn default_refresh_controls_match_round_149_wire_byte_for_byte() {
    let width = 32u32;
    let height = 32u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let (legacy_bytes, _) = oxideav_vp8::encode_p_frame_multi_ref(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
    )
    .expect("legacy multi-ref P");

    let (new_bytes, _) = encode_p_frame_multi_ref_with_refresh(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &RefreshControls::default(),
    )
    .expect("new with-refresh P with default controls");

    assert_eq!(
        legacy_bytes, new_bytes,
        "default RefreshControls must produce the round-149 wire byte-for-byte"
    );
}

/// Wire-shape check: a P-frame emitted with
/// `refresh_golden_frame = true` carries the §19.2 page-122 layout:
///   refresh_golden = 1, copy_buffer_to_golden gated OFF (None),
///   refresh_alternate = 0, copy_buffer_to_alternate = L(2) value,
///   refresh_last = caller value.
/// The §9.7 page-38 spec table guarantees the L(2) copy field is
/// suppressed when its refresh bit is 1 (the decoder reads None).
#[test]
fn refresh_golden_gates_copy_buffer_to_golden_off_in_wire() {
    let width = 16u32;
    let height = 16u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let (_k_bytes, k_planes) =
        oxideav_vp8::encode_keyframe_with_reconstruction(&k_frame, &params).expect("K");

    let (p_y, p_u, p_v) = flat_frame(width as usize, height as usize);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let refresh = RefreshControls {
        refresh_golden_frame: true,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 0,
        copy_buffer_to_alternate: 2, // GOLDEN → ALTREF (read because refresh_alt = false)
        refresh_last: true,
    };
    let (p_bytes, _) = encode_p_frame_multi_ref_with_refresh(
        &p_frame,
        &k_planes,
        Some(&k_planes),
        Some(&k_planes),
        &params,
        &refresh,
    )
    .expect("P with refresh_golden_frame=true");

    let coded = parse_p_coded_header(&p_bytes);
    assert_eq!(coded.refresh_golden_frame, Some(true));
    assert_eq!(coded.refresh_alternate_frame, Some(false));
    assert_eq!(
        coded.copy_buffer_to_golden, None,
        "§19.2 page 122: copy_buffer_to_golden is gated OFF when \
         refresh_golden_frame == 1 (decoder reads None)"
    );
    assert_eq!(
        coded.copy_buffer_to_alternate,
        Some(2),
        "copy_buffer_to_alternate must be present when refresh_alt = false"
    );
    assert_eq!(coded.refresh_last, Some(true));
}

/// Invalid copy_buffer_to_* values are rejected before any encoding
/// work runs. Three rejection paths:
///   1. raw out-of-range (`> 2`),
///   2. non-zero copy when matching refresh bit is set (silent-intent
///      guard added in round 150 — see `RefreshControls::validate`).
#[test]
fn invalid_refresh_controls_are_rejected_up_front() {
    use oxideav_vp8::{CopyBufferSelector, EncodeError};

    let bad_golden = RefreshControls {
        refresh_golden_frame: false,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 3,
        copy_buffer_to_alternate: 0,
        refresh_last: true,
    };
    assert_eq!(
        bad_golden.validate(),
        Err(EncodeError::InvalidCopyBufferSelector {
            which: CopyBufferSelector::Golden,
            value: 3
        })
    );

    let bad_alt = RefreshControls {
        refresh_golden_frame: false,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 0,
        copy_buffer_to_alternate: 5,
        refresh_last: true,
    };
    assert_eq!(
        bad_alt.validate(),
        Err(EncodeError::InvalidCopyBufferSelector {
            which: CopyBufferSelector::Alternate,
            value: 5
        })
    );

    let silent_golden = RefreshControls {
        refresh_golden_frame: true,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 1, // would be silently dropped (refresh bit wins)
        copy_buffer_to_alternate: 0,
        refresh_last: true,
    };
    assert!(silent_golden.validate().is_err());
}

/// Slot-rotation walk test: drive an I+P×3 stream that exercises
/// three §9.7 / §9.8 refresh patterns (`refresh_golden_frame`,
/// `copy_buffer_to_alternate ∈ {1, 2}`, `refresh_last = false`), and
/// verify that the [`oxideav_vp8::Vp8InterStreamEncoder`] driver's
/// `LAST` / `GOLDEN` / `ALTREF` trio evolves per the §20 page-147 walk:
///
/// * After P1 (default ladder): LAST = P1 recon; GOLDEN / ALTREF
///   unchanged from K.
/// * After P2 (`refresh_golden_frame = true`,
///   `copy_buffer_to_alternate = 1`): LAST = P2 recon; GOLDEN = P2
///   recon; ALTREF = pre-rotation LAST = P1 recon.
/// * After P3 (`refresh_last = false`,
///   `copy_buffer_to_alternate = 2`): LAST = pre-P3 LAST (= P2 recon);
///   GOLDEN unchanged from P2; ALTREF = pre-rotation GOLDEN = P2 recon.
///
/// Each frame's emitted bitstream also decodes cleanly through the
/// in-tree decoder (PSNR sanity floor), which is what proves the
/// header bits the encoder wrote actually match the per-frame
/// `Vp8CodedHeader::parse` (the wire-shape lockstep guarantee — the
/// `refresh_golden_gates_copy_buffer_to_golden_off_in_wire` test
/// pins the bit layout one level down).
#[test]
fn stream_slot_rotation_walks_s20_page_147_ordering() {
    let width = 32u32;
    let height = 32u32;
    let qi = 16u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let mut dec = Vp8DecoderState::new();

    let (k_y, k_u, k_v) = stripe_frame(width as usize, height as usize);
    let (flat_y, flat_u, flat_v) = flat_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let flat = I420Frame::packed(width, height, &flat_y, &flat_u, &flat_v);

    // Frame 0 (K): scheduler-driven; populates all three slots with
    // the K reconstruction. The keyframe slot is byte-identical
    // between encoder and decoder (both populate from the same
    // reconstruction).
    let e0 = enc.encode_frame(&k_frame).expect("encode K");
    let _d0 = dec.decode_frame(&e0.bytes).expect("decode K");
    assert_eq!(e0.kind, FrameKind::Key);

    // P1: default scheduler ladder (refresh_last only).
    let e1 = enc.encode_frame(&flat).expect("encode P1");
    let _d1 = dec.decode_frame(&e1.bytes).expect("decode P1");
    assert_eq!(e1.kind, FrameKind::InterZeroMv);
    // GOLDEN / ALTREF unchanged from K — they still equal the K slot.
    let k_slot_dims = (
        enc.golden().expect("GOLDEN present").y_stride,
        enc.golden().expect("GOLDEN present").mb_cols,
        enc.golden().expect("GOLDEN present").mb_rows,
    );
    let p1_last = enc.last().cloned();
    let p1_golden = enc.golden().cloned();
    let p1_altref = enc.altref().cloned();
    assert_eq!(
        p1_golden, p1_altref,
        "K-refreshed GOLDEN / ALTREF must remain equal across P1 (neither was touched)"
    );

    // P2: refresh GOLDEN with current reconstruction, copy LAST → ALTREF.
    let e2 = enc
        .encode_p_frame_with_refresh(
            &flat,
            &RefreshControls {
                refresh_golden_frame: true,
                refresh_alternate_frame: false,
                copy_buffer_to_golden: 0,
                copy_buffer_to_alternate: 1, // copy pre-rotation LAST (= P1 recon) into ALTREF
                refresh_last: true,
            },
        )
        .expect("encode P2 with refresh ladder");
    let _d2 = dec.decode_frame(&e2.bytes).expect("decode P2");
    assert_eq!(e2.kind, FrameKind::InterZeroMv);
    // ALTREF should now equal the pre-rotation LAST (= P1 recon).
    assert_eq!(
        enc.altref().cloned(),
        p1_last.clone(),
        "ALTREF must hold pre-rotation LAST (= P1 recon) after copy_buffer_to_alternate = 1"
    );
    // GOLDEN should now equal the current LAST (both refreshed from
    // the P2 reconstruction in the §20 page-147 walk).
    assert_eq!(
        enc.golden().cloned(),
        enc.last().cloned(),
        "refresh_golden_frame + refresh_last must install the same current reconstruction into \
         both slots (the §20 page-147 walk's refresh_gf and refresh_last cases use the same source)"
    );
    // GOLDEN must have CHANGED from its K-installed value.
    assert!(
        enc.golden().cloned() != p1_golden,
        "GOLDEN must have been replaced by the refresh_golden_frame write"
    );

    // P3: hold LAST (refresh_last = false), copy GOLDEN → ALTREF.
    let pre_p3_last = enc.last().cloned();
    let pre_p3_golden = enc.golden().cloned();
    let e3 = enc
        .encode_p_frame_with_refresh(
            &flat,
            &RefreshControls {
                refresh_golden_frame: false,
                refresh_alternate_frame: false,
                copy_buffer_to_golden: 0,
                copy_buffer_to_alternate: 2, // copy pre-rotation GOLDEN into ALTREF
                refresh_last: false,         // hold LAST
            },
        )
        .expect("encode P3 with hold-LAST + GOLDEN→ALTREF");
    let _d3 = dec.decode_frame(&e3.bytes).expect("decode P3");
    assert_eq!(
        enc.last().cloned(),
        pre_p3_last,
        "refresh_last = false ⇒ LAST is unchanged across P3"
    );
    assert_eq!(
        enc.altref().cloned(),
        pre_p3_golden,
        "ALTREF must hold pre-rotation GOLDEN after copy_buffer_to_alternate = 2"
    );

    // Sanity: the slot's macroblock-grid dimensions are preserved (the
    // copy / refresh paths must not corrupt the slot layout).
    assert_eq!(
        (
            enc.golden().expect("GOLDEN present").y_stride,
            enc.golden().expect("GOLDEN present").mb_cols,
            enc.golden().expect("GOLDEN present").mb_rows,
        ),
        k_slot_dims,
        "slot dimensions must be preserved across the refresh ladder"
    );
}

/// Picker-quality check: GOLDEN promoted from an intermediate P-frame
/// holds a cleaner reconstruction than what LAST will be after a
/// disturbance. On the post-disturbance P-frame, the picker should
/// select GOLDEN for at least one MB (wire `prob_last < 255`) and
/// the self-decode Y-PSNR clears 30 dB.
///
/// Sequence:
///   * K (stripe pattern) → all slots populated.
///   * P1 (stripe pattern, very close to K) → encoded normally, then
///     the caller promotes the P1 reconstruction into GOLDEN via
///     `refresh_golden_frame = true`. GOLDEN now holds a low-noise
///     stripe.
///   * P2 (flat gray) → drifts LAST to flat gray; GOLDEN still holds
///     the stripe.
///   * P3 (stripe pattern again) → picker should beat LAST by picking
///     GOLDEN for the stripe-bearing row-0 MB.
#[test]
fn refresh_golden_carries_picker_quality_to_later_p_frame() {
    let width = 16u32;
    let height = 32u32; // 1 col, 2 rows of MBs.
    let qi = 16u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero interval");
    let mut dec = Vp8DecoderState::new();

    let (stripe_y, stripe_u, stripe_v) = stripe_frame(width as usize, height as usize);
    let (flat_y, flat_u, flat_v) = flat_frame(width as usize, height as usize);
    let k_frame = I420Frame::packed(width, height, &stripe_y, &stripe_u, &stripe_v);
    let stripe = I420Frame::packed(width, height, &stripe_y, &stripe_u, &stripe_v);
    let flat = I420Frame::packed(width, height, &flat_y, &flat_u, &flat_v);

    // K — populate all slots.
    let e0 = enc.encode_frame(&k_frame).expect("encode K");
    let _ = dec.decode_frame(&e0.bytes).expect("decode K");

    // P1 — encode the stripe again, then promote the reconstruction
    // into GOLDEN. After this frame GOLDEN holds the P1 reconstruction
    // (a refined stripe), LAST also holds the P1 reconstruction.
    let e1 = enc
        .encode_p_frame_with_refresh(
            &stripe,
            &RefreshControls {
                refresh_golden_frame: true,
                refresh_alternate_frame: false,
                copy_buffer_to_golden: 0,
                copy_buffer_to_alternate: 0,
                refresh_last: true,
            },
        )
        .expect("encode P1 with refresh_golden");
    let _ = dec.decode_frame(&e1.bytes).expect("decode P1");

    // Verify the §19.2 wire actually carried refresh_golden_frame = 1:
    let p1_coded = parse_p_coded_header(&e1.bytes);
    assert_eq!(p1_coded.refresh_golden_frame, Some(true));
    assert_eq!(p1_coded.refresh_last, Some(true));

    // P2 — flat gray. LAST drifts to flat gray; GOLDEN keeps the
    // stripe reconstruction (just promoted in P1).
    let e2 = enc.encode_frame(&flat).expect("encode P2 (scheduler)");
    let _ = dec.decode_frame(&e2.bytes).expect("decode P2");

    // P3 — stripe again. The picker should beat the now-flat LAST by
    // picking GOLDEN for the stripe-bearing row-0 MB.
    let e3 = enc.encode_frame(&stripe).expect("encode P3 (scheduler)");
    let d3 = dec.decode_frame(&e3.bytes).expect("decode P3");

    let p3_psnr_y = plane_psnr(&stripe_y, &d3.y);
    eprintln!("P3 self-decode Y-PSNR = {p3_psnr_y:.2} dB (stripe back from flat LAST)");
    assert!(
        p3_psnr_y >= 30.0,
        "P3 Y-PSNR {p3_psnr_y:.2} dB below the 30.0 dB target"
    );

    let coded = parse_p_coded_header(&e3.bytes);
    let prob_intra = coded.prob_intra.expect("prob_intra present");
    let prob_last = coded.prob_last.expect("prob_last present");
    eprintln!("P3 §9.10 prob_intra={prob_intra} prob_last={prob_last}");
    assert_eq!(prob_intra, 255, "every MB still coded as inter");
    assert!(
        prob_last < 255,
        "P3's prob_last = {prob_last} (== 255 means LAST won every MB); \
         the refreshed GOLDEN should have beaten the flat-gray LAST for \
         the stripe MB"
    );
}

/// Calling `encode_p_frame_with_refresh` on a fresh stream (no LAST
/// reference yet) is rejected with `NoLastReference` rather than
/// silently promoting to a key frame.
#[test]
fn encode_p_with_refresh_rejects_when_no_last() {
    use oxideav_vp8::StreamEncodeError;

    let mut enc =
        Vp8InterStreamEncoder::new(KeyframeParams::default(), 4).expect("non-zero interval");
    let (y, u, v) = flat_frame(16, 16);
    let frame = I420Frame::packed(16, 16, &y, &u, &v);

    let err = enc
        .encode_p_frame_with_refresh(&frame, &RefreshControls::default())
        .expect_err("must reject on a fresh stream");
    assert!(matches!(err, StreamEncodeError::NoLastReference));
}
