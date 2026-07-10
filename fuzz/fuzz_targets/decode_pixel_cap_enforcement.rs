#![no_main]

//! Fuzz: the §9.1 declared-frame-size DoS cap (`FrameTooLarge`).
//!
//! Round-405 added a caller-configurable `max_pixels_per_frame` cap to
//! the stateless [`decode_vp8_with_max_pixels`] entry point and the
//! stateful [`Vp8DecoderState::with_max_pixels_per_frame`] builder: a
//! key frame whose declared `width × height` exceeds the cap is refused
//! with [`DecodeError::FrameTooLarge`] *before* the decoder reserves its
//! macroblock grid, so a tiny header declaring an enormous frame cannot
//! drive a large allocation. Every other target either uses the fixed
//! `accept_dimensions` gate (never exercising the cap surface) or the
//! default cap (which by construction never fires on a legal 14-bit VP8
//! frame). This target drives the cap boundary directly on attacker
//! bytes.
//!
//! Two legs, split on the first input byte so both the reject and the
//! accept branch stay OOM-safe:
//!
//!  * **Reject leg** — set the cap one pixel *below* the declared size.
//!    The cap must fire and no allocation may happen, so this is safe
//!    even for the wire-legal `16_383 × 16_383 ≈ 268 Mpx` extreme the
//!    other targets have to gate out. Oracle: the result is exactly
//!    `Err(FrameTooLarge { pixels: declared, cap: declared - 1 })`.
//!  * **Accept leg** — only for frames inside the fuzz runner's alloc
//!    budget (`accept_dimensions`). Set the cap at the declared size so
//!    the cap never fires; any `Ok` decode must report visible
//!    dimensions whose product is within the cap. Cross-drives the
//!    stateful path with the same cap for panic-freedom.
//!
//! Errors other than a panic are fine — the reject leg asserts a
//! *specific* error, the accept leg tolerates any `Err` (truncation,
//! malformed header, …) and only constrains the `Ok` shape.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::state::Vp8DecoderState;
use oxideav_vp8::{decode_vp8_with_max_pixels, frame_header::Vp8FrameHeader, DecodeError};
use oxideav_vp8_fuzz::accept_dimensions;

fuzz_target!(|data: &[u8]| {
    // Need at least one selector byte + a parseable header.
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    let Ok(header) = Vp8FrameHeader::parse(rest) else {
        return;
    };
    // Only key frames carry §9.1 width / height; inter frames (None)
    // never reach the cap check.
    let (Some(w), Some(h)) = (header.width, header.height) else {
        return;
    };
    let (w, h) = (u32::from(w), u32::from(h));
    if w == 0 || h == 0 {
        return;
    }
    let declared = u64::from(w) * u64::from(h);

    if sel & 1 == 0 {
        // Reject leg: cap strictly below the declared size. No plane is
        // ever allocated, so a 268-Mpx declared frame is safe here.
        let cap = declared - 1; // declared >= 1 (w, h >= 1)
        match decode_vp8_with_max_pixels(rest, cap) {
            Err(DecodeError::FrameTooLarge { pixels, cap: got_cap }) => {
                assert_eq!(pixels, declared, "reported pixels must equal declared");
                assert_eq!(got_cap, cap, "reported cap must equal configured cap");
            }
            other => panic!("cap below declared must yield FrameTooLarge, got {other:?}"),
        }
    } else {
        // Accept leg: only for frames inside the fuzz alloc budget.
        if !accept_dimensions(w, h) {
            return;
        }
        let cap = declared; // inclusive — the cap never fires here
        if let Ok(frame) = decode_vp8_with_max_pixels(rest, cap) {
            let out = u64::from(frame.width) * u64::from(frame.height);
            assert!(out <= cap, "decoded {out} px must stay within cap {cap}");
        }
        // The stateful builder must honour the same cap without panicking.
        let mut st = Vp8DecoderState::new().with_max_pixels_per_frame(cap);
        let _ = st.decode_frame(rest);
    }
});
