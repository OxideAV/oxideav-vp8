#![no_main]

//! Fuzz: the ARNR temporal-filter altref synthesiser
//! (`arnr::build_arnr_altref` / `ArnrConfig` / `ArnrPicture`).
//!
//! The only fuzz coverage the ARNR layer had before this target was
//! indirect: `altref_stream_encode_decode` drives it through
//! `Vp8AltrefStreamEncoder::push_frame`, which always hands it
//! tightly-packed frames of identical, pre-validated dimensions with a
//! stream-chosen `center`. The direct public surface is wider, and the
//! pixel loops inside `build_arnr_altref` do strided plane arithmetic
//! (`pel`'s clamp-then-index, `copy_plane`'s `r * stride + w` walk, the
//! halved-chroma `(cbx + c).min(cw - 1)` folds) that only mis-shaped
//! direct inputs can stress:
//!
//! * **Strided (non-packed) planes.** Every plane gets an independent
//!   attacker-chosen stride ≥ the packed minimum, with the plane buffer
//!   sized exactly `stride * rows` — one row of over-read past a
//!   mis-multiplied index is an ASan hit.
//! * **Odd dimensions.** Widths / heights over the full 1..=48 range so
//!   the `div_ceil(2)` chroma geometry and the partial trailing 16×16
//!   block (`bw`/`bh` < 16) are the common case, not the exception.
//! * **Raw `strength` bytes.** `ArnrConfig::new` clamps 0..=255 into
//!   0..=`MAX_STRENGTH`; the config used is asserted to stay inside the
//!   documented envelope.
//! * **The rejection legs.** A window member with mismatched dimensions
//!   must surface `EncodeError::ReferenceDimensionsMismatch`; a
//!   zero-dimension center must surface `EncodeError::InvalidDimensions`
//!   — both without touching a single plane byte.
//!
//! Oracles beyond panic-freedom:
//!
//! 1. **Geometry** — the output picture is tightly packed at the
//!    documented sizes (`w×h` luma, `ceil(w/2)×ceil(h/2)` chroma).
//! 2. **Identity** — `strength == 0` or a single-frame window must
//!    reproduce the center frame's visible pixels exactly (the
//!    documented pass-through contract).
//! 3. **Fixed point** — when every window frame carries pixel content
//!    identical to the center's, the difference-driven blend degenerates
//!    to a weighted mean of equal values: the output must equal the
//!    center exactly, at every strength.
//! 4. **Re-entry** — `ArnrPicture::as_i420()` must itself be a valid
//!    single-frame window: re-running the filter on it at strength 0
//!    reproduces it byte-for-byte.
//!
//! Caps: dimensions ≤ 48×48 luma, window ≤ 4 frames, stride pad ≤ 8 —
//! the whole-pel refinement search is O(blocks × frames × candidates)
//! and stays well under a millisecond per iteration at these bounds.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::arnr::{build_arnr_altref, ArnrConfig, ArnrPicture};
use oxideav_vp8::{EncodeError, I420Frame};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 8;
const MAX_FRAMES: usize = 4;
const MAX_DIM: u32 = 48;

/// One owned strided plane set, built from the payload byte pool.
struct OwnedFrame {
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl OwnedFrame {
    /// Materialise a `width × height` frame whose luma/chroma strides
    /// carry independent pads, filling pixels from `pool` (tiled).
    fn synth(width: u32, height: u32, y_pad: usize, uv_pad: usize, pool: &[u8], salt: u8) -> Self {
        let w = width as usize;
        let h = height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let y_stride = w + y_pad;
        let uv_stride = cw + uv_pad;
        let fill = |len: usize, tweak: u8| -> Vec<u8> {
            (0..len)
                .map(|i| {
                    let base = if pool.is_empty() {
                        i as u8
                    } else {
                        pool[i % pool.len()]
                    };
                    base ^ tweak
                })
                .collect()
        };
        OwnedFrame {
            width,
            height,
            y_stride,
            uv_stride,
            y: fill(y_stride * h, salt),
            u: fill(uv_stride * ch, salt ^ 0x55),
            v: fill(uv_stride * ch, salt ^ 0xaa),
        }
    }

    fn as_i420(&self) -> I420Frame<'_> {
        I420Frame {
            width: self.width,
            height: self.height,
            y: &self.y,
            u: &self.u,
            v: &self.v,
            y_stride: self.y_stride,
            uv_stride: self.uv_stride,
        }
    }

    /// Tightly-packed copy of the visible pixels (the identity-oracle
    /// reference).
    fn packed_planes(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let w = self.width as usize;
        let h = self.height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let pack = |plane: &[u8], stride: usize, pw: usize, ph: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(pw * ph);
            for r in 0..ph {
                out.extend_from_slice(&plane[r * stride..r * stride + pw]);
            }
            out
        };
        (
            pack(&self.y, self.y_stride, w, h),
            pack(&self.u, self.uv_stride, cw, ch),
            pack(&self.v, self.uv_stride, cw, ch),
        )
    }
}

/// Assert the documented tightly-packed output geometry.
fn assert_geometry(pic: &ArnrPicture, width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    assert_eq!(pic.width, width, "output width drift");
    assert_eq!(pic.height, height, "output height drift");
    assert_eq!(pic.y.len(), w * h, "luma plane length drift");
    assert_eq!(pic.u.len(), cw * ch, "U plane length drift");
    assert_eq!(pic.v.len(), cw * ch, "V plane length drift");
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = 1 + u32::from(data[0]) % MAX_DIM;
    let height = 1 + u32::from(data[1]) % MAX_DIM;
    let frame_count = 1 + usize::from(data[2]) % MAX_FRAMES;
    let center = usize::from(data[3]) % frame_count;
    // Raw byte: exercises the ArnrConfig::new clamp on 0..=255.
    let cfg = ArnrConfig::new(data[4]);
    assert!(
        cfg.strength <= ArnrConfig::MAX_STRENGTH,
        "ArnrConfig::new must clamp into the documented envelope"
    );
    let y_pad = usize::from(data[5]) % 9;
    let uv_pad = usize::from(data[6]) % 9;
    let flags = data[7];
    let pool = &data[HEADER_BYTES..];

    // Fixed-point leg (flag bit 1): every frame carries the center's
    // pixel content (salt held constant), so the blend must be exact.
    let fixed_point = (flags & 0x02) != 0;

    let frames_owned: Vec<OwnedFrame> = (0..frame_count)
        .map(|i| {
            let salt = if fixed_point { 0x11 } else { 0x11 ^ (i as u8) };
            OwnedFrame::synth(width, height, y_pad, uv_pad, pool, salt)
        })
        .collect();
    let frames: Vec<I420Frame<'_>> = frames_owned.iter().map(|f| f.as_i420()).collect();

    let pic = match build_arnr_altref(&frames, center, &cfg) {
        Ok(p) => p,
        Err(e) => panic!(
            "build_arnr_altref rejected a well-formed window: {e:?} \
             ({width}x{height}, {frame_count} frames, center {center}, cfg {cfg:?})"
        ),
    };
    assert_geometry(&pic, width, height);

    let (cy, cu, cv) = frames_owned[center].packed_planes();
    if cfg.strength == 0 || frame_count == 1 {
        assert_eq!(
            pic.y, cy,
            "strength-0 / single-frame luma must pass through"
        );
        assert_eq!(pic.u, cu, "strength-0 / single-frame U must pass through");
        assert_eq!(pic.v, cv, "strength-0 / single-frame V must pass through");
    } else if fixed_point {
        assert_eq!(
            pic.y, cy,
            "identical-window blend must be a luma fixed point"
        );
        assert_eq!(pic.u, cu, "identical-window blend must be a U fixed point");
        assert_eq!(pic.v, cv, "identical-window blend must be a V fixed point");
    }

    // Re-entry: the produced picture is itself a valid single-frame
    // window; strength 0 must reproduce it byte-for-byte.
    let reborrowed = [pic.as_i420()];
    let echo = build_arnr_altref(&reborrowed, 0, &ArnrConfig::new(0))
        .expect("as_i420 output must be accepted as a window member");
    assert_eq!(echo.y, pic.y, "as_i420 re-entry luma drift");
    assert_eq!(echo.u, pic.u, "as_i420 re-entry U drift");
    assert_eq!(echo.v, pic.v, "as_i420 re-entry V drift");

    // Rejection leg 1: a mismatched window member (needs >= 2 frames so
    // the mismatch lands off-center).
    if frame_count >= 2 && (flags & 0x01) != 0 {
        let other_w = if width == MAX_DIM {
            width - 1
        } else {
            width + 1
        };
        let odd = OwnedFrame::synth(other_w, height, y_pad, uv_pad, pool, 0x77);
        let mut window: Vec<I420Frame<'_>> = frames_owned.iter().map(|f| f.as_i420()).collect();
        let victim = if center == 0 { 1 } else { 0 };
        window[victim] = odd.as_i420();
        match build_arnr_altref(&window, center, &cfg) {
            Err(EncodeError::ReferenceDimensionsMismatch { .. }) => {}
            other => panic!("mismatched window member must be rejected, got {other:?}"),
        }
    }

    // Rejection leg 2: a zero-dimension center must fail cleanly before
    // any plane access.
    if (flags & 0x04) != 0 {
        let empty: [u8; 0] = [];
        let degenerate = I420Frame {
            width: 0,
            height,
            y: &empty,
            u: &empty,
            v: &empty,
            y_stride: 0,
            uv_stride: 0,
        };
        match build_arnr_altref(&[degenerate], 0, &cfg) {
            Err(EncodeError::InvalidDimensions { .. }) => {}
            other => panic!("zero-width center must be rejected, got {other:?}"),
        }
    }
});
