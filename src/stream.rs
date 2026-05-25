//! Multi-frame VP8 keyframe encoder driver
//! ([`Vp8KeyframeStreamEncoder`]).
//!
//! This module is the encoder-side counterpart of
//! [`crate::state::Vp8DecoderState`]: it owns the per-stream state a
//! sequence of VP8 frames needs (frame count, locked dimensions, the
//! §9 three-slot reference-frame buffer) and exposes a single
//! [`Vp8KeyframeStreamEncoder::encode_frame`] entry that turns one
//! input I420 picture into one VP8 keyframe's bytes while updating
//! that state.
//!
//! ## Scope
//!
//! Every emitted frame is a **key frame** — independently decodable, no
//! cross-frame prediction. The reference-frame slot machinery is wired
//! up because every key frame implicitly refreshes all three slots per
//! RFC 6386 §9.7 / §9.8 (the `if (key_frame) … hdr->refresh_last = 1`
//! body referenced in §19.2's header listing): a downstream decoder
//! observing the emitted stream sees its `LAST`, `GOLDEN`, and `ALTREF`
//! slots overwritten with the same picture after every frame.
//! Maintaining the slots in the encoder mirrors that, so the eventual
//! inter-encoder round drops in without further plumbing — it will
//! just need to (a) flip the `key_frame` bit, (b) consult the slots
//! for motion-compensation source pixels, and (c) honour the §9.7
//! `refresh_*` / `copy_buffer_to_*` rules to decide which slot(s) the
//! frame refreshes.
//!
//! Inter prediction itself, reference-frame selection (§16.2), per-MB
//! motion vectors (§17), and motion compensation (§18) are
//! intentionally out of scope for this round.
//!
//! ## Per-frame self-decode invariant
//!
//! Each emitted frame's bytes decode through the crate's own
//! [`crate::state::Vp8DecoderState`] driver to within the §14 quantiser
//! distortion of the input picture. The
//! `encoder_keyframe_stream.rs` integration test pins this on a
//! synthetic 5-frame sequence (different pattern per frame, mid
//! quantiser, whole-frame PSNR ≥ 30 dB).
//!
//! ## Reference-slot lifecycle
//!
//! Internally the driver reuses [`crate::encoder::encode_keyframe_with_reconstruction`]
//! so the *exact* post-§15 macroblock-aligned reconstruction the
//! decoder will rebuild from the emitted bytes is available without a
//! re-decode. After every successful `encode_frame` call the three
//! [`crate::state::RefFrameSlot`]s are atomically replaced with a clone
//! of that reconstruction, matching the
//! [`crate::state::Vp8DecoderState::decode_key_frame`] update logic
//! one-for-one.
//!
//! ## Reference
//!
//! * RFC 6386 §4 page 8 — "The first frame of a VP8 stream is always a
//!   key frame".
//! * RFC 6386 §9.1 / §19.2 — the `key_frame` bit + the keyframe-only
//!   `start_code` `0x9d 0x01 0x2a` in the frame tag.
//! * RFC 6386 §9.7 / §9.8 — `refresh_golden_frame`,
//!   `refresh_alternate_frame`, `refresh_last`. For a key frame all
//!   three are forced to 1 implicitly (the on-wire bits are
//!   key-frame-suppressed; the listing in §19.2 only emits them on
//!   `if (!key_frame)`).

use crate::encoder::{encode_keyframe_with_reconstruction, EncodeError, I420Frame, KeyframeParams};
use crate::state::RefFrameSlot;

/// Errors surfaced by [`Vp8KeyframeStreamEncoder::encode_frame`].
///
/// Wraps the per-frame [`EncodeError`] surface and adds the
/// cross-frame-only failures (a dimensions change between frames,
/// which the §9.1 keyframe-only-resize rule would technically allow
/// for *another* key frame but which most container clients treat as
/// a stream split — we reject it here to keep the contract explicit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEncodeError {
    /// A frame after the first was supplied with different dimensions
    /// than the first. The driver treats the dimensions as locked at
    /// the first `encode_frame` call.
    DimensionsChanged {
        /// `(width, height)` of the first frame fed in.
        first: (u32, u32),
        /// `(width, height)` of the offending later frame.
        got: (u32, u32),
    },
    /// The underlying per-frame encoder rejected the inputs (validator
    /// failure or §13 token failure). Carries the underlying
    /// [`EncodeError`].
    Frame(EncodeError),
}

impl core::fmt::Display for StreamEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamEncodeError::DimensionsChanged { first, got } => write!(
                f,
                "vp8 stream encode: frame dimensions {}x{} differ from \
                 stream's first frame {}x{} (dimensions are locked at \
                 the first encode_frame call)",
                got.0, got.1, first.0, first.1
            ),
            StreamEncodeError::Frame(e) => write!(f, "vp8 stream encode: {e}"),
        }
    }
}

impl std::error::Error for StreamEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamEncodeError::DimensionsChanged { .. } => None,
            StreamEncodeError::Frame(e) => Some(e),
        }
    }
}

impl From<EncodeError> for StreamEncodeError {
    fn from(e: EncodeError) -> Self {
        StreamEncodeError::Frame(e)
    }
}

/// Multi-frame VP8 keyframe encoder driver.
///
/// One instance owns the cross-frame state of a single VP8 elementary
/// stream: the frame counter, the locked-after-first-frame dimensions,
/// and the §9 three-slot reference-frame buffer
/// (`LAST` / `GOLDEN` / `ALTREF`).
///
/// All emitted frames are key frames this round. Each successful
/// [`Self::encode_frame`] call:
///
/// 1. Validates that this frame's dimensions match the first frame's
///    (or, on the first call, locks them in).
/// 2. Calls [`encode_keyframe_with_reconstruction`] with the driver's
///    [`KeyframeParams`].
/// 3. Replaces all three reference slots with a clone of the
///    macroblock-aligned post-§15 reconstruction (§9.7 / §9.8 keyframe
///    refresh).
/// 4. Increments the frame counter.
/// 5. Returns the emitted bytes.
///
/// Construct with [`Self::new`] and drive with [`Self::encode_frame`].
/// [`Self::frame_count`], [`Self::dimensions`], and the per-slot
/// accessors [`Self::last`], [`Self::golden`], [`Self::altref`]
/// expose the running state.
#[derive(Debug, Clone)]
pub struct Vp8KeyframeStreamEncoder {
    params: KeyframeParams,
    /// Visible dimensions of the first frame fed in. Locked after the
    /// first successful `encode_frame` call; `None` before then.
    dimensions: Option<(u32, u32)>,
    /// Number of frames successfully encoded so far.
    frame_count: u64,
    /// `LAST` reference slot. `None` before the first frame; populated
    /// (and replaced) on every successful `encode_frame` call.
    last: Option<RefFrameSlot>,
    /// `GOLDEN` reference slot. Same lifecycle as
    /// [`Self::last`] this round (every key frame refreshes all three).
    golden: Option<RefFrameSlot>,
    /// `ALTREF` reference slot. Same lifecycle as [`Self::last`].
    altref: Option<RefFrameSlot>,
}

impl Vp8KeyframeStreamEncoder {
    /// Build a fresh stream encoder configured with the supplied
    /// per-frame [`KeyframeParams`]. Same parameters are applied to
    /// every encoded frame in this stream.
    ///
    /// The frame counter starts at 0 and all three reference slots
    /// start as `None`. The dimensions are not locked until the first
    /// `encode_frame` call.
    pub fn new(params: KeyframeParams) -> Self {
        Vp8KeyframeStreamEncoder {
            params,
            dimensions: None,
            frame_count: 0,
            last: None,
            golden: None,
            altref: None,
        }
    }

    /// Encode one frame of the stream as a VP8 key frame.
    ///
    /// On the first call this also locks the stream's dimensions to
    /// `frame`'s visible width/height. Every subsequent call must
    /// supply a frame with the same dimensions; a mismatch is
    /// surfaced as [`StreamEncodeError::DimensionsChanged`] and leaves
    /// the encoder state unchanged.
    ///
    /// Returns the raw bytes of one VP8 elementary-stream frame
    /// (`one packet = one frame`) ready to be fed to a container muxer
    /// (e.g. IVF) or decoded directly through
    /// [`crate::decode_vp8`] / [`crate::state::Vp8DecoderState::decode_frame`].
    pub fn encode_frame(&mut self, frame: &I420Frame<'_>) -> Result<Vec<u8>, StreamEncodeError> {
        let dims = (frame.width, frame.height);
        match self.dimensions {
            Some(locked) if locked != dims => {
                return Err(StreamEncodeError::DimensionsChanged {
                    first: locked,
                    got: dims,
                });
            }
            _ => {}
        }

        let (bytes, planes) = encode_keyframe_with_reconstruction(frame, &self.params)?;

        // §9.7 / §9.8 keyframe slot refresh — all three slots take a
        // clone of the post-§15 macroblock-aligned reconstruction.
        // Mirrors `Vp8DecoderState::decode_key_frame`'s slot installation.
        let slot = RefFrameSlot::from_keyframe_planes(&planes);
        self.last = Some(slot.clone());
        self.golden = Some(slot.clone());
        self.altref = Some(slot);

        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(bytes)
    }

    /// Number of frames successfully encoded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Stream dimensions, locked at the first successful
    /// `encode_frame` call. `None` before then.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.dimensions
    }

    /// Borrow the current `LAST` reference slot. `None` before the
    /// first frame.
    pub fn last(&self) -> Option<&RefFrameSlot> {
        self.last.as_ref()
    }

    /// Borrow the current `GOLDEN` reference slot. `None` before the
    /// first frame.
    pub fn golden(&self) -> Option<&RefFrameSlot> {
        self.golden.as_ref()
    }

    /// Borrow the current `ALTREF` reference slot. `None` before the
    /// first frame.
    pub fn altref(&self) -> Option<&RefFrameSlot> {
        self.altref.as_ref()
    }

    /// Borrow the [`KeyframeParams`] applied to every frame this
    /// stream emits.
    pub fn params(&self) -> &KeyframeParams {
        &self.params
    }
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Vp8DecoderState;

    fn flat_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let w = width as usize;
        let h = height as usize;
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        (vec![y; w * h], vec![u; cw * ch], vec![v; cw * ch])
    }

    #[test]
    fn fresh_stream_state() {
        let enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
        assert_eq!(enc.frame_count(), 0);
        assert!(enc.dimensions().is_none());
        assert!(enc.last().is_none());
        assert!(enc.golden().is_none());
        assert!(enc.altref().is_none());
    }

    #[test]
    fn first_frame_locks_dimensions_and_populates_all_slots() {
        let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
        let (y, u, v) = flat_frame(32, 32, 128, 128, 128);
        let frame = I420Frame::packed(32, 32, &y, &u, &v);
        let bytes = enc.encode_frame(&frame).expect("encode first frame");
        assert!(!bytes.is_empty(), "frame bytes non-empty");
        assert_eq!(enc.frame_count(), 1);
        assert_eq!(enc.dimensions(), Some((32, 32)));
        assert!(enc.last().is_some());
        assert!(enc.golden().is_some());
        assert!(enc.altref().is_some());
        // §9.7 / §9.8: a keyframe refreshes all three slots with the
        // same reconstruction.
        let last = enc.last().unwrap();
        let golden = enc.golden().unwrap();
        let altref = enc.altref().unwrap();
        assert_eq!(last.y, golden.y);
        assert_eq!(last.y, altref.y);
        assert_eq!(last.u, golden.u);
        assert_eq!(last.v, altref.v);
    }

    #[test]
    fn dimensions_change_rejected() {
        let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
        let (y, u, v) = flat_frame(32, 32, 128, 128, 128);
        let frame = I420Frame::packed(32, 32, &y, &u, &v);
        enc.encode_frame(&frame).expect("first frame");
        let (y2, u2, v2) = flat_frame(48, 48, 64, 200, 50);
        let frame2 = I420Frame::packed(48, 48, &y2, &u2, &v2);
        let err = enc
            .encode_frame(&frame2)
            .expect_err("differently-sized second frame");
        assert!(matches!(
            err,
            StreamEncodeError::DimensionsChanged {
                first: (32, 32),
                got: (48, 48)
            }
        ));
        // Failure must not advance the frame counter.
        assert_eq!(enc.frame_count(), 1);
    }

    #[test]
    fn three_frame_stream_decodes_through_state_driver() {
        // Drive 3 trivial flat frames through the encoder, then replay
        // the bytes through `Vp8DecoderState::decode_frame` and
        // confirm each frame round-trips.
        let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
        let pixels = [
            (100u8, 110u8, 120u8),
            (50u8, 200u8, 30u8),
            (180u8, 80u8, 200u8),
        ];
        let mut frames_bytes = Vec::new();
        for (y, u, v) in &pixels {
            let (yp, up, vp) = flat_frame(32, 32, *y, *u, *v);
            let frame = I420Frame::packed(32, 32, &yp, &up, &vp);
            frames_bytes.push(enc.encode_frame(&frame).expect("encode"));
        }
        assert_eq!(enc.frame_count(), 3);

        let mut dec = Vp8DecoderState::new();
        for bytes in &frames_bytes {
            let out = dec.decode_frame(bytes).expect("decode");
            assert_eq!(out.width, 32);
            assert_eq!(out.height, 32);
        }
    }

    #[test]
    fn slot_state_replaced_each_frame() {
        // Each frame should overwrite the previous slot contents
        // (not append, not preserve). Encode two distinctly-colored
        // flat frames and confirm the slots reflect the *second*
        // frame's reconstruction.
        let mut enc = Vp8KeyframeStreamEncoder::new(KeyframeParams::default());
        let (y1, u1, v1) = flat_frame(16, 16, 50, 50, 50);
        let frame1 = I420Frame::packed(16, 16, &y1, &u1, &v1);
        enc.encode_frame(&frame1).expect("frame 1");
        let after_first = enc.last().unwrap().y.clone();

        let (y2, u2, v2) = flat_frame(16, 16, 200, 200, 200);
        let frame2 = I420Frame::packed(16, 16, &y2, &u2, &v2);
        enc.encode_frame(&frame2).expect("frame 2");
        let after_second = enc.last().unwrap().y.clone();

        assert_ne!(
            after_first, after_second,
            "slot must reflect the second frame, not the first"
        );
    }
}
