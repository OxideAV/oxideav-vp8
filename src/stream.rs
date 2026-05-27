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

use crate::coded_header::TokenProbUpdates;
use crate::encoder::{
    encode_keyframe_with_reconstruction,
    encode_keyframe_with_reconstruction_and_fitted_token_prob_updates, encode_p_frame_multi_ref,
    encode_p_frame_multi_ref_with_fitted_token_prob_updates,
    encode_p_frame_multi_ref_with_intra_pick, encode_p_frame_multi_ref_with_refresh_and_intra_pick,
    EncodeError, I420Frame, KeyframeParams, LoopFilterDeltas, RefreshControls,
};
use crate::frame::KeyframePlanes;
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
    /// [`Vp8InterStreamEncoder::encode_p_frame_with_refresh`] was called
    /// before the stream had emitted its first frame, so no `LAST`
    /// reference is available. The caller should drive at least one
    /// frame through the scheduler first.
    NoLastReference,
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
            StreamEncodeError::NoLastReference => write!(
                f,
                "vp8 stream encode: encode_p_frame_with_refresh called before \
                 the stream emitted its first frame (LAST reference slot is empty)"
            ),
        }
    }
}

impl std::error::Error for StreamEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamEncodeError::DimensionsChanged { .. } => None,
            StreamEncodeError::Frame(e) => Some(e),
            StreamEncodeError::NoLastReference => None,
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

    /// Encode one frame of the stream as a VP8 key frame, with an
    /// automatically-fitted §13.4 `token_prob_update()` payload per
    /// [`crate::encoder::encode_keyframe_with_reconstruction_and_fitted_token_prob_updates`].
    ///
    /// Drop-in fitted companion to [`Self::encode_frame`]: same K/P
    /// scheduling rules (every emitted frame is a key frame on this
    /// driver), same dimension-lock semantics, same §9.7 / §9.8
    /// three-slot refresh — only the bitstream differs (the fitter is
    /// allowed to *shrink* the wire by overlaying observed-counts
    /// probabilities on the §13.5 defaults, with the round-157 fitter's
    /// safety guard that never *grows* the wire relative to the
    /// caller-driven `None` baseline).
    ///
    /// The returned bytes always decode through
    /// [`crate::state::Vp8DecoderState::decode_frame`] and every
    /// compliant VP8 decoder; the §9 reference-frame slots are refreshed
    /// with the matching reconstruction (the round-157 safety-guard
    /// fall-back returns the **default-pass** planes alongside the
    /// default-pass bytes, so the slot state stays consistent with the
    /// wire on both fitter outcomes).
    ///
    /// Closes the round-157 / round-158 follow-up identified in
    /// [`crate::encoder::encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`]
    /// ("Out of round-158 scope: threading the fitter into
    /// `Vp8KeyframeStreamEncoder` / `Vp8InterStreamEncoder`").
    pub fn encode_frame_with_fitted_token_prob_updates(
        &mut self,
        frame: &I420Frame<'_>,
    ) -> Result<Vec<u8>, StreamEncodeError> {
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

        let (bytes, planes) =
            encode_keyframe_with_reconstruction_and_fitted_token_prob_updates(frame, &self.params)?;

        // §9.7 / §9.8 keyframe slot refresh — identical to
        // `Self::encode_frame`: a key frame refreshes all three slots
        // with the post-§15 reconstruction. The fitter's matching-planes
        // guarantee means `planes` is the reconstruction that matches
        // the emitted `bytes` regardless of which pass (default or
        // fitted) won the safety-guard comparison.
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

// ───────────────── Multi-frame I + P stream driver (Phase 10) ────────────────
//
// `Vp8InterStreamEncoder` extends the keyframe driver by interleaving
// ZERO_MV P-frames between key frames per a caller-specified keyframe
// interval (or a per-frame force-keyframe flag). It uses
// `encode_keyframe_with_reconstruction` for the K-frame path and
// `encode_p_frame_zero_mv` for the P-frame path, and maintains the §9
// three-slot reference-frame ladder one-for-one with
// `Vp8DecoderState::decode_frame`:
//
//   * A keyframe replaces all three slots with the post-§15
//     reconstruction (§9.7 / §9.8 — every refresh / copy bit is forced
//     to 1 for a key frame).
//   * A ZERO_MV P-frame replaces LAST only (§9.7 — the P-frame encoder
//     emits `refresh_last = 1`, all other refresh / copy bits 0).
//
// Reference: RFC 6386 §9 (frame-header layout for K vs P), §9.7
// (refresh ladder), §9.8 (`refresh_last` interpretation), §16
// (interframe header), §17 (motion-vector layer), §18 (motion
// compensation — ZERO_MV path is the §18 identity copy).

/// Per-frame keyframe scheduling decision for
/// [`Vp8InterStreamEncoder`].
///
/// Returned by [`Vp8InterStreamEncoder::next_frame_is_keyframe`] and by
/// the [`EncodedStreamFrame::is_keyframe`] field of every encoded
/// output; the caller can use it to confirm the K/P interleave the
/// encoder picked agreed with its expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// VP8 key frame — independently decodable, refreshes all three
    /// reference slots (§9.7 / §9.8).
    Key,
    /// VP8 inter frame coded as ZERO_MV against LAST — predicts from
    /// the LAST reference at MV (0, 0) per §16.2 / §18, refreshes LAST
    /// only (§9.7 inter refresh ladder).
    InterZeroMv,
}

/// One frame emitted by [`Vp8InterStreamEncoder::encode_frame`].
///
/// Bundles the raw VP8 elementary-stream bytes with the K-vs-P
/// classification and the running frame index, so the caller can sink
/// the bytes into a container while logging or asserting the
/// scheduling pattern.
#[derive(Debug, Clone)]
pub struct EncodedStreamFrame {
    /// Raw VP8 elementary-stream bytes for this frame (one packet =
    /// one frame). Suitable for IVF muxing or for
    /// [`crate::state::Vp8DecoderState::decode_frame`].
    pub bytes: Vec<u8>,
    /// Whether the encoder coded this frame as a key frame or as a
    /// ZERO_MV P-frame.
    pub kind: FrameKind,
    /// 0-based frame index inside this stream — equal to the encoder's
    /// `frame_count()` at the moment this frame was emitted minus one.
    pub frame_index: u64,
}

impl EncodedStreamFrame {
    /// Convenience: `true` iff this frame was coded as a key frame.
    pub fn is_keyframe(&self) -> bool {
        matches!(self.kind, FrameKind::Key)
    }
}

/// Multi-frame VP8 I + P stream encoder.
///
/// One instance drives a sequence of source I420 pictures into a VP8
/// elementary stream of alternating key frames and ZERO_MV P-frames,
/// owning every piece of cross-frame state the §9 / §16 layers need:
///
/// * the frame counter,
/// * the locked-after-first-frame visible dimensions,
/// * the §9 three-slot reference-frame buffer
///   (`LAST` / `GOLDEN` / `ALTREF`),
/// * the keyframe scheduling parameters (interval + per-call override).
///
/// ## Scheduling
///
/// The encoder picks K or P per frame using two inputs:
///
/// 1. A non-zero `keyframe_interval` configured at construction —
///    every `keyframe_interval`th frame (0, K, 2K, 3K, …) is a key
///    frame, the rest are ZERO_MV P-frames.
/// 2. A per-call `force_keyframe` flag on
///    [`Self::encode_frame_with_force`] — set to `true`, the next
///    emitted frame is a key frame regardless of the interval, and
///    the interval re-anchors to the forced index.
///
/// The first frame fed into a fresh encoder is **always** a key
/// frame — there is no prior reference to predict from, matching
/// [`crate::state::Vp8DecoderState`]'s rejection of an interframe with
/// no prior key frame.
///
/// ## Reference-slot lifecycle
///
/// After every successful `encode_frame` call the three reference
/// slots are updated to match what
/// [`crate::state::Vp8DecoderState::decode_frame`] would do for the
/// just-emitted frame:
///
/// * **Key frame** (§9.7 / §9.8) — all three slots are replaced with
///   a clone of the post-§15 reconstruction.
/// * **ZERO_MV P-frame** (§9.7) — `refresh_last = 1` only, so the LAST
///   slot is replaced with the post-§15 reconstruction; GOLDEN and
///   ALTREF are left untouched.
///
/// This mirrors the underlying [`encode_p_frame_zero_mv`] refresh
/// ladder one-for-one.
///
/// ## Per-frame self-decode invariant
///
/// Every emitted frame's bytes are decodable through
/// [`crate::state::Vp8DecoderState::decode_frame`], and a sequence of
/// frames replays into the same per-frame pictures the encoder was
/// fed (to within the §14 quantiser distortion). The
/// `encoder_inter_stream.rs` integration test pins this on a synthetic
/// 10-frame I420 sequence at keyframe interval 4, requiring per-frame
/// PSNR ≥ 30 dB at a mid quantiser.
///
/// ## Scope (this round)
///
/// * Every P-frame is ZERO_MV from LAST — no motion search, no
///   NEARESTMV / NEARMV / NEWMV / SPLITMV, no GOLDEN / ALTREF
///   selection. Quality on natural content beyond slow translation is
///   bounded by the §14 quantiser absorbing the residual.
/// * The §9.5 partition count is whatever the caller put in
///   [`KeyframeParams::nbr_of_dct_partitions`] for K-frames; P-frames
///   stay single-partition this round (the underlying
///   [`encode_p_frame_zero_mv`] hardwires 1).
/// * Mid-stream resize is rejected with
///   [`StreamEncodeError::DimensionsChanged`].
#[derive(Debug, Clone)]
pub struct Vp8InterStreamEncoder {
    params: KeyframeParams,
    keyframe_interval: u64,
    dimensions: Option<(u32, u32)>,
    frame_count: u64,
    /// Frame index that received the most recent key frame. Used to
    /// re-anchor the interval after a forced keyframe so the next
    /// K-frame is `keyframe_interval` frames later, not always at a
    /// multiple of `keyframe_interval` from the absolute start.
    last_keyframe_index: Option<u64>,
    last: Option<RefFrameSlot>,
    golden: Option<RefFrameSlot>,
    altref: Option<RefFrameSlot>,
    /// Across-frame §9.4 `ref_frame_delta[]` state in the §20.6
    /// `{CURRENT, LAST, GOLDEN, ALTREF}` order. Threaded into every
    /// P-frame's [`oxideav_vp8::LoopFilterDeltas::effective`] call so
    /// the encoder's §15 post-walk filter matches what the decoder
    /// derives from the wire. Reset to `[0; 4]` on every key frame
    /// (RFC 6386 §9.4 — key frames begin a fresh delta sequence).
    carried_ref_deltas: [i16; 4],
    /// Across-frame §9.4 `mode_delta[]` state in the §20.6 `{B_PRED,
    /// ZERO_MV, OTHER_MV, SPLIT_MV}` order. Same lifecycle as
    /// `carried_ref_deltas`.
    carried_mode_deltas: [i16; 4],
}

impl Vp8InterStreamEncoder {
    /// Build a fresh I + P stream encoder.
    ///
    /// `keyframe_interval` controls automatic keyframe scheduling:
    ///
    /// * `1` — every frame is a key frame (degenerates to the
    ///   keyframe-only driver).
    /// * `N > 1` — frames 0, N, 2N, 3N, … are key frames; frames in
    ///   between are ZERO_MV P-frames.
    /// * `0` — rejected; the keyframe interval must be at least 1 or
    ///   the encoder would never emit a key frame.
    ///
    /// Returns `None` for `keyframe_interval == 0` to make the bad
    /// configuration impossible-by-construction at the call site.
    pub fn new(params: KeyframeParams, keyframe_interval: u64) -> Option<Self> {
        if keyframe_interval == 0 {
            return None;
        }
        Some(Vp8InterStreamEncoder {
            params,
            keyframe_interval,
            dimensions: None,
            frame_count: 0,
            last_keyframe_index: None,
            last: None,
            golden: None,
            altref: None,
            carried_ref_deltas: [0; 4],
            carried_mode_deltas: [0; 4],
        })
    }

    /// Decide whether the next [`Self::encode_frame`] call would emit
    /// a key frame, given the current frame index and the last forced
    /// keyframe anchor — without actually encoding anything.
    ///
    /// Useful for a caller that wants to pre-roll metadata (IVF
    /// keyframe markers, container random-access entries) before
    /// handing the frame in. The decision is:
    ///
    /// * `FrameKind::Key` if no frame has been encoded yet (first
    ///   frame must be a key frame), or if `frame_count -
    ///   last_keyframe_index >= keyframe_interval`.
    /// * `FrameKind::InterZeroMv` otherwise.
    pub fn next_frame_is_keyframe(&self) -> FrameKind {
        match self.last_keyframe_index {
            None => FrameKind::Key,
            Some(anchor) => {
                if self
                    .frame_count
                    .saturating_sub(anchor)
                    .ge(&self.keyframe_interval)
                {
                    FrameKind::Key
                } else {
                    FrameKind::InterZeroMv
                }
            }
        }
    }

    /// Encode one frame of the stream using the configured keyframe
    /// interval to pick K vs P.
    ///
    /// On the first call this also locks the stream's dimensions to
    /// `frame`'s visible width/height. Every subsequent call must
    /// supply a frame with the same dimensions; a mismatch is
    /// surfaced as [`StreamEncodeError::DimensionsChanged`] and leaves
    /// the encoder state unchanged.
    ///
    /// Returns the emitted bytes wrapped in an
    /// [`EncodedStreamFrame`] so the caller knows which kind of frame
    /// it received without re-parsing the §9.1 tag bit.
    pub fn encode_frame(
        &mut self,
        frame: &I420Frame<'_>,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
        self.encode_frame_with_force(frame, false)
    }

    /// Encode one frame of the stream, with an optional override that
    /// forces it to be coded as a key frame regardless of the
    /// configured interval.
    ///
    /// A `force_keyframe = true` call **also re-anchors the interval**:
    /// after the forced K-frame at frame index `i`, the next automatic
    /// keyframe lands at `i + keyframe_interval`, not at the original
    /// multiple of `keyframe_interval` from the absolute start. This
    /// keeps the inter-keyframe spacing predictable in the presence of
    /// scene-cut overrides.
    pub fn encode_frame_with_force(
        &mut self,
        frame: &I420Frame<'_>,
        force_keyframe: bool,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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

        let scheduled = self.next_frame_is_keyframe();
        let must_be_key = scheduled == FrameKind::Key || force_keyframe;

        // We can only emit a P-frame if we actually hold a LAST
        // reference; otherwise we must promote to a key frame even
        // though the scheduler said "P". (Belt-and-suspenders: the
        // scheduler already returns Key for the first frame, so this
        // arm is only ever taken if a future variant of the API lets
        // the caller drop the slots externally.)
        let emit_key = must_be_key || self.last.is_none();

        let (bytes, planes, kind) = if emit_key {
            let (b, p) = encode_keyframe_with_reconstruction(frame, &self.params)?;
            (b, p, FrameKind::Key)
        } else {
            // Safe: `emit_key` is false ⇒ `self.last` is Some by the
            // expression above. We pass all three §9 reference slots
            // (`LAST` required, `GOLDEN` / `ALTREF` optional) to the
            // multi-ref encoder so it can score each MB against every
            // available reference and emit the §16.2 `ref_frame_tree`
            // selector bits per MB. Until a future round wires up the
            // §9.7 `refresh_golden_frame` / `refresh_alternate_frame`
            // ladder, GOLDEN / ALTREF stay frozen at the most-recent
            // keyframe's reconstruction — they still beat LAST for
            // MBs whose source content matches the keyframe (e.g.
            // after a brief disturbance returns to the original
            // frame).
            let last_planes = ref_slot_to_keyframe_planes(
                self.last.as_ref().expect("LAST slot present for P-frame"),
            );
            let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
            let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
            let (b, p) = encode_p_frame_multi_ref(
                frame,
                &last_planes,
                golden_planes.as_ref(),
                altref_planes.as_ref(),
                &self.params,
            )?;
            (b, p, FrameKind::InterZeroMv)
        };

        let frame_index = self.frame_count;

        // ---- Reference-slot update (§9.7) -----------------------------
        let new_slot = RefFrameSlot::from_keyframe_planes(&planes);
        match kind {
            FrameKind::Key => {
                // §9.7 / §9.8 keyframe — every slot is refreshed with
                // the same reconstruction.
                self.last = Some(new_slot.clone());
                self.golden = Some(new_slot.clone());
                self.altref = Some(new_slot);
                self.last_keyframe_index = Some(frame_index);
                // §9.4 key frames begin a fresh delta sequence — clear
                // the carried state so the next P-frame's effective
                // deltas start from 0 (matching the decoder's behaviour
                // after a key frame).
                self.carried_ref_deltas = [0; 4];
                self.carried_mode_deltas = [0; 4];
            }
            FrameKind::InterZeroMv => {
                // §9.7 inter refresh ladder used by encode_p_frame_zero_mv:
                // refresh_last = 1, refresh_golden_frame = 0,
                // refresh_alternate_frame = 0, copy_buffer_to_* = 0.
                // Only LAST changes. The §9.4 carried-delta state stays
                // unchanged (the scheduler-driven path emits
                // `loop_filter_adj_enable = 0`, so this frame's effective
                // deltas are all 0 and we have no fresh values to carry).
                self.last = Some(new_slot);
            }
        }

        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind,
            frame_index,
        })
    }

    /// Encode one frame using the configured keyframe interval to pick
    /// K vs P, with the round-160 / round-161 §11 intra-within-inter
    /// MB picker engaged on every P-frame.
    ///
    /// Drop-in `intra-pick` companion to [`Self::encode_frame`]: the
    /// scheduling decision (K vs P) and the §9.7 reference-slot ladder
    /// are identical to the non-intra-pick path; only the per-MB picker
    /// changes. On every emitted P-frame the per-MB picker scores the
    /// full §11.2 × §11.4 whole-block intra grid (4 luma × 4 chroma =
    /// 16 candidates, `B_PRED` excluded) against the running in-frame
    /// neighbours in addition to the §16 inter ladder, and picks
    /// whichever of (best inter pick, J-best intra) wins on
    /// `J + lambda * is_inter_mb-bit`. When the intra candidate wins
    /// on at least one MB the §9.10 `prob_intra` byte drops below 255
    /// and the §16.1 intra-mode-tree path emits on those MBs.
    ///
    /// K-frames go through the same
    /// [`encode_keyframe_with_reconstruction`] path as
    /// [`Self::encode_frame`] — the intra-pick flag only affects the
    /// inter (P-frame) arm. The §9.4 carried-delta state, the §9.7
    /// reference-slot rotation, and the dimensions-lock semantics are
    /// unchanged.
    ///
    /// Wire compatibility: on any source where the round-161 picker
    /// never selects an intra MB (e.g. a slow-translation flat
    /// gradient where ZERO_MV already absorbs the residual), the
    /// emitted bytes match [`Self::encode_frame`] modulo the
    /// `prob_intra` byte's drop from 255 to 1 (≈ 6 extra bits ≈ 1
    /// extra byte per P-frame). This is the same bound the bare
    /// `encode_p_frame_multi_ref_with_intra_pick` entry-point already
    /// pins under `tests/encoder_pframe_intra_pick.rs`.
    pub fn encode_frame_with_intra_pick(
        &mut self,
        frame: &I420Frame<'_>,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
        self.encode_frame_with_force_and_intra_pick(frame, false)
    }

    /// Encode one frame with the round-160 / round-161 intra-pick
    /// engaged **and** an optional `force_keyframe` override that
    /// re-anchors the keyframe interval, exactly mirroring
    /// [`Self::encode_frame_with_force`].
    ///
    /// Companion to [`Self::encode_frame_with_intra_pick`]: the
    /// `force_keyframe` semantics (re-anchoring the interval after a
    /// forced K-frame at frame index `i` so the next automatic K-frame
    /// lands at `i + keyframe_interval`) match
    /// [`Self::encode_frame_with_force`] exactly. On the K-frame arm
    /// the intra-pick flag is a no-op — every MB in a key frame is
    /// intra by construction.
    pub fn encode_frame_with_force_and_intra_pick(
        &mut self,
        frame: &I420Frame<'_>,
        force_keyframe: bool,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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

        let scheduled = self.next_frame_is_keyframe();
        let must_be_key = scheduled == FrameKind::Key || force_keyframe;
        let emit_key = must_be_key || self.last.is_none();

        let (bytes, planes, kind) = if emit_key {
            let (b, p) = encode_keyframe_with_reconstruction(frame, &self.params)?;
            (b, p, FrameKind::Key)
        } else {
            // Inter arm — same LAST/GOLDEN/ALTREF plane harvesting as
            // `encode_frame_with_force`, but the per-MB picker is the
            // round-160 / round-161 intra-within-inter widened
            // candidate set.
            let last_planes = ref_slot_to_keyframe_planes(
                self.last.as_ref().expect("LAST slot present for P-frame"),
            );
            let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
            let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
            let (b, p) = encode_p_frame_multi_ref_with_intra_pick(
                frame,
                &last_planes,
                golden_planes.as_ref(),
                altref_planes.as_ref(),
                &self.params,
            )?;
            (b, p, FrameKind::InterZeroMv)
        };

        let frame_index = self.frame_count;

        // ---- Reference-slot update (§9.7) — identical to
        // `encode_frame_with_force`. The intra-pick flag does NOT
        // affect the §9.7 ladder; `planes` is the reconstruction that
        // matches `bytes` regardless of which per-MB pick won.
        let new_slot = RefFrameSlot::from_keyframe_planes(&planes);
        match kind {
            FrameKind::Key => {
                self.last = Some(new_slot.clone());
                self.golden = Some(new_slot.clone());
                self.altref = Some(new_slot);
                self.last_keyframe_index = Some(frame_index);
                // §9.4 — key frames start a fresh delta sequence.
                self.carried_ref_deltas = [0; 4];
                self.carried_mode_deltas = [0; 4];
            }
            FrameKind::InterZeroMv => {
                // §9.7 inter refresh ladder used by the multi-ref
                // P-frame encoder's default `RefreshControls`:
                // refresh_last = 1 only.
                self.last = Some(new_slot);
            }
        }

        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind,
            frame_index,
        })
    }

    /// Encode one frame using the configured keyframe interval to pick
    /// K vs P, with an automatically-fitted §13.4 `token_prob_update()`
    /// payload on every emitted frame.
    ///
    /// Drop-in fitted companion to [`Self::encode_frame`] / the K-frame
    /// arm of [`Self::encode_frame_with_force`]: scheduling, dimension-
    /// lock semantics, and the §9.7 reference-slot ladder are identical;
    /// only the bitstream differs. Each emitted frame independently
    /// runs the round-157 (keyframe) or round-158 (inter) two-pass
    /// fitter — the §13.4 payload is decided per frame from that frame's
    /// observed counts, never carried over to the next.
    ///
    /// Wire compatibility: on a frame where no §13.4 slot crosses the
    /// fitter's saving threshold, the safety-guard fall-back returns
    /// the bytes [`Self::encode_frame_with_force`] would have emitted
    /// for the same kind of frame.
    ///
    /// **Carried-base assumption.** The inter fitter assumes the prior
    /// key frame was emitted with the §13.5 defaults (i.e. an
    /// `encode_keyframe`-equivalent base table). This entry-point is
    /// the closure of that loop: K-frames go through
    /// [`encode_keyframe_with_reconstruction_and_fitted_token_prob_updates`]
    /// which itself emits a fitted-on-defaults wire — the decoder
    /// rebuilds `coeff_probs[4][8][3][11]` from defaults overlaid with
    /// the K-frame's payload, then the next P-frame's fitter again
    /// overlays its own payload on top of the §13.5 defaults. This
    /// matches what [`crate::state::Vp8DecoderState::decode_inter_frame`]
    /// does (`overlay_token_probs(self.coeff_probs, &coded.token_prob_updates)`
    /// where `self.coeff_probs` was reset to §13.5 defaults by the prior
    /// key frame).
    pub fn encode_frame_with_fitted_token_prob_updates(
        &mut self,
        frame: &I420Frame<'_>,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
        self.encode_frame_with_force_and_fitted_token_prob_updates(frame, false)
    }

    /// Encode one frame, with an optional `force_keyframe` override and
    /// the round-157 / round-158 fitter applied to every emitted frame.
    ///
    /// Companion to [`Self::encode_frame_with_force`] /
    /// [`Self::encode_frame_with_fitted_token_prob_updates`]. The
    /// `force_keyframe` semantics (re-anchoring the interval) match
    /// [`Self::encode_frame_with_force`] exactly.
    pub fn encode_frame_with_force_and_fitted_token_prob_updates(
        &mut self,
        frame: &I420Frame<'_>,
        force_keyframe: bool,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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

        let scheduled = self.next_frame_is_keyframe();
        let must_be_key = scheduled == FrameKind::Key || force_keyframe;
        let emit_key = must_be_key || self.last.is_none();

        let (bytes, planes, kind) = if emit_key {
            let (b, p) = encode_keyframe_with_reconstruction_and_fitted_token_prob_updates(
                frame,
                &self.params,
            )?;
            (b, p, FrameKind::Key)
        } else {
            // Mirrors the non-fitted arm exactly — same LAST/GOLDEN/ALTREF
            // plane harvesting, same multi-ref picker, only the fitted
            // inter entry-point swapped in.
            let last_planes = ref_slot_to_keyframe_planes(
                self.last.as_ref().expect("LAST slot present for P-frame"),
            );
            let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
            let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
            let (b, p) = encode_p_frame_multi_ref_with_fitted_token_prob_updates(
                frame,
                &last_planes,
                golden_planes.as_ref(),
                altref_planes.as_ref(),
                &self.params,
            )?;
            (b, p, FrameKind::InterZeroMv)
        };

        let frame_index = self.frame_count;

        // ---- Reference-slot update (§9.7) — identical to
        // `encode_frame_with_force`. The fitter does NOT affect the
        // §9.7 ladder; the fitter's matching-planes guarantee means
        // `planes` is the reconstruction that matches `bytes` regardless
        // of which pass won.
        let new_slot = RefFrameSlot::from_keyframe_planes(&planes);
        match kind {
            FrameKind::Key => {
                self.last = Some(new_slot.clone());
                self.golden = Some(new_slot.clone());
                self.altref = Some(new_slot);
                self.last_keyframe_index = Some(frame_index);
                // §9.4 — key frames start a fresh delta sequence.
                self.carried_ref_deltas = [0; 4];
                self.carried_mode_deltas = [0; 4];
            }
            FrameKind::InterZeroMv => {
                // §9.7 inter refresh ladder used by the multi-ref
                // P-frame encoder's default `RefreshControls`:
                // refresh_last = 1 only.
                self.last = Some(new_slot);
            }
        }

        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind,
            frame_index,
        })
    }

    /// Encode one P-frame with a caller-supplied §9.7 / §9.8
    /// reference-slot refresh pattern.
    ///
    /// This is the slot-rotation companion to
    /// [`crate::encoder::encode_p_frame_multi_ref_with_refresh`]: it
    /// emits a P-frame whose header carries the requested `refresh`
    /// bits and then evolves the driver's `LAST` / `GOLDEN` / `ALTREF`
    /// slots per the §20 page-147 walk (`copy_arf → copy_gf →
    /// refresh_gf → refresh_arf → refresh_last`), so the next
    /// `encode_frame*` call sees the same slot trio the in-tree
    /// decoder would after consuming the same wire.
    ///
    /// Pre-conditions:
    ///
    /// * A `LAST` slot must be present (the stream must have emitted at
    ///   least one prior frame). If not, the call returns
    ///   [`StreamEncodeError::NoLastReference`] — the caller should
    ///   drive at least one frame through the scheduler
    ///   ([`Self::encode_frame`] or [`Self::encode_frame_with_force`])
    ///   first.
    /// * Dimensions must match the stream's locked dimensions; a
    ///   mismatch is surfaced as
    ///   [`StreamEncodeError::DimensionsChanged`].
    /// * `refresh` is forwarded to
    ///   [`crate::encoder::encode_p_frame_multi_ref_with_refresh`],
    ///   which runs [`RefreshControls::validate`]; invalid
    ///   `copy_buffer_to_*` selectors surface as
    ///   [`EncodeError::InvalidCopyBufferSelector`] wrapped in
    ///   [`StreamEncodeError::Frame`].
    ///
    /// The keyframe scheduler is **bypassed** by this entry-point: the
    /// caller has asked for a specific refresh pattern, and forcing a
    /// key frame here would lose it.
    ///
    /// The slot rotation runs after the bitstream is emitted, mirroring
    /// the §20 page-147 ordering verbatim:
    ///
    /// 1. `copy_buffer_to_alternate` (1 = LAST → ALTREF, 2 = GOLDEN → ALTREF).
    /// 2. `copy_buffer_to_golden` (1 = LAST → GOLDEN, 2 = ALTREF → GOLDEN).
    /// 3. `refresh_golden_frame` (replace GOLDEN with current reconstruction).
    /// 4. `refresh_alternate_frame` (replace ALTREF with current reconstruction).
    /// 5. `refresh_last` (replace LAST with current reconstruction).
    ///
    /// `last_keyframe_index` is **not** touched (this is a P-frame).
    pub fn encode_p_frame_with_refresh(
        &mut self,
        frame: &I420Frame<'_>,
        refresh: &RefreshControls,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
        self.encode_p_frame_with_refresh_and_lf_deltas(frame, refresh, &LoopFilterDeltas::default())
    }

    /// Encode one P-frame with a caller-supplied §9.7 / §9.8
    /// reference-slot refresh pattern, with the round-160 / round-161
    /// §11 intra-within-inter MB picker engaged.
    ///
    /// Combines [`Self::encode_p_frame_with_refresh`] (caller-driven
    /// refresh ladder + stream-side slot rotation per §20 page-147)
    /// with the round-161 picker. On every MB the picker scores the
    /// full §11.2 × §11.4 whole-block intra grid (4 luma × 4 chroma)
    /// against the running in-frame neighbours in addition to the §16
    /// inter ladder, and picks whichever of (best inter, J-best intra)
    /// wins on `J + lambda * is_inter_mb-bit`.
    ///
    /// Pre-conditions and slot-rotation mirror
    /// [`Self::encode_p_frame_with_refresh`] exactly:
    ///
    /// * A `LAST` slot must be present —
    ///   [`StreamEncodeError::NoLastReference`] otherwise.
    /// * Dimensions must match the stream's locked dimensions —
    ///   [`StreamEncodeError::DimensionsChanged`] otherwise.
    /// * `refresh` is forwarded to
    ///   [`crate::encoder::encode_p_frame_multi_ref_with_refresh_and_intra_pick`],
    ///   which runs [`RefreshControls::validate`]; invalid selectors
    ///   surface as [`EncodeError::InvalidCopyBufferSelector`] wrapped
    ///   in [`StreamEncodeError::Frame`].
    /// * The keyframe scheduler is **bypassed** (the caller is asking
    ///   for a P-frame with a specific refresh).
    ///
    /// Slot-rotation runs after the bitstream is emitted, mirroring
    /// the §20 page-147 walk (`copy_arf → copy_gf → refresh_gf →
    /// refresh_arf → refresh_last`). `last_keyframe_index` is **not**
    /// touched. The §9.4 carried-delta state is **not** updated this
    /// round — the picker engages the round-160 inter path with
    /// [`LoopFilterDeltas::default`] (matching the bare-encoder
    /// `encode_p_frame_multi_ref_with_refresh_and_intra_pick`
    /// signature), so the effective deltas resolve to `0` and there
    /// is no fresh value to carry. Callers that need both intra-pick
    /// AND §9.4 deltas should sit one round above this — the
    /// composition fits naturally into the
    /// `…_intra_pick_and_lf_deltas` companion that a follow-up round
    /// will expose; this round's scope is exactly the intra-pick
    /// thread, not the full Cartesian product.
    pub fn encode_p_frame_with_refresh_and_intra_pick(
        &mut self,
        frame: &I420Frame<'_>,
        refresh: &RefreshControls,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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
        let last_slot = self
            .last
            .as_ref()
            .ok_or(StreamEncodeError::NoLastReference)?;

        let last_planes = ref_slot_to_keyframe_planes(last_slot);
        let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
        let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
        let (bytes, planes) = encode_p_frame_multi_ref_with_refresh_and_intra_pick(
            frame,
            &last_planes,
            golden_planes.as_ref(),
            altref_planes.as_ref(),
            &self.params,
            refresh,
        )?;

        let frame_index = self.frame_count;

        // ---- §9.7 / §9.8 reference-slot rotation -----------------------
        // Identical to `encode_p_frame_with_refresh_and_lf_deltas`: the
        // intra-pick flag governs per-MB candidate scoring only and
        // does NOT alter the §9.7 slot ladder. The §20 page-147 walk
        // is `copy_arf → copy_gf → refresh_gf → refresh_arf →
        // refresh_last`; the "copy" cases consult pre-refresh state,
        // so capture the pre-rotation slots into temporaries first.
        let current_slot = RefFrameSlot::from_keyframe_planes(&planes);
        let pre_last = self.last.clone();
        let pre_golden = self.golden.clone();
        let pre_altref = self.altref.clone();

        let mut new_altref = pre_altref.clone();
        match refresh.copy_buffer_to_alternate {
            1 => new_altref = pre_last.clone(),
            2 => new_altref = pre_golden.clone(),
            _ => {}
        }
        let mut new_golden = pre_golden.clone();
        match refresh.copy_buffer_to_golden {
            1 => new_golden = pre_last.clone(),
            2 => new_golden = pre_altref.clone(),
            _ => {}
        }
        if refresh.refresh_golden_frame {
            new_golden = Some(current_slot.clone());
        }
        if refresh.refresh_alternate_frame {
            new_altref = Some(current_slot.clone());
        }
        let new_last = if refresh.refresh_last {
            Some(current_slot.clone())
        } else {
            pre_last
        };

        self.last = new_last;
        self.golden = new_golden;
        self.altref = new_altref;
        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind: FrameKind::InterZeroMv,
            frame_index,
        })
    }

    /// Encode one P-frame with a caller-supplied §9.7 / §9.8
    /// reference-slot refresh pattern **and** a caller-supplied §9.4
    /// `mb_lf_adjustments()` per-reference / per-mode loop-filter
    /// delta layer.
    ///
    /// Companion to [`Self::encode_p_frame_with_refresh`] that exposes
    /// the §9.4 `loop_filter_adj_enable` /
    /// `mode_ref_lf_delta_update` + per-slot delta fields through
    /// [`crate::encoder::LoopFilterDeltas`]. The stream encoder threads
    /// the across-frame carried delta state per RFC 6386 §9.4 ("the
    /// values from the previous frame are used, unless they are updated
    /// in the current header") — the caller does not need to track it.
    ///
    /// Wire compatibility: passing
    /// [`crate::encoder::LoopFilterDeltas::default`] (with
    /// `enabled = false`) reproduces
    /// [`Self::encode_p_frame_with_refresh`] byte-for-byte, including
    /// when called repeatedly through the carried-state mechanism
    /// (`enabled = false` resolves effective deltas to 0 regardless
    /// of carried state).
    ///
    /// Slot-rotation and pre-conditions match
    /// [`Self::encode_p_frame_with_refresh`].
    pub fn encode_p_frame_with_refresh_and_lf_deltas(
        &mut self,
        frame: &I420Frame<'_>,
        refresh: &RefreshControls,
        lf_deltas: &LoopFilterDeltas,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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
        let last_slot = self
            .last
            .as_ref()
            .ok_or(StreamEncodeError::NoLastReference)?;

        let last_planes = ref_slot_to_keyframe_planes(last_slot);
        let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
        let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
        let (bytes, planes) = crate::encoder::encode_p_frame_multi_ref_with_refresh_and_lf_deltas(
            frame,
            &last_planes,
            golden_planes.as_ref(),
            altref_planes.as_ref(),
            &self.params,
            refresh,
            lf_deltas,
            self.carried_ref_deltas,
            self.carried_mode_deltas,
        )?;

        let frame_index = self.frame_count;

        // ---- §9.4 across-frame delta carry ----------------------------
        // The decoder updates its carried state with the effective
        // values it just resolved (carried + present updates). Mirror
        // that exactly so the next frame's effective deltas line up.
        let (eff_ref, eff_mode) =
            lf_deltas.effective(self.carried_ref_deltas, self.carried_mode_deltas);
        if lf_deltas.enabled {
            // RFC 6386 §9.4: when adj is enabled, the carried state for
            // the NEXT frame is what this frame's effective deltas
            // resolved to (whether they came from updates or carry).
            // When adj is DISABLED this frame, the spec keeps the
            // carried state untouched — the disabled-this-frame case is
            // not a "reset" event, it just means no delta is applied
            // this frame; the deltas persist for whenever the next
            // frame re-enables the feature.
            self.carried_ref_deltas = eff_ref;
            self.carried_mode_deltas = eff_mode;
        }

        // ---- §9.7 / §9.8 reference-slot rotation -----------------------
        // Mirror the §20 page-147 ordering exactly (the same walk the
        // decoder runs in `Vp8DecoderState`):
        //   copy_buffer_to_alternate → copy_buffer_to_golden →
        //   refresh_golden_frame → refresh_alternate_frame →
        //   refresh_last. The "copy" cases consult the slot state from
        //   BEFORE the refresh writes, so we capture pre-state in
        //   temporaries first.
        let current_slot = RefFrameSlot::from_keyframe_planes(&planes);
        let pre_last = self.last.clone();
        let pre_golden = self.golden.clone();
        let pre_altref = self.altref.clone();

        let mut new_altref = pre_altref.clone();
        match refresh.copy_buffer_to_alternate {
            1 => new_altref = pre_last.clone(),
            2 => new_altref = pre_golden.clone(),
            _ => {}
        }
        let mut new_golden = pre_golden.clone();
        match refresh.copy_buffer_to_golden {
            1 => new_golden = pre_last.clone(),
            2 => new_golden = pre_altref.clone(),
            _ => {}
        }
        if refresh.refresh_golden_frame {
            new_golden = Some(current_slot.clone());
        }
        if refresh.refresh_alternate_frame {
            new_altref = Some(current_slot.clone());
        }
        let new_last = if refresh.refresh_last {
            Some(current_slot.clone())
        } else {
            pre_last
        };

        self.last = new_last;
        self.golden = new_golden;
        self.altref = new_altref;
        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind: FrameKind::InterZeroMv,
            frame_index,
        })
    }

    /// Encode one P-frame with a caller-supplied §9.7 / §9.8 refresh
    /// pattern, §9.4 `mb_lf_adjustments()` delta layer, **and** §13.4
    /// `token_prob_update()` payload.
    ///
    /// Companion to [`Self::encode_p_frame_with_refresh_and_lf_deltas`]
    /// that exposes the §13.4 per-position
    /// `coeff_prob_update_flag` / `coeff_prob` sub-block through
    /// [`crate::coded_header::TokenProbUpdates`]. The encoder writes the
    /// replacement layer into the first-partition header and codes the
    /// §13.3 residual tokens against the merged
    /// `coeff_probs[4][8][3][11]` table (§13.5 defaults overlaid with
    /// the caller's per-position values), exactly mirroring what the
    /// decoder's `decode_inter_frame` rebuilds from the same wire.
    ///
    /// Wire compatibility: passing `token_updates = None` (or an
    /// all-`None` array) reproduces
    /// [`Self::encode_p_frame_with_refresh_and_lf_deltas`] byte-for-byte
    /// — every §13.4 flag is 0 and the §13.5 defaults stay in force.
    ///
    /// Slot-rotation and pre-conditions match
    /// [`Self::encode_p_frame_with_refresh`].
    ///
    /// **Assumption on carried entropy state.** This entry-point assumes
    /// the prior key frame was emitted with the §13.5 defaults (i.e.
    /// either [`crate::encoder::encode_keyframe`] /
    /// [`crate::encoder::encode_keyframe_with_reconstruction`], or
    /// [`crate::encoder::encode_keyframe_with_token_prob_updates`]
    /// called with an all-`None` array). The standard stream entry-points
    /// [`Self::encode_frame`] / [`Self::encode_frame_with_force`] both
    /// satisfy this since they go through
    /// [`crate::encoder::encode_keyframe_with_reconstruction`]. Mixing a
    /// non-default-base keyframe with this entry-point is out of round-
    /// 156 scope.
    pub fn encode_p_frame_with_refresh_and_lf_deltas_and_token_updates(
        &mut self,
        frame: &I420Frame<'_>,
        refresh: &RefreshControls,
        lf_deltas: &LoopFilterDeltas,
        token_updates: Option<&TokenProbUpdates>,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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
        let last_slot = self
            .last
            .as_ref()
            .ok_or(StreamEncodeError::NoLastReference)?;

        let last_planes = ref_slot_to_keyframe_planes(last_slot);
        let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
        let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
        let (bytes, planes) =
            crate::encoder::encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates(
                frame,
                &last_planes,
                golden_planes.as_ref(),
                altref_planes.as_ref(),
                &self.params,
                refresh,
                lf_deltas,
                self.carried_ref_deltas,
                self.carried_mode_deltas,
                token_updates,
            )?;

        let frame_index = self.frame_count;

        // ---- §9.4 across-frame delta carry ----------------------------
        // Same lifecycle rule as `encode_p_frame_with_refresh_and_lf_deltas`:
        // adj-enabled frames update the carried state with this frame's
        // effective deltas; adj-disabled frames leave it unchanged. The
        // §13.4 token-prob layer does NOT affect the §9.4 delta carry.
        let (eff_ref, eff_mode) =
            lf_deltas.effective(self.carried_ref_deltas, self.carried_mode_deltas);
        if lf_deltas.enabled {
            self.carried_ref_deltas = eff_ref;
            self.carried_mode_deltas = eff_mode;
        }

        // ---- §9.7 / §9.8 reference-slot rotation -----------------------
        // Identical to `encode_p_frame_with_refresh_and_lf_deltas`: token-
        // prob updates do NOT alter the §9.7 slot ladder (they govern
        // residual coding only).
        let current_slot = RefFrameSlot::from_keyframe_planes(&planes);
        let pre_last = self.last.clone();
        let pre_golden = self.golden.clone();
        let pre_altref = self.altref.clone();

        let mut new_altref = pre_altref.clone();
        match refresh.copy_buffer_to_alternate {
            1 => new_altref = pre_last.clone(),
            2 => new_altref = pre_golden.clone(),
            _ => {}
        }
        let mut new_golden = pre_golden.clone();
        match refresh.copy_buffer_to_golden {
            1 => new_golden = pre_last.clone(),
            2 => new_golden = pre_altref.clone(),
            _ => {}
        }
        if refresh.refresh_golden_frame {
            new_golden = Some(current_slot.clone());
        }
        if refresh.refresh_alternate_frame {
            new_altref = Some(current_slot.clone());
        }
        let new_last = if refresh.refresh_last {
            Some(current_slot.clone())
        } else {
            pre_last
        };

        self.last = new_last;
        self.golden = new_golden;
        self.altref = new_altref;
        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind: FrameKind::InterZeroMv,
            frame_index,
        })
    }

    /// Encode one P-frame with a caller-supplied §9.7 / §9.8
    /// reference-slot refresh pattern, a caller-supplied §9.4
    /// `mb_lf_adjustments()` per-reference / per-mode loop-filter delta
    /// layer, **and** the round-160 / round-161 §11 intra-within-inter
    /// MB picker engaged.
    ///
    /// Composition of [`Self::encode_p_frame_with_refresh_and_lf_deltas`]
    /// (round 151 §9.4 layer + across-frame carried-delta state) and
    /// [`Self::encode_p_frame_with_refresh_and_intra_pick`] (round 162
    /// stream thread of the round-160 / 161 picker). The round 162 next-
    /// step ladder named exactly this composition — "compose the §9.4
    /// `mb_lf_adjustments()` deltas with the intra-pick on the refresh
    /// path (`encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick`)"
    /// — as the follow-up that would expose both knobs together. Round
    /// 163 lands it.
    ///
    /// Internally calls
    /// [`crate::encoder::encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick`]
    /// (the matching bare-encoder wrapper). The stream encoder threads
    /// the across-frame carried delta state per RFC 6386 §9.4 ("the
    /// values from the previous frame are used, unless they are updated
    /// in the current header") — the caller does not need to track it;
    /// the carry-update rule is identical to
    /// [`Self::encode_p_frame_with_refresh_and_lf_deltas`] (adj-enabled
    /// frames write back the effective deltas; adj-disabled frames leave
    /// the carry untouched). The intra-pick toggle does NOT affect the
    /// §9.4 delta carry — it governs per-MB candidate scoring only.
    ///
    /// Wire compatibility:
    ///
    /// * Passing [`crate::encoder::LoopFilterDeltas::default`] (with
    ///   `enabled = false`) reproduces
    ///   [`Self::encode_p_frame_with_refresh_and_intra_pick`]
    ///   byte-for-byte. The §9.4 layer is gated on `lf_deltas.enabled`
    ///   exactly as on the non-intra-pick path.
    /// * On a source where intra never beats inter the wire matches
    ///   [`Self::encode_p_frame_with_refresh_and_lf_deltas`] modulo a
    ///   ~6 bits ≈ 1 byte frame-constant difference at the §9.10
    ///   `prob_intra` byte (sentinel `255` → fitted `1`), matching the
    ///   bound documented on the bare-encoder intra-pick path.
    ///
    /// Pre-conditions, slot-rotation (§20 page-147 walk
    /// `copy_arf → copy_gf → refresh_gf → refresh_arf → refresh_last`),
    /// and error surface (`NoLastReference`, `DimensionsChanged`) match
    /// [`Self::encode_p_frame_with_refresh`] exactly. `last_keyframe_index`
    /// is **not** touched.
    pub fn encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick(
        &mut self,
        frame: &I420Frame<'_>,
        refresh: &RefreshControls,
        lf_deltas: &LoopFilterDeltas,
    ) -> Result<EncodedStreamFrame, StreamEncodeError> {
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
        let last_slot = self
            .last
            .as_ref()
            .ok_or(StreamEncodeError::NoLastReference)?;

        let last_planes = ref_slot_to_keyframe_planes(last_slot);
        let golden_planes = self.golden.as_ref().map(ref_slot_to_keyframe_planes);
        let altref_planes = self.altref.as_ref().map(ref_slot_to_keyframe_planes);
        let (bytes, planes) =
            crate::encoder::encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick(
                frame,
                &last_planes,
                golden_planes.as_ref(),
                altref_planes.as_ref(),
                &self.params,
                refresh,
                lf_deltas,
                self.carried_ref_deltas,
                self.carried_mode_deltas,
            )?;

        let frame_index = self.frame_count;

        // ---- §9.4 across-frame delta carry ----------------------------
        // Identical to `encode_p_frame_with_refresh_and_lf_deltas` /
        // `encode_p_frame_with_refresh_and_lf_deltas_and_token_updates`:
        // adj-enabled frames update the carried state with this frame's
        // effective deltas; adj-disabled frames leave it unchanged. The
        // §11 intra-pick toggle does NOT affect the §9.4 delta carry —
        // it governs per-MB candidate scoring only.
        let (eff_ref, eff_mode) =
            lf_deltas.effective(self.carried_ref_deltas, self.carried_mode_deltas);
        if lf_deltas.enabled {
            self.carried_ref_deltas = eff_ref;
            self.carried_mode_deltas = eff_mode;
        }

        // ---- §9.7 / §9.8 reference-slot rotation -----------------------
        // Mirror the §20 page-147 ordering exactly (the same walk the
        // decoder runs in `Vp8DecoderState` and the same walk every other
        // refresh-aware entry-point on this struct runs):
        //   copy_buffer_to_alternate → copy_buffer_to_golden →
        //   refresh_golden_frame → refresh_alternate_frame →
        //   refresh_last. The "copy" cases consult the slot state from
        //   BEFORE the refresh writes, so we capture pre-state in
        //   temporaries first.
        let current_slot = RefFrameSlot::from_keyframe_planes(&planes);
        let pre_last = self.last.clone();
        let pre_golden = self.golden.clone();
        let pre_altref = self.altref.clone();

        let mut new_altref = pre_altref.clone();
        match refresh.copy_buffer_to_alternate {
            1 => new_altref = pre_last.clone(),
            2 => new_altref = pre_golden.clone(),
            _ => {}
        }
        let mut new_golden = pre_golden.clone();
        match refresh.copy_buffer_to_golden {
            1 => new_golden = pre_last.clone(),
            2 => new_golden = pre_altref.clone(),
            _ => {}
        }
        if refresh.refresh_golden_frame {
            new_golden = Some(current_slot.clone());
        }
        if refresh.refresh_alternate_frame {
            new_altref = Some(current_slot.clone());
        }
        let new_last = if refresh.refresh_last {
            Some(current_slot.clone())
        } else {
            pre_last
        };

        self.last = new_last;
        self.golden = new_golden;
        self.altref = new_altref;
        self.dimensions = Some(dims);
        self.frame_count += 1;
        Ok(EncodedStreamFrame {
            bytes,
            kind: FrameKind::InterZeroMv,
            frame_index,
        })
    }

    /// Borrow the across-frame §9.4 `ref_frame_delta[]` carried state
    /// (in `{CURRENT, LAST, GOLDEN, ALTREF}` order). Cleared to
    /// `[0; 4]` on every key frame per RFC 6386 §9.4.
    pub fn carried_ref_deltas(&self) -> [i16; 4] {
        self.carried_ref_deltas
    }

    /// Borrow the across-frame §9.4 `mode_delta[]` carried state
    /// (in `{B_PRED, ZERO_MV, OTHER_MV, SPLIT_MV}` order). Cleared
    /// to `[0; 4]` on every key frame per RFC 6386 §9.4.
    pub fn carried_mode_deltas(&self) -> [i16; 4] {
        self.carried_mode_deltas
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

    /// Configured keyframe interval.
    pub fn keyframe_interval(&self) -> u64 {
        self.keyframe_interval
    }

    /// 0-based frame index of the most recent key frame, or `None` if
    /// no frame has been encoded yet.
    pub fn last_keyframe_index(&self) -> Option<u64> {
        self.last_keyframe_index
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

/// Borrow a [`RefFrameSlot`] back into the [`KeyframePlanes`] shape
/// the P-frame encoder consumes as its `reference` argument.
///
/// The two structs hold the same fields (the §9 reference-frame buffer
/// is the same shape as a freshly-reconstructed key frame's planes);
/// this helper is just the field-by-field translation. We clone the
/// plane buffers because `encode_p_frame_zero_mv` takes
/// `&KeyframePlanes` and we don't want to leak the slot lifetime into
/// the encoder's call.
fn ref_slot_to_keyframe_planes(slot: &RefFrameSlot) -> KeyframePlanes {
    KeyframePlanes {
        y: slot.y.clone(),
        u: slot.u.clone(),
        v: slot.v.clone(),
        y_stride: slot.y_stride,
        uv_stride: slot.uv_stride,
        mb_cols: slot.mb_cols,
        mb_rows: slot.mb_rows,
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
    fn inter_stream_rejects_zero_keyframe_interval() {
        assert!(Vp8InterStreamEncoder::new(KeyframeParams::default(), 0).is_none());
    }

    #[test]
    fn inter_stream_first_frame_is_always_key() {
        let mut enc =
            Vp8InterStreamEncoder::new(KeyframeParams::default(), 4).expect("non-zero interval");
        assert_eq!(enc.next_frame_is_keyframe(), FrameKind::Key);

        let (y, u, v) = flat_frame(32, 32, 128, 128, 128);
        let frame = I420Frame::packed(32, 32, &y, &u, &v);
        let out = enc.encode_frame(&frame).expect("encode first frame");
        assert!(out.is_keyframe(), "first frame must be a key frame");
        assert_eq!(out.frame_index, 0);
        // §9.1 bit 0 of byte 0 = frame_type; key frame ⇒ 0.
        assert_eq!(out.bytes[0] & 0x01, 0, "key frame frame_type bit");
        // §9.1 keyframe start code at bytes 3..6.
        assert_eq!(
            &out.bytes[3..6],
            &[0x9d, 0x01, 0x2a],
            "key frame start code"
        );
        assert_eq!(enc.last_keyframe_index(), Some(0));
    }

    #[test]
    fn inter_stream_picks_p_after_first_with_interval_4() {
        let mut enc =
            Vp8InterStreamEncoder::new(KeyframeParams::default(), 4).expect("non-zero interval");
        let (y, u, v) = flat_frame(32, 32, 64, 128, 192);
        let frame = I420Frame::packed(32, 32, &y, &u, &v);

        // Frame 0 — K
        let f0 = enc.encode_frame(&frame).expect("frame 0");
        assert_eq!(f0.kind, FrameKind::Key);
        // Frame 1, 2, 3 — P
        for i in 1..=3u64 {
            assert_eq!(
                enc.next_frame_is_keyframe(),
                FrameKind::InterZeroMv,
                "frame {i} should be P"
            );
            let f = enc.encode_frame(&frame).expect("p frame");
            assert_eq!(f.kind, FrameKind::InterZeroMv, "frame {i}");
            assert_eq!(f.frame_index, i);
            // §9.1 bit 0 of byte 0 = 1 for inter.
            assert_eq!(f.bytes[0] & 0x01, 0x01, "P-frame frame_type bit");
        }
        // Frame 4 — K again.
        assert_eq!(enc.next_frame_is_keyframe(), FrameKind::Key);
        let f4 = enc.encode_frame(&frame).expect("frame 4");
        assert!(f4.is_keyframe(), "frame 4 must be K at interval 4");
        assert_eq!(enc.last_keyframe_index(), Some(4));
    }

    #[test]
    fn inter_stream_force_keyframe_reanchors_interval() {
        let mut enc =
            Vp8InterStreamEncoder::new(KeyframeParams::default(), 4).expect("non-zero interval");
        let (y, u, v) = flat_frame(32, 32, 100, 110, 120);
        let frame = I420Frame::packed(32, 32, &y, &u, &v);

        // K, P, P-forced-K, P, P, P, K (re-anchored at frame 2)
        let kinds = [
            (false, FrameKind::Key),         // 0 — first-ever
            (false, FrameKind::InterZeroMv), // 1
            (true, FrameKind::Key),          // 2 — forced
            (false, FrameKind::InterZeroMv), // 3
            (false, FrameKind::InterZeroMv), // 4 (would be K w/o re-anchor)
            (false, FrameKind::InterZeroMv), // 5
            (false, FrameKind::Key),         // 6 — re-anchored interval
        ];
        for (i, (force, expected)) in kinds.iter().enumerate() {
            let out = enc
                .encode_frame_with_force(&frame, *force)
                .unwrap_or_else(|e| panic!("frame {i}: {e}"));
            assert_eq!(out.kind, *expected, "frame {i} kind mismatch");
        }
        assert_eq!(enc.last_keyframe_index(), Some(6));
    }

    #[test]
    fn inter_stream_p_frame_refreshes_last_only() {
        let mut enc =
            Vp8InterStreamEncoder::new(KeyframeParams::default(), 100).expect("non-zero interval");
        let (y0, u0, v0) = flat_frame(32, 32, 60, 130, 200);
        let frame0 = I420Frame::packed(32, 32, &y0, &u0, &v0);
        enc.encode_frame(&frame0).expect("frame 0 K");
        let golden_after_k = enc.golden().expect("golden after K").clone();
        let altref_after_k = enc.altref().expect("altref after K").clone();

        // Big content change so the P-frame residual genuinely reshapes
        // LAST.
        let (y1, u1, v1) = flat_frame(32, 32, 200, 50, 90);
        let frame1 = I420Frame::packed(32, 32, &y1, &u1, &v1);
        enc.encode_frame(&frame1).expect("frame 1 P");

        // GOLDEN and ALTREF must be byte-identical to their state
        // after the K (the P-frame's §9.7 ladder leaves them alone).
        assert_eq!(enc.golden().expect("golden present").y, golden_after_k.y);
        assert_eq!(enc.golden().expect("golden present").u, golden_after_k.u);
        assert_eq!(enc.golden().expect("golden present").v, golden_after_k.v);
        assert_eq!(enc.altref().expect("altref present").y, altref_after_k.y);
        assert_eq!(enc.altref().expect("altref present").u, altref_after_k.u);
        assert_eq!(enc.altref().expect("altref present").v, altref_after_k.v);

        // LAST should now reflect the P-frame's reconstruction, which
        // for this big content change is no longer equal to the K's
        // reconstruction (= golden's contents).
        let last_after_p = enc.last().expect("last after P");
        assert_ne!(
            last_after_p.y, golden_after_k.y,
            "LAST must change after a P-frame, GOLDEN must not"
        );
    }

    #[test]
    fn inter_stream_dimensions_change_rejected() {
        let mut enc =
            Vp8InterStreamEncoder::new(KeyframeParams::default(), 4).expect("non-zero interval");
        let (y, u, v) = flat_frame(32, 32, 128, 128, 128);
        let frame = I420Frame::packed(32, 32, &y, &u, &v);
        enc.encode_frame(&frame).expect("first");
        let (y2, u2, v2) = flat_frame(48, 48, 64, 200, 50);
        let frame2 = I420Frame::packed(48, 48, &y2, &u2, &v2);
        let err = enc
            .encode_frame(&frame2)
            .expect_err("resize should be rejected");
        assert!(matches!(
            err,
            StreamEncodeError::DimensionsChanged {
                first: (32, 32),
                got: (48, 48),
            }
        ));
        assert_eq!(enc.frame_count(), 1);
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
