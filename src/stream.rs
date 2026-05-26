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

use crate::encoder::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref,
    encode_p_frame_multi_ref_with_refresh, EncodeError, I420Frame, KeyframeParams, RefreshControls,
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
            }
            FrameKind::InterZeroMv => {
                // §9.7 inter refresh ladder used by encode_p_frame_zero_mv:
                // refresh_last = 1, refresh_golden_frame = 0,
                // refresh_alternate_frame = 0, copy_buffer_to_* = 0.
                // Only LAST changes.
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
        let (bytes, planes) = encode_p_frame_multi_ref_with_refresh(
            frame,
            &last_planes,
            golden_planes.as_ref(),
            altref_planes.as_ref(),
            &self.params,
            refresh,
        )?;

        let frame_index = self.frame_count;

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
