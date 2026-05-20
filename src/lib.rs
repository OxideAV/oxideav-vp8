//! # oxideav-vp8
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-20 audit).
//!
//! The prior implementation was retired under the workspace clean-room
//! policy and the crate is being re-implemented from scratch against
//! RFC 6386, using only material under `docs/` and black-box
//! validator binaries.
//!
//! Currently landed: VP8 boolean (range) entropy decoder, the
//! foundational primitive every higher-level decode step is built on
//! (RFC 6386 §7). See [`bool_decoder`].
//!
//! Frame header, macroblock decode, loop filter, and the encoder are
//! all still scaffolded — the top-level `decode_vp8` / `encode_vp8_*`
//! entry points return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

pub mod bool_decoder;

pub use bool_decoder::{BoolDecoder, BoolDecoderError};

#[cfg(feature = "registry")]
use oxideav_core::RuntimeContext;

/// Crate-local error type. Until the clean-room rebuild lands every
/// public API path returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "oxideav-vp8: orphan-rebuild scaffold — no decoder/encoder wired up"
        )
    }
}

impl std::error::Error for Error {}

/// Decode a VP8 elementary stream.
pub fn decode_vp8(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Encode a VP8 keyframe.
pub fn encode_vp8_keyframe(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the orphan-rebuild scaffold registers
/// nothing into the runtime context.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("vp8", register);
