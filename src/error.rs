//! Local error type used by `oxideav-vp8`'s standalone (no
//! `oxideav-core`) public API.
//!
//! When the `registry` feature is enabled, [`Vp8Error`] gains a
//! `From<Vp8Error> for oxideav_core::Error` impl (defined in
//! [`crate::registry`]) so the trait-side surface (`Decoder` /
//! `Encoder`) can keep returning `oxideav_core::Result<T>` while the
//! underlying decode/encode functions stay framework-free.

use std::fmt;

/// `Result` alias scoped to `oxideav-vp8`. Standalone (no `oxideav-core`)
/// callers see this; framework callers convert via the gated
/// `From<Vp8Error> for oxideav_core::Error` impl.
pub type Result<T> = std::result::Result<T, Vp8Error>;

/// Error variants returned by `oxideav-vp8`'s standalone API.
///
/// The variants mirror the subset of `oxideav_core::Error` the codec
/// can hit. The crate intentionally avoids surfacing transport (`Io`)
/// or framework-specific (`FormatNotFound`, `CodecNotFound`) errors —
/// those originate in callers that are already linking `oxideav-core`.
#[derive(Debug)]
pub enum Vp8Error {
    /// The bitstream is malformed (bad sync code, truncated header,
    /// over-long partition table, etc.).
    InvalidData(String),
    /// The bitstream uses a feature this decoder doesn't implement,
    /// or the encoder was asked to emit a frame format it doesn't
    /// support.
    Unsupported(String),
    /// End of stream — no more packets / frames forthcoming.
    Eof,
    /// More input is required before another frame can be produced
    /// (decoder) or another packet can be flushed (encoder).
    NeedMore,
}

impl Vp8Error {
    /// Construct an [`Vp8Error::InvalidData`] from a stringy message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Construct an [`Vp8Error::Unsupported`] from a stringy message.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}

impl fmt::Display for Vp8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Eof => write!(f, "end of stream"),
            Self::NeedMore => write!(f, "need more data"),
        }
    }
}

impl std::error::Error for Vp8Error {}
