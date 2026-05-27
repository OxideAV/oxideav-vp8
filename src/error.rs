//! Public error surface for `oxideav-vp8`.
//!
//! `Vp8Error` is the **single public error type** downstream consumers
//! build their own `From<oxideav_vp8::error::Vp8Error>` adapters against.
//! Notably:
//!
//! * `oxideav-webp`'s lossy VP8 path: `From<oxideav_vp8::error::Vp8Error>
//!   for WebpError` collapses 1-1 (variants intentionally share names so
//!   the adapter is a single `match`).
//! * Any other in-tree codec that wraps a VP8 substream (none right now;
//!   reserved for future).
//!
//! ## Shape
//!
//! Four variants, chosen to mirror the workspace's `WebpError` /
//! `oxideav_core::Error` shape exactly:
//!
//! * [`InvalidData`](Vp8Error::InvalidData) — the bitstream was
//!   syntactically wrong (truncated, bad token, out-of-range header
//!   field, …). Carries a free-form message.
//! * [`Unsupported`](Vp8Error::Unsupported) — the bitstream is valid VP8
//!   per RFC 6386 but uses a feature this build does not implement
//!   (deferred per the README ladder). Carries a message that names the
//!   feature.
//! * [`Eof`](Vp8Error::Eof) — the decode/encode pipeline reached the end
//!   of the input stream cleanly (used by the `oxideav_core::Decoder` /
//!   `Encoder` adapters; see [`crate::decoder::Vp8Decoder`]).
//! * [`NeedMore`](Vp8Error::NeedMore) — the pipeline needs another
//!   packet/frame before it can advance (`send_packet` then
//!   `receive_frame`).
//!
//! ## Standalone build
//!
//! This module is reachable with `default-features = false`. It does not
//! pull in `oxideav-core` — every variant is built from `std` primitives
//! only — so an embedded image / video pipeline that wants the VP8
//! bitstream layer without the runtime registry still gets a usable
//! public error type.

use core::fmt;

/// Crate-wide public error surface. See the [module docs](self) for the
/// design rationale and the four-variant shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vp8Error {
    /// The bitstream was syntactically malformed: a truncated partition,
    /// a token that escapes its tree, a header field out of its
    /// RFC 6386-permitted range, etc. The message is free-form and meant
    /// for human consumption.
    InvalidData(String),
    /// The bitstream is valid per RFC 6386 but exercises a feature this
    /// build does not implement yet (e.g. an interframe arriving at the
    /// stateless [`crate::decode_vp8`] entry point). The message names
    /// the unsupported feature.
    Unsupported(String),
    /// The decode/encode pipeline has consumed every packet it was
    /// given and there is nothing more to emit. Surfaced through the
    /// `oxideav_core::Decoder` / `Encoder` adapter when `flush` has
    /// run and the queue is empty.
    Eof,
    /// The decode/encode pipeline needs another input before it can
    /// produce output. Surfaced through `receive_frame` /
    /// `receive_packet` after a `send_*` / `flush` boundary when the
    /// next stage of the pipeline is data-starved.
    NeedMore,
}

impl Vp8Error {
    /// Convenience constructor — accepts anything that converts to
    /// `String`, so a `&'static str` literal works without a cast.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Vp8Error::InvalidData(msg.into())
    }

    /// Convenience constructor — accepts anything that converts to
    /// `String`.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Vp8Error::Unsupported(msg.into())
    }
}

impl fmt::Display for Vp8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vp8Error::InvalidData(msg) => write!(f, "oxideav-vp8: invalid data: {msg}"),
            Vp8Error::Unsupported(msg) => write!(f, "oxideav-vp8: unsupported: {msg}"),
            Vp8Error::Eof => f.write_str("oxideav-vp8: end of stream"),
            Vp8Error::NeedMore => f.write_str("oxideav-vp8: need more input"),
        }
    }
}

impl std::error::Error for Vp8Error {}

/// Crate-local convenience alias — every public entry point that can
/// fail returns this.
pub type Result<T> = core::result::Result<T, Vp8Error>;

// ───── Adapters that fold the existing sub-error types into `Vp8Error`.

impl From<crate::decoder::DecodeError> for Vp8Error {
    fn from(e: crate::decoder::DecodeError) -> Self {
        match e {
            crate::decoder::DecodeError::Unsupported(msg) => Vp8Error::Unsupported(msg.to_string()),
            other => Vp8Error::InvalidData(other.to_string()),
        }
    }
}

impl From<crate::Error> for Vp8Error {
    fn from(e: crate::Error) -> Self {
        match e {
            crate::Error::NotImplemented => {
                Vp8Error::Unsupported("requested operation not implemented".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_helper_accepts_static_str() {
        let e = Vp8Error::invalid("boom");
        assert_eq!(e, Vp8Error::InvalidData("boom".to_string()));
    }

    #[test]
    fn unsupported_helper_accepts_owned_string() {
        let s = String::from("interframe");
        let e = Vp8Error::unsupported(s);
        assert_eq!(e, Vp8Error::Unsupported("interframe".to_string()));
    }

    #[test]
    fn display_uses_crate_prefix() {
        let e = Vp8Error::InvalidData("truncated partition".into());
        let rendered = format!("{e}");
        assert!(rendered.starts_with("oxideav-vp8:"));
        assert!(rendered.contains("truncated partition"));
    }

    #[test]
    fn eof_and_need_more_have_no_payload() {
        // These are unit variants — guard against an accidental future
        // payload addition (which would be a breaking change for
        // downstream `From` matches).
        let _eof = Vp8Error::Eof;
        let _need = Vp8Error::NeedMore;
    }

    #[test]
    fn is_a_std_error() {
        fn assert_std_error<E: std::error::Error + Send + Sync + 'static>() {}
        assert_std_error::<Vp8Error>();
    }
}
