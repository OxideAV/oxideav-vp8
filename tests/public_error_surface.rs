//! Public-surface lock-test for `Vp8Error`.
//!
//! Downstream consumers (notably `oxideav-webp`'s lossy VP8 path) need
//! `Vp8Error` to be reachable from the crate root so they can build a
//! `From<oxideav_vp8::Vp8Error>` adapter against it. This test locks
//! that public surface: it imports `Vp8Error` *via the crate root only*
//! (no `oxideav_vp8::lib::` reach-around), constructs each variant, and
//! checks the [`std::error::Error`] / [`From`] machinery is wired up.
//!
//! A regression that hides `Vp8Error` behind a private module path, or
//! that drops one of the variants / `From` impls, will fail to compile
//! here — which is exactly the breakage we want surfaced before a
//! downstream crate's CI catches it.

use oxideav_vp8::{DecodeError, Error, Vp8Error};
use std::error::Error as StdError;

#[test]
fn vp8_error_constructs_decode_variant_via_from() {
    // Pick the simplest DecodeError variant: ZeroDimension is a plain
    // unit, no payload needed.
    let inner = DecodeError::ZeroDimension;
    let err: Vp8Error = inner.clone().into();
    assert!(matches!(err, Vp8Error::Decode(DecodeError::ZeroDimension)));

    // `source()` must point at the wrapped DecodeError so downstream
    // error-chain printers (e.g. anyhow / miette) walk through it.
    let source = err
        .source()
        .expect("Vp8Error::Decode should carry a source");
    // The wrapped error's Display string must appear when we ask
    // for the source's Display, not the wrapper's.
    let source_msg = format!("{source}");
    assert_eq!(source_msg, format!("{inner}"));
}

#[test]
fn vp8_error_constructs_encode_variant_via_from() {
    let inner = Error::NotImplemented;
    let err: Vp8Error = inner.into();
    assert!(matches!(err, Vp8Error::Encode(Error::NotImplemented)));

    let source = err
        .source()
        .expect("Vp8Error::Encode should carry a source");
    let source_msg = format!("{source}");
    assert_eq!(source_msg, format!("{inner}"));
}

#[test]
fn vp8_error_display_delegates_to_wrapped_error() {
    let decode_err = Vp8Error::Decode(DecodeError::ZeroDimension);
    assert_eq!(
        format!("{decode_err}"),
        format!("{}", DecodeError::ZeroDimension),
    );

    let encode_err = Vp8Error::Encode(Error::NotImplemented);
    assert_eq!(
        format!("{encode_err}"),
        format!("{}", Error::NotImplemented),
    );
}

#[test]
fn vp8_error_is_a_std_error() {
    // Static assertion via trait objects — if `Vp8Error` ever loses its
    // `std::error::Error` impl, this stops compiling.
    fn assert_is_error<E: StdError + Send + Sync + 'static>() {}
    assert_is_error::<Vp8Error>();
}

#[test]
fn vp8_error_can_be_cloned_and_compared() {
    // Cloning + equality are part of the surface — downstream callers
    // routinely stash an error in a struct and compare against a known
    // sentinel.
    let a = Vp8Error::Encode(Error::NotImplemented);
    let b = a.clone();
    assert_eq!(a, b);

    let c = Vp8Error::Decode(DecodeError::ZeroDimension);
    assert_ne!(a, c);
}
