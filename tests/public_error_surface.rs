//! Public-surface lock-test for `Vp8Error`.
//!
//! Downstream consumers (notably `oxideav-webp`'s lossy VP8 path) need
//! `Vp8Error` to be reachable from the crate root AND from the documented
//! `oxideav_vp8::error::Vp8Error` module path so they can build a
//! `From<oxideav_vp8::error::Vp8Error> for WebpError` adapter against it.
//! This test locks both:
//!
//! * The crate-root re-export — `use oxideav_vp8::Vp8Error;`.
//! * The module path — `use oxideav_vp8::error::Vp8Error;`.
//! * The four-variant shape (`InvalidData(String)` / `Unsupported(String)`
//!   / `Eof` / `NeedMore`) — the same shape `WebpError` uses, so the
//!   webp adapter is a 1-to-1 match.
//! * `From` adapters from the crate's internal error types
//!   ([`oxideav_vp8::DecodeError`] / [`oxideav_vp8::Error`]) — so a
//!   crate-internal call path can bubble its error straight through `?`.
//! * The `std::error::Error` / `Display` / `Clone` / `PartialEq`
//!   machinery.
//!
//! A regression that hides `Vp8Error` behind a private module path, or
//! that renames / drops one of the four variants, will fail to compile
//! here — which is exactly the breakage we want surfaced before a
//! downstream crate's CI catches it.

use oxideav_vp8::{DecodeError, Error, Vp8Error};
use std::error::Error as StdError;

#[test]
fn vp8_error_reachable_via_module_path_too() {
    // Both paths must refer to the same type.
    let via_root: Vp8Error = Vp8Error::Eof;
    let via_module: oxideav_vp8::error::Vp8Error = via_root.clone();
    assert_eq!(via_root, via_module);
}

#[test]
fn vp8_error_has_invalid_data_variant_with_string_payload() {
    let e = Vp8Error::InvalidData("truncated partition".into());
    let s = format!("{e}");
    assert!(s.contains("truncated partition"));
    assert!(s.contains("invalid data"));
}

#[test]
fn vp8_error_has_unsupported_variant_with_string_payload() {
    let e = Vp8Error::Unsupported("interframe".into());
    let s = format!("{e}");
    assert!(s.contains("interframe"));
    assert!(s.contains("unsupported"));
}

#[test]
fn vp8_error_has_eof_unit_variant() {
    let e = Vp8Error::Eof;
    let s = format!("{e}");
    assert!(s.contains("end of stream"));
}

#[test]
fn vp8_error_has_need_more_unit_variant() {
    let e = Vp8Error::NeedMore;
    let s = format!("{e}");
    assert!(s.contains("need more"));
}

#[test]
fn vp8_error_helper_constructors_accept_string_like() {
    let e1 = Vp8Error::invalid("static");
    let e2 = Vp8Error::invalid(String::from("owned"));
    assert!(matches!(e1, Vp8Error::InvalidData(_)));
    assert!(matches!(e2, Vp8Error::InvalidData(_)));

    let e3 = Vp8Error::unsupported("interframe");
    assert!(matches!(e3, Vp8Error::Unsupported(_)));
}

#[test]
fn vp8_error_converts_from_decode_error() {
    // A non-Unsupported DecodeError maps to InvalidData.
    let inner = DecodeError::ZeroDimension;
    let err: Vp8Error = inner.clone().into();
    assert!(matches!(err, Vp8Error::InvalidData(_)));

    // DecodeError::Unsupported(...) maps to Vp8Error::Unsupported(...).
    let unsup = DecodeError::Unsupported("interframe");
    let err: Vp8Error = unsup.into();
    assert!(matches!(err, Vp8Error::Unsupported(msg) if msg.contains("interframe")));
}

#[test]
fn vp8_error_converts_from_legacy_crate_error() {
    let inner = Error::NotImplemented;
    let err: Vp8Error = inner.into();
    assert!(matches!(err, Vp8Error::Unsupported(_)));
}

#[test]
fn vp8_error_is_a_std_error_send_sync_static() {
    // Static assertion via trait objects — if `Vp8Error` ever loses its
    // `std::error::Error` impl, this stops compiling. The
    // `Send + Sync + 'static` bound is what `Box<dyn Error>` returns
    // from `core::error::Error::source` requires.
    fn assert_is_error<E: StdError + Send + Sync + 'static>() {}
    assert_is_error::<Vp8Error>();
}

#[test]
fn vp8_error_can_be_cloned_and_compared() {
    // Cloning + equality are part of the surface — downstream callers
    // routinely stash an error in a struct and compare against a known
    // sentinel.
    let a = Vp8Error::InvalidData("boom".into());
    let b = a.clone();
    assert_eq!(a, b);

    let c = Vp8Error::Unsupported("interframe".into());
    assert_ne!(a, c);

    let eof1 = Vp8Error::Eof;
    let eof2 = Vp8Error::Eof;
    assert_eq!(eof1, eof2);
}
