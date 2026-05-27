//! Registry plumbing — restores the 0.1.13 `oxideav_vp8::registry::*`
//! module path so historical consumers keep building.
//!
//! The current master exposes the registration entry point at
//! [`crate::decoder::register`] (via `oxideav_core::register!`). This
//! module re-exports it under its original module path and adds the
//! companion 0.1.13 `register_codecs` / `register_containers` helpers
//! so consumers that wired the three-function shape keep compiling.
//!
//! ## Standalone build
//!
//! The whole module is gated on the `registry` feature. With
//! `default-features = false` it is not built — consumers that opt out
//! of `oxideav-core` cannot use the registry path by definition.

#![cfg(feature = "registry")]

pub use crate::decoder::{register, register_codecs};

/// Register IVF + WebM container-side bindings for the VP8 codec.
///
/// The 0.1.13 surface exposed this alongside [`register`] /
/// [`register_codecs`]; today the container layer lives in sibling
/// crates (`oxideav-ivf`, `oxideav-mkv`), so this hook is a no-op for
/// now. The symbol is preserved so historical consumers that called it
/// during framework boot-strap keep compiling.
pub fn register_containers(_ctx: &mut oxideav_core::RuntimeContext) {
    // Container plumbing is now a sibling-crate responsibility. Keeping
    // this entry point as a tolerated no-op preserves the historical
    // 0.1.13 three-function shape without re-introducing a container
    // layer inside `oxideav-vp8`.
}
