//! `oxideav-core` integration layer for `oxideav-vp8`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-vp8` with `default-features = false`
//! and skip the `oxideav-core` dependency entirely.
//!
//! The module exposes:
//! * [`register`] / [`register_codecs`] / [`register_containers`] — the
//!   `CodecRegistry` / `ContainerRegistry` entry points the umbrella
//!   `oxideav` crate calls during framework initialisation.
//! * The `From<Vp8Error> for oxideav_core::Error` conversion that lets
//!   the trait-side `Decoder` / `Encoder` impls (still living in
//!   `decoder.rs` / `encoder.rs`) bubble bitstream errors up through
//!   the framework error type.

use oxideav_core::ContainerRegistry;
use oxideav_core::RuntimeContext;
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

use crate::error::Vp8Error;
use crate::ivf;
use crate::CODEC_ID_STR;

/// Convert a [`Vp8Error`] into the framework-shared
/// `oxideav_core::Error` so trait impls in this crate can use `?` on
/// errors returned by the framework-free decode/encode functions.
impl From<Vp8Error> for oxideav_core::Error {
    fn from(e: Vp8Error) -> Self {
        match e {
            Vp8Error::InvalidData(s) => oxideav_core::Error::InvalidData(s),
            Vp8Error::Unsupported(s) => oxideav_core::Error::Unsupported(s),
            Vp8Error::Eof => oxideav_core::Error::Eof,
            Vp8Error::NeedMore => oxideav_core::Error::NeedMore,
        }
    }
}

/// Register the VP8 codec (decoder + encoder) into the supplied
/// [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("vp8_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384);
    // AVI FourCC claims — `VP80` is canonical, `VP8 ` (trailing space)
    // is the Google-blessed variant found in some .avi files.
    reg.register(
        CodecInfo::new(cid.clone())
            .capabilities(caps)
            .decoder(make_decoder)
            .tags([CodecTag::fourcc(b"VP80"), CodecTag::fourcc(b"VP8 ")]),
    );

    let enc_caps = CodecCapabilities::video("vp8_sw_enc")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16383, 16383);
    reg.register(
        CodecInfo::new(cid)
            .capabilities(enc_caps)
            .encoder(make_encoder),
    );
}

/// Register the IVF container demuxer + muxer + extension + probe into
/// the supplied [`ContainerRegistry`].
pub fn register_containers(reg: &mut ContainerRegistry) {
    ivf::register(reg);
}

/// Unified registration entry point — installs the VP8 codec into the
/// codec sub-registry and the IVF container into the container
/// sub-registry of the supplied [`RuntimeContext`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("vp8", register);

fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    crate::decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    crate::encoder::make_encoder(params)
}

#[cfg(test)]
mod register_tests {
    use super::*;

    #[test]
    fn register_via_runtime_context_installs_both_sides() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(crate::CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&id),
            "VP8 decoder factory not installed via RuntimeContext"
        );
        assert!(
            ctx.codecs.has_encoder(&id),
            "VP8 encoder factory not installed via RuntimeContext"
        );
        assert_eq!(
            ctx.containers.container_for_extension("ivf"),
            Some("ivf"),
            "IVF container extension not installed via RuntimeContext"
        );
    }
}
