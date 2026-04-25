# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.0...v0.1.1) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- raise pframe PSNR bars + add regressions for mv-ref-probs fix
- fix MV ref probs to index per-column, correct RFC 6386 tables
- correct chroma MV averaging to RFC 6386 §18.1 formula
- fix intra-4x4 B_VR / B_VL / B_HD prediction formulas
- fix inter-frame header field order (golden before alt)
- fix KF_BMODE_PROB table, TL defaults, and U/V token order
- add BSD-3-Clause attribution for libvpx-derived code
- release v0.0.4

## [0.1.0](https://github.com/OxideAV/oxideav-vp8/compare/v0.0.3...v0.1.0) - 2026-04-19

### Other

- bump to 0.1.0 as vp8 is needed for webp
- finish the encoder — all intra modes, SPLIT_MV, loop filter
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
- claim AVI FourCCs via oxideav-codec CodecTag registry
- NEAREST/NEAR + quarter-pel ME + intra-in-P fallback in P-frame encoder
- clippy + rustfmt cleanup
- update README to match NEWMV + IVF muxer additions
- add write-side IVF muxer
- test NEWMV path recovers horizontal pan losslessly
