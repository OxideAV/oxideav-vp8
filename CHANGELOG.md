# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Alt-ref / golden-ref planning in the encoder. New
  `Vp8EncoderConfig` exposes `golden_interval` (default 8 P-frames),
  `alt_ref_interval` (default 13), `enable_multi_ref` and `enable_rdo`
  knobs; the encoder maintains LAST / GOLDEN / ALTREF reference slots
  on the schedule and emits the matching `refresh_golden_frame` /
  `refresh_alternate_frame` flags in the inter header.
- Per-MB reference-frame selection across LAST / GOLDEN / ALTREF, with
  `find_near_mvs_enc` now ref-aware (neighbours whose ref_frame
  differs from ours contribute nothing, exactly mirroring the
  decoder's RFC 6386 §16.3 walk).
- Lagrangian rate-distortion mode decision (`D + λ·R`) replacing the
  prior SAD-only picker. λ = `lambda_scale * QP^2 / 256` with a
  default scale of 218 (≈0.85·256). Rate is approximated from the
  per-MB mode-info bool-coded bits (skip / intra-vs-inter / ref / MV
  tree / MV deltas) using a 256-entry log2 LUT.
- New `make_encoder_with_config(params, Vp8EncoderConfig)` constructor
  for callers that want fine-grained control over the new knobs.
- Six new integration tests in `tests/encoder_altref_rdo.rs` covering
  the refresh-flag schedule, multi-ref vs single-ref BD-rate
  comparison, end-to-end decode at high PSNR, ffmpeg cross-decode
  validation, and the legacy single-ref opt-out path.

### Changed

- `prob_last` / `prob_gf` are now signaled at 128 (neutral) when any
  GOLDEN / ALTREF slot is populated, so the per-MB reference-frame
  bits cost about 1 bit each on average. With multi-ref disabled we
  retain the legacy `prob_last = 1` setting that makes
  `read_bool(1) == false` (REF_LAST) near-free.

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
