# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Integration tests against the `docs/video/vp8/` fixture corpus (17
  tests in `tests/docs_corpus.rs`, covering every fixture documented
  in `docs/video/vp8/vp8-fixtures-and-traces.md`). Each test demuxes
  the per-fixture `input.ivf`, decodes through `Vp8Decoder`, and
  scores per-plane pixel-match against `expected.yuv`. Three tiers:
  `BitExact` (gates CI), `ReportOnly` (logs divergence without
  failing while the underlying bug is queued), and `Ignored` (WebM
  container, deferred until oxideav-mkv is wired in). Negative case
  `yuv422-not-supported` covered via two checks: the decoder accepts
  the libvpx auto-converted yuv420 stream, and the encoder does not
  panic on a 4:2:2-shaped input.
- `[workspace]` self-contained marker in `Cargo.toml` mirroring the
  pattern in `oxideav-tta`: lets `cargo test` from inside the crate
  run hermetically without walking up into the parent monorepo
  workspace (which may carry mid-development sibling crates that
  fail to load).

### Bit-exact corpus fixtures

- `tiny-i-only-16x16` (1 MB, DC-only Y2, filter_level=1)
- `i-only-loopfilter-off` (filter_level=0, all-zero coeffs)
- `partition-padding-16x16-4parts` (1 MB, 4 token partitions)

### Report-only corpus fixtures

The remaining 13 fixtures decode but diverge from the reference YUV.
Each is tagged with a `TODO(vp8-corpus)` in `tests/docs_corpus.rs`
that names the fixture and the observed pixel-match percentage. Two
recurring root causes:

- B_PRED neighbour-context propagation between adjacent MBs (already
  noted in `tests/decode_keyframe.rs` and `src/lib.rs` module docs):
  `i-only-64x64` (92.94%), `q-low` (86.91%), `gradient-and-noise-128x128`
  (89.44%), `segment-4-partitions` (92.14%), `webm-mux-vs-ivf-ivf`
  (92.94%), `vp8-with-loopfilter-mode-simple` (61.02%).
- Loop filter / dequant boundary at high qindex with active deblocker:
  `q-high` (12.56% — qi=127 + filter_level=38),
  `i-only-loopfilter-high` (40.72% — filter_level=33).
- Inter / multi-frame fixtures inherit the keyframe bug since the
  P-frames reference a divergent keyframe: `i-frame-then-p-frame-64x64`
  (84.96%), `golden-update-cycle` (82.86%), `altref-arnr-on` (79.07%),
  `small-roi-segmentation` (22.96%).

## [0.1.2](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.1...v0.1.2) - 2026-05-02

### Other

- per-frame scene-cut adaptation
- per-MB segment maps with per-segment quantiser deltas
- per-MB context-adaptive ref-frame probabilities
- alt-ref / golden-ref planning + Lagrangian RDO mode decision
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

### Added

- Per-frame scene-cut adaptation in the encoder. New
  `Vp8EncoderConfig` knobs `enable_scene_cut` (default on),
  `scene_cut_threshold` (default 4.0), `scene_cut_quant_boost`
  (default 8) and `scene_cut_boost_frames` (default 4). The encoder
  watches each incoming source frame's per-pixel luma
  mean-absolute-difference (MAD) versus the previous source frame and
  flags a scene cut when the MAD exceeds
  `mean(MAD) + threshold · stddev(MAD)` over the running 16-frame
  window (also gated by an absolute floor of 12 luma units to avoid
  spurious cuts on quiet content). On a flagged cut the next frame is
  forced to a keyframe, the LAST / GOLDEN / ALTREF reference slots
  are dropped, and the post-cut N frames receive a linearly-tapered
  qindex boost so the rebuild GOP starts at higher quality. The
  legacy `make_encoder_with_qindex` constructor keeps the detector
  off so handcrafted small-frame test sequences stay bit-exact with
  pre-#166 behaviour. Six new integration tests in
  `tests/encoder_scene_cut.rs` cover the splice-point detection, the
  no-false-positive guarantee on steady-pan content, the post-cut
  PSNR regression, the legacy-cadence opt-out, and ffmpeg
  cross-decode validation.

- Per-MB segment maps (RFC 6386 §10). The encoder classifies each MB
  by source-luma 16×16 variance into one of four segments, then
  emits the segmentation block (`segmentation_enabled = 1`,
  `update_map = 1`, `update_data = 1`, `abs_delta = 0`) with the
  per-segment quantiser deltas from `Vp8EncoderConfig`. Per-MB
  segment_id bits are written before the skip flag using
  entropy-matched `tree_probs`. Default deltas `[-8, -4, 0, +4]`
  put smooth content into the low-qi segments (extra bits where the
  eye notices banding) and high-variance content into the high-qi
  segments. On a synthetic mixed-content fixture the bit-saving
  variant `[0, +2, +6, +12]` shrinks the bitstream by ~14% with
  sub-1 dB PSNR cost. Disable with `enable_segments = false` to
  recover the legacy single-segment encoding bit-for-bit.
- Decoder: per-MB dequant now consults `header.segmentation` +
  `info.segment_id` so segmented streams (from this encoder or
  libvpx) reconstruct correctly. Honours both `abs_delta` modes.
- Five new integration tests in `tests/encoder_segments.rs` covering
  the frame-header signalling on/off, the byte-size shrink on the
  mixed clip, the encoder ↔ in-tree-decoder roundtrip with segments
  on, and ffmpeg cross-decode validation.

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

- Per-MB context-adaptive `prob_intra` / `prob_last` / `prob_gf` (RFC
  6386 §9.10 field J): the encoder now runs the P-frame in two passes
  — pass 1 makes per-MB mode/ref decisions and accumulates the actual
  ref-frame distribution (intra / LAST / GOLDEN / ALT counts) without
  touching the bool encoder; pass 2 picks the entropy-matched prob
  triple from those counts (`round(256 · n_zero / total)`, clamped to
  1..=255) and emits the frame with the optimised probs. Replaces the
  prior fixed `200 / 128 / 128` (multi-ref) and `200 / 1 / 128`
  (single-ref) literals.
- Real-content byte-size deltas on the existing fixtures: SMPTE-bars
  30-frame clip −13.4% (1336 → 1157 B); gray 30-frame clip −22.9%
  (782 → 603 B); Mandelbrot 30-frame clip −1.25% (14201 → 14023 B,
  small because residual dominates intra-heavy frames). ffmpeg
  cross-decodes every fixture cleanly.
- New tests in `tests/encoder_ref_prob_context.rs` pin the per-frame
  prob triple against the actual MB ref-frame distribution and the
  total-bytes regression on the SMPTE fixture; new unit tests in
  `encoder.rs` cover `optimal_prob_8` rounding/clamping and the
  prob-triple emit on an all-SKIP P-frame.

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
