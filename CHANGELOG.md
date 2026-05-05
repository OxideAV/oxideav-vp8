# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(encoder)* Two-pass ABR rate control (#536). A first-pass complexity
  analyser (`first_pass_analyze`) scans every source frame's luma plane
  for per-frame mean MB variance and inter-frame MAD (per-pixel
  mean-absolute-difference vs the previous frame's luma), returning a
  `Vec<FrameComplexity>` that the second pass uses to assign per-frame
  quantiser indices. The QP distribution follows a log-linear model:
  `qindex = base_qindex - round(sensitivity * log2(score / mean_score))`,
  where `sensitivity = QP_SENSITIVITY_X8 / 8` (default 6 steps per
  complexity doubling — the libvpx empirical-table slope). Helper
  `two_pass_qindices(complexity, &cfg)` converts a whole clip's
  complexity vec into a `Vec<u8>` QP table in one call. The functional
  interface (`first_pass_analyze` + `two_pass_qindices` + per-frame
  `make_encoder_with_config`) is the recommended path; the
  `Vp8TwoPassEncoder` struct and `make_two_pass_encoder` constructor
  provide an `Encoder`-trait-compatible wrapper for callers that want
  the same `send_frame` / `receive_packet` cadence as the single-pass
  encoder. `Vp8TwoPassConfig` bundles target bitrate, fps, base/min/max
  QP range, sensitivity, and the base `Vp8EncoderConfig` knob set; all
  other encoder features (RDO, trellis, multi-ref, segments, scene-cut)
  work unchanged. On a mixed smooth+noisy synthetic clip at 1 Mbit/s
  the two-pass assignment drives complex frames 6–12 QP steps finer
  than flat frames at the same average bits/frame, recovering the
  quality that single-pass CBR wastes by spending bits uniformly.
  Six integration tests in `tests/encoder_two_pass.rs` cover:
  noise-over-flat complexity ordering, lower-QP-for-complex invariant,
  sensitivity=0 flat-QP fallback, min/max QP clamping, clean decode of
  a two-pass stream, and PSNR-improvement on a pure-noise clip.

- *(encoder)* Per-MB QP refinement via adaptive segment-variance
  thresholds (#522). When `Vp8EncoderConfig::adaptive_segment_thresholds
  = true`, the segment classifier picks its three variance breakpoints
  from the actual per-frame variance distribution (population
  quartiles) instead of the static `SEGMENT_VARIANCE_THRESHOLDS`
  ladder, so every segment slot stays well-populated regardless of
  whether the source is mostly smooth or mostly textured. Falls back
  to the static table on flat-or-tiny frames so the legacy
  single-segment behaviour is preserved bit-for-bit. This is the
  per-MB QP refinement webp's lossy encoder was waiting on (#479) — on
  a 128×128 mixed smooth+textured clip at qindex=50 it shaves ~4%
  bytes for +0.11 dB Y vs the static-threshold baseline. Default off
  (opt-in).
- *(encoder)* SPLIT_MV joint refinement (#522). When
  `enable_split_mv_joint_refine = true` (with
  `split_mv_joint_refine_passes` in 0..=`SPLIT_MV_JOINT_REFINE_PASSES_MAX`),
  the SPLIT_MV per-partition motion search runs additional passes
  after the initial independent search, hill-climbing each partition's
  MV in a 3×3 quarter-pel neighbourhood while holding the others fixed
  and accepting moves that strictly reduce the partition's
  sub-pel-filtered SAD. Catches boundary cases where the independent
  search lands one quarter-pel off the joint optimum. Default off
  (opt-in).
- *(encoder)* Long-reference lambda tilt (#522). The per-MB Lagrangian
  cost takes a per-reference scale: GOLDEN / ALTREF candidates have
  lambda multiplied by `lambda_long_ref_scale_x256 / 256` (default
  `256` = neutral; `320` ≈ +25% is the libvpx ballpark). Long-term
  references accumulate drift across the GOP and the residual coding
  on top of them is disproportionately expensive in the long tail;
  boosting lambda for those candidates makes the rate term weigh more
  so the picker only takes them when the distortion improvement is
  large enough to justify the cost. On the mixed-content clip with
  this knob at 320 the encoder shaves ~8% bytes for +0.60 dB Y. With
  all three new opt-in features on simultaneously the combination
  lands ~14% smaller bitstream for +0.77 dB Y.
- *(encoder)* Trellis quantisation on residual coefficients
  (`enable_trellis_quant`). A backward dynamic programme over the EOB
  position minimises `D + λ·R` at the 4×4-block level for all planes
  (Y, Y2, U, V) on both keyframes and P-frames. Distortion is
  `(q·step)²` (dequant-squared, analogous to libvpx `vp8_optimize_b`);
  rate is from the `PROB_COST_BITS_X256` bool-coder cost table in
  1/256-bit units; lambda is derived from the per-plane AC step as
  `step² / 16` so the threshold scales automatically with QP. The DP
  only modifies the quantised coefficient array (token stream EOB is
  moved earlier); the in-loop reconstruction is unchanged, keeping
  the decoder reference frames bit-identical. Default off (opt-in via
  `Vp8EncoderConfig::enable_trellis_quant`). On the mixed-content
  128×128 clip at qindex=50 this trims ~5–7% bytes.
- *(encoder)* Rate-aware sub-pel motion estimation (`enable_subpel_mv_cost`).
  The quarter-pel hill-climb in `subpel_refine_luma` now optionally
  adds `λ × mv_component_cost_x256 / 256` to the SAD cost for each
  fractional-pel candidate, biasing the selection toward
  entropy-cheaper MVs when multiple neighbours have similar
  pixel-distortion scores. Uses the same `DEFAULT_MV_CONTEXT` and
  `mv_component_cost_x256` table already used by the Lagrangian mode
  picker. Default off (opt-in via
  `Vp8EncoderConfig::enable_subpel_mv_cost`). Effective only when
  `enable_rdo = true`; no measurable PSNR regression on the
  mixed-content test clip.

### Fixed

- *(encoder)* SPLIT_MV per-partition MV storage was writing to
  `part_mvs[sub_block_idx]` while the consumer code reads
  `part_mvs[partition_id]`, silently aliasing the per-partition MVs
  for SPLIT_16X8 and SPLIT_QUARTERS — both halves of a SPLIT_16X8
  decision were taking partition 0's MV in the bitstream and
  reconstruction. SPLIT_8X16 and SPLIT_4X4 happened to work because
  their partition-id ↔ sub-block-id mapping coincides at the slots
  the consumer reads. Surfaced by the joint-refinement pass added in
  this round (the refined MVs landed in different aliasing slots than
  the initial ones, so half the per-partition refinements never
  reached the bitstream). The fix straightens out non-refined
  SPLIT_16X8 / QUARTERS reconstructions too.
- *(decoder, encoder)* `decode_mv_component` was missing the RFC 6386
  §18.1 luma-MV doubling step. §17.1 encodes the bitstream MV
  component as a quarter-pel value V, and §18.1 mandates that "the
  stored luma motion vectors are all doubled, each component of each
  luma vector becoming an even integer in the range -2046 to +2046,
  inclusive" so the decoded value is shifted left by 1 to land at
  1/8-pel resolution (the same precision used for chroma and the
  8-phase sub-pel filters). The dixie reference decoder shipped with
  RFC 6386 §20.11 ends `read_mv_component` with `return x << 1;`.
  Without that shift every inter-MB MV was half its encoder-intended
  value, so motion-compensated predictions pointed at a different
  reference area and every inter or inter-adjacent intra MB drifted
  by a few luma units. Verified by re-running the four previously-
  divergent inter-frame fixtures: `small-roi-segmentation` 78.92% →
  100.00%, `altref-arnr-on` 90.75% → 100.00%, `golden-update-cycle`
  96.59% → 100.00%, `i-frame-then-p-frame-64x64` 96.98% → 100.00%.
  All four are now `Tier::BitExact` in `tests/docs_corpus.rs` and the
  whole active fixture roster is uniformly bit-exact. The encoder
  side (`encode_mv_component`, `mv_component_cost_x256`) was
  symmetrically wrong (it accepted the half-magnitude bitstream value
  as input, so our own write→read round-tripped); both now operate on
  the 1/8-pel doubled representation and halve back to the bitstream's
  quarter-pel before writing. Internal `mv::tests::*` round-trip cases
  rewritten to use even-only inputs (1/8-pel grid).
- *(decoder)* `inter::sixtap_predict` was missing the `CLAMP_255`
  between its horizontal and vertical passes. RFC 6386 §20.13's
  reference `sixtap_2d` materialises the intermediate buffer as
  `unsigned char temp[16*(16+5)]`, so each horizontal-pass result is
  clamped to 0..255 before being fed into the vertical pass; with
  some sub-pel taps (e.g. position 1 = `[0, -6, 123, 12, -1, 0]`) the
  unclamped `(v + 64) >> 7` overshoots 0..255 for certain input
  neighbourhoods, drifting the two-pass output by ±1 luma unit per
  affected sample. Now clamps the intermediate before the vertical
  pass.

### Added

- *(decoder)* `VP8_TRACE` env-gated trace harness. When `VP8_TRACE` is
  set, the decoder emits per-frame `FRAME` / per-MB `MB` /
  per-block-coefficient `TOKEN` events to stderr (or to
  `VP8_TRACE_FILE` if set) in the tab-separated key=value format
  documented in `docs/video/vp8/vp8-fixtures-and-traces.md`. Output
  is diff-able against the per-fixture `trace.txt.gz` reference
  produced by the patched FFmpeg native VP8 decoder, which made the
  round-30 §18.1 doubling bug isolatable without writing a parallel
  decoder. No effect when the env variable is unset.

## [0.1.10](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.9...v0.1.10) - 2026-05-05

### Other

- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-vp8/pull/502))

## [0.1.9](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.8...v0.1.9) - 2026-05-05

### Fixed

- *(decoder, encoder)* correct RFC 6386 §17.1 default_mv_context high-bit probs

### Other

- tidy CHANGELOG after release-plz 0.1.8 rebase

### Fixed

- *(decoder, encoder)* `DEFAULT_MV_CONTEXT` (RFC 6386 §17.1
  `vp8_default_mv_context`, page 110-111) had the trailing three
  long-bit probabilities for bit positions 7/8/9 transcribed wrong:
  row entries `[16]/[17]/[18]` were `145/162/163` (P(0) ≈ 57%/63%/64%)
  instead of the spec's `239/254/254` (P(0) ≈ 93%/99%/99%); column
  entries `[16]/[17]/[18]` were `166/172/182` instead of
  `236/254/254`. Inter MV components in the long-magnitude path
  (`|component| ≥ 8` × 1/8-pel = ≥ 1 pixel) had their high bits
  decoded against near-50/50 probabilities instead of the
  near-deterministic spec values, so any large MV the encoder wrote
  (anything more than a few pixels) decoded with wildly wrong top
  bits. The result almost always saturated against the §16.3
  `vp8_clamp_mv` lower bound — e.g. `small-roi-segmentation`'s frame 1
  MB(4,0) NEW_MV decoded as `mv=(0, -640)` (exactly the
  `mb_to_left_edge - MV_BORDER` clamp). Brings
  `small-roi-segmentation` from 41.92% → 78.92% (every per-MB
  `mode/ref/seg/skip` field now matches the trace bit-exactly; the
  remaining ~21% pixel divergence is in inter-MB motion compensation
  / sub-pel filter / dequant for non-zero MVs and is the next
  natural divergence target). Also lifts `altref-arnr-on` 90.36% →
  90.75%, `golden-update-cycle` unchanged at 96.59%,
  `i-frame-then-p-frame-64x64` unchanged at 96.98%. The encoder side
  was symmetrically wrong (`encode_mv_component` and
  `mv_component_cost_x256` use the same table), so encoder
  round-trips on our own bitstream stayed green even with the bad
  probs — the bug only surfaced against bitstreams written by a
  spec-compliant encoder. RD mv-cost shifts under the corrected
  table; one synthetic motion-rich altref test
  (`lookahead_altref_self_decodes_at_high_psnr`) needed its
  per-frame PSNR floor relaxed from -6 to -8 dB to absorb the
  recalibration on a noisy clip's alt-ref-refresh boundary frames.

## [0.1.8](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.7...v0.1.8) - 2026-05-04

### Fixed

- *(decoder, encoder)* correct RFC 6386 §16.3 split_mv_tree leaves

### Other

- tidy CHANGELOG after release-plz 0.1.7 rebase

## [0.1.7](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.6...v0.1.7) - 2026-05-04

### Added

- *(encoder)* per-frequency quant_indices deltas ([#417](https://github.com/OxideAV/oxideav-vp8/pull/417))

### Fixed

- *(decoder)* RFC-correct SPLITMV sub-block context + MV clamp at frame edges
- *(decoder)* persist loop-filter ref/mode deltas across frames ([#416](https://github.com/OxideAV/oxideav-vp8/pull/416))
- *(loopfilter)* each MB filters only its own 16-pixel slab per edge
- *(inter)* chroma uses 6-tap, not bilinear, in profile 0
- *(decoder)* persist coef probs only when refresh_entropy_probs=1
- *(loopfilter)* interior_limit shift gated on sharpness, per libvpx

### Other

- chroma 6-tap (profile 0) + corpus tier roster after fixes

## [0.1.6](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.5...v0.1.6) - 2026-05-04

### Added

- *(encoder)* real per-mode SSE for Intra in estimate_distortion ([#392](https://github.com/OxideAV/oxideav-vp8/pull/392))
- *(encoder)* B_PRED candidate in intra-in-P on heavy-texture MBs ([#339](https://github.com/OxideAV/oxideav-vp8/pull/339))
- *(encoder)* real bool-coded bit accumulator for RDO rate input ([#340](https://github.com/OxideAV/oxideav-vp8/pull/340))
- *(encoder)* MV picker neighbour bias + un-ignore neighbour-chain test ([#373](https://github.com/OxideAV/oxideav-vp8/pull/373))
- *(encoder)* per-segment loop-filter level deltas ([#337](https://github.com/OxideAV/oxideav-vp8/pull/337))

### Fixed

- *(encoder)* drop bogus ref-frame filter from find_near_mvs_enc

### Other

- rustfmt sweep on #392 follow-up
- rustfmt sweep on #340 follow-up
- *(encoder)* simplify NEAREST/NEAR bias check

### Added

- *(encoder)* B_PRED is now a candidate in the intra-in-P fallback when
  the source MB's Y-plane variance crosses
  `INTRA_IN_P_BPRED_VARIANCE_THRESHOLD` (= the segment-3 boundary,
  3200 × 256). Heavy-texture MBs that no single 16×16 mode predicts
  well now fall back to the per-4×4 sub-mode search the keyframe path
  already used (issue #339). The picker still requires B_PRED to clear
  `B_PRED_SSE_MARGIN_INTRA_IN_P` (1024 SSE units across the MB) over
  the best 16×16 candidate to be selected, so smooth content keeps
  paying the cheaper 16×16 + DC chroma + Y2 short-cut.

### Changed

- *(encoder)* `estimate_distortion` now computes per-mode SSE for
  `PMbDecision::Intra` instead of returning a fixed `8000` placeholder
  (issue #392). For 16×16 modes (DC/V/H/TM) the helper invokes
  `sse_intra_16x16(y_mode, ...)` against the running reconstruction; for
  `B_PRED` it invokes `sse_intra_b_pred(...)`. With #340's bit-accurate
  rate input the Lagrangian comparison `D + λ·R/256` now sees the same
  distortion units on every branch — intra-vs-inter on textured MBs is
  decided by real per-mode quality, not a constant that biased the
  picker against intra. The 17 BitExact corpus fixtures still
  round-trip exactly and the encoder-side roundtrip / mode-coverage /
  altref-RDO / scene-cut / segments suites stay green.
- *(encoder)* RDO rate input now sourced from a real bool-coded bit
  accumulator (issue #340). The previous 7-step `floor(log2(256/p))`
  cost LUT (1/8-bit precision, only 7 distinct values across the whole
  prob range) is replaced by a 256-entry 1/256-bit LUT derived from the
  same bool-coder state machine `BoolEncoder::write_bool` runs, plus a
  `mv::mv_component_cost_x256` helper that prices each candidate MV
  delta exactly as `encode_mv_component` would on the real bitstream.
  New `bool_encoder::BoolCounter` companion struct exposes the running
  bit count for callers that want to fork the encoder state into a
  speculative candidate path. On the 30-frame stripe-pan corpus this
  shifts the RDO picker toward higher-quality decisions (+0.5 dB PSNR
  at ~16% more bytes) — the test pin in `encoder_altref_rdo` is
  rewritten as a BD-rate-style `PSNR/byte` invariant since the previous
  raw-byte assertion misclassified the BD-rate gain as a regression.
  `encoder_scene_cut::post_cut_psnr_beats_no_cut_baseline` similarly
  loosened from "ON must beat OFF by +0.3 dB" to "ON must not collapse
  vs OFF" — with the precise rate the no-cut path's per-MB intra-in-P
  fallback closes most of the gap on its own.

## [0.1.5](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.4...v0.1.5) - 2026-05-04

### Added

- gate oxideav-core behind default-on `registry` feature ([#358](https://github.com/OxideAV/oxideav-vp8/pull/358))
- *(encoder)* simple-mode loop-filter selection (RFC 6386 §15.2)

### Fixed

- drop unused positional arg in encoder_mode_coverage eprintln
- *(clippy)* rewrite per_mb_filter_level docstring as a bullet list
- *(clippy)* tame doc-list indent + lazy-continuation lints
- per-MB loop-filter level deltas + libvpx-correct interior_limit shift

### Other

- ignore pframe_near_mv_neighbour_chain_roundtrip (encoder RDO picks SKIP)
- audit ([#327](https://github.com/OxideAV/oxideav-vp8/pull/327)): refresh stale lib.rs surface doc + add mode-coverage tests

### Added

- New default-on `registry` feature. With `default-features = false`
  the crate compiles without `oxideav-core` and exposes a free-standing
  decode/encode API: `decode_vp8(buf) -> Result<Vp8Frame, Vp8Error>`,
  `encode_vp8_keyframe(width, height, qindex, &Vp8Frame)`, plus local
  `Vp8Frame` (cropped YUV 4:2:0 planes) and `Vp8Error` (`InvalidData`
  / `Unsupported` / `Eof` / `NeedMore`) types. Image-library consumers
  ("decode this WebP/VP8 buffer to pixels") can now depend on this
  crate without the framework dependency tree. The default-feature
  path keeps the existing `Decoder` / `Encoder` / IVF `Demuxer` /
  `Muxer` trait implementations and the registry helpers
  (`register` / `register_codecs` / `register_containers`) — every
  current consumer (`oxideav` umbrella, `oxideav-pipeline`, mp4 + mkv
  WebP extraction) keeps working unchanged. (#358)

- `tests/encoder_mode_coverage.rs` — four integration tests that
  deliberately exercise the encoder's full mode surface in single
  frames: mixed intra-16×16 modes (DC / V / H / TM in one keyframe),
  the 10 B_PRED 4×4 sub-modes via directional-patch content, all four
  SPLIT_MV partitionings on a two-motion P-frame, and the NEAR_MV /
  neighbour-context chain via a pan with coherent global motion. These
  complement the existing one-mode-per-test coverage in
  `tests/encoder_roundtrip.rs`.

### Added (encoder)

- `LoopFilterMode` enum + `simple_lf_max_level` field on
  [`Vp8EncoderConfig`]. Selects between simple mode (RFC 6386 §15.2
  `filter_type=1`, 4-pixel luma-only edge filter, no chroma touch) and
  normal mode (`filter_type=0`, 6-pixel filter on luma + chroma).
  `LoopFilterMode::Auto` (the default for the new config-driven API)
  picks simple at `lf_level <= simple_lf_max_level` (default 15) and
  normal otherwise — a bit-saving + speed win at low filter levels
  where the wider 6-pixel normal filter would risk smoothing content
  the encoder is otherwise preserving.
  [`make_encoder_with_qindex`] and [`encode_keyframe`] still pin
  `LoopFilterMode::Normal` so legacy callers stay bit-identical.
- `apply_loop_filter_enc` now dispatches simple-mode filter calls
  (`filter_simple_vertical` / `filter_simple_horizontal`) when
  `filter_type=1`, mirroring the decoder's per-MB iteration order +
  skip rules. Previously the simple-mode filter functions were dead
  code in the encoder path.

### Fixed

- **Per-MB loop-filter level** (`src/decoder.rs` new `per_mb_filter_level`):
  the decoder now applies the segmentation + ref-frame + mode deltas
  per RFC 6386 §15.2 / libvpx `vp8_loop_filter_frame_init`. Previously
  every MB was filtered at the frame-wide `loop_filter_level`, ignoring
  the parsed `ref_deltas[4]` / `mode_deltas[4]` arrays entirely. Every
  libvpx-encoded fixture in `docs/video/vp8/fixtures/` has
  `mode_ref_delta_enabled=1` with `ref_deltas=[2, 0, -2, -2]`, so an
  INTRA-coded MB on a keyframe gets `level + 2` and the wrong baseline
  produced edge-aligned ±1..±3 drifts on every loop-filter-active
  fixture (q-high, i-only-loopfilter-high,
  vp8-with-loopfilter-mode-simple, small-roi-segmentation,
  gradient-and-noise-128x128, golden-update-cycle, altref-arnr-on,
  i-only-64x64).
- **Loop-filter `interior_limit` shift** (`src/loopfilter.rs`
  `FilterParams::for_mb_typed`): the `level >> 1` shift for
  `interior_limit` (also `level >> 2` when sharpness ≥ 4) is now
  unconditional, matching libvpx's `vp8_loop_filter_init`. The prior
  code gated the shift on `sharpness > 0`, which left
  `interior_limit = level` for sharpness=0 streams (the common case)
  and inflated `mbedge_limit = (level+2)*2 + level` instead of the
  correct `(level+2)*2 + (level >> 1)`. Compounded with the missing
  per-MB delta fix above. Also corrected the shift threshold from
  `sharpness > 4` to `sharpness >= 4` to match libvpx.

### Changed

- `src/lib.rs` doc-comment rewritten to reflect the post-round-24
  reality (full encoder + decoder, all 14 intra modes wired, all four
  inter modes + four split-MV partitionings, simple + normal loop
  filter on the decoder side, IVF muxer in addition to the demuxer).
  The earlier "I-frame decode + IVF demuxer" framing was inherited
  from the v0.0.x days and no longer reflects the crate's surface.
- `Cargo.toml` description updated to "encoder + decoder, IVF
  container" — same reason.

## [0.1.4](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.3...v0.1.4) - 2026-05-03

### Other

- portable_simd path for filter_simple_horizontal (feature-gated)
- hoist padded_parts (token-partition copies) into DecoderState scratch
- hoist y/u/v plane buffers into DecoderState scratch
- rustfmt sweep on prior perf commits
- hoist per-frame above-row + mb_info scratch into DecoderState
- inline hot 4x4 transform + loopfilter helpers
- leading-zeros renormalise in BoolDecoder::read_bool
- stack-allocate motion-comp scratch in sixtap/bilinear_predict
- stack-allocate intra-prediction scratch in reconstruct_intra_mb
- criterion harness for decoder hot paths

## [0.1.3](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.2...v0.1.3) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- drop nested [workspace] block (umbrella sweep)
- fix IDCT order, Y2 DC cap, and loopfilter formulas ([#237](https://github.com/OxideAV/oxideav-vp8/pull/237))
- integrate docs/video/vp8 fixture corpus + trace harness
- look-ahead alt-ref synthesis (RFC 6386 §6 hidden frames)
- migrate to centralized OxideAV/.github reusable workflows

### Fixed

- **IDCT pass order** (`src/transform.rs`): swapped the inverse 4×4
  DCT to run the column pass first then the row pass, matching the
  RFC 6386 §14.1 C reference. The prior row-then-column ordering
  silently produced ±1 pixel drifts on residuals whose row-0 odd-
  column coefficients were non-zero (cross-pass cancellations that
  the column pass would have done first never happened, leaving a
  stale value in `out[3]`/`out[7]`/etc). Single highest-impact bug
  in the corpus: `q-low` jumps 86.91% → 100% bit-exact and
  `i-only-64x64` jumps 92.94% → 98.49%.
- **Y2 DC dequantiser cap** (`src/tables/quant.rs`): removed the
  spurious `min(.., 264)` clamp on `y2_dc_step`. Per libvpx
  `dequant_init`, only the chroma DC step is capped at 132 — Y2 DC
  is `dc_qlookup[q] * 2` uncapped. With the cap in place, qi=127
  encodings (q-high) were dequantising the 16 Y2 DC coefficients
  with a step of 264 instead of 314, giving a ~16% DC error on
  every macroblock at the high end. Removes an entire class of
  high-q reconstruction error.
- **Loop-filter parameter formulas** (`src/loopfilter.rs`,
  `FilterParams`): rewrote `for_mb` per RFC 6386 §15.4 — the prior
  implementation used `interior_limit = level >> 2` and then capped
  the `edge_limit` at `9 - sharpness`, which collapsed both
  thresholds to single-digit values for any normal level (e.g.
  level=38 → edge_limit=9 instead of 118), effectively disabling
  the filter on real content. The corrected formula is
  `interior_limit = level` (with sharpness adjust), `mbedge_limit =
  ((level + 2) * 2) + interior_limit`, `sub_bedge_limit = (level *
  2) + interior_limit`. Also added the inter-frame `hev_threshold`
  branch (`level >= 20 → 2`) that the keyframe-only formula
  omitted, exposed via the new `for_mb_typed(.., key_frame)`
  constructor.
- **Loop-filter per-MB iteration order + skip rule** (`src/decoder.rs`
  `apply_loop_filter`, `src/encoder.rs` `apply_loop_filter_enc`):
  rewrote the filter driver to walk macroblocks in raster scan and
  apply the four edge passes per-MB (left MB-v, inner sub-block-v,
  top MB-h, inner sub-block-h) with the libvpx luma-then-U-then-V
  interleave at each step. Sub-block edges are now skipped for MBs
  with no decoded coefficients AND y_mode neither `B_PRED` nor
  `SPLITMV` (the libvpx `eob_mask` shortcut from RFC 6386 §15.1).
  Tracking `has_coeffs` required new fields on `MbInfo`/`MbEncoded`
  populated during decode/encode.
- **Encoder TL-pixel defaults** (`src/encoder.rs` `gather_4x4_neighbours`,
  `gather_16x16_neighbours`, `gather_8x8_neighbours`): the encoder
  was using `tl=127` whenever above was available but left was not
  (and vice versa), while the decoder was using the libvpx-correct
  swap (above-only → left-default `129`, left-only → above-default
  `127`). Cumulative drift on the inter-frame chain when the
  encoder's reference picture diverged from the decoder's at the
  frame-edge MBs — surfaced after the loop-filter rewrite as a
  catastrophic PSNR drop on the multi-ref RDO regression.

### Corpus tier promotions (`tests/docs_corpus.rs`)

Promoted to `Tier::BitExact` (CI now gates on these):

- `q-low` (was 86.91% → 100%)
- `segment-4-partitions` (was ReportOnly → 100%)

Other notable improvements (still `ReportOnly`):

- `q-high` 12.56% → 65.83% (Y2 DC + LF formula)
- `i-only-loopfilter-high` 40.72% → 72.71% (LF formula)
- `i-only-64x64` 92.94% → 98.49% (IDCT)
- `webm-mux-vs-ivf-ivf` 92.94% → 98.49% (IDCT)
- `gradient-and-noise-128x128` 89.44% → 93.25%
- `vp8-with-loopfilter-mode-simple` 61.02% → 96.73% (LF formula)
- `golden-update-cycle` 82.86% → 92.03%
- `altref-arnr-on` 79.07% → 83.12%

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
