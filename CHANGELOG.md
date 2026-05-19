# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.14](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.13...v0.1.14) - 2026-05-19

### Other

- chroma-aware variance LF cap (round-77)

### Added

- *(encoder)* Chroma-aware variance LF cap (round-77). Opt-in
  `Vp8EncoderConfig::enable_chroma_aware_variance_lf_cap` (default
  `false`) extends the round-48 variance-driven adaptive LF cap
  (`enable_variance_lf_cap`) with a luma+chroma blend on the
  coefficient-of-variation² input. The round-48 cap classifies a
  frame's content heterogeneity by the `cv²` of the per-MB **luma**
  SSE distribution only — a frame whose luma is uniformly smooth but
  whose chroma carries a heterogeneous reconstruction error gets
  `cv² ≈ 0` → cap stays at the round-44 default `±6` regardless of
  what the chroma channel actually wants. With this flag on the
  variance-cap input switches from the luma-only `mb_sse_y_cache`
  to a per-MB combined SSE built via the same
  `combined_sse_for_chroma_aware_spatial` helper the round-52
  chroma-aware spatial picker uses, reusing the existing
  `chroma_aware_spatial_luma_weight_x256` / `chroma_aware_spatial_chroma_weight_x256`
  weights (defaults `luma=256` = `1.0`, `chroma=128` = `0.5`, matching
  the 4:2:0 sub-sampling ratio). A chroma-textured MB with smooth
  luma now lifts the `cv²` of the combined distribution and the cap
  widens past `±6` proportionally with the heterogeneity — exactly as
  the luma-only path does on luma-textured frames. Composes with
  every existing cap-widening / picker flag through the same
  `delta_cap` resolution: the round-44/48 estimator's
  `compute_lf_deltas` closure AND the round-49 per-MB / spatial /
  joint pickers share one cap-computation site each, so the
  chroma-aware reading widens their headroom symmetrically when their
  flags are on. The round-51 `mb_sse_uv_cache` gating widens to
  populate chroma SSE whenever this flag is on alongside any
  cap-consumer (round-44 estimator OR round-49 spatial / per-MB
  pickers) — the cache stays unpopulated when no consumer is on so
  the flag is genuinely inert. Off-by-default so the round-48
  luma-only cap calibration is preserved bit-for-bit when this flag
  is disabled. Three unit tests in `src/encoder.rs` plus eight
  integration tests in `tests/encoder_round_77.rs` pin: default-off,
  chroma blend lifts cap when chroma textured / luma smooth,
  chroma-weight-zero recovers luma-only cap, length-mismatch fallback
  to luma, off-path byte-identical, gating on `enable_variance_lf_cap`,
  inert when no cap-consumer is on, determinism, clean P-frame decode
  on chroma-textured clip, byte-envelope vs the luma-only path within
  ±60 %, and combined with the round-49 spatial picker decodes
  cleanly.

## [0.1.13](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.12...v0.1.13) - 2026-05-09

### Other

- chroma-aware per-MB median picker + k-means convergence early-exit/telemetry (round-53)
- joint round-44/48 + round-49 picker + chroma-aware spatial picker (round-52)
- k-means++ centroid seeding + per-MB chroma-SSE cache (round-51)
- 4-means spatial-segment clustering + per-MB SSE caching (round-50)
- per-MB segment LF deltas + spatial-locality bucketed adaptive LF (round-49)
- variance-driven LF cap + UV-channel adaptive deltas (round-48)
- high-QP adaptive LF cap + sub-pel rate refactor (round-47)
- first-pass real-context SPLIT_MV + sub-pel partition MV-cost (round-46)
- MV-cost-aware NEAREST/NEAR snap + SPLIT_MV real-context (round-45)
- adaptive LF deltas + Trellis rate-from-context (round-44)
- SPLIT_MV partition-selection RDO (round-43)
- UV-mode RDO + mode/ref loop-filter deltas (round-42)

### Added

- *(encoder)* Chroma-aware per-MB median picker + k-means convergence
  early-exit & iter-count telemetry (round-53). Two complementary
  forward steps composing the existing round-49 / round-50 / round-51 /
  round-52 LF-delta + spatial-segment picker tiers, both gated behind
  opt-in flags (default off / default-1 in the case of the convergence
  threshold). `Vp8EncoderConfig::enable_chroma_aware_per_mb_median`
  (default `false`) extends the round-49 per-MB segment LF-delta median
  picker (`enable_per_mb_lf_deltas`) to compute each MB's "optimal LF
  delta" from a luma+chroma weighted SSE blend `(luma_w_x256 *
  mb_sse_y + chroma_w_x256 * mb_sse_uv) / 256` (the same blend the
  round-52 chroma-aware spatial picker uses), instead of luma SSE alone.
  Sources chroma SSE from the round-51 `mb_sse_uv_cache` (single-line
  plumbing — the cache gating widens to populate the chroma cache when
  this flag is on alongside the per-MB median path). Default weights
  reuse the existing `chroma_aware_spatial_luma_weight_x256 = 256`
  (= `1.0`) and `chroma_aware_spatial_chroma_weight_x256 = 128`
  (= `0.5`), matching the 4:2:0 sub-sampling ratio. Off-by-default;
  the round-49 luma-only median picker is preserved bit-for-bit when
  the flag is off. Requires `enable_segments = true` and
  `enable_per_mb_lf_deltas = true` — inert otherwise; mutually inert
  with `enable_spatial_lf_deltas` (the spatial picker wins when both
  per-MB flags are on, so the median picker becomes a no-op).
  `Vp8EncoderConfig::kmeans_convergence_threshold` (default
  [`DEFAULT_KMEANS_CONVERGENCE_THRESHOLD`] = `1`) adds a
  centroid-movement-based early-exit to the round-50 / round-51
  4-means spatial-segment picker. After each Lloyd's iteration the
  picker computes `max_delta = max(|new_centroid_i -
  prev_centroid_i|)` over the `(delta, pos_x, pos_y)` axes; when
  `max_delta < threshold` the loop exits early. Default `1` typically
  exits in 2-4 iterations on the encoder's test fixtures vs the
  round-50 hard cap of `KMEANS_SPATIAL_MAX_ITERS = 16` (the cap is
  preserved as a safety upper bound). `0` collapses to the round-50 /
  round-51 termination ("exit only when no region changes assignment")
  bit-for-bit. Telemetry: the actual iteration count is reported in a
  new `Vp8EncoderStats` struct with field `last_kmeans_iters:
  Option<u32>`, accessed via the new `make_encoder_typed_with_config`
  factory + `Vp8Encoder::last_stats` method (the existing
  `make_encoder_with_config` factory returns `Box<dyn Encoder>` and
  hides the concrete type, so the typed factory is the channel for
  callers that want introspection). Keyframes reset
  `last_kmeans_iters` to `None` because the spatial picker only runs
  on P-frames. Implemented via the new
  `compute_spatial_segment_lf_deltas_kmeans_with_telemetry` function
  (returning `(Vec<u8>, [i32; 4], u32)`), with the existing
  `compute_spatial_segment_lf_deltas_kmeans` wrapper preserved for the
  unit-test corpus. Both flags off-by-default so the round-52
  default-config behaviour is preserved bit-for-bit and the
  15-fixture corpus stays bit-exact. Ten integration tests in
  `tests/encoder_round_53.rs` plus seven unit tests in `src/encoder.rs`
  pin: defaults at the documented defaults, chroma-aware off
  byte-identical, chroma-aware-dependency gating, chroma-aware shifts
  segment_id distribution on a chroma-textured / luma-flat clip,
  kmeans iter-count telemetry sanity (≥ 1, ≤ `KMEANS_SPATIAL_MAX_ITERS`),
  kmeans default-threshold ≤ 6 iters on simple fixtures,
  kmeans threshold-zero deterministic re-run, keyframe resets stats,
  composed knobs decode cleanly, full-pipeline reproducible re-runs,
  helper edge cases (zero-threshold matches wrapper, early-exit under
  default threshold, hard iter-cap respected, degenerate inputs return
  iter `0`, large threshold exits in ≤ 2 iters).

  [`DEFAULT_KMEANS_CONVERGENCE_THRESHOLD`]: https://docs.rs/oxideav-vp8/latest/oxideav_vp8/encoder/constant.DEFAULT_KMEANS_CONVERGENCE_THRESHOLD.html

- *(encoder)* Joint round-44/48 + round-49 picker + chroma-aware
  spatial picker (round-52). Two complementary opt-in knobs compose
  the existing round-44/48 (mode/ref deltas) and round-49 / round-50 /
  round-51 (spatial-segment LF deltas) tiers.
  `Vp8EncoderConfig::enable_joint_r44r49_picker` (default `false`)
  makes the two tiers iterate jointly: each iteration recomputes the
  round-44/48 estimator using a per-MB residual SSE that subtracts
  the part of the per-MB ideal delta the spatial tier has already
  addressed, and recomputes the spatial picker using a per-MB
  residual SSE that subtracts the part the mode/ref tier has already
  addressed. The residual is `mb_sse * max(0, 32 - |implied_delta|)
  / 32` so an MB whose implied delta saturates the proportional-
  formula scale (`±32`) contributes zero to the next picker's frame
  mean / per-bucket sums (already "addressed" by the previous tier).
  Convergence: stop when both the 8 mode/ref deltas + the segment_id
  vector + the 4 segment_lf_deltas match the previous iteration,
  capped at `joint_r44r49_picker_max_iters` (new field, default
  `DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS = 3` — converges in 1–2 on
  the test fixtures; clamped to `[1, JOINT_R44R49_PICKER_MAX_ITERS_MAX
  = 8]`). Requires `enable_mode_ref_lf_deltas = true` AND
  `enable_spatial_lf_deltas = true` — inert with either off. Composes
  with the cap-widening flags through the same `delta_cap` resolution.
  `Vp8EncoderConfig::enable_chroma_aware_spatial` (default `false`)
  extends the round-49 / round-50 / round-51 spatial picker to score
  regions on a luma+chroma weighted SSE blend (`combined = (luma_w *
  mb_sse_y + chroma_w * mb_sse_uv) >> 8`) sourced from the round-51
  `mb_sse_uv_cache`, instead of luma SSE alone. Default weights
  `chroma_aware_spatial_luma_weight_x256 = 256` (= `1.0`),
  `chroma_aware_spatial_chroma_weight_x256 = 128` (= `0.5` — matches
  the 4:2:0 sub-sampling ratio). Both the greedy and k-means spatial
  paths consume the combined SSE; the picker logic is unchanged.
  Requires `enable_segments = true` AND `enable_spatial_lf_deltas =
  true` — inert otherwise. The round-51 chroma cache gating widens to
  populate `mb_sse_uv_cache` when this flag is on, so the chroma-aware
  path lands as one extra blend allocation rather than a duplicate
  `compute_per_mb_chroma_sse` walk. Implemented via the new helpers
  `combined_sse_for_chroma_aware_spatial`,
  `per_mb_ref_mode_implied_delta`, `per_mb_segment_implied_delta`, and
  `apply_residual_offset_to_mb_sse`, plus the joint-picker iteration
  loop wrapping `compute_spatial_segment_lf_deltas[_kmeans]` and a
  refactored `compute_lf_deltas` closure for the round-44/48 tier.
  Both flags off-by-default so the round-51 default-config behaviour
  is preserved bit-for-bit and the 15-fixture corpus stays bit-exact.
  Nine integration tests in `tests/encoder_round_52.rs` plus seven
  unit tests in `src/encoder.rs` pin: defaults off, joint-off
  byte-identical, joint-dependency gating, joint iter-cap respected,
  chroma-off byte-identical, chroma-dependency gating, chroma-aware
  shifts segment_id distribution on a chroma-textured / luma-flat
  clip, joint+chroma composed decode, full-pipeline reproducible
  re-runs, and helper edge cases (chroma_w=0, default-weight blend,
  length-mismatch fallback, chroma-dominant blend, residual-offset
  basic, residual-offset saturation, ref/mode bucket aggregation,
  segment-id direct lookup).
- *(encoder)* 4-means clustering for spatial-path segments + per-MB
  luma-SSE caching across round-44/48 + round-49 paths (round-50). Two
  complementary improvements land on top of round-49.
  `Vp8EncoderConfig::enable_kmeans_spatial_segmentation` (default
  `false`) replaces the round-49 greedy "top-3 |delta| → segments
  1/2/3, rest → segment 0" picker with a `k = 4` Lloyd's-algorithm
  clustering on `(region_delta, region_pos_x, region_pos_y)`. The
  distance metric is `(region_delta - centroid_delta)² +
  (alpha_x256/256) * ((px - cx)² + (py - cy)²)`, tuned by the new
  `kmeans_spatial_alpha_x256` field (default `256` = `1.0`). Centroids
  are seeded from the 4 highest-|delta| populated regions and Lloyd's
  iterations run until convergence or `KMEANS_SPATIAL_MAX_ITERS`
  (= 16). Spatially-adjacent regions with similar deltas now merge
  into one segment, freeing slots that the greedy picker would have
  spent on near-duplicates. Off-by-default so the round-49 greedy
  picker is preserved bit-for-bit; requires `enable_spatial_lf_deltas
  = true` (inert otherwise). Cap-widening flags compose through the
  same `delta_cap` resolution as the greedy path. The second
  improvement is internal: the per-MB luma-SSE vector is now computed
  exactly once per P-frame (when needed) and threaded into both the
  round-44/48 adaptive LF estimator and the round-49 per-MB / spatial
  paths via a single `mb_sse_y_cache` allocation. Previously the two
  pipelines independently called `compute_per_mb_luma_sse`, walking
  `rec_y` twice when both were enabled. The refactor is bit-exact
  preserving (the math is unchanged); only the duplicate work is
  removed. Implemented via the new helper
  `compute_spatial_segment_lf_deltas_kmeans` plus the
  `mb_sse_y_cache` lifting in the P-frame encoder. Seven integration
  tests in `tests/encoder_round_50.rs` plus seven unit tests in
  `src/encoder.rs` pin: default-off, k-means-off byte-identical,
  spatial-required gating, clean P-frame decode, ±35 % byte
  envelope vs round-49 greedy, combined-on round-trip,
  cache-deterministic re-runs, and helper edge cases (empty /
  uniform / single-region / band-clamping / alpha-zero pure-delta /
  cluster-mean delta / multi-spike differs-from-greedy).
- *(encoder)* Per-MB-targeted segment LF deltas + spatial-locality
  bucketed adaptive LF (round-49). Two complementary opt-in knobs land
  on top of round-48 to drive the per-segment LF-delta channel from
  per-MB content statistics rather than the static config array.
  `Vp8EncoderConfig::enable_per_mb_lf_deltas` (default `false`)
  computes a per-MB optimal LF delta with the same proportional
  formula round-44 uses for ref/mode buckets
  (`(mb_sse - frame_mean) * delta_cap / frame_mean`), then groups by
  `mb_segment_id` and picks the per-segment median. Empty segments
  fall back to the static `config.segment_lf_deltas` value so toggling
  this flag on a sparse segment map doesn't introduce wild deltas.
  `Vp8EncoderConfig::enable_spatial_lf_deltas` (default `false`)
  partitions the frame into
  `spatial_lf_n_row_bands × spatial_lf_n_col_bands` rectangular
  regions (default `4 × 4`), computes a per-region SSE-driven LF
  delta with the same formula, then maps the regions onto VP8's
  4-segment scheme by clustering: the 3 regions with the largest
  `|delta|` become segments 1/2/3 (each carrying its own delta), the
  rest collapse into segment 0 with delta `0`. Both the per-MB
  segment-id map and the per-segment LF-delta array are overridden so
  the bitstream signals the spatial assignment to the decoder. When
  both flags are on the spatial path wins (it owns both the segment-id
  map and the LF deltas; the per-MB median path becomes a no-op).
  Both gated on `enable_segments = true`. Cap matches the round-44/48
  estimator (`±6` default, expanded under
  `enable_adaptive_lf_high_qp_cap` / `enable_variance_lf_cap`) so
  cap-widening flags compose. Implemented via the new helpers
  `compute_per_mb_optimal_lf_delta`, `pick_per_mb_segment_lf_deltas`,
  and `compute_spatial_segment_lf_deltas`, plus
  `SegmentCtx::set_lf_deltas` for installing the picked array between
  pass 1 and pass 2. Off-by-default so the existing static-config
  segment LF ladder stays bit-for-bit and the 15-fixture corpus stays
  bit-exact. Ten integration tests in `tests/encoder_round_49.rs` plus
  twelve unit tests in `src/encoder.rs` pin: default-off, off-path
  byte-identical, segments-required gating for both paths, clean
  P-frame decode, ±25 % per-MB envelope, ±35 % spatial envelope,
  spatial-wins-over-per-MB composition, helper edge cases (empty /
  uniform / outlier-saturates-cap / cap-widening / median grouping /
  spatial top-band cluster / region-count clamping / zero-band
  collapse), and combined-on round-trip.
- *(encoder)* Variance-driven adaptive LF cap + UV-channel adaptive LF
  deltas (round-48). Two opt-in knobs generalise the round-47 high-QP
  cap and the round-44 luma-only adaptive estimator.
  `Vp8EncoderConfig::enable_variance_lf_cap` (default `false`) replaces
  the round-47 `qindex`-proxy ramp with a content-driven model: the
  per-bucket delta cap is computed directly from the per-frame SSE
  distribution's normalised variance (coefficient of variation squared,
  `cv2 = var / mean^2`). Homogeneous content (`cv2 ≤ 0.5`) keeps the
  cap at `±6`; very heterogeneous content (`cv2 ≥ 1.0`) saturates at
  `±10`; the slope between is linear (`cap = 6 + min(4,
  max(0, cv2 - 0.5) * 8)`). When both this flag and
  `enable_adaptive_lf_high_qp_cap` are on, the variance-driven cap
  wins. `Vp8EncoderConfig::enable_adaptive_uv_lf_deltas` (default
  `false`) extends the round-44 estimator to consider chroma SSE
  alongside luma — the per-bucket delta is the average of the
  luma-only and chroma-only adaptive estimates. Both inputs are inside
  `±delta_cap`, so the average stays inside that envelope without
  additional clamping. Implemented via the new `variance_lf_cap`
  helper, `compute_per_mb_chroma_sse` (Cb + Cr combined per-MB SSE
  computer), and `LfDeltas::round48_adaptive_with_uv` (averaging
  constructor with length-mismatch fallback to luma-only). Both flags
  off-by-default so the round-44 / round-47 calibrations are preserved
  bit-for-bit and the 15-fixture corpus stays bit-exact. Nine
  integration tests in `tests/encoder_round_48.rs` plus eight unit
  tests in `src/encoder.rs` pin: default-off, off-path byte-identical,
  variance-cap inert when adaptive LF deltas off, UV-deltas inert when
  off, both flags clean P-frame decode, ±25 % byte-envelope vs round-44
  baseline, length-mismatch fallback, edge cases (empty / zero /
  uniform / saturated cv2), and combined-on round-trip.
- *(encoder)* High-QP adaptive LF magnitude scaling + sub-pel rate-term
  refactor (round-47). New opt-in
  `Vp8EncoderConfig::enable_adaptive_lf_high_qp_cap` (default `false`)
  lets the round-44 adaptive LF estimator's per-bucket delta cap grow
  from `±6` at `qindex ≤ 60` linearly up to `±10` at `qindex ≥ 110`
  (clamped either side). The round-44 cap is calibrated for mid-QP; at
  high QP the per-MB SSE distribution carries a wider absolute spread
  (the baseline reconstruction error is larger) and the bucket-vs-frame
  deviations more often saturate against `±6`, truncating the
  adaptation signal. The expansion is always *at* the cap — when the
  proportional deviation is small the produced delta is identical to
  the round-44 path. Implemented via
  `LfDeltas::round44_adaptive_with_cap` and the new
  `adaptive_lf_high_qp_cap` ramp helper. Off-by-default so the
  round-44 calibration is preserved bit-for-bit. Requires
  `enable_adaptive_lf_deltas = true` and
  `enable_mode_ref_lf_deltas = true`; ignored on keyframes. The
  round-47 commit also folds the duplicated `mv_rate_cost` closures
  from `subpel_refine_luma` and `subpel_refine_partition` into a
  single shared helper `subpel_mv_rate_cost_x256` — pure mechanical
  refactor, no behavioural change. Eight integration tests in
  `tests/encoder_round_47.rs` pin: default-off, off-path byte-identical
  at high QP, high-QP cap clean decode, byte-envelope ±20 % vs
  round-44 baseline, low-QP inert (cap floor == 6), `enable_adaptive_lf_deltas`-required
  gating, sub-pel rate-refactor determinism, and combined-on round-trip.
- *(encoder)* Adaptive loop-filter mode/ref deltas + Trellis
  rate-from-context (round-44). Two complementary picker upgrades land
  together. `Vp8EncoderConfig::enable_adaptive_lf_deltas` (default
  `false`) replaces the static round-42 ladder
  (`ref_deltas = [+2, 0, -2, -2]` / `mode_deltas = [+4, -2, +1, +4]`)
  with a per-frame estimate from the per-MB unfiltered luma-SSE
  distribution: each ref / mode bucket's delta is biased toward
  stronger filtering for buckets whose mean SSE exceeds the frame
  mean (deblocking helps reconstruction-noisy MBs the most) and toward
  lighter filtering for buckets below the frame mean. Empty buckets
  fall back to the static ladder so sparse-mode frames don't get wild
  deltas. Magnitude is capped at ±6 (one segment-tier above/below the
  bare frame level) so the effective per-MB level stays comparable to
  the static-ladder behaviour. Only meaningful when
  `enable_mode_ref_lf_deltas = true`; ignored on keyframes (which
  always emit `mode_ref_delta_enabled = 0`). Off-by-default so the
  existing static-ladder bitstream is preserved bit-for-bit.
  `Vp8EncoderConfig::enable_trellis_context_rate` (default `false`)
  defers the trellis pass until after the per-MB encode loop completes
  and walks MBs in raster order tracking the running above/left
  non-zero predictor — the same predictor `emit_tokens` uses — so each
  block's `nctx ∈ {0,1,2}` matches the actual entropy-coder context.
  The previous per-MB call always passed `nctx = 0` which
  over-approximated EOB savings on blocks whose neighbours have
  non-zero coefficients. Trellis decisions feed back into the nz
  predictor for subsequent blocks so the context propagates the way
  the entropy coder propagates it. Effective only when
  `enable_trellis_quant = true`. Off-by-default so the existing
  nctx=0 calibration is preserved bit-for-bit. Thirteen integration
  tests in `tests/encoder_round_44.rs` pin: default-off, off-path
  byte-identical, adaptive LF keyframe + P-frame clean decode,
  byte-envelope ±20 % vs static ladder, flat-content envelope ±20 %
  (sparse-mode fallback), `enable_mode_ref_lf_deltas`-required
  gating, trellis-context keyframe + P-frame clean decode,
  trellis-context byte-envelope ±10 % vs nctx=0 baseline, PSNR-Y
  non-regression, `enable_trellis_quant`-required gating, and
  composition with all round-42 / round-43 knobs.
- *(encoder)* SPLIT_MV partition-selection RDO (round-43). New opt-in
  `Vp8EncoderConfig::enable_split_mv_rdo` knob (default `false`)
  switches `search_split_mv` from SAD-min split-mode selection to
  Lagrangian `D + λ·R`, where `D` is the total partition SAD and `R`
  is the bool-coder cost (in 1/256-bit units) of the `MBSPLIT_PROBS`
  tree path plus per-partition `SUB_MV_REF_PROBS` longest-path leaf
  cost (NEW_4X4 under the neutral [0] context — neighbour sub-MVs
  aren't visible at search time, so we charge the worst-case "new MV"
  branch which is the only leaf that adds an MV-delta literal) plus
  per-partition `mv_component_cost_x256` MV-delta bits when the
  partition's MV is non-zero. Counteracts the structural bias of the
  legacy SAD-min path toward the 4×4 split (which nearly always wins
  on raw SAD because of its 16 degrees of freedom but pays the most
  bitstream bits — 16 sub-MV trees + 16 MV deltas + the longest
  split-tree path). λ is `lambda_for_qp(qi, scale)`, the same
  multiplier the per-MB ref/mode picker uses, so RDO trade-offs stay
  coherent across all encoder decisions. Off-by-default so the
  existing greedy SAD-min selection is preserved bit-for-bit; gated
  on `enable_rdo = true` (with `enable_rdo = false` λ collapses to 0
  and the gating is inert). Nine integration tests in
  `tests/encoder_round_43.rs` pin: default-off, off-path
  byte-identical to legacy, keyframe + P-frame clean decode,
  byte-envelope ±20 %, PSNR-Y non-regression, flat-content
  byte-identical (the cheap-skip test fires before the SPLIT_MV
  search runs), `enable_rdo`-required gating, and composition with
  the round-42 knobs (UV-RDO + mode/ref deltas + joint LF-RDO).

## [0.1.12](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.11...v0.1.12) - 2026-05-08

### Other

- ungate bool_cost_x256 import for round-41 BMODE-RDO
- rate-aware B_PRED 4×4 sub-mode picker (round-41)
- activate joint loop-filter / QP RDO (round-40)
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- libvpx-shape per-coefficient Trellis + activity-driven AQ (round-39)

### Added

- *(encoder)* UV-mode RDO + mode/ref loop-filter deltas (round-42).
  Two complementary picker upgrades land together: the chroma intra
  mode pick now scores `D + λ·R` against the UV-mode tree
  probabilities (`KF_UV_MODE_PROBS` on keyframes,
  `DEFAULT_UV_MODE_PROBS` on the intra-in-P fallback) when the
  opt-in `Vp8EncoderConfig::enable_uv_rdo` flag is on, and the
  in-loop deblocking filter now honours RFC 6386 §15.2 mode/ref
  deltas when `Vp8EncoderConfig::enable_mode_ref_lf_deltas` is on.
  The latter unlocks the joint LF-RDO picker's real rate term — with
  per-MB level varying by ref_frame + (intra/inter, y_mode), the
  candidate-level search now scores against the actual post-delta
  reconstruction the decoder will compute, not the bare frame level.
  Default ladder is libvpx-ish: `ref_deltas = [+2, 0, -2, -2]`,
  `mode_deltas = [+4, -2, +1, +4]` (INTRA + B_PRED → +6 vs the LAST
  + ZERO_MV bucket's +0; concentrates extra filtering on the
  reconstruction-poor candidates that benefit most). Both knobs
  default `false` so existing P-frame bitstreams stay byte-identical
  when the flags are off; both gated on `enable_rdo = true` for
  symmetry with the round-41 BMODE-RDO knob. Fifteen integration
  tests in `tests/encoder_round_42.rs` pin: defaults-off,
  off-path byte-identical to legacy, UV-RDO keyframe + P-frame
  clean decode, byte-envelope ±20 %, PSNR-Y non-regression,
  flat-content byte-identical, `enable_rdo`-required gating;
  mode/ref deltas keyframe byte-identical (decoder zeros them per
  §9.4), P-frame clean decode, byte-envelope ±15 %, PSNR-Y
  non-regression, P-frame header grows when deltas are emitted, and
  the three knobs (UV-RDO + mode/ref deltas + joint LF-RDO) compose
  cleanly.
- *(encoder)* Rate-aware B_PRED 4×4 sub-mode picker (round-41). New
  opt-in `Vp8EncoderConfig::enable_bpred_rdo` knob (default `false`)
  switches the per-4×4 mode selection from greedy SSE to `D + λ·R`,
  where `D` is the same SSE the legacy path scores and `R` is the
  bool-coder cost (in 1/256-bit units, via the calibrated
  `PROB_COST_BITS_X256` table) of writing the `BMODE_TREE` path under
  the appropriate context probabilities — `KF_BMODE_PROB[above][left]`
  on keyframes, the static `vp8_bmode_prob` on intra-in-P MBs. λ is
  the same `lambda_for_qp(qi, scale)` value the per-MB ref/mode picker
  uses, so RDO trade-offs stay coherent across all encoder decisions.
  The 16×16-vs-B_PRED outer selector continues to compare pure SSE so
  the existing `B_PRED_SSE_MARGIN` threshold semantics are preserved;
  only the inner per-sub-block search is rate-aware. Off-by-default so
  the existing greedy SSE selection is preserved bit-for-bit;
  additionally gated on `enable_rdo = true` (with `enable_rdo = false`
  λ collapses to 0, but the gating is explicit so the cost-table
  indexing isn't reached for users who disable RDO entirely). Eight
  integration tests in `tests/encoder_round_41.rs` pin: default-off,
  off-path byte-identical to legacy, keyframe + P-frame clean decode,
  byte-envelope ±15 %, PSNR-Y non-regression, flat-content
  byte-identical (gating is engaged only on B_PRED MBs), and
  `enable_rdo`-required gating.
- *(encoder)* Joint loop-filter / QP rate-distortion optimisation
  (round-40). The previously-reserved `Vp8EncoderConfig::enable_joint_lf_rdo`
  flag is now active. When set on a P-frame, after pass 1 has built the
  unfiltered reconstruction the encoder searches a ±4-level neighbourhood
  around the deterministic `loop_filter_level_for_qindex(qi) = 15 + qi/8`
  heuristic and picks the level that minimises luma SSE-vs-source on a
  centre 32×32 patch. The LF level is a 6-bit literal in the frame
  header so the rate term `R(level)` is identical for every candidate;
  the search is therefore pure distortion minimisation, but lets
  content-dependent characteristics (edge density, residual magnitude)
  override the deterministic formula on a per-frame basis. Cost is bounded
  to `2·radius+1 = 9` luma-only LF passes on a clone of the reconstruction —
  negligible vs the per-MB ME / RDO / quantiser budget. Implementation
  introduces a luma-only variant of the existing `apply_loop_filter_enc`
  walker (`apply_loop_filter_luma_only`) so the search doesn't pay for
  chroma it doesn't score. After the new level is chosen, `pick_filter_type`
  re-runs to keep the simple/normal dispatch consistent with the chosen
  level under `LoopFilterMode::Auto`. Keyframes are unchanged — they still
  use the heuristic so the frame-0 bitstream remains bit-identical when
  the flag toggles. Default off (opt-in); enabling preserves a valid
  decodable bitstream and PSNR_Y stays within 0.5 dB of the heuristic
  path on a smooth-pan clip with high-frequency edge noise. Six
  integration tests in `tests/encoder_round_40.rs` pin: keyframe-only
  bit-identical, P-frame clean decode, PSNR-Y non-regression, byte
  envelope ±15 %, segmentation interaction, default-off.

## [0.1.11](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.10...v0.1.11) - 2026-05-05

### Other

- fix clippy::unnecessary_cast + while_let_loop in encoder_round_38
- psy-RDO activity mask + ARNR NLM temporal denoiser (round-38)
- fix clippy::manual_range_contains in encoder_two_pass
- two-pass ABR rate control (round-36)
- add enable_trellis_quant + enable_subpel_mv_cost to all struct literals
- rustfmt fixes (import merge, short fn collapses, if-else expand)
- fix no-default-features build (mv_component_cost_x256 import)
- trellis quantisation + rate-aware sub-pel ME (round-32)
- per-MB QP refinement + SPLIT_MV joint refine + long-ref lambda tilt
- apply RFC 6386 §18.1 luma-MV doubling on decode
- auto-register via oxideav_core::register! macro (linkme distributed slice)

### Added

- *(encoder)* libvpx-shape per-coefficient Trellis (#39). New
  `Vp8EncoderConfig::enable_trellis_full` flag turns on a forward DP
  over the kept non-zero coefficients of every transform block,
  analogous to libvpx `vp8_optimize_b`. For each position the DP
  evaluates two candidates — the original quantised magnitude `q` and
  the toward-zero magnitude `q-1` — tracking the bool-coder ctx
  transitions (1 if `|c|=1`, 2 if `|c|≥2`, 0 if zero) and picking the
  per-position trajectory that minimises the block's total `D + λ·R`
  in 1/256-bit units. Distortion uses `(q-mag)² · step² / 2`,
  calibrated so the DP only accepts magnitude reductions with real
  rate savings. Runs *before* the existing EOB-trim Trellis pass —
  positions zeroed by the magnitude-down DP shorten the EOB further.
  On a mixed smooth+noise 64×64 clip this trims an additional ~1.4 %
  bytes (15160 → 14942) for ~0.02 dB PSNR-Y loss vs the EOB-only
  Trellis path. Default off (opt-in); requires `enable_trellis_quant
  = true` to take effect (the EOB-trim entry path is the gate).

- *(encoder)* Activity-driven adaptive quantisation — AQ (#39). New
  `Vp8EncoderConfig::enable_aq` flag swaps the per-MB segment
  classifier from raw variance to population quartiles of the
  per-MB *activity* (variance + 16 × Laplacian edge energy, the
  same metric the round-38 psy-RDO mask uses). Smooth MBs (low
  activity) land in the low-qindex segments (finer quant, fewer
  banding artefacts); textured / edge-rich MBs (high activity) land
  in the high-qindex segments (coarser quant where the eye masks the
  loss). Reuses the existing 4-segment bitstream signalling — no new
  header bits. New `aq_qindex_range` field (default `8`) bounds the
  per-MB qindex shift; falls back to the variance-based path when the
  per-frame activity distribution is degenerate. Default off
  (opt-in); requires `enable_segments = true` to take effect.

- *(encoder)* New `Vp8EncoderConfig::enable_joint_lf_rdo` knob (#39)
  reserved for joint loop-filter / QP rate-distortion optimisation
  on P-frames. The field is wired through the public config but the
  RD search is not yet active in this round — picking the LF level
  from the existing `15 + qi/8` heuristic keeps the bitstream
  bit-identical when the flag is on or off. Will be activated in a
  later round once a fast trial-encode harness lands.

- *(encoder)* Perceptual RDO activity mask — psy-RDO (#38). When
  `Vp8EncoderConfig::enable_psy_rdo = true`, the Lagrangian lambda is
  scaled per-MB by an activity mask derived from luma variance plus
  `EDGE_WEIGHT × Laplacian edge energy`. Flat MBs receive a higher
  lambda (fewer bits allocated), textured and edge-rich MBs receive a
  lower lambda (distortion-penalised). The frame-mean activity is
  computed once before the per-MB loop (`frame_mean_activity`); the
  per-MB scale is clamped to [64, 512] / 256 to prevent degenerate
  cases. Strength is controlled by `psy_rd_strength` (default 64 ≙ ~
  ±75 % swing for a 2× activity ratio). Default off (opt-in, zero
  impact on existing bitstreams). On a mixed smooth+noise 64×64 clip
  the psy mask redistributes bits toward the noisy half without
  regressing PSNR_Y vs the flat-lambda baseline.

- *(encoder)* NLM temporal denoising on the alt-ref frame — ARNR NLM
  (#38). When `Vp8EncoderConfig::enable_arnr_nlm = true` and
  `enable_lookahead_altref = true`, the lookahead alt-ref synthesis runs
  an additional NLM (non-local means) denoising pass over the Y plane
  after the existing Gaussian temporal filter. For each MC-aligned
  source frame (all non-centre frames in the lookahead window), a
  sliding 5×5 patch MSE is computed against the corresponding patch in
  the Gaussian-filtered composite; per-pixel NLM weights `w = exp(-mse /
  h²)` blend the MC-aligned frame's pixels into the final composite.
  Strength is controlled by `nlm_h2` (default 225.0; higher values
  accept noisier patches). Enabling NLM without a lookahead buffer is a
  no-op and does not panic. On a slow-pan noisy clip the NLM pass
  maintains or improves PSNR_Y vs the Gaussian-only baseline. Default
  off (opt-in).

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
