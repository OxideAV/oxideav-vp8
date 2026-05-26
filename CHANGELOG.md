# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Added

* **VP8 encoder Phase 11: §16.4 SPLITMV per-sub-block motion-vector
  walk in the rate-distortion picker (RFC 6386 §16.4 / §17.2 / §18.1)**
  — extends the round-146 P-frame J = SAD + λ·bits picker from the
  four whole-MB `mv_ref_tree` leaves to all five (the fifth leaf,
  SPLITMV, evaluates the four §16.4 partition shapes
  (`MvPartition::TopBottom` / `LeftRight` / `Quarters` / `Mv16`) per
  MB):

  ```
  J(split, p) = sum_groups group_SAD
              + λ · (mv_ref_tree("1111") bits
                   + mvpartition_tree(p) bits
                   + sum_groups (sub_mv_ref_tree mode bits
                               + NEW4X4 §17.2 component bits))
  ```

  For each partition shape the picker evaluates each group's MV by
  combining a whole-pixel sub-block diamond search around the clamped
  `near.mvs[0]` "best" predictor (the §17 base the decoder's NEW4X4
  differential adds to) with a §16.4 `sub_mv_ref` mode pick (LEFT4X4 /
  ABOVE4X4 / ZERO4X4 / NEW4X4) priced at the §17.2 context the anchor
  sub-block's left/above neighbours produce; the lowest-J group SAD +
  mode bits wins. The lowest-J partition across the four shapes is the
  SPLITMV candidate; it wins the per-MB picker when its total J is
  strictly lower than the best whole-MB candidate's. Ties go to the
  whole-MB path (SPLITMV carries strictly more bits than ZEROMV at
  identical distortion).

  The SPLITMV reconstruction path runs through
  [`oxideav_vp8::predict_split_mv`] + [`oxideav_vp8::reconstruct_split_mv_mb`]
  (already in place since the decoder side landed), with a new
  encoder-local `transform_split_mv_mb` for the per-sub-block luma
  forward DCT (no Y2 carve-out per §14.2 page 76 — every Y sub-block
  codes coefficient 0..=15 under the Y1 quantiser, mirroring B_PRED).
  The §13.3 token-emit pass routes SPLITMV MBs through
  `encode_mb_tokens(use_bpred = true)` so the §13.3 walker skips block
  24 and threads the §13.1 / §20.16 predictor contexts the same way
  B_PRED does. The §15 loop-filter geometry records SPLITMV MBs with
  `y_mode = B_PRED` so the §15.1 "filter internal edges" rule fires
  per spec.

  On the wire the encoder emits the §16.2 "1111" path (4 high-prob
  true bools against `mv_ref_probs`), then the §16.4
  `mvpartition_tree` partition id (1..=3 bool reads against
  `MV_PARTITION_PROBS`), then for each partition group: the §16.4
  `sub_mv_ref_tree` mode at the group anchor's left/above context, and
  (NEW4X4 only) the §17.2 row + column component differential written
  against the default `MvContexts`. SPLITMV neighbours feed the §20.11
  per-sub-block lookups via `MbInfo::is_split = true` +
  `MbInfo::split_mvs = Some([Mv; 16])`, so subsequent census + neighbour
  walks see the same per-sub-block detail the decoder will reconstruct.

  Validated end-to-end on a new `tests/encoder_pframe_splitmv.rs`
  test: a 2-frame I+P sequence with a **divergent per-quadrant
  translation** (each 16×16 MB's four 8×8 quadrants each shift by
  their own whole-pixel vector — TL `(+2, +2)`, TR `(-2, +2)`, BL
  `(-2, -2)`, BR `(+2, -2)`). No single whole-MB MV can simultaneously
  align all four quadrants, so the §16.4 `Quarters` partition (four
  independent 2×2 groups) cleanly wins. The picker emits
  **16 of 16 SPLITMV MBs** and the self-decode Y-PSNR clears **39.65 dB**
  at `yac_qi = 32`. The round-146 uniform-translation test still
  validates: the picker now emits **15 of 24 NEARESTMV MBs + 9 of 24
  SPLITMV MBs** (was 23 of 24 NEARESTMV + 1 of 24 NEWMV) with the
  self-decode Y-PSNR **48.43 dB** (was 48.14 dB — modest gain). The
  round-145 quarter-pixel test now picks 2 of 16 NEWMV + 1 of 16
  SPLITMV at **48.93 dB**, the round-144 half-pixel test still picks 1
  of 16 NEWMV at **58.19 dB**, and the round-143 whole-pixel
  translated-feature test picks 1 of 16 NEWMV + 1 of 16 SPLITMV at
  **50.34 dB** — the SPLITMV picker is selectively engaged on content
  that benefits and stays out of the way on whole-MB-friendly content.

  Test infrastructure: the four pre-existing P-frame test parsers
  (`encoder_pframe_{newmv, halfpel, qpel, nearestmv}.rs`) all gained a
  `match InterMode::Split` arm that drives through
  [`oxideav_vp8::decode_split_mv`] so the encoder's SPLITMV emission
  round-trips back to the same `split_mvs[15]` (the §16.3 `MbInfo::mv`
  convention) the encoder recorded. Assertions that previously pinned
  a specific count of NEWMV / NEARESTMV MBs now accept either the
  whole-MB path or its SPLITMV equivalent (a SPLITMV neighbour can
  propagate a half-/quarter-pixel NEWMV vector through LEFT4X4 /
  ABOVE4X4 at much lower bit cost). The flat-scene picker assertion
  now checks the resolved per-MB MV is `(0, 0)` regardless of mode
  rather than the mode itself, since on a flat scene with all-intra
  neighbours the §16.3 `mv_ref_probs[0] = 7` makes "0" cost ~5 bits
  while the SUBMVREF_LEFT_ABOVE_ZED LEFT4X4 path costs ~0.3 bits per
  group — SPLITMV with all-LEFT4X4 sub-block modes can legitimately
  underrun ZEROMV's bit cost.

  No new public surface; the picker change is internal to
  `encode_p_frame_zero_mv`. `EncodeError::UnsupportedInterMode` is
  reserved for forward-compatibility (the round-147 picker handles all
  five §16.2 leaves). Tests: 474 → 475 (+1 in
  `encoder_pframe_splitmv.rs`).

  GOLDEN / ALTREF source selection, multi-partition inter, and the
  per-MB §9.4 mode / ref delta layer remain follow-up rounds.

  [`oxideav_vp8::predict_split_mv`]: oxideav_vp8::predict_split_mv
  [`oxideav_vp8::reconstruct_split_mv_mb`]: oxideav_vp8::reconstruct_split_mv_mb
  [`oxideav_vp8::decode_split_mv`]: oxideav_vp8::decode_split_mv

* **VP8 encoder Phase 11: §16.2 NEARESTMV / NEARMV candidates in the
  rate-distortion picker (RFC 6386 §16.2 / §16.3)** — widens the
  round-145 P-frame J = SAD + λ·bits picker from {ZEROMV, NEWMV} to
  all four whole-MB `mv_ref_tree` leaves by also scoring the §16.3
  census-derived `near.mvs[1]` (NEARESTMV) and `near.mvs[2]` (NEARMV)
  candidates. The two new candidates are clamped through the same
  per-MB `MvClampRect` the decoder's `resolve_inter_mb_mv` uses, then
  scored through the §18.3 sixtap-aware `mb_luma_sad_at_mv` evaluator
  (neighbour MVs can land at any §17 quarter-pixel position). Tie-break
  order is bit-cost-ascending — ZEROMV ≻ NEARESTMV ≻ NEARMV ≻ NEWMV
  — so when two candidates produce equal SAD the lower-bit
  `mv_ref_tree` path wins. A NEARESTMV / NEARMV whose clamped MV is
  `(0, 0)` is dropped (ZEROMV uses strictly fewer bits at the same
  SAD, so emitting one would waste a bit); NEWMV likewise drops on a
  `(0, 0)` search result or an out-of-§17.1-range differential.

  No new public surface: the picker change is internal to
  `encode_p_frame_zero_mv`. `EncodeError::UnsupportedInterMode` now
  only surfaces on a resolved `SPLITMV` (still deferred); its `Display`
  message updated to list the four supported leaves.

  Validated end-to-end on a new `tests/encoder_pframe_nearestmv.rs`
  test: a 2-frame I+P sequence with a uniform whole-pixel translation
  `(+4, +8)` luma px of a high-frequency-content plane. With the
  extended picker the first MB to detect motion emits NEWMV; the §16.3
  census then propagates that vector into subsequent MBs' nearest slot
  via the left / above-left / above neighbour walk. The picker emits
  **23 of 24 NEARESTMV MBs and 1 of 24 NEWMV MBs** (the seed) and the
  self-decode Y-PSNR clears **48.14 dB** at `yac_qi = 4`. A second
  flat-scene test pins that NEARESTMV / NEARMV / NEWMV are not emitted
  when ZEROMV ties on SAD (`mv_ref_tree` bit-cost-ascending tie-break
  must hold). The round-145 quarter-pixel test still picks 9 of 16
  quarter-pixel-only NEWMV MBs at **48.85 dB**, the round-144
  half-pixel test still picks 3 of 16 half-pixel-grid NEWMV MBs at
  **56.80 dB**, and the round-143 whole-pixel test still picks 4 of
  16 whole-pixel NEWMV MBs at **50.34 dB** — the existing tests'
  NEWMV emissions did not flip to NEARESTMV (each scene's neighbour
  census does not produce a useful nearest candidate for the NEWMV
  MBs in question, so the bit-cost-ascending tie-break protects the
  half- / quarter-pel codepaths from drift).

* **VP8 encoder Phase 11: §18.3 quarter-pixel motion-search refinement
  (RFC 6386 §17.1 / §18.3)** — extends the round-144 P-frame picker to
  follow its `half_pixel_refine_luma` post-pass with a
  `quarter_pixel_refine_luma` second post-pass that probes the 8
  quarter-pixel offsets (±`QUARTER_PIXEL_STEP` on each of the row / col
  axes around the half-pixel center, excluding the center) and keeps
  whichever 16×16 luma SAD is smallest. Each quarter-pixel candidate is
  evaluated through the §18.3 six-tap synthesis (`filter_block_4x4`
  under the version=0 bicubic tap-set), indexed by
  `(stored_luma_mv(mv) & 7)` — a §17 quarter-pixel offset of magnitude
  1 selects the `1/4`-position filter row (`{ 2, -11, 108, 36, -8, 1 }`)
  or `3/4`-position row (the reverse) depending on the parity of the
  existing half-pixel component, distinct from the `1/2` symmetric row
  exercised by the round-144 half-pixel refinement. Tie-breaks prefer
  the half-pixel center — fewer §17.2 component bits to code. §17.1
  clamping is applied per candidate: a quarter-pixel offset that walks
  past `±1023` collapses back onto an already-evaluated MV and is
  skipped.

  Public API additions:
  * `oxideav_vp8::quarter_pixel_refine_luma(reference, mb_col, mb_row,
    src_y, half_pixel_center) -> SearchResult` — the §18.3 quarter-
    pixel refinement entry, expected to be called on the result of a
    `half_pixel_refine_luma` post-pass.
  * `oxideav_vp8::QUARTER_PIXEL_STEP = 1` — one quarter-pixel step in
    §17 quarter-pixel units (per §17.1, V is a signed integer in
    quarter-pixels of luma displacement). After §18.1 doubling this
    maps to the §18.3 eighth-pixel fraction `2` (`1/4` tap row).

  Validated end-to-end on a new `tests/encoder_pframe_qpel.rs` test: a
  2-frame I+P sequence whose P-frame source is the §18.3 sixtap
  synthesis of the I-frame at MV (0, `+QUARTER_PIXEL_STEP`) — a
  +0.25 px horizontal shift fundamentally unreachable from a half-
  pixel-only descent. With the quarter-pixel refinement the picker
  emits 9 of 16 quarter-pixel-only NEWMV MBs (mv & 1 != 0) and the
  self-decode Y-PSNR clears **48.85 dB** at `yac_qi = 4`. The round-144
  half-pixel test (`encoder_pframe_halfpel.rs`) still picks the same
  3 of 16 half-pixel-grid NEWMV MBs and lands the same **57.0 dB**
  Y-PSNR — the tie-break ("equal SAD ⇒ keep the half-pixel center")
  protects the half-pixel codepath from drift. The round-143 whole-
  pixel translation test (`encoder_pframe_newmv.rs`) likewise still
  picks 4 of 16 whole-pixel NEWMV MBs at **50.34 dB**. Five new
  `motion_search.rs` unit tests pin: flat-source tie-break, exact
  quarter-pixel shift recovery (using a stepped high-frequency plane
  since a linear ramp degenerates under sixtap `u8` rounding), descent
  never increases SAD, refinement at a whole-pixel center, §17.1 clamp
  safety at the boundary.

* **VP8 encoder Phase 11: §18.3 half-pixel motion-search refinement
  (RFC 6386 §17.1 / §18.3)** — extends the round-143 P-frame picker to
  follow its whole-pixel `small_diamond_search_luma` descent with a
  `half_pixel_refine_luma` post-pass that probes the 8 half-pixel
  offsets (±`HALF_PIXEL_STEP` on each of the row / col axes around the
  whole-pixel center, excluding the center) and keeps whichever 16×16
  luma SAD is smallest. Each half-pixel candidate is evaluated through
  the §18.3 six-tap synthesis (`filter_block_4x4` under the version=0
  bicubic tap-set the encoder commits to in its frame tag), so a
  sub-pixel MV the picker picks is a SAD the decoder reproduces
  bit-for-bit. Tie-breaks prefer the whole-pixel center — fewer §17.2
  component bits to code. §17.1 clamping is applied per candidate: a
  half-pixel offset that walks past `±1023` collapses back onto an
  already-evaluated MV and is skipped.

  Public API additions:
  * `oxideav_vp8::half_pixel_refine_luma(reference, mb_col, mb_row,
    src_y, whole_pixel_center) -> SearchResult` — the §18.3 half-pixel
    refinement entry, expected to be called on the result of a
    `small_diamond_search_luma` descent.
  * `oxideav_vp8::mb_luma_sad_at_mv(reference, mb_col, mb_row, src_y,
    mv)` — the §18.3 sixtap-aware SAD evaluator (works at any §17.1
    MV, including half- and quarter-pixel positions). Exposed publicly
    so a future hex / square / quarter-pel search shape can reuse it.
  * `oxideav_vp8::HALF_PIXEL_STEP = 2` — one half-pixel step in §17
    quarter-pixel units (after §18.1 doubling this maps to the §18.3
    symmetric half-pixel tap row, `{ 3, -16, 77, 77, -16, 3 }`).

  Validated end-to-end on a new `tests/encoder_pframe_halfpel.rs` test:
  a 2-frame I+P sequence whose P-frame source is the §18.3 sixtap
  synthesis of the I-frame at MV (0, `+HALF_PIXEL_STEP`) — a +0.5 px
  horizontal shift that is fundamentally unreachable from a
  whole-pixel-only descent. With the refinement the picker emits 3 of
  16 half-pixel-grid NEWMV MBs and the self-decode Y-PSNR clears
  **57.0 dB** at `yac_qi = 4`. The round-143 whole-pixel translation
  test still picks the same 4 of 16 whole-pixel NEWMV MBs and lands the
  same **50.3 dB** Y-PSNR at `yac_qi = 32` — the tie-break ("equal SAD
  ⇒ keep the whole-pixel center") protects the whole-pixel codepath
  from drift. Five new `motion_search.rs` unit tests pin: flat-source
  tie-break, exact half-pixel shift recovery, descent never increases
  SAD, `mb_luma_sad_at_mv` ≡ `mb_luma_sad_at_whole_mv` at whole-pixel,
  §17.1 clamp safety at the boundary.

* **VP8 encoder Phase 11: consume motion search into per-MB ZEROMV/NEWMV
  rate-distortion pick (RFC 6386 §16.2 / §16.3 / §17 / §18)** — wires
  the round-142 `motion_search` primitive into
  `encode_p_frame_zero_mv`'s per-MB loop, so the encoder now picks
  between `ZEROMV` and whole-pixel `NEWMV` against `LAST` based on a
  non-normative `J = SAD + lambda * bits` trade. Per MB:
  * A `small_diamond_search_luma` descent runs against the clamped
    §16.3 "best" predictor (the running `find_near_mvs[CNT_BEST]`),
    bounded at 8 iterations.
  * `J_zero = SAD_at_(0,0) + lambda * bits(ZEROMV path)` and
    `J_new = SAD_at_searched_mv + lambda * (bits(NEWMV path) + §17
    component bits)` are compared; lower J wins, ties to ZEROMV. The
    NEWMV differential is `chosen_mv - clamp_mv(near.mvs[0])`, exactly
    matching the decoder's `resolve_inter_mb_mv` add. A differential
    that wraps outside `[-1023, +1023]` is treated as `+inf` cost so
    that candidate is dropped.
  * `lambda` reuses the keyframe RD picker's `q^2 / 32` shape against
    the luma AC dequant factor.
  * On NEWMV the encoder emits the §16.2 `mv_ref_tree` path "1110"
    against the §16.3 census-derived probs, then the §17.2 MV
    differential via the new public `write_mv` (against the §17.2
    default `MvContexts` — `mv_prob_update()` is still emitted with
    every F-gate = 0 so the decoder reads at the same defaults).
  * The §18 prediction at a non-zero whole-pixel MV is the §18.2 /
    §18.3 whole-pixel copy (fractional bits are zero ⇒ no sub-pixel
    filter pass runs).

  Public API additions:
  * `oxideav_vp8::write_mv_component(enc, ctx, value)` — §17.1
    `read_mvcomponent` inverse on the production `BoolEncoder`.
  * `oxideav_vp8::write_mv(enc, contexts, mv)` — §17.2 `read_mv`
    inverse, row-then-column.
  * `oxideav_vp8::mv_component_bits(ctx, value)` and
    `oxideav_vp8::mv_bits(contexts, mv)` — fractional-bit cost of
    a §17 MV / component against a `MvContext` (`-log2(P(bit))`
    accumulator, mirrors the emit control flow exactly so the cost
    equals the bits the real pass will emit modulo per-partition
    renormalisation).
  * New `EncodeError::UnsupportedInterMode { mode }` for the picker
    contract — non-NEWMV / non-ZEROMV resolved modes surface as an
    error rather than panicking, so a future picker can roll out
    incrementally.

  Validation: a new `tests/encoder_pframe_newmv.rs` integration test
  encodes a 2-frame I+P sequence with a clean +4-pixel diagonal
  translation of a 16×16 feature square. The encoder picks NEWMV for
  4 of 16 macroblocks (the 4 the feature crosses) and the self-decode
  Y-plane PSNR clears 50 dB at `yac_qi = 32` (vs. ~30–35 dB the
  ZEROMV-only path would deliver on the same scene). A second test
  pins that the picker stays on ZEROMV when no motion is possible
  (flat scene). The existing 10-frame stream + 64×64 I+P slow-drift
  tests still pass — the picker prefers ZEROMV on low-motion content.

  Out of scope (later rounds): half- / quarter-pel refinement (§18.3),
  `NEARESTMV` / `NEARMV` / `SPLITMV` mode candidates, `GOLDEN` /
  `ALTREF` source selection.

  Tests: 453 → 460 (+5 public-API round-trip + bit-cost monotonicity
  in `motion_vector.rs`, +2 NEWMV / ZEROMV picker tests in
  `encoder_pframe_newmv.rs`).

* **VP8 encoder Phase 11 begin: whole-pixel motion-search primitive
  (RFC 6386 §17.1 / §18.1 / §20.14)** — new `crate::motion_search`
  module wires the smallest piece of infrastructure a non-zero MV
  codepath needs:
  * `block_sad_16x16(src, pred) -> u32` — pixel-wise sum of absolute
    differences for two 16×16 luma blocks.
  * `LumaRef<'_> { plane, stride, width, height }` — a borrow of one
    reference frame's luma plane (bundled into a single argument so
    the search functions stay under clippy's `too_many_arguments`
    limit).
  * `SearchResult { mv: Mv, sad: u32 }` — the descent result.
  * `mb_luma_sad_at_whole_mv(reference, mb_col, mb_row, src_y, mv) ->
    u32` — fetches a 16×16 reference patch at the §17 quarter-pixel
    MV (whole-pixel only this round) via the existing §20.14
    edge-replicating `fetch_block_whole_pixel` and returns SAD vs
    source. Debug builds assert the MV is on the whole-pixel grid
    (`mv % WHOLE_PIXEL_STEP == 0`) and inside §17.1's `[-1023, +1023]`
    range.
  * `small_diamond_search_luma(reference, mb_col, mb_row, src_y,
    center, max_iters) -> SearchResult` — small-diamond (4-neighbour:
    N / S / W / E at ±1 whole pixel each) integer-pixel descent from
    `center`. Snaps `center` to the whole-pixel grid (toward zero) +
    clamps to `[MV_MIN, MV_MAX]` up front; each candidate is similarly
    clamped + snapped. Terminates when no neighbour improves the SAD
    or after `max_iters` iterations. `max_iters = 0` returns the SAD
    at the snapped/clamped center without any neighbour exploration.
  * Constants `MV_MIN: i16 = -1023`, `MV_MAX: i16 = 1023`,
    `WHOLE_PIXEL_STEP: i16 = 4`.

  Nothing in this module touches the bitstream encoder yet —
  `encode_p_frame_zero_mv` still hardwires every MB to ZEROMV at
  (0, 0). The NEWMV emit path that consumes this search result, plus
  the half- / quarter-pel refinement, the §16.3 mv-cost weighting
  (lambda * MV-coding-bits added to SAD), and the GOLDEN / ALTREF
  source-selection layer remain follow-up rounds. New public
  re-exports from `oxideav_vp8`: `block_sad_16x16`, `LumaRef`,
  `SearchResult`, `mb_luma_sad_at_whole_mv`,
  `small_diamond_search_luma`, `MV_MIN`, `MV_MAX`,
  `WHOLE_PIXEL_STEP`. 14 new unit tests in `motion_search.rs` cover:
  pure-SAD identities (identical inputs / one-pixel delta / saturated
  / known manual sum), SAD-at-zero-MV consistency with
  `block_sad_16x16`, exact-translation convergence (horizontal
  2-whole-pixel; diagonal 2-row + 3-col), descent invariants
  (never increases SAD from snapped center), §17.1 range clamp of an
  `i16::MAX` / `i16::MIN` center, off-picture edge-replicate safety,
  snap-to-whole-pixel coercion of a sub-pixel center, max_iters = 0
  no-op probe, identical-source / center-MV stability, and a
  `SearchResult` Copy/Eq contract pin. Tests: 439 → 453 (+14).

* **VP8 encoder Phase 10: multi-frame I + P stream driver
  (RFC 6386 §9 / §9.7 / §9.8 / §16 / §17 / §18)** — new
  `Vp8InterStreamEncoder` extends Phase 8's keyframe driver to
  interleave ZERO_MV P-frames between key frames per a caller-specified
  `keyframe_interval`. The encoder picks K-or-P per frame, maintains
  the §9 three-slot reference-frame ladder (`LAST` / `GOLDEN` /
  `ALTREF`) one-for-one with `Vp8DecoderState::decode_frame`, and
  emits an `EncodedStreamFrame { bytes, kind: FrameKind::{Key,
  InterZeroMv}, frame_index }` per call. Slot updates honour the §9.7
  refresh ladder: a key frame refreshes all three slots (§9.7 / §9.8);
  a ZERO_MV P-frame refreshes LAST only (`refresh_last = 1`,
  `refresh_golden_frame = 0`, `refresh_alternate_frame = 0`), matching
  the bit pattern `encode_p_frame_zero_mv` writes. The first frame of
  a fresh encoder is always a key frame (no prior reference to predict
  from); a per-call `force_keyframe` override re-anchors the interval
  so the next automatic K is `keyframe_interval` frames after the
  forced K, not at the original absolute multiple. Mid-stream resize
  surfaces as `StreamEncodeError::DimensionsChanged`;
  `Vp8InterStreamEncoder::new` returns `None` for
  `keyframe_interval == 0`. New public types `Vp8InterStreamEncoder`,
  `EncodedStreamFrame`, `FrameKind` (re-exported from `crate::stream`).
  Validated on a synthetic 10-frame I420 sequence at
  `keyframe_interval = 4` (K-P-P-P-K-P-P-P-K-P): every frame's
  self-decode through `Vp8DecoderState` clears 30 dB at
  `yac_qi = 32`, range 33.96 dB (frame 7, last P of a GOP) to 45.21
  dB (frames 4 / 8, fresh K), 10-frame mean **41.35 dB** on a 48×32
  source. New `tests/encoder_inter_stream.rs` pins the per-frame PSNR
  floor end-to-end
  (`ten_frame_inter_stream_mid_quant_meets_30db_per_frame`), the §9.1
  K-vs-P frame-tag shape
  (`inter_stream_frame_tag_matches_classification`), and the
  forced-K re-anchor behaviour
  (`inter_stream_forced_keyframe_decodes_and_reanchors`). Six new
  `stream.rs` unit tests cover the scheduler, the slot refresh
  ladder, and the input validators. Tests: 430 → 439.

* **VP8 encoder Phase 9 begin: minimum-viable P-frame encoder
  (RFC 6386 §9.1 / §9.7 / §9.10 / §16 / §17 / §18)** — new
  `encode_p_frame_zero_mv` emits a valid VP8 P-frame where every
  macroblock is coded as inter / ZEROMV / LAST. Frame tag carries
  `frame_type = 1` per §9.1 (no resize, no start code); the §19.2
  coded header writes the inter refresh ladder
  (`refresh_golden_frame = 0` / `refresh_alternate_frame = 0` /
  `copy_buffer_to_golden = 0` / `copy_buffer_to_alternate = 0` /
  `refresh_last = 1` / `refresh_entropy_probs = 0`), `prob_intra = 255`
  / `prob_last = 255` so every MB reads as inter/LAST at minimum cost,
  the §17.2 `mv_prob_update()` block as 38 zero F-gates, and per-MB
  emits `mb_skip_coeff → is_inter_mb → ref_frame selector →
  inter-mode-tree leaf "0" (ZEROMV)` against the §16.3 census-driven
  probability table the decoder evolves identically. The §18 prediction
  reduces to an identity copy of LAST at MV (0,0); residual = source -
  prediction is forward-DCT'd, the Y2 DCs collected via §14.3 forward
  WHT, all blocks quantised against `MbDequantFactors::from_base_and_deltas`
  and token-coded via the existing intra §13.3 walk (single partition
  this round). All-zero residuals emit as skip MBs (§11.1). The §15
  loop filter runs over the inter reconstruction when
  `params.loop_filter_level != 0`. New `EncodeError::ReferenceDimensionsMismatch`
  validates the supplied reference frame's macroblock-aligned dimensions.
  Validated on a synthetic 64×64 I+P sequence (slow brightness drift
  between frames): self-decode through `Vp8DecoderState` produces
  **whole-frame PSNR 43.78 dB** at `yac_qi = 32` (Y 43.60 dB / U 44.15
  dB / V 44.15 dB), comfortably clearing the round's 30 dB bar. New
  tests in `tests/encoder_pframe_roundtrip.rs`:
  `i_plus_p_zero_mv_self_decode_psnr_clears_30db_at_mid_qi` pins the
  PSNR floor end-to-end; `p_frame_emits_inter_frame_tag` pins the §9.1
  `frame_type = 1` / no-start-code shape;
  `p_frame_reference_dimensions_mismatch_rejected` pins the validator.
  This unblocks the real-MV / motion-search encoder rounds (NEARESTMV
  / NEARMV / NEWMV / SPLITMV / GOLDEN / ALTREF). Tests: 427 → 430.

* **VP8 encoder Phase 8: multi-frame keyframe stream driver
  (RFC 6386 §4 / §9.1 / §9.7 / §9.8)** — new
  `Vp8KeyframeStreamEncoder` consumes a sequence of `I420Frame`s and
  emits a sequence of independently-decodable VP8 key frames, owning
  the cross-frame state a real stream needs: a frame counter,
  dimensions locked at the first `encode_frame` call, and the §9
  three-slot reference-frame buffer (`LAST` / `GOLDEN` / `ALTREF`).
  Every emitted frame implicitly refreshes all three slots per the §9.7
  / §9.8 keyframe rule, mirroring `Vp8DecoderState::decode_key_frame`'s
  slot-installation logic one-for-one. New
  `encode_keyframe_with_reconstruction` companion to `encode_keyframe`
  returns both the bytes and the macroblock-aligned post-§15
  reconstructed planes, avoiding a re-decode on the slot-refresh path.
  Mid-stream resize is surfaced as `StreamEncodeError::DimensionsChanged`
  (failed calls leave the counter unchanged). Validated on a synthetic
  5-frame I420 sequence at `qi = 32` on a 48×32 source: per-frame
  self-decode PSNR through `Vp8DecoderState` ranges 45.36–48.53 dB
  (mean 46.90 dB), comfortably above the 30 dB round target. New tests
  in `tests/encoder_keyframe_stream.rs`:
  `five_frame_keyframe_stream_mid_quant_meets_30db_per_frame` pins the
  per-frame PSNR floor through `Vp8DecoderState`;
  `every_emitted_frame_is_a_keyframe` confirms the §9.1
  `key_frame == 0` bit + `0x9d 0x01 0x2a` start code on every emitted
  frame and that each frame independently decodes from a fresh
  `Vp8DecoderState`; `first_frame_matches_standalone_encode_keyframe_psnr`
  pins byte-equality between the stream encoder's first frame and the
  standalone `encode_keyframe`; `stream_rejects_mid_stream_resize`
  pins the dimension-lock error path. Plus 4 unit tests inside
  `stream.rs` covering the fresh-state contract, first-frame slot
  population, slot replacement across frames, and dimension-change
  rejection. Tests: 423 → 427.

* **VP8 encoder Phase 7: §15 loop filter wired into the keyframe driver
  (RFC 6386 §9.4 / §15)** — `encode_keyframe` now honours a non-zero
  `KeyframeParams::loop_filter_level` (0..=63) and a matching
  `sharpness_level` (0..=7). After the per-MB raster walk completes, the
  encoder runs the §15.1 normal filter over its own reconstruction
  buffer via the existing decoder-side `filter_frame`, so the encoder's
  self-decode produces the same pixels the decoder will reproduce from
  the bitstream. The §9.4 `mode_ref_lf_delta_enabled` flag stays at 0
  this round (per-MB delta layer not yet emitted); segmentation also
  stays off, so the per-MB level resolves to the frame base in every
  case. Validation pre-walk: `loop_filter_level > 63` →
  `EncodeError::LoopFilterLevelOutOfRange`; `sharpness_level > 7` →
  `EncodeError::SharpnessLevelOutOfRange`. At `48×32 qi=32` the
  whole-frame self-decode PSNR moves from 43.29 dB (level 0) to
  43.30 / 44.67 / 44.23 dB at levels 1 / 8 / 24 — every level decodes
  cleanly through `decode_vp8` and the §9.4 fields round-trip in the
  parsed `Vp8FrameHeader`. New tests in
  `encoder_keyframe_roundtrip.rs`: `keyframe_loop_filter_levels_roundtrip`
  pins the {0, 1, 8, 24} sweep + the "filter actually ran" PSNR-vs-baseline
  invariant; `keyframe_loop_filter_level_out_of_range_rejected` and
  `keyframe_sharpness_level_out_of_range_rejected` pin the validators;
  `keyframe_sharpness_level_roundtrip` sweeps `{0, 1, 4, 7}` at filter
  level 16. Adds a `KeyframeParams::sharpness_level` field (`0` by
  default — value-compatible with prior callers that used the struct
  literal form, modulo the explicit-init). No external implementation
  consulted. Tests: 7 → 11 in `encoder_keyframe_roundtrip.rs`.

* **VP8 encoder Phase 6: multi-partition DCT output (RFC 6386 §9.5 /
  §19.2 / §20.4)** — `encode_keyframe` generalises to the four-value
  `log2_nbr_of_dct_partitions` table. A new
  `KeyframeParams::nbr_of_dct_partitions` field accepts 1 / 2 / 4 / 8
  (validated up-front before the long mode-pick walk, surfacing
  `EncodeError::InvalidDctPartitionCount` for any other value); the §9.5
  header bit is emitted through the existing
  `write_token_partition_count`; per-macroblock token data is dispatched
  to the right `BoolEncoder` instance by the §20.4 row-loop (row `r` →
  partition `r % N`); each partition finalises independently with the
  §7.3 4-byte flush trailer; and a `(N - 1) × 3`-byte §9.5 size table of
  24-bit little-endian lengths is prepended when `N > 1`. The §13.3
  above-context stays column-wise and frame-lived (shared across
  partitions, mirroring the decoder's `decode_residuals`); the left
  context resets at every row so it does not cross partitions. The
  reconstruction is byte-for-byte unchanged across all four partition
  counts — at `128×128 qi=32` the self-decode PSNR is 45.9549 dB at
  every choice, with byte counts rising monotonically (242 → 246 → 256
  → 274). New tests in `encoder_keyframe_roundtrip.rs`:
  `keyframe_multi_partition_psnr_matches_single_partition_baseline`
  pins identical reconstruction at 1 / 2 / 4 / 8 against the
  1-partition baseline; `keyframe_multi_partition_short_frame_roundtrip`
  covers a frame with fewer MB rows than partitions; and
  `keyframe_invalid_partition_count_rejected` pins the validator.

* **VP8 encoder Phase 5: rate-distortion intra mode selection** — the
  SAD-only mode picker is replaced by a Lagrangian rate-distortion
  search. Each candidate mode is run through the full §14 chain (predict
  → FDCT → Y2/WHT → quantise → dequantise → inverse-transform →
  reconstruct) and scored by `J = SSD + lambda * R`: the distortion `D`
  is the exact self-decode SSD against the source, and the rate `R` is
  the §13.3 token bits (estimated by summing `-log2(p)` over each §7.3
  boolean the token writer would emit) plus the §11.2 / §11.4
  mode-signal bits. `lambda = q² / 32` is derived from the luma AC
  quant step; rate-distortion is a non-normative encoder choice (RFC
  6386 specifies only decoding). Applied to the whole-block luma pick,
  the chroma `uv_mode` pick, and the per-4×4 `B_PRED` sub-block modes;
  the whole-block-vs-`B_PRED` luma decision compares total RD cost. On a
  64×64 natural test frame the RD picker produces smaller streams **and**
  higher self-decode PSNR than the prior SAD picker at every quantiser
  (e.g. qi 32: 1003 → 919 bytes, 34.56 → 35.00 dB). Pinned by
  `rd_beats_sad_baseline_size_and_quality` and
  `rd_keyframe_holds_psnr_floor_on_natural_frame`. No public API change.

* **VP8 encoder Phase 4: per-frame key-frame raster driver (RFC 6386
  §9 / §11 / §19.2)** — new `encode_keyframe(&I420Frame, &KeyframeParams)`
  entry that walks a source I420 picture macroblock-by-macroblock in
  raster order and assembles a complete VP8 key frame: §9.1 frame tag +
  key-frame extension, §19.2 boolean-coded first (control) partition
  (§9.2–§9.11 header fields + the §11 macroblock-mode layer), and a
  single §19.2 DCT partition carrying every non-skipped macroblock's
  §13.3 token data. Per macroblock the driver gathers the
  reconstructed-neighbour strips from the already-encoded frame buffer
  (reusing the decoder's own `gather_neighbors` / `write_mb` walker),
  runs the §12.2 whole-block / §11.3 `B_PRED` intra mode pick via
  `encode_mb_block_set_with_neighbors`, then dequantises and
  reconstructs through the decoder's `decode_keyframe_mb_non_bpred` /
  `decode_keyframe_mb_bpred` orchestrators and writes the result back —
  so the next macroblock predicts from the exact pixels the decoder
  will. The §13.3 non-zero token contexts (one `above` per macroblock
  column, frame-lived; `left` reset per row) and the §11.3
  cross-macroblock `B_PRED` sub-block-mode contexts thread MB-to-MB; an
  all-zero-residual macroblock is coded as an §11.1 skip (and clears its
  predictor slots like the decoder's skip path). Partial right / bottom
  macroblocks of a non-multiple-of-16 frame are edge-replicated into the
  padding region. New public surface: `I420Frame`, `KeyframeParams`,
  `encode_keyframe`, and a `BoolEncoder::write_treed` helper (the §8.1
  `treed_read` walk run in reverse) used by the §11 mode layer. No
  external implementation was consulted. Single key frame, single DCT
  partition, SAD-only mode pick (no RD bit-cost term), loop filter level
  0 only this round. Black-box validated end-to-end in
  `tests/encoder_keyframe_roundtrip.rs` — encode a synthetic gradient +
  flat-region I420 frame, decode via the crate's own `decode_vp8`, and
  measure whole-frame PSNR:
  * 48×32, `yac_qi = 32` (mid quant): **41.50 dB** (target ≥ 30 dB).
  * 32×32, `yac_qi = 32`: **43.65 dB**.
  * 48×32, `yac_qi = 8`: **49.10 dB**.
  * 40×24 (non-multiple-of-16): **39.81 dB**.
  A `mode_layer_roundtrips_through_decoder_parser` unit test in
  `src/encoder.rs` pins the §11 mode-layer writer against the decoder's
  `parse_key_frame_macroblock_modes`.
* **VP8 encoder Phase 3b: B_PRED 4×4 sub-block intra mode pick (RFC
  6386 §11.3 / §12.3)** — the per-MB luma decision now also evaluates
  the `B_PRED` path, in which the 16×16 luma plane is encoded as sixteen
  independent 4×4 sub-blocks, each choosing the SAD-minimising one of the
  ten §12.3 sub-modes (`B_DC` / `B_TM` / `B_VE` / `B_HE` / `B_LD` /
  `B_RD` / `B_VR` / `B_VL` / `B_HD` / `B_HU`) against the source, with
  in-place neighbour evolution — every sub-block predicts from the
  already-reconstructed (predictor + dequantised residue) pixels of the
  sub-blocks above and to its left, including the §12.3 right-edge
  above-right fixup. The encoder reuses the decoder's `predict_b4x4`
  kernel and `inverse_dct_4x4` / `add_residue_4x4` so the reconstruction
  it evolves against is exactly the one the decoder produces. A top-level
  luma decision picks `B_PRED` over the best whole-block mode iff its
  total prediction SAD is strictly lower (a flat / single-edge region in
  matching neighbours stays whole-block). When `B_PRED` wins the
  macroblock carries no Y2 block: the sixteen Y sub-blocks keep their own
  DC and are token-coded through the `YNoY2` plane (`has_y2 = false`).
  The chosen sixteen sub-modes ride on `EncodedMb::b_subblock_modes`
  (`Some` iff `y_mode == B`, else `None`), feeding the decoder's
  `decode_keyframe_mb_bpred` walk. No rate-distortion bit-cost term and
  no inter prediction this round. Validated by 3 new unit tests in
  `src/encoder.rs`:
  * `diagonal_subblock_mb_picks_bpred_and_decodes_above_30db` — a
    macroblock built from per-4×4-sub-block diagonal tiles (which no
    single whole-block mode follows) flips to `B_PRED` and reconstructs
    at **≈ 54 dB** PSNR at `yac_qi = 4`, with a genuine per-sub-block
    mode mix (`Tm` / `Ve` / `He` / `Dc`).
  * `bpred_neighbour_evolution_roundtrips_at_low_q` — the same diagonal
    MB at `yac_qi = 0` clears a 40 dB floor, confirming the encoder's
    neighbour evolution matches the decoder's bit-for-bit.
  * `flat_mb_keeps_whole_block_luma_mode` — a flat MB in matching flat
    neighbours stays on a whole-block mode (`y_mode != B`, no sub-modes).
  The three Phase 2 flat / textured roundtrip tests
  (`mb_block_set_roundtrip_flat_color_recovers_within_one_lsb`,
  `…_at_q16_holds_within_2_lsb`, `isolated_mb_textured_roundtrips_above_30db`)
  now decode through a mode-aware helper, since an isolated (off-frame
  neighbour) flat or textured MB can legitimately pick `B_PRED`.
* **VP8 encoder Phase 3: whole-block intra mode pick (RFC 6386
  §12.2)** — the per-MB driver now evaluates all four §12.2 whole-block
  intra modes (`DC_PRED` / `V_PRED` / `H_PRED` / `TM_PRED`) for the
  16×16 luma plane and the 8×8 chroma planes, picking the SAD-minimising
  mode independently for luma and chroma (no rate-distortion term yet),
  and residual-codes the source against that mode's prediction instead
  of a flat 128. Prediction uses the crate's shared `intra_predict`
  kernels (`predict_y16x16` / `predict_uv8x8`) — the exact kernels the
  decoder reconstructs with — so the residual the encoder subtracts is
  the residue the decoder adds back. The picked `y_mode` / `uv_mode` are
  surfaced on `EncodedMb`. New public surface:
  `encode_mb_block_set_with_neighbors`, which scores the picker against
  caller-supplied reconstructed-neighbour strips (`reconstruct::MbNeighbors`)
  rather than off-frame defaults; `encode_mb_block_set` is now a thin
  wrapper over it with all-off-frame neighbours. Validated by 4 new
  unit tests in `src/encoder.rs`:
  * `mode_pick_chooses_v_pred_for_column_constant_mb` — a
    column-constant (horizontally-varying) MB whose `above` neighbour
    equals the column pattern; the picker chooses `V_PRED` for luma and
    chroma and the decode reconstructs **bit-exact** (∞ dB) at
    `yac_qi = 8`.
  * `mode_pick_chooses_h_pred_for_row_constant_mb` — a row-constant
    (vertically-varying) MB with a matching `left` neighbour; picks
    `H_PRED`, reconstructs bit-exact.
  * `mode_pick_chooses_tm_pred_for_planar_ramp_mb` — a planar ramp
    `clamp(L_i + A_j − P)` with matching above / left / corner; picks
    `TM_PRED`, reconstructs bit-exact.
  * `isolated_mb_textured_roundtrips_above_30db` — a textured MB with
    off-frame neighbours through `encode_mb_block_set`; reconstructs at
    ≈ 44–45 dB PSNR (luma / U / V) at `yac_qi = 8`.
  The three flat-colour Phase 2 roundtrip tests were updated to
  reconstruct against the picked modes (a flat block off 128 now favours
  a non-DC default) and to additionally verify the chroma planes.
  `B_PRED`, inter prediction, a true rate-distortion search, and the
  per-frame raster driver remain deferred to subsequent rounds.

  Lib test count: 382 → 386 (4 new in `encoder::tests`).

* **VP8 encoder Phase 2: per-MB block-set wiring (RFC 6386 §13.3 /
  §14.2)** — new `encode_mb_block_set` driver in `src/encoder.rs` that
  takes a single 16×16 Y + 8×8 Cb + 8×8 Cr macroblock at known
  quantiser index and produces a §13.3 token-coded byte stream that
  decodes (via the crate's own `decode_mb_coeffs` +
  `MbDequantFactors::dequantize` + `decode_keyframe`) back to within
  ≤ 1 LSB of the input pixels on a flat-colour MB at `yac_qi = 0`.
  The driver implements the encode-side inverse of §14.2:
  * §12.2 `DC_PRED` constant 128 prediction (no above / left
    neighbours, matching the single-MB test fixture);
  * §14.4 forward DCT per Y / U / V 4×4 sub-block (16 + 4 + 4 = 24);
  * Collection of the 16 Y DCs into a Y2 block + §14.3 forward WHT;
  * Zeroing of each Y sub-block's DC (now carried by Y2);
  * §14.1 / §20.4 quantisation (round-half-away-from-zero) against
    the six `MbDequantFactors` (`y1_dc/y1_ac/y2_dc/y2_ac/uv_dc/uv_ac`);
  * §13.3 token-walk in residual order Y2 → 16 Y (`YAfterY2`) →
    4 U (`UV`) → 4 V (`UV`), threaded through fresh above / left
    `MbEntropyCtx` with the §20.16 `left_context_index` /
    `above_context_index` slot mapping so the per-block first-position
    probability index matches what `decode_mb_coeffs` reads.
  Returned [`EncodedMb`] carries the raw-quantised `MbCoeffs` (for
  fixture inspection / roundtrip tests), the finished bool-encoder
  byte stream, and the non-zero block count. New public surface:
  `MbPixels`, `EncodedMb`, `encode_mb_block_set`. Validated by 3 new
  unit tests in `src/encoder.rs`:
  * `mb_block_set_roundtrip_flat_color_recovers_within_one_lsb` —
    sweeps flat pixel values 100/110/128/140/160/200 at `yac_qi = 0`;
    every reconstructed luma pixel is within ≤ 1 LSB of the input.
  * `mb_block_set_constant_128_emits_all_eob_blocks` — proves that
    a zero-residual MB (constant 128) produces zero non-zero blocks
    and the bytes decode to all-zero coefficients.
  * `mb_block_set_roundtrip_flat_color_at_q16_holds_within_2_lsb` —
    same flat-MB roundtrip at `yac_qi = 16`, recovered within ≤ 2 LSB.
  RD-driven mode selection / quantiser-step picks, non-DC prediction
  modes, B_PRED / SPLITMV, inter prediction, and the per-frame raster
  driver that threads `MbEntropyCtx` columns across an N-MB frame
  are the next encoder rounds; this round lands only the per-MB
  block-set walker.

  Lib test count: 379 → 382 (3 new in `encoder::tests`).
  Standalone (no-default-features) lib tests: 374 → 377.

* **VP8 encoder Phase 2 begin: §14 forward 4×4 DCT and WHT
  primitives (RFC 6386 §14.3 / §14.4)** — new `src/forward_transform.rs`
  module exposing `forward_dct_4x4`, `forward_wht_4x4`, and
  `raster_to_scan`, the encoder partners of the §14.3 / §14.4 inverse
  transforms and the §20.16 zig-zag reorder. The two transforms are
  mechanically derived as the transpose of the §14.3 / §14.4 inverse
  listings (the §14.4 preamble itself notes the transform is *"a
  classical 2-D inverse discrete cosine transform"*); both reuse the
  same `COSPI8_SQRT2_MINUS1 = 20091` / `SINPI8_SQRT2 = 35468`
  fixed-point constants the §14.4 inverse uses so the forward /
  inverse rounding shapes track each other. Module-level docs walk
  through the matrix derivation (`M * M^T = 4*I`, `T_inv * T_inv^T =
  4*I`, hence `FDCT(p) = round((T_fwd * p * T_fwd^T) / 2)`) so the
  algebraic provenance is recorded inline. Validated by 8 new unit
  tests covering uniform-block DC concentration, FDCT↔IDCT round-trip
  on uniform + gradient + random small inputs, FWHT↔IWHT round-trip on
  uniform inputs, the all-zero block, and the `raster_to_scan`
  zig-zag inverse. A new integration test
  `tests/encoder_transform_roundtrip.rs` (7 tests) ties these
  primitives into the existing §13 `TokenEncoder`: FDCT → quantize
  (§14.1 Y1 factors) → raster-to-scan → `TokenEncoder::encode_block`
  → finish → `BoolDecoder::init` → `decode_block` → scan-to-raster →
  §14.1 `dequant_block` → `inverse_dct_4x4`, recovering the per-MB
  residual pixels. The synthetic flat-color 4×4 (pixel value = 128,
  residual = 12) round-trips at **48.13 dB PSNR** at `yac_qi = 32`,
  well above the round-131 ≥ 35 dB target; at `yac_qi = 0` the chain
  is lossless across the full `0..=64` flat-value sweep. The per-MB
  block-set wiring (Y2 DC seeding, 24/25-block walk, RD-driven mode
  + quant selection) is the next encoder round; this round lands only
  the primitive transforms + the per-block roundtrip proof.

  Lib test count: 371 → 379 (8 new in `forward_transform`).
  Standalone (no-default-features) lib tests: 366 → 374.
  Integration test count grows by 7 (new
  `tests/encoder_transform_roundtrip.rs`).

* **VP8 encoder Phase 2: §13 DCT-token block encoder (RFC 6386
  §13.2 / §13.3)** — new `encode_coeff_block` + `TokenEncoder` API in
  `src/encoder.rs` that walks the §13.2 `coeff_tree` over the existing
  §7.3 `BoolEncoder` to emit a single 16-coefficient sub-block at a
  caller-supplied resolved `coeff_probs[4][8][3][11]` table.
  Surface:
  * `classify_coeff_token(abs_value: u16) -> DctToken` — the §13.2
    alphabet classifier (Dct0..Dct4, Cat1..Cat6) shared by every
    per-coefficient encode site.
  * `encode_coeff_block(enc, block_type, coeff_probs, above_has_nonzero,
    left_has_nonzero, coeffs) -> Result<usize, TokenEncodeError>` —
    the free-function entry. Walks `coeffs[first_coeff..16]` in scan
    order, picks each position's token, emits the boolean-coded tree
    path against `coeff_probs[plane][band][ctx3]`, writes any Cat extra
    bits (PCAT1..PCAT6) + the universal sign bit at probability 128,
    rolls `ctx3` to the new coefficient's magnitude class and tracks
    `prev_was_zero` for the §13.2 EOB-branch-skip rule. Emits an
    explicit `Eob` token after the last non-zero coefficient unless
    the block is fully populated (the §13.2 "implicit EOB after the
    last coefficient" rule).
  * `TokenEncoder` struct — stateful wrapper that owns a
    `BoolEncoder` + resolved `CoeffProbs` table, exposing
    `encode_block` / `finish` / `bytes_written`. Intended for the
    later round that will stream many blocks into a single DCT
    partition.
  * `TokenEncodeError::CoefficientOutOfRange { index, value }` —
    rejects coefficients whose magnitude exceeds the §13.2 Cat6
    maximum (`67 + 2047 = 2114`).

  Validation: 8 new unit tests in `src/encoder.rs` covering the
  full §13.2 alphabet classifier, all-zero blocks at every
  `BlockType`, single non-zero coefficient at every position and
  every plane, one value inside every Cat1..Cat6 range with both
  signs, a fully-populated 16-entry block exercising the implicit
  EOB rule, sparse interior-zero patterns exercising the
  `prev_was_zero` EOB-branch-skip, all four (above × left)
  neighbour-predictor combinations, the out-of-range rejection
  path, the `TokenEncoder` wrapper byte-for-byte matches the
  free-function path, and a 64-trial pseudo-random sweep of
  block-type / coefficient pattern / neighbour combinations. Each
  test encodes the block into bytes with the production
  `BoolEncoder`, hands the bytes back to the matching
  `dct_tokens::decode_block` walk, and asserts the recovered
  `[i16; 16]` equals the input — i.e. byte-identical at the
  coefficient layer per the round goal. No RD search, no quant
  policy, no per-MB block-set wiring (deferred to a subsequent
  round). Lib test count: 363 → 371.

* **VP8 encoder Phase 1: §9 frame-header writers + silent-keyframe
  path (RFC 6386 §7.3 + §9.1–§9.11)** — new `src/encoder.rs` module
  exposing the §7.3 `BoolEncoder` (a Rust port of the `bool_encoder` /
  `write_bool` / `flush_bool_encoder` C listing embedded in RFC 6386
  §7.3) and the §9.x frame-header writer subroutines:

  * `write_frame_tag` — §9.1 3-byte tag + key-frame extension
    (`0x9d 0x01 0x2a` start code + 14-bit width + 2-bit horizontal
    scale + 14-bit height + 2-bit vertical scale, little-endian).
  * `patch_first_partition_size` — back-patch the 19-bit
    `first_partition_size` after the first partition is fully
    written, preserving the frame_type / version / show_frame bits.
  * `write_segment_update_flags` — §9.3 segment-update toggle (Phase 1
    only supports the disabled path).
  * `write_loop_filter` — §9.4 `filter_type` / `loop_filter_level (6)`
    / `sharpness_level (3)` + the `mb_lf_adjustments()` enable bit
    (Phase 1 only supports `loop_filter_adj_enable = false`).
  * `write_token_partition_count` — §9.5 `log2_nbr_of_dct_partitions`,
    accepting `count ∈ {1, 2, 4, 8}` and emitting `log2(count)`.
  * `write_quant_indices` — §9.6 baseline `y_ac_qi (L7)` plus the five
    presence-gated `L(4)+L(1)` deltas for ydc / y2dc / y2ac / uvdc /
    uvac. Phase 1 emits the baseline with every delta omitted.
  * `write_no_token_prob_updates` — the §9.9 / §13.4 1056-flag
    "every flag = 0" path against the position-specific
    `coeff_update_probs` table (NOT a flat 128), so the decoder
    consumes exactly the bits the encoder writes.
  * `write_mb_no_skip_coeff` — §9.10 / §9.11 toggle + optional
    `prob_skip_false (L8)`.

  And the top-level driver:

  * `encode_silent_keyframe(SilentKeyframeParams)` — composes the
    writers above plus a §11 macroblock-prediction loop that emits
    `mb_skip_coeff = 1` / `y_mode = DC_PRED` / `uv_mode = DC_PRED`
    for every MB, finishes with the §7.3 4-byte flush trailer for
    the first partition, then emits the §9.5 size table + one §7.3
    flush trailer per DCT partition. Output is a structurally-valid
    VP8 key frame that decodes through the crate's own
    `decode_vp8` and through `ffmpeg -c:v vp8` when wrapped in IVF.
  * `oxideav_vp8::encoder::make_encoder()` — direct factory paired
    with the workspace's dual-API convention; sits alongside the
    existing `oxideav_core::register!` registry path.
  * `oxideav_vp8::encode_vp8_keyframe` — now delegates to
    `encode_silent_keyframe` instead of returning
    `Error::NotImplemented`, so the legacy entry point starts producing
    real bytes.

  Validation: 15 new unit tests in `src/encoder.rs` (bool-encoder ↔
  bool-decoder round-trip on a 1024-element pseudo-random sequence;
  frame-tag round-trip through `Vp8FrameHeader::parse`;
  patch-first-partition-size preservation; loop-filter / sharpness /
  partition-count / quant-index validation; silent-keyframe
  round-trip through `decode_vp8` at 16×16 / 32×32 / 48×16 / 16×48 /
  32×24; round-trip at all four legal partition counts; coded-header
  re-parse of the first partition asserting every §9.x field;
  bit-budget upper-bound on the 16×16 frame). Plus 3 new
  integration tests in `tests/encoder_external_decode.rs` that pipe
  the emitted frame through `ffmpeg -c:v vp8 -f rawvideo` (16×16,
  64×64, 48×32) and assert ffmpeg accepts the bitstream and produces
  a YUV420P picture of the expected byte length.

  Out of scope for Phase 1 (deferred to subsequent rounds): pixel-aware
  mode selection, DCT/WHT residual encoding (§13), rate-distortion
  optimisation, inter-frame encoding (§16 / §17), segmentation
  (§9.3's non-trivial path), per-MB loop-filter deltas (§9.4's
  non-trivial path), and the §10 per-segment quantiser override.

* **Public `Vp8Error` umbrella error at the crate root** — a new
  `pub enum Vp8Error { Decode(DecodeError), Encode(Error) }` exposed
  from `lib.rs`, with `Display` / `std::error::Error` (with `source()`
  delegation) / `From<DecodeError>` / `From<Error>` impls. Downstream
  consumers (notably `oxideav-webp`'s lossy VP8 path) can now build
  their own `From<oxideav_vp8::Vp8Error>` adapters against a single
  stable symbol instead of having to spell out every per-module sub-error.
  A new `tests/public_error_surface.rs` integration test imports
  `Vp8Error` from the crate root and exercises every variant + the
  `From` machinery + `Display` delegation + `source()` chaining, so a
  future visibility regression fails to compile rather than silently
  breaking webp's build.

* **Top-level interframe `Vp8DecoderState::decode_frame` driver
  (RFC 6386 §9 / §16)** — new `src/state.rs` module owning the §9
  three-slot reference-frame buffer (`LAST` / `GOLDEN` / `ALTREF`),
  threading the §9.10 entropy / intra-mode / MV carry-state across
  frames (with the `refresh_entropy_probs` rollback per the §20
  `saved_entropy` reference pattern), and rotating the slots per the §9
  `copy_buffer_to_alternate` / `copy_buffer_to_golden` /
  `refresh_golden_frame` / `refresh_alternate_frame` / `refresh_last`
  ladder. The per-frame walker dispatches each macroblock to the §16
  intra-on-interframe path or the §16.2 / §16.3 / §18 inter path,
  threads the `MbInfo` neighbour records left-to-right + above row, and
  resolves the per-MB vector (ZEROMV / NEARESTMV / NEARMV / NEWMV /
  SPLITMV) before reconstruction.

  Decoder surface (`src/state.rs`):

  * `Vp8DecoderState { last, golden, altref, coeff_probs, intra_probs,
    mv_contexts, sign_bias, ref_lf_deltas, mode_lf_deltas, .. }` — the
    persistent decoder state. `new()` / `default()` start empty; the
    first packet must be a key frame.
  * `RefFrameSlot { y, u, v, y_stride, uv_stride, mb_cols, mb_rows }` —
    one slot of the three-slot ladder, sized to the macroblock-aligned
    plane buffers the §18 motion-comp fetch consumes.
  * `Vp8DecoderState::decode_frame(&mut self, bytes) ->
    Result<Vp8DecodedFrame, DecodeError>` — the top-level driver. Parses
    the §9.1 + §19.2 headers, dispatches the §11/§16 macroblock layer,
    reconstructs each MB (intra orchestrator on intra MBs, §18 inter
    motion-comp on inter MBs), runs the §15.1 loop filter, updates the
    entropy carry-state per §9.10, rotates the reference slots per §9,
    and emits a visible-cropped I420 picture.

  Loop-filter extensions (`src/loop_filter.rs`):

  * `FrameFilterConfig` gains `ref_delta_last` / `ref_delta_golden` /
    `ref_delta_altref` and `zero_mv_mode_delta` / `other_mv_mode_delta`
    / `split_mv_mode_delta` so the full §9.4 four-entry delta ladder
    fits in one config; `FrameFilterConfig::interframe(header,
    carried_ref_deltas, carried_mode_deltas)` builds the interframe
    config (carrying the §9.4 "deltas persist until updated" rule
    across frames); `FrameFilterConfig::ref_deltas()` /
    `mode_deltas()` extract the four-entry arrays the state caches for
    the next frame.
  * `calculate_mb_filter_level_inter(config, segment_id, ref_frame,
    inter_mode, y_mode_for_bpred)` — the §20.6 `calculate_filter_parameters`
    body with the full ref-frame + mode-bucket branching (an intra MB
    consults `ref_delta[CURRENT]` and `mode_delta[B_PRED]`; an inter
    MB consults `ref_delta[LAST/GOLDEN/ALTREF]` and
    `mode_delta[ZERO/SPLIT/OTHER]`).
  * `filter_inter_frame(planes, modes, coeffs, ref_frames, inter_modes,
    config)` — the §15 raster walker that takes a per-MB ref-frame +
    inter-mode slice and uses `calculate_mb_filter_level_inter` per MB
    (vs the keyframe-only `filter_frame` which assumes CURRENT_FRAME).

  Bool-decoder extension (`src/bool_decoder.rs`):

  * `BoolDecoder::init_partition(bytes)` — the §20-reference tolerant
    init: a sub-2-byte DCT partition (which tiny ALTREF-style frames can
    produce) zero-initialises the `value` register and presents an
    empty input slice, matching the §20 spec `init_bool_decoder` `else`
    branch. Used in the per-frame partition setup in `state.rs` and
    the (existing) `decoder::decode_vp8` keyframe path.

  Decoder-trait integration (`src/decoder.rs`):

  * `Vp8Decoder` (the `oxideav_core::Decoder` impl) now owns a
    `Vp8DecoderState` and routes `receive_frame` through
    `decode_frame`, so the trait surface handles inter frames too. The
    only `Error::Unsupported` it returns is "interframe arrived before
    any key frame" (a stream-mis-feed error).

  Public-in-crate helpers added on `frame.rs` / `decoder.rs`:
  `gather_neighbors_public` / `write_mb_public` / `carve_dct_partitions`
  / `crop_to_visible_public` — the bottom-half intra reconstruction and
  partition-carving primitives the new driver shares with the keyframe
  path.

  Test additions (9 new tests, 4 of them multi-frame bit-exact):

  * `state::tests::fresh_state_rejects_interframe` — the no-keyframe
    refusal.
  * `state::tests::key_frame_populates_all_three_reference_slots` —
    §9.7 / §9.8 implicit refresh.
  * `state::tests::stateful_key_frame_decode_matches_stateless` —
    re-runs three single-keyframe fixtures through the stateful driver
    and asserts bit-exact match vs the stateless `decode_vp8`.
  * `state::tests::i_frame_then_p_frame_64x64_key_frame_decodes_bit_exact`
    — sub-test verifying the I frame round-trips on its own.
  * `state::tests::i_frame_then_p_frame_64x64_p_frame_first_diff_report`
    — diagnostic that prints the first divergent byte if any.
  * `state::tests::i_frame_then_p_frame_64x64_decodes_bit_exact` —
    end-to-end 1 I + 1 P bit-exact against the libvpx YUV (the round's
    target fixture).
  * `state::tests::golden_update_cycle_decodes_bit_exact` — 5-frame
    fixture with mid-GOP golden refresh, bit-exact.
  * `state::tests::altref_arnr_on_decodes_bit_exact` — 10-frame fixture
    with `auto-alt-ref` + ARNR, bit-exact.
  * `decoder::tests::registry_tests::vp8_decoder_decodes_multi_frame_through_trait_api`
    — end-to-end 2-frame decode through the `Decoder` trait surface.

  Test count: 346 (default features) / 341 (standalone) — was 309/305.

* **§16.4 SPLITMV per-sub-block motion-vector decoding + §18 SPLITMV
  reconstruction path** — appended to `src/near_mv.rs` and
  `src/motion_comp.rs`. The walk that turns a §16.2 `Split`
  inter-mode resolution into sixteen Y-sub-block vectors plus the
  matching §18.1-averaged chroma vectors, and the SPLITMV §18
  reconstruction wired through them.

  Decoder surface:

  * `MvPartition` (`TopBottom` / `LeftRight` / `Quarters` / `Mv16`)
    + `MV_PARTITIONS[4][16]` + `MV_PARTITION_TREE` (`{-3, 2, -2, 4,
    -0, -1}`) + `MV_PARTITION_PROBS` (`{110, 111, 150}`) — the §20.13
    `split_mv_tree` / `split_mv_probs` / `mv_partitions` tables.
    `read_mv_partition(dec)` walks the tree.
  * `SubMvRefMode` (`Left4x4` / `Above4x4` / `Zero4x4` / `New4x4`)
    + `SUBMV_REF_TREE` + `SUBMV_REF_PROBS[5][3]` — the §20.13
    `sub_mv_ref_tree` + context-keyed `submv_ref_probs2` table.
    `submv_ref_context(left, above)` derives one of five contexts per
    §16.4 `vp8_mvCont` (NORMAL / LEFT_ZED / ABOVE_ZED /
    LEFT_ABOVE_SAME / LEFT_ABOVE_ZED); `submv_ref(dec, left, above)`
    reads the tree under the context-selected probability row.
  * `above_block_mv(this, above, b)` / `left_block_mv(this, left, b)`
    — the §20.11 neighbour-sub-block MV lookups. Top-row / left-column
    anchors consult the neighbour MB (SPLITMV → its bottom-row /
    right-column sub-block; otherwise its whole-MB vector; intra → 0);
    interior anchors read the current MB's already-filled sub-block.
  * `decode_split_mv(dec, above, left, best, mv_ctx)` — the §16.4 /
    §20.11 `decode_split_mv` partition walk. Reads the partition id,
    then per group finds the anchor sub-block, runs `submv_ref`, picks
    the partition vector (`LEFT4x4` / `ABOVE4x4` neighbour copy,
    `ZERO4x4` zero, `NEW4x4`-adds-decoded-diff-to-`best`), and fills
    every member sub-block. Returns `SplitMvResult { partition,
    split_mvs }`.
  * `MbInfo` gains `split_mvs: Option<[Mv; 16]>` — the §20.5
    `mb_info.split.mvs` array; populated when a neighbour was coded
    SPLITMV so the next MB's `above_block_mv` / `left_block_mv` can
    borrow the correct sub-block vector.

  Reconstruction surface (`src/motion_comp.rs`):

  * `chroma_idx_for_luma_subblock(b)` — the §18.1 / §20.11
    `(b>>1&1) + (b>>2&2)` luma→chroma mapping (`{0,1,4,5}→0`, etc.).
  * `split_chroma_mvs(luma_mvs)` — the §18.1 chroma derivation:
    averages the four luma (stored-doubled) vectors per chroma slot
    via the §18.1 `avg()` primitive (sign-aware divide-by-8 with
    rounding).
  * `predict_split_mv(reference, mb_col, mb_row, split_luma_mvs,
    full_pixel, filters)` — the SPLITMV §18.2/§18.3 prediction buffer:
    sixteen luma sub-blocks each interpolated with their own
    §18.1-doubled vector (per-sub-block `filter_block_4x4` dispatch),
    four chroma sub-blocks under the §18.1 averaged vectors. No §18.1
    secondary clamp (per §18.1 page 114 "secondary clamping is not
    performed for SPLITMV macroblocks").
  * `reconstruct_split_mv_mb(...)` — the SPLITMV analogue of
    `reconstruct_inter_mb`. No Y2 / DC-in-Y residue (§14.2 "for
    SPLITMV the 0th Y coefficients are part of the residue signal"),
    so each Y sub-block's full 16 coefficients go straight through
    the inverse DCT.

  End-to-end driver:

  * `decode_split_mv_mb(...)` — the SPLITMV analogue of
    `decode_inter_mb`: runs the §16.3 census + §16.2 inter-mode tree,
    asserts a `Split` resolution, runs `decode_split_mv`, then drives
    `reconstruct_split_mv_mb`. Returns the reconstructed pixels +
    `SplitMvResult` (caller stores `split_mvs[15]` as the MB's `mv`
    per dixie `this->base.mv = this->split.mvs[15]` and
    `Some(split_mvs)` as the next neighbour's `MbInfo::split_mvs`).

  Twenty-eight new unit tests: spec-verbatim table / tree shape for
  every new table, `submv_ref_context` bucket coverage, partition-tree
  round-trip for every shape, sub-MV-ref tree round-trip for every
  mode + every context, neighbour-MV lookup coverage (intra /
  non-split / split / internal for both `above_block_mv` and
  `left_block_mv`), all-ZERO4x4 `decode_split_mv` for every partition
  shape, per-mode SPLITMV semantics (ZERO/NEW top-bottom, ABOVE4x4
  top-bottom, LEFT4x4 left-right, per-sub-block NEW4x4 Mv16),
  `chroma_idx_for_luma_subblock` grouping against the §18.1
  enumeration, `split_chroma_mvs` reduces to `chroma_mv` on a uniform
  field, and two byte-exact `decode_split_mv_mb` end-to-end
  reconstructions (zero-split co-located copy, TopBottom distinct
  halves with whole-pixel-after-doubling shift). Test count: 337
  (was 309).

* **§16.2 / §16.3 / §18.1 near/nearest motion-vector census + inter-mode
  tree** — new `src/near_mv.rs`, the inter-prediction slice that decides
  *which* vector a whole-MB inter macroblock uses. `find_near_mvs` is the
  §16.3 / §20.11 `vp8_find_near_mvs` spatial census: it surveys the
  above / left / above-left neighbours (`MbInfo` records, `MbInfo::border`
  for the §16.3 1-MB off-frame border of 0,0 vectors), accumulates the
  weighted candidate list (above/left weight 2, above-left weight 1) with
  the §16.3 dedupe, the SPLITMV-merge-with-NEAREST rule, the
  near↔nearest swap, and the best := nearest store, returning the
  `best / nearest / near` candidates plus the four-entry `cnt` census.
  `mv_bias` applies the §16.3 sign-bias negation (`SignBias` carries the
  §9.7 `sign_bias_golden` / `sign_bias_alternate` bits, indexed by the
  dixie `reference_frame` enum). `mv_ref_probs` derives the four §16.2
  tree probabilities from `cnt` via `MV_COUNTS_TO_PROBS` (the §20.13
  `mv_counts_to_probs[6][4]` / `vp8_mode_contexts` table), and
  `read_inter_mode` walks `MV_REF_TREE` (the §20.13 `mv_ref_tree`) into an
  `InterMode` (`Zero` / `Nearest` / `Near` / `New` / `Split`). `clamp_mv`
  / `MvClampRect::for_mb` are the §16.3 / §18.1 / §20.11 `clamp_mv`
  one-MB-border clamp (quarter-pixel, the §20.11 eighth-pixel bounds
  halved consistently). `resolve_inter_mb_mv` ties census → probs → mode
  → per-mode vector: `ZEROMV` zero, `NEARESTMV` / `NEARMV` clamped
  candidate, `NEWMV` clamped-best + decoded §17 differential (the §18.1
  secondary clamp), `SPLITMV` reported with the clamped best base.
  `decode_inter_mb` is the end-to-end integration entry: it runs the
  resolution then drives `motion_comp::reconstruct_inter_mb` with the
  resolved vector, so a whole-MB inter macroblock decodes from bitstream
  to reconstructed Y/U/V pixels (`InterMbError::SplitNotSupported` carries
  the best base for the deferred §16.4 walk). 32 new unit tests (tree /
  table shape, every inter-mode round-trip incl. under default census
  probs, census all-border / single-neighbour / intra-skip / zero-vector
  CNT_ZERO scoring / dedupe / near↔nearest swap / SPLITMV weighting / sign
  bias, clamp bounds + confinement, `mv_ref_probs` per-column indexing,
  per-mode `resolve_inter_mb_mv`, and four byte-exact `decode_inter_mb`
  end-to-end reconstructions — ZEROMV co-located copy, NEARESTMV neighbour
  vector, NEWMV best+differential, SPLITMV error surface). §16.4 SPLITMV
  per-sub-block walk remains a follow-up.

* **§18.3 sub-pixel motion compensation (sixtap luma + bilinear)** —
  extends `src/motion_comp.rs` with the §18.3 fractional-MV interpolation
  path. `SIXTAP_FILTERS` / `BILINEAR_FILTERS` are the §18.3 `filters` /
  `BilinearFilters` 8×6 tap tables (each row summing to 128);
  `FilterSet` / `filter_set_for_version` reproduce the §20.14
  `version == 0 ? sixtap : bilinear` selection (both planes share the
  frame's set). `interp` is the §18.3 single-sample six-tap
  `clamp255((Σ p·fil + 64) >> 7)`; `sixtap_horiz` / `sixtap_vert` /
  `sixtap_2d` are the §20.14 horizontal-then-vertical convolutions with
  the byte-clamped 9-row intermediate. `fetch_block_halo` is the §20.14
  `build_mc_border` 9×9 support-halo fetch (the 4×4 block plus the
  two-before / three-after taps, edge-replicated, block origin at (2,2)).
  `filter_block_4x4` is the §20.14 `filter_block` dispatcher (whole-pixel
  copy or six-tap synthesis per sub-block). `predict_inter_mb` /
  `reconstruct_inter_mb` are the full non-SPLITMV prediction +
  §14-residue path: unlike the retained whole-pixel-only
  `predict_inter_mb_whole_pixel` / `reconstruct_inter_mb_whole_pixel`,
  they interpolate sub-pixel vectors directly instead of returning
  `MotionCompError::SubPixelNotSupported`. 19 new unit tests (tap-table
  values + sum-to-128 + DC pass-through, version→set selection, `interp`
  formula incl. negative-tap clamp, `sixtap_2d` byte-exact against an
  independent §18.3 Hinterp/Vinterp transcription over every (mx,my)
  fraction for both sets, halo edge replication, `filter_block`
  dispatch, and whole-MB sub-pixel prediction / reconstruction incl.
  legacy-path agreement on whole-pixel vectors). §16.3
  near/nearest/best census and §16.4 SPLITMV remain later rounds.

* **§16.2 / §18 whole-pixel interframe motion compensation** — new
  `src/motion_comp.rs`, the first inter-prediction slice that *consumes*
  the §17 motion vectors. `select_ref_frame` reads the §16.2
  `prob_last` / `prob_gf` reference-frame selector into a `RefFrame`
  (`Last` / `Golden` / `AltRef`). `stored_luma_mv` applies the §18.1
  stored-luma doubling (quarter-pel → eighth-pel), `chroma_mv` the §18.1
  `avg()` chroma averaging (cross-checked against the §20.14
  `(c + 1 + (c >> 31) * 2) / 2` closed form), and `apply_full_pixel` the
  version-3 full-pel-chroma truncation (`& ~7`). `fetch_block_whole_pixel`
  is the §20.14 `build_mc_border` edge-replicated 4×4 reference fetch
  specialised to whole-pixel offsets. `predict_inter_mb_whole_pixel`
  assembles the §18.2 whole-MB prediction buffer for a non-SPLITMV
  macroblock (one vector for all sixteen Y sub-blocks, the averaged
  chroma vector for the eight chroma sub-blocks) reading a borrowed
  `ReferencePlanes`, and `reconstruct_inter_mb_whole_pixel` folds in the
  §14 dequantized residual (Y2 WHT seeding + per-sub-block inverse DCT +
  §14.5 `clamp255` summation) to complete inter-MB reconstruction.
  Sub-pixel (§18.3 sixtap / bilinear) vectors are refused with
  `MotionCompError::SubPixelNotSupported`; the §16.3 near/nearest/best
  census and §16.4 SPLITMV remain later rounds. 23 new unit tests
  (reference-frame selector read order, §18.1 adjustments incl. the
  §20.14 closed-form cross-check, `build_mc_border` edge replication,
  whole-MB prediction copy / offset / rejection, and inter-MB
  reconstruction skip + DC-residue paths).

## [0.2.0](https://github.com/OxideAV/oxideav-vp8/compare/v0.1.13...v0.1.14) - 2026-05-24

### Other

- Add §17 motion-vector component decoder (inter-frame path start)
- Wire top-level decode_vp8 driver + oxideav_core::Decoder (key frames)
- §15.1 loop-filter frame geometry over decode_keyframe planes
- §14.1 Y2/chroma dequant scaling + bitstream→dequant wrapper
- §13.3 per-macroblock DCT-coefficient token walk
- per-frame keyframe raster walker (RFC 6386 §12 / §14.2)
- §11.3/§12.3 B_PRED macroblock reconstruction (per-sub-block intra walker)
- §14.2 per-macroblock reconstruction orchestration (non-B_PRED)
- §16.1 interframe intra-predicted macroblock mode decoding
- loop-filter per-segment kernels (RFC 6386 §15)
- dequantization + inverse transforms (RFC 6386 §14)
- §13 DCT-coefficient token decode (coeff_tree walker + extra-bits)
- §12 pixel-shape kernels (16×16 Y, 8×8 UV, 10× 4×4 sub-block)
- key-frame §11 mode layer (Y / UV trees + sub-block modes)
- add §9.10 inter-only tail + §17.2 mv_prob_update
- clean-room rebuild round 3 — boolean-coded frame header (RFC 6386 §19.2)
- clean-room rebuild round 2 — uncompressed frame header (RFC 6386 §9.1)
- clean-room rebuild round 1 — boolean (range) entropy decoder
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **§17 motion-vector component decoder** — new `src/motion_vector.rs`,
  the first element of the inter-frame (§16) prediction path. Motion
  vectors share an identical wire format in `NEWMV` (whole-MB) and
  `NEW4x4` (SPLITMV sub-block) modes, so this is the shared primitive both
  call sites consume. `read_mv_component` implements §17.1
  `read_mvcomponent`: the `mvpis_short` short-vs-long range selector, the
  `small_mvtree` tree-coded short form (`0 <= A <= 7`, transcribed as
  `SMALL_MVTREE`), the independent-bit long form (`8 <= A <= 1023`) with
  the implicit-bit-3 rule, and the conditional sign — returning a signed
  quarter-pixel component `-1023..=1023`. `read_mv` implements §17.2
  `read_mv` (row context then column context → raw differential `Mv`).
  `resolve_mv_contexts` applies the round-4 §17.2 `mv_prob_update()`
  overlays onto a base `MvContexts` (key-frame default via
  `default_mv_contexts`, cross-interframe persistence by passing the
  previous resolved contexts as `base`), turning the parsed updates into
  the live decoding tables. The §16.2 `mv_ref` tree, §16.3
  `find_near_mvs` census, §16.4 SPLITMV walk, §18.1 doubling / clamp, and
  §18 sub-pixel interpolation are deferred to later rounds — `read_mv`
  returns the raw differential vector. New surface: `Mv`, `MvContext`,
  `MvContexts`, `SMALL_MVTREE`, `read_mv_component`, `read_mv`,
  `resolve_mv_contexts`, `default_mv_contexts`. Adds 15 tests
  (round-tripping a test-side bool encoder through every §17.1/§17.2
  branch); total 238.

* **Top-level `decode_vp8` per-frame driver + `oxideav_core::Decoder`** —
  new `src/decoder.rs` wires the previously-isolated keyframe pieces into
  a single end-to-end entry point. `decode_vp8(bytes)` takes the raw bytes
  of one VP8 frame, parses the §9.1 uncompressed header and the §19.2
  boolean-coded frame header, runs the §11 / §19.3 macroblock prediction
  layer, carves the §9.5 DCT partitions (3-byte LE size table + the §20.4
  round-robin row striping, one `BoolDecoder` per consumed partition with
  a persistent cursor), decodes + §14.1-dequantizes each macroblock's §13
  residuals, reconstructs via `decode_keyframe`, applies the §15.1
  loop-filter post-pass, and crops to the §9.1 visible dimensions —
  returning an I420 `Vp8DecodedFrame`. Non-key frames (§16 inter) return a
  clean `DecodeError::Unsupported` (no stub-decode). The default-on
  `registry` feature additionally exposes `Vp8Decoder` (an
  `oxideav_core::Decoder` impl), `make_decoder`, and `register` /
  `register_codecs` (registering codec id `"vp8"` with the `VP80` / `vp08`
  / `V_VP8` container tags); `register` is wired through the
  `oxideav_core::register!` dispatch hook in `lib.rs`. The keyframe decode
  chain (bitstream → dequant → reconstruct → loop-filter → pixels) is now
  **bit-exact** against the libvpx/ffmpeg black-box reference on ten VP8
  conformance fixtures (16×16, 64×64, 32×32, 128×128; 1- and 4-partition;
  loop-filter off / level-1 / level-33 / level-38; simple-filter mode),
  vendored under `tests/fixtures/`. Two intra-prediction correctness fixes
  landed alongside: the §20.14 `fixup_left` corner rule — a left-frame-edge
  macroblock's `(-1,-1)` corner pixel is 129 (was defaulting to 127),
  which `TM_PRED` / `B_TM_PRED` read — and the off-top corner default for
  the non-`B_PRED` `TM_PRED` path (now 127, was 0). New surface:
  `decode_vp8`, `DecodeError`, `Vp8DecodedFrame`, and (gated) `Vp8Decoder`
  / `make_decoder` / `register` / `register_codecs` / `VP8_CODEC_ID`. Adds
  17 tests (non-keyframe → Unsupported, truncation / zero-dimension error
  paths, the partition-table carve, ten bit-exact fixture decodes, and the
  `Decoder`-trait integration). The crate's prior placeholder
  `decode_vp8`/`register` stubs and `Error::NotImplemented` decode path are
  replaced; `encode_vp8_keyframe` remains scaffolded.

* **§15.1 loop-filter frame geometry** — `filter_frame` in
  `src/loop_filter.rs`, the per-frame post-pass that applies the existing
  §15.2 simple / §15.3 normal kernels across a reconstructed
  `KeyframePlanes`. Walks macroblocks in raster order and runs the four
  §15.1 page-86 steps in order — left inter-MB vertical edge (skipped on
  the leftmost column), three internal vertical subblock edges (1/4, 1/2,
  3/4 width; one centre edge for chroma), top inter-MB horizontal edge
  (skipped on the topmost row), three internal horizontal subblock edges
  — with chroma analogues for the normal filter (the simple filter is
  luma-only per §15.2). Steps 2 and 4 are skipped when the MB is neither
  `B_PRED` nor `SPLITMV` *and* has no coded coefficient (the §20.6 annex
  note: the gate is the decoded-coefficient count, not the bitstream skip
  flag, so the pass inspects the dequantized `MbCoeffs` directly), and the
  whole MB is skipped when its resolved level is 0 (§15 page 84). The
  per-MB level derivation, `calculate_mb_filter_level`, implements the
  §20.6 `dixie.c` `calculate_filter_parameters` body — part of RFC 6386,
  not external source: base `loop_filter_level`, the §10 per-segment
  override (delta adds / absolute replaces, clamped `0..=63`), then the
  §9.4 reference + `B_PRED` mode deltas (clamped again). `LoopFilterParams`
  `mbedge_limit` / `sub_bedge_limit` already equal the §20.6 `2*E + I`
  disabling metric, so they pass straight into the kernels as the
  `edge_limit` argument. `FrameFilterConfig` carries the resolved frame
  state, with `FrameFilterConfig::keyframe` building it from a
  `Vp8CodedHeader` (resolving the per-segment LF levels and the §9.4
  current-frame / `B_PRED` deltas). New surface: `filter_frame`,
  `calculate_mb_filter_level`, `FrameFilterConfig`, `MAX_REF_LF_DELTAS`,
  `MAX_MODE_LF_DELTAS`, `MAX_MB_SEGMENTS`. Adds 15 tests (level derivation:
  base, segment delta/absolute, dual clamp, ref/mode deltas, delta-disable;
  frame geometry: level-0 no-op, a hand-derived normal MB-edge rewrite of
  the six straddling pixels, leftmost-column left-edge skip, simple-filter
  luma-only, the coeff/`B_PRED`-gated subblock steps, and the
  header→config resolution). Total 206 tests.

* **§14.1 dequantization scaling + bitstream→dequant wrapper** — new
  `src/dequant.rs` module closing the last §14 spec gap. `MbDequantFactors`
  computes the six §14.1 dequant factors (Y1 DC/AC, Y2 DC/AC, chroma DC/AC)
  per the §20.4 `dixie.c` `dequant_init` rules — part of RFC 6386, not
  external source: Y1 DC/AC use the `dc_qlookup` / `ac_qlookup` tables
  directly; Y2 DC = `dc_q × 2`; Y2 AC = `ac_q × 155 / 100` floored at 8;
  chroma DC = `dc_q` capped at 132; chroma AC = `ac_q`; every index goes
  through the §20.4 `clamp_q` 0..=127 saturation. `from_quant_indices`
  derives the frame-level factors from a `QuantIndices` (base `q = yac_qi`,
  each §9.6 delta applied per plane); `for_segment` layers the §10
  per-segment quantizer override (absolute replaces the base, delta adds to
  it, per-plane deltas still apply). `dequantize(&mut MbCoeffs)` scales a
  raw quantized bundle in place (coefficient 0 × DC, 1..=15 × AC, products
  in `i32` stored as `i16` per §14.1 page 76). `decode_and_dequantize_mb`
  is the wrapper that runs `decode_mb_coeffs` then `dequantize`, turning the
  token partition straight into the pre-dequantized `MbCoeffs` that
  `decode_keyframe` consumes — completing the keyframe decode chain
  bitstream → dequant → reconstruct → pixels. New surface:
  `MbDequantFactors`, `decode_and_dequantize_mb`, `UV_DC_MAX`, `Y2_AC_MIN`.
  `MbCoeffs` now derives `PartialEq` / `Eq`. Adds 15 tests (each scaling
  rule in isolation including the <8 Y2-AC floor and 132 chroma-DC cap and
  the truncating `×155/100`; index clamping; the §10 segment delta/absolute
  derivation; an independent re-derivation of the §20.4 `dequant_init` body
  for the §9.6 worked vector; per-plane in-place dequant; the wrapper
  matching `decode_mb_coeffs` + `dequantize`; and a full keyframe decode
  through the wired chain proving a larger quantizer lifts luma further off
  the prediction). Total 191 tests.

* **§13.3 per-macroblock DCT-coefficient token walk** — new
  `decode_mb_coeffs(dec, has_y2, mb_skip_coeff, coeff_probs, above, left)
  -> Result<MbCoeffs, MbCoeffError>` in `src/dct_tokens.rs`, the
  integration layer over the per-block `decode_block` primitive that feeds
  `decode_keyframe` straight from the bitstream. Walks the 24/25 residual
  blocks of one macroblock in the §13 `residual_data()` order — the §14.2
  Y2 (WHT) block first when present, then the sixteen Y 4×4 DCT blocks,
  then the four U and four V chroma blocks — selecting the `Y2` /
  `YAfterY2` / `YNoY2` / `UV` plane per block, threading the §13.3 above /
  left non-zero predictor context through the §20.16 `left_context_index` /
  `above_context_index` slot tables (a nine-entry `MbEntropyCtx`: four Y,
  two U, two V, one Y2), and updating both referenced predictor slots with
  each block's non-zero status for the macroblocks below and to the right.
  Honours the §13.1 `mb_skip_coeff` short-circuit with the §20.16
  `reset_mb_context` rule (clear the eight Y/U/V slots; clear the Y2 slot
  only when the macroblock would have carried a Y2 block, preserving it
  across skipped `B_PRED` / `SPLITMV` macroblocks per the "most recent
  macroblock that has a Y2 block" rule). Reorders each decoded block into
  raster (natural) order via the §20.16 `zigzag[16]` table — the layout
  the §14 inverse transforms consume. The emitted coefficients are the
  **raw quantized** token values; the §14.1 Y2 / chroma dequant scaling
  remains a documented spec gap (§14.1 page 77 defers it to `dixie.c`
  §20.4). New surface: `decode_mb_coeffs`, `MbEntropyCtx`, `MbCoeffError`,
  `ZIGZAG`, `MB_ENTROPY_CTX_LEN`. Adds 7 tests (zig-zag permutation
  round-trip; §20.16 context-index table spot-check; skip-MB zeroing and
  the `B_PRED` Y2-preservation rule; a synthetic MB round-trip recovering
  the exact per-block raster layout across all planes; an empty-block
  predictor-slot clear; and two-adjacent-MB left-context propagation with
  a fresh-context negative control proving the propagation is
  load-bearing).
* **§12 / §14.2 per-frame keyframe raster walker** — new
  `decode_keyframe(mb_cols, mb_rows, modes, coeffs) ->
  Result<KeyframePlanes, FrameError>` in `src/frame.rs`, the layer above
  the per-MB orchestrators. Iterates a key frame's macroblocks in
  raster-scan order; for each MB it assembles the `MbNeighbors` strips
  from the already-reconstructed full-frame plane buffers (the bottom row
  of the MB above, the rightmost column of the MB to the left, the
  top-left corner pixel, and — for `B_PRED` — the four §12.3
  above-and-to-the-right pixels), selects `decode_keyframe_mb_bpred` vs
  `decode_keyframe_mb_non_bpred` per the MB's decoded luma mode, and
  writes the reconstructed 16×16 luma + two 8×8 chroma blocks into the
  I420 `KeyframePlanes`. Off-frame edges are reported as `None` (not a
  127 / 129 fill) so the §12.2 `DC_PRED` averaging distinguishes genuine
  visible pixels from the out-of-bounds defaults. The §12.3 above-right
  extension applies the rightmost-macroblock `(-1,15)` clamp (and is left
  `None` on the top MB row, where the orchestrator fills 127). New
  surface: `decode_keyframe`, `KeyframePlanes`, `MbCoeffs`, `FrameError`.
  Consumes caller-supplied pre-dequantized `MbCoeffs` because the §13.3
  per-MB token walk and §14.1 Y2/chroma dequant scaling remain documented
  spec gaps. Adds 10 tests (2×2 round-trip through the walker; rightmost-MB
  above-right clamp; non-rightmost genuine above-right; top-row `None`;
  cross-MB neighbour propagation; B_PRED frame walk; error paths).
* **§11.3 / §12.3 `B_PRED` macroblock reconstruction** — new
  `decode_keyframe_mb_bpred(subblock_modes, uv_mode, mb_skip_coeff,
  neighbors, y_coeffs_dequant, u_coeffs_dequant, v_coeffs_dequant)
  -> Result<ReconstructedMb, ReconstructError>` in `src/reconstruct.rs`,
  the companion to the 16×16-mode orchestrator. Drives the sixteen
  4×4 luma sub-blocks of a `B_PRED` macroblock with the §12.3
  per-sub-block neighbour evolution: each sub-block is predicted with
  `predict_b4x4` (one of the ten `B_DC_PRED` … `B_HU_PRED` sub-modes),
  inverse-DCT'd and residue-added **in place** before the next
  sub-block in raster order reads it — so a sub-block's `above` /
  `left` / top-left `P` neighbours are the already-reconstructed
  pixels, exactly as §20.14's reference `b_pred()` loop does. The
  working luma buffer carries a one-pixel top-border row + left-border
  column + a four-pixel above-right extension; the §12.3 right-edge
  "above-right" fixup copies sub-block 3's `(-1,16)..=(-1,19)` pixels
  down into the border slots above sub-blocks 7 / 11 / 15 (the
  reference `copy_down`). A `B_PRED` macroblock has no Y2 block, so —
  per §13 / §14.2 — no inverse-WHT seeding is applied; the 0th
  coefficient of each Y sub-block is taken verbatim from its own
  residue. Chroma uses the ordinary 8×8 §12.2 path. The §12.3
  `B_HD_PRED` `svg2p(E+1)` erratum (task #957) is handled in the
  pre-existing `predict_b4x4` kernel (read as `avg2p`). New surface:
  - `MbNeighbors::y_above_right: Option<[u8; 4]>` — the four
    above-and-to-the-right luma pixels the right-edge sub-blocks share
    (`None` on the top MB row → 127; the caller applies the
    rightmost-MB `(-1,15)` clamp).
  - `ReconstructError::MissingSubblockModes` — surfaced when the
    `B_PRED` entry is called without the sixteen 4×4 sub-block modes.

  Ten new unit tests: missing-modes error; top-left-corner DC settling
  to 128; per-sub-mode (all ten) skip-MB match against the standalone
  `predict_b4x4` kernel for sub-block (0,0); left- and above-neighbour
  evolution (sub-block (0,1)/(1,0) responding to sub-block (0,0)'s
  residue); the right-edge above-right fixup propagating into
  sub-blocks 3 / 7 / 15; top-row above-right defaulting to 127;
  a full mixed-mode MB end-to-end (skip-vs-run invariance + residue
  lift); and chroma using the 8×8 mode independent of luma. Total:
  159 tests across nine modules (up from 149).

* **§14.2 per-macroblock reconstruction orchestration** — new
  `src/reconstruct.rs` module. Ties the previously-isolated transform
  / prediction / summation primitives together for one macroblock
  whose 16×16 luma mode is one of the four non-`B_PRED` modes
  (`DC_PRED` / `V_PRED` / `H_PRED` / `TM_PRED`). New surface:
  - `decode_keyframe_mb_non_bpred(y_mode, uv_mode, mb_skip_coeff,
    neighbors, y2_coeffs_dequant, y_coeffs_dequant, u_coeffs_dequant,
    v_coeffs_dequant) -> Result<ReconstructedMb, ReconstructError>`
    runs the §14.2 four-step recipe: (1) inverse-WHT the Y2 block
    and seed each Y sub-block's coefficient 0 with the matching
    `wht_output[i*4+j]` element (§14.2 first-paragraph index rule);
    (2) inverse-DCT all sixteen Y and eight chroma sub-blocks
    (§14.2 second paragraph); (3) apply the §12 intra-prediction
    kernel selected by the §11 mode record; (4) sum with `clamp255`
    per §14.5.
  - `MbNeighbors { y_above, y_left, y_topleft, u_above, u_left,
    u_topleft, v_above, v_left, v_topleft }` — the per-MB pixel
    context the §12 kernels read, each field `Option`-wrapped so
    frame-edge MBs trigger the spec's default-substitution rules.
  - `ReconstructedMb { y: [u8; 256], u: [u8; 64], v: [u8; 64] }` —
    the predictor-plus-residue output, before loop filtering.
  - `ReconstructError::BPredNotSupported` — surfaced when called
    with `IntraYMode::B`. The `B_PRED` branch needs a per-sub-block
    intra walker that evolves the neighbour context after each
    4×4 reconstruction (§12.3); that is the next layer up.
  - `mb_skip_coeff` short-circuit (§11.1): zero residue → output
    equals prediction, skipping the WHT / DCT / summation.

  Dequantization is the caller's responsibility because §14.1
  Y2 / chroma dequant scaling is a documented spec gap that defers
  to the reference decoder `dixie.c` source (excluded under the
  clean-room policy). Keeping dequant out of this signature lets
  the §14.2 orchestration land without depending on that gap; a
  future wrapper can accept raw tokens + a quantiser-index bundle
  once the gap docs land.

  Eleven new unit tests: `B_PRED` MB rejection; top-left-corner
  skip MB matching `DEFAULT_TOPLEFT_DC` (128) in every plane; skip
  MB with `V_PRED` matching standalone predictor output; zero-residue
  non-skip path equals the skip path; Y2 DC-only seeding lifting all
  sixteen Y sub-blocks; Y2 off-diagonal seeding proving the
  `i * 4 + j` sub-block index rule; `V_PRED` with no `above`
  substituting `DEFAULT_ABOVE_PIXEL` (127); `H_PRED` with no
  `left` substituting `DEFAULT_LEFT_PIXEL` (129); §14.5 `clamp255`
  saturation high (all-255) and low (all-0); and an
  `extract_4x4`/`insert_4x4` helper round-trip. Total: 150 tests
  across eight modules (up from 139).
* **Interframe intra-predicted macroblock mode decoding** per RFC 6386
  §16.1 (extends `src/macroblock.rs`). The §16.1 layer is
  structurally analogous to the §11 key-frame layer but uses different
  trees and probability tables. New surface:
  - `IF_YMODE_PROB_DEFAULTS = [112, 86, 140, 37]`,
    `IF_UV_MODE_PROB_DEFAULTS = [162, 101, 204]` and
    `IF_BMODE_PROB = [120, 90, 79, 133, 87, 85, 80, 111, 151]` — the
    three default probability tables (the first two are dynamic and may
    be overridden per frame; the bmode table is fixed and shared by
    every sub-block, with **no** above/left context — that's a
    key-frame-only behaviour).
  - `InterFrameIntraProbs::for_frame_header(previous, header)` — the
    resolved per-frame Y/UV probability state. On a key frame, both
    dynamic tables reset to the §16.1 defaults per the section's last
    paragraph; on an interframe, the resolved state is the previous
    state with the §9.10 F-gated `intra_y_mode_prob_update` /
    `intra_uv_mode_prob_update` overlays applied wholesale (or
    carried forward unchanged when the override block is absent).
  - `parse_inter_frame_intra_macroblock_modes(dec, probs, segment_id,
    mb_skip_coeff)` — decode one §16.1 intra MB: the Y mode
    (`ymode_tree` against `probs.y_mode_prob`), the sixteen sub-block
    modes when Y is `B_PRED` (`bmode_tree` against the fixed
    `IF_BMODE_PROB`), and the UV mode (`uv_mode_tree` against
    `probs.uv_mode_prob`). The optional `segment_id` and
    `mb_skip_coeff` precede the intra-vs-inter discriminator on
    interframes and are read before this entry point; they pass
    through to the returned `MacroblockModes` unchanged.
* The internal §16.1 `IF_YMODE_TREE` is the eight-entry
  `[-DC_PRED, 2, 4, 6, -V_PRED, -H_PRED, -TM_PRED, -B_PRED]` listing.
  Its first slot disagrees with `KF_YMODE_TREE` — that's the §16.1
  characterisation: the root left-leaf encodes `DC_PRED` ("0") rather
  than the key-frame's `B_PRED` ("0"). The leaves at the second level
  also differ in ordering.
* Twelve new unit tests covering: spec-listing transcription of the
  three §16.1 default tables; the `IF_YMODE_TREE` shape (matching the
  spec literal listing and explicitly distinct from `KF_YMODE_TREE`'s
  root); round-trip of all five Y modes through `IF_YMODE_TREE`; all
  four UV modes through the shared `UV_MODE_TREE` with the §16.1
  defaults; all ten sub-block modes through the shared `BMODE_TREE`
  with the fixed `IF_BMODE_PROB`; a non-`B_PRED` MB round-trip with
  the optional pass-through fields elided; a `B_PRED` MB round-trip
  with a sixteen-entry mixed sub-block pattern that exercises every
  `IntraBmode` plus the optional `segment_id` / `mb_skip_coeff`
  pass-through; the key-frame reset rule of
  `InterFrameIntraProbs::for_frame_header`; the interframe
  carry-forward when no override block is present; the wholesale
  Y+UV overlay when both are present; the mixed Y-only overlay; and
  the `Default` impl matching `defaults()`.
* The inter-predicted §16.2 / §16.3 / §16.4 branch (`mv_ref` tree,
  near/nearest/best census, motion-vector clamping, split-prediction
  sub-block walk) and the §17 motion-vector component decoding remain
  out of scope for this round.

* **Loop-filter per-segment kernels** per RFC 6386 §15 (new module
  `src/loop_filter.rs`). Each routine operates on a caller-supplied
  contiguous pixel window (the spec's "segment" symmetrically
  straddling one edge), so all routines are agnostic to
  horizontal-vs-vertical edge orientation. Surface:
  - §15.2 helpers `clamp_s8` (the spec's `c`), `u2s`, `s2u`, and
    `common_adjust` (the shared core adjustment; 4-tap with outer taps
    or 2-tap without; returns the signed `a` the subblock filter uses).
  - §15.2 `simple_segment` — the simple luma-only filter gated by the
    `abs(p0-q0)*2 + abs(p1-q1)/2 <= edge_limit` metric.
  - §15.3 `subblock_filter` — the normal inter-subblock filter
    (`filter_yes` enable test, `hev` high-edge-variance test, and the
    low-variance half-magnitude inner-pixel adjustment).
  - §15.3 `mb_filter` — the wider inter-macroblock `MBfilter` (six
    pixels adjusted with 3/7, 2/7, 1/7 decaying magnitude under low
    variance; `common_adjust` fall-back under high variance).
  - §15.4 `LoopFilterParams::derive` — computes `interior_limit`,
    `hev_threshold`, `mbedge_limit`, and `sub_bedge_limit` from a
    resolved per-macroblock `loop_filter_level`, the frame
    `sharpness_level`, and the key-frame flag (both `hev_threshold`
    ladders, the sharpness shift+cap on `interior_limit`, the two
    edge-limit formulas).
* Twenty-three new unit tests covering: §15.2 clamp saturation; `u2s` /
  `s2u` round trip over all 256 pixel values + known points +
  out-of-range clamps; §15.4 interior-limit derivation under no / low /
  high sharpness (cap + floor-to-1), both `hev_threshold` ladders at
  every boundary, and the edge-limit formulas (including the max-level
  fit); §15.2 simple-filter skip-vs-adjust plus two hand-derived
  `common_adjust` cases (with / without outer taps, re-deriving the
  spec arithmetic inline); §15.3 subblock / MB filter skip, low-hev
  inner-pixel adjustment, and high-hev fall-back branches; a fully
  hand-derived `mb_filter` low-variance case asserting all eight output
  pixels; and a base-offset test proving the kernels leave the
  surrounding buffer untouched.
* The §15.1 macroblock-by-macroblock filter geometry (raster-order edge
  walk + the §15.1 page-86 step-2/4 skip rule) and the §9.4 / §10
  per-macroblock `loop_filter_level` derivation are out of scope for
  this round; they belong to the per-macroblock reconstruction walk and
  call these kernels.

* **Dequantization and inverse transforms** per RFC 6386 §14 (new
  module `src/inverse_transform.rs`). All primitives operate on
  caller-supplied 4×4 arrays in raster (natural) order. Surface:
  - `DC_QLOOKUP[128]` / `AC_QLOOKUP[128]` — the §14.1 page-77 dequant
    lookup tables, transcribed verbatim and verified byte-for-byte
    against the RFC. `QINDEX_RANGE = 128`.
  - `clamp_qindex` — saturates a delta-adjusted 7-bit quantiser index
    into the `0..=127` table domain.
  - `Y1DequantFactors::from_indices(yac_qi, ydc_delta)` — the Y1-plane
    DC/AC factor computation per §14.1 (*"Lookup values ... are
    directly used in the DC and AC coefficients in Y1"*): DC from
    `dc_qlookup[clamp(yac_qi + ydc_delta)]`, AC from
    `ac_qlookup[clamp(yac_qi)]`.
  - `dequant_block` — multiplies a `[i16; 16]` (DC × dc_factor,
    AC × ac_factor) in `i32`, stored back as `i16` per §14.1.
  - `inverse_wht_4x4` — faithful port of §14.3's
    `vp8_short_inv_walsh4x4_c` (two passes, `(x + 3) >> 3` rounding);
    `inverse_wht_4x4_dc_only` — the single-non-zero-DC fast path
    `vp8_short_inv_walsh4x4_1_c`.
  - `inverse_dct_4x4` — faithful port of §14.4's `short_idct4x4llm_c`
    with the fixed-point constants `cospi8sqrt2minus1 = 20091` and
    `sinpi8sqrt2 = 35468`, two passes, `(x + 4) >> 3` rounding.
  - `add_residue_4x4` / `add_residue` / `clamp255` — the §14.5
    predictor + residue summation with 32-bit-precision sums saturated
    to 8-bit via the §14.5 `clamp255`.
* Eighteen new unit tests covering: dequant-table shape + spec-value
  spot-checks (verified against the RFC); qindex clamping at both ends;
  Y1 DC/AC lookup selection including delta over/underflow clamping;
  per-block DC-vs-AC scaling; `clamp255` boundary behaviour; WHT
  general-vs-fast-path equivalence over a value sweep; a hand-derived
  two-value WHT input traced through both passes; DCT zero-input,
  DC-only flat-block, and DC-rounding cases; a single-AC-coefficient
  DCT case re-deriving the spec arithmetic inline (proving the cosine
  constants land in the right lanes and produce a gradient); a full
  mixed-block DCT re-derivation guarding against a row/column
  transpose; and §14.5 summation saturation at both clamp ends, the
  slice-vs-4×4 form agreement, and the zero-residue identity.

### Spec gaps surfaced

* **§14.1 page 77 Y2 / chroma dequant scaling.** The RFC gives the raw
  `dc_qlookup` / `ac_qlookup` tables and says Y1 uses them directly,
  but the Y2-DC, Y2-AC, chroma-DC, and chroma-AC factors *"undergo
  either scaling or clamping before the multiplies. Details ... can be
  found in related lookup functions in dixie.c (Section 20.4)."* Those
  rules are not in the RFC body and `dixie.c` is off-limits reference
  source, so the four non-Y1 factors are not computed in this round.
  Recommend a clean-room note giving the Y2/chroma scaling/clamping.
* **§13 page 60 zig-zag scan order.** §13 names the coefficient
  ordering "zig-zag" but the 16-entry scan-to-raster permutation
  appears only in the reference `idct_add.c` (Section 20.8). The §14
  transforms here operate in raster order; the integration round must
  supply the reordering. Recommend a clean-room note with the
  permutation array.

* **DCT-coefficient token decoding** per RFC 6386 §13 (new module
  `src/dct_tokens.rs`). Walks the §13.2 `coeff_tree` against the
  `coeff_probs[4][8][3][11]` table populated by round 3's
  `token_prob_update()` and recovers a `[i16; 16]` of quantised
  coefficients per 4×4 sub-block. Surface:
  - `BlockType` — the §13.3 plane-type discriminator (`YAfterY2` /
    `Y2` / `UV` / `YNoY2`); `first_coeff()` returns 1 / 0 / 0 / 0;
    `plane_index()` returns the outermost `coeff_probs` index.
  - `DctToken` — the twelve §13.2 alphabet symbols (`Dct0..Dct4`,
    `Cat1..Cat6`, `Eob`).
  - `CoeffProbs` — the resolved `[[[[u8; 11]; 3]; 8]; 4]` table.
  - `COEFF_BANDS` — the §13.3 position-to-band lookup
    `[0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7]`.
  - `DEFAULT_COEFF_PROBS` — the §13.5 default token-probability
    table, all 4 × 8 × 3 × 11 = 1056 probabilities transcribed
    verbatim from RFC 6386.
  - `merge_default_token_probs` — overlays a `TokenProbUpdates`
    (from a parsed `Vp8CodedHeader`) onto `DEFAULT_COEFF_PROBS`
    to produce a resolved table; `None` entries leave defaults in
    place.
  - `decode_block` — the per-sub-block primitive. Walks
    `coeff_tree` from the appropriate start index per §13.2's
    "skip dct_eob branch when previous coefficient was DCT_0"
    rule, reads §13.2 `DCTextra` extra-bits against
    `Pcat1..Pcat6` for each `Cat*` token, reads the fixed-prob-128
    sign bit for non-zero values, and rolls over the §13.3 `ctx3`
    (`0` / `1` / `2`) for the next coefficient based on the
    just-decoded absolute value.
  - `DctTokenError` — wraps `BoolDecoderError` and surfaces
    `InvalidTokenIndex` for a corrupt tree-table.
* Eighteen new unit tests covering: default-probs transcription
  shape + four-plane spot-checks; `coeff_bands[16]` listing;
  `BlockType::first_coeff()` / `plane_index()`;
  `merge_default_token_probs` identity + overlay behaviour;
  immediate EOB → all-zero block; single-DCT1 round trip at
  position 0; negative-value round trip; round-trip of one value
  inside each of cat1..cat6 (thirteen magnitudes including range
  boundaries 5/6/7/10/11/18/19/34/35/66/67/100/2048);
  `BlockType::YAfterY2` skipping coefficient 0; a dense block with
  mixed positive / negative / cat3 / cat4 values across non-adjacent
  positions; an updates-overlaid round trip proving `decode_block`
  reads from the caller's resolved table (not defaults);
  `prev_token_skips_eob_branch` exercising the §13.2 EOB-skip rule
  through a leading run of zeros; all four `(above, left)`
  predictor combinations for the first-coefficient `ctx3` seed;
  a 16-position fully-occupied block that emits no EOB at all; and
  cat6 maximum value `categoryBase[5] + 0x7FF = 2114` exercising
  the 11-extra-bit `DCTextra` path.

### Spec gaps surfaced

* **§13.3 page 67 token-loop pseudocode.** The trailing
  `prevCoeffWasZero = true;` is a transcription error; setting the
  flag unconditionally true would let EOB follow a non-zero
  coefficient on every iteration after the first, contradicting
  §13.2's "if the preceding coefficient is a DCT_0, decoding will
  skip the first branch" statement. We implement
  `prevCoeffWasZero = (token == DCT_0)`. Recommend an RFC 6386
  erratum.

* **Intra-prediction pixel kernels** per RFC 6386 §12 (new module
  `src/intra_predict.rs`). Pure pixel-shape kernels operating on small
  neighbour-array inputs — no entropy decoding, no IDCT, no loop
  filter is performed yet. Surface:
  - `predict_y16x16_dc` / `_v` / `_h` / `_tm` — the four 16×16 luma
    modes per §12.3 (which forwards to the §12.2 formulas at the
    larger block size). `_v` / `_h` / `_tm` take `&[u8; 16]`
    above / left buffers; `_dc` takes `Option<&[u8; 16]>` so callers
    can encode "this edge is off-frame" and trigger the §12.2 fallback
    rules (single-edge average; top-left → constant 128).
  - `predict_uv8x8_dc` / `_v` / `_h` / `_tm` — the four 8×8 chroma
    modes per §12.2, same shape as the luma kernels at 8×8.
  - `predict_b4x4` — the ten 4×4 sub-block modes per §12.3:
    `B_DC_PRED`, `B_TM_PRED`, `B_VE_PRED`, `B_HE_PRED`, `B_LD_PRED`,
    `B_RD_PRED`, `B_VR_PRED`, `B_VL_PRED`, `B_HD_PRED`, `B_HU_PRED`.
    Takes the 8-pixel above row (positions `(-1, 0)..=(-1, 7)` of the
    sub-block's coordinate frame — the lower four are directly above
    the sub-block, the upper four are the "extra" pixels the §12.3
    right-edge fixup defines), the 4-pixel left column, and the
    single corner pixel `P`. Builds the spec's `E[0..=8]` array
    internally.
  - `predict_y16x16` / `predict_uv8x8` dispatchers that route on the
    decoded `IntraYMode` / `IntraUvMode` enum and apply the §12 page-50
    out-of-bounds defaults (127 above, 129 left) when an edge is
    `None`.
  - Public constants `DEFAULT_ABOVE_PIXEL = 127`,
    `DEFAULT_LEFT_PIXEL = 129`, `DEFAULT_TOPLEFT_DC = 128` for
    callers building neighbour buffers from raw frame data.
* Twenty-seven new unit tests with hand-derived expected pixels:
  - 16×16 luma: DC full-neighbour rounding, DC top-row-only,
    DC top-left default 128, V copies above, H copies left, TM
    matches the spec formula on a non-trivial ramp, TM clamping to
    0 and 255, dispatch routing all five modes,
    V dispatch off-frame default, H dispatch off-frame default
    (11 tests).
  - 8×8 chroma: DC full-neighbour rounding, DC left-column-only,
    DC top-left default 128, dispatch V/H/TM (4 tests).
  - 4×4 sub-block: DC averaging of 8 pixels, TM clamped formula,
    VE smoothed above row, HE smoothed left column with the
    `avg3(L[2], L[3], L[3])` bottom-row special case, LD top-left
    + bottom-right `avg3(A[6], A[7], A[7])` synthetic pixel, RD
    diagonal propagation through `E[4]`, VR mixing avg2p / avg3p,
    VL handling the right-extension `above[4..=7]` pixels, HD
    treating the spec's `svg2p` typo as `avg2p`, HD avg2p / avg3p
    disambiguator (proves we picked avg2p, since a constant-input
    test cannot distinguish them), HU bottom-row L[3] fill, HU
    top-row using the smoothed left column (11 tests).
  - One cross-cutting "flat input → flat output" sanity check that
    exercises every one of the ten sub-block modes (catches missing
    pixel writes in the kernels — found the original bug where my
    `B_HU_PRED` kernel forgot to set `B[1][2]`, `B[1][3]`, and
    `B[2][0]`).
* Updated `lib.rs` to re-export the new module's public API.

### Reference

RFC 6386 §12 in full (pages 50–59). The 16×16 / 8×8 mode shape and
the out-of-bounds defaults (127 / 129) come from §12.2, the DC
rounding formula `(sum + (1 << (shf - 1))) >> shf` is in §12.2
page 52, the TM clamp is in §12.2 page 53, and the 4×4 mode
listings (including the `E[0..=8]` array layout and the `avg2` /
`avg3` / `avg2p` / `avg3p` helper definitions) are in §12.3.

### Spec gaps surfaced

* **§12.3 page 58, B_HD_PRED listing.** The raw RFC text reads
  `svg2p(E + 1)` for the second pixel of row 3. No `svg2p` function
  is defined anywhere in the spec; the three sibling diagonal modes
  (B_VR_PRED, B_VL_PRED, B_HU_PRED) all use `avg2p` at the
  analogous position with no typo. We treat the token as `avg2p`
  and include `b4x4_hd_avg2p_vs_avg3p_disambiguator` to prove the
  resulting output discriminates between the two functions (rather
  than coincidentally agreeing on the test input). Recommend an
  RFC 6386 erratum.

* **Key-frame macroblock mode layer** per RFC 6386 §11
  (`parse_key_frame_macroblock_modes` in `src/macroblock.rs`).
  Consumes a `BoolDecoder` positioned immediately after the §19.2
  header and returns `Vec<MacroblockModes>` for the frame in raster
  order. Each record carries:
  - `segment_id` (§10) — `Some(0..=3)` when the frame enabled both
    `segmentation_enabled` and `update_mb_segmentation_map`; decoded
    by walking `mb_segment_tree` against the resolved
    `mb_segment_tree_probs[3]` (defaulting to 255 for entries whose
    `segment_prob_update_flag` was 0);
  - `mb_skip_coeff` (§11.1) — read against `prob_skip_false` only
    when `mb_no_skip_coeff` is set; forced to `false` otherwise;
  - `y_mode` (§11.2) — `kf_ymode_tree` walk against
    `KF_YMODE_PROB = {145, 156, 163, 128}` (one of `DC_PRED` /
    `V_PRED` / `H_PRED` / `TM_PRED` / `B_PRED`);
  - `subblock_modes` (§11.3 / §11.5) — sixteen `IntraBmode` values
    in raster j=0..15 order, present iff `y_mode == B_PRED`. Each
    is decoded against the `KF_BMODE_PROB[above][left][9]` row of
    the §11.5 `[10][10][9]` table (transcribed verbatim). The
    "above" / "left" indices are derived per §11.3: top-edge
    sub-blocks inherit the above macroblock's bottom row,
    left-edge sub-blocks inherit the left macroblock's right
    column, frame-edge predictors default to `B_DC_PRED`, and a
    non-`B_PRED` neighbouring macroblock projects its 16x16 luma
    mode to a single sub-block context (`DC->B_DC`, `V->B_VE`,
    `H->B_HE`, `TM->B_TM`);
  - `uv_mode` (§11.4) — `uv_mode_tree` walk against
    `KF_UV_MODE_PROB = {142, 114, 183}`.
* **`Vp8CodedHeader::parse_with_decoder`** — new public entry point
  that returns both the parsed header and the still-mutable
  `BoolDecoder`, so the macroblock layer can keep reading from the
  same partition without replaying §19.2. The existing
  `Vp8CodedHeader::parse` is now a thin wrapper that drops the
  decoder.
* New public types: `IntraYMode`, `IntraUvMode`, `IntraBmode`,
  `MacroblockModes`, `MacroblockError`.
* Nine new unit tests covering: exhaustive leaf round-trips for
  `kf_ymode_tree` (5 leaves), `uv_mode_tree` (4 leaves),
  `bmode_tree` (10 leaves), and `mb_segment_tree` (4 leaves); an
  end-to-end 2-macroblock decode exercising a `DC_PRED` MB plus a
  `B_PRED` MB with sixteen sub-block modes (verifying the cross-MB
  context buffer wiring); a segmentation + `mb_skip_coeff` path
  exercising the optional §10 / §11.1 prefix bits; an
  interframe-header guard test; and `KF_YMODE_PROB` /
  `KF_UV_MODE_PROB` / `KF_BMODE_PROB` spot-checks that would catch a
  transcription typo.
* This round stays structural: no actual pixel prediction (§12),
  DCT-coefficient decode (§13), motion-vector decode (§17), IDCT
  (§14) or loop filter (§15) is performed yet. The returned modes
  are the input to subsequent rounds.

* **Boolean-coded frame-header §9.10 inter-only tail** completing the
  §19.2 syntax table. The existing `Vp8CodedHeader::parse` now decodes,
  after `prob_skip_false`, every remaining field listed for an
  interframe in §9.10:
  - `prob_intra` (L8), `prob_last` (L8), `prob_gf` (L8) — the three
    macroblock reference-selection probabilities;
  - the `F? L(8) × 4` block of intra-Y mode probability replacements
    (§9.10 / §16.1) — when the F is 0 the four §16.1 default values
    `{112, 86, 140, 37}` remain in force and the bitstream omits the
    four L(8) values;
  - the `F? L(8) × 3` block of intra-UV mode probability replacements
    (defaults `{162, 101, 204}`);
  - `mv_prob_update()` per RFC 6386 §17.2 — two 19-position
    MV_CONTEXTs (row then column), each position is an `F? P(7)`
    update read at the per-position `MV_UPDATE_PROBS` probability
    (transcribed verbatim from the spec's `vp8_mv_update_probs[2]`)
    with the spec's `x ? x<<1 : 1` P(7) reconstruction. Key frames
    omit the entire tail (`prob_intra` etc. all collapse to `None`).
  The MV update-probabilities table and the `default_mv_context[2]`
  table are both transcribed verbatim from §17.2 and the latter is
  re-exported as the public `DEFAULT_MV_CONTEXT` constant so the
  macroblock-decode round can seed its `MV_CONTEXT mvc[2]` table.
* New public types: `MvProbUpdates` (alias for
  `[[Option<u8>; 19]; 2]`); plus the `MV_PROB_COUNT = 19` constant
  matching §17 `MVPcount`. Six new fields on `Vp8CodedHeader`
  (`prob_intra`, `prob_last`, `prob_gf`, `intra_y_mode_prob_update`,
  `intra_uv_mode_prob_update`, `mv_prob_update`) — each `Option`-typed
  to surface key-frame-vs-inter framing at the type level.
* Six new unit tests covering: the new `key_frame_omits_section_9_10_tail`
  invariant; an extended `interframe_refresh_block_full_path`
  asserting on the full-zeroed tail; `prob_intra` / `prob_last` /
  `prob_gf` round-trip across `0 / 128 / 255`; the gated Y intra-mode
  block (F = 1, four overrides); the gated UV intra-mode block;
  `mv_prob_update()` exercising both branches of the §17.2
  `x ? x<<1 : 1` reconstruction and the per-position `MV_UPDATE_PROBS`
  read; and a `mv_default_context_matches_spec_listing` sanity check
  that would surface a transcription typo in the default-MV table.

* **Boolean-coded frame-header prefix** per RFC 6386 §19.2
  (`Vp8CodedHeader::parse` in `src/coded_header.rs`). Reads through
  the `BoolDecoder` over the `first_partition_size`-byte control
  partition that follows the uncompressed frame header and decodes
  every field up to and including `prob_skip_false`:
  key-frame-only `color_space` / `clamping_type` (§9.2);
  `segmentation_enabled` and the full `update_segmentation()`
  sub-block (§9.3); `filter_type`, 6-bit `loop_filter_level`,
  3-bit `sharpness_level`, and `mb_lf_adjustments()` (§9.4); 2-bit
  `log2_nbr_of_dct_partitions` (§9.5) plus the convenient
  `nbr_of_dct_partitions = 1 << log2_nbr_of_dct_partitions`;
  `quant_indices()` (§9.6); `token_prob_update()` over the
  `[4][8][3][11]` DCT context table — each `coeff_prob_update_flag`
  is read against the per-position `coeff_update_probs` table from
  §13.4 (NOT a flat 128), followed by the optional `L(8)`
  replacement probability; the inter-frame-only refresh / copy /
  sign-bias ladder of §9.7 / §9.8 (`refresh_golden_frame`,
  `refresh_alternate_frame`, conditional `copy_buffer_to_golden` /
  `copy_buffer_to_alternate`, `sign_bias_golden`, `sign_bias_alternate`,
  `refresh_last`); `refresh_entropy_probs` (every frame); and
  `mb_no_skip_coeff` with its conditional 8-bit `prob_skip_false`
  (§9.10 / §9.11). The `[4][8][3][11]` `COEFF_UPDATE_PROBS` table
  used to gate `token_prob_update()` reads is transcribed verbatim
  from RFC 6386 §13.4.
* Nine new unit tests in `coded_header::tests`, covering: minimal
  key-frame round trip with every optional block absent; maxed-out
  filter / partition / quantiser fields; the full
  `mb_lf_adjustments` sub-block with mixed signs across the eight
  delta entries; `update_segmentation()` with mixed-presence
  quantiser deltas, loop-filter deltas, and segment-probability
  entries; quant_indices with all five deltas present (positive,
  negative, max-magnitude, zero); the inter-frame refresh ladder
  including `copy_buffer_to_golden`; `mb_no_skip_coeff = 0`
  omitting `prob_skip_false`; `InputTooShort` surfaced through the
  `CodedHeaderError::BoolDecoder` wrapper; and an end-to-end parse
  of the 23-byte first partition extracted from
  `docs/video/vp8/fixtures/tiny-i-only-16x16/input.ivf` that
  matches every field surfaced in the fixture's `trace.txt`.

* **Uncompressed frame header parser** per RFC 6386 §9.1 / §19.1
  (`Vp8FrameHeader::parse` in `src/frame_header.rs`). Splits the
  3-byte little-endian frame tag into `key_frame`, 3-bit `version`,
  `show_frame`, and 19-bit `first_partition_size`. Maps `version` to
  the §9.1 Table 1 `ReconstructionFilter` / `LoopFilterPolicy`
  enums. On key frames validates the `0x9d 0x01 0x2a` start code and
  splits each of the two LE 16-bit size words into a 14-bit
  dimension and a 2-bit `ScaleCode`. Surfaces `header_bytes_consumed`
  (3 for interframes, 10 for key frames) so callers can advance to
  the first (control) partition. Nine unit tests, including a
  sanity-check parse of the actual first 10 bytes of
  `docs/video/vp8/fixtures/tiny-i-only-16x16/input.ivf`.
* **Boolean (range) entropy decoder** per RFC 6386 §7
  (`BoolDecoder` in `src/bool_decoder.rs`). This is the foundational
  primitive every higher-level VP8 decode step reads through. Surface:
  `init`, `read_bool`, `read_literal`, `read_signed_literal`, plus
  `range()` / `value()` / `remaining_input()` accessors for testing.
  An `EndOfStream` error is surfaced explicitly when renormalisation
  needs a byte the partition no longer has, rather than silently
  returning stale bits.
* Nine unit tests covering init validation, round-trip against an
  in-test reference encoder over a probability sweep, literal /
  signed-literal handling (including the spec's `-1`-initialised
  signed accumulator), the `128 ≤ range ≤ 255` invariant across
  reads, and the end-of-stream signal.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained. License also reset to clean MIT.

  Every public API path other than the new `BoolDecoder` still
  returns `Error::NotImplemented`. A clean-room re-implementation of
  the frame header, macroblock decode, prediction, IDCT, and loop
  filter is planned for subsequent rounds.
