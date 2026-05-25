# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Added

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
