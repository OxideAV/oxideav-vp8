# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Added

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
