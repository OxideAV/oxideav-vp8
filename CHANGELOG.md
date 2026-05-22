# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Added

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
