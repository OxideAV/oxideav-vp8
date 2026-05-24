# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status — 2026-05-24 (round 13)

**Clean-room rebuild in progress.** The prior implementation was
retired under the workspace clean-room policy after a provenance audit
on 2026-05-20. Rebuild work tracks RFC 6386 exclusively, with
black-box `ffmpeg` invocations as the only validator.

### Landed

**Round 1 (2026-05-20).** `BoolDecoder` — the VP8 boolean (range)
entropy decoder of RFC 6386 §7. Every higher-level decode step in VP8
(frame header, macroblock mode, motion vectors, DCT tokens) reads
through this primitive, so it is the foundation everything else builds
on. Surface:

  * `init(&[u8])` — load the first two bytes of a partition into the
    `value` register; reject inputs shorter than two bytes.
  * `read_bool(prob: u8)` — read one boolean coded at probability
    `prob/256` of being zero.
  * `read_literal(num_bits)` — `num_bits` flags read MSB-first.
  * `read_signed_literal(num_bits)` — sign-bit-then-magnitude with
    the spec's `-1`-initialised accumulator.
  * Surfaces `EndOfStream` when renormalisation needs a byte the
    partition no longer has.

**Round 2 (2026-05-21).** `Vp8FrameHeader::parse` — the uncompressed
VP8 frame header per RFC 6386 §9.1 / §19.1. The 3-byte little-endian
frame tag is split into `key_frame`, 3-bit `version`, `show_frame`,
and 19-bit `first_partition_size`. The `version` field is mapped to
the §9.1 Table 1 `ReconstructionFilter` / `LoopFilterPolicy` enums.
On key frames the mandatory `0x9d 0x01 0x2a` start code is validated
and the two 16-bit little-endian size words are split into 14-bit
width / height and 2-bit horizontal / vertical scale codes. The
parser surfaces `header_bytes_consumed` (3 for interframes, 10 for
key frames) so callers can advance the cursor to the start of the
first (control) partition.

**Round 3 (2026-05-21).** `Vp8CodedHeader::parse` — the
boolean-coded frame-header prefix of RFC 6386 §19.2, read over the
`first_partition_size`-byte control partition that immediately
follows the uncompressed header. Decoded fields:

  * key-frame-only `color_space` / `clamping_type` (§9.2);
  * `segmentation_enabled` and the full `update_segmentation()`
    sub-block — per-segment quantiser + loop-filter deltas and the
    optional `segment_prob` table (§9.3);
  * `filter_type`, 6-bit `loop_filter_level`, 3-bit `sharpness_level`
    (§9.4);
  * `mb_lf_adjustments()` — the `loop_filter_adj_enable` toggle, the
    `mode_ref_lf_delta_update` follow-up, and the four reference-frame
    and four prediction-mode `delta_magnitude (L6) + delta_sign (L1)`
    entries (§9.4);
  * 2-bit `log2_nbr_of_dct_partitions` (§9.5), surfaced together with
    the decoded `nbr_of_dct_partitions ∈ {1, 2, 4, 8}`;
  * `quant_indices()` — baseline `y_ac_qi (L7)` and the five
    `present? + L4 magnitude + L1 sign` deltas for ydc / y2dc / y2ac /
    uvdc / uvac (§9.6);
  * `token_prob_update()` — the full 4 × 8 × 3 × 11 sweep of
    `coeff_prob_update_flag`s, each read against the per-position
    `coeff_update_probs` table transcribed verbatim from §13.4 (NOT
    flat probability 128), followed by the optional L(8) replacement
    probability. The decoded probability table is exposed for the
    later macroblock-decoder round; the immediate purpose of consuming
    it here is to reach the `mb_no_skip_coeff` bit that follows it;
  * `refresh_entropy_probs` (every frame), the inter-frame-only
    refresh / copy / sign-bias ladder
    (`refresh_golden_frame`, `refresh_alternate_frame`, optional
    `copy_buffer_to_golden`, optional `copy_buffer_to_alternate`,
    `sign_bias_golden`, `sign_bias_alternate`, `refresh_last` —
    §9.7 / §9.8);
  * `mb_no_skip_coeff` and the conditional `prob_skip_false (L8)`
    (§9.10 / §9.11).

**Round 4 (2026-05-21).** Closes the §19.2 syntax table by adding the
remaining inter-frame-only tail — every field that §9.10 lists after
`prob_skip_false` and the §17.2 motion-vector probability updates:

  * `prob_intra` / `prob_last` / `prob_gf` — three L(8) probabilities
    governing intra-vs-inter and reference-frame selection at the
    macroblock level (§9.10 / §16);
  * a single F gate followed by four `L(8)` overrides for the intra-Y
    mode probabilities (§16.1; defaults `{112, 86, 140, 37}` apply
    when the F is 0 and the four L(8)s are absent);
  * the analogous F-gated block of three `L(8)` intra-UV mode
    probability overrides (defaults `{162, 101, 204}`);
  * `mv_prob_update()` from §17.2 — two 19-position MV_CONTEXTs
    (row then column), each position is `F? P(7)` where the F is
    read at the per-position `MV_UPDATE_PROBS` value (the spec's
    `vp8_mv_update_probs[2]`, transcribed verbatim) and the L(7) `x`
    reconstructs to `x << 1` when non-zero, else `1`. The
    `default_mv_context[2]` table from §17.2 is also transcribed
    verbatim and re-exported as the public `DEFAULT_MV_CONTEXT`
    constant for the macroblock-decode round to seed `mvc[2]` from.

**Round 5 (2026-05-22).** Adds the key-frame macroblock mode layer
(RFC 6386 §11): `parse_key_frame_macroblock_modes` consumes a
[`BoolDecoder`] positioned immediately after the §19.2 header and
returns one `MacroblockModes` record per macroblock in raster order.
Each record carries:

  * `segment_id` (`§10`) — `Some(0..=3)` when the frame enabled both
    `segmentation_enabled` and `update_mb_segmentation_map`; the
    2-bit code is walked through `mb_segment_tree` against
    `mb_segment_tree_probs[3]` (defaulting to 255 for entries whose
    `segment_prob_update_flag` was 0 per §9.3 item 5);
  * `mb_skip_coeff` (`§11.1`) — read against `prob_skip_false` when
    `mb_no_skip_coeff` is set; forced to `false` otherwise;
  * `y_mode` (`§11.2`) — `kf_ymode_tree` walk against the constant
    `KF_YMODE_PROB = {145, 156, 163, 128}`; one of `DC_PRED` /
    `V_PRED` / `H_PRED` / `TM_PRED` / `B_PRED`;
  * `subblock_modes` (`§11.3` / `§11.5`) — `Some([IntraBmode; 16])`
    iff `y_mode == B_PRED`. Each of the sixteen 4x4 sub-blocks is
    decoded against `KF_BMODE_PROB[above][left]` (the §11.5
    `[10][10][9]` table, transcribed verbatim) using the `bmode_tree`
    from §11.2. Cross-macroblock context tracking handles §11.3
    items 2/3/4 — top-edge sub-blocks inherit the above-MB's bottom
    row, left-edge sub-blocks inherit the left-MB's right column,
    frame-edge predictors default to `B_DC_PRED`, and non-`B_PRED`
    macroblocks project their 16x16 luma mode to a constant
    sub-block context (`DC->B_DC`, `V->B_VE`, `H->B_HE`,
    `TM->B_TM`);
  * `uv_mode` (`§11.4`) — `uv_mode_tree` walk against the constant
    `KF_UV_MODE_PROB = {142, 114, 183}`.

The new `Vp8CodedHeader::parse_with_decoder` entry point returns
both the parsed header and the still-mutable `BoolDecoder` so the
macroblock layer can keep reading from the same partition without
replaying §19.2.

This round stays **structural**: no actual pixel prediction (§12),
DCT-coefficient decode (§13), motion-vector decode (§17), IDCT (§14)
or loop filter (§15) is performed yet. The returned modes are the
input to subsequent rounds.

**Round 6 (2026-05-22).** Adds the intra-prediction pixel kernels
(RFC 6386 §12): a new `src/intra_predict.rs` module implementing
all four 16×16 luma modes (`DC_PRED` / `V_PRED` / `H_PRED` /
`TM_PRED`), all four 8×8 chroma modes (the same four modes at 8×8
per §12.2), and all ten 4×4 sub-block modes (`B_DC_PRED`,
`B_TM_PRED`, `B_VE_PRED`, `B_HE_PRED`, `B_LD_PRED`, `B_RD_PRED`,
`B_VR_PRED`, `B_VL_PRED`, `B_HD_PRED`, `B_HU_PRED`). Each kernel is
a pure pixel-shape primitive operating on small caller-supplied
neighbour arrays: an `above` row, a `left` column, and a single
corner pixel `P` (the §12.3 `A[-1] == L[-1]` value). No entropy
decode, no IDCT, and no loop filter is performed in this round.

  * The 16×16 and 8×8 DC modes accept `Option<&[u8; N]>` for
    `above` / `left` to encode the §12.2 page-51 fallback rules:
    when only one edge is on-frame the DC is the rounded average
    of that one edge's `N` pixels; when both are off-frame the
    block is filled with the constant 128 (NOT the average of the
    127 / 129 defaults).
  * The 16×16 / 8×8 dispatchers apply the §12 page-50 defaults
    (127 above, 129 left) to `V_PRED` / `H_PRED` / `TM_PRED` when
    a directly-required edge is off-frame.
  * `predict_b4x4` takes an 8-pixel `above` row covering positions
    `(-1, 0) .. (-1, 7)`. The lower four are directly above the
    sub-block, the upper four are the "extra" pixels the §12.3
    right-edge fixup defines (for sub-blocks 7 / 11 / 15 these are
    the same four pixels used by sub-block 3, and at the rightmost
    macroblock in each row they are clamped to position `(-1, 15)`).
    Computing the fixup is the caller's responsibility; the kernel
    just reads the buffer it's handed.

The new module surfaces three public constants — `DEFAULT_ABOVE_PIXEL`
(127), `DEFAULT_LEFT_PIXEL` (129), `DEFAULT_TOPLEFT_DC` (128) — so
callers can build neighbour buffers from raw frame data without
re-deriving the spec constants.

Sixty-nine unit tests across five modules: the forty-two carried
forward from rounds 1 / 2 / 3 / 4 / 5 plus twenty-seven new round-6
tests with hand-derived expected pixels for every kernel — DC
rounding under each of the three edge-availability cases (full /
single-edge / both-off), V / H copy semantics, TM both as a generic
formula match and as floor/ceiling clamp probes, the
mode-dispatch routing including the off-frame default fall-through,
every one of the ten 4×4 modes individually (including the
`avg3(L[2], L[3], L[3])` bottom-row special case in B_HE_PRED, the
`avg3(A[6], A[7], A[7])` synthetic-pixel special case in B_LD_PRED,
the right-extension `above[4..=7]` consumption in B_VL_PRED, and a
disambiguating test that proves B_HD_PRED uses `avg2p` rather than
the spec's typo'd `svg2p`), and a cross-cutting "flat input → flat
output" check that catches missing pixel-writes in any of the ten
sub-block kernels.

**Round 7 (2026-05-22).** Adds the DCT-coefficient token-tree decoder
(RFC 6386 §13). A new `src/dct_tokens.rs` walks the `coeff_tree` of
§13.2 against the `coeff_probs[4][8][3][11]` table populated by round
3's `token_prob_update()` and recovers a `[i16; 16]` of quantised
coefficients per 4×4 sub-block. The full §13.5 default
probability table is transcribed verbatim (4 × 8 × 3 × 11 = 1056
probabilities) and exposed as `DEFAULT_COEFF_PROBS`. New surface:

  * `BlockType` — the §13.3 plane discriminator (`YAfterY2` / `Y2`
    / `UV` / `YNoY2`), with `first_coeff()` returning the §13.3
    `firstCoeff` value (1 for plane 0, 0 for the other three) and
    `plane_index()` returning the outermost `coeff_probs` index.
  * `DctToken` — the twelve-symbol §13.2 token alphabet (`Dct0..Dct4`,
    `Cat1..Cat6`, `Eob`).
  * `CoeffProbs` — the resolved `[[[[u8; 11]; 3]; 8]; 4]` table type.
  * `COEFF_BANDS` — the §13.3 position-to-band lookup.
  * `merge_default_token_probs` — overlays a `TokenProbUpdates` (from
    the parsed `Vp8CodedHeader`) onto `DEFAULT_COEFF_PROBS` to
    produce the resolved table.
  * `decode_block` — the per-sub-block primitive: walks
    `coeff_tree` (starting at index 0 when EOB is legal, index 2
    after a `Dct0` per §13.2's "skip dct_eob branch" rule), reads
    the §13.2 `DCTextra` extra-bits for the six `Cat*` tokens
    against the fixed `Pcat1..Pcat6` probability lists, reads the
    fixed-prob-128 sign bit for non-zero values, and rolls over
    the §13.3 `ctx3` (`0` / `1` / `2`) for the next coefficient
    based on the just-decoded absolute value.

The round stays decoder-side and per-block: dequantisation, IDCT,
the per-macroblock walker, and the §13.3 above/left non-zero-block
predictor maintenance are all explicitly **not** in scope here —
they belong to the §14 IDCT and the round-7+ integration layer.

Eighteen new unit tests round-trip a test-side bool encoder + a
recursive `coeff_tree` walker through `decode_block` and cover:
default-probs transcription shape + spot-checks on four planes, the
`coeff_bands[16]` listing, `BlockType::first_coeff()` / `plane_index()`,
`merge_default_token_probs` identity + overlay behaviour, immediate
EOB → all-zero block, single-DCT1 round trip, negative round trip,
each of the cat1..cat6 ranges (13 specific magnitudes including
range boundaries and 2114 = `categoryBase[5] + 0x7FF`), the
`YAfterY2` plane skipping coefficient 0, a dense block with mixed
positive / negative / cat3 / cat4 values, an updates-overlaid round
trip proving `decode_block` reads from the caller table rather than
defaults, a `prev_token_skips_eob_branch` test that exercises the
§13.2 EOB-skip-after-`Dct0` rule, all four `ctx3` seed values from
(above, left) ∈ {0,1}², a 16-position fully-occupied block that
emits no EOB, and the cat6 maximum 2114 value with 11-bit
`DCTextra` exercised.

**Round 8 (2026-05-23).** Adds the dequantization tables and the
inverse transforms of RFC 6386 §14 in a new
`src/inverse_transform.rs`, all operating on caller-supplied 4×4
arrays in raster (natural) order:

  * `DC_QLOOKUP[128]` / `AC_QLOOKUP[128]` — the two §14.1 page-77
    dequant lookup tables, transcribed verbatim (verified
    byte-for-byte against the RFC). `QINDEX_RANGE = 128` and
    `clamp_qindex` saturate a delta-adjusted 7-bit index into the
    table domain.
  * `Y1DequantFactors::from_indices(yac_qi, ydc_delta)` — the Y1-plane
    factor computation per §14.1's *"Lookup values from the above two
    tables are directly used in the DC and AC coefficients in Y1"*:
    DC = `dc_qlookup[clamp(yac_qi + ydc_delta)]`, AC =
    `ac_qlookup[clamp(yac_qi)]`. `dequant_block` multiplies a 4×4 of
    coefficients (DC × dc_factor, AC × ac_factor) in `i32` and stores
    back as `i16`.
  * `inverse_wht_4x4` — a faithful port of §14.3's
    `vp8_short_inv_walsh4x4_c` (two passes, `(x + 3) >> 3` rounding),
    plus `inverse_wht_4x4_dc_only` for the single-non-zero-DC fast
    path `vp8_short_inv_walsh4x4_1_c`.
  * `inverse_dct_4x4` — a faithful port of §14.4's `short_idct4x4llm_c`
    using the two 16-bit fixed-point constants `cospi8sqrt2minus1 =
    20091` and `sinpi8sqrt2 = 35468`, two passes of the On2 4-point
    1-D inverse DCT, `(x + 4) >> 3` second-pass rounding.
  * `add_residue_4x4` / `add_residue` / `clamp255` — the §14.5
    predictor + residue summation, each pixel computed at 32-bit
    precision and saturated to 8-bit via the §14.5 `clamp255`.

Eighteen new unit tests: table shape + spec-value spot-checks (verified
against the RFC), qindex clamping at both ends, Y1 DC/AC factor lookup
selection, per-block DC/AC scaling, `clamp255` boundary behaviour, the
WHT general-vs-fast-path equivalence over a value sweep, a hand-derived
two-value WHT input traced through both passes, the DCT DC-only
flat-block and rounding cases, a single-AC-coefficient DCT case
re-deriving the spec arithmetic inline (proving the cosine constants
land in the right lanes and produce a gradient, not a flat block), a
full mixed-block DCT re-derivation guarding against a row/column
transpose, and the §14.5 summation saturation at both clamp ends.
Total: 104 tests across six modules.

This round stays per-block and raster-order. The §14.2 macroblock
orchestration (Y2 → 16 Y-DC seeding, the 24/25-block walk), the
zig-zag → raster coefficient reordering, and the Y2/chroma dequant
scaling are all explicitly **not** in scope — see the two §14 spec
gaps below.

### Spec gaps surfaced (round 8)

**§14.1 page 77 — Y2 / chroma dequant scaling (RESOLVED round 15).** The
RFC body gives the raw `dc_qlookup` / `ac_qlookup` tables and states Y1
uses them directly, but the Y2-DC, Y2-AC, chroma-DC, and chroma-AC
factors *"undergo either scaling or clamping before the multiplies.
Details ... can be found in related lookup functions in dixie.c (Section
20.4)."* Section 20.4 (`dixie.c`) is the RFC's own reference-decoder
annex — part of RFC 6386, not external source — so its `dequant_init`
rules are in-spec: Y2 DC × 2, Y2 AC × 155/100 floored at 8, chroma DC
capped at 132, with the §20.4 `clamp_q` index saturation. Round 15
implements all six factors in `src/dequant.rs`.

**§13 page 60 — zig-zag scan order (RESOLVED round 14).** §13 names the
coefficient ordering "zig-zag" but the §13 body gives no permutation
array. The 16-entry scan-to-raster permutation is, however, present in
the §20.16 (tokens.c) reference annex as `zigzag[16]` — part of the RFC
itself. The round-14 §13.3 per-MB walk (`decode_mb_coeffs`) reorders each
decoded block into raster order using that table (`ZIGZAG`), closing the
gap; the §14 transforms continue to operate in raster order.

### Spec gap surfaced

**§13.3 page 67 pseudocode.** The token loop ends with the literal
statement `prevCoeffWasZero = true;` — i.e. *unconditionally true*.
That is a transcription error: the field controls whether the next
iteration's tree-walk starts at index 2 (skipping the dct_eob
branch) per §13.2's "if the preceding coefficient is a DCT_0,
decoding will skip the first branch" statement. Unconditionally
true would mean every coefficient after the first allows
`eob-after-non-zero`, which contradicts the §13.2 wording. We
implement `prevCoeffWasZero = (token == DCT_0)`. The
`prev_token_skips_eob_branch` test proves the round-trip works
either way the encoder writes it. Recommend an RFC 6386 erratum.

**Round 9 (2026-05-24).** Adds the loop-filter per-segment kernels of
RFC 6386 §15 in a new `src/loop_filter.rs`, all operating on a
caller-supplied contiguous pixel window (the spec's "segment" — the
2/4/6/8 pixels symmetrically straddling one edge), so the routines are
agnostic to horizontal-vs-vertical edge orientation just as the RFC's
reference routines are:

  * §15.2 helpers — `clamp_s8` (the spec's `c`), `u2s` / `s2u`, and
    `common_adjust`, the shared core edge adjustment (4-tap with outer
    taps, or 2-tap without), returning the signed `a` the subblock
    filter consumes;
  * §15.2 `simple_segment` — the simple luma-only filter gated by the
    `abs(p0-q0)*2 + abs(p1-q1)/2 <= edge_limit` metric;
  * §15.3 normal filter — the `filter_yes` enable test, the `hev`
    high-edge-variance test, `subblock_filter` (inter-subblock variant,
    with the low-variance half-magnitude inner-pixel adjustment), and
    `mb_filter` (the wider inter-macroblock `MBfilter` touching six
    pixels with 3/7, 2/7, 1/7 decaying magnitude, falling back to
    `common_adjust` under high variance);
  * §15.4 `LoopFilterParams::derive` — computes `interior_limit`,
    `hev_threshold`, `mbedge_limit`, and `sub_bedge_limit` from a
    resolved per-macroblock `loop_filter_level`, the frame
    `sharpness_level`, and the key-frame flag (the key-frame vs.
    interframe `hev_threshold` ladders, the sharpness shift+cap on
    `interior_limit`, and the two edge-limit formulas).

The round stays per-segment and primitive. The §15.1 macroblock-by-
macroblock filter *geometry* (the raster-order walk gathering the 16
luma / 8 chroma segments straddling each of the four edges per MB, the
ordered four filtering steps, and the §15.1 page-86 skip rule) and the
§9.4 / §10 derivation of the per-macroblock `loop_filter_level` itself
(segment override + reference-frame / prediction-mode deltas) are
explicitly **not** in scope — they belong to the per-macroblock
reconstruction walk (the integration round), which calls these kernels.

Twenty-three new unit tests: §15.2 clamp saturation, the `u2s` / `s2u`
round trip over all 256 pixel values + known points + out-of-range
clamps; §15.4 interior-limit derivation under no / low / high sharpness
(including the cap and the floor-to-1), both `hev_threshold` ladders at
every boundary, and the edge-limit formulas (including the max-level
fit); §15.2 simple-filter skip-vs-adjust plus two hand-derived
`common_adjust` cases (with and without outer taps, re-deriving the
spec arithmetic inline); §15.3 subblock / MB filter skip, low-hev
(inner-pixel adjustment), and high-hev (fall-back) branches, a fully
hand-derived `mb_filter` low-variance case asserting all eight output
pixels, and a base-offset test proving the kernels leave the
surrounding buffer untouched. Total: 127 tests across seven modules.

**Round 10 (2026-05-24).** Adds the interframe intra-predicted
macroblock-mode layer of RFC 6386 §16.1 (extending
`src/macroblock.rs`). The §16.1 layout mirrors §11 structurally but
uses different trees and probability tables:

  * `IF_YMODE_PROB_DEFAULTS = [112, 86, 140, 37]`,
    `IF_UV_MODE_PROB_DEFAULTS = [162, 101, 204]`, and the fixed
    `IF_BMODE_PROB = [120, 90, 79, 133, 87, 85, 80, 111, 151]` (a
    single nine-tuple — no above/left context, unlike the §11.5
    `[10][10][9]` key-frame table);
  * `InterFrameIntraProbs::for_frame_header(previous, header)` — the
    per-frame Y/UV probability state. On a key frame, both dynamic
    tables reset to the §16.1 defaults per the section's last
    paragraph; on an interframe, the resolved state is `previous`
    with the §9.10 F-gated `intra_y_mode_prob_update` /
    `intra_uv_mode_prob_update` overlays applied wholesale (or
    carried forward unchanged when the override block is `None`);
  * `parse_inter_frame_intra_macroblock_modes(dec, probs, segment_id,
    mb_skip_coeff)` — decode one §16.1 intra MB. Reads the Y mode
    (`IF_YMODE_TREE` against `probs.y_mode_prob`; the root left-leaf
    is `DC_PRED`, not `B_PRED` as on key frames), the sixteen
    sub-block modes when Y is `B_PRED` (shared `BMODE_TREE` against
    `IF_BMODE_PROB`, every sub-block reads the same nine-tuple), and
    the UV mode (shared `UV_MODE_TREE` against `probs.uv_mode_prob`).
    The optional `segment_id` (§10) and `mb_skip_coeff` (§11.1) bits
    precede the intra-vs-inter discriminator on interframes and are
    consumed before this entry point — the caller passes them in and
    they round-trip into the returned `MacroblockModes`.

Twelve new unit tests: spec-listing transcription of all three §16.1
default tables; the `IF_YMODE_TREE` shape literal match and an
explicit `IF_YMODE_TREE[0] != KF_YMODE_TREE[0]` divergence check;
round-trip of all five Y modes through `IF_YMODE_TREE` with the
default probabilities; all four UV modes through the shared
`UV_MODE_TREE` with `IF_UV_MODE_PROB_DEFAULTS`; all ten sub-block
modes through `BMODE_TREE` with `IF_BMODE_PROB`; a non-`B_PRED` MB
round-trip with elided optional fields; a `B_PRED` MB round-trip
with a sixteen-entry mixed sub-block pattern that exercises every
`IntraBmode` plus `segment_id = Some(2)` and `mb_skip_coeff = true`
pass-through; key-frame reset of the dynamic state; interframe
carry-forward when no overlay block is present; wholesale Y+UV
overlay when both are present; mixed Y-only overlay; and the
`Default` impl matching `defaults()`. Total: 139 tests across seven
modules.

**Round 11 (2026-05-24).** Adds the §14.2 per-macroblock
reconstruction orchestrator — the glue that ties together the
previously-isolated transform / prediction / summation primitives
(new `src/reconstruct.rs`).

* `decode_keyframe_mb_non_bpred(y_mode, uv_mode, mb_skip_coeff,
  neighbors, y2_coeffs_dequant, y_coeffs_dequant, u_coeffs_dequant,
  v_coeffs_dequant) -> Result<ReconstructedMb, ReconstructError>` —
  runs the §14.2 four-step recipe for one macroblock whose Y mode is
  one of the four 16×16 modes: (1) inverse-WHT the Y2 block and seed
  each Y sub-block's coefficient 0 with `wht_output[i*4+j]` per the
  §14.2 first-paragraph index rule; (2) inverse-DCT all sixteen Y
  and eight chroma sub-blocks (the §14.2 second-paragraph
  "24 inversions are independent" statement); (3) apply the §12
  intra-prediction kernel selected by the §11 mode record; (4) sum
  with `clamp255` (§14.5).
* `MbNeighbors { y_above, y_left, y_topleft, u_above, u_left,
  u_topleft, v_above, v_left, v_topleft }` — the surrounding pixel
  context the §12 kernels read. All fields are `Option`; absence
  invokes the spec's default-substitution rules in
  [`intra_predict`] (127 / 129 / 128).
* `ReconstructedMb { y: [u8; 256], u: [u8; 64], v: [u8; 64] }` —
  the predictor-plus-residue output for the macroblock, before
  loop filtering.
* `ReconstructError::BPredNotSupported` — surfaced when called with
  `y_mode == IntraYMode::B`. The `B_PRED` path needs a
  per-sub-block intra-walker that re-uses each sub-block's
  reconstructed pixels as the next sub-block's `above`/`left`
  (§12.3 / §11.3 right-edge fixup); that is the next layer up.
* `mb_skip_coeff` short-circuit (§11.1): when `true`, the entire
  residue is zero by definition, so the orchestrator skips the WHT /
  DCT / summation work and returns the prediction directly.

Why dequantization is the caller's responsibility, not this
orchestrator's (as of round 11): the §14.1 Y2 / chroma dequant
scaling and the §13 zig-zag → raster reordering were both open at
round 11 — keeping them out of this function's signature let §14.2
land then and let a convenience wrapper slot in later. (Both are now
closed: the zig-zag in round 14, the §14.1 scaling in round 15's
`decode_and_dequantize_mb` — the §20.4 rules are RFC-internal, not
external source.) Y1 factors are in `Y1DequantFactors::from_indices`.

Eleven new unit tests: `B_PRED` MB rejection with the proper error;
top-left-corner skip MB with `DC_PRED` everywhere returning the
spec's `DEFAULT_TOPLEFT_DC` (128) in every plane; skip MB with
`V_PRED` and known above-strips matching standalone
`predict_y16x16_v` / `predict_uv8x8_v` output; zero-residue
non-skip MB equalling the skip MB output (the §14.2 path runs
but contributes 0); a Y2 DC-only seeding test exercising the
§14.2 first-paragraph rule end-to-end; a Y2 off-diagonal seeding
test proving the `i*4+j` index rule (`y2[0]=8` + `y2[4]=8` →
WHT → distinct sub-block residues in rows 0..1 vs rows 2..3);
`V_PRED` with no `above` substituting `DEFAULT_ABOVE_PIXEL` (127)
across both luma and chroma; `H_PRED` with no `left` substituting
`DEFAULT_LEFT_PIXEL` (129); §14.5 `clamp255` saturation both
high (every Y pixel → 255) and low (every Y pixel → 0); and a
helper round-trip test guarding the `extract_4x4` / `insert_4x4`
plane-stride math against off-by-one. Total: 150 tests across
eight modules.

**Round 12 (2026-05-24).** Adds the §11.3 / §12.3 `B_PRED`
macroblock reconstruction orchestrator — the per-sub-block intra
walker the round-11 16×16 path deferred (`src/reconstruct.rs`).

* `decode_keyframe_mb_bpred(subblock_modes, uv_mode, mb_skip_coeff,
  neighbors, y_coeffs_dequant, u_coeffs_dequant, v_coeffs_dequant)
  -> Result<ReconstructedMb, ReconstructError>` — drives the sixteen
  4×4 luma sub-blocks in raster order, interleaving predict →
  inverse-DCT → add-residue **per sub-block** so each sub-block's
  reconstructed pixels become the next sub-block's `above` / `left` /
  top-left `P` neighbours (the §12.3 neighbour evolution; mirrors
  §20.14's in-place `b_pred()` loop). Each sub-block selects one of
  the ten `B_DC_PRED` … `B_HU_PRED` kernels in `predict_b4x4`.
* §12.3 right-edge "above-right" fixup: the working luma buffer
  carries a top-border row + left-border column + a four-pixel
  above-right extension; sub-block 3's `(-1,16)..=(-1,19)` pixels are
  copied down into the border slots above sub-blocks 7 / 11 / 15
  (`copy_down`). On the top MB row those four pixels are 127; the
  caller supplies the rightmost-MB `(-1,15)` clamp.
* No Y2 / inverse-WHT seeding — a `B_PRED` MB has no Y2 block (§13 /
  §14.2); each Y sub-block's 0th coefficient comes from its own
  residue. Chroma uses the ordinary 8×8 §12.2 path. The §12.3
  `B_HD_PRED` `svg2p(E+1)` erratum (task #957) is handled in the
  pre-existing `predict_b4x4` kernel (read as `avg2p`).
* `MbNeighbors::y_above_right: Option<[u8; 4]>` — the four
  above-right luma pixels (`None` on the top MB row → 127).
* `ReconstructError::MissingSubblockModes` — for a `B_PRED` call
  whose sixteen-mode record is absent.

Ten new unit tests: missing-modes error; top-left-corner DC
settling to a uniform 128; per-sub-mode (all ten) skip-MB match
against the standalone `predict_b4x4` kernel for sub-block (0,0);
left- and above-neighbour evolution (sub-blocks (0,1) / (1,0)
responding to sub-block (0,0)'s residue); the right-edge above-right
fixup propagating into sub-blocks 3 / 7 / 15; top-row above-right
defaulting to 127; a full mixed-mode MB end-to-end (skip-vs-run
invariance + residue lift); and chroma using the 8×8 mode
independent of luma. Total: 159 tests across nine modules.

**Round 13 (2026-05-24).** Adds the per-frame keyframe raster
walker — the layer above the round-11 / round-12 per-MB
orchestrators (`src/frame.rs`).

* `decode_keyframe(mb_cols, mb_rows, modes, coeffs) ->
  Result<KeyframePlanes, FrameError>` — iterates a key frame's
  macroblocks in raster-scan order. For each MB it assembles the
  `MbNeighbors` strips from the already-reconstructed full-frame
  plane buffers, selects `decode_keyframe_mb_bpred` (when the luma
  mode is `B_PRED`) or `decode_keyframe_mb_non_bpred` (the four
  16×16 modes), and writes the reconstructed 16×16 luma + two 8×8
  chroma blocks into the I420 `KeyframePlanes`.
* Neighbour assembly follows §12: the bottom row of the MB above
  (`y_above`), the rightmost column of the MB to the left
  (`y_left`), the `(-1,-1)` corner (`*_topleft`), and the chroma
  analogues. Off-frame edges are reported as `None` — **not** a
  127 / 129 fill — so the §12.2 `DC_PRED` averaging distinguishes
  genuinely-visible pixels from the out-of-bounds defaults (the
  top-row average-of-8-left, the left-column average-of-8-above,
  and the constant 128 top-left case).
* §12.3 above-right extension: `y_above_right` is the four
  `(-1,16)..=(-1,19)` pixels (the bottom row of the
  already-built MB above-and-to-the-right) for interior MBs; for
  the **rightmost** MB in a non-top row those four are clamped to
  the `(-1,15)` value (§12.3 page 55, mirroring §20.14's per-row
  "extend the last row by four pixels"); on the top MB row the
  field is `None` so the orchestrator fills 127.
* New surface: `decode_keyframe`, `KeyframePlanes` (Y / U / V
  `Vec<u8>` + strides + `mb_cols` / `mb_rows`), `MbCoeffs`
  (pre-dequantized Y2 / 16 Y / 4 U / 4 V), `FrameError`
  (`EmptyFrame`, `MacroblockCountMismatch`, indexed `Macroblock`).
* Caller supplies pre-dequantized `MbCoeffs`: at round 12 the §13.3
  per-MB token walk and the §14.1 Y2/chroma dequant scaling were both
  open, so this round landed the frame-level raster geometry without
  depending on them. (Both are now closed — round 14 token walk, round
  15 dequant; `decode_and_dequantize_mb` produces the `MbCoeffs`.)

Ten new unit tests: a 2×2-MB synthetic key frame round-tripping
through the walker (output matches an independent hand-gathered
per-MB run); the rightmost-MB above-right `(-1,15)` clamp (with a
non-flat above row so the clamp is meaningful) plus a B_PRED MB
consuming it; the non-rightmost MB taking the genuine `(-1,16..20)`
pixels; the top-row `None` above-right; cross-MB neighbour
propagation (a V_PRED MB below copying the residue-lifted
reconstructed row of the MB above); an all-B_PRED 2×2 frame walk;
plus the `EmptyFrame`, `MacroblockCountMismatch`, and indexed
`MissingSubblockModes` error paths. Total: 169 tests across ten
modules.

**Round 14 (2026-05-24).** Adds the §13.3 per-macroblock token walk
— the missing link that feeds `decode_keyframe` straight from the
bitstream, layered over the round-7 `decode_block` primitive
(`src/dct_tokens.rs`).

* `decode_mb_coeffs(dec, has_y2, mb_skip_coeff, coeff_probs, above,
  left) -> Result<MbCoeffs, MbCoeffError>` — walks the 24/25 residual
  blocks of one macroblock in the §13 `residual_data()` order: the
  §14.2 Y2 (WHT) block first when `has_y2`, then the sixteen Y 4×4 DCT
  blocks (plane `YAfterY2` when Y2 is present, else `YNoY2`), then the
  four U and four V chroma blocks (plane `UV`). Each block runs the
  round-7 `decode_block` token loop; the result is reordered into
  raster (natural) order via the §20.16 `zigzag[16]` table.
* Above/left non-zero predictor threading: a nine-entry
  `MbEntropyCtx` (four Y, two U, two V, one Y2) per direction, indexed
  per block by the §20.16 `left_context_index[25]` /
  `above_context_index[25]` slot tables — Y subblocks share a left
  slot per subblock row and an above slot per subblock column. Each
  decoded block writes its non-zero status back into both referenced
  slots (§13.3 "the two predictors referenced by the block are
  replaced") so later blocks below/to-the-right read the correct
  third-dimension probability context. The caller maintains one
  `above` context per MB column and a single rolling `left`.
* §13.1 skip short-circuit with the §20.16 `reset_mb_context` rule:
  a `mb_skip_coeff` MB reads no tokens and clears the eight Y/U/V
  slots; the Y2 slot is cleared **only** when the MB carries a Y2
  block, preserving it across skipped `B_PRED` / `SPLITMV` MBs (the
  §13.3 "most recent macroblock that has a Y2 block" rule).
* New surface: `decode_mb_coeffs`, `MbEntropyCtx`, `MbCoeffError`,
  `ZIGZAG`, `MB_ENTROPY_CTX_LEN`.
* The emitted coefficients are the **raw quantized** token values:
  the §14.1 Y2 / chroma dequant scaling remains a documented spec gap
  (§14.1 page 77 defers it to `dixie.c` §20.4), so `decode_mb_coeffs`
  does not multiply by any dequant factor. The zig-zag → raster
  reordering — previously a §14 gap — is closed here using the §20.16
  annex `zigzag[16]` table.

Seven new unit tests: the zig-zag table is a bijection on 0..16 and
round-trips scan↔raster; the §20.16 left/above context-index tables
match the annex listing (including the Y2 slot 8); a skip MB yields
all-zero coefficients and zeroes its predictor slots; a skipped
`B_PRED` MB preserves the Y2 slot; a synthetic MB with distinctive
per-plane coefficients round-trips to the exact per-block raster
layout (Y2 + YAfterY2 first-coeff-1 luma + chroma DC) with matching
post-MB context; an empty block clears its predictor slot even when
an earlier block set it; and two horizontally-adjacent MBs sharing a
rolling `left` context recover MB1 correctly only with the propagated
context (a fresh-context negative control desyncs the range decoder,
proving the propagation is load-bearing). Total: 176 tests across ten
modules.

**Round 15 (2026-05-24).** Closes the last §14 spec gap — the §14.1
Y2 / chroma dequant scaling — and wires the bitstream→dequant→
reconstruct→pixels chain end to end (new `src/dequant.rs`).

* `MbDequantFactors` — the six §14.1 dequant factors (Y1 DC/AC, Y2
  DC/AC, chroma DC/AC) for one macroblock's segment, computed per the
  §20.4 `dixie.c` `dequant_init` rules (part of RFC 6386): Y1 DC/AC
  use the `dc_qlookup` / `ac_qlookup` tables directly; **Y2 DC = dc_q
  × 2**; **Y2 AC = ac_q × 155 / 100, floored at 8**; **chroma DC =
  dc_q, capped at 132**; chroma AC = ac_q. Every index goes through
  the §20.4 `clamp_q` 0..=127 saturation. The `* 155 / 100` is
  integer arithmetic (truncating).
* `MbDequantFactors::from_quant_indices(&QuantIndices)` — the
  frame-level derivation (base `q = yac_qi`, each §9.6 delta applied
  per plane). `MbDequantFactors::for_segment(&QuantIndices,
  segment_quant, absolute)` layers the §10 per-segment quantizer
  override: absolute mode replaces the base index, delta mode adds to
  it; the five per-plane deltas still apply on top.
* `MbDequantFactors::dequantize(&mut MbCoeffs)` — scales a raw
  (quantized) `MbCoeffs` in place (coefficient 0 × DC factor, 1..=15 ×
  AC factor per block, products in `i32` stored back as `i16` per
  §14.1 page 76).
* `decode_and_dequantize_mb(...)` — the bitstream→dequant wrapper:
  runs `decode_mb_coeffs` then `dequantize`, turning the token
  partition straight into the pre-dequantized `MbCoeffs` that
  `decode_keyframe` consumes. This completes the keyframe decode chain
  bitstream → dequant → reconstruct → pixels.
* New surface: `MbDequantFactors`, `decode_and_dequantize_mb`,
  `UV_DC_MAX`, `Y2_AC_MIN`; `MbCoeffs` now derives `PartialEq` / `Eq`.

Fifteen new unit tests: each scaling rule in isolation (Y1 direct
lookups; Y2 DC ×2; Y2 AC ×155/100 truncation + the <8 floor lifting
6→8; chroma DC 132 cap; chroma AC delta); index clamping at both ends
through the factors; the §10 segment delta vs absolute base derivation
keeping per-plane deltas; an independent re-derivation of the §20.4
`dequant_init` body for the §9.6 worked vector (q=64 with five
deltas); per-plane in-place `dequantize`; the wrapper matching
`decode_mb_coeffs` + `dequantize` on a real `BoolDecoder`; and a full
1×1 keyframe decode through the wired chain proving a larger quantizer
moves reconstructed luma further from the flat-128 prediction. Total:
191 tests across eleven modules.

### Not yet landed

The inter-predicted §16.2 / §16.3 / §16.4 branch of interframes
(`mv_ref` tree, near/nearest/best census + the three-neighbour
weighted score, motion-vector clamping, split-prediction sub-block
walk); motion-vector component decoding (§17.1) against the updated
`MV_CONTEXT`s; the §15.1 loop-filter geometry (raster-order edge
walk over `decode_keyframe`'s plane output + the §15.1 page-86
step-2/4 skip rule + the §9.4 / §10 per-macroblock
`loop_filter_level` derivation) that drives the round-9 §15.2 /
§15.3 / §15.4 per-segment kernels; the encoder. All top-level entry
points (`decode_vp8`, `encode_vp8_keyframe`) still return
`Error::NotImplemented`.

## Clean-room sources

* RFC 6386 — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).
* Black-box invocations of the `ffmpeg` *binary* as an opaque
  validator (no source consulted).

No external library source — libvpx, libaom, libavcodec/vp8\*, etc. —
is permitted as a reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
