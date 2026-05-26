# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status — 2026-05-26 (round 143)

**Encoder Phase 11 — per-MB ZEROMV / NEWMV rate-distortion pick.** The
round-142 `motion_search` primitive is now consumed by
`encode_p_frame_zero_mv`: every macroblock runs an 8-iteration
`small_diamond_search_luma` descent against the clamped §16.3 "best"
predictor, then the §-non-normative
`J = SAD + lambda * (mode_bits + §17 mv_bits)` trade picks between
ZEROMV (search-skipped) and whole-pixel NEWMV. Lambda reuses the
keyframe RD picker's `q^2 / 32` shape. The §17.2 NEWMV differential is
`chosen_mv - clamp_mv(near.mvs[0])`, exactly matching the decoder's
`resolve_inter_mb_mv` add. A differential that wraps outside §17.1's
`[-1023, +1023]` window is treated as `+inf` cost so that candidate is
dropped. Ties between candidates go to ZEROMV (fewer bits, no MV
component bits).

New public surface: `write_mv_component(enc, ctx, value)` /
`write_mv(enc, contexts, mv)` (§17.2 emit, paired with the existing
`read_mv` / `read_mv_component`), `mv_component_bits(ctx, value)` /
`mv_bits(contexts, mv)` (fractional `-log2(P(bit))` cost used by the
RD picker), and `EncodeError::UnsupportedInterMode { mode }` for the
picker contract (non-{ZEROMV,NEWMV} resolved modes surface as an error
rather than panicking so a future picker can roll out incrementally).

Validation: a new `tests/encoder_pframe_newmv.rs` integration test
encodes a 2-frame I+P sequence with a clean +4-luma-pixel diagonal
translation of a 16×16 feature square. The encoder picks NEWMV for
4 of 16 macroblocks (the 4 the feature crosses) and the self-decode
Y-plane PSNR clears **50.3 dB** at `yac_qi = 32` (vs. the ZEROMV-only
path which would absorb the translation through the §14 quantiser and
crater PSNR on this contrast level). A second test pins that the
picker stays on ZEROMV when no motion is possible (flat scene), and
five new `motion_vector.rs` unit tests round-trip
`write_mv_component` / `write_mv` through the production `BoolEncoder`
+ §17 decoder and pin `mv_component_bits` monotonicity.

Half- / quarter-pel refinement (§18.3), `NEARESTMV` / `NEARMV` /
`SPLITMV` candidates, and `GOLDEN` / `ALTREF` source selection remain
follow-up rounds. Tests: 453 → 460 (+5 in `motion_vector.rs`, +2 in
`encoder_pframe_newmv.rs`).

## Status — 2026-05-26 (round 142)

**Encoder Phase 11 begin — whole-pixel motion-search primitive.** New
`crate::motion_search` module wires the smallest infrastructure a
non-zero MV codepath needs: a 16×16 luma SAD evaluator at any §17.1
whole-pixel MV plus a small-diamond integer-pixel descent that finds a
local SAD minimum from a caller-supplied center MV. The MV is in §17
quarter-pixel units (whole-pixel = multiples of 4) and is clamped into
`[MV_MIN, MV_MAX] = [-1023, +1023]` before fetching; the underlying
reference fetch (`fetch_block_whole_pixel`) edge-replicates per §20.14
`build_mc_border`, so a candidate that walks the patch off the picture
is safe. The descent visits the 4-neighbour (N / S / E / W at ±1 whole
pixel each) ring until no neighbour improves the SAD or after
`max_iters` iterations.

New public surface: `block_sad_16x16`, `LumaRef<'_>`, `SearchResult
{ mv: Mv, sad: u32 }`, `mb_luma_sad_at_whole_mv`,
`small_diamond_search_luma`, `MV_MIN` / `MV_MAX` / `WHOLE_PIXEL_STEP`
constants. No bitstream emit yet — `encode_p_frame_zero_mv` still
hardwires every MB to ZEROMV at (0, 0); the NEWMV emit path that
consumes this search result is a follow-up round.

Validated on 14 new unit tests in `motion_search.rs`: pure-SAD
identities (identical / one-pixel / saturated / known-manual), SAD-at-
zero-MV consistency with `block_sad_16x16`, exact-translation
convergence (horizontal 2-whole-pixel, diagonal 2+3 whole-pixel),
descent invariants (never increases SAD), §17.1 range clamp of an
`i16::MAX` center, off-picture edge-replicate safety, snap-to-whole-
pixel coercion of a sub-pixel center, and a `SearchResult` Copy/Eq
contract pin. Tests: 439 → 453 (+14, all in `motion_search.rs`).

## Status — 2026-05-26 (round 141)

**Encoder Phase 10 — multi-frame I + P stream driver.** New
`Vp8InterStreamEncoder` extends Phase 8's keyframe driver to interleave
ZERO_MV P-frames between key frames per a caller-specified
`keyframe_interval` (or a per-call `force_keyframe` override). The
encoder picks K-or-P per frame, maintains the §9 three-slot
reference-frame ladder (`LAST` / `GOLDEN` / `ALTREF`) one-for-one with
`Vp8DecoderState::decode_frame`, and emits one
`EncodedStreamFrame { bytes, kind: FrameKind::{Key,InterZeroMv},
frame_index }` per call. Slot updates honour the §9.7 refresh ladder:
a key frame refreshes all three slots (§9.7 / §9.8); a ZERO_MV P-frame
refreshes LAST only (`refresh_last = 1`, GOLDEN / ALTREF untouched) —
matching the bit pattern the underlying `encode_p_frame_zero_mv`
writes. The first frame of a fresh encoder is always coded as a key
frame (no prior reference to predict from); a forced K-frame
re-anchors the interval so the next automatic K is
`keyframe_interval` frames later, not at the original absolute
multiple. Mid-stream resize is surfaced as
`StreamEncodeError::DimensionsChanged`. `Vp8InterStreamEncoder::new`
returns `None` for `keyframe_interval == 0`.

Round target validated: a synthetic 10-frame I420 sequence at
`keyframe_interval = 4` produces K-P-P-P-K-P-P-P-K-P; replaying the
encoder's bytes through `Vp8DecoderState` clears the 30 dB bar on
every frame at `qi = 32` on a 48×32 source:

| frame | kind | self-decode PSNR (dB) |
|-------|------|------------------------|
| 0 | K | 43.94 |
| 1 | P | 43.94 |
| 2 | P | 39.25 |
| 3 | P | 34.18 |
| 4 | K | 45.21 |
| 5 | P | 44.72 |
| 6 | P | 39.11 |
| 7 | P | 33.96 |
| 8 | K | 45.21 |
| 9 | P | 43.94 |

10-frame mean **41.35 dB**. The expected drift within a GOP (each
P-frame's reconstruction becomes the next P-frame's LAST, so quant
distortion accumulates) shows as the 43.94 → 33.96 dB sag from
frame 4 to frame 7; the next K snaps PSNR back. Three new tests in
`tests/encoder_inter_stream.rs` pin the per-frame PSNR floor, the
§9.1 K-vs-P frame-tag shape, and the forced-K re-anchor behaviour;
six new unit tests in `stream.rs` cover the scheduler, the slot
refresh ladder, and the input validators. Single partition for both
K and P this round (§9.5). Real motion search, multi-partition for
inter, and GOLDEN / ALTREF source selection remain the next encoder
rounds. Tests: 430 → 439.

## Status — 2026-05-26 (round 140)

**Encoder Phase 9 begin — minimum-viable P-frame encoder.** New
`encode_p_frame_zero_mv` emits a structurally-valid VP8 P-frame whose
every macroblock is coded as inter / ZEROMV / LAST. The §9.1 frame tag
carries `frame_type = 1` (no resize, no start code); the §19.2 coded
header writes the inter refresh ladder (`refresh_last = 1`, all other
refresh / copy bits 0), `prob_intra = 255` / `prob_last = 255` so every
MB reads as inter/LAST at minimum cost, the §17.2 `mv_prob_update()`
block as 38 zero F-gates against the per-position update-probability
table, and per-MB emits `mb_skip_coeff → is_inter_mb → ref_frame
selector → inter-mode-tree leaf "0" (ZEROMV)` against the §16.3
census-driven probability table the decoder evolves identically. The §18
prediction reduces to an identity copy of the LAST reference at MV
(0,0); residual = source − prediction is forward-DCT'd, the sixteen Y
DCs collected via §14.3 forward WHT into Y2, all blocks quantised
through `MbDequantFactors::from_base_and_deltas` and token-coded via the
existing intra §13.3 walk. All-zero residual MBs emit as skip MBs
(§11.1). The §15 loop filter runs over the inter reconstruction when
`params.loop_filter_level != 0`. A new
`EncodeError::ReferenceDimensionsMismatch` validates that the supplied
reference's macroblock-aligned dimensions match the source frame's.

Round target validated: a synthetic 64×64 I+P sequence (slow constant
brightness drift between frames) round-trips through
`Vp8DecoderState` at `yac_qi = 32` with whole-frame self-decode
**PSNR 43.78 dB** (Y 43.60 dB / U 44.15 dB / V 44.15 dB), comfortably
clearing the 30 dB round bar. Three new tests in
`tests/encoder_pframe_roundtrip.rs` pin the PSNR floor end-to-end, the
§9.1 inter-frame-tag shape, and the reference-dimensions validator.
Single partition this round (§9.5); multi-partition for inter, NEARESTMV
/ NEARMV / NEWMV / SPLITMV, GOLDEN / ALTREF source selection, and motion
search are the next encoder rounds. Tests: 427 → 430.

## Status — 2026-05-26 (round 139)

**Encoder Phase 8 — multi-frame keyframe stream driver.** New
`Vp8KeyframeStreamEncoder` consumes a sequence of `I420Frame`s and
emits a sequence of independently-decodable VP8 key frames, owning
the cross-frame state a real stream needs: a frame counter, the
dimensions locked at the first `encode_frame` call, and the §9
three-slot reference-frame buffer (`LAST` / `GOLDEN` / `ALTREF`).
Every emitted frame implicitly refreshes all three slots per RFC 6386
§9.7 / §9.8's keyframe rule, mirroring `Vp8DecoderState`'s
`decode_key_frame` slot-installation logic one-for-one so the
inter-encoder round drops in without further plumbing. A new
`encode_keyframe_with_reconstruction` returns both the bytes and the
macroblock-aligned post-§15 reconstructed planes, avoiding a re-decode
on the slot-refresh path. Mid-stream resize is surfaced as
`StreamEncodeError::DimensionsChanged`; failed calls leave the
counter unchanged.

Round target validated: a synthetic 5-frame I420 sequence (per-frame
shifted diagonal luma gradient + walking 128-flat square + per-frame
chroma DC), encoded then decoded through `Vp8DecoderState`, hits
**per-frame self-decode PSNR 45.36–48.53 dB (mean 46.90 dB)** at
`qi = 32` on a 48×32 source, comfortably clearing the 30 dB bar. A
companion test confirms every emitted frame has the §9.1
`key_frame == 0` bit and the `0x9d 0x01 0x2a` start code, and that
each frame independently decodes from a fresh `Vp8DecoderState`. A
third test pins byte-equality between the stream encoder's first
frame and the standalone `encode_keyframe`. Tests: 423 → 427 (+4 in
the new `encoder_keyframe_stream.rs`, +4 in `stream.rs` unit tests).

## Status — 2026-05-26 (round 138)

**Encoder Phase 7 — §15 loop filter wired into the keyframe driver.**
`encode_keyframe` now honours a non-zero
`KeyframeParams::loop_filter_level` (`0..=63`) and a matching
`KeyframeParams::sharpness_level` (`0..=7`). The encoder runs the
§15.1 normal filter over its own reconstruction buffer after the per-MB
raster walk completes (§15 page 84 — *"After the predictor and residue
have been summed for every macroblock, the filter is applied to the
edges between adjacent macroblocks and the edges between adjacent
subblocks"*), reusing the decoder-side `filter_frame` so the encoder's
self-decode produces the same pixels the decoder will reproduce from
the bitstream. The §9.4 `mode_ref_lf_delta_enabled` flag stays at 0 this
round (per-MB delta layer not yet emitted); segmentation also stays
off, so the per-MB level resolves to the frame base in every case.

Pre-walk validators surface `EncodeError::LoopFilterLevelOutOfRange`
for `loop_filter_level > 63` and
`EncodeError::SharpnessLevelOutOfRange` for `sharpness_level > 7`. On a
48×32 synthetic source at `qi = 32`, the whole-frame self-decode PSNR
sweeps as:

| `loop_filter_level` | self-decode PSNR (dB) |
|---------------------|-----------------------|
| 0                   | 43.29 |
| 1                   | 43.30 |
| 8                   | 44.67 |
| 24                  | 44.22 |

Every level decodes cleanly through `decode_vp8`; non-zero levels
actually alter the reconstruction (a "filter actually ran" invariant
test asserts the level-0 PSNR is not reproduced at any non-zero level).
New tests in `encoder_keyframe_roundtrip.rs`:
`keyframe_loop_filter_levels_roundtrip` pins the {0, 1, 8, 24} sweep;
`keyframe_loop_filter_level_out_of_range_rejected` and
`keyframe_sharpness_level_out_of_range_rejected` pin the validators;
`keyframe_sharpness_level_roundtrip` sweeps `{0, 1, 4, 7}` at filter
level 16. Tests: 7 → 11 in `encoder_keyframe_roundtrip.rs`.

## Status — 2026-05-26 (round 137)

**Encoder Phase 6 — multi-partition DCT output landed.** `encode_keyframe`
generalises to the §9.5 four-value `log2_nbr_of_dct_partitions` table:
the new `KeyframeParams::nbr_of_dct_partitions` field accepts 1 / 2 / 4
/ 8, the §9.5 header bit is written through the existing
`write_token_partition_count`, and per-MB token data is dispatched to
the right `BoolEncoder` by the §20.4 round-robin (row `r` → partition
`r % N`). Each partition is finalised independently with the §7.3 4-byte
flush trailer (§4 page 9 — "All partitions are decoded using separate
instances of the boolean entropy decoder"); a `(N-1) * 3`-byte §9.5
size table of 24-bit little-endian lengths precedes the partition
bodies when `N > 1`. The §13.3 above-context stays column-wise and
frame-lived (shared across partitions, matching the decoder's
`decode_residuals`); the left context resets per row and so does not
cross partitions.

The output is a layout reorganisation, not a coding change: at
`128×128 qi=32` the self-decode PSNR is bit-exact 45.9549 dB at every
partition count, and the byte count grows monotonically with `N` from
242 → 246 → 256 → 274 B (one §7.3 flush + one §9.5 size-table entry
per additional partition). A new
`keyframe_multi_partition_psnr_matches_single_partition_baseline` test
pins identical reconstruction across 1 / 2 / 4 / 8 against the
1-partition baseline; `keyframe_multi_partition_short_frame_roundtrip`
covers a 32×32 frame whose two macroblock rows leave six of the eight
partitions empty (still valid per §20.4); and
`keyframe_invalid_partition_count_rejected` pins the new pre-walk
validator against every non-{1,2,4,8} input. Tests: 3 → 7 in
`encoder_keyframe_roundtrip.rs`.

## Status — 2026-05-25 (round 136)

**Encoder Phase 5 — rate-distortion intra mode selection landed.** The
SAD-only picker is replaced by a Lagrangian rate-distortion search:
every candidate mode is run through the full §14 chain (predict → FDCT
→ Y2/WHT → quantise → dequantise → inverse-transform → reconstruct) and
scored by `J = SSD + lambda * R`, where the distortion `D` is the
**exact self-decode** SSD against the source and the rate `R` is the
§13.3 token bits (priced by summing `-log2(p)` over each §7.3 boolean,
mirroring the token writer) plus the §11.2 / §11.4 mode-signal bits.
`lambda = q² / 32` is derived from the luma AC quant step (RD is
non-normative — RFC 6386 specifies only decoding). Applied to the
whole-block luma pick, the chroma `uv_mode` pick, and per-4×4 `B_PRED`
sub-block modes; the luma whole-block-vs-`B_PRED` decision compares
total RD cost (charging the §11.2 B-flag).

Measured against the r135 SAD picker on a 64×64 natural test frame, RD
wins on **both** axes at every quantiser:

| qi | SAD bytes → RD | SAD PSNR → RD |
|----|----------------|---------------|
| 16 | 1467 → 1366    | 39.29 → 39.62 dB |
| 32 | 1003 → 919     | 34.56 → 35.00 dB |
| 48 | 731 → 615      | 31.74 → 31.94 dB |
| 64 | 383 → 288      | 29.92 → 29.95 dB |

Smaller files **and** higher PSNR throughout (PSNR/byte up ~8–33%). A
`rd_beats_sad_baseline_size_and_quality` test pins the no-regression
guarantee; `rd_keyframe_holds_psnr_floor_on_natural_frame` holds a
30 dB floor at qi 16/32/48. Inter prediction and a full per-token
trellis remain the next encoder rounds.

Lib test count: 390 → 392.

## Status — 2026-05-25 (round 135)

**Encoder Phase 4 — per-frame key-frame raster driver landed.** The
crate now exposes `encode_keyframe(&I420Frame, &KeyframeParams)`: a
top-level driver that walks a source I420 picture macroblock-by-
macroblock in raster order and assembles a complete VP8 key frame —
§9 frame header + §19.2 first (control) partition + a single §19.2 DCT
partition. For each macroblock it gathers the reconstructed-neighbour
strips from the already-encoded part of the frame (reusing the
decoder's own `gather_neighbors`), runs the §12.2 / §11.3 intra mode
pick (`encode_mb_block_set_with_neighbors`), then dequantises and
reconstructs through the **decoder's** §14.2 / §12.3 orchestrators and
writes the result back — so the next macroblock predicts from the exact
pixels the decoder will. Both the §13.3 token contexts (one `above`
per column, `left` reset per row) and the §11.3 cross-macroblock
`B_PRED` sub-block-mode contexts thread MB-to-MB; an all-zero-residual
MB is coded as an §11.1 skip. Partial right / bottom macroblocks of a
non-multiple-of-16 frame are edge-replicated. The §11 mode layer is
written via a new `BoolEncoder::write_treed` (the §8.1 `treed_read`
walk in reverse).

Black-box validation (`tests/encoder_keyframe_roundtrip.rs`): encode a
synthetic gradient + flat-region I420 frame, decode it via the crate's
own `decode_vp8`, measure whole-frame PSNR:

* 48×32, `yac_qi = 32` (mid quant): **41.50 dB** (target ≥ 30 dB).
* 32×32, `yac_qi = 32`: **43.65 dB**.
* 48×32, `yac_qi = 8`: **49.10 dB**.
* 40×24 (non-multiple-of-16), `yac_qi = 32`: **39.81 dB**.

A `mode_layer_roundtrips_through_decoder_parser` unit test pins the §11
writer against the decoder's `parse_key_frame_macroblock_modes`. Inter
prediction, RD bit-cost mode search, multi-partition DCT, a non-zero
loop filter, and a multi-frame driver remain the next encoder rounds.

Lib test count: 389 → 390.

## Status — 2026-05-25 (round 134)

**Encoder Phase 3b — B_PRED 4×4 sub-block intra mode pick landed.** The
per-MB luma decision now also evaluates the §11.3 / §12.3 `B_PRED`
path: the 16×16 luma plane can be encoded as sixteen independent 4×4
sub-blocks, each choosing the SAD-minimising one of the ten §12.3
sub-modes (`B_DC` / `B_TM` / `B_VE` / `B_HE` / `B_LD` / `B_RD` /
`B_VR` / `B_VL` / `B_HD` / `B_HU`) against the source, with **in-place
neighbour evolution** — every sub-block predicts from the already-
reconstructed (predictor + dequantised residue) pixels of the
sub-blocks above and to its left, including the §12.3 right-edge
above-right fixup. The encoder reuses the decoder's `predict_b4x4`
kernel and `inverse_dct_4x4` / `add_residue_4x4`, so the reconstruction
it evolves against is exactly what the decoder produces.

A top-level luma decision picks `B_PRED` over the best whole-block mode
iff its total prediction SAD is strictly lower (a flat / single-edge
region with matching neighbours stays whole-block). When `B_PRED`
wins the macroblock has no Y2 block: the sixteen Y sub-blocks keep
their own DC and are token-coded through the `YNoY2` plane
(`has_y2 = false`). The chosen sixteen sub-modes ride on
`EncodedMb::b_subblock_modes` (`Some` iff `y_mode == B`), feeding the
decoder's `decode_keyframe_mb_bpred` walk. Still a SAD picker — no RD
bit-cost term, no inter prediction.

Validated by 3 new unit tests:

* `diagonal_subblock_mb_picks_bpred_and_decodes_above_30db` — a
  macroblock of per-4×4-sub-block diagonal tiles (no single whole-block
  mode follows them) flips to `B_PRED` and reconstructs at **≈ 54 dB**
  PSNR at `yac_qi = 4`, with a genuine per-sub-block mode mix.
* `bpred_neighbour_evolution_roundtrips_at_low_q` — the same MB at
  `yac_qi = 0` clears a 40 dB floor (encoder evolution == decoder's).
* `flat_mb_keeps_whole_block_luma_mode` — a flat MB in matching flat
  neighbours stays whole-block (`y_mode != B`).

Inter prediction, a true RD search, and the per-frame raster driver
remain the next encoder rounds.

Lib test count: 386 → 389.

## Status — 2026-05-25 (round 133)

**Encoder Phase 3 — whole-block intra mode pick landed.** The per-MB
driver now selects among the four §12.2 whole-block intra modes
(`DC_PRED` / `V_PRED` / `H_PRED` / `TM_PRED`) instead of forcing
`DC_PRED`. For each macroblock it evaluates every candidate's
prediction (via the shared `intra_predict` kernels the decoder already
uses), scores it by SAD against the source, and residual-codes against
the lowest-SAD mode — independently for the 16×16 luma plane and the
shared 8×8 chroma `uv_mode`. This is a SAD picker, not a
rate-distortion search (no bit-cost term yet). The picked
`y_mode` / `uv_mode` ride on `EncodedMb`.

A new `encode_mb_block_set_with_neighbors` entry scores the picker
against caller-supplied reconstructed-neighbour strips
(`reconstruct::MbNeighbors`); `encode_mb_block_set` is now a thin
wrapper that passes all-off-frame neighbours (the §12 127 / 129 / 128
defaults), preserving its isolated-MB behaviour.

Validated by 4 new unit tests:

* `mode_pick_chooses_v_pred_for_column_constant_mb` — a
  column-constant (horizontally-varying) MB whose `above` neighbour
  matches the pattern; picks `V_PRED` for luma + chroma and decodes
  **bit-exact** at `yac_qi = 8`.
* `mode_pick_chooses_h_pred_for_row_constant_mb` — a row-constant
  (vertically-varying) MB with matching `left`; picks `H_PRED`,
  bit-exact.
* `mode_pick_chooses_tm_pred_for_planar_ramp_mb` — a planar ramp
  `clamp(L_i + A_j − P)`; picks `TM_PRED`, bit-exact.
* `isolated_mb_textured_roundtrips_above_30db` — a textured isolated
  MB through `encode_mb_block_set`; reconstructs at ≈ 44–45 dB PSNR.

`B_PRED`, inter prediction, a true RD search, and the per-frame raster
driver remain the next encoder rounds.

Lib test count: 382 → 386.

## Status — 2026-05-25 (round 132)

**Encoder Phase 2 — per-MB block-set wiring landed.** The crate now
exposes `encode_mb_block_set` in `src/encoder.rs`: a per-macroblock
encoder driver that takes a single 16×16 Y + 8×8 Cb + 8×8 Cr
macroblock (`MbPixels`) at a chosen quantiser index and produces an
`EncodedMb` carrying the raw-quantised `MbCoeffs` plus the §13.3
token-coded byte stream the existing `decode_mb_coeffs` consumes.

The driver implements the encode-side inverse of the §14.2 / §12.2
reconstruction orchestrator:

1. **Prediction**: §12.2 `DC_PRED` with no above / left neighbours →
   flat 128 across every plane; residual is `pixel - 128`.
2. **Forward transforms**: §14.4 forward DCT per 4×4 sub-block (16 Y,
   4 U, 4 V); the 16 Y DCs are collected into a Y2 block and §14.3
   forward-WHT'd. Each Y sub-block's DC is then zeroed (now carried by
   Y2).
3. **Quantisation**: §14.1 / §20.4 round-half-away-from-zero division
   by the six `MbDequantFactors` (`y1_dc/y1_ac/y2_dc/y2_ac/uv_dc/uv_ac`)
   — the natural inverse of `MbDequantFactors::dequantize`.
4. **Token coding**: §13.3 walk in residual order Y2 → 16 Y
   (`YAfterY2`) → 4 U (`UV`) → 4 V (`UV`), threaded through fresh
   above / left `MbEntropyCtx` with the §20.16 `left_context_index` /
   `above_context_index` slot mapping so the per-block first-position
   probability index matches what `decode_mb_coeffs` reads on the
   other side.

Validated by 3 new unit tests:

* `mb_block_set_roundtrip_flat_color_recovers_within_one_lsb` —
  sweeps flat pixel values 100/110/128/140/160/200 at `yac_qi = 0`;
  every reconstructed luma pixel is within **≤ 1 LSB** of the input.
* `mb_block_set_constant_128_emits_all_eob_blocks` — a zero-residual
  MB (constant 128) emits zero non-zero blocks and round-trips to
  all-zero coefficients.
* `mb_block_set_roundtrip_flat_color_at_q16_holds_within_2_lsb` —
  same flat-MB roundtrip at `yac_qi = 16`; recovered within ≤ 2 LSB.

What lands this round is the per-MB block-set walker. RD-driven mode
selection / quantiser-step picks, non-DC prediction modes (V / H /
TM / B_PRED for luma, V / H / TM for chroma), inter prediction, and
the per-frame raster driver that threads `MbEntropyCtx` columns
through an N-MB frame are the next encoder rounds.

Lib test count: 379 → 382; standalone (no-default-features) lib
tests: 374 → 377.

## Status — 2026-05-25 (round 131)

**Encoder Phase 2 begin — §14 forward 4×4 DCT + WHT primitives landed
and wired into `TokenEncoder`.** A new `src/forward_transform.rs`
module ships `forward_dct_4x4`, `forward_wht_4x4`, and
`raster_to_scan` — the encoder-side partners of the §14.3 / §14.4
inverse transforms and the §20.16 zig-zag reorder. The two forward
transforms are mechanically derived as the transpose of the §14.3 /
§14.4 inverse listings (the §14.4 preamble itself notes the transform
is *"a classical 2-D inverse discrete cosine transform"*); both reuse
the same `COSPI8_SQRT2_MINUS1 = 20091` / `SINPI8_SQRT2 = 35468`
fixed-point constants the §14.4 inverse uses, so the forward / inverse
rounding shapes track each other. The module-level docstring records
the matrix algebra (`M * M^T = 4 * I`, `T_inv * T_inv^T = 4 * I`,
hence `FDCT(p) = round((T_fwd * p * T_fwd^T) / 2)` and
`FWHT(p) = round((M * p * M) / 2)`) so the derivation is part of the
crate's clean-room provenance.

The new integration test `tests/encoder_transform_roundtrip.rs`
proves the chain end-to-end on a synthetic flat-color 4×4 block:
FDCT → quantize (§14.1 Y1 factors) → raster-to-scan →
`TokenEncoder::encode_block` → finish → `BoolDecoder::init` →
`decode_block` → scan-to-raster → `dequant_block` →
`inverse_dct_4x4`. The recovered residual hits **48.13 dB PSNR** at
`yac_qi = 32` on a flat-12 residual (well above the round-131 ≥ 35 dB
target); at `yac_qi = 0` (where the DC lookup is 4 and the dequant
exactly inverts the quant) the chain is bit-exact lossless across the
full `0..=64` flat-value sweep.

What lands this round is the §14 *primitives* + the per-block roundtrip
proof — the per-MB block-set wiring (Y2 DC seeding across the sixteen Y
sub-blocks, the 24/25-block walk, the RD-driven mode + quant-step
selection) is the next encoder round. Lib test count: 371 → 379;
standalone (no-default-features): 366 → 374; integration tests grow
by 7 (the new `encoder_transform_roundtrip.rs`).

## Status — 2026-05-25 (round 128)

**Encoder Phase 2 — §13 DCT-token block encoder landed.** The crate
now exposes `encode_coeff_block` + `TokenEncoder` in `src/encoder.rs`:
a Phase-2 §13.2 / §13.3 inverse of `dct_tokens::decode_block` that
walks the §13.2 `coeff_tree` against the resolved
`coeff_probs[4][8][3][11]` table and emits a single 16-coefficient
sub-block through the existing §7.3 `BoolEncoder`. Per coefficient
the encoder classifies the magnitude into the twelve-symbol DCT
alphabet (Dct0..Dct4, Cat1..Cat6, Eob), records the bit path the
decoder will read for that leaf (entering the tree at index 2 to
bypass the EOB branch when the previous coefficient was a literal
`DCT_0`), writes any Cat extra bits against `PCAT1..PCAT6`, and
emits the universal sign bit at probability 128; `ctx3` rolls to
the coefficient's magnitude class on the way to the next position.
EOB is emitted explicitly after the last non-zero coefficient unless
all 16 (or 15 for `YAfterY2`) positions are non-zero, in which case
§13.2's "implicit EOB" rule applies. This is the bitstream-side
prerequisite for the §14 quant + RD-driven encode path; this round
ships the entropy coder only — no quant policy, no RD search, no
per-MB block-set wiring yet. Validated by 8 new unit tests that
encode → decode → byte-compare at the coefficient layer across the
full §13.2 alphabet, every `BlockType`, every position, every
neighbour-predictor combination, and a 64-trial pseudo-random
sweep. Lib test count: 363 → 371; standalone (no-default-features):
358 → 366.

## Status — 2026-05-25 (round 125)

**Encoder Phase 1 (frame-header writers + silent-keyframe path)
landed.** A new `src/encoder.rs` ships the RFC 6386 §7.3 `BoolEncoder`
plus the §9.1 / §9.3 / §9.4 / §9.5 / §9.6 / §9.7 / §9.9 / §9.11
frame-header writer subroutines (`write_frame_tag`,
`write_segment_update_flags`, `write_loop_filter`,
`write_token_partition_count`, `write_quant_indices`,
`write_no_token_prob_updates`, `write_mb_no_skip_coeff`,
`patch_first_partition_size`) and a top-level
`encode_silent_keyframe(params)` that composes them into a
structurally-valid all-zero-quantization key frame. Every macroblock
is coded as `mb_skip_coeff = 1` (`prob_skip_false = 1`) with `DC_PRED`
luma + chroma, so the DCT partition carries no token data and the
encoder neither selects modes nor quantizes — that lands in a later
round once the §13 / §14 encode side ships. The emitted bytes
round-trip through the crate's own decoder at multiple dimensions
(16×16, 32×32, 32×24, 48×16, 64×64) and at every legal §9.5 partition
count (1 / 2 / 4 / 8); `ffmpeg -c:v vp8` accepts the same bytes
wrapped in IVF (`tests/encoder_external_decode.rs`). A direct-API
`oxideav_vp8::encoder::make_encoder` factory parallels the registry
path per the workspace dual-API convention; `encode_vp8_keyframe`
(legacy entry) now delegates to the same path instead of returning
`NotImplemented`.

## Status — 2026-05-25 (round 120)

**Clean-room rebuild in progress.** The prior implementation was
retired under the workspace clean-room policy after a provenance audit
on 2026-05-20. Rebuild work tracks RFC 6386 exclusively, with
black-box `ffmpeg` / `libvpx` invocations as the only validator.

**Multi-frame interframe decode is now complete and bit-exact.** As of
round 120 the new `Vp8DecoderState` driver
(`src/state.rs`, `decode_frame(&mut self, bytes)`) owns the
RFC 6386 §9 three-slot reference-frame buffer (`LAST` / `GOLDEN` /
`ALTREF`), threads the §9.10 entropy / intra-mode / MV carry-state
across frames (including the `refresh_entropy_probs` saved-entropy
rollback per the §20 reference pattern), and rotates the slots per the
§9 `copy_buffer_to_alternate` / `copy_buffer_to_golden` /
`refresh_golden_frame` / `refresh_alternate_frame` / `refresh_last`
ladder. The per-frame walker dispatches each macroblock to either the
keyframe intra path or the §16 inter path: an inter MB reads the §16.2
reference-frame bits, runs the §16.3 census + §16.2 mode tree against
the carried `MbInfo` neighbour records, resolves the per-mode vector
(ZERO / NEAREST / NEAR / NEW / SPLITMV), and reconstructs pixels via
the §18 motion-comp kernels fetching from the correct reference slot;
an intra MB on an interframe uses the §16.1 intra-on-interframe tree.
The §15 loop-filter post-pass now uses the full §9.4 reference + mode
delta ladder (the new `FrameFilterConfig::interframe` /
`filter_inter_frame`) so each MB's level reflects its ref frame
(`ref_delta[LAST/GOLDEN/ALTREF]`) and §16.2 mode bucket
(`mode_delta[ZERO/OTHER/SPLIT]`).

End-to-end **byte-for-byte identical** to libvpx/ffmpeg reference
output on three multi-frame fixtures:

* `i-frame-then-p-frame-64x64` — 1 I + 1 P (the round's target),
* `golden-update-cycle` — 5 frames with mid-GOP golden refresh,
* `altref-arnr-on` — 10 frames with `auto-alt-ref` + ARNR.

Plus the ten single-keyframe fixtures from earlier rounds, all still
passing through the new stateful driver. Test count: 361 lib + 3
ffmpeg-external + 5 public-error-surface (default features) / 356 lib
(standalone) — was 346 / 341 in round 120. The Phase 1 encoder
(silent-keyframe path) ships in round 125; the §13 / §14 RD-driven
encode side is still future work.

**Round 122 surface tweak.** A public `Vp8Error` umbrella enum is now
re-exported from the crate root (`pub enum Vp8Error { Decode(DecodeError),
Encode(Error) }`) with `Display` / `std::error::Error` / `From<DecodeError>`
/ `From<Error>` impls. This unblocks downstream consumers (notably
`oxideav-webp`'s lossy VP8 path) that need a single stable error symbol
to build a `From<oxideav_vp8::Vp8Error>` adapter against. The existing
per-module errors (`DecodeError`, `Error`, etc.) are unchanged. A
`tests/public_error_surface.rs` integration test locks the visibility.

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

**Round 16 (2026-05-24).** Adds the §15.1 loop-filter *frame geometry*
— `filter_frame` in `src/loop_filter.rs`, the per-frame post-pass that
drives the round-9 §15.2 / §15.3 / §15.4 per-segment kernels across a
reconstructed `KeyframePlanes`.

  * `filter_frame(planes, modes, coeffs, config)` walks macroblocks in
    raster order and runs the four §15.1 page-86 steps in order: (1)
    left inter-MB vertical edge (skipped on the leftmost column), (2)
    three internal vertical subblock edges at 1/4, 1/2, 3/4 of the luma
    width plus one centre edge per chroma block, (3) top inter-MB
    horizontal edge (skipped on the topmost row), (4) three internal
    horizontal subblock edges. Normal filter does luma + both chroma;
    the simple filter is luma-only (§15.2). The ordering is load-bearing
    — many pixels straddle two edges and are filtered twice.
  * Steps 2 and 4 are skipped when the MB is neither `B_PRED` nor
    `SPLITMV` *and* has no coded coefficient. Per the §20.6 annex note,
    the gate is the decoded-coefficient count, so the pass inspects the
    dequantized `MbCoeffs` rather than the bitstream skip flag. The whole
    MB is skipped when its resolved level is 0 (§15 page 84).
  * `calculate_mb_filter_level` implements the §20.6 `dixie.c`
    `calculate_filter_parameters` body: base `loop_filter_level`, the §10
    per-segment override (delta adds / absolute replaces, clamped
    `0..=63`), then the §9.4 reference + `B_PRED` mode deltas (clamped
    again). The §15.4 `LoopFilterParams` `mbedge_limit` / `sub_bedge_limit`
    already equal the §20.6 `2*E + I` disabling metric, so they pass
    straight into the kernels.
  * `FrameFilterConfig` carries the resolved frame state;
    `FrameFilterConfig::keyframe` builds it from a `Vp8CodedHeader`
    (resolving per-segment LF levels and the §9.4 current-frame / `B_PRED`
    deltas — a key frame has no prior persisted delta state).
  * New surface: `filter_frame`, `calculate_mb_filter_level`,
    `FrameFilterConfig`, `MAX_REF_LF_DELTAS`, `MAX_MODE_LF_DELTAS`,
    `MAX_MB_SEGMENTS`.

Fifteen new unit tests: level derivation (base, segment delta/absolute,
the dual `0..=63` clamp, ref + `B_PRED` mode deltas, delta-disable); a
hand-derived normal MB-edge rewrite of the six straddling pixels
(`p2..q2`) at a 100/110 boundary; level-0 whole-MB no-op; leftmost-column
left-edge skip (with the horizontal edge still touching column 0);
simple-filter luma-only; the coeff-gated vs `B_PRED`-forced subblock
steps; and the header→`FrameFilterConfig` resolution. Total: 206 tests
across eleven modules.

**Round 17 (2026-05-24).** Wires the top-level per-frame decode driver —
`decode_vp8` and the `oxideav_core::Decoder` integration (new
`src/decoder.rs`) — tying every prior round into one end-to-end key-frame
decoder.

  * `decode_vp8(bytes) -> Result<Vp8DecodedFrame, DecodeError>` — one
    packet = one frame. Parses the §9.1 uncompressed header
    (`Vp8FrameHeader`), rejects non-key frames as
    `DecodeError::Unsupported`, parses the §19.2 boolean-coded header
    (`Vp8CodedHeader`) and the §11 / §19.3 macroblock prediction layer
    (`parse_key_frame_macroblock_modes`), carves the §9.5 DCT partitions
    (the 3-byte little-endian size table + the §20.4 round-robin row
    striping `row → partition r % n`, one `BoolDecoder` per consumed
    partition with its cursor persisting across the rows that share it),
    decodes + §14.1-dequantizes each macroblock's §13 residuals
    (`decode_and_dequantize_mb`, with above/left non-zero predictor
    threading and the per-segment dequant factors), reconstructs the frame
    (`decode_keyframe`), runs the §15.1 loop-filter post-pass
    (`filter_frame`), and crops to the §9.1 visible width / height,
    emitting an I420 `Vp8DecodedFrame`.
  * Behind the default-on `registry` feature: `Vp8Decoder` (an
    `oxideav_core::Decoder` impl — `send_packet` queues, `receive_frame`
    runs `decode_vp8` and yields a `Frame::Video`), `make_decoder`, and
    `register` / `register_codecs` registering codec id `"vp8"` with the
    `VP80` / `vp08` / `V_VP8` container tags. `register` is wired through
    the `oxideav_core::register!` dispatch hook in `lib.rs`.
  * **Bit-exact** against the libvpx/ffmpeg reference (RFC 6386 §2 makes
    exact reconstructed pixels part of the spec) on ten conformance
    fixtures vendored under `tests/fixtures/`: 16×16, 32×32, 64×64,
    128×128; one- and four-DCT-partition; loop-filter off / level 1 /
    level 33 / level 38; and the §15.2 simple-filter mode.
  * Two intra-prediction corrections surfaced while reaching bit-exactness
    on multi-macroblock frames: (a) the §20.14 `fixup_left` corner rule —
    a left-frame-edge macroblock's `(-1,-1)` corner is 129 (the kernels
    were defaulting it to 127), read by `TM_PRED` / `B_TM_PRED`; and (b)
    the non-`B_PRED` `TM_PRED` off-top corner default, now 127 (was 0).

Seventeen new tests: non-keyframe → `Unsupported`; the truncation /
zero-dimension error paths; the §9.5 partition-table carve (single,
two-partition, truncated-table, truncated-body); ten bit-exact fixture
decodes; and the `Vp8Decoder` trait integration (NeedMore before a packet,
inter-frame `Unsupported`, the tiny-keyframe decode through the trait API
with pts round-trip, and registry enumeration). Total: 223 tests across
twelve modules.

**Round 18 (2026-05-24).** Lands the §17 motion-vector component decoder
— the first element of the inter-frame (§16) prediction path (new
`src/motion_vector.rs`). Motion vectors appear in both `NEWMV` (whole-MB)
and `NEW4x4` (SPLITMV sub-block) modes with an identical wire format, so
this is the shared primitive both call sites will consume:

  * `read_mv_component(dec, ctx)` — §17.1 `read_mvcomponent`. Reads the
    `mvpis_short` range selector; for the short form (`0 <= A <= 7`) walks
    the §17.1 `small_mvtree` (transcribed as `SMALL_MVTREE`) reading
    probabilities at `ctx[MVPshort + (node >> 1)]`; for the long form
    (`8 <= A <= 1023`) reads bits 0–2 then 9–4 from `ctx[MVPbits + i]` and
    applies the implicit-bit-3 rule (bit 3 is *not* coded when `A <= 15`,
    since a long-coded value is `>= 8`); reads the sign at `ctx[MVPsign]`
    only for a non-zero magnitude. Returns the signed quarter-pixel
    component, `-1023..=1023`.
  * `read_mv(dec, contexts)` — §17.2 `read_mv`: row component against
    `contexts[0]` then column against `contexts[1]`, returning a raw
    differential `Mv { row, col }`.
  * `resolve_mv_contexts(base, updates)` — applies the round-4 §17.2
    `mv_prob_update()` overlays (`Some(prob)` replaces, `None` keeps the
    base) onto a base `MvContexts`, turning the parsed updates into the
    live decoding tables. `default_mv_contexts()` seeds from
    `DEFAULT_MV_CONTEXT` for the §17.2 "set to defaults every key frame"
    rule; passing the previous frame's resolved contexts as `base` gives
    the §17.2 cross-interframe persistence.
  * New surface: `Mv`, `MvContext`, `MvContexts`, `SMALL_MVTREE`,
    `read_mv_component`, `read_mv`, `resolve_mv_contexts`,
    `default_mv_contexts`.

The round stays at the component layer: the §16.2 `mv_ref` tree, the
§16.3 `vp8_find_near_mvs` near/nearest/best census, the §16.4 SPLITMV
sub-block walk, the §18.1 stored-luma doubling / range clamp / chroma
averaging, and the §18 sub-pixel interpolation are explicitly **not** in
scope — `read_mv` returns the raw differential vector the
inter-prediction layer adds to a reference base.

Fifteen new unit tests round-trip a test-side VP8 bool encoder (mirroring
the proven `dct_tokens` / `bool_decoder` test encoder) through the §17
routines: `SMALL_MVTREE` shape + the `MVPindices` offset arithmetic; all
short values `-7..=7`; the zero-has-no-sign-bit short-circuit; long values
at the boundaries (8, 15, 16, 31, 511, 512, 1023 and negatives); the
implicit-bit-3 boundary (8..=15 with no coded bit 3, 16/17/24/25 with it);
the ±1023 extremes; `read_mv` reading row-then-column contexts plus a
load-bearing swapped-context negative control; `resolve_mv_contexts`
identity / overlay / cross-frame persistence / round-trip-under-resolved-
context; and the `Mv` default. Total: 238 tests across thirteen modules.

**Round 19 (2026-05-24).** Lands the first inter-prediction slice that
*consumes* the §17 motion vectors: §16.2 reference-frame selection and
§18 whole-pixel motion compensation (new `src/motion_comp.rs`). VP8
motion vectors carry a sub-pixel fraction that §18.3 sixtap / bilinear
interpolation resolves; that interpolation is large and deferred, so this
round lands the *whole-pixel* path — the §18.3 page-115 "the prediction
subblock is simply copied" case (the §20.14 `filter_block` special case
`mx | my == 0`).

  * `select_ref_frame(dec, prob_last, prob_gf)` — §16.2 reference
    selector. `B(prob_last) == 0` → `Last`; else `B(prob_gf)` picks
    `Golden` (0) / `AltRef` (1). Returns a `RefFrame`.
  * `stored_luma_mv(mv)` — §18.1 stored-luma doubling (quarter-pel →
    eighth-pel, ±2046 range). `chroma_mv(luma_mv)` — §18.1 `avg()` chroma
    averaging of the single repeated whole-MB vector, cross-checked
    against the §20.14 closed form `(c + 1 + (c >> 31) * 2) / 2`.
    `apply_full_pixel(mv)` — §18.1 version-3 full-pel-chroma truncation
    (`& ~7`). `whole_pixel_fraction_is_zero(mv)` — the §18.3 whole-pixel
    test (`(row & 7) | (col & 7) == 0`).
  * `fetch_block_whole_pixel(...)` — the §20.14 `build_mc_border`
    edge-replicated 4×4 reference fetch, specialised to whole-pixel
    offsets (out-of-plane reads clamp to the nearest edge pixel; integer
    offset is `mv >> 3` with sign propagation, §18.2).
  * `predict_inter_mb_whole_pixel(reference, mb_col, mb_row, luma_mv,
    full_pixel)` — §18.2 whole-MB prediction buffer for a non-SPLITMV MB
    (one vector for all sixteen Y sub-blocks, the averaged chroma vector
    for the eight chroma sub-blocks), reading a borrowed `ReferencePlanes`
    (the reference frame's I420 planes). Refuses a sub-pixel vector with
    `MotionCompError::SubPixelNotSupported`.
  * `reconstruct_inter_mb_whole_pixel(...)` — prediction + §14 dequantized
    residual (Y2 WHT seeding + per-sub-block inverse DCT + §14.5
    `clamp255` summation), honouring the §11.1 `mb_skip_coeff`
    short-circuit. The inter analogue of
    `decode_keyframe_mb_non_bpred`.
  * New surface: `RefFrame`, `ReferencePlanes`, `MotionCompError`,
    `select_ref_frame`, `stored_luma_mv`, `chroma_mv`, `apply_full_pixel`,
    `whole_pixel_fraction_is_zero`, `fetch_block_whole_pixel`,
    `predict_inter_mb_whole_pixel`, `reconstruct_inter_mb_whole_pixel`.

The round stays at whole-pixel motion compensation for the four whole-MB
inter modes; the §18.3 sub-pixel interpolation, the §16.3
`vp8_find_near_mvs` near/nearest/best census (this slice takes the
*resolved* per-MB vector as an input), and the §16.4 SPLITMV per-sub-block
walk are explicitly **not** in scope.

Twenty-three new unit tests: the §16.2 selector read order (`Last` /
`Golden` / `AltRef` paths + distinct-prob wiring) round-tripped through a
test-side VP8 bool encoder; the §18.1 adjustments (doubling, `avg()`
formula, the §20.14 closed-form cross-check across a spread of
eighth-pel inputs, full-pel truncation, the whole-pixel test); the
§20.14 `build_mc_border` edge replication (zero offset, integer offset,
left / top / bottom-right corner clamps); whole-MB prediction (zero-MV
copy of the matching reference MB, whole-pixel-offset shift, sub-pixel
luma + chroma rejection, full-pixel-version acceptance); and inter-MB
reconstruction (skip == prediction, a DC-residue path verified against
the public transform primitives, sub-pixel rejection). Total: 261 tests
across fourteen modules.

**Round 117 (2026-05-24).** §18.3 sub-pixel motion compensation, wired
into `predict_inter_mb` / `reconstruct_inter_mb` so non-zero-fraction
vectors reconstruct correctly. VP8 motion vectors carry an eighth-pixel
fraction (`mv & 7` per component); when either fraction is non-zero,
§18.3 synthesises the missing samples via a horizontal then a vertical
one-dimensional six-tap convolution. New surface in
`src/motion_comp.rs`:

  * `SIXTAP_FILTERS` / `BILINEAR_FILTERS` — the §18.3 `filters` /
    `BilinearFilters` 8×6 tap tables (each row sums to 128: "DC is always
    passed"). `FilterSet` + `filter_set_for_version(v)` reproduce the
    §20.14 `version == 0 ? sixtap : bilinear` selection; both luma and
    chroma share the frame's one set.
  * `interp(fil, support)` — the §18.3 single-sample six-tap
    `clamp255((Σ p·fil + 64) >> 7)`. `sixtap_horiz` / `sixtap_vert` /
    `sixtap_2d(halo, mx, my, filters)` — the §20.14 convolutions, with
    the byte-clamped 9-row intermediate (negative partial sums clamp to
    0, exactly as the reference's `unsigned char` temp buffer).
  * `fetch_block_halo(...)` — the §20.14 `build_mc_border` 9×9
    edge-replicated support fetch (4×4 block + two-before / three-after
    taps; block origin at halo `(2,2)`). `filter_block_4x4(...)` — the
    §20.14 `filter_block` dispatcher: whole-pixel copy or `sixtap_2d`.
  * `predict_inter_mb(reference, mb_col, mb_row, luma_mv, full_pixel,
    filters)` / `reconstruct_inter_mb(..)` — the full non-SPLITMV
    prediction + §14-residue path, routing each sub-block through
    `filter_block_4x4`. The round-19 whole-pixel-only
    `predict_inter_mb_whole_pixel` / `reconstruct_inter_mb_whole_pixel`
    entry points are retained (they still refuse sub-pixel vectors).

Nineteen new unit tests: the tap-table values + sum-to-128 + DC
pass-through + bilinear-centre-taps shape; the version→set selection;
the `interp` formula incl. a negative-tap `clamp255` floor;
`sixtap_2d` byte-exact against an **independent** §18.3 Hinterp/Vinterp
transcription over every `(mx, my)` fraction for both filter sets
(plus flat-halo DC, whole-fraction copy, and a horizontal-only
known-value check); `fetch_block_halo` window + block-origin + corner
clamp; `filter_block` whole-pixel-copy vs sub-pixel dispatch; and the
whole-MB sub-pixel `predict_inter_mb` / `reconstruct_inter_mb` (per-block
agreement, filter-set sensitivity, skip == prediction, a sub-pixel
DC-residue path, and whole-pixel agreement with the round-19 legacy
path). Total: 281 tests across fourteen modules.

**Round 118 (2026-05-24).** §16.2 / §16.3 / §18.1 near/nearest
motion-vector census + inter-mode tree — `src/near_mv.rs`, the slice that
decides *which* vector a whole-MB inter macroblock uses (the round-19 /
round-117 motion-compensation slices take the resolved vector as input).
New surface:

  * `find_near_mvs(above, left, aboveleft, current_ref, sign_bias)` — the
    §16.3 / §20.11 `vp8_find_near_mvs` spatial census. Surveys the three
    neighbours (`MbInfo` records; `MbInfo::border()` is the §16.3 1-MB
    off-frame border of 0,0 vectors), accumulating the weighted candidate
    list (above / left weight 2, above-left weight 1) with the §16.3
    dedupe, the SPLITMV-merge-with-NEAREST rule, the near↔nearest swap,
    and the best := nearest store; returns the `best / nearest / near`
    candidates plus the four-entry `cnt` census. `SignBias` carries the
    §9.7 `sign_bias_golden` / `sign_bias_alternate` bits and drives the
    §16.3 `mv_bias` negation.
  * `mv_ref_probs(cnt)` — the §16.3 / §20.13 `vp8_mv_ref_probs`:
    `probs[i] = MV_COUNTS_TO_PROBS[cnt[i]][i]` (the `mv_counts_to_probs`
    / `vp8_mode_contexts` table). `read_inter_mode(dec, probs)` walks
    `MV_REF_TREE` (the §20.13 `mv_ref_tree`) into an `InterMode`.
  * `clamp_mv(mv, bounds)` / `MvClampRect::for_mb(...)` — the §16.3 /
    §18.1 / §20.11 `clamp_mv` one-MB-border clamp (quarter-pixel; the
    §20.11 eighth-pixel bounds halved consistently).
  * `resolve_inter_mb_mv(...)` — census → probs → mode → the single
    per-MB vector: ZEROMV zero, NEARESTMV / NEARMV clamped candidate,
    NEWMV clamped-best + decoded §17 differential (the §18.1 secondary
    clamp), SPLITMV reported with the clamped best base.
  * `decode_inter_mb(...)` — the end-to-end integration entry: runs the
    resolution then drives `reconstruct_inter_mb` with the resolved
    vector, so a whole-MB inter macroblock decodes from bitstream to
    reconstructed Y/U/V pixels.

Thirty-two new unit tests: tree / table shape, every inter-mode
round-trip (incl. under the default census probs), census coverage
(all-border, single-neighbour, intra-skip, zero-vector CNT_ZERO scoring,
dedupe, near↔nearest swap, SPLITMV weighting, sign-bias negate /
no-negate), clamp bounds + confinement, `mv_ref_probs` per-column
indexing, per-mode `resolve_inter_mb_mv`, and four byte-exact
`decode_inter_mb` end-to-end reconstructions (ZEROMV co-located copy,
NEARESTMV neighbour vector, NEWMV best + differential, SPLITMV error
surface). Total: 309 tests across fifteen modules.

**Round 119 (2026-05-24).** §16.4 SPLITMV per-sub-block motion-vector
decoding — the per-MB walk that turns a SPLITMV mode resolution into
sixteen Y-sub-block vectors plus four §18.1-averaged chroma vectors —
and the §18 SPLITMV reconstruction path. New surface:

  * `MvPartition` (`TopBottom` / `LeftRight` / `Quarters` / `Mv16`) +
    `MV_PARTITIONS[4][16]` (the §20.13 `mv_partitions` table) +
    `MV_PARTITION_TREE` (`{-3, 2, -2, 4, -0, -1}`, the §20.13
    `split_mv_tree`) + `MV_PARTITION_PROBS` (`{110, 111, 150}`).
    `read_mv_partition(dec)` walks the tree.
  * `SubMvRefMode` (`Left4x4` / `Above4x4` / `Zero4x4` / `New4x4`) +
    `SUBMV_REF_TREE` + `SUBMV_REF_PROBS[5][3]` (the §20.13
    `submv_ref_probs2`). `submv_ref_context(left, above)` derives one of
    five contexts (NORMAL / LEFT_ZED / ABOVE_ZED / LEFT_ABOVE_SAME /
    LEFT_ABOVE_ZED) per §16.4 `vp8_mvCont`; `submv_ref(dec, left, above)`
    reads the tree against the context-selected probability row.
  * `above_block_mv(this, above, b)` / `left_block_mv(this, left, b)` —
    the §20.11 neighbour-sub-block MV lookups: for top-row / left-column
    anchors look up the neighbour MB (SPLITMV neighbour → its bottom-row
    or right-column sub-block; otherwise its whole-MB vector; intra → 0);
    for interior anchors use the current MB's already-filled sub-block.
  * `decode_split_mv(dec, above, left, best, mv_ctx)` — the §16.4 /
    §20.11 `decode_split_mv` partition walk: reads the partition id,
    then per group finds the anchor sub-block, runs `submv_ref`, picks
    the partition vector (`LEFT4x4` / `ABOVE4x4` / `ZERO4x4` /
    `NEW4x4`-adds-diff-to-best), and fills every member sub-block.
    Returns `SplitMvResult { partition, split_mvs }`.
  * `MbInfo::split_mvs: Option<[Mv; 16]>` — the §20.5 `mb_info.split.mvs`
    array, populated when a neighbour was coded SPLITMV so the next MB's
    `above_block_mv` / `left_block_mv` lookups can borrow the correct
    sub-block vector.
  * `chroma_idx_for_luma_subblock(b)` + `split_chroma_mvs(luma_mvs)` —
    the §18.1 chroma derivation: maps each luma sub-block onto its
    chroma slot (`{0,1,4,5}→0`, etc.) and averages the four luma
    (stored-doubled) vectors per chroma slot via the §18.1 `avg()`
    primitive (sign-aware divide-by-8).
  * `predict_split_mv(reference, ...)` + `reconstruct_split_mv_mb(...)`
    — the SPLITMV §18 prediction + reconstruction path: sixteen luma
    sub-blocks each interpolated with their own §18.1-doubled vector
    (per-sub-block `filter_block_4x4` dispatch), four chroma sub-blocks
    interpolated with the §18.1 averaged vectors, no Y2 / DC-in-Y
    residue (§14.2 "for SPLITMV the 0th Y coefficients are part of the
    residue signal"). No §18.1 secondary clamp (per §18.1 page 114).
  * `decode_split_mv_mb(...)` — the SPLITMV analogue of
    `decode_inter_mb`: runs the §16.3 census + §16.2 inter-mode tree,
    asserts a Split resolution, runs `decode_split_mv`, then drives
    `reconstruct_split_mv_mb`. Returns the reconstructed pixels +
    `SplitMvResult` (caller stores `split_mvs[15]` as the MB's `mv` and
    `Some(split_mvs)` as the next neighbour's `MbInfo::split_mvs`).

Twenty-eight new unit tests: spec-verbatim table / tree shape for
`MV_PARTITIONS` / `MV_PARTITION_TREE` / `MV_PARTITION_PROBS` /
`SUBMV_REF_TREE` / `SUBMV_REF_PROBS`, `submv_ref_context` bucket
coverage, partition-tree round-trip across every shape, sub-MV-ref
tree round-trip across every mode + every context, neighbour-MV lookup
coverage (intra / non-split / split / internal for both `above_block_mv`
and `left_block_mv`), all-ZERO4x4 `decode_split_mv` for every partition
shape (verifies every group fills), per-mode SPLITMV semantics
(ZERO/NEW top-bottom, ABOVE4x4 top-bottom, LEFT4x4 left-right,
per-sub-block NEW4x4 Mv16), `chroma_idx_for_luma_subblock` grouping
against the §18.1 enumeration, `split_chroma_mvs` reduces to `chroma_mv`
on a uniform field, and two byte-exact `decode_split_mv_mb` end-to-end
reconstructions (zero-split co-located copy, TopBottom distinct halves
with whole-pixel-after-doubling shift). Total: 337 tests across sixteen
modules.

### Not yet landed

The top-level interframe `decode_vp8` driver — the per-MB census +
SPLITMV / inter / intra dispatch threaded across a whole frame with the
reference-frame plumbing (current `decode_vp8(&[u8])` decodes one frame
in isolation; an interframe needs a previously decoded reference to
predict against, requiring a multi-frame decoding-context API) — and the
encoder. The slice-level building blocks are now complete: §16.2 / §16.3
mode + MV resolution (round 118), §16.4 SPLITMV per-sub-block walk
(round 119), §18 whole-pixel motion compensation (round 19), §18.3
sub-pixel motion compensation (round 117), and SPLITMV §18
reconstruction (round 119) — both whole-MB inter (`decode_inter_mb`) and
SPLITMV (`decode_split_mv_mb`) macroblocks decode end-to-end at the
slice level. `decode_vp8` still returns `DecodeError::Unsupported` for
any non-key frame, and `encode_vp8_keyframe` still returns
`Error::NotImplemented`.

The round-16 `filter_frame` geometry targets the key-frame case (every
MB is intra / `CURRENT_FRAME`); the inter-frame mode/reference delta
ladder (the other three `mode_delta` slots and the non-current
`ref_delta` slots, plus the persisted-across-frames delta state) will be
wired when the §16 inter-prediction branch lands.

## Clean-room sources

* RFC 6386 — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).
* Black-box invocations of the `ffmpeg` / `libvpx` *binary* as an opaque
  validator (no source consulted): the `tests/fixtures/*/expected.yuv`
  reference pictures are that validator's output.

No external library source — libvpx, libaom, libavcodec/vp8\*, etc. —
is permitted as a reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
