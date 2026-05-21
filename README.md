# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status — 2026-05-22 (round 5)

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

Forty-two unit tests across four modules: the thirty-three carried
forward from rounds 1 / 2 / 3 / 4 plus nine new round-5 tests —
exhaustive `kf_ymode_tree` / `uv_mode_tree` / `bmode_tree` /
`mb_segment_tree` leaf round-trips (5 + 4 + 10 + 4 = 23 paths in 4
tests); an end-to-end 2-macroblock decode that exercises a `DC_PRED`
MB plus a `B_PRED` MB with sixteen sub-block modes feeding the
cross-MB context buffers; a segmentation + `mb_skip_coeff` path
exercising the optional §10 / §11.1 prefix bits; an interframe-header
guard test; and two `KF_YMODE_PROB` / `KF_UV_MODE_PROB` /
`KF_BMODE_PROB` transcription spot-checks.

### Not yet landed

Interframe macroblock prediction records (§16) including
`ymode_tree` / `bmode_prob[9]` decoding and the inter-prediction
mode tree (mv_nearest / mv_near / zero4x4 / new4x4 / split4x4);
intra-prediction pixel synthesis (§12); DCT-coefficient decoding
(§13) against the probability table populated by
`token_prob_update()`; motion-vector decoding (§17) against the
updated `MV_CONTEXT`s; IDCT and inverse-WHT (§14); the loop filter
(§15); the encoder. All top-level entry points (`decode_vp8`,
`encode_vp8_keyframe`) still return `Error::NotImplemented`.

## Clean-room sources

* RFC 6386 — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).
* Black-box invocations of the `ffmpeg` *binary* as an opaque
  validator (no source consulted).

No external library source — libvpx, libaom, libavcodec/vp8\*, etc. —
is permitted as a reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
