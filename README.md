# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status — 2026-05-21 (round 3)

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

Twenty-seven unit tests across the three modules: the nine bool-coder
round-trip cases from round 1; the nine frame-header tests from round
2; plus nine new coded-header tests (minimal-key-frame round trip,
maxed-out filter / partition / quantiser fields, every cell of the
`mb_lf_adjustments` sub-block with mixed signs, full
`update_segmentation()` block with mixed presence flags,
quant-indices with all five deltas present, the inter-frame
refresh-ladder full path including `copy_buffer_to_golden` =
"copy alt to golden", `mb_no_skip_coeff` = 0 omitting
`prob_skip_false`, short-input rejection through the bool-coder
`InputTooShort` surface, and an end-to-end parse of the 23-byte
first partition extracted from
`docs/video/vp8/fixtures/tiny-i-only-16x16/input.ivf` that
reproduces every field listed in the fixture's `trace.txt`:
`filter_simple = 0`, `filter_level = 1`, `filter_sharpness = 0`,
`lf_delta_enabled = 1`, `num_partitions = 1`, `seg_enabled = 0`,
`mbskip_enabled = 1`, `prob_skip = 255`, `qi_y_ac = 4`, every
quantiser delta absent, `update_probs = 1`).

### Not yet landed

`prob_intra` / `prob_last` / `prob_gf` and intra-mode / intra-chroma
probability updates (inter frames); `mv_prob_update()` (inter frames);
macroblock decode; intra / inter prediction; IDCT and inverse-WHT;
loop filter; and the encoder. All top-level entry points
(`decode_vp8`, `encode_vp8_keyframe`) still return
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
