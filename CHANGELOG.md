# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Changed — §12.3 B_TM_PRED routed through the shared SIMD TM kernel (round 334)

The §12.3 4×4 sub-block `B_TM_PRED` arm of `predict_b4x4` previously ran
its own inline scalar `4×4` double loop with a per-pixel `clamp255`. It
now routes through the shared §12.2 TM dispatcher (`predict_tm`), so on
the nightly-only `simd` feature it takes the existing
`core::simd::Simd<i16, N>` row kernel at width N = 4 alongside the §12.2
16/8 block sizes; stable builds take the byte-identical scalar fill.
`B_TM_PRED` is one of the most-selected sub-block intra modes on detailed
keyframe content, so this widens SIMD coverage onto a hot intra path with
no new kernel code (the kernel was already generic over N).

* The SIMD TM kernel's `i16` arithmetic reproduces the scalar `i32`
  `clamp255(L_r + A_c - P)` exactly at every width: each intermediate
  lies in `-255..=510`, far inside `i16`, so no lane wraps before the
  `simd_clamp(0, 255)`. `predict_tm_simd_matches_scalar_on_stress_inputs`
  now sweeps width 4 in addition to 16 / 8, asserting the SIMD and scalar
  fills agree bit-for-bit across the clamp-floor / clamp-ceiling
  endpoints, opposing ramps, alternating extremes, and pseudo-random
  triples.
* New `predict_b4x4_tm_matches_independent_scalar_formula` locks the
  public `predict_b4x4` Tm arm to an independent inline scalar reference
  across a clamp-straddling fixture and 64 deterministic random triples,
  proving the routing did not change the emitted bytes.
* `predict_tm_parity_pair` (the doc-hidden fuzz probe feeding
  `decode_stream_token_descent`) now accepts width 4 so any scalar/SIMD
  divergence on the §12.3 sub-block TM surfaces as a fuzz finding.

### Added — §12.2 DC-prediction edge-sum SIMD kernel (round 329)

The nightly-only `simd` feature now accelerates the averaging numerator
of the §12.2 DC intra predictor — the mode selected on the majority of
flat-region macroblocks (both 16×16 luma and 8×8 chroma). The per-edge
horizontal byte sum is rewritten as a single widening
`core::simd::Simd<u16, N>::reduce_sum` for the two §12.2 widths
(N = 16 luma, N = 8 chroma); the `predict_dc_with_optional_edges` two- /
one-edge averaging paths dispatch to it under the `simd` feature and fall
back to the scalar accumulate on every stable build.

* A `u16` lane cannot overflow: the widest edge holds 16 bytes of value
  255, summing to 4080, well inside `u16::MAX`. Integer addition is
  associative, so the tree reduction yields the identical total as the
  scalar left-fold — the rounding shift that follows is byte-identical.
* `dc_edge_sum_simd_matches_scalar_on_stress_inputs` proves the SIMD and
  scalar sums agree bit-for-bit across all-zero, all-255 (the maximum),
  the §12.2 off-frame defaults (127 / 129), ramps, alternating extremes,
  and 16 pseudo-random edges per width.
  `dc_edge_sum_drives_predict_dc_public_entry_points` confirms the public
  `predict_y16x16_dc` / `predict_uv8x8_dc` surface still emits the
  expected fill against an independent scalar average.
* New `dc_edge_sum_parity_pair` doc-hidden probe wired into the
  `decode_stream_token_descent` fuzz target so any scalar/SIMD divergence
  on attacker-shaped neighbour bytes becomes a finding. No behaviour
  change on the default (scalar) build.

### Added — `panic_free_split_mv_predict` fuzz target (round 318)

New `cargo fuzz` target driving the §16.4 / §18 whole-MB SPLITMV
prediction synthesiser `motion_comp::predict_split_mv` directly under an
attacker-shaped MV envelope — a surface no existing harness reached:
stream-decode fuzzers reach it only through §16.4 mode decode whose
sixteen vectors are bounded by the sub-MV-mode prediction tree, and the
single-sub-block fuzzers never run the whole-MB chroma-MV §18.1 averaging
/ per-sub-block doubling / raster pack.

* Drives an arbitrary `(MB grid ≤ 4×4, target MB position, sixteen luma
  vectors, version / full-pixel flag)` across four MV-shaping classes
  (verbatim full-`i16`, small-signed mid-plane, whole-pixel copy branch,
  degenerate all-equal SPLITMV). Per §18.1 page 114 SPLITMV applies no
  secondary clamp, so the §20.14 edge-replication inside `filter_block_4x4`
  is exercised densely by the adversarial vectors.
* Equivalence oracle on every iteration: each of the sixteen luma + four
  U + four V sub-blocks must equal, byte for byte, the `filter_block_4x4`
  block recomputed independently from `stored_luma_mv` (luma) or the
  `split_chroma_mvs` §18.1 average (chroma), with `apply_full_pixel`
  applied when the version-3 flag is set — a panic on any drift.
* Round-318 ASan run (nightly + default `simd`): ~2.85 M executions in
  61 s (cov 283 / ft 912, ~48 K exec/s, peak RSS 415 MB) from an empty
  seed; zero crashes / leaks / OOMs. No `src/` change was needed — the
  synthesiser is panic-free and byte-exact across the full envelope.

### Added — §15 loop-filter SIMD kernels (round 314, 2026-06-15)

The nightly-only `simd` feature now accelerates the §15 loop filter, the
heaviest decode-side stage on a fully coded frame. New
`core::simd::Simd<i32, 4>` kernels for the §15.2 simple filter and the
§15.3 normal subblock / MB filters process 4 independent edge segments
(rows of a vertical edge, columns of a horizontal edge) per group; the
frame-geometry edge loops dispatch to them under the `simd` feature.

* Byte-exact with the scalar per-segment kernels: the vector path
  computes all four lanes unconditionally and selects between the
  filtered and original pixel per lane with a `Mask` derived from the
  same §15.3 `filter_yes` / §15.2 edge-metric gate, so each lane runs the
  identical i32 add / `clamp_s8` / shift sequence. Three parity tests
  drive a 70-window stress matrix across five limit combinations.
* `loop_filter_frame` A/B (320×240, criterion median): keyframe normal
  352.7 → 196.5 µs (−44 %), keyframe simple 81.0 → 49.3 µs (−39 %), inter
  normal 342.6 → 205.6 µs (−40 %). See `BENCHMARKS.md`.
* The default (stable) scalar build is byte-identical to before this
  round; only the `simd` feature changes behaviour (and only in timing,
  not output).

### Added — automatic §9.7 golden-frame refresh cadence on the streaming encoder (round 309, 2026-06-15)

`Vp8InterStreamEncoder` gains an automatic §9.7 `refresh_golden_frame`
schedule, closing the documented gap where the scheduler-driven
`encode_frame` / `encode_frame_with_force` path only ever refreshed LAST
on P-frames — leaving GOLDEN frozen at the most-recent key frame's
reconstruction until the next key frame.

* New builder `with_golden_interval(n)` + getter `golden_interval()`.
  `n = 0` (the default after `new`) keeps the historical
  `refresh_last`-only auto path **byte-for-byte**; `n > 0` makes every
  `n`-th P-frame after a key frame also emit `refresh_golden_frame = 1`,
  writing the current reconstruction into the encoder's GOLDEN slot in
  lockstep with what the decoder writes from the wire.
* New predicate `next_p_frame_refreshes_golden()` reports — without
  encoding — whether the next scheduler-driven P-frame is a golden
  boundary (false for a disabled cadence or when the next frame is a key
  frame, since key frames already refresh every slot per §9.7).
* The cadence is counted in P-frames since the last key frame and resets
  on every key frame (automatic or forced), so the golden boundary
  re-anchors after a scene-cut keyframe override.

Routes through the existing `encode_p_frame_multi_ref_with_refresh`
primitive with `RefreshControls { refresh_golden_frame: true,
refresh_last: true, .. }`; ALTREF and the §20 `copy_buffer_*` paths are
untouched on the auto schedule (full manual control stays available via
the `encode_p_frame_with_refresh*` family). Five new lib tests: scheduled
GOLDEN update + frozen-otherwise, cadence reset on keyframe,
`golden_interval = 0` byte-identity against the plain auto path, a 1K+4P
scheduled-golden stream decoding cleanly through `Vp8DecoderState`, and the
disabled-by-default predicate. Lib tests 498 → 503 stable. No public
breakage; `new` is unchanged.

## [0.2.4](https://github.com/OxideAV/oxideav-vp8/compare/v0.2.3...v0.2.4) - 2026-06-15

### Other

- *(vp8)* shipped vs clone-everything A/B for §9 slot rotation (round 308 profile)
- §11 key-frame MB mode-info tree walk target (round 307)
- register-local read_literal fixed-prob-128 loop + literal A/B coverage (round 306 profile)
- add keyframe_stream_encode_decode target for Vp8KeyframeStreamEncoder (round 305)
- rename ffmpeg_oracle → blackbox_oracle (validator-name neutralization)
- *(decoder)* contiguous-plane fast path in crop_to_visible (round 304 profile)
- add decode_multi_partition_carve for §9.5 multi-DCT-partition layout walk
- collapse loop-filter coeff side-band to per-MB flag (round 302 profile-opt)
- add ivf_demux_decode_walk IVF container demux-loop target (round 301)
- drop dead per-frame coeff/side-band default-fill (round 300 profile-opt)
- add inter_stream_encode_decode_sequence multi-frame stream target (round 299)
- add dequantize_mb §14.1 dequantization micro-bench (round 298)
- r297 decode profile + §14.4 DC-only fast-path A/B cell
- add decoder_trait_packet_lifecycle target for the oxideav_core::Decoder trait driver (round 296)
- §7.3 bool-decoder primitive in isolation (round 295)
- §14.4 DC-only IDCT-add fast path — −14% whole-frame inter decode
- §14.1 dequant factor-derivation target + fix i32 add-overflow panic
- *(r292)* isolate §12.3 predict_b4x4 sub-block intra path + refresh decoder hotspot map
- lockstep §13.4 token_prob_update flag-read loop (round 291 profile-opt)
- §12/§14 keyframe-reconstruct decode-core target (round 290)
- move-minimising §9 reference-slot rotation (round 289 profile-opt)
- §9.7/§9.8 reference-slot rotation harness + ranked-table refresh (round 288)
- §16 inter-MB mode-info decode target (round 287)
- fuse §14.4 IDCT + §14.5 add-clamp residue pass (round 286 profile-opt)
- round-285 ranked-table shares recomputed against the exact profile sample totals
- r283/r284 hot-path harnesses + ranked hotspot table (round 285)
- move committed seed corpus into corpus/decode_stream_token_descent/seeds/ (round 284 follow-up)
- decode_stream_token_descent target + scalar/SIMD parity probes; fix decode_vp8 short-DCT-partition reject (round 284)
- fused §13 token descent + batched bool renormalisation (round 283)
- decoder-side whole-frame coverage + full-suite refresh (round 282)
- fused whole-pixel SAD scoring (round 281)
- pixel-exact encode→decode lockstep differential target (round 280)
- MB-batched sub-pixel SAD scoring (round 279)
- round-278 whole-frame keyframe path measured under nightly + simd
- compile-time SPLITMV partition-group table (round 277)
- MV-cost log2 table + allocation-free tree walks (round 276)
- document panic_free_filter_block_into in fuzz/README target table (r275)
- panic_free_filter_block_into r275 — §16.4 SPLITMV strided-write filter_block_4x4_into (RFC 6386 §16.4 / §18.3 / §20.14)
- filter_block_4x4_into strided-write primitive + round-274 SPLITMV write-strategy A/B (negative result)
- add panic_free_mb_batch_motion_comp target for round-270/271/272 MB-batched motion-comp primitives
- batch whole-pixel non-SPLITMV MB fetch (r272 depth)
- MB-scale §18.3 chroma batching (sixtap_mb_chroma) — round 271
- MB-scale §18.3 luma batching (sixtap_mb_luma) + SIMD partner (round 270)
- SIMD §18.3 sixtap_2d sub-pixel kernel under the `simd` feature (round 269)
- SIMD §12.2 TM_PRED intra kernel under the `simd` feature (round 268)
- SIMD §14.1 dequantize kernel under the `simd` feature (round 267)
- panic_free_encode_decode_e2e r266 — per-frame symmetric encode→decode round-trip (RFC 6386 §7 / §9 / §11 / §13 / §14 / §15)
- panic_free_inter_mb_reconstruct r265 — §16 inter-MB reconstruction surface (RFC 6386 §16 / §16.2 / §16.4 / §18)
- panic_free_loop_filter_writeback r263 — §9.4 / §19.2 loop-filter parameter writeback (RFC 6386 §9.4 / §19.2 / §15.4)
- panic_free_transform_4x4_roundtrip r262 — §14 transform / dequant / residue-summation primitives (RFC 6386 §14)
- panic_free_bool_codec r261 — §7 boolean range coder primitives (RFC 6386 §7)
- loop_filter_mb_edge r260 — §15.3 `MBfilter` wide deblock partner (RFC 6386 §15.3 / §15.4)
- panic_free_intra_predict_kernels r259 — §12 intra-prediction pixel kernels (RFC 6386 §12)
- block_sad_16x16 SIMD partner + leaf bench r258: §17 SAD primitive (RFC 6386 §17 / §18.3)
- panic_free_sixtap_subpel r257 — §18.3 / §20.14 sub-pixel synthesis primitives (RFC 6386 §18.3 / §20.14)
- panic_free_motion_search_descent r256 — §17.1 / §18.3 luma MV picker (RFC 6386 §17 / §18.3 / §20.14)
- motion_search_descent r255 — wall-time A/B target for the §17.1 / §18.3 luma MV picker (RFC 6386 §17 / §18.3)
- §13.2 walk-order anchor r254: byte-equal ENC_PCAT1..6 + ENC_CAT_BASE against the RFC 6386 §13.2 spec listing
- §17.2 walk-order anchor r253: byte-equal MV_UPDATE_PROBS_FLAT against the spec MV_UPDATE_PROBS table (RFC 6386 §17.2)
- §13.4 walk-order anchor r252: in-crate byte-equivalence against the actual COEFF_UPDATE_PROBS_FLAT (RFC 6386 §13.4)
- forward_wht_4x4 scalar+SIMD r251: rewrite listings into canonical (a1,b1,c1,d1) butterfly shape mirroring §14.3 inverse_wht_4x4 (RFC 6386 §14.3)
- forward_dct_4x4_simd r250: rewrite listing into canonical (a1,b1,c1,d1) butterfly shape, matching the round-249 scalar listing (RFC 6386 §14.4)
- forward_dct_4x4_scalar r249: rewrite listing into canonical (a1,b1,c1,d1) butterfly shape mirroring §14.4 inverse (RFC 6386 §14.4)
- drop release-plz.toml — use release-plz defaults across the workspace
- forward_dct_4x4 SIMD dispatch split r247: route public dispatcher to scalar even under `simd` feature (RFC 6386 §14.4)
- scrub artefacts to positive-only language
- tokens fuzz r237: panic_free_token_block target (RFC 6386 §13.2 / §13.3)
- loopfilter fuzz r232: panic_free_loopfilter_segment target (RFC 6386 §15)
- BENCHMARKS r226: soften whole-frame extrapolation, leave verification to a future profile-depth round
- forward_transform SIMD r226: §14.3 + §14.4 forward `core::simd<i32,4>` rewrites (RFC 6386 §14)
- encoder bench r220: forward_transform_4x4 micro-bench (RFC 6386 §14.3 / §14.4)
- encoder fuzz r213: panic_free_two_pass_stream multi-frame target (RFC 6386 §9.7)
- encoder fuzz r207: panic_free_encode_keyframe target (RFC 6386 §11/§13/§14/§15)
- encoder r204: precompute §13.2 token bit paths (RFC 6386 §13.2)
- round 200: cargo-fuzz harness suite + nested fuzz crate
- round 194: rate_control_qi_sweep bench + published 10-row trade-off curve

### Added — shipped vs clone-everything A/B for the §9 slot rotation in `ref_slot_rotation` (round 308 profile, 2026-06-15)

Profile round. No decoder/encoder logic change — a bench-measurement
correction. The round-288 `ref_slot_rotation` harness measured only a
*clone-everything* stand-in (`rotate_*` rows, six populated `Vec` clones
per frame) and reported ~12–13 µs/frame as the "slot copy cost", naming
copy-on-write as the next profile-opt target sized against that number.
But the shipped `rotate_reference_slots` is already a move-based
*minimal-copy* rotation: it moves each owned source into the last
destination that selects it and clones only a genuinely multi-fed source,
so on the common refresh-last path it performs **zero** plane clones.

This round adds a `#[doc(hidden)]` measurement shim
`bench_rotate_reference_slots` (a thin pass-through to the unchanged
private rotation) and three `shipped_*` bench rows that drive the shipped
rotation under the same three flag combinations as the existing
clone-everything rows. The A/B (320×240, criterion median):

| flags             | clone-everything | shipped (4 input clones) | Δ     |
| ----------------- | ---------------- | ------------------------ | ----- |
| `refresh_last_only` | 13.19 µs        | 8.71 µs                  | −34 % |
| `refresh_all`       | 16.21 µs        | 14.31 µs                 | −12 % |
| `copy_gf_arf`       | 17.20 µs        | 8.40 µs                  | −51 % |

The residual `shipped_*` cost is dominated by the four by-value input
clones the bench performs to set up owned arguments each iteration —
which `decode_frame` does **not** pay (it hands the rotation
`self.{last,golden,altref}.take()`, already-owned values it moves). The
genuine remaining copy work is the §16 cross-slot-copy shape (one
pre-rotation slot feeding two destinations → one clone) and the key-frame
init (one reconstruction → three slots → two clones), which only a
copy-on-write (`Rc`/`Arc`-backed) slot representation can remove; the
move-based rotation already removes the common-path clones. The standing
copy-on-write profile-opt target should be sized against the `shipped_*`
floor, not the `rotate_*` ceiling. See `BENCHMARKS.md` §"Round 308".

### Added — `panic_free_kf_mb_mode_decode` fuzz target: §11 key-frame macroblock mode-info tree walk (round 307 fuzz, 2026-06-15)

Fuzz round. New libFuzzer target `panic_free_kf_mb_mode_decode` drives the
§11 key-frame macroblock *mode-info* tree walk
(`macroblock::parse_key_frame_macroblock_modes`) directly off a bool-decoder
partition — the per-macroblock mode-decode side of the key-frame path no
existing target reached in isolation. It synthesises an attacker-shaped
`Vp8CodedHeader` (segmentation gate, `mb_segment_tree_probs` with the §9.3
item-5 255-fallback per entry, `prob_skip_false`) plus an attacker-shaped
bool partition, then exercises the §10 segment-id 4-leaf tree, the §11.1
`mb_skip_coeff` read, the §11.2 key-frame Y-mode tree, the §11.3 / §11.5
sixteen-sub-block `B_PRED` walk (including the cross-macroblock `above` /
`left` sub-block-mode predictor bookkeeping), and the §11.4 chroma-mode
tree. The partition uses `BoolDecoder::init_partition` (the §20 short-input
fallback) so a truncated / empty partition is a first-class input. The
reconstruction target `panic_free_keyframe_reconstruct` feeds
`decode_keyframe` *already-decoded* `MacroblockModes` and never walks these
trees; the bitstream-gated targets reach the walk only behind the §9.1 /
§19.2 validation gates, so the only mode probabilities they ever feed it
are a valid header's self-consistent ones. Oracle: panic-freedom plus, on
the `Ok` path, one decoded entry per macroblock and every decoded
`segment_id` inside the documented `0..=3` envelope. ~4.7M executions
clean. No `src/` change.

### Changed — `read_literal` register-local fixed-prob-128 loop + `bool_decoder_read` literal coverage (round 306 profile, 2026-06-15)

Profile round. `BoolDecoder::read_literal` is now a register-local
specialisation of the generic `num_bits × read_bool(128)` loop: the four
decoder registers (`range` / `value` / `bit_count` / `input`) are hoisted
into locals for the whole accumulator loop, and at the fixed probability 128
the §7.2 interval split collapses from a 32-bit multiply to a single shift
(`1 + (((range - 1) * 128) >> 8)` == `1 + ((range - 1) >> 1)`). Byte-for-byte
identical to the prior loop, including the mid-literal `EndOfStream` error
path, proven by two new lib tests (`read_literal_fast_matches_generic_loop`
sweeping widths 1..=16 with full state agreement, and
`read_literal_fast_matches_generic_on_end_of_stream`). Strict
instruction-count reduction (one fewer multiply + three fewer field reloads
per coded bit); the win sits below the per-run measurement noise on a loaded
machine, so it is kept as a no-regression micro-improvement rather than a
claimed speedup. The named r304 PROFILE-OPT target (copy-on-write
reference-slot representation behind a versioned `RefFrameSlot` API) was
deferred as too large / too risky for a single safe round (the §9 slot
rotation is already move-minimised) per the round's "do not force a risky
change" guidance. The `bool_decoder_read` bench gained `read_literal_mixed_width`
(widths 1..=16) and `read_signed_literal_7b_8k` (the §17 MV-component
`read_signed_literal` idiom) so those register-local loops have width-varied
A/B targets the flat width-8 case lacked. No bitstream / public-API change.

### Added — `keyframe_stream_encode_decode` fuzz target for the `Vp8KeyframeStreamEncoder` multi-frame driver (round 305 fuzz, 2026-06-15)

Fuzz round. New `cargo-fuzz` target `keyframe_stream_encode_decode`
(27th target) drives the public `Vp8KeyframeStreamEncoder` all-key-frame
stream driver — a cross-frame encoder surface no existing target reached
(`panic_free_encode_keyframe` is one-shot; `panic_free_two_pass_stream`
and `inter_stream_encode_decode_sequence` are the *inter* stream
drivers). Exercises both the plain `encode_frame` path and the §13.4
fitted companion `encode_frame_with_fitted_token_prob_updates` (the
round-157 token-prob fitter reached through a stream lifetime for the
first time), selected per frame by a fuzz bitmap over a sequence of up to
four ≤ 64 × 64 I420 frames. Three driver-specific oracles beyond
panic-freedom: the dimension-lock state machine (a non-first frame fed
altered dimensions must surface `StreamEncodeError::DimensionsChanged`
without advancing the frame counter), the §9.7 / §9.8 three-slot keyframe
refresh (`last == golden == altref` after every success, plus
counter-advances-by-one and dimensions-locked checks), and per-frame
self-decode through a fresh `Vp8DecoderState` to the locked visible
geometry (every decoded plane byte folded into FNV-1a as a short-write
oracle). Round-305 smoke pass (nightly, default `simd`): 21 415
executions in 31 s plus a 26 s confirmatory run (cov 4784 / ft 24762,
940-input corpus from empty seed), zero crashes. No `src/` change needed.

### Changed — contiguous-plane fast path in `crop_to_visible` (round 304 profile, 2026-06-14)

Profile round. `decoder::crop_to_visible` (the per-frame visible-crop emit,
the standing `_platform_memmove` hotspot's #3 caller at 52 / 605 samples)
now collapses its per-row `extend_from_slice` loop into a single contiguous
`to_vec()` whenever the visible width equals the macroblock-padded stride
(`w == y_stride` / `uvw == uv_stride` — the common case for 16-multiple
dimensions). The strided per-row path is retained byte-for-byte for the
genuine-crop case (visible < stride). Bit-identical output; a new lib test
`crop_to_visible_contiguous_matches_strided` sweeps aligned / cropped /
height-truncated cases against an always-strided reference. Whole-frame A/B
sits within measurement noise at benched sizes (crop is ≈ 0.7 % of decode
self-time) but the path is never slower than the prior loop and the benefit
scales with frame area — kept as a no-regression micro-improvement, no risky
change forced. The fresh r304 `sample(1)` profile also attributed every
`_platform_memmove` sample to its `oxideav_vp8` caller (see BENCHMARKS.md
round 304): the residual memmove headroom is concentrated in
`state::decode_frame`'s already-move-minimised slot rotation and the
`decode_mb_coeffs` return-by-value / load-bearing `MbCoeffs` zero-init, with
the named whole-frame PROFILE-OPT target still the copy-on-write
reference-slot representation behind a versioned `RefFrameSlot` API change.
Stable lib 496 (+1), nightly + `simd` lib 498 (+1).

### Added — `decode_multi_partition_carve` fuzz target for the §9.5 multi-DCT-partition layout walk (round 303 fuzz, 2026-06-14)

Fuzz round. New `cargo-fuzz` target (the crate's 26th) covering the §9.5
multi-DCT-partition decode path that the existing raw-bytes decode targets
essentially never reach: `decoder::carve_dct_partitions`' multi-partition
truncation / overshoot branches (`TruncatedPartitionSizes`,
`TruncatedDctPartition`, the `consumed > body.len()` overshoot scan) and
the §20.4 row-interleaved `r % nbr_of_dct_partitions` token descent.
Random libFuzzer bytes almost never survive §9.1 / §19.2 header validation
long enough to set the partition count to 2 / 4 / 8 behind a
self-consistent size table, so those branches sat almost unfuzzed.

The target builds a structurally valid 2 / 4 / 8-partition key frame with
`encode_keyframe`, asserts a clean round-trip back to the source geometry
(every decoded plane byte folded into an FNV-1a accumulator), then applies
four header-located mutations before re-decoding each mutant: corrupt one
3-byte LE size word, truncate inside the DCT section, perturb the 19-bit
`first_partition_size` field, and shrink one declared size to a
smaller-but-legal value (re-routing the per-partition `BoolDecoder` read
windows to drive several §7.3 out-of-data renormalisation tails at once).
Mutation offsets are computed from the freshly parsed `Vp8FrameHeader`
(`header_bytes_consumed`, `first_partition_size`), so no container framing
is re-implemented. Dimensions capped at 64 × 64.

Round-303 smoke pass (nightly, default `simd`): ~30 000 executions in 46 s
plus a 21 s confirmatory run (cov 3279 / ft 14877, 1008-input corpus from
empty seed) on aarch64-apple-darwin, zero crashes. No `src/` change
needed; `fuzz/Cargo.toml` registers the new `[[bin]]`. No public-surface
change.

### Changed — collapse the per-frame loop-filter coefficient side-band to a per-MB flag (round 302 profile-opt, 2026-06-14)

Profile round. The §15 loop filter's step-2/4 internal-edge decision
consults the per-MB coefficients only through `mb_has_coeffs` — a single
boolean per macroblock ("does this MB carry any non-zero DCT
coefficient"). The stateful interframe decoder
(`Vp8DecoderState::decode_frame`) previously held a whole-frame
`Vec<MbCoeffs>` (≈ 800 bytes / MB) alive solely to feed that one boolean
reduction after the frame finished decoding, even though each MB's
coefficients are fully consumed by reconstruction inside the per-MB loop.

The decode loop now computes `mb_has_coeffs` inline — while the
freshly-decoded `mb_coeffs` is still hot in cache — and stores only a
`bool` per MB, dispatching to new internal `filter_frame_flags` /
`filter_inter_frame_flags` cores that take a `&[bool]` "has-coeffs" slice.
The public `filter_frame` / `filter_inter_frame` signatures are unchanged
(thin wrappers that collapse `&[MbCoeffs]` to the flag slice). This drops
the per-frame `Vec<MbCoeffs>` allocation + write/read memory traffic on
the inter path (the win scales with frame size: the eliminated buffer is
`800 bytes × mb_count`).

A/B (`--warm-up-time 2 --measurement-time 8`, three interleaved runs each,
Apple M4-class aarch64): `inter_decode_short_clip/inter_decode_4f_128x128_qi32`
116.2 / 116.5 / 118.6 µs → 115.5 / 115.1 / 113.5 µs (≈ −1…−3 %, every
post-run median below baseline). `keyframe_decode` is flat (the keyframe
path retains the full `Vec<MbCoeffs>` for reconstruction, so no change
there). Bit-exact: two exhaustive equivalence tests
(`filter_frame_flags_matches_coeffs` / `filter_inter_frame_flags_matches_coeffs`)
sweep all 2⁶ per-MB occupancy masks × every Y-mode × simple/normal config
asserting the flag path equals the coeffs path byte-for-byte; the full
stable lib suite (495), nightly + `simd` lib suite (497), and all in-tree
integration tests (encode→decode pixel lockstep, P-frame roundtrips,
inter-stream, two-pass, `blackbox_oracle` black-box validator) pass on both
toolchains.

### Added — `ivf_demux_decode_walk` IVF container demux-loop fuzz target (round 301, 2026-06-14)

Fuzz round. New `cargo-fuzz` target (the crate's 25th) exercising the
IVF container demux walk feeding the stateful decoder — the realistic
`.ivf`-playback path no existing target reaches. `parse_headers` calls
`ivf::parse_header` / `ivf::parse_frame_header` once each on the raw
front bytes (no cursor advance, no second record, no payload feed);
`panic_free_decoder_state` / `decoder_trait_packet_lifecycle` drive
`Vp8DecoderState::decode_frame` from a synthetic `[u16-LE len][payload]`
packetiser that inherits no container framing. The new target parses the
32-byte global header, then walks frame records from offset 32 — carving
`data[off + 12 .. off + 12 + size]` for each 32-bit attacker-controlled
payload `size` and advancing the cursor — feeding each carved payload to
one persistent `Vp8DecoderState` so the §9.7 LAST / GOLDEN / ALTREF
reference ladder runs as multi-frame `.ivf` playback would drive it. The
`off + 12 + size` term is the integer-overflow / out-of-range surface a
hostile `size` field targets; the walk uses checked arithmetic + a
saturating bounds clamp so a corrupt `size` ends the walk cleanly. Every
decoded Y / U / V byte is folded into an FNV-1a accumulator (short-write
oracle). Seeded with two real fixture-derived intact `.ivf` files
(`i-only-64x64`, `i-frame-then-p-frame`). Hard caps: 8 KiB input,
≤ 64 records/iteration. ASan campaign (nightly + default `simd`):
~230 000 executions across two runs (cov 3560 / ft 13091, 831-input
corpus from the two seeds); zero crashes / leaks / OOMs — no `src/`
change needed. Library `src/` untouched; only `fuzz/` + READMEs.

### Changed — drop dead per-frame coefficient/side-band default-fill (round 300, 2026-06-14)

Profile-opt round. A `sample(1)` decode profile flagged a `__bzero` /
`memset` cluster (≈ 5 % of self-time) as a doubly-wasted per-frame
default-fill: both the §13 residual decode (`decoder::decode_residuals`,
`state::decode_intra_residuals`) and the §16 interframe driver
(`Vp8DecoderState::decode_frame`) built their per-MB output vectors with
`vec![default(); mb_count]` (or `with_capacity` + `resize(default())`)
and then overwrote **every** slot via an indexed write inside the
raster-order decode loop — so the bulk default-fill (800 bytes × every
MB on the `Vec<MbCoeffs>` lane, plus a `sentinel_mode` clone per
`modes_out` slot on the inter path) was never read. Replaced with
`Vec::with_capacity` + in-loop `push`; decode is exactly raster-ordered
(`raster = mb_row*mb_cols + mb_col`) so the vector contents are
bit-for-bit identical with the capacity reserved up front. Measured
≈ −3 % whole-frame inter decode and ≈ −2 % keyframe decode (every post
run below every pre run, interleaved A/B, nightly + `simd`,
Apple M4-class aarch64). Bytes-identical: full lib suite (493 stable /
495 nightly + `simd`) and all 37 integration test binaries green on both
toolchains; `cargo clippy --all-targets -D warnings` clean on both. See
`BENCHMARKS.md` round 300.

### Added — `inter_stream_encode_decode_sequence` multi-frame stream fuzz target (round 299, 2026-06-14)

New `cargo-fuzz` target (the 24th) covering the previously-unfuzzed
cross-frame stream layer: `Vp8InterStreamEncoder` driven across a
fuzz-shaped sequence of up to 12 small (≤ 48 × 48) I420 frames at a
fuzz-chosen keyframe interval with per-frame `force_keyframe`
overrides, each emitted frame fed into one long-lived
`Vp8DecoderState`. This is the first fuzz path to reach the §9.7
reference-refresh ladder across frames, the keyframe scheduler /
`force_keyframe` re-anchoring, the §9.4 `ref_frame_delta[]` /
`mode_delta[]` carry, the locked dimension guard, and the §16.2 / §18
ZERO_MV inter-prediction decode path (every prior encode target stops
at a single key frame). Oracles beyond panic-freedom: encode must not
reject an in-range frame, the stateful decoder must accept the
encoder's own output, the emitted K/P classification must match the
scheduler's pre-encode verdict (or be Key when forced), and decoded
visible dimensions + plane lengths must match the §9.1 geometry.
Round-299 ASan campaign (nightly, default `simd`): ~29 000 executions
(cov 4831 / ft 24338, 897-input corpus from empty seed), zero crashes /
leaks / OOMs. No `src/` change was needed — the panic-free decode
invariant held across the new surface. Test counts unchanged (fuzz-only
change).

### Added — `dequantize_mb` §14.1 dequantization micro-bench (round 298, 2026-06-14)

New criterion bench `dequantize_mb` (the 22nd) isolating the §14.1
dequantization layer, previously measured only folded inside the
whole-frame decode benches. Five functions across two layers: the §20.4
`dequant_init` factor derivation (`MbDequantFactors::from_quant_indices`
per-frame ≈ 2.32 ns, `for_segment` per-segment delta/absolute ≈ 2.9 ns —
six `dc_qlookup`/`ac_qlookup` reads through `clamp_qindex` plus the `*2` /
`*155/100` scalings and `>132` / `<8` clamps) and the per-MB apply
(`MbDequantFactors::dequantize`, 400 coefficient×factor multiplies over
all twenty-five 4×4 blocks of a macroblock) at two coefficient densities.
Headline finding: the apply is occupancy-independent — sparse (183.0 ns)
and dense (184.1 ns) macroblocks cost within 1 %, because the scalar
`dequant_block` walks all sixteen lanes of every block unconditionally;
this fixed-cost shape is exactly what the optional `core::simd`
`dequant_block` path (round 267) maps onto. Inputs synthesised in-bench
(no committed fixtures). Bench-only change — no decode/encode-path edit,
the full lib suite is unchanged.

### Added — decode profile + §14.4 DC-only fast-path A/B bench cell (round 297, 2026-06-14)

Profiled a whole-keyframe decode (`sample` over a tight `decode_vp8`
loop on a 320×240 qi=32 keyframe): `inverse_dct_4x4_scalar` is the #1
top-of-stack symbol of the decode path. On this shared/saturated host
the isolated `inverse_transform_4x4` single-call micro-bench swings
±20 % run-to-run, so no register-form scalar rewrite cleared the noise
floor — the §14.4 scalar listing is left byte-for-byte unchanged (no
`src/` change). Added an `idct_dc_only` A/B group to the
`idct_add_residue_fusion` bench isolating the DC-only residue shape
(every AC coefficient zero) that `inverse_dct_4x4_add_into`
short-circuits through its §14.4 DC-only fast path. Per-MB
(24 sub-blocks, same-binary A/B): ~110 ns DC-only vs ~254 ns
full-butterfly — the fast path is ~2.3× cheaper, quantifying the win it
already buys for the common well-predicted sub-block. Bench-only change.

### Added — `decoder_trait_packet_lifecycle` fuzz target: the `oxideav_core::Decoder` trait driver (round 296, 2026-06-14)

New `cargo-fuzz` target (the 23rd) hardening the public
`oxideav_core::Decoder` trait impl (`Vp8Decoder`) — the framework
decode entry point, distinct from the direct API (`decode_vp8`,
`Vp8DecoderState::decode_frame`) every other target uses. The harness
walks the full `send_packet` → `receive_frame` → `flush` → `reset`
lifecycle against a length-prefixed packet stream seeded from the
crate's own 13 fixture-derived IVF streams, exercising surfaces no
direct-API target reaches: the `Packet` clone into the internal
`VecDeque`, the `NeedMore` / `Eof` state transitions across the EOF
latch, the queue rebuild on `reset`, and the
`From<Vp8DecodedFrame> for VideoFrame` conversion (computed luma /
`ceil(w/2)` chroma strides + `pts` copy). Two lifecycle oracles beyond
panic-freedom: post-`flush` drain must surface `Eof` (never
`NeedMore`); post-`reset` empty receive must surface `NeedMore` (never
`Eof`); every produced plane byte is folded into an FNV-1a accumulator
so a stride / length mismatch surfaces under ASan. Per-keyframe §9.1
dimension cap, ≤ 16 KiB / ≤ 12 packets per iteration. Round-296 ASan
campaign (nightly + default `simd`): ~500 000 executions across two
runs (cov 3934 / ft 18688, 2629-input corpus from the 13 seeds); zero
crashes / leaks / OOMs — **no `src/` change was needed** (hardening,
not a bug fix). Fuzz-crate-only change (new target + committed seed
corpus + an `oxideav-core` dep on the fuzz sub-crate for the trait
types); no public-surface change.

### Added — `bool_decoder_read` bench: §7.3 boolean entropy primitive in isolation (round 295, 2026-06-14)

New `criterion` bench `bool_decoder_read` isolates the §7.3
`BoolDecoder::read_bool` / `read_literal` primitive and its batched
renormalisation step — the most-invoked decode primitive (every coded
bit flows through it), previously measured only folded inside the §13
token descent and whole-frame decode. Three regimes, each driven by a
partition produced by the crate's own §7.3 `BoolEncoder` and decoded
once at setup with a full bit/byte-equality assertion:
`read_bool_skewed_64k` (prob 248, renorm fast path), `read_bool_balanced_64k`
(prob 128 fair coin, renorm worst case), and `read_literal_8b_8k`
(the §9 / §17 `L(n)` idiom). Measured (Apple M4-class aarch64, criterion
`--quick`): skewed 151.8 µs / balanced 182.3 µs / literal 152.8 µs — the
fair-coin regime is **≈ 20 % slower per bool** (≈ 2.78 ns vs ≈ 2.32 ns),
confirming the renormalisation shift + byte-refill as the dominant
per-bit cost. Measurement-only: bench harness plus the partitions it
synthesises, no edit to any decode path. See `BENCHMARKS.md` round 295.

### Fixed — §14.1 dequant factor-derivation `i32` add overflow + new fuzz target (round 293, 2026-06-14)

New `cargo-fuzz` target `panic_free_dequant_factors_mb` over the §14.1 /
§20.4 dequant *factor-derivation* and full-macroblock dequant-apply
surface (`MbDequantFactors::from_base_and_deltas` / `from_quant_indices`
/ `for_segment` / `dequantize`) plus the §13.3 → §14.1 token → dequant
wrapper (`decode_and_dequantize_mb`). The pre-existing transform-
primitive target reached §14.1 only at the single-block leaf
(`dequant_block` with factors handed in directly) and the token target
stopped at `decode_mb_coeffs` without deriving factors — so the §20.4
`dequant_init` factor construction and the whole-MB apply were directly
under-fuzzed.

The target **found a real `attempt to add with overflow` panic**: the
internal `q + delta` index additions in `from_base_and_deltas` (and the
§10 per-segment `y_ac_qi + segment_quant` base add) panicked in a debug
build when an out-of-range base/delta pair was supplied through the
public `i32` API, even though `clamp_qindex` was meant to saturate the
index into `0..=127`. **Fixed** by forming every index sum with
`saturating_add` so the documented clamp does its job. A real bitstream's
§9.6 `u8` base index plus `i8` plane deltas never reach the `i32` edge,
so every decoded byte is unchanged; the fix only hardens the public
`i32`-typed API against arbitrary callers. New regression test
`extreme_base_and_deltas_saturate_without_overflow` anchors saturation at
both cliff ends across both the direct and per-segment derivation paths.

ASan campaign (nightly + default `simd`): 4 055 331 executions in 201 s
(cov 338 / ft 602); zero crashes / leaks / OOMs after the fix. The
existing 21 fuzz targets are unaffected.

### Added — `intra_predict_b4x4` §12.3 sub-block intra bench (round 292, 2026-06-14)

Bench-coverage round (no behaviour change, decoded bytes identical). New
`benches/intra_predict_b4x4.rs` isolates the §12.3 4×4 B_PRED sub-block
intra predictor (`intra_predict::predict_b4x4`) — the per-sub-block
directional kernel a B_PRED macroblock invokes sixteen times, the
dominant luma intra mode on detailed keyframe content and previously
only ever exercised folded inside the whole-frame keyframe-decode bench.
Covers all ten §12.3 directional modes individually (`b_dc … b_hu`) plus
a full sixteen-sub-block macroblock loop (`bpred_mb_16_subblocks`), over
a non-flat ramp neighbour context so the directional arithmetic can't be
constant-folded. Measured: the ten modes cluster at 4.3–4.7 ns; a full
B_PRED MB's luma intra prediction is ≈ 84.6 ns — well below its per-MB
§13 token-decode + reconstruct cost, confirming B_PRED intra prediction
is a regression-floor target, not a decoder hotspot. `BENCHMARKS.md`
gains the round-292 section + a refreshed ranked decoder hotspot map
(token decode #1, reference-slot copy churn #2, `reconstruct_inter_mb`
sub-pel #3). Pure additive: one bench binary + its registration + docs,
no `src/`/`tests/` edits, every decode path byte-for-byte unchanged.

### Changed — §13.4 `token_prob_update()` lockstep flag-read loop (round 291, 2026-06-14)

Profile-opt round landing the round-288/289 named decoder target: the
§13.4 `coded_header::parse_token_prob_update` per-frame flag-read loop
(standing decoder rank-5 self-time, ≈ 5 %; 1056 update flags = 4×8×3×11,
each read at its position-specific `COEFF_UPDATE_PROBS` probability). The
pre-291 loop carried a 4-deep `enumerate()` whose `(i,j,k,t)` indices
existed only to re-index `COEFF_UPDATE_PROBS[i][j][k][t]` — four bounds
checks + index arithmetic per flag, re-traversing the table the loop was
already walking structurally. The fixed walk order lets the output array
and probability table be zipped in exact lockstep, so the inner read
indexes neither array. Arithmetic, flag order, and per-position
probabilities are unchanged; output is **bit-identical** by construction,
anchored by the new `parse_token_prob_update_matches_indexed_reference`
test (mixed `Some`/`None` payload exercising the `L(8)` literal branch
across all four planes, byte-equal + same-stream-position against a
verbatim copy of the pre-291 4-level-indexed loop) plus the full lib
suite (491 stable / 493 nightly+`simd`), every roundtrip/decode
integration binary (37), and the `blackbox_oracle` validator. Measured
**−10.6 %** (`parse_no_updates`, the common no-update frame) / **−15.7 %**
(`parse_sparse_updates`) on the new `token_prob_update` micro-bench
(criterion p < 0.05, same-session A/B). See `BENCHMARKS.md` §"Round 291".

### Added — `token_prob_update` bench: §13.4 flag-read loop (round 291, 2026-06-14)

New `token_prob_update` criterion binary isolating the §13.4
`parse_token_prob_update` flag-read loop (the whole-frame decode benches
fold it inside the rest of the §9 header parse). Drives
`bench_parse_token_prob_update` over a header partition pre-encoded by the
crate's own §13.4 writer, validated by a full payload-equality assertion
at setup. Two shapes: `parse_no_updates` (the common all-`None` frame) and
`parse_sparse_updates` (scattered `Some(prob)` replacements exercising the
`read_literal(8)` branch). Bench-only `#[doc(hidden)]` re-exports added:
`coded_header::bench_parse_token_prob_update` and the now-`pub`
`encoder::COEFF_UPDATE_PROBS_FLAT`.

### Changed — move-minimising §9 reference-slot rotation (round 289, 2026-06-13)

Profile-opt round landing the round-288 named decoder target: the §9 /
§20-page-147 reference-frame slot rotation in
`Vp8DecoderState::decode_frame` (the `memmove`/`memset` #3 self-time
cluster, ≈ 15 %). The rotation previously cloned every input — the
just-decoded frame, the three entry slots, and again into each refreshed
destination (up to six populated `Vec` clones/frame). It now resolves each
new `LAST`/`GOLDEN`/`ALTREF` to a symbolic source (`rotate_reference_slots`)
and **moves** each owned source into its destination, cloning only on
genuine fan-out; the visible-cropped output is built from `planes` first so
the just-decoded slot consumes `planes` by move. The common
`refresh_last`-only ladder now does zero plane copies (the keyframe path
moves `planes` into one of its three slots). Output is bit-identical — the
rotation only selects and replaces whole slots — anchored by the new
`rotation_matches_clone_everything_reference` test (every entry-slot
population × every refresh-control combination, byte-equal against the prior
clone-everything ladder) plus the full roundtrip/oracle suite. Measured
**−9.3 %** (refresh-last) / **−10.6 %** (golden+altref cadence) whole-stream
on the `ref_slot_rotation/decode_1k8p_*` benches (criterion p < 0.05,
same-session A/B). See `BENCHMARKS.md` §"Round 289".

### Added — `ref_slot_rotation` bench: §9.7/§9.8 reference-slot rotation (round 288, 2026-06-13)

Bench-only (measurement) round; decoder and encoder byte-identical (no
`src/` logic change). Adds the `ref_slot_rotation` criterion binary, the
isolated harness for the round-285/286 ranked decoder **runner-up** — the
§9.7/§9.8 reference-frame slot rotation (`RefFrameSlot::clone` churn the
`memmove`/`memset` family attributes to, the #3 self-time cluster at
≈ 15–16 % on the inter decode profile). Two layers: a `rotate_*` micro
that replicates the exact `decode_frame` temporaries (`current_slot` +
`pre_{last,golden,altref}` + `new_{altref,golden,last}`) over public
`RefFrameSlot` fields across three flag combinations (refresh-last-only /
refresh-all / cross-slot-copy), and two `decode_*` whole-stream decodes
through `Vp8DecoderState` under a refresh-last vs golden/altref cadence.
The micro measures ~12.2 µs/frame of slot copying even on the common
refresh-last path (six populated `Vec` clones); the whole-stream rows
show the golden/altref cadence adds only ~1.3 %, confirming the rotation
is a near-fixed per-frame cost driven by the unconditional clones, not
refresh frequency. Names the next profile-opt target: copy-on-write /
slot-swap planes (`Rc`/`Arc`-backed or `mem::swap` of the owned current
reconstruction into LAST). See `BENCHMARKS.md` §"Round 288".

### Added — fuzz target for the §16 inter-MB mode-info decode path (round 287, 2026-06-13)

Fuzz-depth round; decoder and encoder behaviour byte-identical (no `src/`
logic change). Adds `fuzz/fuzz_targets/panic_free_near_mv_mode_decode.rs`,
a panic-freedom harness for the §16.2 / §16.3 / §16.4 / §17 inter-MB
*mode-info decode* surface that the prior twenty targets leave cold.

Surface driven directly from a bool-coder partition plus adversarial
neighbour `MbInfo` records, sign-bias flags, and attacker-tiled MV
probability contexts: `decode_inter_mb` / `decode_split_mv_mb` (the §18
end-to-end integration entry points), `resolve_inter_mb_mv`,
`find_near_mvs` (the §16.3 spatial census), `decode_split_mv` (the §16.4
partition walk), `read_inter_mode` / `read_mv_partition` / `submv_ref`
(the three bool-tree walks), `above_block_mv` / `left_block_mv` (the
§20.11 neighbour-MV lookups across all sixteen sub-blocks, including the
SPLITMV `b+12` / `b+3` branches and the intra-zero fallback),
`mv_ref_probs`, `submv_ref_context`, and `clamp_mv`. The round-263
`panic_free_inter_mb_reconstruct` target feeds vectors into the §16
*reconstruction* orchestrators directly; this new target exercises the
bool-decoder-driven §16.3 census → §16.2 tree → §16.4 SPLITMV walk → §17
NEWMV differential that *produces* those vectors — a path the
`decode_vp8` targets reach only through self-consistent §9.7 reference
state. The harness asserts the §16.3 census counts stay inside the
documented `0..=5` `mv_ref_probs` index envelope on every iteration.

ASan campaign (nightly, default `simd` feature, macOS aarch64, cross-
seeded from the inter-MB-reconstruct and decode-stream corpora): ~11.0 M
executions in 181 s (~60.8 K exec/s), peak RSS 572 MB, 1316 new corpus
units, **zero crashes / leaks / OOMs / assertion failures**. The
scheduled `Fuzz` workflow auto-discovers the new target (now 21).

### Changed — fused §14.4 IDCT + §14.5 add-clamp residue pass (round 286, 2026-06-13)

Profile-opt round targeting the round-285 #1 decoder hotspot:
`reconstruct_inter_mb`'s §14.4/§14.5 residue pass. The per-sub-block
`inverse_dct_4x4` → `extract_4x4` → `add_residue_4x4` → `insert_4x4`
four-buffer round-trip is replaced by a single
`inverse_dct_4x4_add_into` helper that folds the second inverse-DCT pass
with the §14.5 add-clamp written straight into the strided prediction
raster. Applied to all three inter-reconstruction entry points
(`reconstruct_inter_mb`, `reconstruct_inter_mb_whole_pixel`,
`reconstruct_split_mv_mb`).

Output is bit-identical: every in-tree fixture (`i-frame-then-p-frame`,
`golden-update-cycle`, `altref-arnr-on`, plus all keyframe fixtures)
still decodes byte-exact against its `expected.yuv`, and a new
`fused_idct_add_into_matches_unfused_sequence` unit test proves the
fused helper equals the prior sequence over a randomized sweep of
coefficients, strided positions, and predictors on both the scalar and
SIMD paths.

The SIMD (nightly + `simd`) path — the one the round-283/285 profile
measured — folds the store lane-wide and is faster on the per-MB
`reconstruct_inter_mb` bench: sub-pixel full-residue 462 → 426 ns
(-8 %), whole-pixel full-residue 323 → 285 ns (-12 %), skip unchanged
(no residue pass). The residue-pass-only delta (full minus skip) drops
-16 % (sub-pixel) / -58 % (whole-pixel). The scalar (stable / default-CI)
path keeps the contiguous-buffer sequence behind the same helper because
a strided fold regressed it ~6 % there (the round-274 scratch-then-copy
lesson), so the default build is unchanged. New A/B bench
`benches/idct_add_residue_fusion.rs` times both shapes in one run.

### Added — hot-path bench harnesses + ranked hotspot table (round 285, 2026-06-12)

Bench-only round; decoder and encoder behaviour byte-identical (no
`src/` change). Three new criterion binaries close the harness gaps the
round-283 profile left, and fresh `sample(1)` profiles of both inter
benches produce a ranked hotspot table in `BENCHMARKS.md`:

* `benches/token_decode.rs` — the fused §13.2 token descent in
  isolation: `decode_mb_coeffs` over 64-MB token partitions produced by
  the crate's own §13 `TokenEncoder` and verified coefficient-exact at
  setup (dense ≈ 7.8 µs/MB, sparse ≈ 0.75 µs/MB), plus an inter-heavy
  1-keyframe + 11-P-frame 176×144 whole-stream decode
  (1.36 ms stable / 1.09 ms nightly + `simd`).
* `benches/reconstruct_inter_mb.rs` — the round-283 #1 decoder
  self-time symbol in its three workload shapes (sub-pixel
  full-residue 569/464 ns, whole-pixel full-residue 424/344 ns,
  skip 280/273 ns, stable / nightly + `simd`).
* `benches/subpel_sad_scoring.rs` — the encoder's #1 cluster
  decomposed: one §17 sub-pixel `mb_luma_sad_at_mv` candidate
  ≈ 184 ns = halo fetch 22 + whole-MB §18.3 convolution ≈ 156 +
  SAD ≈ 6 — the micro-bench the standing "sub-pixel SAD without patch
  materialisation" candidate required.

The ranked table names the next profile-opt target:
`reconstruct_inter_mb` §14.4/§14.5 residue fusion (the per-sub-block
`extract_4x4`/`insert_4x4`/scratch round-trips are ~85 ns/MB of pure
data movement; fuse the inverse-DCT output pass with the add-clamp over
the strided prediction raster). Runner-up: per-frame reference-slot
plane-copy churn (slot-swap / copy-on-write).

### Fixed — `decode_vp8` rejected spec-legal frames with a sub-2-byte consumed DCT partition (round 284, 2026-06-12)

Fuzz-depth round on the round-283 hot-path rewrite. The new
`decode_stream_token_descent` target's cross-entry-point differential
(fresh `Vp8DecoderState::decode_frame` vs one-shot `decode_vp8` on the
same first packet) found a real divergence within its first minute of
execution: `decode_residuals` (the `decode_vp8` keyframe path)
initialised consumed DCT partitions with the strict control-partition
`BoolDecoder::init`, which rejects `len < 2` with `InputTooShort`,
while the stateful keyframe path (`state::decode_intra_residuals`)
uses the tolerant §20.2 `init_bool_decoder` form
(`BoolDecoder::init_partition`: `sz < 2` → zero-initialised value
register, empty input) — what the RFC 6386 §20.2 reference listing
does. A spec-legal key frame whose DCT-partition carve leaves a
consumed partition shorter than 2 bytes therefore decoded through
`Vp8DecoderState` but errored through `decode_vp8`.

`decode_residuals` now uses `BoolDecoder::init_partition`, matching
the stateful path and the §20.2 reference. Regression pinned with the
fuzz-found 57-byte witness in
`tests/decode_short_dct_partition_parity.rs` (both entry points must
accept it and produce byte-identical planes); the witness is also
committed as a corpus seed. Fixture-corpus byte-identity: all 13
`tests/fixtures/*/input.ivf` streams (27 frames, 205 056 decoded
plane bytes) hash to FNV-1a `65266afd221ea43c` before AND after the
fix, on stable and on nightly + `simd` — the fix only changes
behaviour on streams the one-shot path previously rejected.

### Added — fuzz depth on the round-283 fused token-descent decode path (round 284, 2026-06-12)

* **`decode_stream_token_descent` fuzz target** — full-frame
  multi-packet decode driver aimed at the round-283 rewrite (fused
  §13.2 descent, §20.16 zigzag-direct writes, §7.3 batched
  renormalisation), with the suite's first committed seed corpus: the
  13 `tests/fixtures/*/input.ivf` streams (keyframes AND inter
  frames) re-framed as `[u16-LE len][payload]` packet sequences, so
  mutations corrupt real token partitions and inter-frame mode/MV
  data against valid reference state. Oracles: cross-entry-point
  differential (above), an FNV-1a fold over every decoded plane byte,
  and a scalar-vs-SIMD kernel differential.
* **Scalar↔SIMD parity probes** (`#[doc(hidden)]`, `simd`-gated,
  behaviour-neutral — nothing in the decode/encode pipeline calls
  them): `dequant::dequant_block_parity_pair` (§14.1),
  `inverse_transform::inverse_wht_4x4_parity_pair` /
  `inverse_dct_4x4_parity_pair` (§14.3 / §14.4),
  `intra_predict::predict_tm_parity_pair` (§12.2), and
  `motion_comp::sixtap_2d_parity_pair` /
  `sixtap_mb_luma_parity_pair` / `sixtap_mb_chroma_parity_pair`
  (§18.3 / §20.14). Each runs the private scalar and SIMD
  implementations on the same input and returns both results so the
  fuzz harness turns any divergence into a finding — extending the
  fixed-stress-set in-tree equivalence tests to attacker-shaped
  inputs.
* **Fuzz crate defaults to `simd`** — cargo-fuzz always builds on
  nightly, so the SIMD dispatch path (what production nightly builds
  run) is what gets fuzzed by default across all 19 targets, while
  the differential leg keeps the scalar kernels covered in the same
  process. `--no-default-features` fuzzes the pure-scalar dispatch.
* **Scheduled `Fuzz` workflow** (`.github/workflows/fuzz.yml`) —
  daily ASan run over every discovered target via the shared
  `crate-fuzz` reusable workflow (30-minute budget, corpus cached
  across runs).

### Performance — fused §13 token descent + batched bool-decoder renormalisation (round 283, 2026-06-12)

Takes the round-282 ranked candidate #1 (§13 token decode, ≈ 31 % of
inter decode under `simd`) with the decoder-side mirror of the
round-204 encoder playbook. Three pieces, all output-bit-identical:

* **Fused branch-coded §13.2 descent** (`dct_tokens::decode_block_core`)
  — the per-coefficient generic `COEFF_TREE` table walk + double
  token-enum dispatch is written out branch-by-branch over the fixed
  tree; each leaf flows straight into its consequence and the §13.2
  "skip dct_eob after DCT_0" rule becomes an inner zero-run loop, so
  the `prevCoeffWasZero` flag and per-position restart disappear.
* **Write-order-table raster output** — `decode_mb_coeffs` hands the
  §20.16 `ZIGZAG` table to the core as the write order, landing every
  coefficient in its raster slot as it is decoded; the scan-order
  scratch block, the 16-lane `scan_to_raster` permute, and the
  per-block return copy are gone. The public `decode_block` keeps its
  scan-order contract via an identity table.
* **Batched bool-decoder renormalisation** (`bool_decoder`) — the §7.3
  bit-at-a-time doubling loop (up to 7 dependent iterations per
  `read_bool`) collapses into a single `leading_zeros`-derived shift
  plus at most one input-byte splice. Decoder-wide: every header,
  mode, MV, and token bool benefits.

Measured (three interleaved pre/post pairs per config, 30 s
measurement, Apple M4 / aarch64): `keyframe_decode` **−7.2 %** stable
/ **−9.7 %** nightly + `simd` (152.98 → 141.95 µs / 121.40 →
109.66 µs); `inter_decode_short_clip` **−7.8 %** / **−8.5 %**
(189.12 → 174.37 µs / 159.77 → 146.25 µs); non-overlapping pre/post
populations in all four configs. The §13 pair's self-time share drops
≈ 31 % → ≈ 24 % and cedes profile #1 to `reconstruct_inter_mb`.

Bit-identity: two new equivalence anchors
(`batched_renormalize_matches_bit_at_a_time_listing`,
`fused_descent_matches_generic_tree_walk` — lib 486 → 488 stable,
488 → 490 nightly + `simd`), the full 38-target suite, and a 55-frame
decode-side byte-hash A/B (3 resolutions × 3 quantisers × 6 frames
through `Vp8DecoderState` + the 320×240 keyframe through
`decode_vp8`; FNV-1a `ec93aa4f7f728ebe` over 1 324 800 decoded plane
bytes) identical pre-/post-change on both toolchains. See
`BENCHMARKS.md` round 283.

### Benchmarks — decoder-side coverage + full-suite refresh (round 282, 2026-06-12)

Bench-only round; `src/` is byte-identical to round 281. Two new
criterion benches close the decoder-side coverage gaps:

* `benches/inter_decode_short_clip.rs` — the decode half of the §16
  inter roundtrip: `Vp8DecoderState` replaying the 1K+3P 128×128
  stream the `inter_encode_short_clip` clip produces (stream built in
  setup, unmeasured). Baseline on Apple M4 / aarch64: **188.2 µs /
  348 Mpx/s stable, 161.7 µs / 405 Mpx/s nightly + `simd`** (−14 %).
* `benches/loop_filter_frame.rs` — the whole-frame §15 / §20.6 deblock
  pass (`filter_frame` normal + simple, `filter_inter_frame` normal)
  on a fully-coded 20×15-MB frame at filter level 26: 334.6 / 78.5 /
  346.9 µs stable. The per-edge primitives were already benched
  (`loop_filter_normal`, `loop_filter_mb_edge`); the frame-level pass
  — per-MB level resolution, §15.1 skip rules, raster edge cascade —
  was not.

The full regression table in `BENCHMARKS.md` was re-measured on both
toolchains (whole-frame benches at `--measurement-time 30`, micro at
`--quick`), confirming the r278–r281 state within session drift; the
`rate_control_qi_sweep` outputs are byte-identical to the round-194
record while its wall column dropped ~30 % (the accumulated r204–r281
encoder wins at constant output). Fresh `sample(1)` decoder profiles
(keyframe stable, inter stable, inter simd) rank the next decoder
candidates: §13 token decode (`dct_tokens::decode_block` +
`decode_mb_coeffs`, ≈ 31 % of inter decode under `simd`), per-frame
reference-slot plane copying in `Vp8DecoderState::decode_frame`
(`memmove` family ≈ 11–14 %, mostly `RefFrameSlot::clone`), and the
§13.4 `parse_token_prob_update` per-frame fixed cost (≈ 5 %).

### Performance — fused whole-pixel SAD scoring in motion search (round 281, 2026-06-12)

Closes the round-279 BENCHMARKS candidate "whole-pixel SAD scoring
batching". A fresh `sample(1)` profile of the inter-encode bench put
`motion_comp::fetch_block_whole_pixel` at #1 self-time (2290 of ~9276
in-process samples) and `encoder::group_sad_at_whole_mv` at #3 (1371):
the §17 integer-pixel diamond descent (`mb_luma_sad_at_whole_mv`) and
the SPLITMV group scorer still fetched sixteen (or per-group fewer)
separate 4×4 patches per whole-pixel candidate. Both scorers now run a
fused fetch-and-SAD: a whole-pixel candidate's prediction is a direct
window into the reference plane, so when the 16×16 MB-extent source
region is strictly in-bounds (the dominant case) the SAD is accumulated
straight off the reference and source rows — no patch materialisation,
no source sub-block extraction copy. Border-straddling candidates fall
back to the batched `fetch_luma_mb_whole_pixel` (non-SPLITMV) or the
per-member `fetch_block_whole_pixel` path (SPLITMV groups), preserving
the §20.14 `build_mc_border` edge replication bit-for-bit.

* Measured (Apple M4 / aarch64, `--measurement-time 30`, three
  interleaved pre/post pairs per config):
  `inter_encode_short_clip/inter_encode_4f_128x128_qi32` — see
  `BENCHMARKS.md` round 281 for the full table.
* Bit-identical output: every candidate SAD is unchanged (new
  equivalence anchors
  `whole_mv_sad_matches_per_subblock_fetch_assembly`,
  `whole_mv_sad_matches_assembly_with_padded_stride`,
  `group_sad_fused_fast_path_matches_per_subblock_fetch` sweep
  in-bounds + all four border directions across every §16.4 partition
  shape), plus a 54-frame FNV-1a byte-hash A/B (3 resolutions × 6
  frames × 3 quantisers through `Vp8InterStreamEncoder`) identical
  pre-/post-change on stable and nightly + `simd`.

### Fuzzing — pixel-exact encode→decode lockstep differential target (round 280, 2026-06-12)

New `cargo-fuzz` target `encode_decode_pixel_lockstep` (the suite's
eighteenth), plus its deterministic in-CI companion suite
`tests/encoder_decoder_pixel_lockstep.rs` (5 anchors). No `src/`
changes.

* **Oracle** — the round-264 `panic_free_encode_decode_e2e` target
  stitches `encode_keyframe` → `decode_vp8` but only asserts the §9.1
  visible width / height round-trip; decoded pixels were never
  compared against anything. The new target uses the encoder's own
  post-§15 reconstruction planes (returned by
  `encode_keyframe_with_reconstruction_and_token_updates`; per the
  §15.1 lockstep contract a compliant decoder reproduces them exactly)
  as a bit-exact differential oracle: every visible Y / U / V byte of
  the decoder's output is asserted equal, so a single-pixel drift in
  the §12 intra-prediction / §14 dequant + inverse-transform / §15
  loop-filter / §9.1 visible-crop chain panics. Parameters are
  normalised into their legal §9.4 / §9.5 / §9.6 ranges, so an `Err`
  from either half, any dimension / MB-grid drift, or any pixel
  mismatch is a finding rather than a silent early-return.
* **Coverage gaps closed** — (1) first pixel-content oracle in the
  fuzz suite; (2) non-MB-aligned dimensions (raw 1..=64 × 1..=144 luma
  px, the tall end populating all 8 §9.5 DCT partitions via the §20.4
  `row % N` round-robin), putting the partial-macroblock
  edge-replication padding / §15-on-padded-raster / visible-crop seam
  on the hot path — the e2e target only ever encodes whole-MB frames;
  (3) the §13.4 `token_prob_update()` **write** path (previously
  unfuzzed; `parse_headers` / `panic_free_token_block` cover the read
  side only) with raw 0..=255 probability bytes — the full L(8) wire
  range, wider than the `[1, 255]` band the in-tree fitter emits.
* **Run** — 5-input seed corpus (partial-MB, tall 8-partition frame,
  §13.4 probability extremes, 1×1 strip, lf-skip + updates), then
  79 957 execs over 661 s wall on aarch64-apple-darwin under ASan
  (~120 exec/s — every iteration runs the full encode + decode
  pipeline): `cov: 3710, ft: 17209`, corpus grew to 1 294 inputs, peak
  RSS 472 MiB, **zero findings** — no panic, no encode / decode
  rejection, no dimension drift, no pixel drift.
* **Test delta** — +5 integration tests (partial-MB normal filter,
  simple-filter §9.4 extremes, 1-pixel strips, 8-partition
  populated / empty layouts, §13.4 L(8) extremes 0 / 255 under both
  the §15 skip and filtered paths). Full crate suite green: 664
  passed, 0 failed.

Whole-frame `inter_encode_short_clip` A/B under nightly + `simd`
(closing the round-278 symmetric next-round candidate) measured the
`simd` dispatch at only **−1.8 %** on the inter path and attributed the
gap: `motion_comp::sixtap_2d` was the #1 self-time symbol on *both*
scalar and simd builds (≈ 2160 / 2186 of ~7700 samples), living in
`mb_luma_sad_at_mv` — the §17 half-/quarter-pixel refinement scored
each of its 17 candidates per MB as sixteen separate
`filter_block_4x4` calls (sixteen overlapping 9×9 halo fetches +
sixteen 4×4 six-tap passes), a shape the rounds 270–271 MB-batched
kernels never reached.

* `mb_luma_sad_at_mv` now synthesises a candidate exactly the way
  `predict_inter_mb`'s luma half does: a whole-pixel candidate is one
  contiguous `fetch_luma_mb_whole_pixel` 16×16 fetch; a sub-pixel
  candidate is one 21×21 `fetch_luma_mb_halo` fetch + one whole-MB
  `sixtap_mb_luma` §18.3 pass (RFC 6386 §18.1: all sixteen luma
  sub-blocks of a non-SPLITMV candidate share the one MV). Byte-exact
  with the per-sub-block tiling by the existing equivalence tests, so
  every candidate SAD, MV decision, and emitted bit is unchanged.
* Measured (interleaved extended-measurement A/B, Apple M4):
  `inter_encode_short_clip/inter_encode_4f_128x128_qi32` **−16.6 %**
  stable default (8.907 → 7.431 ms), **−18.3 %** nightly scalar (mean
  9.024 → 7.368 ms), **−21.5 %** nightly + `simd` (mean 8.859 →
  6.953 ms); the simd-vs-scalar whole-frame gap widens from −1.8 % to
  **−5.6 %** now that the search leg runs the batched (and on `simd`,
  vectorised) kernel. Full tables + attribution in `BENCHMARKS.md`
  round 279.
* Bit-identity proof: full stable suite (483 lib + integration) +
  nightly `simd` lib suite (485) green, plus a 54-frame
  `Vp8InterStreamEncoder` byte-hash A/B (3 resolutions × 6 frames ×
  3 quantisers, keyframe interval 3) — identical FNV-1a pre-/post-
  change on stable and under nightly + `simd`.

### Documentation — whole-frame keyframe path measured under nightly + `simd` (round 278, 2026-06-11)

Measurement-only depth round closing the standing BENCHMARKS candidate
"whole-frame `keyframe_encode` re-measure under nightly + `simd`". No
code change — the round publishes a trustworthy whole-frame number for
the `simd` feature on the keyframe path, replacing the round-247
sub-percent prediction (which predated the round-267 dequant and
round-268 TM_PRED kernels and under-counted the inlined inverse-DCT
win):

* `keyframe_encode/encode_keyframe_320x240_qi32`: nightly scalar mean
  5.456 ms → nightly + `simd` mean 4.957 ms (**−9.2 %**, 14.08 → 15.49
  Mpx/s).
* `keyframe_decode/decode_keyframe_320x240_qi32`: nightly scalar mean
  151.8 µs → nightly + `simd` mean 120.0 µs (**−21.0 %**, 506 → 640
  Mpx/s).
* Stable 1.95 default-features anchors (5.454 ms / 154.7 µs) agree with
  the nightly scalar column, so the delta is the `simd` dispatch, not
  the compiler version.

Methodology: 30 s measurement time (no `--quick`), three interleaved
scalar/simd run pairs (non-overlapping populations; consecutive
same-config change estimates ≤ 2 %), nightly toolchain on both A/B
columns, separate target dirs. `sample(1)` attribution: the scalar
encode profile's #2 self-time symbol `inverse_dct_4x4` (≈ 16 %, 24
calls per MB in the §11 RD-reconstruct loop) disappears from the
top-of-stack list under `simd`; `predict_y16x16_tm` drops 32 → 7
samples. Full table + analysis in `BENCHMARKS.md` § Round 278.

### Changed — compile-time SPLITMV partition-group table (round 277, 2026-06-11)

Profile-guided allocation elimination in the §16.4 SPLITMV encoder
path, closing the round-276 "remaining allocator churn" candidate. The
fresh inter-encode profile attributed essentially all of the residual
malloc/free samples to `encoder::partition_groups` (a per-call
`Vec<Vec<usize>>` rebuilt for every (macroblock, partition shape) the
SPLITMV scorer visits) plus the two per-candidate `Vec`s inside
`SplitMvCandidate`. Two changes, bit-identical encoder output:

* `partition_groups` now returns `&'static PartitionGroups` from a
  table built **at compile time** (`const fn`) from the §20.13
  `MV_PARTITIONS` constant — same single source of truth, zero runtime
  work, zero allocation. Group slices keep the raster (ascending)
  member order, so `group[0]` remains the §16.4 anchor.
* `SplitMvCandidate::{submv_modes, submv_new_diffs}` switch from
  per-candidate `Vec`s to fixed `[_; 16]` arrays indexed by group id
  (entries past the partition's group count stay at their fill value
  and are never read).

Measured on the `inter_encode_short_clip` bench (Apple M4, 30 s
measurement): 8.96 ms → 8.83 ms (−1.4 %); the sample(1) call-tree
allocator-family count drops 794 → 40 samples and the allocator
disappears from the ≥5-sample top-of-stack list entirely. Output is
bit-identical (full suite + an 18-frame 3-resolution A/B byte hash).

### Changed — MV-cost `log2` lookup table + allocation-free tree walks (round 276, 2026-06-11)

Profile-guided micro-optimisation of the encoder's RD-costing hot path,
closing the BENCHMARKS "remaining allocator churn" candidate's biggest
remaining offenders. Three changes, bit-identical encoder output:

* `motion_vector::mv_component_bits` now prices each §17.1 bool through
  the encoder's round-170 `-log2(p / 256)` lookup table instead of an
  inline libm `log2` per bit — the round-276 inter-encode profile's #6
  self-time symbol (411 samples) drops to zero. The table read returns
  the exact double the inline expression computed for every
  `(prob, value)` pair, locked by a new full-range regression test
  (`mv_component_bits_matches_reference_over_full_range`, `==` on f64
  across `-1023..=1023` × spec-default + clamp-corner contexts).
* `small_mv_bits` / `write_small_mv` replace the recursive
  `Vec`-allocating DFS with a direct §17.1 depth-3 descent: the
  `small_mvtree` listing's own comments give the leaf↔path
  correspondence (`0 = "000"` … `7 = "111"`), so the path bits are the
  leaf's 3-bit binary expansion MSB-first; the descent still walks
  `SMALL_MVTREE` for the node-halved probability offsets and
  debug-asserts the landing leaf.
* `encoder::treed_bits` and `BoolEncoder::write_treed` share a new
  `treed_find_path` helper that runs the same DFS into a stack
  `[bool; 16]` (deepest in-crate tree: `BMODE_TREE`, 9 internal nodes)
  instead of a per-call `Vec<bool>`.

Measured (criterion `--quick`, Apple M4 / aarch64, stable):
`inter_encode_4f_128x128_qi32` 10.32 → 9.38 ms (**−9.1 %**),
`encode_keyframe_320x240_qi32` 5.83 → 5.53 ms (**−5.2 %**). The
post-change profile confirms libm `log2` gone, the `mv_component_bits`
family 305 → 123 self-samples, and the malloc/free family ~540 → ~300.
See BENCHMARKS.md "Round 276" for the full A/B + profile evidence.
Lib tests 481 → 482. No change to encode/decode output bytes.

### Added — `filter_block_4x4_into` strided-write primitive + round-274 SPLITMV write-strategy A/B (round 274, 2026-06-11)

Round 274 closes the long-standing BENCHMARKS next-round candidate "SPLITMV
whole-pixel sub-block batching" — with a **measured negative result** that
keeps the production path unchanged and documents why.

SPLITMV macroblocks (RFC 6386 §16.4) carry sixteen distinct luma motion
vectors (plus four chroma), so the MB-scale shared-halo batch landed in
rounds 270–272 cannot apply — each sub-block must be synthesised
independently. The only freedom left is *how* each per-sub-block result
lands in the macroblock raster:

* **`filter_block_4x4_into`** (new public fn) — the strided-write companion
  of `filter_block_4x4`: synthesises one 4×4 sub-block and writes it
  directly into a destination raster at `(dst_x, dst_y)` / `dst_stride`,
  with no intermediate `[u8; 16]`. The whole-pixel branch copies source rows
  straight into the destination (§18.3 "simply copied" / §20.14
  `build_mc_border` edge replication on the border-straddle path); the
  sub-pixel branch delegates the pixel computation to `filter_block_4x4`
  verbatim (so the `sixtap_2d` SIMD dispatch and its byte-exactness proof
  carry) and writes the result strided.

* **The shipped `predict_split_mv` path stays scratch-then-copy.** A new
  `motion_comp_subpel_luma/splitmv_predict_*` criterion bench measures the
  two write strategies, which produce byte-identical output: the
  scratch-copy form (`filter_block_4x4` → `[u8; 16]` → four contiguous 4-byte
  row copies) runs ~17 % FASTER than the strided-write form (398.8 ns vs
  480.5 ns, Apple M4 / aarch64, criterion `--quick`). The contiguous block
  lets the compiler vectorise the per-row writes, where scattered strided
  writes into a stride-16 raster cannot. `predict_split_mv` therefore keeps
  the scratch path; the round-270/271/272 MB-batch candidate list's SPLITMV
  entry is closed negative.

`filter_block_4x4_into` is retained as a public primitive (useful for
callers that already own a destination raster) with two equivalence
guards: `filter_block_4x4_into_matches_filter_block_4x4` (strided write ==
`filter_block_4x4` across whole-pixel / sub-pixel / corner-clamp cases) and
`strided_into_assembly_matches_predict_split_mv` (a full strided-write MB
assembly == the shipped `predict_split_mv`, sixteen distinct vectors, both
`full_pixel` polarities, in-bounds + corner MB). Lib tests 479 → 481. No
change to encode / decode output bytes.

### Added — `panic_free_mb_batch_motion_comp` fuzz target (round 273, 2026-06-10)

Round 273 adds the fifteenth `cargo-fuzz` target, closing a coverage gap
opened by the rounds 270–272 MB-batching work: the six public MB-scale
§18.3 / §20.14 batched motion-compensation primitives
(`fetch_luma_mb_halo` + `sixtap_mb_luma`, `fetch_chroma_mb_halo` +
`sixtap_mb_chroma`, `fetch_luma_mb_whole_pixel` /
`fetch_chroma_mb_whole_pixel`) landed *after* the round-257
`panic_free_sixtap_subpel` target was written, so no existing harness
reached them. The fourteen targets above hit §18 only through the §17
motion-search descent ladder (which snaps every per-candidate MV to a
sub-block grid and never reaches the MB-scale orchestrator) or through
`decode_vp8` / `Vp8DecoderState::decode_frame` /
`encode_p_frame_multi_ref` (which gate the MB-scale fetch behind a
fully-formed reference picture + §9.7 refresh state machine, so the
§20.14 `build_mc_border` clamp inside the MB-halo fetch never sees an
origin parked across a picture boundary by an arbitrary `i16` vector).

`panic_free_mb_batch_motion_comp` drives the MB-scale surface directly
with an attacker-shaped `(plane dimension, MB origin, MV, fractional
offset, filter set, border-position class)` envelope — every border
class (mid-plane fast path, top-left corner, bottom-right corner,
adversarial full-`i16` MV) and every `(mx, my) ∈ {0..7}²` fraction
across both §18.3 filter sets. Three equivalence cross-checks are
asserted on every iteration (panic on mismatch): the 21×21 luma halo and
the 13×13 chroma halo must each contain every per-sub-block 9×9
`fetch_block_halo` window at offset `(sb*4, sc*4)`, and the whole-pixel
MB luma / chroma copy must equal the per-sub-block
`fetch_block_whole_pixel` assembly tiled into the MB raster — the
round-270 / 271 / 272 in-tree containment invariants, now re-asserted
under the attacker-shaped border-clamp envelope the mid-plane-only
in-tree tests never reach.

26-second smoke pass landed `cov: 355, ft: 569, corp: 65/1063b` across
810 049 iterations from an empty seed on aarch64-apple-darwin at
~31 155 exec/s, peak RSS 495 MiB, zero panics. No library / public-API
change; fuzz-crate only.

### Added — whole-pixel non-SPLITMV MB batching (`fetch_luma_mb_whole_pixel` / `fetch_chroma_mb_whole_pixel`) (round 272, 2026-06-10)

Round 272 closes the round-271 BENCHMARKS next-round candidate
"whole-pixel non-SPLITMV MB batching" — the whole-pixel analogue of the
round-270 / round-271 sub-pixel MB-batching work. When the shared §18.1
motion vector of a non-SPLITMV inter macroblock is *whole-pixel*
(`mv & 7 == 0` per component), the §18.3 prediction is "simply copied"
(no convolution), so the whole 16×16 luma / 8×8 chroma block is one
contiguous source region rather than sixteen / four overlapping 4×4 ones:

* **`fetch_luma_mb_whole_pixel`** (new public fn) — the MB-scale analogue
  of `fetch_block_whole_pixel`: fetches the whole 16×16 luma block in one
  pass at integer offset `(mb_x, mb_y) + (mv >> 3)`, replicating any
  out-of-plane read at the nearest edge pixel (§20.14 `build_mc_border`),
  with the same in-bounds contiguous-row fast path / per-pixel clamp
  fallback split as the per-sub-block fetch. Row-major, stride 16,
  matching the `ReconstructedMb::y` layout.
* **`fetch_chroma_mb_whole_pixel`** (new public fn) — the chroma analogue:
  fetches the whole 8×8 chroma block in one pass, row-major stride 8.
* **`predict_inter_mb` whole-pixel branches rewired** — the luma branch
  now issues one `fetch_luma_mb_whole_pixel` instead of sixteen
  `fetch_block_whole_pixel` copies; each chroma plane issues one
  `fetch_chroma_mb_whole_pixel` instead of four. Byte-identical output
  (both read the same contiguous source region under the shared §18.1
  vector); the gain is pure gather amortisation — one bounds check and one
  border-straddle decision per MB instead of per sub-block.

Five new equivalence / clamp tests anchor the batched fetch against the
per-sub-block `fetch_block_whole_pixel` assembly: in-bounds luma + chroma
byte-equality, top-left luma + bottom-right chroma border-clamp, and a
real corner-MB prediction through `predict_inter_mb`
(`predict_inter_mb_whole_pixel_at_border_uses_mb_batch_clamp`, covering Y
+ U + V). The existing `reconstruct_inter_mb_matches_legacy_for_whole_pixel`
test continues to anchor the full reconstruct path. Lib test count
474 → 479.

New `motion_comp_subpel_luma/mb_*_whole_pixel_*` criterion benches measure
the batched fetch against the per-sub-block assembly on a 64×64
deterministic source (Apple M4 / aarch64, criterion `--quick`):

| Bench | Per-sub-block | Batched | Delta |
|---|---:|---:|---:|
| whole 16×16 luma copy | 46.89 ns | **13.13 ns** | **−72 %** |
| whole 8×8 chroma copy | 8.49 ns | **4.74 ns** | **−44 %** |

This is the path `predict_inter_mb` takes for every whole-pixel
non-SPLITMV inter MB (the common case for low-motion content where the
§17 search snaps to the integer grid). No new decode / encode feature
coverage; clean-room from RFC 6386 §18.2 / §20.14.

### Added — MB-scale §18.3 chroma batching (`sixtap_mb_chroma`) (round 271, 2026-06-10)

Round 271 closes the round-270 BENCHMARKS next-round candidate "MB-scale
§18.3 chroma batching". On a non-SPLITMV inter macroblock the four chroma
sub-blocks of each plane share one §18.1 averaged motion vector
([`chroma_mv`]), so the six-tap support of the whole 8×8 chroma block is
one contiguous region rather than four overlapping ones — the chroma
analogue of the round-270 16×16-luma path:

* **`fetch_chroma_mb_halo`** (new public fn) — the chroma analogue of
  `fetch_luma_mb_halo`: fetches one `(8+5)×(8+5) = 13×13` edge-replicated
  halo (§20.14 `build_mc_border`) for the whole 8×8 chroma block, block
  origin at `halo[(2, 2)]`, with the same in-bounds fast-path / clamp
  fallback split.
* **`sixtap_mb_chroma`** (new public fn) — the MB-scale `sixtap_2d` over an
  8×8 chroma block: horizontal pass of 13 rows × 8 cols then vertical pass
  of 8 rows × 8 cols over the 13×13 halo, producing the 8×8 chroma block in
  one two-pass convolution. Byte-identical to applying `sixtap_2d` to each
  of the four 4×4 sub-blocks (the §18.3 `interp` dot product per output
  sample is independent of how the support is tiled, and the
  horizontal-pass intermediate is clamped identically).
* **`sixtap_mb_chroma` dispatcher** — scalar (`sixtap_mb_chroma_scalar`,
  default on stable / nightly-without-`simd`) vs SIMD
  (`sixtap_mb_chroma_simd`, `#[cfg(feature = "simd")]`). The SIMD path
  widens each pass to `Simd<i32, 8>`: one eight-lane vector per output row
  (tap k's eight lanes are the contiguous source run
  `halo[r*13 + k ..][..8]`), six widen-multiply-accumulates per row in
  place of 48 scalar MACs, with the clamped horizontal intermediate
  resident in `i32` vectors so the vertical pass runs with zero loads.
  Lane type is `i32` for the same §18.3 overflow reason as `sixtap_2d`
  (`[-8160, 40800]` dot-product span past `i16::MAX`).

`predict_inter_mb`'s chroma loop now routes a sub-pixel `uvmv` (U and V
planes) through the batched path and keeps the per-sub-block whole-pixel
copy fast path for a whole-pixel vector; the SPLITMV chroma path is
unchanged (its four chroma sub-blocks carry distinct §18.1-averaged vectors
by construction, so the shared-vector MB halo doesn't apply).

Six new tests (stable lib 468 → 474, nightly + `simd` lib 470 → 476):

* `sixtap_mb_chroma_matches_per_subblock_path` — the whole-MB synthesis is
  byte-exact against four separate `sixtap_2d` calls over the corresponding
  9×9 sub-halos carved from the 13×13 MB halo, for every `(mx, my)` and
  both §18.3 filter sets.
* `sixtap_mb_chroma_simd_matches_scalar_on_stress_inputs` — dispatcher vs
  `sixtap_mb_chroma_scalar` over flat extremes, opposing ramps,
  alternating-extreme checker, and a deterministic LCG set × all 64
  `(mx, my)` × both filter sets (the primary SIMD safety net on
  nightly + `simd`).
* `fetch_chroma_mb_halo_matches_subblock_halos_in_bounds` /
  `fetch_chroma_mb_halo_clamps_at_top_left_corner` — the MB halo contains
  every per-sub-block 9×9 halo as a window in-bounds, and replicates the
  nearest edge pixel exactly like `build_mc_border` at the corner.
* `predict_inter_mb_chroma_sub_pixel_matches_per_subblock_path` /
  `predict_inter_mb_chroma_sub_pixel_at_border_uses_mb_halo_clamp` — a real
  mid-plane MB and a corner MB (0,0) sub-pixel chroma prediction through the
  batched path (and its border-clamp fallback) still equal the
  per-sub-block `filter_block_4x4` path on both U and V.

New `motion_comp_subpel_luma` bench points
`mb_chroma_batched_8x8` / `mb_chroma_per_subblock_8x8`
(`aarch64-apple-darwin`, criterion `--quick`, nightly toolchain for both
columns): scalar batched 43.9 ns, SIMD batched 38.6 ns, per-sub-block
partner ≈ 67.7 ns — **−43 %** end-to-end (SIMD batched vs per-sub-block)
and **−12 %** SIMD-over-scalar on the batched path. The bulk of the win is
the amortised single 13×13 fetch + tighter loop (the scalar batched path
alone is −35 % vs per-sub-block); the wider `i32×8` lanes add the further
−12 %. Verified passing on nightly 1.97 with `simd` and on stable.

### Added — MB-scale §18.3 luma batching (`sixtap_mb_luma`) (round 270, 2026-06-10)

Round 270 lands the round-269 BENCHMARKS next-round candidate "MB-scale
§18.3 batching". All sixteen luma sub-blocks of a non-SPLITMV inter
macroblock share one motion vector (§18.1), so the six-tap support of
the whole 16×16 luma block is one contiguous region rather than sixteen
overlapping ones. `predict_inter_mb`'s sub-pixel luma path now exploits
that:

* **`fetch_luma_mb_halo`** (new public fn) — the MB-scale analogue of
  `fetch_block_halo`: fetches one `(16+5)×(16+5) = 21×21` edge-replicated
  halo (§20.14 `build_mc_border`) for the whole 16×16 luma block, block
  origin at `halo[(2, 2)]`, with the same in-bounds fast-path / clamp
  fallback split as the per-sub-block fetch.
* **`sixtap_mb_luma`** (new public fn) — the MB-scale `sixtap_2d`:
  horizontal pass of 21 rows × 16 cols then vertical pass of 16 rows ×
  16 cols over the 21×21 halo, producing the 16×16 luma block in one
  two-pass convolution. Byte-identical to applying `sixtap_2d` to each of
  the sixteen 4×4 sub-blocks (the §18.3 `interp` dot product per output
  sample is independent of how the support is tiled, and the
  horizontal-pass intermediate is clamped identically).
* **`sixtap_mb_luma` dispatcher** — scalar (`sixtap_mb_luma_scalar`,
  default on stable / nightly-without-`simd`) vs SIMD
  (`sixtap_mb_luma_simd`, `#[cfg(feature = "simd")]`). The SIMD path
  widens each pass to `Simd<i32, 16>`: one sixteen-lane vector per output
  row (tap k's sixteen lanes are the contiguous source run
  `halo[r*21 + k ..][..16]`), six widen-multiply-accumulates per row in
  place of 96 scalar MACs, with the clamped horizontal intermediate
  resident in `i32` vectors so the vertical pass runs with zero loads.
  Lane type is `i32` for the same §18.3 overflow reason as `sixtap_2d`
  (`[-8160, 40800]` dot-product span past `i16::MAX`).

`predict_inter_mb`'s luma loop now routes a sub-pixel `ymv` through the
batched path and keeps the per-sub-block whole-pixel copy fast path for
a whole-pixel vector; the chroma path is unchanged (the four chroma
sub-blocks span only 8×8 and gain little from MB-scale batching, and the
two §18.1 averaged-vector / SPLITMV cases keep their per-sub-block
dispatch).

Five new tests (stable lib 463 → 468, nightly + `simd` lib 465 → 470):

* `sixtap_mb_luma_matches_per_subblock_path` — the whole-MB synthesis is
  byte-exact against sixteen separate `sixtap_2d` calls over the
  corresponding 9×9 sub-halos carved from the 21×21 MB halo, for every
  `(mx, my)` fraction pair and both §18.3 filter sets.
* `sixtap_mb_luma_simd_matches_scalar_on_stress_inputs` — dispatcher vs
  `sixtap_mb_luma_scalar` over flat extremes, opposing ramps,
  alternating-extreme checker, and a deterministic LCG set × all 64
  `(mx, my)` × both filter sets (the primary SIMD safety net on
  nightly + `simd`).
* `fetch_luma_mb_halo_matches_subblock_halos_in_bounds` /
  `fetch_luma_mb_halo_clamps_at_top_left_corner` — the MB halo contains
  every per-sub-block 9×9 halo as a window in-bounds, and replicates the
  nearest edge pixel exactly like `build_mc_border` at the corner.
* `predict_inter_mb_sub_pixel_at_border_uses_mb_halo_clamp` — a real
  corner-MB (0,0) sub-pixel prediction through the MB-halo border-clamp
  fallback still equals the per-sub-block `filter_block_4x4` path.

New `motion_comp_subpel_luma` bench points
`mb_luma_batched_16x16` / `mb_luma_per_subblock_16x16`
(`aarch64-apple-darwin`, criterion `--quick`): scalar batched 158.8 ns,
SIMD batched 140.2 ns, per-sub-block partner ≈ 260–268 ns — **−47 %**
end-to-end (SIMD batched vs per-sub-block) and **−12 %** SIMD-over-scalar
on the batched path. The bulk of the win is the amortised single 21×21
fetch + tighter loop (the scalar batched path alone is −41 % vs
per-sub-block); the wider `i32×16` lanes add the further −12 %. Verified
passing on nightly 1.97 with `simd` and on stable.

### Added — §18.3 six-tap sub-pixel SIMD kernel (round 269, 2026-06-10)

Round 269 is a depth-mode SIMD round extending the nightly-only `simd`
feature onto the §18.3 / §20.14 six-tap sub-pixel interpolation kernel
`sixtap_2d` — the round-170 profile's #4 self-time symbol on the inter
encode and the candidate-list "next SIMD target" since then.

`src/motion_comp.rs` `sixtap_2d` is now a dispatcher:

* **scalar** (`sixtap_2d_scalar`) — the §20.14 two-pass
  `sixtap_horiz` / `sixtap_vert` composition, unchanged, the default
  on stable and on nightly without `simd`.
* **SIMD** (`sixtap_2d_simd`, `#[cfg(feature = "simd")]`) — each
  convolution row's four §18.3 `interp` dot products become one
  `Simd<i32, 4>` vector (tap k's four support lanes are the contiguous
  run `halo[r*9 + k ..][..4]`), so the six taps are six widen-multiply-
  accumulates per row in place of 24 scalar MACs. The horizontal
  pass's clamped intermediate stays resident in `i32` vectors (every
  lane already in `0..=255` after the lane-wise clamp, so the vertical
  pass reads the exact sample values the scalar listing's 8-bit `temp`
  buffer would hold, without a u8 round trip).

Lane-type note: the round-170 candidate list suggested a
`Simd<i16, 8>` stripe, but the §18.3 dot product over `u8` support
spans `[-32·255, 160·255] = [-8160, 40800]` (the ½-displacement row
`{3, -16, 77, 77, -16, 3}` has positive-tap sum 160) — past
`i16::MAX`, so a single `i16` accumulator wraps. A parity-split
two-accumulator `i16×8` two-row-stripe variant was implemented and
measured during the round (every tap-parity class partial sum fits
`i16`) but benched no better than scalar on the MB-scale workload and
~+15 % worse on `filter_block_4x4`, so the four-lane `i32` form —
which matches the 4×4 sub-block geometry exactly — is the one that
shipped.

Two new tests (stable lib 461 → 463, nightly + `simd` lib 463 → 465):

* `sixtap_2d_simd_matches_scalar_on_stress_inputs` — dispatcher vs
  scalar over 13 halos (all-floor, all-ceiling, opposing ramps,
  alternating-extreme checker, 8 deterministic LCG halos) × all 64
  `(mx, my)` eighth-pixel fraction pairs × both §18.3 filter sets.
* `sixtap_2d_accumulator_extremes_match_scalar` — drives the §18.3
  dot product to both extremes (+40800 → clamp ceiling, −8160 → clamp
  floor) through output column 0 under the ½-displacement taps,
  pinning the exact overflow region that forces the `i32` lanes.

Measured on the round-170 `motion_comp_subpel_luma` bench
(`aarch64-apple-darwin`, criterion `--quick`, triple-run vs the same
nightly scalar baseline): `mb_sixtap_2d_16x4x4` 271.5 → 248.5 ns
(**−8.5 %**), `filter_block_4x4_sub3x5` 24.87 → 23.55 ns (**−5.3 %**).
The modest margin (vs TM_PRED's −87.7 %) is expected: the scalar
`interp` loop over fixed-size arrays was already auto-vectorising
well, so the explicit kernel's win comes mostly from keeping the
intermediate in vector registers rather than from new parallelism.

### Added — §12.2 TM_PRED intra SIMD kernel (round 268, 2026-06-10)

Round 268 is a depth-mode SIMD round extending the nightly-only `simd`
feature onto the §12.2 TM_PRED intra-prediction kernel — the only
§12.2 mode with per-pixel arithmetic (`X_{rc} = clamp255(L_r + A_c -
P)` over all 256 luma / 64 chroma cells; DC / V / H are fills and row
copies the compiler already vectorises).

`src/intra_predict.rs` `predict_tm` is now a dispatcher:

* **scalar** (`predict_tm_scalar`) — the longhand §12.2 double loop,
  unchanged, the default on stable and on nightly without `simd`, and
  the fallback for any non-§12.2 block width.
* **SIMD** (`predict_tm_simd::<N>`, `#[cfg(feature = "simd")]`, N = 16
  luma / 8 chroma) — forms the row-invariant column term `A_c - P`
  once as a `Simd<i16, N>` vector, then per row adds a splat of `L_r`,
  clamps every lane into `0..=255` with `simd_clamp`, and narrows back
  to `u8`. The `i16` working type reproduces the scalar `i32`
  arithmetic exactly (every intermediate lies in `-255..=510`), and
  the post-clamp `cast::<u8>()` of a value already in `0..=255` equals
  the scalar `as u8`, so the byte-equivalence is unconditional.

Two new tests: `predict_tm_simd_matches_scalar_on_stress_inputs`
asserts the dispatcher is byte-exact against the scalar fallback at
both widths across the clamp endpoints (floor `-255`, ceiling `510`),
flat mid-range, opposing ramps, alternating extremes, and 16
deterministic LCG triples per width; and
`predict_tm_public_entry_points_route_through_dispatcher` pins the
public `predict_y16x16_tm` / `predict_uv8x8_tm` bytes to the scalar
listing on a clamp-straddling input. Both run on stable
(scalar-vs-scalar, harmless) and are the primary safety net on
nightly + `simd`. Verified passing on nightly 1.97 with `simd`.

A `predict_y16x16_tm` entry joins the round-258 `intra_predict_dc16`
criterion bench as the A/B anchor: 44.15 ns scalar → 5.46 ns SIMD
(**−87.7 %**) on `aarch64-apple-darwin` (criterion `--quick`) — the
largest per-kernel SIMD delta in the crate so far (the 16 rows
collapse from 256 scalar clamp chains to 16 vector ops on a hoisted
column term).

Note for `simd` builders: current nightlies' `core::simd` dropped the
`LaneCount<N>: SupportedLaneCount` bound (`Simd<T, N>` is now generic
over any `const N: usize`), so `predict_tm_simd` carries no
lane-count where-clause.

### Added — §14.1 dequantize SIMD primitive (round 267, 2026-06-10)

Round 267 is a depth-mode SIMD round extending the nightly-only `simd`
feature from the §14.3 / §14.4 transforms onto the §14.1 dequantize
hot path. RFC 6386 §14.1 (page 76) multiplies every decoded coefficient
of a 4×4 block by one of two factors — the DC factor for coefficient 0,
the AC factor for coefficients 1..=15 — with each product formed in
`i32` and stored back as `i16`. The sixteen multiplies are fully
independent (no cross-lane dependency), so they map onto a single
16-wide vector.

`src/dequant.rs` `dequant_block` is now a dispatcher:

* **scalar** (`dequant_block_scalar`) — the longhand multiply loop,
  unchanged, the default on stable and on nightly without `simd`.
* **SIMD** (`dequant_block_simd`, `#[cfg(feature = "simd")]`) — widens
  the `i16` block to `Simd<i32, 16>` (sign-extending `cast`), multiplies
  lane-wise against a per-lane factor vector (`dc_factor` in lane 0,
  `ac_factor` in lanes 1..=15), and truncates each product back to `i16`
  with `cast::<i16>()`. The int→int `cast` truncates exactly like the
  scalar `as i16`, including the i16-overflow wrap-around.

A new `dequant_block_simd_matches_scalar_on_stress_inputs` test asserts
the dispatcher is byte-exact against the scalar fallback across all-zero,
DC-only, single-AC-per-lane, mixed-sign, and i16-overflow stress blocks
(`[i16::MAX; 16] × 440`, `[i16::MIN; 16] × 440`, and a 16-lane
near-extreme pattern) — the overflow fixtures are what distinguish
truncating `cast` from a saturating one. The test runs on stable
(scalar-vs-scalar, harmless) and is the primary safety net on nightly +
`simd` (SIMD-vs-scalar). Verified passing on nightly 1.97 with `simd`.

### Added — `panic_free_inter_mb_reconstruct` fuzz target (round 265, 2026-06-09)

Round 265 is the depth-mode fuzz round on the §16 inter-MB
reconstruction surface (RFC 6386 §16 / §16.2 / §16.4 / §18 / §18.1 /
§18.3). The thirteen existing fuzz targets reach §16 only indirectly:
the decode-side targets gate the inter-MB path behind a fully-formed
previous keyframe + §9.7 reference refresh state machine; the
encode-side targets reach `reconstruct_inter_mb` through
`encode_p_frame_multi_ref` / `Vp8TwoPassEncoder::encode_frame` and
feed the §14 residue blocks with §9.6-clamped quantiser tables; the
round-256 `panic_free_motion_search_descent` and round-257
`panic_free_sixtap_subpel` targets reach the §18.3 primitive layer
but never the §16 macroblock-level reconstruction orchestrator; the
round-262 `panic_free_transform_4x4_roundtrip` target reaches §14
but never feeds the residue into a §16 reconstruct call.

The new `fuzz/fuzz_targets/panic_free_inter_mb_reconstruct.rs` drives
the §16 macroblock-level orchestrator directly with attacker-shaped
`(mb_col, mb_row, luma_mv, full_pixel, mb_skip_coeff, y2_coeffs,
y_coeffs, u_coeffs, v_coeffs)` tuples on every iteration. A 2-bit
path discriminator selects between the three reconstruction entry
points, and each path also drives its `predict_*` residue-free
counterpart for two cross-checks:

1. **§18.1 fractional gate ↔ `SubPixelNotSupported`.** On the §16.2
   whole-pixel path, the harness recomputes the §18.1 stored-luma
   doubling + chroma-average + (optional) full-pel truncation gate
   itself; the dispatcher's `MotionCompError::SubPixelNotSupported`
   return must agree with that gate on every input.

2. **§11.1 skip short-circuit.** On every path, setting
   `mb_skip_coeff = true` collapses the reconstruct call into the
   prediction (no residue is added); the harness asserts
   `reconstruct == predict` byte-equal on every skip-enabled input,
   across all three §16 paths.

Surface covered:

* `reconstruct_inter_mb_whole_pixel` + `predict_inter_mb_whole_pixel`
  — §16.2 non-SPLITMV whole-pixel path.
* `reconstruct_inter_mb` + `predict_inter_mb` — §16.2 / §18.3 full
  sub-pixel path (both filter sets reachable via
  `filter_set_for_version`).
* `reconstruct_split_mv_mb` + `predict_split_mv` — §16.4 SPLITMV
  path with sixteen per-luma-sub-block vectors derived from the
  payload.
* `select_ref_frame` — §16.2 reference-frame discriminator (every
  `RefFrame` variant reachable from a short attacker partition).
* §18.1 vector-adjustment primitives `stored_luma_mv`, `chroma_mv`,
  `apply_full_pixel`, `whole_pixel_fraction_is_zero`,
  `chroma_idx_for_luma_subblock`, `split_chroma_mvs`,
  `filter_set_for_version` — driven directly on every iteration so
  the dispatch table is exercised on every input.

Smoke pass: 25-second `cargo +nightly fuzz run
panic_free_inter_mb_reconstruct -j 2 -- -max_total_time=25
-rss_limit_mb=2048` from an empty seed on aarch64-apple-darwin
landed `cov: 437, ft: 1005, corp: 105` across **2 375 327
iterations** at ~43 700 exec/s, zero panics. Peak RSS bounded by
the 9-MB plane cap (`mb_cols ≤ 3`, `mb_rows ≤ 3` ⇒ ≤ 9 × 384 B
per-MB array = 3 456 B per `ReferencePlanes` allocation).

### Added — `panic_free_loop_filter_writeback` fuzz target (round 263, 2026-06-09)

Round 263 is the depth-mode fuzz round on the §9.4 / §19.2 loop-
filter parameter writeback layer of the public encoder PLUS the small
§9.5 / §9.6 / §9.10 / §9.11 sibling writers reached by the same §19.2
frame-header walk. The twelve existing fuzz targets reach the encoder
writeback layer only indirectly through `encode_keyframe` /
`Vp8TwoPassEncoder::encode_frame`, which feed NORMALISED parameter
bytes that the upstream `KeyframeParams` builder clamped against the
§9.4 / §9.5 / §9.6 fields; the round-261 `panic_free_encode_keyframe`
reaches `write_loop_filter` via the keyframe encoder which always
writes `adj_enable = 0`, never `write_loop_filter_with_deltas` and
never `LoopFilterDeltas::validate` / `::effective`. The round-232
`panic_free_loopfilter_segment` target reaches §15 only through the
per-segment kernel layer — the kernels consume the §15.4 derived
(`hev_threshold`, `interior_limit`, `edge_limit`) triple, not the
§9.4 wire form.

The new `fuzz/fuzz_targets/panic_free_loop_filter_writeback.rs` drives
the §9.4 / §19.2 wire-format writer surface directly:

1. **§9.4 baseline writer.** `write_loop_filter(enc, filter_type,
   level, sharp, adj_enable=false)` — covers the `filter_type` /
   `level` / `sharp` rejection cliffs (`level > 63` /
   [`EncodeError::LoopFilterLevelOutOfRange`]; `sharp > 7` /
   [`EncodeError::SharpnessLevelOutOfRange`]). Passes `adj_enable=false`
   unconditionally to honour the Phase-1 `debug_assert`.

2. **§9.4 + §19.2 full writer.** `write_loop_filter_with_deltas(enc,
   filter_type, level, sharp, &deltas)` — covers every `(enabled,
   update, per-slot Some|None)` ladder branch of the
   `mb_lf_adjustments()` block. The four per-reference + four
   per-mode slots are seeded from attacker bytes; the presence-nibble
   selects which slots carry `Some(v)`.

3. **§9.4 validate().** `LoopFilterDeltas::validate()` — covers the
   `|v| > 63` cliff on each of the eight per-slot magnitudes
   ([`EncodeError::LoopFilterDeltaOutOfRange`]).

4. **§15.4 / §20.6 effective().** `LoopFilterDeltas::effective(
   carried_ref, carried_mode)` — covers every `(enabled, update,
   per-slot Some|None)` × (carried `[i16; 4]` × 2) cross-product;
   cross-checked against a hand-rolled §20.6 oracle on every input.

5. **§9.6 quant indices.** `write_quant_indices(enc, y_ac_qi, …)` —
   covers the `y_ac_qi > 127` cliff
   ([`EncodeError::QuantIndexOutOfRange`]). The five per-`Option<i8>`
   delta slots are pre-clamped to `-15..=15` (the §9.6 `L(4) + L(1)`
   field envelope; the writer's documented contract panics on `|v| >=
   16`, analogous to `add_residue`'s length-mismatch panic in the
   round-262 target).

6. **§9.5 token partition count.** `write_token_partition_count(enc,
   count)` — covers the `count ∉ {1, 2, 4, 8}` cliff
   ([`EncodeError::InvalidDctPartitionCount`]); every other byte
   triggers the rejection branch.

7. **§9.10 / §9.11 mb-skip-coeff.** `write_mb_no_skip_coeff(enc,
   enabled, prob_skip_false)` — covers the gated literal arm.

A round-trip leg feeds the `write_loop_filter_with_deltas` output
back into [`BoolDecoder::init`] and walks the §19.2 field schedule
(`filter_type` at prob 128, `L(6)` level, `L(3)` sharp, `L(1)`
adj_enable, gated `L(1)` update + 4 ref + 4 mode `(present, L(6)
magnitude, L(1) sign)` slot triples), asserting every read value
equals what the encoder wrote. Any asymmetry between
`write_loop_filter_with_deltas` and the §19.2 wire layout (field-order
swap, stray bit, sign-vs-magnitude order transposition) surfaces as a
`panic!` from the harness' equality assertion — the same shape the
round-261 `panic_free_bool_codec` target locked at the primitive
layer, now tight against the structured §9.4 field schedule.

**Smoke results.** 26-second smoke run on aarch64-apple-darwin
landed `cov: 406, ft: 632, corp: 80/2295b` across 5 235 001 iterations
from an empty seed at 201 346 exec/s, zero panics.

**Backfill.** The same round backfills the row `panic_free_token_block`
(round 237) and `panic_free_bool_codec` (round 261) into
`fuzz/README.md`'s targets table — both rows were missing since their
respective rounds added the targets to `fuzz/Cargo.toml` without
updating the README table.

Round 263 marks the **thirteenth** fuzz target (`fuzz_targets/`
files: 13). Stable lib tests: 458 (unchanged); nightly `+ simd`: 460
(unchanged).

Touched files in round 263:

* `fuzz/Cargo.toml` — added the `[[bin]]` block for
  `panic_free_loop_filter_writeback` and extended the metadata
  comment that lists each target's role.
* `fuzz/fuzz_targets/panic_free_loop_filter_writeback.rs` — new file
  (≈350 lines).
* `fuzz/README.md` — backfilled `panic_free_token_block` (round 237)
  and `panic_free_bool_codec` (round 261) rows; added the new
  `panic_free_loop_filter_writeback` row + caps + run / corpus
  entries.
* `README.md` — bumped target count 12 → 13; added the new bullet
  with round-263 smoke results.
* `CHANGELOG.md` — this entry.

### Added — `panic_free_transform_4x4_roundtrip` fuzz target (round 262, 2026-06-09)

Round 262 is the depth-mode fuzz round on the §14 transform / dequant
/ residue-summation primitive layer — the §14.3 forward + inverse
WHT, §14.4 forward + inverse DCT, §14.1 `dequant_block`, §14.5
`add_residue` / `add_residue_4x4`, §14.3 `inverse_wht_4x4_dc_only`
fast path, §20.16 `raster_to_scan` permutation, and the §9.6
`clamp_qindex` / §14.5 `clamp255` saturating caps. The eleven
existing fuzz targets reach §14 only indirectly: the four decode-
side targets (`panic_free_decode_keyframe` / `_decoder_state` /
`parse_headers` / `panic_free_token_block`) feed bytes through
`decode_vp8` / `Vp8DecoderState::decode_frame` / `Vp8FrameHeader::
parse` / `decode_block`, each of which gates §14 behind a fully-
formed §9 / §11 / §13 / §14.1 state machine — the inverse path is
exercised only against well-formed dequantised residuals and the
forward path is never exercised. The two encode-side targets
(`panic_free_encode_keyframe` / `_two_pass_stream`) run a §11 mode
pick → §14 forward transform → §13 token emission chain — the
forward DCT / WHT are exercised but only against §9.6-clamped
residual magnitudes determined by the upper-layer encoder logic.
The remaining five harnesses (`panic_free_loopfilter_segment` /
`panic_free_motion_search_descent` / `panic_free_sixtap_subpel` /
`panic_free_intra_predict_kernels` / `panic_free_bool_codec`) don't
touch §14 at all.

The new `fuzz/fuzz_targets/panic_free_transform_4x4_roundtrip.rs`
drives the §14 primitive surface directly across four legs:

1. **§14.3 WHT round-trip.** Forward `forward_wht_4x4` on an
   attacker-shaped `[i16; 16]` residual seed (mid-magnitude / ±255 /
   ±1023 §14.2 cliff — the documented §14.4 inverse-DCT envelope,
   chosen so the intermediate `i32` butterfly multiplies by
   `SINPI8_SQRT2 = 35468` stay inside `i32`) followed by
   `inverse_wht_4x4` on the result. Panic-free for every
   `(residual_seed_mode, [i16; 16])` combination.
2. **§14.4 DCT round-trip.** Same shape with `forward_dct_4x4`
   followed by `inverse_dct_4x4`.
3. **§14.1 dequant + §14.5 residue-sum leg.** `dequant_block` with
   attacker-chosen `(dc_factor, ac_factor)` cliff values
   (`i16::MIN` / `i16::MAX` / `0` plus the §14.1 4..=255 envelope)
   on a fresh copy of the residual — the §14.1 contract's `i32`
   product wrapping cast back to `i16` is panic-free on every cliff
   triple. The §14.5 `add_residue_4x4` and `add_residue` (arbitrary-
   length form, byte-equality assertion against the fixed-size form
   on equal-length inputs) are exercised against the §14.2-bounded
   residual + a constant predictor.
4. **§14.3 DC-only fast-path + §20.16 zigzag + §9.6 / §14.5 cap
   primitives.** `inverse_wht_4x4_dc_only(dc, out)` is asserted
   byte-equal to `inverse_wht_4x4([dc, 0, …, 0], out)` for every
   `dc ∈ [i16::MIN, i16::MAX]` (the §14.3 fast-path equivalence
   contract). `raster_to_scan(residual)` is asserted to be a
   permutation via multiset equality between input and output.
   `clamp_qindex` is exercised at the `i32::MIN` / `i32::MAX` cliff
   endpoints with a panic-free `idx < QINDEX_RANGE` assertion, and
   `clamp255` at the same cliff endpoints.

Each leg is independently flag-gated so libFuzzer can isolate
coverage to the per-leg primitive surface. The harness is
allocation-free — every buffer is a stack-resident `[i16; 16]` /
`[u8; 16]`. Header is 7 bytes (flags + residual_seed_mode +
dc/ac_factor classes + pred / dc_only seeds); the residual
window is 32 bytes (16 × 2-byte LE halfwords); minimum input is
39 B; max is the libFuzzer 4 KiB default.

`fuzz/Cargo.toml` adds the new `[[bin]]` entry. `README.md`'s
"Fuzz harnesses" bullet count goes from 11 to 12 and the new
target picks up its own bullet with the §14 surface enumeration.
`fuzz/README.md` adds a row to the targets table.

26-second smoke pass landed `cov: 264, ft: 387, corp: 48/1836b`
across 1 000 000 iterations from an empty seed on
aarch64-apple-darwin at 250 000 exec/s, zero panics — the
primitive-layer kernel runs at a high exec/s rate comparable to
the round-259 `panic_free_intra_predict_kernels` target.

No source change. No public-API change. Test counts unchanged.

### Added — `panic_free_bool_codec` fuzz target (round 261, 2026-06-08)

Round 261 is the depth-mode fuzz round on the §7 boolean range coder
— the lowest-level entropy primitive both decode and encode paths
funnel through. The ten existing fuzz targets reach §7 only
indirectly: the four decode harnesses (`panic_free_decode_keyframe` /
`_decoder_state` / `parse_headers` / `panic_free_token_block`) feed
bytes through `decode_vp8` / `Vp8DecoderState::decode_frame` /
`Vp8FrameHeader::parse` / `decode_block`, every one of which calls
`BoolDecoder::init` once at the top of a partition and then issues
`read_bool` / `read_literal` against probabilities determined by the
higher-level decode state — the attacker controls the bytes but the
probability schedule is locked to whatever the upper layers compute.
The four encode harnesses (`panic_free_encode_keyframe` /
`_two_pass_stream`) reach `BoolEncoder` only through
`encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which run a
valid §9 / §11 / §13 / §15 encode chain on top of it — the bool
coder is the final stage, never driven with attacker-shaped (prob,
value) schedules in isolation.

The new `fuzz/fuzz_targets/panic_free_bool_codec.rs` drives the §7
primitive surface directly across three legs:

1. **Stand-alone decode.** `BoolDecoder::init` (the §7.3
   `init_bool_decoder` 2-byte minimum) and
   `BoolDecoder::init_partition` (the §20 reference's short-input
   fallback that tolerates `sz < 2` with `value = 0` and an empty
   input — the 0- and 1-byte legs small inter MBs land on) are both
   attempted against the attacker bytes. The successful initial
   state is then walked with an attacker-shaped (op-type, prob,
   num_bits) schedule of `read_bool` / `read_literal` /
   `read_signed_literal` calls; `InputTooShort` / `EndOfStream`
   errors are accepted as normal returns (the target only fails on
   panic).
2. **Round-trip encode → decode.** A `BoolEncoder` is fed an
   attacker-shaped (op-type, prob, value, num_bits) schedule via
   `write_bool` / `write_literal` / `write_signed_literal`; the
   encoder is `finish`-ed and the resulting partition is fed back
   into a fresh `BoolDecoder::init` that replays the same schedule
   and asserts every read recovers what was written. Any asymmetry
   in the `split = 1 + (((range - 1) * prob) >> 8)` arithmetic or
   in the §7.3 `add_one_to_output` carry propagation surfaces as a
   mismatch on the read side. The `write_signed_literal` round-trip
   pairs against `read_literal(num_bits) + read_bool(128)` (its
   actual §9.3 / §9.4 / §9.6 inverse) rather than
   `read_signed_literal`, which uses a different §7.3 convention
   (sign-first, magnitude-second) and is documented in the harness
   to NOT be the symmetric inverse.
3. **`write_treed` round-trip.** A 7-entry tree mirroring the §11
   `kf_ymode_tree` shape is encoded with an attacker-chosen leaf and
   per-node probabilities; the same partition is decode-walked with
   the same probability schedule and the recovered leaf is asserted
   to match the written leaf. The §8.1 `treed_read` ↔ `write_treed`
   pair is exercised end-to-end.

Input layout: 7-byte header (`flags`, `op_count`, `read_back_count`,
`leaf_sel`, three tree-node probability bytes) followed by 3-byte op
records (`op_type`, `prob`, `payload`). `op_count` is capped at 64
to bound per-iteration memory; the encoder's output is bounded by
`op_count * 5` bytes (a worst-case per-bool 32-bit shift+flush),
≤ 320 bytes per iteration. Total input cap is libFuzzer's default
4 KiB.

26-second smoke pass landed `cov: 286, ft: 1162, corp: 232/8981b`
across 1 693 989 iterations from an empty seed on
aarch64-apple-darwin at 65 153 exec/s, zero panics. The primitive
runs at ~3× the throughput of the §15 loopfilter target
(`panic_free_loopfilter_segment`, ~830× the per-iteration speed of
the keyframe-level harnesses) so libFuzzer mutates aggressively on
top of the §7.3 carry-propagation envelope.

Documentation: the harness's prologue is the §7 read-/write-half
audit reference (which kernels are reachable from which existing
harness, why this direct primitive-layer target is needed, and how
the signed-literal asymmetry is intentional). `fuzz/Cargo.toml`'s
header block is updated to enumerate the new target alongside the
existing ten.

### Added — `loop_filter_mb_edge` criterion bench (round 260, 2026-06-08)

Round 260 is the depth-mode bench round on the §15.3 deblock-filter
hot path — the heaviest sibling of the round-170 `loop_filter_normal`
bench, which covers only `subblock_filter` + `simple_segment`. The
new `benches/loop_filter_mb_edge.rs` adds five criterion points
arranged so a future SIMD / unroll rewrite can compare across
branches and across siblings on the same input:

* `mb_filter_wide` — the §15.3 `MBfilter` low-edge-variance branch:
  the three decaying `27/18/9` weight adjustments that update six of
  the eight straddling pixels. This is the per-call hot path on
  every MB-to-MB edge (up to two times per non-skipped luma MB plus
  the chroma analogues), and the kernel a SIMD rewrite would
  primarily target. Round-260 baseline: ~5.8 ns / call.
* `mb_filter_hev` — the §15.3 `MBfilter` high-edge-variance branch:
  the inner-window `common_adjust` outer-tap fallback when `hev`
  trips. Round-260 baseline: ~5.4 ns / call.
* `subblock_filter_low_variance` — head-to-head against the
  partner: the §15.3 sub-block low-variance branch on the *same*
  8-pixel segment so a reader can read off the wide / narrow
  per-call ratio directly. The round-170 `loop_filter_normal` bench
  measures `subblock_filter` at `hev=true` (high-variance branch);
  this entry covers the *low*-variance branch so both branches have
  a baseline number on file. Round-260 baseline: ~5.0 ns / call.
* `common_adjust_outer_taps` and `common_adjust_no_outer` — the
  §15.2 4-pixel inner core that `simple_segment` calls directly and
  that both normal-filter kernels funnel into for their inner-window
  step. Two entries for the two outer-tap polarities. Round-260
  baselines: ~2.9 ns and ~2.6 ns / call.

The bench reuses the same `[120, 122, 124, 126, 130, 132, 134, 136]`
low-variance ramp as `loop_filter_normal.rs` (chosen so `filter_yes`
accepts at `interior_limit = 4`, `edge_limit = 16`) so the two
deblock micro-benches share an input and successive measurements
can compare directly.

Registered as a `harness = false` `[[bench]]` in `Cargo.toml`. No
src/ changes; additive bench-only commit. CI for benches runs
`cargo check --benches` (the round-170 contract), so the new entry
inherits the existing CI guard.

### Added — `panic_free_intra_predict_kernels` fuzz target (round 259, 2026-06-08)

Round 259 is the depth-mode fuzz round on the §12 intra-prediction
pixel-kernel surface — the primitive layer the eight existing fuzz
targets (`panic_free_decode_keyframe`,
`panic_free_decoder_state`, `parse_headers`,
`panic_free_encode_keyframe`, `panic_free_two_pass_stream`,
`panic_free_loopfilter_segment`, `panic_free_token_block`,
`panic_free_motion_search_descent`, `panic_free_sixtap_subpel`)
reach only indirectly through the top-level decode / encode
entry points and that the round-258 `intra_predict_dc16` criterion
bench likewise only exercises three of the eleven public §12
kernels against a fixed `[128u8; 16] / [129u8; 16]` neighbour pair.

The new target drives every public §12 kernel directly:

* The four 16×16 luma primitives `predict_y16x16_dc` / `_v` / `_h` /
  `_tm`. The DC primitive is exercised in all four `(above, left)`
  `Option` polarities so the top-left-fallback (`DEFAULT_TOPLEFT_DC`)
  and the two single-edge fallbacks are reached on every input.
* The `predict_y16x16` dispatcher across every variant of
  `IntraYMode`, including the `B → None` short-circuit.
* The four 8×8 chroma partners `predict_uv8x8_dc` / `_v` / `_h` /
  `_tm` with the same `Option` polarity envelope.
* The `predict_uv8x8` dispatcher across every variant of
  `IntraUvMode`.
* The ten-arm `predict_b4x4` dispatcher across every variant of
  `IntraBmode` — `Dc`, `Tm`, `Ve`, `He`, `Ld`, `Rd`, `Vr`, `Vl`,
  `Hd`, `Hu`. Each diagonal arm references different positions of
  the synthetic §12.3 `E[0..=8]` array and the `above[4..=7]`
  right-extension pixels; sweeping every arm against the same
  attacker-shaped `(above, left, p)` triple exercises the
  assignment-list arithmetic of every diagonal mode.
* A chained leg re-feeds the 16×16 luma TM output's first row /
  column into the chroma neighbour pair so a kernel-output-as-
  kernel-input data-flow shape (cross-plane neighbour reuse, as the
  §11 / §12 macroblock walker performs) is also exercised.

Input layout: 63-byte header (1 flags + 1 `IntraBmode` selector +
1 `p` + 16-byte luma `above` + 16-byte luma `left` + 8-byte chroma
`above` + 8-byte chroma `left` + 8-byte b4x4 `above` + 4-byte b4x4
`left`). Inputs shorter than 63 bytes early-return so libFuzzer
learns the boundary; inputs longer than 4 KiB also early-return as
defence-in-depth against the libFuzzer default. Every kernel writes
into a fixed-size stack-allocated `[u8; 256]` / `[u8; 64]` /
`[u8; 16]`; no heap touches.

A 21-second smoke pass on `aarch64-apple-darwin` landed:

```
cov: 525, ft: 1300, corp: 31/1892b
2 288 663 iterations, exec/s 108 983, rss 409 MiB, zero panics.
```

The §12 primitive kernel runs ~2.6 × the per-iteration coverage of
the round-232 `panic_free_loopfilter_segment` smoke pass with no
heap allocation, and ~1.2 × the exec/s of the round-257
`panic_free_sixtap_subpel` target (both sub-pixel synthesis and
intra prediction are pure stack-allocated pixel primitives, but
intra prediction's kernel envelope is meaningfully smaller — no
`fetch_block_halo` boundary clamp).

Files touched: `fuzz/fuzz_targets/panic_free_intra_predict_kernels.rs`
(new, ~210 LOC); `fuzz/Cargo.toml` (new `[[bin]]` stanza +
comment-block description); `fuzz/README.md` (Targets table row +
input-bytes envelope row in the OOM-cap table + new
`cargo +nightly fuzz run` line); crate `README.md` (round-259
fuzz-target description block + round-259 sentence in the headline
`simd` cell mirroring the round-258 / -257 pattern). Test counts
unchanged (no new lib tests): stable lib 458, nightly + `simd` lib
460.

### Added — `block_sad_16x16` SIMD partner + `block_sad_16x16_single_pair` bench (round 258, 2026-06-08)

Round 258 is the SIMD-depth round on the §17 SAD primitive — the
per-candidate distortion metric every stage of the round-255 luma
motion-search descent ladder collapses to once the per-candidate
prediction has been synthesised. `motion_search_descent.rs` in the
r255 bench profiled `block_sad_16x16` as a ~6.4 ns leaf on
`aarch64-apple-darwin`, called 16× per MB by `mb_luma_sad_at_mv` and
once per MB by `mb_luma_sad_at_whole_mv`. The leaf has the simplest
possible SIMD-friendly shape: 256 packed bytes = 16 rows × 16 bytes,
linear `Σ |s - p|` per-lane reduction with a single horizontal sum
at the end.

`block_sad_16x16_simd` (gated behind the existing `simd` feature,
the same nightly-only `core::simd::Simd` surface the
round-226 / round-247 inverse-transform rewrites already use) pulls
each row through `Simd<u8, 16>::simd_max - simd_min` for the per-lane
absolute difference, widens to `Simd<u16, 16>` and accumulates
across all 16 rows (worst-case `16 × 255 = 4_080` per lane stays
inside `u16`), then closes with a single `reduce_sum()` widened to
`u32`. The byte-equivalence test
`block_sad_simd_matches_scalar_on_stress_inputs` walks a 21-entry
stress set (full-zero / full-saturated extremes, alternating-column
and alternating-row deltas, checkerboard sign-flip, sparse single-
row / single-column deltas, pseudo-random with two seeds, half-block
splits, vertical-stripe split, tiny perturbations, equal-magnitude
sign-flip interleave, ramp-vs-ramp constant offset, both gradient
shapes the descent ladder uses) and is asserted bit-equal against
`block_sad_16x16_scalar` on every input.

`benches/motion_search_descent.rs` grows a new
`block_sad_16x16_single_pair` micro-bench so the SAD leaf has a
stable A/B target inside the same harness the descent stages live
in (the round-255 `small_diamond_search_luma_iters_8`,
`half_pixel_refine_luma_8_offsets`,
`quarter_pixel_refine_luma_8_offsets`, and
`full_descent_whole_half_quarter` numbers).

#### Dispatch decision

The `--quick` numbers on `aarch64-apple-darwin` showed a trade-off:
the SIMD leaf is **−36 %** in isolation (4.08 ns vs 6.43 ns) but
inlining it into the 16-call-per-MB `mb_luma_sad_at_mv` body
regresses `half_pixel_refine_luma_8_offsets` and
`quarter_pixel_refine_luma_8_offsets` by **+13 %** each (2.74 µs →
3.12 µs). The likely cause is increased NEON register pressure
across the surrounding `filter_block_4x4` loop pessimising LLVM's
scheduling around the leaf. The public `block_sad_16x16` therefore
routes to `block_sad_16x16_scalar` under every feature
configuration — same shape as the round-247 dispatch decision for
`forward_dct_4x4` — keeping the descent stages on their fastest
measured shape. The `_simd` listing stays compiled + tested under
the `simd` feature so a future round can re-target it (e.g. on a
host where the regression flips, or with an `#[inline(never)]`
wrapper that prevents the LLVM scheduling spill into
`mb_luma_sad_at_mv`).

| Bench | r255 stable | r258 stable | r258 nightly + simd direct-call | Δ direct-call vs scalar |
|---|---:|---:|---:|---:|
| `motion_search_descent/block_sad_16x16_single_pair` | (new) | 6.27 ns | 4.08 ns | **−35 %** |
| `motion_search_descent/small_diamond_search_luma_iters_8` | 279.2 ns | 278.4 ns | 275.8 ns | ±0 % |
| `motion_search_descent/half_pixel_refine_luma_8_offsets` | 2.74 µs | 2.71 µs | 3.12 µs *(if dispatched to SIMD)* | +13 % (rejected) |
| `motion_search_descent/quarter_pixel_refine_luma_8_offsets` | 2.75 µs | 2.70 µs | 3.13 µs *(if dispatched to SIMD)* | +13 % (rejected) |
| `motion_search_descent/full_descent_whole_half_quarter` | 5.75 µs | 5.64 µs | 5.71 µs | ±1 % |

Test counts: stable lib **458** (+1 vs r257), nightly + `simd` lib
**460** (+2 vs r257 — the public-dispatch equivalence test plus the
direct SIMD-vs-scalar equivalence test, mirroring the
`wht_simd_matches_scalar_on_stress_inputs` shape from
`src/inverse_transform.rs`). No `#[ignore]`; no version bump; no
`Cargo.lock` committed; `oxideav-core = "0.1"`. Wall: read
`docs/video/vp8/` (RFC 6386 §17 / §18.3), `oxideav-core` public
API, and the agent's own crate only; no external library source, no
web search, no third-party crate, no source-reading of any reference
codec implementation; black-box validator usage unchanged.

### Added — `panic_free_sixtap_subpel` fuzz target covering the §18.3 / §20.14 sub-pixel synthesis primitives (round 257, 2026-06-08)

`fuzz/fuzz_targets/panic_free_sixtap_subpel.rs` is a new libFuzzer
harness for the §18 primitive surface — `filter_block_4x4`,
`sixtap_2d`, `fetch_block_halo`, `fetch_block_whole_pixel`, and
`filter_set_for_version`. The round-256
`panic_free_motion_search_descent` target reaches these only through
the §17 motion-search descent ladder, which by construction snaps
every per-candidate MV to the half- or quarter-pixel grid — so
`mv & 7` only ever indexes a subset of the 64 (mx, my) ∈ {0..7}²
fractional combinations the §18.3 tap table indexes. The round-225
`motion_comp_subpel_luma` criterion bench similarly only exercises a
fixed `(mx, my) = (6, 6)` choice against a mid-plane MB so the §20.14
`build_mc_border` edge-replication clamp inside `fetch_block_halo`
stays cold. That left every fractional offset, every filter-set arm
(sixtap `version == 0` vs bilinear other versions selected via
`filter_set_for_version`), and every border-position class
(top-left corner, bottom-right corner, adversarial, mid-plane fast
path) directly under-fuzzed.

The harness drives the (plane-dimension, sub-block-origin, MV,
filter-set) envelope across plane axes ∈ {16, 24, 32, 40} per
dimension; 4×4 sub-block origin saturated against `(width - 4,
height - 4)`; eighth-pixel MV constructed inline so the chosen
fractional `(mx, my)` is honoured regardless of the integer offset;
filter-set version byte drawn from the input so both `Sixtap` and
`Bilinear` arms are exercised. A border-class selector in the flags
byte forces the integer MV to (a) keep the halo strictly inside the
plane (mid-plane fast path), (b) push the origin past the top-left
edge (every halo row / column clamped), (c) push it past the
bottom-right edge (symmetric), or (d) pass the raw signed
`i16`-range bytes through unchanged so the §20.14 clamp absorbs the
full envelope. An 81-byte halo seeded directly from the input
payload also feeds `sixtap_2d` so the convolution sees byte patterns
the §20.14 fetch could never produce (non-monotonic adjacent rows
swinging the partial sum between extremes within a single tap
window) — the `(a + 64) >> 7` rounding and `clamp255` saturation
surface are the primary panic candidates that pattern targets.

Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
entry); plane ≤ 40 × 40 pixels (matches the r256 motion_search_descent
target for memory-footprint consistency); no internal iteration so
every per-iteration work bound is determined by the input header.
The reference-plane allocation stays under 2 KiB; everything else
(9×9 halo, 4×4 output, the `temp` buffer inside `sixtap_2d`) is
stack-allocated. The harness is panic-freedom-only; output
equivalence against the reference decoder remains the responsibility
of the `tests/blackbox_oracle.rs` round-trip suite.

### Added — `panic_free_motion_search_descent` fuzz target covering the §17.1 / §18.3 luma MV picker (round 256, 2026-06-08)

`fuzz/fuzz_targets/panic_free_motion_search_descent.rs` is a new
libFuzzer harness for the §17 / §18.3 primitive surface —
`small_diamond_search_luma`, `half_pixel_refine_luma`,
`quarter_pixel_refine_luma`, `mb_luma_sad_at_whole_mv`, and
`mb_luma_sad_at_mv`. The pre-existing fuzz targets reached this
surface only indirectly through `encode_keyframe` /
`Vp8TwoPassEncoder::encode_frame`, which gate the per-MB motion-vector
picker behind a fully-formed I420 frame + §11 mode picker + §13 token
emitter cascade, leaving the §17 / §18.3 primitive layer
under-fuzzed — in particular the §20.14 edge-replication clamp inside
`fetch_block_halo` that the round-255 `motion_search_descent`
criterion bench never visits (the bench pins the MB at `(1, 1)` inside
a 64×64 plane, well clear of the clamp).

The harness drives the (mb-position, mv-center, plane-dimension,
source-block) envelope across plane axes ∈ {16, 24, 32, 40} per
dimension, MB origin saturated against `width / 16` and `height / 16`,
center MV pre-clamped into `[MV_MIN, MV_MAX]`, and `max_iters` capped
at `8`. The flags byte picks the descent stage (whole-pixel only / +
half-pixel / + quarter-pixel / full ladder), the source-block seed
(gradient vs constant), and the per-candidate evaluator sweep
(`mb_luma_sad_at_whole_mv` 5-probe whole-pixel ring vs
`mb_luma_sad_at_mv` 3×3 quarter-pixel ring) — so every §18.3
fractional offset `(mx, my) ∈ {0, 2, 4, 6}²` is exercised, not just
the ones the descent happened to land on. The reference plane is a
single tile-extended `Vec<u8>` (≤ ~1.6 KiB at the 40×40 max); the
16×16 source block is stack-allocated. Fuzz-only — no library code or
public surface change. Test counts unchanged (stable lib 457, nightly
+ `simd` lib 458).

### Added — `motion_search_descent` criterion bench covering the §17.1 / §18.3 luma MV picker (round 255, 2026-06-08)

`benches/motion_search_descent.rs` is a new criterion micro-bench
that attributes wall-time to the three stages of the encoder-side
luma motion-search ladder — `small_diamond_search_luma`
(whole-pixel descent), `half_pixel_refine_luma` (the 8 half-pixel
offsets around the diamond pick) and `quarter_pixel_refine_luma`
(the 8 quarter-pixel offsets around the half-pixel pick) — plus a
composite `full_descent_whole_half_quarter` number for the whole
per-MB pick. Every encoded inter MB walks the ladder, so the
search-shape layer is one of the hottest §18 paths in the
encoder; no existing bench attributed wall-time to it (the
round-170 `inter_encode_short_clip` wraps the descent inside the
§11 mode picker + §13 token emit + §15 loop-filter cascade and so
cannot isolate a delta from a future descent-shape rewrite, and
the round-170 `motion_comp_subpel_luma` bench sits one layer
*below* it on the §18.3 sixtap kernel only).

Inputs: deterministic 64×64 luma planes with a mixed-frequency
gradient (mirroring the round-170 `motion_comp_subpel_luma` input
shape so the two micro-benches compare directly) and the MB
placed at `(mb_col, mb_row) = (1, 1)` so the search window stays
clear of the §20.14 edge-replication clamp. Headline numbers on
the dev machine (M4, `--quick`): whole-pixel ≈ 277 ns,
half-pixel ≈ 2.70 µs, quarter-pixel ≈ 2.70 µs, full ladder
≈ 5.68 µs per MB — confirming the half/quarter §18.3 sixtap
synthesis cost dominates and any future SIMD fan-out on
`mb_luma_sad_at_mv` is the highest-return target. Bench-only;
no public surface or library-code change. Test counts unchanged
(stable lib 457, nightly + `simd` lib 458).

### Added — §13.2 `ENC_PCAT*` + `ENC_CAT_BASE` anchored against the RFC 6386 spec listing (round 254, 2026-06-08)

The encoder's per-cat-token `DCTextra` writer in `cat_extras()`
emits each cat-residual MSB-first against six terminator-stripped
probability lists (`ENC_PCAT1..ENC_PCAT6` at `encoder.rs`) and offsets
the recovered residual by `ENC_CAT_BASE[c]` to land in the §13.2
range for cat`c`. The decoder reads the same MSB-first bit sequence
via `DCTextra(d, p)` at `read_bool(d, *p)` against the matching
`Pcat<n>` list, then adds `categoryBase[c]`. If a Pcat byte or a
`CAT_BASE` offset ever drifts here — a typo, a transposed pair, a
dropped trailing zero swept into the slice — the encoder would
emit a bit at `p1` while a third-party reference reader consumed
the same bit at `p2`, silently producing a bool-coder range that
looks valid but means a wholly different integer to the reader.
Self-roundtrip CI would still pass (encoder + decoder drift
together), but a third-party decoder would diverge on the first
cat-token in the bitstream.

The new in-crate test
`encoder::tests::enc_pcat_and_cat_base_match_spec_listing` closes
that gap by anchoring each of the six `ENC_PCAT<n>` lists and the
6-entry `ENC_CAT_BASE` array byte-for-byte against the literal
RFC 6386 §13.2 listing (the `Pcat1..Pcat6` arrays at the `DCTextra`
definition and `categoryBase[6]` immediately preceding the
`vp8_dct_value_cost` cost table). Per-list length checks catch a
future regression that swept the trailing `0` terminator into the
slice or dropped a probability; the cross-check against
`cat_extras()` catches a match-arm reorder that desynced the
returned `(base, list)` pair from the cat index; the non-cat-token
sweep catches a future regression that started returning
`Some(...)` for `Dct0..Dct4` or `Eob`. Lib-test only — no public
surface change. Test counts: stable lib 457, nightly + `simd` lib
458 (each +1 over round 253).

### Added — §17.2 `MV_UPDATE_PROBS_FLAT` anchored against the spec 2×19 table (round 253, 2026-06-08)

The encoder's inter-frame entry-points emit the §17.2
`mv_prob_update()` no-update block by walking a flat 38-entry copy
of the RFC 6386 §17.2 `vp8_mv_update_probs[2]` table
(`MV_UPDATE_PROBS_FLAT` at `encoder.rs`). The canonical 2×19
transcription lives in `coded_header.rs` as `MV_UPDATE_PROBS` and
drives the decoder's `parse_mv_prob_update` per-position
`read_bool(MV_UPDATE_PROBS[i][j])`. If the two transcriptions drifted
— a typo, a row/column swap, an off-by-one in the flat walk — the
encoder would emit each F flag at a different probability than the
decoder consumes it at, silently producing a bool-coder range that
looks valid but means a wholly different bitstream to a third-party
reference reader. Self-roundtrip CI would catch it after a real
encode runs, but not at the constants level.

The new in-crate test
`encoder::tests::mv_update_probs_flat_matches_spec_table` closes
that gap by promoting `coded_header::MV_UPDATE_PROBS` to `pub(crate)`
and comparing `MV_UPDATE_PROBS_FLAT[i*MV_PROB_COUNT + j]` byte-for-byte
against `MV_UPDATE_PROBS[i][j]` for `i in 0..2`, `j in 0..19`. A
length sanity-check on the flat array catches a future
`MV_PROB_COUNT` redefinition that would shrink the spec table
without the encoder's `[u8; 38]` literal noticing.

Test counts: stable lib 456 (+1 over round 252), nightly + `simd`
lib 457 (+1 over round 252). No behavioural change — strengthens the
existing §17.2 constant-table regression net.

### Added — §13.4 walk-order byte-equivalence anchored against the actual spec flag table (round 252, 2026-06-08)

The external test
`tests/encoder_token_prob_updates.rs::write_token_prob_updates_all_none_matches_no_update_writer`
validates the byte equivalence of `write_no_token_prob_updates` and
`write_token_prob_updates(all-None)` against a flat
`[128u8; 1056]` placeholder for the §13.4
`coeff_update_probs[4][8][3][11]` flag-probability table. That
placeholder never exercises the rare extreme-probability splits the
real §13.4 table actually contains (entries as low as `5`, as high
as `255`).

The new in-crate test
`encoder::tests::write_no_token_prob_updates_matches_all_none_against_spec_flag_probs`
closes the gap by anchoring the same byte equivalence against the
actual `COEFF_UPDATE_PROBS_FLAT` table (the crate-local flat view of
`coeff_update_probs[4][8][3][11]`). The §13.4 four-nested-`do/while`
walk `(i=0..4, j=0..8, k=0..3, t=0..11)` is identical between the
two writers, and on the all-`None` path each writer emits
`write_bool(p, false)` at the same `p` for every position with no
follow-up `L(8)` — so the byte streams must match. The test catches
a future regression where one writer subtly diverges from the other
on the extreme-probability splits (e.g. a refactor that switches one
writer to `write_bit` at a hard-coded probability, or skips a slot
when the flag probability is `0` / `255`) while the
flat-`[128u8; 1056]` placeholder test would still pass.

Test counts: stable lib 455 (+1 over round 251), nightly + `simd` lib
456 (+1 over round 251). No behavioural change — strengthens the
existing §13.4 walk-order regression net.

### Changed — `forward_wht_4x4` scalar + SIMD rewritten in canonical butterfly shape, mirroring the §14.3 inverse listing (round 251, 2026-06-07)

Round 249 reorganised `forward_dct_4x4_scalar` into the canonical
`(a1, b1, c1, d1)` partial-sum butterfly form, and round 250 propagated
the same shape into `forward_dct_4x4_simd`. Round 251 closes the
remaining gap on the §14 forward transforms by giving
`forward_wht_4x4_scalar` and `forward_wht_4x4_simd` the same
treatment: both listings now mirror the §14.3 inverse listing
`inverse_wht_4x4` line-for-line.

The §14.3 inverse listing pairs `(ip[0], ip[12])` and `(ip[4], ip[8])`
in each pass with the assignments

```
a1 = ip[0] + ip[12];   b1 = ip[4] + ip[8]
c1 = ip[4] - ip[8];    d1 = ip[0] - ip[12]
op[0] = a1 + b1;   op[4] = c1 + d1;   op[8] = a1 - b1;   op[12] = d1 - c1
```

Because the WHT matrix `M` is symmetric and `M * M = 4 * I`, the
forward and inverse 1-D Walsh-Hadamard transforms use the same
butterfly — only the final rounding differs (`round_div2(x)` on the
forward side, `(x + 3) >> 3` on the inverse side, the two together
producing the round-trip `(8v + 3) >> 3 = v` for a uniform-DC input
of value `v`). Before the rewrite the forward listings paired
`(i0, i4)` / `(i8, i12)` for the column pass and `(r0, r1)` /
`(r2, r3)` for the row pass, with the four outputs assigned in a
different `(a1+b1, a1-b1, c1-d1, c1+d1)` shape. After the rewrite
each pass uses the identical pair-selection and butterfly assignment
as the §14.3 inverse, so the forward / inverse pairs sit side-by-side
in the source.

The rearrangement is bit-exact: each butterfly output unfolds to the
same four-term sum it did before the refactor (`tmp[0] = (i0 + i12) +
(i4 + i8) = i0 + i4 + i8 + i12`, `tmp[4] = (i4 - i8) + (i0 - i12) =
i0 + i4 - i8 - i12`, etc.), and integer addition on `i32` /
`Simd<i32, 4>` is associative on each lane. A new regression guard
`fwht_scalar_matches_direct_derivation_listing` anchors the refactored
scalar listing against the unfactored direct-derivation form
`forward_wht_4x4_listing` (which writes each output as the literal
four-term row-of-`M` sum, `o0 = i0 + i4 + i8 + i12`, etc.) on the
21-input stress matrix — parallel to round 249's
`fdct_scalar_matches_direct_derivation_listing` for the §14.4 forward
DCT.

Bit-exactness between the scalar and SIMD WHT paths continues to be
covered by the existing `fwht_forward_simd_matches_scalar_on_stress_inputs`
test, which routes through the public `forward_wht_4x4` dispatcher
(SIMD under `simd`, scalar otherwise). The chain on nightly + `simd`
is therefore

```
forward_wht_4x4_simd  ==  forward_wht_4x4_scalar  ==  forward_wht_4x4_listing
```

over the 21-input stress matrix.

Test counts: stable lib 454, nightly + `simd` lib 455 (each +1 over
round 250 for the new `_listing` regression guard). `cargo fmt
--check` clean. `cargo clippy --all-targets --no-deps -- -D warnings`
clean on stable and on nightly + `simd`.

The public dispatcher `forward_wht_4x4` is unchanged: it routes to
`forward_wht_4x4_simd` under the `simd` feature and to
`forward_wht_4x4_scalar` otherwise. (The `forward_dct_4x4` dispatcher
remains pinned to scalar per the round-247 `BENCHMARKS.md`
observation; that decision is independent of the WHT path because the
WHT butterfly is multiply-free and the SIMD lane-wide adds /
shifts pipeline well.)

### Changed — `forward_dct_4x4_simd` rewritten in canonical butterfly shape, matching the round-249 scalar listing (round 250, 2026-06-07)

Round 249 reorganised `forward_dct_4x4_scalar` into the canonical
`(a1, b1, c1, d1)` partial-sum butterfly form that the §14.4 inverse
listing `inverse_dct_4x4_scalar` uses. Round 250 propagates the same
shape into `forward_dct_4x4_simd`, the `core::simd::Simd<i32, 4>`
partner kept compiled under the `simd` feature for the byte-
equivalence assertion. Before the rewrite the SIMD listing held the
two even outputs (`t0`, `t2`) as flat four-term sums (`row0 + row1 +
row2 + row3` and `row0 - row1 - row2 + row3`) and the two odd outputs
(`t1`, `t3`) as flat four-term `c_mul` / `s_mul` chains
(`c0 + s1 - s2 - c3` and `s0 - c1v + c2v - s3`). After the rewrite
each pass groups the partial sums into `(a1, b1)` pair-sums shared
between the two even outputs and `(c1, d1)` pair-differences for the
odd outputs, matching the scalar listing line-for-line:

```
a1 = row0 + row3;   b1 = row1 + row2
c1 = (c_mul(row0) - c_mul(row3)) + (s_mul(row1) - s_mul(row2))   // o4
d1 = (s_mul(row0) - s_mul(row3)) - (c_mul(row1) - c_mul(row2))   // o12
t0 = a1 + b1;   t1 = c1;   t2 = a1 - b1;   t3 = d1
```

Each `c_mul` / `s_mul` lane operation is evaluated separately on its
own row-vector — `c_mul(row0)` and `c_mul(row3)` are computed
independently before their lane-wise difference — never collapsed
into `c_mul(row0 - row3)`, which would change the lane-wise
`>> splat(16)` truncation result. The partial-sum reorder is on
associative lane-wise i32 add / sub, so each lane's final byte is
identical to the previously-flat sum.

Bit-exactness is preserved by the existing
`fdct_forward_simd_matches_scalar_on_stress_inputs` test (nightly +
`simd` only), which asserts `forward_dct_4x4_simd` produces identical
bytes against `forward_dct_4x4_scalar` over the same 21-input stress
matrix round 249's scalar refactor was anchored against (DC-only
across 10 magnitudes, single-AC at every position, mixed gradients,
near-i16 extremes). The scalar listing in turn is anchored against
the unfactored direct-derivation form by round 249's
`fdct_scalar_matches_direct_derivation_listing`, so the chain is

```
forward_dct_4x4_simd  ==  forward_dct_4x4_scalar  ==  forward_dct_4x4_listing
```

over the 21-input stress matrix.

Test counts unchanged: stable lib 453, nightly + `simd` lib 454,
`--no-default-features` lib 448. `cargo fmt --check` clean.
`cargo clippy --all-targets --no-deps -- -D warnings` clean on stable,
on `--no-default-features`, and on nightly + `simd`.

The round-247 public dispatcher decision is unchanged: the public
`forward_dct_4x4` still routes to `forward_dct_4x4_scalar` under every
feature configuration on `aarch64-apple-darwin` (the lane-wide
multiply-heavy chain regresses against the scalar straight-line
listing on this host — see `BENCHMARKS.md` round-247 entry). The
SIMD listing stays compiled under the `simd` feature so a future
round on a host where the multiply-heavy SIMD path pipelines better
can re-target the dispatcher without an intervening listing rewrite.

### Changed — `forward_dct_4x4_scalar` rewritten in canonical butterfly shape mirroring §14.4 inverse listing (round 249, 2026-06-07)

Next-round refinement on top of round 247's SIMD dispatch split.
Round 247 kept the §14.4 forward DCT routed through `forward_dct_4x4_scalar`
under every feature configuration; the scalar listing itself was still
in the unfactored direct-derivation form (`o0 = i0 + i4 + i8 + i12;
o4 = c_mul(i0) + s_mul(i4) - s_mul(i8) - c_mul(i12); ...`), shaped
unlike the §14.4 `inverse_dct_4x4_scalar`'s canonical `(a1, b1, c1, d1)`
partial-sum butterfly form. Round 249 reorganises the listing into that
canonical shape so forward and inverse paths share the same visual
structure:

* Even outputs collect into a single pair of partial sums:
  `a1 = i0 + i12`, `b1 = i4 + i8`, then `o0 = a1 + b1` / `o8 = a1 - b1`.
  This mirrors the inverse's `op[0] = a1 + d1` / `op[12] = a1 - d1`
  shape (transposed: the inverse pairs `(i0, i8)`, the forward pairs
  `(i0, i12)`).
* Odd outputs collect their fixed-point multiplies into a `c1` / `d1`
  pair-difference form:
  `c1 = (c_mul(i0) - c_mul(i12)) + (s_mul(i4) - s_mul(i8))` for `o4`,
  `d1 = (s_mul(i0) - s_mul(i12)) - (c_mul(i4) - c_mul(i8))` for `o12`.
  Each `c_mul` / `s_mul` call is evaluated separately (not collapsed
  into `c_mul(i0 - i12)`, which would be non-equivalent under the
  fixed-point `>> 16` truncation) so the rounding stays bit-exact
  against the unfactored listing.

Bit-exactness is verified by a new `forward_dct_4x4_listing` private
reference function (the unfactored direct-derivation form, kept as a
regression oracle) and the test
`fdct_scalar_matches_direct_derivation_listing` that asserts
`forward_dct_4x4_scalar` produces identical bytes against
`forward_dct_4x4_listing` over the 21-input stress matrix (DC-only
across 10 magnitudes, single-AC at every position, mixed gradients,
near-i16 extremes). The pre-existing `fdct_forward_simd_matches_scalar_on_stress_inputs`
(nightly + `simd`) re-runs after the refactor and continues to pass,
confirming the SIMD path is still byte-exact against the refactored
scalar (the SIMD listing was independently shaped on the same
butterfly form in round 226, so the two now visually agree as well as
producing identical bytes).

Test counts: stable lib 452 → 453 (one new regression test); nightly
+ `simd` lib 453 → 454. The 23-bench `forward_transform_4x4/forward_dct_4x4`
`--quick` numbers are within criterion noise envelope of the round-247
baseline on both stable (10.82 → 10.83 ns) and nightly + `simd`
(9.81 → 10.05 ns); the refactor is a readability / shape-parity change,
not a perf optimisation. `cargo fmt --check` + `cargo clippy
--all-targets --no-deps -- -D warnings` both clean.

### Changed — `forward_dct_4x4` SIMD dispatch split (round 247, 2026-06-07)

Closes the round-226 deferred next-round candidate
*"Split the `forward_dct_4x4` SIMD dispatch — round 226 keeps the
forward DCT routed through SIMD under `simd` for shape parity with the
WHT, even though the bench shows a small (~+8 %) regression."*

The §14.4 `forward_dct_4x4` SIMD path runs the same lane-wide
`c_mul` / `s_mul` chain (8 i32 multiplies per pass × 2 passes) plus a
`round_div2_simd` mask + select, and on `aarch64-apple-darwin` doesn't
pipeline as well as the scalar straight-line code (re-measured at
11.69 ns SIMD vs 10.67 ns scalar this round, matching the round-226
note). The §14.3 `forward_wht_4x4` SIMD path has no multiplies in the
butterfly and stays −18 % under scalar, so it keeps the SIMD dispatch.

* `forward_dct_4x4` now calls `forward_dct_4x4_scalar` directly under
  every feature configuration (no `#[cfg(feature = "simd")]` arm).
* `forward_dct_4x4_simd` stays compiled under the `simd` feature with
  `#[allow(dead_code)]` so the byte-equivalence assertion still has a
  symbol to call.
* `fdct_forward_simd_matches_scalar_on_stress_inputs` is now gated
  `#[cfg(feature = "simd")]` and calls `_simd` directly against
  `_scalar` over the 21-input stress matrix — the equivalence proof is
  preserved regardless of the public dispatch.
* A new `fdct_public_dispatch_is_scalar` test runs on every
  configuration and asserts `forward_dct_4x4(input) == forward_dct_4x4_scalar(input)`
  so a future round can't accidentally re-route the dispatcher without
  flipping that assertion too.

Measured impact under nightly + `simd`:

| Bench | r226 SIMD | r247 SIMD | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_dct_4x4` | 11.69 ns | **9.81 ns** | **−16.2 %** |
| `forward_transform_4x4/forward_wht_4x4` | 8.94 ns | 8.74 ns | −2.2 % |

Stable (no `simd`): unchanged within criterion `--quick` noise envelope
(forward DCT 10.67 → 10.82 ns; forward WHT 10.81 → 10.84 ns). The 452-
test stable lib suite and the 11-test nightly + `simd` forward-transform
suite all pass; the new equivalence-direction test fires on both
configurations.

### Added — `panic_free_token_block` fuzz target (round 237, 2026-06-05)

Fuzz-depth round: closes the gap where the six pre-existing fuzz
targets (`panic_free_decode_keyframe`, `panic_free_decoder_state`,
`parse_headers`, `panic_free_encode_keyframe`, `panic_free_two_pass_stream`,
`panic_free_loopfilter_segment`) all reach §13 only indirectly through
`decode_vp8` / `Vp8DecoderState::decode_frame` / `encode_keyframe` /
`Vp8TwoPassEncoder::encode_frame`, which gate the per-block token walk
behind a fully-formed frame-header + coded-header + dequant state. The
new seventh target drives the §13 primitive surface directly with an
attacker-shaped `(probability override list, predictor lattice,
bool partition)` envelope.

Surface covered:

* `dct_tokens::decode_block(dec, block_type, coeff_probs,
  above_has_nonzero, left_has_nonzero, &mut coeffs) -> Result<usize,
  DctTokenError>` — the §13.2 per-sub-block token loop, walking the
  eleven-internal-node coefficient tree, the `Cat1..Cat6` extra-bits
  ladder, the `prev_was_zero` skip-eob branch, and the `ctx3`
  rollover. All four `BlockType` variants (`YAfterY2` / `Y2` / `UV` /
  `YNoY2`) are reached via two flag bits of the input header.
* `dct_tokens::decode_mb_coeffs(dec, has_y2, mb_skip_coeff,
  coeff_probs, above, left) -> Result<MbCoeffs, MbCoeffError>` — the
  §13.3 25-block macroblock walk in `(Y2, 16 Y, 4 U, 4 V)` order,
  with both `has_y2` polarities and the `mb_skip_coeff` short-circuit
  through `MbEntropyCtx::reset_for_skip`.
* `dct_tokens::merge_default_token_probs(updates) -> CoeffProbs` —
  folded over a `TokenProbUpdates` seeded from up to 32
  `(plane, band, ctx, pos, prob)` tuples per iteration, with a
  toggleable "every slot replaced" seed for the §19.2 maximum-update
  envelope.
* `bool_decoder::BoolDecoder::init_partition` — the §20-reference
  short-tolerant init, so 0- and 1-byte partitions still drive the
  renormalisation tail-case rather than returning early at harness
  entry.

Input layout (consumed from the front of the libFuzzer `data`):

| Bytes | Meaning |
|------:|---------|
| `[0]`  | flags byte (path selector, `has_y2`, `mb_skip_coeff`, above/left nonzero predictors, `BlockType` selector, default vs all-128 prob seed) |
| `[1]`  | override count `n`, saturated to `0..=32` |
| `[2 .. 2+5n]` | `n` five-byte override tuples `(plane, band, ctx, pos, prob)`, each modulo its dimension upper bound |
| next 2 bytes  | above-context 9-bit nonzero bitmap (`MB_ENTROPY_CTX_LEN`) |
| next 2 bytes  | left-context  9-bit nonzero bitmap |
| remainder     | bool-decoder partition bytes |

Caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness entry
as defence-in-depth). `MAX_OVERRIDES = 32` ensures every position
slot of one `(plane, band, ctx)` triple can be replaced via the
override channel alone (16 slots per triple × 2 triples).

Smoke pass:

      21 s wall on aarch64-apple-darwin
      1 765 349 iterations
      cov: 1418, ft: 2172, corp: 296/8566b
      exec/s 84 064 sustained
      peak RSS 316 MiB
      no panics, no aborts, no OOB indices

The §13 primitive surface gets ~7 × the coverage envelope of the
§15 loop-filter target while running at ~84 k exec/s. The full
§19.2 token-prob-update lattice is reachable through the harness'
override channel without any encoder pre-amble.

### Added — `panic_free_loopfilter_segment` fuzz target (round 232, 2026-06-04)

Fuzz-depth round: closes the gap where the four pre-existing fuzz
targets (`panic_free_decode_keyframe`, `panic_free_decoder_state`,
`parse_headers`, `panic_free_encode_keyframe`, `panic_free_two_pass_stream`)
all reach §15 only through `decode_vp8` / `Vp8DecoderState::decode_frame`
/ `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which gate the
per-segment loop-filter primitives behind a fully-formed reconstruction
raster. The new sixth target drives the §15 primitive surface directly
with an attacker-shaped `(seg.len(), base)` envelope.

Surface covered:

* `loop_filter::common_adjust(use_outer_taps, seg, base) -> i32` —
  the §15.2 core 4-pixel adjustment shared by both filter types.
* `loop_filter::simple_segment(edge_limit, seg, base)` — the §15.2
  4-pixel simple filter (luma edges only on the decode side; the
  primitive itself doesn't enforce that).
* `loop_filter::subblock_filter(hev_threshold, interior_limit,
  edge_limit, seg, base)` — the §15.3 8-pixel normal inter-subblock
  filter.
* `loop_filter::mb_filter(hev_threshold, interior_limit, edge_limit,
  seg, base)` — the §15.3 8-pixel normal inter-macroblock filter.
* `loop_filter::LoopFilterParams::derive(loop_filter_level,
  sharpness_level, key_frame)` — the §15.4 parameter derivation
  (saturating-sub cap, `interior_limit==0→1` floor, key-frame vs.
  interframe hev-ladder).

Input layout: 7 header bytes (`loop_filter_level`, `sharpness_level`,
raw `hev_threshold`, raw `interior_limit`, raw `edge_limit`, flag byte
for `(key_frame, use_outer_taps, prefer_simple)`, `base` selector)
followed by the segment payload tiled into a working buffer of length
`max(8, payload.len())`. `base` is masked so `base + 8 <= buf.len()`
unconditionally; both kernel families read up to 8 bytes past `base`.

Coverage budget: 4 KiB input cap (libFuzzer default; re-checked at
harness entry as defence-in-depth); single `Vec<u8>` allocation per
iteration.

Smoke pass — 21 seconds, empty seed, aarch64-apple-darwin:
`cov: 202, ft: 475, corp: 157/2944b` across 5 819 579 iterations,
zero panics. Throughput ~290 000 exec/s — the primitive-layer kernel
runs ~830 × faster per iteration than `panic_free_two_pass_stream`
(at 6244 it/20 s) because no encoder reconstruction raster is
allocated and the §15 kernels each operate on at most an 8-byte
segment.

Files: [`fuzz/fuzz_targets/panic_free_loopfilter_segment.rs`](./fuzz/fuzz_targets/panic_free_loopfilter_segment.rs)
(new target), [`fuzz/Cargo.toml`](./fuzz/Cargo.toml) (one `[[bin]]`
section + header doc rewrite to "six targets"),
[`fuzz/README.md`](./fuzz/README.md) (target table + run-command
list + smoke-pass numbers), [`README.md`](./README.md) (Fuzz
harnesses section bumped from five to six targets with the round-232
prose). No `src/` change; no behaviour change; the new target is a
read-only stressor of an already-stable public surface.

### Added — `forward_wht_4x4` / `forward_dct_4x4` SIMD rewrites (round 226, 2026-06-04)

SIMD-depth round: closes the round-220 next-round candidate. The
public `forward_wht_4x4` and `forward_dct_4x4` are now dispatchers
(SIMD on nightly + `simd`, scalar otherwise) matching the round-180
inverse-side rewrite shape. The forward primitives sit behind the
same `simd` cargo feature the inverse primitives use; no new feature
flags, no new dependencies.

* `forward_wht_4x4_simd` — `core::simd::Simd<i32, 4>` rewrite of the
  §14.3 forward WHT. Holds the input as four row-vectors (lane `j`
  of row `i` is `input[i*4 + j]`), runs the four-column butterfly as
  four parallel lane-wide adds / subs, transposes to put each row of
  the intermediate into a row-vector, runs the row butterfly the
  same way, and applies the symmetric `round_div2` `/2` step
  lane-wide via a shared `round_div2_simd` helper.
* `forward_dct_4x4_simd` — `core::simd::Simd<i32, 4>` rewrite of the
  §14.4 forward DCT. Same layout. The `c_mul` / `s_mul` fixed-point
  multiplies become `Simd::splat(K) * v` / `>> Simd::splat(16)` lane-
  wide chains that produce identical bytes to scalar `(x * K) >> 16`
  because the SIMD spec defines i32 lane multiplies as wrapping and
  signed-i32 lane right-shift as arithmetic.
* `round_div2_simd` — lane-wide port of scalar `round_div2`. The two
  arithmetic branches (`(v + 1) >> 1` and `-((-v + 1) >> 1)`) are
  computed unconditionally and merged with `simd_ge(0).select(...)`,
  followed by `simd_clamp(i16::MIN, i16::MAX)` so the final clamp the
  scalar path applies before the `as i16` truncation is mirrored.
  Uses `core::simd::Select` and `core::simd::cmp::SimdOrd`.

Equivalence proof:
`forward_transform::tests::fdct_forward_simd_matches_scalar_on_stress_inputs`
and `…::fwht_forward_simd_matches_scalar_on_stress_inputs` run a
21-input stress set (all-zero, DC-only across 10 magnitudes,
single-AC at every of the 15 positions, the bench's mixed pattern,
and two high-AC mid-range patterns including alternating-sign) and
assert public-dispatch byte-equality against the renamed `_scalar`
variants. Full 452-test lib suite passes on both stable (scalar
dispatch) and nightly + `simd` (SIMD dispatch).

Headline numbers on `aarch64-apple-darwin` (criterion `--quick`):

* `forward_transform_4x4/forward_wht_4x4`: 10.74 ns → **8.72 ns**
  (**−18.8 %**) — the clearest win in the suite next to the round-180
  inverse WHT.
* `forward_transform_4x4/forward_dct_4x4`: 10.71 ns → 11.54 ns
  (+7.7 %) — the multiply-heavy DCT plus the lane-wide `round_div2`
  cost more per call than the scalar path's straight-line arithmetic.
  Kept routed through SIMD for shape parity with the WHT and so the
  byte-exact equivalence test fires; a future round can split the
  dispatch.

Files: [`src/forward_transform.rs`](./src/forward_transform.rs)
(public dispatchers `forward_wht_4x4` / `forward_dct_4x4`; new
`_scalar` + `_simd` siblings; shared `round_div2_simd` helper; two
new `*_simd_matches_scalar` stress tests);
[`README.md`](./README.md) (feature-table SIMD row extended to cover
the forward partners + Δ numbers);
[`BENCHMARKS.md`](./BENCHMARKS.md) (round-226 section with the A/B
table and the inverse-side comparison). No `Cargo.toml` change — the
`simd` feature and the `[bench]` entry already exist from earlier
rounds.

### Added — `forward_transform_4x4` micro-bench (round 220, 2026-06-03)

Bench-depth round: publishes the long-missing A/B target for the
§14.3 forward WHT (`forward_wht_4x4`) and the §14.4 forward DCT
(`forward_dct_4x4`) — the encoder partners of the round-170 /
round-180 inverse-transform primitives. Up to now the only encoder
benches that touched the forward path were the whole-frame
`keyframe_encode` and `inter_encode_short_clip` jobs, which drive
the forward primitives inside the §11 intra picker + §13 token
emit + §15 loop-filter cascade and so cannot attribute a wall-time
delta to a forward-transform rewrite in isolation.

The new bench mirrors `inverse_transform_4x4.rs`'s input layout
(same DC-heavy 4×4 residual block, same `SAMPLE_INPUT` constant)
so a side-by-side read of the forward and inverse files lines
the two passes up sample-for-sample. Headline baseline numbers
on Apple M4 / aarch64, criterion `--quick`:

* `forward_transform_4x4/forward_dct_4x4` — 10.13 ns
* `forward_transform_4x4/forward_wht_4x4` — 10.48 ns
* `inverse_transform_4x4/inverse_dct_4x4` — 10.25 ns (reference)
* `inverse_transform_4x4/inverse_wht_4x4` — 9.76 ns (reference)

The forward DCT and §14.4 inverse DCT cost within a percent of
each other (shared butterfly shape, shared fixed-point constants).
The forward WHT is ~8 % more expensive than the inverse WHT because
the forward path's symmetric `round_div2` rounds every output
sample (16 calls per invocation) where the §14.3 inverse path's
matching `(x + 3) >> 3` doesn't carry the negative-symmetric branch.

Per-MB call count is 24 × `forward_dct_4x4` (16 Y + 8 chroma)
plus optionally 1 × `forward_wht_4x4` (MBs with a Y2 DC plane),
which lands the forward path at ~76 µs / frame on 320 × 240
(~1.3 % of the 5.81 ms `keyframe_encode` wall time) — not the
hot path but now visible at criterion's micro-bench resolution,
ready as an A/B target for a future SIMD / unroll rewrite parallel
to the round-180 `inverse_dct_4x4_simd` work.

Files: [`benches/forward_transform_4x4.rs`](./benches/forward_transform_4x4.rs)
(new); [`Cargo.toml`](./Cargo.toml) (new `[[bench]]` entry);
[`BENCHMARKS.md`](./BENCHMARKS.md) (round-220 section with the
forward + inverse side-by-side table and the next-round
`forward_dct_4x4` SIMD candidate). No source changes — the bench
target uses the existing `oxideav_vp8::forward_dct_4x4` /
`oxideav_vp8::forward_wht_4x4` crate-root re-exports unchanged.

### Added — multi-frame `panic_free_two_pass_stream` fuzz target (round 213, 2026-06-03)

Fuzz-target-depth round: extends the round-207 four-target suite
with a fifth target that exercises the public
`Vp8TwoPassEncoder::first_pass_analyze` → `encode_frame` loop over a
multi-frame sequence. This is the only encoder fuzz target that
reaches `encode_p_frame_multi_ref` — the §9.7 reference-frame
refresh ladder, the keyframe-vs-Pframe switching state machine, and
the complexity-aware qindex picker that the round-207 keyframe-only
target by definition cannot touch.

Input layout: 8-byte header (visible width / height in MB-units,
base `qindex`, `lf_level`, `golden_interval`, `alt_ref_interval`,
frame count `1..=4`, scene-cut bitmap) followed by one
`bits_per_mb` byte per frame, then a tiled pixel payload. The
fuzzer-supplied scene-cut bitmap can override the analysed flag
per-frame so the §9.7 force-keyframe-on-scene-cut path is exercised
even mid-stream; the per-frame `bits_per_mb` bytes are rescaled into
the qindex picker's full 0..=1024 envelope so the complexity-aware
delta path is also covered. `golden_interval`, `alt_ref_interval`,
`qindex`, and `lf_level` are fed raw so the encoder's structured
`Vp8Error` surface is exercised in the same iteration loop as the
happy-path encode chain. Frame count capped at 4 and per-axis
dimensions at 128 px (i.e. 4 × 128×128 ≈ 96 KiB of luma per
iteration) so the per-iteration memory + wall-time budget stays
inside libFuzzer's defaults even with the full inter-prediction
pipeline + reference-reconstruction carry-over running on every
frame after the first.

A 20-second smoke pass on aarch64-apple-darwin (nightly +
`cargo fuzz run`) landed `cov: 3672` (216 new coverage edges over
the encoder-only target's 2790) and `ft: 19072` features across 6244
iterations from an empty seed, no panics. The new harness is
documented in [`fuzz/README.md`](./fuzz/README.md) alongside the
existing four targets and called out in the top-level README's
"Fuzz harnesses" section.

Files: [`fuzz/fuzz_targets/panic_free_two_pass_stream.rs`](./fuzz/fuzz_targets/panic_free_two_pass_stream.rs)
(new); [`fuzz/Cargo.toml`](./fuzz/Cargo.toml) (binary entry +
intro comment updated from "Four targets" to "Five");
[`fuzz/README.md`](./fuzz/README.md) (target table row + OOM-cap
rows + run command); [`README.md`](./README.md) (Fuzz-harnesses
section bumped from four to five targets).

### Added — encoder-side `panic_free_encode_keyframe` fuzz target (round 207, 2026-06-02)

Depth-mode round: extends the round-200 fuzz harness suite with a
fourth target that exercises the public encoder driver
`encode_keyframe(&I420Frame, &KeyframeParams)`. The previous three
targets cover the decode-side surface only; this one closes the encode
side so the panic-freedom contract now holds on every public entry
point the crate documents.

The target consumes a 7-byte header from the front of the libFuzzer
input — `(mb_w, mb_h)` (each normalised into 1..=16 MB-units i.e.
16..=256 luma px so the per-iteration raster stays inside the same
~96 KiB OOM envelope as `panic_free_decode_keyframe`), then `y_ac_qi`,
`loop_filter_level`, `sharpness_level`, `nbr_of_dct_partitions`, and
`filter_type` raw bytes. The four numeric knobs are NOT pre-clamped
into their wire-legal sub-ranges: the goal is to exercise the
parameter-rejection paths
(`QuantIndexOutOfRange` / `LoopFilterLevelOutOfRange` /
`SharpnessLevelOutOfRange` / `InvalidDctPartitionCount`) in the same
iteration loop as the happy-path §11 / §14 / §13 / §15 chain. The
tail of the input tiles modular-indexed across the three I420 planes
so a short fuzz seed still produces fully-populated pixel data and the
intra mode picker has real content to score.

Smoke pass on `aarch64-apple-darwin` (Apple M-series, nightly
libFuzzer): 17 585 iterations in 31 s from an empty corpus, 2 790
coverage edges, 541-input corpus, zero panics. Subsequent 60 s sweep
hit 28 000+ cumulative runs across both invocations, still zero
panics, no `artifacts/panic_free_encode_keyframe/` crash inputs.

The fuzz crate stays a nested workspace (`[workspace] members = ["."]`
in its `Cargo.toml`) so it remains NOT pulled into the umbrella's
`crates/*` glob — the four targets run on demand
(`cargo +nightly fuzz run <target>`), not as part of umbrella CI.

No `src/` changes; this round adds a regression baseline for
encoder-side fuzz-discoverable defects parallel to the decoder side
landed in round 200.

### Changed — `token_to_bit_path` precomputed (round 204, 2026-06-01)

Closes the round-170 `BENCHMARKS.md` follow-up *"`encoder::token_to_bit_path::descend`
— the function walks a small tree and ends up RD-scoring the same token paths
repeatedly. A precomputed token-to-path table would remove the descent
entirely."*

`token_to_bit_path` previously allocated a fresh `Vec<(usize, bool)>` and
ran a recursive `ENC_COEFF_TREE` descent on every call — and was hit at
least three times per coefficient (encoder block writer, RD bit-cost
estimator, and the §13.4 token-prob counts fitter). The descent is a
pure function of `(start_index ∈ {0, 2}, token ∈ 12 alphabet entries)`,
so all 24 cells (one is the §13.2-forbidden `start = 2, Eob` tombstone)
are materialised once at module load through `std::sync::LazyLock` into
`TOKEN_BIT_PATHS: [[([TokenBitStep; 7], u8); 12]; 2]`. The function now
returns `&'static [TokenBitStep]` — a single index-and-slice with zero
per-token allocation. Path widths cap at 7 (`Cat3..Cat6` from root); the
fixed-width buffer fits in static storage with no heap involvement.

The new `encoder::tests::token_bit_path_table_matches_tree_descent`
re-runs the original recursive descent inline and asserts every
reachable cell agrees on length, every prob_index, and every bit. The
unreachable `(start = 2, Eob)` cell stays a `length = 0` tombstone, so a
caller accidentally requesting it will trip the in-function
`debug_assert`. Bit-identical encoder output across the full 450-test
lib suite + every existing integration test.

Bench delta (Apple M4 / aarch64, criterion `--quick`,
`CARGO_TARGET_DIR=/tmp/oxideav-vp8-r204-target cargo bench
--bench keyframe_encode --bench inter_encode_short_clip`):

* `keyframe_encode_320x240_qi32`: 8.51 ms → **5.97 ms (−29.8 %)**,
  9.03 → **12.87 Mpx/s (+43 %)**.
* `inter_encode_4f_128x128_qi32`: 10.82 ms → **10.20 ms (−5.7 %)**,
  6.06 → **6.42 Mpx/s (+5.9 %)**.

The keyframe-encode delta is the headline number: removing the
per-coefficient `Vec` allocation moves `malloc` / `free` (which sat at
≈ 50 self-samples per pair on the round-170 profile) out of the encode
hot path entirely. The inter-encode delta is smaller because motion
search and reconstruction dilute the token-emission share of total time.

### Added — `cargo-fuzz` harness suite (round 200, 2026-06-01)

Depth-mode round: stands up a new `fuzz/` nested workspace with three
libFuzzer targets, none of which existed in any prior round. The
contract across all three is panic-freedom — public APIs MUST surface
malformed input as a `Result::Err`, never via panic, abort,
debug-arithmetic overflow, or out-of-bounds index.

| Target | Surface |
|--------|---------|
| `panic_free_decode_keyframe` | `decode_vp8` — one-shot keyframe decode end-to-end. Pre-flighted by a §9.1 dimension cap (256 × 256 luma pixels) so wire-legal 14-bit width / height extremes don't OOM the runner. |
| `panic_free_decoder_state`   | `Vp8DecoderState::decode_frame` driven over a length-prefixed packet sequence — the §9.7 LAST / GOLDEN / ALTREF refresh ladder, i.e. the *extreme reference-frame dependency* path a one-shot decode call can't reach. |
| `parse_headers`              | Six pure-parse entry points: `frame_tag::parse_header`, `frame_tag::parse_keyframe_header`, `frame_header::Vp8FrameHeader::parse`, `coded_header::Vp8CodedHeader::parse` (key + inter), `ivf::parse_header`, `ivf::parse_frame_header`. The §19.2 walk in particular exercises `update_segmentation`, `mb_lf_adjustments`, `quant_indices`, `token_prob_update`, and `mv_prob_update` — the malformed-segmentation-map / LF-delta / token-prob-update paths the depth-mode brief explicitly called out. |

Initial smoke pass on `aarch64-apple-darwin` (Apple M-series,
nightly libFuzzer): 200 000 iterations on `parse_headers` in ~4 s,
300 000 iterations on each of `panic_free_decode_keyframe` and
`panic_free_decoder_state` in ~1 s each — zero panics across
800 000 combined iterations starting from an empty seed corpus.
The targets are intended to run continuously on demand
(`cargo +nightly fuzz run <target>`), not as part of umbrella CI;
the fuzz crate is a nested workspace (`[workspace] members = ["."]`)
so it is NOT pulled into the umbrella's `crates/*` glob.

No `src/` changes; this round adds a regression baseline for
fuzz-discoverable defects.

### Added — `rate_control_qi_sweep` criterion bench + published trade-off curve (round 194, 2026-05-31)

Depth-mode round: extends the existing criterion bench suite with a
new `rate_control_qi_sweep` macro-bench that walks
`KeyframeParams::y_ac_qi` — the §9.6 baseline quantiser index, the
principal rate-control knob on `encode_keyframe` — across ten
representative values (`8 / 16 / 24 / 32 / 40 / 48 / 56 / 72 / 96 /
120`) on the same deterministic 320×240 I420 source the round-170
`keyframe_encode` bench uses.

The bench produces both wall-time throughput (Mpx/s) and per-call
output bytes (printed via a one-shot probe encode in the bench
prologue + tabulated in `BENCHMARKS.md`). Trade-off curve on Apple M4
/ aarch64 (criterion `--quick`):

* qi=8   → 1701 B at  8.23 Mpx/s
* qi=32  →  595 B at  9.16 Mpx/s (default)
* qi=120 →  299 B at 10.51 Mpx/s

Across the full sweep: −83 % bytes, +28 % throughput. Both axes are
strictly monotonic; the steepest byte segment is qi=8 → qi=16
(−60 %). The encode-wall-time drop is consistent with the round-170
profile — higher `y_ac_qi` produces shorter token streams + more EOB
early-exits inside `encoder::token_to_bit_path::descend` and
`encoder::estimate_block_bits`, the two top self-time symbols on the
encoder hot path.

No source / encoder changes; this round adds measured evidence
people can tune against and a regression baseline (re-run after any
encoder change and compare the per-qi columns). See
`BENCHMARKS.md` §"Round 194 — rate-control `y_ac_qi` sweep" for the
full 10-row table and reading.


## [0.2.3](https://github.com/OxideAV/oxideav-vp8/compare/v0.2.2...v0.2.3) - 2026-05-29

### Other

- inverse_dct_4x4 SIMD r180: §14.4 vectorisation parallel to round-170 WHT (RFC 6386 §14.4)

### Added — `inverse_dct_4x4` SIMD + byte-exact stress tests (round 180, 2026-05-29)

Followed the round-170 `inverse_wht_4x4_simd` deferred TODO from
`BENCHMARKS.md` ("`inverse_dct_4x4` SIMD ... deferred so this round
ships one SIMD primitive that's been A/B-proven against scalar"). The
public `inverse_dct_4x4` now dispatches at compile time between
`inverse_dct_4x4_scalar` (the unchanged RFC 6386 §14.4 listing) and
`inverse_dct_4x4_simd` (a `core::simd::Simd<i32, 4>` rewrite of the
two §14.4 passes), matching the round-170 `inverse_wht_4x4` dispatch
shape exactly.

The SIMD layout maps lane `j` of each row-vector onto column `j` of the
input matrix; the §14.4 column-pass butterfly + fixed-point multiplies
(`(x * SINPI8_SQRT2) >> 16` etc.) then vectorise as four parallel SIMD
operations across the lanes. After the column pass we transpose the
4×4 i32 matrix so the row-pass runs the same butterfly across the
intermediate, applying the `(x + 4) >> 3` rounding lane-wide. No
external SIMD reference consulted — the layout is derived from the
§14.4 listing in RFC 6386 directly.

Added two byte-exact stress tests (`dct_simd_matches_scalar_on_stress_inputs`
and `wht_simd_matches_scalar_on_stress_inputs`) that compare the public
dispatch against the scalar listing on 21 inputs: DC-only at 10
magnitudes (±8 to ±4096), single-AC at every one of the 15 non-DC
positions, plus two near-extreme mixed gradients. The same suite
backfills equivalence coverage for the round-170 WHT SIMD path. Both
tests pass on stable (where the dispatch is identity) and on
nightly + `simd` (where the dispatch points at the SIMD rewrite).

Bench numbers (round 180, criterion `--quick`,
`inverse_transform_4x4/inverse_dct_4x4`): scalar 10.07–10.32 ns,
SIMD 9.51–10.02 ns (Apple M4, aarch64). Modest 1–5 % drop — the §14.4
DCT has 8 fixed-point multiplies per pass against the §14.3 WHT's
zero, so the SIMD margin is smaller than the round-170 WHT's −23.9 %
(the WHT moves end-to-end as parallel adds/subs; the DCT's multiplies
serialise inside each lane). No whole-frame regression on the seven
criterion macro-benches under `cargo bench -p oxideav-vp8 -- --quick`.

## [0.2.2](https://github.com/OxideAV/oxideav-vp8/compare/v0.2.1...v0.2.2) - 2026-05-27

### Other

- remove API-COMPAT-0.1.13.md spec file; tests/api_compat_0_1_13.rs is the contract
- second pass — remove libvpx from src/state.rs (×5) + tests/fixtures/README.md (×2)
- remove libvpx fixture-provenance attestations from src/lib.rs + src/decoder.rs
- round 170: criterion benches + sample-profile-driven optimisations + real simd feature
- remove remaining libwebp-style mentions from src/encoder.rs + tests/public_quality_mapping.rs
- scrub libwebp-style → WebP-canonical naming
- add standalone-API + blackbox-oracle end-to-end interop suites
- replace hand-waved imports with real working API examples
- rewrite as production-ready overview, drop per-round chronology
- encoder TwoPass r168: real two-pass rate control replaces Tier-3 stub

### Added — criterion benches, sample-profile-driven optimisations, real `simd` feature (round 170, 2026-05-27)

New `benches/` directory ships seven criterion micro/macro benches —
`keyframe_encode` (320×240 keyframe at qi32), `keyframe_decode`
(consuming the same), `inter_encode_short_clip` (4-frame 128×128 inter
through `Vp8InterStreamEncoder`), `inverse_transform_4x4`,
`motion_comp_subpel_luma`, `intra_predict_dc16`,
`loop_filter_normal` — wired to `criterion = "0.5"` as a
dev-dependency. Each bench synthesises its inputs in-bench (no
committed fixtures). The bench harness is `cargo bench -p oxideav-vp8
--bench <name> -- --quick`; see `BENCHMARKS.md` for the running command
matrix + baseline + post-optimisation numbers.

A `sample(1)`-based PID-attach profile of the slowest benches revealed
three concrete optimisation targets:

1. **`encoder::bool_bits` `log2` lookup table.** RD scoring calls
   `bool_bits` for every emitted bit of every candidate block;
   `f64::log2` was the #1 self-time symbol on the inter encode and
   the #6 on the keyframe encode. Replaced the per-call `log2` with a
   precomputed 256-entry `LazyLock<[f64; 256]>` table
   (`BIT_COST_BY_FALSE_PROB[p] = -log2(p / 256)`, p = 0 floored at
   8.0 to preserve the original `prob.max(1)` + `1/256` clamp). Drove
   −21.7 % on `inter_encode_short_clip` and −8.4 % on
   `keyframe_encode`. Bit-identical encoder output.

2. **`motion_comp::fetch_block_whole_pixel` / `fetch_block_halo`
   in-bounds fast paths.** Profile evidence: ~90 self-samples between
   them on the inter bench. Added an `src_x0 >= 0 && src_y0 >= 0 &&
   src_x0 + N <= w && src_y0 + N <= h` early branch that issues each
   output row as a single `copy_from_slice` — no per-pixel
   `isize.clamp()`, no per-byte bounds checks. The edge-replication
   slow path is preserved for border MBs. Drove −30 % on
   `filter_block_4x4` and contributed to the inter-encode delta.
   Byte-exact reconstruction on every existing test.

3. **`inverse_wht_4x4` `core::simd::Simd<i32, 4>` rewrite, gated
   `simd` feature, nightly-only.** §14.3's two-pass butterfly maps
   directly onto a 4-lane `Simd<i32, 4>` layout (one lane per column
   in the first pass; transpose; one lane per row in the second
   pass; lane-wide `(x + 3) >> 3` rounding). Public
   `inverse_wht_4x4` dispatches at compile time between the scalar
   and SIMD kernels; both are byte-exact on every fixture. The
   `simd` feature, previously a reserved no-op carried over from
   0.1.13, now gates this path and ships behind a
   `#![cfg_attr(feature = "simd", feature(portable_simd))]` lib-root
   attribute. −24 % on the `inverse_wht_4x4` micro-bench
   (9.83 ns → 7.48 ns on `aarch64-apple-darwin`).

All optimisations were designed from RFC 6386 directly — no external
codec / SIMD reference consulted. The SIMD layout is derived from the
§14.3 listing's structure (column-independent first pass, row-
independent second pass after transpose). The scalar path stays
bit-for-bit identical to the spec listing.

`Cargo.toml` gains `[dev-dependencies] criterion = "0.5"` + seven
`[[bench]] harness = false` stanzas. The `simd` feature comment is
rewritten to reflect its new role.

### Added — end-to-end interop + standalone-API tests (round 169, 2026-05-27)

Two new test files extend the crate's interop coverage:

* **`tests/standalone_e2e.rs`** — 8 tests that drive the public surface
  reachable under `cargo test -p oxideav-vp8 --no-default-features`.
  Covers the keyframe roundtrip (`encode_keyframe` → `decode_vp8`,
  PSNR-Y ≥ 30 dB), the multi-frame inter roundtrip
  (`Vp8TwoPassEncoder::encode_frame` → `Vp8DecoderState::decode_frame`,
  PSNR-Y ≥ 28 dB), the two-pass schedule (`first_pass_analyze` +
  `two_pass_qindices` must vary across complexity-varied frames), the
  IVF container roundtrip (`ivf::write_header` + `ivf::write_frame` →
  `ivf::parse_header` + `ivf::parse_frame_header`), the `quality_to_qindex`
  table (0 → 127, 75 → 32, 100 → 0, NaN → 127, out-of-range clamps),
  and empty / single-frame two-pass edge cases. Compiles + passes
  under both `--no-default-features` and the default-features build —
  every imported symbol resolves without the `registry` feature.

* **`tests/blackbox_oracle.rs`** — 4 tests that cross-validate against
  `ffmpeg` as a black-box oracle. Direction A drives the real
  `encode_keyframe` (the §11 + §13 + §14 RD encoder, not the Phase-1
  silent path that `encoder_external_decode.rs` already covers) →
  IVF-wrap → `ffmpeg -i pipe:0 -f rawvideo` and asserts the recovered
  YUV420P planes clear PSNR-Y ≥ 30 dB. Direction B is the reverse:
  synthetic source → `ffmpeg -c:v vp8 -f ivf` → `ivf::parse_header` +
  `ivf::parse_frame_header` → `Vp8DecoderState::decode_frame`, with
  PSNR-Y ≥ 25 dB on the recovered frames. Two fixture sizes per
  direction (64×64 and 320×240). Both directions skip via
  `eprintln! + return` when `ffmpeg` isn't on `$PATH` (never
  `#[ignore]`). Locally observed PSNR-Y: direction A 42.34 dB
  (64×64) / 48.99 dB (320×240); direction B 61.54 dB / 52.11 dB.

`ffmpeg` is invoked only as a binary; its source remains off-limits
per the workspace clean-room policy. No third-party Rust crate, no
`libvpx` / `libavcodec` source, no web search, no `WebFetch`.

### Added — two-pass encoder real bodies (round 168, 2026-05-27)

Round 167 landed the `Vp8TwoPass*` family as type-shape stubs whose
bodies returned `Vp8Error::Unsupported("two-pass encoder not yet
implemented in this release")`.  This round replaces every stub body
with a real implementation built from a clean-room rate-control design
sourced exclusively from RFC 6386 §9.6 + the in-tree single-pass
primitives.

* **`first_pass_analyze`** (both the free function and
  `Vp8TwoPassEncoder::first_pass_analyze`) — single linear pass over
  each input frame's luma plane computing mean-absolute-deviation vs
  the previous frame (motion proxy) and per-frame variance (spatial
  activity proxy).  Combined into a `bits_per_mb` cost surrogate
  (`α·log2(1+mad) + β·log2(1+var)`).  Scene-cut detection is gated on
  `DEFAULT_SCENE_CUT_THRESHOLD` and `SCENE_CUT_ABS_FLOOR`.
* **`two_pass_qindices`** — distributes per-frame `qindex` around
  `config.base.qindex` so heavier-than-mean frames receive lower qindex
  (better quality) and lighter-than-mean frames receive higher qindex,
  with the delta clamped to `DEFAULT_AQ_QINDEX_RANGE` either side of the
  baseline and again to RFC 6386 §9.6 `0..=127`.  Scene-cut frames
  subtract `DEFAULT_SCENE_CUT_QUANT_BOOST`.
* **`two_pass_qindex_for_frame`** — stateless single-frame picker;
  applies the scene-cut boost and validates `config.base.qindex`.
* **`Vp8TwoPassEncoder::encode_frame`** — selects keyframe vs P-frame
  (first call, scene cut, or `golden_interval` elapsed → keyframe),
  builds a `KeyframeParams` at the resolved per-frame qindex, drives
  `encode_keyframe_with_reconstruction` or `encode_p_frame_multi_ref`,
  and stashes the reconstruction as the next-frame `LAST` reference.

Tests: `tests/two_pass_roundtrip.rs` — nine tests covering the four-
frame solid→gradient→checker→noise clip (per-frame stats + schedule +
end-to-end decode through `Vp8DecoderState`), scene-cut boost,
out-of-range rejection, empty-input handling, and the fallback
"encode_frame without prior first_pass_analyze" path.

`tests/api_compat_0_1_13.rs::api_compat_0_1_13_encoder_constructors`
updated: the surface-lock test now asserts the **live** two-pass
behaviour (empty input succeeds, single-frame complexity returns a
valid 0..=127 qindex) instead of the previous "must return Err" stub
contract.

## [0.2.1](https://github.com/OxideAV/oxideav-vp8/compare/v0.2.0...v0.2.1) - 2026-05-27

### Other

- 0.1.13 public-surface widen per API-COMPAT-0.1.13.md
- API-COMPAT-0.1.13.md — minimum public surface from crates.io 0.1.13
- encoder API finalize r166: public-surface lock for oxideav-webp lossy-VP8 binding
- encoder Phase 18 r165: §11 intra-pick parallel-composed with §13.4 fitter on the stream-driver refresh + §9.4 deltas axis (RFC 6386 §11 / §13.4 / §13.5 / §9.4 / §9.7)
- encoder Phase 17 r164: §13.4 fitter composed with §9.4 deltas on the stream-driver refresh path (RFC 6386 §13.4 / §13.5 / §9.4 / §9.7)
- encoder Phase 16 r163: §9.4 mb_lf_adjustments() deltas + §11 intra-pick composed on the stream-driver refresh path (RFC 6386 §9.4 / §11 / §9.7 / §16.1)
- encoder Phase 16 r162: §11 intra-within-inter MB picker threaded into Vp8InterStreamEncoder (RFC 6386 §11 / §9.7 / §9.8 / §16.1)
- encoder Phase 16 r161: §11 intra-within-inter MB picker widened to 4 Y × 4 UV whole-block grid (RFC 6386 §11.2 / §11.4 / §16.1)
- encoder Phase 16: §11 / §12.2 intra-within-inter MB picking — first cut (RFC 6386 §11 / §12.2 / §16.1)
- encoder Phase 15: §13.4 fitter threaded into the multi-frame stream drivers (RFC 6386 §13.4 / §13.5 / §9.7)
- encoder Phase 14: §13.4 token-prob observed-counts fitter for the inter (P-frame) path (RFC 6386 §13.4 / §13.5)
- encoder Phase 13: §13.4 token-prob observed-counts fitter for the keyframe path (RFC 6386 §13.4 / §13.5)
- encoder Phase 12: §13.4 token-prob caller-driven update layer for the inter (P-frame) path (RFC 6386 §13.4 / §13.5)
- encoder Phase 12: §13.4 token-prob caller-driven update layer for the keyframe path (RFC 6386 §13.4 / §13.5)
- expose §9.4 filter_type knob through KeyframeParams (RFC 6386 §9.4 / §15.2 / §15.3)
- dequantise inter MB picker forward-transform coeffs before reconstruction (RFC 6386 §14.1)
- encoder Phase 11: §9.4 caller-driven mb_lf_adjustments delta layer (RFC 6386 §9.4 / §19.2 / §20.6)
- encoder Phase 11: §9.7 / §9.8 caller-driven reference-slot refresh patterns (RFC 6386 §9.7 / §9.8 / §19.2 / §20)
- encoder Phase 11: §9.5 / §20.4 multi-partition inter token output (RFC 6386 §9.5 / §20.4 / §13.3)
- encoder Phase 11: §16.2 ref_frame_tree GOLDEN/ALTREF selector in the RD picker (RFC 6386 §9.10/§16.2/§16.3)
- encoder Phase 11: §16.4 SPLITMV per-sub-block MV walk in the RD picker (RFC 6386 §16.4 / §17.2 / §18.1)
- encoder Phase 11: §16.2 NEARESTMV/NEARMV in the RD picker (RFC 6386 §16.2/§16.3)
- encoder Phase 11: §18.3 quarter-pixel motion-search refinement (RFC 6386 §17.1 / §18.3)
- encoder Phase 11: §18.3 half-pixel motion-search refinement (RFC 6386 §17.1 / §18.3)
- encoder Phase 11: per-MB ZEROMV/NEWMV rate-distortion pick (RFC 6386 §16.2 / §16.3 / §17 / §18)
- encoder Phase 11 begin: whole-pixel motion-search primitive (RFC 6386 §17.1 / §18.1 / §20.14)
- encoder Phase 10: multi-frame I + P stream driver (RFC 6386 §9 / §16 / §17 / §18)
- encoder Phase 9 begin: minimum-viable P-frame encoder (RFC 6386 §16 / §17 / §18)
- encoder Phase 8: multi-frame keyframe stream driver (RFC 6386 §4 / §9.7 / §9.8)
- wire §15 loop filter into the keyframe driver (Phase 7)
- encoder Phase 6: multi-partition DCT output (RFC 6386 §9.5 / §20.4)
- encoder Phase 5: rate-distortion intra mode selection (RFC 6386 §13 / §14)
- encoder Phase 4: per-frame keyframe raster driver (RFC 6386 §9 / §11 / §19.2)
- encoder Phase 3b: B_PRED 4×4 sub-block intra mode pick (RFC 6386 §11.3 / §12.3)
- encoder Phase 3: whole-block intra mode pick (RFC 6386 §12.2)
- encoder Phase 2: per-MB block-set wiring (RFC 6386 §13.3 / §14.2)
- encoder Phase 2 begin: §14 forward 4×4 DCT + WHT primitives (RFC 6386 §14.3/§14.4)
- encoder Phase 2: §13 DCT-token block encoder (RFC 6386 §13.2/§13.3)
- rephrase BoolEncoder docstring (no behaviour change)
- Phase 1 — §9 frame-header writers + silent-keyframe path
- expose public Vp8Error umbrella at crate root
- refresh decode_vp8 scope docstring (no behaviour change)
- multi-frame Vp8DecoderState driver + bit-exact P-frame decode
- §16.4 SPLITMV per-sub-block MV decoding + §18 SPLITMV reconstruction
- §16.2/§16.3/§18.1 near/nearest MV census + inter-mode tree
- §18.3 sub-pixel motion compensation (sixtap + bilinear)
- Add §16.2/§18 whole-pixel interframe motion compensation

### Added — 0.1.13 public-surface widen (round 167, 2026-05-27)

Per `crates/oxideav-vp8/API-COMPAT-0.1.13.md` (the parent's contract
landed at `d2d6b12`), this round restores every public symbol the
crates.io `oxideav-vp8 0.1.13` release exposed so historical consumers
pinned to `oxideav-vp8 = "0.1"` can upgrade transparently. The earlier
round 166 finalize covered only the webp-binding subset; this widens it
to the full pre-orphan surface.

* **`CODEC_ID_STR`** — crate-root constant set to `"vp8"`.
* **`Vp8Frame`** — public type alias for `Vp8DecodedFrame`.
* **`Result`** — re-export of `error::Result<T>` at the crate root.
* **`decode_frame`** (registry-gated) — legacy alias of `decode_vp8`
  returning `oxideav_core::VideoFrame`.
* **`frame_tag` module** — new module with `FrameTag`, `FrameType`,
  `KeyframeHeader`, `ParsedHeader`, `parse_header`,
  `parse_keyframe_header`. Delegates to `Vp8FrameHeader::parse`.
* **`ivf` module** — IVF container helpers (`IvfHeader`, `IvfFrameHeader`,
  `parse_header`, `write_header`, `write_frame`, `parse_frame_header`).
  Reachable under `--no-default-features`.
* **`registry` module** (registry-gated) — `register`, `register_codecs`,
  `register_containers` (the last is a tolerated no-op since container
  layers live in sibling crates today).
* **Module-path aliases** (`fdct`, `inter`, `intra`, `loopfilter`, `mv`,
  `tables`, `tokens`, `transform`, `bool_encoder`) — re-export shells
  over the current master's module layout so the historical 0.1.13
  paths keep resolving without renaming the underlying files.
* **`Vp8Encoder` / `Vp8EncoderConfig` / `Vp8EncoderStats`** — typed
  direct-API encoder handle plus its config + running statistics.
  `Vp8Encoder::encode_keyframe` delegates to the existing
  `encode_keyframe` driver; reachable under `--no-default-features`.
* **`LoopFilterMode`** — `Auto` / `Normal` / `Simple` enum (default
  `Auto`). Persisted on `Vp8EncoderConfig`; the encoder body still emits
  the current single behaviour (the enum locks the surface).
* **`Vp8TwoPassEncoder` / `Vp8TwoPassConfig` / `FrameComplexity`** —
  surface only (Tier-3 stub). Every two-pass method / free fn
  (`first_pass_analyze`, `two_pass_qindex_for_frame`,
  `two_pass_qindices`, `make_two_pass_encoder`, `encode_frame`) returns
  `Vp8Error::Unsupported("two-pass encoder not yet implemented in this
  release")`. The rate-control algorithm is intentionally deferred —
  this round locks the type shapes only.
* **`make_encoder_with_config`** (registry-gated) — framework-side
  factory taking a full `Vp8EncoderConfig`. Routes through
  `make_encoder_with_qindex` for now.
* **`make_encoder_typed_with_config`** — `Vp8Encoder::new` convenience.
* **Encoder constants** — 28 documented `DEFAULT_*` / `*_MAX` / `*_MIN`
  literals (`DEFAULT_QINDEX = 50`, `DEFAULT_GOLDEN_INTERVAL = 8`,
  `DEFAULT_ALT_REF_INTERVAL = 16`, etc.) for downstream pattern-matching.
* **Cargo feature `simd`** — no-op flag declared verbatim from the
  0.1.13 manifest so historical consumers that set it explicitly keep
  building.
* **`FrameHeader`** — type alias for `Vp8FrameHeader` at the crate root.
* **`tests/api_compat_0_1_13.rs`** — compile-only assertion suite that
  binds every Crate-root re-export under both feature configurations.
  Locks the API in place; if a future change removes or re-shapes any
  symbol, this file stops compiling.

Verified under both `cargo build -p oxideav-vp8` and
`cargo build -p oxideav-vp8 --no-default-features` (the latter pulls
zero deps per `cargo tree --no-default-features`). All pre-existing
447 / 442 in-tree tests still pass.

### Added — public-API finalize (round 166, 2026-05-27)

* **`oxideav_vp8::error` module + flat `Vp8Error`** — public-surface
  finalize for binding-compatible downstream consumers (notably
  `oxideav-webp`'s lossy VP8 path per
  `crates/oxideav-webp/API-COMPAT-0.1.2.md`). The new `Vp8Error` is a
  four-variant flat enum (`InvalidData(String)` / `Unsupported(String)`
  / `Eof` / `NeedMore`) that maps 1-to-1 to `WebpError`. Reachable via
  both `oxideav_vp8::Vp8Error` (crate-root re-export) and
  `oxideav_vp8::error::Vp8Error` (module path). Standalone-compatible —
  no `oxideav-core` dep, so an embedded image / video pipeline that
  pulls `oxideav-vp8` with `--no-default-features` still gets a usable
  public error type. `From<DecodeError>` / `From<Error>` adapters
  collapse the crate's internal error types into the flat shape so
  internal call paths bubble through `?`.
* **`encoder::make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>>`**
  — the framework `oxideav_core::Encoder` factory entry point. Wraps a
  `Vp8FrameEncoder` adapter around the historical `encode_keyframe`
  direct API. Each `send_frame(Frame::Video)` produces one keyframe
  `Packet`. Validates width/height/pixel-format/qindex up-front with
  clean `Error::invalid` / `Error::unsupported` errors.
* **`encoder::make_encoder_with_quality(params, quality: f32)`** —
  libwebp-style `0.0..=100.0` quality (higher = better), translates
  through `quality_to_qindex`.
* **`encoder::make_encoder_with_qindex(params, qindex: u8)`** —
  explicit VP8 §9.6 `y_ac_qi` (`0..=127`, lower = better).
* **`encoder::quality_to_qindex(quality: f32) -> u8`** — the pure
  libwebp-style quality → §9.6 `y_ac_qi` mapping
  (`round((100 - quality) * 1.27)`, NaN → 127, clamped to `0..=127`).
  Standalone-compatible (no `oxideav-core` dep) so an embedder can
  pick a qindex without building the framework adapter.
* **Crate-root re-exports**: `Vp8Error`, `quality_to_qindex` (always
  reachable); `make_encoder`, `make_encoder_with_quality`,
  `make_encoder_with_qindex` (`#[cfg(feature = "registry")]`-gated).
* **Three new lock-test files** (`tests/public_error_surface.rs`,
  `tests/public_quality_mapping.rs`,
  `tests/public_factory_surface.rs`) pinning the surface — 31 tests
  in total. Compile failure / red run = a regression to one of the
  contracts above.

### Changed — public-API finalize (round 166, 2026-05-27)

* **`encoder::make_encoder()`** — the historical pre-r166 no-arg
  factory that returned a `SilentKeyframeEncoder` is renamed
  `encoder::make_silent_keyframe_encoder()`. The
  `SilentKeyframeEncoder` type and its `encode_keyframe(&[u8], u32, u32)`
  method are unchanged. The historical lower-level path
  `encode_silent_keyframe(SilentKeyframeParams::new(w, h))` (the
  function the helper wrapped) is unaffected.
* **`Vp8Error`** — flattens from the pre-r166 nested-enum shape
  (`Decode(DecodeError)` / `Encode(Error)`) to the four-variant flat
  shape above. `DecodeError` / `Error` remain public types; their
  `From` adapters into `Vp8Error` now collapse onto `InvalidData(_)` /
  `Unsupported(_)` per the README mapping table.

### Added — round 165 (2026-05-27, prior)

* **§11 intra-pick parallel-composed with the §13.4 fitter on the
  stream-driver refresh + §9.4 deltas axis (RFC 6386 §11 / §13.4 /
  §13.5 / §9.4 / §9.7)** — round 165 closes the r164 next-step ladder
  item (5) ("parallel fitter composition on the intra-pick + refresh +
  lf-deltas axis (combining r163 + r164 — the picker on the fitted
  refresh path)"). Two new entry-points:
  * **Bare encoder:**
    `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates`
    — argument shape matches
    `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`
    exactly. Two-pass: pass-1 encodes with §13.5 defaults + `pick_intra
    = true` and collects per-position branch counts via the
    `encode_p_frame_multi_ref_inner_with_counts_and_pick` `counts`
    side-channel; pass-2 re-encodes with the fitted
    `TokenProbUpdates` payload, `pick_intra = true` again, so the RD
    picker re-scores against the merged probability table. The
    round-158 `bytes_fitted <= bytes_default` safety guard carries
    through unchanged — on fall-back we also drop the pass-2 planes so
    a streaming caller's next-frame LAST never mis-matches the
    decoder's reconstruction.
  * **Stream driver:**
    `Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates`
    — threads the across-frame §9.4 carried-delta state per RFC 6386
    §9.4 identically to every other refresh + lf-deltas sibling
    (adj-enabled frames write back the effective deltas; adj-disabled
    frames leave the carry untouched). Neither the §11 picker nor the
    §13.4 fitter perturbs the §9.4 carry — both govern per-MB /
    residual decisions only. Pre-conditions, slot-rotation
    (§20 page-147 walk), and error surface (`NoLastReference`,
    `DimensionsChanged`) match `encode_p_frame_with_refresh` exactly.
  Wire compatibility:
    * Whenever the fitter falls back (no slot crossed the saving
      threshold or pass-2 grew the wire), the bytes are byte-equal to
      `encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick` on
      the same inputs.
    * Whenever the fitter wins, the bytes are byte-equal to the
      bare-encoder composition.
    * In every case the wire is `<=` the round-163 intra-pick default
      — the round-158 bare-encoder safety guard lifted into the
      composed stream driver.
  Same carried-base assumption as the round-158 / round-164 siblings:
  the prior key frame must have been emitted with the §13.5 defaults
  (i.e. via `encode_keyframe` /
  `encode_keyframe_with_reconstruction` / the stream driver's
  `encode_frame` ladder, which satisfy this). Mixing a fitted keyframe
  (`encode_frame_with_fitted_token_prob_updates`) with this entry-
  point is out of round-165 scope.
  Pins (`tests/encoder_inter_stream_intra_pick_fitted_lf_deltas.rs`,
  5 cases): bare-encoder composition byte-equality on a K+P sequence
  with self-decode round-trip; per-frame fitted-composed wire never
  grows vs. round-163 caller-driven intra-pick default; §9.4 carry
  advance / persist / partial-update / adj-disabled / keyframe-reset
  semantics under both intra-pick and fitter engaged; `NoLastReference`
  + `DimensionsChanged` guards.
  Tests: 549 → 554 (+5).
* **§13.4 fitter composed with §9.4 deltas on the stream-driver refresh
  path (RFC 6386 §13.4 / §13.5 / §9.4 / §9.7)** — round 164 closes
  the r163 next-step ladder item (5) ("analogous
  `_with_fitted_token_prob_updates` composition on the refresh +
  lf-deltas axis"). The new
  `Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`
  mirrors the existing
  `encode_p_frame_with_refresh_and_lf_deltas_and_token_updates`
  argument shape (caller-supplied `refresh` + `lf_deltas`, carried
  `[i16; 4]` / `[i16; 4]` state threaded by the stream driver) and
  dispatches to the round-158 bare-encoder
  `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`.
  The bare-encoder's `bytes_fitted <= bytes_default` safety guard
  carries through unchanged: whenever the fitter's saving threshold is
  not crossed (or pass-2's re-encode is larger than pass-1's defaults),
  the stream emits the default-encode bytes — byte-equal to
  `encode_p_frame_with_refresh_and_lf_deltas` on the same inputs.
  Whenever the fitter wins, the bytes are byte-equal to the bare-encoder
  composition. The §9.4 across-frame carry rule is unchanged
  (adj-enabled frames write back the effective deltas; adj-disabled
  frames leave the carry untouched); the §13.4 fitter does NOT
  perturb the §9.4 carry — it governs residual-token coding only,
  identical to the caller-driven token-updates sibling. Pre-conditions,
  slot-rotation (§20 page-147 walk), and error surface
  (`NoLastReference`, `DimensionsChanged`) match
  `encode_p_frame_with_refresh` exactly.
  Pins (`tests/encoder_inter_stream_fitted_lf_deltas.rs`, 5 cases):
  bare-encoder composition byte-equality; fitted-stream wire never
  grows vs. caller-driven default; §9.4 carry advance/persist/reset
  semantics under the fitter; `NoLastReference` + `DimensionsChanged`
  guards.
  Tests: 544 → 549 (+5).
* **§9.4 `mb_lf_adjustments()` deltas + §11 intra-pick — composed on
  the stream-driver refresh path (RFC 6386 §9.4 / §11 / §9.7 / §16.1)**
  — round 163 closes the r162 next-step ladder item #3 ("compose the
  §9.4 `mb_lf_adjustments()` deltas with the intra-pick on the refresh
  path"). Two new entry-points:
  * **Bare encoder:**
    `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick`
    — argument shape matches
    `encode_p_frame_multi_ref_with_refresh_and_lf_deltas` exactly (caller
    supplies `refresh`, `lf_deltas`, `carried_ref_deltas`,
    `carried_mode_deltas`); the intra-pick toggle is implicit (always
    engaged on this entry-point, matching
    `encode_p_frame_multi_ref_with_refresh_and_intra_pick`).
  * **Stream driver:**
    `Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas_and_intra_pick`
    — threads the across-frame §9.4 carried-delta state per RFC 6386
    §9.4 the same way
    `encode_p_frame_with_refresh_and_lf_deltas` /
    `encode_p_frame_with_refresh_and_lf_deltas_and_token_updates` do
    (adj-enabled frames write back the effective deltas; adj-disabled
    frames leave the carry untouched). The §11 picker toggle does NOT
    affect the §9.4 delta carry — it governs per-MB candidate scoring
    only.

  Pure composition — no new picker or §9.4 logic. The bare wrapper
  forwards to `encode_p_frame_multi_ref_inner_with_counts_and_pick`
  with `pick_intra = true` (matching
  `encode_p_frame_multi_ref_with_refresh_and_intra_pick`) and the
  caller-supplied `lf_deltas` + carry inputs (matching
  `encode_p_frame_multi_ref_with_refresh_and_lf_deltas`); the stream
  method mirrors the carry-update + §20 page-147 slot-rotation walk
  used on every refresh-aware sibling. Wire compatibility: calling
  with `LoopFilterDeltas::default()` reproduces
  `encode_p_frame_with_refresh_and_intra_pick` byte-for-byte (pinned),
  so every pre-r163 caller of the round-162 intra-pick refresh path
  keeps the exact wire it had.

  Five new test pins in
  `tests/encoder_inter_stream_intra_pick_lf_deltas.rs`:
  (1) disabled-deltas wire matches the round-162 intra-pick-only path
  byte-for-byte; (2) composed bytes on a K + P with both knobs engaged
  match the bare-encoder composition; (3) the §9.4 across-frame carry
  rule applies on the composed path identically to the non-intra-pick
  sibling (fresh deltas advance the carry, `update = false` carries
  through, partial updates merge per-slot, adj-disabled frames leave
  the carry untouched, keyframes reset to zero); (4) `NoLastReference`
  error before any LAST exists; (5) dimensions-lock preserved on the
  composed path. Tests: 539 → 544.

* **§11 intra-within-inter MB picker threaded into
  `Vp8InterStreamEncoder` (RFC 6386 §11 / §9.7 / §9.8 / §16.1)** —
  round 162 closes the r161 follow-up's #2 ("thread the intra-pick
  into the `Vp8InterStreamEncoder` stream driver — currently only
  the bare-encoder entry-points carry the toggle"). Three new opt-in
  entry-points on `Vp8InterStreamEncoder` mirror the existing
  `encode_frame*` family:
  * `encode_frame_with_intra_pick(frame)` — scheduler-driven
    drop-in companion to `encode_frame`. K-frames go through
    `encode_keyframe_with_reconstruction`; every emitted P-frame
    engages the round-161 picker. §9.7 reference-slot ladder,
    dimensions-lock, and §9.4 carried-delta reset match
    `encode_frame` exactly.
  * `encode_frame_with_force_and_intra_pick(frame, force_keyframe)`
    — same plus the `force_keyframe` override that re-anchors the
    keyframe interval, mirroring `encode_frame_with_force`.
  * `encode_p_frame_with_refresh_and_intra_pick(frame, refresh)` —
    direct P-frame call with caller-driven §9.7 / §9.8 refresh
    pattern. Bypasses the keyframe scheduler. Pre-conditions, the
    §20 page-147 slot rotation, and the error surface
    (`NoLastReference`, `DimensionsChanged`) match
    `encode_p_frame_with_refresh` exactly.

  Pure plumbing — no new picker logic. Each new entry-point harvests
  the §9 reference-slot trio into the `KeyframePlanes` shape, calls
  the matching bare-encoder intra-pick entry-point
  (`encode_p_frame_multi_ref_with_intra_pick` /
  `encode_p_frame_multi_ref_with_refresh_and_intra_pick`), and runs
  the §9.7 / §9.8 slot-rotation walk. Composition is byte-identical
  to a caller that drives the bare-encoder entry-point through the
  same slot-harvest steps by hand. The intra-pick is opt-in: existing
  entry-points are unchanged, so every pre-r162 caller keeps the
  exact wire it had.

  Six new test pins in `tests/encoder_inter_stream_intra_pick.rs`:
  byte-equality with the bare-encoder composition on a K + P
  sequence; scheduler invariants on `interval = 4`; force-keyframe
  re-anchoring on the intra-pick path; refresh-driven slot rotation
  (`refresh_golden_frame = true` updates GOLDEN to the P-frame
  reconstruction); `NoLastReference` error before any LAST exists;
  dimensions-lock preserved. Tests: 533 → 539.

* **§11 intra-within-inter MB picker widened from DC-only to the full
  4 Y modes × 4 UV modes whole-block grid (RFC 6386 §11.2 / §11.4 /
  §16.1)** — round 161 extends the round-160 picker per its own
  next-step ladder item #2 ("intra all-four-Y + all-four-UV mode RD
  picking"). The per-MB picker now scores every
  (`y_mode ∈ {Dc, V, H, Tm}`, `uv_mode ∈ {Dc, V, H, Tm}`)
  whole-block intra combination — 16 candidates in total, `B_PRED`
  excluded (the per-sub-block intra walker is a separate fitter
  family) — picks the J-best, and only then trades it against the
  inter winner on `J + lambda * is_inter_mb-bit` exactly the way
  r160 did. No new public entry-points: the round-160
  `encode_p_frame_multi_ref_with_intra_pick` and
  `encode_p_frame_multi_ref_with_refresh_and_intra_pick` callers get
  the widened picker transparently. The picker's strict-`<` tie-break
  on J means encode-wire bytes stay byte-identical to round 160 on
  any source where `(Dc, Dc)` would have won there; sources with
  structured content (vertical gradients, horizontal stripes, planar
  ramps) now select V / H / TM luma respectively, dropping the §14
  residual magnitude where the matched mode's prediction is
  accurate. Decoder side: zero changes — the §16.1
  `IF_YMODE_TREE` / `UV_MODE_TREE` walks already emitted
  `intra_y_modes[raster].leaf()` / `intra_uv_modes[raster].leaf()`,
  so the only change downstream of the picker is the value range
  those slots can hold (`{Dc, V, H, Tm}` instead of always `Dc`).
  Pinned by `encoder::tests::pick_intra_mb_all_selects_v_h_tm_for_structured_sources`
  (V_PRED / H_PRED / TM_PRED matched-source asserts plus the
  flat-grey ⇒ `(Dc, Dc)` smoke check on which r160 wire compatibility
  hinges). All round-160 self-decode + wire-non-growth pins in
  `encoder_pframe_intra_pick.rs` still pass unchanged.

* **§11 / §12.2 intra-within-inter MB picking — first cut (DC_PRED Y +
  DC_PRED UV, RFC 6386 §11 / §12.2 / §16.1)** — round 160 lands the
  opt-in bit of round 159's next-step ladder item #1: the per-MB
  picker, when called through the new
  `encode_p_frame_multi_ref_with_intra_pick` (or
  `encode_p_frame_multi_ref_with_refresh_and_intra_pick`)
  entry-point, additionally scores a §12.2 DC_PRED intra candidate
  against the running in-frame neighbours and picks whichever of
  (best inter pick, intra DC) wins on `J + lambda * is_inter_mb-bit`.
  J formula matches the inter picker's metric (Y-plane SAD on the
  prediction residual, before residual coding / reconstruction) so
  the cross-candidate trade is apples-to-apples. §9.10 `prob_intra`
  is fitted to the picker's observed (intra, inter) per-MB count
  distribution via `fit_prob_l8(count_intra, count_inter)`. Decoder
  side: zero changes — the bytes re-enter
  `Vp8DecoderState::decode_frame`'s existing inter path; the §16.1
  `parse_inter_frame_intra_macroblock_modes` walker + the keyframe
  per-MB reconstructor handle the intra-on-interframe branch the
  same way they handle a key frame's intra MBs. Existing
  entry-points (`encode_p_frame_multi_ref` and its
  `_with_refresh` / `_with_token_updates` / `_with_lf_deltas`
  family) stay byte-identical to the pre-r160 wire (the new
  `pick_intra` toggle threads through the inner driver, defaulting
  off on every pre-r160 caller). Round 160's scope is a single
  intra candidate; subsequent rounds extend the picker to all four
  whole-block Y modes and all four chroma modes. New
  `tests/encoder_pframe_intra_pick.rs` (3 tests) pins the picker
  selecting intra on a black-K + bright-P pattern, the safety guard
  bounding wire growth on a perfect-match source (≤ +4 bytes
  slack), and end-to-end I + P + P self-decode at Y-PSNR ≥ 30 dB
  per frame at mid quantiser.

* **§13.4 fitter threaded into the multi-frame stream drivers (RFC 6386
  §13.4 / §13.5 / §9.7)** — round 159 closes the round-157 / round-158
  follow-up ("Out of round-158 scope: threading the fitter into
  `Vp8KeyframeStreamEncoder` / `Vp8InterStreamEncoder`"). Three new
  surfaces stack on top of the existing fitter entries:
  `encode_keyframe_with_reconstruction_and_fitted_token_prob_updates`
  (the planes-returning companion of
  `encode_keyframe_with_fitted_token_prob_updates`, shaped the same way
  `encode_keyframe_with_reconstruction` relates to `encode_keyframe` —
  bytes byte-identical to the no-reconstruction fitter, planes are the
  post-§15 macroblock-aligned reconstruction the §9 reference-frame
  buffer wants for the `LAST` / `GOLDEN` / `ALTREF` ladder),
  `Vp8KeyframeStreamEncoder::encode_frame_with_fitted_token_prob_updates`,
  and the inter pair
  `Vp8InterStreamEncoder::encode_frame_with_fitted_token_prob_updates` /
  `encode_frame_with_force_and_fitted_token_prob_updates`. The stream
  drivers reuse the K/P scheduler unchanged — only the per-frame
  bitstream emission swaps to the fitter; the §9.7 / §9.8 three-slot
  refresh and §9.4 across-frame delta carry stay identical to the
  caller-driven path. Round 158's fitter "matching reconstruction
  planes on safety-guard fall-back" guarantee is honoured by both new
  entries so a streaming caller's next-frame `LAST` slot stays
  consistent with the wire on either fitter outcome.

  Wire shrinkage on the new
  `tests/encoder_stream_fitted_token_prob_updates.rs` integration test
  at y_ac_qi = 32 (synthetic source, 6-frame I + P sequence,
  keyframe interval 3):

    * K-frame stream, 4 frames: -3 / +0 / -5 / +0 B (one fitted slot
      crosses the threshold on three of four frames).
    * I + P stream, 6 frames: K(-3) / P(-14) / P(-14) / K(+0) / P(-52)
      / P(-38) — every emitted P-frame's residual amortises the §13.4
      header cost.

  Tests: 523 → 529 (+6) — pin (a) fitted-keyframe-stream never grows
  the wire frame-by-frame relative to the default-stream wire, (b)
  fitted-keyframe-stream bytes replay through `Vp8DecoderState` at
  PSNR ≥ 30 dB / frame on the 5-frame mid-quantiser target, (c) the
  §9.7 / §9.8 keyframe slot-refresh invariant survives the fitter
  (all three slots byte-equal after a fitted K), (d) fitted-inter-
  stream wire is `<=` default-inter-stream wire on every frame of an
  I + P interleave (kind matches default frame-by-frame — fitter has
  zero effect on scheduling), (e) the same self-decode ≥ 30 dB target
  holds on the inter path, (f) `force_keyframe` re-anchors the
  interval the same way as the non-fitted entry-point.

* **§13.4 `token_prob_update()` observed-counts fitter for the inter
  (P-frame) encoder (RFC 6386 §13.4 / §13.5)** — round 158's mirror of
  the round-157 keyframe fitter, closing the natural symmetry between
  the keyframe and inter paths. Three new surfaces:
  `count_inter_frame_branches` (the inter analogue of
  `count_keyframe_branches` — same shape, plus an explicit
  `use_bpred_per_mb: &[bool]` argument because the inter picker stamps
  `IntraYMode::Dc` onto every MB and the "no Y2" decision can't be
  recovered from `y_mode`),
  `encode_p_frame_multi_ref_with_fitted_token_prob_updates` (the
  high-level thin-wrapper entry that uses `RefreshControls::default` /
  `LoopFilterDeltas::default` / `[0; 4]` carried state), and
  `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates`
  (the full-surface entry that exposes all §9.4 / §9.7 / §9.8 knobs
  alongside the fitter). Both entries take the same two-pass approach
  as the keyframe fitter: pass 1 encodes with §13.5 defaults and
  collects per-position branch counts via the new
  `encode_p_frame_multi_ref_inner_with_counts` private side-channel;
  `fit_token_prob_updates` then derives the §13.4 payload; pass 2
  re-encodes with that payload through the round-156 caller-driven
  entry. A `bytes_fitted <= bytes_default` safety guard ships whichever
  wire is smaller (with matching reconstruction planes, so a streaming
  caller's next-frame LAST slot stays consistent regardless of which
  pass won). Synthetic-frame measurements at `y_ac_qi = 32` (smooth
  ramp, default reference, ZEROMV-favouring inter residual): 32×32 ramp
  unchanged (safety-guard fallback — too small to amortise the §13.4
  transmission cost), 64×64 ramp -8.2 %, 128×128 ramp -19.0 %,
  256×256 ramp -30.0 %. The savings rise with frame area for the same
  reason as the keyframe fitter: a 1056-position header amortises over
  more residual when there are more macroblocks. Decoder side: no
  changes — the inter path's existing
  `Vp8DecoderState::decode_inter_frame` →
  `overlay_token_probs(self.coeff_probs, &coded.token_prob_updates)`
  consumes the fitted payload exactly as it did the round-156 caller-
  driven payload. New regression test
  `encoder_pframe_fitted_token_prob_updates.rs` (6 tests) pins (a) the
  high-level entry never grows the wire relative to the default
  inter wire (the safety guard); (b) the fitted inter wire decodes
  through `Vp8DecoderState` after its I-frame predecessor clearing the
  same 25 dB PSNR floor the r155 / r156 / r157 tests pin; (c) the
  fitted §19.2 header round-trips through `Vp8CodedHeader::parse` with
  every recovered `Some(p)` in `[1, 255]`; (d)
  `count_inter_frame_branches` honours `mb_skip_coeff` (skip MBs emit
  no counts); (e) the thin-wrapper and full-surface entries produce
  byte-for-byte the same wire when the full-surface caller passes
  `default`s; (f) `fit_token_prob_updates` derives a strictly-smaller
  wire on a noisy 64×64 residual.
* **§13.4 `token_prob_update()` observed-counts fitter for the keyframe
  encoder (RFC 6386 §13.4 / §13.5)** — round 155 wired the keyframe
  caller-driven layer; round 156 mirrored it on the inter path; round
  157 closes the natural follow-up by letting the encoder *fit* the
  §13.4 payload from observed counts. Four new public surfaces:
  `BranchCounts` (the per-position `(zeros, ones)` counter type for the
  4×8×3×11 table), `empty_branch_counts` / `count_block_branches` /
  `count_mb_branches` / `count_keyframe_branches` (the count-collector
  walkers, lockstep replicas of `encode_coeff_block` /
  `encode_mb_tokens` / the keyframe token-encode pass — they only
  record branch counts, no entropy bits are written),
  `fit_token_prob_updates(counts, min_saving_bits)` (the bit-cost
  fitter — emits `Some(p_obs)` only when the body bit saving exceeds
  the §13.4 transmission cost plus a small re-encode-drift guard),
  and `encode_keyframe_with_fitted_token_prob_updates` (the high-level
  two-pass driver: encode-with-defaults → collect counts → fit →
  re-encode → ship the smaller wire). The fitter's safety guard
  (`bytes_fitted <= bytes_default`) guarantees the entry-point never
  *grows* the wire. Synthetic-frame measurements at `y_ac_qi = 32`:
  -9.6 % (32×32 ramp), -23.4 % (64×64 checker+gradient), -33.6 %
  (128×128 quadratic radial). New regression test
  `encoder_fitted_token_prob_updates.rs` (8 tests) pins (a) the fitter
  is a strict no-op on empty counts; (b) `p_new == p_old` short-
  circuits to no update; (c) the high-level entry never grows the
  wire; (d) the fitted wire decodes through `decode_vp8` clearing the
  same 25 dB PSNR floor the r155 / r156 tests pin; (e) the high-level
  entry returns either the default bytes or strictly-smaller bytes;
  (f) the fitted §19.2 header round-trips through `Vp8CodedHeader::parse`
  with every recovered `Some(p)` in `[1, 255]`; (g) `count_keyframe_
  branches` honours `mb_skip_coeff` (skip MBs emit no counts);
  (h) `fit_token_prob_updates` emits a near-255 `p_new` at a hand-
  loaded 1024:1 zero-biased slot.

* **§13.4 `token_prob_update()` caller-driven layer for the inter
  (P-frame) encoder (RFC 6386 §13.4 / §13.5)** — round 155 landed the
  caller-driven layer on the keyframe path; round 156 mirrors the
  pattern on the inter path. Two new public entry-points:
  `encode_p_frame_multi_ref_with_token_updates` (a thin wrapper using
  `RefreshControls::default` / `LoopFilterDeltas::default` and
  `[0; 4]` carried delta state) and
  `encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates`
  (the full surface with §9.7 refresh + §9.4 LF deltas + §13.4 tokens
  in one call). A matching `Vp8InterStreamEncoder` method
  (`encode_p_frame_with_refresh_and_lf_deltas_and_token_updates`)
  drives the stream encoder through the new payload while preserving
  the §9.4 carried-delta state and §9.7 slot rotation. The decoder's
  inter path already overlays `coded.token_prob_updates` on its
  carried entropy state (`Vp8DecoderState::decode_inter_frame`), so
  the encoder codes residual tokens against the same merged table the
  decoder rebuilds for this frame. `token_updates = None` (or an
  all-`None` array) reproduces the round-155 inter wire byte-for-byte;
  the new full-surface entry-point with default refresh + LF deltas +
  no token updates reproduces the round-151 wire byte-for-byte. Note:
  this entry-point assumes the prior key frame was emitted with the
  §13.5 defaults — the stream driver's `encode_frame*` keyframe path
  satisfies this. Inter `refresh_entropy_probs` stays `false` (per
  §9.10 row 1 of the encoder's hardwired ladder), so the overlay is
  THIS-frame-only — well-suited to a per-frame fit from observed
  token counts in a future round. New regression test
  `encoder_pframe_token_prob_updates.rs` (5 tests) pins (a) no-op
  equivalence: `token_updates = None` and all-`None` array both
  reproduce the round-155 inter wire; (b) non-trivial wire divergence
  + sound self-decode through `Vp8DecoderState` clearing 25 dB PSNR
  on the I+P pair; (c) per-position recovery through
  `Vp8CodedHeader::parse` on the new inter wire; (d) full inter
  entry-point with all defaults reproduces the round-151 wire
  byte-for-byte; (e) `Vp8InterStreamEncoder` stream method preserves
  the no-op equivalence and a follow-up P-frame with no updates still
  self-decodes against the saved entry state.

* **§13.4 `token_prob_update()` caller-driven layer for the keyframe
  encoder (RFC 6386 §13.4 / §13.5)** — three new public surfaces wire
  the encoder up for caller-supplied per-position
  `coeff_probs[i][j][k][t]` replacements. `write_token_prob_updates`
  is the 1056-position writer paired with the existing parser in
  `Vp8CodedHeader::parse`; it walks `[i][j][k][t]` in §13.4's nested
  `do/while` order and emits one flag at
  `coeff_update_probs[i][j][k][t]` per position plus an `L(8)` value
  whenever the slot is `Some(prob)`. `encode_keyframe_with_token_prob_updates`
  is a new keyframe entry-point that merges the supplied
  `TokenProbUpdates` onto the §13.5 defaults via
  `merge_default_token_probs` and threads the merged `coeff_probs`
  table into both the picker's RD estimate and the §13.3 token-encode
  pass, then writes the §13.4 layer so the decoder rebuilds the same
  merged table. `encode_keyframe_with_reconstruction_and_token_updates`
  exposes the same machinery alongside the post-§15 reconstruction
  planes. An all-`None` updates array reproduces the round-154 wire
  byte-for-byte (the no-op path retains the §13.5 defaults verbatim).
  New regression test `encoder_token_prob_updates.rs` (4 tests) pins
  (a) no-op equivalence to the round-154 wire; (b) non-trivial wire
  divergence + sound round-trip through `decode_vp8` clearing 25 dB
  PSNR; (c) per-position recovery through `Vp8CodedHeader::parse`;
  (d) writer-level agreement with `write_no_token_prob_updates` on
  the all-`None` payload.

* **`KeyframeParams::filter_type` exposes the §9.4 `filter_type` knob
  to the encoder (RFC 6386 §9.4 / §15.2 / §15.3)** — the §9.4 1-bit
  field selecting the §15.3 normal vs §15.2 simple loop filter was
  previously hardwired to `false` (normal) on every encoder entry
  point. Round 154 adds a `filter_type: bool` field to
  `KeyframeParams` and threads it into both (a) the `write_loop_filter`
  / `write_loop_filter_with_deltas` wire emit (the bit the decoder
  reads back) and (b) `FrameFilterConfig::simple` for the encoder's
  own §15 post-walk filter pass over its reconstruction buffer, so
  encoder-vs-decoder pixel lockstep holds at both branches.
  `KeyframeParams::default()` keeps `filter_type = false` so every
  pre-r154 wire is byte-identical. New regression test
  `encoder_pframe_simple_filter.rs` pins lockstep at both settings on
  a seam-crossing fixture and pins that the two settings produce
  observably different decoded Y planes (the §15.2 simple kernel is
  edge-only / luma-only and cannot agree with §15.3 on real content).

### Fixed

* **Encoder inter MB picker now dequantises forward-transform output
  before reconstruction (RFC 6386 §14.1)** — the `pick_mb_for_ref`
  picker handed still-quantised coefficients to
  `motion_comp::reconstruct_inter_mb` /
  `motion_comp::reconstruct_split_mv_mb`, both of which document that
  they consume *dequantised* coefficients (the keyframe encoder path
  performs the same dequant on a copy of `MbCoeffs` before calling
  its keyframe-mode reconstructor). The defect made the encoder's
  stored P-frame reconstruction diverge from the decoder's
  self-decode by the §14.1 dequant factor on every coded sub-block,
  so the next P-frame's reference and the decoder's reference were
  the same buffer-shape but different pixels — leaving downstream
  `_p_reconstruction` consumers (loop-filter post-walk pre-image,
  reference-slot refresh) out of sync. The fix mirrors the keyframe
  pattern: dequantise a separate copy of `raw_coeffs` purely for the
  §14.2 / §14.5 reconstruction step, keep the original quantised
  `raw_coeffs` for the §13 token-emit path on the same call site.
  New regression test (`encoder_pframe_loop_filter_recon.rs`) pins
  encoder-recon == decoder-recon byte-for-byte at
  `loop_filter_level = 0` (isolates the §14 recon path) and at
  `loop_filter_level = 32` (ensures §15 runs against the same
  pre-filter pixels on both ends).

### Added

* **VP8 encoder Phase 11: §9.4 caller-driven per-reference /
  per-mode `mb_lf_adjustments()` delta layer (RFC 6386 §9.4 /
  §19.2 / §20.6)** — round 150 (`e6df803`) landed the §9.7 / §9.8
  caller-driven refresh layer; round 151 closes the last documented
  "lacks" tail on the inter encoder. The decoder already honoured
  the §9.4 deltas
  (`loop_filter::calculate_mb_filter_level_inter`); round 151
  exposes the encoder's transmit path through a new public
  `LoopFilterDeltas` struct (`enabled`, `update`,
  `ref_frame_delta[4]`, `mode_delta[4]`), a new
  `LoopFilterDeltaSlot` enum, and a new public
  `write_loop_filter_with_deltas` writer.

  New entry-points:
    - `encode_p_frame_multi_ref_with_refresh_and_lf_deltas(frame,
      last, golden, altref, params, refresh, lf_deltas,
      carried_ref_deltas, carried_mode_deltas)` — standalone
      multi-reference P-frame encoder with both `RefreshControls`
      and `LoopFilterDeltas` exposed. Callers supply the
      across-frame carried state (use `[0; 4]` for a single-shot
      encode).
    - `Vp8InterStreamEncoder::encode_p_frame_with_refresh_and_lf_deltas(frame,
      refresh, lf_deltas)` — stream-driver method that threads the
      across-frame carried state internally per RFC 6386 §9.4
      ("the values from the previous frame are used, unless they
      are updated in the current header"). The carried state is
      reset to `[0; 4]` on every key frame per §9.4 (key frames
      begin a fresh sequence).
    - `Vp8InterStreamEncoder::carried_ref_deltas` /
      `carried_mode_deltas` accessors for the across-frame carry
      inspection / testing.
    - `LoopFilterDeltas::effective(carried_ref, carried_mode)` ⇒
      `(eff_ref, eff_mode)` — public helper that resolves a
      frame's effective per-slot deltas given the carried state.
      Used by the §15 post-walk filter inside the encoder and
      exposed so callers can inspect their carry threading.

  Magnitude validation: per-slot values outside `-63..=63` are
  rejected with the new
  `EncodeError::LoopFilterDeltaOutOfRange { which: LoopFilterDeltaSlot,
  value: i16 }` variant.

  `LoopFilterDeltas::default()` (`enabled = false`) reproduces the
  round-150 wire byte-for-byte, so every existing call site stays
  unchanged.

  Validation: 7 new integration tests in
  `tests/encoder_pframe_lf_deltas.rs` pin (a) round-150 wire
  equivalence for the default deltas; (b) header round-trip for
  non-trivial `enabled = true, update = true` values through
  `Vp8CodedHeader::parse`; (c) the `enabled = true, update = false`
  wire shape (decoder reads `None` for every slot); (d) decoder
  data-flow honouring (same source encoded with zero vs `(+20,
  +15)` deltas produces observably-different decoded Y planes);
  (e) stream encoder across-frame carry + keyframe reset; (f)
  magnitude rejection on both `LoopFilterDeltas::validate` and the
  encoder entry-point; (g) the spec carry-rule pinned in
  `LoopFilterDeltas::effective`. Tests: 487 → 494.

* **VP8 encoder Phase 11: §9.7 / §9.8 caller-driven reference-slot
  refresh patterns (RFC 6386 §9.7 / §9.8 / §19.2 / §20)** — round 149
  baked the inter-frame refresh ladder to `refresh_last = 1` and every
  other §9.7 / §9.8 bit `0`. Round 150 exposes the five
  reference-slot bits through a new `RefreshControls` struct
  (`refresh_golden_frame`, `refresh_alternate_frame`,
  `copy_buffer_to_golden`, `copy_buffer_to_alternate`, `refresh_last`)
  and threads them into `encode_p_frame_multi_ref_with_refresh` (new
  public entry-point) and `Vp8InterStreamEncoder::encode_p_frame_with_refresh`
  (new public method on the stream driver).

  The wire emission follows the §19.2 page-122 listing exactly: the
  L(2) `copy_buffer_to_*` fields are gated on the matching `if
  (!refresh_*_frame)` (the decoder reads `None` for the copy field
  when the refresh bit is 1). `RefreshControls::validate` rejects both
  raw out-of-range copy values (`> 2`) and the silent-intent
  combination `refresh_*_frame = true && copy_buffer_to_* != 0`
  (which the §19.2 gating would silently drop) via the new
  `EncodeError::InvalidCopyBufferSelector { which, value }` /
  `CopyBufferSelector` types.

  The stream driver's slot rotation mirrors the §20 page-147
  ordering verbatim (`copy_arf → copy_gf → refresh_gf → refresh_arf
  → refresh_last`, with the "copy" cases reading from the
  pre-rotation slot state), the same walk
  `Vp8DecoderState::decode_frame` runs, so the encoder's
  `LAST` / `GOLDEN` / `ALTREF` trio evolves the same way the in-tree
  decoder would after consuming the same bytes.

  `RefreshControls::default` reproduces the round-149 ladder
  (`refresh_last = true`, every other bit `0`), so
  `encode_p_frame_multi_ref` becomes a wrapper over the new
  with-refresh entry-point and the wire is byte-identical to round
  149 for the default case (`tests/encoder_pframe_refresh_ladder.rs::
  default_refresh_controls_match_round_149_wire_byte_for_byte`
  pins this).

  Validation: a new `tests/encoder_pframe_refresh_ladder.rs`
  integration test exercises:
    1. byte-for-byte round-149 wire equivalence for the default
       controls;
    2. §19.2 page-122 wire shape — `refresh_golden_frame = 1`
       gates `copy_buffer_to_golden` off (decoder reads `None`)
       while `copy_buffer_to_alternate` survives unchanged;
    3. `RefreshControls::validate` rejection of out-of-range and
       silent-intent combinations up front;
    4. the §20 page-147 slot-rotation walk on a 4-frame I+P×3
       stream — after `refresh_golden_frame = true,
       copy_buffer_to_alternate = 1` on P2, ALTREF holds the
       pre-rotation LAST (= P1 recon) and GOLDEN holds the current
       reconstruction; after `refresh_last = false,
       copy_buffer_to_alternate = 2` on P3, LAST is unchanged from
       P2 and ALTREF holds the pre-rotation GOLDEN;
    5. picker-quality consequence — promoting a clean P1
       reconstruction into GOLDEN (via
       `refresh_golden_frame = true` on P1) lets the picker beat
       the now-flat-gray LAST on a subsequent P3 stripe-back-from-
       flat frame, with a 30 dB Y-PSNR floor and wire
       `prob_last < 255` (at least one MB picked GOLDEN);
    6. `Vp8InterStreamEncoder::encode_p_frame_with_refresh` rejects
       a fresh-stream call with the new
       `StreamEncodeError::NoLastReference` variant.

  Tests: 481 → 487 (+6 in `encoder_pframe_refresh_ladder.rs`).

* **VP8 encoder Phase 11: §9.5 / §20.4 multi-partition inter token
  output (RFC 6386 §9.5 / §20.4 / §13.3)** — extends the round-137
  multi-partition keyframe writer to P-frames. `encode_p_frame_multi_ref`
  now honours `params.nbr_of_dct_partitions` (1 / 2 / 4 / 8) for inter
  frames as well: macroblock rows are distributed round-robin across
  the partitions per the §20.4 row-loop (row `r` → partition
  `r % N`), each partition is its own `BoolEncoder` instance
  finalised with its own §7.3 4-byte flush trailer, and the wire
  prepends `(N - 1) * 3` bytes of size table per the §9.5 layout.
  The §13.3 above-context is column-wise and frame-lived so it is
  shared across partitions; the "left" context resets at every
  macroblock-row boundary so it never has to cross a partition seam.

  The residual coding inside each partition is bit-identical to the
  1-partition case, so the self-decoded picture is unchanged across
  all four legal counts — the layout choice only trades a small
  size overhead (`(N - 1) * 3 + (N - 1) * 4` bytes per frame) for the
  decoder-side parallelism the §9.5 split was designed to enable.
  Validation: a new `tests/encoder_pframe_multi_partition.rs`
  integration test sweeps `N ∈ {1, 2, 4, 8}` on a 32×128 (8 MB row)
  I + P pair at `yac_qi = 16` and asserts the P-frame self-decode
  Y-PSNR is bit-identical across all four counts (**47.01 dB** for
  every partition count) and the encoded byte length grows
  monotonically (80 → 84 → 94 → 114 bytes). Two regression-guard
  tests cover the short-frame case (mb_rows < N, leaving partitions
  unused) and the rejection of out-of-range counts with
  `EncodeError::InvalidDctPartitionCount` before the per-MB pick
  walk runs.

  Tests: 478 → 481 (+3 in `encoder_pframe_multi_partition.rs`).

* **VP8 encoder Phase 11: §16.2 `ref_frame_tree` GOLDEN / ALTREF
  reference-frame selector in the rate-distortion picker
  (RFC 6386 §9.10 / §16.2 / §16.3)** — extends the round-147 P-frame
  picker from a single reference (LAST) to scoring each available
  reference (LAST + optional GOLDEN + optional ALTREF) per MB and
  emitting whichever wins:

  ```
  J(ref, mode, mv) = SAD(mv, ref)
                   + λ · (mv_ref_tree_bits(mode)
                        + ref_frame_tree_bits(ref)
                        + §17 mv bits if NEWMV / NEW4X4
                        + §16.4 partition / sub_mv_ref / NEW4X4 bits
                          if SPLITMV)
  ```

  `ref_frame_tree_bits` follows §16.2: LAST → `B(prob_last)` reads
  `false` (1 bool); GOLDEN → `true, false` (2 bools); ALTREF →
  `true, true` (2 bools). The picker uses `prob_last = prob_gf = 128`
  for scoring (neutral 1-bit-per-branch prior); after every MB is
  picked, the wire `prob_last` / `prob_gf` are fitted to the observed
  per-MB distribution via
  `fit_prob_l8(count_false, count_true) = floor(256·count_false/total)`
  clamped to `1..=255`, so the §16.2 selector bits compress against
  an on-distribution prior.

  The §16.3 `find_near_mvs` census is per-ref (neighbour MVs only
  count toward `near.mvs[]` when their recorded `ref_frame` matches
  the candidate ref), so the picker scores every reference against
  its own population of neighbour predictors. Reconstruction reads
  from the winning ref's planes — a single P-frame can mix LAST /
  GOLDEN / ALTREF predictors per MB.

  `Vp8InterStreamEncoder` now threads all three §9 reference slots
  through to the new picker; the §9.7 refresh ladder is unchanged
  (`refresh_last = 1`, GOLDEN / ALTREF stay frozen at the most-recent
  keyframe's reconstruction), so GOLDEN / ALTREF naturally beat LAST
  for MBs whose source content matches the keyframe after a brief
  disturbance.

  New public surface: `encode_p_frame_multi_ref(frame, last,
  golden: Option<&KeyframePlanes>, altref: Option<&KeyframePlanes>,
  params)`. Backward-compatible: `encode_p_frame_zero_mv(frame, ref,
  params)` is a thin wrapper calling the new path with
  `golden = altref = None`.

  Validation: a new `tests/encoder_pframe_goldenref.rs` integration
  test encodes a 3-frame I+P+P sequence where the picture content
  whipsaws (K = high-contrast stripe pattern, P1 = flat gray
  drifting LAST, P2 = back to original stripe). On P2 the picker
  selects GOLDEN over LAST for the stripe MB (wire `prob_last` lands
  at **128** — half the MBs picked non-LAST) and the self-decode
  Y-PSNR clears **49.76 dB** at `yac_qi = 16`. Two regression-guard
  tests pin the no-GOLDEN-passed and identical-refs cases to
  `prob_last = 255` (the picker correctly collapses to LAST-only
  when GOLDEN / ALTREF provide no distortion gain). All existing
  inter-encoder tests (splitmv 39.65 dB, uniform translation
  48.43 dB, quarter-pixel 48.93 dB, half-pixel 58.19 dB, translated
  feature 50.34 dB) pass with bit-for-bit-equivalent PSNR.

  Tests: 475 → 478 (+3 in `encoder_pframe_goldenref.rs`).

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
