# oxideav-vp8 fuzz harnesses

Panic-free harnesses for the public encode and decode API of
`oxideav-vp8`. Each target feeds arbitrary libFuzzer bytes through
one layer of the RFC 6386 stack and asserts that no input —
well-formed, malformed, or hostile — causes a panic, abort,
debug-arithmetic overflow, or out-of-bounds index.

## Targets

| Target | Surface under test | What it exercises |
|--------|--------------------|-------------------|
| `panic_free_decode_keyframe` | `decode_vp8` | One-shot keyframe decode end-to-end (§9.1 header → §19.2 coded header → §11 / §12 / §13 / §14 / §15 pipeline). Pre-flighted by the §9.1 dimension cap below. |
| `panic_free_decoder_state`   | `Vp8DecoderState::decode_frame` | Stateful multi-packet driver. Exercises the §9.7 reference-frame refresh ladder (LAST / GOLDEN / ALTREF) — the extreme-reference-dependency path a one-shot decode call can never reach. |
| `parse_headers`              | `frame_tag::parse_header`, `frame_tag::parse_keyframe_header`, `frame_header::Vp8FrameHeader::parse`, `coded_header::Vp8CodedHeader::parse` (key + inter), `ivf::parse_header`, `ivf::parse_frame_header` | Pure-parse layer. The §19.2 coded-header walk routes through `update_segmentation`, `mb_lf_adjustments`, `quant_indices`, `token_prob_update`, and `mv_prob_update`. |
| `panic_free_encode_keyframe` | `encode_keyframe(&I420Frame, &KeyframeParams)` | Public encoder driver. Drives both the happy-path §11 intra mode pick → §14 forward transform → §13 token emission → §15 loop-filter reconstruct chain AND the parameter-rejection surface (raw `y_ac_qi` / `loop_filter_level` / `sharpness_level` / `nbr_of_dct_partitions` bytes are fed without normalisation so the encoder's `QuantIndexOutOfRange` / `LoopFilterLevelOutOfRange` / `SharpnessLevelOutOfRange` / `InvalidDctPartitionCount` paths are exercised in addition to the legal-range cases). |
| `panic_free_two_pass_stream` | `Vp8TwoPassEncoder::first_pass_analyze` + `Vp8TwoPassEncoder::encode_frame` (multi-frame loop) | Public multi-frame encoder driver. The only target that reaches `encode_p_frame_multi_ref` (the §9.7 reference-frame refresh ladder, keyframe-vs-Pframe switching state machine, complexity-aware qindex picker). Per-frame `bits_per_mb` and a scene-cut bitmap are fed from the input tail so the qindex-delta envelope and the force-keyframe-on-scene-cut path are exercised even on the first-pass-skipped fallback. Frame count capped at 4 and per-axis dimensions at 128 px to bound per-iteration memory / wall time. |
| `panic_free_loopfilter_segment` | `common_adjust`, `simple_segment`, `subblock_filter`, `mb_filter`, `LoopFilterParams::derive` | §15 per-segment loop-filter primitives. Drives the `(seg.len(), base)` slice-arithmetic envelope plus the four §15.4 `(loop_filter_level, sharpness_level, key_frame)` axis combinations directly, exercising the saturating-clamp / `interior_limit==0→1` floor / hev-ladder branches the higher-level decode and encode harnesses can only reach via a fully-formed reconstruction raster. Both the derived parameter set and an independent raw-byte triple are fed to each kernel, with a snapshot-and-restore step between calls so each primitive sees a fresh segment. A chained-pass leg (`simple_segment` → `mb_filter`, or `subblock_filter` → `mb_filter`) exercises state hand-off across primitives. |
| `panic_free_motion_search_descent` | `small_diamond_search_luma`, `half_pixel_refine_luma`, `quarter_pixel_refine_luma`, `mb_luma_sad_at_whole_mv`, `mb_luma_sad_at_mv` | §17.1 / §18.3 luma motion-search descent ladder. Drives the (mb-position, mv-center, plane-dimension, source-block) envelope into the §20.14 edge-replication clamp inside `fetch_block_halo` and across every §18.3 sixtap fractional offset — the corner of the configuration space the round-255 `motion_search_descent` criterion bench never visits (it pins the MB at (1, 1) inside a 64×64 plane so the SAD landscape stays clear of the clamp). Plane width / height ∈ {16, 24, 32, 40} per axis; MB origin saturated against `width / 16` and `height / 16` so the macroblock stays inside the plane; center MV pre-clamped into `[MV_MIN, MV_MAX]`; `max_iters` capped at `8`. The flags byte picks the descent stage (whole / + half / + quarter / full ladder), the source-block seed (gradient vs constant), and the per-candidate evaluator sweep (`mb_luma_sad_at_whole_mv` 5-probe whole-pixel ring vs `mb_luma_sad_at_mv` 3×3 quarter-pixel ring). |
| `panic_free_sixtap_subpel` | `filter_block_4x4`, `sixtap_2d`, `fetch_block_halo`, `fetch_block_whole_pixel`, `filter_set_for_version` | §18.3 / §20.14 sub-pixel synthesis primitives. The round-256 motion_search_descent target reaches these only through the §17 descent ladder, which by construction snaps every per-candidate MV to the half- or quarter-pixel grid (so `mv & 7` only ever indexes a subset of the 64 (mx, my) ∈ {0..7}² tap-table rows). The round-225 `motion_comp_subpel_luma` criterion bench similarly only exercises a fixed `(mx, my) = (6, 6)` choice against a mid-plane MB. This target drives every fractional offset, both filter-set arms (sixtap `version == 0` vs bilinear other versions, selected via `filter_set_for_version`), and every border-position class (top-left corner, bottom-right corner, adversarial, mid-plane fast path) directly. An 81-byte halo seeded straight from the input also feeds `sixtap_2d` so the convolution sees byte patterns the §20.14 clamp would never produce. |
| `panic_free_intra_predict_kernels` | `predict_y16x16_dc`, `predict_y16x16_v`, `predict_y16x16_h`, `predict_y16x16_tm`, `predict_y16x16` (dispatcher), `predict_uv8x8_dc`, `predict_uv8x8_v`, `predict_uv8x8_h`, `predict_uv8x8_tm`, `predict_uv8x8` (dispatcher), `predict_b4x4` (10-arm dispatcher) | §12 intra-prediction pixel kernels. The seven decode / encode harnesses above reach §12 only indirectly through `decode_vp8` / `Vp8DecoderState::decode_frame` / `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which gate the per-block prediction kernels behind a fully-formed reconstruction raster (every neighbour pixel pre-sourced from the same valid frame; every `above`-extension pixel for right-edge sub-blocks pre-clamped per §12.3). The round-258 `intra_predict_dc16` criterion bench only exercises `predict_y16x16_dc` / `_v` / `_h` against a fixed `[128u8; 16] / [129u8; 16]` neighbour pair. This target drives the §12 primitive surface directly: every `Option<&[u8; 16]>` polarity for `predict_y16x16_dc` (top-left fallback path included), every variant of `IntraYMode` through `predict_y16x16` (including the `B → None` short-circuit), every variant of `IntraUvMode` through `predict_uv8x8`, and every variant of `IntraBmode` through `predict_b4x4` — sweeping every assignment-list arm of the six §12.3 diagonal modes (`Ld`, `Rd`, `Vr`, `Vl`, `Hd`, `Hu`) over the synthetic `E[0..=8]` array and the `above[4..=7]` right-extension pixels. A chained leg re-feeds the 16×16 luma TM output's first row / column into the chroma neighbour pair so a kernel-output-as-kernel-input data-flow shape (cross-plane neighbour reuse) is also exercised. |
| `panic_free_transform_4x4_roundtrip` | `forward_dct_4x4`, `inverse_dct_4x4`, `forward_wht_4x4`, `inverse_wht_4x4`, `inverse_wht_4x4_dc_only`, `dequant_block`, `add_residue_4x4`, `add_residue`, `raster_to_scan`, `clamp_qindex`, `clamp255` | §14 transform / dequant / residue-summation primitive layer. The eleven existing fuzz targets reach §14 only indirectly: the four decode-side targets feed inverse-only via well-formed dequantised residuals gated by the §9 / §11 / §13 / §14.1 state machine; the two encode-side targets feed forward-only via §9.6-clamped residual magnitudes; the five primitive-layer targets above don't touch §14 at all. This target drives the §14 primitive surface directly across four legs: (1) §14.3 WHT forward → inverse round-trip on an attacker-shaped `[i16; 16]` residual seed (mid-magnitude / ±255 / ±1023 §14.2 cliff — the documented §14.4 inverse-DCT envelope, chosen so the intermediate `i32` butterfly multiplies by `SINPI8_SQRT2 = 35468` stay inside `i32`); (2) §14.4 DCT forward → inverse round-trip on the same residual; (3) §14.1 `dequant_block` with attacker-chosen `(dc_factor, ac_factor)` cliff values (`i16::MIN` / `i16::MAX` / `0` plus the §14.1 4..=255 envelope) followed by §14.5 `add_residue_4x4` vs `add_residue` byte-equality on the §14.2-bounded residual; (4) §14.3 `inverse_wht_4x4_dc_only(dc)` asserted byte-equal to `inverse_wht_4x4([dc, 0, …, 0])` for every `dc ∈ [i16::MIN, i16::MAX]`, the §20.16 `raster_to_scan` permutation asserted via multiset equality between input and output, plus `clamp_qindex` (§9.6) and `clamp255` (§14.5) at their `i32::MIN` / `i32::MAX` cliff endpoints. |

The contract these harnesses enforce is **panic-freedom on the
public API surface**, not output equivalence.

## OOM caps

VP8's §9.1 wire format encodes visible width / height as 14-bit
fields each, so a wire-legal extreme yields ~268 Mpx — whose I420
reconstruction raster is ~384 MiB. That would OOM the fuzz runner
instantly and yield no useful coverage. Each target that allocates
gating data short-circuits before the decoder runs:

| Cap | Value | Source |
|-----|-------|--------|
| Max luma pixels per decoded frame (`panic_free_decode_keyframe`) | 256 × 256 (65 536) | I420 raster stays under ~100 KiB |
| Max input length (`panic_free_decoder_state`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth |
| Max packets per iteration (`panic_free_decoder_state`) | 32 | Bounds per-iteration wall time so exec/s stays comparable to the single-frame target |
| Max luma pixels per encoded frame (`panic_free_encode_keyframe`) | 256 × 256 (65 536) | Width / height are normalised to `1 + (b % 16)` MB units so the dimensions land in the same 16..=256 px range as the decode target's cap |
| Max input length (`panic_free_encode_keyframe`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth |
| Max frames per iteration (`panic_free_two_pass_stream`) | 4 | Bounds per-iteration wall time; the §9.7 keyframe / golden / alt-ref schedule turns over within 4 frames at the smaller frame size used here |
| Max luma pixels per encoded frame (`panic_free_two_pass_stream`) | 128 × 128 (16 384) | Tighter than the keyframe-only target so 4 frames × the full pipeline (forward transform → token emit → §15 loop filter → reconstruction storage for the next frame's reference) stays inside the per-iteration memory cap |
| Max input length (`panic_free_two_pass_stream`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth |
| Max input length (`panic_free_loopfilter_segment`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth. The working buffer is `max(8, payload.len())` bytes; no further allocation |
| Max plane dimensions (`panic_free_motion_search_descent`) | 40 × 40 (1 600 px) | Plane axes drawn from {16, 24, 32, 40} so the reference-plane allocation stays well under 2 KiB; the 16×16 source block is stack-allocated |
| Max diamond iterations (`panic_free_motion_search_descent`) | 8 | Bounds per-iteration wall time; 8 iterations × 4 neighbours = 32 SAD probes covers a dimensional cross-section of the search-plane / 4 ≈ 8 pixels, well past the bench's representative value |
| Max input length (`panic_free_motion_search_descent`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth |
| Max plane dimensions (`panic_free_sixtap_subpel`) | 40 × 40 (1 600 px) | Plane axes drawn from {16, 24, 32, 40} for consistency with the round-256 motion_search_descent target; the 81-byte halo and 16-byte 4×4 output are stack-allocated |
| Max input length (`panic_free_sixtap_subpel`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth. The harness has no internal iteration — every per-iteration work bound is determined by the input header |
| Min input length (`panic_free_intra_predict_kernels`) | 63 B | The harness reads a 63-byte header (1 flags + 1 bmode selector + 1 `p` + 16-byte luma `above` + 16-byte luma `left` + 8-byte chroma `above` + 8-byte chroma `left` + 8-byte b4x4 `above` + 4-byte b4x4 `left`); inputs shorter than that early-return so libFuzzer learns the boundary |
| Max input length (`panic_free_intra_predict_kernels`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth. Every kernel writes into a fixed-size stack-allocated `[u8; 256]` / `[u8; 64]` / `[u8; 16]`; no heap touches |
| Min input length (`panic_free_transform_4x4_roundtrip`) | 39 B | The harness reads a 7-byte header (flags + residual_seed_mode + dc/ac_factor classes + pred / dc_only seeds) plus a 32-byte residual window (16 × 2-byte LE halfwords); inputs shorter than that early-return so libFuzzer learns the boundary |
| Max input length (`panic_free_transform_4x4_roundtrip`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth. Every buffer is a fixed-size stack-allocated `[i16; 16]` / `[u8; 16]`; no heap touches |

The `parse_headers` target has **no** dimension cap — it allocates
nothing beyond the parsers' own internal state, so even wire-extreme
inputs only cost a few microseconds.

The `panic_free_decoder_state` target does NOT cap per-packet
dimensions: the very edge case that interesting fuzz inputs
exercise is "a state machine that latched onto frame 0's geometry
and is now being fed frame 1 with a different one." Pre-gating the
width would hide it.

## Run

```sh
cd crates/oxideav-vp8/fuzz
cargo +nightly fuzz run panic_free_decode_keyframe
cargo +nightly fuzz run panic_free_decoder_state
cargo +nightly fuzz run parse_headers
cargo +nightly fuzz run panic_free_encode_keyframe
cargo +nightly fuzz run panic_free_two_pass_stream
cargo +nightly fuzz run panic_free_loopfilter_segment
cargo +nightly fuzz run panic_free_motion_search_descent
cargo +nightly fuzz run panic_free_sixtap_subpel
cargo +nightly fuzz run panic_free_intra_predict_kernels
cargo +nightly fuzz run panic_free_transform_4x4_roundtrip
```

`cargo-fuzz` requires the nightly toolchain for libFuzzer's
sanitiser instrumentation. The crate itself builds on stable; only
the fuzz binaries need nightly.

## Corpus

The repository ships **no** seed corpus. libFuzzer starts from empty
and discovers structure on its own; the targets each converge
on coverage of their respective surface within a few minutes on a
single core. A 20-second smoke run on `panic_free_two_pass_stream`
landed `cov: 3672, ft: 19072` across 6244 iterations at round 213.
A 21-second smoke run on `panic_free_loopfilter_segment` landed
`cov: 202, ft: 475, corp: 157/2944b` across 5 819 579 iterations at
round 232 (the primitive-layer kernel runs ~830 × faster per
iteration than the multi-frame two-pass encoder).

## CI

The fuzz crate is intentionally a separate nested workspace
(`[workspace] members = ["."]` in its `Cargo.toml`) so it is NOT
pulled into the umbrella's `crates/*` glob. The umbrella CI does
not run fuzz iterations; the targets are exercised on demand by
maintainers and during pre-release hardening.
