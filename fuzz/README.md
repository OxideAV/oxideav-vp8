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
