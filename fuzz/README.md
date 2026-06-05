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
| `panic_free_motion_comp_subpel` | `fetch_block_whole_pixel`, `fetch_block_halo`, `sixtap_2d`, `filter_block_4x4`, `interp`, `filter_set_for_version`, `stored_luma_mv`, `chroma_mv`, `apply_full_pixel`, `whole_pixel_fraction_is_zero`, `predict_inter_mb_whole_pixel`, `predict_inter_mb` | §18.3 / §20.14 sub-pixel motion-compensation primitives. Drives the `(plane, stride, w, h, blk_x, blk_y, mv, version)` envelope so the §20.14 `build_mc_border` edge-replication clamp (whole-pixel and 9×9 halo branches), the eighth-pel phase selector across all 8 positions of both filter sets, the `(a + 64) >> 7 → clamp255` rounding ladder in `interp`, and the version-routed sixtap-vs-bilinear filter-set pick are all exercised at the primitive layer. Both phase pairs (`mv & 7` and `(mv ^ flags) & 7`) and both tap tables are driven per iteration so a single fuzz seed reaches more than one branch of the dispatcher; the optional §18.2 / §18.4 MB-level wrappers (`predict_inter_mb_whole_pixel`, `predict_inter_mb`) aggregate the 16×16 luma + 8×8 chroma per-sub-block fetches under a flag-gated leg. |

The harnesses use **no oracle** and depend on no external
implementation. The contract is panic-freedom, not output
equivalence.

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
| Max input length (`panic_free_motion_comp_subpel`) | 4 KiB | libFuzzer default; re-checked at harness entry as defence-in-depth |
| Max plane dimensions (`panic_free_motion_comp_subpel`) | 64 × 64 px, stride ≤ 80 | Plane allocation is `stride × h` ≤ ~5 KiB; the §20.14 `build_mc_border` clamp is fully exercised at this size and per-iteration memory stays inside the runner's cap |
| Max MB-grid (`panic_free_motion_comp_subpel` MB-aggregate leg) | 2 × 2 MB (32 × 32 px luma + 16 × 16 px chroma per plane) | The §18.2 / §18.4 wrappers demand a strict MB-aligned plane; the flag-gated leg synthesises a 1..=2 MB-wide × 1..=2 MB-tall plane so the 16×16 / 8×8 aggregate is reached without inflating per-iteration alloc |

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
cargo +nightly fuzz run panic_free_motion_comp_subpel
```

`cargo-fuzz` requires the nightly toolchain for libFuzzer's
sanitiser instrumentation. The crate itself builds on stable; only
the fuzz binaries need nightly.

## Corpus

The repository ships **no** seed corpus. libFuzzer starts from empty
and discovers structure on its own; the seven targets each converge
on coverage of their respective surface within a few minutes on a
single core. A 20-second smoke run on `panic_free_two_pass_stream`
landed `cov: 3672, ft: 19072` across 6244 iterations at round 213.
A 21-second smoke run on `panic_free_loopfilter_segment` landed
`cov: 202, ft: 475, corp: 157/2944b` across 5 819 579 iterations at
round 232 (the primitive-layer kernel runs ~830 × faster per
iteration than the multi-frame two-pass encoder). A 16-second smoke
run on `panic_free_motion_comp_subpel` landed
`cov: 344, ft: 958, corp: 147/2471b` across 1 197 792 iterations at
round 237 — sustained throughput ~74 800 exec/s on a single core,
peak RSS 456 MiB — establishing the §18.3 primitive harness as
mid-tier between the multi-frame encoder and the §15 segment
harness (the §18.3 halo allocates a 9×9 byte buffer plus a 9×4
intermediate per `sixtap_2d` call, so the per-iteration cost sits
between the multi-frame state machine and the in-place §15
kernels).

## CI

The fuzz crate is intentionally a separate nested workspace
(`[workspace] members = ["."]` in its `Cargo.toml`) so it is NOT
pulled into the umbrella's `crates/*` glob. The umbrella CI does
not run fuzz iterations; the targets are exercised on demand by
maintainers and during pre-release hardening.
