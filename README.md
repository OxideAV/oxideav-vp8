# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386). Decoder and encoder both at
production status as of 2026-05-27.

* RFC 6386 key + inter decode, bit-exact against the reference output
  on 10+ multi-frame fixtures (mid-GOP golden refresh, 10-frame
  auto-alt-ref, ARNR).
* Phase-2 encoder with SPLITMV, GOLDEN / ALTREF, multi-partition,
  RefreshControls, LoopFilterDeltas, §11 intra picker, §13.4
  token-probability fitter, and a complexity-aware two-pass
  rate-control family. Encoder hot path is allocation-free for
  per-coefficient token emission as of round 204 (§13.2 token bit
  paths precomputed into a 24-cell static table; **−30 % keyframe
  encode wall time, +43 % throughput** on the criterion bench).
* The full crates.io `0.1.13` public surface is reachable, both with
  the default `registry` build and under `--no-default-features`.
  Compile-only assertion suite:
  per-symbol contract and [`tests/api_compat_0_1_13.rs`](./tests/api_compat_0_1_13.rs)
  for the compile-only assertion suite.

## Install

```toml
# With oxideav-core for use under the OxideAV runtime:
[dependencies]
oxideav-vp8 = "0.2"

# Or standalone, without any framework dependency:
[dependencies]
oxideav-vp8 = { version = "0.2", default-features = false }
```

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Pulls `oxideav-core` and the framework-trait factories (`make_encoder` / `make_decoder` returning `Box<dyn Encoder>` / `Box<dyn Decoder>`) plus `Vp8Decoder` (the `oxideav_core::Decoder` impl) and the `register*` entry points. |
| `simd` | — | **Nightly-only.** Switches the §14.3 / §14.4 4×4 transform primitives — both inverse partners (`inverse_wht_4x4`, `inverse_dct_4x4`) and the §14.3 forward partner (`forward_wht_4x4`) — over to `core::simd::Simd<i32, 4>` rewrites. Every SIMD primitive is byte-exact against the scalar path on a 21-input stress set (DC-only across 10 magnitudes, single-AC at every position, mixed gradients). Headline micro-bench numbers on `aarch64-apple-darwin` (criterion `--quick`): inverse WHT 9.4 → 7.5 ns (≈ −20 %); inverse DCT 10.07 → ~9.5 ns (≈ −1 to −5 %); forward WHT 10.74 → 8.72 ns (≈ −19 %). The §14.4 forward DCT (`forward_dct_4x4`) is intentionally left on the scalar path even under `simd` as of round 247: the lane-wide `c_mul` / `s_mul` multiply-heavy chain + `round_div2_simd` mask + select doesn't pipeline as well as the scalar straight-line code on `aarch64-apple-darwin` (re-measured 11.69 ns SIMD vs 10.67 ns scalar). The `_simd` implementation stays compiled under the `simd` feature with a `cfg(feature = "simd")` byte-equivalence assertion so a future round can re-target the dispatcher on a host where the multiply-heavy SIMD path flips back to a win. Requires a nightly toolchain because `core::simd` is itself nightly. Default (stable) builds use the scalar §14.3 / §14.4 paths unchanged. See `BENCHMARKS.md` for the A/B numbers + the round-170 / round-180 / round-226 / round-247 profiles that motivated each primitive's vectorisation decision. Round 249 reorganised the `forward_dct_4x4_scalar` listing into the canonical `(a1, b1, c1, d1)` partial-sum butterfly shape mirroring §14.4 `inverse_dct_4x4_scalar`, bit-exact against a private `forward_dct_4x4_listing` regression oracle (the unfactored direct-derivation form) on the same 21-input stress matrix — readability / spec-shape parity, no perf change. Round 250 propagates the same canonical butterfly shape into `forward_dct_4x4_simd` (the partner SIMD listing kept compiled under the `simd` feature for the byte-equivalence assertion). The SIMD and scalar §14.4 forward listings now agree visually as well as producing identical bytes; the `fdct_forward_simd_matches_scalar_on_stress_inputs` byte-exact assertion on the 21-input stress matrix continues to pass. Round 251 closes the §14.3 side of the same shape-parity work: `forward_wht_4x4_scalar` and `forward_wht_4x4_simd` are reorganised into the canonical `(a1, b1, c1, d1)` butterfly form mirroring the §14.3 inverse listing `inverse_wht_4x4` line-for-line — only the final-pass rounding differs (`round_div2(x)` forward vs `(x + 3) >> 3` inverse). A new `fwht_scalar_matches_direct_derivation_listing` regression guard anchors the refactored scalar listing against the unfactored direct-derivation form on the same 21-input stress matrix. Test counts: stable lib 454, nightly + `simd` lib 455 (each +1 over round 250). Round 252 lands an in-crate §13.4 walk-order regression anchor (`encoder::tests::write_no_token_prob_updates_matches_all_none_against_spec_flag_probs`) that strengthens the byte-equivalence assertion between `write_no_token_prob_updates` and `write_token_prob_updates(all-None)` by replacing the placeholder `[128u8; 1056]` flag-probability table with the actual §13.4 `COEFF_UPDATE_PROBS_FLAT` spec table — closes a gap where a future divergent-writer refactor could pass the external placeholder test (`tests/encoder_token_prob_updates.rs`) while drifting on the extreme-probability splits (`5`, `255`) the real §13.4 `coeff_update_probs[4][8][3][11]` table contains. Test counts: stable lib 455, nightly + `simd` lib 456 (each +1 over round 251). Round 253 extends the same anchor pattern to the §17.2 `mv_prob_update()` writer path: a new in-crate test `encoder::tests::mv_update_probs_flat_matches_spec_table` byte-equates the encoder-side `MV_UPDATE_PROBS_FLAT` (the 38-entry flat copy each inter-frame entry-point walks when emitting the `mv_prob_update()` all-zero F-gate block) against the canonical 2×19 `coded_header::MV_UPDATE_PROBS` spec table that `parse_mv_prob_update` reads each F flag at (transcribed from RFC 6386 §17.2 `vp8_mv_update_probs[2]`). Catches a transcription drift between the encoder's flat copy and the decoder's spec-side table at constants level, before any self-roundtrip encode would notice. `coded_header::MV_UPDATE_PROBS` is promoted from private to `pub(crate)` for the anchor; no public-surface change. Test counts: stable lib 456, nightly + `simd` lib 457 (each +1 over round 252). Round 254 reaches further into the §13 token-coder by anchoring the encoder's six per-cat extra-bits probability lists and category-offset array against the literal RFC 6386 §13.2 spec listing: a new in-crate test `encoder::tests::enc_pcat_and_cat_base_match_spec_listing` byte-equates each of `ENC_PCAT1..ENC_PCAT6` (the terminator-stripped probability prefixes `cat_extras()` returns for cat1..cat6 tokens) and the 6-entry `ENC_CAT_BASE` array against the spec `Pcat1..Pcat6 = {159,0} / {165,145,0} / {173,148,140,0} / {176,155,140,135,0} / {180,157,141,134,130,0} / {254,254,243,230,196,177,153,140,133,130,129,0}` and `categoryBase[6] = {5,7,11,19,35,67}` literals. Per-list length checks catch a future regression that swept the trailing `0` terminator into the slice; a cross-check against `cat_extras()` catches a match-arm reorder; a non-cat-token sweep catches a regression that started returning `Some(...)` for `Dct0..Dct4` or `Eob`. Lib-test only — no public surface change. Test counts: stable lib 457, nightly + `simd` lib 458 (each +1 over round 253). Round 258 adds the §17 SAD primitive partner: `block_sad_16x16_simd` pulls the 16×16 (src, pred) pair through `Simd<u8, 16>` per-row `max - min` absdiff into a `Simd<u16, 16>` row accumulator with a single `reduce_sum()` close, byte-exact against the scalar listing on a 21-input stress set (`block_sad_simd_matches_scalar_on_stress_inputs`). The new `motion_search_descent/block_sad_16x16_single_pair` leaf bench measures the SIMD leaf at 4.08 ns (−36 % vs 6.43 ns scalar) in isolation, but inlining it into the 16-call-per-MB `mb_luma_sad_at_mv` body regresses `half_pixel_refine_luma_8_offsets` / `quarter_pixel_refine_luma_8_offsets` by ~+13 % (NEON register pressure spilling into LLVM's scheduling around the surrounding `filter_block_4x4` loop), so the public `block_sad_16x16` dispatcher continues to route to the scalar partner — same shape as round-247 `forward_dct_4x4`. The `_simd` listing stays compiled under the `simd` feature for the byte-equivalence proof and as a future re-target target (e.g. on a host where the regression flips, or with an `#[inline(never)]` wrapper that prevents the LLVM scheduling spill). Test counts: stable lib 458, nightly + `simd` lib 460 (each +1 / +2 over round 257; the nightly +2 picks up both the public-dispatch equivalence test and the direct `block_sad_16x16_simd`-vs-scalar equivalence test). Round 259 adds the §12 intra-prediction primitive-layer fuzz target `panic_free_intra_predict_kernels`: drives the eleven public §12 kernels (`predict_y16x16_dc` / `_v` / `_h` / `_tm` plus the `predict_y16x16` dispatcher, `predict_uv8x8_dc` / `_v` / `_h` / `_tm` plus the `predict_uv8x8` dispatcher, and the ten-arm `predict_b4x4` sub-block dispatcher) directly with attacker-shaped (`above`, `left`, `p`) triples — the corner of the configuration space the seven existing decode / encode harnesses can never reach (they gate §12 behind a fully-formed reconstruction raster) and the round-258 `intra_predict_dc16` criterion bench leaves cold (it only exercises three of the eleven kernels against a fixed `[128u8; 16] / [129u8; 16]` neighbour pair). The harness sweeps every `Option<&[u8; 16]>` polarity for the DC primitives (including the top-left fallback path where neither edge is present), every variant of `IntraYMode` / `IntraUvMode` / `IntraBmode` against the same input, and re-feeds the 16×16 luma TM output's first row / column into the chroma neighbour slot for a cross-plane kernel-output-as-kernel-input leg. 21-second smoke pass landed `cov: 525, ft: 1300, corp: 31/1892b` across 2 288 663 iterations from an empty seed on aarch64-apple-darwin at 108 983 exec/s, zero panics. Test counts unchanged (no new lib tests). |

## Standalone use (no `oxideav-core`)

### Decode one VP8 frame

`decode_vp8` is the keyframe entry point. For inter-frame sequences
the caller drives the stateful `Vp8DecoderState` that keeps the LAST
/ GOLDEN / ALTREF reference slots:

```rust
use oxideav_vp8::{decode_vp8, Vp8DecoderState};

// One-shot single-frame decode of a VP8 keyframe.
let frame = decode_vp8(&vp8_keyframe_bytes)?;
let (w, h) = (frame.width as usize, frame.height as usize);
assert_eq!(frame.y.len(), w * h);
assert_eq!(frame.u.len(), w.div_ceil(2) * h.div_ceil(2));
assert_eq!(frame.v.len(), w.div_ceil(2) * h.div_ceil(2));

// Multi-frame inter decode — one Vp8DecoderState per stream.
let mut state = Vp8DecoderState::new();
for packet in vp8_frame_packets {
    let frame = state.decode_frame(packet)?;
    // consume frame.y / frame.u / frame.v (8-bit I420, tightly packed)
}
```

`Vp8DecodedFrame` (also re-exported as `Vp8Frame`) carries the visible
`width` / `height` plus three tightly-packed `Vec<u8>` planes.

### Encode one VP8 keyframe

```rust
use oxideav_vp8::{encode_keyframe, I420Frame, KeyframeParams};

let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
let params = KeyframeParams::new(width, height);
let vp8_bytes: Vec<u8> = encode_keyframe(&frame, &params)?;

// The output is a raw VP8 keyframe bitstream (3-byte tag + 7-byte
// start code + partitions). Wrap it in an IVF / RIFF-WEBP / Matroska
// container as needed for your downstream consumer.
```

For a multi-frame inter encoder use `Vp8Encoder` + `Vp8EncoderConfig`
(both reachable standalone) and feed successive `I420Frame`s; see
`tests/encoder_inter_stream.rs` for a worked example.

### Two-pass rate-control encode

```rust
use oxideav_vp8::encoder::{
    first_pass_analyze, two_pass_qindices,
    Vp8TwoPassConfig, Vp8TwoPassEncoder,
};

// First pass: cheap per-frame complexity stats (no encode).
let stats = first_pass_analyze(&i420_frames);

// Second pass: per-frame qindex distribution + actual encode.
let config = Vp8TwoPassConfig::default();        // wraps a base Vp8EncoderConfig
let qindices = two_pass_qindices(&stats, &config)?;
let encoder = Vp8TwoPassEncoder::new(config);
let packets = encoder.encode(&i420_frames, &stats)?;
```

The algorithm distributes per-frame qindex around `config.base.qindex`
so heavier-than-mean frames get lower qindex (better quality) and
lighter frames get higher qindex (smaller bytes), with scene-cut
detection forcing extra-quality keyframes.

### Container helpers

The `ivf` module gives you a standalone IVF reader / writer for the
common case of `*.ivf` test fixtures:

```rust
use oxideav_vp8::ivf::{IvfHeader, parse_header, write_header, write_frame};

let mut out = Vec::new();
let hdr = IvfHeader::vp8(width, height, fps_num, fps_den);
out.extend_from_slice(&write_header(&hdr));
for (pts, frame_bytes) in &timed_packets {
    write_frame(&mut out, *pts, frame_bytes);
}
```

### Quality knob

If you have a `0.0..=100.0` quality scalar (the WebP-canonical
convention) and want the matching VP8 qindex:

```rust
use oxideav_vp8::encoder::quality_to_qindex;
let qindex = quality_to_qindex(75.0);          // → 32
let qindex = quality_to_qindex(100.0);         // → 0   (best)
let qindex = quality_to_qindex(f32::NAN);      // → 127 (worst, safe)
```

### Rate-control trade-off (the `y_ac_qi` knob)

`KeyframeParams::y_ac_qi` (range `0..=127`, default `32`, §9.6) is the
principal rate-control knob on the encoder — every other §9.6 quantiser
delta defaults to 0, so a single dial moves the DC + AC luma / chroma
quantiser bank in lockstep. Lower = higher quality / larger output;
higher = lower quality / smaller output.

The `rate_control_qi_sweep` criterion bench walks the dial across ten
representative values on a fixed 320×240 deterministic source:

```sh
cargo bench -p oxideav-vp8 --bench rate_control_qi_sweep -- --quick
```

Headline trade-off (Apple M4 / aarch64, criterion `--quick`,
output bytes are bench-prologue stderr lines): qi=8 → 1701 B at
8.2 Mpx/s; qi=32 → 595 B at 9.2 Mpx/s; qi=120 → 299 B at 10.5 Mpx/s
(`−83 %` bytes, `+28 %` throughput across the full sweep). See
`BENCHMARKS.md` round-194 section for the complete 10-row table.

### Motion-search descent ladder (the §17.1 / §18.3 inter picker)

`motion_search_descent` attributes wall-time to the three stages
of the per-MB luma MV picker — whole-pixel diamond, half-pixel
ring, quarter-pixel ring — plus a composite full-ladder number:

```sh
cargo bench -p oxideav-vp8 --bench motion_search_descent -- --quick
```

Headline numbers (Apple M4 / aarch64, criterion `--quick`):
whole-pixel `small_diamond_search_luma` ≈ 277 ns,
`half_pixel_refine_luma` ≈ 2.70 µs,
`quarter_pixel_refine_luma` ≈ 2.70 µs,
`full_descent_whole_half_quarter` ≈ 5.68 µs per MB. The
half/quarter §18.3 sixtap synthesis cost dominates the whole-pixel
SAD by ~10×, so any future SIMD fan-out on `mb_luma_sad_at_mv` is
the highest-return target on the §17 search-shape layer.

Round 258 added a `block_sad_16x16_single_pair` leaf bench (~6.27 ns
stable scalar; the SAD primitive every stage of the descent ladder
collapses to once the per-candidate prediction has been
synthesised) and a `block_sad_16x16_simd` partner — `Simd<u8, 16>`
per-row `max - min` absdiff into a `Simd<u16, 16>` row accumulator,
single `reduce_sum()` close, gated behind the existing `simd`
feature, byte-exact against the scalar listing on a 21-input stress
set. The SIMD leaf is **−36 %** in isolation (4.08 ns) but inlining
it into the 16-call-per-MB `mb_luma_sad_at_mv` body pessimised the
half-/quarter-pixel refine stages by **+13 %** each (NEON register
pressure across the surrounding `filter_block_4x4` loop spilling
into LLVM's scheduling), so the public dispatcher continues to
route to `block_sad_16x16_scalar` — same shape as the round-247
dispatch decision for `forward_dct_4x4`. The `_simd` listing stays
compiled + tested as a future re-target target.

### Wide deblock filter (the §15.3 `MBfilter` partner)

Round 260 added `loop_filter_mb_edge`, the partner of the round-170
`loop_filter_normal` bench. The round-170 entry measures the
`subblock_filter` + `simple_segment` per-call cost; the round-260
partner covers the heavier wider `mb_filter` (the §15.3 `MBfilter`
kernel that fires on every MB-to-MB edge — up to twice per
non-skipped luma MB plus the chroma analogues) plus a leaf number
for the §15.2 `common_adjust` inner core that both filters funnel
into:

```sh
cargo bench -p oxideav-vp8 --bench loop_filter_mb_edge -- --quick
```

Headline numbers (Apple M4 / aarch64, criterion `--quick`):
`mb_filter_wide` ≈ 5.8 ns (three decaying `27/18/9` weight
adjustments over six pixels — the heaviest deblock path),
`mb_filter_hev` ≈ 5.4 ns (inner `common_adjust` outer-tap fallback
when `hev` trips), `subblock_filter_low_variance` ≈ 5.0 ns
(head-to-head on the same 8-pixel segment — the round-170 entry
measures the high-variance branch, this one the low-variance
branch so both branches have a baseline), `common_adjust_outer_taps`
≈ 2.9 ns, `common_adjust_no_outer` ≈ 2.6 ns (the §15.2 4-pixel
leaf). The wide MB-edge kernel is ~16 % heavier than the sub-block
kernel per call on the same input, confirming the deblock-path
wall-time picture is two-pronged: future SIMD / unroll work needs
to target both poles. Same input as `loop_filter_normal` so the two
deblock micro-benches compare directly.

## With the OxideAV runtime (`registry` feature on, the default)

```rust
use oxideav_core::RuntimeContext;
use oxideav_vp8::CODEC_ID_STR;                  // = "vp8"

let mut ctx = RuntimeContext::new();
oxideav_vp8::register(&mut ctx);
// ctx.codecs now has a "vp8" Decoder + Encoder factory.

// Or directly call the factories:
use oxideav_vp8::encoder::{make_encoder_with_quality, make_encoder_with_qindex};
let enc = make_encoder_with_quality(&params, 75.0)?;   // Box<dyn Encoder>
let enc = make_encoder_with_qindex(&params, 32)?;
```

Both factories return `Box<dyn oxideav_core::Encoder>` and integrate
with the OxideAV pipeline through `send_frame` / `receive_packet`.

## Clean-room sources

Implementation is derived entirely from the public format spec:

* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).

The two-pass rate-control algorithm is intentionally outside RFC scope
(the spec is silent on rate control by design) — the design is
clean-room, sourced from the in-tree per-MB activity primitives plus
log-MAD + log-variance first-pass statistics.

Fixture `expected.yuv` reference pictures are produced by black-box
invocations of the reference decoder *binary*; no third-party codec
library source is consulted.

## Interop test coverage

The crate ships two interop suites layered on top of the per-stage
unit tests:

* [`tests/standalone_e2e.rs`](./tests/standalone_e2e.rs) — keyframe,
  multi-frame inter, two-pass, IVF container, and `quality_to_qindex`
  end-to-end checks against the `default-features = false` standalone
  surface. Every imported symbol must resolve without the `registry`
  feature; passes under both feature configurations.
* [`tests/ffmpeg_oracle.rs`](./tests/ffmpeg_oracle.rs) — bidirectional
  cross-validation against `ffmpeg` as a black-box oracle. Direction A
  is *our encode → ffmpeg decode* (PSNR-Y ≥ 30 dB on the recovered
  picture); direction B is *ffmpeg encode → our decode* via the
  crate's own `ivf::parse_header` walker (PSNR-Y ≥ 25 dB). Skips
  cleanly via `eprintln! + return` when `ffmpeg` isn't on `$PATH` —
  never `#[ignore]`.

## Fuzz harnesses

The crate ships thirteen `cargo-fuzz` targets under [`fuzz/`](./fuzz/)
that exercise the public encode and decode surface for panic-freedom:

* `panic_free_decode_keyframe` — one-shot `decode_vp8`, dimension-gated
  at 256 × 256.
* `panic_free_decoder_state` — multi-packet `Vp8DecoderState::decode_frame`
  drives the §9.7 LAST / GOLDEN / ALTREF refresh ladder.
* `parse_headers` — `frame_tag::parse_header` / `parse_keyframe_header`,
  `Vp8FrameHeader::parse`, `Vp8CodedHeader::parse` (key + inter — the
  §19.2 segmentation-map / MB-LF-adjustments / token-prob-update /
  MV-prob-update walk), and the IVF framing layer.
* `panic_free_encode_keyframe` — `encode_keyframe(&I420Frame,
  &KeyframeParams)`. Drives both the happy path through the §11 intra
  mode pick → §14 forward transform → §13 token emission → §15
  loop-filter reconstruct chain AND the parameter-rejection surface
  (raw `y_ac_qi` / `loop_filter_level` / `sharpness_level` /
  `nbr_of_dct_partitions` bytes fed unnormalised so the encoder's
  `QuantIndexOutOfRange` / `LoopFilterLevelOutOfRange` /
  `SharpnessLevelOutOfRange` / `InvalidDctPartitionCount` returns are
  also exercised). Dimension-gated to 256 × 256 by normalising the
  width / height bytes into 1..=16 MB-units (16..=256 luma px).
* `panic_free_two_pass_stream` — `Vp8TwoPassEncoder::first_pass_analyze`
  + `Vp8TwoPassEncoder::encode_frame` over a multi-frame sequence
  (round 213). The only public encoder fuzz target that reaches
  `encode_p_frame_multi_ref` (the §9.7 reference-frame refresh ladder,
  the keyframe-vs-Pframe switching state machine, and the
  complexity-aware qindex picker). A scene-cut bitmap and per-frame
  `bits_per_mb` bytes from the input tail drive the force-keyframe
  + qindex-delta envelope even on the first-pass-skipped fallback.
  Frame count capped at 4 and per-axis dimensions at 128 px to bound
  per-iteration memory and wall time. 20-second smoke pass landed
  `cov: 3672, ft: 19072` across 6244 iterations from an empty seed on
  aarch64-apple-darwin, zero panics.
* `panic_free_loopfilter_segment` — the public §15 per-segment loop-filter
  primitives `common_adjust`, `simple_segment`, `subblock_filter`,
  `mb_filter`, plus the §15.4 `LoopFilterParams::derive` parameter
  derivation (round 232). The four decode / encode harnesses above
  reach §15 only through `decode_vp8` / `Vp8DecoderState::decode_frame`
  / `encode_keyframe`, which gate the per-segment kernels behind a
  fully-formed reconstruction raster; this target drives them
  directly with an attacker-shaped `(seg.len(), base)` envelope.
  Both the derived `(hev_threshold, interior_limit, edge_limit)`
  triple from §15.4 AND an independent raw-byte triple feed each
  kernel, with snapshot-and-restore between calls so each primitive
  sees a fresh segment. A chained-pass leg (`simple_segment` →
  `mb_filter`, or `subblock_filter` → `mb_filter`) exercises
  state hand-off across primitives. 21-second smoke pass landed
  `cov: 202, ft: 475, corp: 157/2944b` across 5 819 579 iterations
  from an empty seed on aarch64-apple-darwin, zero panics — the
  primitive-layer kernel runs ~830 × faster per iteration than
  `panic_free_two_pass_stream`.
* `panic_free_token_block` — the public §13.2 / §13.3 token-coding
  primitives `decode_block`, `decode_mb_coeffs`, and
  `merge_default_token_probs`, plus the §13.3 `MbEntropyCtx` above /
  left predictor lattice (round 237). The six harnesses above all
  reach §13 only indirectly through `decode_vp8` /
  `Vp8DecoderState::decode_frame` / `encode_keyframe` /
  `Vp8TwoPassEncoder::encode_frame`, which gate the per-block token
  walk behind a fully-formed frame-header + coded-header + dequant
  state; this target drives the §13 primitive surface directly. An
  attacker-shaped `(probability override list, predictor lattice,
  bool partition)` envelope hits every node of the §13.2 coefficient
  tree, every `Cat1..Cat6` extra-bits leg, every per-position slot of
  `coeff_probs[4][8][3][11]` (default seed AND all-128 seed —
  the §19.2 "every slot replaced by `token_prob_update`" envelope),
  the §13.3 `decode_mb_coeffs` 25-block `(Y2, 16 Y, 4 U, 4 V)` walk
  with both `has_y2` polarities, and the `mb_skip_coeff`
  short-circuit through `MbEntropyCtx::reset_for_skip`. A 9-bit
  bitmap per predictor vector seeds the `MB_ENTROPY_CTX_LEN`
  above / left flags so the §13.3 lattice covers every starting
  state. 21-second smoke pass landed `cov: 1418, ft: 2172, corp:
  296/8566b` across 1 765 349 iterations from an empty seed on
  aarch64-apple-darwin, peak RSS 316 MiB, zero panics — the §13
  primitive surface gets ~7 × the coverage envelope of the §15
  loop-filter target while running at 84 064 exec/s.
* `panic_free_motion_search_descent` — the public §17.1 / §18.3 luma
  motion-search descent ladder (`small_diamond_search_luma`,
  `half_pixel_refine_luma`, `quarter_pixel_refine_luma`) plus the
  per-candidate evaluators `mb_luma_sad_at_whole_mv` /
  `mb_luma_sad_at_mv` (round 256). Drives the (mb-position, mv-center,
  plane-dimension, source-block) envelope into the §20.14 edge-
  replication clamp inside `fetch_block_halo` and across every §18.3
  sixtap fractional offset.
* `panic_free_sixtap_subpel` — the public §18.3 / §20.14 sub-pixel
  synthesis primitives `filter_block_4x4`, `sixtap_2d`,
  `fetch_block_halo`, `fetch_block_whole_pixel`, and
  `filter_set_for_version` (round 257). The motion_search_descent
  target reaches these only through the §17 descent ladder, which by
  construction snaps every per-candidate MV to the half- or
  quarter-pixel grid; the round-225 `motion_comp_subpel_luma`
  criterion bench only exercises a fixed `(mx, my) = (6, 6)` choice
  against a mid-plane MB. This target drives every (mx, my) ∈ {0..7}²
  fractional combination, both filter-set arms (sixtap `version == 0`
  vs bilinear other versions), and every border-position class
  (top-left corner, bottom-right corner, adversarial, mid-plane fast
  path) directly. An 81-byte halo seeded from the input also feeds
  `sixtap_2d` so the convolution sees byte patterns the §20.14 clamp
  would never produce. 26-second smoke pass landed `cov: 181, ft:
  337, corp: 37/846b` across 2 286 492 iterations from an empty seed
  on aarch64-apple-darwin at 87 942 exec/s, zero panics.
* `panic_free_intra_predict_kernels` — the public §12 intra-prediction
  pixel kernels: the four 16×16 luma primitives `predict_y16x16_dc` /
  `_v` / `_h` / `_tm` plus the `predict_y16x16` dispatcher, the four
  8×8 chroma primitives `predict_uv8x8_dc` / `_v` / `_h` / `_tm` plus
  the `predict_uv8x8` dispatcher, and the ten-arm `predict_b4x4`
  sub-block dispatcher across every `IntraBmode` (round 259). The
  seven decode / encode harnesses above reach §12 only indirectly
  through `decode_vp8` / `Vp8DecoderState::decode_frame` /
  `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which gate
  the per-block prediction kernels behind a fully-formed
  reconstruction raster (every neighbour pixel pre-sourced from the
  same valid frame; every `above`-extension pixel for right-edge
  sub-blocks pre-clamped per §12.3). The round-258 `intra_predict_dc16`
  criterion bench only exercises `predict_y16x16_dc` / `_v` / `_h`
  against a fixed `[128u8; 16] / [129u8; 16]` neighbour pair. This
  target drives the §12 primitive surface directly: every
  `Option<&[u8; 16]>` polarity for `predict_y16x16_dc` (top-left
  fallback path included), every variant of `IntraYMode` through the
  16×16 dispatcher (including the `B → None` short-circuit), every
  variant of `IntraUvMode` through the 8×8 dispatcher, and every
  variant of `IntraBmode` through `predict_b4x4` — sweeping every
  assignment-list arm of the six §12.3 diagonal modes (`Ld`, `Rd`,
  `Vr`, `Vl`, `Hd`, `Hu`) over the synthetic `E[0..=8]` array and the
  `above[4..=7]` right-extension pixels. A chained leg re-feeds the
  16×16 luma TM output's first row / column into the chroma neighbour
  pair so a kernel-output-as-kernel-input data-flow shape (cross-plane
  neighbour reuse) is also exercised. 21-second smoke pass landed
  `cov: 525, ft: 1300, corp: 31/1892b` across 2 288 663 iterations
  from an empty seed on aarch64-apple-darwin at 108 983 exec/s, zero
  panics.
* `panic_free_bool_codec` — the public §7 boolean range coder
  primitives driven directly (round 261). Decode side:
  `BoolDecoder::init` (§7.3 `init_bool_decoder` with the 2-byte
  `InputTooShort` rejection) and `BoolDecoder::init_partition`
  (the §20 reference's short-input fallback that tolerates `sz < 2`
  with `value = 0`); plus `read_bool(prob)`, `read_literal(num_bits)`
  ∈ 0..=32, and `read_signed_literal(num_bits)` ∈ 0..=31 (incl. the
  `num_bits == 0 → 0` short-circuit). Encode side: `BoolEncoder::new`,
  `write_bool` / `write_literal` / `write_signed_literal` /
  `write_treed` / `finish` (§7.3 `init_bool_encoder` /
  `add_one_to_output` / `flush_bool_encoder`). An encode-then-decode
  round-trip leg locks the `write_bool` ↔ `read_bool` and
  `write_literal` ↔ `read_literal` halves in §7.3 lockstep against an
  attacker-shaped (op-type, prob, value, num_bits) schedule, so any
  asymmetry in the `split = 1 + (((range - 1) * prob) >> 8)`
  arithmetic or in the `add_one_to_output` carry propagation surfaces
  as a mismatch on the read side. A separate `write_treed` round-trip
  leg encodes a leaf of a 7-entry `kf_ymode_tree`-shaped tree and
  decode-walks it with the same per-node probabilities so the §8.1
  `treed_read` ↔ `write_treed` pair is exercised end-to-end. The ten
  fuzz targets above reach §7 only through `decode_vp8` /
  `Vp8DecoderState::decode_frame` / `Vp8FrameHeader::parse` /
  `decode_block` (decode) or `encode_keyframe` /
  `Vp8TwoPassEncoder::encode_frame` (encode); the bool coder is
  always the final stage and never driven with attacker-shaped
  probability schedules in isolation. This target does. 26-second
  smoke pass landed `cov: 286, ft: 1162, corp: 232/8981b` across
  1 693 989 iterations from an empty seed on aarch64-apple-darwin at
  65 153 exec/s, zero panics.
* `panic_free_transform_4x4_roundtrip` — the public §14 transform /
  dequant / residue-summation primitive layer driven directly (round
  262). Forward DCT (`forward_dct_4x4`) / inverse DCT
  (`inverse_dct_4x4`) and forward WHT (`forward_wht_4x4`) / inverse
  WHT (`inverse_wht_4x4`) round-trip on an attacker-shaped `[i16; 16]`
  residual seed (mid-magnitude / ±255 / ±1023 §14.2 cliff — the
  documented §14.4 inverse-DCT envelope, chosen so the intermediate
  `i32` butterfly multiplies by `SINPI8_SQRT2 = 35468` stay inside
  `i32`); `dequant_block` with attacker-chosen `(dc_factor, ac_factor)`
  cliff values (`i16::MIN` / `i16::MAX` / `0` plus the §14.1 4..=255
  envelope) on a fresh coeff copy — the §14.1 contract's `i32` product
  wrapping cast back to `i16` is panic-free on every cliff triple;
  `add_residue_4x4` / `add_residue` (§14.5 fixed-size vs arbitrary-
  length form, byte-equality assertion on equal-length inputs) against
  the §14.2-bounded residual + a constant predictor;
  `inverse_wht_4x4_dc_only` (§14.3 single-non-zero-DC fast path)
  asserted byte-equal to `inverse_wht_4x4([dc, 0, …, 0])` for every
  `dc ∈ [i16::MIN, i16::MAX]`; the §20.16 `raster_to_scan` permutation
  asserted via multiset equality between input and output; plus the
  `clamp_qindex` (§9.6) and `clamp255` (§14.5) saturating-cap
  primitives at their `i32::MIN` / `i32::MAX` cliff endpoints. The
  eleven fuzz targets above reach §14 only indirectly: the four
  decode-side targets feed inverse-only via well-formed dequantised
  residuals gated by the §9 / §11 / §13 / §14.1 state machine; the
  two encode-side targets feed forward-only via §9.6-clamped residual
  magnitudes determined by the upper-layer encoder; the five
  primitive-layer targets above don't touch §14 at all. No existing
  harness round-trips the §14.3 / §14.4 forward + inverse pair on
  attacker-shaped residuals at the §14.2 cliff envelope, drives
  `dequant_block` with cliff factor pairs, exercises
  `inverse_wht_4x4_dc_only` standalone with cliff `dc`, or asserts the
  §20.16 `raster_to_scan` permutation invariant directly. This target
  does all four. 26-second smoke pass landed `cov: 264, ft: 387, corp:
  48/1836b` across 1 000 000 iterations from an empty seed on
  aarch64-apple-darwin at 250 000 exec/s, zero panics.
* `panic_free_loop_filter_writeback` — the public §9.4 / §19.2
  loop-filter parameter writeback layer of the encoder PLUS the small
  §9.5 / §9.6 / §9.10 / §9.11 sibling writers reached by the same
  §19.2 frame-header walk (round 263). Drives `write_loop_filter`
  (§9.4 baseline) AND `write_loop_filter_with_deltas` (§9.4 +
  §19.2 `mb_lf_adjustments()` full), plus `LoopFilterDeltas::validate`
  (§9.4 per-slot `|v| <= 63` magnitude check) and
  `LoopFilterDeltas::effective` (§15.4 / §20.6 carried-state
  resolution, cross-checked against a hand-rolled oracle on every
  `(enabled, update, per-slot Some|None)` ladder branch),
  `write_quant_indices` (§9.6, with the `y_ac_qi > 127` rejection
  branch reached on every iteration that picks an out-of-range byte),
  `write_token_partition_count` (§9.5, with the `count ∉ {1, 2, 4, 8}`
  [`EncodeError::InvalidDctPartitionCount`] rejection branch
  exhaustively covered across `0..=255`), and `write_mb_no_skip_coeff`
  (§9.10 / §9.11). The twelve fuzz targets above reach the encoder
  writeback layer only indirectly through `encode_keyframe` /
  `Vp8TwoPassEncoder::encode_frame`, which feed NORMALISED parameter
  bytes that the upstream `KeyframeParams` builder clamped against the
  §9.4 / §9.5 / §9.6 fields; the round-261 `panic_free_encode_keyframe`
  reaches `write_loop_filter` via the keyframe encoder which always
  writes `adj_enable = 0`, never `write_loop_filter_with_deltas` and
  never `LoopFilterDeltas::validate` / `::effective`. A round-trip leg
  feeds the `write_loop_filter_with_deltas` output back into
  `BoolDecoder::init` and walks the §19.2 field schedule
  (`filter_type` at prob 128, `L(6)` level, `L(3)` sharp, `L(1)`
  adj_enable, gated `L(1)` update + 4 ref + 4 mode `(present, L(6)
  magnitude, L(1) sign)` slot triples), asserting every read value
  equals what the encoder wrote — any asymmetry between
  `write_loop_filter_with_deltas` and the §19.2 wire layout surfaces
  as a `panic!` from the harness' equality assertion. 26-second smoke
  pass landed `cov: 406, ft: 632, corp: 80/2295b` across 5 235 001
  iterations from an empty seed on aarch64-apple-darwin at 201 346
  exec/s, zero panics.

Initial smoke pass: 800 000 combined iterations on the three decode
targets + 17 500+ iterations on the encode target (2790 coverage edges,
541-input corpus from empty seed in 31 s on aarch64-apple-darwin),
zero panics across the board. See [`fuzz/README.md`](./fuzz/README.md)
for caps, run instructions, and the rationale behind each target's
pre-flight gating.

## License

MIT. See [`LICENSE`](./LICENSE).
