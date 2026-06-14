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
| `simd` | — | **Nightly-only.** Switches the §14.3 / §14.4 4×4 transform primitives — both inverse partners (`inverse_wht_4x4`, `inverse_dct_4x4`) and the §14.3 forward partner (`forward_wht_4x4`) — over to `core::simd::Simd<i32, 4>` rewrites. Every SIMD primitive is byte-exact against the scalar path on a 21-input stress set (DC-only across 10 magnitudes, single-AC at every position, mixed gradients). Headline micro-bench numbers on `aarch64-apple-darwin` (criterion `--quick`): inverse WHT 9.4 → 7.5 ns (≈ −20 %); inverse DCT 10.07 → ~9.5 ns (≈ −1 to −5 %); forward WHT 10.74 → 8.72 ns (≈ −19 %). The §14.4 forward DCT (`forward_dct_4x4`) is intentionally left on the scalar path even under `simd` as of round 247: the lane-wide `c_mul` / `s_mul` multiply-heavy chain + `round_div2_simd` mask + select doesn't pipeline as well as the scalar straight-line code on `aarch64-apple-darwin` (re-measured 11.69 ns SIMD vs 10.67 ns scalar). The `_simd` implementation stays compiled under the `simd` feature with a `cfg(feature = "simd")` byte-equivalence assertion so a future round can re-target the dispatcher on a host where the multiply-heavy SIMD path flips back to a win. Requires a nightly toolchain because `core::simd` is itself nightly. Default (stable) builds use the scalar §14.3 / §14.4 paths unchanged. See `BENCHMARKS.md` for the A/B numbers + the round-170 / round-180 / round-226 / round-247 profiles that motivated each primitive's vectorisation decision. Round 249 reorganised the `forward_dct_4x4_scalar` listing into the canonical `(a1, b1, c1, d1)` partial-sum butterfly shape mirroring §14.4 `inverse_dct_4x4_scalar`, bit-exact against a private `forward_dct_4x4_listing` regression oracle (the unfactored direct-derivation form) on the same 21-input stress matrix — readability / spec-shape parity, no perf change. Round 250 propagates the same canonical butterfly shape into `forward_dct_4x4_simd` (the partner SIMD listing kept compiled under the `simd` feature for the byte-equivalence assertion). The SIMD and scalar §14.4 forward listings now agree visually as well as producing identical bytes; the `fdct_forward_simd_matches_scalar_on_stress_inputs` byte-exact assertion on the 21-input stress matrix continues to pass. Round 251 closes the §14.3 side of the same shape-parity work: `forward_wht_4x4_scalar` and `forward_wht_4x4_simd` are reorganised into the canonical `(a1, b1, c1, d1)` butterfly form mirroring the §14.3 inverse listing `inverse_wht_4x4` line-for-line — only the final-pass rounding differs (`round_div2(x)` forward vs `(x + 3) >> 3` inverse). A new `fwht_scalar_matches_direct_derivation_listing` regression guard anchors the refactored scalar listing against the unfactored direct-derivation form on the same 21-input stress matrix. Test counts: stable lib 454, nightly + `simd` lib 455 (each +1 over round 250). Round 252 lands an in-crate §13.4 walk-order regression anchor (`encoder::tests::write_no_token_prob_updates_matches_all_none_against_spec_flag_probs`) that strengthens the byte-equivalence assertion between `write_no_token_prob_updates` and `write_token_prob_updates(all-None)` by replacing the placeholder `[128u8; 1056]` flag-probability table with the actual §13.4 `COEFF_UPDATE_PROBS_FLAT` spec table — closes a gap where a future divergent-writer refactor could pass the external placeholder test (`tests/encoder_token_prob_updates.rs`) while drifting on the extreme-probability splits (`5`, `255`) the real §13.4 `coeff_update_probs[4][8][3][11]` table contains. Test counts: stable lib 455, nightly + `simd` lib 456 (each +1 over round 251). Round 253 extends the same anchor pattern to the §17.2 `mv_prob_update()` writer path: a new in-crate test `encoder::tests::mv_update_probs_flat_matches_spec_table` byte-equates the encoder-side `MV_UPDATE_PROBS_FLAT` (the 38-entry flat copy each inter-frame entry-point walks when emitting the `mv_prob_update()` all-zero F-gate block) against the canonical 2×19 `coded_header::MV_UPDATE_PROBS` spec table that `parse_mv_prob_update` reads each F flag at (transcribed from RFC 6386 §17.2 `vp8_mv_update_probs[2]`). Catches a transcription drift between the encoder's flat copy and the decoder's spec-side table at constants level, before any self-roundtrip encode would notice. `coded_header::MV_UPDATE_PROBS` is promoted from private to `pub(crate)` for the anchor; no public-surface change. Test counts: stable lib 456, nightly + `simd` lib 457 (each +1 over round 252). Round 254 reaches further into the §13 token-coder by anchoring the encoder's six per-cat extra-bits probability lists and category-offset array against the literal RFC 6386 §13.2 spec listing: a new in-crate test `encoder::tests::enc_pcat_and_cat_base_match_spec_listing` byte-equates each of `ENC_PCAT1..ENC_PCAT6` (the terminator-stripped probability prefixes `cat_extras()` returns for cat1..cat6 tokens) and the 6-entry `ENC_CAT_BASE` array against the spec `Pcat1..Pcat6 = {159,0} / {165,145,0} / {173,148,140,0} / {176,155,140,135,0} / {180,157,141,134,130,0} / {254,254,243,230,196,177,153,140,133,130,129,0}` and `categoryBase[6] = {5,7,11,19,35,67}` literals. Per-list length checks catch a future regression that swept the trailing `0` terminator into the slice; a cross-check against `cat_extras()` catches a match-arm reorder; a non-cat-token sweep catches a regression that started returning `Some(...)` for `Dct0..Dct4` or `Eob`. Lib-test only — no public surface change. Test counts: stable lib 457, nightly + `simd` lib 458 (each +1 over round 253). Round 258 adds the §17 SAD primitive partner: `block_sad_16x16_simd` pulls the 16×16 (src, pred) pair through `Simd<u8, 16>` per-row `max - min` absdiff into a `Simd<u16, 16>` row accumulator with a single `reduce_sum()` close, byte-exact against the scalar listing on a 21-input stress set (`block_sad_simd_matches_scalar_on_stress_inputs`). The new `motion_search_descent/block_sad_16x16_single_pair` leaf bench measures the SIMD leaf at 4.08 ns (−36 % vs 6.43 ns scalar) in isolation, but inlining it into the 16-call-per-MB `mb_luma_sad_at_mv` body regresses `half_pixel_refine_luma_8_offsets` / `quarter_pixel_refine_luma_8_offsets` by ~+13 % (NEON register pressure spilling into LLVM's scheduling around the surrounding `filter_block_4x4` loop), so the public `block_sad_16x16` dispatcher continues to route to the scalar partner — same shape as round-247 `forward_dct_4x4`. The `_simd` listing stays compiled under the `simd` feature for the byte-equivalence proof and as a future re-target target (e.g. on a host where the regression flips, or with an `#[inline(never)]` wrapper that prevents the LLVM scheduling spill). Test counts: stable lib 458, nightly + `simd` lib 460 (each +1 / +2 over round 257; the nightly +2 picks up both the public-dispatch equivalence test and the direct `block_sad_16x16_simd`-vs-scalar equivalence test). Round 259 adds the §12 intra-prediction primitive-layer fuzz target `panic_free_intra_predict_kernels`: drives the eleven public §12 kernels (`predict_y16x16_dc` / `_v` / `_h` / `_tm` plus the `predict_y16x16` dispatcher, `predict_uv8x8_dc` / `_v` / `_h` / `_tm` plus the `predict_uv8x8` dispatcher, and the ten-arm `predict_b4x4` sub-block dispatcher) directly with attacker-shaped (`above`, `left`, `p`) triples — the corner of the configuration space the seven existing decode / encode harnesses can never reach (they gate §12 behind a fully-formed reconstruction raster) and the round-258 `intra_predict_dc16` criterion bench leaves cold (it only exercises three of the eleven kernels against a fixed `[128u8; 16] / [129u8; 16]` neighbour pair). The harness sweeps every `Option<&[u8; 16]>` polarity for the DC primitives (including the top-left fallback path where neither edge is present), every variant of `IntraYMode` / `IntraUvMode` / `IntraBmode` against the same input, and re-feeds the 16×16 luma TM output's first row / column into the chroma neighbour slot for a cross-plane kernel-output-as-kernel-input leg. 21-second smoke pass landed `cov: 525, ft: 1300, corp: 31/1892b` across 2 288 663 iterations from an empty seed on aarch64-apple-darwin at 108 983 exec/s, zero panics. Test counts unchanged (no new lib tests). Round 267 extends the `simd` feature onto the §14.1 dequantize hot path: `src/dequant.rs` `dequant_block` becomes a scalar/SIMD dispatcher, with `dequant_block_simd` (`#[cfg(feature = "simd")]`) widening the `i16` 4×4 block to `Simd<i32, 16>`, multiplying lane-wise against a per-lane factor vector (`dc_factor` in lane 0, `ac_factor` in lanes 1..=15), and truncating each product back to `i16` with `cast::<i16>()` — int→int `cast` truncates exactly like the scalar `as i16`, including the i16-overflow wrap. The 16-wide layout maps the sixteen fully-independent coefficient×factor multiplies onto a single vector with no cross-lane dependency, so the byte-equivalence is unconditional. `dequant_block_simd_matches_scalar_on_stress_inputs` asserts dispatcher == scalar across all-zero, DC-only, single-AC-per-lane, mixed-sign, and i16-overflow blocks (`[i16::MAX; 16] × 440`, `[i16::MIN; 16] × 440`, and a near-extreme 16-lane pattern) — the overflow fixtures are what distinguish truncating `cast` from a saturating one. Verified passing on nightly 1.97 + `simd` and on stable (scalar-vs-scalar). Test counts: stable lib 459, nightly + `simd` lib 461 (each +1 over round 259; nightly +1 for the dispatcher-vs-scalar equivalence test under `simd`). Round 268 extends the `simd` feature onto the §12.2 TM_PRED intra kernel — the only §12.2 mode with per-pixel arithmetic (`X_{rc} = clamp255(L_r + A_c - P)`; DC / V / H are fills and row copies the compiler already vectorises): `src/intra_predict.rs` `predict_tm` becomes a scalar/SIMD dispatcher, with `predict_tm_simd::<N>` (`#[cfg(feature = "simd")]`, N = 16 luma / 8 chroma) hoisting the row-invariant column term `A_c - P` into a `Simd<i16, N>` vector and emitting each row as splat-add + `simd_clamp(0, 255)` + `u8` narrow. Every intermediate lies in `-255..=510` so the `i16` lanes reproduce the scalar `i32` arithmetic exactly; byte-equivalence enforced at both widths across the clamp endpoints, ramps, alternating extremes, and deterministic LCG triples (`predict_tm_simd_matches_scalar_on_stress_inputs`, plus a public-entry-point anchor `predict_tm_public_entry_points_route_through_dispatcher`). New `predict_y16x16_tm` entry in the `intra_predict_dc16` bench: 44.15 ns scalar → 5.46 ns SIMD (**−87.7 %**) on `aarch64-apple-darwin` — the largest per-kernel SIMD delta in the crate so far. Test counts: stable lib 461, nightly + `simd` lib 463 (each +2 over round 267). Round 269 extends the `simd` feature onto the §18.3 / §20.14 six-tap sub-pixel interpolation kernel — the round-170 inter-encode profile's #4 self-time symbol: `src/motion_comp.rs` `sixtap_2d` becomes a scalar/SIMD dispatcher, with `sixtap_2d_simd` (`#[cfg(feature = "simd")]`) computing each convolution row's four §18.3 `interp` dot products as one `Simd<i32, 4>` vector (tap k's support lanes are the contiguous run `halo[r*9 + k ..][..4]`) and keeping the horizontal pass's clamped intermediate resident in `i32` vectors so the vertical pass runs with zero loads. The lanes must be `i32`: the §18.3 dot product over `u8` support spans `[-8160, 40800]` (the ½-displacement row `{3, -16, 77, 77, -16, 3}` has positive-tap sum 160 → 160·255 > `i16::MAX`), so the round-170 candidate note's `Simd<i16, 8>` stripe wraps — a parity-split two-accumulator `i16×8` variant was implemented and measured during the round but benched no better than scalar (and ~+15 % worse on `filter_block_4x4`), so the four-lane `i32` form shipped. Byte-equivalence enforced across 13 halos × all 64 `(mx, my)` fraction pairs × both §18.3 filter sets (`sixtap_2d_simd_matches_scalar_on_stress_inputs`) plus both dot-product extremes (`sixtap_2d_accumulator_extremes_match_scalar`). `motion_comp_subpel_luma` bench, nightly scalar → nightly SIMD: `mb_sixtap_2d_16x4x4` 271.5 → 248.5 ns (**−8.5 %**), `filter_block_4x4_sub3x5` 24.87 → 23.55 ns (**−5.3 %**). Test counts: stable lib 463, nightly + `simd` lib 465 (each +2 over round 268). Round 270 lands the round-269 BENCHMARKS candidate "MB-scale §18.3 batching": all sixteen luma sub-blocks of a non-SPLITMV inter MB share one motion vector (§18.1), so `predict_inter_mb`'s sub-pixel luma path now fetches one 21×21 halo (`fetch_luma_mb_halo`) and synthesises the whole 16×16 luma block in a single two-pass convolution (`sixtap_mb_luma`) instead of sixteen overlapping 9×9 `fetch_block_halo` + `sixtap_2d` calls. The SIMD partner `sixtap_mb_luma_simd` (`#[cfg(feature = "simd")]`) widens each pass to `Simd<i32, 16>` — one sixteen-lane vector per output row (tap k's lanes are the contiguous run `halo[r*21 + k ..][..16]`), the horizontal-pass intermediate staying resident in `i32` vectors so the vertical pass runs with zero loads. Byte-exact against the per-sub-block path (`sixtap_mb_luma_matches_per_subblock_path`) and against the scalar listing (`sixtap_mb_luma_simd_matches_scalar_on_stress_inputs`), with the MB-halo border clamp cross-checked against `fetch_block_halo` (`fetch_luma_mb_halo_matches_subblock_halos_in_bounds`, `fetch_luma_mb_halo_clamps_at_top_left_corner`) and through a real corner-MB prediction (`predict_inter_mb_sub_pixel_at_border_uses_mb_halo_clamp`). New `motion_comp_subpel_luma/mb_luma_batched_16x16` bench, nightly: scalar batched 158.8 ns and SIMD batched 140.2 ns vs the per-sub-block partner `mb_luma_per_subblock_16x16` ≈ 260–268 ns — **−47 %** end-to-end (batched SIMD vs per-sub-block) and **−12 %** SIMD-over-scalar on the batched path. Test counts: stable lib 468, nightly + `simd` lib 470 (each +5 over round 269 — four MB-scale equivalence / clamp tests plus the SIMD-vs-scalar stress test). Round 271 lands the round-270 BENCHMARKS candidate "MB-scale §18.3 chroma batching" — the chroma analogue of the round-270 luma path: the four chroma sub-blocks of each plane on a non-SPLITMV inter MB share one §18.1 averaged motion vector, so `predict_inter_mb`'s sub-pixel chroma path now fetches one 13×13 halo (`fetch_chroma_mb_halo`) and synthesises the whole 8×8 chroma block in a single two-pass convolution (`sixtap_mb_chroma`) instead of four overlapping 9×9 `fetch_block_halo` + `sixtap_2d` calls per plane. The SIMD partner `sixtap_mb_chroma_simd` (`#[cfg(feature = "simd")]`) widens each pass to `Simd<i32, 8>` — one eight-lane vector per output row (tap k's lanes are the contiguous run `halo[r*13 + k ..][..8]`), the horizontal-pass intermediate staying resident in `i32` vectors so the vertical pass runs with zero loads. Byte-exact against the per-sub-block path (`sixtap_mb_chroma_matches_per_subblock_path`) and the scalar listing (`sixtap_mb_chroma_simd_matches_scalar_on_stress_inputs`), with the MB-halo border clamp cross-checked against `fetch_block_halo` (`fetch_chroma_mb_halo_matches_subblock_halos_in_bounds`, `fetch_chroma_mb_halo_clamps_at_top_left_corner`) and through real mid-plane + corner-MB predictions on both U and V (`predict_inter_mb_chroma_sub_pixel_matches_per_subblock_path`, `predict_inter_mb_chroma_sub_pixel_at_border_uses_mb_halo_clamp`). New `motion_comp_subpel_luma/mb_chroma_batched_8x8` bench, nightly: scalar batched 43.9 ns and SIMD batched 38.6 ns vs the per-sub-block partner `mb_chroma_per_subblock_8x8` ≈ 67.7 ns — **−43 %** end-to-end (batched SIMD vs per-sub-block) and **−12 %** SIMD-over-scalar on the batched path. Test counts: stable lib 474, nightly + `simd` lib 476 (each +6 over round 270 — five MB-scale equivalence / clamp / real-prediction tests plus the SIMD-vs-scalar stress test). Round 272 lands the round-271 BENCHMARKS candidate "whole-pixel non-SPLITMV MB batching" — the whole-pixel analogue of the round-270 / round-271 sub-pixel MB-batching work, and a pure-scalar gather amortisation (no SIMD): when the shared §18.1 vector of a non-SPLITMV inter MB is whole-pixel (`mv & 7 == 0` per component) the §18.3 prediction is a copy, so the whole 16×16 luma / 8×8 chroma block is one contiguous source region. `predict_inter_mb`'s whole-pixel luma branch now issues one `fetch_luma_mb_whole_pixel` (16×16, stride 16) and each chroma plane one `fetch_chroma_mb_whole_pixel` (8×8, stride 8) instead of sixteen / four 4×4 `fetch_block_whole_pixel` copies — one bounds check + one border-straddle decision per MB rather than per sub-block, both with the same in-bounds contiguous-row fast path / per-pixel §20.14 `build_mc_border` clamp fallback. Byte-exact against the per-sub-block assembly five ways (`fetch_luma_mb_whole_pixel_matches_per_subblock_in_bounds`, `fetch_chroma_mb_whole_pixel_matches_per_subblock_in_bounds`, `fetch_luma_mb_whole_pixel_clamps_at_top_left_corner`, `fetch_chroma_mb_whole_pixel_clamps_at_bottom_right_corner`, and a real corner-MB prediction `predict_inter_mb_whole_pixel_at_border_uses_mb_batch_clamp` covering Y + U + V), with the existing `reconstruct_inter_mb_matches_legacy_for_whole_pixel` still anchoring the full reconstruct path. New `motion_comp_subpel_luma/mb_*_whole_pixel_*` benches (Apple M4 / aarch64, criterion `--quick`): whole 16×16 luma copy 46.89 → 13.13 ns (**−72 %**), whole 8×8 chroma copy 8.49 → 4.74 ns (**−44 %**). Test counts: stable lib 479 (+5 over round 271). |

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

### Decoder-side whole-frame benches (round 282)

Round 282 closed the two decoder-side bench gaps and refreshed the
full regression table (see `BENCHMARKS.md` round 282):

```sh
cargo bench -p oxideav-vp8 --bench inter_decode_short_clip
cargo bench -p oxideav-vp8 --bench loop_filter_frame -- --quick
```

* `inter_decode_short_clip/inter_decode_4f_128x128_qi32` —
  `Vp8DecoderState` replaying the 1-keyframe + 3-P-frame stream the
  `inter_encode_short_clip` clip encodes to: **188 µs (348 Mpx/s)
  stable, 162 µs (405 Mpx/s) under nightly + `simd`** on Apple M4 /
  aarch64. The decode half of the §16 inter roundtrip finally has a
  whole-frame baseline next to `keyframe_decode` (154 µs stable /
  123 µs simd at 320×240).
* `loop_filter_frame/*` — the whole-frame §15 / §20.6 deblock pass
  (`filter_frame` keyframe normal ≈ 335 µs and simple ≈ 78 µs,
  `filter_inter_frame` normal ≈ 347 µs at 320×240, every MB coded —
  the fully-coded worst case the synthetic decode streams never
  reach).
* The round-282 `sample(1)` decoder profiles rank the next decoder
  candidates: §13 token decode (`decode_block`, ≈ 31 % of inter
  decode under `simd`), per-frame reference-slot plane copying in
  `Vp8DecoderState` (≈ 11–14 %), and the §13.4
  `parse_token_prob_update` per-frame fixed cost (≈ 5 %).

### Fused §13 token descent + batched renormalisation (round 283)

Round 283 took that ranked list's #1. The §13.2 coefficient-token
descent is now written out branch-by-branch over the fixed
`coeff_tree` (no per-step tree-table loads, no token-enum
re-dispatch; the "skip dct_eob after DCT_0" rule is an inner
zero-run loop), the per-macroblock walk lands each coefficient in
its raster slot as it is decoded (the §20.16 zig-zag table is the
write order — the scratch block + permute + copy pass are gone), and
the §7.3 bool-decoder renormalisation collapses its bit-at-a-time
doubling loop into one `leading_zeros`-derived shift, which every
header/mode/MV/token bool in the decoder shares. Whole-frame:
**−7.2 % / −9.7 %** keyframe decode and **−7.8 % / −8.5 %** inter
decode (stable / nightly + `simd`), decoded output bit-identical
(two equivalence-anchor tests + a 55-frame decode-side byte-hash
A/B on both toolchains — see `BENCHMARKS.md` round 283).

### Hot-path harnesses + ranked hotspot table (round 285)

Round 285 (bench-only, no `src/` change) gave the r283/r284 hot paths
their own A/B instruments and re-ranked both pipelines with fresh
`sample(1)` profiles (see `BENCHMARKS.md` round 285):

```sh
cargo bench -p oxideav-vp8 --bench token_decode
cargo bench -p oxideav-vp8 --bench reconstruct_inter_mb -- --quick
cargo bench -p oxideav-vp8 --bench subpel_sad_scoring -- --quick
```

* `token_decode` — the fused §13.2 descent in isolation
  (`decode_mb_coeffs` over self-encoded, setup-verified 64-MB token
  partitions: dense ≈ 7.8 µs/MB, sparse ≈ 0.75 µs/MB) plus an
  inter-heavy 1K + 11P 176×144 whole-stream decode (1.36 ms stable /
  1.09 ms simd, 224 / 279 Mpx/s).
* `reconstruct_inter_mb` — the round-283 #1 decoder symbol in its
  three shapes (sub-pixel full-residue 569/464 ns, whole-pixel
  424/344 ns, skip 280/273 ns stable/simd); the decomposition shows
  ~95 % of the whole-pixel cost in the §14.4/§14.5 residue chain.
* `subpel_sad_scoring` — one §17 sub-pixel SAD candidate ≈ 184 ns
  (fetch 22 / convolve ≈ 156 / SAD ≈ 6), the micro-bench the standing
  fused-row-SAD candidate required.
* Ranked tables (decoder + encoder) land in `BENCHMARKS.md`; the named
  next profile-opt target is **`reconstruct_inter_mb` §14.4/§14.5
  residue fusion** (IDCT output pass fused with the add-clamp over the
  strided prediction raster, dropping the per-sub-block
  `extract_4x4`/`insert_4x4` scratch round-trips).

### Fused §14.4 IDCT + §14.5 add-clamp residue pass (round 286)

Round 286 lands the round-285 named target. The per-sub-block
`inverse_dct_4x4` → `extract_4x4` → `add_residue_4x4` → `insert_4x4`
four-buffer round-trip in all three inter-reconstruction entry points
(`reconstruct_inter_mb`, `reconstruct_inter_mb_whole_pixel`,
`reconstruct_split_mv_mb`) collapses into one `inverse_dct_4x4_add_into`
helper that folds the second inverse-DCT pass with the §14.5 add-clamp
written straight into the strided prediction raster.

Output is bit-identical (every in-tree inter + keyframe fixture still
decodes byte-exact, plus a randomized fused-vs-unfused equivalence test
on both paths). On the SIMD path the store folds lane-wide, measuring
**−8 %** sub-pixel / **−12 %** whole-pixel on the per-MB
`reconstruct_inter_mb` bench (residue-pass-only delta −16 % / −58 %).
The scalar (default-CI) path keeps the contiguous-buffer sequence behind
the same helper — a strided fold regressed it there (the round-274
scratch-then-copy lesson), so the default build is unchanged. A/B bench:
`cargo bench -p oxideav-vp8 --bench idct_add_residue_fusion`. Next
target: per-frame reference-slot plane-copy churn (see `BENCHMARKS.md`
round 286).

### Decode profile + §14.4 DC-only fast-path A/B (round 297)

A whole-keyframe decode profile (`sample` over a tight `decode_vp8`
loop on a 320×240 qi=32 keyframe built by `encode_keyframe`) ranked
`inverse_dct_4x4_scalar` the **#1** top-of-stack symbol of the decode
path, ahead of `decode_keyframe_mb_non_bpred`, `decode_block_core`,
and the `predict_y16x16_tm` intra kernel. The existing
`inverse_transform_4x4` micro-bench measures only the full high-AC
transform; on this shared/saturated host the isolated single-call
micro-bench swings ±20 % run-to-run, so no register-form scalar
rewrite cleared the noise floor and the §14.4 scalar listing is left
byte-for-byte unchanged.

Instead, round 297 adds an `idct_dc_only` A/B group to the
`idct_add_residue_fusion` bench that isolates the residue shape a real
decode actually produces most: the **DC-only** sub-block (every AC
coefficient zero), which `inverse_dct_4x4_add_into` short-circuits
through its §14.4 DC-only fast path (`(input[0]+4)>>3` added uniformly,
no butterfly). Measured per-MB (24 sub-blocks, same binary, immune to
cross-run drift): **~110 ns** DC-only vs **~254 ns** full-butterfly —
the fast path is ~2.3× cheaper, quantifying the win it already buys.
A/B bench: `cargo bench -p oxideav-vp8 --bench idct_add_residue_fusion`.

### `ref_slot_rotation` bench — §9.7/§9.8 reference-slot rotation (round 288)

Round 288 (bench-only, no `src/` change) gives the round-286 runner-up —
the §9.7/§9.8 reference-frame slot rotation (`RefFrameSlot::clone` churn,
the `memmove`/`memset` #3 decoder self-time cluster at ≈ 15 %) — its
first isolated A/B instrument:

```sh
cargo bench -p oxideav-vp8 --bench ref_slot_rotation -- --quick
```

* `rotate_*` micro — the §20 page-147 rotation walk in isolation across
  three flag combinations: even the common `refresh_last`-only ladder
  costs **~12.2 µs/frame** of slot copying (six populated `Vec` clones);
  `refresh_all` ~15.2 µs, cross-slot `copy_gf_arf` ~16.2 µs.
* `decode_1k8p_*` whole-stream — decode through `Vp8DecoderState` under a
  refresh-last (1.86 ms) vs golden/altref-cadence (1.89 ms) stream; the
  ~1.3 % gap confirms the rotation is a near-**fixed** per-frame cost
  driven by the unconditional clones, not refresh frequency.
* Named next profile-opt target: **copy-on-write / slot-swap planes**
  (`Rc`/`Arc`-backed slots so `pre_*` capture + GOLDEN/ALTREF
  pass-through become pointer bumps, or `mem::swap` of the owned current
  reconstruction into LAST) — byte-identity provable with the existing
  decode-side byte-hash A/B. See `BENCHMARKS.md` round 288.

### Move-minimising reference-slot rotation (round 289)

Round 289 lands the round-288 named target. `Vp8DecoderState::decode_frame`
staged the §9 slot rotation by cloning every input — the just-decoded
frame (`current_slot = planes.clone()`), the three entry slots (`pre_*`),
and again into each refreshed destination. The rotation is a pure
*select-and-replace* over whole slots, so it now resolves each new
`LAST`/`GOLDEN`/`ALTREF` to a symbolic source (`rotate_reference_slots`)
and **moves** each owned source into its destination, cloning only when a
source genuinely feeds more than one slot. The visible-cropped output is
built from `planes` first so the just-decoded slot consumes `planes` by
move. The common `refresh_last`-only ladder now does **zero** plane copies
(the current frame moves into LAST; GOLDEN/ALTREF pass through by move),
where it previously did six populated `Vec` clones. The keyframe path
similarly moves `planes` into one of its three slots.

Output is bit-identical: the rotation only selects and replaces whole
slots. Anchored by `rotation_matches_clone_everything_reference` (every
entry-slot population × every refresh-control combination, asserting
byte-equality against the prior clone-everything ladder) plus the full
roundtrip/oracle suite (490 stable / 492 nightly+simd lib tests, the
`i_frame_then_p_frame` / `golden_update_cycle` / `altref_arnr` bit-exact
decode tests, and the `ffmpeg_oracle` black-box validator). Measured
**−9.3 %** (refresh-last) / **−10.6 %** (golden+altref cadence)
whole-stream on the round-288 `ref_slot_rotation/decode_1k8p_*` benches
(363 → 404 and 352 → 403 Melem/s; criterion p < 0.05, same-session A/B).
A/B bench: `cargo bench -p oxideav-vp8 --bench ref_slot_rotation -- 'decode_1k8p'`.

### §13.4 `token_prob_update()` lockstep flag-read loop (round 291)

Round 291 lands the round-288/289 named decoder rank-5 target — the §13.4
`coded_header::parse_token_prob_update` per-frame flag-read loop (≈ 5 %
decoder self-time; 1056 update flags = 4×8×3×11, each read at its
position-specific `COEFF_UPDATE_PROBS` probability). The loop carried a
4-deep `enumerate()` whose `(i,j,k,t)` indices existed only to re-index
`COEFF_UPDATE_PROBS[i][j][k][t]` — four bounds checks + index arithmetic
per flag, re-traversing the table the loop already walked structurally.
The fixed walk order lets the output array and probability table be zipped
in exact lockstep, so the inner read indexes neither array.

Output is bit-identical: same flags, same order, same per-position
probabilities. Anchored by `parse_token_prob_update_matches_indexed_reference`
(a mixed `Some`/`None` payload exercising the `L(8)` literal branch across
all four planes, byte-equal + same-stream-position against a verbatim copy
of the pre-291 4-level-indexed loop) plus the full lib suite (491 stable /
493 nightly+simd), all 37 roundtrip/decode integration binaries, and the
`ffmpeg_oracle` validator — decoded bytes unchanged across the corpus.
Measured **−10.6 %** (`parse_no_updates`, the common no-update frame) /
**−15.7 %** (`parse_sparse_updates`) on the new `token_prob_update`
micro-bench (criterion p < 0.05, same-session A/B). A/B bench:
`cargo bench -p oxideav-vp8 --bench token_prob_update`. See `BENCHMARKS.md`
round 291.

Round 294 lands a §14.4 **DC-only IDCT-add fast path** — the round-294
profile's rank-2 decoder symbol (`inverse_dct_4x4_add_into`, ≈ 16 % of
inter decode). The §14.4 inverse DCT + §14.5 add-clamp was applied to
every coded residual sub-block, including the very common case where
only the DC coefficient is non-zero. For a DC-only block both separable
passes carry only the DC term, so the full transform collapses to a
single uniform residue `(input[0] + 4) >> 3` added (clamped) to all
sixteen predictor pixels — the butterfly, transpose, and SIMD lane work
are skipped entirely (and an all-zero rounded DC returns with the
prediction untouched). The guard sits in the shared dispatcher so both
the scalar and `simd` paths benefit; the general path is unchanged for
any block with a non-zero AC coefficient. Output is bit-identical:
`dc_only_add_into_matches_general_path` sweeps every DC magnitude across
the §14.2 envelope (both clamp saturations) at every strided sub-block
position against an independent full-`inverse_dct_4x4` → add-clamp
reference; the full lib suite (493 stable / 495 nightly+simd), the
roundtrip/decode integration binaries, and a 10-fixture decode-side
byte-hash A/B (162 048 plane bytes, FNV-1a `cf33bace1d44adff`, identical
pre/post on both stable scalar and nightly + `simd`) all confirm the
decoded corpus is unchanged. Measured **−14.0 %** on whole-frame inter
decode (`inter_decode_4f_128x128_qi32`, 530 → 616 Mpx/s; four
non-overlapping A/B pairs); the textured token-heavy 12-frame stream is
noise-bounded at −0.4 %. See `BENCHMARKS.md` round 294.

Round 295 adds `bool_decoder_read`, the first bench to isolate the §7.3
boolean entropy decoder primitive itself — `BoolDecoder::read_bool` /
`read_literal` and the batched renormalisation step, the most-invoked
decode primitive (every coded bit flows through it), previously measured
only folded inside the §13 token descent and whole-frame decode. Three
regimes, each driven by a partition produced by the crate's own §7.3
`BoolEncoder` and decoded once at setup with a full bit/byte-equality
assertion: `read_bool_skewed_64k` (prob 248, renorm fast path) 151.8 µs,
`read_bool_balanced_64k` (prob 128 fair coin, renorm worst case) 182.3 µs,
and `read_literal_8b_8k` (the §9 / §17 `L(n)` idiom) 152.8 µs. Headline
finding: the fair-coin regime is **≈ 20 % slower per bool** (≈ 2.78 ns vs
≈ 2.32 ns), so the renormalisation shift + byte-refill is the dominant
per-bit cost; `read_literal` adds no measurable overhead beyond its
constituent `read_bool` calls. Measurement-only — no decode-path change,
the full lib suite is unchanged. A/B bench: `cargo bench -p oxideav-vp8
--bench bool_decoder_read`. See `BENCHMARKS.md` round 295.

Round 298 adds `dequantize_mb`, isolating the §14.1 dequantization layer
— previously measured only folded inside the whole-frame decode benches.
Two layers: the §20.4 `dequant_init` factor derivation
(`MbDequantFactors::from_quant_indices` per-frame ≈ 2.32 ns +
`for_segment` per-segment ≈ 2.9 ns — six `dc_qlookup`/`ac_qlookup` reads
through `clamp_qindex` plus the `*2` / `*155/100` scalings and `>132` /
`<8` clamps) and the per-MB apply (`MbDequantFactors::dequantize`, 400
coefficient×factor multiplies over all twenty-five 4×4 blocks). Headline
finding: the apply is **occupancy-independent** — sparse (183.0 ns) and
dense (184.1 ns) residual macroblocks cost within 1 %, because the scalar
`dequant_block` walks all sixteen lanes of every block unconditionally
(`0 * factor` is still a multiply-and-store). That fixed-cost shape is
exactly what the optional `core::simd` `dequant_block` path (round 267)
maps onto. Measurement-only — no decode/encode-path change, the full lib
suite is unchanged. A/B bench: `cargo bench -p oxideav-vp8 --bench
dequantize_mb`. See `BENCHMARKS.md` round 298.

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

The crate ships twenty-four `cargo-fuzz` targets under [`fuzz/`](./fuzz/)
that exercise the public encode and decode surface for panic-freedom
(plus, where a target carries an equivalence leg, byte-exact agreement
between the paired surfaces):

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
* `panic_free_inter_mb_reconstruct` — the §16 inter-MB reconstruction
  surface driven directly: `reconstruct_inter_mb_whole_pixel` (§16.2
  non-SPLITMV whole-pixel), `reconstruct_inter_mb` (§16.2 / §18.3 full
  sub-pixel), and `reconstruct_split_mv_mb` (§16.4 SPLITMV) plus their
  `predict_*` residue-free counterparts (round 265). The thirteen
  fuzz targets above reach §16 only indirectly through `decode_vp8` /
  `Vp8DecoderState::decode_frame` / `encode_p_frame_multi_ref` /
  `Vp8TwoPassEncoder::encode_frame`, which gate the inter-MB path
  behind a fully-formed previous keyframe + §9.7 reference refresh
  state machine; the round-256 `panic_free_motion_search_descent` and
  round-257 `panic_free_sixtap_subpel` targets reach the §18.3
  sub-pixel synthesis primitive layer but never the §16 macroblock-
  level reconstruction orchestrator; the round-262
  `panic_free_transform_4x4_roundtrip` target reaches §14 but never
  feeds the residue into a §16 reconstruct call. This target drives
  the §16 orchestrator directly with attacker-shaped `(mb_col, mb_row,
  luma_mv, full_pixel, mb_skip_coeff, y2_coeffs, y_coeffs, u_coeffs,
  v_coeffs)` tuples and cross-checks (a) the §18.1 fractional gate
  against the dispatcher's `MotionCompError::SubPixelNotSupported`
  return on the whole-pixel path, and (b) the §11.1 `mb_skip_coeff`
  short-circuit (`reconstruct == predict` byte-equal on every input
  that sets the skip flag) on all three §16 paths. Also drives
  `select_ref_frame` over a short attacker partition (every `RefFrame`
  variant reachable) and the §18.1 vector-adjustment primitives
  (`stored_luma_mv`, `chroma_mv`, `apply_full_pixel`,
  `whole_pixel_fraction_is_zero`, `chroma_idx_for_luma_subblock`,
  `split_chroma_mvs`, `filter_set_for_version`) on every iteration.
  25-second smoke pass landed `cov: 437, ft: 1005, corp: 105` across
  2 375 327 iterations from an empty seed on aarch64-apple-darwin at
  ~43 700 exec/s, zero panics.
* `panic_free_mb_batch_motion_comp` — the MB-scale §18.3 / §20.14
  batched motion-compensation primitives landed in rounds 270–272:
  `fetch_luma_mb_halo` + `sixtap_mb_luma` (21×21 halo → 16×16 luma
  convolution), `fetch_chroma_mb_halo` + `sixtap_mb_chroma` (13×13 halo
  → 8×8 chroma convolution), and the whole-pixel non-SPLITMV copy paths
  `fetch_luma_mb_whole_pixel` / `fetch_chroma_mb_whole_pixel`
  (round 273). These six public functions synthesise (or copy) a whole
  macroblock's prediction in one pass and landed *after* the round-257
  `panic_free_sixtap_subpel` target was written, so no existing harness
  reaches them: the fourteen targets above hit §18 only through the §17
  motion-search descent ladder (which snaps every per-candidate MV to a
  sub-block grid and never reaches the MB-scale orchestrator) or through
  `decode_vp8` / `Vp8DecoderState::decode_frame` /
  `encode_p_frame_multi_ref` (which gate the MB-scale fetch behind a
  fully-formed reference picture + §9.7 refresh state machine, so the
  §20.14 `build_mc_border` clamp inside the MB-halo fetch never sees an
  origin parked across a picture boundary by an arbitrary `i16` vector).
  This target drives the MB-scale surface directly with an
  attacker-shaped `(plane dimension, MB origin, MV, fractional offset,
  filter set, border-position class)` envelope — every border class
  (mid-plane fast path, top-left corner, bottom-right corner,
  adversarial full-`i16` MV) and every `(mx, my) ∈ {0..7}²` fraction
  across both §18.3 filter sets — plus three equivalence cross-checks
  asserted on every iteration (panic on mismatch): the 21×21 luma halo
  and the 13×13 chroma halo must each contain every per-sub-block 9×9
  `fetch_block_halo` window at offset `(sb*4, sc*4)`, and the
  whole-pixel MB luma / chroma copy must equal the per-sub-block
  `fetch_block_whole_pixel` assembly tiled into the MB raster — the
  round-270 / 271 / 272 in-tree containment invariants, now re-asserted
  under the attacker-shaped border-clamp envelope the mid-plane-only
  in-tree tests never reach. 26-second smoke pass landed `cov: 355,
  ft: 569, corp: 65/1063b` across 810 049 iterations from an empty seed
  on aarch64-apple-darwin at ~31 155 exec/s, peak RSS 495 MiB, zero
  panics.
* `panic_free_filter_block_into` — the §16.4 SPLITMV strided-write
  motion-comp primitive `filter_block_4x4_into` landed in round 274: the
  companion of `filter_block_4x4` that synthesises one 4×4 sub-block
  (§20.14 `filter_block`) and writes it directly into a destination
  raster at `(dst_x, dst_y)` / `dst_stride` — whole-pixel branch copies
  source rows straight in (§18.3 "simply copied" / §20.14
  `build_mc_border` edge replication on the border-straddle path),
  sub-pixel branch delegates to `filter_block_4x4` and writes strided. It
  landed *after* the round-257 `panic_free_sixtap_subpel` target (which
  drives `filter_block_4x4` itself), so no existing harness reaches its
  destination-raster triple: the in-tree round-274 equivalence tests
  drive fixed inputs on one mid-plane geometry, and `predict_split_mv`
  keeps the shipped scratch-copy form so the decode / encode stack never
  exposes `(dst_x, dst_y, dst_stride)` to an attacker. This target drives
  it directly with an attacker-shaped `(plane dimension, block origin,
  mv, fraction, filter set, border-position class, destination geometry)`
  envelope — every border class (mid-plane fast path, top-left corner,
  bottom-right corner, adversarial full-`i16` MV), both whole-pixel and
  sub-pixel vectors, both §18.3 filter sets, and a `dst_stride` / `dst_x`
  / `dst_y` swept across `16..=32` — plus two equivalence cross-checks
  asserted on every iteration (panic on mismatch): the 4×4 footprint at
  `(dst_x, dst_y)` must equal `filter_block_4x4`'s `[u8; 16]` block byte
  for byte (the round-274 `filter_block_4x4_into_matches_filter_block_4x4`
  invariant, under the attacker-shaped clamp + destination envelope the
  in-tree test never reaches), and every destination byte outside the 4×4
  footprint must retain its pre-fill sentinel (a stride / length
  regression that strided past the window is caught here). 31-second smoke
  pass landed `cov: 248, ft: 369, corp: 63/1738b` across 3 961 583
  iterations from an empty seed on aarch64-apple-darwin at ~127 793
  exec/s, peak RSS 428 MiB, zero panics.
* `panic_free_encode_decode_e2e` — per-frame symmetric
  `encode_keyframe` → `decode_vp8` round-trip (round 264): every
  `(I420Frame, KeyframeParams)` pair the encoder accepts is decoded in
  the same iteration, with the §9.1 visible width / height asserted to
  round-trip. MB-aligned dimensions (1..=16 MB units per axis).
* `encode_decode_pixel_lockstep` — pixel-exact encode→decode lockstep
  differential (round 280). Strengthens the e2e target's
  dimensions-only oracle to full pixel content: the decoder's visible
  Y / U / V planes are asserted byte-equal to the encoder's own
  post-§15 reconstruction (returned by
  `encode_keyframe_with_reconstruction_and_token_updates`; per the
  §15.1 lockstep contract a compliant decoder reproduces those exact
  pixels), so a single-pixel drift anywhere in the §12 / §14 / §15 /
  §9.1-crop chain panics. Dimensions are raw (non-MB-aligned) luma
  pixels — width 1..=64, height 1..=144 (the tall end populating all
  8 §9.5 DCT partitions) — so the partial-macroblock padding / crop
  seam is hot on nearly every iteration, and half of all iterations
  thread a fuzz-shaped sparse §13.4 `token_prob_update()` payload (raw
  0..=255 probability bytes) through the only fuzz path that drives
  the §13.4 **write** side. Parameters are normalised into their legal
  ranges, so an `Err` from either half is itself a finding. The
  deterministic in-CI companions live in
  `tests/encoder_decoder_pixel_lockstep.rs` (5 anchors: partial-MB +
  normal filter, simple-filter extremes, 1-px strips, 8-partition
  populated/empty layouts, §13.4 L(8) probability extremes 0 / 255).
* `decode_stream_token_descent` — full-frame multi-packet decode
  driver aimed at the round-283 hot-path rewrite (fused §13.2 token
  descent, §20.16 zigzag-direct coefficient writes, §7.3 batched
  bool-decoder renormalisation), landed in round 284. Seeded with the
  13 fixture IVF streams (keyframes AND inter frames) re-framed as
  length-prefixed packet sequences so mutations corrupt real §13
  token partitions against valid reference state. Three oracles:
  fresh-`Vp8DecoderState`-vs-`decode_vp8` cross-entry-point
  differential (Ok/Err polarity + byte-identical planes — caught the
  round-284 short-DCT-partition fix within its first minute), an
  FNV-1a fold over every decoded plane byte, and a scalar-vs-SIMD
  differential over the §14.1 / §14.3 / §14.4 / §12.2 / §18.3 kernel
  pairs (the fuzz crate builds with `simd` by default, so the SIMD
  dispatch path is fuzzed while the differential keeps the scalar
  kernels covered). Daily scheduled `Fuzz` workflow runs every target
  under ASan since round 284.
* `panic_free_near_mv_mode_decode` — the §16.2 / §16.3 / §16.4 / §17
  inter-MB *mode-info decode* surface, landed in round 287. Drives
  `decode_inter_mb` / `decode_split_mv_mb` (the §18 end-to-end
  integration entry points), `resolve_inter_mb_mv`, `find_near_mvs`
  (the §16.3 spatial census), `decode_split_mv` (the §16.4 partition
  walk), the three bool-tree walks `read_inter_mode` /
  `read_mv_partition` / `submv_ref`, the §20.11 neighbour-MV lookups
  `above_block_mv` / `left_block_mv` across all sixteen sub-blocks
  (SPLITMV `b+12` / `b+3` branches + intra-zero fallback), plus
  `mv_ref_probs` / `submv_ref_context` / `clamp_mv` — all fed directly
  from a bool-coder partition with adversarial neighbour `MbInfo`
  records, sign-bias flags, and attacker-tiled §17 MV probability
  contexts. This is the bool-decoder-driven §16.3 census → §16.2 tree →
  §16.4 SPLITMV walk → §17 NEWMV differential that *produces* the
  vectors the round-263 `panic_free_inter_mb_reconstruct` target feeds
  in directly — a path the `decode_vp8` targets reach only through
  self-consistent §9.7 reference state. Asserts the §16.3 census counts
  stay inside the documented `0..=5` `mv_ref_probs` index envelope on
  every iteration. Round-287 ASan campaign (nightly, default `simd`):
  ~11.0 M executions in 181 s (~60.8 K exec/s), peak RSS 572 MB, zero
  crashes / leaks / OOMs.
* `panic_free_keyframe_reconstruct` — the §12 / §14 key-frame intra
  *reconstruction core* driven directly via `frame::decode_keyframe`,
  landed in round 290. The full-stream targets
  (`panic_free_decode_keyframe`, `decode_stream_token_descent`) reach
  this reconstruction walker only behind the §9.1 frame-tag and §19.2
  coded-header validation gates, so a random or fixture-derived
  bitstream rarely drives it through the full Cartesian product of intra
  modes, B_PRED sub-block-mode mixes, skip flags, and hostile
  post-dequant coefficient magnitudes. This target bypasses the parser:
  it materialises an arbitrary `mb_cols × mb_rows` grid (≤ 16 MBs) of
  `MacroblockModes` + already-dequantized `MbCoeffs` straight from the
  fuzz bytes and feeds them to `decode_keyframe`, forcing every §12
  16×16 luma mode (`DC`/`V`/`H`/`TM_PRED`), the §11.3 `B_PRED` path with
  all ten §11.2 4×4 sub-block predictors in arbitrary arrangements,
  every §11.4 chroma mode, the `mb_skip_coeff` short-circuit, and
  arbitrary `i16` coefficients through the §11/§12 neighbour gather →
  §14.3 inverse WHT → §14.4 inverse DCT → §14.5 add-and-clamp pass. The
  oracle is panic-freedom plus full plane materialisation (every output
  byte folded into an FNV-1a accumulator so ASan reads the entire
  claimed raster — a short-write / stale-stride bug surfaces here) plus
  the documented `mb_cols·16 × mb_rows·16` luma / `·8 × ·8` chroma
  geometry asserts. Round-290 ASan campaign (nightly, default `simd`):
  ~2.15 M executions in 151 s, plus a 0.31 M-iteration corpus-replay
  reaching a 687-PC / 2957-feature plateau (303-input corpus); zero
  crashes / leaks / OOMs.
* `panic_free_dequant_factors_mb` — the §14.1 / §20.4 dequant
  *factor-derivation* and full-macroblock dequant-apply surface, plus
  the §13.3 → §14.1 token → dequant wrapper, landed in round 293. The
  pre-existing transform-primitive target reaches §14.1 only at the
  single-block leaf (`dequant_block` with the two factors handed in
  directly) and the token target stops at `decode_mb_coeffs` without
  ever deriving factors or scaling a block — so the §20.4 `dequant_init`
  factor construction (`MbDequantFactors::from_base_and_deltas` /
  `from_quant_indices` / `for_segment`), the whole-MB apply
  (`dequantize` over the Y2 + 16 Y + 4 U + 4 V block set) and the
  bitstream → dequant wrapper (`decode_and_dequantize_mb`) were
  directly under-fuzzed. The harness drives all three from raw bytes,
  sweeping the base index and the five §9.6 plane deltas across the full
  `i32` range including the `i32::MIN` / `i32::MAX` cliff endpoints, with
  every macroblock coefficient seeded at the §14.1 `i16` product cliffs.
  This target **found a real `attempt to add with overflow` panic**: the
  internal `q + delta` additions in `from_base_and_deltas` (and the
  §10 per-segment `y_ac_qi + segment_quant` base add) panicked in a
  debug build when an out-of-range base/delta pair was supplied through
  the public `i32` API, even though `clamp_qindex` was meant to saturate
  the index. Fixed in the same commit by forming every index sum with
  `saturating_add` so the documented clamp does its job (a real
  bitstream's §9.6 `u8` base + `i8` deltas never reach the edge, so
  decode output is unchanged). Round-293 ASan campaign (nightly, default
  `simd`): 4 055 331 executions in 201 s (cov 338 / ft 602), zero
  crashes / leaks / OOMs after the fix; a regression test
  (`extreme_base_and_deltas_saturate_without_overflow`) anchors the
  saturation at both cliff ends.
* `decoder_trait_packet_lifecycle` — the `oxideav_core::Decoder` trait
  driver (`Vp8Decoder`), landed in round 296. Every other decode target
  enters through the direct API (`decode_vp8`, `Vp8DecoderState`); this
  one walks the *framework* entry point through its full `send_packet`
  → `receive_frame` → `flush` → `reset` lifecycle. That exercises two
  surfaces no direct-API target reaches: the packet/frame plumbing (the
  `Packet` clone into the internal `VecDeque`, the `NeedMore` / `Eof`
  state transitions across the EOF latch, the queue rebuild on `reset`)
  and the `From<Vp8DecodedFrame> for VideoFrame` conversion (computed
  luma / chroma strides, `pts` copy). Two lifecycle oracles beyond
  panic-freedom: post-`flush` drain must surface `Eof` (never
  `NeedMore`), and post-`reset` empty receive must surface `NeedMore`
  (never `Eof`); every produced plane byte is folded into an FNV-1a
  accumulator so a stride/length mismatch in the conversion surfaces
  under ASan. Length-prefixed packet input (`split_packets`), seeded
  from the crate's own fixture-derived IVF streams; per-keyframe §9.1
  dimension cap, ≤ 16 KiB / ≤ 12 packets per iteration. Round-296 ASan
  campaign (nightly, default `simd`): ~500 000 executions across two
  runs (cov 3934 / ft 18688, 2629-input corpus from the 13 fixture
  seeds), zero crashes / leaks / OOMs; no `src/` change was needed.
* `inter_stream_encode_decode_sequence` — the multi-frame
  `Vp8InterStreamEncoder` driver, landed in round 299. Every other
  encode target stops at a *single* key frame (`encode_keyframe`,
  `encode_keyframe_with_reconstruction*`), leaving the cross-frame
  stream layer — the §9.7 reference-refresh ladder (a key frame
  replaces all three slots, a ZERO_MV P-frame refreshes LAST only),
  the keyframe-interval scheduler with its `force_keyframe`
  re-anchoring, the across-frame §9.4 `ref_frame_delta[]` /
  `mode_delta[]` carry, and the locked-after-first-frame dimension
  guard — never driven by a fuzzer. The harness pushes a fuzz-shaped
  sequence of up to 12 small (≤ 48 × 48) I420 frames at a fuzz-chosen
  keyframe interval (with per-frame `force_keyframe` overrides) through
  one encoder and feeds each emitted frame into one long-lived
  `Vp8DecoderState`, so the decoder carries its own §9 reference slots
  forward and reconstructs the ZERO_MV P-frames from the LAST it kept —
  the §16.2 / §18 inter-prediction path no single-frame target reaches.
  Every parameter is normalised in-range, so an encode `Err`, a decode
  `Err`, a K/P-classification drift against the scheduler's own verdict,
  or a visible-dimension drift is itself a finding; decoded plane
  lengths are asserted against the §9.1 geometry. Round-299 ASan
  campaign (nightly, default `simd`): ~29 000 executions across two
  runs (cov 4831 / ft 24338, 897-input corpus from empty seed), zero
  crashes / leaks / OOMs; no `src/` change was needed.

Initial smoke pass: 800 000 combined iterations on the three decode
targets + 17 500+ iterations on the encode target (2790 coverage edges,
541-input corpus from empty seed in 31 s on aarch64-apple-darwin),
zero panics across the board. See [`fuzz/README.md`](./fuzz/README.md)
for caps, run instructions, and the rationale behind each target's
pre-flight gating.

## License

MIT. See [`LICENSE`](./LICENSE).
