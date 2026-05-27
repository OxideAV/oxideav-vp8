# oxideav-vp8 benchmarks

Round 170 (2026-05-27) wired up `criterion` benches for the
`oxideav-vp8` encode + decode hot paths and used the resulting profiles
to retarget three concrete optimisations. This document records the
baseline + post-optimisation numbers so future rounds have a stable
A/B reference.

## Running

```sh
# Stable + default features — every bench runs the scalar paths.
cargo bench -p oxideav-vp8 --bench keyframe_encode --bench keyframe_decode \
            --bench inter_encode_short_clip --bench inverse_transform_4x4 \
            --bench motion_comp_subpel_luma --bench intra_predict_dc16 \
            --bench loop_filter_normal -- --quick

# Nightly + simd feature — `inverse_wht_4x4` goes through the
# `core::simd::Simd<i32, 4>` rewrite. Tests and other benches are
# byte-identical to the stable path.
RUSTC=$(rustup which --toolchain nightly rustc) \
RUSTDOC=$(rustup which --toolchain nightly rustdoc) \
rustup run nightly cargo bench -p oxideav-vp8 --features simd \
    --bench inverse_transform_4x4 -- --quick
```

Bench inputs are synthesised in-bench (no committed fixture files); the
encoder benches use a deterministic gradient + flat-square I420 source
that exercises both intra paths so the per-MB mode picker does real
work. The micro-benches drive a single fixed input so successive
measurements compare apples-to-apples.

## Hardware

Apple M4-class aarch64, macOS 25.1, rustc 1.95.0 (stable) / 1.97.0-nightly
(simd). `--quick` measurements (criterion's reduced-iteration mode).

## Headline results — round 170

| Bench | Baseline | Post-opt (stable) | Post-opt (nightly+simd) | Δ |
|---|---:|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 9.06 ms | **8.30 ms** | (same as stable) | **−8.4 %** |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 153.3 µs | 153.6 µs | 154.7 µs | ±0.2 % |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 14.35 ms | **11.24 ms** | 11.07 ms | **−21.7 %** |
| `inverse_transform_4x4/inverse_dct_4x4` | 10.23 ns | 10.06 ns | 10.57 ns | −1.7 % |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.83 ns | 9.38 ns | **7.48 ns** | **−23.9 %** (simd) |
| `motion_comp_subpel_luma/filter_block_4x4_sub3x5` | 35.40 ns | **24.65 ns** | (same) | **−30.4 %** |
| `motion_comp_subpel_luma/mb_sixtap_2d_16x4x4` | 267.6 ns | 264.3 ns | (same) | −1.2 % |
| `intra_predict_dc16/predict_y16x16_dc` | 4.88 ns | 4.77 ns | (same) | −2.3 % |
| `intra_predict_dc16/predict_y16x16_v` | 3.66 ns | 3.67 ns | (same) | ±0.2 % |
| `intra_predict_dc16/predict_y16x16_h` | 3.75 ns | 3.73 ns | (same) | −0.4 % |
| `loop_filter_normal/subblock_filter_4_4` | 2.16 ns | 2.12 ns | (same) | −1.9 % |
| `loop_filter_normal/simple_segment_4` | 2.12 ns | 2.12 ns | (same) | ±0.0 % |

Throughput on the whole-frame benches:

* `keyframe_encode_320x240_qi32`: 8.48 → **9.25 Mpx/s** (+9 %)
* `keyframe_decode_320x240_qi32`: ≈ **501 Mpx/s** (decoder was already
  cheap; the round-170 optimisations target the encode-heavy paths)
* `inter_encode_4f_128x128_qi32`: 4.57 → **5.83 Mpx/s** (+28 %)

## Profile evidence (pre-optimisation)

`sample(1)`-based PID-attach profile of the `keyframe_encode` and
`inter_encode_short_clip` bench binaries while looping under
`--measurement-time 30`. Top self-time symbols (15 s wall, 1 ms
interval):

### `keyframe_encode_320x240_qi32`

| Self-samples | Symbol |
|---:|---|
| 59 | `encoder::token_to_bit_path::descend` |
| 52 | `_xzm_free` (libsystem_malloc) |
| 50 | `encoder::estimate_block_bits` |
| 42 | `encoder::encode_mb_block_set_with_neighbors` |
| 41 | `_xzm_xzone_malloc` |
| 29 | `log2` (libsystem_m) |
| 17 | `inverse_transform::inverse_dct_4x4` |

### `inter_encode_4f_128x128_qi32`

| Self-samples | Symbol |
|---:|---|
| 60 | `log2` (libsystem_m) |
| 55 | `motion_comp::fetch_block_whole_pixel` |
| 54 | `_xzm_free` |
| 39 | `motion_comp::sixtap_2d` |
| 37 | `_xzm_xzone_malloc` |
| 35 | `motion_comp::fetch_block_halo` |
| 33 | `encoder::token_to_bit_path::descend` |
| 29 | `encoder::estimate_block_bits` |

## Optimisations landed (round 170)

### 1. `bool_bits` `log2` lookup table — encoder RD scoring

Profile evidence: `log2` was the #6 self-time symbol on the keyframe
encode and the #1 on the inter encode, called from
`encoder::estimate_block_bits` → `bool_bits` for every emitted bit of
every candidate block during RD scoring.

Rewrite: `bool_bits` now consults a 256-entry `LazyLock<[f64; 256]>`
lookup table, `BIT_COST_BY_FALSE_PROB[p] = -log2(p / 256)`, with the
`p = 0` slot floored at `-log2(1/256) = 8.0` to preserve the original
`prob.max(1)` + `1/256` clamp. The `value == true` branch reads slot
`256 - p` (with `p = 0` pre-clamped to slot 255) so a single i/o pair
per call replaces the FP log2.

**Wins:** drove the −21.7 % on `inter_encode_short_clip` and the
−8.4 % on `keyframe_encode`. Bit-identical encoder output on every
fixture in the existing test suite.

### 2. `motion_comp::fetch_block_*` in-bounds fast paths

Profile evidence: `fetch_block_whole_pixel` (#2) and `fetch_block_halo`
(#6) together accounted for ~90 self-samples on the inter bench. Each
function ran a per-pixel `isize.clamp(0, w-1)` even when the entire
block / halo landed strictly inside the plane (the common case mid-
frame).

Rewrite: both functions branch on `src_x0 >= 0 && src_y0 >= 0 &&
src_x0 + N <= w && src_y0 + N <= h`. In the in-bounds case each output
row becomes a single `copy_from_slice` over a contiguous `N`-byte run
of the source plane — no per-pixel clamps, no per-byte bounds checks.
The edge-replication slow path stays bit-identical for border MBs.

**Wins:** drove the −30 % on `filter_block_4x4` and contributed to the
inter-encode delta. Byte-exact reconstruction on the existing
encoder/decoder roundtrip + inter-stream tests.

### 3. `inverse_wht_4x4` `core::simd::Simd<i32, 4>` rewrite (simd feature)

Profile evidence: not in the top-5 self-time of either bench (the WHT
fires once per Y2-bearing MB) but it's the cleanest vectorisation
target — the §14.3 listing is two passes of independent column /
row butterflies which map 1:1 onto a 4-lane `Simd<i32, 4>` layout
(one lane per column for the first pass; transpose; one lane per row
for the second pass).

Rewrite: `inverse_wht_4x4` now dispatches at compile time between
`inverse_wht_4x4_scalar` (bit-identical to the prior RFC 6386 listing)
and `inverse_wht_4x4_simd` (the `core::simd` rewrite). The SIMD layout
is derived directly from the §14.3 spec listing — no external SIMD
reference consulted. The `simd` feature is nightly-only because
`core::simd` is itself nightly.

**Wins:** −23.9 % on the `inverse_wht_4x4` micro-bench (9.83 → 7.48 ns
on nightly+simd). Doesn't yet move the whole-frame numbers because
the WHT is a small fraction of total decode time; the same rewrite
pattern applied to `inverse_dct_4x4` in a future round is the natural
next step.

## What didn't get touched yet (next-round candidates)

* **`encoder::token_to_bit_path::descend` (#1 self-time on encode)** —
  the function walks a small tree and ends up RD-scoring the same
  token paths repeatedly. A precomputed token-to-path table would
  remove the descent entirely.
* **Allocator churn (`malloc` / `free` ≈ 90 self-samples on the
  encoder)** — likely `Vec`-on-Vec inside the per-MB inner loop. A
  pass with a `SmallVec` / fixed-size `[Vec; …]` cache should hit it.
* **`sixtap_2d` (#4 on inter)** — the inner 6-tap convolution is a
  natural SIMD target (`Simd<i16, 8>` for an 8-pixel-wide stripe).
  Held back this round to keep the SIMD-feature surface to one
  primitive.
* **`inverse_dct_4x4` SIMD** — same layout as the WHT but with the
  §14.4 `(t1 * SINPI8_SQRT2) >> 16` fixed-point multiplies. A
  `Simd<i32, 4>` rewrite is straightforward; deferred so this round
  ships one SIMD primitive that's been A/B-proven against scalar.
