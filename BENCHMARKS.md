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

## Round 180 — `inverse_dct_4x4` SIMD

Round 180 (2026-05-29) landed the round-170 deferred follow-on: a
`core::simd::Simd<i32, 4>` rewrite of `inverse_dct_4x4` parallel to the
existing `inverse_wht_4x4_simd`. Same dispatch shape — the public
`inverse_dct_4x4` calls `inverse_dct_4x4_simd` on nightly + `simd` and
`inverse_dct_4x4_scalar` (the spec listing, unchanged) elsewhere.

Byte-exact against the scalar listing on a 21-input stress set in
`src/inverse_transform.rs::tests::dct_simd_matches_scalar_on_stress_inputs`
(DC-only over 10 magnitudes from ±8 to ±4096, single-AC at every one
of the 15 non-DC positions, plus two near-extreme mixed gradients).
The same suite was added for the WHT
(`wht_simd_matches_scalar_on_stress_inputs`) so the round-170 path now
has the same dense equivalence proof as the new round-180 path.

| Bench | r170 stable | r180 stable | r180 nightly+simd | Δ |
|---|---:|---:|---:|---:|
| `inverse_transform_4x4/inverse_dct_4x4` | 10.06 ns | 10.07–10.32 ns | **9.51–10.02 ns** | **−1 to −5 %** (simd) |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.38 ns | 9.4–10.0 ns | 7.36–8.03 ns | unchanged from r170 |

The §14.4 inverse DCT carries 8 fixed-point multiplies per pass
against the §14.3 WHT's zero — that's where the smaller r180 SIMD
margin comes from. The WHT moves end-to-end as four parallel
adds/subs per pass; the DCT has multiply-then-shift dependencies that
serialise inside each lane. Both passes still vectorise correctly
across the four-lane width, just at a lower speedup than the WHT.
The headline value of the round is the byte-exact equivalence and
the now-symmetric §14.3 / §14.4 SIMD shape, not a runtime headline
number — `inverse_dct_4x4` fires once per Y / U / V sub-block (16 +
4 + 4 = 24 calls per MB on decode) vs `inverse_wht_4x4` once per
Y2-bearing MB, so the cumulative whole-frame impact tracks the small
per-call delta scaled by the call count.

## Round 194 — rate-control `y_ac_qi` sweep

Round 194 (2026-05-31) adds a depth-mode bench
(`rate_control_qi_sweep`) that walks `KeyframeParams::y_ac_qi` — the
§9.6 baseline quantiser index, the principal rate-control knob on
`encode_keyframe` — across ten representative values on the same
deterministic 320×240 I420 source the round-170 `keyframe_encode` bench
uses. The goal is *not* a new optimisation but a published trade-off
curve readers can tune against: every other §9.6 quantiser delta
defaults to 0 in `KeyframeParams`, so a single `y_ac_qi` value moves the
DC + AC luma / chroma quantiser bank in lockstep.

```sh
CARGO_TARGET_DIR=/tmp/oxideav-vp8-target \
    cargo bench -p oxideav-vp8 --bench rate_control_qi_sweep -- --quick
```

Headline numbers (Apple M4 / aarch64, criterion `--quick`, source =
mixed luma gradient + centre flat-128 square, 320×240 I420):

| `y_ac_qi` | Output | bpp | Encode wall | Throughput |
|---:|---:|---:|---:|---:|
|   8 | 1701 B | 0.18 |  9.33 ms |  8.23 Mpx/s |
|  16 |  676 B | 0.07 |  8.48 ms |  9.05 Mpx/s |
|  24 |  612 B | 0.06 |  8.44 ms |  9.10 Mpx/s |
|  32 |  595 B | 0.06 |  8.38 ms |  9.16 Mpx/s |
|  40 |  480 B | 0.05 |  8.16 ms |  9.41 Mpx/s |
|  48 |  466 B | 0.05 |  8.21 ms |  9.35 Mpx/s |
|  56 |  461 B | 0.05 |  7.92 ms |  9.69 Mpx/s |
|  72 |  360 B | 0.04 |  7.61 ms | 10.09 Mpx/s |
|  96 |  355 B | 0.04 |  7.54 ms | 10.18 Mpx/s |
| 120 |  299 B | 0.03 |  7.31 ms | 10.51 Mpx/s |

Reading the curve:

* The byte cost falls monotonically with `y_ac_qi` — a strict
  expected-direction sanity check on the quantiser path. The jump
  between qi=8 and qi=16 (1701 → 676 B, ~−60 %) is the steepest
  segment; everything past qi=32 lives in the lossy-but-flat tail of
  the dial.
* Encode wall time falls in the same direction (9.33 → 7.31 ms, −22 %)
  because larger quantisers produce shorter token streams + more EOB
  early-exits inside `encoder::token_to_bit_path::descend` and
  `encoder::estimate_block_bits` — the two top self-time symbols
  identified in the round-170 profile (`BENCHMARKS.md` §Profile
  evidence above). The trend is consistent with that profile: the
  rate-control knob and the encoder's hot path are coupled through
  the token bit count.
* Throughput delta across the sweep is +28 % (8.23 → 10.51 Mpx/s) —
  picking a higher qi for previewing / draft modes is a real wall-time
  win, not just a bytes-on-the-wire win.
* The synthetic source compresses to small outputs at every qi
  (DC-heavy gradient + uniform flat square is the easy case for the
  intra picker) — production-content numbers will be larger in
  absolute bytes but the *shape* of the qi/bytes/throughput trade-off
  carries.

This bench is intended as a published baseline: re-run after any
encoder change and compare the per-qi byte and wall-time columns to
catch regressions on a single dial that drives the whole rate-control
surface.

## Round 204 — `token_to_bit_path` precomputation

Round 204 (2026-06-01) lands the round-170 follow-up
*"`encoder::token_to_bit_path::descend` — the function walks a small
tree and ends up RD-scoring the same token paths repeatedly. A
precomputed token-to-path table would remove the descent entirely."*

`token_to_bit_path` is hit at least three times per coefficient in the
encoder hot path — once by the block writer, once by the RD bit-cost
estimator, and once by the §13.4 token-prob counts fitter. Before
round 204 each call allocated a fresh `Vec<(usize, bool)>` and ran a
recursive `ENC_COEFF_TREE` descent. The descent is a pure function of
`(start_index ∈ {0, 2}, token ∈ 12 entries)`, so all 24 cells are now
materialised once at module load through `std::sync::LazyLock` into a
fixed-shape `[[([TokenBitStep; 7], u8); 12]; 2]` table. Path widths cap
at 7 (`Cat3..Cat6` from the root); the unreachable `(start = 2, Eob)`
cell is a `length = 0` tombstone. The function returns
`&'static [TokenBitStep]` — index-and-slice, zero per-token allocation.

| Bench | r194 stable | r204 stable | Δ |
|---|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 8.51 ms | **5.97 ms** | **−29.8 %** |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 10.82 ms | **10.20 ms** | **−5.7 %** |

Throughput on the whole-frame benches (Apple M4 / aarch64,
criterion `--quick`):

* `keyframe_encode_320x240_qi32`: 9.03 → **12.87 Mpx/s** (+43 %)
* `inter_encode_4f_128x128_qi32`: 6.06 → **6.42 Mpx/s** (+5.9 %)

The keyframe-encode delta is the headline. The round-170 sample profile
counted `_xzm_xzone_malloc` + `_xzm_free` at ≈ 93 combined self-samples
across the encoder bench; removing the per-coefficient `Vec` allocation
collapses that into the static table's single one-shot init. The inter
delta is smaller because motion search + reconstruction dilute the
token-emission share of total time.

Equivalence proof:
`encoder::tests::token_bit_path_table_matches_tree_descent` re-runs the
original recursive descent inline and asserts every reachable cell
agrees on length, every prob_index, and every bit. Plus the full
450-test lib suite + every existing integration test reach unchanged
byte-exact output through the new dispatch.

## Round 220 — `forward_transform_4x4` micro-bench

Round 220 (2026-06-03) closes a long-standing gap in the published
A/B surface: the §14.3 forward WHT (`forward_wht_4x4`) and the §14.4
forward DCT (`forward_dct_4x4`) — the encoder partners of the
round-170 / round-180 inverse-side primitives — had no published
micro-bench. The whole-frame `keyframe_encode` and
`inter_encode_short_clip` benches drive them inside the §11 intra
picker + §13 token emit + §15 loop-filter cascade, which makes
attributing a wall-time delta to a forward-transform rewrite
ambiguous. The new `forward_transform_4x4` bench mirrors
`inverse_transform_4x4`'s input layout (the same representative
DC-heavy 4×4 residual block) so a side-by-side read of the two files
lines up the forward and inverse passes one-for-one.

Headline baseline (Apple M4 / aarch64, criterion `--quick`):

| Bench | Wall time |
|---|---:|
| `forward_transform_4x4/forward_dct_4x4` | 10.13 ns |
| `forward_transform_4x4/forward_wht_4x4` | 10.48 ns |
| `inverse_transform_4x4/inverse_dct_4x4` | 10.25 ns |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.76 ns |

Observations:

* The forward DCT and the §14.4 inverse DCT are within a percent of
  each other (10.13 vs 10.25 ns) — the §14.3 / §14.4 forward and
  inverse passes share the same fixed-point constants
  (`COSPI8_SQRT2_MINUS1`, `SINPI8_SQRT2`), the same butterfly
  shape, and the same number of `c_mul` / `s_mul` invocations per
  pass. The forward path's per-row `round_div2` (the symmetric
  round-away-from-zero step that doesn't appear in the inverse)
  costs the small remaining delta.
* The forward WHT is ~8 % more expensive than the inverse WHT
  (10.48 vs 9.76 ns) — the forward path's symmetric `round_div2`
  fires on every output sample (16 calls per `forward_wht_4x4`)
  where the §14.3 inverse path's matching `(x + 3) >> 3` rounds
  without the negative-symmetric branch.
* Per-MB call count: 24 × `forward_dct_4x4` (16 Y + 8 chroma) +
  potentially 1 × `forward_wht_4x4` (for MBs with a Y2 DC plane).
  At 10.13 + 10.48 ns this is ≈ 253 ns / MB on the forward side
  alone, which at 320 × 240 (300 MBs / frame) is ~76 µs per frame
  — ~1.3 % of the 5.81 ms `keyframe_encode` wall time. Reads as:
  the forward primitives are not the encoder hot path (token emit
  + RD scoring dwarf them), but they're now visible at criterion's
  micro-bench resolution, ready as an A/B target for any future
  SIMD / unroll work on the encoder side parallel to the round-180
  `inverse_dct_4x4_simd` rewrite.

The bench input matches `inverse_transform_4x4.rs::SAMPLE_INPUT`
verbatim so a future `forward_dct_4x4_simd` rewrite would produce
the same per-call delta shape across the two pairs.

## Round 226 — `forward_wht_4x4` / `forward_dct_4x4` SIMD rewrites

Round 226 (2026-06-04) closes the round-220 next-round candidate: the
§14.3 forward WHT and §14.4 forward DCT now dispatch through a
`core::simd::Simd<i32, 4>` rewrite under the `simd` cargo feature,
parallel to the round-180 `inverse_*_simd` rewrites. The layout
mirrors the inverse SIMD path one-for-one: hold the input as four
`Simd<i32, 4>` row-vectors (lane `j` of row `i` carries `input[i*4 +
j]`); the first pass operates as four parallel lane-wide column
butterflies + (for the DCT) `c_mul` / `s_mul` lane-wide multiplies;
transpose; run the second pass the same way; apply the symmetric
`round_div2` step lane-wide via a `Mask::select(pos_branch,
neg_branch)` over `simd_ge(0)` (matching scalar `round_div2` bit-
for-bit — the two arithmetic branches are unconditional).

Headline `--quick` numbers (Apple M4 / aarch64), comparing scalar
(public dispatcher on stable, no `simd` feature) against the new SIMD
path (public dispatcher on nightly + `simd`):

| Bench | Scalar | SIMD | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_wht_4x4` | 10.74 ns | **8.72 ns** | **−18.8 %** |
| `forward_transform_4x4/forward_dct_4x4` | 10.71 ns | 11.54 ns | +7.7 % |

Observations:

* The forward WHT SIMD path is the cleanest win in the suite next to
  the round-180 inverse WHT (also ≈ −20 %). Both transforms are
  pure butterfly chains over an i32 column axis; the four-lane
  layout collapses the per-column scalar loop into a straight-line
  add / sub sequence. The bulk of the −19 % comes from the second-
  pass `round_div2` step folding into four lane-wide `Mask::select`s
  in place of four scalar branch-pairs.
* The forward DCT SIMD path is +8 % slower than scalar — the
  multiply-heavy `c_mul` / `s_mul` chain (8 i32 lane-wide multiplies
  per pass × 2 passes) doesn't pipeline as well as the scalar
  straight-line code, and the lane-wide `round_div2_simd` adds a
  mask + select that the scalar path avoids. This is the same shape
  the round-180 inverse DCT measurement showed (≈ −1 to −5 % over
  scalar) — the forward direction's per-pass `round_div2` step is
  the extra cost, where the inverse path's `(x + 4) >> 3` is a
  pure shift. The SIMD path is kept enabled under `simd` for shape
  parity with the WHT and so the byte-exact equivalence assertion
  (`fdct_forward_simd_matches_scalar_on_stress_inputs`) is exercised
  on every nightly + `simd` test run; a future round can split the
  dispatch (route `forward_dct_4x4` to scalar even under `simd` if
  the regression matters on a target hardware).
* Per-MB call count: 24 × `forward_dct_4x4` (16 Y + 8 chroma) + at
  most 1 × `forward_wht_4x4`. Under `simd` the forward-side wall is
  24 × 11.54 + 1 × 8.72 ≈ 286 ns / MB, vs scalar 24 × 10.71 + 1 ×
  10.74 ≈ 268 ns / MB — a ~+7 % bias on the forward primitives only,
  whose round-170 share of `keyframe_encode` was ~1.3 %. That puts
  the expected `keyframe_encode` impact at sub-0.1 %, deep below the
  bench's `--quick` noise envelope; verifying empirically by
  re-running `keyframe_encode` under both feature settings is left
  to a future profile-depth round.

Equivalence proof: `forward_transform::tests::{
fdct_forward_simd_matches_scalar_on_stress_inputs,
fwht_forward_simd_matches_scalar_on_stress_inputs}` run a 21-input
stress set (the same matrix shape as the round-180 inverse tests:
all-zero / DC-only at 10 magnitudes / single-AC at every of the
15 positions / mixed gradients / high-AC mid-range, both positive
and alternating-sign) and assert public-dispatch byte-equality with
the renamed `_scalar` variants. The full 452-test lib suite and
every integration test pass on both stable (scalar dispatch) and
nightly + `simd` (SIMD dispatch).

## Round 247 — `forward_dct_4x4` SIMD dispatch split

Round 247 (2026-06-07) closes the round-226 deferred next-round
candidate. The §14.4 `forward_dct_4x4` SIMD path ran the lane-wide
`c_mul` / `s_mul` chain (8 i32 multiplies per pass × 2 passes) plus a
`round_div2_simd` mask + select; round 226 measured it ~+8 % slower
than the scalar straight-line code on `aarch64-apple-darwin` but kept
the SIMD dispatch for shape parity with the §14.3 WHT. Round 247
re-routes the public `forward_dct_4x4` dispatcher to the scalar path
under every feature configuration — `forward_wht_4x4` keeps its SIMD
dispatch unchanged (no multiplies, still −18 % under scalar).

The `_simd` implementation stays compiled under the `simd` feature
(now with `#[allow(dead_code)]` since the dispatcher no longer reaches
it) so the byte-equivalence assertion still has a symbol to call. The
test was renamed `_simd_matches_scalar`-style logic-wise and now calls
`forward_dct_4x4_simd` directly against `forward_dct_4x4_scalar` over
the 21-input stress matrix (the public-dispatch comparison would be
trivially equal after the split). A second
`fdct_public_dispatch_is_scalar` test runs on every configuration and
asserts `forward_dct_4x4 == forward_dct_4x4_scalar` so a future round
can't accidentally re-route the dispatcher without flipping that
assertion too.

Headline `--quick` numbers (Apple M4 / aarch64) comparing the
post-round-226 SIMD dispatch with the round-247 scalar dispatch:

| Bench | r226 SIMD | r247 SIMD | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_dct_4x4` | 11.69 ns | **9.81 ns** | **−16.2 %** |
| `forward_transform_4x4/forward_wht_4x4` | 8.94 ns | 8.74 ns | −2.2 % |

Stable (no `simd`): unchanged within criterion `--quick` noise (forward
DCT 10.67 → 10.82 ns; forward WHT 10.81 → 10.84 ns). Per-MB forward
side under `simd` now drops from 24 × 11.54 + 1 × 8.72 ≈ 286 ns / MB
to 24 × 9.81 + 1 × 8.74 ≈ 244 ns / MB — a ~−15 % bias on the forward
primitives, the inverse of round 226's +7 % bias.

## Round 249 — `forward_dct_4x4_scalar` canonical-butterfly refactor

Round 249 (2026-06-07) reorganises the scalar §14.4 `forward_dct_4x4_scalar`
listing into the canonical `(a1, b1, c1, d1)` partial-sum butterfly form
mirroring the §14.4 `inverse_dct_4x4_scalar` listing. The original form
matched the direct derivation from `T_fwd * input * T_fwd^T` (`o0 = i0
+ i4 + i8 + i12`; eight `c_mul` / `s_mul` calls split across `o4` and
`o12`); the refactored form groups partial sums (`a1 = i0 + i12`, `b1
= i4 + i8` for the evens, `c1` / `d1` carrying the fixed-point
multiply pair-differences for the odds) so forward and inverse paths
share visual shape. Bit-exactness is preserved by never collapsing a
multiply pair-difference into `c_mul(i0 - i12)` (the `>> 16` truncation
is non-linear under sum); each `c_mul` / `s_mul` call still evaluates
separately and the addition reorder is on associative i32 sums. A new
`forward_dct_4x4_listing` private function holds the unfactored
direct-derivation form as a regression oracle, and
`fdct_scalar_matches_direct_derivation_listing` asserts the refactored
`_scalar` produces identical bytes on the 21-input stress matrix.

`--quick` numbers comparing round-247 (post-SIMD-split) and round-249
(post-refactor) on Apple M4 / aarch64:

| Bench | r247 | r249 | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_dct_4x4` (stable) | 10.82 ns | 10.83 ns | within noise |
| `forward_transform_4x4/forward_dct_4x4` (nightly + `simd`) | 9.81 ns | 10.05 ns | +2.4 % (within `--quick` noise envelope) |
| `forward_transform_4x4/forward_wht_4x4` (stable) | 10.84 ns | 10.54 ns | within noise |
| `forward_transform_4x4/forward_wht_4x4` (nightly + `simd`) | 8.74 ns | 8.93 ns | within noise |

The refactor is a readability / spec-shape parity change (LLVM was
already CSE'ing the i0/i12 partial sums under `-O3`); the bench numbers
on both configurations agree on the round-247 baseline within the
criterion `--quick` envelope. The `forward_dct_4x4_simd` listing (which
already used a butterfly-friendly lane-wide form in round 226) now
agrees visually as well as bit-exactly with the scalar.

## Round 269 — `sixtap_2d` SIMD (§18.3 six-tap sub-pixel interpolation)

Round 269 (2026-06-10) closes the long-standing round-170 candidate:
the §18.3 / §20.14 `sixtap_2d` kernel (#4 self-time on the round-170
inter-encode profile) now dispatches through a `core::simd::Simd<i32,
4>` rewrite under the `simd` feature, same dispatcher shape as the §14
transforms / §14.1 dequant / §12.2 TM_PRED rounds. Each convolution
row's four §18.3 `interp` dot products are one four-lane vector — tap
k's support lanes are the contiguous run `halo[r*9 + k ..][..4]` — and
the horizontal pass's clamped intermediate stays resident in `i32`
vectors so the vertical pass runs with zero loads.

Lane-type finding (vs the round-170 candidate note's `Simd<i16, 8>`
stripe): the §18.3 dot product over `u8` support spans `[-8160,
40800]` (the ½-displacement row `{3, -16, 77, 77, -16, 3}` has
positive-tap sum 160; 160 × 255 = 40800 > `i16::MAX`), so a single
`i16` accumulator wraps. A parity-split two-accumulator `i16×8`
two-row-stripe variant (every tap-parity class partial sum provably
fits `i16`: positive class sums cap at 128) was implemented and
measured during the round: `mb_sixtap_2d_16x4x4` ≈ 260–268 ns (no
better than scalar) and `filter_block_4x4_sub3x5` ≈ 28.3–28.7 ns
(~+15 % regression — the per-tap two-row gather eats the lane-width
win). The four-lane `i32` form, which matches the 4×4 sub-block
geometry exactly, is what shipped.

Headline `--quick` numbers (Apple M4 / aarch64, triple-run, nightly
toolchain for both columns so the compiler version cancels):

| Bench | Scalar | SIMD | Δ |
|---|---:|---:|---:|
| `motion_comp_subpel_luma/mb_sixtap_2d_16x4x4` | 271.5 ns | **248.5 ns** | **−8.5 %** |
| `motion_comp_subpel_luma/filter_block_4x4_sub3x5` | 24.87 ns | **23.55 ns** | **−5.3 %** |

Stable (no `simd`) is unchanged: 268.1 ns / 24.81 ns, within noise of
the nightly scalar column. The margin is modest next to TM_PRED's
−87.7 % because the scalar `interp` loop over fixed-size arrays was
already auto-vectorising well; the explicit kernel's win comes mostly
from keeping the two-pass intermediate in vector registers (no
narrow-to-u8 / widen-from-u8 round trip between the passes) rather
than from new parallelism.

Equivalence proof:
`motion_comp::tests::sixtap_2d_simd_matches_scalar_on_stress_inputs`
(13 halos × all 64 `(mx, my)` fraction pairs × both §18.3 filter
sets) plus `sixtap_2d_accumulator_extremes_match_scalar` (both dot-
product extremes through output column 0 under the ½-displacement
taps). The full lib suite passes on stable (463) and nightly + `simd`
(465).

## Round 270 — MB-scale §18.3 luma batching (`sixtap_mb_luma`)

Round 270 (2026-06-10) closes the round-269 next-round candidate
"MB-scale §18.3 batching". The round-269 per-4×4 `sixtap_2d` SIMD win
was capped by the 4-wide block geometry; the real headroom is one layer
up in `predict_inter_mb`, where all sixteen luma sub-blocks of a
non-SPLITMV MB share one §18.1 motion vector. Instead of sixteen
overlapping 9×9 `fetch_block_halo` + `sixtap_2d` calls, the sub-pixel
luma path now fetches one 21×21 halo (`fetch_luma_mb_halo`) and
synthesises the whole 16×16 luma block in a single two-pass convolution
(`sixtap_mb_luma`): a horizontal pass of 21 rows × 16 cols then a
vertical pass of 16 rows × 16 cols. The SIMD partner widens each pass to
`Simd<i32, 16>` — one sixteen-lane vector per output row, the clamped
horizontal intermediate resident in `i32` vectors so the vertical pass
runs with zero loads.

The new `motion_comp_subpel_luma/mb_luma_batched_16x16` bench measures
the whole-MB synthesis against the per-sub-block partner
`mb_luma_per_subblock_16x16` (sixteen `sixtap_2d` calls on the same
sub-pixel workload). Numbers on `aarch64-apple-darwin` (criterion
`--quick`, nightly toolchain for both columns so the compiler version
cancels):

| Bench | Per-sub-block | Batched scalar | Batched SIMD |
|---|---:|---:|---:|
| whole 16×16 luma block | 260–268 ns | **158.8 ns** | **140.2 ns** |

Reading the result:

* **Batched scalar vs per-sub-block: −41 %** (158.8 vs ~267.9 ns). The
  bulk of the MB-scale win is structural, not SIMD: one 21×21 fetch +
  one tight two-pass loop replaces sixteen 9×9 fetches and sixteen
  separate two-pass calls, amortising the per-block border / gather
  setup the round-170 profile attributed ~90 self-samples to
  (`fetch_block_halo` + `fetch_block_whole_pixel`).
* **Batched SIMD vs batched scalar: −12 %** (140.2 vs 158.8 ns). The
  `Simd<i32, 16>` rewrite quadruples the lane width vs round-269's
  `Simd<i32, 4>` 4×4-block form, so each horizontal-pass row is one
  sixteen-lane vector and the per-block setup the 4-wide form paid 16×
  is paid once.
* **Batched SIMD vs per-sub-block: −47 %** end-to-end (140.2 vs
  ~265 ns) — the headline. This is the path `predict_inter_mb` takes for
  every sub-pixel non-SPLITMV inter MB on nightly + `simd`.

Byte-exactness is anchored three ways:
`sixtap_mb_luma_matches_per_subblock_path` (whole-MB vs sixteen
`sixtap_2d` calls over the carved 9×9 sub-halos, every `(mx, my)` × both
filter sets), `sixtap_mb_luma_simd_matches_scalar_on_stress_inputs`
(dispatcher vs scalar over the flat / ramp / checker / LCG stress set),
and `predict_inter_mb_sub_pixel_at_border_uses_mb_halo_clamp` (a real
corner-MB prediction through the MB-halo border-clamp fallback). The
chroma path keeps the per-sub-block dispatch (the 8×8 chroma block gains
little from MB-scale batching, and the §18.1 averaged-vector / SPLITMV
cases use per-sub-block vectors).

## Round 271 — MB-scale §18.3 chroma batching (`sixtap_mb_chroma`)

Round 271 (2026-06-10) closes the round-270 next-round candidate "MB-scale
§18.3 chroma batching". The four chroma sub-blocks of each plane on a
non-SPLITMV inter MB share one §18.1 averaged motion vector, so the six-tap
support of the whole 8×8 chroma block is one contiguous 13×13 region.
`predict_inter_mb`'s sub-pixel chroma path now fetches one 13×13 halo
(`fetch_chroma_mb_halo`) and synthesises the whole 8×8 chroma block in a
single two-pass convolution (`sixtap_mb_chroma`): a horizontal pass of 13
rows × 8 cols then a vertical pass of 8 rows × 8 cols. The SIMD partner
widens each pass to `Simd<i32, 8>` — one eight-lane vector per output row,
the clamped horizontal intermediate resident in `i32` vectors so the
vertical pass runs with zero loads. This is the chroma analogue of the
round-270 16×16-luma path (`sixtap_mb_luma`, 21×21 halo, `Simd<i32, 16>`).

The new `motion_comp_subpel_luma/mb_chroma_batched_8x8` bench measures the
whole-MB synthesis against the per-sub-block partner
`mb_chroma_per_subblock_8x8` (four `sixtap_2d` calls on the same sub-pixel
workload). Numbers on `aarch64-apple-darwin` (criterion `--quick`, nightly
toolchain for both columns so the compiler version cancels):

| Bench | Per-sub-block | Batched scalar | Batched SIMD |
|---|---:|---:|---:|
| whole 8×8 chroma block | 67.7 ns | **43.9 ns** | **38.6 ns** |

Reading the result:

* **Batched scalar vs per-sub-block: −35 %** (43.9 vs 67.7 ns) — one 13×13
  fetch + one tight two-pass loop replaces four 9×9 fetches and four
  separate two-pass calls, amortising the per-block border / gather setup.
* **Batched SIMD vs batched scalar: −12 %** (38.6 vs 43.9 ns) — the
  `Simd<i32, 8>` rewrite halves round-270's `Simd<i32, 16>` lane width to
  match the 8-wide chroma block, so each horizontal-pass row is one
  eight-lane vector and the per-block setup is paid once.
* **Batched SIMD vs per-sub-block: −43 %** end-to-end (38.6 vs 67.7 ns) —
  the path `predict_inter_mb` takes for every sub-pixel non-SPLITMV inter
  MB's chroma planes on nightly + `simd`. The absolute win is smaller than
  the round-270 luma path's (8×8 vs 16×16 block) but the per-output-pixel
  ratio matches.

Byte-exactness is anchored five ways:
`sixtap_mb_chroma_matches_per_subblock_path` (whole-MB vs four `sixtap_2d`
calls over the carved 9×9 sub-halos, every `(mx, my)` × both filter sets),
`sixtap_mb_chroma_simd_matches_scalar_on_stress_inputs` (dispatcher vs
scalar over the flat / ramp / checker / LCG stress set),
`fetch_chroma_mb_halo_matches_subblock_halos_in_bounds` /
`fetch_chroma_mb_halo_clamps_at_top_left_corner` (halo containment +
border-clamp), and two real-prediction tests
(`predict_inter_mb_chroma_sub_pixel_matches_per_subblock_path` mid-plane +
`predict_inter_mb_chroma_sub_pixel_at_border_uses_mb_halo_clamp` corner) on
both U and V. Stable lib 468 → 474; nightly + `simd` lib 470 → 476.

## Round 272 — whole-pixel non-SPLITMV MB batching (`fetch_luma_mb_whole_pixel` / `fetch_chroma_mb_whole_pixel`)

Round 272 (2026-06-10) closes the round-271 next-round candidate
"whole-pixel non-SPLITMV MB batching". When the shared §18.1 motion
vector of a non-SPLITMV inter MB is *whole-pixel* (`mv & 7 == 0` per
component), the §18.3 prediction is a pure copy — no convolution. The
sixteen luma sub-blocks (or four chroma sub-blocks per plane) then share
one contiguous source region at integer offset `(mb_x, mb_y) + (mv >> 3)`,
so the whole block can be fetched in one pass instead of sixteen / four
overlapping 4×4 `fetch_block_whole_pixel` copies. Unlike the round-270 /
round-271 sub-pixel paths this is pure gather amortisation: one bounds
check + one border-straddle decision per MB rather than per sub-block, no
SIMD needed.

`predict_inter_mb`'s whole-pixel luma branch now issues one
`fetch_luma_mb_whole_pixel` (16×16, stride 16) and each chroma plane one
`fetch_chroma_mb_whole_pixel` (8×8, stride 8), both with the same
in-bounds contiguous-row fast path / per-pixel §20.14 `build_mc_border`
clamp fallback split as the per-sub-block fetch.

The new `motion_comp_subpel_luma/mb_*_whole_pixel_*` benches measure the
batched fetch against the per-sub-block assembly on a 64×64 deterministic
source (Apple M4 / aarch64, criterion `--quick`):

| Bench | Per-sub-block | Batched | Delta |
|---|---:|---:|---:|
| whole 16×16 luma copy | 46.89 ns | **13.13 ns** | **−72 %** |
| whole 8×8 chroma copy | 8.49 ns | **4.74 ns** | **−44 %** |

Reading the result: the per-sub-block luma path pays sixteen
`fetch_block_whole_pixel` calls, each with its own in-bounds test and its
own four-row `copy_from_slice` loop into a `[u8; 16]` scratch that is then
re-copied four rows at a time into `out.y`; the batched path does one
in-bounds test and sixteen direct 16-byte row copies straight into
`out.y`. The chroma ratio is smaller (four sub-blocks vs sixteen) but the
same shape. Byte-exactness is anchored five ways against the
per-sub-block `fetch_block_whole_pixel` assembly (in-bounds luma + chroma,
top-left luma + bottom-right chroma clamp, and a real corner-MB
`predict_inter_mb` covering Y + U + V); the existing
`reconstruct_inter_mb_matches_legacy_for_whole_pixel` test anchors the
full reconstruct path. Stable lib 474 → 479.

## Round 274 — SPLITMV write-strategy A/B (negative result)

Round 274 (2026-06-11) closes the rounds 270–272 next-round candidate
"SPLITMV whole-pixel sub-block batching" with a measured negative result.
SPLITMV macroblocks (RFC 6386 §16.4) carry sixteen distinct luma vectors
(plus four chroma), so the MB-scale shared-halo batch (`sixtap_mb_luma` /
`sixtap_mb_chroma`, `fetch_*_mb_whole_pixel`) cannot apply — every sub-block
is synthesised independently. The only remaining freedom is *how* each
per-sub-block result lands in the macroblock raster. Two byte-identical
strategies were benchmarked (`motion_comp_subpel_luma/splitmv_predict_*`):

| Strategy | Wall time |
|---|---:|
| `splitmv_predict_scratch_copy` (shipped) | **398.8 ns** |
| `splitmv_predict_strided_write` (`filter_block_4x4_into`) | 480.5 ns |

* **scratch_copy** builds a contiguous `[u8; 16]` block per sub-block
  (`filter_block_4x4`), then copies four contiguous 4-byte rows into the
  stride-16 (luma) / stride-8 (chroma) raster — the form `predict_split_mv`
  ships.
* **strided_write** writes each synthesised sub-block directly into the
  raster at its strided offset via the new `filter_block_4x4_into`, with no
  intermediate block.

The scratch-copy form wins by ~17 % on Apple M4 / aarch64: the contiguous
`[u8; 16]` lets the compiler vectorise the per-row writes, where the
scattered strided writes into a stride-16 raster cannot. `predict_split_mv`
therefore keeps the scratch path. `filter_block_4x4_into` is retained as a
public primitive (for callers that already own a destination raster) and is
byte-exact against `filter_block_4x4` + a strided copy
(`filter_block_4x4_into_matches_filter_block_4x4`,
`strided_into_assembly_matches_predict_split_mv`). The 16-sub-block
synthesis cost dominates; the write strategy is a ~4–5 % slice of total
per-MB SPLITMV prediction, so neither form moves whole-frame inter encode
materially — the value of the round is the documented A/B closing the
candidate.

## Round 276 — MV-cost `log2` table + allocation-free tree walks

Round 276 (2026-06-11) re-profiled the inter-encode bench per the
standing "remaining allocator churn" candidate below. The fresh
`sample(1)` profile (15 s attach, 1 ms interval, `--measurement-time
30`) put three related symbols high in self-time:

| Self-samples | Symbol |
|---:|---|
| 411 | `log2` (libsystem_m) — #6 overall |
| 200 | `motion_vector::mv_component_bits::small_mv_bits::find_path` |
| 122 | `encoder::treed_bits::find_path` |
| ~540 | `_xzm_free` / `_xzm_xzone_malloc` / `_malloc_zone_malloc` / `RawVec::finish_grow` / `grow_one` combined |

Caller attribution placed the malloc churn squarely under
`mv_component_bits` (per-candidate-MV RD costing inside motion search)
and the §11 mode-RD `treed_bits` walk — exactly the "next biggest
short-lived `Vec`" the round-272 candidate predicted. Three fixes, all
producing bit-identical encoder output:

1. **`mv_component_bits` joins the round-170 `-log2(p/256)` lookup
   table.** The §17.1 MV-component costing still computed an inline libm
   `log2` per priced bool; it now reads the same 256-entry
   `BIT_COST_BY_FALSE_PROB` table the block-RD `bool_bits` has used
   since round 170. For every `(prob, value)` pair the table entry is
   the *exact* double the inline expression produced (same clamps, same
   dyadic arguments), locked by a new full-range regression test
   (`mv_component_bits_matches_reference_over_full_range`: every value
   in `-1023..=1023` × three contexts including a 0/255 clamp-corner
   sweep, `==` on f64 — not approximate).
2. **`small_mv_bits` / `write_small_mv` drop the recursive DFS + `Vec`.**
   §17.1 `small_mvtree` is a perfect depth-3 tree whose spec listing
   comments give the leaf↔path correspondence directly (`0 = "000"` …
   `7 = "111"`), so both walks now descend the tree with the 3-bit
   binary expansion of the leaf (MSB first), still indexing
   `SMALL_MVTREE` for the node-halved probability offsets and
   debug-asserting the landing leaf. No allocation, no search.
3. **`treed_bits` / `BoolEncoder::write_treed` get a fixed-buffer DFS.**
   The shared `treed_find_path` helper writes the path into a stack
   `[bool; 16]` (a §8.1 path visits distinct internal nodes, and the
   deepest tree the crate walks — `BMODE_TREE` — has 9), replacing the
   per-call `Vec<bool>` in both the §11 mode-cost and mode-emit walks.

### Measured A/B (criterion `--quick`, Apple M4 / aarch64, stable)

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 10.32 ms | **9.38 ms** | **−9.1 %** |
| `keyframe_encode/encode_keyframe_320x240_qi32` | 5.83 ms | **5.53 ms** | **−5.2 %** |

Throughput: inter 6.35 → **6.99 Mpx/s**, keyframe 13.17 → **13.89
Mpx/s**. The post-change profile (same attach methodology) confirms the
causal chain: libm `log2` disappears from the top-of-stack list
entirely (411 → 0 samples), the `mv_component_bits` family drops 305 →
123, the tree walk shows up only as the allocation-free
`treed_find_path::dfs` (115 samples), and the malloc/free family drops
~540 → ~300 samples. Encoder output is bit-identical: the RD costs are
the same doubles, so every pick is unchanged — the full fixture /
roundtrip suite passes untouched (lib tests 481 → 482 with the new
full-range cost regression anchor).

## Round 277 — compile-time SPLITMV partition-group table

Round 277 (2026-06-11) ran the attribution pass the round-276 candidate
asked for. A fresh `sample(1)` profile of the inter-encode bench (15 s
attach, 1 ms interval, `--measurement-time 60`) counted ~335
allocator-family top-of-stack self-samples (`_xzm_free` 98,
`_xzm_xzone_malloc` 43, `_malloc_zone_malloc` 24, `finish_grow` 25,
`grow_one` 20, plus dedup/realloc tails). Call-tree attribution put
essentially **all** of it under `encoder::partition_groups` — NOT the
`near_mv` census (which is stack-only) nor the per-frame `stream`
assembly the candidate guessed. `partition_groups` rebuilt a
`Vec<Vec<usize>>` (1 outer + up to 16 inner allocations) on every call,
and the SPLITMV scorer calls it once per (MB, partition shape) — 4× per
MB per reference frame; the per-candidate `SplitMvCandidate`
`submv_modes` / `submv_new_diffs` `Vec` pair accounted for the rest.

Two changes, bit-identical encoder output:

1. **`partition_groups` becomes a compile-time table.** A `const fn`
   decomposes each row of the §20.13 `MV_PARTITIONS` constant into a
   `PartitionGroups { members: [[usize; 16]; 16], len, num_groups }`
   record; the four records live in a `static` and the function now
   returns `&'static PartitionGroups`. Same single source of truth
   (the spec table), zero runtime work, zero allocation. Member order
   inside each group stays raster-ascending so `group[0]` remains the
   §16.4 anchor.
2. **`SplitMvCandidate` drops its two `Vec`s** for fixed `[_; 16]`
   arrays indexed by group id; the emit path already walks exactly
   `num_groups` entries so the tail fill values are never read.

### Measured A/B (criterion, 30 s measurement, Apple M4 / aarch64, stable)

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 8.96 ms | **8.83 ms** | **−1.4 %** |

Throughput: inter 7.31 → **7.39 Mpx/s** (same-session A/B; criterion's
stored-baseline estimate said −2.2 %). The post-change profile confirms
the causal chain: the allocator family disappears from the ≥5-sample
top-of-stack list entirely, and the same call-tree counting script
drops 794 → **40** allocator samples — what remains is the bounded
per-frame `stream` assembly (BoolEncoder output growth), not per-MB
churn. Bit-identity was checked two ways: the full 482-test suite plus
fixture/roundtrip integration suites, and an 18-frame A/B byte-hash
(3 resolutions × 6 frames × 3 quantisers through
`Vp8InterStreamEncoder`) — identical FNV-1a before/after.

## Round 278 — whole-frame keyframe path under nightly + `simd` (measurement round)

Round 278 (2026-06-11) closes the standing "whole-frame `keyframe_encode`
re-measure under nightly + `simd`" candidate. The round-247 note predicted
a sub-percent whole-frame delta, but that arithmetic only counted the
§14.3 / §14.4 *forward* primitives — since then the `simd` feature grew
the §14.1 dequant (round 267), §12.2 TM_PRED (round 268) and §18.3
six-tap (rounds 269–271, inter-only) kernels, all of which sit on the
keyframe encode's §11 RD-reconstruct loop or the decode path. The fresh
measurement says the whole-frame win is now far from sub-percent.

Methodology: `--measurement-time 30 --warm-up-time 3` (no `--quick`),
three interleaved scalar/simd run pairs per bench so thermal drift
cancels, nightly 1.97 toolchain for both A/B columns so the compiler
version cancels, separate `CARGO_TARGET_DIR`s per config, no concurrent
load. Stable 1.95 default-features runs provide the default-path anchor.
`keyframe_decode` rides along as the other half of the keyframe path
(the SIMD kernels mostly live on reconstruct/decode).

| Bench | Stable default | Nightly scalar (3 runs) | Nightly + `simd` (3 runs) | Δ (simd vs scalar) |
|---|---:|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 5.454 ms | 5.385 / 5.482 / 5.500 ms (mean 5.456) | 4.928 / 4.976 / 4.967 ms (mean **4.957**) | **−9.2 %** |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 154.7 µs | 151.0 / 152.7 / 151.6 µs (mean 151.8) | 120.6 / 119.6 / 119.8 µs (mean **120.0**) | **−21.0 %** |

Throughput: keyframe encode 14.08 → **15.49 Mpx/s** (+10 %); keyframe
decode 506 → **640 Mpx/s** (+27 %). Stable default agrees with the
nightly scalar column within run-to-run spread on both benches, so the
whole delta is the `simd` dispatch, not the compiler version. Every
scalar run sits in 5.385–5.500 ms and every simd run in 4.928–4.976 ms
— the two populations don't overlap, so the deltas are far outside the
measurement envelope (criterion's own change estimates between
consecutive same-config runs were ≤ 2 %).

Attribution (`sample(1)` PID-attach, 10 s, 1 ms interval, on the encode
bench under `--measurement-time 60`): the scalar profile's #2 self-time
symbol is `inverse_transform::inverse_dct_4x4` at 1350 samples (≈ 16 %
of in-process time — it fires 24× per MB inside the §11 RD loop's
reconstruct leg *and* per coded block in the final reconstruct); under
`simd` that symbol disappears from the ≥ 5-sample top-of-stack list
entirely (the `Simd<i32, 4>` body inlines into
`encode_mb_block_set_with_neighbors`). Secondary movers:
`intra_predict::predict_y16x16_tm` 32 → 7 samples (the round-268
−87.7 % kernel) and `inverse_wht_4x4` 6 → absent. The takeaway over the
round-180 micro-bench (which measured the inverse DCT SIMD at only −1
to −5 % per isolated call): in the real inlined context — back-to-back
calls over 24 sub-blocks with surrounding dequant + add-and-clamp code
— the vectorised body wins far more than the isolated-call number
suggested. Micro-bench deltas under-predict inlined whole-frame deltas
in both directions (round 226 saw the reverse); whole-frame A/B with
extended measurement time is the deciding instrument.

No code change shipped this round (measurement + attribution only), so
bit-identity is structural; the full stable lib suite (483) + nightly +
`simd` lib suite (485) were re-run green as a sanity anchor.

## Round 279 — whole-frame inter path under nightly + `simd`, and the MB-batched sub-pixel SAD

Round 279 (2026-06-11) closes the round-278 next-round candidate: the
symmetric whole-frame `inter_encode_short_clip` A/B under nightly +
`simd`, same methodology (`--measurement-time 30 --warm-up-time 3`,
three interleaved scalar/simd run pairs, nightly 1.97 for both columns,
separate `CARGO_TARGET_DIR`s, stable 1.95 default-features anchor).

### Measurement (pre-change crate state)

| Bench | Stable default | Nightly scalar (3 runs) | Nightly + `simd` (3 runs) | Δ (simd vs scalar) |
|---|---:|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 8.907 ms | 8.977 / 9.035 / 9.061 ms (mean 9.024) | 8.823 / 8.905 / 8.850 ms (mean **8.859**) | **−1.8 %** |

The populations don't overlap (every scalar run ≥ 8.977 ms, every simd
run ≤ 8.905 ms) so the −1.8 % is real — but it's an order of magnitude
smaller than the keyframe path's −9.2 %, and the attribution explains
why. `sample(1)` PID-attach (10 s, 1 ms interval) on the **scalar**
build puts `motion_comp::sixtap_2d` at #1 self-time (2162 of ~7700
in-process samples); on the **simd** build the same symbol is *still*
#1 at 2186 samples — flat. The call tree shows where it lives: the §17
sub-pixel refinements (`half_pixel_refine_luma` +
`quarter_pixel_refine_luma`) are ≈ 38 % of in-process time, almost all
inside `mb_luma_sad_at_mv`, which synthesised each of the 17 candidate
predictions per searched MB (center + 8 half-pel + 8 quarter-pel) as
**sixteen separate `filter_block_4x4` calls** — sixteen overlapping 9×9
`fetch_block_halo` fetches + sixteen 4×4 `sixtap_2d` convolutions per
candidate. In that shape the 4×4 `sixtap_2d_simd` kernel buys nothing
whole-frame (the rounds 269–271 MB-batched kernels only served the
*reconstruct* leg, one MB-pass per coded MB; the search leg dwarfs it
at 17 candidate syntheses per MB). The simd column's whole −1.8 % is
the §14 transform/dequant kernels on the RD/reconstruct leg
(`inverse_dct_4x4` 277 scalar samples → absent under `simd`), the
round-278 finding scaled to a 4-frame inter clip.

### The flagged hotspot, and the one targeted change

The measurement flags a clear scalar-shape hotspot that is *not* a
missing SIMD kernel: `mb_luma_sad_at_mv` should synthesise a candidate
the same way `predict_inter_mb`'s luma half already does. All sixteen
luma sub-blocks of a non-SPLITMV candidate share the one MV (§18.1), so
the round-270/271 MB-batched primitives apply verbatim:

* whole-pixel candidate (`stored_luma_mv(mv) & 7 == 0` both axes) →
  one contiguous [`fetch_luma_mb_whole_pixel`] 16×16 fetch;
* sub-pixel candidate → one 21×21 [`fetch_luma_mb_halo`] fetch + one
  whole-MB [`sixtap_mb_luma`] §18.3 pass.

Both are byte-exact with the per-sub-block tiling by the existing
equivalence proofs (`predict_inter_mb_sub_pixel_matches_per_block`,
`fetch_luma_mb_whole_pixel_matches_per_subblock_in_bounds`, the border
clamp tests, and the `sixtap_mb_luma` scalar/simd stress tests), so
every candidate SAD — and therefore every MV decision and every emitted
bit — is unchanged.

### Measured A/B (same interleaved extended-measurement methodology)

| Bench (post-change) | Stable default | Nightly scalar (3 runs) | Nightly + `simd` (3 runs) | Δ vs pre-change |
|---|---:|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 7.431 ms (**−16.6 %**) | 7.459 / 7.320 / 7.326 ms (mean 7.368, **−18.3 %**) | 6.955 / 6.952 / 6.953 ms (mean **6.953**, **−21.5 %**) | — |

Throughput: stable default 7.36 → **8.82 Mpx/s**; nightly scalar 7.26 →
**8.90 Mpx/s**; nightly simd 7.40 → **9.43 Mpx/s**. The simd-vs-scalar
whole-frame gap widens from −1.8 % to **−5.6 %** — the search leg now
runs through `sixtap_mb_luma`, whose SIMD body finally gets to act on
the dominant call site. Post-change simd attribution: `sixtap_2d`
disappears from the ≥ 5-sample top-of-stack list entirely;
`fetch_luma_mb_halo` shows 410 samples and `mb_luma_sad_at_mv` 1510
(the inlined batched kernel), down from the ~2960-sample refinement
cluster.

Bit-identity was checked three ways: the full stable suite (483 lib +
integration, 36 green targets), the nightly + `simd` lib suite (485),
and a 54-frame byte-hash A/B — 3 resolutions (64×48 / 128×128 /
176×144) × 6 frames × 3 quantisers (qi 12 / 32 / 96), keyframe
interval 3 so each stream carries two K + four P frames, all through
`Vp8InterStreamEncoder` — FNV-1a `2f655ee6d2a8a303` identical
pre-/post-change on stable *and* under nightly + `simd`.

## Round 281 — fused whole-pixel SAD scoring (`mb_luma_sad_at_whole_mv` / `group_sad_at_whole_mv`)

Round 281 (2026-06-12) closes the round-279 standing candidate
"whole-pixel SAD scoring batching". A fresh `sample(1)` PID-attach
profile (12 s, 1 ms interval, nightly + `simd` build under
`--measurement-time 60`) reproduced the round-279 finding as the
current #1 / #3 self-time symbols:

| Self-samples (of ~9276) | Symbol |
|---:|---|
| 2290 | `motion_comp::fetch_block_whole_pixel` |
| 1784 | `motion_search::mb_luma_sad_at_mv` (sub-pixel leg, round-279 shape) |
| 1371 | `encoder::group_sad_at_whole_mv` |
| 590 | `encoder::encode_p_frame_multi_ref_inner_with_counts_and_pick` |
| 458 | `motion_comp::fetch_luma_mb_halo` |

The §17 integer-pixel diamond descent (`mb_luma_sad_at_whole_mv`) and
the SPLITMV group scorer (`group_sad_at_whole_mv`) still fetched
sixteen (or per-group fewer) separate 4×4 patches per whole-pixel
candidate — sixteen `fetch_block_whole_pixel` bounds checks + scratch
copies per candidate, plus (on the SPLITMV side) a per-member 4×4
source extraction copy. Both scorers now run a **fused fetch-and-SAD**
(the round-279 candidate's parenthetical "a fused fetch-and-SAD that
never materialises the patch"): a whole-pixel candidate's §18.3
prediction is "simply copied", i.e. a direct window into the reference
plane at integer offset `(mb_x, mb_y) + (eighth-pixel mv >> 3)`, so
when the 16×16 MB-extent source region lands strictly in-bounds (the
dominant case mid-frame — and a conservative gate for every SPLITMV
group member) the SAD accumulates straight off the reference and
source rows. No prediction block is built and no source sub-block is
extracted. Border-straddling candidates fall back to the batched
`fetch_luma_mb_whole_pixel` (non-SPLITMV) or the original per-member
`fetch_block_whole_pixel` assembly (SPLITMV groups) — §20.14
`build_mc_border` edge replication preserved bit-for-bit.

### Measured A/B (`--measurement-time 30 --warm-up-time 3`, three interleaved pre/post pairs per config, Apple M4 / aarch64)

| Config | Pre (3 runs) | Post (3 runs) | Δ (means) |
|---|---:|---:|---:|
| stable 1.95 default | 7.403 / 7.183 / 7.130 ms (mean 7.239) | 6.317 / 6.152 / 6.130 ms (mean **6.200**) | **−14.4 %** |
| nightly 1.97 scalar | 7.347 / 7.242 / 7.153 ms (mean 7.247) | 6.308 / 6.247 / 6.139 ms (mean **6.231**) | **−14.0 %** |
| nightly 1.97 + `simd` | 6.910 / 6.818 / 6.775 ms (mean 6.834) | 5.876 / 5.815 / 5.758 ms (mean **5.816**) | **−14.9 %** |

Throughput: stable 9.05 → **10.57 Mpx/s**; nightly scalar 9.04 →
**10.52 Mpx/s**; nightly + `simd` 9.59 → **11.27 Mpx/s**. Every pre
run sits ≥ 7.119 ms (scalar) / ≥ 6.775 ms (simd) and every post run
≤ 6.376 ms / ≤ 5.904 ms — the populations don't overlap in any
config, so the deltas are far outside the measurement envelope. The
pre columns agree with the round-279 post-change numbers (7.43 /
7.37 / 6.95 ms) within session drift.

Post-change attribution (same `sample(1)` methodology):
`fetch_block_whole_pixel` collapses 2290 → **356** self-samples (the
residue is the SPLITMV border fallback + the per-sub-block reconstruct
paths); the fused SAD work now shows in its callers' own bodies
(`group_sad_at_whole_mv` 2007, `mb_luma_sad_at_whole_mv` 179 — tight
row loops instead of call + copy + re-read), and the new #1 is the
round-279-shaped sub-pixel `mb_luma_sad_at_mv` leg (2103), which
already runs the MB-batched §18.3 kernels.

Bit-identity was checked three ways: the full stable suite (38 green
targets, lib 483 → 486 with the new equivalence anchors
`whole_mv_sad_matches_per_subblock_fetch_assembly` — every MB of a
3×2-MB plane × 13 whole-pixel candidates including §17.1-extreme
border-straddlers in all four directions —
`whole_mv_sad_matches_assembly_with_padded_stride`, and
`group_sad_fused_fast_path_matches_per_subblock_fetch` — all four
§16.4 partition shapes × every group × 11 candidates × 3 MB
positions); the nightly + `simd` lib suite (485 → 488); and a 54-frame
byte-hash A/B — 3 resolutions (64×48 / 128×128 / 176×144) × 6 frames
× 3 quantisers (qi 12 / 32 / 96), keyframe interval 3, all through
`Vp8InterStreamEncoder` — FNV-1a `1495730e2f66b0a3` (17 273 bytes)
identical pre-/post-change on stable *and* under nightly + `simd`.
(The hash differs from the round-279 record because the round-281
harness drives a different deterministic source — a drifting gradient
plus a moving high-contrast square so SPLITMV does real work; what
matters is pre == post within the round, on both toolchains.)

## What didn't get touched yet (next-round candidates)

* **~~Remaining allocator churn (`malloc` / `free`)~~** — CLOSED in
  round 277 (see above): the churn was `partition_groups`'
  per-call `Vec<Vec<usize>>` + the `SplitMvCandidate` `Vec` pair, both
  now allocation-free. The residual ~40 call-tree allocator samples are
  the per-frame partition assembly (`BoolEncoder` output `Vec` growth
  in `stream`) — bounded per frame, not per MB, and not worth a
  speculative pre-reserve without a fresh profile pointing at it.
* **~~SPLITMV whole-pixel sub-block batching~~** — closed negative in round
  274 (see the round-274 section above): SPLITMV sub-blocks carry distinct
  vectors so the shared-halo batch can't apply, and the only remaining
  freedom (the per-sub-block write strategy) measured ~17 % SLOWER as a
  strided write than the shipped contiguous `[u8; 16]`-scratch copy. The
  candidate is retired.
* **~~Whole-frame `keyframe_encode` re-measure under nightly + `simd`~~**
  — CLOSED in round 278 (see above): measured **−9.2 %** whole-frame
  keyframe encode and **−21.0 %** keyframe decode under nightly +
  `simd`, attributed primarily to `inverse_dct_4x4_simd` in the §11
  RD-reconstruct loop (scalar profile's #2 self-time symbol at ≈ 16 %,
  gone under `simd`). The round-247 sub-percent prediction predated the
  round-267/268 dequant + TM_PRED kernels and under-counted the inlined
  inverse-DCT win.
* **~~Whole-frame `inter_encode_short_clip` re-measure under nightly +
  `simd`~~** — CLOSED in round 279 (see above): the pre-change simd
  dispatch was worth only **−1.8 %** whole-frame because the dominant
  §18.3 six-tap work sat in the *search* scoring path
  (`mb_luma_sad_at_mv`, 17 per-4×4-tiled candidate syntheses per MB),
  not the reconstruct leg the MB-batched kernels served. Routing that
  scoring path through the round-270/271 MB-batched primitives (bit-
  identical, 54-frame byte-hash proof) measured **−16.6 %** stable /
  **−18.3 %** nightly scalar / **−21.5 %** nightly simd whole-frame,
  and widened the simd-vs-scalar gap to **−5.6 %**.
* **~~Whole-pixel SAD scoring batching~~** — CLOSED in round 281 (see
  above): both whole-pixel scorers (`mb_luma_sad_at_whole_mv` and the
  SPLITMV `group_sad_at_whole_mv`) now run a fused fetch-and-SAD
  straight off the reference rows when the MB-extent source region is
  in-bounds, with the §20.14 border fallback unchanged. Measured
  **−14.4 %** stable / **−14.0 %** nightly scalar / **−14.9 %**
  nightly + `simd` whole-frame on `inter_encode_short_clip`;
  `fetch_block_whole_pixel` self-time collapsed 2290 → 356 samples;
  bit-identical (3 new equivalence anchors + 54-frame byte-hash A/B).
* **Sub-pixel SAD without patch materialisation** — the post-round-281
  profile's #1 is the sub-pixel `mb_luma_sad_at_mv` leg (2103
  self-samples) + `fetch_luma_mb_halo` (590): each half-/quarter-pixel
  candidate still materialises the full 16×16 `sixtap_mb_luma` output
  before `block_sad_16x16` reads it back. A fused variant could SAD
  each output row as the vertical pass produces it (the row lives in
  an `i32` vector already under `simd`), skipping the narrow-to-u8
  store + reload — but the clamp-to-u8 must stay bit-exact, and the
  scalar path's auto-vectorisation may already cover most of the
  margin. Needs a micro-bench first; the round-258 lesson (SIMD leaf
  inlining regressing the surrounding descent) applies.
