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

## What didn't get touched yet (next-round candidates)

* **Remaining allocator churn (`malloc` / `free`)** — after r204 removed
  the token-path `Vec`, the round-170 profile's #2/#4/#5/#7 hits
  (`_xzm_*`) should have shifted; re-profile to find the next biggest
  short-lived `Vec` (the §11 mode picker + `near_mv` MV-candidate
  scratch are the most likely remaining offenders).
* **`sixtap_2d` (#4 on inter)** — the inner 6-tap convolution is a
  natural SIMD target (`Simd<i16, 8>` for an 8-pixel-wide stripe).
  Held back this round to keep the SIMD-feature surface focused on
  the §14 transform primitives.
* **Whole-frame `keyframe_encode` re-measure under nightly + `simd`**
  — round 247's per-primitive `--quick` numbers imply a sub-percent
  whole-frame win, deep below the bench's `--quick` noise envelope. A
  profile-depth round with `--measurement-time` extended could attribute
  it cleanly instead of letting it sit inside the per-frame noise.
