# oxideav-vp8 fuzz harnesses

Panic-free harnesses for the public decode-side API of `oxideav-vp8`.
Each target feeds arbitrary libFuzzer bytes through one layer of the
RFC 6386 decode stack and asserts that no input — well-formed,
malformed, or hostile — causes a panic, abort, debug-arithmetic
overflow, or out-of-bounds index.

## Targets

| Target | Surface under test | What it exercises |
|--------|--------------------|-------------------|
| `panic_free_decode_keyframe` | `decode_vp8` | One-shot keyframe decode end-to-end (§9.1 header → §19.2 coded header → §11 / §12 / §13 / §14 / §15 pipeline). Pre-flighted by the §9.1 dimension cap below. |
| `panic_free_decoder_state`   | `Vp8DecoderState::decode_frame` | Stateful multi-packet driver. Exercises the §9.7 reference-frame refresh ladder (LAST / GOLDEN / ALTREF) — the extreme-reference-dependency path a one-shot decode call can never reach. |
| `parse_headers`              | `frame_tag::parse_header`, `frame_tag::parse_keyframe_header`, `frame_header::Vp8FrameHeader::parse`, `coded_header::Vp8CodedHeader::parse` (key + inter), `ivf::parse_header`, `ivf::parse_frame_header` | Pure-parse layer. The §19.2 coded-header walk routes through `update_segmentation`, `mb_lf_adjustments`, `quant_indices`, `token_prob_update`, and `mv_prob_update`. |

The harnesses are **decode-only**: no oracle, no comparison against
any external implementation. The contract is panic-freedom, not
output equivalence.

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
```

`cargo-fuzz` requires the nightly toolchain for libFuzzer's
sanitiser instrumentation. The crate itself builds on stable; only
the fuzz binaries need nightly.

## Corpus

The repository ships **no** seed corpus. libFuzzer starts from empty
and discovers structure on its own; the three targets each converge
on coverage of their respective surface within a few minutes on a
single core.

## CI

The fuzz crate is intentionally a separate nested workspace
(`[workspace] members = ["."]` in its `Cargo.toml`) so it is NOT
pulled into the umbrella's `crates/*` glob. The umbrella CI does
not run fuzz iterations; the targets are exercised on demand by
maintainers and during pre-release hardening.
