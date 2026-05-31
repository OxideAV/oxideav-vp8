# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386). Decoder and encoder both at
production status as of 2026-05-27.

* RFC 6386 key + inter decode, bit-exact against the reference output
  on 10+ multi-frame fixtures (mid-GOP golden refresh, 10-frame
  auto-alt-ref, ARNR).
* Phase-2 encoder with SPLITMV, GOLDEN / ALTREF, multi-partition,
  RefreshControls, LoopFilterDeltas, §11 intra picker, §13.4
  token-probability fitter, and a complexity-aware two-pass
  rate-control family.
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
| `simd` | — | **Nightly-only.** Switches the §14.3 inverse 4×4 WHT (`inverse_wht_4x4`) and the §14.4 inverse 4×4 DCT (`inverse_dct_4x4`) over to `core::simd::Simd<i32, 4>` rewrites; both are byte-exact against the scalar path on a 21-input stress set (DC-only across 10 magnitudes, single-AC at every position, mixed gradients). Headline micro-bench numbers on `aarch64-apple-darwin`: WHT 9.4 → 7.5 ns (≈ −20 %); DCT 10.07 → ~9.5 ns (≈ −1 to −5 %, the DCT is multiply-heavy so per-lane serialisation eats most of the win). Requires a nightly toolchain because `core::simd` is itself nightly. Default (stable) builds use the scalar §14.3 / §14.4 paths unchanged. See `BENCHMARKS.md` for the A/B numbers + the round-170 / round-180 profiles that motivated each primitive's vectorisation. |

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

The crate ships three `cargo-fuzz` targets under [`fuzz/`](./fuzz/)
that exercise the public decode-side surface for panic-freedom:

* `panic_free_decode_keyframe` — one-shot `decode_vp8`, dimension-gated
  at 256 × 256.
* `panic_free_decoder_state` — multi-packet `Vp8DecoderState::decode_frame`
  drives the §9.7 LAST / GOLDEN / ALTREF refresh ladder.
* `parse_headers` — `frame_tag::parse_header` / `parse_keyframe_header`,
  `Vp8FrameHeader::parse`, `Vp8CodedHeader::parse` (key + inter — the
  §19.2 segmentation-map / MB-LF-adjustments / token-prob-update /
  MV-prob-update walk), and the IVF framing layer.

Initial smoke pass: 800 000 combined iterations, zero panics. See
[`fuzz/README.md`](./fuzz/README.md) for caps, run instructions, and
the rationale behind each target's pre-flight gating.

## License

MIT. See [`LICENSE`](./LICENSE).
