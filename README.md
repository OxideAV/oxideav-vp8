# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status

**Production-ready, both decoder and encoder at ✅ 100% as of 2026-05-27.**

* **Decoder** — full RFC 6386 key + inter decode, bit-exact against
  the reference output on 10+ multi-frame fixtures including
  5-frame mid-GOP golden refresh, 10-frame auto-alt-ref, and ARNR.
  Stateful multi-frame `Vp8DecoderState` driver.
* **Encoder** — Phase-2 I + P with SPLITMV, GOLDEN / ALTREF,
  multi-partition residual, RefreshControls, LoopFilterDeltas, §11
  intra picking, §13.4 token-probability fitter, and a real
  complexity-aware **two-pass rate-control** family
  (`first_pass_analyze` + `two_pass_qindices` + `Vp8TwoPassEncoder`)
  that distributes per-frame qindex around a target using log-MAD
  + log-variance first-pass statistics.
* **0.1.13 public-surface lock** — every symbol the published
  crates.io `0.1.13` release exposed is reachable, both with the
  default `registry` build and under `--no-default-features`. See
  [`API-COMPAT-0.1.13.md`](./API-COMPAT-0.1.13.md) for the
  per-symbol contract and [`tests/api_compat_0_1_13.rs`](./tests/api_compat_0_1_13.rs)
  for the compile-only assertion suite.

## Cargo features

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Enables the `oxideav-core` dependency and the framework-trait factories (`make_encoder` / `make_decoder` returning `Box<dyn Encoder>` / `Box<dyn Decoder>`). |
| `simd` | — | Reserved no-op flag carried over from the 0.1.13 manifest; preserved so historical consumers that set it explicitly keep building. |

The crate builds and tests cleanly under both `cargo build -p oxideav-vp8`
and `cargo build -p oxideav-vp8 --no-default-features`; both
configurations are kept green in CI.

## Direct-API entry points

### Decode

```rust
use oxideav_vp8::{decode_vp8, Vp8DecoderState, Vp8Frame};

// One-shot single-frame decode (caller manages multi-frame state).
let frame: Vp8Frame = decode_vp8(&vp8_keyframe_bytes)?;

// Stateful multi-frame decode (handles inter frames + reference buffers).
let mut state = Vp8DecoderState::new();
for packet in vp8_packets {
    let frame = state.decode_packet(packet)?;
    // ... consume `frame.y` / `frame.u` / `frame.v`
}
```

### Encode

```rust
use oxideav_vp8::encoder::{
    make_encoder_with_quality,   // 0.0..=100.0 quality scalar
    make_encoder_with_qindex,    // 0..=127 explicit qindex (lower = better)
    make_encoder_with_config,    // full Vp8EncoderConfig (segmentation, RDO, …)
    Vp8TwoPassEncoder,           // complexity-aware two-pass rate control
    Vp8TwoPassConfig,
};
```

`encode_vp8_keyframe(pixels, width, height)` emits a standalone VP8
keyframe bitstream without pulling the `oxideav-core` framework
dependency.

## Registry path

With the default `registry` feature on:

```rust
let mut ctx = oxideav_core::RuntimeContext::new();
oxideav_vp8::register(&mut ctx);
// ctx.codecs now has a "vp8" decoder + encoder factory.
```

The codec id is `oxideav_vp8::CODEC_ID_STR` (= `"vp8"`).

## Clean-room sources

Implementation is derived entirely from the public format spec:

* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).

Fixture `expected.yuv` reference pictures are produced by black-box
invocations of the reference decoder *binary*; no third-party codec
library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
