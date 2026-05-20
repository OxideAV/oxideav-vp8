# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status — 2026-05-20

**Clean-room rebuild in progress.** The prior implementation was
retired under the workspace clean-room policy after a provenance audit
on 2026-05-20. Rebuild work tracks RFC 6386 exclusively, with
black-box `ffmpeg` invocations as the only validator.

### Landed (round 1)

* `BoolDecoder` — the VP8 boolean (range) entropy decoder of
  RFC 6386 §7. Every higher-level decode step in VP8 (frame header,
  macroblock mode, motion vectors, DCT tokens) reads through this
  primitive, so it is the foundation everything else builds on.
  Surface:
  * `init(&[u8])` — load the first two bytes of a partition into
    the `value` register; reject inputs shorter than two bytes.
  * `read_bool(prob: u8)` — read one boolean coded at probability
    `prob/256` of being zero.
  * `read_literal(num_bits)` — `num_bits` flags read MSB-first.
  * `read_signed_literal(num_bits)` — sign-bit-then-magnitude with
    the spec's `-1`-initialised accumulator.
  * Surfaces `EndOfStream` when renormalisation needs a byte the
    partition no longer has.

Nine unit tests exercise init validation, round-trip against an
in-test reference encoder over a probability sweep, literal/signed-
literal handling, the range invariant (128 ≤ range ≤ 255 after every
successful read), and the end-of-stream signal.

### Not yet landed

Frame header parsing, macroblock decode, intra/inter prediction, loop
filter, IDCT / inverse-WHT, and the encoder. All top-level entry
points (`decode_vp8`, `encode_vp8_keyframe`) still return
`Error::NotImplemented`.

## Clean-room sources

* RFC 6386 — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).
* Black-box invocations of the `ffmpeg` *binary* as an opaque
  validator (no source consulted).

No external library source — libvpx, libaom, libavcodec/vp8\*, etc. —
is permitted as a reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
