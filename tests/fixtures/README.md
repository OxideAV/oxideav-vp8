# VP8 decode fixtures

Each subdirectory holds one VP8 conformance fixture used by the
`decode_vp8` end-to-end tests in `src/decoder.rs`:

* `input.ivf` — a single-key-frame VP8 elementary stream wrapped in IVF
  (32-byte DKIF header + one 12-byte frame header + the raw VP8 frame).
* `expected.yuv` — the I420 reference picture (`Y` plane, then `U`, then
  `V`, each row-major), produced by an opaque black-box reference
  decoder binary used purely as a validator.

These are public conformance vectors: the input bitstreams were generated
by the reference encoder binary and the expected output is whatever the
reference decoder produces. Under the workspace clean-room policy, black-box
validator *output* is a legitimate oracle — no decoder source is consulted
or reproduced; we compare our bytes to theirs. RFC 6386 §2 makes the exact
reconstructed pixel values part of the specification, so a bit-exact match
against the reference is the correctness bar.

The same fixtures live in the workspace under
`docs/video/vp8/fixtures/<name>/` (with their `notes.md` / `trace.txt`).
They are vendored here so the standalone `oxideav-vp8` crate repo (which
does not carry the workspace `docs/` tree) can `include_bytes!` them at
test-build time.
