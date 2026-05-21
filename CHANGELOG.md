# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Added

* **Uncompressed frame header parser** per RFC 6386 §9.1 / §19.1
  (`Vp8FrameHeader::parse` in `src/frame_header.rs`). Splits the
  3-byte little-endian frame tag into `key_frame`, 3-bit `version`,
  `show_frame`, and 19-bit `first_partition_size`. Maps `version` to
  the §9.1 Table 1 `ReconstructionFilter` / `LoopFilterPolicy`
  enums. On key frames validates the `0x9d 0x01 0x2a` start code and
  splits each of the two LE 16-bit size words into a 14-bit
  dimension and a 2-bit `ScaleCode`. Surfaces `header_bytes_consumed`
  (3 for interframes, 10 for key frames) so callers can advance to
  the first (control) partition. Nine unit tests, including a
  sanity-check parse of the actual first 10 bytes of
  `docs/video/vp8/fixtures/tiny-i-only-16x16/input.ivf`.
* **Boolean (range) entropy decoder** per RFC 6386 §7
  (`BoolDecoder` in `src/bool_decoder.rs`). This is the foundational
  primitive every higher-level VP8 decode step reads through. Surface:
  `init`, `read_bool`, `read_literal`, `read_signed_literal`, plus
  `range()` / `value()` / `remaining_input()` accessors for testing.
  An `EndOfStream` error is surfaced explicitly when renormalisation
  needs a byte the partition no longer has, rather than silently
  returning stale bits.
* Nine unit tests covering init validation, round-trip against an
  in-test reference encoder over a probability sweep, literal /
  signed-literal handling (including the spec's `-1`-initialised
  signed accumulator), the `128 ≤ range ≤ 255` invariant across
  reads, and the end-of-stream signal.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained. License also reset to clean MIT.

  Every public API path other than the new `BoolDecoder` still
  returns `Error::NotImplemented`. A clean-room re-implementation of
  the frame header, macroblock decode, prediction, IDCT, and loop
  filter is planned for subsequent rounds.
