# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386).

## Status — 2026-05-20

**Orphan-rebuild scaffold.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core modules could not be defended against the "no external library
source as reference" rule that governs every crate in this workspace.

Per workspace policy, the only acceptable response is a full
clean-room re-implementation against RFC 6386 and black-box validator
binaries. That work has not yet been scheduled.

Every public entry point currently returns `Error::NotImplemented`.

## Planned clean-room sources

The clean-room rebuild will consult only:

* RFC 6386 — VP8 Data Format and Decoding Guide.
* Black-box invocations of `ffmpeg` (the binary — not its source) as
  an opaque validator.

No external library source — libvpx, libavcodec/vp8*, etc. — is
permitted as a reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
