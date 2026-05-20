# Changelog

All notable changes to `oxideav-vp8` are recorded here.

## [Unreleased]

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained. License also reset to clean MIT.

  Every public API path now returns `Error::NotImplemented`. A
  clean-room re-implementation against RFC 6386 is planned for a
  future round.
