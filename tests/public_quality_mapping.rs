//! Standalone-reachable lock-test for [`oxideav_vp8::quality_to_qindex`].
//!
//! `quality_to_qindex` is the libwebp-style quality (`0.0..=100.0`) to
//! VP8 §9.6 `y_ac_qi` (`0..=127`) mapping that
//! `oxideav-webp`'s lossy path relies on. The function is intentionally
//! pure (no `oxideav-core` dep) so it MUST stay reachable under
//! `--no-default-features` for embedded image / video pipelines that
//! want to pick a `qindex` without building the framework adapter. This
//! test compiles + runs against both feature configurations.

use oxideav_vp8::{encoder, quality_to_qindex};

#[test]
fn module_path_and_crate_root_resolve_to_same_function() {
    assert_eq!(quality_to_qindex(75.0), encoder::quality_to_qindex(75.0));
}

#[test]
fn quality_100_maps_to_qindex_0_best() {
    // round((100 - 100) * 1.27) = 0.
    assert_eq!(quality_to_qindex(100.0), 0);
}

#[test]
fn quality_0_maps_to_qindex_127_worst() {
    // round((100 - 0) * 1.27) = round(127.0) = 127.
    assert_eq!(quality_to_qindex(0.0), 127);
}

#[test]
fn quality_75_maps_to_qindex_32_libwebp_default() {
    // round((100 - 75) * 1.27) = round(31.75) = 32 (half-away-from-zero).
    assert_eq!(quality_to_qindex(75.0), 32);
}

#[test]
fn quality_50_maps_to_qindex_64() {
    // round((100 - 50) * 1.27) = round(63.5) = 64 (half-away-from-zero).
    assert_eq!(quality_to_qindex(50.0), 64);
}

#[test]
fn quality_25_maps_to_qindex_95() {
    // round((100 - 25) * 1.27) = round(95.25) = 95.
    assert_eq!(quality_to_qindex(25.0), 95);
}

#[test]
fn negative_quality_clamps_to_worst() {
    // Negative → clamp to 0 → 127.
    assert_eq!(quality_to_qindex(-1.0), 127);
    assert_eq!(quality_to_qindex(-100.0), 127);
}

#[test]
fn over_max_quality_clamps_to_best() {
    // > 100 → clamp to 100 → 0.
    assert_eq!(quality_to_qindex(101.0), 0);
    assert_eq!(quality_to_qindex(1000.0), 0);
}

#[test]
fn nan_quality_returns_worst_qindex() {
    // Documented: NaN → 127 (the "couldn't tell" / "keep file small"
    // fallback). Pins the contract.
    assert_eq!(quality_to_qindex(f32::NAN), 127);
}

#[test]
fn quality_function_is_monotonic_non_increasing() {
    // Higher quality always yields lower-or-equal qindex.
    let mut prev = quality_to_qindex(0.0);
    let mut q = 1u32;
    while q <= 100 {
        let cur = quality_to_qindex(q as f32);
        assert!(
            cur <= prev,
            "quality_to_qindex({q}) = {cur} > {prev} at q-1: monotonicity broken"
        );
        prev = cur;
        q += 1;
    }
}
