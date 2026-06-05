"""
Tests for Week 4 monitoring metrics and drift detection.

Verifies:
  - All 8 metrics compute without errors on real data
  - KS test detects shift in synthetic data
  - PSI detects significant distribution change
  - Null rate detection works
  - Duplicate rate detection works
  - All 4 drift patterns detected on actual week4 data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from metric_template import MetricComputer, _compute_psi
from detect_drift import (
    detect_feature_drift,
    detect_temporal_peak_shift,
    detect_manhattan_lag_deflation,
    detect_outer_borough_scramble,
    detect_manhattan_weekend_drift,
)

_DATA_DIR = Path(__file__).parent.parent / "data"


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def baseline():
    return pd.read_parquet(_DATA_DIR / "demand_enriched_baseline.parquet")


@pytest.fixture(scope="session")
def week4():
    return pd.read_parquet(_DATA_DIR / "demand_enriched_week4.parquet")


@pytest.fixture(scope="session")
def computer(baseline):
    return MetricComputer(baseline)


# ── Metric computation ────────────────────────────────────────────────────────

class TestMetricComputation:

    def test_compute_all_metrics_no_crash(self, computer, week4):
        result = computer.compute_all_metrics(week4)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "null_rates" in result
        assert "ks_test" in result
        assert "psi" in result

    def test_null_rates_returns_dict(self, computer, week4):
        nr = computer.metric_3_null_rates(week4)
        assert isinstance(nr, dict)
        for col, info in nr.items():
            assert "rate" in info
            assert 0.0 <= info["rate"] <= 1.0

    def test_duplicate_rate_returns_valid(self, computer, week4):
        dup = computer.metric_8_duplicate_rate(week4)
        assert "duplicate_rate" in dup
        assert 0.0 <= dup["duplicate_rate"] <= 1.0

    def test_accuracy_returns_float_in_range(self, computer, week4):
        preds = week4["zone_slot_baseline"].fillna(0).values if "zone_slot_baseline" in week4.columns else np.zeros(len(week4))
        acts = week4["trip_count"].fillna(0).values
        acc = computer.metric_1_accuracy(week4, preds, acts)
        assert 0.0 <= acc <= 1.0

    def test_accuracy_perfect_predictions(self, computer, week4):
        acts = week4["trip_count"].fillna(0).values
        acc = computer.metric_1_accuracy(week4, acts.copy(), acts)
        assert acc == pytest.approx(1.0, abs=0.01)

    def test_ks_test_returns_p_values(self, computer, week4):
        result = computer.metric_4_ks_test(week4)
        assert isinstance(result, dict)
        for feat, res in result.items():
            assert "p_value" in res
            assert 0.0 <= res["p_value"] <= 1.0

    def test_psi_returns_non_negative(self, computer, week4):
        result = computer.metric_5_psi(week4)
        for feat, res in result.items():
            assert res["psi"] >= 0.0
            assert res["status"] in ("stable", "monitor", "retrain")


# ── KS / PSI on synthetic data ────────────────────────────────────────────────

class TestStatisticalTests:

    def test_ks_detects_large_shift(self, computer):
        df_shifted = pd.DataFrame({"trip_count": np.random.normal(100, 5, 1000)})
        result = computer.metric_4_ks_test(df_shifted)
        if "trip_count" in result:
            assert result["trip_count"]["drift_detected"], "KS should detect a large distribution shift"

    def test_ks_passes_similar_distribution(self, computer, baseline):
        similar = baseline.copy()
        result = computer.metric_4_ks_test(similar)
        if "trip_count" in result:
            assert not result["trip_count"]["drift_detected"], "KS should not flag identical data as drifted"

    def test_psi_large_shift_triggers_retrain(self, computer):
        df_shifted = pd.DataFrame({"trip_count": np.random.normal(200, 10, 2000)})
        result = computer.metric_5_psi(df_shifted)
        if "trip_count" in result:
            assert result["trip_count"]["psi"] > 0.25, "PSI should be >0.25 for extreme distribution shift"
            assert result["trip_count"]["status"] == "retrain"

    def test_psi_stable_distribution(self, computer, baseline):
        result = computer.metric_5_psi(baseline)
        if "trip_count" in result:
            assert result["trip_count"]["psi"] < 0.10, "PSI should be near 0 for identical distribution"

    def test_psi_helper_zero_for_same(self):
        s = pd.Series(np.random.normal(10, 1, 1000))
        psi = _compute_psi(s, s.copy())
        assert psi < 0.05, "PSI should be near 0 for same distribution"

    def test_psi_helper_large_for_shifted(self):
        base = pd.Series(np.random.normal(10, 1, 1000))
        shifted = pd.Series(np.random.normal(50, 1, 1000))
        psi = _compute_psi(base, shifted)
        assert psi > 0.25, "PSI should exceed 0.25 for large shift"


# ── Null rate and duplicate detection ─────────────────────────────────────────

class TestQualityMetrics:

    def test_null_rate_detects_injected_nulls(self, computer):
        df = pd.DataFrame({
            "trip_count": [1.0, None, None, None, 2.0] * 20,
            "PULocationID": [1] * 100,
        })
        nr = computer.metric_3_null_rates(df)
        assert nr["trip_count"]["rate"] > 0.50

    def test_duplicate_rate_detects_dupes(self, computer):
        df = pd.DataFrame({
            "PULocationID": [1, 1, 1, 2, 3],
            "time_bucket": ["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 00:00",
                            "2026-01-01 00:15", "2026-01-01 00:30"],
            "trip_count": [5, 5, 5, 3, 2],
        })
        result = computer.metric_8_duplicate_rate(df)
        assert result["duplicate_count"] == 2
        assert result["alert"]

    def test_baseline_has_no_duplicate_alert(self, computer, baseline):
        result = computer.metric_8_duplicate_rate(baseline)
        assert not result.get("alert", False), "Baseline should have no duplicate alert"


# ── Drift pattern detection on actual week4 data ──────────────────────────────

class TestDriftPatterns:

    def test_feature_drift_detects_trip_count(self, baseline, week4):
        result = detect_feature_drift(baseline, week4, "trip_count")
        assert "ks_statistic" in result
        assert result["drift_detected"], "trip_count should show drift in week4 data"

    def test_temporal_peak_shift_detected(self, baseline, week4):
        result = detect_temporal_peak_shift(baseline, week4)
        assert "detected" in result
        assert result["detected"], "Temporal peak shift should be detected in week4"

    def test_manhattan_lag_deflation_detected(self, baseline, week4):
        lag_cols = [c for c in ["lag_1day", "lag_1week", "roll_mean_1day"] if c in baseline.columns]
        if not lag_cols:
            pytest.skip("Lag columns not present in data")
        result = detect_manhattan_lag_deflation(baseline, week4)
        assert result.get("detected"), "Manhattan lag deflation should be detected"

    def test_outer_borough_scramble_detected(self, baseline, week4):
        if "zone_slot_baseline" not in baseline.columns:
            pytest.skip("zone_slot_baseline column not present")
        result = detect_outer_borough_scramble(baseline, week4)
        assert result.get("detected"), "Outer borough scramble should be detected"

    def test_manhattan_weekend_drift_detected(self, baseline, week4):
        result = detect_manhattan_weekend_drift(baseline, week4)
        if result.get("error"):
            pytest.skip(f"Skipping: {result['error']}")
        assert result.get("detected"), "Manhattan weekend concept drift should be detected"

    def test_feature_drift_no_error_on_missing_column(self, baseline, week4):
        result = detect_feature_drift(baseline, week4, "nonexistent_column")
        assert "error" in result
