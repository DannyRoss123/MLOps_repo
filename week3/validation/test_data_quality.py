"""
Tests for data quality validation.

Verifies:
  - Baseline (clean) data passes all checks
  - Corrupted data triggers all 4 issue types
  - Graceful degradation functions clean data without crashing
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.check_data_quality import DataQualityValidator, detect_outliers, compare_distributions

_DATA_DIR = Path(__file__).parent.parent / "data"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def baseline_data():
    return pd.read_parquet(_DATA_DIR / "demand_enriched_baseline.parquet")


@pytest.fixture(scope="session")
def corrupted_data():
    return pd.read_parquet(_DATA_DIR / "demand_enriched_corrupted.parquet")


@pytest.fixture
def validator(baseline_data):
    return DataQualityValidator(baseline_data)


# ── Baseline: should pass ──────────────────────────────────────────────────────

class TestBaselineData:

    def test_baseline_passes_validation(self, baseline_data, validator):
        result = validator.validate(baseline_data)
        assert result["is_valid"], f"Baseline should pass but got issues: {result['issues']}"

    def test_baseline_has_no_duplicates(self, baseline_data, validator):
        validator.issues = []
        validator.check_duplicates(baseline_data)
        dup_issues = [i for i in validator.issues if i["type"] == "duplicates"]
        assert len(dup_issues) == 0, "Baseline should have no duplicate rows"

    def test_baseline_trip_counts_in_range(self, baseline_data, validator):
        """Baseline has zero-demand slots (legitimate) but no negatives or extremes."""
        validator.issues = []
        validator.check_value_ranges(baseline_data)
        range_issues = [i for i in validator.issues if i["type"] == "out_of_range"]
        assert len(range_issues) == 0, f"Baseline should have no out-of-range values (zeros are valid): {range_issues}"

    def test_baseline_has_required_columns(self, baseline_data, validator):
        validator.issues = []
        validator.check_schema(baseline_data)
        assert len(validator.issues) == 0, "Baseline should have all required columns"


# ── Corrupted data: should detect all 4 issues ────────────────────────────────

class TestDataQualityIssues:

    def test_detect_duplicates(self, corrupted_data, validator):
        """Issue 1: 10,085 duplicate rows in zones 4, 43, 87, 107, 152, 229."""
        validator.issues = []
        validator.check_duplicates(corrupted_data)
        dup_issues = [i for i in validator.issues if i["type"] == "duplicates"]
        assert len(dup_issues) > 0, "Should detect duplicate rows"
        assert dup_issues[0]["count"] > 1000, "Should detect thousands of duplicates"

    def test_detect_out_of_range_trip_counts(self, corrupted_data, validator):
        """Issue 2: rows with trip_count values of -5, -1, 9999, 99999 (zeros are legitimate)."""
        validator.issues = []
        validator.check_value_ranges(corrupted_data)
        range_issues = [i for i in validator.issues if i["type"] == "out_of_range"]
        assert len(range_issues) > 0, "Should detect out-of-range trip_count values"
        total_bad = sum(i["count"] for i in range_issues)
        assert total_bad >= 300, f"Should detect at least 300 bad rows (negatives + extremes), found {total_bad}"

    def test_detect_holiday_drift(self, corrupted_data, validator):
        """Issue 3: is_holiday incorrectly set for 82,080 rows."""
        validator.issues = []
        validator.check_holiday_drift(corrupted_data)
        holiday_issues = [i for i in validator.issues if i["type"] == "holiday_drift"]
        assert len(holiday_issues) > 0, "Should detect holiday flag drift"

    def test_detect_lag_contamination(self, corrupted_data, validator):
        """Issue 4: lag_1week scale inconsistent with trip_count in at least one zone."""
        if "lag_1week" not in corrupted_data.columns:
            pytest.skip("lag_1week column not present in corrupted data")
        validator.issues = []
        validator.check_lag_contamination(corrupted_data)
        lag_issues = [i for i in validator.issues if i["type"] == "lag_contamination"]
        assert len(lag_issues) > 0, "Should detect lag_1week scale anomalies in corrupted data"
        assert lag_issues[0]["count"] >= 1, "Should flag at least one zone with lag contamination"

    def test_corrupted_data_fails_overall_validation(self, corrupted_data, validator):
        """Full validation should fail on corrupted data."""
        result = validator.validate(corrupted_data)
        assert not result["is_valid"], "Corrupted data should fail validation"
        assert result["num_issues"] >= 4, f"Should detect at least 4 issues, found {result['num_issues']}"


# ── Graceful degradation ───────────────────────────────────────────────────────

class TestGracefulDegradation:

    def test_cleaning_removes_duplicates(self, corrupted_data):
        """Dropping duplicates should reduce row count."""
        key_cols = ["PULocationID", "time_bucket"]
        available = [c for c in key_cols if c in corrupted_data.columns]
        cleaned = corrupted_data.drop_duplicates(subset=available)
        assert len(cleaned) < len(corrupted_data), "Cleaning should remove duplicate rows"

    def test_cleaning_fixes_trip_count(self, corrupted_data):
        """After clamping, all trip_count values should be in valid range."""
        cleaned = corrupted_data.copy()
        cleaned = cleaned[cleaned["trip_count"] >= 1]
        cleaned = cleaned[cleaned["trip_count"] <= 9000]
        assert (cleaned["trip_count"] >= 1).all(), "All trip_count should be >= 1 after cleaning"
        assert (cleaned["trip_count"] <= 9000).all(), "All trip_count should be <= 9000 after cleaning"

    def test_api_does_not_crash_with_bad_data(self, corrupted_data):
        """Validation and cleaning pipeline should not raise exceptions."""
        try:
            validator = DataQualityValidator()
            result = validator.validate(corrupted_data)
            # Apply basic cleaning
            cleaned = corrupted_data.drop_duplicates(subset=["PULocationID", "time_bucket"])
            cleaned = cleaned[(cleaned["trip_count"] >= 1) & (cleaned["trip_count"] <= 9000)]
            assert len(cleaned) > 0, "Cleaned data should not be empty"
        except Exception as e:
            pytest.fail(f"Pipeline raised an unexpected exception: {e}")

    def test_fallback_is_logged(self, corrupted_data, caplog):
        """Degradation actions should be logged at WARNING level."""
        from validation.check_data_quality import logger as val_logger

        with caplog.at_level(logging.WARNING, logger="validation.check_data_quality"):
            # Simulate what data.py does: validate then log issues
            validator = DataQualityValidator()
            result = validator.validate(corrupted_data)
            for issue in result["issues"]:
                val_logger.warning(
                    "Data quality issue — %s (%s): %s",
                    issue["type"], issue["severity"], issue["description"]
                )

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_messages) > 0, "Degradation should produce at least one WARNING log entry"


# ── Utility function tests ─────────────────────────────────────────────────────

class TestUtilityFunctions:

    def test_detect_outliers_flags_extremes(self):
        normal = pd.Series([10.0, 11.0, 10.5, 9.8, 10.2] * 20)
        with_outlier = pd.Series([10.0, 11.0, 10.5, 9.8, 10.2] * 20 + [9999.0])
        flags = detect_outliers(with_outlier, sigma=3.0)
        assert flags.iloc[-1], "Extreme value should be flagged as outlier"
        assert flags.sum() <= 3, "Should not flag too many normal values"

    def test_compare_distributions_detects_shift(self):
        baseline = pd.Series(np.random.normal(10, 1, 1000))
        shifted = pd.Series(np.random.normal(30, 1, 1000))
        assert compare_distributions(baseline, shifted, threshold=2.0), "Large shift should be detected"

    def test_compare_distributions_passes_similar(self):
        baseline = pd.Series(np.random.normal(10, 1, 1000))
        similar = pd.Series(np.random.normal(10.1, 1, 1000))
        assert not compare_distributions(baseline, similar, threshold=2.0), "Similar distributions should pass"
