"""
Data Quality Validation for NYC Taxi Demand Data.

Detects 4 known corruption types:
  1. Duplicate rows (same zone + time_bucket)
  2. Out-of-range trip_count values (negative or extreme outliers)
  3. is_holiday flag drift (flag set for non-holiday calendar dates)
  4. lag_1week contamination (lag values inconsistent with trip_count scale)
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known US holidays as (month, day) tuples
_HOLIDAYS = {
    (1, 1), (1, 20), (2, 17), (3, 17), (5, 26),
    (7, 4), (9, 1), (10, 13), (10, 31), (11, 11),
    (11, 27), (12, 24), (12, 25), (12, 31),
}


class DataQualityValidator:
    """Validates a demand DataFrame against quality expectations."""

    REQUIRED_COLUMNS = ["PULocationID", "hour", "dayofweek", "trip_count", "is_holiday", "time_bucket"]

    def __init__(self, baseline_df: pd.DataFrame = None):
        self.baseline = baseline_df
        self.issues: List[Dict] = []

    def validate(self, df: pd.DataFrame) -> Dict:
        """Run all checks. Returns dict with is_valid, num_issues, issues."""
        self.issues = []
        self.check_schema(df)
        self.check_null_rates(df)
        self.check_duplicates(df)
        self.check_value_ranges(df)
        self.check_holiday_drift(df)
        if "lag_1week" in df.columns:
            self.check_lag_contamination(df)
        return {
            "is_valid": len(self.issues) == 0,
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    # ── Individual checks ──────────────────────────────────────────────────────

    def check_schema(self, df: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self._add_issue("schema", "critical", f"Missing required columns: {missing}", count=len(missing))

    def check_null_rates(self, df: pd.DataFrame):
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                rate = null_count / len(df)
                severity = "critical" if rate > 0.05 else "medium"
                self._add_issue(
                    "null_rate", severity,
                    f"Column '{col}' has {null_count:,} nulls ({rate:.1%})",
                    count=null_count, column=col,
                )

    def check_duplicates(self, df: pd.DataFrame):
        key_cols = [c for c in ["PULocationID", "time_bucket"] if c in df.columns]
        if len(key_cols) < 2:
            return
        dup_count = int(df.duplicated(subset=key_cols).sum())
        if dup_count > 0:
            self._add_issue(
                "duplicates", "high",
                f"{dup_count:,} duplicate rows (same PULocationID + time_bucket)",
                count=dup_count,
            )

    def check_value_ranges(self, df: pd.DataFrame):
        if "trip_count" not in df.columns:
            return
        # Negative trip counts are physically impossible
        low = int((df["trip_count"] < 0).sum())
        if low > 0:
            self._add_issue(
                "out_of_range", "critical",
                f"{low:,} rows with trip_count < 0 (negative trips impossible)",
                count=low,
            )
        # Values above 9000 per 15-min window are unrealistic for any NYC zone
        high = int((df["trip_count"] > 9000).sum())
        if high > 0:
            self._add_issue(
                "out_of_range", "critical",
                f"{high:,} rows with trip_count > 9000 (extreme outlier — likely data error)",
                count=high,
            )
        if "hour" in df.columns:
            bad = int((~df["hour"].between(0, 23)).sum())
            if bad > 0:
                self._add_issue("out_of_range", "high", f"{bad:,} rows with hour outside 0-23", count=bad)
        if "dayofweek" in df.columns:
            bad = int((~df["dayofweek"].between(0, 6)).sum())
            if bad > 0:
                self._add_issue("out_of_range", "high", f"{bad:,} rows with dayofweek outside 0-6", count=bad)

    def check_holiday_drift(self, df: pd.DataFrame):
        """
        Compare is_holiday flag against the known holiday calendar.
        A mismatch rate above 0.1% indicates systematic flag drift.
        """
        if "is_holiday" not in df.columns or "time_bucket" not in df.columns:
            return
        try:
            dates = pd.to_datetime(df["time_bucket"])
            expected = dates.apply(lambda d: 1 if (d.month, d.day) in _HOLIDAYS else 0)
            mismatch = int((df["is_holiday"] != expected).sum())
            mismatch_rate = mismatch / len(df)
            if mismatch_rate > 0.001:
                self._add_issue(
                    "holiday_drift", "medium",
                    f"{mismatch:,} rows ({mismatch_rate:.2%}) have is_holiday flag inconsistent with calendar",
                    count=mismatch,
                    mismatch_rate=float(mismatch_rate),
                )
        except Exception as e:
            logger.warning("Could not check holiday drift: %s", e)

    def check_lag_contamination(self, df: pd.DataFrame):
        """
        Detect zones where lag_1week is inconsistent with trip_count scale.
        For a clean zone, lag_1week mean should be within 5x of trip_count mean.
        A ratio outside this range indicates the lag was computed from the wrong zone.
        """
        if "PULocationID" not in df.columns or "trip_count" not in df.columns:
            return

        contaminated = []
        for zone_id, grp in df.groupby("PULocationID"):
            if len(grp) < 10:
                continue
            lag_vals = grp["lag_1week"].dropna()
            if len(lag_vals) < 5:
                continue
            tc_mean = grp["trip_count"].mean()
            lag_mean = lag_vals.mean()
            if tc_mean <= 0:
                continue
            ratio = lag_mean / tc_mean
            if ratio > 5.0 or ratio < 0.2:
                contaminated.append(int(zone_id))

        if contaminated:
            self._add_issue(
                "lag_contamination", "high",
                f"lag_1week scale inconsistent with trip_count in {len(contaminated)} zone(s): {contaminated[:10]}",
                count=len(contaminated),
                zones=contaminated,
            )

    # ── Helper ─────────────────────────────────────────────────────────────────

    def _add_issue(self, issue_type: str, severity: str, description: str, count: int = None, **details):
        self.issues.append({
            "type": issue_type,
            "severity": severity,
            "description": description,
            "count": count,
            **details,
        })


# ── Utility functions ──────────────────────────────────────────────────────────

def compare_distributions(baseline: pd.Series, current: pd.Series, threshold: float = 2.0) -> bool:
    """Return True if distributions differ by more than `threshold` standard deviations."""
    if baseline.std() == 0:
        return False
    return abs(current.mean() - baseline.mean()) / baseline.std() > threshold


def detect_outliers(series: pd.Series, baseline_series: pd.Series = None, sigma: float = 3.0) -> pd.Series:
    """Return boolean Series — True where values are outliers."""
    ref = baseline_series if baseline_series is not None else series
    mean, std = ref.mean(), ref.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    return (series - mean).abs() > sigma * std


# ── CLI entry point ────────────────────────────────────────────────────────────

def _run_validation():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _HERE = Path(__file__).parent.parent  # week3/
    baseline_path = _HERE / "data" / "demand_enriched_baseline.parquet"
    corrupted_path = _HERE / "data" / "demand_enriched_corrupted.parquet"

    if not baseline_path.exists():
        logger.error("Baseline data not found: %s", baseline_path)
        sys.exit(1)
    if not corrupted_path.exists():
        logger.error("Current data not found: %s", corrupted_path)
        sys.exit(1)

    logger.info("Loading baseline data...")
    baseline_df = pd.read_parquet(baseline_path)
    logger.info("Loading current data...")
    current_df = pd.read_parquet(corrupted_path)

    logger.info("Running validation...")
    validator = DataQualityValidator(baseline_df)
    result = validator.validate(current_df)

    results_path = _HERE / "validation-results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    print(f"\n{'='*55}")
    print(f"Validation {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Issues found: {result['num_issues']}")
    for issue in result["issues"]:
        print(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['description']}")
    print("=" * 55)

    if not result["is_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    _run_validation()
