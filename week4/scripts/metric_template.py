"""
Monitoring metrics for NYC taxi demand drift detection.

8 metrics covering: performance, data quality, data drift, model health.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def _compute_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    edges = np.percentile(baseline.dropna(), np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    base_counts, _ = np.histogram(baseline.dropna(), bins=edges)
    curr_counts, _ = np.histogram(current.dropna(), bins=edges)
    base_pct = base_counts / max(len(baseline.dropna()), 1)
    curr_pct = curr_counts / max(len(current.dropna()), 1)
    base_pct = np.where(base_pct == 0, 1e-4, base_pct)
    curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


class MetricComputer:
    """Compute monitoring metrics for drift detection."""

    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df

    def metric_1_accuracy(self, new_df: pd.DataFrame, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Overall accuracy: fraction of predictions within 50% of actuals (for non-zero actuals).
        Returns 0-1 float. Baseline ~0.72 on clean data.
        Alert: < 0.60
        """
        mask = actuals > 0
        if mask.sum() == 0:
            return 1.0
        preds = predictions[mask]
        acts = actuals[mask]
        within_tolerance = np.abs(preds - acts) / acts <= 0.50
        return float(within_tolerance.mean())

    def metric_2_accuracy_by_zone(self, new_df: pd.DataFrame, predictions: np.ndarray, actuals: np.ndarray) -> dict:
        """
        Per-zone accuracy (fraction within 50% of actuals for non-zero rows).
        Alert: any zone drops below 0.55.
        """
        if "PULocationID" not in new_df.columns:
            return {}
        results = {}
        for zone_id, idx in new_df.groupby("PULocationID").groups.items():
            acts = actuals[idx]
            preds = predictions[idx]
            mask = acts > 0
            if mask.sum() < 5:
                continue
            within = np.abs(preds[mask] - acts[mask]) / acts[mask] <= 0.50
            results[int(zone_id)] = float(within.mean())
        return results

    def metric_3_null_rates(self, new_df: pd.DataFrame) -> dict:
        """
        Null rates for critical columns.
        Alert: trip_count/PULocationID > 1%; lag features > 2%.
        """
        critical = ["PULocationID", "trip_count", "lag_15min", "lag_1h",
                    "lag_1day", "lag_1week", "roll_mean_1h", "roll_mean_1day", "is_holiday"]
        result = {}
        for col in critical:
            if col in new_df.columns:
                rate = float(new_df[col].isna().mean())
                result[col] = {"rate": rate, "count": int(new_df[col].isna().sum())}
        return result

    def metric_4_ks_test(self, new_df: pd.DataFrame) -> dict:
        """
        KS test comparing trip_count (and lag features) distributions.
        Alert: p-value < 0.05 indicates significant shift.
        """
        features = ["trip_count", "lag_1day", "lag_1week", "roll_mean_1day"]
        results = {}
        for feat in features:
            if feat not in new_df.columns or feat not in self.baseline_df.columns:
                continue
            base = self.baseline_df[feat].dropna()
            curr = new_df[feat].dropna()
            stat, pval = ks_2samp(base, curr)
            results[feat] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "drift_detected": bool(pval < 0.05),
            }
        return results

    def metric_5_psi(self, new_df: pd.DataFrame, bins: int = 10) -> dict:
        """
        Population Stability Index for trip_count and lag features.
        PSI < 0.10: stable; 0.10-0.25: monitor; > 0.25: retrain.
        """
        features = ["trip_count", "lag_1day", "lag_1week", "roll_mean_1day"]
        results = {}
        for feat in features:
            if feat not in new_df.columns or feat not in self.baseline_df.columns:
                continue
            psi = _compute_psi(self.baseline_df[feat], new_df[feat], bins)
            results[feat] = {
                "psi": psi,
                "status": "stable" if psi < 0.10 else ("monitor" if psi < 0.25 else "retrain"),
            }
        return results

    def metric_6_prediction_distribution(self, predictions: np.ndarray) -> dict:
        """
        Prediction distribution health: checks for model collapse (std near zero).
        Alert: std < 0.5 or mean shifts more than 50% from baseline trip_count mean.
        """
        baseline_mean = float(self.baseline_df["trip_count"].mean()) if "trip_count" in self.baseline_df.columns else 17.0
        pred_mean = float(np.mean(predictions))
        pred_std = float(np.std(predictions))
        collapsed = pred_std < 0.5
        mean_shift_pct = abs(pred_mean - baseline_mean) / max(baseline_mean, 1e-6) * 100
        return {
            "mean": pred_mean,
            "std": pred_std,
            "collapsed": collapsed,
            "mean_shift_pct": mean_shift_pct,
            "alert": collapsed or mean_shift_pct > 50,
        }

    def metric_7_data_freshness(self, new_df: pd.DataFrame) -> dict:
        """
        Age of most recent record (requires time_bucket column).
        Alert: most recent record older than 26 hours (missed daily ETL).
        """
        if "time_bucket" not in new_df.columns:
            return {"error": "time_bucket column not found"}
        try:
            latest = pd.to_datetime(new_df["time_bucket"]).max()
            import datetime
            now = pd.Timestamp.now(tz=latest.tz) if latest.tzinfo else pd.Timestamp.now()
            age_hours = float((now - latest).total_seconds() / 3600)
            return {
                "latest_record": str(latest),
                "age_hours": age_hours,
                "alert": age_hours > 26,
            }
        except Exception as e:
            return {"error": str(e)}

    def metric_8_duplicate_rate(self, new_df: pd.DataFrame) -> dict:
        """
        Fraction of rows that are exact duplicates on (PULocationID, time_bucket).
        Alert: duplicate rate > 0.5%.
        """
        key_cols = [c for c in ["PULocationID", "time_bucket"] if c in new_df.columns]
        if len(key_cols) < 2:
            return {"error": "key columns not available"}
        dup_count = int(new_df.duplicated(subset=key_cols).sum())
        rate = dup_count / max(len(new_df), 1)
        return {
            "duplicate_count": dup_count,
            "duplicate_rate": float(rate),
            "alert": rate > 0.005,
        }

    def compute_all_metrics(self, new_df: pd.DataFrame, predictions: np.ndarray = None, actuals: np.ndarray = None) -> dict:
        """Run all 8 metrics and return a combined results dict."""
        if predictions is None and "zone_slot_baseline" in new_df.columns and "trip_count" in new_df.columns:
            predictions = new_df["zone_slot_baseline"].fillna(0).values
            actuals = new_df["trip_count"].fillna(0).values
        if predictions is None:
            predictions = np.zeros(len(new_df))
            actuals = new_df["trip_count"].fillna(0).values if "trip_count" in new_df.columns else np.zeros(len(new_df))

        results = {
            "accuracy": self.metric_1_accuracy(new_df, predictions, actuals),
            "accuracy_by_zone": self.metric_2_accuracy_by_zone(new_df, predictions, actuals),
            "null_rates": self.metric_3_null_rates(new_df),
            "ks_test": self.metric_4_ks_test(new_df),
            "psi": self.metric_5_psi(new_df),
            "prediction_distribution": self.metric_6_prediction_distribution(predictions),
            "data_freshness": self.metric_7_data_freshness(new_df),
            "duplicate_rate": self.metric_8_duplicate_rate(new_df),
        }
        return results
