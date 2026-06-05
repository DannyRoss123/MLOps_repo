"""
Drift detection for Week 4 NYC taxi demand data.

Detects 4 distinct drift patterns between Jan 1-15 baseline and Feb 2-28 data:
  1. Temporal peak shift         — early-morning demand boosted, late-morning reduced
  2. Manhattan lag deflation     — lag_1day/lag_1week/roll_mean_1day deflated ~45% for Manhattan
  3. Outer-borough scramble      — zone_slot_baseline correlation broken in Queens/Brooklyn
  4. Manhattan weekend concept drift — weekend demand down ~28% in Manhattan
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

_DATA_DIR = Path(__file__).parent.parent / "data"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_manhattan(df: pd.DataFrame) -> pd.Series:
    """Best-effort Manhattan flag: use borough_id==0 if present, else PULocationID 1-69."""
    if "borough_id" in df.columns:
        return df["borough_id"] == 0
    return df["PULocationID"].between(1, 69)


def _is_outer_borough(df: pd.DataFrame) -> pd.Series:
    """Queens (PULocationID 129-200) or Brooklyn (70-128) as fallback."""
    if "borough_id" in df.columns:
        return df["borough_id"].isin([1, 2])
    return df["PULocationID"].between(70, 200)


def _is_weekend(df: pd.DataFrame) -> pd.Series:
    if "is_weekend" in df.columns:
        return df["is_weekend"].astype(bool)
    return df["dayofweek"].isin([5, 6])


# ── Drift detectors ─────────────────────────────────────────────────────────

def detect_feature_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict:
    """KS test on a single feature. Returns stat, p-value, and interpretation."""
    if feature not in baseline_df.columns or feature not in new_df.columns:
        return {"feature": feature, "error": "column not found"}
    base = baseline_df[feature].dropna()
    curr = new_df[feature].dropna()
    stat, pval = ks_2samp(base, curr)
    mean_shift = float(curr.mean() - base.mean())
    mean_shift_pct = mean_shift / max(abs(float(base.mean())), 1e-6) * 100
    return {
        "feature": feature,
        "ks_statistic": float(stat),
        "p_value": float(pval),
        "drift_detected": bool(pval < 0.05),
        "baseline_mean": float(base.mean()),
        "current_mean": float(curr.mean()),
        "mean_shift_pct": mean_shift_pct,
    }


def detect_concept_drift_by_segment(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Compare mean trip_count per zone and per hour between baseline and new data.
    Segments with >20% mean shift are flagged as concept drift.
    """
    findings = {}

    # Per-zone mean shift
    if "PULocationID" in baseline_df.columns and "trip_count" in baseline_df.columns:
        base_zone = baseline_df.groupby("PULocationID")["trip_count"].mean()
        curr_zone = new_df.groupby("PULocationID")["trip_count"].mean()
        common = base_zone.index.intersection(curr_zone.index)
        shifts = ((curr_zone[common] - base_zone[common]) / base_zone[common].clip(lower=1e-6) * 100)
        drifted_zones = shifts[shifts.abs() > 20].sort_values(key=abs, ascending=False)
        findings["zone_drift"] = {
            "n_drifted_zones": int(len(drifted_zones)),
            "worst_zones": {int(z): float(v) for z, v in drifted_zones.head(10).items()},
        }

    # Per-hour mean shift
    if "hour" in baseline_df.columns:
        base_hour = baseline_df.groupby("hour")["trip_count"].mean()
        curr_hour = new_df.groupby("hour")["trip_count"].mean()
        common = base_hour.index.intersection(curr_hour.index)
        hour_shifts = ((curr_hour[common] - base_hour[common]) / base_hour[common].clip(lower=1e-6) * 100)
        findings["hour_drift"] = {
            "most_shifted_hours": {int(h): float(v) for h, v in
                                   hour_shifts.reindex(hour_shifts.abs().sort_values(ascending=False).index).head(5).items()}
        }

    return findings


def detect_temporal_peak_shift(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Pattern 1: Early morning (5-7am, slots 20-27) boosted +45%;
               late morning (9-11am, slots 36-43) reduced -35%.
    """
    if "hour" not in baseline_df.columns or "trip_count" not in baseline_df.columns:
        return {"error": "hour or trip_count column missing"}

    early_hours = [5, 6, 7]
    late_hours = [9, 10, 11]

    def mean_by_hours(df, hours):
        return float(df[df["hour"].isin(hours)]["trip_count"].mean())

    base_early = mean_by_hours(baseline_df, early_hours)
    curr_early = mean_by_hours(new_df, early_hours)
    base_late = mean_by_hours(baseline_df, late_hours)
    curr_late = mean_by_hours(new_df, late_hours)

    early_shift_pct = (curr_early - base_early) / max(base_early, 1e-6) * 100
    late_shift_pct = (curr_late - base_late) / max(base_late, 1e-6) * 100

    # KS test on hour distribution of rows
    base_hours_dist = baseline_df["hour"].values
    curr_hours_dist = new_df["hour"].values
    ks_stat, ks_pval = ks_2samp(base_hours_dist, curr_hours_dist)

    detected = abs(early_shift_pct) > 20 or abs(late_shift_pct) > 20 or ks_pval < 0.05

    return {
        "pattern": "temporal_peak_shift",
        "detected": detected,
        "early_morning_shift_pct": round(early_shift_pct, 1),
        "late_morning_shift_pct": round(late_shift_pct, 1),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_pval),
        "interpretation": (
            f"Early morning demand shifted {early_shift_pct:+.1f}%; "
            f"late morning shifted {late_shift_pct:+.1f}%. "
            f"KS test p={ks_pval:.2e} — hour distribution {'drifted' if ks_pval < 0.05 else 'stable'}."
        ),
    }


def detect_manhattan_lag_deflation(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Pattern 2: Manhattan lag_1day/lag_1week/roll_mean_1day deflated ~45%.
    """
    lag_features = ["lag_1day", "lag_1week", "roll_mean_1day"]
    available = [f for f in lag_features if f in baseline_df.columns and f in new_df.columns]
    if not available:
        return {"pattern": "manhattan_lag_deflation", "error": "lag feature columns not found"}

    base_man = baseline_df[_is_manhattan(baseline_df)]
    curr_man = new_df[_is_manhattan(new_df)]

    shifts = {}
    ks_results = {}
    for feat in available:
        bm = base_man[feat].dropna()
        cm = curr_man[feat].dropna()
        if len(bm) < 5 or len(cm) < 5:
            continue
        shift_pct = (float(cm.mean()) - float(bm.mean())) / max(float(bm.mean()), 1e-6) * 100
        ks_stat, ks_pval = ks_2samp(bm, cm)
        shifts[feat] = round(shift_pct, 1)
        ks_results[feat] = {"ks_stat": float(ks_stat), "p_value": float(ks_pval)}

    detected = any(abs(v) > 20 for v in shifts.values())
    return {
        "pattern": "manhattan_lag_deflation",
        "detected": detected,
        "manhattan_rows_baseline": int(len(base_man)),
        "manhattan_rows_current": int(len(curr_man)),
        "feature_shifts_pct": shifts,
        "ks_tests": ks_results,
        "interpretation": (
            f"Manhattan lag features shifted: {shifts}. "
            f"{'Deflation detected (>20% drop) — lag features underestimate actual demand.' if detected else 'No significant deflation.'}"
        ),
    }


def detect_outer_borough_scramble(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Pattern 3: Queens/Brooklyn zone_slot_baseline correlation with trip_count broken.
    Detect via Pearson correlation drop between zone_slot_baseline and trip_count.
    """
    if "zone_slot_baseline" not in baseline_df.columns or "trip_count" not in baseline_df.columns:
        return {"pattern": "outer_borough_scramble", "error": "zone_slot_baseline column not found"}

    outer_base = baseline_df[_is_outer_borough(baseline_df)]
    outer_curr = new_df[_is_outer_borough(new_df)]

    def safe_corr(df):
        sub = df[["zone_slot_baseline", "trip_count"]].dropna()
        if len(sub) < 10:
            return float("nan")
        return float(sub["zone_slot_baseline"].corr(sub["trip_count"]))

    base_corr = safe_corr(outer_base)
    curr_corr = safe_corr(outer_curr)
    corr_drop = base_corr - curr_corr if not (np.isnan(base_corr) or np.isnan(curr_corr)) else float("nan")
    detected = (not np.isnan(corr_drop)) and corr_drop > 0.15

    # Also check PSI on zone_slot_baseline distribution
    from metric_template import _compute_psi
    psi_val = float("nan")
    if len(outer_base) > 0 and len(outer_curr) > 0:
        try:
            psi_val = _compute_psi(outer_base["zone_slot_baseline"], outer_curr["zone_slot_baseline"])
        except Exception:
            pass

    return {
        "pattern": "outer_borough_scramble",
        "detected": bool(detected),
        "baseline_correlation": round(base_corr, 4) if not np.isnan(base_corr) else None,
        "current_correlation": round(curr_corr, 4) if not np.isnan(curr_corr) else None,
        "correlation_drop": round(corr_drop, 4) if not np.isnan(corr_drop) else None,
        "zone_slot_baseline_psi": round(psi_val, 4) if not np.isnan(psi_val) else None,
        "interpretation": (
            f"Queens/Brooklyn zone_slot_baseline vs trip_count correlation: "
            f"{base_corr:.3f} → {curr_corr:.3f} (drop={corr_drop:.3f}). "
            f"{'Correlation broken — baseline feature no longer predictive.' if detected else 'Correlation intact.'}"
        ),
    }


def detect_manhattan_weekend_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Pattern 4: Manhattan weekend demand down ~28% (multiplier 0.72).
    """
    if "trip_count" not in baseline_df.columns:
        return {"pattern": "manhattan_weekend_drift", "error": "trip_count column not found"}

    base_mw = baseline_df[_is_manhattan(baseline_df) & _is_weekend(baseline_df)]
    curr_mw = new_df[_is_manhattan(new_df) & _is_weekend(new_df)]
    base_mwkd = baseline_df[_is_manhattan(baseline_df) & ~_is_weekend(baseline_df)]
    curr_mwkd = new_df[_is_manhattan(new_df) & ~_is_weekend(new_df)]

    if len(base_mw) < 5 or len(curr_mw) < 5:
        return {"pattern": "manhattan_weekend_drift", "error": "insufficient weekend rows"}

    wkend_shift = (float(curr_mw["trip_count"].mean()) - float(base_mw["trip_count"].mean())) / max(float(base_mw["trip_count"].mean()), 1e-6) * 100
    wkday_shift = (float(curr_mwkd["trip_count"].mean()) - float(base_mwkd["trip_count"].mean())) / max(float(base_mwkd["trip_count"].mean()), 1e-6) * 100

    ks_stat, ks_pval = ks_2samp(base_mw["trip_count"].dropna(), curr_mw["trip_count"].dropna())
    detected = abs(wkend_shift) > 15 and (abs(wkend_shift) - abs(wkday_shift)) > 10

    return {
        "pattern": "manhattan_weekend_concept_drift",
        "detected": bool(detected),
        "weekend_demand_shift_pct": round(wkend_shift, 1),
        "weekday_demand_shift_pct": round(wkday_shift, 1),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_pval),
        "interpretation": (
            f"Manhattan weekend demand shifted {wkend_shift:+.1f}% vs weekday {wkday_shift:+.1f}%. "
            f"{'Concept drift: weekend demand pattern changed disproportionately.' if detected else 'No disproportionate weekend drift.'}"
        ),
    }


def main():
    SEP = "=" * 70

    # Load data
    baseline_path = _DATA_DIR / "demand_enriched_baseline.parquet"
    week4_path = _DATA_DIR / "demand_enriched_week4.parquet"
    for p in (baseline_path, week4_path):
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    print(f"\n{SEP}")
    print("DRIFT DETECTION — Baseline (Jan 1-15) vs Week4 (Feb 2-28)")
    print(SEP)

    baseline = pd.read_parquet(baseline_path)
    week4 = pd.read_parquet(week4_path)
    print(f"Baseline rows: {len(baseline):,}  |  Week4 rows: {len(week4):,}")
    print(f"Columns: {list(baseline.columns)}\n")

    # ── Pattern 1: Temporal peak shift ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Pattern 1: Temporal Peak Shift")
    print("─" * 70)
    r1 = detect_temporal_peak_shift(baseline, week4)
    print(f"  Detected:         {r1.get('detected', 'N/A')}")
    print(f"  Early AM shift:   {r1.get('early_morning_shift_pct', 'N/A')}%")
    print(f"  Late AM shift:    {r1.get('late_morning_shift_pct', 'N/A')}%")
    print(f"  KS p-value:       {r1.get('ks_p_value', 'N/A'):.2e}" if 'ks_p_value' in r1 else "")
    print(f"  Interpretation:   {r1.get('interpretation', '')}")

    # ── Pattern 2: Manhattan lag deflation ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Pattern 2: Manhattan Lag Feature Deflation")
    print("─" * 70)
    r2 = detect_manhattan_lag_deflation(baseline, week4)
    print(f"  Detected:         {r2.get('detected', 'N/A')}")
    for feat, shift in r2.get("feature_shifts_pct", {}).items():
        print(f"  {feat:25s}: {shift:+.1f}%")
    print(f"  Interpretation:   {r2.get('interpretation', '')}")

    # ── Pattern 3: Outer borough scramble ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Pattern 3: Outer Borough Baseline Scramble")
    print("─" * 70)
    r3 = detect_outer_borough_scramble(baseline, week4)
    print(f"  Detected:         {r3.get('detected', 'N/A')}")
    print(f"  Baseline corr:    {r3.get('baseline_correlation', 'N/A')}")
    print(f"  Current corr:     {r3.get('current_correlation', 'N/A')}")
    print(f"  Correlation drop: {r3.get('correlation_drop', 'N/A')}")
    print(f"  PSI (zone_slot_baseline): {r3.get('zone_slot_baseline_psi', 'N/A')}")
    print(f"  Interpretation:   {r3.get('interpretation', '')}")

    # ── Pattern 4: Manhattan weekend concept drift ─────────────────────────────
    print(f"\n{'─'*70}")
    print("Pattern 4: Manhattan Weekend Concept Drift")
    print("─" * 70)
    r4 = detect_manhattan_weekend_drift(baseline, week4)
    print(f"  Detected:         {r4.get('detected', 'N/A')}")
    print(f"  Weekend shift:    {r4.get('weekend_demand_shift_pct', 'N/A')}%")
    print(f"  Weekday shift:    {r4.get('weekday_demand_shift_pct', 'N/A')}%")
    print(f"  KS p-value:       {r4.get('ks_p_value', 'N/A'):.2e}" if 'ks_p_value' in r4 else "")
    print(f"  Interpretation:   {r4.get('interpretation', '')}")

    # ── Feature-level KS summary ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Feature-level KS Drift Summary")
    print("─" * 70)
    for feat in ["trip_count", "lag_1day", "lag_1week", "roll_mean_1day", "zone_slot_baseline"]:
        res = detect_feature_drift(baseline, week4, feat)
        if "error" in res:
            continue
        flag = " *** DRIFT" if res["drift_detected"] else ""
        print(f"  {feat:25s}: stat={res['ks_statistic']:.4f}  p={res['p_value']:.2e}  "
              f"mean_shift={res['mean_shift_pct']:+.1f}%{flag}")

    # ── Concept drift by segment ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Concept Drift by Segment")
    print("─" * 70)
    concept = detect_concept_drift_by_segment(baseline, week4)
    n_zones = concept.get("zone_drift", {}).get("n_drifted_zones", 0)
    print(f"  Zones with >20% mean shift: {n_zones}")
    for zone, shift in list(concept.get("zone_drift", {}).get("worst_zones", {}).items())[:5]:
        print(f"    Zone {zone:4d}: {shift:+.1f}%")
    print(f"  Hours most shifted:")
    for hour, shift in concept.get("hour_drift", {}).get("most_shifted_hours", {}).items():
        print(f"    Hour {hour:2d}: {shift:+.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    patterns_detected = sum(1 for r in [r1, r2, r3, r4] if r.get("detected"))
    print(f"\n{SEP}")
    print(f"DRIFT SUMMARY: {patterns_detected}/4 patterns detected")
    for i, (label, r) in enumerate([
        ("Temporal peak shift", r1),
        ("Manhattan lag deflation", r2),
        ("Outer borough scramble", r3),
        ("Manhattan weekend drift", r4),
    ], 1):
        status = "DETECTED" if r.get("detected") else "not detected"
        print(f"  {i}. {label:35s}: {status}")
    print(SEP)

    if patterns_detected > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
