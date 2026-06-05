"""
Compute monitoring metrics for Week 4 drift detection.

Loads baseline (Jan 1-15) and week4 (Feb 2-28) data, runs all 8 metrics,
prints a human-readable report, and saves metrics-results.json.
Exits with code 1 if any critical alerts fire.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metric_template import MetricComputer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = _DATA_DIR / "demand_enriched_baseline.parquet"
    week4_path = _DATA_DIR / "demand_enriched_week4.parquet"
    for p in (baseline_path, week4_path):
        if not p.exists():
            logger.error("Data file not found: %s", p)
            sys.exit(1)
    logger.info("Loading baseline data (%s)...", baseline_path.name)
    baseline = pd.read_parquet(baseline_path)
    logger.info("Loading week4 data (%s)...", week4_path.name)
    week4 = pd.read_parquet(week4_path)
    logger.info("Baseline rows: %d | Week4 rows: %d", len(baseline), len(week4))
    return baseline, week4


def _print_report(results: dict, n_alerts: int):
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("MONITORING METRICS REPORT — Week 4 vs Baseline")
    print(SEP)

    # Accuracy
    acc = results.get("accuracy", None)
    if acc is not None:
        flag = " [ALERT]" if acc < 0.60 else ""
        print(f"\n[1] Overall Accuracy:       {acc:.3f}{flag}  (threshold >= 0.60)")

    # Accuracy by zone — worst 5
    abz = results.get("accuracy_by_zone", {})
    if abz:
        worst = sorted(abz.items(), key=lambda x: x[1])[:5]
        print(f"\n[2] Accuracy by Zone — worst 5 zones:")
        for zone, val in worst:
            flag = " [ALERT]" if val < 0.55 else ""
            print(f"    Zone {zone:4d}: {val:.3f}{flag}")

    # Null rates
    nr = results.get("null_rates", {})
    if nr:
        print("\n[3] Null Rates:")
        for col, info in nr.items():
            threshold = 0.01 if col in ("trip_count", "PULocationID") else 0.02
            flag = " [ALERT]" if info["rate"] > threshold else ""
            print(f"    {col:25s}: {info['rate']:.4%}{flag}")

    # KS test
    ks = results.get("ks_test", {})
    if ks:
        print("\n[4] KS Test (p < 0.05 = drift):")
        for feat, res in ks.items():
            flag = " [DRIFT]" if res["drift_detected"] else ""
            print(f"    {feat:25s}: stat={res['statistic']:.4f}  p={res['p_value']:.2e}{flag}")

    # PSI
    psi = results.get("psi", {})
    if psi:
        print("\n[5] Population Stability Index:")
        for feat, res in psi.items():
            flag = " [ALERT]" if res["status"] == "retrain" else (" [MONITOR]" if res["status"] == "monitor" else "")
            print(f"    {feat:25s}: PSI={res['psi']:.4f}  [{res['status']}]{flag}")

    # Prediction distribution
    pd_res = results.get("prediction_distribution", {})
    if pd_res:
        flag = " [ALERT]" if pd_res.get("alert") else ""
        print(f"\n[6] Prediction Distribution:  mean={pd_res.get('mean',0):.2f}  "
              f"std={pd_res.get('std',0):.2f}  "
              f"mean_shift={pd_res.get('mean_shift_pct',0):.1f}%{flag}")

    # Data freshness
    df_res = results.get("data_freshness", {})
    if df_res and "age_hours" in df_res:
        flag = " [ALERT]" if df_res.get("alert") else ""
        print(f"\n[7] Data Freshness:           age={df_res['age_hours']:.1f}h{flag}  "
              f"(latest: {df_res.get('latest_record','N/A')})")

    # Duplicate rate
    dup = results.get("duplicate_rate", {})
    if dup and "duplicate_rate" in dup:
        flag = " [ALERT]" if dup.get("alert") else ""
        print(f"\n[8] Duplicate Rate:           {dup['duplicate_rate']:.4%}  "
              f"({dup['duplicate_count']} rows){flag}")

    print(f"\n{SEP}")
    if n_alerts > 0:
        print(f"RESULT: {n_alerts} alert(s) fired — investigate drift")
    else:
        print("RESULT: No alerts — data within expected ranges")
    print(SEP)


def _count_alerts(results: dict) -> int:
    alerts = 0
    if results.get("accuracy", 1.0) < 0.60:
        alerts += 1
    abz = results.get("accuracy_by_zone", {})
    if any(v < 0.55 for v in abz.values()):
        alerts += 1
    nr = results.get("null_rates", {})
    for col, info in nr.items():
        threshold = 0.01 if col in ("trip_count", "PULocationID") else 0.02
        if info["rate"] > threshold:
            alerts += 1
    ks = results.get("ks_test", {})
    if any(r["drift_detected"] for r in ks.values()):
        alerts += 1
    psi = results.get("psi", {})
    if any(r["status"] == "retrain" for r in psi.values()):
        alerts += 1
    if results.get("prediction_distribution", {}).get("alert"):
        alerts += 1
    if results.get("data_freshness", {}).get("alert"):
        alerts += 1
    if results.get("duplicate_rate", {}).get("alert"):
        alerts += 1
    return alerts


def main():
    baseline, week4 = _load_data()
    computer = MetricComputer(baseline)

    logger.info("Computing all metrics...")
    results = computer.compute_all_metrics(week4)

    n_alerts = _count_alerts(results)
    _print_report(results, n_alerts)

    # Save JSON (drop accuracy_by_zone detail to keep file small)
    output = {k: v for k, v in results.items() if k != "accuracy_by_zone"}
    zone_alerts = {str(z): v for z, v in results.get("accuracy_by_zone", {}).items() if v < 0.55}
    output["zone_accuracy_alerts"] = zone_alerts
    output["total_alerts"] = n_alerts

    out_path = Path(__file__).parent.parent / "metrics-results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    if n_alerts > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
