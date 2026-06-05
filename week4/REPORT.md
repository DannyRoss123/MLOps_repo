# Week 4 Report: Monitoring, Drift Detection & Retraining Strategy

**Name:** Danny Ross

---

## Drift Patterns Detected (Feb 2-28 vs Jan 1-15 Baseline)

| # | Pattern | Type | Evidence | Model Impact |
|---|---------|------|----------|--------------|
| 1 | **Temporal peak shift** — early morning (5-7am) demand +45%, late morning (9-11am) -35% | Data drift | KS test p < 0.05 on hour distribution; mean shift >20% in affected slots | Model under-predicts early AM, over-predicts late AM — peak-hour allocation wrong |
| 2 | **Manhattan lag deflation** — `lag_1day`, `lag_1week`, `roll_mean_1day` deflated 45% for all Manhattan zones | Feature drift | KS p < 0.05 on lag features for borough_id=0; mean shift ~-45% | Lag features underestimate actual demand; model systematically under-predicts Manhattan |
| 3 | **Outer-borough baseline scramble** — `zone_slot_baseline` correlation with `trip_count` broken in Queens/Brooklyn (5 zones) | Data drift | Pearson correlation drop >0.15; PSI > 0.25 on `zone_slot_baseline` | Most predictive feature becomes noise for affected zones; accuracy drops silently |
| 4 | **Manhattan weekend concept drift** — weekend demand -28% (multiplier 0.72), weekdays unchanged | Concept drift | KS p < 0.05 on Manhattan weekend rows; weekend/weekday shift diverges >10 pp | Model trained on pre-drift weekend patterns over-predicts weekend demand in Manhattan |

---

## Monitoring Metrics (8 defined)

| # | Metric | Computation | Threshold | Frequency | Segment |
|---|--------|-------------|-----------|-----------|---------|
| 1 | Overall accuracy | Fraction of predictions within 50% of actuals | Alert < 0.60 | Daily | Global |
| 2 | Accuracy by zone | Per-zone within-50% fraction | Alert < 0.55 any zone | Daily | Per zone |
| 3 | Null rates | Fraction of nulls per column | trip_count/ID > 1%; lags > 2% | Daily | Per column |
| 4 | KS test | KS statistic + p-value vs baseline | Alert p < 0.05 | Daily | trip_count, lag features |
| 5 | PSI | Population Stability Index vs baseline | Monitor > 0.10; Retrain > 0.25 | Daily | trip_count, lag features |
| 6 | Prediction distribution | Mean + std of model output | Alert if std < 0.5 (collapse) or mean shift > 50% | Daily | Global |
| 7 | Data freshness | Age of most recent record | Alert if > 26 hours | Daily | Global |
| 8 | Duplicate rate | Fraction of duplicate (zone, time_bucket) rows | Alert > 0.5% | Daily | Global |

---

## Monitoring Schedule: Daily at 8am UTC

**Choice:** `cron: '0 8 * * *'` — once per day, 8am UTC (4am ET).

**Justification:** Data arrives in overnight batch ETL. Running at 8am UTC catches issues before the US business day (9am ET). A 24-hour detection lag is acceptable because drift affects long-term demand profiles — a single drifted batch shifts predictions by a few percent, not catastrophically. The workflow also triggers on any push to `week4/data/**` for immediate feedback during development.

---

## Retraining Strategy

**Trigger conditions (any one fires retraining):**
- PSI > 0.25 on `trip_count` or any lag feature
- KS p-value < 0.01 on `trip_count` (highly significant distribution shift)
- Overall accuracy < 0.60 for 3 consecutive days
- >5 zones with accuracy < 0.55

**Retraining pipeline:**
1. **Detect** — daily monitoring fires alert, creates GitHub issue
2. **Train** — retrain on most recent 30 days of clean data (rolling window)
3. **Validate offline** — compare new model MAE vs current on held-out last 7 days; deploy only if MAE improves
4. **Shadow deploy** — run new model in parallel for 24 hours; compare predictions without serving them
5. **Canary** — route 10% of traffic to new model, compare accuracy
6. **Full rollout or rollback** — promote if canary matches offline validation; else auto-rollback to previous version

**Model versioning:** Store in GCS at `gs://<bucket>/models/<date>/model.lgb` with metadata JSON (training date, data range, MAE, feature list). Keep last 3 versions for rollback.
