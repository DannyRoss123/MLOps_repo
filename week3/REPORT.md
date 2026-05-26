# Week 3 Report: Data Quality Validation

**Name:** Danny Ross

---

## Issues Found in Corrupted Dataset

| # | Issue | Rows Affected | Severity | Model Impact |
|---|---|---|---|---|
| 1 | **Duplicate rows** — zones 4, 43, 87, 107, 152, 229 contain repeated (zone, time_bucket) pairs | 10,085 | High | Demand counts artificially inflated; aggregated profiles overestimate true demand, causing the model to over-predict for these zones |
| 2 | **Out-of-range trip_count** — values of -5, -1, 0, 9999, 99999 present | 850 | Critical | Negative trips are physically impossible; extreme values (99,999) corrupt rolling mean features and shift the model's learned demand scale |
| 3 | **is_holiday flag drift** — flag incorrectly set to 1 for ~82,000 rows outside any actual holiday window | 82,080 (1.3%) | Medium | Model routes demand through the holiday profile for non-holiday dates, producing systematically wrong predictions during the affected window |
| 4 | **lag_1week contamination** — lag feature for zones 161, 162, 186 replaced with demand data from zone 237 | All rows for 3 zones | High | The most predictive lag feature becomes uncorrelated with the actual target for these zones, silently degrading forecast accuracy without any visible error |

---

## Validation Schedule: Daily at 6am UTC

**Choice:** `cron: '0 6 * * *'` — once per day at 6am UTC (2am Eastern).

**Justification:**
The demand data is a batch ETL pipeline, not a real-time stream. New data arrives overnight as the prior day's trip records are aggregated and uploaded to GCS. Running validation at 6am UTC catches any ETL corruption before the US business day begins (8am ET), when driver recommendations are most heavily used.

**Trade-offs considered:**

| Frequency | Cost | Detection Lag | Verdict |
|---|---|---|---|
| Every 15 minutes | High (96 runs/day) | ~15 min | Overkill for batch data |
| Hourly | Moderate (24 runs/day) | ~1 hour | Reasonable but wasteful |
| **Daily at 6am** | Low (1 run/day) | ~24 hours | Right fit for overnight ETL |
| Weekly | Very low | Up to 7 days | Too slow — bad data compounds |

A 24-hour detection window is acceptable because corrupted data affects demand *profiles* (historical averages), not real-time predictions. A single bad batch shifts averages slightly but the model remains usable. The workflow also triggers on any push that touches `week3/data/` or `week3/validation/`, providing immediate feedback during development.

---

## Graceful Degradation Strategy

The API must never crash due to data quality issues. The `_validate_and_clean()` function in `backend/data.py` applies four ordered fixes before any data reaches the model:

1. **Duplicates → drop** — `drop_duplicates(subset=["PULocationID", "time_bucket"])`. Deduplication is safe and lossless; the duplicate rows carry no additional information.

2. **Out-of-range trip_count → filter** — Rows where `trip_count < 1` or `trip_count > 9,000` are removed. Values outside this range are physically impossible for a 15-minute NYC taxi window. Remaining rows form a valid, if smaller, dataset.

3. **is_holiday drift → recalculate from date** — The correct holiday flag is recomputed from the `time_bucket` timestamp using the same `HOLIDAYS` lookup table used by the model. If more than 5% of rows have a mismatched flag, all flags are corrected in-place. This is deterministic and requires no external data.

4. **lag_1week contamination → replace with zone median** — For any zone where `lag_1week` mean is more than 5× or less than 0.2× the zone's own `trip_count` mean, the contaminated lag values are replaced with the zone's own median `trip_count`. This is a conservative fallback — the feature becomes less informative but remains correlated with the true target.

Every action is logged at `WARNING` level with row counts, so operators can see exactly what was degraded without reading source code. The API continues serving requests throughout; users receive slightly less accurate predictions rather than errors.
