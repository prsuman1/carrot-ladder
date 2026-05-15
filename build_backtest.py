"""Backtest data builder.

Takes the 24 months of monthly bill files pulled by build_user_data.py, adds
a partial May 1-14 2026 pull if missing, then splits at 2026-04-01:
  - training: bill_date < 2026-04-01  → per-user predictions (1 row/patient)
  - test:     bill_date in [2026-04-01, 2026-05-15)  → actual bills to score

A "new patient" = a test-window bill whose patient_id is NOT in the training
user set. Those bills are dropped (separate new-patient program).

Outputs (overwrites previous versions):
  data/user_summary.parquet   — 1 row per training patient
  data/test_bills.parquet     — test-window bills, returning patients only,
                                with a 'window' column ('april' | 'may')
  data/user_topline.json      — counts + windows + generated_at

Run:
    python build_backtest.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_user_data import (
    DATA_DIR,
    MONTHLY_DIR,
    PATIENT_COL,
    connect,
    month_sql,
    write_atomic_parquet,
)

CUTOFF = date(2026, 4, 1)         # training < CUTOFF, test >= CUTOFF
TEST_END = date(2026, 5, 15)      # exclusive end of test window
APRIL_END = date(2026, 5, 1)      # exclusive end of April

PARTIAL_MAY_PATH = MONTHLY_DIR / "bills_2026_05_partial.parquet"


def pull_partial_may() -> None:
    if PARTIAL_MAY_PATH.exists():
        print(f"  {PARTIAL_MAY_PATH.name} already on disk, skipping")
        return
    print(f"  Pulling May 1-14, 2026 from Redshift...")
    sql = month_sql(PATIENT_COL, date(2026, 5, 1), TEST_END)
    with connect() as conn:
        df = pd.read_sql(sql, conn)
    df = df[df["bill_value_net"] > 0]
    write_atomic_parquet(PARTIAL_MAY_PATH, df)
    print(f"    Wrote {PARTIAL_MAY_PATH.name} ({len(df):,} rows)")


def load_all_bills() -> pd.DataFrame:
    files = sorted(MONTHLY_DIR.glob("bills_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No monthly files in {MONTHLY_DIR}")
    print(f"  Loading {len(files)} monthly files...")
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["bill_id"])
    df["bill_date"] = pd.to_datetime(df["bill_date"]).dt.date
    return df


def build_training_summary(train_bills: pd.DataFrame) -> pd.DataFrame:
    """One row per training patient (bill_count >= 1).

    Light = 1 or 2 training bills; warm = >= 3.
    """
    bv = train_bills["bill_value_net"].astype(float)
    train_bills = train_bills.assign(_bv=bv, _bv2=bv * bv)

    g = train_bills.groupby("patient_id", sort=False)
    agg = g.agg(
        bill_count=("_bv", "size"),
        sum_v=("_bv", "sum"),
        sumsq_v=("_bv2", "sum"),
        min_v=("_bv", "min"),
        max_v=("_bv", "max"),
        last_bill_date=("bill_date", "max"),
    )
    print("  Computing p80 / p90 / p95 per user (slow step)...")
    agg["pred_p80"] = g["_bv"].quantile(0.80)
    agg["pred_p90"] = g["_bv"].quantile(0.90)
    agg["pred_p95"] = g["_bv"].quantile(0.95)
    agg["pred_max"] = agg["max_v"]  # already computed above

    n = agg["bill_count"].astype(float)
    agg["pred_avg"] = agg["sum_v"] / n
    var_full = (agg["sumsq_v"] / n) - agg["pred_avg"] ** 2
    agg["sd_full"] = np.sqrt(var_full.clip(lower=0))

    trim_n = (n - 2).replace(0, np.nan)
    trim_sum = agg["sum_v"] - agg["min_v"] - agg["max_v"]
    trim_sumsq = agg["sumsq_v"] - agg["min_v"] ** 2 - agg["max_v"] ** 2
    pred_trimmed = np.where(n >= 4, trim_sum / trim_n, np.nan)
    trim_var = np.where(n >= 4, (trim_sumsq / trim_n) - pred_trimmed ** 2, np.nan)
    agg["pred_trimmed"] = pred_trimmed
    agg["sd_trimmed"] = np.sqrt(np.clip(trim_var, a_min=0, a_max=None))

    agg = agg[agg["bill_count"] >= 1].copy()
    agg["segment"] = np.where(agg["bill_count"] >= 3, "warm", "light")

    return agg.reset_index()[[
        "patient_id", "bill_count",
        "pred_avg", "pred_p80", "pred_p90", "pred_p95", "pred_max", "pred_trimmed",
        "sd_full", "sd_trimmed",
        "last_bill_date", "segment",
    ]]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: ensure May 1-14, 2026 is on disk")
    pull_partial_may()

    print("\nStep 2: load all monthly bills")
    bills = load_all_bills()
    print(f"  Total bills (deduped): {len(bills):,}")

    print(f"\nStep 3: split at cutoff {CUTOFF}")
    train_bills = bills[bills["bill_date"] < CUTOFF].copy()
    test_bills = bills[(bills["bill_date"] >= CUTOFF) & (bills["bill_date"] < TEST_END)].copy()
    print(f"  Training bills: {len(train_bills):,}")
    print(f"  Test bills (raw): {len(test_bills):,}")

    print("\nStep 4: build training user summary")
    summary = build_training_summary(train_bills)
    n_warm = int((summary["segment"] == "warm").sum())
    n_light = int((summary["segment"] == "light").sum())
    print(f"  Training users: {len(summary):,}  (warm: {n_warm:,}, light: {n_light:,})")

    print("\nStep 5: drop new-patient bills from test")
    train_users = set(summary["patient_id"])
    is_returning = test_bills["patient_id"].isin(train_users)
    n_new = int((~is_returning).sum())
    test_bills_r = test_bills[is_returning].copy()
    print(f"  Excluded (new-patient bills): {n_new:,}")
    print(f"  Test bills (returning patients): {len(test_bills_r):,}")

    test_bills_r["window"] = np.where(
        test_bills_r["bill_date"] < APRIL_END, "april", "may"
    )
    n_apr = int((test_bills_r["window"] == "april").sum())
    n_may = int((test_bills_r["window"] == "may").sum())
    print(f"    April:     {n_apr:,}")
    print(f"    May 1-14:  {n_may:,}")

    print("\nStep 6: write outputs")
    write_atomic_parquet(DATA_DIR / "user_summary.parquet", summary)
    print(f"  Wrote data/user_summary.parquet ({len(summary):,} rows)")
    write_atomic_parquet(DATA_DIR / "test_bills.parquet", test_bills_r)
    print(f"  Wrote data/test_bills.parquet ({len(test_bills_r):,} rows)")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    topline = {
        "n_users": int(len(summary)),
        "n_warm": n_warm,
        "n_light": n_light,
        "n_training_bills": int(len(train_bills)),
        "n_test_bills_april": n_apr,
        "n_test_bills_may": n_may,
        "n_excluded_new_patient": n_new,
        "training_cutoff": CUTOFF.isoformat(),
        "test_window_april": ["2026-04-01", "2026-04-30"],
        "test_window_may": ["2026-05-01", "2026-05-14"],
        "generated_at": generated_at,
    }
    (DATA_DIR / "user_topline.json").write_text(json.dumps(topline, indent=2))
    print("  Wrote data/user_topline.json")
    print(json.dumps(topline, indent=2))


if __name__ == "__main__":
    main()
