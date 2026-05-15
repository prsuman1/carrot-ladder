"""Per-user data builder (offline, run once).

Pulls 24 months of bill-grain payable values from Redshift — one query per
calendar month so Redshift never has to return the full 2-year range in a
single shot — saves each month as a parquet under data/bills_monthly/, then
merges them and computes a one-row-per-patient summary:

    {patient_id, bill_count, pred_avg, pred_p80, pred_trimmed,
     sd_full, sd_trimmed, last_bill_date, segment}

The streamlit simulator (personalized_page.py) reads only the small
user_summary.parquet, never the raw monthlies.

Usage:
    python build_user_data.py                  # pull all 24 months + build summary
    python build_user_data.py --skip-pull      # just rebuild summary from existing monthlies
    python build_user_data.py --months 1       # pull only the most recent 1 month (verify column names)

Resumable: monthly files already on disk are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "data"
MONTHLY_DIR = DATA_DIR / "bills_monthly"

# Patient column name. If Redshift errors with "column does not exist",
# try alternatives: "customer-id", "patient_id", etc.
PATIENT_COL = "patient-id"


def month_sql(patient_col: str, month_start: date, month_end: date) -> str:
    """ZRF-aware bill-value SQL for a single calendar month [start, end).

    Same payable-value rules as build_data.py, plus patient_id in the group key.
    """
    return f"""
WITH line_value AS (
  SELECT s."bill-id",
         s."{patient_col}" AS patient_id,
         s."created-at",
         CASE
           WHEN s."promo-code" LIKE 'ZRF%%' AND s."promo-discount" > 0 AND s."net-quantity" > 1
             THEN s.rate * (s."net-quantity" - 1)
           WHEN s."promo-code" LIKE 'ZRF%%' AND s."promo-discount" > 0
             THEN 0
           ELSE s.rate * s."net-quantity"
         END AS line_payable_gross,
         CASE WHEN s."promo-code" LIKE 'ZRD%%' THEN s."promo-discount" ELSE 0 END AS line_zrd_disc
  FROM "prod2-generico".sales s
  WHERE s."bill-flag" = 'gross'
    AND s."created-at" >= DATE '{month_start.isoformat()}'
    AND s."created-at" <  DATE '{month_end.isoformat()}'
)
SELECT "bill-id" AS bill_id,
       patient_id,
       MIN("created-at")::date AS bill_date,
       SUM(line_payable_gross) - SUM(line_zrd_disc) AS bill_value_net
FROM line_value
GROUP BY "bill-id", patient_id
"""


def connect():
    load_dotenv()
    return psycopg2.connect(
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", 5439)),
        dbname=os.environ["REDSHIFT_DB"],
        user=os.environ["REDSHIFT_USER"],
        password=os.environ["REDSHIFT_PASSWORD"],
    )


def month_filename(month_start: date) -> Path:
    return MONTHLY_DIR / f"bills_{month_start.strftime('%Y_%m')}.parquet"


def pull_month(month_start: date, month_end: date, patient_col: str = PATIENT_COL) -> pd.DataFrame:
    print(f"  Pulling {month_start} → {month_end}...", flush=True)
    sql = month_sql(patient_col, month_start, month_end)
    with connect() as conn:
        df = pd.read_sql(sql, conn)
    print(f"    Got {len(df):,} bills", flush=True)
    return df


def write_atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def month_ranges(num_months: int, end_anchor: date | None = None) -> list[tuple[date, date]]:
    """Return [(start, end_exclusive), ...] for the most recent `num_months`
    full calendar months, oldest first. End anchor = first of current month."""
    if end_anchor is None:
        today = date.today()
        end_anchor = date(today.year, today.month, 1)
    starts = [end_anchor - relativedelta(months=i + 1) for i in range(num_months)]
    starts.sort()
    return [(s, s + relativedelta(months=1)) for s in starts]


def pull_all_months(num_months: int, patient_col: str = PATIENT_COL) -> None:
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    ranges = month_ranges(num_months)
    print(f"Will pull {len(ranges)} months: {ranges[0][0]} → {ranges[-1][1]}")
    for i, (ms, me) in enumerate(ranges, 1):
        out = month_filename(ms)
        if out.exists():
            print(f"[{i}/{len(ranges)}] {ms} — already on disk, skipping")
            continue
        print(f"[{i}/{len(ranges)}] {ms}")
        df = pull_month(ms, me, patient_col)
        df = df[df["bill_value_net"] > 0]
        write_atomic_parquet(out, df)
        print(f"    Wrote {out.name} ({len(df):,} rows kept)")


# --------------------------------------------------------------------------
# Per-user summary
# --------------------------------------------------------------------------


def load_all_monthlies() -> pd.DataFrame:
    files = sorted(MONTHLY_DIR.glob("bills_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No monthly files in {MONTHLY_DIR}")
    print(f"Loading {len(files)} monthly files...")
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["bill_id"])
    print(f"Total bills after dedupe: {len(df):,}")
    return df


def build_user_summary(bills: pd.DataFrame) -> pd.DataFrame:
    """One row per patient. Excludes patients with bill_count == 1 (new patients).

    Vectorised: groupby aggregations + closed-form trimmed-mean derivation
    (trimmed = drop one min and one max; requires bill_count >= 4).
    """
    print("Computing per-user aggregates...")
    bv = bills["bill_value_net"].astype(float)
    bills = bills.assign(_bv=bv, _bv2=bv * bv)

    g = bills.groupby("patient_id", sort=False)
    agg = g.agg(
        bill_count=("_bv", "size"),
        sum_v=("_bv", "sum"),
        sumsq_v=("_bv2", "sum"),
        min_v=("_bv", "min"),
        max_v=("_bv", "max"),
        last_bill_date=("bill_date", "max"),
    )

    print("Computing p80 per user (this is the slowest step)...")
    agg["pred_p80"] = g["_bv"].quantile(0.80)

    n = agg["bill_count"].astype(float)
    agg["pred_avg"] = agg["sum_v"] / n
    var_full = (agg["sumsq_v"] / n) - agg["pred_avg"] ** 2
    agg["sd_full"] = np.sqrt(var_full.clip(lower=0))

    trim_n = n - 2
    trim_sum = agg["sum_v"] - agg["min_v"] - agg["max_v"]
    trim_sumsq = agg["sumsq_v"] - agg["min_v"] ** 2 - agg["max_v"] ** 2
    pred_trimmed = np.where(n >= 4, trim_sum / trim_n.replace(0, np.nan), np.nan)
    trim_var = np.where(n >= 4,
                        (trim_sumsq / trim_n.replace(0, np.nan)) - pred_trimmed ** 2,
                        np.nan)
    agg["pred_trimmed"] = pred_trimmed
    agg["sd_trimmed"] = np.sqrt(np.clip(trim_var, a_min=0, a_max=None))

    agg = agg[agg["bill_count"] >= 2].copy()  # exclude new patients (first-ever bill only)
    agg["segment"] = np.where(agg["bill_count"] >= 3, "warm", "light")

    out = agg.reset_index()[[
        "patient_id", "bill_count",
        "pred_avg", "pred_p80", "pred_trimmed",
        "sd_full", "sd_trimmed",
        "last_bill_date", "segment",
    ]]
    print(f"Users in summary: {len(out):,}  "
          f"(warm: {(out['segment'] == 'warm').sum():,}, "
          f"light: {(out['segment'] == 'light').sum():,})")
    return out


def build_user_topline(summary: pd.DataFrame, bills_total: int, generated_at: str) -> dict:
    return {
        "n_users": int(len(summary)),
        "n_warm": int((summary["segment"] == "warm").sum()),
        "n_light": int((summary["segment"] == "light").sum()),
        "total_bills": int(bills_total),
        "mean_pred_avg": float(summary["pred_avg"].mean()),
        "median_pred_avg": float(summary["pred_avg"].median()),
        "generated_at": generated_at,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=24,
                   help="How many recent full months to pull (default 24).")
    p.add_argument("--skip-pull", action="store_true",
                   help="Skip Redshift pulls; just rebuild user_summary from existing monthlies.")
    p.add_argument("--patient-col", default=PATIENT_COL,
                   help=f'Patient column name in sales (default "{PATIENT_COL}").')
    args = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not args.skip_pull:
        pull_all_months(args.months, args.patient_col)

    bills = load_all_monthlies()
    summary = build_user_summary(bills)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    write_atomic_parquet(DATA_DIR / "user_summary.parquet", summary)
    print(f"Wrote data/user_summary.parquet ({len(summary):,} users)")

    topline = build_user_topline(summary, bills_total=len(bills), generated_at=generated_at)
    (DATA_DIR / "user_topline.json").write_text(json.dumps(topline, indent=2))
    print(f"Wrote data/user_topline.json")
    print(json.dumps(topline, indent=2))


if __name__ == "__main__":
    main()
