"""One-time data fetcher.

Pulls 90 days of bill-grain payable values from Redshift, computes 5 summary
files used by the Streamlit app, and writes them to ./data/.

Run locally with valid .env credentials. The Streamlit app never imports this.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "data"

SQL = """
WITH line_value AS (
  SELECT s."bill-id",
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
    AND s."created-at" >= DATEADD(day, -90, CURRENT_DATE)
    AND s."created-at" <  CURRENT_DATE
)
SELECT "bill-id",
       MIN("created-at")::date AS bill_date,
       SUM(line_payable_gross) - SUM(line_zrd_disc) AS bill_value_net
FROM line_value
GROUP BY "bill-id"
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


def fetch_bills() -> pd.DataFrame:
    print("Connecting to Redshift...")
    with connect() as conn:
        print("Running bill-grain query (90 days, ZRF-aware payable)...")
        df = pd.read_sql(SQL, conn)
    print(f"Pulled {len(df):,} bills")
    return df


def build_topline(bv: np.ndarray, bill_dates: pd.Series, generated_at: str) -> dict:
    return {
        "n_bills": int(bv.size),
        "total_revenue": float(bv.sum()),
        "mean": float(bv.mean()),
        "median": float(np.median(bv)),
        "date_window_start": str(bill_dates.min()),
        "date_window_end": str(bill_dates.max()),
        "generated_at": generated_at,
    }


def build_percentiles(bv: np.ndarray) -> dict:
    pcts = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(bv, pcts)
    return {f"P{p}": float(v) for p, v in zip(pcts, values)}


def build_histogram(bv: np.ndarray) -> pd.DataFrame:
    """Histogram with mixed bin widths.

    - ₹50 bins from 0 to 2000
    - ₹100 bins from 2000 to 5000
    - one overflow row for >= 5000
    """
    edges = np.concatenate([
        np.arange(0, 2001, 50),         # 0,50,...,2000
        np.arange(2100, 5001, 100),     # 2100,...,5000
    ])
    counts, _ = np.histogram(bv, bins=edges)
    rows = []
    for i in range(len(edges) - 1):
        rows.append({
            "bin_lo": int(edges[i]),
            "bin_hi": int(edges[i + 1]),
            "count": int(counts[i]),
        })
    overflow = int((bv >= 5000).sum())
    rows.append({"bin_lo": 5000, "bin_hi": -1, "count": overflow})
    df = pd.DataFrame(rows)
    df["pct"] = df["count"] / df["count"].sum() * 100.0
    return df


def build_cdf(bv: np.ndarray) -> pd.DataFrame:
    thresholds = np.arange(0, 2001, 50)
    sorted_bv = np.sort(bv)
    n = sorted_bv.size
    pct_below = np.searchsorted(sorted_bv, thresholds, side="left") / n * 100.0
    pct_at_or_above = 100.0 - pct_below
    return pd.DataFrame({
        "threshold": thresholds.astype(int),
        "pct_below": pct_below,
        "pct_at_or_above": pct_at_or_above,
    })


def build_value_distribution(bv: np.ndarray) -> pd.DataFrame:
    """₹1 resolution histogram from 0 to 9999, with bin=10000 aggregating ≥ 10000.

    This is the workhorse used for live zone metrics.
    """
    bin_int = np.minimum(np.floor(bv).astype(int), 10000)
    df = pd.DataFrame({"bin": bin_int, "value": bv})
    vd = df.groupby("bin", as_index=False).agg(
        count=("value", "size"),
        sum_value=("value", "sum"),
    )
    vd["bin"] = vd["bin"].astype(np.int32)
    vd["count"] = vd["count"].astype(np.int64)
    vd["sum_value"] = vd["sum_value"].astype(np.float64)
    return vd


def write_atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def write_atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    bills = fetch_bills()

    bv_full = bills["bill_value_net"].astype(float)
    keep_mask = bv_full > 0
    bv = bv_full[keep_mask].values
    bill_dates = bills.loc[keep_mask, "bill_date"]
    dropped = int((~keep_mask).sum())
    print(f"Dropped {dropped:,} bills with bill_value_net <= 0; kept {len(bv):,}")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    topline = build_topline(bv, bill_dates, generated_at)
    write_atomic_json(DATA_DIR / "topline.json", topline)
    print(f"Wrote data/topline.json (n_bills={topline['n_bills']:,})")

    pcts = build_percentiles(bv)
    write_atomic_json(DATA_DIR / "percentiles.json", pcts)
    print("Wrote data/percentiles.json")

    hist = build_histogram(bv)
    write_atomic_parquet(DATA_DIR / "histogram_chart.parquet", hist)
    print(f"Wrote data/histogram_chart.parquet ({len(hist)} rows)")

    cdf = build_cdf(bv)
    write_atomic_parquet(DATA_DIR / "cdf_chart.parquet", cdf)
    print(f"Wrote data/cdf_chart.parquet ({len(cdf)} rows)")

    vd = build_value_distribution(bv)
    write_atomic_parquet(DATA_DIR / "value_distribution.parquet", vd)
    print(f"Wrote data/value_distribution.parquet ({len(vd)} rows)")

    total_size = sum(p.stat().st_size for p in DATA_DIR.iterdir() if p.is_file())
    print(f"Total data/ size: {total_size / 1024:.1f} KB")
    print(f"generated_at = {generated_at}")


if __name__ == "__main__":
    main()
