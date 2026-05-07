"""Pure functions over the value_distribution dataframe.

All input is the small ~7k-row histogram with columns (bin, count, sum_value).
Slider-driven recomputation runs through here — never through cached file loads.
"""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd


def zone_counts(vd: pd.DataFrame, total_n: int, T1: float, T2: float, reach: float) -> Dict[str, float]:
    """Compute the five-zone partition for a (T1, T2) pair under a reach window.

    Zones (every bill falls into exactly one):
      1. Unreachable        bill < lo1
      2. Within reach T1    lo1 <= bill < T1
      3. Past T1, mid-zone  T1 <= bill < lo2
      4. Within reach T2    lo2 <= bill < T2
      5. Past T2            bill >= T2
    """
    lo1 = reach * T1
    lo2 = max(T1, reach * T2)

    bins = vd["bin"].values
    counts = vd["count"].values
    sums = vd["sum_value"].values

    def slice_count_and_sum(lo: float, hi: float):
        m = (bins >= lo) & (bins < hi)
        return int(counts[m].sum()), float(sums[m].sum())

    n_unreach, _ = slice_count_and_sum(0, lo1)
    n_inreach1, sum_inreach1 = slice_count_and_sum(lo1, T1)
    n_mid, _ = slice_count_and_sum(T1, lo2)
    n_inreach2, sum_inreach2 = slice_count_and_sum(lo2, T2)
    m_ge_T2 = bins >= T2
    n_past2 = int(counts[m_ge_T2].sum())

    pct_unreachable = n_unreach / total_n * 100
    pct_within_reach_T1 = n_inreach1 / total_n * 100
    pct_mid_zone = n_mid / total_n * 100
    pct_within_reach_T2 = n_inreach2 / total_n * 100
    pct_ge_T2 = n_past2 / total_n * 100
    pct_ge_T1 = (n_mid + n_inreach2 + n_past2) / total_n * 100

    avg_gap_T1 = (T1 - sum_inreach1 / n_inreach1) if n_inreach1 else math.nan
    avg_gap_T2 = (T2 - sum_inreach2 / n_inreach2) if n_inreach2 else math.nan

    return {
        "pct_unreachable": pct_unreachable,
        "pct_within_reach_T1": pct_within_reach_T1,
        "pct_mid_zone": pct_mid_zone,
        "pct_within_reach_T2": pct_within_reach_T2,
        "pct_ge_T2": pct_ge_T2,
        "pct_ge_T1": pct_ge_T1,
        "avg_gap_T1": avg_gap_T1,
        "avg_gap_T2": avg_gap_T2,
        "lo1": lo1,
        "lo2": lo2,
        "n_unreach": n_unreach,
        "n_within_reach_T1": n_inreach1,
        "n_mid_zone": n_mid,
        "n_within_reach_T2": n_inreach2,
        "n_ge_T2": n_past2,
    }


def score_pair(metrics: Dict[str, float], alpha: float = 1.0) -> Dict[str, float]:
    reach_score = metrics["pct_within_reach_T1"] + metrics["pct_within_reach_T2"]
    waste = metrics["pct_ge_T2"]
    return {
        "reach_score": reach_score,
        "wasted_carrot_pct": waste,
        "adjusted_score": reach_score - alpha * waste,
    }
