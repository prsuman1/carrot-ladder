"""Pure functions over the value_distribution dataframe.

All input is the small ~7k-row histogram with columns (bin, count, sum_value).
Slider-driven recomputation runs through here — never through cached file loads.

Generalized to N tiers (1..5). The thresholds list is the source of truth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import pandas as pd


def zone_counts(vd: pd.DataFrame, total_n: int, thresholds: Sequence[float], reach: float) -> Dict:
    """Compute the (2N+1)-zone partition for a sorted thresholds list.

    Returns a dict with:
      - tiers: list of dicts {tier, T, lo, count_in_reach, pct_in_reach,
                              sum_in_reach, avg_gap, gap_ratio}
      - mid_zones: list of dicts {after_tier, lo, hi, count, pct} (one per gap
                   between consecutive tiers)
      - n_unreachable, pct_unreachable (bills below lo of T1)
      - n_past_top, pct_past_top (bills at or above the top tier — wasted carrot)
    """
    bins = vd["bin"].values
    counts = vd["count"].values
    sums = vd["sum_value"].values

    def slice_count_sum(lo, hi):
        m = (bins >= lo) & (bins < hi)
        return int(counts[m].sum()), float(sums[m].sum())

    los: List[float] = []
    prev = 0.0
    for T in thresholds:
        lo = max(prev, reach * T)
        los.append(lo)
        prev = T

    tiers = []
    for i, T in enumerate(thresholds):
        lo = los[i]
        c, s = slice_count_sum(lo, T)
        gap_ratio = 0.0 if c == 0 else (T - s / c) / T
        tiers.append({
            "tier": i + 1,
            "T": T,
            "lo": lo,
            "count_in_reach": c,
            "pct_in_reach": c / total_n * 100,
            "sum_in_reach": s,
            "avg_gap": (T - s / c) if c > 0 else math.nan,
            "gap_ratio": gap_ratio,
        })

    mid_zones = []
    for i in range(len(thresholds) - 1):
        lo_mz = thresholds[i]
        hi_mz = los[i + 1]
        c, _ = slice_count_sum(lo_mz, hi_mz)
        mid_zones.append({
            "after_tier": i + 1,
            "lo": lo_mz,
            "hi": hi_mz,
            "count": c,
            "pct": c / total_n * 100,
        })

    n_unreach, _ = slice_count_sum(0, los[0])
    n_past_top, _ = slice_count_sum(thresholds[-1], 10**9)

    return {
        "thresholds": list(thresholds),
        "los": los,
        "tiers": tiers,
        "mid_zones": mid_zones,
        "n_unreachable": n_unreach,
        "pct_unreachable": n_unreach / total_n * 100,
        "n_past_top": n_past_top,
        "pct_past_top": n_past_top / total_n * 100,
    }


def expected_lift(zones: Dict, response_rate: float, gap_aware: bool = True) -> Dict:
    """Per-tier earned and free counts (% of all bills) under the single-gift model.

    Each bill receives at most one gift — the highest tier it crosses. So:
      earned_per_tier[i] = response_rate × pct_in_reach[i] × gap_factor[i]
      free_per_tier[i]   = mid_zone_pct[i]                                              # if i < N
                         + pct_in_reach[i+1] × (1 − response_rate × gap_factor[i+1])    # if i+1 ≤ N
                         (or pct_past_top for i = N − 1)

    With gap_aware=True, gap_factor = 1 − gap_ratio.
    """
    n_tiers = len(zones["tiers"])

    earned_per_tier: list[float] = []
    for t in zones["tiers"]:
        gap_factor = (1 - t["gap_ratio"]) if gap_aware else 1.0
        earned_per_tier.append(response_rate * t["pct_in_reach"] * gap_factor)

    free_per_tier: list[float] = []
    for i in range(n_tiers):
        if i < n_tiers - 1:
            next_t = zones["tiers"][i + 1]
            next_gap = (1 - next_t["gap_ratio"]) if gap_aware else 1.0
            non_convert = 1.0 - response_rate * next_gap
            free = zones["mid_zones"][i]["pct"] + non_convert * next_t["pct_in_reach"]
        else:
            free = zones["pct_past_top"]
        free_per_tier.append(free)

    earned_total = sum(earned_per_tier)
    free_total = sum(free_per_tier)

    return {
        "per_tier": earned_per_tier,
        "earned_per_tier": earned_per_tier,
        "total": earned_total,
        "earned_total": earned_total,
        "free_per_tier": free_per_tier,
        "free_total": free_total,
    }


def money_breakdown(
    lift: Dict,
    zones: Dict,
    gift_costs: Sequence[float],
    n_bills: int,
    months_in_window: float = 3.0,
    rgm_rate: float = 0.36,
) -> Dict:
    """Per-tier and total ₹/month earn (RGM) and burn (gift cost).

    Burn at tier i = (earned + free) × gift_cost_i — every gift handed out costs cash.
    Revenue at tier i = avg_gap × earned_bills (incremental top-line from program).
    RGM = Revenue × rgm_rate (the contribution-margin slice that's actually money).
    """
    pm = lambda count_pct: (count_pct / 100.0) * n_bills / months_in_window
    per_tier = []
    total_rev = total_rgm = total_burn = 0.0
    for i, t in enumerate(zones["tiers"]):
        earned_bills = pm(lift["earned_per_tier"][i])
        free_bills = pm(lift["free_per_tier"][i])
        avg_gap = t["avg_gap"]
        if avg_gap is None or (isinstance(avg_gap, float) and avg_gap != avg_gap):
            avg_gap = 0.0
        revenue = avg_gap * earned_bills
        rgm = revenue * rgm_rate
        burn = (earned_bills + free_bills) * gift_costs[i]
        per_tier.append({
            "tier": i + 1,
            "T": t["T"],
            "gift_cost": gift_costs[i],
            "earned_bills": earned_bills,
            "free_bills": free_bills,
            "avg_gap": avg_gap,
            "revenue": revenue,
            "rgm": rgm,
            "burn": burn,
            "net": rgm - burn,
        })
        total_rev += revenue
        total_rgm += rgm
        total_burn += burn
    return {
        "per_tier": per_tier,
        "revenue": total_rev,
        "rgm": total_rgm,
        "burn": total_burn,
        "net": total_rgm - total_burn,
        "rgm_rate": rgm_rate,
    }


def score_pair(zones: Dict, lift: Dict, alpha: float = 1.0) -> Dict:
    """Generalized score for an N-tier configuration under the single-gift model.

    reach_score        = sum of pct_in_reach across tiers (theoretical addressable %)
    wasted_carrot_pct  = lift['free_total'] (sum of per-tier free gifts)
    adjusted_score     = reach_score − α × wasted_carrot_pct
    expected_lift      = lift['total'] (= earned_total %)
    adjusted_expected  = expected_lift − α × wasted_carrot_pct
    """
    reach = sum(t["pct_in_reach"] for t in zones["tiers"])
    waste = lift["free_total"]
    lift_total = lift["total"]
    return {
        "reach_score": reach,
        "wasted_carrot_pct": waste,
        "adjusted_score": reach - alpha * waste,
        "expected_lift": lift_total,
        "adjusted_expected": lift_total - alpha * waste,
    }
