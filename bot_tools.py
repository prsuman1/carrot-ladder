"""Carrot Bot tool layer.

Defines the five tools the LLM can invoke (JSON schemas + Python implementations)
and a single `dispatch(name, args)` entry point. All math goes through
`metrics.py` so the bot's numbers always agree with the dashboard.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data_loader import load_topline, load_percentiles, load_value_distribution
from metrics import (
    breakeven_cashback,
    expected_lift,
    money_breakdown,
    score_pair,
    zone_counts,
)

MONTHS_IN_WINDOW = 3.0


# ---------------------------------------------------------------------------
# Cached data accessors
# ---------------------------------------------------------------------------
def _vd() -> pd.DataFrame:
    return load_value_distribution()


def _topline() -> dict:
    return load_topline()


def _percentiles() -> dict:
    return load_percentiles()


def _n_bills() -> int:
    return int(_topline()["n_bills"])


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _round_dict(d: Dict[str, Any], dp: int = 4) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, dp)
        elif isinstance(v, list):
            out[k] = [round(x, dp) if isinstance(x, float) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _round_dict(v, dp)
        else:
            out[k] = v
    return out


def get_data_summary() -> Dict[str, Any]:
    t = _topline()
    p = _percentiles()
    n = int(t["n_bills"])
    return {
        "n_bills_90d": n,
        "n_bills_per_month": int(round(n / MONTHS_IN_WINDOW)),
        "total_revenue_90d": round(float(t["total_revenue"]), 2),
        "revenue_per_month": round(float(t["total_revenue"]) / MONTHS_IN_WINDOW, 2),
        "mean_bill": round(float(t["mean"]), 2),
        "median_bill": round(float(t["median"]), 2),
        "percentiles": {k: round(float(v), 2) for k, v in p.items()},
        "date_window_start": t["date_window_start"],
        "date_window_end": t["date_window_end"],
        "data_generated_at": t["generated_at"],
        "notes": [
            "All bill counts and revenue are 90-day cumulative; divide by 3 for monthly avg.",
            "Distribution is heavy right-skewed: median < mean by ~2x.",
            "Zeno = pharmacy + daily-use FMCG (not pharmacy-only).",
        ],
    }


def get_dashboard_state() -> Dict[str, Any]:
    """Read the user's CURRENT sidebar widget values from session_state.

    Returns sane defaults if a widget hasn't been rendered yet (chat-first user).
    """
    ss = st.session_state
    n_tiers = int(ss.get("__n_tiers_radio__", 2))
    thresholds = []
    DEFAULT_T = [399, 899, 1299, 1699, 1999]
    DEFAULT_GIFT = [10, 25, 50, 100, 150]
    for i in range(n_tiers):
        thresholds.append(int(ss.get(f"T_{i + 1}", DEFAULT_T[i])))
    gift_costs = [float(ss.get(f"gift_{i + 1}", DEFAULT_GIFT[i])) for i in range(n_tiers)]

    reach_pct = int(ss.get("__reach_slider__", 75))
    response_pct = int(ss.get(
        next((k for k in ss.keys() if "response" in k.lower()), "_unset"),
        10,
    ))
    # Fallback: response slider key isn't custom; Streamlit auto-keys it. We use defaults.
    overshoot_pct = 30
    rgm_pct = 36
    alpha = 1.0
    gap_aware = True

    # Compute current state
    vd = _vd()
    n_bills = _n_bills()
    z = zone_counts(vd, n_bills, thresholds, reach_pct / 100.0)
    lf = expected_lift(z, response_pct / 100.0, gap_aware)
    s = score_pair(z, lf, alpha)
    mb = money_breakdown(lf, z, gift_costs, n_bills, MONTHS_IN_WINDOW,
                         rgm_pct / 100.0, 1.0 + overshoot_pct / 100.0)
    coverage = sum(lf["earned_per_tier"]) + sum(lf["free_per_tier"])

    return {
        "thresholds": thresholds,
        "n_tiers": n_tiers,
        "reach_pct": reach_pct,
        "response_pct": response_pct,
        "gap_aware": gap_aware,
        "alpha": alpha,
        "overshoot_pct": overshoot_pct,
        "rgm_pct": rgm_pct,
        "gift_costs": gift_costs,
        "current_metrics": {
            "lift_pct": round(lf["total"], 2),
            "waste_pct": round(lf["free_total"], 2),
            "coverage_pct": round(coverage, 2),
            "adjusted_expected": round(s["adjusted_expected"], 2),
            "revenue_per_month": round(mb["revenue"], 2),
            "rgm_per_month": round(mb["rgm"], 2),
            "burn_per_month": round(mb["burn"], 2),
            "net_per_month": round(mb["net"], 2),
            "roi_pct": (round(mb["net"] / mb["burn"] * 100, 2)
                        if mb["burn"] > 0 else None),
        },
    }


def evaluate_ladder(
    thresholds: List[int],
    reach_pct: int = 70,
    response_pct: int = 50,
    gap_aware: bool = True,
    overshoot_pct: int = 30,
    rgm_pct: int = 36,
    cashback_pct: float = None,
    gift_costs: List[float] = None,
) -> Dict[str, Any]:
    if cashback_pct is None and gift_costs is None:
        return {"error": "Provide either cashback_pct (e.g. 3 for 3%) or gift_costs (list of ₹/tier)."}
    if not thresholds or any(t < 100 for t in thresholds):
        return {"error": "thresholds must be a non-empty list of integers ≥ 100."}
    thresholds = sorted(int(t) for t in thresholds)
    vd = _vd()
    n_bills = _n_bills()
    reach = reach_pct / 100.0
    response = response_pct / 100.0
    kappa = 1.0 + overshoot_pct / 100.0
    rgm = rgm_pct / 100.0
    if gift_costs is None:
        gift_costs = [(cashback_pct / 100.0) * T for T in thresholds]
    if len(gift_costs) != len(thresholds):
        return {"error": f"len(gift_costs)={len(gift_costs)} != len(thresholds)={len(thresholds)}."}

    z = zone_counts(vd, n_bills, thresholds, reach)
    lf = expected_lift(z, response, gap_aware)
    s = score_pair(z, lf, 1.0)
    mb = money_breakdown(lf, z, gift_costs, n_bills, MONTHS_IN_WINDOW, rgm, kappa)
    coverage = sum(lf["earned_per_tier"]) + sum(lf["free_per_tier"])
    pm = lambda pct: int(round(pct / 100.0 * n_bills / MONTHS_IN_WINDOW))

    per_tier = []
    for i, t in enumerate(z["tiers"]):
        per_tier.append({
            "tier": i + 1,
            "T": t["T"],
            "gift_cost": round(float(gift_costs[i]), 2),
            "avg_gap": round(float(t["avg_gap"]) if t["avg_gap"] == t["avg_gap"] else 0.0, 2),
            "earned_pct": round(lf["earned_per_tier"][i], 2),
            "free_pct": round(lf["free_per_tier"][i], 2),
            "earned_bills_per_month": pm(lf["earned_per_tier"][i]),
            "free_bills_per_month": pm(lf["free_per_tier"][i]),
            "revenue_per_month": round(mb["per_tier"][i]["revenue"], 2),
            "rgm_per_month": round(mb["per_tier"][i]["rgm"], 2),
            "burn_per_month": round(mb["per_tier"][i]["burn"], 2),
            "net_per_month": round(mb["per_tier"][i]["net"], 2),
        })

    return {
        "inputs": {
            "thresholds": thresholds,
            "n_tiers": len(thresholds),
            "reach_pct": reach_pct,
            "response_pct": response_pct,
            "gap_aware": gap_aware,
            "overshoot_pct": overshoot_pct,
            "rgm_pct": rgm_pct,
            "cashback_pct": cashback_pct,
            "gift_costs": [round(g, 2) for g in gift_costs],
        },
        "totals": {
            "lift_pct": round(lf["total"], 2),
            "waste_pct": round(lf["free_total"], 2),
            "coverage_pct": round(coverage, 2),
            "coverage_bills_per_month": pm(coverage),
            "adjusted_expected": round(s["adjusted_expected"], 2),
            "revenue_per_month": round(mb["revenue"], 2),
            "rgm_per_month": round(mb["rgm"], 2),
            "burn_per_month": round(mb["burn"], 2),
            "net_per_month": round(mb["net"], 2),
            "roi_pct": (round(mb["net"] / mb["burn"] * 100, 2)
                        if mb["burn"] > 0 else None),
        },
        "per_tier": per_tier,
    }


def find_breakeven(
    thresholds: List[int],
    reach_pct: int = 70,
    response_pct: int = 50,
    gap_aware: bool = True,
    overshoot_pct: int = 30,
    rgm_pct: int = 36,
) -> Dict[str, Any]:
    thresholds = sorted(int(t) for t in thresholds)
    vd = _vd()
    n_bills = _n_bills()
    be = breakeven_cashback(
        vd, thresholds, reach_pct / 100.0, response_pct / 100.0, gap_aware,
        n_bills, MONTHS_IN_WINDOW, rgm_pct / 100.0, 1.0 + overshoot_pct / 100.0,
    )
    return {
        "inputs": {
            "thresholds": thresholds,
            "reach_pct": reach_pct,
            "response_pct": response_pct,
            "gap_aware": gap_aware,
            "overshoot_pct": overshoot_pct,
            "rgm_pct": rgm_pct,
        },
        "breakeven_cashback_pct": round(be * 100, 2),
        "interpretation": (
            f"At a per-tier cashback of {be*100:.2f}% (gift_i = {be*100:.2f}% × T_i), "
            f"this ladder breaks even (Net = ₹0). Rates below this are profitable; above, "
            f"the program loses money under the given response/RGM/overshoot assumptions."
        ),
    }


def search_ladders(
    n_tiers_min: int = 1,
    n_tiers_max: int = 4,
    reach_options: List[int] = None,
    response_pct: int = 50,
    gap_aware: bool = True,
    overshoot_pct: int = 30,
    rgm_pct: int = 36,
    cashback_pct: float = 3.0,
    objective: str = "net",
    min_coverage_pct: float = 0.0,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Coarse grid (₹50, T ∈ [200, 2500], spacing ≥ ₹100) search over n_tiers and
    reach. Returns top_k ladders by `objective` (net | roi | lift | adjusted_expected)
    that satisfy `min_coverage_pct`.
    """
    if objective not in {"net", "roi", "lift", "adjusted_expected"}:
        return {"error": f"objective must be one of net|roi|lift|adjusted_expected, got {objective}"}
    n_tiers_max = min(n_tiers_max, 4)  # bounded for latency
    n_tiers_min = max(n_tiers_min, 1)
    if reach_options is None:
        reach_options = [70, 80]

    grid = list(range(200, 2501, 50))
    spacing = 100
    vd = _vd()
    n_bills = _n_bills()
    pm = lambda pct: int(round(pct / 100.0 * n_bills / MONTHS_IN_WINDOW))

    found = []
    for n in range(n_tiers_min, n_tiers_max + 1):
        for reach_pct in reach_options:
            reach = reach_pct / 100.0
            for combo in itertools.combinations(grid, n):
                if n > 1 and not all(combo[i + 1] - combo[i] >= spacing for i in range(n - 1)):
                    continue
                z = zone_counts(vd, n_bills, list(combo), reach)
                lf = expected_lift(z, response_pct / 100.0, gap_aware)
                s = score_pair(z, lf, 1.0)
                cov = sum(lf["earned_per_tier"]) + sum(lf["free_per_tier"])
                if cov < min_coverage_pct:
                    continue
                gifts = [(cashback_pct / 100.0) * T for T in combo]
                mb = money_breakdown(lf, z, gifts, n_bills, MONTHS_IN_WINDOW,
                                     rgm_pct / 100.0, 1.0 + overshoot_pct / 100.0)
                metrics = {
                    "thresholds": list(combo),
                    "reach_pct": reach_pct,
                    "n_tiers": n,
                    "coverage_pct": round(cov, 2),
                    "lift_pct": round(lf["total"], 2),
                    "waste_pct": round(lf["free_total"], 2),
                    "adjusted_expected": round(s["adjusted_expected"], 2),
                    "revenue_per_month": round(mb["revenue"], 2),
                    "rgm_per_month": round(mb["rgm"], 2),
                    "burn_per_month": round(mb["burn"], 2),
                    "net_per_month": round(mb["net"], 2),
                    "roi_pct": (round(mb["net"] / mb["burn"] * 100, 2)
                                if mb["burn"] > 0 else None),
                    "coverage_bills_per_month": pm(cov),
                }
                if objective == "net":
                    score = mb["net"]
                elif objective == "roi":
                    score = (mb["net"] / mb["burn"]) if mb["burn"] > 0 else -1e18
                elif objective == "lift":
                    score = lf["total"]
                else:  # adjusted_expected
                    score = s["adjusted_expected"]
                found.append((score, metrics))

    if not found:
        return {"error": "no ladders satisfy the constraints; try lowering min_coverage_pct."}
    found.sort(key=lambda r: r[0], reverse=True)
    top = [m for _, m in found[:top_k]]
    return {
        "inputs": {
            "n_tiers_range": [n_tiers_min, n_tiers_max],
            "reach_options_pct": reach_options,
            "response_pct": response_pct,
            "gap_aware": gap_aware,
            "overshoot_pct": overshoot_pct,
            "rgm_pct": rgm_pct,
            "cashback_pct": cashback_pct,
            "objective": objective,
            "min_coverage_pct": min_coverage_pct,
        },
        "n_evaluated": len(found),
        "top": top,
    }


# ---------------------------------------------------------------------------
# Tool registry — schemas (OpenAI tool-call format) + dispatch
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_data_summary",
            "description": (
                "Frozen dataset facts: bill counts (90-day and per month), total revenue, "
                "mean/median/percentiles, and the date window. Call when user asks anything "
                "about the underlying data distribution."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dashboard_state",
            "description": (
                "The user's CURRENT dashboard sidebar settings (thresholds, reach, response, "
                "RGM, overshoot, gift costs) and all derived metrics (lift, waste, coverage, "
                "adjusted_expected, revenue/RGM/burn/net per month, ROI). Call when the user "
                "says 'my', 'current', 'this config', or similar."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_ladder",
            "description": (
                "Compute everything for an explicit ladder configuration: zones, lift, waste, "
                "coverage, revenue, RGM, burn, net, ROI, and per-tier breakdown. Pass either "
                "cashback_pct (uniform per-tier %) OR explicit gift_costs (₹ per tier)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thresholds": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 100, "maximum": 5000},
                        "description": "Sorted ladder of T values in ₹.",
                    },
                    "reach_pct": {"type": "integer", "minimum": 50, "maximum": 99, "default": 70},
                    "response_pct": {"type": "integer", "minimum": 0, "maximum": 100, "default": 50},
                    "gap_aware": {"type": "boolean", "default": True,
                                  "description": "If true, scale conversion by (1 − gap_ratio)."},
                    "overshoot_pct": {"type": "integer", "minimum": 0, "maximum": 100, "default": 30,
                                      "description": "Customer overshoot above the gap, %."},
                    "rgm_pct": {"type": "integer", "minimum": 1, "maximum": 100, "default": 36,
                                "description": "Gross margin %."},
                    "cashback_pct": {"type": "number",
                                     "description": "Per-tier cashback %, e.g. 3 for 3%. Either this or gift_costs."},
                    "gift_costs": {"type": "array", "items": {"type": "number"},
                                   "description": "Explicit ₹ gift per tier (alternative to cashback_pct)."},
                },
                "required": ["thresholds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_breakeven",
            "description": (
                "Binary-search the per-tier cashback rate (as % of each T) at which Net = ₹0 "
                "for the given ladder under the given response/RGM/overshoot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thresholds": {"type": "array", "items": {"type": "integer"}},
                    "reach_pct": {"type": "integer", "default": 70},
                    "response_pct": {"type": "integer", "default": 50},
                    "gap_aware": {"type": "boolean", "default": True},
                    "overshoot_pct": {"type": "integer", "default": 30},
                    "rgm_pct": {"type": "integer", "default": 36},
                },
                "required": ["thresholds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_ladders",
            "description": (
                "Coarse exhaustive search across (n_tiers, reach, ladder) at ₹50 grid. "
                "Returns top_k ladders ranked by `objective`. Bounded to n_tiers ≤ 4 for "
                "latency. Use when the user asks 'find me the best…' or 'what's optimal under…'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_tiers_min": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1},
                    "n_tiers_max": {"type": "integer", "minimum": 1, "maximum": 4, "default": 4},
                    "reach_options": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 50, "maximum": 99},
                        "default": [70, 80],
                    },
                    "response_pct": {"type": "integer", "default": 50},
                    "gap_aware": {"type": "boolean", "default": True},
                    "overshoot_pct": {"type": "integer", "default": 30},
                    "rgm_pct": {"type": "integer", "default": 36},
                    "cashback_pct": {"type": "number", "default": 3.0,
                                     "description": "Per-tier cashback as % of T."},
                    "objective": {
                        "type": "string",
                        "enum": ["net", "roi", "lift", "adjusted_expected"],
                        "default": "net",
                    },
                    "min_coverage_pct": {"type": "number", "default": 0.0},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
                "required": [],
            },
        },
    },
]

_DISPATCH = {
    "get_data_summary": get_data_summary,
    "get_dashboard_state": get_dashboard_state,
    "evaluate_ladder": evaluate_ladder,
    "find_breakeven": find_breakeven,
    "search_ladders": search_ladders,
}


def dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a tool call by name with arguments dict. Returns a JSON-serializable dict."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args) if args else fn()
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
