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

from data_loader import (
    load_percentiles,
    load_test_bills,
    load_topline,
    load_user_summary,
    load_user_topline,
    load_value_distribution,
)
from metrics import (
    breakeven_cashback,
    expected_lift,
    money_breakdown,
    score_pair,
    zone_counts,
)
from personalized_metrics import (
    DEFAULT_BASE_T,
    DEFAULT_CAP_BILLS,
    DEFAULT_CASHBACK,
    DEFAULT_MARGIN_PCT,
    DEFAULT_N_TIERS,
    DEFAULT_NUDGE_STEP,
    DEFAULT_OVERSHOOT_PCT,
    DEFAULT_PREDICTION_METHOD,
    DEFAULT_REACH_PCT,
    DEFAULT_REDEMPTION_PCT,
    DEFAULT_RESPONSE_PCT,
    DEFAULT_SEGMENT,
    score_config,
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
# Personalized Carrot tools (mirror personalized_page.py / personalized_metrics)
# ---------------------------------------------------------------------------

def _personalized_data():
    return load_user_summary(), load_test_bills_april(), load_user_topline()


def load_test_bills_april():
    df = load_test_bills()
    return df[df["window"] == "april"].copy()


def _personalized_current_settings_from_session() -> Dict[str, Any]:
    """Read the Personalized Carrot page's sidebar widgets from session_state.

    Falls back to module defaults if the user has never visited the page in
    this streamlit session.
    """
    ss = st.session_state
    # n_tiers radio (defaults to 3, same as page)
    n_tiers = int(ss.get("Number of tiers", DEFAULT_N_TIERS))
    base_T: List[int] = []
    cashback_per_tier: List[float] = []
    for i in range(n_tiers):
        base_T.append(int(ss.get(f"pT_{i + 1}", DEFAULT_BASE_T[i])))
        cashback_per_tier.append(float(ss.get(f"pG_{i + 1}", DEFAULT_CASHBACK[i])))

    # The page uses radio labels — translate back to internal segment key
    segment_label = ss.get("Include", "Warm + Light (≥1 bill)")
    segment = "warm_only" if segment_label.startswith("Warm only") else "warm_light"

    response_pct = int(ss.get("Default response rate (per bucket)",
                              ss.get("Response rate", DEFAULT_RESPONSE_PCT)))

    # Per-bucket response rates from session_state, falling back to the global
    # default. 5 buckets per tier (the 5 nudgeable gap categories).
    bucket_keys = ["le_50", "50_100", "100_200", "200_400", "gt_400"]
    bucket_response_rates: Dict[int, Dict[str, int]] = {}
    for t_idx in range(1, n_tiers + 1):
        bucket_response_rates[t_idx] = {
            bkey: int(ss.get(f"resp_t{t_idx}_b{bkey}", response_pct))
            for bkey in bucket_keys
        }

    return {
        "segment": segment,
        "cap_bills": int(ss.get("Cap bill_count at", DEFAULT_CAP_BILLS)),
        "prediction_method": ss.get("Per-user predicted bill", DEFAULT_PREDICTION_METHOD),
        "base_T": base_T,
        "cashback_per_tier": cashback_per_tier,
        "nudge_step": int(ss.get("Nudge step ₹", DEFAULT_NUDGE_STEP)),
        "reach_pct": int(ss.get("Reach window (T1 only)", DEFAULT_REACH_PCT)),
        "response_pct": response_pct,
        "redemption_pct": int(ss.get("Redemption rate", DEFAULT_REDEMPTION_PCT)),
        "overshoot_pct": int(ss.get("Revenue overshoot κ", DEFAULT_OVERSHOOT_PCT)),
        "margin_pct": int(ss.get("Gross margin", DEFAULT_MARGIN_PCT)),
        "bucket_response_rates": bucket_response_rates,
    }


def get_personalized_state() -> Dict[str, Any]:
    """The user's CURRENT Personalized Carrot config + April backtest metrics.

    Use this when the user asks about 'my', 'current', 'this config' on the
    Personalized Carrot page. Mirrors get_dashboard_state() for the Dashboard.
    """
    users, april, top = _personalized_data()
    settings = _personalized_current_settings_from_session()
    result = score_config(users, april, **settings)
    result["data_window"] = {
        "training_cutoff": top["training_cutoff"],
        "test_window_april": top["test_window_april"],
        "n_test_bills_april_total": top["n_test_bills_april"],
        "n_excluded_new_patient_bills": top["n_excluded_new_patient"],
    }
    return result


def evaluate_personalized_config(
    prediction_method: str = None,
    base_T: List[int] = None,
    cashback_per_tier: List[float] = None,
    nudge_step: int = None,
    reach_pct: int = None,
    response_pct: int = None,
    redemption_pct: int = None,
    overshoot_pct: int = None,
    margin_pct: int = None,
    segment: str = None,
    cap_bills: int = None,
    bucket_response_rates: Dict[int, Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score a hypothetical config without touching the user's sidebar.

    Any argument omitted falls back to the CURRENT session state (or page
    default if the user hasn't visited the page). Use this for 'what if'
    explorations like 'what if I switched to P95 and raised T1 to 300?'.

    `bucket_response_rates` overrides the per-tier per-bucket conversion rates.
    Shape: {tier_idx: {bucket_key: pct}} where bucket_key is one of
    le_50 / 50_100 / 100_200 / 200_400 / gt_400. Missing entries fall back
    to the current sidebar value for that bucket.
    """
    users, april, _ = _personalized_data()
    current = _personalized_current_settings_from_session()

    # Merge per-bucket override into current state's per-bucket rates
    merged_bucket_rates = dict(current.get("bucket_response_rates", {}))
    if bucket_response_rates:
        for tier_idx, bucket_dict in bucket_response_rates.items():
            tier_idx = int(tier_idx)
            base = dict(merged_bucket_rates.get(tier_idx, {}))
            for bkey, rate in bucket_dict.items():
                base[bkey] = float(rate)
            merged_bucket_rates[tier_idx] = base

    settings = {
        "prediction_method": prediction_method or current["prediction_method"],
        "base_T": list(base_T) if base_T is not None else current["base_T"],
        "cashback_per_tier": list(cashback_per_tier) if cashback_per_tier is not None else current["cashback_per_tier"],
        "nudge_step": nudge_step if nudge_step is not None else current["nudge_step"],
        "reach_pct": reach_pct if reach_pct is not None else current["reach_pct"],
        "response_pct": response_pct if response_pct is not None else current["response_pct"],
        "redemption_pct": redemption_pct if redemption_pct is not None else current["redemption_pct"],
        "overshoot_pct": overshoot_pct if overshoot_pct is not None else current["overshoot_pct"],
        "margin_pct": margin_pct if margin_pct is not None else current["margin_pct"],
        "segment": segment or current["segment"],
        "cap_bills": cap_bills if cap_bills is not None else current["cap_bills"],
        "bucket_response_rates": merged_bucket_rates,
    }
    # If base_T changed but cashback list wasn't passed, repad cashback list to match length
    if len(settings["cashback_per_tier"]) != len(settings["base_T"]):
        K = len(settings["base_T"])
        settings["cashback_per_tier"] = list(DEFAULT_CASHBACK[:K])
    return score_config(users, april, **settings)


# Search grids for find_best_personalized_config — kept small for latency
_SEARCH_GRIDS = {
    "prediction_method": ["Average", "P65", "P70", "P75", "P80", "P90", "P95", "Max"],
    "nudge_step": [0, 25, 50, 75, 100, 150, 200],
    "base_t1": [150, 200, 250, 300, 350],
    "base_t2": [400, 450, 550, 650, 750],
    "base_t3": [600, 700, 850, 1000, 1200],
    "cashback_pct_per_tier": [0.02, 0.04, 0.06, 0.08, 0.10],
    "reach_pct_t1": [50, 60, 70, 80, 90],
    "response_pct": [20, 30, 40, 50, 60],
    "redemption_pct": [40, 60, 70, 80],
    "overshoot_pct": [10, 20, 30, 40, 50],
}
_GRID_HARD_CAP = 200
_VALID_OBJECTIVES = {"max_net", "max_rgm", "max_cashback_rate", "max_revenue"}


def find_best_personalized_config(
    objective: str = "max_net",
    constraints: Dict[str, float] = None,
    search_params: List[str] = None,
    fixed: Dict[str, Any] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Grid search over the personalized config space; return top-k configs.

    objective: "max_net" / "max_rgm" / "max_cashback_rate" / "max_revenue".
    constraints (optional): filters like {"cashback_rate_min": 50, "net_min": 0,
        "bills_min": 100000}.
    search_params: which dimensions to vary. Default ["prediction_method", "nudge_step"].
        Supported: prediction_method, nudge_step, base_t1, base_t2, base_t3,
        cashback_pct_per_tier, reach_pct_t1, response_pct, redemption_pct,
        overshoot_pct.
    fixed (optional): lock other settings to these values.
    top_k: max configs to return (default 5).

    Always uses the CURRENT 3-tier ladder structure (base_t1/2/3); the
    cashback_pct_per_tier knob scales all tiers' cashback together as a % of T.
    Refuses to run if cartesian grid > 200 evaluations.
    """
    if objective not in _VALID_OBJECTIVES:
        return {"error": f"objective must be one of {sorted(_VALID_OBJECTIVES)}, got {objective!r}"}
    constraints = constraints or {}
    search_params = search_params or ["prediction_method", "nudge_step"]
    fixed = fixed or {}

    for p in search_params:
        if p not in _SEARCH_GRIDS:
            return {"error": f"unknown search_params entry {p!r}. "
                             f"Supported: {sorted(_SEARCH_GRIDS)}"}

    # Compute grid size
    sizes = [len(_SEARCH_GRIDS[p]) for p in search_params]
    grid_size = 1
    for s in sizes:
        grid_size *= s
    if grid_size > _GRID_HARD_CAP:
        return {"error": f"grid_size {grid_size} > hard cap {_GRID_HARD_CAP}; "
                         f"narrow search_params (currently {search_params})."}

    # Build cartesian product
    axis_values = [_SEARCH_GRIDS[p] for p in search_params]
    combos = list(itertools.product(*axis_values))

    current = _personalized_current_settings_from_session()

    def _settings_for(combo: tuple) -> Dict[str, Any]:
        """Build settings dict from current + fixed + this combo."""
        out = {
            "prediction_method": current["prediction_method"],
            "base_T": list(current["base_T"]),
            "cashback_per_tier": list(current["cashback_per_tier"]),
            "nudge_step": current["nudge_step"],
            "reach_pct": current["reach_pct"],
            "response_pct": current["response_pct"],
            "redemption_pct": current["redemption_pct"],
            "overshoot_pct": current["overshoot_pct"],
            "margin_pct": current["margin_pct"],
            "segment": current["segment"],
            "cap_bills": current["cap_bills"],
            "bucket_response_rates": current.get("bucket_response_rates"),
        }
        # Apply fixed overrides first
        for k, v in fixed.items():
            if k == "cashback_pct_per_tier":
                out["cashback_per_tier"] = [round(v * t, 2) for t in out["base_T"]]
            elif k == "reach_pct_t1":
                out["reach_pct"] = int(v)
            elif k in ("base_t1", "base_t2", "base_t3"):
                idx = int(k[-1]) - 1
                if idx < len(out["base_T"]):
                    out["base_T"][idx] = int(v)
            else:
                out[k] = v
        # Then apply combo
        for p, v in zip(search_params, combo):
            if p == "cashback_pct_per_tier":
                out["cashback_per_tier"] = [round(v * t, 2) for t in out["base_T"]]
            elif p == "reach_pct_t1":
                out["reach_pct"] = int(v)
            elif p in ("base_t1", "base_t2", "base_t3"):
                idx = int(p[-1]) - 1
                if idx < len(out["base_T"]):
                    out["base_T"][idx] = int(v)
            else:
                out[p] = v
        # Validation: tier values must be strictly ascending
        bT = out["base_T"]
        if any(bT[i + 1] <= bT[i] for i in range(len(bT) - 1)):
            return None
        # Re-pad cashback list if length doesn't match
        if len(out["cashback_per_tier"]) != len(out["base_T"]):
            out["cashback_per_tier"] = list(DEFAULT_CASHBACK[:len(out["base_T"])])
        return out

    users, april, _ = _personalized_data()
    objective_key_map = {
        "max_net": "net",
        "max_rgm": "rgm",
        "max_cashback_rate": "cashback_rate_pct",
        "max_revenue": "revenue",
    }
    obj_key = objective_key_map[objective]

    rows: List[Dict[str, Any]] = []
    invalid_combo = 0
    for combo in combos:
        settings = _settings_for(combo)
        if settings is None:
            invalid_combo += 1
            continue
        r = score_config(users, april, **settings)
        if "error" in r:
            continue
        totals = r["totals"]
        # Apply constraints
        ok = True
        cr = totals["cashback_rate_pct"]
        net = totals["net"]
        bills = totals["bills_scored"]
        if "cashback_rate_min" in constraints and cr < constraints["cashback_rate_min"]:
            ok = False
        if "cashback_rate_max" in constraints and cr > constraints["cashback_rate_max"]:
            ok = False
        if "net_min" in constraints and net < constraints["net_min"]:
            ok = False
        if "bills_min" in constraints and bills < constraints["bills_min"]:
            ok = False
        if not ok:
            continue
        rows.append({
            "score": totals[obj_key],
            "settings": r["settings"],
            "totals": totals,
            "per_tier": r["per_tier"],
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    top = []
    for i, row in enumerate(rows[:top_k]):
        top.append({
            "rank": i + 1,
            "settings": row["settings"],
            "totals": row["totals"],
            "per_tier": row["per_tier"],
        })

    return {
        "objective": objective,
        "constraints": constraints,
        "search_params": search_params,
        "fixed": fixed,
        "grid_size": grid_size,
        "evaluated": grid_size - invalid_combo,
        "passing_constraints": len(rows),
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

TOOL_SCHEMAS.extend([
    {
        "type": "function",
        "function": {
            "name": "get_personalized_state",
            "description": (
                "PERSONALIZED CARROT page — the user's current sidebar config "
                "(segment, prediction method, base tiers, cashback per tier, nudge step, "
                "reach %, response %, redemption %, overshoot, margin) + April 2026 "
                "backtest metrics (Net, Revenue, RGM, Burn, Cashback rate, per-tier "
                "breakdown of Auto-qualified / In-reach / Unreachable / Nudged). "
                "Call when the user asks 'my', 'current', 'this config' about the "
                "Personalized Carrot model (one-tier-per-user). Do NOT confuse with "
                "get_dashboard_state which is for the separate Dashboard ladder."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_personalized_config",
            "description": (
                "PERSONALIZED CARROT — score a hypothetical config without changing "
                "the user's sidebar. Any argument omitted falls back to the user's "
                "CURRENT sidebar value. Use for 'what if' questions like 'what if I "
                "switched from P80 to P95?' or 'what if I raised T1 to 300 and nudge "
                "to 100?'. Returns the same shape as get_personalized_state. "
                "Don't sweep manually with many calls — use find_best_personalized_config "
                "when the user asks for 'the best' under conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction_method": {
                        "type": "string",
                        "enum": ["Average", "P65", "P70", "P75", "P80", "P90", "P95", "Max"],
                        "description": "Which per-user predicted-bill estimator to use.",
                    },
                    "base_T": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 50, "maximum": 5000},
                        "description": "Sorted base tier values, e.g. [250, 450, 700]. Length = number of tiers.",
                    },
                    "cashback_per_tier": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "₹ cashback per tier (same length as base_T).",
                    },
                    "nudge_step": {"type": "integer", "minimum": 0, "maximum": 300,
                                   "description": "Slot increment between tiers in ₹."},
                    "reach_pct": {"type": "integer", "minimum": 50, "maximum": 95,
                                  "description": "Reach window % — applies only to T1."},
                    "response_pct": {"type": "integer", "minimum": 0, "maximum": 100,
                                     "description": "% of in-reach customers who top up."},
                    "redemption_pct": {"type": "integer", "minimum": 0, "maximum": 100,
                                       "description": "% of issued cashback that gets redeemed."},
                    "overshoot_pct": {"type": "integer", "minimum": 0, "maximum": 100,
                                      "description": "κ — extra % spend above gap on conversion."},
                    "margin_pct": {"type": "integer", "minimum": 1, "maximum": 100,
                                   "description": "Gross margin %."},
                    "segment": {"type": "string", "enum": ["warm_only", "warm_light"],
                                "description": "Filter: only warm users (≥3 bills) or warm+light (≥1)."},
                    "cap_bills": {"type": "integer", "minimum": 10, "maximum": 10000,
                                  "description": "Drop synthetic walk-in accounts above this bill count."},
                    "bucket_response_rates": {
                        "type": "object",
                        "description": (
                            "Per-tier per-bucket conversion overrides. Shape: "
                            "{tier_idx (1-based): {bucket_key: pct}} where bucket_key is one of "
                            "'le_50', '50_100', '100_200', '200_400', 'gt_400'. "
                            "Any missing tier/bucket falls back to the user's current sidebar value. "
                            "Use to explore 'what if T2 le_50 converts at 40% but T2 gt_400 at 1%?'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_best_personalized_config",
            "description": (
                "PERSONALIZED CARROT — grid-search optimiser. Returns the top-k "
                "configs ranked by the chosen objective, optionally filtered by "
                "constraints. Use when the user asks 'find the best config that…' "
                "or 'optimal X under condition Y'. Don't manually loop with "
                "evaluate_personalized_config. Cartesian grid size is capped at 200; "
                "if you need more dimensions than fit, run two narrower searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "enum": ["max_net", "max_rgm", "max_cashback_rate", "max_revenue"],
                        "default": "max_net",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Optional filters: cashback_rate_min, cashback_rate_max, net_min, bills_min.",
                        "properties": {
                            "cashback_rate_min": {"type": "number"},
                            "cashback_rate_max": {"type": "number"},
                            "net_min": {"type": "number"},
                            "bills_min": {"type": "number"},
                        },
                    },
                    "search_params": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["prediction_method", "nudge_step", "base_t1", "base_t2",
                                     "base_t3", "cashback_pct_per_tier", "reach_pct_t1",
                                     "response_pct", "redemption_pct", "overshoot_pct"],
                        },
                        "default": ["prediction_method", "nudge_step"],
                        "description": "Dimensions to vary. Internal grids are small; cartesian-cap is 200.",
                    },
                    "fixed": {
                        "type": "object",
                        "description": "Lock settings to specific values (same keys as search_params, plus base_T as a full list).",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": [],
            },
        },
    },
])

_DISPATCH = {
    "get_data_summary": get_data_summary,
    "get_dashboard_state": get_dashboard_state,
    "evaluate_ladder": evaluate_ladder,
    "find_breakeven": find_breakeven,
    "search_ladders": search_ladders,
    "get_personalized_state": get_personalized_state,
    "evaluate_personalized_config": evaluate_personalized_config,
    "find_best_personalized_config": find_best_personalized_config,
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
