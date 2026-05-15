"""Personalized Carrot — pure scoring functions (no streamlit deps).

Used by both `personalized_page.py` (UI) and `bot_tools.py` (chatbot tools).
The page passes its sidebar values; the bot tools pass either current
session-state values or hypothetical overrides — but the math itself is
identical and lives only here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (mirror the streamlit page; used when widgets haven't been touched)
# ---------------------------------------------------------------------------
DEFAULT_BASE_T = [250, 450, 700, 1100, 1500]
DEFAULT_CASHBACK = [10, 20, 40, 80, 120]
DEFAULT_N_TIERS = 3
DEFAULT_NUDGE_STEP = 50
DEFAULT_REACH_PCT = 75
DEFAULT_RESPONSE_PCT = 40
DEFAULT_REDEMPTION_PCT = 70
DEFAULT_OVERSHOOT_PCT = 30
DEFAULT_MARGIN_PCT = 36
DEFAULT_PREDICTION_METHOD = "P80"
DEFAULT_SEGMENT = "warm_light"  # or "warm_only"
DEFAULT_CAP_BILLS = 500

PRED_METHOD_TO_COL = {
    "Average": "pred_avg",
    "P80": "pred_p80",
    "P90": "pred_p90",
    "P95": "pred_p95",
    "Max": "pred_max",
}


def build_slots(base_T: List[int], nudge: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted slot array + parallel base-tier-index array.

    Slot at index i represents an assignable per-user threshold; slot_base[i]
    is the base-tier index whose cashback rate applies.

    For tier i < K−1: slots = {T_i, T_i+N, T_i+2N, ...} stopping strictly before T_{i+1}.
    For top tier K: a single slot = T_K (no upward extension).
    """
    slots: list[float] = []
    slot_base: list[int] = []
    K = len(base_T)
    for i in range(K):
        if i == K - 1:
            slots.append(float(base_T[i]))
            slot_base.append(i)
            continue
        upper = float(base_T[i + 1])
        v = float(base_T[i])
        if nudge <= 0:
            slots.append(v)
            slot_base.append(i)
            continue
        while v < upper:
            slots.append(v)
            slot_base.append(i)
            v += nudge
    return np.asarray(slots, dtype=float), np.asarray(slot_base, dtype=int)


def assign_threshold(pred: np.ndarray, base_T: List[int],
                     nudge: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-user threshold + base-tier index. base_idx == -1 means excluded
    (predicted bill ≥ top base tier — high spender, separate program)."""
    slots, slot_base = build_slots(base_T, nudge)
    top_T = float(base_T[-1])
    idx = np.searchsorted(slots, pred, side="right")
    in_range = (idx < len(slots)) & (pred < top_T)
    safe = np.clip(idx, 0, len(slots) - 1)
    threshold = np.where(in_range, slots[safe], np.nan)
    base_idx = np.where(in_range, slot_base[safe], -1)
    return threshold.astype(float), base_idx.astype(int)


def pred_column(method: str) -> str:
    return PRED_METHOD_TO_COL.get(method, "pred_p80")


def _settings_echo(segment, cap_bills, pred_method, base_T, cashback_per_tier,
                   nudge_step, reach_pct, response_pct, redemption_pct,
                   overshoot_pct, margin_pct) -> Dict[str, Any]:
    return {
        "segment": segment,
        "cap_bills": int(cap_bills),
        "prediction_method": pred_method,
        "base_tiers": [int(t) for t in base_T],
        "cashback_per_tier": [float(c) for c in cashback_per_tier],
        "nudge_step": int(nudge_step),
        "reach_pct_t1_only": int(reach_pct),
        "response_pct": int(response_pct),
        "redemption_pct": int(redemption_pct),
        "overshoot_pct": int(overshoot_pct),
        "margin_pct": int(margin_pct),
    }


def score_config(
    users_df: pd.DataFrame,
    april_bills_df: pd.DataFrame,
    *,
    segment: str = DEFAULT_SEGMENT,
    cap_bills: int = DEFAULT_CAP_BILLS,
    prediction_method: str = DEFAULT_PREDICTION_METHOD,
    base_T: List[int] = None,
    cashback_per_tier: List[float] = None,
    nudge_step: int = DEFAULT_NUDGE_STEP,
    reach_pct: int = DEFAULT_REACH_PCT,
    response_pct: int = DEFAULT_RESPONSE_PCT,
    redemption_pct: int = DEFAULT_REDEMPTION_PCT,
    overshoot_pct: int = DEFAULT_OVERSHOOT_PCT,
    margin_pct: int = DEFAULT_MARGIN_PCT,
) -> Dict[str, Any]:
    """Full pipeline: filter users → pick prediction → assign threshold →
    score actual April bills → aggregate. Returns a JSON-friendly dict.

    Implements the asymmetric reach window: T1 (base_idx==0) keeps the
    reach-% cutoff; T2+ has no unreachable bucket.
    """
    if base_T is None:
        base_T = list(DEFAULT_BASE_T[:DEFAULT_N_TIERS])
    base_T = [int(t) for t in base_T]
    K = len(base_T)
    if cashback_per_tier is None:
        cashback_per_tier = list(DEFAULT_CASHBACK[:K])
    cashback_per_tier = [float(c) for c in cashback_per_tier]
    if len(cashback_per_tier) != K:
        return {"error": f"cashback_per_tier length {len(cashback_per_tier)} != base_T length {K}"}

    reach = reach_pct / 100.0
    response_rate = response_pct / 100.0
    redemption_rate = redemption_pct / 100.0
    kappa = 1.0 + overshoot_pct / 100.0
    rgm_rate = margin_pct / 100.0

    # 1. Filter users
    u = users_df[users_df["bill_count"] <= cap_bills].copy()
    if segment == "warm_only":
        u = u[u["segment"] == "warm"]

    # 2. Pick prediction column
    col = pred_column(prediction_method)
    if col not in u.columns:
        return {"error": f"prediction column '{col}' not in user_summary"}
    u["pred"] = u[col]
    u = u[u["pred"] > 0].reset_index(drop=True)

    # 3. Assign threshold
    th, bi = assign_threshold(u["pred"].to_numpy(dtype=float), base_T, nudge_step)
    u["threshold"] = th
    u["base_idx"] = bi
    u["cashback"] = np.where(bi >= 0, np.take(cashback_per_tier, np.clip(bi, 0, K - 1)), np.nan)

    n_excluded = int((u["base_idx"] == -1).sum())
    users_assigned = u[u["base_idx"] >= 0]
    users_per_tier = [int((users_assigned["base_idx"] == k).sum()) for k in range(K)]

    # 4. Merge with April bills
    bs = april_bills_df.merge(
        users_assigned[["patient_id", "threshold", "cashback", "base_idx"]],
        on="patient_id", how="inner",
    )
    n = len(bs)
    settings = _settings_echo(segment, cap_bills, prediction_method, base_T,
                              cashback_per_tier, nudge_step, reach_pct,
                              response_pct, redemption_pct, overshoot_pct, margin_pct)
    if n == 0:
        empty_per_tier = [{
            "tier": k + 1, "base": base_T[k], "cashback": cashback_per_tier[k],
            "users_assigned": users_per_tier[k], "bills": 0,
            "unreachable": 0, "in_reach": 0, "auto_qualified": 0,
            "nudged_expected": 0.0, "revenue": 0.0, "rgm": 0.0,
            "burn": 0.0, "net": 0.0,
        } for k in range(K)]
        return {
            "settings": settings,
            "totals": {"bills_scored": 0, "users_assigned": int(len(users_assigned)),
                       "users_excluded_high_spender": n_excluded,
                       "cashback_rate_pct": 0.0, "revenue": 0.0, "rgm": 0.0,
                       "burn": 0.0, "net": 0.0,
                       "n_unreach": 0, "n_inreach": 0, "n_auto": 0,
                       "nudged_total": 0.0},
            "per_tier": empty_per_tier,
        }

    B = bs["bill_value_net"].to_numpy(dtype=float)
    V = bs["threshold"].to_numpy(dtype=float)
    BI = bs["base_idx"].to_numpy(dtype=int)

    # 5. Score bills (asymmetric reach: T1 only)
    auto = B >= V
    is_t1 = BI == 0
    unreach = is_t1 & (B < reach * V)
    inreach = (~auto) & (~unreach)
    gap = V - B
    gf = np.where(inreach, 1.0, 0.0)  # flat (gap-aware removed)
    pc = response_rate * gf

    n_unreach = int(unreach.sum())
    n_inreach = int(inreach.sum())
    n_auto = int(auto.sum())

    per_tier = []
    total_revenue = 0.0
    total_burn = 0.0
    total_nudged = 0.0

    for k in range(K):
        mk = BI == k
        bills_k = int(mk.sum())
        if bills_k == 0:
            per_tier.append({
                "tier": k + 1, "base": base_T[k], "cashback": cashback_per_tier[k],
                "users_assigned": users_per_tier[k], "bills": 0,
                "unreachable": 0, "in_reach": 0, "auto_qualified": 0,
                "nudged_expected": 0.0, "revenue": 0.0, "rgm": 0.0,
                "burn": 0.0, "net": 0.0,
            })
            continue
        n_unr_k = int(unreach[mk].sum())
        n_ir_k = int(inreach[mk].sum())
        n_aut_k = int(auto[mk].sum())
        m_nudged = mk & inreach
        nudged_k = float(pc[m_nudged].sum())
        rev_k = float((pc[m_nudged] * gap[m_nudged] * kappa).sum())
        burn_k = (n_aut_k + nudged_k) * cashback_per_tier[k] * redemption_rate
        rgm_k = rev_k * rgm_rate

        per_tier.append({
            "tier": k + 1,
            "base": base_T[k],
            "cashback": cashback_per_tier[k],
            "users_assigned": users_per_tier[k],
            "bills": bills_k,
            "unreachable": n_unr_k,
            "in_reach": n_ir_k,
            "auto_qualified": n_aut_k,
            "nudged_expected": round(nudged_k, 2),
            "revenue": round(rev_k, 2),
            "rgm": round(rgm_k, 2),
            "burn": round(burn_k, 2),
            "net": round(rgm_k - burn_k, 2),
        })
        total_revenue += rev_k
        total_burn += burn_k
        total_nudged += nudged_k

    total_rgm = total_revenue * rgm_rate
    cashback_rate = (total_nudged + n_auto) / n * 100.0

    return {
        "settings": settings,
        "totals": {
            "bills_scored": n,
            "users_assigned": int(len(users_assigned)),
            "users_excluded_high_spender": n_excluded,
            "cashback_rate_pct": round(cashback_rate, 2),
            "revenue": round(total_revenue, 2),
            "rgm": round(total_rgm, 2),
            "burn": round(total_burn, 2),
            "net": round(total_rgm - total_burn, 2),
            "n_unreach": n_unreach,
            "n_inreach": n_inreach,
            "n_auto": n_auto,
            "nudged_total": round(total_nudged, 2),
        },
        "per_tier": per_tier,
    }
