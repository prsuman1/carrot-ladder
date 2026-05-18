"""Ladder optimisation for the Personalized Carrot model.

Standalone module (no streamlit imports) used by:
  - bot_tools.design_personalized_ladder — exposes the optimiser to the chatbot
  - prediction_lab/optimize_ladder.py — standalone sandbox script

Math matches personalized_metrics.score_config exactly: asymmetric reach
(T1 only), per-bucket conversion rates, single-tier-per-user assignment.

The speed trick: precompute per-bill `(pred, actual_bill)` numpy arrays once,
then every ladder evaluation is pure numpy (no pandas merge). ~30 ms per call.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (mirror personalized_metrics)
# ---------------------------------------------------------------------------
BUCKET_KEYS_5 = ["le_50", "50_100", "100_200", "200_400", "gt_400"]
BUCKET_EDGES = np.array([50.0, 100.0, 200.0, 400.0])
DEFAULT_BUCKET_RATES_PCT = {  # user's standard rates (also the fall-back)
    "le_50": 50.0, "50_100": 30.0, "100_200": 15.0, "200_400": 0.0, "gt_400": 0.0,
}

PRED_METHOD_TO_COL = {
    "Average": "pred_avg", "P65": "pred_p65", "P70": "pred_p70",
    "P75": "pred_p75", "P80": "pred_p80", "P90": "pred_p90",
    "P95": "pred_p95", "Max": "pred_max",
}

TIER_GRID = list(range(50, 2001, 25))   # ₹50..₹2000 in ₹25 steps
MIN_GAP = 100                            # adjacent tiers must differ ≥ ₹100
PERTURB_DELTAS = [-800, -400, -200, -100, -75, -50, -25,
                  25, 50, 75, 100, 200, 400, 800]


# ---------------------------------------------------------------------------
# Slot building + threshold assignment
# ---------------------------------------------------------------------------
def build_slots(base_T: List[int], nudge: int) -> Tuple[np.ndarray, np.ndarray]:
    slots: list[float] = []
    slot_base: list[int] = []
    K = len(base_T)
    for i in range(K):
        if i == K - 1:
            slots.append(float(base_T[i])); slot_base.append(i); continue
        upper = float(base_T[i + 1])
        v = float(base_T[i])
        if nudge <= 0:
            slots.append(v); slot_base.append(i); continue
        while v < upper:
            slots.append(v); slot_base.append(i); v += nudge
    return np.asarray(slots, float), np.asarray(slot_base, int)


def assign_threshold(pred: np.ndarray, base_T: List[int], nudge: int):
    slots, slot_base = build_slots(base_T, nudge)
    top_T = float(base_T[-1])
    idx = np.searchsorted(slots, pred, side="right")
    in_range = (idx < len(slots)) & (pred < top_T)
    safe = np.clip(idx, 0, len(slots) - 1)
    threshold = np.where(in_range, slots[safe], np.nan)
    base_idx = np.where(in_range, slot_base[safe], -1)
    return threshold, base_idx


# ---------------------------------------------------------------------------
# Fast scorer — vectorised, no pandas in inner loop
# ---------------------------------------------------------------------------
def score_ladder_fast(
    pred: np.ndarray,
    B: np.ndarray,
    ladder: List[int],
    bucket_rates: Dict[int, Dict[str, float]],
    cashback_pct: float,
    nudge: int,
    reach_pct: int,
    redemption_pct: int,
    overshoot_pct: int,
    margin_pct: int,
) -> Dict[str, float]:
    """Score actual April bills `B` against thresholds derived from `pred` +
    `ladder`. Returns the standard metrics dict."""
    K = len(ladder)
    V, BI = assign_threshold(pred, ladder, nudge)
    valid = BI >= 0
    if not valid.any():
        return {"net": 0.0, "cashback_rate_pct": 0.0, "revenue": 0.0,
                "rgm": 0.0, "burn": 0.0, "n": 0}
    Vv = V[valid]; Bv = B[valid]; BIv = BI[valid]

    cashback = np.array([cashback_pct / 100.0 * t for t in ladder])
    reach = reach_pct / 100.0
    redemption = redemption_pct / 100.0
    overshoot = 1.0 + overshoot_pct / 100.0
    margin = margin_pct / 100.0

    auto = Bv >= Vv
    is_t1 = BIv == 0
    unreach = is_t1 & (Bv < reach * Vv)
    inreach = (~auto) & (~unreach)
    gap = Vv - Bv

    bucket_idx = np.clip(np.searchsorted(BUCKET_EDGES, gap, side="left"), 0, 4)

    # Build K × 5 rate matrix from bucket_rates (per-tier)
    rate_matrix = np.zeros((K, 5))
    for tk in range(K):
        tier_rates = bucket_rates.get(tk + 1, DEFAULT_BUCKET_RATES_PCT)
        for bi, bkey in enumerate(BUCKET_KEYS_5):
            rate_matrix[tk, bi] = tier_rates.get(bkey, DEFAULT_BUCKET_RATES_PCT[bkey]) / 100.0
    rate_per_bill = rate_matrix[BIv, bucket_idx]
    pc = np.where(inreach, rate_per_bill, 0.0)

    n_auto_pt = np.zeros(K)
    nudged_pt = np.zeros(K)
    revenue_pt = np.zeros(K)
    for k in range(K):
        mk = BIv == k
        if not mk.any():
            continue
        n_auto_pt[k] = auto[mk].sum()
        m_nud = mk & inreach
        nudged_pt[k] = pc[m_nud].sum()
        revenue_pt[k] = (pc[m_nud] * gap[m_nud] * overshoot).sum()

    revenue = float(revenue_pt.sum())
    rgm = revenue * margin
    burn = float(((n_auto_pt + nudged_pt) * cashback * redemption).sum())
    net = rgm - burn
    n = len(Bv)
    cashback_rate = (nudged_pt.sum() + n_auto_pt.sum()) / n * 100.0 if n > 0 else 0.0

    return {"net": float(net), "cashback_rate_pct": float(cashback_rate),
            "revenue": revenue, "rgm": float(rgm), "burn": burn, "n": int(n)}


# ---------------------------------------------------------------------------
# Hill-climb utilities
# ---------------------------------------------------------------------------
def random_ladder(N: int, rng: np.random.Generator) -> List[int]:
    while True:
        picks = sorted(rng.choice(TIER_GRID, size=N, replace=False).tolist())
        if all(picks[i + 1] - picks[i] >= MIN_GAP for i in range(N - 1)):
            return picks


def smart_starts(N: int) -> List[List[int]]:
    """A handful of intentional starting points spanning the space.
    Helps hill climbing escape weak basins from purely random starts."""
    if N == 1:
        return [[50], [200], [500], [1000]]
    if N == 2:
        return [[50, 200], [50, 1000], [200, 500], [500, 1500]]
    # N >= 3: spread starting points
    starts = []
    # All-low (catches bulk distribution)
    starts.append([50 + i * 100 for i in range(N)])
    # Two-low-plus-spread (the winning shape we discovered manually)
    if N >= 4:
        starts.append([50, 150] + [int(1500 - (N - 3 - j) * 100) for j in range(N - 2)])
    if N == 3:
        starts.append([50, 150, 1500])
    # Evenly spread
    step = (1500 - 50) // (N - 1)
    starts.append([50 + step * i for i in range(N)])
    # High-end clustered
    starts.append([200, 400, 600] + [int(1200 + j * 100) for j in range(N - 3)] if N > 3
                  else [200, 500, 1000][:N])
    # Filter to valid
    return [s for s in starts if _valid(s)]


def _valid(ladder: List[int]) -> bool:
    if any(t not in TIER_GRID for t in ladder):
        return False
    return all(ladder[i + 1] - ladder[i] >= MIN_GAP for i in range(len(ladder) - 1))


def _neighbours(ladder: List[int]):
    for i, t in enumerate(ladder):
        for d in PERTURB_DELTAS:
            new_t = t + d
            if new_t < TIER_GRID[0] or new_t > TIER_GRID[-1]:
                continue
            cand = list(ladder)
            cand[i] = new_t
            cand_sorted = sorted(cand)
            if cand_sorted == ladder:
                continue
            if _valid(cand_sorted):
                yield cand_sorted


def _hill_climb(score_fn, start: List[int], max_iter: int, eval_budget: List[int]):
    """Greedy hill climb. `eval_budget` is a single-element list used as a
    mutable counter so the optimiser can stop early when total evals exceed cap."""
    cur = list(start)
    cur_s = score_fn(cur)
    eval_budget[0] += 1
    for _ in range(max_iter):
        if eval_budget[0] >= eval_budget[1]:
            break
        improved = False
        best_n = cur
        best_s = cur_s
        for n in _neighbours(cur):
            if eval_budget[0] >= eval_budget[1]:
                break
            s = score_fn(n)
            eval_budget[0] += 1
            if s["net"] > best_s["net"]:
                best_n = n
                best_s = s
                improved = True
        if not improved:
            break
        cur = best_n
        cur_s = best_s
    return cur, cur_s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _resolve_objective(name: str) -> str:
    name = name.lower()
    table = {"max_net": "net", "max_rgm": "rgm",
             "max_cashback_rate": "cashback_rate_pct", "max_revenue": "revenue"}
    if name not in table:
        raise ValueError(f"objective must be one of {list(table.keys())}, got {name!r}")
    return table[name]


def _passes_constraints(metrics: Dict[str, float], constraints: Dict[str, float]) -> bool:
    if not constraints:
        return True
    if "cashback_rate_min" in constraints and metrics["cashback_rate_pct"] < constraints["cashback_rate_min"]:
        return False
    if "cashback_rate_max" in constraints and metrics["cashback_rate_pct"] > constraints["cashback_rate_max"]:
        return False
    if "net_min" in constraints and metrics["net"] < constraints["net_min"]:
        return False
    if "bills_min" in constraints and metrics["n"] < constraints["bills_min"]:
        return False
    return True


def design_ladder(
    users_df: pd.DataFrame,
    april_df: pd.DataFrame,
    *,
    objective: str = "max_net",
    bucket_response_rates: Dict[int, Dict[str, float]] | None = None,
    cashback_pct_of_tier: float = 3.0,
    prediction_method: str = "P80",
    n_tier_range: Tuple[int, int] = (3, 5),
    constraints: Dict[str, float] | None = None,
    nudge_step: int = 50,
    reach_pct: int = 75,
    redemption_pct: int = 70,
    overshoot_pct: int = 30,
    margin_pct: int = 36,
    n_restarts: int = 2,
    max_iter: int = 20,
    max_evals: int = 1500,
    top_k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Search the ladder space (N tiers + values) for the config that
    maximises `objective`. Random-restart hill climbing, vectorised scorer.

    Returns:
        {
          "best": {"ladder", "totals"},
          "top_k": [...],
          "pareto": [...],     # Pareto-frontier on (Net, Cashback rate)
          "settings": {...},   # echo of fixed params used
          "n_evaluated": int,
          "runtime_s": float,
        }

    Per-bucket rates default to {le_50: 50, 50_100: 30, 100_200: 15,
    200_400: 0, gt_400: 0} for any tier/bucket not specified by the caller.
    """
    t0 = time.time()
    obj_key = _resolve_objective(objective)

    # Precompute per-bill arrays (one merge, never again)
    col = PRED_METHOD_TO_COL.get(prediction_method)
    if col is None:
        raise ValueError(f"prediction_method must be one of {list(PRED_METHOD_TO_COL)}, got {prediction_method!r}")
    u = users_df[["patient_id", col]].rename(columns={col: "pred"})
    u = u[u["pred"].notna() & (u["pred"] > 0)]
    joined = april_df[["patient_id", "bill_value_net"]].merge(u, on="patient_id", how="inner")
    pred = joined["pred"].to_numpy(dtype=float)
    B = joined["bill_value_net"].to_numpy(dtype=float)

    # Normalise bucket_response_rates: ensure dict-of-dicts. Caller may pass
    # a flat dict (uniform per tier) or a nested dict (per-tier). Stored as
    # a nested dict keyed by tier; missing tiers fall back to defaults.
    bucket_rates: Dict[int, Dict[str, float]] = {}
    if bucket_response_rates:
        # If it looks like {bucket: pct} (no integer keys), treat as uniform.
        flat = all(isinstance(k, str) for k in bucket_response_rates.keys())
        if flat:
            uniform = {k: float(v) for k, v in bucket_response_rates.items()}
            # Will be applied to every tier dynamically by score_ladder_fast
            # (via DEFAULT_BUCKET_RATES_PCT fallback). To make it uniform across
            # all tiers, store under a sentinel tier key 0 — but score_ladder_fast
            # expects per-tier keys. Materialise per-tier on the fly inside scorer
            # using N at the time we know it. Simpler: build a per-tier dict
            # in the scorer call below by populating all tiers from `uniform`.
            for tk in range(1, 11):
                bucket_rates[tk] = dict(uniform)
        else:
            for tk, sub in bucket_response_rates.items():
                bucket_rates[int(tk)] = {str(k): float(v) for k, v in sub.items()}

    rng = np.random.default_rng(seed)

    def score_fn(ladder: List[int]) -> Dict[str, float]:
        return score_ladder_fast(
            pred, B, ladder, bucket_rates,
            cashback_pct=cashback_pct_of_tier, nudge=nudge_step,
            reach_pct=reach_pct, redemption_pct=redemption_pct,
            overshoot_pct=overshoot_pct, margin_pct=margin_pct,
        )

    eval_budget = [0, max_evals]  # [used, cap]
    all_results: List[Tuple[List[int], Dict[str, float]]] = []
    per_n: Dict[int, Tuple[List[int], Dict[str, float]]] = {}

    n_min, n_max = int(n_tier_range[0]), int(n_tier_range[1])
    for N in range(n_min, n_max + 1):
        # Combine smart seeds + random restarts (smart seeds are very cheap;
        # they help climbs land in the right basin).
        starts = smart_starts(N) + [random_ladder(N, rng) for _ in range(n_restarts)]
        # Dedupe identical starts
        seen_starts = set()
        unique_starts = []
        for s in starts:
            k = tuple(s)
            if k not in seen_starts:
                seen_starts.add(k)
                unique_starts.append(s)
        per_n_best = None
        for start in unique_starts:
            if eval_budget[0] >= eval_budget[1]:
                break
            best_l, best_s = _hill_climb(score_fn, start, max_iter, eval_budget)
            all_results.append((best_l, best_s))
            if per_n_best is None or best_s["net"] > per_n_best[1]["net"]:
                per_n_best = (best_l, best_s)
        if per_n_best is not None:
            per_n[N] = per_n_best

    # Filter by constraints, then rank by chosen objective
    valid_results = [(l, s) for l, s in all_results if _passes_constraints(s, constraints or {})]
    valid_results.sort(key=lambda r: r[1][obj_key], reverse=True)

    # Dedupe by ladder
    seen = set(); top = []
    for l, s in valid_results:
        key = tuple(l)
        if key in seen:
            continue
        seen.add(key)
        top.append({
            "ladder": list(l),
            "n_tiers": len(l),
            "totals": {
                "net": round(s["net"], 2),
                "cashback_rate_pct": round(s["cashback_rate_pct"], 2),
                "revenue": round(s["revenue"], 2),
                "rgm": round(s["rgm"], 2),
                "burn": round(s["burn"], 2),
                "roi_pct": round((s["rgm"] - s["burn"]) / s["burn"] * 100, 1) if s["burn"] > 0 else None,
                "bills_scored": s["n"],
            },
        })
        if len(top) >= top_k:
            break

    # Pareto frontier (Net vs Cashback rate) over deduped valid results
    deduped: Dict[Tuple[int, ...], Dict[str, float]] = {}
    for l, s in valid_results:
        deduped[tuple(l)] = s
    pareto: List[Dict[str, Any]] = []
    items = list(deduped.items())
    for k_l, k_s in items:
        dominated = False
        for o_l, o_s in items:
            if o_l == k_l:
                continue
            if ((o_s["net"] > k_s["net"] and o_s["cashback_rate_pct"] >= k_s["cashback_rate_pct"]) or
                (o_s["net"] >= k_s["net"] and o_s["cashback_rate_pct"] > k_s["cashback_rate_pct"])):
                dominated = True
                break
        if not dominated:
            pareto.append({
                "ladder": list(k_l),
                "n_tiers": len(k_l),
                "net": round(k_s["net"], 2),
                "cashback_rate_pct": round(k_s["cashback_rate_pct"], 2),
                "roi_pct": round((k_s["rgm"] - k_s["burn"]) / k_s["burn"] * 100, 1) if k_s["burn"] > 0 else None,
            })
    pareto.sort(key=lambda x: x["net"], reverse=True)

    return {
        "best": top[0] if top else None,
        "top_k": top,
        "pareto": pareto[:10],
        "per_n_best": {N: {"ladder": l, "net": round(s["net"], 2),
                           "cashback_rate_pct": round(s["cashback_rate_pct"], 2)}
                       for N, (l, s) in per_n.items()},
        "settings": {
            "objective": objective,
            "bucket_response_rates_used": bucket_rates if bucket_rates else "default (50/30/15/0/0 per bucket)",
            "cashback_pct_of_tier": cashback_pct_of_tier,
            "prediction_method": prediction_method,
            "n_tier_range": list(n_tier_range),
            "constraints": constraints or {},
            "nudge_step": nudge_step, "reach_pct": reach_pct,
            "redemption_pct": redemption_pct, "overshoot_pct": overshoot_pct,
            "margin_pct": margin_pct,
            "n_restarts": n_restarts, "max_iter": max_iter, "max_evals": max_evals,
        },
        "n_evaluated": eval_budget[0],
        "runtime_s": round(time.time() - t0, 2),
    }
