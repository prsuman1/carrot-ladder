"""Personalized Carrot — April 2026 Backtest (single-tier-per-user model).

Each returning patient gets ONE assigned threshold and ONE cashback rate
derived from base tiers + nudge + their predicted bill. Cashback fires only
when their actual bill reaches *their* threshold — there's no auto-cashback
for crossing a lower tier they don't own.

Bills that hit their threshold are split into:
  - Nudged:        in-reach, customer topped up → incremental revenue + burn
  - Auto-qualified: already past threshold → burn only, no incremental revenue
  - Unreachable:   below reach window → no cashback at all

UI is built fresh around this model (no zone diagram, no Earned/Free language).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_test_bills, load_user_summary, load_user_topline
from personalized_metrics import assign_threshold, build_slots, pred_column

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
TIER_COLORS = ["#FFEB84", "#F4D03F", "#E8B130", "#C99214", "#A87100"]
EXCLUDED_COLOR = "#BFBFBF"
PLOTLY_TEMPLATE = "plotly_white"
GRID_COLOR = "rgba(0,0,0,0.08)"
TEXT_COLOR = "#222222"
HEADING = "#1F4E78"
MUTED = "#606060"
NET_POS = "#1B7A3F"
NET_NEG = "#B11A1A"
COLOR_UNREACH = "#F8696B"
COLOR_INREACH = "#F4D03F"
COLOR_AUTO = "#9BC2E6"


def fmt_inr(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"₹{x:,.0f}"


def fmt_inr_cr(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    a = abs(x)
    if a >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


def fmt_inr_signed(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "−" if x < 0 else "+"
    return f"{sign}₹{abs(x):,.0f}"


def fmt_inr_signed_cr(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "−" if x < 0 else "+"
    a = abs(x)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:,.2f} L"
    return f"{sign}₹{a:,.0f}"


def fmt_pct(x: float, dp: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{dp}f}%"


# ---------------------------------------------------------------------------
# Load (cached)
# ---------------------------------------------------------------------------
users_full = load_user_summary()
test_bills_full = load_test_bills()
top = load_user_topline()
april_bills_full = test_bills_full[test_bills_full["window"] == "april"].copy()
N_APRIL_TOTAL = len(april_bills_full)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<h2 style='margin-top:0;color:{HEADING}'>Personalized Carrot</h2>"
        f"<div style='font-size:0.85rem;color:{MUTED};margin-top:-0.3rem'>April 2026 backtest · single tier per user</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div style='font-size:0.82rem;color:{MUTED};line-height:1.5;margin-top:0.6rem'>
<b>Training</b>&nbsp;&nbsp;< {top['training_cutoff']} · {top['n_training_bills']:,} bills<br>
<b>Users</b>&nbsp;&nbsp;{top['n_users']:,} (warm {top['n_warm']:,} · light {top['n_light']:,})<br>
<b>April bills</b>&nbsp;&nbsp;{top['n_test_bills_april']:,}<br>
<b>Excluded (new)</b>&nbsp;&nbsp;{top['n_excluded_new_patient']:,}
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Segment")
    segment_choice = st.radio(
        "Include",
        options=["Warm only (≥3 bills)", "Warm + Light (≥1 bill)"],
        index=1,
    )
    cap_bills = st.number_input(
        "Cap bill_count at",
        min_value=10, max_value=10000, value=500, step=50,
        help="Drop synthetic walk-in accounts with unrealistic bill counts.",
    )

    st.subheader("Prediction method")
    pred_method = st.radio(
        "Per-user predicted bill",
        options=["Average", "P80", "P90", "P95", "Max"],
        index=1,
        help=(
            "Average — mean of training bills.\n"
            "P80 / P90 / P95 — value below which that % of training bills fall (higher = more conservative).\n"
            "Max — highest training bill (most conservative; nearly zero auto-qualified)."
        ),
    )

    st.subheader("Base ladder")
    n_tiers = st.radio("Number of tiers", options=[1, 2, 3, 4, 5],
                       index=2, horizontal=True)
    DEFAULT_T = [250, 450, 700, 1100, 1500]
    DEFAULT_GIFT = [10, 20, 40, 80, 120]
    base_T: list[int] = []
    gift_costs: list[float] = []
    for i in range(n_tiers):
        prev = base_T[-1] if base_T else None
        slider_min = (prev + 50) if prev is not None else 100
        default_val = DEFAULT_T[i] if prev is None else max(DEFAULT_T[i], prev + 100)
        T = st.slider(f"Base T{i + 1} (₹)", min_value=slider_min, max_value=3000,
                      value=default_val, step=25, key=f"pT_{i + 1}")
        g = st.number_input(f"T{i + 1} cashback (₹)", min_value=0, max_value=2000,
                            value=DEFAULT_GIFT[i], step=5, key=f"pG_{i + 1}")
        base_T.append(int(T))
        gift_costs.append(float(g))

    st.subheader("Nudge")
    nudge_amount = st.slider(
        "Nudge step ₹", min_value=0, max_value=300, value=50, step=5,
        help=(
            "Between consecutive base tiers, generate slot values at nudge-step intervals. "
            "Each user is assigned the smallest slot value strictly greater than their predicted bill."
        ),
    )

    st.subheader("Behaviour")
    reach_pct = st.slider("Reach window (T1 only)", min_value=50, max_value=95, value=75, step=5,
                          format="%d%%",
                          help=("Applies ONLY to T1 (the smallest tier). A T1 bill below "
                                "reach × threshold is 'unreachable' (no cashback). "
                                "T2+ has no reach cutoff — every bill below threshold is "
                                "in-reach (converts at the flat response rate)."))
    reach = reach_pct / 100.0
    response_pct = st.slider("Response rate", min_value=0, max_value=100, value=40, step=1,
                             format="%d%%",
                             help="Of bills in-reach, what % top up to cross their threshold.")
    response_rate = response_pct / 100.0
    redemption_pct = st.slider("Redemption rate", min_value=0, max_value=100, value=70, step=5,
                               format="%d%%",
                               help="Of cashback issued, what % gets redeemed (= real burn).")
    redemption_rate = redemption_pct / 100.0

    st.subheader("Money")
    overshoot_pct = st.slider("Revenue overshoot κ", min_value=0, max_value=100, value=30, step=5,
                              format="%d%%",
                              help="Customer crossing threshold spends ~(1+κ) × gap.")
    kappa = 1.0 + overshoot_pct / 100.0
    rgm_pct = st.slider("Gross margin", min_value=5, max_value=60, value=36, step=1,
                        format="%d%%")
    rgm_rate = rgm_pct / 100.0


# ---------------------------------------------------------------------------
# Slot building + tier assignment now come from personalized_metrics (shared
# with the chatbot tools). Page-local code below uses these directly.
# ---------------------------------------------------------------------------
# Filter + assign
# ---------------------------------------------------------------------------
users = users_full[users_full["bill_count"] <= cap_bills].copy()
if segment_choice.startswith("Warm only"):
    users = users[users["segment"] == "warm"]

if pred_method == "Average":
    users["pred"] = users["pred_avg"]
elif pred_method == "P80":
    users["pred"] = users["pred_p80"]
elif pred_method == "P90":
    users["pred"] = users["pred_p90"]
elif pred_method == "P95":
    users["pred"] = users["pred_p95"]
else:  # Max
    users["pred"] = users["pred_max"]

users = users[users["pred"] > 0].reset_index(drop=True)

th, bi = assign_threshold(users["pred"].to_numpy(dtype=float), base_T, nudge_amount)
users["threshold"] = th
users["base_idx"] = bi
users["cashback"] = np.where(bi >= 0, np.take(gift_costs, np.clip(bi, 0, len(gift_costs) - 1)), np.nan)

n_users_total = len(users)
n_excluded = int((users["base_idx"] == -1).sum())
users_assigned = users[users["base_idx"] >= 0]

# ---------------------------------------------------------------------------
# Score April bills against the assigned thresholds
# ---------------------------------------------------------------------------
bills_scored = april_bills_full.merge(
    users_assigned[["patient_id", "threshold", "cashback", "base_idx"]],
    on="patient_id", how="inner",
)


def score(df: pd.DataFrame, nudge_override: int | None = None,
          base_T_override: list[int] | None = None) -> dict:
    """Score actual bills against each user's single assigned threshold.

    If overrides given, re-assign thresholds first (used by nudge-sensitivity sweep)."""
    if nudge_override is not None or base_T_override is not None:
        # Re-assign thresholds for the sweep
        bT = base_T_override if base_T_override is not None else base_T
        nd = nudge_override if nudge_override is not None else nudge_amount
        u_local = users_full[users_full["bill_count"] <= cap_bills].copy()
        if segment_choice.startswith("Warm only"):
            u_local = u_local[u_local["segment"] == "warm"]
        if pred_method == "Average":
            u_local["pred"] = u_local["pred_avg"]
        elif pred_method == "P80":
            u_local["pred"] = u_local["pred_p80"]
        elif pred_method == "P90":
            u_local["pred"] = u_local["pred_p90"]
        elif pred_method == "P95":
            u_local["pred"] = u_local["pred_p95"]
        else:  # Max
            u_local["pred"] = u_local["pred_max"]
        u_local = u_local[u_local["pred"] > 0].reset_index(drop=True)
        th_l, bi_l = assign_threshold(u_local["pred"].to_numpy(dtype=float), bT, nd)
        u_local["threshold"] = th_l
        u_local["base_idx"] = bi_l
        u_local["cashback"] = np.where(bi_l >= 0,
                                       np.take(gift_costs, np.clip(bi_l, 0, len(gift_costs) - 1)),
                                       np.nan)
        u_local = u_local[u_local["base_idx"] >= 0]
        df = april_bills_full.merge(
            u_local[["patient_id", "threshold", "cashback", "base_idx"]],
            on="patient_id", how="inner",
        )

    n = len(df)
    K = len(base_T)
    if n == 0:
        return {"n": 0, "cashback_rate_pct": 0.0, "n_unreach": 0,
                "n_inreach": 0, "n_auto": 0,
                "nudged_pt": np.zeros(K), "auto_pt": np.zeros(K),
                "inreach_pt": np.zeros(K), "unreach_pt": np.zeros(K),
                "bills_pt": np.zeros(K),
                "revenue_pt": np.zeros(K), "burn_pt": np.zeros(K),
                "users_assigned_pt": np.zeros(K),
                "revenue": 0.0, "rgm": 0.0, "burn": 0.0, "net": 0.0}

    B = df["bill_value_net"].to_numpy(dtype=float)
    V = df["threshold"].to_numpy(dtype=float)
    C = df["cashback"].to_numpy(dtype=float)
    BI = df["base_idx"].to_numpy(dtype=int)

    # Asymmetric reach window: T1 (base_idx==0) keeps the reach cutoff;
    # T2+ has no unreachable bucket — every bill below threshold is in-reach.
    auto = B >= V
    is_t1 = (BI == 0)
    unreach = is_t1 & (B < reach * V)
    inreach = (~auto) & (~unreach)
    gap = V - B
    gap_factor = np.where(inreach, 1.0, 0.0)  # flat (gap-aware scaling removed)
    p_convert = response_rate * gap_factor  # only nonzero for in-reach

    n_unreach = int(unreach.sum())
    n_inreach = int(inreach.sum())
    n_auto = int(auto.sum())

    # Per base-tier aggregates
    nudged_pt = np.zeros(K)
    auto_pt = np.zeros(K)
    inreach_pt = np.zeros(K)
    unreach_pt = np.zeros(K)
    bills_pt = np.zeros(K)
    revenue_pt = np.zeros(K)
    burn_pt = np.zeros(K)
    for k in range(K):
        mk = BI == k
        if not mk.any():
            continue
        bills_pt[k] = int(mk.sum())
        unreach_pt[k] = int(unreach[mk].sum())
        inreach_pt[k] = int(inreach[mk].sum())
        auto_pt[k] = int(auto[mk].sum())
        # Nudged expected = sum of p_convert across in-reach bills for this tier
        nudged_pt[k] = float(p_convert[mk & inreach].sum())
        # Revenue = nudged × gap × κ
        m_nudged = mk & inreach
        if m_nudged.any():
            revenue_pt[k] = float((p_convert[m_nudged] * gap[m_nudged] * kappa).sum())
        # Burn = (auto + nudged) × cashback × redemption
        burn_pt[k] = (auto_pt[k] + nudged_pt[k]) * gift_costs[k] * redemption_rate

    users_assigned_pt = np.zeros(K)
    # Recompute users-assigned for this score call (in case overrides are used)
    if nudge_override is not None or base_T_override is not None:
        u_counts = u_local.groupby("base_idx").size()
    else:
        u_counts = users_assigned.groupby("base_idx").size()
    for k in range(K):
        if k in u_counts.index:
            users_assigned_pt[k] = int(u_counts.loc[k])

    revenue = float(revenue_pt.sum())
    rgm = revenue * rgm_rate
    burn = float(burn_pt.sum())
    # Cashback rate = expected % of bills earning cashback = (nudged + auto) / n
    cashback_rate = (float(nudged_pt.sum()) + n_auto) / n * 100.0 if n > 0 else 0.0
    return {
        "n": n, "cashback_rate_pct": cashback_rate,
        "n_unreach": n_unreach, "n_inreach": n_inreach, "n_auto": n_auto,
        "nudged_pt": nudged_pt, "auto_pt": auto_pt,
        "inreach_pt": inreach_pt, "unreach_pt": unreach_pt,
        "bills_pt": bills_pt,
        "revenue_pt": revenue_pt, "burn_pt": burn_pt,
        "users_assigned_pt": users_assigned_pt,
        "revenue": revenue, "rgm": rgm, "burn": burn,
        "net": rgm - burn,
    }


res = score(bills_scored)
total_nudged = float(res["nudged_pt"].sum())
total_auto = float(res["auto_pt"].sum())

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.title("Personalized Carrot — April 2026 Backtest")
st.caption(
    f"Each returning patient gets **one** personalized threshold based on their "
    f"predicted next bill. We score **{N_APRIL_TOTAL:,}** actual April bills against "
    f"those assignments. New patients ({top['n_excluded_new_patient']:,} bills) are excluded — "
    f"they're on a different program."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        f"""
**Slot values.** Between consecutive base tiers, generate slot values at nudge-step intervals.
With tiers `[250, 450, 700, 1100]` and nudge `₹50`:

| Tier | Slot values | Cashback rate |
|---|---|---|
| T₁ = 250 | 250, 300, 350, 400 | T₁'s cashback |
| T₂ = 450 | 450, 500, 550, 600, 650 | T₂'s cashback |
| T₃ = 700 | 700, 750, …, 1050 | T₃'s cashback |
| T₄ = 1100 | 1100 only | T₄'s cashback |

**Assignment.** Each user gets the **smallest slot strictly greater than their predicted bill**:
- Predicted ₹130 → slot ₹250 (T₁'s cashback). Threshold to hit: ₹250.
- Predicted ₹570 → slot ₹600 (T₂'s cashback ₹20). Threshold to hit: ₹600.
- Predicted ₹670 → slot ₹700 (T₃'s cashback). Threshold to hit: ₹700.
- Predicted ≥ ₹1100 → **excluded** (high-spender; not a target).

**Outcomes per actual bill.** Asymmetric: T1 keeps a reach cutoff; T2+ has none (every bill below threshold is nudgable).

For a **T1** bill `B` with threshold `V`, reach window `r` (default 75%):
- `B ≥ V` → **Auto-qualified**. Cashback paid. *No incremental revenue.*
- `r × V ≤ B < V` → **In-reach**. With probability = `response rate` they top up to V → **Nudged**. Burn + incremental revenue.
- `B < r × V` → **Unreachable**. No cashback.

For a **T2+** bill (no reach cutoff):
- `B ≥ V` → **Auto-qualified** (same).
- `B < V` → **In-reach**. Conversion at flat `response rate`.

**Cashback rate** = expected % of bills earning cashback = `(nudged + auto-qualified) / total bills`.
**Net** = Revenue × Margin − Burn.
"""
    )

st.divider()

# ---------- 2. Headline panel ----------
net_color = NET_POS if res["net"] >= 0 else NET_NEG
cb_rate = res["cashback_rate_pct"]
cb_color = HEADING
status = (
    f"At your current settings the program is "
    f"<b style='color:{net_color}'>{'PROFITABLE' if res['net'] >= 0 else 'LOSING MONEY'}</b>. "
    f"Cashback fires on <b>{cb_rate:.1f}%</b> of April bills."
)
st.markdown(
    f"""
<div style='border-left:6px solid {net_color}; background:#F8F9FA; padding:18px 22px; border-radius:6px'>
  <div style='font-size:0.95rem; color:{MUTED}; margin-bottom:6px'>April 2026 · 30 days · {res['n']:,} bills scored</div>
  <div style='display:flex; align-items:baseline; gap:32px; flex-wrap:wrap'>
    <div>
      <div style='font-size:0.85rem; color:{MUTED}'>NET</div>
      <div style='font-size:2.6rem; font-weight:700; color:{net_color}; line-height:1.1'>
        {fmt_inr_signed_cr(res['net'])}
      </div>
    </div>
    <div>
      <div style='font-size:0.85rem; color:{MUTED}'>CASHBACK RATE</div>
      <div style='font-size:2.6rem; font-weight:700; color:{cb_color}; line-height:1.1'>
        {fmt_pct(cb_rate)}
      </div>
    </div>
  </div>
  <div style='margin-top:12px; color:{TEXT_COLOR}; font-size:1rem; line-height:1.5'>{status}</div>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Revenue (incremental)", fmt_inr_cr(res["revenue"]),
          help="From nudged conversions only: gap × κ × p_convert, summed.")
m2.metric(f"RGM ({rgm_pct}%)", fmt_inr_cr(res["rgm"]),
          help="Revenue × Gross Margin.")
m3.metric("Burn (redeemed)", fmt_inr_cr(res["burn"]),
          help=f"(Nudged + Auto-qualified) × cashback × {redemption_pct}% redemption.")
m4.metric("Nudged vs Auto-qualified",
          f"{int(total_nudged):,} / {int(total_auto):,}",
          help=("Nudged = bills where the program caused the upsell (in-reach, customer topped up). "
                "Auto-qualified = bills that crossed threshold without any nudge."))

st.divider()

# ---------- Per-tier outcome cards (April-only) ----------
K = len(base_T)
st.header("What happened in April per tier")
st.caption("For each base tier, how the actual April bills landed against assigned thresholds.")

for k in range(K):
    n_bills_k = int(res["bills_pt"][k])
    if n_bills_k == 0:
        continue
    n_unr = int(res["unreach_pt"][k])
    n_ir = int(res["inreach_pt"][k])
    n_aut = int(res["auto_pt"][k])
    nudged_k = float(res["nudged_pt"][k])
    rev_k = float(res["revenue_pt"][k])
    rgm_k = rev_k * rgm_rate
    burn_k = float(res["burn_pt"][k])
    net_k = rgm_k - burn_k
    net_color_k = NET_POS if net_k >= 0 else NET_NEG
    pct_unr = n_unr / n_bills_k * 100
    pct_ir = n_ir / n_bills_k * 100
    pct_aut = n_aut / n_bills_k * 100

    # Per-tier stacked bar — T1 has 3 segments (unreachable / in-reach / auto),
    # T2+ has 2 segments (in-reach / auto). The score function already enforces
    # that n_unr == 0 for T2+.
    fig_t = go.Figure()
    if k == 0 and n_unr > 0:
        fig_t.add_trace(go.Bar(
            x=[n_unr], y=[""], orientation="h",
            marker=dict(color=COLOR_UNREACH, line=dict(width=0)),
            text=f"Unreachable<br>{n_unr:,} ({pct_unr:.1f}%)",
            textposition="inside", insidetextanchor="middle",
            hoverinfo="skip", showlegend=False,
        ))
    fig_t.add_trace(go.Bar(
        x=[n_ir], y=[""], orientation="h",
        marker=dict(color=COLOR_INREACH, line=dict(width=0)),
        text=f"In-reach<br>{n_ir:,} ({pct_ir:.1f}%)<br>→ nudged {nudged_k:,.0f}",
        textposition="inside", insidetextanchor="middle",
        hoverinfo="skip", showlegend=False,
    ))
    fig_t.add_trace(go.Bar(
        x=[n_aut], y=[""], orientation="h",
        marker=dict(color=COLOR_AUTO, line=dict(width=0)),
        text=f"Auto-qualified<br>{n_aut:,} ({pct_aut:.1f}%)",
        textposition="inside", insidetextanchor="middle",
        hoverinfo="skip", showlegend=False,
    ))
    fig_t.update_layout(
        template=PLOTLY_TEMPLATE, barmode="stack", height=110,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    st.markdown(
        f"""
<div style='border-left:5px solid {TIER_COLORS[k]}; background:#FCFCFD;
            padding:14px 18px; border-radius:6px; margin-top:0.6rem'>
  <div style='display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:12px'>
    <div style='font-size:1.05rem; font-weight:700; color:{HEADING}'>
      T{k+1} = ₹{base_T[k]} &nbsp;·&nbsp; cashback ₹{int(gift_costs[k])}
      &nbsp;·&nbsp; <span style='color:{MUTED};font-weight:500'>{n_bills_k:,} bills</span>
    </div>
    <div style='font-size:1.2rem; font-weight:700; color:{net_color_k}'>
      Net {fmt_inr_signed(net_k)}
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_t, width="stretch")

    cols = st.columns(4)
    cols[0].metric("Nudged conversions", f"{nudged_k:,.0f}",
                   help="In-reach × p_convert (expected). Drives incremental revenue.")
    cols[1].metric("Auto-qualified", f"{n_aut:,}",
                   help="Bills already past threshold without any nudge.")
    cols[2].metric("Revenue", fmt_inr(rev_k))
    cols[3].metric("Burn", fmt_inr(burn_k))

st.divider()

# ---------- Actual April bill distribution ----------
slots_arr, _ = build_slots(base_T, nudge_amount)
st.header("Actual April bill distribution")
st.caption("Where the real April bills landed. Same ladder/slot lines overlaid.")
B_arr = bills_scored["bill_value_net"].to_numpy(dtype=float)
hist_max_b = float(min(np.percentile(B_arr, 99), max(base_T) * 1.15))
edges_b = np.arange(0, hist_max_b + 25, 25)
counts_b, _ = np.histogram(B_arr, bins=edges_b)

fig_act = go.Figure()
fig_act.add_trace(go.Bar(
    x=(edges_b[:-1] + edges_b[1:]) / 2, y=counts_b, width=25,
    marker=dict(color="#A8D08D", line=dict(width=0)),
    hovertemplate="₹%{x:.0f}<br>%{y:,} bills<extra></extra>",
    showlegend=False,
))
for s in slots_arr:
    if 0 < s < hist_max_b and s not in base_T:
        fig_act.add_vline(x=float(s),
                          line=dict(color="rgba(0,0,0,0.18)", width=1, dash="dot"))
for i, T in enumerate(base_T):
    if T <= hist_max_b:
        fig_act.add_vline(x=float(T),
                          line=dict(color="black", width=1.4),
                          annotation_text=f"T{i+1}=₹{T}",
                          annotation_position="top")
fig_act.update_layout(
    template=PLOTLY_TEMPLATE, height=340,
    margin=dict(l=10, r=10, t=30, b=40),
    xaxis_title="Actual bill ₹", yaxis_title="Bills",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR), showlegend=False,
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, range=[0, hist_max_b]),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
)
st.plotly_chart(fig_act, width="stretch")

st.divider()

# ---------- 7. Method comparison ----------
st.header("Prediction-method comparison")
st.caption("Same ladder + behaviour; vary only the prediction method. Bold = currently selected.")


def score_for_method(method: str) -> dict:
    u_local = users_full[users_full["bill_count"] <= cap_bills].copy()
    if segment_choice.startswith("Warm only"):
        u_local = u_local[u_local["segment"] == "warm"]
    if method == "Average":
        u_local["pred"] = u_local["pred_avg"]
    elif method == "P80":
        u_local["pred"] = u_local["pred_p80"]
    elif method == "P90":
        u_local["pred"] = u_local["pred_p90"]
    elif method == "P95":
        u_local["pred"] = u_local["pred_p95"]
    else:  # Max
        u_local["pred"] = u_local["pred_max"]
    u_local = u_local[u_local["pred"] > 0].reset_index(drop=True)
    th_l, bi_l = assign_threshold(u_local["pred"].to_numpy(dtype=float), base_T, nudge_amount)
    u_local["threshold"] = th_l
    u_local["base_idx"] = bi_l
    u_local["cashback"] = np.where(bi_l >= 0,
                                   np.take(gift_costs, np.clip(bi_l, 0, len(gift_costs) - 1)),
                                   np.nan)
    excl = int((u_local["base_idx"] == -1).sum())
    u_assigned = u_local[u_local["base_idx"] >= 0]
    bs = april_bills_full.merge(
        u_assigned[["patient_id", "threshold", "cashback", "base_idx"]],
        on="patient_id", how="inner",
    )
    r = score(bs)
    r["excluded_users"] = excl
    r["assigned_users"] = len(u_assigned)
    return r


def _net_color_fn(val: str) -> str:
    if val.startswith("+"):
        return f"color: {NET_POS}; font-weight: 600"
    if val.startswith("−"):
        return f"color: {NET_NEG}; font-weight: 600"
    return ""


mc_rows = []
for m in ["Average", "P80", "P90", "P95", "Max"]:
    r = score_for_method(m)
    mc_rows.append({
        "Method": ("**" + m + "**") if m == pred_method else m,
        "April Bills": f"{r['n']:,}",
        "Cashback rate": fmt_pct(r["cashback_rate_pct"]),
        "Nudged": f"{r['nudged_pt'].sum():,.0f}",
        "Auto-qualified": f"{int(r['auto_pt'].sum()):,}",
        "Revenue": fmt_inr(r["revenue"]),
        "Burn": fmt_inr(r["burn"]),
        "Net": fmt_inr_signed(r["net"]),
    })
st.dataframe(
    pd.DataFrame(mc_rows).style.map(_net_color_fn, subset=["Net"]),
    hide_index=True, width="stretch",
)

st.divider()

# ---------- 8. Nudge sensitivity ----------
st.header("Nudge sensitivity")
st.caption("Sweep nudge ₹ from 0 to 200 holding all other settings constant.")

nudge_grid = list(range(0, 201, 20))
net_curve, cb_curve, rgm_curve, burn_curve = [], [], [], []
for nd in nudge_grid:
    r_nd = score(bills_scored, nudge_override=nd)
    net_curve.append(r_nd["net"])
    cb_curve.append(r_nd["cashback_rate_pct"])
    rgm_curve.append(r_nd["rgm"])
    burn_curve.append(r_nd["burn"])

fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(x=nudge_grid, y=net_curve, mode="lines+markers",
                              name="Net ₹", line=dict(color=NET_POS, width=2.5)))
fig_sens.add_trace(go.Scatter(x=nudge_grid, y=rgm_curve, mode="lines",
                              name="RGM", line=dict(color="#8FBF8F", width=1.5, dash="dot")))
fig_sens.add_trace(go.Scatter(x=nudge_grid, y=burn_curve, mode="lines",
                              name="Burn", line=dict(color=NET_NEG, width=1.5, dash="dot")))
fig_sens.add_trace(go.Scatter(x=nudge_grid, y=cb_curve, mode="lines+markers",
                              name="Cashback rate %", yaxis="y2",
                              line=dict(color=HEADING, width=2.5)))
fig_sens.add_hline(y=0, line=dict(color="#999999", width=1))
fig_sens.add_vline(x=nudge_amount,
                   line=dict(color="#F8696B", width=1.5, dash="dot"),
                   annotation_text=f"current ₹{nudge_amount}")
fig_sens.update_layout(
    height=400, margin=dict(l=10, r=10, t=30, b=40),
    template=PLOTLY_TEMPLATE, font=dict(color=TEXT_COLOR),
    xaxis=dict(title="Nudge ₹", showgrid=True, gridcolor=GRID_COLOR),
    yaxis=dict(title="₹", showgrid=True, gridcolor=GRID_COLOR),
    yaxis2=dict(title="Cashback rate %", overlaying="y", side="right", range=[0, 100]),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
)
st.plotly_chart(fig_sens, width="stretch")

st.divider()
st.markdown(
    f"<div style='font-size:0.78rem;color:{MUTED}'>"
    f"Source: <code>data/user_summary.parquet</code> + <code>data/test_bills.parquet</code> "
    f"(<code>{top['generated_at']}</code>) — "
    f"training cutoff {top['training_cutoff']}, "
    f"test window {top['test_window_april'][0]}→{top['test_window_april'][1]}."
    f"</div>",
    unsafe_allow_html=True,
)
