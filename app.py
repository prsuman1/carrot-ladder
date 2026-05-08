"""AOV Ladder Threshold — Streamlit UI.

Reads only pre-computed summary files from data/. Never imports psycopg2.
Slider changes recompute zone metrics on a small in-memory histogram; file
loads are cached via @st.cache_data so they run only once.

Generalized to N tiers (1..5) with an expected-response-rate model and an
optional auto-optimizer that searches the threshold grid for the best
adjusted_expected score.
"""

from __future__ import annotations

import itertools
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    load_cdf_chart,
    load_histogram_chart,
    load_percentiles,
    load_topline,
    load_value_distribution,
)
from metrics import expected_lift, money_breakdown, score_pair, zone_counts

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AOV Ladder Threshold",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZONE_COLORS_PALETTE = {
    "unreachable": "#F8696B",
    "in_reach": ["#FFEB84", "#F4D03F", "#E8B130", "#C99214", "#A87100"],
    "mid_zone": ["#E5E5E5", "#D9D9D9", "#CCCCCC", "#BFBFBF", "#B3B3B3"],
    "past_top": "#9BC2E6",
}

# App is locked to light mode — see .streamlit/config.toml for the Streamlit chrome.
PLOTLY_TEMPLATE = "plotly_white"
GRID_COLOR = "rgba(0,0,0,0.08)"
TEXT_COLOR = "#222222"
HEADING    = "#1F4E78"
MUTED      = "#606060"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


MONTHS_IN_WINDOW = 3.0  # 90-day SQL pull → 3 months


def per_month(n: float) -> float:
    return n / MONTHS_IN_WINDOW


def fmt_inr(x: float) -> str:
    return f"₹{x:,.0f}"


def fmt_inr_cr(x: float) -> str:
    return f"₹{x / 1e7:,.2f} Cr"


def fmt_inr_signed(x: float) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    sign = "−" if x < 0 else "+"
    return f"{sign}₹{abs(x):,.0f}"


def fmt_pct(x: float, dp: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{dp}f}%"


def fmt_dt(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso


def get_git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unversioned"


@st.cache_data
def revenue_concentration(vd: pd.DataFrame) -> pd.DataFrame:
    """Lorenz-style cumulative curve: cumulative % of bills (sorted by value, highest first)
    vs cumulative % of revenue."""
    df = vd.sort_values("bin", ascending=False).reset_index(drop=True)
    cum_bills = df["count"].cumsum().to_numpy(dtype=float)
    cum_rev = df["sum_value"].cumsum().to_numpy(dtype=float)
    total_bills = float(df["count"].sum())
    total_rev = float(df["sum_value"].sum())
    out = pd.DataFrame({
        "pct_bills":   cum_bills / total_bills * 100.0,
        "pct_revenue": cum_rev   / total_rev   * 100.0,
    })
    return pd.concat([pd.DataFrame({"pct_bills": [0.0], "pct_revenue": [0.0]}), out],
                     ignore_index=True)


def revenue_at_pct_bills(curve: pd.DataFrame, x_pct: float) -> float:
    return float(np.interp(x_pct, curve["pct_bills"].values, curve["pct_revenue"].values))


def concentration_figure(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
        name="Equal distribution",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=curve["pct_bills"], y=curve["pct_revenue"],
        mode="lines",
        line=dict(color=HEADING, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,78,120,0.10)",
        name="Revenue concentration",
        hovertemplate="Top %{x:.1f}% of bills<br>= %{y:.1f}% of revenue<extra></extra>",
        showlegend=False,
    ))
    y_at_10 = revenue_at_pct_bills(curve, 10.0)
    fig.add_shape(type="line", x0=10, x1=10, y0=0, y1=y_at_10,
                  line=dict(color=TEXT_COLOR, width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, x1=10, y0=y_at_10, y1=y_at_10,
                  line=dict(color=TEXT_COLOR, width=1, dash="dot"))
    fig.add_annotation(
        x=10, y=y_at_10, ax=40, ay=-30,
        text=f"Top 10% of bills<br>= {y_at_10:.0f}% of revenue",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
        arrowcolor=TEXT_COLOR, font=dict(color=TEXT_COLOR, size=11),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="Revenue concentration",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Top % of bills (sorted high → low value)",
                   range=[0, 100], showgrid=True, gridcolor=GRID_COLOR, ticksuffix="%"),
        yaxis=dict(title="% of total revenue",
                   range=[0, 100], showgrid=True, gridcolor=GRID_COLOR, ticksuffix="%"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def run_optimizer(vd, total_n, n_tiers, reach, response_rate, gap_aware, alpha):
    """Coarse-grid search over tier combinations. Returns the (T1..TN) that
    maximizes adjusted_expected at the current α and reach. Spacing constraint:
    consecutive thresholds differ by ≥ ₹100."""
    grid = list(range(200, 2001, 50))
    best = None
    best_score = -1e18
    for combo in itertools.combinations(grid, n_tiers):
        if n_tiers > 1 and not all(combo[i + 1] - combo[i] >= 100 for i in range(n_tiers - 1)):
            continue
        z = zone_counts(vd, total_n, list(combo), reach)
        lf = expected_lift(z, response_rate, gap_aware)
        s = score_pair(z, lf, alpha)
        if s["adjusted_expected"] > best_score:
            best_score = s["adjusted_expected"]
            best = {
                "thresholds": list(combo),
                "expected_lift": lf["total"],
                "reach_score": s["reach_score"],
                "waste": s["wasted_carrot_pct"],
                "adjusted_expected": s["adjusted_expected"],
            }
    return best


# ---------------------------------------------------------------------------
# Load summaries (cached)
# ---------------------------------------------------------------------------
topline = load_topline()
percentiles = load_percentiles()
hist_df = load_histogram_chart()
cdf_df = load_cdf_chart()
vd = load_value_distribution()
N_BILLS = topline["n_bills"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<h2 style='margin-top:0;color:{HEADING}'>AOV Ladder Threshold</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div style='font-size:0.85rem;color:{MUTED};line-height:1.5'>
<b>Generated</b>&nbsp;&nbsp;{fmt_dt(topline['generated_at'])}<br>
<b>Bills</b>&nbsp;&nbsp;{topline['n_bills']:,}<br>
<b>Window</b>&nbsp;&nbsp;{topline['date_window_start']} → {topline['date_window_end']}
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    n_tiers = st.radio(
        "Number of tiers",
        options=[1, 2, 3, 4, 5],
        index=1,
        horizontal=True,
    )

    st.subheader("Thresholds")
    DEFAULT_T = [399, 899, 1299, 1699, 1999]
    DEFAULT_GIFT = [10, 25, 50, 100, 150]
    thresholds: list[int] = []
    for i in range(n_tiers):
        prev = thresholds[-1] if thresholds else None
        slider_min = (prev + 50) if prev is not None else 200
        default_val = DEFAULT_T[i] if prev is None else max(DEFAULT_T[i], prev + 100)
        key = f"T_{i + 1}"
        # Pre-seed session state before the widget is created. Streamlit then
        # reads the value from session_state and we don't need to pass `value=`,
        # which would otherwise trigger a "default value + session state" warning.
        if key not in st.session_state:
            st.session_state[key] = default_val
        else:
            try:
                cur = int(st.session_state[key])
            except (TypeError, ValueError):
                cur = default_val
            st.session_state[key] = max(slider_min, min(2500, cur))
        T = st.slider(
            f"T{i + 1}",
            min_value=slider_min,
            max_value=2500,
            step=25,
            key=key,
        )

        gift_key = f"gift_{i + 1}"
        if gift_key not in st.session_state:
            st.session_state[gift_key] = DEFAULT_GIFT[i]
        st.number_input(
            f"T{i + 1} gift (₹)",
            min_value=0,
            max_value=10000,
            step=5,
            format="%d",
            key=gift_key,
            help=f"Cash value of the gift handed out when a bill crosses T{i + 1}.",
        )

        thresholds.append(int(T))

    gift_costs = [float(st.session_state[f"gift_{i + 1}"]) for i in range(n_tiers)]

    st.subheader("Reach window")
    reach_pct = st.slider(
        "Reach window",
        min_value=60, max_value=95, value=75, step=5,
        format="%d%%",
        label_visibility="collapsed",
        help="A bill is 'within reach' of T if reach × T ≤ bill < T.",
    )
    reach = reach_pct / 100.0

    st.subheader("Expected response rate")
    response_pct = st.slider(
        "Expected response rate",
        min_value=0, max_value=100, value=10, step=1,
        format="%d%%", label_visibility="collapsed",
        help="Of customers within reach of a tier, what % actually add the items needed?",
    )
    response_rate = response_pct / 100.0
    gap_aware = st.checkbox(
        "Scale response by gap size",
        value=True,
        help="ON: bills with smaller shortfalls convert at a higher rate than bills with larger shortfalls.",
    )

    st.subheader("Waste penalty (α)")
    alpha = st.slider("α", min_value=0.0, max_value=3.0, value=1.0, step=0.5)

    st.subheader("Gross margin (%)")
    rgm_pct = st.slider(
        "Gross margin",
        min_value=5, max_value=60, value=36, step=1,
        format="%d%%", label_visibility="collapsed",
        help="Of the incremental top-line revenue earned, what % converts to gross margin (RGM)?",
    )
    rgm_rate = rgm_pct / 100.0

    st.divider()
    optimize_clicked = st.button(
        "Find best automatically",
        width="stretch",
        help="Search the threshold space and pick the (T1..TN) that maximizes adjusted_expected at the current α and reach.",
    )

# ---------------------------------------------------------------------------
# Live metric computation (small dataframe, fast)
# ---------------------------------------------------------------------------
zones = zone_counts(vd, N_BILLS, thresholds, reach)
lift = expected_lift(zones, response_rate, gap_aware)
sc = score_pair(zones, lift, alpha)
money = money_breakdown(lift, zones, gift_costs, N_BILLS, MONTHS_IN_WINDOW, rgm_rate)

# ---------------------------------------------------------------------------
# Section 1 — Header
# ---------------------------------------------------------------------------
st.title("AOV Ladder Threshold")
st.caption(
    "Pick a tier ladder on the left. Every chart and number on this page reflects "
    "those thresholds against a ZRF-aware payable bill value. "
    "**All bill counts and revenue figures are shown as monthly averages over the last 3 months.**"
)

# ---------------------------------------------------------------------------
# Section 2 — How this works (collapsed)
# ---------------------------------------------------------------------------
with st.expander("How this works", expanded=False):
    st.markdown(
        """
**Goal.** A ladder of N upsell thresholds (T1 < T2 < … < TN) drives a tiered
carrot ("spend ₹X more to unlock Y"). This page shows how the bill-value
distribution — last 90 days, real sales only — maps onto the resulting
spending zones for any ladder you pick.

**Payable bill value (ZRF-aware).**
- Normal line: `rate × net-quantity`.
- ZRD line (rupee discount coupon): same as above, then subtract `promo-discount`.
- ZRF line, qty > 1 (free item coupon): customer pays for `qty − 1` units.
- ZRF line, qty = 1: customer pays 0 — the only unit is the free one.

**Zones.** With N tiers, every bill falls into exactly one of `2N + 1` zones:
unreachable, in-reach-Tᵢ for i ∈ 1..N, mid-zone-(i→i+1) for i ∈ 1..N-1, and
past-top.

**Reach window.** A bill of value `b` is "within reach" of threshold `T` if
`reach × T ≤ b < T`. Larger reach % = stricter window.

**Expected response.** *Addressable* = % of bills inside any in-reach zone.
*Expected lift* applies an assumed response rate, optionally scaled by how
close the bill was to the threshold (gap-aware).
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Data summary
# ---------------------------------------------------------------------------
st.header("Data summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bills /month", f"{int(round(per_month(topline['n_bills']))):,}")
c2.metric("Revenue /month", fmt_inr_cr(per_month(topline["total_revenue"])))
c3.metric("Mean", fmt_inr(topline["mean"]))
c4.metric("Median", fmt_inr(topline["median"]))

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Percentiles")
    pct_rows = [
        {"Percentile": k, "Bill value (₹)": f"{v:,.0f}"}
        for k, v in percentiles.items()
    ]
    st.dataframe(pd.DataFrame(pct_rows), hide_index=True, width="stretch")

with right:
    st.subheader("Revenue concentration")
    curve = revenue_concentration(vd)
    p10 = revenue_at_pct_bills(curve, 10.0)
    p20 = revenue_at_pct_bills(curve, 20.0)
    p50 = revenue_at_pct_bills(curve, 50.0)

    k1, k2, k3 = st.columns(3)
    k1.metric("Top 10% of bills", fmt_pct(p10, 1), help="Share of total revenue from the highest-value 10% of bills")
    k2.metric("Top 20% of bills", fmt_pct(p20, 1))
    k3.metric("Top 50% of bills", fmt_pct(p50, 1))

    st.plotly_chart(concentration_figure(curve), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Optimizer result (above zones, only when button clicked)
# ---------------------------------------------------------------------------
if optimize_clicked:
    with st.spinner(f"Searching threshold space for best {n_tiers}-tier ladder..."):
        best = run_optimizer(vd, N_BILLS, n_tiers, reach, response_rate, gap_aware, alpha)
    if best is None:
        st.warning("Optimizer found no valid combinations. Try different settings.")
    else:
        bills_moved_pm = int(round(per_month(best["expected_lift"] / 100 * N_BILLS)))
        bills_wasted_pm = int(round(per_month(best["waste"] / 100 * N_BILLS)))
        st.success(
            f"**Best ladder:** T = {best['thresholds']}  →  "
            f"expected lift **{best['expected_lift']:.2f}%** "
            f"(~{bills_moved_pm:,} bills/month), "
            f"waste **{best['waste']:.2f}%** (~{bills_wasted_pm:,} bills/month), "
            f"adjusted_expected **{best['adjusted_expected']:.2f}**.  "
            "Move the sliders to match these thresholds to apply."
        )

# ---------------------------------------------------------------------------
# Section 4 — Threshold zones diagram
# ---------------------------------------------------------------------------
st.header("Threshold zones")


def zones_figure(zones: dict, max_value: int) -> go.Figure:
    los = zones["los"]
    thresholds = zones["thresholds"]

    segments = [
        (0, los[0], f"Unreachable<br>{int(round(per_month(zones['n_unreachable']))):,} bills/month",
         ZONE_COLORS_PALETTE["unreachable"]),
    ]
    for i, t in enumerate(zones["tiers"]):
        segments.append((
            t["lo"], t["T"],
            f"In reach T{i + 1}<br>[₹{t['lo']:.0f}, ₹{t['T']})<br>{int(round(per_month(t['count_in_reach']))):,} bills/month",
            ZONE_COLORS_PALETTE["in_reach"][i],
        ))
        if i < len(zones["mid_zones"]):
            mz = zones["mid_zones"][i]
            if mz["hi"] > mz["lo"]:
                segments.append((
                    mz["lo"], mz["hi"],
                    f"Mid {i + 1}→{i + 2}<br>[₹{mz['lo']:.0f}, ₹{mz['hi']:.0f})<br>{int(round(per_month(mz['count']))):,} bills/month",
                    ZONE_COLORS_PALETTE["mid_zone"][i],
                ))
    segments.append((
        thresholds[-1], max_value,
        f"Past top<br>(≥ ₹{thresholds[-1]})<br>{int(round(per_month(zones['n_past_top']))):,} bills/month",
        ZONE_COLORS_PALETTE["past_top"],
    ))

    fig = go.Figure()
    for lo, hi, label, color in segments:
        width = max(hi - lo, 0)
        if width <= 0:
            continue
        fig.add_trace(go.Bar(
            x=[width], y=[""], orientation="h", base=lo,
            marker=dict(color=color, line=dict(width=0)),
            text=label, textposition="inside", insidetextanchor="middle",
            showlegend=False, hoverinfo="skip",
        ))
    for i, T in enumerate(thresholds):
        fig.add_vline(
            x=T,
            line=dict(color="black", width=1.5),
            annotation_text=f"T{i + 1} = ₹{T}",
            annotation_position="top",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        height=220,
        margin=dict(l=10, r=10, t=60, b=40),
        xaxis_title="Bill value (₹)",
        xaxis=dict(range=[0, max_value], showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(visible=False, showgrid=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
    )
    return fig


zones_max = max(int(thresholds[-1] * 1.4), 1300)
st.plotly_chart(zones_figure(zones, max_value=zones_max), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Live metrics
# ---------------------------------------------------------------------------
st.header("Live metrics")

addressable_pct = sc["reach_score"]
addressable_bills = sum(t["count_in_reach"] for t in zones["tiers"])
addressable_bills_pm = int(round(per_month(addressable_bills)))
expected_bills_pm = int(round(per_month(lift["total"] / 100 * N_BILLS)))
wasted_bills_pm = int(round(per_month(sc["wasted_carrot_pct"] / 100 * N_BILLS)))

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Total addressable",
    fmt_pct(addressable_pct),
    help=f"~{addressable_bills_pm:,} bills/month are within reach of any tier",
)
k2.metric(
    "Expected lift",
    fmt_pct(lift["total"]),
    help=f"At {response_pct}% response rate, ~{expected_bills_pm:,} bills/month will actually move",
)
k3.metric(
    "Wasted carrot %",
    fmt_pct(sc["wasted_carrot_pct"]),
    help=(
        f"~{wasted_bills_pm:,} bills/month receive a gift without changing behavior "
        "(Mid zones + non-converters of higher tiers + bills past the top tier). "
        "Each bill gets one gift — the highest tier it crossed."
    ),
)
k4.metric(
    "Adjusted (expected)",
    f"{sc['adjusted_expected']:.2f}",
    help="expected_lift − α × wasted_carrot_pct (the honest score)",
)

st.subheader("Per-tier breakdown")
tier_rows = []
for i, t in enumerate(zones["tiers"]):
    per_tier_lift = lift["per_tier"][i]
    bills_moved_pm = int(round(per_month(per_tier_lift / 100 * N_BILLS)))
    free_pm = int(round(per_month(lift["free_per_tier"][i] / 100 * N_BILLS)))
    tier_rows.append({
        "Tier": f"T{i + 1}",
        "Threshold": fmt_inr(t["T"]),
        "% addressable": round(t["pct_in_reach"], 2),
        "Avg gap": fmt_inr(t["avg_gap"]) if not np.isnan(t["avg_gap"]) else "—",
        "Per-tier expected lift %": round(per_tier_lift, 2),
        "Bills moved /month": f"{bills_moved_pm:,}",
        "Free /month": f"{free_pm:,}",
    })
st.dataframe(pd.DataFrame(tier_rows), hide_index=True, width="stretch")

st.markdown(
    f"""
<div style='font-size:0.95rem;color:{TEXT_COLOR};line-height:1.6;margin-top:0.5rem'>
Address pool reaches <b>{addressable_bills_pm:,}</b> bills/month (theoretical max).<br>
At <b>{response_pct}%</b> response rate, ~<b>{expected_bills_pm:,}</b> bills/month will actually move.<br>
<b>{wasted_bills_pm:,}</b> bills/month receive a gift without changing behavior (Mid zones + failed nudges + bills already past the top tier).
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Section 5b — Money model (₹)
# ---------------------------------------------------------------------------
st.header("Money model (₹)")
st.caption(
    f"**Earn** = avg gap × earned bills × {rgm_pct}% gross margin = RGM. "
    "**Burn** = (earned + free) bills × gift cost per tier — every gift handed out costs cash. "
    "**Net** = RGM − Burn. Tune gift values inline with each T in the sidebar."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Revenue /month",
    fmt_inr(money["revenue"]),
    help="Sum across tiers of avg_gap × bills_moved (incremental top-line from program-induced conversions).",
)
m2.metric(
    f"RGM /month ({rgm_pct}%)",
    fmt_inr(money["rgm"]),
    help=f"Revenue × {rgm_pct}% gross margin — actual contribution earned.",
)
m3.metric(
    "Burn /month",
    fmt_inr(money["burn"]),
    help="Sum across tiers of (earned + free) bills × gift cost. Total cash spent on gifts.",
)
m4.metric(
    "Net /month",
    fmt_inr_signed(money["net"]),
    delta=("Profitable" if money["net"] >= 0 else "Loss"),
    delta_color=("normal" if money["net"] >= 0 else "inverse"),
    help="RGM − Burn. Positive means the program pays for itself.",
)

st.subheader("Per-tier money breakdown")
money_rows = []
for r in money["per_tier"]:
    money_rows.append({
        "Tier": f"T{r['tier']}",
        "Threshold": fmt_inr(r["T"]),
        "Gift cost": fmt_inr(r["gift_cost"]),
        "Avg gap": fmt_inr(r["avg_gap"]),
        "Revenue /mo": fmt_inr(r["revenue"]),
        f"RGM /mo ({rgm_pct}%)": fmt_inr(r["rgm"]),
        "Burn /mo": fmt_inr(r["burn"]),
        "Net /mo": fmt_inr_signed(r["net"]),
    })
df_money = pd.DataFrame(money_rows)


def _net_color(val: str) -> str:
    if val.startswith("+"):
        return "color: #1B7A3F; font-weight: 600"
    if val.startswith("−"):
        return "color: #B11A1A; font-weight: 600"
    return ""


styled_money = df_money.style.map(_net_color, subset=["Net /mo"])
st.dataframe(styled_money, hide_index=True, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 6 — Distribution charts
# ---------------------------------------------------------------------------
st.header("Distribution")

show_tail = st.checkbox("Include ₹2000–₹5000 tail", value=False)


def histogram_figure(hist_df: pd.DataFrame, zones: dict, show_tail: bool) -> go.Figure:
    df = hist_df.copy()
    df = df[df["bin_hi"] != -1]
    if not show_tail:
        df = df[df["bin_hi"] <= 2000]
    df["mid"] = (df["bin_lo"] + df["bin_hi"]) / 2
    df["width"] = df["bin_hi"] - df["bin_lo"]
    df["count_pm"] = df["count"] / MONTHS_IN_WINDOW

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["mid"], y=df["count_pm"], width=df["width"],
        marker=dict(color="#9BC2E6", line=dict(width=0)),
        hovertemplate="₹%{customdata[0]}–₹%{customdata[1]}<br>Bills/month: %{y:,.0f}<extra></extra>",
        customdata=df[["bin_lo", "bin_hi"]].values,
        showlegend=False,
    ))
    x_max = float(df["bin_hi"].max())

    los = zones["los"]
    thresholds = zones["thresholds"]

    fig.add_vrect(x0=0, x1=min(los[0], x_max),
                  fillcolor=ZONE_COLORS_PALETTE["unreachable"], opacity=0.12,
                  line_width=0, layer="below")
    for i, t in enumerate(zones["tiers"]):
        if t["lo"] >= x_max:
            break
        fig.add_vrect(x0=t["lo"], x1=min(t["T"], x_max),
                      fillcolor=ZONE_COLORS_PALETTE["in_reach"][i], opacity=0.25,
                      line_width=0, layer="below")
        if i < len(zones["mid_zones"]):
            mz = zones["mid_zones"][i]
            if mz["lo"] >= x_max:
                continue
            fig.add_vrect(x0=mz["lo"], x1=min(mz["hi"], x_max),
                          fillcolor=ZONE_COLORS_PALETTE["mid_zone"][i], opacity=0.20,
                          line_width=0, layer="below")
    if thresholds[-1] < x_max:
        fig.add_vrect(x0=thresholds[-1], x1=x_max,
                      fillcolor=ZONE_COLORS_PALETTE["past_top"], opacity=0.18,
                      line_width=0, layer="below")
    for T in thresholds:
        if T <= x_max:
            fig.add_vline(x=T, line=dict(color="black", width=1.2, dash="dot"))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Bill-value histogram",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis_title="Bill value (₹)",
        yaxis_title="Bills /month",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
    )
    return fig


def cdf_figure(cdf_df: pd.DataFrame, thresholds: list[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cdf_df["threshold"],
        y=cdf_df["pct_at_or_above"],
        mode="lines",
        line=dict(color=HEADING, width=2),
        name="% bills ≥ threshold",
        hovertemplate="≥ ₹%{x}: %{y:.2f}%<extra></extra>",
    ))
    x_max = cdf_df["threshold"].max()
    for i, T in enumerate(thresholds):
        if T <= x_max:
            fig.add_vline(
                x=T,
                line=dict(color="black", width=1.2, dash="dot"),
                annotation_text=f"T{i + 1} = ₹{T}",
                annotation_position="top",
            )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Cumulative distribution",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis_title="Threshold (₹)",
        yaxis_title="% of bills ≥ threshold",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, range=[0, 100]),
        showlegend=False,
    )
    return fig


h_col, c_col = st.columns(2)
with h_col:
    st.plotly_chart(histogram_figure(hist_df, zones, show_tail), width="stretch")
with c_col:
    st.plotly_chart(cdf_figure(cdf_df, thresholds), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 7 — Trade-off explorer (vary the top tier)
# ---------------------------------------------------------------------------
top_label = f"T{n_tiers}"
if n_tiers == 1:
    fixed_label = "(no other tiers)"
elif n_tiers == 2:
    fixed_label = "T1"
else:
    fixed_label = f"T1..T{n_tiers - 1}"
st.header(f"Trade-off explorer (vary {top_label}, hold {fixed_label})")

T_TOP_GRID = [699, 799, 899, 999, 1199]
top_held = thresholds[:-1] if n_tiers > 1 else []
floor = (top_held[-1] + 100) if top_held else 200
candidates = [t for t in T_TOP_GRID if t >= floor]

current_top = thresholds[-1]
rows = []
for t_top in candidates:
    combo = top_held + [t_top]
    z = zone_counts(vd, N_BILLS, combo, reach)
    lf = expected_lift(z, response_rate, gap_aware)
    s = score_pair(z, lf, alpha)
    top_tier = z["tiers"][-1]
    rows.append({
        f"{top_label}": fmt_inr(t_top),
        f"% in reach {top_label}": round(top_tier["pct_in_reach"], 2),
        f"Avg gap to {top_label}": fmt_inr(top_tier["avg_gap"]) if not np.isnan(top_tier["avg_gap"]) else "—",
        f"% past {top_label} (top-tier free)": round(z["pct_past_top"], 2),
        "Total waste %": round(s["wasted_carrot_pct"], 2),
        "Total expected lift %": round(lf["total"], 2),
        "Adjusted (expected)": round(s["adjusted_expected"], 2),
        "_t_int": t_top,
    })

if rows:
    df_to = pd.DataFrame(rows)
    current_mask = df_to["_t_int"] == current_top

    def highlight_current(row):
        return [
            "background-color: #FFF4CE; font-weight: 600" if current_mask.iloc[row.name] else ""
            for _ in row
        ]

    styled = df_to.drop(columns=["_t_int"]).style.apply(highlight_current, axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")
else:
    st.info(f"No valid candidates: {top_label} must exceed ₹{floor - 1}. Adjust earlier tiers down.")

st.divider()

# ---------------------------------------------------------------------------
# Section 8 — α-sensitivity
# ---------------------------------------------------------------------------
st.header("α-sensitivity")

alpha_grid = np.arange(0.0, 3.001, 0.25)
adj_expected_curve = [lift["total"] - a * sc["wasted_carrot_pct"] for a in alpha_grid]

alpha_fig = go.Figure()
alpha_fig.add_trace(go.Scatter(
    x=alpha_grid,
    y=adj_expected_curve,
    mode="lines+markers",
    line=dict(color=HEADING, width=2),
    marker=dict(size=6),
    hovertemplate="α=%{x:.2f}: adjusted_expected=%{y:.2f}<extra></extra>",
))
alpha_fig.add_vline(
    x=alpha,
    line=dict(color="#F8696B", width=1.5, dash="dot"),
    annotation_text=f"current α = {alpha:.2f}",
    annotation_position="top",
)
alpha_fig.add_hline(y=0, line=dict(color="#999999", width=1))
alpha_fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=320,
    margin=dict(l=10, r=10, t=30, b=40),
    xaxis_title="α (waste penalty)",
    yaxis_title="Adjusted (expected)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR),
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
    showlegend=False,
)
st.plotly_chart(alpha_fig, width="stretch")

threshold_str = ", ".join(f"T{i + 1}=₹{T}" for i, T in enumerate(thresholds))
st.caption(
    f"At the current ladder ({threshold_str}, reach = {int(reach * 100)}%, "
    f"response = {response_pct}%), expected_lift = {lift['total']:.2f}%, "
    f"wasted = {sc['wasted_carrot_pct']:.2f}%. "
    "Adjusted (expected) is linear in α and crosses zero where waste starts dominating expected lift."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 9 — Diagnostic footer
# ---------------------------------------------------------------------------
git_rev = get_git_rev()
st.markdown(
    f"<div style='font-size:0.78rem;color:{MUTED}'>Data generated "
    f"<code>{topline['generated_at']}</code> from "
    f"<code>prod2-generico.sales</code>. App build commit "
    f"<code>{git_rev}</code>.</div>",
    unsafe_allow_html=True,
)
