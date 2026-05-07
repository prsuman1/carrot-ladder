"""AOV Ladder Threshold — Streamlit UI.

Reads only pre-computed summary files from data/. Never imports psycopg2.
Slider changes recompute zone metrics on a small in-memory histogram; file
loads are cached via @st.cache_data so they run only once.
"""

from __future__ import annotations

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
from metrics import score_pair, zone_counts

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AOV Ladder Threshold",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZONE_COLORS = {
    "unreachable": "#F8696B",
    "within_reach_T1": "#FFEB84",
    "mid_zone": "#D9D9D9",
    "within_reach_T2": "#A9D18E",
    "past_T2": "#9BC2E6",
}

# Theme detection — Streamlit ≥1.45 exposes st.context.theme.type
try:
    theme_base = st.context.theme.type
except Exception:
    theme_base = st.get_option("theme.base") or "light"
IS_DARK = theme_base == "dark"

PLOTLY_TEMPLATE = "plotly_dark" if IS_DARK else "plotly_white"
GRID_COLOR = "rgba(255,255,255,0.10)" if IS_DARK else "rgba(0,0,0,0.08)"
TEXT_COLOR = "#E6E6E6" if IS_DARK else "#222222"
HEADING    = "#9DC3E6" if IS_DARK else "#1F4E78"
MUTED      = "#A0A0A0" if IS_DARK else "#606060"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_inr(x: float) -> str:
    return f"₹{x:,.0f}"


def fmt_inr_cr(x: float) -> str:
    return f"₹{x / 1e7:,.2f} Cr"


def fmt_pct(x: float, dp: int = 1) -> str:
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
    """Linear interpolation: at x_pct% of top bills, what % of revenue?"""
    return float(np.interp(x_pct, curve["pct_bills"].values, curve["pct_revenue"].values))


def concentration_figure(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        line=dict(color=GRID_COLOR.replace("0.08", "0.35").replace("0.10", "0.35"), width=1, dash="dot"),
        name="Equal distribution",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=curve["pct_bills"], y=curve["pct_revenue"],
        mode="lines",
        line=dict(color=HEADING, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,78,120,0.10)" if not IS_DARK else "rgba(157,195,230,0.12)",
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
    st.subheader("Thresholds")
    T1 = st.slider("T1", min_value=200, max_value=999, value=399, step=25)
    T2 = st.slider("T2", min_value=500, max_value=2000, value=899, step=25)

    if T2 <= T1:
        new_T2 = min(T1 + 50, 2000)
        st.warning(f"T2 must be greater than T1. Auto-bumping T2 to ₹{new_T2}.")
        T2 = new_T2

    st.subheader("Reach window")
    reach_pct = st.slider(
        "Reach window",
        min_value=60, max_value=95, value=75, step=5,
        format="%d%%",
        label_visibility="collapsed",
        help="A bill is 'within reach' of T if reach × T ≤ bill < T. Larger reach % = stricter window.",
    )
    reach = reach_pct / 100.0

    st.subheader("Waste penalty (α)")
    alpha = st.slider("α", min_value=0.0, max_value=3.0, value=1.0, step=0.5)

# ---------------------------------------------------------------------------
# Live metric computation (small dataframe, fast)
# ---------------------------------------------------------------------------
m = zone_counts(vd, N_BILLS, T1, T2, reach)
sc = score_pair(m, alpha)

# ---------------------------------------------------------------------------
# Section 1 — Header
# ---------------------------------------------------------------------------
st.title("AOV Ladder Threshold")
st.caption(
    "Pick a (T1, T2) pair on the left. Every chart and number on this page "
    "reflects that pair on a ZRF-aware payable bill value."
)

# ---------------------------------------------------------------------------
# Section 2 — How this works (collapsed)
# ---------------------------------------------------------------------------
with st.expander("How this works", expanded=False):
    st.markdown(
        """
**Goal.** Two upsell thresholds, T1 and T2, drive a tiered carrot
("spend ₹X more to unlock Y"). This page shows how the bill-value
distribution — last 90 days, real sales only — maps onto five spending
zones for any (T1, T2) you pick.

**Payable bill value (ZRF-aware).**
- Normal line: `rate × net-quantity`.
- ZRD line (rupee discount coupon): same as above, then subtract `promo-discount`.
- ZRF line, qty > 1 (free item coupon): customer pays for `qty − 1` units.
- ZRF line, qty = 1: customer pays 0 — the only unit is the free one.

**Five zones (every bill falls into exactly one).**
1. **Unreachable.** Bill below the reach window for T1.
2. **Within reach T1.** Bill close enough to T1 to be nudged across.
3. **Past T1, mid-zone.** Already cleared T1 but too far below T2 to nudge.
4. **Within reach T2.** Close enough to T2 to be nudged across.
5. **Past T2.** Already at or above T2 — the carrot is "wasted" on them.

**Reach window.** A bill of value `b` is "within reach" of threshold `T`
if `reach × T ≤ b < T`. The reach toggle (75% / 80%) controls how
aggressive the nudge cone is.
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Data summary
# ---------------------------------------------------------------------------
st.header("Data summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("N bills", f"{topline['n_bills']:,}")
c2.metric("Total revenue", fmt_inr_cr(topline["total_revenue"]))
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
# Section 4 — Threshold zones diagram
# ---------------------------------------------------------------------------
st.header("Threshold zones")


def zones_figure(T1: int, T2: int, reach: float, m: dict, max_value: int = 1300) -> go.Figure:
    lo1 = m["lo1"]
    lo2 = m["lo2"]
    zones = [
        (
            0,
            lo1,
            f"Unreachable<br>{m['n_unreach']:,} bills",
            ZONE_COLORS["unreachable"],
        ),
        (
            lo1,
            T1,
            f"Within reach T1<br>[₹{lo1:.0f}, ₹{T1})<br>{m['n_within_reach_T1']:,} bills",
            ZONE_COLORS["within_reach_T1"],
        ),
        (
            T1,
            lo2,
            f"Past T1, mid-zone<br>[₹{T1}, ₹{lo2:.0f})<br>{m['n_mid_zone']:,} bills",
            ZONE_COLORS["mid_zone"],
        ),
        (
            lo2,
            T2,
            f"Within reach T2<br>[₹{lo2:.0f}, ₹{T2})<br>{m['n_within_reach_T2']:,} bills",
            ZONE_COLORS["within_reach_T2"],
        ),
        (
            T2,
            max_value,
            f"Past T2<br>(≥ ₹{T2})<br>{m['n_ge_T2']:,} bills",
            ZONE_COLORS["past_T2"],
        ),
    ]

    fig = go.Figure()
    for lo, hi, label, color in zones:
        width = max(hi - lo, 0)
        if width <= 0:
            continue
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[""],
                orientation="h",
                base=lo,
                marker=dict(color=color, line=dict(width=0)),
                text=label,
                textposition="inside",
                insidetextanchor="middle",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_vline(
        x=T1,
        line=dict(color="black", width=1.5),
        annotation_text=f"T1 = ₹{T1}",
        annotation_position="top",
    )
    fig.add_vline(
        x=T2,
        line=dict(color="black", width=1.5),
        annotation_text=f"T2 = ₹{T2}",
        annotation_position="top",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        height=200,
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis_title="Bill value (₹)",
        xaxis=dict(range=[0, max_value], showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(visible=False, showgrid=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
    )
    return fig


zones_max = max(int(T2 * 1.3), 1300)
st.plotly_chart(zones_figure(T1, T2, reach, m, max_value=zones_max), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Live metrics
# ---------------------------------------------------------------------------
st.header("Live metrics")

z1, z2, z3, z4, z5 = st.columns(5)
z1.metric("Unreachable", fmt_pct(m["pct_unreachable"]))
z2.metric("Within reach T1", fmt_pct(m["pct_within_reach_T1"]))
z3.metric("Past T1, mid-zone", fmt_pct(m["pct_mid_zone"]))
z4.metric("Within reach T2", fmt_pct(m["pct_within_reach_T2"]))
z5.metric("Past T2", fmt_pct(m["pct_ge_T2"]))

g1, g2 = st.columns(2)
g1.metric("Avg gap to T1 (within reach)", fmt_inr(m["avg_gap_T1"]) if not np.isnan(m["avg_gap_T1"]) else "—")
g2.metric("Avg gap to T2 (within reach)", fmt_inr(m["avg_gap_T2"]) if not np.isnan(m["avg_gap_T2"]) else "—")

s1, s2, s3 = st.columns(3)
s1.metric("Reach score", fmt_pct(sc["reach_score"], 2))
s2.metric("Wasted carrot %", fmt_pct(sc["wasted_carrot_pct"], 2))
s3.metric("Adjusted score", f"{sc['adjusted_score']:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Section 6 — Distribution charts
# ---------------------------------------------------------------------------
st.header("Distribution")

show_tail = st.checkbox("Include ₹2000–₹5000 tail", value=False)


def histogram_figure(hist_df: pd.DataFrame, T1: int, T2: int, m: dict, show_tail: bool) -> go.Figure:
    df = hist_df.copy()
    df = df[df["bin_hi"] != -1]  # drop overflow row
    if not show_tail:
        df = df[df["bin_hi"] <= 2000]
    df["mid"] = (df["bin_lo"] + df["bin_hi"]) / 2
    df["width"] = df["bin_hi"] - df["bin_lo"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["mid"],
            y=df["count"],
            width=df["width"],
            marker=dict(color="#9BC2E6", line=dict(width=0)),
            hovertemplate="₹%{customdata[0]}–₹%{customdata[1]}<br>Bills: %{y:,}<extra></extra>",
            customdata=df[["bin_lo", "bin_hi"]].values,
            showlegend=False,
        )
    )
    x_max = df["bin_hi"].max()
    # translucent bands for the four zone boundaries
    fig.add_vrect(
        x0=0, x1=m["lo1"],
        fillcolor=ZONE_COLORS["unreachable"], opacity=0.12, line_width=0, layer="below",
    )
    fig.add_vrect(
        x0=m["lo1"], x1=T1,
        fillcolor=ZONE_COLORS["within_reach_T1"], opacity=0.25, line_width=0, layer="below",
    )
    fig.add_vrect(
        x0=T1, x1=m["lo2"],
        fillcolor=ZONE_COLORS["mid_zone"], opacity=0.20, line_width=0, layer="below",
    )
    fig.add_vrect(
        x0=m["lo2"], x1=min(T2, x_max),
        fillcolor=ZONE_COLORS["within_reach_T2"], opacity=0.25, line_width=0, layer="below",
    )
    if T2 <= x_max:
        fig.add_vrect(
            x0=T2, x1=x_max,
            fillcolor=ZONE_COLORS["past_T2"], opacity=0.18, line_width=0, layer="below",
        )
    fig.add_vline(x=T1, line=dict(color="black", width=1.2, dash="dot"))
    fig.add_vline(x=T2, line=dict(color="black", width=1.2, dash="dot"))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Bill-value histogram",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis_title="Bill value (₹)",
        yaxis_title="Bills",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
    )
    return fig


def cdf_figure(cdf_df: pd.DataFrame, T1: int, T2: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf_df["threshold"],
            y=cdf_df["pct_at_or_above"],
            mode="lines",
            line=dict(color=HEADING, width=2),
            name="% bills ≥ threshold",
            hovertemplate="≥ ₹%{x}: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=T1, line=dict(color="black", width=1.2, dash="dot"),
                  annotation_text=f"T1 = ₹{T1}", annotation_position="top")
    if T2 <= cdf_df["threshold"].max():
        fig.add_vline(x=T2, line=dict(color="black", width=1.2, dash="dot"),
                      annotation_text=f"T2 = ₹{T2}", annotation_position="top")
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
    st.plotly_chart(
        histogram_figure(hist_df, T1, T2, m, show_tail),
        width="stretch",
    )
with c_col:
    st.plotly_chart(cdf_figure(cdf_df, T1, T2), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 7 — Trade-off explorer
# ---------------------------------------------------------------------------
st.header("Trade-off explorer (vary T2, hold T1)")

T2_grid = [699, 799, 899, 999, 1199]
rows = []
for t2 in T2_grid:
    if t2 <= T1:
        continue
    mm = zone_counts(vd, N_BILLS, T1, t2, reach)
    ss = score_pair(mm, alpha)
    rows.append({
        "T2": f"₹{t2}",
        "% within reach T2": round(mm["pct_within_reach_T2"], 2),
        "Avg gap to T2": fmt_inr(mm["avg_gap_T2"]) if not np.isnan(mm["avg_gap_T2"]) else "—",
        "% already ≥ T2": round(mm["pct_ge_T2"], 2),
        "Reach score": round(ss["reach_score"], 2),
        "Adjusted score": round(ss["adjusted_score"], 2),
        "_T2_int": t2,
    })

if rows:
    df_to = pd.DataFrame(rows)
    current = df_to["_T2_int"] == T2

    def highlight_current(row):
        return [
            "background-color: #FFF4CE; font-weight: 600" if current.iloc[row.name] else ""
            for _ in row
        ]

    styled = df_to.drop(columns=["_T2_int"]).style.apply(highlight_current, axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")
else:
    st.info("Current T1 is above all T2 candidates — adjust T1 lower.")

st.divider()

# ---------------------------------------------------------------------------
# Section 8 — α-sensitivity
# ---------------------------------------------------------------------------
st.header("α-sensitivity")

alpha_grid = np.arange(0.0, 3.001, 0.25)
adj_scores = [
    score_pair(m, a)["adjusted_score"] for a in alpha_grid
]

alpha_fig = go.Figure()
alpha_fig.add_trace(
    go.Scatter(
        x=alpha_grid,
        y=adj_scores,
        mode="lines+markers",
        line=dict(color=HEADING, width=2),
        marker=dict(size=6),
        hovertemplate="α=%{x:.2f}: adjusted=%{y:.2f}<extra></extra>",
    )
)
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
    yaxis_title="Adjusted score",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR),
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
    showlegend=False,
)
st.plotly_chart(alpha_fig, width="stretch")

st.caption(
    f"At the current pair (T1 = ₹{T1}, T2 = ₹{T2}, reach = {int(reach * 100)}%), "
    f"reach_score = {sc['reach_score']:.2f}, wasted = {sc['wasted_carrot_pct']:.2f}. "
    "Adjusted score crosses zero where the waste penalty starts dominating reach."
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
