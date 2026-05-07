"""Cached loaders for the pre-computed summary files in data/.

These are the only file reads in the runtime app. Slider changes do not
re-trigger them — they hit the @st.cache_data layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"


@st.cache_data
def load_topline() -> dict:
    return json.loads((DATA / "topline.json").read_text())


@st.cache_data
def load_percentiles() -> dict:
    return json.loads((DATA / "percentiles.json").read_text())


@st.cache_data
def load_histogram_chart() -> pd.DataFrame:
    return pd.read_parquet(DATA / "histogram_chart.parquet")


@st.cache_data
def load_cdf_chart() -> pd.DataFrame:
    return pd.read_parquet(DATA / "cdf_chart.parquet")


@st.cache_data
def load_value_distribution() -> pd.DataFrame:
    return pd.read_parquet(DATA / "value_distribution.parquet")
