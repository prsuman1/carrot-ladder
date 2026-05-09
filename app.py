"""Streamlit multipage entrypoint.

Two pages:
  1. Dashboard — the AOV Ladder Threshold dashboard (lives in dashboard_page.py).
  2. AI Chat   — Carrot Bot, powered by Gemini 2.5 Pro via OpenRouter
                  (lives in chat_page.py).

The page config + sidebar nav are owned here; each page module contains its own
sidebar widgets and main-area content.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AOV Ladder Threshold",
    layout="wide",
    initial_sidebar_state="expanded",
)

dashboard_page = st.Page(
    "dashboard_page.py",
    title="Dashboard",
    default=True,
)
chat_page = st.Page(
    "chat_page.py",
    title="AI Chat",
)

st.navigation([dashboard_page, chat_page]).run()
