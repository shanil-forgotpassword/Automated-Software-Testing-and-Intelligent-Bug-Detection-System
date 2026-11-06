from __future__ import annotations

import logging

import streamlit as st

from codeguard.ui import render_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(
    page_title="CodeGuard AI",
    page_icon="assets/codeguard-logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_app()
