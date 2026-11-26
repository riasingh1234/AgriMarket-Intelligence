"""AgriMarket Intelligence - run with:  streamlit run app.py"""
import traceback

import streamlit as st

st.set_page_config(page_title="AgriMarket Intelligence", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

from utils import ui                                    # noqa: E402
from utils.data_loader import DataError, load_features  # noqa: E402
from views import (anomaly, arrivals, dashboard, explorer, prediction,  # noqa: E402
                   price_analysis, seasonal, state_comparison, statistics)

PAGES = {
    "🏠 Dashboard": dashboard,
    "💰 Price Analysis": price_analysis,
    "🚚 Arrivals Analysis": arrivals,
    "🗺️ State Comparison": state_comparison,
    "🌦️ Seasonal Analysis": seasonal,
    "📊 Statistical Analysis": statistics,
    "🔮 Price Prediction": prediction,
    "🚨 Anomaly Detection": anomaly,
    "🗂️ Data Explorer & Reports": explorer,
}


def main():
    ui.inject_css()
    with st.sidebar:
        st.markdown('<div class="brand">🌾 AgriMarket Intelligence</div>'
                    '<div class="brand-sub">Agricultural Price Analytics, Supply Analysis and Forecasting System for India</div>',
                    unsafe_allow_html=True)
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    try:
        feat = load_features()
    except DataError as exc:
        st.error(f"Data problem: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()
    with st.sidebar:
        st.divider()
        st.caption(f"📅 {ui.fmt_month(feat['month'].min())} - {ui.fmt_month(feat['month'].max())}")
        st.caption(f"🌾 {feat['commodity'].nunique()} commodities · 🗺️ {feat['state'].nunique()} states")
        st.caption("Prices: Rs per quintal · Arrivals: tonnes")
    try:
        PAGES[choice].render()
    except DataError as exc:
        st.error(f"Data problem: {exc}")
    except Exception as exc:
        st.error(f"Something went wrong on this page: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


main()
