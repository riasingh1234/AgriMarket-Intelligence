"""Shared UI helpers: theme CSS, KPI cards, filters, charts, tables, downloads."""
import html
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config import ALL_STATES, COLORS, PALETTE

_NEW_WIDTH_API = False
try:
    _v = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
    _NEW_WIDTH_API = _v >= (1, 50)
except Exception:
    pass


def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, #F2F7FF 0%, #FFFFFF 45%); }}
    [data-testid="stSidebar"] {{ background: #EAF1FB; border-right: 1px solid #D5E2F5; }}
    h1, h2, h3, h4 {{ color: {COLORS['navy']}; font-weight: 700; }}
    .block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
    .page-header {{ background: linear-gradient(90deg, {COLORS['navy']} 0%, {COLORS['blue']} 100%);
        color: white; padding: 20px 26px; border-radius: 14px; margin-bottom: 18px; }}
    .page-header h2 {{ color: white; margin: 0; font-size: 1.7rem; }}
    .page-header p {{ color: #DCE8FA; margin: 4px 0 0 0; font-size: 0.95rem; }}
    .kpi-card {{ background: white; border: 1px solid #D9E5F7; border-left: 5px solid {COLORS['blue']};
        border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(11,42,91,0.06);
        margin-bottom: 10px; min-height: 108px; }}
    .kpi-label {{ color: #5B6B86; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi-value {{ color: {COLORS['navy']}; font-size: 1.55rem; font-weight: 700; margin: 2px 0; line-height: 1.25; }}
    .kpi-sub {{ color: #6B7A94; font-size: 0.8rem; }}
    .insight {{ background: #EEF5FF; border: 1px solid #CFE0F7; border-radius: 10px;
        padding: 10px 14px; margin-bottom: 8px; color: {COLORS['navy']}; font-size: 0.92rem; }}
    .section-title {{ color: {COLORS['navy']}; font-size: 1.15rem; font-weight: 700;
        margin: 18px 0 6px 0; padding-bottom: 4px; border-bottom: 2px solid #D5E2F5; }}
    .brand {{ color: {COLORS['navy']}; font-weight: 800; font-size: 1.25rem; line-height: 1.2; }}
    .brand-sub {{ color: #5B6B86; font-size: 0.8rem; margin-bottom: 10px; }}
    [data-testid="stDataFrame"] {{ border: 1px solid #D9E5F7; border-radius: 10px; }}
    </style>""", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-header"><h2>{html.escape(title)}</h2>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, subtext)."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{html.escape(str(label))}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>'
            f'<div class="kpi-sub">{html.escape(str(sub))}</div></div>', unsafe_allow_html=True)


def insights(lines):
    for line in lines:
        st.markdown(f'<div class="insight">💡 {html.escape(line)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------- formatting
def fmt_inr(x, decimals=0):
    return "–" if x is None or pd.isna(x) else f"₹{x:,.{decimals}f}"


def fmt_compact(x):
    if x is None or pd.isna(x):
        return "–"
    a = abs(x)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{x / div:,.2f}{suffix}"
    return f"{x:,.0f}"


def fmt_pct(x, decimals=1):
    return "–" if x is None or pd.isna(x) else f"{x:+.{decimals}f}%"


def fmt_month(ts):
    return "–" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%b %Y")


# --------------------------------------------------------------- charts/tables
def style_fig(fig, title=None, height=420):
    fig.update_layout(
        template="plotly_white", height=height, colorway=PALETTE,
        font=dict(family="Segoe UI, Arial, sans-serif", color=COLORS["navy"], size=13),
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)))
    return fig


def plot(fig, key=None):
    if _NEW_WIDTH_API:
        try:
            return st.plotly_chart(fig, width="stretch", key=key)
        except Exception:
            pass
    return st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, height=None):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%b %Y")
    d = d.round(2)
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    if _NEW_WIDTH_API:
        try:
            return st.dataframe(d, width="stretch", **kw)
        except Exception:
            pass
    return st.dataframe(d, use_container_width=True, **kw)


def download_csv(df: pd.DataFrame, label: str, filename: str, key: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)


def empty_state(msg="No data available for the selected filters."):
    st.info(msg)


# --------------------------------------------------------------------- filters
def commodity_picker(df, key, label="Commodity", multi=False, default=None):
    opts = sorted(df["commodity"].dropna().unique())
    if multi:
        return st.multiselect(label, opts, default=default if default is not None else opts[:3], key=key)
    return st.selectbox(label, opts, key=key)


def state_picker(df, commodities, key, label="State", allow_all=True):
    commodities = [commodities] if isinstance(commodities, str) else list(commodities)
    opts = sorted(df[df["commodity"].isin(commodities)]["state"].dropna().unique())
    if allow_all:
        opts = [ALL_STATES] + opts
    return st.selectbox(label, opts, key=key)


def apply_state(df, state):
    return df if state in (None, ALL_STATES) else df[df["state"] == state]


def month_range(df, key, label="Period"):
    months = sorted(df["month"].dropna().unique())
    if len(months) < 2:
        return (months[0], months[0]) if months else (None, None)
    months = [pd.Timestamp(m) for m in months]
    return st.select_slider(label, options=months, value=(months[0], months[-1]),
                            format_func=lambda t: t.strftime("%b %Y"), key=key)
