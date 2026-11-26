import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import analytics as an, ui
from utils.config import ARR, COLORS, PRICE
from utils.data_loader import load_features, valid_prices


def render():
    ui.page_header("Arrivals Analysis", "Arrival trends, commodity comparison and price-versus-arrivals relationship")
    df = valid_prices(load_features()).dropna(subset=[ARR])
    c1, c2 = st.columns([2, 1])
    with c1:
        commodities = ui.commodity_picker(df, "ar_commodities", "Commodities to compare", multi=True)
    if not commodities:
        return ui.empty_state("Select at least one commodity.")
    with c2:
        state = ui.state_picker(df, commodities, "ar_state")
    start, end = ui.month_range(df[df["commodity"].isin(commodities)], "ar_range")
    sub = an.filter_df(ui.apply_state(df, state), commodities, start=start, end=end)
    if sub.empty:
        return ui.empty_state()

    ui.kpi_row([
        ("Total arrivals", ui.fmt_compact(sub[ARR].sum()), "tonnes"),
        ("Avg monthly arrivals", ui.fmt_compact(sub.groupby("month")[ARR].sum().mean()), "tonnes per month"),
        ("Peak month", ui.fmt_month(sub.groupby("month")[ARR].sum().idxmax()), "highest total arrivals"),
        ("Records", f"{len(sub):,}", "state-month observations"),
    ])

    logy = st.checkbox("Log scale (helps when commodities differ greatly in volume)", value=len(commodities) > 1, key="ar_log")
    trend = sub.groupby(["month", "commodity"])[ARR].sum().reset_index()
    fig = px.line(trend, x="month", y=ARR, color="commodity", labels={ARR: "Arrivals (t)", "month": ""})
    if logy:
        fig.update_yaxes(type="log")
    ui.plot(ui.style_fig(fig, "Monthly arrivals trend", 430))

    a, b = st.columns(2)
    comp = sub.groupby("commodity")[ARR].sum().sort_values(ascending=False).reset_index()
    with a:
        fig = px.bar(comp, x="commodity", y=ARR, labels={ARR: "Total arrivals (t)", "commodity": ""})
        fig.update_traces(marker_color=COLORS["blue"])
        ui.plot(ui.style_fig(fig, "Total arrivals by commodity", 380))
    with b:
        fig = px.box(sub, x="commodity", y=ARR, log_y=True, labels={ARR: "Arrivals per state-month (t)", "commodity": ""})
        ui.plot(ui.style_fig(fig, "Distribution of arrivals (log scale)", 380))

    ui.section("Price versus arrivals")
    pc = st.selectbox("Commodity for price-arrival analysis", commodities, key="ar_pc")
    p = an.filter_df(ui.apply_state(df, state), pc, start=start, end=end)
    p = p[p[ARR] > 0]
    if len(p) < 10:
        return ui.empty_state("Not enough records for a price-versus-arrivals analysis.")
    pear = np.log(p[ARR]).corr(p[PRICE])
    spear = p[ARR].corr(p[PRICE], method="spearman")
    ui.kpi_row([
        ("Pearson r (log arrivals vs price)", f"{pear:.2f}", "linear relationship"),
        ("Spearman rho", f"{spear:.2f}", "rank relationship"),
        ("Observations", f"{len(p):,}", "state-month points"),
    ])
    x, y = np.log10(p[ARR].to_numpy()), p[PRICE].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    fig = px.scatter(p, x=ARR, y=PRICE, color="state", log_x=True, opacity=0.6,
                     labels={ARR: "Arrivals (t, log scale)", PRICE: "Price (Rs/qtl)"})
    fig.add_trace(go.Scatter(x=10 ** xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                             line=dict(color=COLORS["navy"], width=3, dash="dash")))
    ui.plot(ui.style_fig(fig, f"{pc}: price vs arrivals", 450))
    strength = "weak" if abs(spear) < 0.3 else "moderate" if abs(spear) < 0.6 else "strong"
    direction = "negative (more supply, lower price)" if spear < 0 else "positive"
    ui.insights([f"The rank correlation is {spear:.2f}: a {strength} {direction} relationship. "
                 "Correlation does not prove that arrivals cause price changes."])

    ms = an.monthly_agg(p)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ms.index, y=ms[PRICE], name="Avg price", line=dict(color=COLORS["blue"], width=2.5)), secondary_y=False)
    fig.add_trace(go.Bar(x=ms.index, y=ms[ARR], name="Total arrivals", marker_color="rgba(76,155,232,0.35)"), secondary_y=True)
    fig.update_yaxes(title_text="Price (Rs/qtl)", secondary_y=False)
    fig.update_yaxes(title_text="Arrivals (t)", secondary_y=True)
    ui.plot(ui.style_fig(fig, "Price and arrivals over time", 400))
