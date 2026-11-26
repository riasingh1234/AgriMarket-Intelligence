import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui
from utils.config import ARR, COLORS, PRICE
from utils.data_loader import load_features, load_full, load_quality, load_stats, valid_prices


def render():
    ui.page_header("Dashboard", "National overview of agricultural prices and arrivals across Indian states")
    feat, full, stats, quality = load_features(), load_full(), load_stats(), load_quality()
    valid = valid_prices(feat)
    if valid.empty:
        return ui.empty_state("No valid price records found in the dataset.")

    miss = quality.set_index("column")["missing_percent"].to_dict() if "column" in quality else {}
    top_price = stats.sort_values("mean_price", ascending=False).iloc[0] if len(stats) else None
    top_vol = stats.sort_values("coefficient_of_variation_pct", ascending=False).iloc[0] if len(stats) else None

    ui.kpi_row([
        ("Commodities (full data)", f"{full['commodity'].nunique():,}", f"{feat['commodity'].nunique()} focus commodities analysed"),
        ("States / UTs", f"{full['state'].nunique()}", "reporting markets"),
        ("Avg price (focus)", ui.fmt_inr(valid[PRICE].mean()), "Rs per quintal"),
        ("Avg arrivals / record", ui.fmt_compact(valid[ARR].mean()), "tonnes per state-month"),
        ("Records (focus)", f"{len(valid):,}", f"{ui.fmt_month(valid['month'].min())} - {ui.fmt_month(valid['month'].max())}"),
    ])
    ui.kpi_row([
        ("Highest avg price", top_price["commodity"] if top_price is not None else "–",
         ui.fmt_inr(top_price["mean_price"]) + " / qtl" if top_price is not None else ""),
        ("Most volatile", top_vol["commodity"] if top_vol is not None else "–",
         f"CV {top_vol['coefficient_of_variation_pct']:.1f}%" if top_vol is not None else ""),
        ("Missing price (full data)", f"{miss.get(PRICE, float('nan')):.1f}%", "left as missing, never imputed"),
        ("Focus-data missing", f"{feat[PRICE].isna().mean() * 100:.1f}%", "price & arrivals missing together"),
        ("Months covered", f"{feat['month'].nunique()}", "monthly observations"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        avg = valid.groupby("commodity")[PRICE].mean().sort_values()
        fig = go.Figure(go.Bar(x=avg.values, y=avg.index, orientation="h", marker_color=COLORS["blue"],
                               text=[ui.fmt_inr(v) for v in avg.values], textposition="outside"))
        fig.update_xaxes(title="Average price (Rs/qtl)")
        ui.plot(ui.style_fig(fig, "Average price by commodity", 480))
    with c2:
        idx = an.market_price_index(valid)
        fig = go.Figure(go.Scatter(x=idx.index, y=idx.values, mode="lines", fill="tozeroy",
                                   line=dict(color=COLORS["navy"], width=2.5), name="Index"))
        fig.update_yaxes(title="Index (100 = each commodity's own average)", rangemode="normal")
        ui.plot(ui.style_fig(fig, "Equal-weighted market price index", 480))

    c3, c4 = st.columns(2)
    with c3:
        st_arr = feat.dropna(subset=[ARR]).groupby("state")[ARR].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=st_arr.index, y=st_arr.values, labels={"x": "State", "y": "Total arrivals (t)"})
        fig.update_traces(marker_color=COLORS["sky"])
        ui.plot(ui.style_fig(fig, "Top 10 states by total arrivals", 400))
    with c4:
        grp = full.groupby("group").size().sort_values(ascending=False).reset_index(name="records")
        fig = px.pie(grp, names="group", values="records", hole=0.55)
        ui.plot(ui.style_fig(fig, "Records by commodity group (full data)", 400))

    ui.section("Market insights")
    lines = []
    if top_price is not None:
        lines.append(f"{top_price['commodity']} has the highest average price at {ui.fmt_inr(top_price['mean_price'])} per quintal.")
    if top_vol is not None:
        lines.append(f"{top_vol['commodity']} is the most volatile commodity (coefficient of variation {top_vol['coefficient_of_variation_pct']:.1f}%).")
    yoy = an.latest_yoy(valid)
    if not yoy.empty:
        up, dn = yoy.sort_values("yoy_pct").iloc[-1], yoy.sort_values("yoy_pct").iloc[0]
        lines.append(f"Year-on-year (latest month): biggest rise is {up['commodity']} ({ui.fmt_pct(up['yoy_pct'])}); "
                     f"biggest fall is {dn['commodity']} ({ui.fmt_pct(dn['yoy_pct'])}).")
    if not st_arr.empty:
        lines.append(f"{st_arr.index[0]} records the highest total arrivals among the focus commodities.")
    ui.insights(lines)

    if len(stats):
        with st.expander("Commodity statistics from the EDA stage (commodity_statistics.csv)"):
            ui.table(stats)
