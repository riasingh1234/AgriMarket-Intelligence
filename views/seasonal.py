import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui
from utils.config import COLORS, MONTH_NAMES, PRICE, SEASON_ORDER
from utils.data_loader import load_features, valid_prices


def render():
    ui.page_header("Seasonal Analysis", "Monthly patterns, seasonal price variation and year-wise comparison")
    feat = load_features()
    df = valid_prices(feat)
    c1, c2 = st.columns(2)
    with c1:
        commodity = ui.commodity_picker(df, "se_commodity")
    with c2:
        state = ui.state_picker(df, commodity, "se_state")
    sub = ui.apply_state(df[df["commodity"] == commodity], state)
    d = an.seasonal_frame(an.monthly_agg(sub), an.season_map(feat))
    if len(d) < 12:
        return ui.empty_state("At least 12 months of data are needed for seasonal analysis.")

    idx = an.seasonal_index(d)
    if idx.dropna().empty:
        return ui.empty_state("Not enough complete years to compute a seasonal index.")
    peak, trough = idx.idxmax(), idx.idxmin()
    sea = d.groupby("season")[PRICE].mean().reindex(SEASON_ORDER)
    ui.kpi_row([
        ("Seasonal peak", MONTH_NAMES[peak - 1], f"index {idx.max():.1f}"),
        ("Seasonal low", MONTH_NAMES[trough - 1], f"index {idx.min():.1f}"),
        ("Seasonal swing", f"{idx.max() - idx.min():.1f} pts", "peak minus low index"),
        ("Priciest season", str(sea.idxmax()), ui.fmt_inr(sea.max())),
    ])

    a, b = st.columns(2)
    with a:
        colors = [COLORS["accent"] if i + 1 == peak else COLORS["blue"] for i in range(12)]
        fig = go.Figure(go.Bar(x=MONTH_NAMES, y=idx.values, marker_color=colors))
        fig.add_hline(y=100, line_dash="dash", line_color=COLORS["grey"])
        fig.update_yaxes(title="Seasonal index (100 = yearly average)")
        ui.plot(ui.style_fig(fig, "Seasonal price index by month", 400))
    with b:
        fig = px.box(d, x="season", y=PRICE, category_orders={"season": SEASON_ORDER},
                     labels={PRICE: "Price (Rs/qtl)", "season": ""})
        ui.plot(ui.style_fig(fig, "Price distribution by season", 400))

    ui.section("Year-wise comparison")
    fig = px.line(d.reset_index(), x="month_name", y=PRICE, color=d["year"].astype(str).values,
                  category_orders={"month_name": MONTH_NAMES}, markers=True,
                  labels={PRICE: "Price (Rs/qtl)", "month_name": "", "color": "Year"})
    ui.plot(ui.style_fig(fig, "Monthly price by year", 430))

    heat = d.pivot_table(index="year", columns="month_num", values=PRICE).reindex(columns=range(1, 13))
    heat.columns = MONTH_NAMES
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Rs/qtl"))
    ui.plot(ui.style_fig(fig, "Average price heatmap (year x month)", 380))

    ui.insights([f"Prices of {commodity} tend to peak around {MONTH_NAMES[peak - 1]} and bottom out around "
                 f"{MONTH_NAMES[trough - 1]} (index {idx.max():.0f} vs {idx.min():.0f}, trend removed)."])
    out = idx.rename("seasonal_index").reset_index().assign(month=lambda x: x["month_num"].map(lambda m: MONTH_NAMES[m - 1]))
    ui.download_csv(out, "Download seasonal index (CSV)", f"seasonal_{commodity[:20]}.csv", "se_dl")
