import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui
from utils.config import ARR, COLORS, PRICE
from utils.data_loader import load_features, valid_prices


def render():
    ui.page_header("State Comparison", "Compare prices and arrivals across Indian states with rankings")
    df = valid_prices(load_features())
    c1, c2 = st.columns([1, 2])
    with c1:
        commodity = ui.commodity_picker(df, "sc_commodity")
    d0 = df[df["commodity"] == commodity]
    with c2:
        start, end = ui.month_range(d0, "sc_range")
    sub = an.filter_df(d0, start=start, end=end)
    if sub.empty:
        return ui.empty_state()

    rank = sub.groupby("state").agg(avg_price=(PRICE, "mean"), max_price=(PRICE, "max"),
                                    min_price=(PRICE, "min"), price_std=(PRICE, "std"),
                                    total_arrivals=(ARR, "sum"), avg_arrivals=(ARR, "mean"),
                                    months=("month", "nunique")).reset_index()
    rank["price_rank"] = rank["avg_price"].rank(ascending=False, method="min").astype(int)
    rank["arrivals_rank"] = rank["total_arrivals"].rank(ascending=False, method="min").astype(int)
    rank = rank.sort_values("price_rank")

    hi, lo = rank.iloc[0], rank.iloc[-1]
    ui.kpi_row([
        ("States reporting", f"{len(rank)}", commodity),
        ("Highest avg price", hi["state"], ui.fmt_inr(hi["avg_price"])),
        ("Lowest avg price", lo["state"], ui.fmt_inr(lo["avg_price"])),
        ("Price spread", ui.fmt_inr(hi["avg_price"] - lo["avg_price"]), "highest minus lowest state"),
        ("Top arrivals", rank.sort_values("total_arrivals", ascending=False).iloc[0]["state"], "highest total volume"),
    ])

    n = st.slider("States to show", 3, max(3, len(rank)), min(10, len(rank)), key="sc_n") if len(rank) > 3 else len(rank)
    a, b = st.columns(2)
    with a:
        t = rank.nlargest(n, "avg_price").sort_values("avg_price")
        fig = go.Figure(go.Bar(x=t["avg_price"], y=t["state"], orientation="h", marker_color=COLORS["blue"]))
        fig.update_xaxes(title="Average price (Rs/qtl)")
        ui.plot(ui.style_fig(fig, f"Top {n} states by average price", 420))
    with b:
        t = rank.nlargest(n, "total_arrivals").sort_values("total_arrivals")
        fig = go.Figure(go.Bar(x=t["total_arrivals"], y=t["state"], orientation="h", marker_color=COLORS["sky"]))
        fig.update_xaxes(title="Total arrivals (t)")
        ui.plot(ui.style_fig(fig, f"Top {n} states by total arrivals", 420))

    fig = px.scatter(rank, x="total_arrivals", y="avg_price", text="state", size="months",
                     labels={"total_arrivals": "Total arrivals (t)", "avg_price": "Average price (Rs/qtl)"})
    fig.update_traces(textposition="top center", marker_color=COLORS["blue"])
    ui.plot(ui.style_fig(fig, "States: average price vs total arrivals", 430))

    ui.section("State price trends")
    states = sorted(sub["state"].unique())
    pick = st.multiselect("Compare states", states, default=rank.head(4)["state"].tolist(), key="sc_states")
    if pick:
        t = sub[sub["state"].isin(pick)]
        fig = px.line(t, x="month", y=PRICE, color="state", labels={PRICE: "Price (Rs/qtl)", "month": ""})
        ui.plot(ui.style_fig(fig, "Monthly price by state", 420))

    ui.section("Price heatmap across commodities (state average / commodity average x 100)")
    full = an.filter_df(df, start=start, end=end)
    piv = full.groupby(["state", "commodity"])[PRICE].mean().unstack()
    rel = piv / piv.mean() * 100
    fig = px.imshow(rel, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color="Index"))
    ui.plot(ui.style_fig(fig, "Relative price index (100 = all-state average of that commodity)", 650))

    ui.section("Ranking table")
    ui.table(rank)
    ui.download_csv(rank, "Download state ranking (CSV)", f"state_ranking_{commodity[:20]}.csv", "sc_dl")
