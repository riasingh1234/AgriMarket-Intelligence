import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui
from utils.config import COLORS, PALETTE, PRICE
from utils.data_loader import load_features, valid_prices


def render():
    ui.page_header("Price Analysis", "Historical price trends, moving averages, extremes and monthly changes")
    df = valid_prices(load_features())
    c1, c2, c3 = st.columns([1.2, 1, 1.6])
    with c1:
        commodity = ui.commodity_picker(df, "pa_commodity")
    with c2:
        state = ui.state_picker(df, commodity, "pa_state")
    with c3:
        d0 = df[df["commodity"] == commodity]
        start, end = ui.month_range(d0, "pa_range")
    mas = st.multiselect("Moving averages (months)", [3, 6, 12], default=[3, 6], key="pa_ma")

    raw = an.filter_df(ui.apply_state(d0, state), start=start, end=end)
    if raw.empty:
        return ui.empty_state()
    ms = an.with_changes(an.monthly_agg(raw))
    s = ms[PRICE].dropna()
    hi, lo = raw.loc[raw[PRICE].idxmax()], raw.loc[raw[PRICE].idxmin()]
    change = (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 and s.iloc[0] else float("nan")

    ui.kpi_row([
        ("Latest price", ui.fmt_inr(s.iloc[-1]), ui.fmt_month(s.index[-1])),
        ("Average price", ui.fmt_inr(s.mean()), f"{len(s)} months"),
        ("Highest price", ui.fmt_inr(hi[PRICE]), f"{hi['state']}, {ui.fmt_month(hi['month'])}"),
        ("Lowest price", ui.fmt_inr(lo[PRICE]), f"{lo['state']}, {ui.fmt_month(lo['month'])}"),
        ("Change over period", ui.fmt_pct(change), "first vs latest month"),
    ])

    label = "All-state average" if state == "All States" else state
    fig = go.Figure(go.Scatter(x=ms.index, y=ms[PRICE], name="Price", mode="lines",
                               line=dict(color=COLORS["blue"], width=2.5), connectgaps=False))
    for i, w in enumerate(sorted(mas)):
        fig.add_scatter(x=ms.index, y=ms[PRICE].rolling(w, min_periods=w).mean(), name=f"{w}-month MA",
                        mode="lines", line=dict(width=2, dash="dash", color=PALETTE[i + 3]))
    fig.update_yaxes(title="Rs per quintal")
    ui.plot(ui.style_fig(fig, f"{commodity} - price trend ({label})", 450))

    a, b = st.columns(2)
    with a:
        mom = ms["price_mom_pct"].dropna()
        fig = go.Figure(go.Bar(x=mom.index, y=mom.values,
                               marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in mom.values]))
        fig.update_yaxes(title="% change")
        ui.plot(ui.style_fig(fig, "Month-on-month price change (%)", 380))
    with b:
        yoy = ms["price_yoy_pct"].dropna()
        if yoy.empty:
            ui.empty_state("Year-on-year change needs at least 13 months of data.")
        else:
            fig = go.Figure(go.Scatter(x=yoy.index, y=yoy.values, mode="lines", fill="tozeroy",
                                       line=dict(color=COLORS["navy"])))
            fig.update_yaxes(title="% change")
            ui.plot(ui.style_fig(fig, "Year-on-year price change (%)", 380))

    ui.section("Largest monthly moves")
    m = ms.dropna(subset=["price_mom_pct"]).reset_index()
    cols = ["month", PRICE, "price_mom_pct"]
    x, y = st.columns(2)
    with x:
        st.markdown("**Top 5 increases**")
        ui.table(m.nlargest(5, "price_mom_pct")[cols])
    with y:
        st.markdown("**Top 5 decreases**")
        ui.table(m.nsmallest(5, "price_mom_pct")[cols])

    ui.section("Monthly data")
    out = ms.reset_index().rename(columns={"index": "month"})[["month", PRICE, "price_mom_pct", "price_yoy_pct", "n_states"]]
    ui.table(out, height=320)
    ui.download_csv(out, "Download monthly price data (CSV)", f"price_{commodity[:20]}.csv", "pa_dl")
