import plotly.express as px
import streamlit as st

from utils import analytics as an, ui
from utils.config import ARR, COLORS, PRICE
from utils.data_loader import load_features, valid_prices

TYPE_COLORS = {"Price Spike": COLORS["red"], "Price Drop": COLORS["blue"],
               "Arrivals Surge": COLORS["accent"], "Arrivals Drop": COLORS["green"]}


@st.cache_data(show_spinner=False)
def compute(df, z, pct):
    return an.detect_anomalies(df, z_threshold=z, min_abs_pct=pct)


def render():
    ui.page_header("Anomaly Detection", "Unusual month-on-month price increases, decreases and arrival changes")
    df = valid_prices(load_features())
    c1, c2, c3 = st.columns(3)
    with c1:
        z = st.slider("Sensitivity (robust z-score)", 2.5, 8.0, 4.0, 0.5, key="an_z",
                      help="Higher = fewer, more extreme anomalies")
    with c2:
        pct = st.slider("Minimum % change", 5, 100, 25, 5, key="an_pct")
    with c3:
        types = st.multiselect("Anomaly types", list(TYPE_COLORS), default=list(TYPE_COLORS), key="an_types")
    with st.spinner("Scanning all commodity-state series..."):
        res = compute(df, z, pct)
    if res.empty:
        return ui.empty_state("No anomalies found. Try lowering the sensitivity or minimum % change.")

    f1, f2 = st.columns(2)
    with f1:
        comm = st.multiselect("Commodity", sorted(res["commodity"].unique()), key="an_c")
    with f2:
        stt = st.multiselect("State", sorted(res["state"].unique()), key="an_s")
    out = res[res["anomaly_type"].isin(types)]
    if comm:
        out = out[out["commodity"].isin(comm)]
    if stt:
        out = out[out["state"].isin(stt)]
    if out.empty:
        return ui.empty_state()

    cnt = out["anomaly_type"].value_counts()
    ui.kpi_row([
        ("Total anomalies", f"{len(out):,}", f"of {len(df):,} price records"),
        ("Price spikes", f"{cnt.get('Price Spike', 0):,}", "unusual increases"),
        ("Price drops", f"{cnt.get('Price Drop', 0):,}", "unusual decreases"),
        ("Arrival changes", f"{cnt.get('Arrivals Surge', 0) + cnt.get('Arrivals Drop', 0):,}", "surges and drops"),
    ])

    a, b = st.columns(2)
    with a:
        t = out.groupby([out["month"].dt.to_period("Q").dt.to_timestamp(), "anomaly_type"]).size().reset_index(name="count")
        t.columns = ["quarter", "anomaly_type", "count"]
        fig = px.bar(t, x="quarter", y="count", color="anomaly_type", color_discrete_map=TYPE_COLORS)
        fig.update_layout(barmode="stack", hovermode="closest")
        ui.plot(ui.style_fig(fig, "Anomalies over time (per quarter)", 380))
    with b:
        t = out.groupby("commodity").size().sort_values().reset_index(name="count")
        fig = px.bar(t, x="count", y="commodity", orientation="h")
        fig.update_traces(marker_color=COLORS["blue"])
        ui.plot(ui.style_fig(fig, "Anomalies by commodity", 380))

    ui.section("Inspect a series")
    c1, c2 = st.columns(2)
    with c1:
        sc = st.selectbox("Commodity", sorted(df["commodity"].unique()), key="an_sc")
    with c2:
        ss = ui.state_picker(df, sc, "an_ss", allow_all=False)
    s = df[(df["commodity"] == sc) & (df["state"] == ss)]
    marks = res[(res["commodity"] == sc) & (res["state"] == ss) & res["anomaly_type"].isin(types)]
    fig = px.line(s, x="month", y=PRICE, labels={PRICE: "Price (Rs/qtl)", "month": ""})
    fig.update_traces(line_color=COLORS["navy"])
    if not marks.empty:
        fig.add_scatter(x=marks["month"], y=marks[PRICE], mode="markers", name="Anomaly",
                        marker=dict(size=12, color=COLORS["red"], symbol="diamond"))
    ui.plot(ui.style_fig(fig, f"{sc} - {ss}", 380))

    ui.section("Anomaly list (largest first)")
    ui.table(out, height=380)
    ui.download_csv(out, "Download anomalies (CSV)", "anomalies.csv", "an_dl")
    st.caption("Method: change vs the previous month (consecutive months only) is compared with that "
               "commodity-state's own history using a robust z-score (median/MAD).")
