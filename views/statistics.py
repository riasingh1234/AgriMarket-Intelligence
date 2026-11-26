import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, ui
from utils.config import ARR, COLORS, PRICE
from utils.data_loader import load_features, valid_prices


def render():
    ui.page_header("Statistical Analysis", "Descriptive statistics, volatility, outliers and correlations")
    df = valid_prices(load_features())
    tab1, tab2, tab3, tab4 = st.tabs(["Descriptive statistics", "Volatility", "Outliers", "Correlations"])

    with tab1:
        metric = st.radio("Variable", [PRICE, ARR], horizontal=True, key="st_var")
        by = st.radio("Group by", ["commodity", "state"], horizontal=True, key="st_by")
        data = df.dropna(subset=[metric])
        res = an.group_stats(data, by, metric)
        if res.empty:
            return ui.empty_state()
        ui.table(res)
        ui.download_csv(res, "Download statistics (CSV)", f"stats_{by}_{metric}.csv", "st_dl")

    with tab2:
        res = an.group_stats(df, "commodity", PRICE).sort_values("volatility_cv_pct", ascending=False)
        fig = px.bar(res, x="commodity", y="volatility_cv_pct", labels={"volatility_cv_pct": "CV (%)", "commodity": ""})
        fig.update_traces(marker_color=COLORS["blue"])
        ui.plot(ui.style_fig(fig, "Price volatility (coefficient of variation) by commodity", 400))
        commodity = ui.commodity_picker(df, "st_vol_c")
        ms = an.with_changes(an.monthly_agg(df[df["commodity"] == commodity]))
        roll = ms["price_mom_pct"].rolling(6, min_periods=4).std()
        fig = go.Figure(go.Scatter(x=roll.index, y=roll.values, line=dict(color=COLORS["navy"], width=2.5), fill="tozeroy"))
        fig.update_yaxes(title="Std. dev. of monthly % change")
        ui.plot(ui.style_fig(fig, f"{commodity}: rolling 6-month volatility", 380))

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            commodity = ui.commodity_picker(df, "st_out_c")
        with c2:
            state = ui.state_picker(df, commodity, "st_out_s")
        sub = ui.apply_state(df[df["commodity"] == commodity], state)
        if len(sub) < 8:
            return ui.empty_state("Not enough records for outlier detection.")
        mask = an.iqr_outlier_mask(sub[PRICE])
        ui.kpi_row([("Records", f"{len(sub):,}", "valid prices"), ("Outliers (IQR rule)", f"{int(mask.sum())}", f"{mask.mean() * 100:.1f}% of records")])
        fig = px.box(sub, y=PRICE, points="outliers", labels={PRICE: "Price (Rs/qtl)"})
        ui.plot(ui.style_fig(fig, "Box plot with outliers", 400))
        out = sub[mask][["month", "commodity", "state", PRICE, ARR]].sort_values(PRICE, ascending=False)
        ui.table(out, height=300)
        ui.download_csv(out, "Download outliers (CSV)", "outliers.csv", "st_out_dl")

    with tab4:
        mode = st.radio("Correlation between", ["Price & arrivals features", "Commodity prices"], horizontal=True, key="st_cm")
        if mode == "Price & arrivals features":
            commodity = ui.commodity_picker(df, "st_corr_c")
            cols = [PRICE, ARR, "price_lag_1", "price_lag_3", "price_lag_6", "price_ma_3", "price_ma_6",
                    "arrivals_lag_1", "arrivals_ma_3", "price_mom_pct", "arrivals_mom_pct"]
            corr = df[df["commodity"] == commodity][cols].corr()
        else:
            piv = df.groupby(["month", "commodity"])[PRICE].mean().unstack()
            corr = piv.corr(min_periods=12)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
        ui.plot(ui.style_fig(fig, "Correlation heatmap (Pearson)", 600))
