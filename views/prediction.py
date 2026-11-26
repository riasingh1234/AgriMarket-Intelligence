import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an, forecasting as fc, ui
from utils.config import COLORS, PALETTE, PRICE
from utils.data_loader import load_features, valid_prices


@st.cache_data(show_spinner=False)
def _run(series, horizon, models):
    return fc.run_forecast(series, horizon, models)


def render():
    ui.page_header("Price Prediction", "Commodity-wise and state-wise forecasts with honest, leakage-free evaluation")
    df = valid_prices(load_features())
    mode = st.radio("Forecast type", ["Commodity-wise (average of all states)", "State-wise (commodity within one state)"],
                    horizontal=True, key="pr_mode")
    c1, c2, c3 = st.columns(3)
    with c1:
        commodity = ui.commodity_picker(df, "pr_commodity")
    d0 = df[df["commodity"] == commodity]
    state_mode = mode.startswith("State")
    with c2:
        if state_mode:
            state = ui.state_picker(df, commodity, "pr_state", allow_all=False)
        else:
            st.selectbox("State", ["All States"], disabled=True, key="pr_state_dis")
            state = None
    with c3:
        horizon = st.selectbox("Forecast period (months)", [3, 6, 12], index=1, key="pr_h")
    models = st.multiselect("Models", fc.ML_MODELS, default=fc.ML_MODELS, key="pr_models")
    if not models:
        return ui.empty_state("Select at least one model.")

    if state_mode:
        counts = d0.groupby("state")["month"].nunique().sort_values(ascending=False).rename("months_of_data").reset_index()
        counts["forecastable"] = counts["months_of_data"] >= fc.MIN_OBS
        with st.expander("History available per state for this commodity"):
            ui.table(counts)
        raw = d0[d0["state"] == state].groupby("month")[PRICE].mean()
        label = f"{commodity} - {state}"
    else:
        raw = an.monthly_agg(d0)[PRICE].dropna()
        label = f"{commodity} - all-state average"

    series, info = fc.prepare_series(raw)
    if series.empty:
        return ui.empty_state("No price history for this selection.")
    with st.spinner("Training models and evaluating on the hold-out period..."):
        res = _run(series, horizon, tuple(models))
    if not res["ok"]:
        return st.warning(res["message"] + " Try another state or the commodity-wise forecast.")

    notes = list(res["warnings"])
    if info["gaps_filled"]:
        notes.append(f"{info['gaps_filled']} missing month(s) were linearly interpolated (gaps up to 3 months only).")
    if info["months_dropped"]:
        notes.append(f"{info['months_dropped']} early month(s) were ignored because of a longer data gap.")
    if series.index[-1] < d0["month"].max() - pd.DateOffset(months=2):
        notes.append(f"Series ends {ui.fmt_month(series.index[-1])}, earlier than the dataset's latest month.")
    for n in notes:
        st.warning(n)

    m = res["metrics"]
    best = res["best_model"]
    gain = (1 - res["best_rmse"] / res["baseline_rmse"]) * 100 if res["baseline_rmse"] else float("nan")
    ui.kpi_row([
        ("Observations", f"{len(series)}", f"{ui.fmt_month(series.index[0])} - {ui.fmt_month(series.index[-1])}"),
        ("Train / test months", f"{res['n_train']} / {res['test_size']}", "chronological split"),
        ("Best model", best, f"RMSE {res['best_rmse']:,.1f}"),
        ("vs naive baseline", ui.fmt_pct(-gain) if gain == gain else "–", "RMSE change (negative = better)"),
        (f"Forecast {horizon}m end", ui.fmt_inr(res["future"][best].iloc[-1]) if best in res["future"] else "–", ui.fmt_month(res["future"].index[-1])),
    ])

    show = [x for x in models if x in res["future"]]
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, name="Actual", line=dict(color=COLORS["navy"], width=3)))
    ts, te = res["test_pred"].index[0], res["test_pred"].index[-1]
    fig.add_vrect(x0=ts.to_pydatetime(), x1=te.to_pydatetime(), fillcolor="#EAF1FB", opacity=0.6, line_width=0, layer="below")
    fig.add_scatter(x=res["test_pred"].index, y=res["test_pred"][fc.NAIVE], name="Naive baseline (test)",
                    line=dict(color=COLORS["grey"], dash="dot"))
    for i, name in enumerate(show):
        col = PALETTE[(i + 2) % len(PALETTE)]
        fig.add_scatter(x=res["test_pred"].index, y=res["test_pred"][name], name=f"{name} (test)", line=dict(color=col, dash="dash"))
        fig.add_scatter(x=res["future"].index, y=res["future"][name], name=f"{name} forecast", mode="lines+markers",
                        line=dict(color=col, width=2.5))
        if name in res["lower"].columns:
            fig.add_scatter(x=list(res["upper"].index) + list(res["lower"].index[::-1]),
                            y=list(res["upper"][name]) + list(res["lower"][name][::-1]), fill="toself",
                            fillcolor="rgba(76,155,232,0.15)", line=dict(width=0), name="ARIMA 80% interval", hoverinfo="skip")
    fig.update_yaxes(title="Price (Rs/qtl)")
    ui.plot(ui.style_fig(fig, f"Actual vs predicted - {label} (shaded = hold-out test window)", 500))

    ui.section("Model evaluation on the hold-out test window")
    ui.table(m)
    if best in (fc.NAIVE,) or gain <= 0:
        st.info("No model clearly beats the naive baseline here, so prices for this series are hard to forecast. "
                "Treat the forecast as indicative only.")
    else:
        st.success(f"{best} improves on the naive baseline by {gain:.1f}% (RMSE).")
    st.caption("R² can be negative on short test windows: it means the model does worse than predicting the test-period mean.")

    ui.section("Forecast")
    out = res["future"][show + [fc.NAIVE]].copy()
    out.index.name = "month"
    out = out.reset_index()
    ui.table(out)
    ui.download_csv(out, "Download forecast (CSV)", f"forecast_{commodity[:20]}_{horizon}m.csv", "pr_dl")
    ui.download_csv(m, "Download model metrics (CSV)", "model_metrics.csv", "pr_dl2")
    with st.expander("How the evaluation avoids data leakage"):
        st.markdown(
            "- The **last few months** (about 20%, between 6 and 12) are held out as the test set.\n"
            "- Models are fitted on earlier months only, then forecast the whole test window **recursively**, "
            "so test actuals are never used as inputs.\n"
            "- Features are past prices (lags 1, 2, 3, 6) and the calendar month only.\n"
            "- For the future forecast, models are refitted on the full history.\n"
            "- Baselines: last observed value (naive) and same month last year (seasonal naive).")
