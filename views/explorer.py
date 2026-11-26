import pandas as pd
import streamlit as st

from utils import analytics as an, ui
from utils.config import ARR, MONTH_NAMES, PRICE
from utils.data_loader import (load_features, load_full, load_quality, load_selected,
                               load_stats, valid_prices)

DATASETS = {
    "Feature-engineered data (15 commodities)": load_features,
    "Selected 15 commodities": load_selected,
    "Cleaned data (all commodities)": load_full,
    "Commodity statistics": load_stats,
    "Data quality report": load_quality,
}
MAX_PREVIEW = 2000


@st.cache_data(show_spinner="Building reports...")
def build_reports():
    feat = load_features()
    v = valid_prices(feat)
    smap = an.season_map(feat)
    reports = {}
    reports["Commodity summary"] = an.group_stats(v, "commodity", PRICE)
    reports["State summary"] = an.group_stats(v, "state", PRICE)
    reports["Commodity x State summary"] = an.group_stats(v, ["commodity", "state"], PRICE)
    yearly = v.groupby(["commodity", "year"])[PRICE].mean().unstack().round(2).reset_index()
    yearly.columns = [str(c) for c in yearly.columns]
    reports["Yearly average price"] = yearly
    rows = []
    for c, g in v.groupby("commodity"):
        idx = an.seasonal_index(an.seasonal_frame(an.monthly_agg(g), smap))
        if idx.dropna().empty:
            continue
        rows.append({"commodity": c, **{m: round(x, 1) for m, x in zip(MONTH_NAMES, idx.values)}})
    reports["Seasonal index (100 = yearly avg)"] = pd.DataFrame(rows)
    reports["Anomalies (default settings)"] = an.detect_anomalies(v)
    return reports


def render():
    ui.page_header("Data Explorer & Reports", "Search, filter, preview and download datasets and analysis reports")
    tab1, tab2 = st.tabs(["Data explorer", "Analysis reports"])

    with tab1:
        name = st.selectbox("Dataset", list(DATASETS), key="ex_ds")
        df = DATASETS[name]()
        out = df
        c1, c2, c3 = st.columns(3)
        if "commodity" in df.columns:
            with c1:
                sel = st.multiselect("Commodity", sorted(df["commodity"].dropna().unique()), key="ex_c")
            if sel:
                out = out[out["commodity"].isin(sel)]
        if "state" in df.columns:
            with c2:
                sel = st.multiselect("State", sorted(df["state"].dropna().unique()), key="ex_s")
            if sel:
                out = out[out["state"].isin(sel)]
        if "group" in df.columns:
            with c3:
                sel = st.multiselect("Group", sorted(df["group"].dropna().unique()), key="ex_g")
            if sel:
                out = out[out["group"].isin(sel)]
        if "month" in df.columns:
            start, end = ui.month_range(df, "ex_range")
            out = an.filter_df(out, start=start, end=end)
        text = st.text_input("Search text (commodity, state or group)", key="ex_q")
        if text.strip():
            mask = pd.Series(False, index=out.index)
            for col in ("commodity", "state", "group"):
                if col in out.columns:
                    mask |= out[col].astype(str).str.contains(text.strip(), case=False, regex=False)
            if not {"commodity", "state", "group"} & set(out.columns):
                mask = out.astype(str).apply(lambda c: c.str.contains(text.strip(), case=False, regex=False)).any(axis=1)
            out = out[mask]
        only_valid = st.checkbox("Hide rows with missing price", value=False, key="ex_valid") if PRICE in out.columns else False
        if only_valid:
            out = out.dropna(subset=[PRICE])

        ui.kpi_row([("Rows matching", f"{len(out):,}", f"of {len(df):,}"), ("Columns", f"{out.shape[1]}", name)])
        if out.empty:
            ui.empty_state()
        else:
            if len(out) > MAX_PREVIEW:
                st.caption(f"Showing the first {MAX_PREVIEW:,} rows. The download contains all {len(out):,} rows.")
            ui.table(out.head(MAX_PREVIEW), height=420)
            ui.download_csv(out, "Download filtered data (CSV)", "filtered_data.csv", "ex_dl")

    with tab2:
        try:
            reports = build_reports()
        except Exception as exc:
            return st.error(f"Could not build reports: {exc}")
        for i, (title, rep) in enumerate(reports.items()):
            with st.expander(f"{title} ({len(rep):,} rows)"):
                if rep.empty:
                    st.info("No rows for this report.")
                    continue
                ui.table(rep.head(200), height=300)
                fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "").replace("=", "")
                ui.download_csv(rep, "Download report (CSV)", f"{fname}.csv", f"rep_{i}")
