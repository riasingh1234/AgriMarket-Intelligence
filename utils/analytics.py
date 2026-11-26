"""Pure pandas analytics helpers (no Streamlit code, easy to test)."""
import numpy as np
import pandas as pd

from utils.config import ARR, PRICE


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def filter_df(df, commodities=None, states=None, start=None, end=None):
    out = df
    commodities, states = _as_list(commodities), _as_list(states)
    if commodities:
        out = out[out["commodity"].isin(commodities)]
    if states:
        out = out[out["state"].isin(states)]
    if start is not None:
        out = out[out["month"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["month"] <= pd.Timestamp(end)]
    return out


def pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    """Percent change that never forward-fills missing months."""
    prev = s.shift(periods)
    out = (s / prev - 1) * 100
    return out.replace([np.inf, -np.inf], np.nan)


def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly series: mean price, total arrivals and number of reporting states."""
    if df.empty:
        return pd.DataFrame(columns=[PRICE, ARR, "n_states"])
    g = df.groupby("month")
    out = pd.DataFrame({
        PRICE: g[PRICE].mean(),
        ARR: g[ARR].sum(min_count=1),
        "n_states": g["state"].nunique(),
    })
    return out.sort_index().asfreq("MS")


def with_changes(ms: pd.DataFrame) -> pd.DataFrame:
    out = ms.copy()
    out["price_mom_pct"] = pct_change(out[PRICE], 1)
    out["price_yoy_pct"] = pct_change(out[PRICE], 12)
    out["arrivals_mom_pct"] = pct_change(out[ARR], 1)
    return out


def market_price_index(df: pd.DataFrame) -> pd.Series:
    """Equal-weighted index: each commodity's monthly mean price / its own average x 100."""
    m = df.dropna(subset=[PRICE]).groupby(["commodity", "month"])[PRICE].mean().reset_index()
    m["idx"] = m[PRICE] / m.groupby("commodity")[PRICE].transform("mean") * 100
    return m.groupby("month")["idx"].mean().sort_index()


def latest_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Latest month vs same month a year earlier, per commodity (all-state mean price)."""
    piv = df.dropna(subset=[PRICE]).groupby(["month", "commodity"])[PRICE].mean().unstack()
    if piv.empty:
        return pd.DataFrame(columns=["commodity", "yoy_pct"])
    last = piv.index.max()
    prev = last - pd.DateOffset(months=12)
    if prev not in piv.index:
        return pd.DataFrame(columns=["commodity", "yoy_pct"])
    yoy = ((piv.loc[last] / piv.loc[prev] - 1) * 100).dropna()
    return yoy.rename("yoy_pct").reset_index()


# ---------------------------------------------------------------- statistics
def group_stats(df: pd.DataFrame, by, col: str = PRICE) -> pd.DataFrame:
    """Mean, median, std, variance, volatility (CV %), IQR and IQR-rule outlier counts."""
    d = df.dropna(subset=[col])
    rows = []
    for key, x in d.groupby(by)[col]:
        if len(x) == 0:
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        mean = x.mean()
        row = dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,)))
        row.update({
            "records": len(x), "mean": mean, "median": x.median(), "std_dev": x.std(),
            "variance": x.var(), "min": x.min(), "max": x.max(),
            "volatility_cv_pct": (x.std() / mean * 100) if mean else np.nan,
            "q1": q1, "q3": q3, "iqr": iqr, "skewness": x.skew(),
            "outliers": n_out, "outlier_pct": n_out / len(x) * 100,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def iqr_outlier_mask(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


# ------------------------------------------------------------------ seasonal
def season_map(df: pd.DataFrame) -> dict:
    """month number -> season, taken from the 'season' column of the dataset."""
    d = df.dropna(subset=["season"])
    return d.groupby("month_num")["season"].agg(lambda x: x.mode().iloc[0]).to_dict()


def seasonal_frame(ms: pd.DataFrame, smap: dict) -> pd.DataFrame:
    d = ms.dropna(subset=[PRICE]).copy()
    d["year"] = d.index.year
    d["month_num"] = d.index.month
    d["month_name"] = d.index.strftime("%b")
    d["season"] = d["month_num"].map(smap)
    return d


def seasonal_index(d: pd.DataFrame) -> pd.Series:
    """Average of (month price / that year's mean price) x 100. Removes the long-term trend."""
    if d.empty:
        return pd.Series(dtype=float)
    cnt = d.groupby("year")[PRICE].transform("count")
    dd = d[cnt >= 6].copy()
    if dd.empty:
        return pd.Series(dtype=float)
    dd["idx"] = dd[PRICE] / dd.groupby("year")[PRICE].transform("mean") * 100
    return dd.groupby("month_num")["idx"].mean().reindex(range(1, 13))


# ------------------------------------------------------------------ anomalies
def _robust_z(d: pd.DataFrame, col: str, by: list, min_obs: int) -> pd.Series:
    med = d.groupby(by)[col].transform("median")
    dev = (d[col] - med).abs()
    mad = dev.groupby([d[b] for b in by]).transform("median")
    cnt = d.groupby(by)[col].transform("count")
    z = 0.6745 * (d[col] - med) / mad
    return z.where((mad > 0) & (cnt >= min_obs))


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.5,
                     min_abs_pct: float = 15.0, min_obs: int = 12) -> pd.DataFrame:
    """Flag unusual month-on-month price and arrival changes.

    Changes are computed only between consecutive months of the same commodity+state.
    A change is flagged when its robust z-score (median/MAD, within that series) exceeds
    z_threshold AND the change is at least min_abs_pct percent.
    """
    cols = ["month", "commodity", "state", "anomaly_type", "change_pct",
            "robust_z", PRICE, ARR]
    d = df.dropna(subset=[PRICE]).sort_values(["commodity", "state", "month"]).copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["_mi"] = d["month"].dt.year * 12 + d["month"].dt.month
    g = d.groupby(["commodity", "state"], sort=False)
    consecutive = (d["_mi"] - g["_mi"].shift(1)) == 1
    prev_p, prev_a = g[PRICE].shift(1), g[ARR].shift(1)
    d["price_chg"] = ((d[PRICE] / prev_p - 1) * 100).where(consecutive & (prev_p > 0))
    d["arr_chg"] = ((d[ARR] / prev_a - 1) * 100).where(consecutive & (prev_a > 0))
    d["price_chg"] = d["price_chg"].replace([np.inf, -np.inf], np.nan)
    d["arr_chg"] = d["arr_chg"].replace([np.inf, -np.inf], np.nan)
    by = ["commodity", "state"]
    d["price_z"] = _robust_z(d, "price_chg", by, min_obs)
    d["arr_z"] = _robust_z(d, "arr_chg", by, min_obs)

    parts = []
    rules = [
        ("Price Spike", "price_chg", "price_z", lambda c, z: (z > z_threshold) & (c >= min_abs_pct)),
        ("Price Drop", "price_chg", "price_z", lambda c, z: (z < -z_threshold) & (c <= -min_abs_pct)),
        ("Arrivals Surge", "arr_chg", "arr_z", lambda c, z: (z > z_threshold) & (c >= min_abs_pct)),
        ("Arrivals Drop", "arr_chg", "arr_z", lambda c, z: (z < -z_threshold) & (c <= -min_abs_pct)),
    ]
    for label, chg, zc, rule in rules:
        hit = d[rule(d[chg], d[zc])]
        if hit.empty:
            continue
        p = hit[["month", "commodity", "state", PRICE, ARR]].copy()
        p["anomaly_type"] = label
        p["change_pct"] = hit[chg].round(2)
        p["robust_z"] = hit[zc].round(2)
        parts.append(p[cols])
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["abs_z"] = out["robust_z"].abs()
    return out.sort_values("abs_z", ascending=False).drop(columns="abs_z").reset_index(drop=True)
