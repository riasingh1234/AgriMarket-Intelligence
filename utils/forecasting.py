"""Price forecasting: baselines, Linear Regression, Random Forest and ARIMA.

Leakage control
---------------
* Chronological split: the LAST `test_size` months are the test set.
* Models are fitted on the training months only and forecast the whole test window
  recursively (their own predictions feed the lag features) - the test actuals are never used.
* For the final future forecast, models are refitted on the full history.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MIN_OBS = 24
LAGS = (1, 2, 3, 6)
MAX_LAG = max(LAGS)
NAIVE = "Naive Baseline"
SNAIVE = "Seasonal Naive Baseline"
LINEAR = "Linear Regression"
FOREST = "Random Forest"
ARIMA_NAME = "ARIMA"
ML_MODELS = [LINEAR, FOREST, ARIMA_NAME]
ARIMA_ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]


# ------------------------------------------------------------- series prep
def prepare_series(raw: pd.Series):
    """Make a clean, continuous monthly series.

    Gaps of up to 3 months are linearly interpolated; if a longer gap exists, only the
    most recent unbroken stretch is used. Returns (series, info).
    """
    info = {"gaps_filled": 0, "months_dropped": 0, "first": None, "last": None}
    s = raw.dropna().sort_index()
    if s.empty:
        return s, info
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full)
    was_missing = s.isna()
    s = s.interpolate(limit=3, limit_area="inside")
    ok = s.notna().to_numpy()
    start = len(ok)
    while start > 0 and ok[start - 1]:
        start -= 1
    info["months_dropped"] = int(start)
    s = s.iloc[start:]
    info["gaps_filled"] = int(was_missing.iloc[start:].sum())
    s = s.clip(lower=0.01)
    if len(s):
        info["first"], info["last"] = s.index[0], s.index[-1]
    return s, info


# --------------------------------------------------------- ML feature logic
def _row_features(hist_log, month):
    m = month.month
    lags = [hist_log[-l] for l in LAGS]
    return lags + [np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]


def _build_xy(values_log, index):
    X, y = [], []
    for i in range(MAX_LAG, len(values_log)):
        X.append(_row_features(list(values_log[:i]), index[i]))
        y.append(values_log[i])
    return np.array(X), np.array(y)


def _recursive_predict(model, hist_log, start_month, steps):
    hist = list(hist_log)
    lo, hi = min(hist) - 0.7, max(hist) + 0.7          # keep forecasts within ~x0.5 - x2 of history
    preds = []
    for k in range(steps):
        month = start_month + pd.DateOffset(months=k)
        p = float(model.predict(np.array([_row_features(hist, month)]))[0])
        p = min(max(p, lo), hi)
        hist.append(p)
        preds.append(p)
    return np.expm1(np.array(preds))


def _arima_forecast(hist: pd.Series, steps: int):
    from statsmodels.tsa.arima.model import ARIMA   # imported here so the app still runs without it
    y = pd.Series(np.log1p(hist.to_numpy(float)), index=pd.DatetimeIndex(hist.index, freq="MS"))
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_ORDERS:
            try:
                res = ARIMA(y, order=order).fit()
                if best is None or res.aic < best.aic:
                    best = res
            except Exception:
                continue
        if best is None:
            raise RuntimeError("ARIMA could not be fitted on this series.")
        fc = best.get_forecast(steps=steps)
        ci = fc.conf_int(alpha=0.2)                    # 80% interval
    return (np.expm1(fc.predicted_mean.to_numpy()),
            np.expm1(ci.iloc[:, 0].to_numpy()), np.expm1(ci.iloc[:, 1].to_numpy()))


def forecast_with(name: str, hist: pd.Series, steps: int):
    """Forecast `steps` months after the end of `hist` using ONLY `hist`. Returns (mean, lower, upper)."""
    vals = hist.to_numpy(float)
    start = hist.index[-1] + pd.DateOffset(months=1)
    none = None
    if name == NAIVE:
        return np.repeat(vals[-1], steps), none, none
    if name == SNAIVE:
        if len(vals) < 12:
            raise ValueError("Need at least 12 months for the seasonal baseline.")
        return np.array([vals[-12 + (k % 12)] for k in range(steps)]), none, none
    if name in (LINEAR, FOREST):
        logv = np.log1p(vals)
        X, y = _build_xy(logv, hist.index)
        if len(X) < 8:
            raise ValueError("Not enough training rows after creating lag features.")
        model = (LinearRegression() if name == LINEAR else
                 RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42))
        model.fit(X, y)
        return _recursive_predict(model, logv, start, steps), none, none
    if name == ARIMA_NAME:
        return _arima_forecast(hist, steps)
    raise ValueError(f"Unknown model: {name}")


def _metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2 = r2_score(actual, pred) if len(actual) > 1 and np.var(actual) > 0 else np.nan
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return mae, rmse, r2, mape


def run_forecast(series: pd.Series, horizon: int, models=None) -> dict:
    """Evaluate on a chronological hold-out, then forecast `horizon` months ahead."""
    models = list(models) if models else list(ML_MODELS)
    res = {"ok": False, "message": "", "warnings": [], "series": series}
    n = len(series)
    if n < MIN_OBS:
        res["message"] = (f"Only {n} usable monthly observations - at least {MIN_OBS} are needed "
                          f"for a reliable forecast.")
        return res
    if n < 36:
        res["warnings"].append(f"Limited history ({n} months): treat the forecast with caution.")

    test_size = int(np.clip(round(n * 0.2), 6, 12))
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    names = [NAIVE, SNAIVE] + models
    test_pred, future, lower, upper, rows = {}, {}, {}, {}, []
    fut_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

    for name in names:
        try:
            p_test, _, _ = forecast_with(name, train, test_size)          # train only -> no leakage
            p_fut, lo, hi = forecast_with(name, series, horizon)          # refit on all history
        except Exception as exc:                                          # model-level failure is not fatal
            res["warnings"].append(f"{name} skipped: {exc}")
            continue
        test_pred[name], future[name] = p_test, p_fut
        if lo is not None:
            lower[name], upper[name] = lo, hi
        mae, rmse, r2, mape = _metrics(test.to_numpy(), p_test)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE %": mape})

    if NAIVE not in test_pred or all(r["Model"] in (NAIVE, SNAIVE) for r in rows):
        res["message"] = "No forecasting model could be fitted for this series."
        return res

    metrics = pd.DataFrame(rows)
    base_rmse = metrics.loc[metrics["Model"] == NAIVE, "RMSE"].iloc[0]
    metrics["Beats Naive Baseline"] = np.where(
        metrics["Model"].isin([NAIVE]), "-", np.where(metrics["RMSE"] < base_rmse, "Yes", "No"))
    metrics = metrics.sort_values("RMSE").reset_index(drop=True)
    candidates = metrics[~metrics["Model"].isin([NAIVE, SNAIVE])]
    best = candidates.iloc[0]["Model"] if len(candidates) else NAIVE

    res.update({
        "ok": True, "test_size": test_size, "n_train": len(train),
        "test_pred": pd.DataFrame(test_pred, index=test.index).assign(Actual=test.to_numpy()),
        "future": pd.DataFrame(future, index=fut_idx),
        "lower": pd.DataFrame(lower, index=fut_idx), "upper": pd.DataFrame(upper, index=fut_idx),
        "metrics": metrics, "best_model": best,
        "best_rmse": float(candidates.iloc[0]["RMSE"]) if len(candidates) else np.nan,
        "baseline_rmse": float(base_rmse),
    })
    return res
