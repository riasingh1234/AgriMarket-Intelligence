"""Logic smoke test: python tests/smoke_test.py  (uses the real CSVs, no browser)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from utils import analytics as an, forecasting as fc  # noqa: E402
from utils.config import PRICE  # noqa: E402

df = pd.read_csv("data/feature_engineered_data.csv", parse_dates=["month"]).dropna(subset=[PRICE])
for com in df["commodity"].unique():
    s, _ = fc.prepare_series(an.monthly_agg(df[df.commodity == com])[PRICE])
    r = fc.run_forecast(s, 6)
    assert r["ok"], (com, r["message"])
    assert not r["future"].isna().any().any()
print("forecasting OK for", df["commodity"].nunique(), "commodities")
print("anomalies:", len(an.detect_anomalies(df)))
print("stats rows:", len(an.group_stats(df, "commodity")))
