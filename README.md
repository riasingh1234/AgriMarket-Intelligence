# 🌾 AgriMarket Intelligence
**Agricultural Price Analytics, Supply Analysis and Forecasting System for India**

A Streamlit dashboard built on the datasets produced in the Colab preprocessing / feature engineering / EDA stage.
Everything is computed from the CSV files in `data/` - no sample numbers, no APIs, no API keys.

## Setup and run

Requires Python 3.9+.

```bash
cd AgriMarket_Intelligence

# (optional) virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Project structure
```
app.py                     entry point + sidebar navigation
requirements.txt
.streamlit/config.toml     light theme
data/                      the 5 uploaded CSV files
utils/
  config.py                paths, column names, colours
  data_loader.py           cached CSV loaders with column validation
  analytics.py             statistics, seasonal index, anomaly detection (pure pandas)
  forecasting.py           baselines, Linear Regression, Random Forest, ARIMA
  ui.py                    CSS, KPI cards, filters, chart/table/download helpers
views/                     one file per module (dashboard, price_analysis, arrivals,
                           state_comparison, seasonal, statistics, prediction,
                           anomaly, explorer)
tests/smoke_test.py        quick logic test (no browser needed)
```

## Data used
| File | Use |
|---|---|
| `feature_engineered_data.csv` | main source for all analysis modules (15 commodities, 32 states, with season/lag/MA columns) |
| `cleaned_data.csv` | dashboard totals (all 459 commodities) and explorer |
| `selected_15_commodities.csv`, `commodity_statistics.csv`, `data_quality_report.csv` | explorer, dashboard KPIs |

Columns: `month, commodity_id, commodity, group, state, arrivals_t (tonnes), price_rs_qtl (Rs/quintal)` plus engineered columns.
About 23% of price/arrival values are missing in the source; they are **never imputed** - those rows are excluded from calculations.

## Forecasting method
- Series: monthly price for a commodity (all-state mean) or a commodity in one state.
- Needs at least **24 usable months**; gaps up to 3 months are linearly interpolated (reported on screen), longer gaps cut the series to the latest unbroken stretch.
- **Chronological split**: last ~20% (6-12 months) is the test set. Models train on earlier months only and forecast the test window recursively - no future information is used.
- Models: Linear Regression and Random Forest (lag 1/2/3/6 of log price + month seasonality), ARIMA (best of a few orders by AIC, 80% interval).
- Baselines: naive (last value) and seasonal naive (same month last year).
- Metrics: MAE, RMSE, R², MAPE. R² can be negative on short test windows.
- Final forecasts (3, 6 or 12 months) refit on the full history.

## Anomaly detection
Month-on-month change (consecutive months only) per commodity-state, flagged when its robust z-score (median/MAD) and the minimum % change thresholds are both exceeded. Both are adjustable.

## Troubleshooting
- *File not found*: keep the CSVs in `data/`.
- *ARIMA missing*: `pip install statsmodels` (other models still work without it).
- First load of `cleaned_data.csv` (about 12 MB) takes a few seconds; it is cached afterwards.
