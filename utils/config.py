"""Project-wide constants: file names, column names and colours."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "features": "feature_engineered_data.csv",
    "selected": "selected_15_commodities.csv",
    "cleaned": "cleaned_data.csv",
    "stats": "commodity_statistics.csv",
    "quality": "data_quality_report.csv",
}

# Actual column names used in the uploaded datasets
PRICE = "price_rs_qtl"      # Rs per quintal
ARR = "arrivals_t"          # tonnes
BASE_COLS = ["month", "commodity_id", "commodity", "group", "state", ARR, PRICE]
FEATURE_COLS = BASE_COLS + [
    "year", "month_num", "month_name", "quarter", "season",
    "price_lag_1", "arrivals_lag_1", "price_lag_3", "arrivals_lag_3",
    "price_lag_6", "arrivals_lag_6", "price_ma_3", "price_ma_6",
    "arrivals_ma_3", "arrivals_ma_6", "price_mom_pct", "price_yoy_pct",
    "arrivals_mom_pct",
]

COLORS = {
    "navy": "#0B2A5B", "blue": "#1F5FBF", "sky": "#4C9BE8", "light": "#EAF1FB",
    "accent": "#F28C28", "green": "#2E9E6B", "red": "#D64545", "grey": "#8A97AB",
}
PALETTE = ["#1F5FBF", "#0B2A5B", "#4C9BE8", "#F28C28", "#2E9E6B",
           "#8E6BD1", "#D64545", "#17A2B8", "#B58B00", "#6C7A89"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
ALL_STATES = "All States"
