"""Cached loaders for the uploaded CSV files. All data comes from /data."""
import pandas as pd
import streamlit as st

from utils.config import ARR, BASE_COLS, BASE_DIR, DATA_DIR, FEATURE_COLS, FILES, PRICE


class DataError(Exception):
    """Raised when a data file is missing or has unexpected columns."""


def _path(key: str):
    name = FILES[key]
    for folder in (DATA_DIR, BASE_DIR):
        if (folder / name).exists():
            return folder / name
    raise DataError(f"'{name}' was not found. Place it inside the 'data/' folder next to app.py.")


def _check(df: pd.DataFrame, required: list, name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{name} is missing expected columns: {missing}")


@st.cache_data(show_spinner="Loading feature-engineered dataset...")
def load_features() -> pd.DataFrame:
    df = pd.read_csv(_path("features"), parse_dates=["month"])
    _check(df, FEATURE_COLS, FILES["features"])
    for c in (PRICE, ARR):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["commodity", "state", "month"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading selected commodities dataset...")
def load_selected() -> pd.DataFrame:
    df = pd.read_csv(_path("selected"), parse_dates=["month"])
    _check(df, BASE_COLS, FILES["selected"])
    return df


@st.cache_data(show_spinner="Loading full cleaned dataset (all commodities)...")
def load_full() -> pd.DataFrame:
    df = pd.read_csv(_path("cleaned"), parse_dates=["month"])
    _check(df, BASE_COLS, FILES["cleaned"])
    for c in (PRICE, ARR):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_stats() -> pd.DataFrame:
    return pd.read_csv(_path("stats"))


@st.cache_data(show_spinner=False)
def load_quality() -> pd.DataFrame:
    return pd.read_csv(_path("quality"))


def valid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable price (missing values are never filled with fake data)."""
    return df.dropna(subset=[PRICE])
