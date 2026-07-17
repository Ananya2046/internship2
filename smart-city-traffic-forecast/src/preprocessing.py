"""
preprocessing.py
================
Module for data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw traffic dataset.

    Steps:
    - Convert DateTime to pandas datetime
    - Remove duplicates
    - Handle missing values
    - Sort by DateTime and Junction
    - Drop the ID column (not needed for analysis)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataset.
    """
    df = df.copy()

    # Convert DateTime
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # Drop ID column — not useful for modeling
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Remove duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Removing {n_dupes} duplicate rows.")
        df = df.drop_duplicates()

    # Handle missing values
    n_nulls = df.isnull().sum().sum()
    if n_nulls > 0:
        print(f"Found {n_nulls} null values. Forward-filling...")
        df = df.fillna(method="ffill")

    # Sort
    df = df.sort_values(by=["Junction", "DateTime"]).reset_index(drop=True)

    print(f"Preprocessing complete. Shape: {df.shape}")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the DateTime column.

    Features extracted:
    - Hour, Day, Month, Year, Weekday (0=Monday), WeekOfYear
    - IsWeekend (Saturday=5, Sunday=6)
    - DayName

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with DateTime column.

    Returns
    -------
    pd.DataFrame
        Dataset with additional time features.
    """
    df = df.copy()

    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Year"] = df["DateTime"].dt.year
    df["Weekday"] = df["DateTime"].dt.weekday  # 0=Monday
    df["WeekOfYear"] = df["DateTime"].dt.isocalendar().week.astype(int)
    df["DayName"] = df["DateTime"].dt.day_name()
    df["IsWeekend"] = (df["Weekday"] >= 5).astype(int)

    print(f"Time features extracted. Columns: {list(df.columns)}")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling window features per junction.

    Lag features:
    - lag_1h: Previous hour traffic
    - lag_24h: Same hour previous day
    - lag_168h: Same hour previous week

    Rolling features:
    - rolling_24h_mean: 24-hour rolling average
    - rolling_7d_mean: 7-day (168h) rolling average

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with time features.

    Returns
    -------
    pd.DataFrame
        Dataset with lag and rolling features.
    """
    df = df.copy()
    df = df.sort_values(by=["Junction", "DateTime"]).reset_index(drop=True)

    lag_dfs = []
    for junction in df["Junction"].unique():
        jdf = df[df["Junction"] == junction].copy()
        jdf = jdf.sort_values("DateTime")

        # Lag features
        jdf["lag_1h"] = jdf["Vehicles"].shift(1)
        jdf["lag_24h"] = jdf["Vehicles"].shift(24)
        jdf["lag_168h"] = jdf["Vehicles"].shift(168)

        # Rolling features
        jdf["rolling_24h_mean"] = jdf["Vehicles"].rolling(window=24, min_periods=1).mean()
        jdf["rolling_7d_mean"] = jdf["Vehicles"].rolling(window=168, min_periods=1).mean()

        lag_dfs.append(jdf)

    df = pd.concat(lag_dfs, ignore_index=True)
    df = df.sort_values(by=["Junction", "DateTime"]).reset_index(drop=True)

    print(f"Lag and rolling features added. Shape: {df.shape}")
    return df


def get_junction_data(df: pd.DataFrame, junction: int) -> pd.DataFrame:
    """
    Filter dataset for a specific junction.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    junction : int
        Junction number (1, 2, 3, or 4).

    Returns
    -------
    pd.DataFrame
        Filtered dataset for the given junction.
    """
    jdf = df[df["Junction"] == junction].copy()
    jdf = jdf.sort_values("DateTime").reset_index(drop=True)
    return jdf


if __name__ == "__main__":
    from data_loading import load_data

    df = load_data()
    df = preprocess_data(df)
    df = extract_time_features(df)
    df = add_lag_features(df)
    print(df.head(10))
    print(df.info())
