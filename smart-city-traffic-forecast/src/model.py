"""
model.py
========
Module for building and training traffic forecasting models.
Supports SARIMA and Prophet, trained separately per junction.
"""

import os
import warnings
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

warnings.filterwarnings("ignore")


def train_test_split_ts(df: pd.DataFrame, test_days: int = 30) -> tuple:
    """
    Split time-series data into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Junction-specific dataset sorted by DateTime.
    test_days : int
        Number of days to use for testing.

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    df = df.sort_values("DateTime").reset_index(drop=True)
    cutoff = df["DateTime"].max() - pd.Timedelta(days=test_days)
    train = df[df["DateTime"] <= cutoff].copy()
    test = df[df["DateTime"] > cutoff].copy()

    print(f"  Train: {train.shape[0]} rows ({train['DateTime'].min()} to {train['DateTime'].max()})")
    print(f"  Test:  {test.shape[0]} rows ({test['DateTime'].min()} to {test['DateTime'].max()})")

    return train, test


def build_sarima_model(train_series: pd.Series, junction: int,
                       seasonal_period: int = 24) -> dict:
    """
    Build a SARIMA model using auto_arima for parameter selection.

    Uses daily seasonality (m=24) since data is hourly.
    To keep fitting tractable, we use a subset for auto_arima
    parameter search and then fit the full model.

    Parameters
    ----------
    train_series : pd.Series
        Training time series (Vehicles) — must be a pd.Series with a
        DatetimeIndex so that SARIMAX can properly extend forecasts.
    junction : int
        Junction number (for logging).
    seasonal_period : int
        Seasonal period (24 for hourly data with daily seasonality).

    Returns
    -------
    dict
        Dictionary with model object, order, seasonal_order, and fitted values.
    """
    print(f"\n  [SARIMA] Junction {junction}: Finding optimal parameters...")

    # Ensure we have a proper pd.Series (convert numpy arrays if needed)
    if not isinstance(train_series, pd.Series):
        train_series = pd.Series(train_series)

    # Use a small sample for fast auto_arima parameter search (last 500 pts)
    sample = train_series.iloc[-500:] if len(train_series) > 500 else train_series

    try:
        auto_model = pm.auto_arima(
            sample,
            seasonal=True,
            m=seasonal_period,
            max_p=2, max_q=2,       # reduced from 3
            max_P=1, max_Q=1,       # reduced from 2
            max_d=1, max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            n_fits=10,              # reduced from 30
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
    except Exception:
        # Fallback to a sensible default if auto_arima fails
        order = (1, 1, 1)
        seasonal_order = (1, 0, 0, seasonal_period)

    print(f"  [SARIMA] Junction {junction}: Best order={order}, seasonal_order={seasonal_order}")

    # Fit SARIMAX on last 2000 points for tractability
    fit_series = train_series.iloc[-2000:] if len(train_series) > 2000 else train_series

    model = SARIMAX(
        fit_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False, maxiter=100)
    print(f"  [SARIMA] Junction {junction}: AIC={fitted_model.aic:.2f}")

    return {
        "model": fitted_model,
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": fitted_model.aic,
        "n_train": len(fit_series),
    }


def forecast_sarima(model_result: dict, steps: int) -> np.ndarray:
    """
    Generate forecasts using a fitted SARIMA model.

    Parameters
    ----------
    model_result : dict
        Output from build_sarima_model().
    steps : int
        Number of steps to forecast.

    Returns
    -------
    np.ndarray
        Forecasted values.
    """
    forecast = model_result["model"].forecast(steps=steps)
    # Ensure non-negative forecasts
    forecast = np.maximum(forecast, 0)
    # Handle both pd.Series and np.ndarray outputs
    if isinstance(forecast, pd.Series):
        return forecast.values
    return np.asarray(forecast)


def build_prophet_model(train_df: pd.DataFrame, junction: int) -> object:
    """
    Build a Prophet model for a single junction.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with DateTime and Vehicles columns.
    junction : int
        Junction number (for logging).

    Returns
    -------
    object
        Fitted Prophet model.
    """
    from prophet import Prophet

    print(f"\n  [Prophet] Junction {junction}: Training model...")

    prophet_df = train_df[["DateTime", "Vehicles"]].copy()
    prophet_df.columns = ["ds", "y"]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )
    model.fit(prophet_df)

    print(f"  [Prophet] Junction {junction}: Model trained.")
    return model


def forecast_prophet(model, periods: int, freq: str = "h") -> pd.DataFrame:
    """
    Generate forecasts using a fitted Prophet model.

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    periods : int
        Number of periods to forecast.
    freq : str
        Frequency string ('h' for hourly).

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with ds, yhat, yhat_lower, yhat_upper.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    forecast["yhat"] = np.maximum(forecast["yhat"], 0)
    return forecast.tail(periods)


def train_all_junctions(df: pd.DataFrame, test_days: int = 30,
                        use_prophet: bool = True) -> dict:
    """
    Train forecasting models for all junctions.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed dataset.
    test_days : int
        Number of days for test split.
    use_prophet : bool
        Whether to also train Prophet models.

    Returns
    -------
    dict
        Nested dictionary with results per junction and model type.
        Structure: {junction: {"sarima": {...}, "prophet": {...},
                               "train": df, "test": df}}
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    results = {}
    junctions = sorted(df["Junction"].unique())

    for junction in junctions:
        print(f"\n{'─' * 40}")
        print(f"Junction {junction}")
        print(f"{'─' * 40}")

        jdf = df[df["Junction"] == junction].sort_values("DateTime").reset_index(drop=True)
        train, test = train_test_split_ts(jdf, test_days=test_days)

        junction_result = {"train": train, "test": test}

        # ── SARIMA ──
        try:
            # Pass a pd.Series with DatetimeIndex so SARIMAX can extend
            # forecasts properly (using numpy array loses the time index).
            train_series = train.set_index("DateTime")["Vehicles"]
            sarima_result = build_sarima_model(train_series, junction)
            sarima_forecast = forecast_sarima(sarima_result, steps=len(test))
            junction_result["sarima"] = {
                "model": sarima_result,
                "forecast": sarima_forecast,
                "actual": test["Vehicles"].values,
            }
        except Exception as e:
            print(f"  [SARIMA] Junction {junction}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            junction_result["sarima"] = None

        # ── Prophet ──
        if use_prophet:
            try:
                prophet_model = build_prophet_model(train, junction)
                prophet_forecast = forecast_prophet(prophet_model, periods=len(test))
                junction_result["prophet"] = {
                    "model": prophet_model,
                    "forecast": prophet_forecast["yhat"].values,
                    "actual": test["Vehicles"].values,
                }
            except Exception as e:
                print(f"  [Prophet] Junction {junction}: FAILED — {e}")
                junction_result["prophet"] = None
        else:
            junction_result["prophet"] = None

        results[junction] = junction_result

    print(f"\n{'=' * 60}")
    print("MODEL TRAINING COMPLETE")
    print(f"{'=' * 60}")

    return results


def forecast_future(df: pd.DataFrame, results: dict,
                    future_days: int = 30) -> dict:
    """
    Generate future traffic forecasts beyond the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    results : dict
        Training results from train_all_junctions().
    future_days : int
        Number of days to forecast into the future.

    Returns
    -------
    dict
        Dictionary mapping junction -> forecast DataFrame.
    """
    print("\n" + "=" * 60)
    print(f"FORECASTING NEXT {future_days} DAYS")
    print("=" * 60)

    future_forecasts = {}
    steps = future_days * 24  # hourly

    for junction in sorted(results.keys()):
        jresult = results[junction]
        jdf = df[df["Junction"] == junction].sort_values("DateTime")
        last_date = jdf["DateTime"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                     periods=steps, freq="h")

        forecasts = {"DateTime": future_dates}

        # SARIMA forecast
        if jresult.get("sarima") is not None:
            try:
                sarima_model = jresult["sarima"]["model"]["model"]
                sarima_fc = sarima_model.forecast(steps=steps)
                forecasts["SARIMA"] = np.maximum(sarima_fc.values, 0)
            except Exception as e:
                print(f"  Junction {junction} SARIMA future forecast failed: {e}")

        # Prophet forecast
        if jresult.get("prophet") is not None:
            try:
                prophet_model = jresult["prophet"]["model"]
                prophet_fc = forecast_prophet(prophet_model, periods=steps)
                forecasts["Prophet"] = prophet_fc["yhat"].values
            except Exception as e:
                print(f"  Junction {junction} Prophet future forecast failed: {e}")

        future_df = pd.DataFrame(forecasts)
        future_forecasts[junction] = future_df
        print(f"  Junction {junction}: Forecasted {steps} hours ({future_days} days)")

    return future_forecasts


def save_models(results: dict, output_dir: str = None):
    """
    Save trained model results to disk.

    Parameters
    ----------
    results : dict
        Training results from train_all_junctions().
    output_dir : str, optional
        Directory to save models.
    """
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "models")

    os.makedirs(output_dir, exist_ok=True)

    for junction, jresult in results.items():
        if jresult.get("sarima") is not None:
            path = os.path.join(output_dir, f"sarima_junction_{junction}.pkl")
            with open(path, "wb") as f:
                pickle.dump(jresult["sarima"]["model"], f)
            print(f"Saved SARIMA model: {path}")

    print("Models saved.")


if __name__ == "__main__":
    from data_loading import load_data
    from preprocessing import preprocess_data, extract_time_features

    df = load_data()
    df = preprocess_data(df)
    df = extract_time_features(df)

    results = train_all_junctions(df, test_days=30, use_prophet=True)
    future = forecast_future(df, results, future_days=30)

    for junction, fdf in future.items():
        print(f"\nJunction {junction} future forecast:")
        print(fdf.head(10))
