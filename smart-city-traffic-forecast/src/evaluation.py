"""
evaluation.py
=============
Module for model evaluation, metrics computation, and results visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150
JUNCTION_COLORS = {1: "#2196F3", 2: "#FF5722", 3: "#4CAF50", 4: "#9C27B0"}


def _get_figures_dir() -> str:
    """Get the reports/figures directory path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        Dictionary with MAE and RMSE.
    """
    # Ensure arrays are the same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}


def evaluate_all_junctions(results: dict) -> pd.DataFrame:
    """
    Evaluate all models across all junctions and produce a summary table.

    Parameters
    ----------
    results : dict
        Output from model.train_all_junctions().

    Returns
    -------
    pd.DataFrame
        Evaluation metrics table.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    rows = []
    for junction in sorted(results.keys()):
        jresult = results[junction]

        # SARIMA
        if jresult.get("sarima") is not None:
            actual = jresult["sarima"]["actual"]
            predicted = jresult["sarima"]["forecast"]
            metrics = compute_metrics(actual, predicted)
            rows.append({
                "Junction": junction,
                "Model": "SARIMA",
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            })

        # Prophet
        if jresult.get("prophet") is not None:
            actual = jresult["prophet"]["actual"]
            predicted = jresult["prophet"]["forecast"]
            metrics = compute_metrics(actual, predicted)
            rows.append({
                "Junction": junction,
                "Model": "Prophet",
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
            })

    metrics_df = pd.DataFrame(rows)

    print("\n" + metrics_df.to_string(index=False))
    print("\n" + "=" * 60)

    # Save to CSV
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "reports", "evaluation_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")

    return metrics_df


def plot_actual_vs_predicted(results: dict) -> str:
    """
    Plot actual vs predicted values on test set for each junction.

    Parameters
    ----------
    results : dict
        Output from model.train_all_junctions().

    Returns
    -------
    str
        Path to saved figure.
    """
    n_junctions = len(results)
    fig, axes = plt.subplots(n_junctions, 1, figsize=(16, 5 * n_junctions), sharex=False)
    if n_junctions == 1:
        axes = [axes]

    for i, junction in enumerate(sorted(results.keys())):
        ax = axes[i]
        jresult = results[junction]
        test_df = jresult["test"]
        dates = test_df["DateTime"].values

        actual = test_df["Vehicles"].values

        ax.plot(dates, actual, color="#333333", linewidth=1,
                alpha=0.7, label="Actual")

        if jresult.get("sarima") is not None:
            sarima_pred = jresult["sarima"]["forecast"][:len(dates)]
            ax.plot(dates[:len(sarima_pred)], sarima_pred,
                    color="#E91E63", linewidth=1.5, alpha=0.85,
                    label="SARIMA", linestyle="--")

        if jresult.get("prophet") is not None:
            prophet_pred = jresult["prophet"]["forecast"][:len(dates)]
            ax.plot(dates[:len(prophet_pred)], prophet_pred,
                    color="#00BCD4", linewidth=1.5, alpha=0.85,
                    label="Prophet", linestyle="-.")

        ax.set_title(f"Junction {junction} — Actual vs Forecast (Test Period)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Vehicles", fontsize=12)
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("DateTime", fontsize=12)

    plt.suptitle("Model Performance: Actual vs Predicted",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "actual_vs_predicted.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> str:
    """
    Bar chart comparing MAE and RMSE across junctions and models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Evaluation metrics from evaluate_all_junctions().

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, metric in enumerate(["MAE", "RMSE"]):
        ax = axes[i]
        pivot = metrics_df.pivot(index="Junction", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, color=["#E91E63", "#00BCD4"],
                   edgecolor="white", width=0.7)

        ax.set_title(f"{metric} by Junction and Model",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Junction", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(fontsize=11)
        ax.tick_params(axis="x", rotation=0)

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=9, padding=3)

    plt.suptitle("Model Evaluation Metrics Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "metrics_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_future_forecast(future_forecasts: dict) -> str:
    """
    Plot future traffic forecasts for each junction.

    Parameters
    ----------
    future_forecasts : dict
        Output from model.forecast_future().

    Returns
    -------
    str
        Path to saved figure.
    """
    n = len(future_forecasts)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, junction in enumerate(sorted(future_forecasts.keys())):
        ax = axes[i]
        fdf = future_forecasts[junction]
        dates = fdf["DateTime"]

        if "SARIMA" in fdf.columns:
            ax.plot(dates, fdf["SARIMA"], color="#E91E63",
                    linewidth=1.2, alpha=0.8, label="SARIMA Forecast")

        if "Prophet" in fdf.columns:
            ax.plot(dates, fdf["Prophet"], color="#00BCD4",
                    linewidth=1.2, alpha=0.8, label="Prophet Forecast")

        ax.set_title(f"Junction {junction} — 30-Day Future Forecast",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Predicted Vehicles", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Date", fontsize=12)

    plt.suptitle("Future Traffic Forecast (Next 30 Days)",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "future_forecast.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def identify_future_peak_periods(future_forecasts: dict) -> pd.DataFrame:
    """
    Identify peak traffic periods from future forecasts.

    Parameters
    ----------
    future_forecasts : dict
        Output from model.forecast_future().

    Returns
    -------
    pd.DataFrame
        Peak period analysis for each junction.
    """
    print("\n" + "=" * 60)
    print("FUTURE PEAK TRAFFIC PERIODS")
    print("=" * 60)

    rows = []
    for junction in sorted(future_forecasts.keys()):
        fdf = future_forecasts[junction].copy()

        # Use SARIMA forecast if available, else Prophet
        if "SARIMA" in fdf.columns:
            forecast_col = "SARIMA"
        elif "Prophet" in fdf.columns:
            forecast_col = "Prophet"
        else:
            continue

        fdf["Hour"] = fdf["DateTime"].dt.hour
        fdf["DayName"] = fdf["DateTime"].dt.day_name()
        fdf["Date"] = fdf["DateTime"].dt.date

        # Peak hours
        hourly_avg = fdf.groupby("Hour")[forecast_col].mean()
        peak_hours = hourly_avg.nlargest(5).index.tolist()

        # Peak days
        daily_avg = fdf.groupby("Date")[forecast_col].mean()
        peak_dates = daily_avg.nlargest(5).index.tolist()

        # Peak day of week
        dow_avg = fdf.groupby("DayName")[forecast_col].mean()
        peak_dow = dow_avg.idxmax()

        rows.append({
            "Junction": junction,
            "Peak_Hours": sorted(peak_hours),
            "Peak_Day_of_Week": peak_dow,
            "Top_5_Peak_Dates": peak_dates,
            "Max_Forecasted_Traffic": round(fdf[forecast_col].max(), 1),
            "Avg_Forecasted_Traffic": round(fdf[forecast_col].mean(), 1),
        })

        print(f"\nJunction {junction}:")
        print(f"  Peak Hours: {sorted(peak_hours)}")
        print(f"  Busiest Day of Week: {peak_dow}")
        print(f"  Max Forecasted Traffic: {fdf[forecast_col].max():.1f}")
        print(f"  Avg Forecasted Traffic: {fdf[forecast_col].mean():.1f}")

    peak_df = pd.DataFrame(rows)
    print("\n" + "=" * 60)

    return peak_df


def run_evaluation(results: dict, future_forecasts: dict) -> dict:
    """
    Run the complete evaluation pipeline.

    Parameters
    ----------
    results : dict
        Output from model.train_all_junctions().
    future_forecasts : dict
        Output from model.forecast_future().

    Returns
    -------
    dict
        Dictionary with all evaluation outputs.
    """
    eval_output = {}

    eval_output["metrics"] = evaluate_all_junctions(results)
    eval_output["actual_vs_predicted_plot"] = plot_actual_vs_predicted(results)
    eval_output["metrics_comparison_plot"] = plot_metrics_comparison(eval_output["metrics"])
    eval_output["future_forecast_plot"] = plot_future_forecast(future_forecasts)
    eval_output["future_peaks"] = identify_future_peak_periods(future_forecasts)

    return eval_output


if __name__ == "__main__":
    print("Run via main.py for full pipeline evaluation.")
