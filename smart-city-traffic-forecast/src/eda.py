"""
eda.py
======
Module for Exploratory Data Analysis and visualization.
All plots are saved to reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (14, 6)
FIGSIZE_LARGE = (16, 10)
DPI = 150

# Colors for 4 junctions
JUNCTION_COLORS = {1: "#2196F3", 2: "#FF5722", 3: "#4CAF50", 4: "#9C27B0"}
JUNCTION_NAMES = {1: "Junction 1", 2: "Junction 2", 3: "Junction 3", 4: "Junction 4"}


def _get_figures_dir() -> str:
    """Get the reports/figures directory path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def plot_traffic_trends(df: pd.DataFrame) -> str:
    """
    Plot daily traffic trends per junction over time.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with DateTime and Vehicles columns.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE, sharex=False)
    axes = axes.ravel()

    for i, junction in enumerate(sorted(df["Junction"].unique())):
        jdf = df[df["Junction"] == junction].copy()
        daily = jdf.groupby(jdf["DateTime"].dt.date)["Vehicles"].mean().reset_index()
        daily.columns = ["Date", "AvgVehicles"]
        daily["Date"] = pd.to_datetime(daily["Date"])

        ax = axes[i]
        ax.plot(daily["Date"], daily["AvgVehicles"],
                color=JUNCTION_COLORS.get(junction, "#333"),
                linewidth=1.2, alpha=0.8)
        ax.fill_between(daily["Date"], daily["AvgVehicles"],
                        alpha=0.15, color=JUNCTION_COLORS.get(junction, "#333"))
        ax.set_title(f"Junction {junction} — Daily Average Traffic",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Avg Vehicles / Hour", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Traffic Trends Over Time Per Junction",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "traffic_trends.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_hourly_distribution(df: pd.DataFrame) -> str:
    """
    Plot average hourly traffic distribution per junction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Hour and Vehicles columns.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    hourly = df.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()

    for junction in sorted(hourly["Junction"].unique()):
        jdata = hourly[hourly["Junction"] == junction]
        ax.plot(jdata["Hour"], jdata["Vehicles"],
                marker="o", markersize=5, linewidth=2.5,
                color=JUNCTION_COLORS.get(junction, "#333"),
                label=JUNCTION_NAMES.get(junction, f"Junction {junction}"))

    ax.set_title("Average Hourly Traffic Distribution by Junction",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Average Number of Vehicles", fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "hourly_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_weekday_vs_weekend(df: pd.DataFrame) -> str:
    """
    Compare weekday vs weekend traffic patterns per junction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with IsWeekend, Hour, and Vehicles columns.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.ravel()

    for i, junction in enumerate(sorted(df["Junction"].unique())):
        jdf = df[df["Junction"] == junction]
        ax = axes[i]

        for is_weekend, label, color, ls in [(0, "Weekday", "#1976D2", "-"),
                                              (1, "Weekend", "#E64A19", "--")]:
            subset = jdf[jdf["IsWeekend"] == is_weekend]
            hourly_avg = subset.groupby("Hour")["Vehicles"].mean()
            ax.plot(hourly_avg.index, hourly_avg.values,
                    marker="o", markersize=4, linewidth=2.2,
                    color=color, linestyle=ls, label=label)

        ax.set_title(f"Junction {junction}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour", fontsize=11)
        ax.set_ylabel("Avg Vehicles", fontsize=11)
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Weekday vs Weekend Traffic Patterns",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "weekday_vs_weekend.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_monthly_trends(df: pd.DataFrame) -> str:
    """
    Plot monthly average traffic per junction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Month, Year, and Vehicles columns.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    df["YearMonth"] = df["DateTime"].dt.to_period("M")
    monthly = df.groupby(["Junction", "YearMonth"])["Vehicles"].mean().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    for junction in sorted(monthly["Junction"].unique()):
        jdata = monthly[monthly["Junction"] == junction]
        ax.plot(range(len(jdata)), jdata["Vehicles"].values,
                marker="s", markersize=4, linewidth=2,
                color=JUNCTION_COLORS.get(junction, "#333"),
                label=JUNCTION_NAMES.get(junction, f"Junction {junction}"))

    # Use junction 1 for x-tick labels
    j1_data = monthly[monthly["Junction"] == sorted(monthly["Junction"].unique())[0]]
    tick_positions = range(0, len(j1_data), max(1, len(j1_data) // 10))
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([j1_data["YearMonth"].iloc[i] for i in tick_positions],
                       rotation=45, ha="right")

    ax.set_title("Monthly Average Traffic Trends", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Average Vehicles / Hour", fontsize=12)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Clean up temp column
    df.drop(columns=["YearMonth"], inplace=True, errors="ignore")

    path = os.path.join(_get_figures_dir(), "monthly_trends.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_heatmap(df: pd.DataFrame) -> str:
    """
    Plot heatmap of average traffic by hour and day of week per junction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Hour, Weekday, and Vehicles columns.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.ravel()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for i, junction in enumerate(sorted(df["Junction"].unique())):
        jdf = df[df["Junction"] == junction]
        pivot = jdf.pivot_table(values="Vehicles", index="Weekday",
                                columns="Hour", aggfunc="mean")
        pivot.index = [day_labels[j] for j in pivot.index]

        ax = axes[i]
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, annot=False,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Junction {junction}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour", fontsize=11)
        ax.set_ylabel("Day of Week", fontsize=11)

    plt.suptitle("Traffic Heatmap: Hour × Day of Week",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "traffic_heatmap.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_day_of_week_distribution(df: pd.DataFrame) -> str:
    """
    Box plot of traffic distribution by day of week.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with DayName and Vehicles columns.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.ravel()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    for i, junction in enumerate(sorted(df["Junction"].unique())):
        jdf = df[df["Junction"] == junction]
        ax = axes[i]
        sns.boxplot(data=jdf, x="DayName", y="Vehicles", order=day_order,
                    palette="Set2", ax=ax, fliersize=2)
        ax.set_title(f"Junction {junction}", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Vehicles", fontsize=11)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Traffic Distribution by Day of Week",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(_get_figures_dir(), "day_of_week_boxplot.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def identify_peak_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify peak traffic hours for each junction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Hour, Junction, and Vehicles columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with peak hour analysis per junction.
    """
    results = []
    for junction in sorted(df["Junction"].unique()):
        jdf = df[df["Junction"] == junction]
        hourly = jdf.groupby("Hour")["Vehicles"].agg(["mean", "std", "max"]).reset_index()
        hourly.columns = ["Hour", "AvgVehicles", "StdVehicles", "MaxVehicles"]

        # Peak hours = top 5 hours by average traffic
        top5 = hourly.nlargest(5, "AvgVehicles")
        peak_hours = sorted(top5["Hour"].tolist())

        # Overall stats
        overall_avg = jdf["Vehicles"].mean()
        overall_max = jdf["Vehicles"].max()

        results.append({
            "Junction": junction,
            "PeakHours": peak_hours,
            "OverallAvgVehicles": round(overall_avg, 2),
            "OverallMaxVehicles": overall_max,
            "TopHourAvg": round(top5["AvgVehicles"].max(), 2),
        })

    peak_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("PEAK TRAFFIC HOURS ANALYSIS")
    print("=" * 60)
    for _, row in peak_df.iterrows():
        print(f"\nJunction {row['Junction']}:")
        print(f"  Peak Hours: {row['PeakHours']}")
        print(f"  Avg Vehicles: {row['OverallAvgVehicles']}")
        print(f"  Max Vehicles: {row['OverallMaxVehicles']}")
        print(f"  Busiest Hour Avg: {row['TopHourAvg']}")
    print("=" * 60)

    return peak_df


def run_eda(df: pd.DataFrame) -> dict:
    """
    Run the complete EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with all time features.

    Returns
    -------
    dict
        Dictionary with paths to all generated figures and peak hour data.
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    results = {}

    results["traffic_trends"] = plot_traffic_trends(df)
    results["hourly_distribution"] = plot_hourly_distribution(df)
    results["weekday_vs_weekend"] = plot_weekday_vs_weekend(df)
    results["monthly_trends"] = plot_monthly_trends(df)
    results["heatmap"] = plot_heatmap(df)
    results["day_boxplot"] = plot_day_of_week_distribution(df)
    results["peak_hours"] = identify_peak_hours(df)

    print(f"\nEDA complete. {len(results) - 1} figures saved to reports/figures/")
    return results


if __name__ == "__main__":
    from data_loading import load_data
    from preprocessing import preprocess_data, extract_time_features

    df = load_data()
    df = preprocess_data(df)
    df = extract_time_features(df)
    run_eda(df)
