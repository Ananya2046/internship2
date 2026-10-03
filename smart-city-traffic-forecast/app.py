import streamlit as st
import pandas as pd
import numpy as np
import os

# Set up page layout
st.set_page_config(
    page_title="Smart City Traffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
METRICS_PATH = os.path.join(REPORTS_DIR, "evaluation_metrics.csv")

st.title("🚦 Smart City Traffic Forecasting System")
st.markdown("Interactive control panel to monitor historical trends and evaluate **30-day predictive horizons** with **SARIMA** and **Prophet**.")
st.write("---")

# 📊 Sidebar Controls
st.sidebar.header("🕹️ Control Center")
junction = st.sidebar.selectbox("Select Traffic Junction", [1, 2, 3, 4], index=0)
model_choice = st.sidebar.radio("Select Prediction Engine", ["Both", "SARIMA", "Prophet"], index=0)

# Load forecast data
forecast_file = os.path.join(REPORTS_DIR, f"forecast_junction_{junction}.csv")
forecast_df = None

if os.path.exists(forecast_file):
    try:
        forecast_df = pd.read_csv(forecast_file)
        if "DateTime" in forecast_df.columns:
            forecast_df["DateTime"] = pd.to_datetime(forecast_df["DateTime"])
            forecast_df.set_index("DateTime", inplace=True)
    except Exception as e:
        st.sidebar.error(f"Error loading forecast data: {e}")

# Load metrics data
metrics_df = None
if os.path.exists(METRICS_PATH):
    try:
        raw_metrics = pd.read_csv(METRICS_PATH)
        if not raw_metrics.empty and "Junction" in raw_metrics.columns:
            metrics_df = raw_metrics[raw_metrics["Junction"] == junction]
    except Exception:
        pass

# Determine plot columns
plot_cols = []
if forecast_df is not None:
    available_cols = forecast_df.columns.tolist()
    if model_choice == "SARIMA":
        if "SARIMA" in available_cols:
            plot_cols.append("SARIMA")
        else:
            st.warning(f"⚠️ Junction {junction}: SARIMA forecast column not found in report.")
    elif model_choice == "Prophet":
        if "Prophet" in available_cols:
            plot_cols.append("Prophet")
        else:
            st.warning(f"⚠️ Junction {junction}: Prophet forecast column not found in report.")
    elif model_choice == "Both":
        if "SARIMA" in available_cols:
            plot_cols.append("SARIMA")
        if "Prophet" in available_cols:
            plot_cols.append("Prophet")
        if not plot_cols:
            st.warning(f"⚠️ Junction {junction}: No model forecast columns found in report.")

# Top KPIs Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Selected Junction", f"Junction {junction}")

with kpi2:
    if forecast_df is not None and plot_cols:
        primary_col = plot_cols[0]
        avg_val = forecast_df[primary_col].mean()
        st.metric(f"Avg Forecast ({primary_col})", f"{avg_val:.1f} veh/hr")
    else:
        st.metric("Avg Forecast", "N/A")

with kpi3:
    if forecast_df is not None and plot_cols:
        primary_col = plot_cols[0]
        max_val = forecast_df[primary_col].max()
        st.metric(f"Peak Forecast ({primary_col})", f"{max_val:.1f} veh/hr")
    else:
        st.metric("Peak Forecast", "N/A")

with kpi4:
    if metrics_df is not None and not metrics_df.empty:
        best_row = metrics_df.sort_values("MAE").iloc[0]
        st.metric("Top Model (Lowest MAE)", f"{best_row['Model']} ({best_row['MAE']:.2f})")
    else:
        st.metric("Top Model", "N/A")

st.write("")

# 📈 Main Dashboard Body
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🎯 Model Performance")
    if metrics_df is not None and not metrics_df.empty:
        display_metrics = metrics_df.copy()
        if model_choice != "Both":
            filtered = display_metrics[display_metrics["Model"] == model_choice]
            if not filtered.empty:
                display_metrics = filtered
        st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    else:
        st.info("No evaluation metrics found. Run the pipeline to generate metrics.")

with col2:
    st.subheader(f"🔮 30-Day Traffic Forecast: Junction {junction}")
    if forecast_df is not None and plot_cols:
        st.line_chart(forecast_df[plot_cols], use_container_width=True)
    elif forecast_df is None:
        st.error(f"Missing forecast CSV for Junction {junction}. Please run the pipeline first.")
    else:
        st.warning("No forecast data available for the chosen model filter.")

st.write("---")
st.caption("✨ Designed as an interactive addition to the Smart City Traffic Forecasting System pipeline.")