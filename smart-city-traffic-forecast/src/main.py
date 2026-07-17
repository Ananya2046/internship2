"""
main.py
=======
Main entry point for the Traffic Forecasting System.
Orchestrates the full pipeline: Load → Preprocess → EDA → Model → Evaluate.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Add src to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loading import load_data, inspect_data
from preprocessing import preprocess_data, extract_time_features, add_lag_features
from eda import run_eda
from model import train_all_junctions, forecast_future, save_models
from evaluation import run_evaluation


def main():
    """Execute the full traffic forecasting pipeline."""
    start_time = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  SMART CITY TRAFFIC FORECASTING SYSTEM                   ║")
    print("║  Traffic Analysis & Prediction for 4 Junctions           ║")
    print("╚" + "═" * 58 + "╝")

    # ──────────────────────────────────────────
    # Step 1: Load Data
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 1: Loading Dataset...")
    df = load_data()
    inspection = inspect_data(df)

    # ──────────────────────────────────────────
    # Step 2: Preprocessing
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 2: Preprocessing Data...")
    df = preprocess_data(df)
    df = extract_time_features(df)
    df_with_lags = add_lag_features(df)

    # ──────────────────────────────────────────
    # Step 3: Exploratory Data Analysis
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 3: Exploratory Data Analysis...")
    eda_results = run_eda(df)

    # ──────────────────────────────────────────
    # Step 4: Model Training
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 4: Training Forecasting Models...")
    model_results = train_all_junctions(df, test_days=30, use_prophet=True)

    # ──────────────────────────────────────────
    # Step 5: Future Forecasting
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 5: Generating Future Forecasts...")
    future_forecasts = forecast_future(df, model_results, future_days=30)

    # Save forecast CSVs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(project_root, "reports")
    for junction, fdf in future_forecasts.items():
        csv_path = os.path.join(reports_dir, f"forecast_junction_{junction}.csv")
        fdf.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # ──────────────────────────────────────────
    # Step 6: Evaluation
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 6: Evaluating Models...")
    eval_output = run_evaluation(model_results, future_forecasts)

    # ──────────────────────────────────────────
    # Step 7: Save Models
    # ──────────────────────────────────────────
    print("\n\n🔹 STEP 7: Saving Models...")
    save_models(model_results)

    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n\n" + "═" * 60)
    print("✅ PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  EDA figures saved to: reports/figures/")
    print(f"  Forecast CSVs saved to: reports/")
    print(f"  Evaluation metrics saved to: reports/evaluation_metrics.csv")
    print(f"  Models saved to: models/")
    print("═" * 60)

    return {
        "inspection": inspection,
        "eda_results": eda_results,
        "model_results": model_results,
        "future_forecasts": future_forecasts,
        "eval_output": eval_output,
    }


if __name__ == "__main__":
    results = main()
