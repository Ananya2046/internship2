# 🚦 Smart City Traffic Forecasting System

A comprehensive traffic forecasting system for smart city planning that analyzes traffic patterns across four urban junctions and predicts future traffic volumes using time-series forecasting models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

---

## 📋 Project Overview

The government wants to manage traffic efficiently across four junctions in a smart city. Traffic varies based on working days, holidays, and special occasions. This system:

- **Analyzes** historical traffic patterns across 4 junctions
- **Forecasts** future traffic using SARIMA and Prophet models
- **Identifies** peak traffic periods for each junction
- **Supports** infrastructure planning with data-driven insights

## 🎯 Problem Statement

Build a traffic forecasting system that:
1. Loads and preprocesses hourly traffic data from 4 city junctions
2. Performs exploratory data analysis to uncover traffic trends
3. Engineers time-based and lag features for modeling
4. Trains SARIMA and Prophet models separately for each junction
5. Evaluates models using MAE and RMSE metrics
6. Forecasts future traffic and identifies peak periods

## 📂 Dataset

- **Source**: [Traffic Prediction Dataset](https://drive.google.com/file/d/1y61cDyuO9Zrp1fSchWcAmCxk0B6SMx7X/view)
- **Records**: 48,120 hourly observations
- **Features**: DateTime, Junction (1-4), Vehicles, ID
- **Time Range**: November 2015 — June 2017

## 📁 Repository Structure

```
smart-city-traffic-forecast/
│
├── data/
│   └── traffic.csv              # Raw dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loading.py          # Data loading & inspection
│   ├── preprocessing.py         # Preprocessing & feature engineering
│   ├── eda.py                   # Exploratory data analysis & plots
│   ├── model.py                 # SARIMA & Prophet model training
│   ├── evaluation.py            # Metrics, comparison & visualization
│   └── main.py                  # Full pipeline orchestration
│
├── notebooks/
│   └── analysis.ipynb           # Jupyter notebook with full analysis
│
├── reports/
│   ├── final_report.md          # Professional project report
│   ├── evaluation_metrics.csv   # Model performance metrics
│   ├── forecast_junction_*.csv  # Forecast results per junction
│   └── figures/                 # All generated plots
│
├── models/                      # Saved trained models
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Steps to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-city-traffic-forecast.git
cd smart-city-traffic-forecast
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ensure Dataset is in `data/`
Place the `traffic.csv` file in the `data/` directory (included in the repo).

### 4. Run the Full Pipeline
```bash
python src/main.py
```

This will:
- Load and preprocess the data
- Generate EDA visualizations in `reports/figures/`
- Train SARIMA and Prophet models for all 4 junctions
- Evaluate models and save metrics
- Generate 30-day future forecasts
- Save all results to `reports/`

### 5. (Optional) Run Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

## 📊 Results Summary

### Models Used
| Model | Description |
|-------|-------------|
| **SARIMA** | Seasonal ARIMA with hourly seasonality (m=24), auto-tuned via `pmdarima` |
| **Prophet** | Facebook's Prophet with daily, weekly, and yearly seasonality |

### Key Findings
- **Junction 2** has the highest traffic volume across all periods
- **Peak hours** are typically between 7-9 AM and 4-7 PM (rush hours)
- **Weekday traffic** is significantly higher than weekend traffic
- **Traffic shows strong daily and weekly seasonality**

### Evaluation Metrics
Models are evaluated on a held-out test set (last 30 days) using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

See `reports/evaluation_metrics.csv` for detailed results per junction.

### Generated Visualizations
- Traffic trends over time per junction
- Hourly traffic distribution
- Weekday vs weekend patterns
- Monthly traffic trends
- Traffic heatmap (Hour × Day of Week)
- Day-of-week box plots
- Actual vs predicted comparisons
- Future forecast plots
- Metrics comparison charts

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** — Data manipulation
- **numpy** — Numerical computing
- **matplotlib / seaborn** — Visualization
- **statsmodels** — SARIMA modeling
- **pmdarima** — Auto ARIMA parameter selection
- **prophet** — Facebook Prophet forecasting
- **scikit-learn** — Evaluation metrics

## 📄 Report

A comprehensive project report is available at `reports/final_report.md` covering:
1. Introduction
2. Dataset Description
3. Data Preprocessing
4. Exploratory Data Analysis
5. Model Building
6. Model Evaluation
7. Forecast Results
8. Conclusion & Recommendations

## 📝 License

This project is for educational and research purposes.

## 👤 Author

Smart City Traffic Forecasting Project — Data Science Internship
