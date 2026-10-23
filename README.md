# FlowCast: Smart Traffic Flow Prediction System
> **An End-to-End AI System for Forecasting Urban Traffic Volume, Congestion, Travel Time, and Incident Risk Across an Arterial Network.**  
> *Meridian Mobility Systems (MMS) · Northline Corridor AI Engineering Project*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚦 Overview

**FlowCast** is a production-grade traffic intelligence decision-support platform built for metropolitan traffic operations centers. Operating on inductive loop and radar sensor telemetry across the **25-segment Northline Corridor**, FlowCast ingests over **176,000 multi-source records** (traffic logs, meteorological feeds, and calendar events) to deliver short-horizon predictions (30–120 minutes ahead) with quantified uncertainty.

The project fulfills all engineering requirements of the **FlowCast PRD v1.0**, featuring:
- **Data Engineering Pipeline:** Ingestion, quarantine, deduplication, physical outlier correction, causal feature generation, and leakage protection.
- **First-Principles Mathematics:** Linear Regression trained from scratch using pure NumPy matrix multiplications and Mini-Batch Gradient Descent.
- **Classical ML Suite:** Linear Regression, Ridge, Decision Trees, Random Forests, XGBoost, and Support Vector Machines.
- **Deep Learning Recurrent Model:** Multi-task 2-layer LSTM sequence forecaster built and trained from scratch in PyTorch with CUDA acceleration.
- **Interactive Analytics Dashboard:** Streamlit application providing 9 real-time operational views, scenario simulator, and batch CSV upload.

---

## 📊 Benchmark Scoreboard (May 2025 Test Window)

Evaluated on the unseen, held-out test split (**May 1 – May 31, 2025**; 36,267 records):

| Paradigm | Model Architecture | Primary Target | Test RMSE | Test MAE | Test MAPE (%) | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **From-Scratch Math** | **Scratch Linear Regression (NumPy GD)** | Traffic Volume | 75.94 | 54.47 | 15.01% | 0.9296 |
| **Classical ML** | Ridge Regression | Traffic Volume | 75.32 | 53.98 | 14.75% | 0.9307 |
| **Classical ML** | Decision Tree Regressor | Traffic Volume | 65.35 | 44.00 | 10.23% | 0.9478 |
| **Classical ML** | Random Forest Regressor | Traffic Volume | 58.72 | 39.75 | **9.29%** | 0.9579 |
| **Classical ML** | **Gradient Boosting (XGBoost)** | Traffic Volume | **57.74** | **39.54** | **9.35%** | **0.9593** |
| **Deep Learning** | **FlowCast LSTM (PyTorch Scratch)** | Traffic Volume | 59.38 | 40.61 | 9.68% | 0.9569 |

### Operational Success Targets Audit:
- ✅ **Volume Forecast Accuracy:** Achieved **9.29% MAPE** (beats PRD target $\le 12.0\%$).
- ✅ **Congestion Classification:** Achieved **0.7635 Macro-F1** and **87.05% Accuracy** across 4 classes.
- ✅ **Travel Time Estimation:** Achieved **1.08 min RMSE** and **9.91% MAPE**.
- ✅ **Mathematical Transparency:** First-principles gradient descent verified with $R^2 = 0.9296$.
- ✅ **Full Reproducibility:** Entire pipeline reruns end-to-end via one script.

---

## 📁 Repository Structure

```
Traffic Flow Pred/
├── config.yaml                    # Master pipeline configuration & hyperparameters
├── requirements.txt               # Pinned Python package dependencies
├── run_pipeline.py                # Master 1-command reproduction script
├── README.md                      # Project documentation and reproduction guide
├── data/
│   ├── raw/                       # Source CSV tables as delivered
│   │   ├── traffic_sensor_log.csv
│   │   ├── weather_observations.csv
│   │   └── calendar_events.csv
│   ├── interim/                   # Quarantined anomalies and interim merged sets
│   └── processed/                 # Analysis-ready, versioned Parquet dataset (176k rows)
├── src/
│   ├── __init__.py
│   ├── ingest.py                  # M1: Schema validation & quarantine
│   ├── clean.py                   # M2: Deduplication, outlier repair & merge
│   ├── features.py                # M3: Causal lags, rolling stats & encodings
│   ├── eda.py                     # M4: EDA charts & data quality reporting
│   ├── ml_models.py               # M5: Classical ML & NumPy Scratch LR
│   ├── dl_model.py                # M6: PyTorch sequence LSTM engine
│   ├── evaluate.py                # Metrics scoreboard & comparison plots
│   └── utils.py                   # Logging, seed setting & config loader
├── models/                        # Persisted models (.joblib, .pt), cards & scoreboard
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_cleaning.ipynb          # Data cleaning and audit
│   └── 03_modelling.ipynb         # Model training and benchmarking
├── dashboard/
│   └── app.py                     # M7: Streamlit Operations Dashboard (11 views)
├── reports/
│   ├── data_quality.md            # Comprehensive Data Quality Audit Report
│   ├── final_report.md            # Engineering Capstone Technical Report
│   └── figures/                   # High-resolution publication figures
└── tests/
    ├── __init__.py
    └── test_pipeline.py           # Unit and integration test suite
```

---

## 🚀 Quickstart & Pipeline Reproduction

### 1. Installation
Ensure Python 3.11+ is installed. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Reproduction
Run the complete pipeline from raw data to trained models and evaluation benchmarks:
```bash
python run_pipeline.py
```
This single command executes:
1. Raw data ingestion & validation (M1)
2. Data cleaning, outlier imputation, and source merging (M2)
3. Causal feature engineering with zero lookahead leakage (M3)
4. EDA and data quality reporting with 5 figures (M4)
5. Classical ML models + NumPy from-scratch gradient descent baseline (M5)
6. Deep learning sequence LSTM model training with PyTorch & CUDA (M6)
7. Evaluation scoreboard compilation and benchmarking figures export.

### 3. Launching the Interactive Dashboard
Launch the operations dashboard in your browser:
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore:
- **Live Prediction:** Segment volume, travel time, and congestion forecasts.
- **Historical Trends:** Diurnal peak profiles and day-of-week demand.
- **Congestion Heatmap:** Spatial-temporal 2D detector occupancy grid.
- **Road Comparison:** Side-by-side capacity utilization and volume rankings.
- **Model Performance:** Head-to-head model comparison bar charts.
- **Feature Importance:** Top predictive feature drivers.
- **Forecast Visualisation:** 7-day actual vs. predicted overlay with confidence intervals.
- **Prediction Confidence:** Residual distributions and uncertainty bands.
- **Weather vs Traffic:** Rain and visibility degradation impact curves.
- **What-If Scenario Simulator:** Interactive slider simulator for operational planning.
- **Batch Upload & Reports:** CSV batch inference and summary download.

### 4. Running Tests
Execute the unit and integration test suite:
```bash
python tests/test_pipeline.py
```

---

## 📐 Mathematical Formulation Highlights

### First-Principles NumPy Gradient Descent
The baseline regressor explicitly models the relationship without scikit-learn:
$$\hat{y} = X w + b$$
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$
$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

### Causal Feature Rolling Statistics (Leakage-Free)
Rolling volume and speed statistics are calculated strictly over previous windows using `.shift(1)`:
$$\mu_{\text{roll}}(t) = \frac{1}{k} \sum_{i=1}^{k} y_{t-i}, \quad \sigma_{\text{roll}}(t) = \sqrt{\frac{1}{k-1} \sum_{i=1}^{k} (y_{t-i} - \mu_{\text{roll}}(t))^2}$$

---

## 👥 Credits & Authorship
- **Author:** Dhanush — AI Engineering Division, Meridian Mobility Systems
- **Programme:** Capstone / Final-Year AI Engineering Project
- **Classification:** Internal Engineering — Approved for Build
