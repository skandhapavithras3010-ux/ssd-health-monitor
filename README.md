# 💾 # AI-Based SSD Health Monitoring System

Predictive SSD health monitoring using Machine Learning techniques for health classification, remaining useful life (RUL) estimation, and anomaly detection.

## Dashboard Preview

![SSD Health Dashboard](reports/dashboard_preview.png)

## Features

- SSD Health Classification (Healthy / Warning / Critical)
- Remaining Useful Life (RUL) Prediction
- Anomaly Detection using Isolation Forest
- Composite Health Index (0–100 scoring system)
- Interactive Streamlit-based real-time dashboard

##  Model Performance

- Classification Model: Random Forest Classifier  
- Regression Model: Random Forest Regressor (RUL prediction)  
- Anomaly Detection: Isolation Forest (5% contamination)  
- Dataset Size: 8,000 SSD telemetry records  
- Feature Set: 6 SMART-inspired disk health features

## System Overview

The system simulates SSD telemetry data and applies machine learning models to:
- Predict SSD health state
- Estimate remaining useful life
- Detect abnormal behavior patterns
- Compute a unified health score (0–100)
  
## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Models

- Random Forest Classifier
- Random Forest Regressor
- Isolation Forest

## Project Structure

```text
data/        - Dataset and processed files
models/      - Trained ML models
notebooks/   - EDA, preprocessing and model development
reports/     - Visualizations and evaluation results
scripts/     - Verification and testing scripts
dashboard/   - Streamlit dashboard
```
