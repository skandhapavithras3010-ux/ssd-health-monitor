# 💾 # AI-Based SSD Health Monitoring System

Predictive SSD health monitoring using Machine Learning techniques for health classification, remaining useful life (RUL) estimation, and anomaly detection.

## Dashboard Preview

![SSD Health Dashboard](reports/dashboard_preview.png)

## Problem Statement

Unexpected SSD failures can lead to downtime, data loss, and increased maintenance costs. This project explores machine learning techniques to predict SSD health, estimate remaining useful life (RUL), and detect anomalous behavior before failure occurs.

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
- Dataset Size: 8,000 synthetic SSD telemetry records  
- Feature Set: 6 SMART-inspired disk health features

## Results

### SSD Health Classification
- Accuracy: 94%
- Weighted F1 Score: 94%

### Remaining Useful Life (RUL) Prediction
- Mean Absolute Error (MAE): 258 hours
- Root Mean Squared Error (RMSE): 320 hours
- RMSE corresponds to 3.2% of the full RUL range

### Anomaly Detection
- Isolation Forest detected 416 anomalous SSD records
- Anomalies represented 5.2% of the total dataset

### Feature Importance Analysis
- Wear Level contributed approximately 80% of total feature importance
- Bad Blocks, Write Cycles, and Power-On Hours were secondary indicators of SSD degradation
  
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
## Future Improvements

- Train and validate models on real-world SMART telemetry datasets
- Integrate SHAP-based model explainability
- Enable real-time SSD monitoring and alert generation
- Deploy the dashboard using Streamlit Cloud
- Explore deep learning approaches for RUL prediction
