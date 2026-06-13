import joblib
import numpy as np

print("=== PIPELINE TEST ===\n")

# Load saved artifacts
scaler = joblib.load("models/scaler.pkl")
clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regression.pkl")
iso = joblib.load("models/anomaly_detector.pkl")

mapping = joblib.load("models/label_mapping.pkl")
reverse_mapping = {v: k for k, v in mapping.items()}

# Sample SSD:
# High wear, high temperature, many errors
sample = np.array([
    [62, 85, 35000, 80000, 0.08, 9]
])

# Scale features
scaled = scaler.transform(sample)

# Predictions
health_code = clf.predict(scaled)[0]
health = reverse_mapping[health_code]

rul = int(reg.predict(scaled)[0])

anomaly = iso.predict(scaled)[0]
anomaly_text = "Yes" if anomaly == -1 else "No"

# Results
print(f"Health:  {health}")
print(f"RUL:     {rul} hours")
print(f"Anomaly: {anomaly_text}")

print("\nPipeline test: PASSED")