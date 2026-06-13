import os

MODELS = [
    "models/classifier.pkl",
    "models/regression.pkl",
    "models/anomaly_detector.pkl",
    "models/scaler.pkl",
    "models/label_mapping.pkl",
]

all_ok = True

print("=== MODEL VERIFICATION ===\n")

for path in MODELS:
    exists = os.path.exists(path)
    size = os.path.getsize(path) // 1024 if exists else 0

    status = "OK" if exists else "MISSING"

    print(f"[{status}] {path} ({size} KB)")

    if not exists:
        all_ok = False

print("\nAll models ready:", all_ok)