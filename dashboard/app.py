import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SSD Health Monitor",
    page_icon="💾",
    layout="wide"
)

st.title("💾 AI-Based SSD Health Monitoring System")
st.markdown("*Predictive flash storage health using Random Forest + Isolation Forest*")


@st.cache_resource
def load_models():
    return {
        "scaler": joblib.load("models/scaler.pkl"),
        "clf": joblib.load("models/classifier.pkl"),
        "reg": joblib.load("models/regression.pkl"),
        "iso": joblib.load("models/anomaly_detector.pkl"),
        "mapping": joblib.load("models/label_mapping.pkl"),
    }


models = load_models()

# Reverse mapping
reverse_mapping = {v: k for k, v in models["mapping"].items()}

# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.header("Device Parameters")
st.sidebar.markdown("Adjust the sliders to simulate a device:")

temp = st.sidebar.slider("Temperature (°C)", 20, 80, 45)

wear = st.sidebar.slider("Wear Level (%)", 0, 100, 30)

poh = st.sidebar.number_input(
    "Power-On Hours",
    min_value=100,
    max_value=50000,
    value=10000,
    step=500
)

wc = st.sidebar.number_input(
    "Write Cycles",
    min_value=500,
    max_value=100000,
    value=20000,
    step=1000
)

ecc = st.sidebar.number_input(
    "ECC Error Rate",
    min_value=0.0,
    max_value=0.5,
    value=0.01,
    step=0.005,
    format="%.3f"
)

bb = st.sidebar.number_input(
    "Bad Blocks",
    min_value=0,
    max_value=50,
    value=2,
    step=1
)

# ── Prediction Pipeline ─────────────────────────────────────────

feature_names = [
    "temperature",
    "wear_level",
    "power_on_hours",
    "write_cycles",
    "ECC_error_rate",
    "bad_blocks"
]

features = pd.DataFrame(
    [[temp, wear, poh, wc, ecc, bb]],
    columns=feature_names
)

scaled = models["scaler"].transform(features)

health_code = models["clf"].predict(scaled)[0]
health = reverse_mapping[health_code]

rul = int(models["reg"].predict(scaled)[0])

anomaly = models["iso"].predict(scaled)[0]

# ── Health Index ────────────────────────────────────────────────

hi = round(
    max(
        0,
        min(
            100,
            (1 - (temp - 20) / 60) * 15
            + (1 - wear / 100) * 25
            + (1 - min(ecc / 0.2, 1)) * 25
            + (1 - min(bb / 20, 1)) * 15
            + (min(rul / 10000, 1)) * 20,
        ),
    ),
    1,
)

# ── Metric Cards ────────────────────────────────────────────────

status_icon = {
    "Healthy": "🟢",
    "Warning": "🟡",
    "Critical": "🔴"
}

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Health Status",
    f"{status_icon.get(health, '⚪')} {health}"
)

c2.metric(
    "Remaining Life",
    f"{rul:,} hrs"
)

c3.metric(
    "Anomaly Detected",
    "⚠️ Yes" if anomaly == -1 else "✅ No"
)

c4.metric(
    "Health Index",
    f"{hi}/100"
)

st.divider()

# ── Feature Profile Chart ───────────────────────────────────────

st.subheader("Feature Profile (Normalised 0–1)")

feat_names = [
    "Temperature",
    "Wear Level",
    "Power-On Hrs",
    "Write Cycles",
    "ECC Rate",
    "Bad Blocks"
]

feat_vals = [
    temp / 80,
    wear / 100,
    poh / 50000,
    wc / 100000,
    min(ecc / 0.5, 1),
    min(bb / 50, 1),
]

bar_colors = [
    "#E74C3C" if v > 0.7
    else "#F39C12" if v > 0.4
    else "#2ECC71"
    for v in feat_vals
]

fig, ax = plt.subplots(figsize=(9, 3))

ax.barh(
    feat_names,
    feat_vals,
    color=bar_colors
)

ax.set_xlim(0, 1)

ax.set_xlabel(
    "Normalised value (higher = worse condition)"
)

ax.axvline(
    0.7,
    color="#E74C3C",
    linestyle="--",
    alpha=0.6,
    label="Critical threshold"
)

ax.axvline(
    0.4,
    color="#F39C12",
    linestyle="--",
    alpha=0.6,
    label="Warning threshold"
)

ax.legend(loc="lower right")

ax.set_title(
    f"Device Feature Profile — Health Index: {hi}/100"
)

plt.tight_layout()

st.pyplot(fig)

plt.close()

st.caption(
    "Model: Random Forest (Classification + Regression) | "
    "Anomaly Detection: Isolation Forest"
)