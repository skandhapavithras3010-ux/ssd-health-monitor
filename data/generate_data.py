import numpy as np
import pandas as pd
 
np.random.seed(42)
n = 8000
 
temperature    = np.random.normal(45, 10, n).clip(20, 80)
wear_level     = np.random.uniform(0, 100, n)
power_on_hours = np.random.randint(100, 50000, n)
write_cycles   = np.random.randint(500, 100000, n)
ECC_error_rate = np.random.exponential(0.02, n)
bad_blocks     = np.random.poisson(2, n)
 
# RUL decreases as wear and errors increase (realistic relationship)
RUL = (
    10000
    - wear_level * 80
    - bad_blocks * 200
    - ECC_error_rate * 1000
    + np.random.normal(0, 300, n)
).clip(0, 10000)
 
# Label health based on RUL thresholds
def label(rul):
    if rul > 6000:  return "Healthy"
    elif rul > 2000: return "Warning"
    else:            return "Critical"
 
health_status = [label(r) for r in RUL]
 
df = pd.DataFrame({
    "temperature":    temperature.round(1),
    "wear_level":     wear_level.round(1),
    "power_on_hours": power_on_hours,
    "write_cycles":   write_cycles,
    "ECC_error_rate": ECC_error_rate.round(4),
    "bad_blocks":     bad_blocks,
    "RUL":            RUL.round(0).astype(int),
    "health_status":  health_status
})
 
df.to_csv("data/ssd_data.csv", index=False)
print("Dataset saved!")
print(df["health_status"].value_counts())
print(df.describe())
