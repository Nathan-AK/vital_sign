
import pandas as pd
from pathlib import Path

CSV_FILE = Path("stress-dataset/dataset/esp32_stress_dataset.csv")

df = pd.read_csv(CSV_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Separate valid and invalid HR
valid_hr = df[df["heart_rate"] > 0].copy()
invalid_hr = df[df["heart_rate"] <= 0].copy()

print("\n========== VALID DATA AUDIT ==========")

print("\nTotal samples:", len(df))
print("Valid HR samples:", len(valid_hr))
print("Invalid HR samples:", len(invalid_hr))

print("\nValid HR percentage:",
      round(len(valid_hr) / len(df) * 100, 2), "%")

print("\n========== VALID SENSOR STATISTICS ==========")

print("\nHeart Rate:")
print(valid_hr["heart_rate"].describe())

print("\nSkin Temperature:")
print(valid_hr["skin_temp"].describe())

print("\nMotion Level:")
print(valid_hr["motion_level"].describe())

print("\n========== LABEL DISTRIBUTION ==========")

print("\nAll samples:")
print(df["stress_status"].value_counts().sort_index())

print("\nSamples with valid HR:")
print(valid_hr["stress_status"].value_counts().sort_index())

print("\n========== LABEL VS HEART RATE ==========")

print(
    valid_hr.groupby("stress_status")["heart_rate"]
    .agg(["count", "mean", "min", "max"])
)

print("\n========== SENSOR RANGE CHECK ==========")

print("Temperature below 25°C:",
      (df["skin_temp"] < 25).sum())

print("Temperature above 40°C:",
      (df["skin_temp"] > 40).sum())

print("Motion above 3:",
      (df["motion_level"] > 3).sum())

print("\n========== TIMESTAMP CHECK ==========")

intervals = df["timestamp"].diff().dt.total_seconds()

print("Average interval:", intervals.mean(), "seconds")
print("Largest interval:", intervals.max(), "seconds")

print("\n========== AUDIT COMPLETE ==========")
