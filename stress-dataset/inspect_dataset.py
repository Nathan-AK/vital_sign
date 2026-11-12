
import pandas as pd
from pathlib import Path

# Dataset location
CSV_FILE = Path("stress-dataset/dataset/esp32_stress_dataset.csv")

print("========== DATASET INSPECTION ==========")

# Check file
if not CSV_FILE.exists():
    print("ERROR: CSV dataset tidak ditemukan!")
    exit()

# Read CSV
df = pd.read_csv(CSV_FILE)

# Basic information
print("\n[1] DATASET INFORMATION")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Column names:", df.columns.tolist())

# Preview
print("\n[2] FIRST 5 ROWS")
print(df.head())

print("\n[3] LAST 5 ROWS")
print(df.tail())

# Missing values
print("\n[4] MISSING VALUES")
print(df.isnull().sum())

# Data types
print("\n[5] DATA TYPES")
print(df.dtypes)

# Statistics
print("\n[6] STATISTICAL SUMMARY")
print(df.describe())

# Label distribution
print("\n[7] STRESS STATUS DISTRIBUTION")
print(df["stress_status"].value_counts().sort_index())

# Duplicate data
print("\n[8] DUPLICATE ROWS")
print(df.duplicated().sum())

# Invalid values
print("\n[9] INVALID SENSOR READINGS")
print("Invalid HR:", (df["heart_rate"] <= 0).sum())
print("Invalid skin temperature:", (df["skin_temp"] <= 0).sum())
print("Invalid motion:", (df["motion_level"] < 0).sum())

print("\n========== INSPECTION COMPLETE ==========")
