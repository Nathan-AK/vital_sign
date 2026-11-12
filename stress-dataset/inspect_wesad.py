
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
PKL_FILE = BASE_DIR / "dataset" / "wesad" / "S2" / "S2.pkl"

print("Loading WESAD dataset...")

with open(PKL_FILE, "rb") as file:
    data = pickle.load(file, encoding="latin1")

print("\n=== DATASET STRUCTURE ===")
print("Top-level keys:", data.keys())

print("\n=== SUBJECT ===")
print("Subject:", data.get("subject"))

print("\n=== SIGNAL STRUCTURE ===")
signal = data["signal"]

for device in signal:
    print(f"\nDevice: {device}")

    for sensor, values in signal[device].items():
        arr = np.asarray(values)
        print(
            f"  {sensor}: shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

print("\n=== LABEL INFORMATION ===")
labels = np.asarray(data["label"])

print("Label shape:", labels.shape)
print("Label distribution:")

for label, count in sorted(Counter(labels.tolist()).items()):
    print(f"  Label {label}: {count:,} samples")

print("\nWESAD inspection completed.")
