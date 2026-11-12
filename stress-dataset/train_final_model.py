
from pathlib import Path
import json
import joblib
import pandas as pd

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "dataset" / "wesad_all_features.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "stress_random_forest_final.joblib"
METADATA_FILE = MODEL_DIR / "final_model_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# FEATURES & TARGET
# ==========================================

FEATURES = [
    "heart_rate",
    "skin_temp",
    "motion_level"
]

TARGET = "target"

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading WESAD dataset...")

df = pd.read_csv(DATASET_FILE)

required_columns = FEATURES + [TARGET, "subject"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

X = df[FEATURES]
y = df[TARGET].astype(int)

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\n===== FINAL TRAINING DATASET =====")

print("Samples:", len(df))
print("Subjects:", df["subject"].nunique())

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nSubject list:")
print(sorted(df["subject"].unique()))

# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Final Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\n===== FEATURE IMPORTANCE =====")

for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{feature}: {importance:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, MODEL_FILE)

metadata = {
    "model_name": "Random Forest",
    "dataset": "WESAD",
    "samples": int(len(df)),
    "subjects": sorted(df["subject"].unique().tolist()),
    "features": FEATURES,
    "target": TARGET,
    "classes": {
        "0": "Non-stress",
        "1": "Stress"
    },
    "parameters": {
        "n_estimators": 300,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42
    },
    "training_date": datetime.now().isoformat(),
    "evaluation_reference": {
        "method": "Leave-One-Subject-Out",
        "accuracy": 0.8442,
        "precision": 0.6245,
        "recall": 0.7725,
        "f1_score": 0.6907,
        "roc_auc": 0.8845
    }
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n===== FINAL MODEL SAVED =====")
print("Model:", MODEL_FILE)
print("Metadata:", METADATA_FILE)

print("\nFinal training completed!")
