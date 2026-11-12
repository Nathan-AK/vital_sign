
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset" / "wesad_all_features.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "heart_rate",
    "skin_temp",
    "motion_level",
]

TARGET = "target"

TRAIN_SUBJECTS = [
    "S2", "S3", "S4",
    "S10", "S11", "S13", "S14", "S15"
]

TEST_SUBJECTS = ["S16", "S17"]

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading WESAD dataset...")

df = pd.read_csv(DATASET_FILE)

required_columns = FEATURES + [TARGET, "subject"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df.dropna(subset=required_columns).copy()

# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================

train_df = df[df["subject"].isin(TRAIN_SUBJECTS)].copy()
test_df = df[df["subject"].isin(TEST_SUBJECTS)].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing dataset is empty.")

if set(train_df["subject"]) & set(test_df["subject"]):
    raise ValueError("Subject leakage detected!")

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)

print("\n===== DATASET SPLIT =====")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining subjects:")
print(sorted(train_df["subject"].unique()))

print("\nTesting subjects:")
print(sorted(test_df["subject"].unique()))

print("\nTraining labels:")
print(y_train.value_counts().sort_index())

print("\nTesting labels:")
print(y_test.value_counts().sort_index())

# Both classes must exist in training and testing
if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError("Both classes must exist in training and testing.")

# ============================================================
# RANDOM FOREST TRAINING
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training completed.")

# ============================================================
# MODEL EVALUATION
# ============================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

stress_class_index = list(model.classes_).index(1)
stress_probabilities = probabilities[:, stress_class_index]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
roc_auc = roc_auc_score(y_test, stress_probabilities)

print("\n===== MODEL EVALUATION =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-stress", "Stress"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values(
    by="importance",
    ascending=False,
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))

# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

model_path = MODEL_DIR / "stress_random_forest.joblib"
metadata_path = MODEL_DIR / "model_metadata.json"

joblib.dump(model, model_path)

metadata = {
    "model": "RandomForestClassifier",
    "features": FEATURES,
    "target": TARGET,
    "classes": {
        "0": "Non-stress",
        "1": "Stress",
    },
    "train_subjects": TRAIN_SUBJECTS,
    "test_subjects": TEST_SUBJECTS,
    "training_samples": int(len(train_df)),
    "testing_samples": int(len(test_df)),
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    },
}

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("\n===== SAVED FILES =====")
print("Model:", model_path)
print("Metadata:", metadata_path)
print("\nTraining pipeline completed successfully.")
