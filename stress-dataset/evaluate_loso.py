
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset" / "wesad_all_features.csv"

FEATURES = ["heart_rate", "skin_temp", "motion_level"]

df = pd.read_csv(DATASET_FILE)

X = df[FEATURES]
y = df["target"].astype(int)
groups = df["subject"]

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

logo = LeaveOneGroupOut()

print("Running Leave-One-Subject-Out validation...")

predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=logo,
    method="predict",
    n_jobs=1,
)

probabilities = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=logo,
    method="predict_proba",
    n_jobs=1,
)

stress_index = 1
stress_probabilities = probabilities[:, stress_index]

print("\n===== LOSO RESULTS =====")
print("Subjects:", groups.nunique())
print("Samples:", len(df))

print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y, predictions, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y, predictions, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y, stress_probabilities):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Non-stress", "Stress"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions, labels=[0, 1]))

print("\n===== RESULTS PER SUBJECT =====")

results = pd.DataFrame({
    "subject": groups,
    "actual": y,
    "predicted": predictions,
})

for subject, group in results.groupby("subject"):
    print(
        f"{subject}: "
        f"Accuracy={accuracy_score(group['actual'], group['predicted']):.3f}, "
        f"F1={f1_score(group['actual'], group['predicted'], zero_division=0):.3f}"
    )
