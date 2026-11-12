
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

logo = LeaveOneGroupOut()

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ]),
}

results = []

print("===== MODEL COMPARISON: LOSO =====")
print(f"Subjects: {groups.nunique()}")
print(f"Samples : {len(df)}")

for name, model in models.items():
    print(f"\nEvaluating {name}...")

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

    # Target classes are 0 = non-stress, 1 = stress.
    stress_probabilities = probabilities[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    auc = roc_auc_score(y, stress_probabilities)

    cm = confusion_matrix(y, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc,
    })

    print(f"\n===== {name} RESULTS =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        target_names=["Non-stress", "Stress"],
        zero_division=0,
    ))

comparison = pd.DataFrame(results)

print("\n===== FINAL COMPARISON =====")
print(comparison.to_string(index=False))

output_file = BASE_DIR / "model_comparison_loso.csv"
comparison.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
