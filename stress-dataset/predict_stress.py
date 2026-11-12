
from pathlib import Path
import joblib
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "stress_random_forest_final.joblib"

FEATURES = [
    "heart_rate",
    "skin_temp",
    "motion_level"
]

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_stress(heart_rate, skin_temp, motion_level):

    input_data = pd.DataFrame([{
        "heart_rate": heart_rate,
        "skin_temp": skin_temp,
        "motion_level": motion_level
    }], columns=FEATURES)

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    class_index = list(model.classes_).index(prediction)

    confidence = probabilities[class_index] * 100

    status = "STRESS" if prediction == 1 else "NON-STRESS"

    return {
        "prediction": int(prediction),
        "status": status,
        "confidence": round(float(confidence), 2)
    }

# ==========================================
# TEST PREDICTIONS
# ==========================================

test_cases = [
    {
        "name": "Example 1",
        "heart_rate": 70,
        "skin_temp": 32.5,
        "motion_level": 1.0
    },
    {
        "name": "Example 2",
        "heart_rate": 110,
        "skin_temp": 32.0,
        "motion_level": 5.0
    },
    {
        "name": "Example 3",
        "heart_rate": 85,
        "skin_temp": 33.0,
        "motion_level": 2.0
    }
]

print("\n===== STRESS PREDICTION TEST =====")

for case in test_cases:

    result = predict_stress(
        case["heart_rate"],
        case["skin_temp"],
        case["motion_level"]
    )

    print(f"\n{case['name']}")
    print(f"Heart Rate : {case['heart_rate']} BPM")
    print(f"Skin Temp  : {case['skin_temp']} °C")
    print(f"Motion     : {case['motion_level']}")

    print(f"Prediction : {result['status']}")
    print(f"Confidence : {result['confidence']}%")
