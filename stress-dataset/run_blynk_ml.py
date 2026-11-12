
import os
import time
import requests
import joblib
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

TOKEN = os.getenv("BLYNK_AUTH_TOKEN")

MODEL_FILE = BASE_DIR / "models" / "stress_random_forest_final.joblib"

BLYNK_URL = "https://blynk.cloud/external/api"

POLL_INTERVAL = 3

FEATURES = [
    "heart_rate",
    "skin_temp",
    "motion_level"
]

# ==========================================
# VALIDATE CONFIGURATION
# ==========================================

if not TOKEN:
    raise ValueError(
        "BLYNK_AUTH_TOKEN tidak ditemukan di stress-dataset/.env"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_FILE}"
    )

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# ==========================================
# BLYNK FUNCTIONS
# ==========================================

def get_sensor_data():

    response = requests.get(
        f"{BLYNK_URL}/getAll",
        params={"token": TOKEN},
        timeout=(5, 15)
    )

    response.raise_for_status()

    data = response.json()

    heart_rate = float(data["v0"])
    skin_temp = float(data["v1"])
    motion_level = float(data["v2"])

    return heart_rate, skin_temp, motion_level


def send_prediction(prediction, confidence):

    confidence = round(float(confidence), 2)

    # Update V5
    response_v5 = requests.get(
        f"{BLYNK_URL}/update",
        params={
            "token": TOKEN,
            "v5": int(prediction)
        },
        timeout=(5, 15)
    )

    response_v5.raise_for_status()

    # Update V6
    response_v6 = requests.get(
        f"{BLYNK_URL}/update",
        params={
            "token": TOKEN,
            "v6": confidence
        },
        timeout=(5, 15)
    )

    response_v6.raise_for_status()

    return True

# ==========================================
# ML PREDICTION
# ==========================================

def predict_stress(heart_rate, skin_temp, motion_level):

    input_data = pd.DataFrame(
        [{
            "heart_rate": heart_rate,
            "skin_temp": skin_temp,
            "motion_level": motion_level
        }],
        columns=FEATURES
    )

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    class_index = list(model.classes_).index(prediction)

    confidence = float(
        probabilities[class_index] * 100
    )

    return prediction, confidence


# ==========================================
# MAIN PROGRAM
# ==========================================

print("\n====================================")
print("      REAL-TIME BLYNK ML SYSTEM")
print("====================================")

# ------------------------------------------
# START MAIN LOOP
# ------------------------------------------

print("Reading sensor data from Blynk...")
print("Press CTRL + C to stop.\n")

while True:

    try:

        # READ SENSOR DATA
        heart_rate, skin_temp, motion_level = get_sensor_data()

        # VALIDATE SENSOR DATA
        if heart_rate <= 0:

            print("Invalid heart rate. Waiting...")
            time.sleep(POLL_INTERVAL)
            continue

        if not 0 <= skin_temp <= 50:

            print("Invalid skin temperature. Skipping...")
            time.sleep(POLL_INTERVAL)
            continue

        # RUN ML MODEL
        prediction, confidence = predict_stress(
            heart_rate,
            skin_temp,
            motion_level
        )

        status = (
            "STRESS"
            if prediction == 1
            else "NON-STRESS"
        )

        # SEND PREDICTION
        send_prediction(
            prediction,
            confidence
        )

        # VERIFY BLYNK VALUES
        # saved_prediction, saved_confidence = verify_prediction()

        # DISPLAY RESULTS
        print("------------------------------------")

        print(f"Heart Rate     : {heart_rate:.2f} BPM")
        print(f"Skin Temp      : {skin_temp:.2f} °C")
        print(f"Motion Level   : {motion_level:.2f}")

        print(f"ML Prediction  : {status}")
        print(f"ML Confidence  : {confidence:.2f}%")

        print("------------------------------------\n")

    except requests.exceptions.RequestException as error:

        print(f"Blynk connection error: {error}")

    except (KeyError, TypeError, ValueError) as error:

        print(f"Invalid sensor data: {error}")

    except KeyboardInterrupt:

        print("\nML system stopped.")
        break

    except Exception as error:

        print(f"Unexpected error: {error}")

    time.sleep(POLL_INTERVAL)
