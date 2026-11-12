
import os
import csv
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
CSV_FILE = BASE_DIR / "dataset" / "esp32_stress_dataset.csv"

URL = "https://blynk.cloud/external/api/getAll"
POLL_INTERVAL = 1

# Load token dari stress-dataset/.env
load_dotenv(ENV_FILE)

TOKEN = os.getenv("BLYNK_AUTH_TOKEN")

headers = [
    "timestamp",
    "heart_rate",
    "skin_temp",
    "motion_level",
    "stress_status",
    "buzzer_state"
]

if not TOKEN:
    raise ValueError(
        "BLYNK_AUTH_TOKEN tidak ditemukan. Periksa file stress-dataset/.env"
    )

CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def get_pin(data, pin):
    value = data.get(pin.lower())

    if isinstance(value, list):
        value = value[0] if value else None

    if value is None:
        raise ValueError(f"Data {pin} tidak ditemukan di Blynk")

    return float(value)


print("====================================")
print(" BLYNK SENSOR DATA COLLECTOR")
print("====================================")
print(f"CSV: {CSV_FILE}")
print("Mengambil data dari Blynk...")
print("Tekan CTRL+C untuk berhenti.\n")

session = requests.Session()
total_saved = 0

try:
    while True:
        try:
            response = session.get(
                URL,
                params={"token": TOKEN},
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            heart_rate = get_pin(data, "V0")
            skin_temp = get_pin(data, "V1")
            motion = get_pin(data, "V2")
            status = int(get_pin(data, "V3"))
            buzzer = int(get_pin(data, "V4"))

            timestamp = datetime.now().isoformat(
                timespec="seconds"
            )

            # Simpan data ke CSV
            with open(CSV_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    heart_rate,
                    skin_temp,
                    motion,
                    status,
                    buzzer
                ])

            total_saved += 1

            print(
                f"[{total_saved}] "
                f"HR: {heart_rate:.1f} BPM | "
                f"Temp: {skin_temp:.2f} C | "
                f"Motion: {motion:.2f} | "
                f"Status: {status} | "
                f"Buzzer: {buzzer}"
            )

        except requests.exceptions.RequestException as error:
            print("Koneksi Blynk error:", error)

        except (ValueError, KeyError) as error:
            print("Pembacaan data error:", error)

        except Exception as error:
            print("Error:", error)

        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print("\nPengambilan data dihentikan.")

finally:
    session.close()
    print(f"Total data tersimpan: {total_saved}")
    print(f"Lokasi file: {CSV_FILE}")
