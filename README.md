# IoT-Based Vital Sign Monitoring System for Stress Classification

An AIoT-based system that integrates ESP32 and Machine Learning to monitor vital signs and classify stress conditions in real-time.

## Project Overview

This project uses physiological and motion data to predict two conditions: Non-Stress and Stress. Sensor readings are collected by ESP32, processed using a Random Forest model, and displayed through the Blynk IoT dashboard.

## Hardware Components

- ESP32
- MAX30102 – Heart Rate Sensor
- DS18B20 – Skin Temperature Sensor
- MPU6050 – Motion Sensor
- RGB LED
- Active Buzzer

## Technologies Used

- Arduino / PlatformIO
- Python
- Random Forest
- Blynk IoT
- WESAD Dataset
- Scikit-learn

## System Architecture

1. Sensors collect heart rate, skin temperature, and motion data.
2. ESP32 sends sensor readings to Blynk Cloud.
3. A Python application retrieves the data through the Blynk HTTPS REST API.
4. The Random Forest model predicts Non-Stress or Stress.
5. Prediction and confidence values are sent back to Blynk.
6. The ESP32 controls the RGB LED and buzzer based on the ML prediction.

## Machine Learning Evaluation

The Random Forest model was evaluated using Leave-One-Subject-Out (LOSO) validation.

| Metric | Result |
|---|---:|
| Accuracy | 84.42% |
| Precision | 62.45% |
| Recall | 77.25% |
| F1-score | 69.07% |
| ROC-AUC | 88.45% |

## Project Limitations

- This project is an experimental prototype, not a medical diagnostic device.
- Skin temperature is not equivalent to core body temperature.
- Real-time predictions may be affected by differences between the WESAD dataset and the ESP32 sensors.

## Dataset

The model was developed using the WESAD dataset.

Source: https://archive.ics.uci.edu/dataset/465/wesad

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt

# Required Python Libraries
numpy
pandas
scikit-learn
joblib
requests
python-dotenv