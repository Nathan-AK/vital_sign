
#define BLYNK_TEMPLATE_ID "TMPL6p8JiaZvu"
#define BLYNK_TEMPLATE_NAME "Vital Sign"
#define BLYNK_AUTH_TOKEN "ISI_BLYNK_AUTH_TOKEN"

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BlynkSimpleEsp32.h>

#include "MAX30105.h"
#include "heartRate.h"

// ================= WIFI =================

const char *WIFI_SSID = "ISI_WIFI_NAME";
const char *WIFI_PASS = "ISI_PASS_WIFI";

// ================= PIN ==================

#define DS18B20_PIN 4
#define MPU_ADDR 0x68

const bool ACTUATORS_ENABLED = true;

#define LED_R 19
#define LED_G 18
#define LED_B 17

#define BUZZER_PIN 23
#define BUZZER_ON HIGH

// ================= THRESHOLD =================

const float HR_MILD = 90.0;
const float HR_MODERATE = 100.0;
const float HR_HIGH = 120.0;

const float TEMP_LOW = 28.0;
const float TEMP_HIGH = 35.0;

const float MOTION_LIMIT = 3.0;

const long FINGER_THRESHOLD = 50000;

const int HYSTERESIS_COUNT = 3;

// ================= OBJECTS =================

MAX30105 particleSensor;

OneWire oneWire(DS18B20_PIN);
DallasTemperature tempSensor(&oneWire);

// ================= SENSOR STATUS =================

bool maxOk = false;
bool tempOk = false;
bool mpuOk = false;

// ================= SENSOR VALUES =================

bool fingerDetected = false;

long irValue = 0;

float bpm = 0;
float skinTemp = 0;

float motionLevel = 0;
float motionPeak = 0;

// ================= HEART RATE AVERAGE =================

const byte RATE_SIZE = 4;

byte rates[RATE_SIZE] = {0};

byte rateSpot = 0;
byte rateCount = 0;

unsigned long lastBeat = 0;

// ================= RULE-BASED STATUS =================

// 0 = Normal
// 1 = Mild
// 2 = Moderate
// 3 = High

int stressStatus = 0;

int pendingStatus = 0;
int pendingCount = 0;

int buzzerState = 0;

// ================= MACHINE LEARNING STATUS =================

// 0 = Non-Stress
// 1 = Stress
// -1 = Waiting for ML prediction

int mlPrediction = -1;

// Actual ML-based buzzer state
int mlBuzzerState = 0;

// ================= TIME =================

unsigned long lastTick = 0;
unsigned long lastMotionSample = 0;
unsigned long lastBlynkTry = 0;

// ================= FUNCTION DECLARATIONS =================

void updateLed();
void updateBuzzer();

// =========================================================
// HEART RATE
// =========================================================

void readHeartRate()
{
    if (!maxOk)
        return;

    irValue = particleSensor.getIR();

    fingerDetected = irValue > FINGER_THRESHOLD;

    if (!fingerDetected)
    {
        bpm = 0;

        rateCount = 0;
        rateSpot = 0;

        lastBeat = 0;

        return;
    }

    if (checkForBeat(irValue))
    {
        unsigned long now = millis();

        if (lastBeat > 0)
        {
            float current = 60000.0 / (now - lastBeat);

            if (current > 30 && current < 200)
            {
                rates[rateSpot] = (byte)current;

                rateSpot = (rateSpot + 1) % RATE_SIZE;

                if (rateCount < RATE_SIZE)
                    rateCount++;

                int total = 0;

                for (byte i = 0; i < rateCount; i++)
                    total += rates[i];

                bpm = (float)total / rateCount;
            }
        }

        lastBeat = now;
    }

    if (lastBeat > 0 && millis() - lastBeat > 3000)
    {
        bpm = 0;
        rateCount = 0;
    }
}

// =========================================================
// MPU6050 / MPU6500
// =========================================================

int16_t read16()
{
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();

    return (int16_t)((hi << 8) | lo);
}

bool initMpu()
{
    Wire.beginTransmission(MPU_ADDR);

    if (Wire.endTransmission() != 0)
        return false;

    Wire.beginTransmission(MPU_ADDR);

    Wire.write(0x6B);
    Wire.write(0);

    return Wire.endTransmission() == 0;
}

float readMotion()
{
    Wire.beginTransmission(MPU_ADDR);

    Wire.write(0x3B);

    if (Wire.endTransmission(false) != 0)
        return -1;

    if (Wire.requestFrom((uint16_t)MPU_ADDR, (size_t)6, true) != 6)
        return -1;

    float x = read16() / 16384.0 * 9.81;
    float y = read16() / 16384.0 * 9.81;
    float z = read16() / 16384.0 * 9.81;

    float magnitude = sqrt(x * x + y * y + z * z);

    return fabs(magnitude - 9.81);
}

void sampleMotion()
{
    if (!mpuOk)
        return;

    float m = readMotion();

    if (m >= 0 && m > motionPeak)
        motionPeak = m;
}

// =========================================================
// TEMPERATURE
// =========================================================

void readTemperature()
{
    if (!tempOk)
        return;

    float t = tempSensor.getTempCByIndex(0);

    if (t != DEVICE_DISCONNECTED_C && t > -55 && t < 125)
        skinTemp = t;

    tempSensor.requestTemperatures();
}

// =========================================================
// RULE-BASED CLASSIFICATION
// =========================================================

int hrLevel(float hr)
{
    if (hr >= HR_HIGH)
        return 3;

    if (hr >= HR_MODERATE)
        return 2;

    if (hr >= HR_MILD)
        return 1;

    return 0;
}

void classifyCondition()
{
    int candidate = 0;

    if (fingerDetected && bpm > 0 && motionLevel <= MOTION_LIMIT)
    {
        int score = hrLevel(bpm);

        if (tempOk && (skinTemp < TEMP_LOW || skinTemp > TEMP_HIGH))
            score++;

        candidate = min(score, 3);
    }

    if (candidate == stressStatus)
    {
        pendingCount = 0;
    }
    else
    {
        if (candidate == pendingStatus)
        {
            pendingCount++;
        }
        else
        {
            pendingStatus = candidate;
            pendingCount = 1;
        }

        if (pendingCount >= HYSTERESIS_COUNT)
        {
            stressStatus = pendingStatus;
            pendingCount = 0;
        }
    }

}

// =========================================================
// LED PWM COMPATIBILITY
// =========================================================

#if ESP_ARDUINO_VERSION_MAJOR >= 3

#define LED_ATTACH(pin, ch) ledcAttach(pin, 5000, 8)
#define LED_WRITE(pin, ch, duty) ledcWrite(pin, duty)

#else

#define LED_ATTACH(pin, ch) \
    do { \
        ledcSetup(ch, 5000, 8); \
        ledcAttachPin(pin, ch); \
    } while (0)

#define LED_WRITE(pin, ch, duty) ledcWrite(ch, duty)

#endif

// =========================================================
// RGB LED
// =========================================================

void setLed(uint8_t r, uint8_t g, uint8_t b)
{
    LED_WRITE(LED_R, 0, r);
    LED_WRITE(LED_G, 1, g);
    LED_WRITE(LED_B, 2, b);
}

void initActuators()
{
    if (!ACTUATORS_ENABLED)
        return;

    LED_ATTACH(LED_R, 0);
    LED_ATTACH(LED_G, 1);
    LED_ATTACH(LED_B, 2);

    pinMode(BUZZER_PIN, OUTPUT);

    digitalWrite(BUZZER_PIN, !BUZZER_ON);

    setLed(0, 0, 0);
}

// =========================================================
// ML-BASED LED CONTROL
// =========================================================

void updateLed()
{
    if (!ACTUATORS_ENABLED)
        return;

    // Waiting for ML prediction
    if (mlPrediction == -1)
    {
        setLed(0, 0, 0);
        return;
    }

    // NON-STRESS = GREEN
    if (mlPrediction == 0)
    {
        setLed(0, 255, 0);
    }

    // STRESS = RED
    else if (mlPrediction == 1)
    {
        setLed(255, 0, 0);
    }
}

// =========================================================
// ML-BASED BUZZER CONTROL
// =========================================================

void updateBuzzer()
{
    if (!ACTUATORS_ENABLED)
        return;

    bool on = (mlPrediction == 1);

    mlBuzzerState = on ? 1 : 0;

    digitalWrite(
        BUZZER_PIN,
        on ? BUZZER_ON : !BUZZER_ON
    );
}

// =========================================================
// BLYNK ML CALLBACK
// =========================================================

// Receive ML prediction from Python through V5

BLYNK_WRITE(V5)
{
    int receivedPrediction = param.asInt();

    if (receivedPrediction == 0 || receivedPrediction == 1)
    {
        mlPrediction = receivedPrediction;

        updateLed();
        updateBuzzer();

        Serial.println();
        Serial.println("===== ML PREDICTION RECEIVED =====");

        Serial.print("ML Prediction: ");

        if (mlPrediction == 0)
            Serial.println("NON-STRESS (0)");
        else
            Serial.println("STRESS (1)");

        Serial.print("ML LED Status: ");

        Serial.println(
            mlPrediction == 0 ? "GREEN" : "RED"
        );

        Serial.print("ML Buzzer: ");

        Serial.println(
            mlPrediction == 0 ? "OFF" : "ON"
        );

        Serial.println("=================================");
        Serial.println();
    }
}

// =========================================================
// BLYNK CONNECTION
// =========================================================

BLYNK_CONNECTED()
{
    Serial.println("Blynk connected!");

    // Request the latest ML prediction
    Blynk.syncVirtual(V5);
}

void blynkLoop()
{
    if (WiFi.status() != WL_CONNECTED)
        return;

    if (!Blynk.connected())
    {
        if (millis() - lastBlynkTry > 10000)
        {
            lastBlynkTry = millis();

            Blynk.connect(2000);
        }
    }

    if (Blynk.connected())
        Blynk.run();
}

// =========================================================
// ONE SECOND TASK
// =========================================================

void oneSecondTick()
{
    readTemperature();

    motionLevel = motionPeak;

    motionPeak = 0;

    // Rule-based classification
    classifyCondition();

    // ML controls actuators
    updateLed();
    updateBuzzer();

    if (Blynk.connected())
    {
        // Sensor data
        Blynk.virtualWrite(V0, bpm);
        Blynk.virtualWrite(V1, skinTemp);
        Blynk.virtualWrite(V2, motionLevel);

        // Rule-based results
        Blynk.virtualWrite(V3, stressStatus);
        Blynk.virtualWrite(V4, mlBuzzerState);
    }

    // Serial output
    Serial.print(millis());
    Serial.print(',');

    Serial.print(bpm, 1);
    Serial.print(',');

    Serial.print(skinTemp, 2);
    Serial.print(',');

    Serial.print(motionLevel, 2);
    Serial.print(',');

    Serial.print(stressStatus);
    Serial.print(',');

    Serial.print(buzzerState);
    Serial.print(',');

    Serial.print(fingerDetected ? 1 : 0);
    Serial.print(',');

    Serial.println(mlPrediction);
}

// =========================================================
// SETUP
// =========================================================

void setup()
{
    Serial.begin(115200);

    delay(300);

    Wire.begin(21, 22);

    // MAX30102
    maxOk = particleSensor.begin(Wire, I2C_SPEED_STANDARD);

    if (maxOk)
    {
        particleSensor.setup();

        particleSensor.setPulseAmplitudeRed(0x0A);
        particleSensor.setPulseAmplitudeGreen(0);
    }

    // DS18B20
    tempSensor.begin();

    tempOk = tempSensor.getDeviceCount() > 0;

    if (tempOk)
    {
        tempSensor.setResolution(9);

        tempSensor.setWaitForConversion(false);

        tempSensor.requestTemperatures();
    }

    // MPU6050
    mpuOk = initMpu();

    // Actuators
    initActuators();

    // Sensor initialization status
    Serial.print("# MAX30102: ");
    Serial.println(maxOk ? "OK" : "NOT FOUND");

    Serial.print("# DS18B20 : ");
    Serial.println(tempOk ? "OK" : "NOT FOUND");

    Serial.print("# MPU     : ");
    Serial.println(mpuOk ? "OK" : "NOT FOUND");

    // Wi-Fi
    WiFi.mode(WIFI_STA);

    WiFi.setAutoReconnect(true);

    WiFi.begin(WIFI_SSID, WIFI_PASS);

    // Blynk
    Blynk.config(BLYNK_AUTH_TOKEN);

    Serial.println();
    Serial.println("====================================");
    Serial.println("     AIoT ML STRESS MONITORING");
    Serial.println("====================================");

    Serial.println("Waiting for ML prediction...");

    Serial.println(
        "ms,hr,temp,motion,rule_status,rule_buzzer,finger,ml_prediction"
    );
}

// =========================================================
// MAIN LOOP
// =========================================================

void loop()
{
    // Read heart rate
    readHeartRate();

    // Read motion every 100 ms
    if (millis() - lastMotionSample >= 100)
    {
        lastMotionSample = millis();

        sampleMotion();
    }

    // Process ML actuator control
    updateBuzzer();

    // Maintain Blynk connection
    blynkLoop();

    // One-second sensor task
    if (millis() - lastTick >= 1000)
    {
        lastTick = millis();

        oneSecondTick();
    }
}
