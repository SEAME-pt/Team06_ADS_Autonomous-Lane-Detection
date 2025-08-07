#include "SpeedSensor.h"

mcp2515_can* SpeedSensor::can = nullptr;
int SpeedSensor::sensorPin = 3;

const float SpeedSensor::WHEEL_DIAMETER = 0.067;
const float SpeedSensor::WHEEL_CIRCUMFERENCE = PI * SpeedSensor::WHEEL_DIAMETER;
const float SpeedSensor::CONVERT_TO_KMH = 3.6;
const float SpeedSensor::CONVERT_TO_MPH = 2.237;
const bool SpeedSensor::METRIC = true;
const unsigned long SpeedSensor::DEBOUNCE_TIME = 1;
const unsigned long SpeedSensor::NO_PULSE_TIMEOUT = 50;

volatile unsigned long SpeedSensor::lastPulseTime = 0;
volatile unsigned long SpeedSensor::lastValidPulseTime = 0;
volatile unsigned long SpeedSensor::currentPulseTime = 0;
volatile bool SpeedSensor::newPulse = false;

int SpeedSensor::rpm = 0;
float SpeedSensor::speed = 0.0;

float SpeedSensor::speedReadings[AVERAGE_POINTS] = {0.0};
int SpeedSensor::rpmReadings[AVERAGE_POINTS] = {0};
int SpeedSensor::readIndex = 0;

unsigned long SpeedSensor::lastPrintTime = 0;

void SpeedSensor::handlePulse() {
    lastPulseTime = currentPulseTime;
    currentPulseTime = micros();
    newPulse = true;
}

void SpeedSensor::begin(int pin, mcp2515_can* canBus) {
    sensorPin = pin;
    can = canBus;

    pinMode(sensorPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(sensorPin), handlePulse, RISING);

    for (int i = 0; i < AVERAGE_POINTS; i++) {
        speedReadings[i] = 0.0;
        rpmReadings[i] = 0;
    }

    lastPrintTime = millis();
}

void SpeedSensor::loop() {
    checkPulseAndUpdate();
    handleNoPulseTimeout();
}

void SpeedSensor::checkPulseAndUpdate() {
    if (!newPulse) return;

    float timeDiff_ms = (float)(currentPulseTime - lastValidPulseTime) / 1000.0;
    if (timeDiff_ms < DEBOUNCE_TIME) {
        newPulse = false;
        return;
    }

    float timeDiff_sec = timeDiff_ms / 1000.0;
    if (timeDiff_sec <= 0) {
        newPulse = false;
        return;
    }

    computeSpeedAndRpm(timeDiff_sec);
    sendSpeedAndRpm();

    lastValidPulseTime = currentPulseTime;
    lastPrintTime = millis();
    newPulse = false;
}

void SpeedSensor::computeSpeedAndRpm(float timeDiff_sec) {
    float revTime = timeDiff_sec * SLOTS_IN_DISK;
    int rawRpm = round(60.0 / revTime);

    rpmReadings[readIndex] = rawRpm;
    int avgRpm = 0;
    for (int i = 0; i < AVERAGE_POINTS; i++) {
        avgRpm += rpmReadings[i];
    }
    avgRpm /= AVERAGE_POINTS;
    rpm = avgRpm;

    float units = METRIC ? CONVERT_TO_KMH : CONVERT_TO_MPH;
    float rawSpeed = (WHEEL_CIRCUMFERENCE * (rpm / 60.0)) * units;

    speedReadings[readIndex] = rawSpeed;
    float avgSpeed = 0.0;
    for (int i = 0; i < AVERAGE_POINTS; i++) {
        avgSpeed += speedReadings[i];
    }
    avgSpeed /= AVERAGE_POINTS;
    avgSpeed /= 2.0;
    speed = avgSpeed;

    readIndex = (readIndex + 1) % AVERAGE_POINTS;
}

void SpeedSensor::sendSpeedAndRpm() {
    // Send scaled speed as 2-byte uint16
    uint16_t scaledSpeed = (uint16_t)(speed * 10.0);
    byte speedMsg[2] = {
        (byte)((scaledSpeed >> 8) & 0xFF),
        (byte)(scaledSpeed & 0xFF)
    };

    if (can->sendMsgBuf(0x100, 0, 2, speedMsg) == CAN_OK) {
        Serial.println("SENT speed: " + String(speed, 1));
    } else {
        Serial.println("SPEED CAN MESSAGE ERROR!");
    }

    // Send RPM as 2-byte integer
    // byte rpmMsg[2] = {
    //     (byte)((rpm >> 8) & 0xFF),
    //     (byte)(rpm & 0xFF)
    // };

    // if (can->sendMsgBuf(0x200, 0, 2, rpmMsg) == CAN_OK) {
    //     Serial.println("SENT rpm: " + String(rpm));
    // } else {
    //     Serial.println("RPM CAN MESSAGE ERROR!");
    // }
}

void SpeedSensor::handleNoPulseTimeout() {
    if ((millis() - lastValidPulseTime > NO_PULSE_TIMEOUT) &&
        (millis() - lastPrintTime > NO_PULSE_TIMEOUT)) {

        byte speedMsg[2] = { 0, 0 };

        if (can->sendMsgBuf(0x100, 0, 2, speedMsg) == CAN_OK) {
            Serial.println("SENT speed: 0.0");
        } else {
            Serial.println("SPEED CAN MESSAGE ERROR!");
        }

        speedReadings[readIndex] = 0.0;
        readIndex = (readIndex + 1) % AVERAGE_POINTS;
        speed = 0.0;

        // byte rpmMsg[2] = { 0, 0 };

        // if (can->sendMsgBuf(0x200, 0, 2, rpmMsg) == CAN_OK) {
        //     Serial.println("SENT rpm: 0");
        // } else {
        //     Serial.println("RPM CAN MESSAGE ERROR!");
        // }

        lastPrintTime = millis();
    }
}
