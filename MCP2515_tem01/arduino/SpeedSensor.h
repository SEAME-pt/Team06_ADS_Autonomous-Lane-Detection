#ifndef SPEED_SENSOR_H
#define SPEED_SENSOR_H

#include <Arduino.h>
#include "mcp2515_can.h"

class SpeedSensor {
public:
    static void begin(int sensorPin, mcp2515_can* canBus);
    static void loop();

private:
    static void handlePulse();
    static void checkPulseAndUpdate();
    static void computeSpeedAndRpm(float timeDiff_sec);
    static void sendSpeedAndRpm();
    static void handleNoPulseTimeout();

    static mcp2515_can* can;
    static int sensorPin;

    static const int SLOTS_IN_DISK = 20;
    static const int AVERAGE_POINTS = 20;
    static const float WHEEL_DIAMETER;
    static const float WHEEL_CIRCUMFERENCE;
    static const float CONVERT_TO_KMH;
    static const float CONVERT_TO_MPH;
    static const bool METRIC;
    static const unsigned long DEBOUNCE_TIME;
    static const unsigned long NO_PULSE_TIMEOUT;

    static volatile unsigned long lastPulseTime;
    static volatile unsigned long lastValidPulseTime;
    static volatile unsigned long currentPulseTime;
    static volatile bool newPulse;

    static int rpm;
    static float speed;

    static float speedReadings[AVERAGE_POINTS];
    static int rpmReadings[AVERAGE_POINTS];
    static int readIndex;

    static unsigned long lastPrintTime;
};

#endif
