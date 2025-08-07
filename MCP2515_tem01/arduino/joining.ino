#include <SPI.h>
#include "mcp2515_can.h"
#include <Wire.h>
#include "SpeedSensor.h"

// CAN setup
const int SPI_CS_PIN = 9;
mcp2515_can CAN(SPI_CS_PIN);

// ---- SPEED SENSOR SETUP ----

void setupSpeedSensor() {
  Serial.begin(9600);

  if (CAN.begin(CAN_500KBPS) != CAN_OK) {
    while (1);  // CAN init failed
  }

  CAN.setMode(MODE_NORMAL);

  SpeedSensor::begin(3, &CAN);  // Pin 3 as pulse input
}

#define SRF08_ADDR 0x70

// ---- SRF08 SETUP ----
void setupSRF08() {
  Wire.begin();
  delay(100);  // Let the sensor stabilize
}

// ---- SRF08 LOOP ----
void loopSRF08() {
  static unsigned long lastRead = 0;
  if (millis() - lastRead >= 300) {
    lastRead = millis();
    readSRF08();
  }
}

// ---- READ DISTANCE ----
void readSRF08() {
  // Send ranging command (0x51 = range in cm)
  Wire.beginTransmission(SRF08_ADDR);
  Wire.write(0x00);
  Wire.write(0x51);
  Wire.endTransmission();

  delay(70);  // Wait for measurement (~65 ms)

  // Request distance result (high and low byte)
  Wire.beginTransmission(SRF08_ADDR);
  Wire.write(0x02);
  Wire.endTransmission();

  Wire.requestFrom(SRF08_ADDR, 2);
  if (Wire.available() == 2) {
    byte high = Wire.read();
    byte low = Wire.read();
    int distance = (high << 8) | low;

    byte distanceMessage[2];
    distanceMessage[0] = (distance >> 8) & 0xFF;  // High byte
    distanceMessage[1] = distance & 0xFF;         // Low byte

    if (CAN.sendMsgBuf(0x300, 0, 2, distanceMessage) == CAN_OK) {
      Serial.println("SENT distance: " + String(distance));
    } else {
      Serial.println("DISTANCE CAN MESSAGE ERROR!");
    }
  }
}

void setup() {
  setupSpeedSensor();
  setupSRF08();
}

void loop() {
  SpeedSensor::loop();
  loopSRF08();
}
