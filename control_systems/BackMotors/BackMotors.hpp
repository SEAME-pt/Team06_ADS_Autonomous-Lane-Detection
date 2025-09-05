#pragma once

#include <thread>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <csignal>
#include <atomic>
#include <chrono>
#include <linux/i2c-dev.h>

class BackMotors {
private:
    std::string i2c_device;
    const int _motorAddr = 0x60;
    int _fdMotor;

    // Compensações internas (alterar aqui se necessário)
    double _compLeft = 0.875;  // 5% mais força no motor esquerdo
    double _compRight = 1.00; // motor direito normal

public:
    BackMotors();
    ~BackMotors();

    bool init_motors();
    bool setMotorPwm(const int channel, int value);

    // Define a mesma velocidade para ambos (usa compensação interna)
    void setSpeed(int speed);

    // Define velocidades independentes (também usa compensação)
    //void setSpeeds(int leftSpeed, int rightSpeed);

    void writeByteData(int fd, uint8_t reg, uint8_t value);
    uint8_t readByteData(int fd, uint8_t reg);
};
