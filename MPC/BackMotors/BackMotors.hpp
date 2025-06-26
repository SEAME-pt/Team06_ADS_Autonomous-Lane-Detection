#pragma once

#include <thread>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>    // Para close(), usleep()
#include <fcntl.h>     // Para open()
#include <sys/ioctl.h> 
#include <csignal> // Biblioteca para manipulação de sinais
#include <atomic>
#include <chrono> // Para cálculos de tempo
#include <linux/i2c-dev.h>// Interface padrão do Linux para I2C


class BackMotors{
private:
	std::string i2c_device;
	const int _motorAddr = 0x60;
	int _fdMotor;

public:
	BackMotors();
	~BackMotors();
	bool init_motors();
	bool setMotorPwm(const int channel, int value);
	void setSpeed(int speed);

	void writeByteData(int fd, uint8_t reg, uint8_t value);
	uint8_t readByteData(int fd, uint8_t reg);
};
