#ifndef FSERVO_HPP
#define FSERVO_HPP

#include <cmath>
#include <fcntl.h> // Para open()
#include <fstream>
#include <iostream>
#include <linux/i2c-dev.h> // Interface padrão do Linux para I2C
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h> // Para close(), usleep()

#include <csignal> // Biblioteca para manipulação de sinais

#include <atomic>
#include <chrono> // Para cálculos de tempo

// Forward declaration
class FServo;

// Interface for FServo
class IFServo {
public:
  virtual ~IFServo() = default;
  virtual void open_i2c_bus() = 0;
  virtual bool init_servo() = 0;
  virtual bool setServoPwm(const int channel, int on_value, int off_value) = 0;
  virtual void set_steering(int angle) = 0;
  virtual void writeByteData(int fd, uint8_t reg, uint8_t value) = 0;
  virtual uint8_t readByteData(int fd, uint8_t reg) = 0;
};

class FServo : public IFServo {
private:
  std::string i2c_device;
  const int _servoAddr = 0x40;
  const int _maxAngle = 90;
  const int _servoCenterPwm = 320;
  const int _servoLeftPwm = 320 - 150;
  const int _servoRightPwm = 320 + 150;
  const int _sterringChannel = 0;

  int _currentAngle;

public:
  int _fdServo;
  FServo();

  void open_i2c_bus() override;
  ~FServo() override;
  bool init_servo() override;
  bool setServoPwm(const int channel, int on_value, int off_value) override;
  void set_steering(int angle) override;

  void writeByteData(int fd, uint8_t reg, uint8_t value) override;
  uint8_t readByteData(int fd, uint8_t reg) override;
};

#endif