#pragma once

#include "../BackMotors/BackMotors.hpp"
#include "../FServo/FServo.hpp"

#include <linux/joystick.h>
#include <sys/select.h>
#include <cstring>

const unsigned char MAX_AXES = 8;
const unsigned char MAX_BUTTONS = 14;
const int16_t MAX_AXIS_VALUE = 32767; // Valor máximo do eixo no Linux

const unsigned char LEFT_BUTTON_ONE = 6;
const unsigned char LEFT_BUTTON_TWO = 8;

const unsigned char RIGHT_BUTTON_ONE = 7;
const unsigned char RIGHT_BUTTON_TWO = 9;

const unsigned char X_BUTTON = 3;
const unsigned char Y_BUTTON = 4;

const unsigned char A_BUTTON = 0;
const unsigned char B_BUTTON = 1;

const unsigned char SELECT_BUTTON = 10;
const unsigned char START_BUTTON = 11;
const unsigned char HOME_BUTTON = 12;


class BackMotors;
class FServo;

class Controller {
private:
	const std::string	_device = "/dev/input/js0";
	int 				_joyFd = -1;
	char 				_name[128] = "Team06";
	bool				_quit = false;
	bool				_doEvent = false;

	int 				_rawAxes[MAX_AXES];
	float 				_normalizedAxes[MAX_AXES];
	char 				_buttons[MAX_BUTTONS];
	float				_normalizeAxisValue(int value);

public:
	Controller();
	~Controller();

	bool	isQuit() const;
	bool	isConnected() const;
	bool	isEvent() const;

	bool	readEvent();
	float	getAxis(int axis) const;
	int		getRawAxis(int axis) const;
	bool	getButton(int button) const;
};
