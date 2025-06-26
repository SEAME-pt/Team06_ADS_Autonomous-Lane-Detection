#pragma once

#include <chrono>
#include <cstdio>
#include <cstdarg>


#include "BackMotors.hpp"
#include "FServo.hpp"
#include "Controller.hpp"
#include "Logger.hpp"

class ControlAssembly{
private:
	BackMotors	_backMotors;
	FServo		_fServo;
	Controller	_controller;

    double	_accelaration = 0;
    double	_turn = 0;
	bool	_onClick = false;

public:
	ControlAssembly();
	~ControlAssembly();
};