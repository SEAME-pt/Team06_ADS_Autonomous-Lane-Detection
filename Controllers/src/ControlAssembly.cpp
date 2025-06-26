#include "../inc/ControlAssembly.hpp"


ControlAssembly::ControlAssembly()
{
	if (!_backMotors.init_motors()){
		LOG_ERROR("Failed to initialize BackMotors");
		return;
	}

	/* if (!_fServo.init_servo()){
		LOG_ERROR("Failed to initialize FServo");
		return;
	} */

	if (!_controller.isConnected()){
		LOG_ERROR("Failed to initialize Controller");
		return;
	}

	//auto last_update = std::chrono::steady_clock::now();
	//const auto update_interval = std::chrono::milliseconds(16); // ~60fps

	while (true){
		_onClick = false;
		if (!_controller.readEvent()) { break; }
		if (_controller.getButton(X_BUTTON)){
			_onClick = true;
			_backMotors.setSpeed(0);
			//_fServo.set_steering(0);
		}
		if (_controller.getButton(SELECT_BUTTON)){
			std::cout << "START_BUTTON" << std::endl;
		}
		if (_controller.getButton(START_BUTTON)){
			std::cout << "START_BUTTON" << std::endl;
		}
		if (_controller.getButton(HOME_BUTTON)){
			std::cout << "HOME_BUTTON" << std::endl;
		}
		_accelaration *= 0.99;
		//_turn *= 0.99;
		if (std::abs(_turn) < 0.1) {
    		_turn = 0;
		} else {
		    _turn -= _turn * 0.15; // Retorno proporcional
		}

		float force = _controller.getAxis(3);
		if (force != 0){
			_accelaration -= (force * 0.55f);
		}
		_backMotors.setSpeed(static_cast<int>(_accelaration * 1));
		float gear = _controller.getAxis(0);
		if (std::abs(gear) > 0.1f) { // Zona morta
		    // Maior sensibilidade com ajuste exponencial
		    _turn = (gear > 0 ? 1 : -1) * std::pow(std::abs(gear), 1.5f) * 5.0f;
		    if (_turn < -4.5f) _turn = -4.5f;
		    if (_turn > 4.5f) _turn = 4.5f;
		} else {
		    _turn = 0;
		}
		//LOG_INFO("Gear %f", _turn);
		//_fServo.set_steering(static_cast<int>(_turn * 30)); // Multiplicador ajustado

		
		//float gear = _controller.getAxis(0);
		//if (gear != 0){
		//	_turn = (gear * 0.15f) / 0.02f;
		//	if (_turn <= -4.5){
		//		_turn = -4.5;
		//	}
		//	else if (_turn >= 4.5){
		//		_turn = 4.5;
		//	}
		//	LOG_INFO("Gear %f", _turn);
		//}
		//_fServo.set_steering(static_cast<int>(_turn * 25));
	}
	_backMotors.setSpeed(0);
	//_fServo.set_steering(0);
	return;
}

ControlAssembly::~ControlAssembly(){
	_backMotors.setSpeed(0);
	//_fServo.set_steering(0);
}