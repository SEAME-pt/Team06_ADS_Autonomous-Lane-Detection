#include "../inc/Controller.hpp"

Controller::Controller(){
	memset(_rawAxes, 0, sizeof(_rawAxes));
	memset(_normalizedAxes, 0, sizeof(_normalizedAxes));
	memset(_buttons, 0, sizeof(_buttons));

	_joyFd = open(_device.c_str(), O_RDONLY);
	if (_joyFd == -1){
		std::cerr << "Erro ao abrir o dispositivo: " << _device << std::endl;
		return;
	}

	// Pega o nome do joystick
	if (ioctl(_joyFd, JSIOCGNAME(128), _name) < 0){
		std::cerr << "Erro ao obter o nome do joystick" << std::endl;
		return;
	}

	std::cout << "Joystick: " << _name << " aberto com sucesso!" << std::endl;
}

Controller::~Controller()
{
	if (_joyFd != -1){
		close(_joyFd);
		std::cout << "Joystick fechado." << std::endl;
	}
}

bool	Controller::isQuit() const {return _quit;}
bool	Controller::isConnected() const {return _joyFd != -1;}
bool	Controller::isEvent() const {return _doEvent;}

bool	Controller::readEvent()
{
	struct js_event event;

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(_joyFd, &fds);
	_doEvent = false;

	struct timeval tv = {0, 0};

	int ret = select(_joyFd + 1, &fds, nullptr, nullptr, &tv);
	if (ret == -1){
		std::cerr << "Select Error!" << std::endl;
		return false;
	}
	if (ret > 0 && FD_ISSET(_joyFd, &fds)){
		ssize_t bytesRead = read(_joyFd, &event, sizeof(event));

		if (bytesRead == -1){
			std::cerr << "Erro ao ler evento do joystick!" << std::endl;
			return false;
		}
		if (bytesRead == sizeof(event)){
			switch (event.type){
				case JS_EVENT_AXIS:{
					if (event.number < MAX_AXES){
						_rawAxes[event.number] = event.value;
						_normalizedAxes[event.number] = _normalizeAxisValue(event.value);
					}
					_doEvent = true;
					break;
				}
				case JS_EVENT_BUTTON:{
					if (event.number < MAX_BUTTONS){
						_buttons[event.number] = event.value;
					}
					if (event.number == SELECT_BUTTON && event.value == 1){
						_quit = true;
					}
					_doEvent = true;
					break;
				}
				default:
					break;
			}

		}
	}
	return true;
}

float	Controller::getAxis(int axis) const{
    if (axis < 0 || axis >= MAX_AXES){
            return 0;
    }
	return _normalizedAxes[axis];
}

int		Controller::getRawAxis(int axis) const{
	if (axis < 0 || axis >= MAX_AXES){
		return 0;
	}
	return _rawAxes[axis];
}

bool	Controller::getButton(int button) const{
	if (button < 0 || button >= MAX_BUTTONS){
		return false;
	}
	return _buttons[button] == 1;
}

float	Controller::_normalizeAxisValue(int value){
	return (static_cast<float>(value) / MAX_AXIS_VALUE);
}

