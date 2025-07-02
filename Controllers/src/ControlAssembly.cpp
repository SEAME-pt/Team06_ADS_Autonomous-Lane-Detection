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
    

    while (true){
        _onClick = false;
        if (!_controller.readEvent()) { break; }

        // Parar o carro ao pressionar X_BUTTON (botão 0)
        if (_controller.getButton(0)){
            _onClick = true;
            _backMotors.setSpeed(0);
            //_fServo.set_steering(0);
            _accelaration = 0;
            LOG_INFO("Botão X pressionado: Carro parado");
        }

        // Outros botões (mantidos)
        if (_controller.getButton(2)){ // SELECT_BUTTON
            std::cout << "SELECT_BUTTON" << std::endl;
        }
        if (_controller.getButton(3)){ // START_BUTTON
            std::cout << "START_BUTTON" << std::endl;
        }
        if (_controller.getButton(12)){ // HOME_BUTTON
            std::cout << "HOME_BUTTON" << std::endl;
        }

        // Controle de velocidade com R2 (botão 9) e L2 (botão 8)
        const float speed_increment = 50.0f; // Incremento/decremento por segundo
        const float max_speed = 100.0f;     // Limite máximo de velocidade
        const float dt = 0.016f;            // Intervalo de tempo (16 ms)

        if (_controller.getButton(9)) { // R2
            _accelaration += speed_increment * dt;
            if (_accelaration > max_speed) _accelaration = max_speed;
            LOG_INFO("R2 pressionado: Aceleração = %f", _accelaration);
        }
        if (_controller.getButton(8)) { // L2
            _accelaration -= speed_increment * dt;
            if (_accelaration < -max_speed) _accelaration = -max_speed;
            LOG_INFO("L2 pressionado: Aceleração = %f", _accelaration);
        }

        // Controle de velocidade com eixo 3 (mantido)
        float force = _controller.getAxis(3);
        if (force != 0){
            _accelaration -= (force * 0.55f);
            if (_accelaration > max_speed) _accelaration = max_speed;
            if (_accelaration < -max_speed) _accelaration = -max_speed;
            LOG_INFO("Eixo 3 ajustado: Aceleração = %f", _accelaration);
        }

        // Aplicar velocidade aos motores
        _backMotors.setSpeed(static_cast<int>(_accelaration));

        // Controle de direção (mantido)
        if (std::abs(_turn) < 0.1) {
            _turn = 0;
        } else {
            _turn -= _turn * 0.15; // Retorno proporcional
        }

        float gear = _controller.getAxis(0);
        if (std::abs(gear) > 0.1f) { // Zona morta
            _turn = (gear > 0 ? 1 : -1) * std::pow(std::abs(gear), 1.5f) * 5.0f;
            if (_turn < -4.5f) _turn = -4.5f;
            if (_turn > 4.5f) _turn = 4.5f;
        } else {
            _turn = 0;
        }
        //LOG_INFO("Gear %f", _turn);
        //_fServo.set_steering(static_cast<int>(_turn * 30));

        // Atraso para ~60 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    _backMotors.setSpeed(0);
    //_fServo.set_steering(0);
    return;
}

ControlAssembly::~ControlAssembly(){
    _backMotors.setSpeed(0);
    //_fServo.set_steering(0);
}