#include "ZmqSubscriber.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

ZmqSubscriber::ZmqSubscriber(zmq::context_t& context, const std::string& host, int port, std::atomic<double>& speed_var, const std::string& protocol)
    : _context(context),
      _socket(_context, zmq::socket_type::sub),
      _speed_var(speed_var) {
    _connectAddress = protocol + "://" + host + ":" + std::to_string(port);
    std::cout << "ZMQ Subscriber: Conectando a " << _connectAddress << "..." << std::endl;

    try {
        _socket.set(zmq::sockopt::subscribe, "");  // Subscreve a todos os tópicos
        int rcvhwm = 1000;
        _socket.set(zmq::sockopt::rcvhwm, rcvhwm);
        int linger = 0;
        _socket.set(zmq::sockopt::linger, linger);
        _socket.connect(_connectAddress);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Tempo para conexão estabilizar
        _isConnected = true;
        std::cout << "ZMQ Subscriber: Conectado com sucesso." << std::endl;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Subscriber ERROR: Não foi possível conectar: " << e.what() << std::endl;
        _isConnected = false;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Subscriber ERROR: Erro inesperado no construtor: " << e.what() << std::endl;
        _isConnected = false;
    }
}

ZmqSubscriber::~ZmqSubscriber() {
    stop();
    try {
        _socket.close();
        std::cout << "ZMQ Subscriber: Socket fechado." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Subscriber ERROR: Erro no destrutor: " << e.what() << std::endl;
    }
}

void ZmqSubscriber::start() {
    if (!_isConnected || _running.load()) return;
    _running.store(true);
    _receiveThread = std::make_unique<std::thread>(&ZmqSubscriber::receiveLoop, this);
}

void ZmqSubscriber::stop() {
    _running.store(false);
    if (_receiveThread && _receiveThread->joinable()) {
        _receiveThread->join();
    }
}

bool ZmqSubscriber::isConnected() const {
    return _isConnected;
}

void ZmqSubscriber::receiveLoop() {
    while (_running.load()) {
        try {
            zmq::message_t message;
            zmq::recv_result_t recv_result = _socket.recv(message, zmq::recv_flags::dontwait);
            std::cout << "Velocidade recebida! " << std::endl;
            
            if (recv_result) {
                std::string data(static_cast<char*>(message.data()), message.size());
                // Parsing simples e rápida: "speed:0.69;"
                if (data.find("speed:") == 0) {
                    std::string value_str = data.substr(6, data.size() - 7);  // Extrai "0.69"
                    double speed_kmh = std::stod(value_str);
                    double speed_ms = speed_kmh * (5.0 / 18.0);  // Conversão km/h -> m/s
                    _speed_var.store(speed_ms);  // Atualiza atómica
                    std::cout << "Velocidade recebida: " << speed_kmh << " km/h" << std::endl;
                    std::cout << "Velocidade convertida: " << speed_ms << " m/s" << std::endl;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Evita CPU alta em idle
            }
        } catch (const zmq::error_t& e) {
            if (e.num() != EAGAIN) {  // Ignora EAGAIN (sem dados)
                std::cerr << "ZMQ Subscriber ERROR: Erro na receção: " << e.what() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "ZMQ Subscriber ERROR: Erro no loop de receção: " << e.what() << std::endl;
        }
    }
}
