#include "ZmqPublisher.hpp"
#include <thread> // Para std::this_thread::sleep_for
#include <chrono> // Para std::chrono::milliseconds
// #include <sstream> // REMOVIDO: Não precisamos mais para Json::StreamWriterBuilder

// Construtor (Permanece o mesmo na lógica)
ZmqPublisher::ZmqPublisher(zmq::context_t& context, const std::string& host, int port, const std::string& protocol)
    : _context(context),
      _socket(_context, zmq::socket_type::pub),
      _isConnected(false) {

    _bindAddress = protocol + "://" + host + ":" + std::to_string(port);

    std::cout << "ZMQ Publisher: Ligando (binding) a " << _bindAddress << "..." << std::endl;

    try {
        int sndhwm = 1000;
        _socket.set(zmq::sockopt::sndhwm, sndhwm);

        int linger = 0;
        _socket.set(zmq::sockopt::linger, linger);

        _socket.bind(_bindAddress);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        _isConnected = true;
        std::cout << "ZMQ Publisher: Ligado (bound) com sucesso." << std::endl;

    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Publisher ERROR: Nao foi possivel ligar ao endereco: " << e.what() << std::endl;
        _isConnected = false;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Publisher ERROR: Erro inesperado no construtor: " << e.what() << std::endl;
        _isConnected = false;
    }
}

// Destrutor (Permanece o mesmo)
ZmqPublisher::~ZmqPublisher() {
    if (_isConnected) {
        try {
            _socket.close();
            std::cout << "ZMQ Publisher: Socket fechado." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ZMQ Publisher ERROR: Erro no destrutor: " << e.what() << std::endl;
        }
    }
}

// Método para publicar uma mensagem: AGORA ACEITA UMA STRING DIRETAMENTE
bool ZmqPublisher::publishMessage(const std::string& data) {
    if (!_isConnected) {
        std::cerr << "ZMQ Publisher ERROR: Publicador nao esta conectado. Nao pode enviar mensagem." << std::endl;
        return false;
    }

    try {
        // Cria uma mensagem ZMQ diretamente da string
        zmq::message_t message(data.begin(), data.end());

        //std::cout << "ZMQ Publisher: Publicando mensagem: " << data << std::endl;

        zmq::send_result_t send_result = _socket.send(message, zmq::send_flags::none);

        if (!send_result) {
            std::cerr << "ZMQ Publisher ERROR: Falha ao enviar a mensagem." << std::endl;
            return false;
        }

        return true;

    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Publisher ERROR: Erro ao publicar mensagem: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Publisher ERROR: Ocorreu um erro inesperado ao publicar: " << e.what() << std::endl;
        return false;
    }
}

// isConnected() (Permanece o mesmo)
bool ZmqPublisher::isConnected() const {
    return _isConnected;
}