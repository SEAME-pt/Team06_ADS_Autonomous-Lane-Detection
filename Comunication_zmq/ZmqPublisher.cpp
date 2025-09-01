#include "ZmqPublisher.hpp"
#include <thread> // Para std::this_thread::sleep_for
#include <chrono> // Para std::chrono::milliseconds
#include <sstream> // Para Json::StreamWriterBuilder

// Construtor
ZmqPublisher::ZmqPublisher(const std::string& host, int port, const std::string& protocol)
    : _context(zmq::context_t::instance()), // Usa a instância global do contexto ZMQ
      _socket(_context, zmq::socket_type::pub), // Define o tipo de socket como PUB
      _isConnected(false) {

    _bindAddress = protocol + "://" + host + ":" + std::to_string(port);

    std::cout << "ZMQ Publisher: Ligando (binding) a " << _bindAddress << "..." << std::endl;

    try {
        // Configurações opcionais (mas boas práticas para PUB-SUB):
        // Define um alto watermark de envio para evitar bloqueios em caso de subscritores lentos.
        // O subscriber do seu colega tem conflate, então este lado não precisa de um HWM baixo.
        int sndhwm = 1000; // Exemplo: permite até 1000 mensagens na fila de envio
        _socket.set(zmq::sockopt::sndhwm, sndhwm);

        // Define um período de linger de 0 para saída limpa
        int linger = 0;
        _socket.set(zmq::sockopt::linger, linger);

        _socket.bind(_bindAddress); // O publicador faz BIND

        // Pequena pausa para permitir que os subscritores se conectem
        // Em um sistema real, um mecanismo de prontidão mais robusto seria ideal.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        _isConnected = true;
        std::cout << "ZMQ Publisher: Ligado (bound) com sucesso." << std::endl;

    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Publisher ERROR: Nao foi possivel ligar ao endereco: " << e.what() << std::endl;
        _isConnected = false;
        // Não relança a exceção aqui, mas o isConnected() será false.
        // O chamador deve verificar isConnected().
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Publisher ERROR: Erro inesperado no construtor: " << e.what() << std::endl;
        _isConnected = false;
    }
}

// Destrutor
ZmqPublisher::~ZmqPublisher() {
    if (_isConnected) {
        try {
            // Não há disconnect() direto para sockets PUB
            // O fechamento do socket e término do contexto lidam com isso.
            _socket.close(); // Fecha o socket explicitamente
            std::cout << "ZMQ Publisher: Socket fechado." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ZMQ Publisher ERROR: Erro no destrutor: " << e.what() << std::endl;
        }
    }
    // O contexto é uma instância estática e será terminado automaticamente ao final do programa.
}

// Método para publicar uma mensagem
bool ZmqPublisher::publishMessage(const Json::Value& data) {
    if (!_isConnected) {
        std::cerr << "ZMQ Publisher ERROR: Publicador nao esta conectado. Nao pode enviar mensagem." << std::endl;
        return false;
    }

    try {
        // Serializa o objeto JSON para uma string
        Json::StreamWriterBuilder writer;
        std::string jsonString = Json::writeString(writer, data);

        // Cria uma mensagem ZMQ a partir da string JSON
        zmq::message_t message(jsonString.begin(), jsonString.end());

        std::cout << "ZMQ Publisher: Publicando mensagem: " << jsonString << std::endl;

        // Envia a mensagem (não espera resposta no PUB-SUB)
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

// Verifica se o publicador está conectado/pronto
bool ZmqPublisher::isConnected() const {
    return _isConnected;
}