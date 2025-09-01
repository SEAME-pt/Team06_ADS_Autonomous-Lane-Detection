#ifndef ZMQ_PUBLISHER_HPP
#define ZMQ_PUBLISHER_HPP

#include <string>
#include <zmq.hpp> // Inclui a biblioteca ZeroMQ C++
#include <iostream>
// #include <json/json.h> // REMOVIDO: Não precisamos mais de JsonCpp

class ZmqPublisher {
public:
    // Construtor
    ZmqPublisher(zmq::context_t& context, const std::string& host, int port, const std::string& protocol = "tcp");

    // Destrutor
    ~ZmqPublisher();

    // Método para publicar uma mensagem: AGORA ACEITA APENAS UMA STRING
    bool publishMessage(const std::string& data);

    // Verifica se o publicador está conectado/pronto
    bool isConnected() const;

private:
    zmq::context_t& _context; // Referência ao contexto ZMQ
    zmq::socket_t _socket;    // O socket do publicador
    std::string _bindAddress; // Endereço completo para o bind
    bool _isConnected;        // Status da conexão
};

#endif // ZMQ_PUBLISHER_HPP