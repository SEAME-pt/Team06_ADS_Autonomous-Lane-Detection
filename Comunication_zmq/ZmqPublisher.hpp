#ifndef ZMQ_PUBLISHER_HPP
#define ZMQ_PUBLISHER_HPP

#include <string>
#include <zmq.hpp> // Inclui a biblioteca ZeroMQ C++
#include <iostream>
#include <json/json.h> // Para serialização JSON (você precisará de uma biblioteca JSON)

class ZmqPublisher {
public:
    // Construtor
    ZmqPublisher(const std::string& host, int port, const std::string& protocol = "tcp");

    // Destrutor
    ~ZmqPublisher();

    // Método para publicar uma mensagem
    // Aceita um objeto Json::Value para flexibilidade
    bool publishMessage(const Json::Value& data);

    // Verifica se o publicador está conectado/pronto
    bool isConnected() const;

private:
    zmq::context_t& _context; // Referência ao contexto ZMQ (geralmente um global ou de main)
    zmq::socket_t _socket;    // O socket do publicador
    std::string _bindAddress; // Endereço completo para o bind
    bool _isConnected;        // Status da conexão
};

#endif // ZMQ_PUBLISHER_HPP