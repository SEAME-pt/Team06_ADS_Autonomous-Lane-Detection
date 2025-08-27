#ifndef ZMQ_SUBSCRIBER_HPP
#define ZMQ_SUBSCRIBER_HPP

#include <zmq.hpp>
#include <string>
#include <atomic>
#include <thread>
#include <memory>

class ZmqSubscriber {
public:
    // Construtor: Recebe contexto ZMQ, host, porta e referência à variável atómica para atualização
    ZmqSubscriber(zmq::context_t& context, const std::string& host, int port, std::atomic<double>& speed_var, const std::string& protocol = "tcp");

    // Destrutor: Para o thread e fecha o socket
    ~ZmqSubscriber();

    // Inicia o thread de receção
    void start();

    // Para o thread de receção
    void stop();

    // Verifica se está conectado
    bool isConnected() const;

private:
    // Função do thread: Loop de receção e parsing
    void receiveLoop();

    zmq::context_t& _context;
    zmq::socket_t _socket;
    std::string _connectAddress;
    std::atomic<double>& _speed_var;  // Referência à variável atómica para atualização
    std::unique_ptr<std::thread> _receiveThread;
    std::atomic<bool> _running{false};
    bool _isConnected{false};
};

#endif // ZMQ_SUBSCRIBER_HPP
