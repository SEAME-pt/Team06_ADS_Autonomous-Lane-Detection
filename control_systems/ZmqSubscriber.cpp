#include "ZmqSubscriber.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <errno.h>  // For EAGAIN, ETERM
#include <algorithm>  // For std::min

ZmqSubscriber::ZmqSubscriber(zmq::context_t& context, const std::string& host, int port, std::atomic<double>& speed_var, const std::string& protocol)
    : _context(context),
      _socket(_context, zmq::socket_type::sub),
      _speed_var(speed_var) {
    _connectAddress = protocol + "://" + host + ":" + std::to_string(port);
    std::cout << "ZMQ Subscriber: Conectando a " << _connectAddress << "..." << std::endl;

    try {
        // Connect first (like the working example)
        _socket.connect(_connectAddress);
        std::cout << "ZMQ Subscriber: Connected to " << _connectAddress << std::endl;

        // Subscribe to all messages (empty filter) - after connecting
        _socket.set(zmq::sockopt::subscribe, "");
        std::cout << "ZMQ Subscriber: Subscription pattern set (all messages)" << std::endl;

        // Set receive timeout (like the working example)cd
        int timeout = 1000; // 1 second timeout
        _socket.set(zmq::sockopt::rcvtimeo, timeout);
        std::cout << "ZMQ Subscriber: Receive timeout set to " << timeout << "ms" << std::endl;

        // Optional optimizations (but keep them minimal)
        int rcvhwm = 1000;  // Use higher value like working example
        _socket.set(zmq::sockopt::rcvhwm, rcvhwm);

        int linger = 0;
        _socket.set(zmq::sockopt::linger, linger);

        _socket.set(zmq::sockopt::ipv6, 0);
        // Give time for connection to stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
    std::cout << "ZMQ Subscriber: Starting receive thread..." << std::endl;
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
    std::cout << "ZMQ Subscriber: Starting receive loop..." << std::endl;

    int heartbeat_counter = 0;
    auto last_heartbeat = std::chrono::steady_clock::now();
    int messageCount = 0;

    while (_running.load()) {
        try {
            zmq::message_t message;
            auto result = _socket.recv(message, zmq::recv_flags::none);

            if (result) {
                messageCount++;
                // Debug: Show message size and raw content
                size_t msg_size = message.size();
                //std::cout << "ZMQ: Received message #" << messageCount << " of " << msg_size << " bytes" << std::endl;

                std::string data;  // Declare data variable here

                if (msg_size == 0) {
                    std::cout << "ZMQ: Empty message received! Checking for multi-part..." << std::endl;

                    // Check if this is a multi-part message
                    int more = 0;
                    size_t more_size = sizeof(more);
                    _socket.getsockopt(ZMQ_RCVMORE, &more, &more_size);

                    if (more) {
                        std::cout << "ZMQ: Multi-part message detected, receiving next part..." << std::endl;
                        zmq::message_t next_message;
                        auto next_result = _socket.recv(next_message, zmq::recv_flags::none);
                        if (next_result) {
                            data = std::string(static_cast<char*>(next_message.data()), next_message.size());
                            std::cout << "ZMQ: Next part received: '" << data << "' (" << next_message.size() << " bytes)" << std::endl;
                            msg_size = next_message.size();
                        }
                    } else {
                        std::cout << "ZMQ: Single-part empty message - publisher may be sending empty data" << std::endl;
                        continue;
                    }
                } else {
                    // Extract message content (like working example)
                    data = std::string(static_cast<char*>(message.data()), message.size());
                    //std::cout << "ZMQ received: '" << data << "'" << std::endl;
                }

                // Only process if we have actual data
                if (msg_size > 0) {
                    // Parsing simples e rápida: "speed:0.69;"
                    if (data.find("speed:") == 0) {
                        size_t colon_pos = data.find(':');
                        size_t semicolon_pos = data.find(';');

                        if (colon_pos != std::string::npos && semicolon_pos != std::string::npos && semicolon_pos > colon_pos) {
                            std::string value_str = data.substr(colon_pos + 1, semicolon_pos - colon_pos - 1);
                            //std::cout << "ZMQ: Extracted speed value: '" << value_str << "'" << std::endl;
                            try {
                                double speed_mms = std::stod(value_str);
                                double speed_ms = speed_mms / 1000;  // Conversão mm/s -> m/s
                                _speed_var.store(speed_ms);  // Atualiza atómica
/*                                 std::cout << "Velocidade recebida: " << speed_mms << " km/h" << std::endl;
                                std::cout << "Velocidade convertida: " << speed_ms << " m/s" << std::endl; */
                            } catch (const std::exception& e) {
                                std::cerr << "ZMQ Subscriber ERROR: Erro ao converter velocidade '" << value_str << "': " << e.what() << std::endl;
                            }
                        } else {
                            std::cerr << "ZMQ Subscriber ERROR: Formato de mensagem inválido: '" << data << "'" << std::endl;
                            std::cerr << "  colon_pos: " << colon_pos << ", semicolon_pos: " << semicolon_pos << std::endl;
                        }
                    } else {
                        std::cout << "ZMQ Subscriber: Mensagem ignorada (não é velocidade): '" << data << "'" << std::endl;
                    }
                }

            } else {
                // Timeout occurred (like working example)
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 5) {
                    heartbeat_counter++;
                    std::cout << "ZMQ Subscriber: Heartbeat #" << heartbeat_counter << " - no messages received (timeout)" << std::endl;
                    last_heartbeat = now;
                }
            }

        } catch (const zmq::error_t& e) {
            if (e.num() == EAGAIN) {
                std::cout << "ZMQ Subscriber: Receive timeout - waiting for messages..." << std::endl;
            } else if (e.num() == ETERM) {
                std::cout << "ZMQ Subscriber: Context terminated, stopping receive loop" << std::endl;
                break;
            } else {
                std::cerr << "ZMQ Subscriber ERROR: " << e.what() << " (code: " << e.num() << ")" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (const std::exception& e) {
            std::cerr << "ZMQ Subscriber ERROR: Erro no loop de receção: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cout << "ZMQ Subscriber: Receive loop stopped." << std::endl;
}
