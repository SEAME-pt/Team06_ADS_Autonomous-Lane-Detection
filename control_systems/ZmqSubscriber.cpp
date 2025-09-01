#include "ZmqSubscriber.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <errno.h>
#include <algorithm>

ZmqSubscriber::ZmqSubscriber(zmq::context_t& context, const std::string& host, int port, std::atomic<double>& speed_var, const std::string& protocol)
    : _context(context),
      _socket(_context, zmq::socket_type::sub),
      _speed_var(speed_var) {
    _connectAddress = protocol + "://" + host + ":" + std::to_string(port);
    std::cout << "ZMQ Subscriber: Connecting to " << _connectAddress << "..." << std::endl;

    try {
        _socket.connect(_connectAddress);
        std::cout << "ZMQ Subscriber: Connected to " << _connectAddress << std::endl;

        _socket.set(zmq::sockopt::subscribe, "");
        std::cout << "ZMQ Subscriber: Subscription pattern set (all messages)" << std::endl;

        int timeout = 1000;
        _socket.set(zmq::sockopt::rcvtimeo, timeout);
        std::cout << "ZMQ Subscriber: Receive timeout set to " << timeout << "ms" << std::endl;

        int rcvhwm = 1000;
        _socket.set(zmq::sockopt::rcvhwm, rcvhwm);

        int linger = 0;
        _socket.set(zmq::sockopt::linger, linger);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        _isConnected = true;
        std::cout << "ZMQ Subscriber: Successfully connected." << std::endl;

    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Subscriber ERROR: Failed to connect: " << e.what() << std::endl;
        _isConnected = false;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Subscriber ERROR: Unexpected error in constructor: " << e.what() << std::endl;
        _isConnected = false;
    }
}

ZmqSubscriber::~ZmqSubscriber() {
    stop();
    try {
        _socket.close();
        std::cout << "ZMQ Subscriber: Socket closed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ Subscriber ERROR: Error in destructor: " << e.what() << std::endl;
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
                size_t msg_size = message.size();
                std::string data;

                if (msg_size == 0) {
                    std::cout << "ZMQ: Empty message received; checking for multi-part..." << std::endl;

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
                        std::cout << "ZMQ: Single-part empty message — publisher may be sending empty payloads" << std::endl;
                        continue;
                    }
                } else {
                    data = std::string(static_cast<char*>(message.data()), message.size());
                }

                if (msg_size > 0) {
                    if (data.find("speed:") == 0) {
                        size_t colon_pos = data.find(':');
                        size_t semicolon_pos = data.find(';');

                        if (colon_pos != std::string::npos && semicolon_pos != std::string::npos && semicolon_pos > colon_pos) {
                            std::string value_str = data.substr(colon_pos + 1, semicolon_pos - colon_pos - 1);
                            try {
                                double speed_mms = std::stod(value_str);
                                double speed_ms = speed_mms / 1000;  // mm/s -> m/s
                                _speed_var.store(speed_ms);
                            } catch (const std::exception& e) {
                                std::cerr << "ZMQ Subscriber ERROR: Failed to parse speed '" << value_str << "': " << e.what() << std::endl;
                            }
                        } else {
                            std::cerr << "ZMQ Subscriber ERROR: Invalid message format: '" << data << "'" << std::endl;
                            std::cerr << "  colon_pos: " << colon_pos << ", semicolon_pos: " << semicolon_pos << std::endl;
                        }
                    } else {
                        std::cout << "ZMQ Subscriber: Message ignored (not a speed message): '" << data << "'" << std::endl;
                    }
                }
            } else {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 5) {
                    heartbeat_counter++;
                    std::cout << "ZMQ Subscriber: Heartbeat #" << heartbeat_counter << " — no messages received (timeout)" << std::endl;
                    last_heartbeat = now;
                }
            }

        } catch (const zmq::error_t& e) {
            if (e.num() == EAGAIN) {
                std::cout << "ZMQ Subscriber: Receive timeout — waiting for messages..." << std::endl;
            } else if (e.num() == ETERM) {
                std::cout << "ZMQ Subscriber: Context terminated, stopping receive loop" << std::endl;
                break;
            } else {
                std::cerr << "ZMQ Subscriber ERROR: " << e.what() << " (code: " << e.num() << ")" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (const std::exception& e) {
            std::cerr << "ZMQ Subscriber ERROR: Error in receive loop: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cout << "ZMQ Subscriber: Receive loop stopped." << std::endl;
}
