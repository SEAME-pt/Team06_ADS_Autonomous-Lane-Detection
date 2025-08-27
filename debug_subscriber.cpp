#include <zmq.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Debug ZMQ Subscriber starting..." << std::endl;

    try {
        // Create context
        zmq::context_t context(1);

        // Create subscriber socket
        zmq::socket_t subscriber(context, zmq::socket_type::sub);

        // Connect to the publisher
        subscriber.connect("tcp://127.0.0.1:5555");
        std::cout << "Connected to tcp://127.0.0.1:5555" << std::endl;

        // Subscribe to all messages (empty filter means receive all)
        subscriber.set(zmq::sockopt::subscribe, "");

        // Set receive timeout
        subscriber.set(zmq::sockopt::rcvtimeo, 1000); // 1 second timeout

        std::cout << "Waiting for messages..." << std::endl;

        int messageCount = 0;
        int emptyCount = 0;
        while (messageCount < 20) {  // Limit to 20 messages for debugging
            try {
                zmq::message_t message;
                auto result = subscriber.recv(message, zmq::recv_flags::none);

                if (result) {
                    messageCount++;
                    size_t msg_size = message.size();

                    if (msg_size == 0) {
                        emptyCount++;
                        std::cout << "Received EMPTY message #" << messageCount << " (empty count: " << emptyCount << ")" << std::endl;

                        // Check for multi-part
                        int more = 0;
                        size_t more_size = sizeof(more);
                        subscriber.getsockopt(ZMQ_RCVMORE, &more, &more_size);

                        if (more) {
                            std::cout << "  Multi-part message detected!" << std::endl;
                            zmq::message_t next_message;
                            auto next_result = subscriber.recv(next_message, zmq::recv_flags::none);
                            if (next_result) {
                                std::string next_content(static_cast<char*>(next_message.data()), next_message.size());
                                std::cout << "  Next part: '" << next_content << "' (" << next_message.size() << " bytes)" << std::endl;
                            }
                        }
                    } else {
                        // Extract message content
                        std::string content(static_cast<char*>(message.data()), message.size());
                        std::cout << "Received [" << messageCount << "] (" << msg_size << " bytes): '" << content << "'" << std::endl;
                    }
                } else {
                    std::cout << "No message received (timeout)" << std::endl;
                }
            } catch (const zmq::error_t& e) {
                if (e.num() == EAGAIN) {
                    std::cout << "Receive timeout - waiting for messages..." << std::endl;
                } else {
                    std::cerr << "ZMQ error: " << e.what() << std::endl;
                    break;
                }
            }
        }

        std::cout << "Debug complete. Total messages: " << messageCount << ", Empty: " << emptyCount << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
