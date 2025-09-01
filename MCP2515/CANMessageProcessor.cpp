#include "CANMessageProcessor.hpp"
#include <stdexcept>

// Adicionar no construtor (opcional, mas útil):
CANMessageProcessor::CANMessageProcessor() {
    defaultHandler = [](const std::vector<uint8_t>& data) {
        //std::cout << "Mensagem CAN desconhecida ignorada." << std::endl;
    };
}

// Nova função:
void CANMessageProcessor::registerDefaultHandler(MessageHandler handler) {
    if (!handler) {
        throw std::invalid_argument("Default handler cannot be null");
    }
    defaultHandler = handler;
}

void CANMessageProcessor::registerHandler(uint16_t frameID, MessageHandler handler) {
	if (!handler) {
		throw std::invalid_argument("Handler cannot be null");
	}
	handlers[frameID] = handler;
}

// Modificar processMessage:
void CANMessageProcessor::processMessage(uint16_t frameID, const std::vector<uint8_t>& data) {
    auto it = handlers.find(frameID);
    if (it != handlers.end()) {
        it->second(data);
    } else if (defaultHandler) {
        defaultHandler(data);  // Usa default se registado
    } else {
        std::cerr << "No handler for frame ID: " << frameID << " (ignorado)" << std::endl;
        // Não lança exceção - evita crash
    }
}
