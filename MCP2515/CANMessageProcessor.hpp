#ifndef CANMESSAGEPROCESSOR_HPP
#define CANMESSAGEPROCESSOR_HPP

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>

/*!
 * @brief Class that processes CAN messages.
 * @class CANMessageProcessor
 */
class CANMessageProcessor {
public:
	using MessageHandler = std::function<void(const std::vector<uint8_t> &)>;

	CANMessageProcessor();
	~CANMessageProcessor() = default;
	void registerHandler(uint16_t frameID, MessageHandler handler);
	void processMessage(uint16_t frameID, const std::vector<uint8_t> &data);
	// Adicionar ao final da class CANMessageProcessor (antes de private):
	void registerDefaultHandler(MessageHandler handler);  // Novo: Para IDs sem handler específico


private:
	/*! @brief Map of frame IDs to message handlers. */
	std::unordered_map<uint16_t, MessageHandler> handlers;
	MessageHandler defaultHandler;
};

#endif // CANMESSAGEPROCESSOR_HPP
