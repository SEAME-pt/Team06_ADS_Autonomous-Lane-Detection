/*!
 * @file CANMessageProcessor.cpp
 * @brief Implementation of the CANMessageProcessor class.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the CANMessageProcessor
 * class, which processes CAN messages.
 *
 * @note This class is used to process CAN messages and call the appropriate
 * handler for each message.
 *
 * @warning Ensure that the MessageHandler type is properly defined.
 *
 * @see CANMessageProcessor.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "CANMessageProcessor.hpp"
#include <stdexcept>

/*!
 * @brief Construct a new CANMessageProcessor::CANMessageProcessor object
 *
 * @details This constructor initializes the CANMessageProcessor object.
 */
CANMessageProcessor::CANMessageProcessor() {}

/*!
 * @brief Destroy the CANMessageProcessor::CANMessageProcessor object
 *
 * @param frameID
 * @param handler
 * @throws std::invalid_argument if the handler is null
 * @details This method registers a handler for a specific frame ID.
 */
void CANMessageProcessor::registerHandler(uint16_t frameID,
																					MessageHandler handler) {
	if (!handler) {
		throw std::invalid_argument("Handler cannot be null");
	}
	handlers[frameID] = handler;
}

/*!
 * @brief Process a CAN message
 *
 * @param frameID The frame ID of the message.
 * @param data The data of the message.
 * @throws std::runtime_error if no handler is registered for the frame ID.
 * @details This method processes a CAN message by calling the appropriate
 * handler for the frame ID.
 */
void CANMessageProcessor::processMessage(uint16_t frameID,
																				 const std::vector<uint8_t> &data) {
	auto it = handlers.find(frameID);
	if (it != handlers.end()) {
		it->second(data);
	} else {
		throw std::runtime_error("No handler registered for frame ID: " +
														 std::to_string(frameID));
	}
}
