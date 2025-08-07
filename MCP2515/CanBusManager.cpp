/*!
 * @file CanBusManager.cpp
 * @brief Implementation of the CanBusManager class.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the CanBusManager class,
 * which manages the CAN bus communication.
 *
 * @note This class is used to manage the CAN bus communication and
 * process incoming messages.
 *
 * @see CanBusManager.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "CanBusManager.hpp"
#include <iostream>
#include <cstring> // Adicionado para memcpy

/*!
 * @brief Construct a new CanBusManager::CanBusManager object
 *
 * @param controller
 * @details This constructor initializes the CanBusManager object.
 */
CanBusManager::CanBusManager(std::unique_ptr<IMCP2515Controller> controller)
    : mcp2515Controller(std::move(controller)) {}

/*!
 * @brief Destroy the CanBusManager::CanBusManager object
 *
 * @details This destructor stops the worker thread.
 */
CanBusManager::~CanBusManager() {
	stop();
}

/*!
 * @brief Starts the CAN bus manager in a separate thread.
 */
void CanBusManager::start() {
    if (!running) {
        running = true;
        mcp2515Controller->initialize();
        workerThread = std::thread([this] {
            while (running) {
                this->mcp2515Controller->processReading();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
}

/*!
 * @brief Stops the CAN bus manager thread.
 */
void CanBusManager::stop() {
    running = false;
    if (workerThread.joinable()) {
        workerThread.join();
    }
}

/*!
 * @brief Callback function to handle received speed data.
 */
void CanBusManager::handleSpeed(const std::vector<uint8_t>& data) {
    if (data.size() == sizeof(float)) {
        float speed;
        memcpy(&speed, data.data(), sizeof(float));
        std::cout << "Velocidade recebida: " << speed << " m/s" << std::endl;
    }
}

/*!
 * @brief Callback function to handle received RPM data.
 */
void CanBusManager::handleRPM(const std::vector<uint8_t>& data) {
    if (data.size() == sizeof(int32_t)) {
        int32_t rpm;
        memcpy(&rpm, data.data(), sizeof(int32_t));
        std::cout << "RPM recebido: " << rpm << std::endl;
    }
}