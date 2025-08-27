#include "CanBusManager.hpp"
#include <iostream>
#include <cstring> // Adicionado para memcpy
#include <atomic> // Incluir a biblioteca atomic

extern std::atomic<double> current_speed; // Declarar a variável global

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
        current_speed.store(speed); // Atualiza a variável atómica global
        //std::cout << "Speed: " << speed << " m/s" << std::endl;
    }
}

/*!
 * @brief Callback function to handle received RPM data.
 */
void CanBusManager::handleRPM(const std::vector<uint8_t>& data) {
    if (data.size() == sizeof(int32_t)) {
        int32_t rpm;
        memcpy(&rpm, data.data(), sizeof(int32_t));
        std::cout << "RPM: " << rpm << std::endl;
    }
}