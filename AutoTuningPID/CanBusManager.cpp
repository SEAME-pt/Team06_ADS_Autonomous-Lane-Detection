#include "CanBusManager.hpp"
#include <iostream>
#include <cstring>

CanBusManager::CanBusManager(std::unique_ptr<MCP2515Controller> controller)
    : mcp2515Controller_(std::move(controller)), currentSpeed_(0.0f) {}

CanBusManager::~CanBusManager() {
    stop();
}

void CanBusManager::start() {
    if (!running) {
        running = true;
        mcp2515Controller_->initialize();
        workerThread = std::thread([this] {
            while (running) {
                this->mcp2515Controller_->processReading();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
}

void CanBusManager::stop() {
    running = false;
    if (workerThread.joinable()) {
        workerThread.join();
    }
}

void CanBusManager::handleSpeed(const std::vector<uint8_t>& data) {
    if (data.size() == sizeof(float)) {
        memcpy(&currentSpeed_, data.data(), sizeof(float)); // Store speed
        std::cout << "Velocidade recebida: " << currentSpeed_ << " m/s" << std::endl;
    }
}

void CanBusManager::handleRPM(const std::vector<uint8_t>& data) {
    if (data.size() == sizeof(int32_t)) {
        int32_t rpm;
        memcpy(&rpm, data.data(), sizeof(int32_t));
        std::cout << "RPM recebido: " << rpm << std::endl;
    }
}

float CanBusManager::getCurrentSpeed() const {
    return currentSpeed_;
}

void CanBusManager::sendCANMessage(uint16_t id, const std::vector<uint8_t>& data) {
    mcp2515Controller_->sendCANMessage(id, data); // Delegate to MCP2515Controller
}

MCP2515Controller* CanBusManager::getMCP2515Controller() const {
    return mcp2515Controller_.get();
}