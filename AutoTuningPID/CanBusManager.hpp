#pragma once
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include "MCP2515Controller.hpp"

class CanBusManager {
public:
    CanBusManager(std::unique_ptr<MCP2515Controller> controller);
    ~CanBusManager();
    void start();
    void stop();
    void handleSpeed(const std::vector<uint8_t>& data);
    void handleRPM(const std::vector<uint8_t>& data);
    float getCurrentSpeed() const; // Added
    void sendCANMessage(uint16_t id, const std::vector<uint8_t>& data); // Added
    MCP2515Controller* getMCP2515Controller() const;

private:
    std::unique_ptr<MCP2515Controller> mcp2515Controller_;
    std::thread workerThread;
    std::atomic<bool> running{false};
    float currentSpeed_{0.0f}; // Added to store speed
};