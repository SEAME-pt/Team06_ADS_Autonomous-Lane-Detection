#pragma once
#include "IMCP2515Controller.hpp"
#include "ISPIController.hpp"
#include "MCP2515Configurator.hpp"
#include "CANMessageProcessor.hpp"
#include <memory>
#include <vector>

class MCP2515Controller : public IMCP2515Controller {
public:
    MCP2515Controller(std::unique_ptr<ISPIController> spiController,
                      std::unique_ptr<MCP2515Configurator> configurator,
                      std::shared_ptr<CANMessageProcessor> messageProcessor);
    void initialize() override;
    void processReading() override;
    void sendCANMessage(uint16_t id, const std::vector<uint8_t>& data); // Added
    MCP2515Configurator* getConfigurator() const;

private:
    void readMessage(uint8_t address, uint16_t& frameID, std::vector<uint8_t>& data);
    std::unique_ptr<ISPIController> spi;
    std::unique_ptr<MCP2515Configurator> config;
    std::shared_ptr<CANMessageProcessor> processor;
};