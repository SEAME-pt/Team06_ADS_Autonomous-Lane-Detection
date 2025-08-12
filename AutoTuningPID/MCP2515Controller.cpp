#include "MCP2515Controller.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <vector>
#include "CanBusManager.hpp"
#include "MCP2515Configurator.hpp"

namespace MCP2515Constants {
    constexpr uint8_t MODE_NORMAL = 0x00;
    constexpr uint8_t WRITE_INSTRUCTION = 0x02;
    constexpr uint8_t TXB0SIDH = 0x31; // Transmit Buffer 0 Standard ID High
    constexpr uint8_t TXB0D0 = 0x36;   // Transmit Buffer 0 Data
    constexpr uint8_t TXB0DLC = 0x35;  // Transmit Buffer 0 Data Length
    constexpr uint8_t RTS_INSTRUCTION = 0x80; // Request to Send
}
using namespace MCP2515Constants;

MCP2515Controller::MCP2515Controller(std::unique_ptr<ISPIController> spiController,
                                     std::unique_ptr<MCP2515Configurator> configurator,
                                     std::shared_ptr<CANMessageProcessor> messageProcessor)
    : spi(std::move(spiController)), config(std::move(configurator)),
      processor(messageProcessor) {}

void MCP2515Controller::initialize() {
    try {
        spi->openDevice();
        config->resetChip();
        config->configureBaudRate();
        config->configureRXBuffer();
        config->configureFiltersAndMasks();
        config->configureInterrupts();
        config->setMode(MODE_NORMAL);
    } catch (const std::runtime_error& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
    }
}

void MCP2515Controller::processReading() {
    uint8_t status = config->readStatus();
    if (status & 0x01) { // RX0IF bit
        uint16_t frameID;
        std::vector<uint8_t> data;
        readMessage(0x90, frameID, data); // Read RXB0
        processor->processMessage(frameID, data);
        config->clearInterrupts(0x01);
    }
    if (status & 0x02) { // RX1IF bit
        uint16_t frameID;
        std::vector<uint8_t> data;
        readMessage(0x94, frameID, data); // Read RXB1
        processor->processMessage(frameID, data);
        config->clearInterrupts(0x02);
    }
}

void MCP2515Controller::readMessage(uint8_t address, uint16_t &frameID,
                                   std::vector<uint8_t> &data) {
    std::vector<uint8_t> buffer(13); // 1 opcode + 4 ID + 1 DLC + 8 data
    buffer[0] = address;
    spi->spiTransfer(buffer.data(), buffer.size());
    frameID = (static_cast<uint16_t>(buffer[1]) << 3) | (buffer[2] >> 5);
    uint8_t dlc = buffer[5] & 0x0F;
    data.assign(buffer.begin() + 6, buffer.begin() + 6 + dlc);
}

void MCP2515Controller::sendCANMessage(uint16_t id, const std::vector<uint8_t>& data) {
    // Ensure data size is valid (max 8 bytes for CAN)
    if (data.size() > 8) {
        throw std::runtime_error("CAN message data exceeds 8 bytes");
    }

    // Prepare buffer for TXB0
    std::vector<uint8_t> buffer(14); // 1 opcode + 4 ID + 1 DLC + 8 data
    buffer[0] = WRITE_INSTRUCTION;
    buffer[1] = TXB0SIDH;
    buffer[2] = (id >> 3);        // SIDH
    buffer[3] = (id & 0x07) << 5; // SIDL
    buffer[4] = 0x00;             // EID8
    buffer[5] = 0x00;             // EID0
    buffer[6] = data.size();      // DLC
    for (size_t i = 0; i < data.size(); ++i) {
        buffer[7 + i] = data[i];  // Data bytes
    }

    // Write to TXB0
    spi->spiTransfer(buffer.data(), 7 + data.size());

    // Request to send
    uint8_t rts_data = RTS_INSTRUCTION | 0x01; // RTS for TXB0
    spi->spiTransfer(&rts_data, 1);
}

MCP2515Configurator* MCP2515Controller::getConfigurator() const {
    return config.get();
}