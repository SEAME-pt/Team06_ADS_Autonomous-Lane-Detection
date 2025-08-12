/*!
 * @file MCP2515Controller.cpp
 * @brief Implementation of the MCP2515Controller class.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the MCP2515Controller class,
 * which controls the MCP2515 CAN controller.
 *
 * @note This class is used to control the MCP2515 CAN controller and
 * process incoming messages.
 *
 * @see MCP2515Controller.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "MCP2515Controller.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <vector>
#include "CanBusManager.hpp"
#include "MCP2515Configurator.hpp"

// Constantes do MCP2515 (para referencia, ja definidas no MCP2515Configurator.cpp)
namespace MCP2515Constants {
    constexpr uint8_t MODE_NORMAL = 0x00;
}
using namespace MCP2515Constants;

/*!
 * @brief Construct a new MCP2515Controller::MCP2515Controller object
 *
 * @param spiController
 * @param configurator
 * @param messageProcessor
 * @details This constructor initializes the MCP2515Controller object.
 */
MCP2515Controller::MCP2515Controller(std::unique_ptr<ISPIController> spiController,
                                     std::unique_ptr<MCP2515Configurator> configurator,
                                     std::shared_ptr<CANMessageProcessor> messageProcessor)
    : spi(std::move(spiController)), config(std::move(configurator)),
      processor(messageProcessor) {}

/*!
 * @brief Initializes the MCP2515 controller.
 */
void MCP2515Controller::initialize() {
    try {
        spi->openDevice();
        config->resetChip();
        config->configureBaudRate();
        config->configureRXBuffer();
        config->configureFiltersAndMasks();
        config->configureInterrupts();
        
        // NOVO: Coloca o chip em modo de operação normal
        config->setMode(MODE_NORMAL); 

    } catch (const std::runtime_error& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
    }
}

/*!
 * @brief Processes the reading of CAN messages.
 */
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

/*!
 * @brief Reads a CAN message from a specific buffer.
 *
 * @param address The address of the buffer to read from.
 * @param frameID The frame ID of the message.
 * @param data The data of the message.
 */
void MCP2515Controller::readMessage(uint8_t address, uint16_t &frameID,
                                   std::vector<uint8_t> &data) {
    std::vector<uint8_t> buffer(13); // 1 opcode + 4 ID + 1 DLC + 8 data
    buffer[0] = address;
    spi->spiTransfer(buffer.data(), buffer.size());

    // Extracting ID
    frameID = (static_cast<uint16_t>(buffer[1]) << 3) | (buffer[2] >> 5);
    
    uint8_t dlc = buffer[5] & 0x0F;
    data.assign(buffer.begin() + 6, buffer.begin() + 6 + dlc);
}