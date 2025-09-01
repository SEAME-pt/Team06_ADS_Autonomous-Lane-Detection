/*!
 * @file MCP2515Configurator.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the MCP2515Configurator class.
 * @details This file contains the definition of the MCP2515Configurator class,
 * which is responsible for configuring the MCP2515 CAN controller.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef MCP2515CONFIGURATOR_HPP
#define MCP2515CONFIGURATOR_HPP

#include "ISPIController.hpp"
#include <cstdint>
#include <memory>

class MCP2515Configurator {
public:
    explicit MCP2515Configurator(ISPIController& spi);
    ~MCP2515Configurator();

    void resetChip();
    void configureBaudRate();
    void configureRXBuffer();
    void configureFiltersAndMasks();
    void configureInterrupts();
    void clearInterrupts(uint8_t flag);
    uint8_t readStatus();

    // Novas funções
    void setMode(uint8_t mode);
    bool verifyMode(uint8_t expectedMode);

private:
    ISPIController& spiController;

    // Métodos auxiliares
    uint8_t readRegister(uint8_t address);
    void writeRegister(uint8_t address, uint8_t data);
    void writeRegisterMask(uint8_t address, uint8_t mask, uint8_t data);
    void readID(uint8_t address, uint16_t& id);
    void writeID(uint8_t address, uint16_t id);
};

#endif // MCP2515CONFIGURATOR_HPP