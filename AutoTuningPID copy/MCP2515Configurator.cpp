/*!
 * @file MCP2515Configurator.cpp
 * @brief Implementation of the MCP2515Configurator class.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the MCP2515Configurator class,
 * which configures the MCP2515 CAN controller.
 *
 * @see MCP2515Configurator.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "MCP2515Configurator.hpp"
#include <iostream>
#include <stdexcept>
#include <unistd.h> // For usleep

// Definir as constantes do MCP2515 aqui
namespace MCP2515Constants {
    // Comandos de instrução
    constexpr uint8_t RESET_INSTRUCTION = 0xC0;
    constexpr uint8_t READ_INSTRUCTION = 0x03;
    constexpr uint8_t WRITE_INSTRUCTION = 0x02;
    constexpr uint8_t READ_STATUS_INSTRUCTION = 0xA0;
    constexpr uint8_t BIT_MODIFY_INSTRUCTION = 0x05;

    // Registos de controlo
    constexpr uint8_t CANCTRL = 0x0F;
    constexpr uint8_t CANSTAT = 0x0E;
    constexpr uint8_t CNF1 = 0x2A;
    constexpr uint8_t CNF2 = 0x29;
    constexpr uint8_t CNF3 = 0x28;
    constexpr uint8_t RXB0CTRL = 0x60;
    constexpr uint8_t RXB1CTRL = 0x70;
    constexpr uint8_t CANINTE = 0x2B;
    constexpr uint8_t CANINTF = 0x2C;

    // Modos de operação
    constexpr uint8_t MODE_NORMAL = 0x00;
    constexpr uint8_t MODE_CONFIG = 0x80;

    // Bits de interrupção
    constexpr uint8_t RX0IE = 0x01; // Receive Buffer 0 Full Interrupt Enable
    constexpr uint8_t RX1IE = 0x02; // Receive Buffer 1 Full Interrupt Enable

    // Registos de ID e Máscara (apenas exemplos)
    constexpr uint8_t RXM0SIDH = 0x20;
    constexpr uint8_t RXM1SIDH = 0x24;
    constexpr uint8_t RXF0SIDH = 0x00;
    constexpr uint8_t RXF1SIDH = 0x04;
}

using namespace MCP2515Constants;

MCP2515Configurator::MCP2515Configurator(ISPIController& spi)
    : spiController(spi) {}

MCP2515Configurator::~MCP2515Configurator() = default;

void MCP2515Configurator::resetChip() {
    uint8_t txData = RESET_INSTRUCTION;
    spiController.spiTransfer(&txData, 1);
    usleep(100);
}

void MCP2515Configurator::configureBaudRate() {
    writeRegister(CANCTRL, MODE_CONFIG);
    usleep(100);

    writeRegister(CNF1, 0x00);
    writeRegister(CNF2, 0x90);
    writeRegister(CNF3, 0x02);
}

void MCP2515Configurator::configureRXBuffer() {
    writeRegisterMask(RXB0CTRL, 0x60, 0x60);
    writeRegisterMask(RXB1CTRL, 0x60, 0x60);
}

void MCP2515Configurator::configureFiltersAndMasks() {
    writeID(RXM0SIDH, 0x0000);
    writeID(RXM1SIDH, 0x0000);

    writeID(RXF0SIDH, 0x0100);
    writeID(RXF1SIDH, 0x0300);
}

void MCP2515Configurator::configureInterrupts() {
    writeRegister(CANINTE, RX0IE | RX1IE);
}

void MCP2515Configurator::clearInterrupts(uint8_t flag) {
    writeRegisterMask(CANINTF, flag, 0x00);
}

uint8_t MCP2515Configurator::readStatus() {
    uint8_t txData[2] = {READ_STATUS_INSTRUCTION, 0x00};
    spiController.spiTransfer(txData, 2);
    return txData[1];
}

void MCP2515Configurator::setMode(uint8_t mode) {
    writeRegister(CANCTRL, mode);
    usleep(100);
}

bool MCP2515Configurator::verifyMode(uint8_t expectedMode) {
    uint8_t currentMode = readRegister(CANSTAT) & 0xE0;
    return currentMode == expectedMode;
}

uint8_t MCP2515Configurator::readRegister(uint8_t address) {
    uint8_t txData[3] = {READ_INSTRUCTION, address, 0x00};
    spiController.spiTransfer(txData, 3);
    return txData[2];
}

void MCP2515Configurator::writeRegister(uint8_t address, uint8_t data) {
    uint8_t txData[3] = {WRITE_INSTRUCTION, address, data};
    spiController.spiTransfer(txData, 3);
}

void MCP2515Configurator::writeRegisterMask(uint8_t address, uint8_t mask, uint8_t data) {
    uint8_t txData[4] = {BIT_MODIFY_INSTRUCTION, address, mask, data};
    spiController.spiTransfer(txData, 4);
}

void MCP2515Configurator::readID(uint8_t address, uint16_t& id) {
    uint8_t txData[5] = {READ_INSTRUCTION, address, 0x00, 0x00, 0x00};
    spiController.spiTransfer(txData, 5);
    id = (static_cast<uint16_t>(txData[2]) << 3) | (txData[3] >> 5);
}

void MCP2515Configurator::writeID(uint8_t address, uint16_t id) {
    uint8_t txData[4];
    txData[0] = WRITE_INSTRUCTION;
    txData[1] = address;
    txData[2] = (id >> 3);
    txData[3] = (id & 0x07) << 5;
    spiController.spiTransfer(txData, 4);
}