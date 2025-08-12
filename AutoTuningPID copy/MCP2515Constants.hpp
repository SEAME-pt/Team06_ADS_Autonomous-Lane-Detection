/*!
 * @file MCP2515Constants.hpp
 * @brief Definition of constants for the MCP2515 CAN controller.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains various constants and register definitions
 * for interacting with the MCP2515 CAN controller.
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MCP2515CONSTANTS_HPP
#define MCP2515CONSTANTS_HPP

#include <cstdint>

namespace MCP2515Constants {
    // Instruction Commands
    constexpr uint8_t RESET_INSTRUCTION = 0xC0;
    constexpr uint8_t READ_INSTRUCTION = 0x03;
    constexpr uint8_t WRITE_INSTRUCTION = 0x02;
    constexpr uint8_t READ_STATUS_INSTRUCTION = 0xA0;
    constexpr uint8_t BIT_MODIFY_INSTRUCTION = 0x05;
    constexpr uint8_t RTS_TXB0_INSTRUCTION = 0x81; // Request to Send TXB0

    // Control Registers
    constexpr uint8_t CANCTRL = 0x0F;
    constexpr uint8_t CANSTAT = 0x0E;
    constexpr uint8_t CNF1 = 0x2A;
    constexpr uint8_t CNF2 = 0x29;
    constexpr uint8_t CNF3 = 0x28;
    constexpr uint8_t TXB0CTRL = 0x30;
    constexpr uint8_t RXB0CTRL = 0x60;
    constexpr uint8_t RXB1CTRL = 0x70;
    constexpr uint8_t CANINTE = 0x2B;
    constexpr uint8_t CANINTF = 0x2C;

    // Receive Buffer 0 Registers
    constexpr uint8_t RXB0SIDH = 0x61;
    constexpr uint8_t RXB0SIDL = 0x62;
    constexpr uint8_t RXB0DLC = 0x65;
    constexpr uint8_t RXB0D0 = 0x66;

    // Receive Buffer 1 Registers
    constexpr uint8_t RXB1SIDH = 0x71;
    constexpr uint8_t RXB1SIDL = 0x72;
    constexpr uint8_t RXB1DLC = 0x75;
    constexpr uint8_t RXB1D0 = 0x76;

    // Transmit Buffer 0 Registers
    constexpr uint8_t TXB0SIDH = 0x31;
    constexpr uint8_t TXB0SIDL = 0x32;
    constexpr uint8_t TXB0EID8 = 0x33;
    constexpr uint8_t TXB0EID0 = 0x34;
    constexpr uint8_t TXB0DLC = 0x35;
    constexpr uint8_t TXB0D0 = 0x36;

    // Operating Modes
    constexpr uint8_t MODE_NORMAL = 0x00;
    constexpr uint8_t MODE_SLEEP = 0x20;
    constexpr uint8_t MODE_LOOPBACK = 0x40;
    constexpr uint8_t MODE_LISTENONLY = 0x60;
    constexpr uint8_t MODE_CONFIG = 0x80;

    // Interrupt Flags
    constexpr uint8_t RX0IE = 0x01; // Receive Buffer 0 Full Interrupt Enable
    constexpr uint8_t RX1IE = 0x02; // Receive Buffer 1 Full Interrupt Enable
    constexpr uint8_t TX0IE = 0x04; // Transmit Buffer 0 Empty Interrupt Enable
    constexpr uint8_t TX1IE = 0x08; // Transmit Buffer 1 Empty Interrupt Enable
    constexpr uint8_t TX2IE = 0x10; // Transmit Buffer 2 Empty Interrupt Enable
    constexpr uint8_t ERRIE = 0x20; // Error Interrupt Enable
    constexpr uint8_t WAKIE = 0x40; // Wake-up Interrupt Enable
    constexpr uint8_t MERRE = 0x80; // Message Error Interrupt Enable

    // Receive Buffer Operating Modes
    constexpr uint8_t RXM_ALL = 0x60; // Receive all messages
    constexpr uint8_t RXM_VALID_ONLY = 0x00; // Receive only valid messages
}

#endif // MCP2515CONSTANTS_HPP