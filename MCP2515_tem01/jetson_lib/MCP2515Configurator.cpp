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
 * @details This file contains the implementation of the MCP2515Configurator
 * class, which configures the MCP2515 CAN controller.
 *
 * @note This class is used to configure the MCP2515 CAN controller for
 * communication.
 *
 * @warning Ensure that the SPI controller is properly implemented.
 *
 * @see MCP2515Configurator.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "MCP2515Configurator.hpp"
#include <chrono>
#include <thread>

/*!
 * @brief Construct a new MCP2515Configurator::MCP2515Configurator object
 * @param spiController The SPI controller to use for communication.
 * @details This constructor initializes the MCP2515Configurator object with the
 * specified SPI controller.
 */
MCP2515Configurator::MCP2515Configurator(ISPIController &spiController)
		: spiController(spiController) {}

/*!
 * @brief clean up the resources used by the MCP2515Configurator.
 * @returns The chip in configuration mode.
 * @details This function cleans up the resources used putting the MCP2515 to
 * default configuration.
 */
bool MCP2515Configurator::resetChip() {
	sendCommand(RESET_CMD);
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	uint8_t status = readRegister(CANSTAT);
	return (status & 0xE0) == 0x80; // Verify configuration mode
}

/*!
 * @brief Configure the baud rate for the MCP2515.
 * @details This function sets the baud rate for the MCP2515.
 */
void MCP2515Configurator::configureBaudRate() {
	writeRegister(CNF1, 0x00); // Set BRP (Baud Rate Prescaler)
	writeRegister(CNF2, 0x90); // Set Propagation and Phase Segment 1
	writeRegister(CNF3, 0x02); // Set Phase Segment 2
}

/*!
 * @brief Configure the TX buffer for the MCP2515.
 * @details This function configures the TX buffer for the MCP2515.
 */
void MCP2515Configurator::configureTXBuffer() {
	writeRegister(TXB0CTRL, 0x00); // Clear TX buffer control register
}

/*!
 * @brief Configure the RX buffer for the MCP2515.
 * @details This function configures the RX buffer for the MCP2515.
 */
void MCP2515Configurator::configureRXBuffer() {
	writeRegister(RXB0CTRL,
								0x60); // Enable rollover and set RX mode to receive all
}

/*!
 * @brief Configure the filters and masks for the MCP2515.
 * @details This function configures the filters and masks for the MCP2515.
 */
void MCP2515Configurator::configureFiltersAndMasks() {
	writeRegister(0x00, 0xFF); // Set filter 0
	writeRegister(0x01, 0xFF); // Set mask 0
}

/*!
 * @brief Configure the interrupts for the MCP2515.
 * @details This function configures the interrupts for the MCP2515.
 */
void MCP2515Configurator::configureInterrupts() {
	writeRegister(CANINTE, 0x01); // Enable receive interrupt
}

/*!
 * @brief Set the mode for the MCP2515.
 * @param mode The mode to set.
 * @details This function sets the mode for the MCP2515.
 */
void MCP2515Configurator::setMode(uint8_t mode) {
	writeRegister(CANCTRL, mode);
}

/*!
 * @brief Verify the mode of the MCP2515.
 * @param expectedMode The expected mode.
 * @returns True if the mode is as expected, false otherwise.
 * @details This function verifies the mode of the MCP2515.
 */
bool MCP2515Configurator::verifyMode(uint8_t expectedMode) {
	uint8_t mode = readRegister(CANSTAT) & 0xE0;
	return mode == expectedMode;
}

/*!
 * @brief Write a value to a register.
 * @param address The address of the register.
 * @param value The value to write.
 * @details This function writes a value to a register.
 */
void MCP2515Configurator::writeRegister(uint8_t address, uint8_t value) {
	spiController.writeByte(address, value);
}

/*!
 * @brief Read a value from a register.
 * @param address The address of the register.
 * @returns The value read from the register.
 * @details This function reads a value from a register.
 */
uint8_t MCP2515Configurator::readRegister(uint8_t address) {
	return spiController.readByte(address);
}

/*!
 * @brief Send a command to the MCP2515.
 * @param command The command to send.
 * @returns The response from the MCP2515.
 * @details This function sends a command to the MCP2515
 */
void MCP2515Configurator::sendCommand(uint8_t command) {
	uint8_t tx[] = {command};
	spiController.spiTransfer(tx, nullptr, sizeof(tx));
}

/*!
 * @brief Read a CAN message from the MCP2515.
 * @param frameID The frame ID of the message.
 * @returns The data of the message.
 * @details This function reads a CAN message from the MCP2515.
 */
std::vector<uint8_t> MCP2515Configurator::readCANMessage(uint16_t &frameID) {
	std::vector<uint8_t> CAN_RX_Buf;

	if (readRegister(CANINTF) & 0x01) { // Check if data is available
		uint8_t sidh = readRegister(RXB0SIDH);
		uint8_t sidl = readRegister(RXB0SIDL);
		frameID = (sidh << 3) | (sidl >> 5);

		uint8_t len = readRegister(0x65); // Length of the data
		for (uint8_t i = 0; i < len; ++i) {
			CAN_RX_Buf.push_back(readRegister(0x66 + i));
		}

		writeRegister(CANINTF, 0x00); // Clear interrupt flag
	}

	return CAN_RX_Buf;
}

/*!
 * @brief Send a CAN message to the MCP2515.
 * @param frameID The frame ID of the message.
 * @param CAN_TX_Buf The data of the message.
 * @param length1 The length of the data.
 * @details This function sends a CAN message to the MCP2515.
 */
void MCP2515Configurator::sendCANMessage(uint16_t frameID, uint8_t *CAN_TX_Buf,
																				 uint8_t length1) {
	uint8_t tempdata = readRegister(CAN_RD_STATUS);
	writeRegister(TXB0SIDH, (frameID >> 3) & 0xFF);
	writeRegister(TXB0SIDL, (frameID & 0x07) << 5);

	writeRegister(TXB0EID8, 0);
	writeRegister(TXB0EID0, 0);
	writeRegister(TXB0DLC, length1);
	for (uint8_t j = 0; j < length1; ++j) {
		writeRegister(TXB0D0 + j, CAN_TX_Buf[j]);
	}

	if (tempdata & 0x04) { // TXREQ
		std::this_thread::sleep_for(
				std::chrono::milliseconds(10)); // sleep for 0.01 seconds
		writeRegister(TXB0CTRL, 0);         // clean flag
		while (true) {
			if ((readRegister(CAN_RD_STATUS) & 0x04) != 1) {
				break;
			}
		}
	}
	uint8_t rtsCommand = CAN_RTS_TXB0;
	spiController.spiTransfer(&rtsCommand, nullptr, 1);
}
