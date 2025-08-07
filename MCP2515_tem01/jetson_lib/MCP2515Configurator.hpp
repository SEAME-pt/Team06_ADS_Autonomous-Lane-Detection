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
#include <vector>

/*!
 * @brief Class that configures the MCP2515 CAN controller.
 * @class MCP2515Configurator
 */
class MCP2515Configurator {
public:
	explicit MCP2515Configurator(ISPIController &spiController);
	~MCP2515Configurator() = default;

	bool resetChip();
	void configureBaudRate();
	void configureTXBuffer();
	void configureRXBuffer();
	void configureFiltersAndMasks();
	void configureInterrupts();
	void setMode(uint8_t mode);

	bool verifyMode(uint8_t expectedMode);
	std::vector<uint8_t> readCANMessage(uint16_t &frameID);
	void sendCANMessage(uint16_t frameID, uint8_t *CAN_TX_Buf, uint8_t length1);

	static constexpr uint8_t RESET_CMD = 0xC0;
	static constexpr uint8_t CANCTRL = 0x0F;
	static constexpr uint8_t CANSTAT = 0x0E;
	static constexpr uint8_t CNF1 = 0x2A;
	static constexpr uint8_t CNF2 = 0x29;
	static constexpr uint8_t CNF3 = 0x28;
	static constexpr uint8_t TXB0CTRL = 0x30;
	static constexpr uint8_t RXB0CTRL = 0x60;
	static constexpr uint8_t CANINTF = 0x2C;
	static constexpr uint8_t CANINTE = 0x2B;
	static constexpr uint8_t RXB0SIDH = 0x61;
	static constexpr uint8_t RXB0SIDL = 0x62;

	static constexpr uint8_t CAN_RD_STATUS = 0xA0;
	static constexpr uint8_t TXB0SIDH = 0x31;
	static constexpr uint8_t TXB0SIDL = 0x32;
	static constexpr uint8_t TXB0EID8 = 0x33;
	static constexpr uint8_t TXB0EID0 = 0x34;
	static constexpr uint8_t TXB0DLC = 0x35;
	static constexpr uint8_t TXB0D0 = 0x36;
	static constexpr uint8_t CAN_RTS_TXB0 = 0x81;

private:
	/*! @brief Reference to the SPI controller. */
	ISPIController &spiController;

	void writeRegister(uint8_t address, uint8_t value);
	uint8_t readRegister(uint8_t address);
	void sendCommand(uint8_t command);
};

#endif // MCP2515CONFIGURATOR_HPP
