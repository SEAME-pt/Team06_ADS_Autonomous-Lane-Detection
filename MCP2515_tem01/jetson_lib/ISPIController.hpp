/*!
 * @file ISPIController.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the ISPIController interface.
 * @details This file contains the definition of the ISPIController interface,
 * which is responsible for controlling the SPI communication.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef ISPICONTROLLER_HPP
#define ISPICONTROLLER_HPP

#include <cstdint>
#include <string>

/*!
 * @brief Interface for the SPI controller.
 * @class ISPIController
 */
class ISPIController {
public:
	virtual ~ISPIController() = default;
	virtual bool openDevice(const std::string &device) = 0;
	virtual void configure(uint8_t mode, uint8_t bits, uint32_t speed) = 0;
	virtual void writeByte(uint8_t address, uint8_t data) = 0;
	virtual uint8_t readByte(uint8_t address) = 0;
	virtual void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length) = 0;
	virtual void closeDevice() = 0;
};
#endif // ISPICONTROLLER_HPP
