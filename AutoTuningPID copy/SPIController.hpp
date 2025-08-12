/*!
 * @file SPIController.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the SPIController class.
 * @details This file contains the definition of the SPIController class,
 * which is responsible for controlling the SPI bus.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include "ISPIController.hpp"
#include <string>
#include <cstdint>

class SPIController : public ISPIController {
public:
	SPIController(const std::string& devicePath);
	~SPIController() override;

	void openDevice() override;
	void closeDevice() override;
	void spiTransfer(uint8_t* txData, size_t size) override;
	uint8_t readByte() override;
	void writeByte(uint8_t data) override;

private:
	std::string m_devicePath;
	int m_fd;
};

#endif // SPICONTROLLER_HPP