/*!
 * @file SPIController.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the SPIController class.
 * @details This file contains the definition of the SPIController class, which
 * is responsible for controlling the SPI communication.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include "ISPIController.hpp"
#include <cstdint>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

using IoctlFunc = int (*)(int, unsigned long, ...);
using OpenFunc = int (*)(const char *, int, ...);
using CloseFunc = int (*)(int);

/*!
 * @brief Class that controls the SPI communication.
 * @class SPIController inherits from ISPIController
 */
class SPIController : public ISPIController {
public:
	enum class Opcode : uint8_t { Write = 0x02, Read = 0x03 };

	SPIController(IoctlFunc ioctlFunc = ::ioctl, OpenFunc openFunc = ::open,
								CloseFunc closeFunc = ::close);
	~SPIController() override;

	bool openDevice(const std::string &device) override;
	void configure(uint8_t mode, uint8_t bits, uint32_t speed) override;
	void writeByte(uint8_t address, uint8_t data) override;
	uint8_t readByte(uint8_t address) override;
	void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length) override;
	void closeDevice() override;

private:
	/*! @brief File descriptor of the SPI device. */
	int spi_fd;
	/*! @brief Mode of the SPI communication. */
	uint8_t mode;
	/*! @brief Number of bits per word. */
	uint8_t bits;
	/*! @brief Speed of the SPI communication. */
	uint32_t speed;

	/*! @brief Function pointer to the ioctl function. */
	IoctlFunc m_ioctlFunc;
	/*! @brief Function pointer to the open function. */
	OpenFunc m_openFunc;
	/*! @brief Function pointer to the close function. */
	CloseFunc m_closeFunc;

	static constexpr uint8_t DefaultBitsPerWord = 8;
	static constexpr uint32_t DefaultSpeedHz = 1'000'000;
	static constexpr uint8_t DefaultMode = 0;
};

#endif // SPICONTROLLER_HPP
