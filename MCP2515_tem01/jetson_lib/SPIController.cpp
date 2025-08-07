/*!
 * @file SPIController.cpp
 * @brief Implementation of the SPIController class.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the SPIController class,
 * which controls the SPI communication.
 *
 * @note This class is used to control the SPI communication for the MCP2515 CAN
 * controller.
 *
 * @warning Ensure that the SPI device is properly connected and configured on
 * your system.
 *
 * @see SPIController.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "SPIController.hpp"
#include <cstring>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

/*!
 * @brief Construct a new SPIController::SPIController object
 *
 * @param ioctlFunc
 * @param openFunc
 * @param closeFunc
 * @details This constructor initializes the SPIController object with the
 * specified functions.
 */
SPIController::SPIController(IoctlFunc ioctlFunc, OpenFunc openFunc,
														 CloseFunc closeFunc)
		: spi_fd(-1), mode(DefaultMode), bits(DefaultBitsPerWord),
			speed(DefaultSpeedHz), m_ioctlFunc(ioctlFunc), m_openFunc(openFunc),
			m_closeFunc(closeFunc) {}

/*!
 * @brief Destroy the SPIController::SPIController object
 *
 * @details This destructor closes the SPI device.
 */
SPIController::~SPIController() { closeDevice(); }

/*!
 * @brief Open the SPI device.
 *
 * @param device The device to open.
 * @returns True if the device was opened successfully.
 * @throws std::runtime_error if the device cannot be opened.
 * @details This function opens the SPI device with the specified device name.
 */
bool SPIController::openDevice(const std::string &device) {
	spi_fd = m_openFunc(device.c_str(), O_RDWR);
	if (spi_fd < 0) {
		throw std::runtime_error("Failed to open SPI device");
	}
	return true;
}

/*!
 * @brief Configure the SPI device.
 *
 * @param mode The SPI mode.
 * @param bits The number of bits per word.
 * @param speed The speed in Hz.
 * @throws std::runtime_error if the device is not open.
 * @throws std::runtime_error if the SPI mode cannot be set.
 * @throws std::runtime_error if the bits per word cannot be set.
 * @throws std::runtime_error if the speed cannot be set.
 * @details This function configures the SPI device with the specified mode,
 * bits per word, and speed.
 */
void SPIController::configure(uint8_t mode, uint8_t bits, uint32_t speed) {
	if (spi_fd < 0) {
		throw std::runtime_error("SPI device not open");
	}

	this->mode = mode;
	this->bits = bits;
	this->speed = speed;

	if (m_ioctlFunc(spi_fd, SPI_IOC_WR_MODE, &mode) < 0) {
		throw std::runtime_error("Failed to set SPI mode");
	}

	if (m_ioctlFunc(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) {
		throw std::runtime_error("Failed to set SPI bits per word");
	}

	if (m_ioctlFunc(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
		throw std::runtime_error("Failed to set SPI speed");
	}
}

/*!
 * @brief Write a byte to the SPI device.
 *
 * @param address The address to write to.
 * @param data The data to write.
 * @details This function writes a byte to the SPI device at the specified
 * address.
 */
void SPIController::writeByte(uint8_t address, uint8_t data) {
	uint8_t tx[] = {static_cast<uint8_t>(Opcode::Write), address, data};
	spiTransfer(tx, nullptr, sizeof(tx));
}

/*!
 * @brief Read a byte from the SPI device.
 *
 * @param address The address to read from.
 * @returns The byte read from the SPI device.
 * @details This function reads a byte from the SPI device at the specified
 * address.
 */
uint8_t SPIController::readByte(uint8_t address) {
	uint8_t tx[] = {static_cast<uint8_t>(Opcode::Read), address, 0x00};
	uint8_t rx[sizeof(tx)] = {0};
	spiTransfer(tx, rx, sizeof(tx));
	return rx[2];
}

/*!
 * @brief Transfer data over SPI.
 *
 * @param tx The data to transmit.
 * @param rx The data to receive.
 * @param length The length of the data.
 * @throws std::runtime_error if the SPI device is not open.
 * @throws std::runtime_error if the SPI transfer fails.
 * @details This function transfers data over SPI.
 */
void SPIController::spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length) {
	if (spi_fd < 0) {
		throw std::runtime_error("SPI device not open");
	}

	struct spi_ioc_transfer transfer = {};
	transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
	transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
	transfer.len = length;
	transfer.speed_hz = speed;
	transfer.bits_per_word = bits;

	if (m_ioctlFunc(spi_fd, SPI_IOC_MESSAGE(1), &transfer) < 0) {
		throw std::runtime_error("SPI transfer error");
	}
}

/*!
 * @brief Close the SPI device.
 * @details This function closes the SPI device.
 */
void SPIController::closeDevice() {
	if (spi_fd >= 0) {
		m_closeFunc(spi_fd);
		spi_fd = -1;
	}
}
