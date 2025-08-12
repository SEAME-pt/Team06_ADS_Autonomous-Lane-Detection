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
 * which controls the SPI bus.
 *
 * @see SPIController.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "SPIController.hpp"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <cstring>
#include <vector>

SPIController::SPIController(const std::string& devicePath)
    : m_devicePath(devicePath), m_fd(-1) {}

SPIController::~SPIController() {
    closeDevice();
}

void SPIController::openDevice() {
    m_fd = open(m_devicePath.c_str(), O_RDWR);
    if (m_fd < 0) {
        throw std::runtime_error("Could not open SPI device");
    }

    uint8_t mode = SPI_MODE_0;
    uint8_t bits = 8;
    uint32_t speed = 1000000; // 1 MHz

    if (ioctl(m_fd, SPI_IOC_WR_MODE, &mode) < 0) {
        throw std::runtime_error("Could not set SPI mode");
    }
    if (ioctl(m_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) {
        throw std::runtime_error("Could not set bits per word");
    }
    if (ioctl(m_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        throw std::runtime_error("Could not set SPI speed");
    }
}

void SPIController::closeDevice() {
    if (m_fd >= 0) {
        close(m_fd);
        m_fd = -1;
    }
}

void SPIController::spiTransfer(uint8_t* txData, size_t size) {
    std::vector<uint8_t> rxData(size);
    struct spi_ioc_transfer tr = {};
    tr.tx_buf = (unsigned long)txData;
    tr.rx_buf = (unsigned long)rxData.data();
    tr.len = size;
    tr.speed_hz = 1000000;

    if (ioctl(m_fd, SPI_IOC_MESSAGE(1), &tr) < 0) {
        throw std::runtime_error("SPI transfer failed");
    }
    memcpy(txData, rxData.data(), size);
}

uint8_t SPIController::readByte() {
    uint8_t rxData;
    uint8_t txData = 0x00;
    spiTransfer(&txData, 1);
    return txData;
}

void SPIController::writeByte(uint8_t data) {
    spiTransfer(&data, 1);
}