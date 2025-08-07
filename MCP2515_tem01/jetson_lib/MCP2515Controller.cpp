/*!
 * @file MCP2515Controller.cpp
 * @brief Implementation of the MCP2515Controller class.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the MCP2515Controller
 * class, which controls the MCP2515 CAN controller.
 *
 * @note This class is used to control the MCP2515 CAN controller for
 * communication.
 *
 * @warning Ensure that the SPI controller is properly implemented.
 *
 * @see MCP2515Controller.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "MCP2515Controller.hpp"
#include "SPIController.hpp"
#include "NotificationManager.hpp"
#include <QDebug>
#include <QThread>
#include <cstring>
#include <stdexcept>

/*!
 * @brief Construct a new MCP2515Controller::MCP2515Controller object
 * @param spiDevice The SPI device to use for communication.
 * @throw std::runtime_error if the SPI device cannot be opened.
 * @details This constructor initializes the MCP2515Controller object with the
 * specified SPI device.
 */
MCP2515Controller::MCP2515Controller(const std::string &spiDevice)
		: spiController(new SPIController()), configurator(*spiController),
			messageProcessor(), ownsSPIController(true) {
	if (!spiController->openDevice(spiDevice)) {
		throw std::runtime_error("Failed to open SPI device : " + spiDevice);
	}
	setupHandlers();
}

/*!
 * @brief Construct a new MCP2515Controller::MCP2515Controller object
 * @param spiDevice The SPI device to use for communication.
 * @param spiController The SPI controller to use for communication.
 * @throw std::runtime_error if the SPI device cannot be opened.
 * @details This constructor initializes the MCP2515Controller object with the
 * specified SPI device and SPI controller.
 */
MCP2515Controller::MCP2515Controller(const std::string &spiDevice,
																		 ISPIController &spiController)
		: spiController(&spiController), configurator(spiController),
			messageProcessor(), ownsSPIController(false) {
	if (!spiController.openDevice(spiDevice)) {
		throw std::runtime_error("Failed to open SPI device : " + spiDevice);
	}
	setupHandlers();
}

/*!
 * @brief Destroy the MCP2515Controller::MCP2515Controller object
 * @details This destructor closes the SPI device and deletes the SPI controller
 * if it was created by the MCP2515Controller.
 */
MCP2515Controller::~MCP2515Controller() {
	spiController->closeDevice();
	if (this->ownsSPIController) {
		delete this->spiController;
	}
}

/*!
 * @brief Initialize the MCP2515 controller.
 * @throw std::runtime_error if the MCP2515 cannot be reset.
 * @returns True if the MCP2515 is successfully initialized
 * @details This function initializes the MCP2515 controller by resetting the
 * chip and configuring it.
 */
bool MCP2515Controller::init() {
	if (!configurator.resetChip()) {
		throw std::runtime_error("Failed to reset MCP2515");
	}

	configurator.configureBaudRate();
	configurator.configureTXBuffer();
	configurator.configureRXBuffer();
	configurator.configureFiltersAndMasks();
	configurator.configureInterrupts();
	configurator.setMode(0x00);

	if (!configurator.verifyMode(0x00)) {
		throw std::runtime_error("Failed to set MCP2515 to normal mode");
	}

	return true;
}

/*!
 * @brief Start reading CAN messages.
 * @details This function starts reading CAN messages from the MCP2515.
 */
void MCP2515Controller::processReading() {
	while (!stopReadingFlag) {
		uint16_t frameID;
		std::vector<uint8_t> data;
		// qDebug() << "In loop" << stopReadingFlag;
		try {
			data = configurator.readCANMessage(frameID);
			if (!data.empty()) {
				messageProcessor.processMessage(frameID, data);
			}
		} catch (const std::exception &e) {
			qDebug() << "Error while processing CAN message:" << e.what();
		}

		if (stopReadingFlag) {
				break;
		}

		QThread::msleep(10);
	}
}

/*!
 * @brief Stop reading CAN messages.
 * @details This function stops reading CAN messages from the MCP2515.
 */
void MCP2515Controller::stopReading() { stopReadingFlag = true; }

/*!
 * @brief Send a CAN message.
 * @param frameID The frame ID of the message.
 * @param data The data of the message.
 * @details This function sends a CAN message with the specified frame ID and
 * data.
 */
void MCP2515Controller::setupHandlers() {
	messageProcessor.registerHandler(0x100, [this](const std::vector<uint8_t> &data) {
		if (data.size() == 2) {
			uint16_t rawSpeed = (data[0] << 8) | data[1];
			float speed = rawSpeed / 10.0f;
			// std::cout << "Received Speed data: " << speed << " km/h" << std::endl;
			emit speedUpdated(speed);
		}
	});
	
	messageProcessor.registerHandler(0x300, [this](const std::vector<uint8_t> &data) {
		if (data.size() == 2) {
			uint16_t distance = (data[0] << 8) | data[1];
			//emit distanceUpdated(distance);
			// qDebug() << "Distance from CAN:" << distance;
			if (distance > 20) {
				if (!brakeWarningActive) {
					brakeWarningActive = true;
					QString message = QString("Brake!");
					NotificationManager::instance()->showPersistentNotification(message, NotificationLevel::Warning);
				}
			} else {
				if (brakeWarningActive) {
					brakeWarningActive = false;
					NotificationManager::instance()->clearNotification();
				}
			}
		}
	});
}

/*!
 * @brief Check if the stop reading flag is set.
 * @returns True if the stop reading flag is set, false otherwise.
 * @details This function checks if the stop reading flag is set.
 */
bool MCP2515Controller::isStopReadingFlagSet() const {
	return this->stopReadingFlag;
}
