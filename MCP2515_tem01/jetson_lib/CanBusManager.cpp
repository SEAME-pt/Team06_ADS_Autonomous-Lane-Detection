/*!
 * @file CanBusManager.cpp
 * @brief Implementation of the CanBusManager class.
 * @version 0.1
 * @date 2025-01-29
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains the implementation of the CanBusManager class,
 * which manages the CAN bus communication.
 *
 * @note This class uses the MCP2515Controller for CAN bus communication.
 *
 * @warning Ensure that the MCP2515 controller is properly implemented.
 *
 * @see CanBusManager.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "CanBusManager.hpp"
#include <QDebug>
#include "MCP2515Controller.hpp"

/*!
 * @brief Construct a new CanBusManager::CanBusManager object
 *
 * @param spi_device The SPI device to use for communication.
 * @param parent The parent QObject.
 *
 * @details This constructor initializes the CanBusManager with a specified SPI
 * device and sets up the MCP2515 controller.
 */
CanBusManager::CanBusManager(const std::string &spi_device, QObject *parent)
	: QObject(parent)
{
	m_controller = new MCP2515Controller(spi_device);
	ownsMCP2515Controller = true;
	connectSignals();
}

/*!
 * @brief Construct a new CanBusManager::CanBusManager object
 *
 * @param controller The MCP2515 controller to use.
 * @param parent The parent QObject.
 *
 * @details This constructor initializes the CanBusManager with an existing
 * MCP2515 controller.
 */
CanBusManager::CanBusManager(IMCP2515Controller *controller, QObject *parent)
	: QObject(parent)
	, m_controller(controller)
{
	ownsMCP2515Controller = false;
	connectSignals();
}

/*!
 * @brief Destroy the CanBusManager::CanBusManager object
 *
 * @details Cleans up the resources used by the CanBusManager, including
 * stopping the reading thread and deleting the controller if owned.
 */
CanBusManager::~CanBusManager()
{
	if (m_thread) {
		m_controller->stopReading();
		m_thread->disconnect();
		m_thread->quit();
		m_thread->wait();

		delete m_thread;
		m_thread = nullptr;
	}

	if (ownsMCP2515Controller) {
		delete m_controller;
	}
}

void CanBusManager::onSpeedUpdated(float newSpeed)
{
	emit speedUpdated(newSpeed);

	// std::cout << "Speed updated: " << newSpeed << std::endl;

	Publisher::instance(CAR_SPEED_PORT)->publishCarSpeed(CAR_SPEED_TOPIC, newSpeed); // Publish speed to ZeroMQ publisher
}

/*!
 * @brief Connects the signals from the MCP2515 controller to the CanBusManager
 * slots.
 *
 * @details This method sets up the connections between the signals emitted by
 * the MCP2515 controller and the corresponding slots in the CanBusManager.
 */
void CanBusManager::connectSignals()
{
	connect(m_controller, &IMCP2515Controller::speedUpdated, this, &CanBusManager::onSpeedUpdated);
	connect(m_controller, &IMCP2515Controller::rpmUpdated, this, &CanBusManager::rpmUpdated);
}

/*!
 * @brief Initializes the CanBusManager.
 *
 * @details Initializes the MCP2515 controller and starts the reading thread.
 *
 * @returns true if initialization is successful, false otherwise.
 */
bool CanBusManager::initialize()
{
	if (!m_controller->init()) {
		return false;
	}

	m_thread = new QThread(this);
	m_controller->moveToThread(m_thread);

	connect(m_thread, &QThread::started, m_controller, &IMCP2515Controller::processReading);
	connect(m_thread, &QThread::finished, m_controller, &QObject::deleteLater);
	connect(m_thread, &QThread::finished, m_thread, &QObject::deleteLater);

	m_thread->start();
	return true;
}
