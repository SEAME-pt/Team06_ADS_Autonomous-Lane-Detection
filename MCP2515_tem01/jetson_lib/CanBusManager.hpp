/*!
 * @file CanBusManager.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the CanBusManager class.
 * @details This file contains the definition of the CanBusManager class, which
 * is responsible for managing the CAN bus communication.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef CANBUSMANAGER_HPP
#define CANBUSMANAGER_HPP

#include "IMCP2515Controller.hpp"
#include <QObject>
#include <QThread>
#include "../../../ZeroMQ/Publisher.hpp"

/*!
 * @brief Class that manages the CAN bus communication.
 * @class CanBusManager inherits from QObject
 */
class CanBusManager : public QObject {
	Q_OBJECT
public:
	explicit CanBusManager(const std::string &spi_device,
												 QObject *parent = nullptr);
	CanBusManager(IMCP2515Controller *controller, QObject *parent = nullptr);
	~CanBusManager();
	bool initialize();

	QThread *getThread() const { return m_thread; }

	signals:
	/*!
	 * @brief Signal emitted when the speed is updated.
	 * @param newSpeed The new speed value.
	 */
	void speedUpdated(float newSpeed);
	/*!
	 * @brief Signal emitted when the RPM is updated.
	 * @param newRpm The new RPM value.
	 */
	void rpmUpdated(int newRpm);

private slots:
	/*!
	 * @brief Slot to handle the speed update.
	 * @param newSpeed The new speed value.
	 */
	void onSpeedUpdated(float newSpeed);

private:
	/*! @brief Pointer to the IMCP2515Controller object. */
	IMCP2515Controller *m_controller = nullptr;
	/*! @brief Pointer to the QThread object. */
	QThread *m_thread = nullptr;
	/*! @brief Flag to indicate if the MCP2515 controller is owned by the
	 * CanBusManager. */
	bool ownsMCP2515Controller = false;
	/*! @brief Method to connect signals. */
	void connectSignals();
};

#endif // CANBUSMANAGER_HPP
