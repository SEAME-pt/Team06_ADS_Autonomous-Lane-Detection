/*!
 * @file IMCP2515Controller.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the IMCP2515Controller class.
 * @details This file contains the definition of the IMCP2515Controller class,
 * which is responsible for controlling the MCP2515 CAN controller.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef IMCP2515CONTROLLER_HPP
#define IMCP2515CONTROLLER_HPP

#include <QObject>

/*!
 * @brief Interface for the MCP2515 CAN controller.
 * @class IMCP2515Controller inherits from QObject
 */
class IMCP2515Controller : public QObject {
	Q_OBJECT
public:
	virtual ~IMCP2515Controller() = default;
	virtual bool init() = 0;
	virtual void processReading() = 0;
	virtual void stopReading() = 0;
	virtual bool isStopReadingFlagSet() const = 0;

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
};

#endif // IMCP2515CONTROLLER_HPP
