/*!
 * @file MCP2515Controller.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the MCP2515Controller class.
 * @details This file contains the definition of the MCP2515Controller class,
 * which is responsible for controlling the MCP2515 CAN controller.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef MCP2515CONTROLLER_HPP
#define MCP2515CONTROLLER_HPP

#include "CANMessageProcessor.hpp"
#include "IMCP2515Controller.hpp"
#include "ISPIController.hpp"
#include "MCP2515Configurator.hpp"
#include <QObject>
#include <string>
#include <iostream>

/*!
 * @brief Class that controls the MCP2515 CAN controller.
 * @class MCP2515Controller inherits from IMCP2515Controller
 */
class MCP2515Controller : public IMCP2515Controller {
	Q_OBJECT
public:
	explicit MCP2515Controller(const std::string &spiDevice);
	MCP2515Controller(const std::string &spiDevice,
										ISPIController &spiController);

	~MCP2515Controller() override;

	bool init() override;
	void processReading() override;
	void stopReading() override;

	CANMessageProcessor &getMessageProcessor() { return messageProcessor; }
	bool isStopReadingFlagSet() const override;

private:
	/*! @brief Pointer to the ISPIController object. */
	ISPIController *spiController;
	/*! @brief MCP2515Configurator object. */
	MCP2515Configurator configurator;
	/*! @brief CANMessageProcessor object. */
	CANMessageProcessor messageProcessor;
	/*! @brief Flag to indicate if the reading process should stop. */
	bool stopReadingFlag = false;
	/*! @brief Flag to indicate if the SPI controller is owned by the
	 * MCP2515Controller. */
	bool ownsSPIController = false;

	void setupHandlers();
	bool brakeWarningActive = false;
};

#endif // MCP2515CONTROLLER_HPP
