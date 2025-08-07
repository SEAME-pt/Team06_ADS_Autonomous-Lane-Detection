/*!
 * @file IMCP2515Controller.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the IMCP2515Controller class.
 * @details This file contains the definition of the IMCP2515Controller class,
 * which is an interface for the MCP2515 CAN controller.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef IMCP2515CONTROLLER_HPP
#define IMCP2515CONTROLLER_HPP

#include <cstdint>
#include <vector>

/*!
 * @brief Interface for the MCP2515 CAN controller.
 * @class IMCP2515Controller
 */
class IMCP2515Controller {
public:
	virtual ~IMCP2515Controller() = default;

	virtual void initialize() = 0;
	virtual void processReading() = 0;
};

#endif // IMCP2515CONTROLLER_HPP