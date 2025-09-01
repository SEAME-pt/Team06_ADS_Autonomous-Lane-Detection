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

#include "ISPIController.hpp"
#include "IMCP2515Controller.hpp"
#include "MCP2515Configurator.hpp"
#include "CANMessageProcessor.hpp"
#include <memory>
#include <cstdint>
#include <vector>

/*!
 * @brief Concrete class for controlling the MCP2515 CAN controller.
 * @class MCP2515Controller
 */
class MCP2515Controller : public IMCP2515Controller {
public:
	MCP2515Controller(std::unique_ptr<ISPIController> spiController,
											 std::unique_ptr<MCP2515Configurator> configurator,
											 std::shared_ptr<CANMessageProcessor> messageProcessor);
	~MCP2515Controller() override = default;

	void initialize() override;
	void processReading() override;

	// Função para aceder ao configurador para fins de verificação
	MCP2515Configurator* getConfigurator() const {
        return config.get();
    }

private:
	std::unique_ptr<ISPIController> spi;
	std::unique_ptr<MCP2515Configurator> config;
	std::shared_ptr<CANMessageProcessor> processor;

	void readMessage(uint8_t address, uint16_t &frameID, std::vector<uint8_t> &data);
};

#endif // MCP2515CONTROLLER_HPP