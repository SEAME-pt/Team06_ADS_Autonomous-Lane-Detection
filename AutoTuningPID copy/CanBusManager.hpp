/*!
 * @file CanBusManager.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the CanBusManager class.
 * @details This file contains the definition of the CanBusManager class,
 * which manages the CAN bus communication.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef CANBUSMANAGER_HPP
#define CANBUSMANAGER_HPP

#include <memory>
#include <thread>
#include <vector>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include "IMCP2515Controller.hpp"
#include "MCP2515Controller.hpp" // Novo include para o cast

/*!
 * @brief Class for managing the CAN bus communication.
 * @class CanBusManager
 */
class CanBusManager {
public:
    explicit CanBusManager(std::unique_ptr<IMCP2515Controller> controller);
    ~CanBusManager();

    void start();
    void stop();
    void handleSpeed(const std::vector<uint8_t>& data);
    void handleRPM(const std::vector<uint8_t>& data);

    // Função de acesso para o controlador MCP2515
    MCP2515Controller* getMCP2515Controller() const {
        return static_cast<MCP2515Controller*>(mcp2515Controller.get());
    }
    
private:
    std::unique_ptr<IMCP2515Controller> mcp2515Controller;
    std::thread workerThread;
    bool running = false;
    std::mutex mtx;
    std::condition_variable cv;
};

#endif // CANBUSMANAGER_HPP