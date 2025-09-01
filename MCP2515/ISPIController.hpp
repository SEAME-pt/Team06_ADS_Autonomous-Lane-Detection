/*!
 * @file ISPIController.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief Definition of the ISPIController class.
 * @details This file contains the definition of the ISPIController class,
 * which is an interface for the SPI bus communication.
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef ISPICONTROLLER_HPP
#define ISPICONTROLLER_HPP

#include <cstdint>
#include <vector>

/*!
 * @brief Interface for the SPI bus controller.
 * @class ISPIController
 */
class ISPIController {
public:
    virtual ~ISPIController() = default;

    virtual void openDevice() = 0;
    virtual void closeDevice() = 0;
    virtual void spiTransfer(uint8_t* txData, size_t size) = 0;
    virtual uint8_t readByte() = 0;
    virtual void writeByte(uint8_t data) = 0;
};

#endif // ISPICONTROLLER_HPP