#include <iostream>
#include <memory>
#include <cstring>
#include <thread>
#include "SPIController.hpp"
#include "MCP2515Configurator.hpp"
#include "MCP2515Controller.hpp"
#include "CanBusManager.hpp"
#include "CANMessageProcessor.hpp"

// Constantes do MCP2515 (para verificação do modo)
namespace MCP2515Constants {
    constexpr uint8_t MODE_NORMAL = 0x00;
}
using namespace MCP2515Constants;

int main() {
    std::cout << "Iniciando a aplicação..." << std::endl;

    // Crie o controlador SPI (ajuste os parâmetros conforme necessário)
    auto spiController = std::make_unique<SPIController>("/dev/spidev0.0");
    
    // Crie o configurador do MCP2515
    auto configurator = std::make_unique<MCP2515Configurator>(*spiController);
    
    // Crie o processador de mensagens
    auto messageProcessor = std::make_shared<CANMessageProcessor>();

    // Crie o controlador do MCP2515
    auto mcp2515Controller = std::make_unique<MCP2515Controller>(
        std::move(spiController),
        std::move(configurator),
        messageProcessor
    );

    // Crie o gestor do CAN Bus
    CanBusManager canBusManager(std::move(mcp2515Controller));

    // Registe os handlers para o gestor
    messageProcessor->registerHandler(0x100, [&](const std::vector<uint8_t>& data) {
        canBusManager.handleSpeed(data);
    });
    
    messageProcessor->registerHandler(0x300, [&](const std::vector<uint8_t>& data) {
        canBusManager.handleRPM(data);
    });

    try {
        // Inicie o gestor do CAN Bus em uma thread separada
        canBusManager.start();

        // Verifique se o chip está em modo normal
        if (canBusManager.getMCP2515Controller()->getConfigurator()->verifyMode(MODE_NORMAL)) {
            std::cout << "MCP2515 está em modo normal. Pronto para receber mensagens." << std::endl;
        } else {
            std::cerr << "ERRO: MCP2515 não está em modo normal." << std::endl;
        }

        std::cout << "Pressione Ctrl+C para terminar a aplicação." << std::endl;
        
        // Mantém o main thread ativo
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        canBusManager.stop();
    }

    return 0;
}