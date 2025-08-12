#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstring> // Added for memcpy
#include "SPIController.hpp"
#include "MCP2515Configurator.hpp"
#include "MCP2515Controller.hpp"
#include "CanBusManager.hpp"
#include "CANMessageProcessor.hpp"
#include "SpeedPIDController.hpp"
#include "SpeedPIDTuner.hpp"

const float PWM_MIN = 0.0f; // PWM mínimo
const float PWM_MAX = 100.0f; // PWM máximo

namespace MCP2515Constants {
    constexpr uint8_t MODE_NORMAL = 0x00;
}
using namespace MCP2515Constants;

int main() {
    std::cout << "Iniciando a aplicação..." << std::endl;

    // Crie o controlador SPI (ajuste o caminho do dispositivo conforme necessário)
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

    // Parâmetros do PID
    float dt = 0.03f; // Passo de tempo (30 ms)
    float sim_time = 10.0f; // Tempo de simulação/teste
    float v_target = 2.0f; // Velocidade alvo (2 m/s)

    // Registre os handlers para o gestor
    messageProcessor->registerHandler(0x100, [&](const std::vector<uint8_t>& data) {
        canBusManager.handleSpeed(data);
    });
    
    messageProcessor->registerHandler(0x300, [&](const std::vector<uint8_t>& data) {
        canBusManager.handleRPM(data);
    });

    try {
        // Inicie o gestor do CAN Bus
        canBusManager.start();

        // Verifique se o chip está em modo normal
        if (canBusManager.getMCP2515Controller()->getConfigurator()->verifyMode(MODE_NORMAL)) {
            std::cout << "MCP2515 está em modo normal. Pronto para receber mensagens." << std::endl;
        } else {
            std::cerr << "ERRO: MCP2515 não está em modo normal." << std::endl;
            return 1;
        }

        // Inicie o auto-tuning do PID usando dados reais do CAN Bus
        std::cout << "Iniciando auto-tuning do PID com dados reais..." << std::endl;
        auto [kp, ki, kd] = auto_tune_pid(dt, sim_time, v_target, true, &canBusManager);

        std::cout << "Melhores ganhos PID encontrados:\n";
        std::cout << "Kp = " << kp << "\n";
        std::cout << "Ki = " << ki << "\n";
        std::cout << "Kd = " << kd << "\n";

        // Inicialize o controlador PID com os ganhos encontrados
        SpeedPIDController pid(kp, ki, kd, PWM_MIN, PWM_MAX);
        pid.reset();

        // Loop de controle contínuo com PID
        std::cout << "Iniciando loop de controle PID..." << std::endl;
        while (true) {
            float v_current = canBusManager.getCurrentSpeed(); // Obter velocidade real via CAN
            float pwm_output = pid.update(v_current, v_target, dt); // Calcular saída PWM

            std::cout << "Velocidade atual: " << v_current << " m/s, PWM calculado: " << pwm_output << std::endl;

            // Enviar o comando PWM via CAN Bus
            std::vector<uint8_t> pwm_data(4);
            memcpy(pwm_data.data(), &pwm_output, sizeof(float)); // Converter float para bytes
            canBusManager.sendCANMessage(0x200, pwm_data);

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(dt * 1000))); // Sincronizar com dt
        }

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        canBusManager.stop();
    }

    return 0;
}