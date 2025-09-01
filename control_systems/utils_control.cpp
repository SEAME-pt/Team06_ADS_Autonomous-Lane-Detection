#include "utils_control.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>

// ---- LaneControl ----

LaneControl::LaneControl(const std::string& model_path, int width, int height, int fps)
    : trt(model_path), cam(width, height, fps) {}

std::unique_ptr<LaneControl> initLaneControl() {
    try {
        auto laneControl = std::make_unique<LaneControl>("../model.engine", 1280, 720, 30);
        laneControl->cam.start();
        std::cout << "Lógica de controlo de faixas inicializada com sucesso." << std::endl;
        return laneControl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Erro ao inicializar o controlo de faixas: " << e.what() << std::endl;
        std::cerr << "Verifique se o ficheiro do modelo existe em '../engines/model.engine'." << std::endl;
        // Aqui podes adicionar mais lógica, como sair do programa ou retornar nullptr
        return nullptr;
    } catch (const std::exception& e) {
        // Captura exceções genéricas para maior robustez
        std::cerr << "Erro inesperado ao inicializar o controlo de faixas: " << e.what() << std::endl;
        return nullptr;
    }
}

// ---- Motores ----
bool initMotors(BackMotors& backMotors) {
    if (!backMotors.init_motors()) {
        std::cerr << "Falha ao inicializar os motores traseiros." << std::endl;
        return false;
    }
    return true;
}

// ---- Servo ----
bool initServo(FServo& servo) {
    try {
        servo.open_i2c_bus();
        if (!servo.init_servo()) {
            throw std::runtime_error("Falha ao inicializar o servo");
        }
        std::cout << "Servo inicializado com sucesso\n";
        servo.set_steering(0);
        return true;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
}

// ---- CAN Bus ----
std::unique_ptr<CanBusManager> initCanBus(std::shared_ptr<CANMessageProcessor>& messageProcessor) {
    auto spiController = std::make_unique<SPIController>("/dev/spidev0.0");
    auto configurator = std::make_unique<MCP2515Configurator>(*spiController);
    messageProcessor = std::make_shared<CANMessageProcessor>();

    auto mcp2515Controller = std::make_unique<MCP2515Controller>(
        std::move(spiController),
        std::move(configurator),
        messageProcessor
    );

    auto canBusManager = std::make_unique<CanBusManager>(std::move(mcp2515Controller));

    messageProcessor->registerHandler(0x100, [&](const std::vector<uint8_t>& data) {
        canBusManager->handleSpeed(data);
    });
    messageProcessor->registerHandler(0x300, [&](const std::vector<uint8_t>& data) {
        canBusManager->handleRPM(data);
    });

    // Novo: Regista default para evitar erros
    messageProcessor->registerDefaultHandler([](const std::vector<uint8_t>& data) {
        std::cout << "Mensagem CAN desconhecida ignorada." << std::endl;
    });

    try {
        canBusManager->start();
        std::cout << "CanBusManager iniciado. A processar mensagens..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL: Falha ao iniciar CanBusManager: " << e.what() << std::endl;
        return nullptr;
    }

    return canBusManager;
}

// ---- ZMQ ----
ZmqPublisher* initZmq(zmq::context_t& context) {
    const std::string ZMQ_HOST = "127.0.0.1";
    const int ZMQ_PORT = 5558;

    try {
        auto* zmq_publisher = new ZmqPublisher(context, ZMQ_HOST, ZMQ_PORT);
        if (!zmq_publisher->isConnected()) {
            std::cerr << "AVISO: Falha ao inicializar o ZMQ Publisher." << std::endl;
        }
        return zmq_publisher;
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL ao iniciar ZMQ Publisher: " << e.what() << std::endl;
        return nullptr;
    }
}

// ---- HUD ----
void drawHUD(cv::Mat& frame,
             double smoothedFPS,
             double delta,
             double v_actual,
             double motor_pwm,
             double offset,
             double psi,
             int steering_angle) 
{
    int y = 20; 
    int step = 20;

    auto put = [&](const std::string& text, cv::Scalar color) {
        cv::putText(frame, text, cv::Point(10, y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        y += step;
    };

    put("FPS: " + std::to_string(static_cast<int>(smoothedFPS)), cv::Scalar(0, 255, 0));
    put("Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg", cv::Scalar(0, 0, 255));
    put("V_actual: " + std::to_string(v_actual) + " m/s", cv::Scalar(0, 0, 255));
    put("Motor: " + std::to_string(motor_pwm), cv::Scalar(0, 255, 0));
    put("Desv Lat: " + std::to_string(offset), cv::Scalar(255, 100, 0));
    put("Psi(rad): " + std::to_string(psi) + 
        " (deg): " + std::to_string(psi * 180.0 / M_PI), cv::Scalar(0, 255, 0));
    put("Steering: " + std::to_string(steering_angle) + " deg", cv::Scalar(200, 0, 0));
}
