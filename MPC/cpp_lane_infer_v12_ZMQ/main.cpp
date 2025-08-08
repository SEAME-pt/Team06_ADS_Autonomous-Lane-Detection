#include "lane.hpp"
#include "nmpc.hpp"
#include "pid.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>

// Componentes do seu sistema para controlo de motor e CAN Bus
#include "../FServo/FServo.hpp"
#include "../Control/ControlAssembly.hpp"
#include "../BackMotors/BackMotors.hpp"
#include "../../MCP2515/CanBusManager.hpp"
#include "../../MCP2515/MCP2515Controller.hpp"
#include "../../MCP2515/SPIController.hpp"
#include "../../MCP2515/MCP2515Configurator.hpp"
#include "../../MCP2515/CANMessageProcessor.hpp"

#include "ZmqPublisher.hpp"
#include <zmq.hpp>
#include <sstream>

// Variável atómica para armazenar a velocidade atual, acessível de forma segura entre threads
std::atomic<double> current_speed_ms{0.0};

// Função que executa o loop de leitura do CAN Bus numa thread separada
void can_bus_loop(CanBusManager& canBusManager) {
    try {
        canBusManager.start();
        std::cout << "CanBusManager iniciado. A processar mensagens..." << std::endl;
        // O loop 'while(true)' interno ao CanBusManager irá gerir as leituras
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (const std::exception& e) {
        std::cerr << "Erro no CanBusManager: " << e.what() << std::endl;
    }
}

int main() {
    // --- PARTE 1: INICIALIZAÇÃO DA LÓGICA DE CONTROLO DE FAixas ---
    TensorRTInference trt("../model.engine");
    CSICamera cam(1280, 720, 30);
    cam.start();
    
    // Inicialização do NMPC
    NMPCController mpc;

    // --- PARTE 2: INICIALIZAÇÃO DO CONTROLO DE VELOCIDADE E MOTORES ---
    
    // Inicializar os motores traseiros. Usamos BackMotors no lugar de ControlAssembly
    BackMotors backMotors;
    if (!backMotors.init_motors()) {
        std::cerr << "Falha ao inicializar os motores traseiros." << std::endl;
        return 1;
    }

    // Configurar o PID para controlo de velocidade (ajuste estes valores)
    // Velocidade máxima de 8 m/s para frente e trás
    double kp = 50.0, ki = 0.5, kd = 0.1; // Ganhos de exemplo do PID
    double dt = 0.1; // Intervalo de tempo do loop de controlo (100ms)
    PIDController pid(kp, ki, kd, dt, -100.0, 100.0); // A saída do PID é entre -100 e 100 para o setSpeed()
    double v_ideal = 2.0; // Velocidade ideal desejada (em m/s)
    
    // Inicialização do servo
    FServo servo;
    try {
        servo.open_i2c_bus();
        if (!servo.init_servo()) {
            throw std::runtime_error("Falha ao inicializar o servo");
        }
        std::cout << "Servo inicializado com sucesso\n";
        servo.set_steering(0);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // Inicialização do sistema CAN Bus
    auto spiController = std::make_unique<SPIController>("/dev/spidev0.0");
    auto configurator = std::make_unique<MCP2515Configurator>(*spiController);
    auto messageProcessor = std::make_shared<CANMessageProcessor>();

    auto mcp2515Controller = std::make_unique<MCP2515Controller>(
        std::move(spiController),
        std::move(configurator),
        messageProcessor
    );

    CanBusManager canBusManager(std::move(mcp2515Controller));

    // Regista o handler para a mensagem de velocidade CAN (ID 0x100)
    // A função lambda é executada sempre que uma mensagem com ID 0x100 é recebida
    messageProcessor->registerHandler(0x100, [&](const std::vector<uint8_t>& data) {
        if (data.size() == 2) {
            uint16_t rawSpeed = (data[0] << 8) | data[1];
            double speed_kmh = static_cast<double>(rawSpeed) / 10.0;
            double speed_ms = speed_kmh * 1000.0 / 3600.0;
            current_speed_ms.store(speed_ms);
            std::cout << "Velocidade recebida: " << speed_ms << " m/s" << std::endl;
        }
    });

    // Inicia o CanBusManager numa thread separada para não bloquear o loop principal
    std::thread canThread(can_bus_loop, std::ref(canBusManager));
    canThread.detach();

    // --- INICIALIZAÇÃO DO ZMQ ---
    zmq::context_t context(1);
    const std::string ZMQ_HOST = "127.0.0.1";
    const int ZMQ_PORT = 5558;
    ZmqPublisher* zmq_publisher = nullptr;

    try {
        zmq_publisher = new ZmqPublisher(context, ZMQ_HOST, ZMQ_PORT);
        if (!zmq_publisher->isConnected()) {
            std::cerr << "AVISO: Falha ao inicializar o ZMQ Publisher. O middleware pode nao receber dados." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL ao iniciar ZMQ Publisher: " << e.what() << std::endl;
        zmq_publisher = nullptr;
    }

    std::cout << "Pressione 'q' para sair" << std::endl;

    auto lastTime = std::chrono::steady_clock::now();
    double smoothedFPS = 0.0;
    const double alpha = 0.9;
    int frameCount = 0;

    double last_delta = 0.0;

    // --- PARTE 3: LOOP PRINCIPAL DE CONTROLO ---
    while (true) {
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;

        auto currentTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        double currentFPS = 1.0 / elapsed;
        smoothedFPS = smoothedFPS == 0.0 ? currentFPS : alpha * smoothedFPS + (1.0 - alpha) * currentFPS;

        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        LaneData laneData;
        LineIntersect intersect;
        auto result = postprocess(outputs.data(), frame, medianPoints, laneData, intersect);

        // AQUI ESTÁ A ALTERAÇÃO CHAVE: lê a velocidade do sensor CAN
        double v_actual = current_speed_ms.load();

        double offset = intersect.offset;
        double psi = intersect.psi;
        double delta = last_delta;
        if (!std::isnan(offset) && !std::isnan(psi)) {
            delta = -mpc.computeControl(offset, psi, v_actual);
        } else {
            std::cerr << "AVISO: Offset ou Psi invalido (NaN). Usando delta anterior." << std::endl;
        }

        // Usa a velocidade real para o controlo PID
        double motor_signal = pid.compute_control(v_ideal, v_actual);
        backMotors.setSpeed(static_cast<int>(motor_signal));

        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-40, std::min(40, steering_angle));

        servo.set_steering(steering_angle);
        last_delta = delta;

        // ... o seu código de exibição permanece o mesmo ...
        std::cout << "Offset: " << offset << " m, Psi: " << psi * 180.0 / M_PI << " deg, Delta: " << delta * 180.0 / M_PI << " deg" << std::endl;
        
        int lane;
        lane = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);

        // Envio de dados ZMQ
        if (zmq_publisher && zmq_publisher->isConnected()) {
            std::stringstream ss;
            ss << "lane:" << lane;
            zmq_publisher->publishMessage(ss.str());
        }

        // ... o seu código de exibição permanece o mesmo ...
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string deltaText = "Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg";
        cv::putText(result, deltaText, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        std::string vRefText = "V_ideal: " + std::to_string(v_ideal) + " m/s";
        cv::putText(result, vRefText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string vActText = "V_actual: " + std::to_string(v_actual) + " m/s";
        cv::putText(result, vActText, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        std::string motorText = "Motor: " + std::to_string(motor_signal);
        cv::putText(result, motorText, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string desvText = "Desv Lat: " + std::to_string(offset);
        cv::putText(result, desvText, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 2);
        std::string psiText = "Psi(rad): " + std::to_string(psi) + " (deg): " + std::to_string(psi * 180.0 / M_PI);
        cv::putText(result, psiText, cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string steeringText = "Steering: " + std::to_string(steering_angle) + " deg";
        cv::putText(result, steeringText, cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0), 2);

        int centerX = result.cols / 2;
        int centerY = result.rows;
        int lineLength = 300;
        cv::Point lineStart(centerX, centerY - lineLength);
        cv::Point lineEnd(centerX, centerY);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 200, 200), 2);

        cv::Point mediumStart(0, centerY / 2);
        cv::Point mediumEnd(centerX * 2, centerY / 2);
        cv::line(result, mediumStart, mediumEnd, cv::Scalar(250, 200, 200), 2);

        frameCount++;
        cv::imshow("Lane Detection", result);

        if (cv::waitKey(1) == 'q') break;
    }

    // Stop servo and motors
    servo.set_steering(0);
    backMotors.setSpeed(0);
    cam.stop();
    cv::destroyAllWindows();
    canBusManager.stop();
    canThread.join();

    // --- LIBERAÇÃO DO RECURSO ZMQ ---
    if (zmq_publisher) {
        delete zmq_publisher;
        std::cout << "ZMQ Publisher liberado." << std::endl;
    }

    return 0;
}