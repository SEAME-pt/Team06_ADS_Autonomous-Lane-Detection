#include "lane.hpp"
#include "nmpc.hpp"
#include "pid.hpp"
#include "scurve.hpp"
#include "MovingAverage.hpp"
#include "SpeedFilter.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>
#include <csignal>
#include <iomanip>

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
// Flag atómica para controlar a execução do loop principal
std::atomic<bool> keep_running{true};

// Handler de sinal para terminar o programa de forma limpa com Ctrl+C
void signalHandler(int signum) {
    std::cout << "\nSinal de interrupcao (" << signum << ") recebido. A terminar a aplicacao..." << std::endl;
    keep_running.store(false);
}

int main() {
    std::signal(SIGINT, signalHandler);

    // --- PARTE 1: INICIALIZAÇÃO DA LÓGICA DE CONTROLO DE FAixas ---
    TensorRTInference trt("../model.engine");
    CSICamera cam(1280, 720, 30);
    cam.start();
    
    NMPCController mpc;

    // --- PARTE 2: INICIALIZAÇÃO DO CONTROLO DE VELOCIDADE E MOTORES ---
    BackMotors backMotors;
    if (!backMotors.init_motors()) {
        std::cerr << "Falha ao inicializar os motores traseiros." << std::endl;
        return 1;
    }
    
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

//#### Inicialização do sistema CAN Bus  ##########
    auto spiController = std::make_unique<SPIController>("/dev/spidev0.0");
    auto configurator = std::make_unique<MCP2515Configurator>(*spiController);
    auto messageProcessor = std::make_shared<CANMessageProcessor>();

    auto mcp2515Controller = std::make_unique<MCP2515Controller>(
        std::move(spiController),
        std::move(configurator),
        messageProcessor
    );

    CanBusManager canBusManager(std::move(mcp2515Controller));

    // Agora, o handler chama a função `handleSpeed` do objeto `canBusManager`
    messageProcessor->registerHandler(0x100, [&](const std::vector<uint8_t>& data) {
        canBusManager.handleSpeed(data);
    });
    
    // Regista o handler para os RPMs, se necessário
    messageProcessor->registerHandler(0x300, [&](const std::vector<uint8_t>& data) {
        canBusManager.handleRPM(data);
    });

    try {
        canBusManager.start();
        std::cout << "CanBusManager iniciado. A processar mensagens..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL: Falha ao iniciar CanBusManager: " << e.what() << std::endl;
        return 1;
    }
// #################################################

// --- INICIALIZAÇÃO DO ZMQ ---------------
    zmq::context_t context(1);
    const std::string ZMQ_HOST = "127.0.0.1";
    const int ZMQ_PORT = 5558;
    ZmqPublisher* zmq_publisher = nullptr;

    try {
        zmq_publisher = new ZmqPublisher(context, ZMQ_HOST, ZMQ_PORT);
        if (!zmq_publisher->isConnected()) {
            std::cerr << "AVISO: Falha ao inicializar o ZMQ Publisher." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL ao iniciar ZMQ Publisher: " << e.what() << std::endl;
        zmq_publisher = nullptr;
    }
//---------------------------------------------

    std::cout << "Pressione Ctrl+C para sair" << std::endl;

    auto lastTime = std::chrono::steady_clock::now();
    double smoothedFPS = 0.0;
    const double alpha = 0.9;
    double last_smoothed_angle = 0.0; // Para filtro low-pass
    double last_delta = 0.0;
    int frameCount = 0;

    bool img_entry = false;

    /*PID INICIALIZATION*/
    double setpoint_velocity = 1.0; // m/s desejados
    PID pid;
    auto pid_last_time = std::chrono::steady_clock::now();
    double motor_pwm = 0.0;
    /********************/
    
    /*S-CURVE INICIALIZATION*/
    double max_steering_vel_deg_s = 100.0;  // Velocidade máx: 30°/s (lento)
    double max_steering_acc_deg_s2 = 300.0; // Aceleração máx: 100°/s²
    double max_steering_jerk_deg_s3 = 600.0; // Jerk máx: 400°/s³
    SCurveProfile steering_profile(max_steering_vel_deg_s, max_steering_acc_deg_s2, max_steering_jerk_deg_s3);
/***************************/

    MovingAverage filter(5);      // média móvel de 5 amostras
    //SpeedFilter filter(0.1);       // 20% valor novo, 80% suavização

    while (keep_running.load()) {
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

        double v_actual = current_speed_ms.load();
        std::cout << "Speed now: " << v_actual << " m/s" << std::endl;
        
        auto pid_now = std::chrono::steady_clock::now();
        double pid_dt = std::chrono::duration<double>(pid_now - pid_last_time).count();
        if (pid_dt >= 0.02) { // 50 ms → 20 Hz
            motor_pwm = pid.compute(setpoint_velocity, v_actual, pid_dt);
            //backMotors.setSpeed(static_cast<int>(motor_pwm)); // aplica PWM de 0 a 100
            pid_last_time = pid_now;
            //std::cout << "Motor Signal: " << motor_pwm << " PWM" << std::endl;
        }
        
        double v_filtered = filter.update(v_actual); // velocidade suavizada
        std::cout << "Velocidade filtrada: " << v_filtered << std::endl;

        double offset = intersect.offset;
        double psi = intersect.psi;
        double delta = last_delta;
        if (!std::isnan(offset) && !std::isnan(psi)) {
            delta = -mpc.computeControl(offset, psi, 0.7);
        } else {
            std::cerr << "AVISO: Offset ou Psi invalido (NaN). Usando delta anterior." << std::endl;
        }

        // Converte delta para graus (ângulo desejado)
        double target_steering_angle = delta * 180.0 / M_PI;
        double smoothed_steering_angle =target_steering_angle;
        // Aplica perfil S-Curve
        //double smoothed_steering_angle = steering_profile.computeNext(target_steering_angle, elapsed);

/*         // Aplica filtro low-pass leve para suavizar entre ciclos (opcional)
        const double alpha_lowpass = 0.8; // 0.8 = equilíbrio entre suavidade e reatividade
        smoothed_steering_angle = alpha_lowpass * smoothed_steering_angle + (1.0 - alpha_lowpass) * last_smoothed_angle;
        last_smoothed_angle = smoothed_steering_angle; */

        // Limita o ângulo final
        int steering_angle = static_cast<int>(smoothed_steering_angle);
        steering_angle = std::max(-40, std::min(40, steering_angle));
        servo.set_steering(steering_angle);
        last_delta = delta; // Atualiza last_delta

        int lane;
        lane = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);
        if (zmq_publisher && zmq_publisher->isConnected()) {
            std::stringstream ss;
            ss << "lane:" << lane;
            zmq_publisher->publishMessage(ss.str());
        }

        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string deltaText = "Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg";
        cv::putText(result, deltaText, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        std::string vActText = "V_actual: " + std::to_string(v_actual) + " m/s";
        cv::putText(result, vActText, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);        
        std::string vFilterText = "V_Filter: " + std::to_string(v_filtered) + " m/s";
        cv::putText(result, vFilterText, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        std::string motorText = "Motor: " + std::to_string(motor_pwm);
        cv::putText(result, motorText, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string desvText = "Desv Lat: " + std::to_string(offset);
        cv::putText(result, desvText, cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 2);
        std::string psiText = "Psi(rad): " + std::to_string(psi) + " (deg): " + std::to_string(psi * 180.0 / M_PI);
        cv::putText(result, psiText, cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string steeringText = "Steering: " + std::to_string(steering_angle) + " deg";
        cv::putText(result, steeringText, cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0), 2);
        std::string smoothedText = "Smoothed Steering: " + std::to_string(smoothed_steering_angle) + " deg";
        cv::putText(result, smoothedText, cv::Point(10, 2000), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 200), 2);

        img_entry = true;
        frameCount++;
        cv::imshow("Lane Detection", result);

        if (cv::waitKey(1) == 'q') {
            keep_running.store(false);
        }
    }

    servo.set_steering(0);
    backMotors.setSpeed(0);
    cam.stop();
    cv::destroyAllWindows();
    
    canBusManager.stop();
    
    if (zmq_publisher) {
        delete zmq_publisher;
        std::cout << "ZMQ Publisher liberado." << std::endl;
    }

    return 0;
}