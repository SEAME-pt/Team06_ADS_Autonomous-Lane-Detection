#include "lane.hpp"
#include "nmpc.hpp"
#include "pid.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include "../FServo/FServo.hpp"
#include "../Control/ControlAssembly.hpp"

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(1280, 720, 30);
    cam.start();
    
    // Initialize NMPC
    NMPCController mpc;

    // Initialize PID for velocity control: kp, ki, kd, dt, output_min, output_max
    PIDController pid(1.0, 0.1, 0.05, 0.1, -255.0, 255.0); // Example gains and PWM bounds
    double v_ideal = 1.0; // Velocidade ideal fixa (m/s)

    // Initialize servo
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

    // Initialize motor controller
    // ControlAssembly motor; // Uncomment and configure
    // motor.set_acceleration(0.0); // Initialize to zero

    std::cout << "Pressione 'q' para sair" << std::endl;

    auto lastTime = std::chrono::steady_clock::now();
    double smoothedFPS = 0.0;
    const double alpha = 0.9;
    int frameCount = 0;

    // Initialize state
    double last_delta = 0.0; // Store last delta for state update

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

        // Update state (replace with actual odometry if available)
        double v_actual = 1.0; // MUITO IMPORTANTE: Substitua por uma leitura real da velocidade do seu robô.
                               // Se v_actual for sempre 3.0, o MPC não terá a velocidade real do veículo,
                               // o que pode levar a um controle impreciso.
        
        // Check for invalid inputs
        double offset = intersect.offset;
        double psi = intersect.psi;
        double delta = last_delta; // Default to last delta if inputs are invalid
        if (!std::isnan(offset) && !std::isnan(psi)) {
            // Execute NMPC
            // O parâmetro 'theta' (current_theta_rad) foi removido da chamada.
            delta = -mpc.computeControl(offset, psi, v_actual);
        } else {
            std::cerr << "AVISO: Offset ou Psi inválido (NaN). Usando delta anterior." << std::endl;
        }

        // PID for velocity control
        double motor_signal = pid.compute_control(v_ideal, v_actual);

        // Convert delta to degrees and limit
        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-40, std::min(40, steering_angle));

        // Apply controls
        servo.set_steering(steering_angle);
        last_delta = delta; // Store delta for next state update (CRUCIAL para o MPC)

        // Log para depuração
        std::cout << "Offset: " << offset << " m, Psi: " << psi * 180.0 / M_PI << " deg, Delta: " << delta * 180.0 / M_PI << " deg" << std::endl;
        // motor.set_pwm(motor_signal); // Implement motor control (e.g., PWM)

        // Display info
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
        std::string desvText = "Desv Lat: " + std::to_string(offset); // Exibir em m
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
    // motor.set_pwm(0.0); // Implement
    cam.stop();
    cv::destroyAllWindows();
    return 0;
}