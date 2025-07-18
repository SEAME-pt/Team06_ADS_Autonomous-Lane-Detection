#include "lane.hpp"
#include "nmpc.hpp"
#include "pid.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include "../FServo/FServo.hpp"
#include "../Control/ControlAssembly.hpp"

void visualize_pixel_markers(cv::Mat& frame) {
    int y_pos = static_cast<int>(0.95 * frame.rows); // 95% da altura (y = 342 para 448px)
    int y_pos_2 = static_cast<int>(0.5 * frame.rows); // 50% da altura (y =  para 448px)
    bool up = true;
    cv::Mat marker_frame = frame.clone(); // Criar uma cópia para não modificar o original
    
    // Desenhar marcadores a cada 10 pixels ao longo da largura
    for (int x = 0; x < frame.cols; x += 10) {
        cv::circle(marker_frame, cv::Point(x, y_pos), 2, cv::Scalar(0, 0, 255), 0);
        cv::circle(marker_frame, cv::Point(x, y_pos_2), 2, cv::Scalar(0, 0, 255), 0);
        if (x % 10 == 0) {
            std::string label = std::to_string(x);
            if (up == true){
                cv::putText(marker_frame, label, cv::Point(x, y_pos - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0, 0, 0), 0);
                cv::putText(marker_frame, label, cv::Point(x, y_pos_2 - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0, 0, 0), 0);
                up = false;
            }
            else{
                cv::putText(marker_frame, label, cv::Point(x, y_pos + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(255, 255, 255), 0);
                cv::putText(marker_frame, label, cv::Point(x, y_pos_2 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(255, 255, 255), 0);
                up = true;
            }
        }
    }
    cv::imshow("Pixel Markers", marker_frame);
}

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(1280, 720, 30);
    cam.start();
    
    // Initialize NMPC: L, dt, N, delta_max, a_max, w_x, w_y, w_psi, w_v, w_delta, w_a
    NMPCController nmpc(0.15, 0.1, 10, 0.698132, 2.0, 0.1, 0.0, 20.0, 1.0, 20.0, 5.0);
    std::vector<double> x0 = {0.0, 0.0, 0.0, 3.0}; // [x, y, psi, v]

    // Initialize PID for velocity control: kp, ki, kd, dt, output_min, output_max
    PIDController pid(1.0, 0.1, 0.05, 0.1, -255.0, 255.0); // Example gains and PWM bounds

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

        //visualize_pixel_markers(result);

        // Update state
        x0[0] = 0.0; // Assume x=0
        x0[1] = intersect.offset_cm; // Lateral offset
        x0[2] = intersect.psi; // Yaw error
        x0[3] = 3.0; // Replace with actual velocity: motor.get_velocity()

        // Execute NMPC
        std::vector<double> control = nmpc.compute_control(x0, laneData, intersect.psi);
        double delta = -control[0]; // Steering angle (rad)
        double v_ref = x0[3] + control[1] * 0.1; // v_ref = current v + a * dt (NMPC output)

        // PID for velocity control
        double v_actual = x0[3]; // Replace with actual velocity measurement
        double motor_signal = pid.compute_control(v_ref, v_actual);

        // Convert delta to degrees and limit
        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-40, std::min(40, steering_angle));

        // Apply controls
        servo.set_steering(steering_angle);
        // motor.set_pwm(motor_signal); // Implement motor control (e.g., PWM)

        // Display info
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string deltaText = "Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg";
        cv::putText(result, deltaText, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        std::string vRefText = "V_ref: " + std::to_string(v_ref) + " m/s";
        cv::putText(result, vRefText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string vActText = "V_actual: " + std::to_string(v_actual) + " m/s";
        cv::putText(result, vActText, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        std::string motorText = "Motor: " + std::to_string(motor_signal);
        cv::putText(result, motorText, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string desvText = "Desv Lat: " + std::to_string(x0[1]);
        cv::putText(result, desvText, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 2);
        std::string psiText = "Psi(rad): " + std::to_string(x0[2]) + " (deg): " + std::to_string(x0[2] * 180.0 / M_PI);
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