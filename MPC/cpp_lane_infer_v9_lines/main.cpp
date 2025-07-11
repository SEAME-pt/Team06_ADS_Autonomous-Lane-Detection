#include "lane.hpp"
#include "nmpc.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include "../FServo/FServo.hpp"
#include "../Control/ControlAssembly.hpp"


int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(448, 448, 15);
    cam.start();
    
    // Inicializar NMPC   L,  dt, N, delta_max, w_x, w_y, w_psi, w_delta
    NMPCController nmpc(0.15, 0.1, 10, 0.524, 0.1, 0.0, 20.0, 10.0);
    std::vector<double> x0 = {0.0, 0.0, 0.0}; // [x, y, psi]
    
    // Inicializar servo
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
    
    // Inicializar joystick
    //ControlAssembly controlAssembly;

    std::cout << "Pressione 'q' para sair" << std::endl;
    
    auto lastTime = std::chrono::steady_clock::now();
    double smoothedFPS = 0.0;
    const double alpha = 0.9;
    int frameCount = 0;

    while (true) {
//std::cout << "Entrou while " << std::endl;
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;
        auto currentTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        double currentFPS = 1.0 / elapsed;
        smoothedFPS = smoothedFPS == 0.0 ? currentFPS : alpha * smoothedFPS + (1.0 - alpha) * currentFPS;
        
        std::vector<float> input = preprocess_frame(frame);
//std::cout << "preprocess frame check " << std::endl;

        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        LaneData laneData;
        LineIntersect intersect;
        auto result = postprocess(outputs.data(), frame, medianPoints, laneData, intersect);
//std::cout << "postprocess check " << std::endl;

        // Visualizar marcadores de pixels
        //visualize_pixel_markers(result);

        // Atualizar estado com psi do intersect
        x0[0] = 0.0; // Assume x=0 (posição longitudinal inicial)
        x0[1] = intersect.offset_cm; // y é o desvio lateral
        x0[2] = intersect.psi; // Atualiza psi com o erro de yaw

        // Executar NMPC
        std::vector<double> control = nmpc.compute_control(x0, laneData, intersect.psi);
        double delta = - control[0]; // rad
        
        // Converter delta de radianos para graus e limitar
        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-35, std::min(35, steering_angle));
        
        servo.set_steering(steering_angle);
        
         // Exibir FPS e controles
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::string deltaText = "Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg";
        cv::putText(result, deltaText, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        std::string desvText = "Desv Lat: " + std::to_string(x0[1]);
        cv::putText(result, desvText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        std::string psiText = "Psi(rad): " + std::to_string(x0[2]) + " (deg): " + std::to_string(x0[2] * 180.0 / M_PI);
        cv::putText(result, psiText, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        std::string steeringText = "Steering: " + std::to_string(steering_angle) + " deg";
        cv::putText(result, steeringText, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        
        //std::cout << "janela check " << std::endl;
        
        int centerX = result.cols / 2;
        int centerY = result.rows;
        int lineLength = 300;
        cv::Point lineStart(centerX, centerY - lineLength);
        cv::Point lineEnd(centerX, centerY - 20);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 200, 200), 2);

        //deBug(delta, laneData, intersect, frameCount, medianPoints);
        frameCount++;
        cv::imshow("Lane Detection", result);
//std::cout << "imshow check " << std::endl;

        if (cv::waitKey(1) == 'q') break;

        frame.release();
        result.release();
    }

    // Parar servo e motores
    servo.set_steering(0);
    cam.stop();
    cv::destroyAllWindows();
    return 0;
}
