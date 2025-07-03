#include "lane.hpp"
#include "nmpc.hpp"
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

void deBug(double delta, LaneData laneData, LineIntersect intersect, int frameCount, std::vector<cv::Point> medianPoints){

    if (intersect.valid && frameCount % 20 == 0) {
/*         std::cout << "Left Top: (" << intersect.xl_t.x << ", " << intersect.xl_t.y << ")" << std::endl;
        std::cout << "Left Bottom: (" << intersect.xl_b.x << ", " << intersect.xl_b.y << ")" << std::endl;
        std::cout << "Right Top: (" << intersect.xr_t.x << ", " << intersect.xr_t.y << ")" << std::endl;
        std::cout << "Right Bottom: (" << intersect.xr_b.x << ", " << intersect.xr_b.y << ")" << std::endl; */

        std::cout << "offset_cm: " << intersect.offset_cm << std::endl;
        std::cout << "pixels on top: " << intersect.x_px_t << std::endl;
        std::cout << "pixels on bottom: " << intersect.x_px_b << std::endl;
        std::cout << "s(y1): " << intersect.scaleFactor_t << std::endl;
        std::cout << "s(y2): " << intersect.scaleFactor_b << std::endl;
        std::cout << "var a: " << intersect.var_a << std::endl;
        std::cout << "var b: " << intersect.var_b << std::endl;
        std::cout << "Psi: " << intersect.psi << std::endl;
        std::cout << "median points: " << medianPoints << std::endl;

    }
        
/*     if (intersect.valid && frameCount % 20 == 0) {
        std::cout << "ratio: " << intersect.ratio_top << std::endl;
        //std::cout << "xs_b: " << intersect.xs_b << std::endl;
        std::cout << "slope: " << intersect.slope << std::endl;
        std::cout << "Psi: " << intersect.psi << std::endl;
        std::cout << "Psi: " << intersect.psi * 180.0 / M_PI << " deg" << std::endl ;
        std::cout << " Delta: " + std::to_string(delta * 180.0 / M_PI) + "deg" << std::endl << std::endl;
    } */
    /*if (laneData.valid && frameCount % 20 == 0) {
        for (int i = 0; i < laneData.num_points; ++i) {
            std::cout << "  Ponto " << i << ": (" << laneData.points[i].x << ", " << laneData.points[i].y << ")" << std::endl;
        }
    } */
}

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(448, 448, 15);
    cam.start();
    
    // Inicializar NMPC
    NMPCController nmpc(0.15, 0.1, 10, 0.524, 1.0, 5.0, 20.0, 10.0); // L, dt, N, delta_max, w_x, w_y, w_psi, w_delta
    std::vector<double> x0 = {0.0, 0.0, 0.0}; // Estado inicial: [x, y, psi]
    
    // Inicializar servo
    FServo servo;
    try {
        servo.open_i2c_bus();
        if (!servo.init_servo()) {
            throw std::runtime_error("Falha ao inicializar o servo");
        }

        std::cout << "Definindo ângulo de direção para -45 graus...\n";
        servo.set_steering(-45);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::cout << "Definindo ângulo de direção para +45 graus...\n";
        servo.set_steering(45);
        std::this_thread::sleep_for(std::chrono::seconds(2));

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
        auto result = postprocess(outputs[0].data(), outputs[1].data(), frame, medianPoints, laneData, intersect);
        

        // Visualizar marcadores de pixels
        //visualize_pixel_markers(result);

        // Atualizar estado com psi do intersect
        if (intersect.valid) {
            x0[0] = 0.0; // Assume x=0 (posição longitudinal inicial)
            x0[1] = intersect.offset_cm; // y é o desvio lateral
            x0[2] = intersect.psi; // Atualiza psi com o erro de yaw
        }

        // Executar NMPC
        std::vector<double> control = nmpc.compute_control(x0, laneData, intersect.psi);
        double delta = control[0]; // rad
        
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
        
        int centerX = result.cols / 2 + 22.5;
        int centerY = result.rows;
        int lineLength = 200;
        cv::Point lineStart(centerX, centerY - lineLength);
        cv::Point lineEnd(centerX, centerY - 20);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 200, 200), 2);

        deBug(delta, laneData, intersect, frameCount, medianPoints);
        frameCount++;
        cv::imshow("Lane Detection", result);
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
