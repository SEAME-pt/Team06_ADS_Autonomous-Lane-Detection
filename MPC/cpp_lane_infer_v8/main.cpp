#include "lane.hpp"
#include "nmpc.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include "../FServo/FServo.hpp"


void deBug(LineIntersect intersect, int frameCount){

/*     if (intersect.valid && frameCount % 20 == 0) {
        std::cout << "Left Top: (" << intersect.left_top.x << ", " << intersect.left_top.y << ")" << std::endl;
        std::cout << "Left Bottom: (" << intersect.left_bottom.x << ", " << intersect.left_bottom.y << ")" << std::endl;
        std::cout << "Right Top: (" << intersect.right_top.x << ", " << intersect.right_top.y << ")" << std::endl;
        std::cout << "Right Bottom: (" << intersect.right_bottom.x << ", " << intersect.right_bottom.y << ")" << std::endl;
    } */
        
    if (intersect.valid && frameCount % 20 == 0) {
    
        std::cout << "ratio: " << intersect.ratio_top << std::endl;
        std::cout << "xs_b: " << intersect.xs_b << std::endl;
        std::cout << "slope: " << intersect.slope << std::endl;
        std::cout << "Psi: " << intersect.psi << std::endl;
        std::cout << "Psi_rad: " << intersect.psi * 180.0 / M_PI << " deg" << std::endl << std::endl;
    }
    /*         if (laneData.valid && frameCount % 20 == 0) {
    for (int i = 0; i < laneData.num_points; ++i) {
        std::cout << "  Ponto " << i << ": (" << laneData.points[i].x << ", " << laneData.points[i].y << ")" << std::endl;
        std::cout << "x: " << x_ref[i][1] << ", y: " << x_ref[i][0] << ", psi_ref: " << x_ref[i][2] << std::endl;
    }
    //std::cout << "  Ponto: (" << laneData.points[9].x << ", " << laneData.points[9].y << ")" << std::endl;
    std::cout << "  Delta: " + std::to_string(delta * 180.0 / M_PI) + "deg" << std::endl;
    std::cout << " ******************************** X   " << x0[0] << ")" << std::endl;
    std::cout << " ******************************** Y   " << x0[1] << ")" << std::endl;
    std::cout << " ******************************** Psi " << x0[2] << ")" << std::endl;
    std::cout << " ******************************** Vel " << x0[3] << ")" << std::endl;
    std::cout << " ******************************** Acel " << a << ")" << std::endl;
    } */
}

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(448, 448, 15);
    cam.start();
    
    // Inicializar NMPC
    NMPCController nmpc(0.15, 0.1, 10, 0.349, 1.0, 1.0, 0.1, 0.01); // L, dt, N, delta_max, w_x, w_y, w_psi, w_delta
    std::vector<double> x0 = {0.0, 0.0, 0.0}; // Estado inicial: [x, y, psi]
    
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
        
        // Executar NMPC
        std::vector<double> control = nmpc.compute_control(x0, laneData, intersect.psi);
        double delta = control[0]; // rad
        double a = 0; //control[1];     // m/s² (não usado por enquanto)
        
        // Converter delta de radianos para graus e limitar
        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-20, std::min(20, steering_angle));
        
        
        servo.set_steering(steering_angle);
        
        // Exibir FPS e controles
        /*std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2); */
        std::string deltaText = "Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg";
        cv::putText(result, deltaText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        std::string aText = "Accel: " + std::to_string(a) + " m/s^2";
        cv::putText(result, aText, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        std::string psiText = "Psi(rad): " + std::to_string(x0[2]) + " (deg): " + std::to_string(x0[2] * 180.0 / M_PI);
        cv::putText(result, psiText, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        std::string velText = "Vel : " + std::to_string(x0[3]) ;
        cv::putText(result, velText, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        
        int centerX = result.cols / 2;
        int centerY = result.rows;
        int lineLength = 200;
        cv::Point lineStart(centerX, centerY - lineLength);
        cv::Point lineEnd(centerX, centerY - 20);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 250, 250), 2);
        
        deBug(intersect, frameCount);
        frameCount++;
        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;
       
        frame.release();
        result.release();
    }
    
    servo.set_steering(0);
    cam.stop();
    cv::destroyAllWindows();
    return 0;
}
