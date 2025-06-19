#include "lane_detection.hpp"
#include "nmpc_controller.hpp"
#include "polyfit.cpp"
#include <iostream>
#include <chrono>
#include <vector>
#include "../FServo/FServo.hpp"


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

void bicicleModel(std::vector<double>& x0, double a, double delta){
    // Atualizar estado (para próxima iteração) - Dinamic Model
    x0[0] += x0[3] * cos(x0[2]) * 0.1; // x
    x0[1] += x0[3] * sin(x0[2]) * 0.1; // y
    x0[2] += (x0[3] / 0.15) * tan(delta) * 0.1; // psi
    x0[3] += a * 0.1; // vel
}

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(448, 448, 15);
    cam.start();
    
    // Inicializar NMPC
    NMPCController nmpc(0.15, 0.1, 10, 5, 0.5, 2.0); // L=0.15m, dt=0.1s, Np=10, Nc=5, delta_max=0.5rad, a_max=1m/s²
    std::vector<double> x0 = {0.0, 0.0, 0.0, 2.0}; // Estado inicial: [x, y, psi, v]
    
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
        auto result = postprocess(outputs[0].data(), outputs[1].data(), frame, medianPoints, laneData);
        
        // Visualizar marcadores de pixels
        //visualize_pixel_markers(result);
        
        // Executar NMPC
        std::vector<std::vector<double>> x_ref = generateReference(laneData, frameCount);
        std::vector<double> control = nmpc.compute_control(x0, x_ref);
        double delta = control[0]; // rad
        double a = 0; //control[1];     // m/s² (não usado por enquanto)
        
        // Converter delta de radianos para graus e limitar
        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-20, std::min(20, steering_angle));
        
        
        servo.set_steering(steering_angle);
        bicicleModel(x0, a, delta);
        
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

/* // Gerar trajetória de referência
std::vector<std::vector<double>> generateReference(const LaneData& laneData, int frameCount, double v_ref = 1.0, int Np = 10) {
    std::vector<std::vector<double>> x_ref(Np);
    if (!laneData.valid || laneData.num_points < 2) {
        std::cout << " laneData invalid!!!" << "\n";
        for (int k = 0; k < Np; ++k) {
            x_ref[k] = {0.0, 0.0, 0.0, v_ref};
        }
        return x_ref;
    }

    for (int k = 0; k < Np; ++k) {
        int idx = std::min(k * laneData.num_points / Np, laneData.num_points - 1);
        double x = laneData.points[idx].x;
        double y = laneData.points[idx].y;
        int next_idx = std::min(idx + 1, laneData.num_points - 1);
        double dx = laneData.points[next_idx].x - x;
        double dy = laneData.points[next_idx].y - y;
        double psi = atan2(dx, 0);
        x_ref[k] = {y, -x, psi, v_ref};
        if (frameCount % 20 == 0){
            std::cout << "k: " << k << "| x: " << x << "| y: " << y; 
            std::cout << "|  dx: " << dx << "| psi_ref: " << psi << std::endl;
        }
    }
    if (frameCount % 20 == 0)
        std::cout << "************************------*********************" << std::endl;
    return x_ref;
} */


/* std::vector<std::vector<double>> generateReference(const LaneData& laneData, int frameCount, double v_ref = 1.0, int Np = 10) {
    std::vector<std::vector<double>> x_ref(Np);

    if (!laneData.valid || laneData.num_points < 2) {
        for (int k = 0; k < Np; ++k) {
            x_ref[k] = {0.0, 0.0, 0.0, v_ref}; // Estado padrão se inválido
        }
        return x_ref;
    }

    // Número de pontos para suavização (ex., 3 pontos para direção média)
    const int window_size = std::min(3, laneData.num_points);
    std::vector<double> x_vals(window_size), y_vals(window_size);

    for (int k = 0; k < Np; ++k) {
        int idx = std::min(k * (laneData.num_points - 1) / (Np - 1), laneData.num_points - 1);
        // Preencher janela de pontos para suavização
        int start_idx = std::max(0, idx - (window_size - 1) / 2);
        int end_idx = std::min(laneData.num_points - 1, start_idx + window_size - 1);
        int count = end_idx - start_idx + 1;
        for (int i = 0, j = start_idx; j <= end_idx; ++i, ++j) {
            x_vals[i] = laneData.points[j].x;
            y_vals[i] = laneData.points[j].y;
        }

        // Calcular direção média usando uma regressão linear simples
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
        for (int i = 0; i < count; ++i) {
            sum_x += x_vals[i];
            sum_y += y_vals[i];
            sum_xy += x_vals[i] * y_vals[i];
            sum_xx += x_vals[i] * x_vals[i];
        }
        double slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x); // Inclinação da reta
        double theta = atan2(1.0, slope); // Ângulo da direção (aproximação)

        // Ajuste com base no desvio lateral (x_laneData no ponto mais próximo)
        int nearest_idx = laneData.num_points - 1; // Ponto 9 como o mais próximo
        double dx_lateral = laneData.points[nearest_idx].x; // Desvio lateral
        double psi_ref = theta; // Direção base
        if (dx_lateral != 0) {
            // Ajuste relativo ao desvio (simplificação)
            psi_ref += atan2(0.0, dx_lateral); // Correção mínima; idealmente usa distância ao próximo ponto
        }

        // Definir estado de referência
        double x_ref_val = laneData.points[idx].y; // Inverter x e y conforme convenção
        double y_ref_val = -laneData.points[idx].x;
        x_ref[k] = {y_ref_val, x_ref_val, psi_ref, v_ref};

        // Log para depuração
        if (frameCount % 20 == 0){
            std::cout << "k: " << k << ", idx: " << idx << ", x: " << laneData.points[idx].x 
                      << ", y: " << laneData.points[idx].y << ", theta: " << theta 
                      << ", dx_lateral: " << dx_lateral << ", psi_ref: " << psi_ref << std::endl;
        }
    }

    return x_ref;
} */