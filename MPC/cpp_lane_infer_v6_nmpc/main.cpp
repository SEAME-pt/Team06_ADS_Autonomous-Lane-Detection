#include "lane_detection.hpp"
#include "nmpc_controller.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include "../FServo/FServo.hpp"

// Gerar trajetória de referência
std::vector<std::vector<double>> generateReference(const LaneData& laneData, double v_ref = 1.0, int Np = 10) {
    std::vector<std::vector<double>> x_ref(Np);
    if (!laneData.valid || laneData.num_points < 2) {
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
        double psi = atan2(dy, dx);
        x_ref[k] = {y, -x, psi, v_ref};
    }
    return x_ref;
}

// Função para marcar pixels de 10 em 10 a 95% da altura da imagem
void visualize_pixel_markers(cv::Mat& frame) {
    int y_pos = static_cast<int>(0.95 * frame.rows); // 95% da altura (y = 342 para 448px)
    cv::Mat marker_frame = frame.clone(); // Criar uma cópia para não modificar o original

    // Desenhar marcadores a cada 10 pixels ao longo da largura
    for (int x = 0; x < frame.cols; x += 10) {
        cv::circle(marker_frame, cv::Point(x, y_pos), 2, cv::Scalar(0, 0, 255), -1);
        if (x % 100 == 0) {
            std::string label = std::to_string(x);
            cv::putText(marker_frame, label, cv::Point(x, y_pos - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
        }
    }

    // Exibir a imagem com marcadores em uma janela separada
    cv::imshow("Pixel Markers", marker_frame);
}

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(448, 448, 30);
    cam.start();

    // Inicializar NMPC
    NMPCController nmpc(0.15, 0.1, 10, 5, 0.8, 2.0); // L=0.15m, dt=0.1s, Np=10, Nc=5, delta_max=0.8rad, a_max=1m/s²
    std::vector<double> x0 = {0.0, 0.0, 1.57, 1.0}; // Estado inicial: [x, y, psi, v]

    // Inicializar servo
    FServo servo;
    try {
        servo.open_i2c_bus();
        if (!servo.init_servo()) {
            throw std::runtime_error("Falha ao inicializar o servo");
        }
        std::cout << "Servo inicializado com sucesso\n";
        // Calibração inicial (opcional)
/*         servo.set_steering(0);
        sleep(1);
        servo.set_steering(-45);
        sleep(1);
        servo.set_steering(45);
        sleep(1); */
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
        std::vector<std::vector<double>> x_ref = generateReference(laneData);
        std::vector<double> control = nmpc.compute_control(x0, x_ref);
        double delta = control[0]; // rad
        double a = control[1];     // m/s² (não usado por enquanto)

        // Converter delta de radianos para graus e limitar
        int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
        steering_angle = std::max(-45, std::min(45, steering_angle));

        // Enviar ao servo
        servo.set_steering(steering_angle);

        // Atualizar estado (para próxima iteração)
        x0[0] += x0[3] * cos(x0[2]) * 0.1;
        x0[1] += x0[3] * sin(x0[2]) * 0.1;
        x0[2] += (x0[3] / 0.15) * tan(delta) * 0.1;
        x0[3] += a * 0.1;

        // Exibir FPS e controles
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        std::string deltaText = "Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg";
        cv::putText(result, deltaText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        std::string aText = "Accel: " + std::to_string(a) + " m/s^2";
        cv::putText(result, aText, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

        int centerX = result.cols / 2;
        int centerY = result.rows;
        int lineLength = 200;
        cv::Point lineStart(centerX, centerY - lineLength);
        cv::Point lineEnd(centerX, centerY - 20);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 250, 250), 2);

        if (laneData.valid && frameCount % 20 == 0) {
            /* for (int i = 0; i < laneData.num_points; ++i) {
                std::cout << "  Ponto " << i << ": (" << laneData.points[i].x << ", " << laneData.points[i].y << ")\n";
            } */
            std::cout << "  Ponto: (" << laneData.points[9].x << ", " << laneData.points[9].y << ")\n";
            std::cout << "  Delta: " + std::to_string(delta * 180.0 / M_PI) + " deg \n";
        }

        frameCount++;
        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;
    }

    cam.stop();
    cv::destroyAllWindows();
    return 0;
}