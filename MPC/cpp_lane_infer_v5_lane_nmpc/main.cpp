#include "lane_detection.hpp"
#include "nmpc_controller.hpp"
#include <iostream>
#include <chrono>
#include <vector>

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

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(360, 360, 30);
    cam.start();

    // Inicializar NMPC
    NMPCController nmpc(0.15, 0.1, 10, 5, 0.5, 1.0); // L=0.15m, dt=0.1s, Np=10, Nc=5, delta_max=0.5rad, a_max=1m/s²
    std::vector<double> x0 = {0.0, 0.0, 1.57, 0.0}; // Estado inicial: [x, y, psi, v]

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

        // Executar NMPC
        std::vector<std::vector<double>> x_ref = generateReference(laneData);
        std::vector<double> control = nmpc.compute_control(x0, x_ref);
        double delta = control[0]; // rad
        double a = control[1];     // m/s²

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

        if (laneData.valid && frameCount % 20 == 0 ) {
            //std::cout << "LaneData: " << laneData.num_points << " pontos, timestamp: " << laneData.timestamp << "\n";
            for (int i = 0; i < laneData.num_points; ++i) {
                std::cout << "  Ponto " << i << ": (" << laneData.points[i].x << ", " << laneData.points[i].y << ")\n";
            }
            //std::cout << "NMPC: delta = " << delta * 180.0 / M_PI << " deg, a = " << a << " m/s^2\n";
        } /* else {
            std::cout << "LaneData inválido\n";
            } */
           
        frameCount++;
        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;
    }

    cam.stop();
    cv::destroyAllWindows();
    return 0;
}