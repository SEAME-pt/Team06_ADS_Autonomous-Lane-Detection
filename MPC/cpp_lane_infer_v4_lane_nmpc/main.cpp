#include "lane_detection.hpp"
#include <iostream>
#include <chrono>

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(640, 360, 30);
    cam.start();

    std::cout << "Pressione 'q' para sair" << std::endl;

    auto lastTime = std::chrono::steady_clock::now();
    double smoothedFPS = 0.0;
    const double alpha = 0.9;

    while (true) {
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;

        auto currentTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        double currentFPS = 1.0 / elapsed;
        smoothedFPS = smoothedFPS == 0.0 ? currentFPS : alpha * smoothedFPS + (1.0 - alpha) * currentFPS;

        // Inference pipeline
        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        LaneData laneData;
        auto result = postprocess(outputs[0].data(), outputs[1].data(), frame, medianPoints, laneData);

        // Desenhar o FPS na imagem
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(smoothedFPS));
        cv::putText(result, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2);

        int centerX = result.cols / 2 + 10;
        int centerY = result.rows;
        int lineLength = 200;

        cv::Point lineStart(centerX, centerY - lineLength);
        cv::Point lineEnd(centerX, centerY - 20);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 250, 250), 2);  

        // Exibir LaneData para depuração
        if (laneData.valid) {
            std::cout << "LaneData: " << laneData.num_points << " pontos, timestamp: " << laneData.timestamp << "\n";
            for (int i = 0; i < laneData.num_points; ++i) {
                std::cout << "  Ponto " << i << ": (" << laneData.points[i].x << ", " << laneData.points[i].y << ")\n";
            }
        } else {
            std::cout << "LaneData inválido\n";
        }

        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;
    }

    cam.stop();
    cv::destroyAllWindows();
    return 0;
}