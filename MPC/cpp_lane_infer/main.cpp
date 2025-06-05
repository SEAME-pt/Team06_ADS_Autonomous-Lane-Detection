#include "lane_detection.hpp"
#include <iostream>
#include <chrono>

int main() {
    TensorRTInference trt("../model.engine");
    CSICamera cam(640, 360, 30);
    cam.start();

    std::cout << "Pressione 'q' para sair" << std::endl;
    int frameCount = 0;
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;
        
        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        auto result = postprocess(outputs[0].data(), outputs[1].data(), frame, medianPoints);
        
        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;
        
        frameCount++;
        if (frameCount % 30 == 0) {
            auto now = std::chrono::steady_clock::now();
            double fps = 30.0 / std::chrono::duration<double>(now - start).count();
            std::cout << "FPS: " << fps << std::endl;
            start = now;
        }
    }

    cam.stop();
    cv::destroyAllWindows();
    return 0;
}