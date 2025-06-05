#include "lane_detection.hpp"
#include <iostream>
#include <chrono>

// Programa principal: Captura vídeo, realiza inferência e exibe resultados
int main() {
    // Inicializa inferência TensorRT com modelo
    TensorRTInference trt("../model.engine");
    
    // Inicializa câmera CSI com resolução 640x360 e 30 FPS
    CSICamera cam(640, 360, 30);
    cam.start();

    std::cout << "Pressione 'q' para sair" << std::endl;
    int frameCount = 0;
    auto start = std::chrono::steady_clock::now();

    // Loop principal
    while (true) {
        // Lê frame da câmera
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;

        // Pré-processa frame para entrada do modelo
        std::vector<float> input = preprocess_frame(frame);
        
        // Realiza inferência
        auto outputs = trt.infer(input);
        
        // Processa saídas e calcula mediana
        std::vector<cv::Point> medianPoints;
        // Corrigido: Removido 'and()' da chamada de postprocess
        auto [result_mask, result_frame] = postprocess(outputs[0].data(), outputs[1].data(), frame, medianPoints);

        // Exibe máscara processada em uma janela
        //cv::imshow("Lane Detection - Mask", result_mask);
        // Exibe frame original com linhas em outra janela
        cv::imshow("Lane Detection - Frame", result_frame);
        if (cv::waitKey(1) == 'q') break;

        // Calcula e exibe FPS a cada 30 frames
        frameCount++;
        if (frameCount % 30 == 0) {
            auto now = std::chrono::steady_clock::now();
            double fps = 30.0 / std::chrono::duration<double>(now - start).count();
            std::cout << "FPS: " << fps << std::endl;
            start = now;
        }
    }

    // Para câmera e fecha janelas
    cam.stop();
    cv::destroyAllWindows();
    return 0;
}