#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <deque>
#include <algorithm>
#include <cmath>

#include "cam.hpp"
#include "fps.hpp"
#include "infer.hpp"
#include "frame.hpp"

int main() {
    try {
        std::string engine_path = "../best.engine";
        int input_size = 640;
        FrameSkipper::Strategy skip_strategy = FrameSkipper::FIXED;
        int skip_frames = 8;
        double target_fps = 60.0;
        
        std::cout << "Inicializando TensorRT YOLO..." << std::endl;
        TensorRTYOLO detector(engine_path, input_size);
        
        std::cout << "Configurando frame skipper..." << std::endl;
        FrameSkipper frame_skipper(skip_strategy, skip_frames, target_fps);
        
        std::cout << "Inicializando calculadora de FPS..." << std::endl;
        FPSCalculator fps_calculator(30);
        
        std::cout << "Configurando câmera CSI..." << std::endl;
        CSICamera camera(640, 480, 30);
        camera.start();
        
        std::cout << "Sistema inicializado com sucesso!" << std::endl;
        std::cout << "Pressiona 'q' para sair" << std::endl;
        std::cout << "Pressiona '1' para estratégia FIXED" << std::endl;
        std::cout << "Pressiona '2' para estratégia ADAPTIVE" << std::endl;
        std::cout << "Pressiona '3' para estratégia TIME_BASED" << std::endl;
        
        int frame_count = 0;
        int processed_frames = 0;
        int skipped_frames = 0;
        std::vector<Detection> last_detections;
        
        auto stats_start = std::chrono::high_resolution_clock::now();
        int stats_frame_count = 0;
        const double stats_interval = 5.0;
        
        while (true) {
            cv::Mat frame = camera.read();
            if (frame.empty()) continue;
            
            fps_calculator.update();
            frame_count++;
            stats_frame_count++;
            
            std::vector<Detection> detections;
            
            if (frame_skipper.shouldProcessFrame()) {
                auto inference_start = std::chrono::high_resolution_clock::now();
                detections = detector.infer(frame);
                auto inference_end = std::chrono::high_resolution_clock::now();
                
                double inference_time = std::chrono::duration<double>(inference_end - inference_start).count();
                frame_skipper.recordProcessingTime(inference_time);
                
                processed_frames++;
                last_detections = detections;
            } else {
                detections = last_detections;
                skipped_frames++;
            }
            
            // Desenhar detecções
            for (const auto& det : detections) {
                cv::Scalar color = detector.getColor(det.class_id);
                cv::rectangle(frame, det.bbox, color, 2);
                
                std::string label = det.class_name + ": " + std::to_string(det.confidence).substr(0, 4);
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
                cv::rectangle(frame, 
                             cv::Point(det.bbox.x, det.bbox.y - label_size.height - 10),
                             cv::Point(det.bbox.x + label_size.width, det.bbox.y), 
                             color, -1);
                cv::putText(frame, label, 
                           cv::Point(det.bbox.x, det.bbox.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            }
            
            // Estatísticas periódicas
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - stats_start).count();
            
            if (elapsed >= stats_interval) {
                double period_fps = stats_frame_count / elapsed;
                double processing_ratio = (double)processed_frames / frame_count * 100;
                
                std::cout << "Stats (últimos " << stats_interval << "s): FPS=" << std::fixed << std::setprecision(1) << period_fps 
                         << " | Processados: " << processed_frames << "/" << frame_count 
                         << " (" << std::setprecision(1) << processing_ratio << "%) | Saltados: " << skipped_frames << std::endl;
                
                stats_start = current_time;
                stats_frame_count = 0;
            }
            
            // Informações na tela
            double smooth_fps = fps_calculator.getSmoothFPS();
            std::string info_text = "FPS: " + std::to_string(smooth_fps).substr(0, 4) + 
                                   " | Proc: " + std::to_string(processed_frames) + 
                                   " | Skip: " + std::to_string(skipped_frames);
            cv::putText(frame, info_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            std::string strategy_names[] = {"FIXED", "ADAPTIVE", "TIME_BASED"};
            std::string strategy_text = "Estrategia: " + strategy_names[frame_skipper.getStrategy()];
            cv::putText(frame, strategy_text, cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
            
            std::string detections_text = "Deteccoes: " + std::to_string(detections.size());
            cv::putText(frame, detections_text, cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
            
            cv::imshow("TensorRT YOLO", frame);
            
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;
            else if (key == '1') {
                frame_skipper.setStrategy(FrameSkipper::FIXED);
                std::cout << "Estratégia: FIXED" << std::endl;
            }
            else if (key == '2') {
                frame_skipper.setStrategy(FrameSkipper::ADAPTIVE);
                std::cout << "Estratégia: ADAPTIVE" << std::endl;
            }
            else if (key == '3') {
                frame_skipper.setStrategy(FrameSkipper::TIME_BASED);
                std::cout << "Estratégia: TIME_BASED" << std::endl;
            }
        }
        
        camera.stop();
        cv::destroyAllWindows();
        
        // Estatísticas finais
        double final_fps = fps_calculator.getSmoothFPS();
        std::cout << "\n=== Estatísticas Finais ===" << std::endl;
        std::cout << "Total de frames capturados: " << frame_count << std::endl;
        std::cout << "Frames processados: " << processed_frames << std::endl;
        std::cout << "Frames saltados: " << skipped_frames << std::endl;
        std::cout << "Taxa de processamento: " << std::fixed << std::setprecision(1) 
                  << (double)processed_frames/frame_count*100 << "%" << std::endl;
        std::cout << "FPS final (suavizado): " << std::setprecision(1) << final_fps << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
