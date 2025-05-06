#include "infer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "controller.hpp"

std::string gstreamer_pipeline() {
    return "nvarguscamerasrc ! "
           "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
           "nvvidconv ! video/x-raw, format=BGR ! "
           "appsink drop=true max-buffers=1 sync=false";
}

int main()
{
    TRTInfer infer("unet_model_2.engine");

    cv::VideoCapture cap(gstreamer_pipeline(), cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "Falha ao abrir a câmara." << std::endl;
        return -1;
    }
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) 
    {
        std::cerr << "Erro: não conseguiu capturar o primeiro frame!" << std::endl;
        return -1;
    }
    
    Controller controller(frame.cols, frame.rows); 


    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat mask = infer.infer(frame);
        cv::resize(mask, mask, frame.size());

        cv::Mat overlay = frame.clone();
        cv::Mat color_mask;
        cv::cvtColor(mask, color_mask, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 0.7, color_mask, 0.3, 0, overlay);

        // ---------- Calcular ângulo ----------
        float angle = controller.calcular_angulo(mask, overlay);
        cv::putText(overlay, "Angle: " + std::to_string(angle * 180.0 / CV_PI) + " deg",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);

        cv::imshow("Frame", overlay);
        if (cv::waitKey(1) == 27) break; // ESC
    }

    return 0;
}
