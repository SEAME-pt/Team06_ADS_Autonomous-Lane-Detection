#ifndef CSI_CAMERA_HPP
#define CSI_CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <thread>

class CSICamera {
public:
    CSICamera(int width = 640, int height = 360, int fps = 30);
    ~CSICamera();
    void start();
    cv::Mat read();
    void stop();

private:
    void update();
    std::string pipeline_;
    cv::VideoCapture cap_;
    cv::Mat frame_;
    bool running_;
    std::thread thread_;
};

#endif