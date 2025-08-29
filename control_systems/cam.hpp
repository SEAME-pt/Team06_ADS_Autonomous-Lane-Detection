#ifndef CAM_HPP
#define CAM_HPP

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <fstream>
#include <mutex>

class CSICamera {
public:
    CSICamera(int width, int height, int fps);
    void start();
    void stop();
    cv::Mat read() const;

private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::thread thread;
    std::atomic<bool> running{false};
    mutable std::mutex frame_mutex;

    void update();
};

#endif
