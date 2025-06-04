#include "csi_camera.hpp"
#include <iostream>

CSICamera::CSICamera(int width, int height, int fps) : running_(false) {
    pipeline_ = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
                ", height=" + std::to_string(height) + ", format=NV12, framerate=" + std::to_string(fps) +
                "/1 ! nvvidconv flip-method=0 ! video/x-raw, width=" + std::to_string(width) +
                ", height=" + std::to_string(height) + ", format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";
    cap_ = cv::VideoCapture(pipeline_, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
    }
}

CSICamera::~CSICamera() {
    stop();
}

void CSICamera::start() {
    running_ = true;
    thread_ = std::thread(&CSICamera::update, this);
}

void CSICamera::update() {
    while (running_) {
        cv::Mat frame;
        if (cap_.read(frame)) {
            frame_ = frame.clone();
        }
    }
}

cv::Mat CSICamera::read() {
    return frame_.clone();
}

void CSICamera::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    cap_.release();
}