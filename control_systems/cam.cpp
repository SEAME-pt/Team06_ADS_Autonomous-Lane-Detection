#include "cam.hpp"
#include <sstream>
#include <iostream>

CSICamera::CSICamera(int width, int height, int fps) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-mode=4 ! "
             << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height
             << ", format=NV12, framerate=" << fps << "/1 ! "
             << "nvvidconv flip-method=0 ! video/x-raw, width=" << width
             << ", height=" << height << ", format=BGRx ! "
             << "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";

    std::cout << "Trying CSI camera with pipeline: " << pipeline.str() << std::endl;
    cap.open(pipeline.str(), cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cout << "CSI camera failed, trying USB camera fallback..." << std::endl;
        cap.open(0); // Try USB camera fallback
        if (!cap.isOpened()) {
            throw std::runtime_error("CSICamera: failed to open both GStreamer pipeline and USB camera");
        }
        std::cout << "Using USB camera fallback" << std::endl;
    } else {
        std::cout << "CSI camera initialized successfully" << std::endl;
    }
}

void CSICamera::start() {
    running = true;
    thread = std::thread(&CSICamera::update, this);
}

void CSICamera::stop() {
    running = false;
    if (thread.joinable()) thread.join();
    cap.release();
}

cv::Mat CSICamera::read() const {
    std::lock_guard<std::mutex> lock(frame_mutex);
    if (frame.empty()) {
        return cv::Mat();  // Return empty Mat if frame not ready
    }
    return frame.clone();
}

void CSICamera::update() {
    while (running) {
        cv::Mat f;
        cap.read(f);
        if (!f.empty()) {
            cv::resize(f, f, cv::Size(640, 360));
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame = f;
            }
        }
    }
}
