#ifndef CAM_HPP
#define CAM_HPP

// Classe para câmera CSI
class CSICamera {
private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::thread thread;
    std::atomic<bool> running{false};
    mutable std::mutex frame_mutex;

public:
    CSICamera(int width = 640, int height = 480, int fps = 30, int flip_method = 0) {
        std::ostringstream pipeline;
        pipeline << "nvarguscamerasrc ! "
                 << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height
                 << ", format=NV12, framerate=" << fps << "/1 ! "
                 << "nvvidconv flip-method=" << flip_method << " ! "
                 << "video/x-raw, width=" << width << ", height=" << height << ", format=BGRx ! "
                 << "videoconvert ! "
                 << "video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
        
        cap.open(pipeline.str(), cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            std::cerr << "Erro: Não foi possível abrir a câmera CSI. Tentando fallback..." << std::endl;
            cap.open(0);
            if (!cap.isOpened()) {
                throw std::runtime_error("Erro ao abrir câmera");
            }
        }
    }

    void start() {
        running = true;
        thread = std::thread(&CSICamera::update, this);
    }

    void stop() {
        running = false;
        if (thread.joinable()) thread.join();
        cap.release();
    }

    cv::Mat read() const {
        std::lock_guard<std::mutex> lock(frame_mutex);
        return frame.clone();
    }

private:
    void update() {
        while (running) {
            cv::Mat f;
            cap >> f;
            if (!f.empty()) {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame = f;
            }
        }
    }
};

#endif