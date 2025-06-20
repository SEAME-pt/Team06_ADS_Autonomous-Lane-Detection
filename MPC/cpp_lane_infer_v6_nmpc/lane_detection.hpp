#ifndef LANE_DETECTION_HPP
#define LANE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <vector>
#include <iostream>
#include <chrono>


using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct LineCoef {
    double m, b;
    bool valid;
};

struct Buffer {
    void* device;
    float* host;
    size_t size;
};

struct LaneData {
    bool valid;
    double timestamp;
    int num_points;
    struct Point {
        float x;
        float y;
    } points[10];
};

struct LineIntersect {
    cv::Point2f xl_t;    // Interseção da linha esquerda com roi_start_y (pixels)
    cv::Point2f xl_b;    // Interseção da linha esquerda com roi_end_y (pixels)
    cv::Point2f xr_t;    // Interseção da linha direita com roi_start_y (pixels)
    cv::Point2f xr_b;    // Interseção da linha direita com roi_end_y (pixels)
    float ratio_top;     // Razão da posição relativa na margem superior
    float xs_b;          // Ponto x da mediana estimada na margem inferior
    float slope;         // Inclinação da mediana estimada
    float psi;           // Yaw (ângulo) da mediana estimada
    bool valid;
};

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path);
    ~TensorRTInference();

    std::vector<std::vector<float>> infer(const std::vector<float>& inputData);

private:
    Logger logger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    std::vector<Buffer> inputBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<void*> bindings;

    void allocateBuffers();
};

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

    void update();
};

std::vector<float>  preprocess_frame(const cv::Mat& frame);
cv::Mat             postprocess(float* da_output, float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints, LaneData& laneData, LineIntersect& intersect);
LineIntersect       findIntersect(const LineCoef& left_coeffs, const LineCoef& right_coeffs, int height, int width);

#endif