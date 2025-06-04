#ifndef LANE_DETECTION_HPP
#define LANE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <thread>
#include <atomic>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path);
    ~TensorRTInference();

    struct Buffer {
        void* device;
        float* host;
        size_t size;
    };

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

std::vector<float> preprocess_frame(const cv::Mat& frame);
cv::Mat postprocess(float* da_output, float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints);

#endif