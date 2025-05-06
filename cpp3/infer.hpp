// infer.hpp
#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};

class TRTInfer {
public:
    TRTInfer(const std::string& engine_path);
    ~TRTInfer();
    cv::Mat infer(const cv::Mat& input);
    void preprocess(const cv::Mat& img, float* gpu_input);
    cv::Mat postprocess(float* gpu_output);

private:
    Logger logger_; // Logger como membro da classe
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* buffers[2]; // Para input e output
    int inputIndex;
    int outputIndex;
    int batchSize;
    int inputH;
    int inputW;
    cudaStream_t stream;
};

#endif // TRT_INFER_HPP