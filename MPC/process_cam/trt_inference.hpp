#ifndef TRT_INFERENCE_HPP
#define TRT_INFERENCE_HPP

#include <NvInfer.h>
#include <vector>
#include <string>
#include <iostream>
#include <ostream>


// Implementação mínima de ILogger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << msg << std::endl;
        }
    }
};

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path);
    ~TensorRTInference();
    std::vector<float*> infer(float* input_data);

private:
/*     nvinfer1::ILogger logger_; */
    Logger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::vector<void*> buffers_;
    std::vector<int> binding_sizes_;
    cudaStream_t stream_;
};

#endif