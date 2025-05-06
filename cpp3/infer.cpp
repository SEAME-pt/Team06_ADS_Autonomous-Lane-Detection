#include "infer.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>

TRTInfer::TRTInfer(const std::string& engine_path) {
    // Criar runtime com logger
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        throw std::runtime_error("Failed to create IRuntime");
    }

    // Carregar o arquivo .engine
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Engine file não encontrado");
    }

    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    // Deserializar o engine
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_size);
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize ICudaEngine");
    }

    // Criar contexto de execução
    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("Failed to create IExecutionContext");
    }

    // Obter índices de entrada e saída
    inputIndex = engine_->getBindingIndex("input");
    outputIndex = engine_->getBindingIndex("output");

    // Obter dimensões
    auto input_dims = engine_->getBindingDimensions(inputIndex);
    batchSize = input_dims.d[0];
    inputH = input_dims.d[2];
    inputW = input_dims.d[3];

    // Alocar buffers
    size_t input_size = 1 * 3 * inputH * inputW * sizeof(float);
    size_t output_size = 1 * 1 * inputH * inputW * sizeof(float);
    cudaMalloc(&buffers[inputIndex], input_size);
    cudaMalloc(&buffers[outputIndex], output_size);

    // Criar stream CUDA
    cudaStreamCreate(&stream);

    std::cout << "[INFO] Engine carregada com sucesso.\n";
}

TRTInfer::~TRTInfer() {
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    delete context_;
    delete engine_;
    delete runtime_;
}

void TRTInfer::preprocess(const cv::Mat& img, float* gpu_input) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    float host_input[3 * inputH * inputW];
    int idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < inputH; ++y)
            for (int x = 0; x < inputW; ++x)
                host_input[idx++] = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;

    cudaMemcpyAsync(gpu_input, host_input, sizeof(host_input),
                    cudaMemcpyHostToDevice, stream);
}

cv::Mat TRTInfer::postprocess(float* gpu_output) {
    float host_output[inputH * inputW];
    cudaMemcpyAsync(host_output, gpu_output, sizeof(host_output),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat mask(inputH, inputW, CV_8UC1);
    for (int y = 0; y < inputH; ++y)
        for (int x = 0; x < inputW; ++x)
            mask.at<uchar>(y, x) = host_output[y * inputW + x] > 0.5 ? 255 : 0;

    return mask;
}

cv::Mat TRTInfer::infer(const cv::Mat& input) {
    preprocess(input, (float*)buffers[inputIndex]);
    context_->enqueueV2(buffers, stream, nullptr);
    return postprocess((float*)buffers[outputIndex]);
}