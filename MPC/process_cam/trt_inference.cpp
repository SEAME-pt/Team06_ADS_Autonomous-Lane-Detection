#include "trt_inference.hpp"
#include <fstream>
#include <iostream>

TensorRTInference::TensorRTInference(const std::string& engine_path) : logger_() {
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return;
    }

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return;
    }

    cudaStreamCreate(&stream_);
    int num_bindings = engine_->getNbIOTensors();
    buffers_.resize(num_bindings);
    binding_sizes_.resize(num_bindings);

    for (int i = 0; i < num_bindings; i++) {
        auto dims = engine_->getTensorShape(engine_->getIOTensorName(i));
        size_t size = 1;
        for (int d = 0; d < dims.nbDims; d++) size *= dims.d[d];
        binding_sizes_[i] = size * sizeof(float);
        cudaMalloc(&buffers_[i], binding_sizes_[i]);
    }

    // Associar tensores ao contexto
    for (int i = 0; i < num_bindings; i++) {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
            context_->setInputTensorAddress(tensor_name, buffers_[i]);
        } else {
            context_->setOutputTensorAddress(tensor_name, buffers_[i]);
        }
    }
}

TensorRTInference::~TensorRTInference() {
    for (void* buffer : buffers_) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream_);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

std::vector<float*> TensorRTInference::infer(float* input_data) {
    // Copiar dados de entrada para o buffer da GPU
    cudaMemcpyAsync(buffers_[0], input_data, binding_sizes_[0], cudaMemcpyHostToDevice, stream_);

    // Executar inferência
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "Failed to execute inference" << std::endl;
        return std::vector<float*>();
    }

    // Copiar resultados da GPU
    std::vector<float*> outputs;
    for (int i = 1; i < buffers_.size(); i++) {
        float* output = new float[binding_sizes_[i] / sizeof(float)];
        cudaMemcpyAsync(output, buffers_[i], binding_sizes_[i], cudaMemcpyDeviceToHost, stream_);
        outputs.push_back(output);
    }
    cudaStreamSynchronize(stream_); // Garantir que todas as operações sejam concluídas
    return outputs;
}