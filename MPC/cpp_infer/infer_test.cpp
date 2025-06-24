#include <iostream>
#include <chrono>
#include <vector>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <fstream>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path) {
        std::ifstream engineFile(engine_path, std::ios::binary);
        if (!engineFile) throw std::runtime_error("Erro ao abrir engine");

        engineFile.seekg(0, engineFile.end);
        size_t fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);

        std::vector<char> engineData(fsize);
        engineFile.read(engineData.data(), fsize);

        runtime = createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
        context = engine->createExecutionContext();

        allocateBuffers();
    }

    ~TensorRTInference() {
        for (auto& mem : inputBuffers) cudaFree(mem.device);
        for (auto& mem : outputBuffers) cudaFree(mem.device);
    }

    void run_dummy_inference(int iters) {
        std::vector<float> dummy_input(inputBuffers[0].size / sizeof(float), 0.5f);
        cudaMemcpy(inputBuffers[0].device, dummy_input.data(), inputBuffers[0].size, cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            context->executeV2(bindings.data());
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double fps = iters / std::chrono::duration<double>(end - start).count();
        std::cout << "Inference-only FPS: " << fps << std::endl;
    }

private:
    Logger logger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    struct Buffer {
        void* device;
        float* host;
        size_t size;
    };

    std::vector<Buffer> inputBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<void*> bindings;

    void allocateBuffers() {
        int nbBindings = engine->getNbBindings();
        inputBuffers.resize(1);
        outputBuffers.resize(nbBindings - 1);
        bindings.resize(nbBindings);

        for (int i = 0; i < nbBindings; ++i) {
            Dims dims = engine->getBindingDimensions(i);
            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
            size_t typeSize = sizeof(float); // Assumindo float

            void* deviceMem;
            cudaMalloc(&deviceMem, vol * typeSize);
            float* hostMem = new float[vol];

            bindings[i] = deviceMem;
            if (engine->bindingIsInput(i)) {
                inputBuffers[0] = {deviceMem, hostMem, vol * typeSize};
            } else {
                outputBuffers[i - 1] = {deviceMem, hostMem, vol * typeSize};
            }
        }
    }
};

int main() {
    TensorRTInference trt("model.engine");
    trt.run_dummy_inference(100);
    return 0;
}


//g++ -std=c++17 -O2 -I/path/to/tensorrt/include -L/path/to/tensorrt/lib -lnvinfer -lcudart -o infer_test infer_test.cpp
