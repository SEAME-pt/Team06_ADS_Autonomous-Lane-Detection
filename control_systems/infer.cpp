#include "infer.hpp"
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>

TensorRTInference::TensorRTInference(const std::string& engine_path) {
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) throw std::runtime_error(std::string("Lane engine not found: ") + engine_path);

    engineFile.seekg(0, engineFile.end);
    size_t fsize = static_cast<size_t>(engineFile.tellg());
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    runtime = createInferRuntime(logger);
    if (!runtime) throw std::runtime_error("Lane: createInferRuntime() returned null");

    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine) throw std::runtime_error("Lane: deserializeCudaEngine() failed");

    context = engine->createExecutionContext();
    if (!context) throw std::runtime_error("Lane: createExecutionContext() failed");

    allocateBuffers();
}

TensorRTInference::~TensorRTInference() {
    // libertar device mem
    for (auto& mem : inputBuffers)  { if (mem.device) cudaFree(mem.device); }
    for (auto& mem : outputBuffers) { if (mem.device) cudaFree(mem.device); }
    // libertar host mem
    for (auto& mem : inputBuffers)  { delete[] mem.host; }
    for (auto& mem : outputBuffers) { delete[] mem.host; }

    // Nota: APIs destroy() podem estar deprecadas, mas são válidas nas versões Jetson
    if (context) context->destroy();
    if (engine)  engine->destroy();
    if (runtime) runtime->destroy();
}

void TensorRTInference::allocateBuffers() {
    const int nbBindings = engine->getNbBindings();

    // temos 1 input e 1 output buffer
    inputBuffers.resize(1);
    outputBuffers.resize(1);
    bindings.resize(nbBindings, nullptr);

    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            vol *= static_cast<size_t>(dims.d[j]);
        }

        const size_t typeSize = sizeof(float);
        const size_t totalSize = vol * typeSize;

        void* deviceMem = nullptr;
        auto st = cudaMalloc(&deviceMem, totalSize);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("Lane: cudaMalloc failed: ") + cudaGetErrorString(st));
        }

        float* hostMem = new float[vol];
        bindings[i] = deviceMem;

        if (engine->bindingIsInput(i)) {
            inputBuffers[0] = Buffer{deviceMem, hostMem, totalSize};
        } else {
            outputBuffers = Buffer{deviceMem, hostMem, totalSize};
        }
    }
}

std::vector<float> TensorRTInference::infer(const std::vector<float>& inputData) {
    // copiar input para GPU
    cudaMemcpy(inputBuffers.device, inputData.data(), inputBuffers.size, cudaMemcpyHostToDevice);

    // executar a inferência
    context->executeV2(bindings.data());
    auto err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Lane CUDA kernel error: ") + cudaGetErrorString(err));
    }
    auto syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        throw std::runtime_error(std::string("Lane CUDA sync error: ") + cudaGetErrorString(syncErr));
    }

    // copiar output para host
    cudaMemcpy(outputBuffers[0].host, outputBuffers.device, outputBuffers.size, cudaMemcpyDeviceToHost);

    const size_t num_floats = outputBuffers.size / sizeof(float);
    return std::vector<float>(outputBuffers.host, outputBuffers.host + num_floats);
}
