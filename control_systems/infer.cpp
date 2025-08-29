#include "infer.hpp"
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>

TensorRTInference::TensorRTInference(const std::string& engine_path) {
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) {
        throw std::runtime_error(std::string("Lane engine not found: ") + engine_path);
    }

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
    for (auto& mem : inputBuffers)  { if (mem.device) cudaFree(mem.device);  delete[] mem.host; }
    for (auto& mem : outputBuffers) { if (mem.device) cudaFree(mem.device);  delete[] mem.host; }

    if (context) context->destroy();
    if (engine)  engine->destroy();
    if (runtime) runtime->destroy();
}

void TensorRTInference::allocateBuffers() {
    const int nbBindings = engine->getNbBindings();

    inputBuffers.resize(1);
    outputBuffers.resize(1);
    bindings.resize(nbBindings, nullptr);

    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            vol *= static_cast<size_t>(dims.d[j]);
        }

        const size_t totalSize = vol * sizeof(float);

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
            outputBuffers[0] = Buffer{deviceMem, hostMem, totalSize};
        }
    }
}

std::vector<float> TensorRTInference::infer(const std::vector<float>& inputData) {
    // Validate input size
    if (inputData.size() * sizeof(float) > inputBuffers[0].size) {
        throw std::runtime_error("Lane: Input data size exceeds buffer capacity");
    }

    auto copyH2D = cudaMemcpy(inputBuffers[0].device, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (copyH2D != cudaSuccess) {
        throw std::runtime_error(std::string("Lane: cudaMemcpy H2D failed: ") + cudaGetErrorString(copyH2D));
    }

    if (!context->executeV2(bindings.data())) {
        throw std::runtime_error("Lane: executeV2 failed");
    }

    auto err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Lane CUDA kernel error: ") + cudaGetErrorString(err));
    }
    auto syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        throw std::runtime_error(std::string("Lane CUDA sync error: ") + cudaGetErrorString(syncErr));
    }

    auto copyD2H = cudaMemcpy(outputBuffers[0].host, outputBuffers[0].device, outputBuffers[0].size, cudaMemcpyDeviceToHost);
    if (copyD2H != cudaSuccess) {
        throw std::runtime_error(std::string("Lane: cudaMemcpy D2H failed: ") + cudaGetErrorString(copyD2H));
    }

    const size_t num_floats = outputBuffers[0].size / sizeof(float);
    return std::vector<float>(outputBuffers[0].host, outputBuffers[0].host + num_floats);
}
