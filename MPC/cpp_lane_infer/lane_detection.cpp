#include "lane_detection.hpp"
#include "mask_processor.hpp"
#include <iostream>
#include <chrono>

TensorRTInference::TensorRTInference(const std::string& engine_path) {
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

TensorRTInference::~TensorRTInference() {
    for (auto& mem : inputBuffers) cudaFree(mem.device);
    for (auto& mem : outputBuffers) cudaFree(mem.device);
}

void TensorRTInference::allocateBuffers() {
    int nbBindings = engine->getNbBindings();
    inputBuffers.resize(1);
    outputBuffers.resize(nbBindings - 1);
    bindings.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];

        DataType dtype = engine->getBindingDataType(i);
        size_t typeSize = sizeof(float);

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

std::vector<std::vector<float>> TensorRTInference::infer(const std::vector<float>& inputData) {
    cudaMemcpy(inputBuffers[0].device, inputData.data(), inputBuffers[0].size, cudaMemcpyHostToDevice);
    context->executeV2(bindings.data());

    std::vector<std::vector<float>> outputs;
    for (auto& out : outputBuffers) {
        cudaMemcpy(out.host, out.device, out.size, cudaMemcpyDeviceToHost);
        outputs.emplace_back(out.host, out.host + out.size / sizeof(float));
    }
    return outputs;
}

CSICamera::CSICamera(int width, int height, int fps) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc ! "
             << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height
             << ", format=NV12, framerate=" << fps << "/1 ! "
             << "nvvidconv flip-method=0 ! video/x-raw, width=" << width
             << ", height=" << height << ", format=BGRx ! "
             << "videoconvert ! video/x-raw, format=BGR ! appsink";

    cap.open(pipeline.str(), cv::CAP_GSTREAMER);
}

void CSICamera::start() {
    running = true;
    thread = std::thread(&CSICamera::update, this);
}

void CSICamera::stop() {
    running = false;
    if (thread.joinable()) thread.join();
    cap.release();
}

cv::Mat CSICamera::read() const {
    return frame.clone();
}

void CSICamera::update() {
    while (running) {
        cv::Mat f;
        cap.read(f);
        if (!f.empty()) frame = f;
    }
}

std::vector<float> preprocess_frame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 360));
    std::vector<float> inputData(3 * 640 * 360);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 360; ++i) {
            for (int j = 0; j < 640; ++j) {
                inputData[idx++] = resized.at<cv::Vec3b>(i, j)[2 - c] / 255.0f;  // BGR -> RGB
            }
        }
    }
    return inputData;
}

cv::Mat postprocess(float* da_output, float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints) {
    const int height = 360;
    const int width = 640;

    // Criar as máscaras
    cv::Mat da_logits(2, height * width, CV_32FC1, da_output);
    cv::Mat ll_logits(2, height * width, CV_32FC1, ll_output);

    cv::Mat da_mask(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat ll_mask(height, width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < height * width; ++i) {
        float da0 = da_logits.at<float>(0, i);
        float da1 = da_logits.at<float>(1, i);
        float ll0 = ll_logits.at<float>(0, i);
        float ll1 = ll_logits.at<float>(1, i);

        da_mask.at<uchar>(i / width, i % width) = (da1 > da0) ? 255 : 0;
        ll_mask.at<uchar>(i / width, i % width) = (ll1 > ll0) ? 255 : 0;
    }

    // Redimensionar para tamanho original
    cv::Mat da_resized, ll_resized;
    cv::resize(da_mask, da_resized, original_frame.size());
    cv::resize(ll_mask, ll_resized, original_frame.size());

    // Processar a máscara da área transitável para obter bordas e mediana
    MaskProcessor processor;
    cv::Mat mask_output;
    processor.processMask(da_resized, mask_output, medianPoints);

    // Combinar com o frame original
    cv::Mat result = original_frame.clone();
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (da_resized.at<uchar>(y, x) > 100) {
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);  // Azul para área transitável
            }
            if (ll_resized.at<uchar>(y, x) > 100) {
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);  // Verde para linhas
            }
        }
    }

    // Sobrepor as linhas das bordas e a mediana
    result = mask_output.clone();
    return result;
}