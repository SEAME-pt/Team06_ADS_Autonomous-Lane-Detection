#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

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
        for (auto& mem : inputBuffers) {
            cudaFree(mem.device);
        }
        for (auto& mem : outputBuffers) {
            cudaFree(mem.device);
        }
    }

    struct Buffer {
        void* device;
        float* host;
        size_t size;
    };

    std::vector<std::vector<float>> infer(const std::vector<float>& inputData) {
        cudaMemcpy(inputBuffers[0].device, inputData.data(), inputBuffers[0].size, cudaMemcpyHostToDevice);

        context->executeV2(bindings.data());

        std::vector<std::vector<float>> outputs;
        for (auto& out : outputBuffers) {
            cudaMemcpy(out.host, out.device, out.size, cudaMemcpyDeviceToHost);
            outputs.emplace_back(out.host, out.host + out.size / sizeof(float));
        }

        return outputs;
    }

private:
    Logger logger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

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

            DataType dtype = engine->getBindingDataType(i);
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

class CSICamera {
public:
    CSICamera(int width, int height, int fps) {
        std::ostringstream pipeline;
        pipeline << "nvarguscamerasrc ! "
                 << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height
                 << ", format=NV12, framerate=" << fps << "/1 ! "
                 << "nvvidconv flip-method=0 ! video/x-raw, width=" << width
                 << ", height=" << height << ", format=BGRx ! "
                 << "videoconvert ! video/x-raw, format=BGR ! appsink";

        cap.open(pipeline.str(), cv::CAP_GSTREAMER);
    }

    void start() {
        running = true;
        thread = std::thread(&CSICamera::update, this);
    }

    void stop() {
        running = false;
        if (thread.joinable()) thread.join();
        cap.release();
    }

    cv::Mat read() const {
        return frame.clone();
    }

private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::thread thread;
    std::atomic<bool> running{false};

    void update() {
        while (running) {
            cv::Mat f;
            cap.read(f);
            if (!f.empty()) frame = f;
        }
    }
};

std::vector<float> preprocess_frame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(448, 448));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255); // Normaliza

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    std::vector<float> inputData;
    for (int i = 2; i >= 0; --i) { // BGR to RGB
        inputData.insert(inputData.end(), (float*)channels[i].datastart, (float*)channels[i].dataend);
    }
    return inputData;
}

cv::Mat postprocess_outputs(const std::vector<std::vector<float>>& outputs, const cv::Mat& original_frame) {
    cv::Mat da_mask(448, 448, CV_8UC1);
    cv::Mat ll_mask(448, 448, CV_8UC1);

    for (int i = 0; i < 448 * 448; ++i) {
        da_mask.data[i] = outputs[0][i * 2 + 1] > outputs[0][i * 2] ? 255 : 0;
        ll_mask.data[i] = outputs[1][i * 2 + 1] > outputs[1][i * 2] ? 255 : 0;
    }

    cv::resize(da_mask, da_mask, original_frame.size());
    cv::resize(ll_mask, ll_mask, original_frame.size());

    cv::Mat result = original_frame.clone();
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            if (da_mask.at<uchar>(i, j) > 100)
                result.at<cv::Vec3b>(i, j) = {255, 0, 0};
            if (ll_mask.at<uchar>(i, j) > 100)
                result.at<cv::Vec3b>(i, j) = {0, 255, 0};
        }
    }

    return result;
}

cv::Mat postprocess(float* da_output, float* ll_output, cv::Mat& original_frame) {
    const int height = 448;
    const int width = 448;

    // Criar as máscaras com shape (2, 448, 448)
    cv::Mat da_logits(2, height * width, CV_32FC1, da_output);
    cv::Mat ll_logits(2, height * width, CV_32FC1, ll_output);

    // Argmax manual
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

    // Redimensionar para tamanho original do frame
    cv::Mat da_resized, ll_resized;
    cv::resize(da_mask, da_resized, original_frame.size());
    cv::resize(ll_mask, ll_resized, original_frame.size());

    // Aplicar cores sobre o frame
    cv::Mat result = original_frame.clone();
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (da_resized.at<uchar>(y, x) > 100) {
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);  // Azul
            }
            if (ll_resized.at<uchar>(y, x) > 100) {
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);  // Verde
            }
        }
    }

    return result;
}


int main() {
    TensorRTInference trt("model.engine");
    CSICamera cam(448, 448, 15);
    cam.start();

    std::cout << "Pressione 'q' para sair" << std::endl;
    int frameCount = 0;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;

        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        //auto result = postprocess_outputs(outputs, frame);
        auto result = postprocess(outputs[0].data(), outputs[1].data(), frame);
        cv::Point lineStart(0, 224);
        cv::Point lineEnd(447, 224);
        cv::line(result, lineStart, lineEnd, cv::Scalar(250, 250, 250), 2);
        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;

        frameCount++;
        if (frameCount % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            double fps = 10 /  std::chrono::duration<double>(now - start).count();
            std::cout << "FPS: " << fps << std::endl;
            start = now;
        }
    }

    cam.stop();
    cv::destroyAllWindows();
    return 0;
}
