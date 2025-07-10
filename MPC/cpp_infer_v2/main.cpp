#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace nvinfer1;

// Logger para TensorRT
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
        if (!engineFile) throw std::runtime_error("Error engine");

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

    struct Buffer {
        void* device;
        float* host;
        size_t size;
    };

    std::vector<float> infer(const std::vector<float>& inputData) {
        cudaMemcpy(inputBuffers[0].device, inputData.data(), inputBuffers[0].size, cudaMemcpyHostToDevice);
        context->executeV2(bindings.data());
        cudaMemcpy(outputBuffers[0].host, outputBuffers[0].device, outputBuffers[0].size, cudaMemcpyDeviceToHost);
        return std::vector<float>(outputBuffers[0].host, outputBuffers[0].host + outputBuffers[0].size / sizeof(float));
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
        outputBuffers.resize(1);
        bindings.resize(nbBindings);

        for (int i = 0; i < nbBindings; ++i) {
            Dims dims = engine->getBindingDimensions(i);
            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
            size_t typeSize = sizeof(float);

            void* deviceMem;
            cudaMalloc(&deviceMem, vol * typeSize);
            float* hostMem = new float[vol];

            bindings[i] = deviceMem;
            if (engine->bindingIsInput(i)) {
                inputBuffers[0] = {deviceMem, hostMem, vol * typeSize};
            } else {
                outputBuffers[0] = {deviceMem, hostMem, vol * typeSize};
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
         << "nvvidconv ! "
         << "video/x-raw, format=BGRx ! "
         << "videoconvert ! "
         << "video/x-raw, format=BGR ! appsink";
            cap.open(pipeline.str(), cv::CAP_GSTREAMER);
            if (!cap.isOpened()) {
                std::cerr << "Erro: Não foi possível abrir a câmera CSI. Usando fallback." << std::endl;
      
            }
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
    cv::resize(frame, resized, cv::Size(224, 224));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f; // R
    channels[1] = (channels[1] - 0.456f) / 0.224f; // G
    channels[2] = (channels[2] - 0.406f) / 0.225f; // B
    cv::merge(channels,  resized);

    std::vector<float> inputData;
    for (int i = 0; i < 3; ++i) {
        inputData.insert(inputData.end(), (float*)channels[i].datastart, (float*)channels[i].dataend);
    }
    return inputData;
}


// Pós-processamento simples da máscara
cv::Mat postprocess_org(const float* ll_output, cv::Mat& original_frame) {
    const int height = 224, width = 224;
    cv::Mat ll_mask(height, width, CV_32FC1, (void*)ll_output);
    cv::Mat ll_bin;
    cv::threshold(ll_mask, ll_bin, 0.2, 255, cv::THRESH_BINARY);
    ll_bin.convertTo(ll_bin, CV_8UC1);
    cv::Mat ll_resized;
    cv::resize(ll_bin, ll_resized, original_frame.size(), 0, 0, cv::INTER_NEAREST);

 

 
    cv::Mat color_mask = cv::Mat::zeros(original_frame.size(), CV_8UC3);
    color_mask.setTo(cv::Scalar(255, 0, 255), ll_resized);
    cv::Mat overlay;
    cv::addWeighted(original_frame, 0.7, color_mask, 0.7, 0, overlay);
    return overlay;
}


cv::Mat postprocess(const float* ll_output, cv::Mat& original_frame) {
    const int height = 224, width = 224;
    // Cria a máscara a partir da saída do modelo
    cv::Mat ll_mask(height, width, CV_32FC1, (void*)ll_output);
    cv::Mat ll_bin;
    // Threshold agressivo para capturar mais pixels de linha
    cv::threshold(ll_mask, ll_bin, 0.1, 255, cv::THRESH_BINARY);
    ll_bin.convertTo(ll_bin, CV_8UC1);

    // Kernel maior para morfologia e dilatação
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // Fecha buracos nas linhas
    cv::morphologyEx(ll_bin, ll_bin, cv::MORPH_CLOSE, kernel);
    // Engrossa as linhas
    cv::dilate(ll_bin, ll_bin, kernel);

    // Redimensiona suavemente para o tamanho original do frame
    cv::Mat ll_resized;
    cv::resize(ll_bin, ll_resized, original_frame.size(), 0, 0, cv::INTER_LINEAR);

    // Cria máscara colorida para overlay
    cv::Mat color_mask = cv::Mat::zeros(original_frame.size(), CV_8UC3);
    color_mask.setTo(cv::Scalar(255, 0, 255), ll_resized);

    // Overlay mais forte para destacar as linhas
    cv::Mat overlay;
    cv::addWeighted(original_frame, 0.5, color_mask, 1.0, 0, overlay);

    return overlay;
}



int main() {
    try {
        TensorRTInference trt("model.engine");
        CSICamera cam(480, 480, 30);
        cam.start();

        std::cout << "Pressione 'q' para sair" << std::endl;
        int frameCount = 0;
        auto start = std::chrono::steady_clock::now();

        while (true) {
            cv::Mat frame = cam.read();
            if (frame.empty()) continue;

            std::vector<float> input = preprocess_frame(frame);
            auto output = trt.infer(input);
            cv::Mat result = postprocess(output.data(), frame);

            cv::imshow("LineNet Lane Detection", result);

            if (cv::waitKey(1) == 'q') break;

            frameCount++;
            if (frameCount % 30 == 0) {
                auto now = std::chrono::steady_clock::now();
                double fps = 30.0 / std::chrono::duration<double>(now - start).count();
                std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;
                start = now;
            }
        }
        cam.stop();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
//gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink
//sudo systemctl restart nvargus-daemon

