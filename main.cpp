#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>

// Logger personalizado para TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};

const std::string ENGINE_PATH = "unet_model_2.engine";
const int IMG_SIZE = 256;

std::vector<char> loadEngine(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.good())
        throw std::runtime_error("Erro ao abrir o ficheiro engine.");
    return std::vector<char>((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
}

void preprocess(const cv::Mat& img, std::vector<float>& input_data) {
    cv::Mat resized, float_img;
    cv::resize(img, resized, cv::Size(IMG_SIZE, IMG_SIZE));
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < IMG_SIZE; ++y)
            for (int x = 0; x < IMG_SIZE; ++x)
                input_data[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] =
                    float_img.at<cv::Vec3f>(y, x)[c];
}

cv::Mat postprocess(const std::vector<float>& output_data) {
    cv::Mat mask(IMG_SIZE, IMG_SIZE, CV_32F);
    for (int y = 0; y < IMG_SIZE; ++y)
        for (int x = 0; x < IMG_SIZE; ++x)
            mask.at<float>(y, x) = output_data[y * IMG_SIZE + x];
    cv::Mat mask_u8;
    mask *= 255;
    mask.convertTo(mask_u8, CV_8U);
    return mask;
}

int main() {
    // Inicializar o logger
    Logger logger;

    // Configurar pipeline GStreamer para captura de vídeo
    std::string pipeline =
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=400, height=400, "
        "format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, "
        "format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmara!" << std::endl;
        return -1;
    }

    // Carregar o modelo TensorRT
    auto engine_data = loadEngine(ENGINE_PATH);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Falha ao criar IRuntime" << std::endl;
        return -1;
    }

    nvinfer1::ICudaEngine* engine =
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine) {
        std::cerr << "Falha ao deserializar ICudaEngine" << std::endl;
        delete runtime;
        return -1;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Falha ao criar IExecutionContext" << std::endl;
        delete engine;
        delete runtime;
        return -1;
    }

    // Alocar buffers para entrada e saída
    void* buffers[2];
    std::vector<float> input_data(3 * IMG_SIZE * IMG_SIZE);
    std::vector<float> output_data(IMG_SIZE * IMG_SIZE);

    cudaMalloc(&buffers[0], input_data.size() * sizeof(float));
    cudaMalloc(&buffers[1], output_data.size() * sizeof(float));

    // Loop de inferência
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        preprocess(frame, input_data);
        cudaMemcpy(buffers[0], input_data.data(),
                   input_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        context->executeV2(buffers);

        cudaMemcpy(output_data.data(), buffers[1],
                   output_data.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cv::Mat mask = postprocess(output_data);
        // Depuração: Verificar tipo da imagem antes do redimensionamento
        std::cout << "Tipo de mask antes do resize: " << mask.type() << " (CV_8UC1 = " << CV_8UC1 << ")" << std::endl;

        // Redimensionar garantindo que o tipo permaneça CV_8UC1
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
        mask = resized_mask; // Substituir mask pela versão redimensionada

        // Depuração: Verificar tipo da imagem após o redimensionamento
        std::cout << "Tipo de mask após o resize: " << mask.type() << " (CV_8UC1 = " << CV_8UC1 << ")" << std::endl;

        // Garantir que mask é CV_8UC1 antes de applyColorMap
        if (mask.type() != CV_8UC1) {
            cv::Mat temp;
            mask.convertTo(temp, CV_8UC1);
            mask = temp;
            std::cout << "Converteu mask para CV_8UC1" << std::endl;
        }

        cv::Mat overlay;
        cv::applyColorMap(mask, overlay, cv::COLORMAP_JET);
        cv::addWeighted(frame, 0.6, overlay, 0.4, 0, overlay);

        cv::imshow("Mask", mask);
        cv::imshow("Frame", overlay);
        if (cv::waitKey(1) == 27) break; // ESC para sair
    }

    // Liberar recursos
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}