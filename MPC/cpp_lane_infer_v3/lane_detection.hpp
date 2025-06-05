#ifndef LANE_DETECTION_HPP
#define LANE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <thread>
#include <atomic>

using namespace nvinfer1;

// Logger para mensagens do TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// Classe para realizar inferência com modelo TensorRT
class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path); // Construtor: carrega o modelo .engine
    ~TensorRTInference(); // Destrutor: libera memória

    // Estrutura para buffers de entrada/saída
    struct Buffer {
        void* device; // Memória na GPU
        float* host;  // Memória na CPU
        size_t size;  // Tamanho do buffer
    };

    // Realiza inferência no modelo com dados de entrada
    std::vector<std::vector<float>> infer(const std::vector<float>& inputData);

private:
    Logger logger; // Logger para mensagens do TensorRT
    IRuntime* runtime = nullptr; // Runtime do TensorRT
    ICudaEngine* engine = nullptr; // Motor do modelo
    IExecutionContext* context = nullptr; // Contexto de execução

    std::vector<Buffer> inputBuffers; // Buffers de entrada
    std::vector<Buffer> outputBuffers; // Buffers de saída
    std::vector<void*> bindings; // Bindings para TensorRT

    void allocateBuffers(); // Aloca buffers para entrada/saída
};

// Classe para captura de vídeo da câmera CSI no Jetson
class CSICamera {
public:
    CSICamera(int width, int height, int fps); // Construtor: configura pipeline GStreamer
    void start(); // Inicia thread de captura
    void stop(); // Para thread e libera recursos
    cv::Mat read() const; // Lê o frame atual

private:
    cv::VideoCapture cap; // Objeto OpenCV para captura
    cv::Mat frame; // Frame atual
    std::thread thread; // Thread para captura contínua
    std::atomic<bool> running{false}; // Controle de execução

    void update(); // Função executada na thread para atualizar frames
};

// Pré-processa o frame da câmera para entrada do modelo
std::vector<float> preprocess_frame(const cv::Mat& frame);

// Pós-processa as saídas do modelo, retornando apenas o frame com linhas
cv::Mat postprocess(float* da_output, float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints);

#endif