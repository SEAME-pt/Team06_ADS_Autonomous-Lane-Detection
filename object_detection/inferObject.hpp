#ifndef INFEROBJECT_HPP
#define INFEROBJECT_HPP

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

class TensorRTYOLO {
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

    int input_size;
    int num_classes;
    size_t output_size;

    std::map<int, std::string> classes;
    std::map<int, cv::Scalar> colors;

public:
    TensorRTYOLO(const std::string& engine_path, int input_sz = 640) : input_size(input_sz) {
        classes = {
            {0, "STOP"}, {1, "YIELD"}, {2, "SPEED_50"}, {3, "SPEED_80"},
            {4, "LIGHT_RED"}, {5, "LIGHT_GREEN"},
            {6, "LIGHT_YELLOW"}, {7, "CROSSWALK"}, {8, "DANGER"}, {9, "DANGER_CURVE"}
        };
        colors = {
            {0, cv::Scalar(255, 0, 0)}, {1, cv::Scalar(0, 255, 0)}, {2, cv::Scalar(0, 0, 255)},
            {3, cv::Scalar(255, 255, 0)}, {4, cv::Scalar(255, 0, 255)}, {5, cv::Scalar(0, 255, 255)},
            {6, cv::Scalar(128, 0, 128)}, {7, cv::Scalar(0, 128, 255)}, {8, cv::Scalar(128, 128, 0)},
            {9, cv::Scalar(255, 165, 0)}
        };
        num_classes = static_cast<int>(classes.size());

        // Carregar engine
        std::ifstream engineFile(engine_path, std::ios::binary);
        if (!engineFile) {
            throw std::runtime_error("Erro ao carregar engine: " + engine_path);
        }
        engineFile.seekg(0, engineFile.end);
        size_t fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);

        std::vector<char> engineData(fsize);
        engineFile.read(engineData.data(), fsize);

        runtime = createInferRuntime(logger);
        if (!runtime) throw std::runtime_error("YOLO: createInferRuntime() returned null");

        engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
        if (!engine) throw std::runtime_error("YOLO: deserializeCudaEngine() failed");

        context = engine->createExecutionContext();
        if (!context) throw std::runtime_error("YOLO: createExecutionContext() failed");

        allocateBuffers();

        std::cout << "TensorRT Engine carregado: " << engine_path << std::endl;
        std::cout << "Input size: " << input_size << "x" << input_size << std::endl;
        std::cout << "Num classes: " << num_classes << std::endl;
        std::cout << "Output size: " << output_size << std::endl;
    }

    ~TensorRTYOLO() {
        for (auto& mem : inputBuffers) {
            cudaFree(mem.device);
            delete[] mem.host;
        }
        for (auto& mem : outputBuffers) {
            cudaFree(mem.device);
            delete[] mem.host;
        }
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();
    }

    void allocateBuffers() {
        int nbBindings = engine->getNbBindings();
        inputBuffers.resize(1);
        outputBuffers.resize(1);
        bindings.resize(nbBindings);

        for (int i = 0; i < nbBindings; ++i) {
            Dims dims = engine->getBindingDimensions(i);
            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                vol *= dims.d[j];
            }
            size_t typeSize = sizeof(float);
            size_t totalSize = vol * typeSize;

            void* deviceMem = nullptr;
            auto st = cudaMalloc(&deviceMem, totalSize);
            if (st != cudaSuccess) throw std::runtime_error(std::string("YOLO: cudaMalloc failed: ") + cudaGetErrorString(st));
            float* hostMem = new float[vol];

            bindings[i] = deviceMem;

            if (engine->bindingIsInput(i)) {
                inputBuffers[0] = {deviceMem, hostMem, totalSize};
            } else {
                outputBuffers = {deviceMem, hostMem, totalSize};
                output_size = vol;
            }
        }
    }

    std::vector<float> preprocess(const cv::Mat& image, float& scale, int& dw, int& dh) {
        int h = image.rows;
        int w = image.cols;

        scale = std::min(static_cast<float>(input_size) / w, static_cast<float>(input_size) / h);
        int nw = static_cast<int>(scale * w);
        int nh = static_cast<int>(scale * h);

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(nw, nh));

        cv::Mat padded = cv::Mat::ones(input_size, input_size, CV_8UC3) * 114;
        dw = (input_size - nw) / 2;
        dh = (input_size - nh) / 2;

        resized.copyTo(padded(cv::Rect(dw, dh, nw, nh)));
        padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(padded, channels);

        std::vector<float> inputData;
        inputData.reserve(3 * input_size * input_size);
        for (int i = 0; i < 3; ++i) {
            inputData.insert(inputData.end(), (float*)channels[i].datastart, (float*)channels[i].dataend);
        }
        return inputData;
    }

    std::vector<Detection> postprocess(const std::vector<float>& output, float scale, int dw, int dh,
                                       float conf_threshold = 0.3, float nms_threshold = 0.4) {
        std::vector<Detection> detections;
        try {
            // Assumindo formato YOLOv8: [batch, 4+num_classes, num_detections]
            int total_elements = static_cast<int>(output.size());
            int num_detections = total_elements / (4 + num_classes);

            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;

            for (int i = 0; i < num_detections; ++i) {
                float cx = output[i];
                float cy = output[num_detections + i];
                float width = output[2 * num_detections + i];
                float height = output[3 * num_detections + i];

                float max_confidence = 0.0f;
                int best_class_id = 0;

                for (int j = 0; j < num_classes; ++j) {
                    float class_conf = output[(4 + j) * num_detections + i];
                    if (class_conf > max_confidence) {
                        max_confidence = class_conf;
                        best_class_id = j;
                    }
                }

                if (max_confidence > conf_threshold) {
                    float x_center = (cx - dw) / scale;
                    float y_center = (cy - dh) / scale;
                    float w = width / scale;
                    float h = height / scale;

                    float x1 = x_center - w / 2.0f;
                    float y1 = y_center - h / 2.0f;

                    if (x1 >= 0 && y1 >= 0 && w > 0 && h > 0) {
                        boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                                 static_cast<int>(w), static_cast<int>(h)));
                        confidences.push_back(max_confidence);
                        class_ids.push_back(best_class_id);
                    }
                }
            }

            if (!boxes.empty()) {
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

                for (int idx : indices) {
                    Detection det;
                    det.class_id = class_ids[idx];
                    det.class_name = classes.count(det.class_id) ? classes[det.class_id] : std::string("unknown");
                    det.confidence = confidences[idx];
                    det.bbox = boxes[idx];
                    detections.push_back(det);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Erro no postprocessing: " << e.what() << std::endl;
        }
        return detections;
    }

    std::vector<Detection> infer(const cv::Mat& image) {
        try {
            float scale;
            int dw, dh;
            std::vector<float> input_data = preprocess(image, scale, dw, dh);

            std::copy(input_data.begin(), input_data.end(), inputBuffers[0].host);
            cudaMemcpy(inputBuffers.device, inputBuffers.host, inputBuffers.size, cudaMemcpyHostToDevice);

            context->executeV2(bindings.data());
            auto err = cudaPeekAtLastError();
            if (err != cudaSuccess) throw std::runtime_error(std::string("YOLO CUDA kernel error: ") + cudaGetErrorString(err));
            auto syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess) throw std::runtime_error(std::string("YOLO CUDA sync error: ") + cudaGetErrorString(syncErr));

            cudaMemcpy(outputBuffers[0].host, outputBuffers.device, outputBuffers.size, cudaMemcpyDeviceToHost);

            std::vector<float> output(outputBuffers.host, outputBuffers.host + output_size);
            return postprocess(output, scale, dw, dh, 0.6f, 0.4f);
        } catch (const std::exception& e) {
            std::cerr << "Erro durante inferência: " << e.what() << std::endl;
            return {};
        }
    }

    cv::Scalar getColor(int class_id) {
        return colors.count(class_id) ? colors[class_id] : cv::Scalar(255, 255, 255);
    }
};

#endif
