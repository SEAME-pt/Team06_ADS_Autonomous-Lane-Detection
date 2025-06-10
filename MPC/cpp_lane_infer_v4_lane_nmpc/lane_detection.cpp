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
    pipeline << "nvarguscamerasrc sensor-mode=4 ! "
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
                inputData[idx++] = resized.at<cv::Vec3b>(i, j)[2 - c] / 255.0f;
            }
        }
    }
    return inputData;
}

cv::Mat postprocess(float* da_output, float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints, LaneData& laneData) {
    const int height = original_frame.rows;
    const int width = original_frame.cols;

    int roi_start_y = static_cast<int>(0.50 * height); // 180
    int roi_end_y = static_cast<int>(0.95 * height);   // 342
    int roi_height = roi_end_y - roi_start_y;

    cv::Rect roi(0, roi_start_y, width, roi_height);

    cv::Mat da_logits(2, height * width, CV_32FC1, da_output);
    cv::Mat da_mask(height, width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < height * width; ++i) {
        float da0 = da_logits.at<float>(0, i);
        float da1 = da_logits.at<float>(1, i);
        da_mask.at<uchar>(i / width, i % width) = (da1 > da0) ? 255 : 0;
    }

    da_mask(cv::Rect(0, 0, width, roi_start_y)) = 0;
    da_mask(cv::Rect(0, roi_end_y, width, height - roi_end_y)) = 0;

    cv::Mat da_resized;
    cv::resize(da_mask, da_resized, original_frame.size());

    MaskProcessor processor;
    cv::Mat mask_output;
    processor.processMask(da_resized, mask_output, medianPoints);

    cv::Mat result_frame = original_frame.clone();

    laneData.valid = !medianPoints.empty();
    laneData.timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    laneData.num_points = 0;

    if (laneData.valid) {
        int step = medianPoints.size() > 10 ? medianPoints.size() / 10 : 1;
        for (size_t i = 0; i < medianPoints.size() && laneData.num_points < 10; i += step) {
            if (medianPoints[i].y >= roi_start_y && medianPoints[i].y <= roi_end_y) {
                laneData.points[laneData.num_points].x = 0.00617 * (medianPoints[i].x - 320);
                laneData.points[laneData.num_points].y = 0.00617 * (342 - medianPoints[i].y);
                laneData.num_points++;
            }
        }
    }

    if (!medianPoints.empty()) {
        std::vector<cv::Point> left_edge_points, right_edge_points;
        cv::Mat mask_bin = da_resized(roi);
        cv::threshold(mask_bin, mask_bin, 127, 255, cv::THRESH_BINARY);
        for (int y = 0; y < mask_bin.rows; y++) {
            const cv::Mat row = mask_bin.row(y);
            int left_x = -1, right_x = -1;
            for (int x = 0; x < row.cols; x++) {
                if (row.at<uchar>(0, x) == 255) {
                    left_x = x;
                    break;
                }
            }
            for (int x = row.cols - 1; x >= 0; x--) {
                if (row.at<uchar>(0, x) == 255) {
                    right_x = x;
                    break;
                }
            }
            if (left_x != -1) {
                left_edge_points.push_back(cv::Point(left_x, y + roi_start_y));
                right_edge_points.push_back(cv::Point(right_x, y + roi_start_y));
            }
        }

        LineCoefficients left_coeffs = processor.linearRegression(left_edge_points);
        LineCoefficients right_coeffs = processor.linearRegression(right_edge_points);

        if (left_coeffs.valid && right_coeffs.valid) {
            std::vector<cv::Point> left_line_points, right_line_points;
            for (int y = roi_start_y; y < roi_end_y; y++) {
                int left_x = static_cast<int>(left_coeffs.m * y + left_coeffs.b);
                int right_x = static_cast<int>(right_coeffs.m * y + right_coeffs.b);
                if (left_x >= 0 && left_x < result_frame.cols)
                    left_line_points.push_back(cv::Point(left_x, y));
                if (right_x >= 0 && right_x < result_frame.cols)
                    right_line_points.push_back(cv::Point(right_x, y));
            }

            if (!left_line_points.empty() && !right_line_points.empty()) {
                cv::line(result_frame, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2);
                cv::line(result_frame, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2);
            }

            std::vector<cv::Point> roi_median_points;
            for (const auto& p : medianPoints) {
                if (p.y >= roi_start_y && p.y < roi_end_y) {
                    roi_median_points.push_back(p);
                }
            }
            if (roi_median_points.size() >= 2) {
                cv::line(result_frame, roi_median_points.front(), roi_median_points.back(), cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    return result_frame;
}