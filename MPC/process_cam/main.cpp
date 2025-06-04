#include <iostream>
#include <chrono>
#include "trt_inference.hpp"
#include "csi_camera.hpp"
#include "mask_processor.hpp"

cv::Mat preprocessFrame(const cv::Mat& frame) {
    cv::Mat img;
    cv::resize(frame, img, cv::Size(640, 360));
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    std::vector<cv::Mat> reordered = {channels[2], channels[1], channels[0]}; // BGR to RGB
    cv::Mat chw;
    cv::merge(reordered, chw);
    return chw;
}

void postprocessOutputs(float* output1, float* output2, cv::Mat& frame, cv::Mat& da_mask, cv::Mat& ll_mask) {
    int h = frame.rows;
    int w = frame.cols;
    cv::Mat drivable_area(360, 640, CV_32F, output1);
    cv::Mat lane_lines(360, 640, CV_32F, output2);

    cv::Mat da_predict(360, 640, CV_8U);
    cv::Mat ll_predict(360, 640, CV_8U);

    for (int i = 0; i < 360; i++) {
        for (int j = 0; j < 640; j++) {
            da_predict.at<uchar>(i, j) = (drivable_area.at<float>(i, j) > 0.5) ? 255 : 0;
            ll_predict.at<uchar>(i, j) = (lane_lines.at<float>(i, j) > 0.5) ? 255 : 0;
        }
    }

    cv::resize(da_predict, da_mask, cv::Size(w, h));
    cv::resize(ll_predict, ll_mask, cv::Size(w, h));
}

int main() {
    TensorRTInference trt_inference("model.engine");
    CSICamera camera(640, 360, 30);
    camera.start();
    MaskProcessor processor;

    int fps_counter = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat frame = camera.read();
        if (frame.empty()) continue;

        cv::Mat input_data = preprocessFrame(frame);
        std::vector<float*> outputs = trt_inference.infer(input_data.ptr<float>());

        cv::Mat da_mask, ll_mask;
        postprocessOutputs(outputs[0], outputs[1], frame, da_mask, ll_mask);

        cv::Mat result = frame.clone();
        result.setTo(cv::Scalar(255, 0, 0), da_mask > 100);
        result.setTo(cv::Scalar(0, 255, 0), ll_mask > 100);

        cv::Mat mask_output;
        std::vector<cv::Point> median_points;
        processor.processMask(da_mask, mask_output, median_points); // Usar da_mask

        cv::addWeighted(result, 0.7, mask_output, 0.3, 0.0, result);

        fps_counter++;
        if (fps_counter % 30 == 0) {
            auto end_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            float fps = 30000.0f / elapsed;
            std::cout << "FPS: " << fps << std::endl;
            start_time = end_time;
        }

        cv::imshow("Lane Detection", result);
        if (cv::waitKey(1) == 'q') break;

        for (float* output : outputs) delete[] output;
    }

    camera.stop();
    cv::destroyAllWindows();
    return 0;
}