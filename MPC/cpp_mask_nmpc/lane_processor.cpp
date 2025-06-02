#include "lane_processor.hpp"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>

LaneProcessor::LaneProcessor(int Np, double dt) : Np_(Np), dt_(dt) {}

std::vector<std::vector<double>> LaneProcessor::process_lane(const cv::Mat& ll_mask) {
    auto centerline = extract_centerline(ll_mask);
    
    std::vector<std::vector<double>> x_ref(Np_);
    double scale = 0.01; // 1 pixel = 0.01 metros
    for (int k = 0; k < Np_; ++k) {
        if (k < centerline.size()) {
            double x_world = (centerline[k][1] - ll_mask.rows / 2.0) * scale; // Longitudinal
            double y_world = (ll_mask.cols / 2.0 - centerline[k][0]) * scale; // Transversal
            double psi = (k < centerline.size() - 1) ? 
                         std::atan2(centerline[k+1][0] - centerline[k][0], 
                                    centerline[k+1][1] - centerline[k][1]) : 0.0;
            x_ref[k] = {x_world, y_world, psi, 0.5}; // Velocidade constante
        } else {
            double x_world = (centerline.back()[1] - ll_mask.rows / 2.0) * scale;
            double y_world = (ll_mask.cols / 2.0 - centerline.back()[0]) * scale;
            x_ref[k] = {x_world, y_world, 0.0, 0.5};
        }
    }
    return x_ref;
}

std::vector<std::vector<double>> LaneProcessor::extract_centerline(const cv::Mat& ll_mask) {
    cv::Mat blurred;
    cv::GaussianBlur(ll_mask, blurred, cv::Size(7, 7), 0); // Suavização mais forte
    
    std::vector<std::vector<double>> centerline;
    for (int y = ll_mask.rows - 1; y >= ll_mask.rows / 2; y -= 5) { // Reduz intervalo
        std::vector<int> x_coords;
        for (int x = 0; x < ll_mask.cols; ++x) {
            if (blurred.at<uint8_t>(y, x) > 100) {
                x_coords.push_back(x);
            }
        }
        if (!x_coords.empty() && x_coords.size() > 2) {
            std::sort(x_coords.begin(), x_coords.end());
            int mid_idx = x_coords.size() / 2;
            double x_mean = x_coords[mid_idx]; // Usa mediana
            centerline.push_back({x_mean, static_cast<double>(y)});
        }
    }
    return centerline;
}