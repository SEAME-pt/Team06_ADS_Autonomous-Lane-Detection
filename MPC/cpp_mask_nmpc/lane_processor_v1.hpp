#ifndef LANE_PROCESSOR_HPP
#define LANE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class LaneProcessor {
public:
    LaneProcessor(int Np, double dt);
    
    // Processa a máscara de linhas para gerar a trajetória de referência
    std::vector<std::vector<double>> process_lane(const cv::Mat& ll_mask);

private:
    int Np_;    // Horizonte de predição
    double dt_; // Passo de tempo
    
    // Extrai a linha central da máscara
    std::vector<std::vector<double>> extract_centerline(const cv::Mat& ll_mask);
};

#endif