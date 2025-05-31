#include "lane_processor.hpp"
#include <numeric> // Adicionado para std::accumulate
#include <cmath>

LaneProcessor::LaneProcessor(int Np, double dt) : Np_(Np), dt_(dt) {}

std::vector<std::vector<double>> LaneProcessor::process_lane(const cv::Mat& ll_mask) {
    // Extrai a linha central da máscara
    auto centerline = extract_centerline(ll_mask);
    
    // Gera trajetória de referência
    std::vector<std::vector<double>> x_ref(Np_);
    for (int k = 0; k < Np_; ++k) {
        // Suponha que a linha central seja uma parábola para testes
        double x = k * dt_ * 0.5; // Velocidade média de 0.5 m/s
        double y = 0.01 * x * x;  // Trajetória parabólica (teste)
        double psi = std::atan2(2 * 0.01 * x, 1); // Orientação
        x_ref[k] = {x, y, psi, 0.5}; // Velocidade constante de 0.5 m/s
    }
    return x_ref;
}

std::vector<std::vector<double>> LaneProcessor::extract_centerline(const cv::Mat& ll_mask) {
    // Processa a máscara binária (255 para linhas, 0 para fundo)
    std::vector<std::vector<double>> centerline;
    // Para cada linha da imagem, encontra a média dos pixels brancos
    for (int y = 0; y < ll_mask.rows; ++y) {
        std::vector<int> x_coords;
        for (int x = 0; x < ll_mask.cols; ++x) {
            if (ll_mask.at<uint8_t>(y, x) > 100) {
                x_coords.push_back(x);
            }
        }
        if (!x_coords.empty()) {
            double x_mean = std::accumulate(x_coords.begin(), x_coords.end(), 0.0) / x_coords.size();
            centerline.push_back({x_mean, static_cast<double>(y)});
        }
    }
    // Simplificação: retorna apenas alguns pontos para teste
    return centerline;
}