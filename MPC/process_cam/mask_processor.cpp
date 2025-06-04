#include "mask_processor.hpp"
#include <iostream>

MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

LineCoefficients MaskProcessor::linearRegression(const std::vector<cv::Point>& points) {
    LineCoefficients coeffs = {0.0, 0.0, false};
    if (points.size() < 2) return coeffs;

    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumXX = 0.0;
    int n = points.size();

    for (const auto& p : points) {
        double x = p.x; // Usar x como variável independente (horizontal)
        double y = p.y; // Usar y como dependente (vertical)
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
    }

    double denominator = (n * sumXX - sumX * sumX);
    if (std::abs(denominator) < 1e-6) {
        std::cout << "Regressão linear falhou: denominador próximo de zero" << std::endl;
        return coeffs;
    }

    coeffs.m = (n * sumXY - sumX * sumY) / denominator;
    coeffs.b = (sumY * sumXX - sumX * sumXY) / denominator;
    coeffs.valid = true;

    // Depuração: exibir coeficientes
    std::cout << "Regressão linear: m = " << coeffs.m << ", b = " << coeffs.b << std::endl;

    return coeffs;
}

int MaskProcessor::findFirstWhite(const cv::Mat& row) {
    for (int x = 0; x < row.cols; x++) {
        if (row.at<uchar>(0, x) == 255) return x;
    }
    return -1;
}

int MaskProcessor::findLastWhite(const cv::Mat& row) {
    for (int x = row.cols - 1; x >= 0; x--) {
        if (row.at<uchar>(0, x) == 255) return x;
    }
    return -1;
}

void MaskProcessor::processMask(const cv::Mat& mask, cv::Mat& output, 
                               std::vector<cv::Point>& medianPoints,
                               std::vector<cv::Point>& left_line_points,
                               std::vector<cv::Point>& right_line_points) {
    // Binarizar a máscara
    cv::Mat mask_bin;
    cv::threshold(mask, mask_bin, 127, 255, cv::THRESH_BINARY);

    // Vetores para armazenar os pontos das bordas
    std::vector<cv::Point> left_edge_points;
    std::vector<cv::Point> right_edge_points;

    // Dimensões da imagem
    int height = mask_bin.rows;
    int width = mask_bin.cols;

    // Determinar os limites superior e inferior da zona branca
    int top_y = -1, bottom_y = -1;
    for (int y = 0; y < height; y++) {
        const cv::Mat row = mask_bin.row(y);
        if (findFirstWhite(row) != -1) {
            if (top_y == -1) top_y = y;
            bottom_y = y;
        }
    }

    // Depuração: exibir limites
    std::cout << "Limites: top_y = " << top_y << ", bottom_y = " << bottom_y << std::endl;

    // Percorrer cada linha dentro dos limites
    for (int y = top_y; y <= bottom_y; y++) {
        const cv::Mat row = mask_bin.row(y);
        int left_x = findFirstWhite(row);
        int right_x = findLastWhite(row);

        if (left_x != -1) left_edge_points.push_back(cv::Point(left_x, y));
        if (right_x != -1) right_edge_points.push_back(cv::Point(right_x, y));
    }

    // Aplicar regressão linear
    LineCoefficients left_coeffs = linearRegression(left_edge_points);
    LineCoefficients right_coeffs = linearRegression(right_edge_points);

    // Preparar a imagem de saída
    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

    // Limpar vetores de saída
    left_line_points.clear();
    right_line_points.clear();
    medianPoints.clear();

    if (left_coeffs.valid && right_coeffs.valid) {
        // Calcular os pontos das retas dentro dos limites
        for (int x = 0; x < width; x++) {
            int left_y = static_cast<int>(left_coeffs.m * x + left_coeffs.b);
            int right_y = static_cast<int>(right_coeffs.m * x + right_coeffs.b);
            if (left_y >= top_y && left_y <= bottom_y && left_y >= 0 && left_y < height) {
                left_line_points.push_back(cv::Point(x, left_y));
            }
            if (right_y >= top_y && right_y <= bottom_y && right_y >= 0 && right_y < height) {
                right_line_points.push_back(cv::Point(x, right_y));
            }
        }

        // Desenhar as linhas ajustadas na imagem de saída
        if (!left_line_points.empty() && !right_line_points.empty()) {
            cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
            cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
        }

        // Calcular a mediana (trajetória central)
        for (int x = 0; x < width; x++) {
            int left_y = static_cast<int>(left_coeffs.m * x + left_coeffs.b);
            int right_y = static_cast<int>(right_coeffs.m * x + right_coeffs.b);
            int median_y = (left_y + right_y) / 2;
            if (median_y >= top_y && median_y <= bottom_y && median_y >= 0 && median_y < height) {
                medianPoints.push_back(cv::Point(x, median_y));
            }
        }

        // Desenhar a mediana (em verde) na imagem de saída
        if (!medianPoints.empty()) {
            cv::line(output, medianPoints.front(), medianPoints.back(), cv::Scalar(0, 255, 0), 2); // Verde
        }
    }

    // Salvar a imagem com as retas
    cv::imwrite("mask_with_lines.png", output);
}