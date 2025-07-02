#include "mask.hpp"
#include <algorithm> // Para std::min_element

MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

LineCoef MaskProcessor::linearRegression(const std::vector<cv::Point>& points) {
    LineCoef coeffs = {0.0, 0.0, false};
    if (points.size() < 2) return coeffs;

    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumXX = 0.0;
    int n = points.size();

    for (const auto& p : points) {
        double x = p.y;
        double y = p.x;
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
    }

    double denominator = (n * sumXX - sumX * sumX);
    if (abs(denominator) < 1e-6) return coeffs;

    coeffs.m = (n * sumXY - sumX * sumY) / denominator;
    coeffs.b = (sumY * sumXX - sumX * sumXY) / denominator;
    coeffs.valid = true;
    return coeffs;
}

// vai do mais alto ao menor
int MaskProcessor::findFirstWhite(const cv::Mat& row, int first, int width) {
    for (int x = first; x >= 0; x--) { // Corrigido de x >= width para x >= 0
        if (row.at<uchar>(0, x) == 255) return x;
        else if (x == 0) return x;
    }
    return -1;
}

// vai do menor ao mais alto
int MaskProcessor::findLastWhite(const cv::Mat& row, int first, int width) {
    for (int x = first; x < width; x++) {
        if (row.at<uchar>(0, x) == 255) return x;
        else if (x == (width - 1)) return x;
    }
    return -1;
}

void displayMaskAndLines(const cv::Mat& da_mask, const cv::Mat& ll_mask, 
                        const std::vector<cv::Point>& left_line_points,
                        const std::vector<cv::Point>& right_line_points,
                        const std::vector<cv::Point>& medianPoints,
                        const std::vector<cv::Point>& ll_left_points,
                        const std::vector<cv::Point>& ll_right_points) {
    cv::Mat viz;
    cv::cvtColor(da_mask, viz, cv::COLOR_GRAY2BGR);

    // Desenhar bordas (vermelho e azul) e mediana (verde)
    if (!left_line_points.empty() && !right_line_points.empty()) {
        cv::line(viz, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
        cv::line(viz, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
    }
    if (!medianPoints.empty()) {
        cv::line(viz, medianPoints.front(), medianPoints.back(), cv::Scalar(0, 255, 0), 2); // Verde
    }

    // Desenhar regressões lineares de ll_mask (amarelo)
    if (!ll_left_points.empty()) {
        cv::line(viz, ll_left_points.front(), ll_left_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo para esquerda
    }
    if (!ll_right_points.empty()) {
        cv::line(viz, ll_right_points.front(), ll_right_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo para direita
    }

        // Desenhar linhas de ll_mask em amarelo
    int height = da_mask.rows;
    int width = da_mask.cols;
    int roi_start_y = static_cast<int>(0.50 * height);
    int roi_end_y = static_cast<int>(0.95 * height);
    for (int y = roi_start_y; y < roi_end_y; y++) {
        const cv::Mat row = ll_mask.row(y);
        for (int x = 0; x < width; x++) {
            if (row.at<uchar>(0, x) == 255)
                cv::circle(viz, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1); // Amarelo
        }
    }

    cv::imshow("Lane Detection", viz);
    cv::waitKey(1); // Atualiza a janela em tempo real
}

void MaskProcessor::processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints) {
    cv::Mat mask_bin = da_mask.clone();
    cv::threshold(mask_bin, mask_bin, 127, 255, cv::THRESH_BINARY);

    std::vector<cv::Point> left_edge_points;
    std::vector<cv::Point> right_edge_points;
    std::vector<cv::Point> ll_left_points;  // Pontos de ll_mask para regressão esquerda
    std::vector<cv::Point> ll_right_points; // Pontos de ll_mask para regressão direita

    int height = mask_bin.rows;
    int width = mask_bin.cols;
    const int roi_start_y = static_cast<int>(0.50 * height);
    const int roi_end_y = static_cast<int>(0.95 * height);
    const int y_step = 2; // Processar a cada 2 linhas para eficiência

    // Identificar limites verticais da pista no ROI
    int first_y = -1, last_y = -1;
    for (int y = roi_start_y; y <= roi_end_y; y += y_step) {
        const cv::Mat row = mask_bin.row(y);
        int center_x = width / 2;
        if (findFirstWhite(row, center_x, 0) != -1) {
            if (first_y == -1) first_y = y;
            last_y = y;
        }
    }

    // Se não encontrou pista, limpar medianPoints e sair
    if (first_y == -1 || last_y == -1) {
        medianPoints.clear();
        cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);
        return;
    }
    
    // Coletar pontos de ll_mask no ROI para regressão
    for (int y = first_y; y <= roi_end_y; y += y_step) {
        const cv::Mat row = ll_mask.row(y);
        int center_x = width / 2;
        int left_x = findFirstWhite(row, center_x, 0);
        int right_x = findLastWhite(row, center_x, width);

        if (left_x != -1 && right_x != -1) {
            // Dividir pontos de ll_mask em esquerda e direita com base no centro
            if (left_x < center_x) ll_left_points.push_back(cv::Point(left_x, y));
            if (right_x > center_x) ll_right_points.push_back(cv::Point(right_x, y));
        }
    }

    // Coletar pontos das bordas no ROI para da_mask, usando pontos de ll_mask como ponto de partida
    for (int y = first_y; y <= last_y; y += y_step) {
        const cv::Mat row = mask_bin.row(y); // Pega a linha y da imagem mask_bin (matriz binária de da_mask).
        int center_x = width / 2; // Define o centro horizontal da imagem como ponto de partida padrão.

        // Encontrar o ponto mais próximo em ll_left_points e ll_right_points para o y atual
        int ll_left_start = 0; // Inicializa o ponto de partida esquerdo como o centro.
        int ll_right_start = width - 1; // Inicializa o ponto de partida direito como o centro.
        for (const auto& p : ll_left_points) { // Itera sobre os pontos da borda esquerda de ll_mask.
            if (abs(p.y - y) < y_step) { // Verifica se o y do ponto está dentro de y_step (ex.: 2 pixels) do y atual.
                ll_left_start = p.x; // Se sim, usa o x desse ponto como novo ponto de partida esquerdo.
                break; // Sai do loop ao encontrar o primeiro ponto válido.
            }
        }
        for (const auto& p : ll_right_points) { // Itera sobre os pontos da borda direita de ll_mask.
            if (abs(p.y - y) < y_step) { // Verifica se o y do ponto está dentro de y_step do y atual.
                ll_right_start = p.x; // Se sim, usa o x desse ponto como novo ponto de partida direito.
                break; // Sai do loop ao encontrar o primeiro ponto válido.
            }
        }

        int right_x = findFirstWhite(row, ll_right_start, center_x); // Busca o primeiro pixel branco à direita a partir de ll_right_start.
        int left_x = findLastWhite(row, ll_left_start, center_x); // Busca o primeiro pixel branco à esquerda a partir de ll_left_start.

        if (left_x != -1) left_edge_points.push_back(cv::Point(left_x, y)); // Adiciona o ponto esquerdo à lista se encontrado.
        if (right_x != -1) right_edge_points.push_back(cv::Point(right_x, y)); // Adiciona o ponto direito à lista se encontrado.
    }

    LineCoef left_coeffs = linearRegression(left_edge_points);
    LineCoef right_coeffs = linearRegression(right_edge_points);
    LineCoef ll_left_coeffs = linearRegression(ll_left_points);
    LineCoef ll_right_coeffs = linearRegression(ll_right_points);

    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

    // Processar apenas se ambas as linhas forem válidas
    std::vector<cv::Point> left_line_points, right_line_points;
    std::vector<cv::Point> ll_left_line_points, ll_right_line_points;
    if (left_coeffs.valid && right_coeffs.valid) {
        for (int y = first_y; y <= last_y; y++) {
            int left_x = static_cast<int>(left_coeffs.m * y + left_coeffs.b);
            int right_x = static_cast<int>(right_coeffs.m * y + right_coeffs.b);
            if (left_x >= 0 && left_x < width) left_line_points.push_back(cv::Point(left_x, y));
            if (right_x >= 0 && right_x < width) right_line_points.push_back(cv::Point(right_x, y));
        }

        if (!left_line_points.empty() && !right_line_points.empty()) {
            cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
            cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
        }
    }

    if (ll_left_coeffs.valid && ll_right_coeffs.valid) {
        for (int y = first_y; y <= last_y; y++) {
            int ll_left_x = static_cast<int>(ll_left_coeffs.m * y + ll_left_coeffs.b);
            int ll_right_x = static_cast<int>(ll_right_coeffs.m * y + ll_right_coeffs.b);
            if (ll_left_x >= 0 && ll_left_x < width) ll_left_line_points.push_back(cv::Point(ll_left_x, y));
            if (ll_right_x >= 0 && ll_right_x < width) ll_right_line_points.push_back(cv::Point(ll_right_x, y));
        }

        if (!ll_left_line_points.empty() && !ll_right_line_points.empty()) {
            cv::line(output, ll_left_line_points.front(), ll_left_line_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo
            cv::line(output, ll_right_line_points.front(), ll_right_line_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo
        }
    }

    medianPoints.clear();
    for (int y = first_y; y <= last_y; y++) {
        int left_x = static_cast<int>(left_coeffs.m * y + left_coeffs.b);
        int right_x = static_cast<int>(right_coeffs.m * y + right_coeffs.b);
        int median_x = (left_x + right_x) / 2;
        if (median_x >= 0 && median_x < width) {
            medianPoints.push_back(cv::Point(median_x, y));
        }
    }

    if (!medianPoints.empty()) {
        cv::line(output, medianPoints.front(), medianPoints.back(), cv::Scalar(0, 255, 0), 2); // Verde
    }

    // Passar ll_mask e pontos calculados para exibição
    displayMaskAndLines(mask_bin, ll_mask, left_line_points, right_line_points, medianPoints, ll_left_line_points, ll_right_line_points);
}