#include "mask.hpp"

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

/**********************************************/

int MaskProcessor::findFirstWhite(const cv::Mat& row, int center_x) {
    for (int x = center_x; x >= 0; x--) {
        if (row.at<uchar>(0, x) != 255) return (x + 1);
        else if (x == 0) return x + 1;
    }
    return -1;
}

int MaskProcessor::findLastWhite(const cv::Mat& row, int center_x) {
    int width = row.cols;
    for (int x = center_x; x < width; x++) {
        if (row.at<uchar>(0, x) != 255) return x - 1;
        else if (x == (width - 1)) return x - 1;
    }
    return -1;
}

/**********************************************/

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

    cv::imshow("Lane Detection", viz);
    cv::waitKey(1); // Atualiza a janela em tempo real
}

void MaskProcessor::processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints,
                                LineCoef& left_coeffs, LineCoef& right_coeffs) {
    cv::Mat mask_bin = da_mask.clone();
    cv::threshold(da_mask, mask_bin, 127, 255, cv::THRESH_BINARY);

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
        if (findFirstWhite(row, center_x) != -1) {
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

    // Coletar pontos das bordas no ROI para da_mask com maior sequência de brancos
    int prev_width = -1; // Armazena a largura da sequência anterior
    for (int y = first_y; y <= last_y; y += y_step) {
        const cv::Mat row = mask_bin.row(y);
        int max_seq_start = -1, max_seq_end = -1, max_seq_length = 0;
        int current_start = -1, current_length = 0;

        // Encontrar a maior sequência contínua de pixels brancos
        for (int x = 0; x < width; x++) {
            if (row.at<uchar>(0, x) == 255) {
                if (current_start == -1) current_start = x; // Início de uma nova sequência
                current_length++;
            } else {
                if (current_length > max_seq_length) {
                    max_seq_length = current_length;
                    max_seq_start = current_start;
                    max_seq_end = x - 1;
                }
                current_start = -1;
                current_length = 0;
            }
        }
        // Verificar o final da linha
        if (current_length > max_seq_length) {
            max_seq_length = current_length;
            max_seq_start = current_start;
            max_seq_end = width - 1;
        }

        // Verificar se a diferença de largura é abrupta (maior que 10 pixels)
        int current_width = (max_seq_end - max_seq_start + 1);
        if (prev_width != -1 && abs(current_width - prev_width) > 10) {
            continue; // Descartar esta linha se a mudança for abrupta
        }
        prev_width = current_width;

        // Guardar as coordenadas do primeiro e último pixel da maior sequência
        if (max_seq_start != -1 && max_seq_end != -1) {
            left_edge_points.push_back(cv::Point(max_seq_start, y));
            right_edge_points.push_back(cv::Point(max_seq_end, y));
        }
    }

    // Coletar pontos de ll_mask no ROI para regressão
    for (int y = first_y; y <= roi_end_y; y += y_step) {
        const cv::Mat row = ll_mask.row(y);
        int center_x = width / 2;
        int left_x = findFirstWhite(row, center_x);
        int right_x = findLastWhite(row, center_x);

        if (left_x != -1 && right_x != -1) {
            // Dividir pontos de ll_mask em esquerda e direita com base no centro
            if (left_x < center_x) ll_left_points.push_back(cv::Point(left_x, y));
            if (right_x > center_x) ll_right_points.push_back(cv::Point(right_x, y));
        }
    }

    left_coeffs = linearRegression(left_edge_points);
    right_coeffs = linearRegression(right_edge_points);
    
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

/*         if (!ll_left_line_points.empty() && !ll_right_line_points.empty()) {
            cv::line(output, ll_left_line_points.front(), ll_left_line_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo
            cv::line(output, ll_right_line_points.front(), ll_right_line_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo
        } */
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
    //displayMaskAndLines(mask_bin, ll_mask, left_line_points, right_line_points, medianPoints, ll_left_line_points, ll_right_points);
}