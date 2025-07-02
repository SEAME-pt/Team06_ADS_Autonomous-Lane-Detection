#include "mask.hpp" // Assumindo que contém as definições necessárias



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

int MaskProcessor::findFirstWhite(const cv::Mat& row) {
    for (int x = 224; x >= 0; x--) {
        if (row.at<uchar>(0, x) != 255) return (x + 1);
        else if (x == 0) return x + 1;
    }
    return -1;
}

int MaskProcessor::findLastWhite(const cv::Mat& row) {
    for (int x = 224; x < row.cols; x++) {
        if (row.at<uchar>(0, x) != 255) return x - 1;
        else if (x == (row.cols - 1)) return x - 1;
    }
    return -1;
}


void displayMaskAndLines(const cv::Mat& da_mask, const cv::Mat& ll_mask, 
                                       const std::vector<cv::Point>& left_line_points,
                                       const std::vector<cv::Point>& right_line_points,
                                       const std::vector<cv::Point>& medianPoints) {
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

    int height = mask_bin.rows;
    int width = mask_bin.cols;
    const int roi_start_y = static_cast<int>(0.50 * height);
    const int roi_end_y = static_cast<int>(0.95 * height);
    const int y_step = 2; // Processar a cada 2 linhas para eficiência

    // Identificar limites verticais da pista no ROI
    int first_y = -1, last_y = -1;
    for (int y = roi_start_y; y <= roi_end_y; y += y_step) {
        const cv::Mat row = mask_bin.row(y);
        if (findFirstWhite(row) != -1) {
            if (first_y == -1) first_y = y;
            last_y = y;
        }
    }

    // Se não encontrou pista, limpar medianPoints e sair
    if (first_y == -1 || last_y == -1) {
        medianPoints.clear();
        cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);
        //displayMaskAndLines(mask_bin, ll_mask, {}, {}, medianPoints);
        return;
    }

    // Coletar pontos das bordas no ROI
    for (int y = first_y; y <= last_y; y += y_step) {
        const cv::Mat row = mask_bin.row(y);
        int left_x = findFirstWhite(row);
        int right_x = findLastWhite(row);

        if (left_x != -1) left_edge_points.push_back(cv::Point(left_x, y));
        if (right_x != -1) right_edge_points.push_back(cv::Point(right_x, y));
    }

    LineCoef left_coeffs = linearRegression(left_edge_points);
    LineCoef right_coeffs = linearRegression(right_edge_points);

    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

    // Processar apenas se ambas as linhas forem válidas
    std::vector<cv::Point> left_line_points, right_line_points;
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
    } else {
        medianPoints.clear();
    }

    // Passar ll_mask e pontos calculados para exibição
    //displayMaskAndLines(mask_bin, ll_mask, left_line_points, right_line_points, medianPoints);
}
