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

int MaskProcessor::findFirstWhite(const cv::Mat& row) {
    for (int x = 0; x < row.cols - 5; x++) {
        if (row.at<uchar>(0, x) == 255 && x > 5) return x;
        else if (row.at<uchar>(0, x) == 255 && x < 5) return -1;
    }
    return -1;
}

int MaskProcessor::findLastWhite(const cv::Mat& row) {
    for (int x = row.cols - 1; x > 5; x--) {
        if (row.at<uchar>(0, x) == 255 && x < row.cols - 5) return x;
        else if (row.at<uchar>(0, x) == 255 && x > row.cols - 5) return -1;
    }
    return -1;
}

void MaskProcessor::processMask(const cv::Mat& mask, cv::Mat& output, std::vector<cv::Point>& medianPoints) {
    cv::Mat mask_bin;
    cv::threshold(mask, mask_bin, 127, 255, cv::THRESH_BINARY);

    std::vector<cv::Point> left_edge_points;
    std::vector<cv::Point> right_edge_points;

    int height = mask_bin.rows;
    int width = mask_bin.cols;

    int top_y = -1, bottom_y = -1;
    for (int y = 0; y < height; y++) {
        const cv::Mat row = mask_bin.row(y);
        if (findFirstWhite(row) != -1) {
            if (top_y == -1) top_y = y;
            bottom_y = y;
        }
    }

    for (int y = top_y; y <= bottom_y; y++) {
        const cv::Mat row = mask_bin.row(y);
        int left_x = findFirstWhite(row);
        int right_x = findLastWhite(row);

        if (left_x == -1 || right_x == -1) break;
        if (left_x != -1) left_edge_points.push_back(cv::Point(left_x, y));
        if (right_x != -1) right_edge_points.push_back(cv::Point(right_x, y));
    }

    LineCoef left_coeffs = linearRegression(left_edge_points);
    LineCoef right_coeffs = linearRegression(right_edge_points);

    std::cout << " left coeffs: " << left_edge_points.size() << std::endl;
    std::cout << "right coeffs: " << right_edge_points.size() << std::endl;


    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

    if (left_coeffs.valid && right_coeffs.valid) {
        std::vector<cv::Point> left_line_points, right_line_points;
        for (int y = top_y; y <= bottom_y; y++) {
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
        for (int y = top_y; y <= bottom_y; y++) {
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
    }
}