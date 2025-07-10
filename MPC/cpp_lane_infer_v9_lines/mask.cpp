#include "mask.hpp"


MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

std::vector<cv::Point> MaskProcessor::linearRegression(const std::vector<cv::Point>& points, int top_y, int bottom_y, int width, LineCoef& coeffs) {
    coeffs = {0.0, 0.0, false};
    //if (points.size() < 2) return coeffs;
    std::vector<cv::Point> edge;
    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumXX = 0.0;
    int n = points.size();
    int y_step = 2;

    for (const auto& p : points) {
        double x = p.y;
        double y = p.x;
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
    }

    double denominator = (n * sumXX - sumX * sumX);
    if (abs(denominator) < 1e-6) {
        return edge;
    }

    coeffs.m = (n * sumXY - sumX * sumY) / denominator;
    coeffs.b = (sumY * sumXX - sumX * sumXY) / denominator;
    coeffs.valid = true;

    for (int y = top_y; y <= bottom_y; y += y_step) {
        int point_x = static_cast<int>(coeffs.m * y + coeffs.b);

        if (point_x >= 0 && point_x < width) edge.push_back(cv::Point(point_x, y));
    }
    return edge;
}

int MaskProcessor::firstWhite(const cv::Mat& row) {
    for (int x = row.cols / 2 - 10; x > 5; x--) {
        if (row.at<uchar>(0, x) == 255) return x;
    }
    return -1;
}

int MaskProcessor::lastWhite(const cv::Mat& row) {
    for (int x = row.cols / 2 + 10; x < row.cols - 5; x++) {
        if (row.at<uchar>(0, x) == 255) return x;
    }
    return -1;
}


void MaskProcessor::processMask(const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints) {
    cv::Mat mask_bin = ll_mask.clone();
    cv::threshold(ll_mask, mask_bin, 127, 255, cv::THRESH_BINARY);
    /*lanes*/
    std::vector<cv::Point> left_edge_points, right_edge_points;

    int height = mask_bin.rows, width = mask_bin.cols;

    int top_y = height / 2, bottom_y = height * 0.95, y_step = 2;

    // Coletar pontos de ll_mask no ROI para regressão
    for (int y = top_y; y <= bottom_y; y += y_step) {
        const cv::Mat row = ll_mask.row(y);
        int left_x = firstWhite(row);
        int right_x = lastWhite(row);

        
        if ((left_x == -1 || right_x == -1) && y < top_y + 50) continue;
        if (left_x == -1 || right_x == -1) break;
        left_edge_points.push_back(cv::Point(left_x, y));
        right_edge_points.push_back(cv::Point(right_x, y));
    }
    
    
    // Verify if de have Edges
    if (left_edge_points.size() < 10) {
        std::cerr << "[Warning] Left Edge Lost" << std::endl;
    } else if (right_edge_points.size() < 10) {
        std::cerr << "[Warning] Right Edge Lost" << std::endl;
    } else {
        bottom_y = (left_edge_points.size() > right_edge_points.size()) 
        ? right_edge_points.back().y 
        : left_edge_points.back().y;
    }
    
    /********Linear Regression *********/
    LineCoef left_coeffs, right_coeffs;
    
    std::vector<cv::Point> left_line_points = linearRegression(left_edge_points, top_y, bottom_y, width, left_coeffs);
    std::vector<cv::Point> right_line_points = linearRegression(right_edge_points, top_y, bottom_y, width, right_coeffs);
    
/*     std::cout << "left          " << left_line_points << std::endl;
    std::cout << "right         " << right_line_points << std::endl; */

    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

    if (left_coeffs.valid && right_coeffs.valid) {
        if (!left_line_points.empty() && !right_line_points.empty()) {

            std::cout << "      entrou         " << std::endl;
            
            cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
            cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
        }
        std::cout << "[" <<  __func__ <<"]" << std::endl
            << "median points back: " << left_line_points.back().x << " " << left_line_points.back().y << " -- " << 
                                        right_line_points.back().x << " " << right_line_points.back().y << std::endl
            << "median points front: " << left_line_points.front().x << " " << left_line_points.front().y << " -- "
                                        << right_line_points.front().x << " " << right_line_points.front().y << std::endl;
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

    // Passar ll_mask e pontos calculados para exibição
    //displayMaskAndLines(mask_bin, ll_mask, left_line_points, right_line_points, medianPoints, ll_left_line_points, ll_right_points);
}