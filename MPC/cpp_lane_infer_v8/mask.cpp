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

/* int MaskProcessor::firstWhite(const cv::Mat& row) {
    for (int x = 0; x < row.cols - 5; x++) {
        if (row.at<uchar>(0, x) == 255 && x > 5) return x;
        else if (row.at<uchar>(0, x) == 255 && x < 5) return -1;
    }
    return -1;
}

int MaskProcessor::lastWhite(const cv::Mat& row) {
    for (int x = row.cols - 1; x > 5; x--) {
        if (row.at<uchar>(0, x) == 255 && x < row.cols - 5) return x;
        else if (row.at<uchar>(0, x) == 255 && x > row.cols - 5) return -1;
    }
    return -1;
} */

int MaskProcessor::firstWhite(const cv::Mat& row) {
    for (int x = row.cols / 2 - 10; x > 5; x--) {
        if (row.at<uchar>(0, x) != 255) return x + 1;
    }
    return -1;
}

int MaskProcessor::lastWhite(const cv::Mat& row) {
    for (int x = row.cols / 2 + 10; x < row.cols - 5; x++) {
        if (row.at<uchar>(0, x) != 255) return x - 1;
    }
    return -1;
}


void MaskProcessor::processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints) {
    cv::Mat mask_bin = da_mask.clone();
    cv::threshold(da_mask, mask_bin, 127, 255, cv::THRESH_BINARY);

    /*area*/
    std::vector<cv::Point> left_edge_points;
    std::vector<cv::Point> right_edge_points;

    /*lanes*/
    std::vector<cv::Point> ll_left_points;  // Pontos de ll_mask para regressão esquerda
    std::vector<cv::Point> ll_right_points; // Pontos de ll_mask para regressão direita

    int height = mask_bin.rows;
    int width = mask_bin.cols;

    int top_y = height / 2, bottom_y = height * 0.95, y_step = 2;

    for (int y = top_y; y <= bottom_y; y += y_step) {
        const cv::Mat row = mask_bin.row(y);
        int left_x = firstWhite(row);
        int right_x = lastWhite(row);

        if ((left_x == -1 || right_x == -1) && y > top_y + 50) break;
        if (left_x != -1) left_edge_points.push_back(cv::Point(left_x, y));
        if (right_x != -1) right_edge_points.push_back(cv::Point(right_x, y));
    }

    // Coletar pontos de ll_mask no ROI para regressão
    for (int y = top_y; y <= bottom_y; y += y_step) {
        const cv::Mat row = ll_mask.row(y);
        int left_x = firstWhite(row);
        int right_x = lastWhite(row);

        if ((left_x == -1 || right_x == -1) && y < top_y + 50) continue;
        if (left_x == -1 || right_x == -1) break;
        ll_left_points.push_back(cv::Point(left_x, y));
        ll_right_points.push_back(cv::Point(right_x, y));
    }

    // Verify if de have Edges
    if (left_edge_points.size() < 10) {
        std::cerr << "[Warning] Left Edge Lost" << std::endl;
    } else if (right_edge_points.size() < 10) {
        std::cerr << "[Warning] Right Edge Lost" << std::endl;
    } else {
        int bottom_y = (left_edge_points.size() > right_edge_points.size()) 
            ? right_edge_points.back().y 
            : left_edge_points.back().y;
    }

    /********Linear Regression *********/
    LineCoef left_coeffs = linearRegression(left_edge_points);
    LineCoef right_coeffs = linearRegression(right_edge_points);
    LineCoef ll_left_coeffs = linearRegression(ll_left_points);
    LineCoef ll_right_coeffs = linearRegression(ll_right_points);

    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point> left_line_points, right_line_points;
    std::vector<cv::Point> ll_left_line_points, ll_right_line_points;
    if (left_coeffs.valid && right_coeffs.valid) {
        for (int y = top_y; y <= bottom_y; y += y_step) {
            int left_x = static_cast<int>(left_coeffs.m * y + left_coeffs.b);
            int right_x = static_cast<int>(right_coeffs.m * y + right_coeffs.b);

            if (left_x >= 0 && left_x < width) left_line_points.push_back(cv::Point(left_x, y));
            if (right_x >= 0 && right_x < width) right_line_points.push_back(cv::Point(right_x, y));
        }

        if (!left_line_points.empty() && !right_line_points.empty()) {
            cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
            cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
        }
        std::cout << "[" <<  __func__ <<"]" << std::endl
            << "median points back: " << left_line_points.back().x << " " << left_line_points.back().y << " -- " << 
                                        right_line_points.back().x << " " << right_line_points.back().y << std::endl
            << "median points front: " << left_line_points.front().x << " " << left_line_points.front().y << " -- "
                                        << right_line_points.front().x << " " << right_line_points.front().y << std::endl;
    }

/*     if (ll_left_coeffs.valid && ll_right_coeffs.valid) {
        for (int y = top_y; y <= bottom_y; y++) {
            int ll_left_x = static_cast<int>(ll_left_coeffs.m * y + ll_left_coeffs.b);
            int ll_right_x = static_cast<int>(ll_right_coeffs.m * y + ll_right_coeffs.b);
            if (ll_left_x >= 0 && ll_left_x < width) ll_left_line_points.push_back(cv::Point(ll_left_x, y));
            if (ll_right_x >= 0 && ll_right_x < width) ll_right_line_points.push_back(cv::Point(ll_right_x, y));
        }

        if (!ll_left_line_points.empty() && !ll_right_line_points.empty()) {
            cv::line(output, ll_left_line_points.front(), ll_left_line_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo
            cv::line(output, ll_right_line_points.front(), ll_right_line_points.back(), cv::Scalar(0, 255, 255), 2); // Amarelo
        }
    } */

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