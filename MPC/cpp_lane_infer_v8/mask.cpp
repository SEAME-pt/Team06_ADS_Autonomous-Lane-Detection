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

bool MaskProcessor::processEdges(const cv::Mat& mask_bin, std::vector<cv::Point>& left_edge, std::vector<cv::Point>& right_edge) {
    int height = mask_bin.rows, width = mask_bin.cols;
    int center_x = width / 2, margin = 5, range = 10;

    left_edge.clear();
    right_edge.clear();

    // Passo 1: Verificar se o pixel central na linha base é branco
    int base_y = static_cast<int>(height * 0.95) - 1;

    // Passo 2: Encontrar 
    // o último pixel branco na sequência para esquerda e direita
    int left_x = -1, right_x = -1;
    int left_start_limit = static_cast<int>(height * 0.75);
    int right_start_limit = static_cast<int>(height * 0.75);

    // Passo 3: Busca de baixo para cima nas linhas subsequentes
    for (int y = base_y; y >= height / 2; --y) {
        const cv::Mat row = mask_bin.row(y);
        int new_right_x = -1, new_left_x = -1;

        if (left_x == -1 && y > left_start_limit){
            for (int x = center_x; x >= margin; --x) {
                if (row.at<uchar>(0, x) == 255 && x < center_x - (center_x / 4)) {
                    left_x = x; 
                }
                else if (left_x != -1)
                    break;
            }
        }
        else if (left_x != -1){
            for (int x = std::min(center_x + 50, left_x + range); x >= std::max(left_x - range, margin); --x) {
                if (row.at<uchar>(0, x) != 255){
                    left_x = x;
                    break;  
                }
                else if (x == std::max(left_x - range, margin)){
                    left_x = left_x + 1;
                }
            }
        }
        
        if (right_x == -1 && y > right_start_limit){
            for (int x = center_x; x < width - margin; ++x) {
                if (row.at<uchar>(0, x) == 255 && x > center_x + (center_x / 4)) {
                    right_x = x; 
                }else if (right_x != -1)
                    break;
            }
        } else if (right_x != -1) {
            for (int x = std::max( center_x - 50, right_x - range); x <= std::min(width - margin, right_x + range) ; ++x) {
                if (row.at<uchar>(0, x) != 255){
                    right_x = x;
                    break;
                }
            }
        }

        if (left_x != -1) left_edge.emplace(left_edge.begin(), left_x, y);

        if (right_x != -1) right_edge.emplace(right_edge.begin(), right_x, y);

    }
    return true;
}


void MaskProcessor::processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, const cv::Mat& original_frame, std::vector<cv::Point>& medianPoints) {
    cv::Mat mask_bin = da_mask.clone();
    cv::threshold(da_mask, mask_bin, 127, 255, cv::THRESH_BINARY);

    /*area*/
    std::vector<cv::Point> left_edge_points, right_edge_points;

    /*lanes*/
    std::vector<cv::Point> ll_left_points, ll_right_points;

    int height = mask_bin.rows, width = mask_bin.cols;

    int top_y = height / 2, bottom_y = height * 0.95, y_step = 2;

    bool findEdges = processEdges(mask_bin, left_edge_points, right_edge_points);

    if (findEdges == false)
        std::cout << "Lane Not Found" << std::endl;

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
    }
    if (right_edge_points.size() < 10) {
        std::cerr << "[Warning] Right Edge Lost" << std::endl;
    }
    if (!left_edge_points.empty() && !right_edge_points.empty()) {
        bottom_y = (left_edge_points.size() > right_edge_points.size()) 
            ? right_edge_points.back().y 
            : left_edge_points.back().y;
    }

    /********Linear Regression *********/
    LineCoef left_coeffs, right_coeffs, ll_left_coeffs, ll_right_coeffs;

    std::vector<cv::Point> left_line_points = linearRegression(left_edge_points, top_y, bottom_y, width, left_coeffs);
    std::vector<cv::Point> right_line_points = linearRegression(right_edge_points, top_y, bottom_y, width, right_coeffs);
    std::vector<cv::Point> ll_left_line_points = linearRegression(ll_left_points, top_y, bottom_y, width, ll_left_coeffs);
    std::vector<cv::Point> ll_right_line_points = linearRegression(ll_right_points, top_y, bottom_y, width, ll_right_coeffs);

    //cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);
    cv::imshow("Lanes", mask_bin);


    output = original_frame.clone();
    if (left_coeffs.valid && right_coeffs.valid) {
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