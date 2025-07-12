#include "mask.hpp"


MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

std::vector<cv::Point> MaskProcessor::linearRegression(const std::vector<cv::Point>& points, int top_y, int bottom_y, int width, LineCoef& coeffs) {
    coeffs = {0.0, 0.0, false};
    std::vector<cv::Point> edge;
    if (points.size() < 2) {
        std::cout << "                                                  no points "<< std::endl;
        return edge;
    }
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

bool MaskProcessor::processEdges(const cv::Mat& mask_bin, std::vector<cv::Point>& left_edge, std::vector<cv::Point>& right_edge) {
    int height = mask_bin.rows, width = mask_bin.cols;
    int center_x = width / 2, margin = 5, range = 20;

    left_edge.clear();
    right_edge.clear();

    // Passo 1: Verificar se o pixel central na linha base é branco
    int base_y = static_cast<int>(height * 0.95) - 1;

    // Passo 2: Encontrar 
    // o último pixel branco na sequência para esquerda e direita
    int left_x = -1, right_x = -1;
    int left_contig = -1, right_contig = -1;
    int left_count = 0, right_count = 0;
    int left_start_limit = static_cast<int>(height * 0.75);
    int right_start_limit = static_cast<int>(height * 0.66);
    bool reset_left = false, reset_right = false;

    // Passo 3: Busca de baixo para cima nas linhas subsequentes
    for (int y = base_y; y >= height / 2; --y) {
        const cv::Mat row = mask_bin.row(y);
        int new_right_x = -1, new_left_x = -1;

        if (left_x == -1 && y > left_start_limit){
            //  std::cout << "          entrou no primeiro" << std::endl;
            for (int x = center_x; x >= margin; --x) {
                if (row.at<uchar>(x) == 255 && x < center_x - (center_x / 4)) {
                    left_x = x; 
                    left_contig ++;
                    left_count ++;
                    break;
                }
            }
        }
        else if (left_x != -1 && left_contig != -1 && left_contig <= 10){
            for (int x = std::min(left_x + range, center_x + 100); x >= std::max(margin, left_x - range); --x) {
                if (row.at<uchar>(x) == 255) {
                    new_left_x = x; 
                    left_count ++;
                    break; 
                }
            }
            if (new_left_x != -1) left_x = new_left_x;
            else left_contig++;
        }
        if (left_contig > 10 && left_count < 30) {
            std::cout << " reset left" << std::endl;
            left_x = -1;
            left_contig = -1;
            left_count = 0;
            reset_left = true;
        }
        
        if (right_x == -1 && y > right_start_limit){
            for (int x = center_x; x < width - margin; ++x) {
                if (row.at<uchar>(x) == 255 && x > center_x + (center_x / 4)) {
                    right_x = x; 
                    right_contig++;
                    right_count++;
                    break;
                }
            }
        } else if (right_x != -1 && right_contig != -1 && right_contig <= 10) {
            for (int x = std::max(center_x, right_x - range); x <= std::min( width - margin, right_x + range); ++x) {
                if (row.at<uchar>(x) == 255) {
                    new_right_x = x;
                    right_count++;
                    break;
                }
            }
            if (new_right_x != -1) right_x = new_right_x;
            else right_contig++;
        }
        if (right_contig > 10 && right_count < 30) {
            right_x = -1;
            right_contig = -1;
            right_count = 0;
            reset_right = true;
        }

        // Atualiza as coordenadas e adiciona os pontos
        if (left_x != -1 && left_contig <= 10 && left_contig != -1) left_edge.emplace(left_edge.begin(), left_x, y);
        if (reset_left == true)  left_edge.clear();

        if (right_x != -1 && right_contig <= 10 && right_contig != -1) right_edge.emplace(right_edge.begin(), right_x, y);
        if (reset_right == true)  right_edge.clear();

    }
    return true;
}


void MaskProcessor::processMask(const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints) {
    cv::Mat mask_bin = ll_mask.clone();
    cv::threshold(ll_mask, mask_bin, 127, 255, cv::THRESH_BINARY);
    /*lanes*/
    std::vector<cv::Point> left_edge_points, right_edge_points;

    int height = mask_bin.rows, width = mask_bin.cols;

    int top_y = height / 2, bottom_y = height * 0.95, y_step = 2;

    bool findEdges = processEdges(mask_bin, left_edge_points, right_edge_points);

    const double a = -0.00000362, b = 0.00221;
    double x_px = 0, sy = 0;
    double d = 0.26;

    if (findEdges == false)
        std::cout << "Lane Not Found" << std::endl;
    
    // Verify if de have Edges
    if (left_edge_points.size() < 10) {
        std::cerr << "[Warning] Left Edge Lost" << std::endl;
    }
    if (right_edge_points.size() < 10) {
        std::cerr << "[Warning] Right Edge Lost" << std::endl;
    } 
/*     if (!left_edge_points.empty() && !right_edge_points.empty()) {
        bottom_y = (left_edge_points.size() > right_edge_points.size()) 
        ? right_edge_points.back().y 
        : left_edge_points.back().y;
    }   */  
    if (left_edge_points.empty() && right_edge_points.empty()){
        std::cerr << "[Warning Break!!!!!!!!!!!!] Edges Lost" << std::endl;

        return;
    } 

    int bottom_left = bottom_y, bottom_right = bottom_y;
    int top_left = top_y , top_right = top_y;
    if (!left_edge_points.empty()){
        bottom_left = left_edge_points.back().y;
        std::cout << " encontrou a esquerda" << std::endl;
    }
    else if (left_edge_points.empty() && !right_edge_points.empty()){
        std::cout << "              Entrou na excepção    left!!!" << std::endl;
        auto it = right_edge_points.begin();
        for (int y = it->y; it != right_edge_points.end(); ++it, y = it->y) {
            sy = a * y + b;
            x_px = d / sy;
            int left_x = it->x - x_px;
            left_edge_points.emplace_back(left_x, y);
        }
        bottom_left = right_edge_points.back().y;
        top_left = right_edge_points.front().y;
    }
    if (!right_edge_points.empty()) {
        std::cout << " encontrou a direita" << std::endl;
        bottom_right = right_edge_points.back().y;
    }
    else if (right_edge_points.empty() && !left_edge_points.empty()){
        std::cout << "              Entrou na excepção  right!!!" << std::endl;
        
        auto it = left_edge_points.begin();
        for (int y = it->y; it != left_edge_points.end(); ++it, y = it->y) {
            sy = a * y + b;
            x_px = d / sy;
            int right_x = it->x + x_px;
            right_edge_points.emplace_back(right_x, y);
        }
        bottom_right = left_edge_points.back().y;
        top_right = left_edge_points.front().y;

    }
    std::cout << "right   back      " << right_edge_points.back() << std::endl;
    std::cout << "left    back     " << left_edge_points.back() << std::endl;    
    std::cout << "right   front      " << right_edge_points.front() << std::endl;
    std::cout << "left    front     " << left_edge_points.front() << std::endl;
    //std::cout << "right   size      " << right_edge_points << std::endl;
    //std::cout << "left   size      " << left_edge_points << std::endl;


    //std::cout << " checking linearRegression " << std::endl;
    /********Linear Regression *********/
    LineCoef left_coeffs, right_coeffs;
    
    std::vector<cv::Point> left_line_points = linearRegression(left_edge_points, top_left, bottom_left, width, left_coeffs);
    std::vector<cv::Point> right_line_points = linearRegression(right_edge_points, top_right, bottom_right, width, right_coeffs);

/*     std::cout << std::endl << "Linear left      back    " << left_line_points.back() << std::endl;
    std::cout << "Linear left      front    " << left_line_points.front() << std::endl;
    std::cout << "Linear right     back    " << right_line_points.back() << std::endl;
    std::cout << "Linear right     front    " << right_line_points.front() << std::endl << std::endl; */

    //std::cout << "Linear left      front    " << left_line_points << std::endl;
    //std::cout << "Linear right      front    " << right_line_points << std::endl;

/*     if (!left_edge_points.empty()){
        cv::Point point_back (left_edge_points.back().x, left_edge_points.back().y);
        cv::Point point_front (left_edge_points.front().x, left_edge_points.front().y);
        cv::circle(output, point_back, 5, cv::Scalar(0,0,0), -1 );
        cv::circle(output, point_front, 5, cv::Scalar(0,0,0), -1 );
    } */
    
    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);
    if (!left_line_points.empty()){
        cv::Point point_left_back (left_line_points.back().x, left_line_points.back().y);
        cv::Point point_left_front (left_line_points.front().x, left_line_points.front().y);
        cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
        cv::circle(output, point_left_back, 5, cv::Scalar(255,100,0), -1 );
        cv::circle(output, point_left_front, 5, cv::Scalar(200,100,0), -1 );
    }
    if (!right_line_points.empty())
        cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
/*     if (!right_line_points.empty() && !left_line_points.empty()){
        std::cout << "[" <<  __func__ <<"]" << std::endl
        << "median points back: " << left_line_points.back().x << " " << left_line_points.back().y << " -- " << 
                                    right_line_points.back().x << " " << right_line_points.back().y << std::endl
        << "median points front: " << left_line_points.front().x << " " << left_line_points.front().y << " -- "
                                    << right_line_points.front().x << " " << right_line_points.front().y << std::endl;
    } */

    medianPoints.clear();
    for (int y = top_y; y <= bottom_y; y++) {
        int left_x = static_cast<int>(left_coeffs.m * y + left_coeffs.b);
        int right_x = static_cast<int>(right_coeffs.m * y + right_coeffs.b);
        int median_x = (left_x + right_x) / 2;
        if (median_x >= 0 && median_x < width) {
           medianPoints.emplace_back(median_x, y);
        }
    }

    if (!medianPoints.empty()) {
        cv::line(output, medianPoints.front(), medianPoints.back(), cv::Scalar(0, 255, 0), 2); // Verde
    }
}