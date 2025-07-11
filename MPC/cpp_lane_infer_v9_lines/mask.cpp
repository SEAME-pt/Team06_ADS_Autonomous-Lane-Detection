#include "mask.hpp"


MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

std::vector<cv::Point> MaskProcessor::linearRegression(const std::vector<cv::Point>& points, int top_y, int bottom_y, int width, LineCoef& coeffs) {
    coeffs = {0.0, 0.0, false};
    std::vector<cv::Point> edge;
    if (points.size() < 2) {
        std::cout << "no points "<< std::endl;
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
    int center_x = width / 2, margin = 20, range = 20;

    left_edge.clear();
    right_edge.clear();

    // Passo 1: Verificar se o pixel central na linha base é branco
    int base_y = static_cast<int>(height * 0.95) - 1;

    // Passo 2: Encontrar o último pixel branco na sequência para esquerda e direita
    int left_x = -1, right_x = -1;
    int left_contig = -1, right_contig = -1;

    // Passo 3: Busca de baixo para cima nas linhas subsequentes
    for (int y = base_y; y >= height / 2; --y) {
        const cv::Mat row = mask_bin.row(y);
        int new_right_x = -1, new_left_x = -1;

        if (left_x == -1 && y > height * 0.66){
            for (int x = center_x; x >= margin; --x) {
                if (row.at<uchar>(x) == 255 && x < width / 2 - 100) {
                    left_x = x; // Último pixel branco na sequência
                    left_contig ++;
                    break;
                }
            }
        }
        else if (left_contig != -1 && left_contig <= 10){
            // Borda esquerda: verificar mesma coordenada, depois direita, depois esquerda
            // Procurar para a direita (left_x até left_x + range, limitado por center_x)
            for (int x = std::min(left_x + range, center_x); x >= std::max(margin, left_x - range); --x) {
                //if (x == left_x + range) std::cout << " X left    " << x << std::endl;
                if (row.at<uchar>(x) == 255) {
                    new_left_x = x; // Último pixel branco na sequência
                    break; // Para ao encontrar um pixel não branco após uma sequência branca
                }
            }
            if (new_left_x != -1) left_x = new_left_x;
            else left_contig++;
        }
        
        if (right_x == -1 && y > height * 0.66){
            // Right edge: buscar do centro para a direita (x cresce)
            for (int x = center_x; x < width - margin; ++x) {
                if (row.at<uchar>(x) == 255 && x > width / 2 + 100) {
                    right_x = x; // Último pixel branco na sequência
                    right_contig++;
                    break;
                }
            }
        } else if (right_contig != -1 && right_contig <= 10) {
            // Procurar para a esquerda (right_x até right_x - range, limitado por center_x)
            for (int x = std::max(center_x, right_x - range); x <= std::min( width - margin, right_x + range); ++x) {
                //if (x == right_x - range) std::cout << "    X right       " << x << std::endl << std::endl;
                if (row.at<uchar>(x) == 255) {
                    new_right_x = x; // Último pixel branco na sequência
                    break; // Para ao encontrar um pixel não branco após uma sequência branca
                }
            }
            if (new_right_x != -1) right_x = new_right_x;
            else right_contig++;
        }

        // Atualiza as coordenadas e adiciona os pontos
        if (left_x != -1 && left_contig <= 10 && right_contig != -1){
            left_edge.emplace(left_edge.begin(), left_x, y);
        }
        if (right_x != -1 && right_contig <= 10 && right_contig != -1) {
            right_edge.emplace(right_edge.begin(), right_x, y);
        }
/*         std::cout << std::endl << "left Point " << left_x << std::endl;
        std::cout << "Right Point " << right_x << std::endl << std::endl; */
    }

    left_edge.erase(
        std::remove_if(left_edge.begin(), left_edge.end(), [margin, width](const cv::Point& pt) {
            return pt.x <= margin || pt.x >= width - margin;
        }),
        left_edge.end()
    );

    right_edge.erase(
        std::remove_if(right_edge.begin(), right_edge.end(), [margin, width](const cv::Point& pt) {
            return pt.x <= margin || pt.x >= width - margin;
        }),
        right_edge.end()
    );
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

    if (findEdges == false)
        std::cout << "Lane Not Found" << std::endl;
    
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
    
    //std::cout << " checking linearRegression " << std::endl;
    /********Linear Regression *********/
    LineCoef left_coeffs, right_coeffs;
    
    std::vector<cv::Point> left_line_points = linearRegression(left_edge_points, top_y, bottom_y, width, left_coeffs);
    std::vector<cv::Point> right_line_points = linearRegression(right_edge_points, top_y, bottom_y, width, right_coeffs);
    
/*     std::cout << "left          " << left_line_points << std::endl;
    std::cout << "right         " << right_line_points << std::endl; */

    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);
    if (!left_line_points.empty())
        cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
    if (!right_line_points.empty())
        cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
    if (!right_line_points.empty() && !left_line_points.empty()){
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
           medianPoints.emplace_back(median_x, y);
        }
    }

    if (!medianPoints.empty()) {
        cv::line(output, medianPoints.front(), medianPoints.back(), cv::Scalar(0, 255, 0), 2); // Verde
    }
}