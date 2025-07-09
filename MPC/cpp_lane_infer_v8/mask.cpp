#include "mask.hpp"


MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

/* LineCoef MaskProcessor::linearRegression(const std::vector<cv::Point>& points) {
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
} */

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

bool MaskProcessor::processEdges(const cv::Mat& mask_bin, std::vector<cv::Point>& left_edge, std::vector<cv::Point>& right_edge) {
    int height = mask_bin.rows, width = mask_bin.cols;
    int center_x = width / 2, margin = 20, range = 25;

    left_edge.clear();
    right_edge.clear();

    // Passo 1: Verificar se o pixel central na linha base é branco
    int base_y = static_cast<int>(height * 0.95) - 1;
    const cv::Mat base_row = mask_bin.row(base_y);

    if (base_row.at<uchar>(0, center_x) != 255) {
        std::cout << "Aviso: O pixel central na linha base (x=" << center_x << ", y=" << base_y << ") não é branco." << std::endl;
        return false;
    }

    // Passo 2: Encontrar o último pixel branco na sequência para esquerda e direita
    int left_x = center_x, right_x = center_x;

    // Left edge: buscar do centro para a esquerda (x decresce)
    for (int x = center_x; x >= margin; --x) {
        if (base_row.at<uchar>(0, x) == 255) {
            left_x = x; // Último pixel branco na sequência
        } else {
            break;
        }
    }

    // Right edge: buscar do centro para a direita (x cresce)
    for (int x = center_x; x < width - margin; ++x) {
        if (base_row.at<uchar>(0, x) == 255) {
            right_x = x; // Último pixel branco na sequência
        } else {
            break;
        }
    }

    // Adicionar pontos da base
    left_edge.push_back(cv::Point(left_x, base_y));
    right_edge.push_back(cv::Point(right_x, base_y));

    // Passo 3: Busca de baixo para cima nas linhas subsequentes
    for (int y = base_y - 1; y >= height / 2; --y) {
        const cv::Mat row = mask_bin.row(y);

        // Borda esquerda: verificar mesma coordenada, depois direita, depois esquerda
        int new_left_x = -1;
        if (row.at<uchar>(0, left_x) == 255) {
            new_left_x = left_x; // Mesma coordenada é branca
        } else {
            // Procurar para a direita (left_x até left_x + range, limitado por center_x)
            for (int x = left_x; x <= std::min(center_x, left_x + range); ++x) {
                if (row.at<uchar>(0, x) == 255) {
                    new_left_x = x; // Último pixel branco na sequência
                } else if (new_left_x != -1) {
                    break; // Para ao encontrar um pixel não branco após uma sequência branca
                }
            }
            // Se não encontrou, procurar para a esquerda (left_x - 1 até left_x - range, limitado por margin)
            if (new_left_x == -1) {
                for (int x = left_x - 1; x >= std::max(margin, left_x - range); --x) {
                    if (row.at<uchar>(0, x) == 255) {
                        new_left_x = x; // Último pixel branco na sequência
                    } else if (new_left_x != -1) {
                        break; // Para ao encontrar um pixel não branco após uma sequência branca
                    }
                }
            }
        }

        // Borda direita: verificar mesma coordenada, depois esquerda, depois direita
        int new_right_x = -1;
        if (row.at<uchar>(0, right_x) == 255) {
            new_right_x = right_x; // Mesma coordenada é branca
        } else {
            // Procurar para a esquerda (right_x até right_x - range, limitado por center_x)
            for (int x = right_x; x >= std::max(center_x, right_x - range); --x) {
                if (row.at<uchar>(0, x) == 255) {
                    new_right_x = x; // Último pixel branco na sequência
                } else if (new_right_x != -1)
                    break; // Para ao encontrar um pixel não branco após uma sequência branca
            }
            // Se não encontrou, procurar para a direita (right_x + 1 até right_x + range, limitado por width - margin)
            if (new_right_x == -1) {
                for (int x = right_x + 1; x <= std::min(width - margin, right_x + range); ++x) {
                    if (row.at<uchar>(0, x) == 255) {
                        new_right_x = x; // Último pixel branco na sequência
                    } else if (new_right_x != -1)
                        break; // Para ao encontrar um pixel não branco após uma sequência branca
                }
            }
        }

        // Se não encontrar pixels brancos em uma das bordas, interrompe a busca
        if (new_left_x == -1 || new_right_x == -1 ) break;

        // Atualiza as coordenadas e adiciona os pontos
        left_x = new_left_x;
        right_x = new_right_x;
        left_edge.emplace(left_edge.begin(), left_x, y);
        right_edge.emplace(right_edge.begin(), right_x, y);
    }

    // Passo 4: Verificar e remover pontos a partir de coordenadas nas bordas da imagem
    for (auto it = left_edge.begin(); it != left_edge.end(); ++it) {
        if (it->x <= margin || it->x >= width - margin) {
            left_edge.erase(it, left_edge.end());
            break;
        }
    }

    for (auto it = right_edge.begin(); it != right_edge.end(); ++it) {
        std::cout << " it x             " << it->x << std::endl;
        if (it->x <= margin || it->x >= width - margin) {
            right_edge.erase(it, right_edge.end());
            break;
        }
    }

    return true;
}


void MaskProcessor::processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints) {
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
    } else if (right_edge_points.size() < 10) {
        std::cerr << "[Warning] Right Edge Lost" << std::endl;
    } else {
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

    cv::cvtColor(mask_bin, output, cv::COLOR_GRAY2BGR);

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