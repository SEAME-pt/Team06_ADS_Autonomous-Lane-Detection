#include "mask.hpp"
#include "lane.hpp"


MaskProcessor::MaskProcessor() {}
MaskProcessor::~MaskProcessor() {}

std::vector<cv::Point> MaskProcessor::linearRegression(const std::vector<cv::Point>& points, int top_y, int bottom_y, int width, LineCoef& coeffs) {
    coeffs = {0.0, 0.0, false};
    std::vector<cv::Point> edge;
    if (points.size() < 3) {
        std::cout << "                                              no points "<< std::endl;
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

void MaskProcessor::processEdges(const cv::Mat& mask_bin, std::vector<cv::Point>& left_edge, std::vector<cv::Point>& right_edge) {
    int height = mask_bin.rows, width = mask_bin.cols;
    int center_x = width / 2, margin = 5, range = 20;

    left_edge.clear();
    right_edge.clear();

    int base_y = static_cast<int>(height * 0.95) - 1;

    int left_x = -1, right_x = -1;
    int left_contig = -1, right_contig = -1;
    int left_count = 0, right_count = 0;
    int left_start_limit = static_cast<int>(height * 0.60);
    int right_start_limit = static_cast<int>(height * 0.60);
    bool reset_left = false, reset_right = false;

    for (int y = base_y; y >= height / 2; --y) {
        const cv::Mat row = mask_bin.row(y);
        int new_right_x = -1, new_left_x = -1;

        if (left_x == -1 && y > left_start_limit){
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
        if (left_contig > 10 && left_count < 50) {
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
        if (right_contig > 10 && right_count < 50) {
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
}

void MaskProcessor::verifyLanes(std::vector<cv::Point>& left_edge_points, std::vector<cv::Point>& right_edge_points, 
                        int& bottom_left, int& bottom_right, int& top_left, int& top_right){

    size_t min_size_line = 50;
    double x_px = 0, sy = 0;
    double d = 0.26;

    if (left_edge_points.size() > min_size_line){
        bottom_left = left_edge_points.back().y;
        std::cout << " encontrou a esquerda" << std::endl;
    }
    else if (left_edge_points.size() < min_size_line && right_edge_points.size() > min_size_line){
        std::cerr << "[Warning] Left Edge Lost" << std::endl;
        auto it = right_edge_points.begin();
        for (int y = it->y; it != right_edge_points.end(); ++it, y = it->y) {
            sy = Asy * y + Bsy;
            x_px = d / sy;
            int left_x = it->x - x_px;
            left_edge_points.emplace_back(left_x, y);
        }
        bottom_left = right_edge_points.back().y;
        top_left = right_edge_points.front().y;
    }

    if (right_edge_points.size() > min_size_line) {
        std::cout << " encontrou a direita" << std::endl;
        bottom_right = right_edge_points.back().y;
    }
    else if (right_edge_points.size()  < min_size_line && left_edge_points.size() > min_size_line){
        std::cerr << "[Warning] Right Edge Lost" << std::endl;
        auto it = left_edge_points.begin();
        for (int y = it->y; it != left_edge_points.end(); ++it, y = it->y) {
            sy = Asy * y + Bsy;
            x_px = d / sy;
            int right_x = it->x + x_px;
            right_edge_points.emplace_back(right_x, y);
        }
        bottom_right = left_edge_points.back().y;
        top_right = left_edge_points.front().y;
        if (!right_edge_points.empty()){
            std::cout << "bottom right" << right_edge_points.back() << std::endl;
            std::cout << "top right" << right_edge_points.front() << std::endl;
        }
        if (!left_edge_points.empty()){
            std::cout << "bottom left" << left_edge_points.back() << std::endl;
            std::cout << "top left" << left_edge_points.front() << std::endl;
        }
    }

    if (left_edge_points.empty() && right_edge_points.empty()){
        std::cerr << "[Warning Break!!!!!!!!!!!!] Edges Lost" << std::endl;
        return;
    }

    /*     std::cout << "right   back      " << right_edge_points.back() << std::endl;
    std::cout << "left    back     " << left_edge_points.back() << std::endl;    
    std::cout << "right   front      " << right_edge_points.front() << std::endl;
    std::cout << "left    front     " << left_edge_points.front() << std::endl; */
    //std::cout << "right   size      " << right_edge_points << std::endl;
    //std::cout << "left   size      " << left_edge_points << std::endl;
}


LineIntersect  MaskProcessor::findIntersect(const LineCoef& left_coeffs, const LineCoef& right_coeffs, int height, int width) {
    LineIntersect intersect;
    intersect.valid = false;

    int roi_start_y = static_cast<int>(0.50 * height); // y = 180
    int roi_end_y = static_cast<int>(0.95 * height);   // y = 340

    if (left_coeffs.valid && right_coeffs.valid) {
        intersect.xl_t = { static_cast<float>(left_coeffs.m * roi_start_y + left_coeffs.b), static_cast<float>(roi_start_y) };
        intersect.xl_b = { static_cast<float>(left_coeffs.m * roi_end_y + left_coeffs.b), static_cast<float>(roi_end_y) };
        intersect.xr_t = { static_cast<float>(right_coeffs.m * roi_start_y + right_coeffs.b), static_cast<float>(roi_start_y) };
        intersect.xr_b = { static_cast<float>(right_coeffs.m * roi_end_y + right_coeffs.b), static_cast<float>(roi_end_y) };
        
        intersect.ratio_top = (intersect.xr_t.x - width / 2.0f) / (intersect.xr_t.x - intersect.xl_t.x);
        intersect.xs_b = intersect.xr_b.x - intersect.ratio_top * (intersect.xr_b.x - intersect.xl_b.x);
        intersect.slope = (intersect.xs_b - width / 2.0f) / (roi_end_y - roi_start_y);
        intersect.psi = std::atan(intersect.slope);
        intersect.valid = true;

        // Pixels on top and bottom image
        intersect.x_px_t = intersect.xr_t.x - intersect.xl_t.x;
        intersect.x_px_b = intersect.xr_b.x - intersect.xl_b.x;

        // Scale Factor | d = s(y).x_px + b | with b = 0
        intersect.scaleFactor_t = intersect.w_real / intersect.x_px_t;
        intersect.scaleFactor_b = intersect.w_real / intersect.x_px_b;

        // var a | s(y) = a * y + b (=) a = delta(s(y)) / delta(y) |,  with b = 0 
        intersect.var_a = (intersect.scaleFactor_b - intersect.scaleFactor_t) / (intersect.xl_b.y - intersect.xl_t.y);

        // var b para o top que será igual ao bottom
        intersect.var_b = intersect.scaleFactor_t - (intersect.var_a * intersect.xl_t.y);
    }
    
    if (intersect.valid) {
        //std::cout << "ratio: " << intersect.ratio_top << std::endl;
        //std::cout << "xs_b: " << intersect.xs_b << std::endl;
        std::cout << "slope: " << intersect.slope << std::endl;
        std::cout << "Psi: " << intersect.psi << std::endl;
        //std::cout << "Psi: " << intersect.psi * 180.0 / M_PI << " deg" << std::endl ;
        //std::cout << " Delta: " + std::to_string(delta * 180.0 / M_PI) + "deg" << std::endl << std::endl;
    }
    return intersect;
}

void MaskProcessor::processMask(const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints, 
                                LineCoef& left_coeffs, LineCoef& right_coeffs, LineIntersect& intersect) {
    
    cv::Mat mask_bin = ll_mask.clone();
    cv::threshold(ll_mask, mask_bin, 127, 255, cv::THRESH_BINARY);
    //LineIntersect intersect;

    std::vector<cv::Point> left_edge_points, right_edge_points;
    int height = 360, width = 640;
    int top_y = height / 2, bottom_y = height * 0.95;
    int bottom_left = bottom_y, bottom_right = bottom_y;
    int top_left = top_y , top_right = top_y;

    processEdges(mask_bin, left_edge_points, right_edge_points);
    verifyLanes(left_edge_points, right_edge_points, bottom_left, bottom_right, top_left, top_right);

    /********Linear Regression *********/    
    std::vector<cv::Point> left_line_points = linearRegression(left_edge_points, top_left, bottom_left, width, left_coeffs);
    std::vector<cv::Point> right_line_points = linearRegression(right_edge_points, top_right, bottom_right, width, right_coeffs);
    
    //intersect = findIntersect(left_coeffs, right_coeffs, 360, 640);

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
    if (!left_line_points.empty())
        cv::line(output, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2); // Vermelho
    if (!right_line_points.empty())
        cv::line(output, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2); // Azul
    if (!right_line_points.empty() && !left_line_points.empty()){
        float sy1 = (Asy * 180 + Bsy) * ((intersect.xrt - intersect.xlt) / std::cos(intersect.slope));
        float sy2 = (Asy * 360 + Bsy) * ((intersect.xrb - intersect.xlb) / std::cos(intersect.slope));

        std::cout << "[" <<  __func__ <<"]" << std::endl
        << "P2: " << sy2 << std::endl
        << "P1: " << sy1 << std::endl
        << "Slope: " << intersect.slope << std::endl << std::endl

        /* << "median points back: " << left_line_points.back().x << " " << left_line_points.back().y << " -- " << 
                                    right_line_points.back().x << " " << right_line_points.back().y << std::endl
        << "median points front: " << left_line_points.front().x << " " << left_line_points.front().y << " -- "
                                    << right_line_points.front().x << " " << right_line_points.front().y << std::endl << std::endl
        << "distance back: " << right_line_points.back().x - left_line_points.back().x << " " << left_line_points.back().y << " -- " << std::endl
        << "distance front: " << right_line_points.front().x - left_line_points.front().x << " " << left_line_points.front().y << std::endl << std::endl */;
    }

    medianPoints.clear();
    for (int y = top_y; y < height; y++) {
        float left_x = (left_coeffs.m * y + left_coeffs.b);
        float right_x = (right_coeffs.m * y + right_coeffs.b);
        float median_x = (left_x + right_x) / 2;
        if (median_x >= 0 && median_x < width) {
           medianPoints.emplace_back(median_x, y);
        }
    }
    /* std::cout << "top: " << medianPoints.front() << std::endl;
    std::cout << "bottom: " << medianPoints.back() << std::endl << std::endl; */

    if (!medianPoints.empty()) {
        cv::line(output, medianPoints.front(), medianPoints.back(), cv::Scalar(200, 200, 200), 2); // Verde
    }
}