#include "controller.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>

Controller::Controller(int frame_width, int frame_height)
{
    width = frame_width;
    height = frame_height;
    center_x = width / 2;
    base_y = height - 10;
    min_distance = 40;
    threshold = 0.5;

    ghost_point = cv::Point(center_x, base_y - 60);
    ghost_time = 0;
    ghost_timeout = 2.0;
}

cv::Point Controller::encontrar_ponto(const cv::Mat& mask)
{
    std::vector<cv::Point> pontos;
    for (int y = 0; y < mask.rows; ++y)
    {
        for (int x = 0; x < mask.cols; ++x)
        {
            if (mask.at<uchar>(y, x) > 127) pontos.emplace_back(x, y);
        }
    }

    if (pontos.empty()) return cv::Point(-1, -1);

    // Encontrar o ponto mais próximo do centro/base
    cv::Point melhor_ponto;
    int menor_dist = 1e6;
    for (const auto& p : pontos)
    {
        int dist = abs(p.y - base_y) + abs(p.x - center_x);
        if ((base_y - p.y) > min_distance && dist < menor_dist)
        {
            menor_dist = dist;
            melhor_ponto = p;
        }
    }

    if (menor_dist == 1e6) return cv::Point(-1, -1);

    return melhor_ponto;
}

float Controller::calcular_angulo(const cv::Mat& mask, cv::Mat& vis_frame)
{
    cv::Point ponto = encontrar_ponto(mask);

    if (ponto.x >= 0)
    {
        point_history.push_back(ponto);
        ghost_point = ponto;
        ghost_time = std::time(nullptr);
    }
    else
    {
        if (std::time(nullptr) - ghost_time < ghost_timeout)
        {
            ponto = ghost_point;
        }
        else
        {
            ponto = cv::Point(center_x, base_y - 60);
        }
    }
 
    if (!point_history.empty())
    {
        cv::Point avg(0, 0);
        for (const auto& pt : point_history)
        {
            avg += pt;
        }
        avg.x /= point_history.size();
        avg.y /= point_history.size();

        float dx = avg.x - center_x;
        float dy = base_y - avg.y;
        float angle = std::atan2(dx, dy);

        angle_history.push_back(angle);
        if (angle_history.size() > 5) angle_history.pop_front();

        float smooth_angle = 0;
        for (float a : angle_history) smooth_angle += a;
        smooth_angle /= angle_history.size();

        // Visualização
        cv::circle(vis_frame, avg, 8, cv::Scalar(0, 255, 0), -1);
        int arrow_x = int(center_x + 60 * std::sin(smooth_angle));
        int arrow_y = int(base_y - 60 * std::cos(smooth_angle));
        cv::arrowedLine(vis_frame, cv::Point(center_x, base_y),
                        cv::Point(arrow_x, arrow_y), cv::Scalar(0, 255, 255),
                        4);

        return smooth_angle;
    }

    return 0.0f;
}
