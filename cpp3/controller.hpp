#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <opencv2/opencv.hpp>
#include <deque>

class Controller {
public:
    Controller(int frame_width, int frame_height);

    float calcular_angulo(const cv::Mat& mask, cv::Mat& vis_frame);

private:
    int width, height;
    int center_x, base_y;
    int min_distance;
    float threshold;

    std::deque<cv::Point> point_history;
    std::deque<float> angle_history;

    cv::Point ghost_point;
    double ghost_time;
    double ghost_timeout;

    cv::Point encontrar_ponto(const cv::Mat& mask);
};

#endif
