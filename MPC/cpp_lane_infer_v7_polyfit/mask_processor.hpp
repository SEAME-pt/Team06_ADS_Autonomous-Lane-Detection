#ifndef MASK_PROCESSOR_HPP
#define MASK_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct LineCoefficients {
    double m, b;
    bool valid;
};

class MaskProcessor {
public:
    MaskProcessor();
    ~MaskProcessor();
    void processMask(const cv::Mat& mask, cv::Mat& output, std::vector<cv::Point>& medianPoints);
    LineCoefficients linearRegression(const std::vector<cv::Point>& points);

private:
    int findFirstWhite(const cv::Mat& row);
    int findLastWhite(const cv::Mat& row);
};

#endif