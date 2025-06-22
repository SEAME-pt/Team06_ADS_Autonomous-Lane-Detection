#ifndef MASK_HPP
#define MASK_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "lane.hpp"

class MaskProcessor {
public:
    MaskProcessor();
    ~MaskProcessor();
    void processMask(const cv::Mat& mask, cv::Mat& output, std::vector<cv::Point>& medianPoints);
    LineCoef linearRegression(const std::vector<cv::Point>& points);

private:
    int findFirstWhite(const cv::Mat& row);
    int findLastWhite(const cv::Mat& row);
};

#endif