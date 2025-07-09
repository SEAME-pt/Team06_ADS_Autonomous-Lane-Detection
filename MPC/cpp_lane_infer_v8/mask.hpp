#ifndef MASK_HPP
#define MASK_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "lane.hpp"

class MaskProcessor {
public:
    MaskProcessor();
    ~MaskProcessor();
    //void processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints, LineCoef& left_coeffs, LineCoef& right_coeffs);
    void processMask(const cv::Mat& da_mask, const cv::Mat& ll_mask, cv::Mat& output, std::vector<cv::Point>& medianPoints);
    //void processMask(const cv::Mat& mask, cv::Mat& output, std::vector<cv::Point>& medianPoints);
    LineCoef linearRegression(const std::vector<cv::Point>& points);
    std::pair<int, int> findEdgesFromCenter(const cv::Mat& row, int center_x, int width);

private:
    int firstWhite(const cv::Mat& row);
    int lastWhite(const cv::Mat& row);    
};

#endif