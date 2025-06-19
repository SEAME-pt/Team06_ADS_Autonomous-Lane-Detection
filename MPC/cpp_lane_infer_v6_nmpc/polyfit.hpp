// polyfit.h
#ifndef POLYFIT_H
#define POLYFIT_H

#include "lane_detection.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

LaneData fitPolynomial(const LaneData& data);
LaneData generateReference(const LaneData& data, int someParam);

#endif
