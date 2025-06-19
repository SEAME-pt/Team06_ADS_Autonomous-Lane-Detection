// polyfit.h
#ifndef POLYFIT_H
#define POLYFIT_H

#include "lane_detection.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>


// Estrutura para coeficientes do polinômio
struct PolyCoefficients {
    double a0, a1, a2; // y = a0 + a1*x + a2*x^2
    bool valid;
};


PolyCoefficients fitPolynomial(const LaneData& data);
std::vector<std::vector<double>> generateReference(const LaneData& data, int someParam);

#endif
