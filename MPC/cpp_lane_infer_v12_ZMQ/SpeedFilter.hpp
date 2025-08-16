#pragma once

class SpeedFilter {
private:
    double alpha;       // fator de suavização (0 < alpha < 1)
    double filtered;    // valor filtrado atual
    bool initialized;   // indica se já inicializou

public:
    SpeedFilter(double alpha = 0.2) 
        : alpha(alpha), filtered(0.0), initialized(false) {}

    double update(double measured) {
        if (!initialized) {
            filtered = measured;
            initialized = true;
        } else {
            filtered = alpha * measured + (1.0 - alpha) * filtered;
        }
        return filtered;
    }
};
