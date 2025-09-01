#pragma once
#include <deque>

class MovingAverage {
private:
    std::deque<double> buffer;
    size_t windowSize;
    double sum;

public:
    MovingAverage(size_t windowSize = 5) 
        : windowSize(windowSize), sum(0.0) {}

    double update(double measured) {
        buffer.push_back(measured);
        sum += measured;

        if (buffer.size() > windowSize) {
            sum -= buffer.front();
            buffer.pop_front();
        }

        return sum / buffer.size();
    }
};
