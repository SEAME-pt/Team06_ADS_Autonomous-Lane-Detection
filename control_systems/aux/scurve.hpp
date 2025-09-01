#ifndef SCURVE_HPP
#define SCURVE_HPP

#include <algorithm>
#include <cmath>

class SCurveProfile {
public:
    SCurveProfile(double max_vel_deg_s, double max_acc_deg_s2, double max_jerk_deg_s3)
        : max_vel_(max_vel_deg_s), max_acc_(max_acc_deg_s2), max_jerk_(max_jerk_deg_s3),
          current_pos_(0.0), current_vel_(0.0), current_acc_(0.0) {}

    double computeNext(double target_pos, double dt) {
        double error = target_pos - current_pos_;

        if (std::abs(error) < 1e-3) {
            current_pos_ = target_pos;
            current_vel_ = 0.0;
            current_acc_ = 0.0;
            return current_pos_;
        }

        double sign = (error > 0.0) ? 1.0 : -1.0;
        double ideal_vel = max_vel_ * sign;
        double desired_acc = (ideal_vel - current_vel_) / dt;
        double delta_acc = desired_acc - current_acc_;
        double max_delta_acc = max_jerk_ * dt;
        delta_acc = std::clamp(delta_acc, -max_delta_acc, max_delta_acc);

        current_acc_ += delta_acc;
        current_acc_ = std::clamp(current_acc_, -max_acc_, max_acc_);
        current_vel_ += current_acc_ * dt;
        current_vel_ = std::clamp(current_vel_, -max_vel_, max_vel_);
        current_pos_ += current_vel_ * dt;

        if ((error > 0.0 && current_pos_ > target_pos) || (error < 0.0 && current_pos_ < target_pos)) {
            current_pos_ = target_pos;
            current_vel_ = 0.0;
            current_acc_ = 0.0;
        }

        return current_pos_;
    }

    void reset(double initial_pos = 0.0) {
        current_pos_ = initial_pos;
        current_vel_ = 0.0;
        current_acc_ = 0.0;
    }

private:
    double max_vel_;
    double max_acc_;
    double max_jerk_;
    double current_pos_;
    double current_vel_;
    double current_acc_;
};

#endif // SCURVE_HPP