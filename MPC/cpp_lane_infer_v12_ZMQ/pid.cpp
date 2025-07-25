#include "pid.hpp"
#include <algorithm>

PIDController::PIDController(double kp, double ki, double kd, double dt, double output_min, double output_max)
    : kp_(kp), ki_(ki), kd_(kd), dt_(dt), output_min_(output_min), output_max_(output_max),
      integral_(0.0), prev_error_(0.0) {}

double PIDController::compute_control(double setpoint, double measured_value) {
    // Calculate error
    double error = setpoint - measured_value;

    // Proportional term
    double p_term = kp_ * error;

    // Integral term
    integral_ += error * dt_;
    double i_term = ki_ * integral_;

    // Derivative term
    double derivative = (error - prev_error_) / dt_;
    double d_term = kd_ * derivative;

    // Compute output
    double output = p_term + i_term + d_term;

    // Limit output
    output = std::max(output_min_, std::min(output_max_, output));

    // Update previous error
    prev_error_ = error;

    return output;
}