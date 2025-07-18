#ifndef PID_HPP
#define PID_HPP

class PIDController {
public:
    PIDController(double kp, double ki, double kd, double dt, double output_min, double output_max);
    double compute_control(double setpoint, double measured_value);

private:
    double kp_, ki_, kd_; // PID gains
    double dt_;           // Time step (s)
    double output_min_, output_max_; // Output limits (e.g., PWM bounds)
    double integral_;     // Accumulated integral term
    double prev_error_;   // Previous error for derivative term
};

#endif