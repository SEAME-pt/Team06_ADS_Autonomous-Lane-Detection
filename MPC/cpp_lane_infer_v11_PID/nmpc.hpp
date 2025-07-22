#ifndef NMPC_HPP
#define NMPC_HPP

#include <casadi/casadi.hpp>
#include <vector>

class NMPCController {
private:
    // Robot parameters
    const double L = 0.15;      // Wheelbase (m)
    const double dt = 0.1;      // Time step (s)
    const double delta_max = 40.0 * M_PI / 180.0; // Max steering angle (radians)

    // NMPC parameters
    const int N = 10;           // Prediction horizon
    const double Q_pos = 1000.0;  // Weight for lateral error
    const double Q_yaw = 500.0;   // Weight for yaw error
    const double R_delta = 1.0; // Weight for control effort

    // CasADi variables
    casadi::Opti opti;          // Optimization problem
    casadi::Slice all;          // Helper for indexing
    std::vector<casadi::MX> x;  // State trajectory
    std::vector<casadi::MX> u;  // Control trajectory
    casadi::MX cost;            // Cost function

    // CasADi function for solving NMPC
    casadi::Function solver;

public:
    NMPCController();
    double computeControl(double offset_cm, double psi, double theta, double v);
    void setState(double x, double y, double theta, double v);
};

#endif // NMPC_HPP