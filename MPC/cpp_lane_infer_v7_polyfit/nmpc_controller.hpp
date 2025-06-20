#ifndef NMPC_CONTROLLER_HPP
#define NMPC_CONTROLLER_HPP

#include <casadi/casadi.hpp>
#include <vector>

class NMPCController {
public:
    NMPCController(double L, double dt, int Np, int Nc, double delta_max, double a_max);
    std::vector<double> compute_control(const std::vector<double>& x0,
                                       const std::vector<std::vector<double>>& x_ref);

private:
    double L_;        // Wheelbase (m)
    double dt_;       // Time step (s)
    int Np_;          // Prediction horizon
    int Nc_;          // Control horizon
    double delta_max_; // Max steering angle (rad)
    double a_max_;    // Max acceleration (m/s²)

    casadi::Opti opti_;
    casadi::MX X_;    // States [x, y, psi, v]
    casadi::MX U_;    // Controls [delta, a]
    casadi::MX x0_param_;
    std::vector<casadi::MX> x_ref_params_;
    casadi::MX x_ref_N_;

    casadi::MX vehicle_model(const casadi::MX& x, const casadi::MX& u);
    void setup_nmpc();
};

#endif