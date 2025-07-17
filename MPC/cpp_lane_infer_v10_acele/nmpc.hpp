#ifndef NMPC_HPP
#define NMPC_HPP

#include <casadi/casadi.hpp>
#include <vector>
#include "lane.hpp"

class NMPCController {
public:
    NMPCController(double L, double dt, int N, double delta_max, double w_x, double w_y, double w_psi, double w_delta);
    std::vector<double> compute_control(const std::vector<double>& x0, const LaneData& laneData, double psi);

private:
    double L_;          // Distância entre eixos (m)
    double dt_;         // Passo de tempo (s)
    int N_;             // Horizonte de previsão
    double delta_max_;  // Limite do ângulo das rodas (rad)
    double w_x_, w_y_, w_psi_, w_delta_; // Pesos da função de custo
    casadi::Function solver_; // Solver CasADi
};

#endif