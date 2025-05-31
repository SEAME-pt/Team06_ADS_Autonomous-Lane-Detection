#ifndef NMPC_CONTROLLER_HPP
#define NMPC_CONTROLLER_HPP

#include <casadi/casadi.hpp>
#include <vector>

class NMPCController {
public:
    // Construtor: inicializa parâmetros do NMPC
    NMPCController(double L, double dt, int Np, int Nc, double delta_max, double a_max);
    
    // Calcula comandos de controle
    std::vector<double> compute_control(const std::vector<double>& x0,
                                       const std::vector<std::vector<double>>& x_ref);

private:
    double L_;          // Distância entre eixos (m)
    double dt_;         // Passo de tempo (s)
    int Np_;            // Horizonte de predição
    int Nc_;            // Horizonte de controle
    double delta_max_;  // Ângulo máximo de direção (rad)
    double a_max_;      // Aceleração máxima (m/s^2)
    casadi::Opti opti_; // Solucionador CasADi
    casadi::MX X_;      // Variável de estados
    casadi::MX U_;      // Variável de controles
    std::vector<casadi::MX> x_ref_params_; // Parâmetros de referência para cada passo
    casadi::MX x_ref_N_; // Parâmetro de referência terminal
    casadi::MX x0_param_; // Parâmetro de estado inicial
    
    // Modelo dinâmico do veículo
    casadi::MX vehicle_model(const casadi::MX& x, const casadi::MX& u);
    
    // Configura o problema de otimização
    void setup_nmpc();
};

#endif