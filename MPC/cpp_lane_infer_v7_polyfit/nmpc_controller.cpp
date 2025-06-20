#include "nmpc_controller.hpp"
#include <cmath>

NMPCController::NMPCController(double L, double dt, int Np, int Nc, double delta_max, double a_max)
    : L_(L), dt_(dt), Np_(Np), Nc_(Nc), delta_max_(delta_max), a_max_(a_max) { // Reduz limites
    setup_nmpc();
}

casadi::MX NMPCController::vehicle_model(const casadi::MX& x, const casadi::MX& u) {
    // Estados: x = [x, y, psi, v]
    casadi::MX x_pos = x(0), y_pos = x(1), psi = x(2), v = x(3);
    // Controles: u = [delta, a]
    casadi::MX delta = u(0), a = u(1);
    
    // Modelo cinemático de bicicleta (discretizado com Euler)
    casadi::MX x_dot = v * casadi::MX::cos(psi);
    casadi::MX y_dot = v * casadi::MX::sin(psi);
    casadi::MX psi_dot = (v / L_) * casadi::MX::tan(delta);
    casadi::MX v_dot = a;
    
    // Próximo estado
    casadi::MX x_next = casadi::MX::vertcat({
        x_pos + dt_ * x_dot,
        y_pos + dt_ * y_dot,
        psi + dt_ * psi_dot,
        v + dt_ * v_dot
    });
    
    return x_next;
}

void NMPCController::setup_nmpc() {
    X_ = opti_.variable(4, Np_ + 1);
    U_ = opti_.variable(2, Nc_);
    
    casadi::DM Q = casadi::DM::diag({10.0, 20.0, 1.0, 0.1});
    casadi::DM R = casadi::DM::diag({0.05, 0.1});
    casadi::DM P = Q;

    casadi::MX cost = 0;
    x_ref_params_.resize(Np_);
    for (int k = 0; k < Np_; ++k) {
        x_ref_params_[k] = opti_.parameter(4, 1);
        casadi::MX x_k = X_(casadi::Slice(), k);
        casadi::MX e_k = x_k - x_ref_params_[k];
        cost += casadi::MX::mtimes({e_k.T(), Q, e_k});
        if (k < Nc_) {
            casadi::MX u_k = U_(casadi::Slice(), k);
            cost += casadi::MX::mtimes({u_k.T(), R, u_k});
        }
    }
    x_ref_N_ = opti_.parameter(4, 1);
    casadi::MX e_N = X_(casadi::Slice(), Np_) - x_ref_N_;
    cost += casadi::MX::mtimes({e_N.T(), P, e_N});
    
    x0_param_ = opti_.parameter(4, 1);
    opti_.subject_to(X_(casadi::Slice(), 0) == x0_param_);
    
    for (int k = 0; k < Np_; ++k) {
        casadi::MX u_k = (k < Nc_) ? U_(casadi::Slice(), k) : U_(casadi::Slice(), Nc_ - 1);
        opti_.subject_to(X_(casadi::Slice(), k + 1) == vehicle_model(X_(casadi::Slice(), k), u_k));
        opti_.subject_to(-delta_max_ <= u_k(0) <= delta_max_);
        opti_.subject_to(-a_max_ <= u_k(1) <= a_max_);
    }

    opti_.minimize(cost);
    casadi::Dict opts;
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = 0;
    opti_.solver("ipopt", opts);
}

std::vector<double> NMPCController::compute_control(const std::vector<double>& x0,
                                                   const std::vector<std::vector<double>>& x_ref) {
    // Verifica se Nc_ é válido
    if (Nc_ <= 0) {
        throw std::runtime_error("Horizonte de controle Nc_ deve ser maior que 0");
    }

    // Define estado inicial
    opti_.set_value(x0_param_, x0);

    // Define trajetória de referência
    for (int k = 0; k < Np_; ++k) {
        opti_.set_value(x_ref_params_[k], x_ref[k]);
    }
    opti_.set_value(x_ref_N_, x_ref[0]); // Última referência

    // Resolve o problema
    auto sol = opti_.solve();

    // Extrai o primeiro controle
    casadi::DM u_opt = sol.value(U_); // Acessa a variável U
    std::vector<double> result = {u_opt(0, 0).scalar(), u_opt(1, 0).scalar()};

    return result;
}