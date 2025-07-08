#include "nmpc.hpp"
#include <stdexcept>

using namespace casadi;

NMPCController::NMPCController(double L, double dt, int N, double delta_max, double w_x, double w_y, double w_psi, double w_delta)
    : L_(L), dt_(dt), N_(N), delta_max_(delta_max), w_x_(w_x), w_y_(w_y), w_psi_(w_psi), w_delta_(w_delta) {
    // Velocidade constante
    const double v = 3.0;

    // Variáveis simbólicas
    SX x = SX::sym("x");
    SX y = SX::sym("y");
    SX psi = SX::sym("psi");
    SX delta = SX::sym("delta");
    SX states = vertcat(x, y, psi);
    SX controls = delta;

    // Modelo dinâmico discretizado
    SX x_next = x + v * cos(psi) * dt;
    SX y_next = y + v * sin(psi) * dt;
    SX psi_next = psi + (v / L) * tan(delta) * dt;
    SX states_next = vertcat(x_next, y_next, psi_next);

    // Função do modelo
    Function f("f", {states, controls}, {states_next});

    // Variáveis de otimização
    std::vector<SX> w; // Todas as variáveis
    std::vector<SX> g; // Restrições
    SX J = 0;          // Função de custo
    std::vector<std::vector<double>> lbw, ubw; // Limites das variáveis
    std::vector<std::vector<double>> lbg, ubg; // Limites das restrições

    // Estados e controles ao longo do horizonte
    std::vector<SX> X(N + 1), U(N);
    for (int k = 0; k < N + 1; ++k) {
        X[k] = SX::sym("X_" + std::to_string(k), 3);
        w.push_back(X[k]);
        lbw.push_back({-inf, -inf, -inf});
        ubw.push_back({inf, inf, inf});
    }
    for (int k = 0; k < N; ++k) {
        U[k] = SX::sym("U_" + std::to_string(k), 1);
        w.push_back(U[k]);
        lbw.push_back({-delta_max});
        ubw.push_back({delta_max});
    }

    // Estado inicial
    SX x0 = SX::sym("x0", 3);
    g.push_back(X[0] - x0);
    lbg.push_back({0, 0, 0});
    ubg.push_back({0, 0, 0});

    // Dinâmica
    for (int k = 0; k < N; ++k) {
        std::vector<SX> f_in = {X[k], U[k]};
        SX xk_next = f(f_in)[0];
        g.push_back(xk_next - X[k + 1]);
        lbg.push_back({0, 0, 0});
        ubg.push_back({0, 0, 0});
    }

    // Função de custo
    SX x_ref = SX::sym("x_ref", 2 * N); // Vetor [x_ref_0, y_ref_0, x_ref_1, y_ref_1, ...]
    for (int k = 0; k < N; ++k) {
        J += w_x * pow(X[k + 1](0) - x_ref(2 * k), 2);
        J += w_y * pow(X[k + 1](1) - x_ref(2 * k + 1), 2);
        J += w_psi * pow(X[k + 1](2), 2); // psi_ref = 0
        if (k < N) {
            J += w_delta * pow(U[k](0), 2);
        }
    }

    // Configuração do solver
    Dict opts;
    opts["ipopt.print_level"] = 0; // Suprime saída do IPOPT
    opts["ipopt.sb"] = "yes";      // Suprime banner do IPOPT
    opts["print_time"] = 0;        // Suprime estatísticas de tempo
    opts["ipopt.max_iter"] = 100;
    opts["ipopt.tol"] = 1e-6;
    SXDict nlp = {{"x", vertcat(w)}, {"f", J}, {"g", vertcat(g)}, {"p", vertcat(x0, x_ref)}};
    solver_ = nlpsol("solver", "ipopt", nlp, opts);
}

std::vector<double> NMPCController::compute_control(const std::vector<double>& x0, const LaneData& laneData, double psi) {
    if (x0.size() < 3) {
        throw std::runtime_error("Estado inicial deve ter pelo menos 3 elementos");
    }
    // Preparar trajetória de referência
    std::vector<double> x_ref(N_ * 2, 0.0);
    int points_used = std::min(laneData.num_points, N_);
    for (int i = 0; i < points_used; ++i) {
        x_ref[2 * i] = laneData.points[i].x;
        x_ref[2 * i + 1] = laneData.points[i].y;
    }
    
    // Preencher com o último ponto, se necessário
    for (int i = points_used; i < N_; ++i) {
        x_ref[2 * i] = laneData.points[points_used - 1].x;
        x_ref[2 * i + 1] = laneData.points[points_used - 1].y;
    }

    // Estado inicial
    std::vector<double> state = {x0[0], x0[1], psi};

    // Preparar argumentos do solver
    std::vector<double> w0((N_ + 1) * 3 + N_, 0.0);
    std::vector<double> lbx((N_ + 1) * 3 + N_, 0.0);
    std::vector<double> ubx((N_ + 1) * 3 + N_, 0.0);
    std::vector<double> lbg((N_ + 1) * 3, 0.0);
    std::vector<double> ubg((N_ + 1) * 3, 0.0);

    // Limites dos estados (sem restrições)
    for (int k = 0; k < N_ + 1; ++k) {
        lbx[3 * k] = -inf;
        lbx[3 * k + 1] = -inf;
        lbx[3 * k + 2] = -inf;
        ubx[3 * k] = inf;
        ubx[3 * k + 1] = inf;
        ubx[3 * k + 2] = inf;
    }

    // Limites dos controles
    for (int k = 0; k < N_; ++k) {
        lbx[(N_ + 1) * 3 + k] = -delta_max_;
        ubx[(N_ + 1) * 3 + k] = delta_max_;
    }

    // Estado inicial
    lbg[0] = ubg[0] = state[0];
    lbg[1] = ubg[1] = state[1];
    lbg[2] = ubg[2] = state[2];

    // Resolver o problema
    std::vector<double> p(3 + 2 * N_);
    p[0] = state[0];
    p[1] = state[1];
    p[2] = state[2];
    for (int i = 0; i < N_; ++i) {
        p[3 + 2 * i] = x_ref[2 * i];
        p[3 + 2 * i + 1] = x_ref[2 * i + 1];
    }

    DMDict arg = {{"x0", w0}, {"lbx", lbx}, {"ubx", ubx}, {"lbg", lbg}, {"ubg", ubg}, {"p", p}};
    DMDict res = solver_(arg);
    std::vector<double> w_opt(res.at("x").get_elements());

    // Extrair o primeiro controle
    return {w_opt[(N_ + 1) * 3]};
}