#include "nmpc.hpp"
#include <iostream>
#include <cmath> // Para M_PI
#include <limits> // Para std::numeric_limits

// Usar o namespace casadi para facilitar
using namespace casadi;

NMPCController::NMPCController(double L, double dt, int N_pred, double max_delta_rad,
                               double Q_offset, double Q_psi, double R_delta_rate)
    : L(L), dt(dt), N_pred(N_pred),
      Q_offset(Q_offset), Q_psi(Q_psi), R_delta_rate(R_delta_rate), // Ordem corrigida
      max_delta_rad(max_delta_rad), // Ordem corrigida
      prev_delta(0.0), prev_x(0.0), prev_y(0.0), prev_theta(0.0) {
    setupCasADiProblem();
}

void NMPCController::setupCasADiProblem() {
    // Definir variáveis simbólicas
    SX x = SX::sym("x");             // Posição x
    SX y = SX::sym("y");             // Posição y
    SX theta = SX::sym("theta");     // Orientação (yaw)
    SX delta = SX::sym("delta");     // Ângulo de direção
    SX v = SX::sym("v");             // Velocidade linear

    // Vetor de estado [x, y, theta]
    SX states = SX::vertcat({x, y, theta});
    // Variável de controle [delta]
    SX controls = SX::vertcat({delta});

    // Parâmetros de entrada para o problema de otimização
    // O estado inicial e a velocidade atual são parâmetros que mudam a cada iteração
    SX initial_state = SX::sym("initial_state", 3); // [x0, y0, theta0]
    SX current_velocity = SX::sym("current_velocity"); // v_actual

    // Definir o modelo de sistema (discretizado)
    // O modelo prevê o próximo estado (x_next, y_next, theta_next)
    SX rhs = SX::vertcat({
        x + current_velocity * cos(theta) * dt,
        y + current_velocity * sin(theta) * dt,
        theta + (current_velocity / L) * tan(delta) * dt
    });

    // Criar uma função para o modelo (f: {states, controls, current_velocity} -> {next_states})
    Function f("f", {states, controls, current_velocity}, {rhs});

    // ------ Problema de otimização ------
    // Variáveis de otimização:
    // Uma matriz para os estados previstos ao longo do horizonte (N_pred+1 estados)
    // Uma matriz para os controles (delta) ao longo do horizonte (N_pred controles)
    SX X = SX::sym("X", 3, N_pred + 1); // X = [x0, x1, ..., xN_pred]
                                       //     [y0, y1, ..., yN_pred]
                                       //     [theta0, theta1, ..., thetaN_pred]
    SX U = SX::sym("U", 1, N_pred);     // U = [delta0, delta1, ..., deltaN_pred-1]

    // Posição inicial (parâmetro)
    SX P = SX::vertcat({initial_state, current_velocity});

    // Restrição inicial: o primeiro estado previsto deve ser igual ao estado inicial
    SX g = SX::vertcat({X(Slice(), 0) - initial_state}); // x0 - initial_state = 0

    // Função custo (J)
    SX obj = 0; // Inicializa o objetivo

    // Loop sobre o horizonte de previsão
    for (int k = 0; k < N_pred; ++k) {
        // Obter estado e controle no passo k
        SX state_k = X(Slice(), k);
        SX control_k = U(Slice(), k);

        // Prever o próximo estado usando o modelo
        SX next_state_k = f(SXVector{state_k, control_k, current_velocity})[0];

        // Adicionar restrição de igualdade: o próximo estado previsto deve ser igual ao próximo estado na variável X
        g = SX::vertcat({g, next_state_k - X(Slice(), k+1)});

        // Termos da função custo
        // Penaliza desvio lateral (y) e erro de orientação (theta)
        // Assumimos que o objetivo é que o veículo esteja em y=0 e theta=0
        // (alinhado com o eixo X, que representa a pista).
        obj += Q_offset * pow(X(1, k), 2); // Penaliza desvio lateral (y_k)
        obj += Q_psi * pow(X(2, k), 2);   // Penaliza erro de orientação (theta_k)

        // Penaliza a taxa de mudança do ângulo de direção
        if (k > 0) {
            obj += R_delta_rate * pow(control_k(0) - U(0, k-1), 2);
        }
    }

    // Criar uma variável para o prev_delta no P
    SX prev_delta_param = SX::sym("prev_delta_param");
    P = SX::vertcat({P, prev_delta_param});

    // Ajustar a penalidade do primeiro delta
    obj += R_delta_rate * pow(U(0, 0) - prev_delta_param, 2);


    // Bounds nas variáveis de controle (delta)
    // Usar DM::ones() e multiplicar pelo valor desejado
    DM lbu = DM::ones(1, N_pred) * (-max_delta_rad); // CasADi DM preenchido com -max_delta_rad
    DM ubu = DM::ones(1, N_pred) * max_delta_rad;    // CasADi DM preenchido com max_delta_rad

    // Bounds nos estados (x, y, theta) - usar infinito do CasADi
    DM lbx_states = DM::ones(3, N_pred + 1) * (-DM::inf()); // Lower bound para estados
    DM ubx_states = DM::ones(3, N_pred + 1) * DM::inf();    // Upper bound para estados

    // Definir as restrições de igualdade (g = 0)
    DM lbg = DM::zeros(g.size1(), g.size2()); // Lower bound para g (zeros)
    DM ubg = DM::zeros(g.size1(), g.size2()); // Upper bound para g (zeros)

    // Criar o problema de otimização
    Dict opts;
    opts["print_time"] = 0;
    opts["ipopt.print_level"] = 0;
    opts["ipopt.print_timing_statistics"] = "no";
    opts["ipopt.max_iter"] = 100;
    opts["ipopt.warm_start_init_point"] = "yes";


    // Setup do problema NLP
    casadi::SXDict nlp = {{"x", SX::vertcat({SX::reshape(X, -1, 1), SX::reshape(U, -1, 1)})},
                          {"p", P},
                          {"f", obj},
                          {"g", g}};

    // Criar o solver
    solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // Inicializar o mapa de argumentos para o solver
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    // Combinar os limites dos estados e controles em um único DM
    arg["lbx"] = DM::vertcat({SX::reshape(lbx_states, -1, 1), SX::reshape(lbu, -1, 1)});
    arg["ubx"] = DM::vertcat({SX::reshape(ubx_states, -1, 1), SX::reshape(ubu, -1, 1)});

    // Inicializar o "warm start" para as variáveis de otimização
    arg["x0"] = DM::zeros(X.size1() * X.size2() + U.size1() * U.size2());
}

// O parâmetro 'current_theta_rad' foi removido da assinatura da função
double NMPCController::computeControl(double offset_m, double psi_rad, double current_velocity_mps) {
    DM initial_state_val = DM({prev_x, offset_m, psi_rad}); // Usando prev_x para warm-start
    DM p_val = DM::vertcat({initial_state_val, DM(current_velocity_mps), DM(prev_delta)});
    arg["p"] = p_val;

    // Defina o ponto inicial para as variáveis de otimização (warm start)
    if (!res.empty()) {
        arg["x0"] = res.at("x");
    }

    // Soluciona o problema de otimização
    res = solver(arg);

    // Extrai a primeira variável de controle (delta)
    DM opt_controls_flat = res.at("x")(Slice(3 * (N_pred + 1), 3 * (N_pred + 1) + N_pred), 0);
    double optimal_delta = static_cast<double>(opt_controls_flat(0)); // Pega o primeiro delta

    // Atualiza o prev_delta para a próxima iteração
    prev_delta = optimal_delta;

    // Atualiza o estado inicial `prev_x` para a próxima iteração (para warm-start)
    DM optimal_states_flat = res.at("x")(Slice(0, 3 * (N_pred + 1)), 0);
    prev_x = static_cast<double>(optimal_states_flat(3)); // Pega o x do próximo passo (k=1)

    return optimal_delta;
}