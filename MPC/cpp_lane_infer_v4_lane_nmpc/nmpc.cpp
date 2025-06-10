#include "lane_detection.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

// Modelo de bicicleta
struct VehicleState {
    float x = 0.0;    // Posição x (m)
    float y = 0.0;    // Posição y (m)
    float psi = 0.0;  // Orientação (rad)
    float v = 1.0;    // Velocidade (m/s)
};

struct Control {
    float delta = 0.0; // Ângulo de direção (rad)
};

const float L = 0.15;  // Wheelbase (m)
const float dt = 0.1; // Passo de tempo (s)
const int N = 10;     // Horizonte de previsão

// Atualiza estado usando modelo de bicicleta
VehicleState updateState(const VehicleState& state, const Control& u) {
    VehicleState next;
    next.x = state.x + state.v * cos(state.psi) * dt;
    next.y = state.y + state.v * sin(state.psi) * dt;
    next.psi = state.psi + (state.v / L) * tan(u.delta) * dt;
    next.v = state.v;
    return next;
}

// Calcula erro lateral e angular
void computeErrors(const VehicleState& state, const LaneData& laneData, float& e, float& psi_e) {
    if (!laneData.valid || laneData.num_points < 2) {
        e = 0.0;
        psi_e = 0.0;
        return;
    }

    // Encontrar ponto mais próximo
    float min_dist = std::numeric_limits<float>::max();
    int closest_idx = 0;
    for (int i = 0; i < laneData.num_points; ++i) {
        float dx = laneData.points[i].x - state.x;
        float dy = laneData.points[i].y - state.y;
        float dist = sqrt(dx * dx + dy * dy);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }

    // Erro lateral (distância com sinal)
    float dx = laneData.points[closest_idx].x - state.x;
    float dy = laneData.points[closest_idx].y - state.y;
    e = sqrt(dx * dx + dy * dy);
    if (dx * sin(state.psi) - dy * cos(state.psi) < 0) e = -e;

    // Erro angular (usar próximo ponto para tangente)
    int next_idx = std::min(closest_idx + 1, laneData.num_points - 1);
    float ref_dx = laneData.points[next_idx].x - laneData.points[closest_idx].x;
    float ref_dy = laneData.points[next_idx].y - laneData.points[closest_idx].y;
    float psi_ref = atan2(ref_dy, ref_dx);
    psi_e = state.psi - psi_ref;
    while (psi_e > M_PI) psi_e -= 2 * M_PI;
    while (psi_e < -M_PI) psi_e += 2 * M_PI;
}

// Função de custo
float computeCost(const std::vector<VehicleState>& states, const LaneData& laneData) {
    float cost = 0.0;
    for (int k = 0; k < states.size(); ++k) {
        float e, psi_e;
        computeErrors(states[k], laneData, e, psi_e);
        cost += 10.0 * e * e + 5.0 * psi_e * psi_e;
    }
    return cost;
}

// NMPC simplificado (gradiente descendente)
Control optimizeControl(const VehicleState& initial_state, const LaneData& laneData) {
    Control u;
    float best_delta = 0.0;
    float best_cost = std::numeric_limits<float>::max();

    // Testar valores de delta
    for (float delta = -0.5; delta <= 0.5; delta += 0.01) {
        std::vector<VehicleState> states(N + 1);
        states[0] = initial_state;
        Control u_temp{delta};

        for (int k = 0; k < N; ++k) {
            states[k + 1] = updateState(states[k], u_temp);
        }

        float cost = computeCost(states, laneData);
        if (cost < best_cost) {
            best_cost = cost;
            best_delta = delta;
        }
    }

    u.delta = best_delta;
    return u;
}

int main() {
    // Ler LaneData de um arquivo (para teste)
    std::ifstream file("../lane_data.txt");
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir lane_data.txt\n";
        return 1;
    }

    LaneData laneData;
    file >> laneData.valid >> laneData.timestamp >> laneData.num_points;
    for (size_t i = 0; i < laneData.num_points; ++i) {
        file >> laneData.points[i].x >> laneData.points[i].y;
    }
    file.close();

    // Estado inicial do veículo
    VehicleState state;

    // Executar NMPC
    Control u = optimizeControl(state, laneData);
    std::cout << "Comando NMPC: delta = " << u.delta << " rad\n";

    return 0;
}