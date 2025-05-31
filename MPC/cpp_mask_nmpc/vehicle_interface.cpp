#include "vehicle_interface.hpp"

VehicleInterface::VehicleInterface() {
    state_ = {0.0, 0.0, 0.0, 0.5}; // Estado inicial: [x=0, y=0, psi=0, v=0.5]
}

void VehicleInterface::apply_control(double delta, double a) {
    // Simula a dinâmica do veículo (Euler)
    double L = 0.15;
    double dt = 0.05;
    double x = state_[0], y = state_[1], psi = state_[2], v = state_[3];
    
    state_[0] = x + dt * v * std::cos(psi);
    state_[1] = y + dt * v * std::sin(psi);
    state_[2] = psi + dt * (v / L) * std::tan(delta);
    state_[3] = v + dt * a;
}

std::vector<double> VehicleInterface::get_state() {
    return state_;
}