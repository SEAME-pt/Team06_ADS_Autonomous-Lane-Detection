#ifndef VEHICLE_INTERFACE_HPP
#define VEHICLE_INTERFACE_HPP

#include <vector>
#include <cmath>

class VehicleInterface {
public:
    VehicleInterface();
    
    // Aplica comandos de controle (simulação)
    void apply_control(double delta, double a);
    
    // Obtém o estado atual (simulado)
    std::vector<double> get_state();

private:
    std::vector<double> state_; // [x, y, psi, v]
};

#endif