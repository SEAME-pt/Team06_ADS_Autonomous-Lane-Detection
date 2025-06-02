#include "nmpc_controller.hpp"
#include "lane_processor.hpp"
#include "vehicle_interface.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Parâmetros
    double L = 0.15;         // Distância entre eixos (m)
    double dt = 0.05;        // Passo de tempo (s)
    int Np = 10;             // Horizonte de predição
    int Nc = 5;              // Horizonte de controle
    double delta_max = 0.349;// Ângulo máximo (rad)
    double a_max = 1.0;      // Aceleração máxima (m/s^2)
    
    // Inicializa módulos
    NMPCController nmpc(L, dt, Np, Nc, delta_max, a_max);
    LaneProcessor lane_proc(Np, dt);
    VehicleInterface vehicle;
    
    // Carrega máscara estática
    cv::Mat ll_mask = cv::imread("../../mask/mask_test01.png", cv::IMREAD_GRAYSCALE);
    if (ll_mask.empty()) {
        std::cerr << "Erro ao carregar a máscara!" << std::endl;
        return -1;
    }
    
    // Processa a máscara
    auto x_ref = lane_proc.process_lane(ll_mask);
    
    // Simulação
    for (int i = 0; i < 10; ++i) { // 100 iterações de teste
        auto x0 = vehicle.get_state();
        auto control = nmpc.compute_control(x0, x_ref);
        vehicle.apply_control(control[0], control[1]);
        
        // Exibe resultados
        std::cout << "Estado: [" << x0[0] << ", " << x0[1] << ", "
                  << x0[2] << ", " << x0[3] << "] "
                  << "Controle: [" << control[0] << ", " << control[1] << "]" << std::endl;
    }
    
    return 0;
}