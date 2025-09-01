// trapezoidal.hpp
#ifndef TRAPEZOIDAL_HPP
#define TRAPEZOIDAL_HPP

class TrapezoidalProfile {
public:
    TrapezoidalProfile(double max_vel_deg_s, double max_acc_deg_s2)
        : max_vel_deg_s_(max_vel_deg_s), max_acc_deg_s2_(max_acc_deg_s2),
          current_angle_(0.0), current_vel_(0.0) {}

    // Calcula o próximo ângulo suavizado
    double computeNextAngle(double target_angle, double dt) {
        // Diferença para o alvo
        double error = target_angle - current_angle_;
        
        // Calcular velocidade desejada (respeitando aceleração)
        double acc_time = max_vel_deg_s_ / max_acc_deg_s2_; // Tempo para atingir velocidade máxima
        double acc_distance = 0.5 * max_vel_deg_s_ * acc_time; // Distância durante aceleração
        
        double desired_vel;
        if (std::abs(error) <= acc_distance) {
            // Desaceleração (triângulo)
            desired_vel = std::sqrt(2.0 * max_acc_deg_s2_ * std::abs(error)) * (error >= 0 ? 1.0 : -1.0);
        } else {
            // Velocidade constante
            desired_vel = max_vel_deg_s_ * (error >= 0 ? 1.0 : -1.0);
        }
        
        // Limitar aceleração
        double delta_vel = desired_vel - current_vel_;
        double max_delta_vel = max_acc_deg_s2_ * dt;
        delta_vel = std::max(-max_delta_vel, std::min(max_delta_vel, delta_vel));
        
        // Atualizar velocidade e ângulo
        current_vel_ += delta_vel;
        current_angle_ += current_vel_ * dt;
        
        // Garantir que não ultrapassa o alvo
        if ((error >= 0 && current_angle_ > target_angle) || (error < 0 && current_angle_ < target_angle)) {
            current_angle_ = target_angle;
            current_vel_ = 0.0;
        }
        
        return current_angle_;
    }

private:
    double max_vel_deg_s_;   // Velocidade angular máxima (graus/s)
    double max_acc_deg_s2_;  // Aceleração angular máxima (graus/s²)
    double current_angle_;    // Ângulo atual
    double current_vel_;     // Velocidade angular atual
};

#endif // TRAPEZOIDAL_HPP