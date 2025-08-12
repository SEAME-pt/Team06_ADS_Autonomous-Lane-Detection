#ifndef AUTOTUNE_HPP
#define AUTOTUNE_HPP

#include <vector>
#include <string>
#include <cstdint>

struct PIDGains {
    double kp;
    double ki;
    double kd;
    bool ok;
    std::string msg;
};

/// runRelayAutotune
/// - backMotors: referência ao controlador de motores (usa setSpeed(int))
/// - setpoint: velocidade alvo em m/s durante o autotune
/// - duration_s: duração máxima do autotune (segundos)
/// - sample_dt: intervalo de amostragem (s)
/// - relay_amplitude: amplitude do relé em percent points (ex. 15 -> +/-15 sobre base)
/// - base_output: nível de saída central (0..100)
/// - save_path: se não vazio, guarda gains para este ficheiro (texto)
PIDGains runRelayAutotune(
    class BackMotors &backMotors,
    double setpoint,
    double duration_s = 20.0,
    double sample_dt = 0.05,
    double relay_amplitude = 20.0,
    int base_output = 60,
    const std::string &save_path = "pid_gains.txt"
);

#endif // AUTOTUNE_HPP
