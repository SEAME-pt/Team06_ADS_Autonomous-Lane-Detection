#include "autotune.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>

// Declaração externa da tua variável atómica (conforme main)
extern std::atomic<double> current_speed_ms;

// Forward declaration da classe BackMotors (usa setSpeed(int), conforme o teu código).
// Inclui o header real se disponível.
class BackMotors {
public:
    bool init_motors() { return true; } // placeholder se não existir header
    void setSpeed(int s) {
        // implementação real já existe no teu projecto
        (void)s;
    }
};

using Clock = std::chrono::steady_clock;

static double clamp_double(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

PIDGains runRelayAutotune(
    BackMotors &backMotors,
    double setpoint,
    double duration_s,
    double sample_dt,
    double relay_amplitude,
    int base_output,
    const std::string &save_path
) {
    PIDGains result;
    result.ok = false;

    // Segurança
    if (sample_dt <= 0.0) sample_dt = 0.05;
    if (duration_s < 2.0) duration_s = 20.0;
    relay_amplitude = std::abs(relay_amplitude);
    int max_output = 100;
    int min_output = 0;

    int u_high = static_cast<int>(clamp_double(base_output + relay_amplitude, min_output, max_output));
    int u_low  = static_cast<int>(clamp_double(base_output - relay_amplitude, min_output, max_output));
    int last_output = base_output;

    std::cout << "[AutoTune] iniciar: setpoint=" << setpoint << " m/s, sample_dt=" << sample_dt
              << "s, duration=" << duration_s << "s, relay +/-" << relay_amplitude
              << ", outputs(" << u_low << "," << u_high << ")\n";

    // buffers
    std::vector<double> times;
    std::vector<double> values;

    // estados para detectar picos / cruzamentos
    bool first_cycle_seen = false;
    std::vector<double> crossing_times; // tempo em que a velocidade cruza o setpoint (subida/descida)
    double start_t = 0;
    auto t0 = Clock::now();

    // Forçamos inicialmente a saída para base_output
    backMotors.setSpeed(base_output);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    double elapsed = 0.0;
    double last_sample_time = 0.0;
    double time_limit = duration_s;

    // Variáveis para tracking de max/min
    double cur_max = -1e9;
    double cur_min = 1e9;

    // Run loop
    while (elapsed < time_limit) {
        auto loop_t = Clock::now();
        elapsed = std::chrono::duration<double>(loop_t - t0).count();

        // read current speed (thread-safe atomic)
        double v = current_speed_ms.load();

        // filtro muito simples: rejeita valores negativos / nan
        if (std::isnan(v) || v < 0) v = 0.0;

        times.push_back(elapsed);
        values.push_back(v);

        cur_max = std::max(cur_max, v);
        cur_min = std::min(cur_min, v);

        // RELAY LOGIC: se v < setpoint => saída = u_high, else u_low
        int out = (v < setpoint) ? u_high : u_low;
        if (out != last_output) {
            backMotors.setSpeed(out);
            last_output = out;
        }

        // detectar cruzamentos com o setpoint (v cruzando setpoint)
        if (times.size() >= 2) {
            double v_prev = values[values.size()-2];
            double t_prev = times[times.size()-2];
            double v_curr = values.back();
            double t_curr = times.back();

            // cruzamento ascendente
            if ( (v_prev < setpoint && v_curr >= setpoint) ||
                 (v_prev > setpoint && v_curr <= setpoint) ) {
                // linear interpolation para estimativa de tempo exato do cruzamento
                double frac = (setpoint - v_prev) / (v_curr - v_prev + 1e-12);
                double crossing_time = t_prev + frac * (t_curr - t_prev);
                crossing_times.push_back(crossing_time);
                // queremos pelo menos alguns cruzamentos para estimar periodo
                if (!first_cycle_seen) {
                    first_cycle_seen = true;
                    start_t = crossing_time;
                }
                std::cout << "[AutoTune] crossing at t=" << crossing_time << "s, v=" << v_curr << "\n";
            }
        }

        // se tivermos N cruzamentos (ex.: 6), podemos estimar Pu
        if (crossing_times.size() >= 6) {
            // calcular períodos entre cruzamentos de mesmo sinal: duas cruzamentos consecutivos são meio-período.
            // Para robustez, tomamos diferenças entre crossing_times[i+2] - crossing_times[i] (período completo)
            std::vector<double> periods;
            for (size_t i = 0; i + 2 < crossing_times.size(); ++i) {
                double p = crossing_times[i+2] - crossing_times[i];
                if (p > 1e-3) periods.push_back(p);
            }
            if (!periods.empty()) {
                double sum = 0.0;
                for (double p : periods) sum += p;
                double Pu = sum / periods.size();
                double amp = (cur_max - cur_min) / 2.0;
                if (amp < 1e-6) {
                    result.msg = "Amplitude too small, no oscillation detected.";
                    std::cerr << "[AutoTune] " << result.msg << std::endl;
                    break;
                }
                // relay amplitude in output units (difference between high and low) divided by 2 is h
                double h = (u_high - u_low) / 2.0;
                // Ku formula from relay method: Ku = (4*h) / (pi * a)
                double Ku = (4.0 * h) / (M_PI * amp);

                std::cout << "[AutoTune] Pu=" << Pu << "s, amp=" << amp << " m/s, h=" << h << "\n";
                std::cout << "[AutoTune] Ku=" << Ku << "\n";

                // Ziegler-Nichols (classic) tuning (continuous PID)
                double Kp = 0.6 * Ku;
                double Ti = 0.5 * Pu; // integral time
                double Td = 0.125 * Pu; // derivative time

                double Ki = (Ti > 1e-9) ? (Kp / Ti) : 0.0; // Ki as in controller (Kp/Ti)
                double Kd = Kp * Td;

                // Guard rails to avoid crazy gains
                if (!std::isfinite(Kp) || !std::isfinite(Ki) || !std::isfinite(Kd)) {
                    result.msg = "Computed gains not finite.";
                    std::cerr << "[AutoTune] " << result.msg << std::endl;
                    break;
                }
                if (Kp < 0) Kp = std::abs(Kp);
                if (Ki < 0) Ki = std::abs(Ki);
                if (Kd < 0) Kd = std::abs(Kd);

                result.kp = Kp;
                result.ki = Ki;
                result.kd = Kd;
                result.ok = true;
                result.msg = "OK";

                // save
                if (!save_path.empty()) {
                    std::ofstream ofs(save_path);
                    if (ofs) {
                        ofs << result.kp << " " << result.ki << " " << result.kd << "\n";
                        ofs.close();
                        std::cout << "[AutoTune] Gains saved to " << save_path << "\n";
                    } else {
                        std::cerr << "[AutoTune] WARNING: Could not save gains to " << save_path << "\n";
                    }
                }

                break;
            }
        }

        // sleep until next sample
        std::this_thread::sleep_for(std::chrono::duration<double>(sample_dt));
    }

    // ensure motors stopped / safe at the end
    backMotors.setSpeed(0);
    std::cout << "[AutoTune] finished. OK=" << result.ok << " msg=" << result.msg << "\n";
    if (!result.ok && result.msg.empty()) result.msg = "Autotune failed or timed out.";

    return result;
}
