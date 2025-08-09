#include "SpeedPIDTuner.hpp"
#include "SpeedPIDController.hpp"
#include <cmath>
#include <limits>

float speed_from_zermq = 0.0f; // Placeholder for the speed from ZMQ service

static float real_velocity(float v_current, float pwm_input, float dt) {
    (void)v_current; // Evita aviso de parâmetro não utilizado
    (void)pwm_input;
    (void)dt;
    return speed_from_zermq; // Placeholder: substituir por leitura real de sensor
}

static float simulate_velocity(float v_current, float pwm_input, float dt) {
    float damping = 0.1f;
    float max_accel = 3.0f;
    float accel = pwm_input * max_accel / 100.0f - damping * v_current;
    return v_current + accel * dt;
}

static float get_velocity(float v_current, float pwm_input, float dt, bool real = true) {
    return (real ? real_velocity(v_current, pwm_input, dt) : simulate_velocity(v_current, pwm_input, dt));
}

static float evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target, bool real = true) {
    SpeedPIDController pid(kp, ki, kd, 0.0f, 100.0f);
    float v = 0.0f;
    float total_error = 0.0f;
    float max_overshoot = 0.0f;
    float final_error = 0.0f;

    for (float t = 0.0f; t <= sim_time; t += dt) {
        float pwm = pid.update(v, v_target, dt);
        v = get_velocity(v, pwm, dt, real);
        float error = std::abs(v_target - v);
        total_error += error * dt;

        if (v > v_target) {
            max_overshoot = std::max(max_overshoot, v - v_target);
        }

        if (t >= sim_time - 1.0f) {
            final_error += error * dt;
        }
    }

    return total_error + 10.0f * max_overshoot + 20.0f * final_error;
}

std::tuple<float, float, float> auto_tune_pid(float dt, float sim_time, float v_target, bool real) {
    float best_score = std::numeric_limits<float>::max();
    float best_kp = 0, best_ki = 0, best_kd = 0;

    for (float kp = 0.1f; kp <= 1.0f; kp += 0.1f) {
        for (float ki = 0.0f; ki <= 0.2f; ki += 0.02f) {
            for (float kd = 0.0f; kd <= 0.2f; kd += 0.02f) {
                float score = evaluate_pid(kp, ki, kd, dt, sim_time, v_target, real);
                if (score < best_score) {
                    best_score = score;
                    best_kp = kp;
                    best_ki = ki;
                    best_kd = kd;
                }
            }
        }
    }

    return {best_kp, best_ki, best_kd};
}