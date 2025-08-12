#include "SpeedPIDTuner.hpp"
#include "SpeedPIDController.hpp"
#include "CanBusManager.hpp"
#include <cstring> // Added for memcpy
#include <cmath>
#include <limits>

static float real_velocity(float v_current, float pwm_input, float dt, CanBusManager* canBusManager) {
    (void)v_current; // Avoid unused parameter warning
    (void)pwm_input;
    (void)dt;
    if (canBusManager) {
        return canBusManager->getCurrentSpeed(); // Get real speed from CAN
    }
    return 0.0f; // Fallback if canBusManager is null
}

static float simulate_velocity(float v_current, float pwm_input, float dt) {
    float damping = 0.1f;
    float max_accel = 3.0f;
    float accel = pwm_input * max_accel / 100.0f - damping * v_current;
    return v_current + accel * dt;
}

static float get_velocity(float v_current, float pwm_input, float dt, bool real = true, CanBusManager* canBusManager = nullptr) {
    return (real ? real_velocity(v_current, pwm_input, dt, canBusManager) : simulate_velocity(v_current, pwm_input, dt));
}

static float evaluate_pid(float kp, float ki, float kd, float dt, float sim_time, float v_target, bool real = true, CanBusManager* canBusManager = nullptr) {
    SpeedPIDController pid(kp, ki, kd, 0.0f, 100.0f);
    float v = 0.0f;
    float total_error = 0.0f;
    float max_overshoot = 0.0f;
    float final_error = 0.0f;

    for (float t = 0.0f; t <= sim_time; t += dt) {
        float pwm = pid.update(v, v_target, dt);
        if (real && canBusManager) {
            std::vector<uint8_t> pwm_data(4);
            memcpy(pwm_data.data(), &pwm, sizeof(float));
            canBusManager->sendCANMessage(0x200, pwm_data); // Send PWM via CAN
        }
        v = get_velocity(v, pwm, dt, real, canBusManager);
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

std::tuple<float, float, float> auto_tune_pid(float dt, float sim_time, float v_target, bool real, CanBusManager* canBusManager) {
    float best_score = std::numeric_limits<float>::max();
    float best_kp = 0, best_ki = 0, best_kd = 0;

    for (float kp = 0.1f; kp <= 1.0f; kp += 0.1f) {
        for (float ki = 0.0f; ki <= 0.2f; ki += 0.02f) {
            for (float kd = 0.0f; kd <= 0.2f; kd += 0.02f) {
                float score = evaluate_pid(kp, ki, kd, dt, sim_time, v_target, real, canBusManager);
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