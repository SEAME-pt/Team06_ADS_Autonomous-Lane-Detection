#pragma once
#include <tuple>

class CanBusManager; // Forward declaration

std::tuple<float, float, float> auto_tune_pid(float dt = 0.1f, float sim_time = 10.0f, float v_target = 2.0f, bool real = true, CanBusManager* canBusManager = nullptr);