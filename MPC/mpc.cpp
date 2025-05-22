#include <cmath>
#include <vector>
#include <iostream>

class VehicleModel {
public:
    // Vehicle parameters
    const float v = 3.0f;      // Speed: 3 m/s
    const float L = 0.15f;     // Wheelbase: 0.15 m
    const float dt = 0.05f;    // Time step: 0.05 s (20 Hz)
    const float meters_per_pixel = 0.001f; // Conversion: 0.001 m/pixel (estimate)

    // State: [y, psi]
    float y;    // Lateral position (meters)
    float psi;  // Heading angle (radians)

    VehicleModel() : y(0.0f), psi(0.0f) {}

    // Update state based on steering angle delta
    void update(float delta) {
        y += dt * v * std::sin(psi);           // y[k+1] = y[k] + dt * v * sin(psi[k])
        psi += dt * (v / L) * std::tan(delta); // psi[k+1] = psi[k] + dt * (v / L) * tan(delta[k])
    }

    // Convert pixel shift to meters
    float convertShift(float pixel_shift) {
        return pixel_shift * meters_per_pixel;
    }
};

class MPCController {
public:
    // MPC parameters
    const int N = 10;          // Prediction horizon
    const float R = 0.1f;      // Control weight
    const float delta_max = M_PI / 6.0f; // Max steering angle: 30 degrees
    const float dt = 0.05f;    // Time step: 0.05 s

    VehicleModel& model;       // Reference to vehicle model

    MPCController(VehicleModel& vehicle) : model(vehicle) {}

    // Compute cost for a given control sequence
    float computeCost(const std::vector<float>& delta, float y_ref) {
        float cost = 0.0f;
        float y = model.y;
        float psi = model.psi;

        // Simulate model over horizon
        for (int k = 0; k < N; ++k) {
            cost += std::pow(y - y_ref, 2) + R * std::pow(delta[k], 2);
            y += dt * model.v * std::sin(psi);
            psi += dt * (model.v / model.L) * std::tan(delta[k]);
        }
        return cost;
    }

    // Simple gradient descent to find optimal steering angle
    float optimize(float y_ref) {
        float delta = 0.0f; // Initial guess
        float learning_rate = 0.01f;
        float cost, cost_plus, cost_minus;
        const int max_iterations = 50;

        for (int i = 0; i < max_iterations; ++i) {
            std::vector<float> delta_seq(N, delta); // Constant control sequence
            cost = computeCost(delta_seq, y_ref);

            // Compute gradient (numerical)
            std::vector<float> delta_plus = delta_seq;
            delta_plus[0] = std::min(std::max(delta + 0.01f, -delta_max), delta_max);
            cost_plus = computeCost(delta_plus, y_ref);

            std::vector<float> delta_minus = delta_seq;
            delta_minus[0] = std::min(std::max(delta - 0.01f, -delta_max), delta_max);
            cost_minus = computeCost(delta_minus, y_ref);

            float gradient = (cost_plus - cost_minus) / 0.02f;
            delta -= learning_rate * gradient;

            // Enforce constraints
            delta = std::min(std::max(delta, -delta_max), delta_max);
        }

        return delta;
    }
};

// Example usage
int main() {
    VehicleModel vehicle;
    MPCController mpc(vehicle);

    // Example: pixel shift from mask (-10 pixels)
    float pixel_shift = -10.0f;
    float y_ref = vehicle.convertShift(pixel_shift); // -0.01 m

    std::cout << "Desired shift: " << y_ref << " meters" << std::endl;

    // Simulate 10 steps
    for (int k = 0; k < 10; ++k) {
        float delta = mpc.optimize(y_ref);
        vehicle.update(delta);
        std::cout << "Step " << k << ": y = " << vehicle.y << " m, psi = " << vehicle.psi
                  << " rad, delta = " << delta * 180.0f / M_PI << " deg" << std::endl;
    }

    return 0;
}