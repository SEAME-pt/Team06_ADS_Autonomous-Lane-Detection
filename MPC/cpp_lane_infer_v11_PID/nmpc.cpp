#include "nmpc.hpp"
#include <iostream>

NMPCController::NMPCController() {
    using namespace casadi;

    // Define state and control variables
    MX x_state = MX::sym("x", 4); // [x, y, theta, v]
    MX u_control = MX::sym("u", 1); // delta

    // Define system dynamics
    MX x_dot = vertcat(
        x_state(3) * cos(x_state(2)),
        x_state(3) * sin(x_state(2)),
        (x_state(3) / L) * tan(u_control),
        0 // Assume constant velocity for simplicity
    );

    // Discretize: x(k+1) = x(k) + dt * x_dot
    MX x_next = x_state + dt * x_dot;

    // Create CasADi function for dynamics
    Function f = Function("f", {x_state, u_control}, {x_next});

    // Optimization problem setup
    x = std::vector<MX>(N + 1); // State trajectory
    u = std::vector<MX>(N);     // Control trajectory
    std::vector<MX> params = {
        opti.variable(4),              // Initial state x0
        opti.parameter(N + 1),         // Reference y (ref_y)
        opti.parameter(N + 1)          // Reference theta (ref_theta)
    };

    x[0] = params[0]; // Initial state
    cost = 0;

    // Build cost function and constraints
    for (int k = 0; k < N; ++k) {
        u[k] = opti.variable(1); // Declare control variable within Opti
        opti.set_initial(u[k], 0.0); // Initialize control to zero
        x[k + 1] = f(std::vector<MX>{x[k], u[k]})[0];

        // Cost: penalize lateral error, yaw error, and control effort
        MX e_y = x[k + 1](1) - params[1](k + 1); // Lateral error
        MX e_theta = x[k + 1](2) - params[2](k + 1); // Yaw error
        cost += Q_pos * e_y * e_y + Q_yaw * e_theta * e_theta + R_delta * u[k] * u[k];

        // Constraints: |delta| <= delta_max
        opti.subject_to(u[k] >= -delta_max);
        opti.subject_to(u[k] <= delta_max);
    }

    // Setup solver with additional IPOPT options for robustness
    opti.minimize(cost);
    casadi::Dict opts;
    opts["ipopt.print_level"] = 0; // Suppress IPOPT output
    opts["print_time"] = 0;
    opts["ipopt.tol"] = 1e-6; // Solver tolerance
    opts["ipopt.max_iter"] = 100; // Maximum iterations
    opti.solver("ipopt", opts);
    solver = opti.to_function("solver", {params[0], params[1], params[2]}, {u[0]});
}

double NMPCController::computeControl(double offset, double psi, double theta, double v) {
    using namespace casadi;

    // Debug: print inputs
    std::cout << "Inputs: offset=" << offset << " m, psi=" << psi << " rad, theta=" << theta << " rad, v=" << v << " m/s" << std::endl;

    // Reference trajectory: aim to reduce errors to zero
    std::vector<double> ref_y(N + 1, 0.0); // Target y = 0 (no lateral error)
    std::vector<double> ref_theta(N + 1, theta + psi); // Target theta to correct psi to zero

    // Current state
    std::vector<double> x0 = {0.0, offset, theta, v}; // x=0 (arbitrary), y=offset, theta, v

    // Convert inputs to DM (numerical) for solver
    DM x0_dm = DM(x0);
    DM ref_y_dm = DM(ref_y);
    DM ref_theta_dm = DM(ref_theta);

    // Call solver with explicit std::vector<DM>
    auto result = solver(std::vector<DM>{x0_dm, ref_y_dm, ref_theta_dm});
    double delta = static_cast<double>(result[0](0));

    // Debug: print computed delta
    std::cout << "Computed delta: " << delta * 180.0 / M_PI << " deg" << std::endl;

    return delta; // Return steering angle in radians
}

void NMPCController::setState(double x, double y, double theta, double v) {
    // Not used in this implementation since state is passed directly to computeControl
    // Included for compatibility with potential state updates
}