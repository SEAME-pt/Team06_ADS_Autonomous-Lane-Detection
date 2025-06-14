# Model Predictive Control (MPC) for Autonomous Driving on Jetson Nano with CARLA
This project focuses on implementing a Model Predictive Control (MPC) for autonomous driving, tested in the CARLA simulator and optimized for the Jetson Nano, a resource-constrained embedded platform. The goal is to provide a practical design, comparing linear and non-linear MPC approaches, and offering implementation guidelines for integration with CARLA.

## What is MPC?
Model Predictive Control (MPC) is an advanced control strategy that uses a dynamic model of the system (here, a vehicle) to predict its future behavior and optimize control actions (e.g., acceleration, braking, steering) over a prediction horizon. It solves a real-time optimization problem, respecting physical constraints and objectives like trajectory tracking or energy efficiency.

## Essential Components of an MPC

An MPC requires the following elements:

* Dynamic Model: Represents the vehicle's behavior (position, velocity, orientation). The kinematic bicycle model is commonly used for its simplicity and effectiveness at moderate speeds.
* Cost Function: Defines optimization goals, such as minimizing trajectory error or control effort for smooth driving.
* Constraints: Include physical limits (e.g., maximum steering angle) and safety constraints (e.g., collision avoidance).
* Prediction Horizon: Number of future time steps considered (e.g., 1 second ahead).
* Control Horizon: Number of steps where control inputs (e.g., steering, acceleration) are optimized.
* Optimization Algorithm: Solves the numerical optimization problem in real-time (e.g., Quadratic Programming for linear MPC, IPOPT for non-linear MPC).
* Discretization: Converts the continuous model into discrete time steps (e.g., 0.1 s) for computational implementation.

## Key Equations for MPC
## 1. Dynamic Model (Kinematic Bicycle Model)
The kinematic bicycle model is widely used in autonomous driving for its simplicity. It describes the vehicle's dynamics based on position $(x, y)$, orientation $\psi$, and velocity $v$. The control inputs are the steering angle $\delta$ and acceleration $a$. The continuous-time equations are:
$$\dot{x} = v \cos(\psi)$$
$$\dot{y} = v \sin(\psi)$$
$$\dot{\psi} = \frac{v}{L} \tan(\delta)$$
$$\dot{v} = a$$
Where:

* $x, y$: Vehicle coordinates in the plane.
* $\psi$: Yaw angle (orientation).
* $v$: Longitudinal velocity.
* $\delta$: Steering angle.
* $a$: Acceleration.
* $L$: Wheelbase (fixed vehicle parameter).

For MPC, these equations are discretized using, e.g., the Euler method:
$$x_{k+1} = x_k + v_k \cos(\psi_k) \cdot \Delta t$$
$$y_{k+1} = y_k + v_k \sin(\psi_k) \cdot \Delta t$$
$$\psi_{k+1} = \psi_k + \frac{v_k}{L} \tan(\delta_k) \cdot \Delta t$$
$$v_{k+1} = v_k + a_k \cdot \Delta t$$
Where $\Delta t$ is the time step (e.g., 0.1 s), and $k$ is the step index.

## 2. Cost Function
The cost function $J$ minimizes the error between the vehicle's current state and the desired trajectory, while penalizing excessive control effort. A typical form is:
$$J = \sum_{k=1}^{N_p} \left[ | \mathbf{x}k - \mathbf{x}{ref,k} |_Q^2 + | \mathbf{u}_k |_R^2 \right]$$
Where:

* $\mathbf{x}_k = [x_k, y_k, \psi_k, v_k]^T$: State vector at step $k$.
* $\mathbf{x}_{ref,k}$: Reference state (desired trajectory).
* $\mathbf{u}_k = [\delta_k, a_k]^T$: Control vector.
* $N_p$: Prediction horizon.
* $Q$: Weight matrix for state errors (e.g., prioritizing position accuracy).
* $R$: Weight matrix for control effort (e.g., penalizing abrupt steering).

## 3. Constraints
Examples of constraints:

* States: Velocity limits ($v_{min} \leq v_k \leq v_{max}$).
* Controls: Steering limits ($\delta_{min} \leq \delta_k \leq \delta_{max}$) and acceleration limits ($a_{min} \leq a_k \leq a_{max}$).
* Safety: Obstacle avoidance.

## 4. Optimization Problem
The MPC solves:
$$\min_{\mathbf{u}} J(\mathbf{x}, \mathbf{u})$$
Subject to:

* Dynamic model: $\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$.
* Constraints: $\mathbf{x}_k \in \mathcal{X}$, $\mathbf{u}_k \in \mathcal{U}$.

## Linear vs. Non-Linear MPC
### Linear MPC

How it works: Linearizes the dynamic model around an operating condition (e.g., $\delta = 0$, $v = v_0$), using a linear model like:

$\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k$.

### Advantages:

* Computationally lightweight, suitable for Jetson Nano's limited resources.
* Fast and reliable Quadratic Programming (QP) solvers (e.g., OSQP).
* Effective for smooth trajectories and moderate speeds.


### Disadvantages:
* Loses accuracy in non-linear conditions (e.g., tight curves, high speeds).
* Less robust to significant dynamic changes.


### Non-Linear MPC (NMPC)

How it works: Uses the non-linear dynamic model directly (e.g., kinematic bicycle equations without linearization).

### Advantages:
* More accurate in complex scenarios (e.g., sharp turns, evasive maneuvers).
* Better handles non-linear vehicle dynamics.


### Disadvantages:
* Computationally intensive, challenging for Jetson Nano without optimization.
* Requires complex solvers (e.g., IPOPT), which may be slow on embedded hardware.
* More complex to implement.


### Implementation Steps

#### 1. Define the Dynamic Model:
* The discretized kinematic bicycle model.
* Vehicle parameters (e.g., $L$, limits for $\delta$ and $a$).


#### 2. Set Up the Cost Function:
* Choose weights $Q$ and $R$ to balance trajectory accuracy and control smoothness.
* Example: $Q = \text{diag}(10, 10, 5, 1)$ for prioritizing position and orientation, $R = \text{diag}(1, 1)$ for controls.


#### 3. Define Constraints:
* Example: $\delta \in [-30^\circ, 30^\circ]$, $a \in [-3 , \text{m/s}^2, 3 , \text{m/s}^2]$, $v \in [0, 20 , \text{m/s}]$.
* Sensors (cameras) for obstacle avoidance constraints.


#### 4. Choose a Solver:
* For linear MPC: Use lightweight solvers like OSQP or qpOASES.
* For NMPC: Use CasADi with IPOPT, optimized for real-time performance.


#### 5. Integrate with CARLA:
* CARLA's Python API to obtain states ($x, y, \psi, v$) and reference trajectories.
* Send control commands ($\delta, a$) to the simulated vehicle.


#### 6. Optimize for Jetson Nano:
* C++ for efficiency.
* Reduce prediction horizon (e.g., $N_p = 10$, $N_c = 2$).
* Compile solvers.


## dependencies

sudo apt-get update
sudo apt-get install libopencv-dev

## Compile process_mask
g++ -o process_mask process_mask.cpp `pkg-config --cflags --libs opencv4`
./process_mask

## Compile dynamic_model
g++ -o dynamic_model dynamic_model.cpp
./dynamic_model

## Compile MPC
g++ -o mpc mpc.cpp -I/usr/include/eigen3
./mpc

## Compile mpc_integrated
g++ -o mpc_integrated mpc_integrated.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/eigen3
./mpc_integrated

## Compile mpc_carla
g++ -o mpc_carla mpc_carla.cpp `pkg-config --cflags --libs opencv4` -I/home/ndo-vale/Desktop/carla/include -L/home/ndo-vale/Desktop/carla/lib -lcarla_client -lboost_python
./mpc_carla

## Compile cpp_infer
g++ -o lane_detec lane_detection.cpp `pkg-config --cflags --libs opencv4` \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart \
-I/usr/include/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu \
-lnvinfer -lnvinfer_plugin -lpthread


