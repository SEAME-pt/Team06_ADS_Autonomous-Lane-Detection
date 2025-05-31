import casadi as ca
import numpy as np

# Parâmetros do veículo
L = 2.9
MAX_STEER = np.deg2rad(30.0)
MAX_ACC = 3.0
MAX_SPEED = 20.0
DT = 0.1

# Parâmetros do NMPC
N = 8
Q = np.diag([5.0, 5.0, 10.0, 10.0])  # Aumentar peso da velocidade
R = np.diag([20.0, 1.0])  # Aumentar penalidade em delta

def setup_nmpc():
    """Configura o controlador NMPC usando CasADi."""
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    psi = ca.SX.sym('psi')
    v = ca.SX.sym('v')
    states = ca.vertcat(x, y, psi, v)
    n_states = states.size1()

    delta = ca.SX.sym('delta')
    a = ca.SX.sym('a')
    controls = ca.vertcat(delta, a)
    n_controls = controls.size1()

    rhs = ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        v / L * ca.tan(delta),
        a
    )

    f = ca.Function('f', [states, controls], [rhs])

    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N + 1)
    P = ca.SX.sym('P', n_states + n_states)

    obj = 0
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        st_ref = P[4:8]
        obj += ca.mtimes([(st - st_ref).T, Q, (st - st_ref)]) + ca.mtimes([con.T, R, con])

    g = [X[:, 0] - P[0:4]]
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        st_next = X[:, k + 1]
        f_value = f(st, con)
        st_next_euler = st + DT * f_value
        g += [st_next - st_next_euler]

    lbx = []
    ubx = []
    lbg = [0.0] * n_states * (N + 1)
    ubg = [0.0] * n_states * (N + 1)

    for k in range(N):
        lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0]
        ubx += [ca.inf, ca.inf, ca.inf, MAX_SPEED]
        lbx += [-MAX_STEER, -MAX_ACC]
        ubx += [MAX_STEER, MAX_ACC]

    lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0]
    ubx += [ca.inf, ca.inf, ca.inf, MAX_SPEED]

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100, 'ipopt.tol': 1e-6}
    nlp = {'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
           'f': obj, 'g': ca.vertcat(*g), 'p': P}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, n_states, n_controls, f, lbx, ubx, N

def compute_control(solver, n_states, n_controls, state_init, state_ref, lbx, ubx, N, prev_x0=None):
    """Calcula os controles ótimos usando o NMPC com warm-start."""
    x0 = prev_x0 if prev_x0 is not None else np.zeros((N + 1) * n_states + N * n_controls)
    for k in range(N + 1):
        x0[k * n_states:(k + 1) * n_states] = state_init

    p = np.concatenate((state_init, state_ref))
    res = solver(x0=x0, p=p, lbg=0, ubg=0, lbx=lbx, ubx=ubx)
    x_opt = res['x']
    u_opt = x_opt[-N * n_controls:].reshape((n_controls, N))
    delta = float(u_opt[0, 0])
    a = float(u_opt[1, 0])

    delta = np.clip(delta, -MAX_STEER, MAX_STEER)
    print(f"NMPC: delta={delta * 180.0 / np.pi:.3f} deg, a={a:.3f} m/s^2")  # Depuração

    X_pred = x_opt[0:(N + 1) * n_states].reshape((n_states, N + 1))
    return delta, a, x_opt, X_pred