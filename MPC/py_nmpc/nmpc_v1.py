import casadi as ca
import numpy as np

# Parâmetros do veículo
L = 2.9  # Distância entre eixos (m)
MAX_STEER = np.deg2rad(20.0)  # Máximo ângulo de direção (rad)
MAX_ACC = 3.0  # Máxima aceleração (m/s^2)
MAX_SPEED = 20.0  # Máxima velocidade (m/s)
DT = 0.05  # Passo de tempo (s)

# Parâmetros do NMPC
N = 5  # Horizonte de predição
Q = np.diag([10.0, 10.0, 5.0, 1.0])  # Pesos para estados [x, y, psi, v]
R = np.diag([1.0, 1.0])  # Pesos para controles [delta, a]

def setup_nmpc():
    """Configura o controlador NMPC usando CasADi."""
    # Variáveis simbólicas
    x = ca.SX.sym('x')  # Posição x
    y = ca.SX.sym('y')  # Posição y
    psi = ca.SX.sym('psi')  # Orientação
    v = ca.SX.sym('v')  # Velocidade
    states = ca.vertcat(x, y, psi, v)
    n_states = states.size1()

    delta = ca.SX.sym('delta')  # Ângulo de direção
    a = ca.SX.sym('a')  # Aceleração
    controls = ca.vertcat(delta, a)
    n_controls = controls.size1()

    # Modelo dinâmico (bicicleta cinemático)
    rhs = ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        v / L * ca.tan(delta),
        a
    )

    # Função de dinâmica
    f = ca.Function('f', [states, controls], [rhs])

    # Variáveis de otimização
    U = ca.SX.sym('U', n_controls, N)  # Controles
    X = ca.SX.sym('X', n_states, N + 1)  # Estados
    P = ca.SX.sym('P', n_states + n_states)  # Estado inicial + referência

    # Função de custo
    obj = 0
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        st_ref = P[4:8]
        obj += ca.mtimes([(st - st_ref).T, Q, (st - st_ref)]) + ca.mtimes([con.T, R, con])

    # Restrições
    g = []
    g += [X[:, 0] - P[0:4]]  # Estado inicial
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        st_next = X[:, k + 1]
        f_value = f(st, con)
        st_next_euler = st + DT * f_value
        g += [st_next - st_next_euler]

    # Restrições de controles e estados
    lbx = []
    ubx = []
    lbg = []
    ubg = []

    for k in range(N):
        lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0]  # v >= 0
        ubx += [ca.inf, ca.inf, ca.inf, MAX_SPEED]
        lbx += [-MAX_STEER, -MAX_ACC]  # delta, a
        ubx += [MAX_STEER, MAX_ACC]
        lbg += [0.0] * n_states
        ubg += [0.0] * n_states

    lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0]  # Último estado
    ubx += [ca.inf, ca.inf, ca.inf, MAX_SPEED]

    # Problema de otimização
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    nlp = {'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
           'f': obj, 'g': ca.vertcat(*g), 'p': P}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, n_states, n_controls

def compute_control(solver, n_states, n_controls, state_init, state_ref):
    """Calcula os controles ótimos usando o NMPC."""
    # Inicializar variáveis de otimização
    x0 = np.zeros((N + 1) * n_states + N * n_controls)
    for k in range(N + 1):
        x0[k * n_states:(k + 1) * n_states] = state_init

    # Configurar parâmetros
    p = np.concatenate((state_init, state_ref))

    # Resolver o problema de otimização
    res = solver(x0=x0, p=p, lbg=0, ubg=0)
    u_opt = res['x'][-N * n_controls:].reshape((n_controls, N))
    delta = float(u_opt[0, 0])
    a = float(u_opt[1, 0])
    return delta, a