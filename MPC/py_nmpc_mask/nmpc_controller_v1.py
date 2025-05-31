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
    """Calcula os controles ótimos usando o NMPC com warm-start e retorna as previsões."""
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
    print(f"NMPC: delta={delta * 180.0 / np.pi:.3f} deg, a={a:.3f} m/s^2")

    X_pred = x_opt[0:(N + 1) * n_states].reshape((n_states, N + 1))
    X_pred_np = np.array(X_pred)  # Converter para array NumPy
    return delta, a, x_opt, X_pred_np

def prepare_reference(center_line):
    """Converte os pontos (x, y) da linha central em (x, y, psi, v) para o NMPC."""
    points = np.array(center_line)
    if len(points) < 2:
        print("Erro: Poucos pontos na linha central para calcular a referência.")
        return np.array([])

    # Calcular a orientação (psi) como a tangente entre pontos consecutivos
    psi_ref = []
    for i in range(len(points) - 1):
        dx = points[i + 1, 0] - points[i, 0]
        dy = points[i + 1, 1] - points[i, 1]
        psi = np.arctan2(dy, dx)
        psi_ref.append(psi)
    psi_ref.append(psi_ref[-1])  # Último psi é o mesmo do penúltimo

    # Definir velocidade inicial (exemplo: v = 5.0 m/s)
    v_ref = [5.0] * len(points)

    # Combinar em (x, y, psi, v)
    reference = np.column_stack((points, psi_ref, v_ref))
    return reference

def main(center_line):
    """Função principal para executar o NMPC e retornar previsões para visualização."""
    # Configurar o NMPC
    solver, n_states, n_controls, f, lbx, ubx, N = setup_nmpc()

    # Preparar a referência a partir da linha central
    reference = prepare_reference(center_line)
    if len(reference) == 0:
        print("Falha ao preparar a referência para o NMPC.")
        return None, None

    state_ref = reference[0]  # Primeira linha como referência inicial

    # Estado inicial do veículo
    state_init = np.array([0.0, 0.0, 0.0, 5.0])  # (x, y, psi, v)

    # Loop de controle
    prev_x0 = None
    X_pred = None
    for i in range(len(reference)):
        delta, a, x_opt, X_pred = compute_control(solver, n_states, n_controls, state_init, state_ref, lbx, ubx, N, prev_x0)
        # Atualizar estado (simulação simples)
        f_value = f(state_init, np.array([delta, a]))
        state_init = state_init + DT * f_value.full().flatten()
        state_ref = reference[min(i + 1, len(reference) - 1)]  # Próxima referência
        prev_x0 = x_opt
        print(f"Estado atual: x={state_init[0]:.2f}, y={state_init[1]:.2f}, psi={state_init[2]:.2f}, v={state_init[3]:.2f}")

    return X_pred, state_init  # Retornar as previsões e o estado atual

if __name__ == "__main__":
    print("Este script precisa ser chamado com uma lista de pontos da linha central.")