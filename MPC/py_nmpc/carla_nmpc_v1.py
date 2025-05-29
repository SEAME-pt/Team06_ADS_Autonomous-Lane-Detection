import carla
import cv2
import numpy as np
import casadi as ca
import time
import math
import random

# Configurações de exibição
WIDTH = 640
HEIGHT = 320

# Parâmetros do veículo
L = 2.9  # Distância entre eixos (m)
MAX_STEER = np.deg2rad(30.0)  # Máximo ângulo de direção (rad)
MAX_ACC = 3.0  # Máxima aceleração (m/s^2)
MAX_SPEED = 20.0  # Máxima velocidade (m/s)
DT = 0.05  # Passo de tempo (s)

# Parâmetros do NMPC
N = 10  # Horizonte de predição
Q = np.diag([10.0, 10.0, 5.0, 1.0])  # Pesos para estados [x, y, psi, v]
R = np.diag([1.0, 1.0])  # Pesos para controles [delta, a]

# Variáveis globais
current_image = None
current_mask = None
frame_count = 0

def process_mask(mask):
    """
    Processa a máscara de segmentação para extrair y_ref e psi_ref.
    Retorna: delta (ângulo de direção sugerido), y_ref, psi_ref, lines_detected.
    """
    # Definir ROI
    roi_x = int(WIDTH * 0.10)
    roi_width = int(WIDTH * 0.80)
    roi_y = int(HEIGHT * 0.5)
    roi_height = int(HEIGHT * 0.5)
    roi = mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Detectar linhas usando Hough Transform
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        # Calcular a linha média
        x_sum, y_sum, count = 0, 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += (x1 + x2) / 2
            y_sum += (y1 + y2) / 2
            count += 1
        if count > 0:
            x_center = x_sum / count + roi_x
            y_center = y_sum / count + roi_y
            y_ref = (x_center - WIDTH / 2) * 0.05  # Escala para metros
            psi_ref = 0.0  # Simplificação: orientação desejada = 0
            delta = np.arctan2(y_ref, 10.0)  # Ângulo aproximado
            return delta, y_ref, psi_ref, True
    return 0.0, 0.0, 0.0, False

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

def control_vehicle(world, vehicle, solver, n_states, n_controls):
    """Controla o veículo usando o NMPC."""
    global current_image, current_mask, frame_count
    running = True
    paused = False
    speed_factor = 0.5
    target_speed = 10.0  # Velocidade desejada (m/s)

    vehicle_control = carla.VehicleControl()

    while running:
        world.tick()

        if not paused:
            # Obter estado do veículo
            transform = vehicle.get_transform()
            x = transform.location.x
            y = transform.location.y
            psi = np.deg2rad(transform.rotation.yaw)
            velocity = vehicle.get_velocity()
            v = np.sqrt(velocity.x**2 + velocity.y**2)

            # Processar máscara para referência
            delta_ref, y_ref, psi_ref, lines_detected = process_mask(current_mask) if current_mask is not None else (0.0, 0.0, 0.0, False)

            # Configurar parâmetros do NMPC
            state_init = np.array([x, y, psi, v])
            state_ref = np.array([x, y + y_ref, psi + psi_ref, target_speed])
            p = np.concatenate((state_init, state_ref))

            # Inicializar variáveis de otimização
            x0 = np.zeros((N + 1) * n_states + N * n_controls)
            for k in range(N + 1):
                x0[k * n_states:(k + 1) * n_states] = state_init

            # Resolver o problema de otimização
            res = solver(x0=x0, p=p, lbg=0, ubg=0)
            u_opt = res['x'][-N * n_controls:].reshape((n_controls, N))
            delta = float(u_opt[0, 0])
            a = float(u_opt[1, 0])


            # Aplicar controles
            vehicle_control.steer = delta / MAX_STEER
            vehicle_control.throttle = max(0.0, a / MAX_ACC) if a > 0 else 0.0
            vehicle_control.brake = -a / MAX_ACC if a < 0 else 0.0
            vehicle.apply_control(vehicle_control)

            # Exibir informações
            if current_image is not None:
                img_display = current_image.copy()
                speed = 3.6 * v
                cv2.putText(img_display, f"Speed: {speed:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_display, f"Steering: {delta * 180.0 / np.pi:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_display, f"Lines Detected: {lines_detected}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img_display, f"MPC Delta: {delta * 180.0 / np.pi:.2f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img_display, f"Speed factor: {speed_factor:.2f}", (WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if paused:
                    cv2.putText(img_display, "PAUSED (press 'R' to resume)", (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('CARLA View', img_display)

                if current_mask is not None:
                    mask_display = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
                    roi_x = int(WIDTH * 0.10)
                    roi_width = int(WIDTH * 0.80)
                    roi_y = int(HEIGHT * 0.5)
                    roi_height = int(HEIGHT * 0.6)
                    cv2.rectangle(mask_display, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
                    cv2.imshow('CARLA Mask', mask_display)

            if lines_detected:
                print(f"Frame {frame_count}: y_ref = {y_ref:.3f} m, psi_ref = {psi_ref:.3f} rad, delta = {delta * 180.0 / np.pi:.3f} deg")
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                running = False
            elif key == ord('p'):
                paused = not paused
                print(f"Navigation {'paused' if paused else 'resumed'}.")
                if paused:
                    vehicle_control.throttle = 0.0
                    vehicle_control.brake = 1.0
                    vehicle.apply_control(vehicle_control)
            elif key == ord('+'):
                speed_factor = min(1.0, speed_factor + 0.05)
                target_speed = MAX_SPEED * speed_factor
                print(f"Speed factor: {speed_factor:.2f}")
            elif key == ord('-'):
                speed_factor = max(0.1, speed_factor - 0.05)
                target_speed = MAX_SPEED * speed_factor
                print(f"Speed factor: {speed_factor:.2f}")

        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                paused = False
                print("Navigation resumed.")
            elif key == 27:
                running = False

    vehicle_control.throttle = 0.0
    vehicle_control.brake = 1.0
    vehicle.apply_control(vehicle_control)
    print("Vehicle stopped.")

def process_image(image):
    """Processa imagens RGB capturadas pela câmera."""
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    current_image = np.reshape(array, (image.height, image.width, 4))[:, :, :3].copy()

def process_segmentation(image):
    """Processa imagens de segmentação para criar máscara binária (ID 24)."""
    global current_mask
    image.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    mask = (arr[:, :, 2] == 24).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    current_mask = mask

def main():
    """Configura a simulação CARLA e executa o NMPC."""
    try:
        print("Conectando ao CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.load_world("/Game/Carla/Maps/Town10HD_Opt")
        world = client.get_world()

        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        blueprint = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint.find('vehicle.tesla.model3')
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
        cam_rgb = blueprint.find('sensor.camera.rgb')
        cam_rgb.set_attribute('image_size_x', str(WIDTH))
        cam_rgb.set_attribute('image_size_y', str(HEIGHT))
        cam_rgb.set_attribute('fov', '90')
        camera = world.spawn_actor(cam_rgb, camera_transform, attach_to=vehicle)

        cam_seg = blueprint.find('sensor.camera.semantic_segmentation')
        cam_seg.set_attribute('image_size_x', str(WIDTH))
        cam_seg.set_attribute('image_size_y', str(HEIGHT))
        cam_seg.set_attribute('fov', '90')
        seg = world.spawn_actor(cam_seg, camera_transform, attach_to=vehicle)

        camera.listen(process_image)
        seg.listen(process_segmentation)

        print("Configurando NMPC...")
        solver, n_states, n_controls = setup_nmpc()

        print("Iniciando navegação...")
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            world.tick()

        control_vehicle(world, vehicle, solver, n_states, n_controls)

    except Exception as e:
        print(f"Erro: {e}")
    finally:
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
        if 'seg' in locals():
            seg.stop()
            seg.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'world' in locals():
            world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        cv2.destroyAllWindows()
        print(f"Sessão terminada. {frame_count} imagens processadas.")

if __name__ == "__main__":
    main()