import carla
import numpy as np
import cvxpy as cp
import cv2
import time
import math
import os

def connect_to_carla(host='localhost', port=2000, retries=5, delay=5.0):
    """Tenta conectar ao CARLA com retries."""
    for attempt in range(retries):
        try:
            client = carla.Client(host, port)
            client.set_timeout(20.0)
            world = client.get_world()
            print(f"Conectado ao CARLA na tentativa {attempt + 1}, mapa:", world.get_map().name)
            return client
        except RuntimeError as e:
            print(f"Tentativa {attempt + 1} falhou: {e}")
            if attempt < retries - 1:
                print(f"Aguardando {delay} segundos antes da próxima tentativa...")
                time.sleep(delay)
    raise RuntimeError("Não foi possível conectar ao CARLA. Verifique se o simulador está rodando.")

def get_vehicle_state(vehicle):
    """Obtém a posição e orientação do veículo."""
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'yaw': math.radians(transform.rotation.yaw),
        'v': math.sqrt(velocity.x**2 + velocity.y**2)
    }

def mpc_control(state, target_y, horizon=10, dt=0.1):
    """Controlador MPC simplificado para movimento inicial."""
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    B = np.array([[0, 0],
                  [dt, 0],
                  [0, 0],
                  [0, dt]])
    
    x0 = np.array([state['x'], state['v'] * math.cos(state['yaw']),
                   state['y'], state['v'] * math.sin(state['yaw'])])
    
    x = cp.Variable((4, horizon + 1))
    u = cp.Variable((2, horizon))
    
    Q = np.diag([0.1, 0.1, 5, 0.1])
    R = np.diag([5.0, 5.0])
    cost = 0
    constraints = [x[:, 0] == x0]
    
    for t in range(horizon):
        cost += cp.quad_form(x[:, t] - np.array([0, 0, target_y, 0]), Q)
        cost += cp.quad_form(u[:, t], R)
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
    
    constraints += [u <= 0.2, u >= -0.2]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    if prob.status != cp.OPTIMAL:
        print("MPC não convergiu!")
        return 0.3, 0.0  # Throttle positivo para movimento
    
    ax, ay = u.value[:, 0]
    throttle = np.clip(ax, -0.2, 0.2)
    steer = np.clip(ay, -0.2, 0.2)
    return throttle, steer

def process_image(image):
    """Converte imagem CARLA para formato OpenCV, exibe e salva."""
    print(f"Imagem recebida: tamanho = {image.width}x{image.height}, frame = {image.frame}")
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    cv2.imshow("Camera Feed", array)
    cv2.waitKey(1)
    # Salvar imagem para debug
    os.makedirs("camera_debug", exist_ok=True)
    cv2.imwrite(f"camera_debug/frame_{image.frame}.png", array)

def main():
    # Conectar ao CARLA
    client = connect_to_carla()
    world = client.get_world()
    
    # Carregar mapa Town01
    try:
        world = client.load_world('Town01')
        print("Mapa Town01 carregado.")
    except RuntimeError as e:
        print(f"Falha ao carregar Town01: {e}. Usando mapa atual.")
    
    # Sincronizar simulação
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    # Obter blueprint do veículo
    for attempt in range(3):
        try:
            blueprint_library = world.get_blueprint_library()
            break
        except RuntimeError as e:
            print(f"Falha ao obter blueprint_library (tentativa {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(5.0)
            else:
                raise RuntimeError("Não foi possível obter blueprint_library.")
    
    vehicle_bp = blueprint_library.filter('model3')[0]
    
    # Spawnar veículo
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Nenhum ponto de spawn disponível!")
    print(f"Total de spawn points: {len(spawn_points)}")
    
    vehicle = None
    for i, spawn_point in enumerate(spawn_points[:5]):  # Tentar primeiros 5 pontos
        try:
            print(f"Tentando spawn point {i}: {spawn_point.location}")
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Veículo spawnado: ID = {vehicle.id}, ativo = {vehicle.is_alive}")
            break
        except RuntimeError as e:
            print(f"Falha ao spawnar no ponto {i}: {e}")
    
    if vehicle is None:
        raise RuntimeError("Não foi possível spawnar o veículo em nenhum ponto!")
    
    # Adicionar sensor de colisão
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collisions = []
    def on_collision(event):
        collisions.append(event)
        print(f"Colisão detetada: {event}")
    collision_sensor.listen(on_collision)
    
    # Adicionar sensor de câmera RGB
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '400')
    camera_bp.set_attribute('image_size_y', '300')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('role_name', 'front_camera')
    camera_bp.set_attribute('enable_postprocess_effects', 'true')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(process_image)
    
    try:
        # Loop principal
        target_y = get_vehicle_state(vehicle)['y']
        frame_count = 0
        while True:
            world.tick()
            frame_count += 1
            
            state = get_vehicle_state(vehicle)
            
            if collisions:
                print("Colisão detetada, reiniciando veículo.")
                vehicle.set_transform(spawn_points[0])
                collisions.clear()
            
            throttle, steer = mpc_control(state, target_y)
            
            control = carla.VehicleControl(
                throttle=throttle if throttle > 0 else 0.0,
                steer=steer,
                brake=-throttle if throttle < 0 else 0.0
            )
            vehicle.apply_control(control)
            
            print(f"Frame {frame_count}: Pos: ({state['x']:.2f}, {state['y']:.2f}), Throttle: {throttle:.2f}, Steer: {steer:.2f}")
            
            time.sleep(0.05)
    
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()
        camera.destroy()
        collision_sensor.destroy()
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        print("Veículo e sensores destruídos.")

if __name__ == '__main__':
    main()