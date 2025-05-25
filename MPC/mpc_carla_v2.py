import carla
import numpy as np
import cv2
import time
import os
import math


def connect_to_carla(host='localhost', port=2000, retries=5, delay=5.0):
    for attempt in range(retries):
        try:
            client = carla.Client(host, port)
            client.set_timeout(20.0)
            world = client.get_world()
            print(f"Conectado ao CARLA na tentativa {attempt + 1}, mapa: {world.get_map().name}")
            return client
        except RuntimeError as e:
            print(f"Tentativa {attempt + 1} falhou: {e}")
            time.sleep(delay)
    raise RuntimeError("Não conectou ao CARLA.")

def get_vehicle_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'v': math.sqrt(velocity.x**2 + velocity.y**2)
    }

def simple_control(state):
    return 0.3, 0.0  # Throttle fixo, sem virar

def process_image(image):
    print(f"Imagem recebida: frame={image.frame}, tamanho={image.width}x{image.height}")
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    cv2.imshow("Camera Feed", array)
    cv2.waitKey(1)
    os.makedirs("camera_debug", exist_ok=True)
    cv2.imwrite(f"camera_debug/frame_{image.frame}.png", array)

def main():
    client = connect_to_carla()
    world = client.get_world()
    
    print(f"Usando mapa atual: {world.get_map().name}")
    time.sleep(5.0)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    time.sleep(2.0)
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    
    spawn_points = world.get_map().get_spawn_points()
    print(f"Total de spawn points: {len(spawn_points)}")
    
    vehicle = None
    for i, spawn_point in enumerate(spawn_points[:10]):
        try:
            print(f"Tentando spawn point {i}: {spawn_point.location}")
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Veículo spawnado: ID={vehicle.id}, ativo={vehicle.is_alive}")
            break
        except RuntimeError as e:
            print(f"Falha ao spawnar no ponto {i}: {e}")
    
    if vehicle is None:
        raise RuntimeError("Não foi possível spawnar o veículo.")
    
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collisions = []
    def on_collision(event):
        collisions.append(event)
        print(f"Colisão detetada: {event}")
    collision_sensor.listen(on_collision)
    
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '400')
    camera_bp.set_attribute('image_size_y', '300')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('role_name', 'front_camera')
    camera_bp.set_attribute('enable_postprocess_effects', 'true')
    camera_bp.set_attribute('sensor_tick', '0.05')
    camera_bp.set_attribute('gamma', '2.2')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f"Câmera spawnada: ID={camera.id}, ativo={camera.is_alive}")
    camera.listen(process_image)
    
    try:
        frame_count = 0
        while True:
            world.tick()
            frame_count += 1
            
            state = get_vehicle_state(vehicle)
            
            if collisions:
                print("Colisão detetada, reiniciando veículo.")
                vehicle.set_transform(spawn_points[0])
                collisions.clear()
            
            throttle, steer = simple_control(state)
            
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=0.0
            )
            vehicle.apply_control(control)
            
            print(f"Frame {frame_count}: Pos: ({state['x']:.2f}, {state['y']:.2f}), Throttle: {throttle:.2f}, Steer: {steer:.2f}")
            time.sleep(0.1)
    
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